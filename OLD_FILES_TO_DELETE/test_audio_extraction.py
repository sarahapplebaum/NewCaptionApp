# test_audio_extraction.py
import sys
import os
from pathlib import Path

# Add your AudioProcessor class here (copy the find_ffmpeg and extract_audio_with_ffmpeg methods)

class AudioProcessor:
    """Handle audio processing with robust ffmpeg detection"""
    
    @staticmethod
    def find_ffmpeg():
        """Find ffmpeg executable with proper macOS app bundle support"""
        print("🔍 Looking for ffmpeg...")
        
        possible_paths = []
        
        # For PyInstaller onefile builds
        if hasattr(sys, '_MEIPASS'):
            possible_paths.extend([
                Path(sys._MEIPASS) / "ffmpeg",
                Path(sys._MEIPASS) / "ffmpeg.exe",
            ])
            print(f"Onefile mode detected: {sys._MEIPASS}")
        
        # For PyInstaller app bundle builds - check ALL possible locations
        if sys.executable.endswith('VideoCaption'):  # We're in an app bundle
            exe_path = Path(sys.executable)
            app_contents = exe_path.parent.parent  # Go up from MacOS to Contents
            
            possible_paths.extend([
                # MacOS directory (same as executable)
                exe_path.parent / "ffmpeg",
                exe_path.parent / "ffprobe",
                
                # Resources directory (where PyInstaller actually puts them)
                app_contents / "Resources" / "ffmpeg",
                app_contents / "Resources" / "ffprobe",
                
                # Frameworks directory (alternative location)
                app_contents / "Frameworks" / "ffmpeg", 
                app_contents / "Frameworks" / "ffprobe",
            ])
            print(f"App bundle mode - Contents dir: {app_contents}")
        
        # For development/system installs
        possible_paths.extend([
            Path.cwd() / "ffmpeg",
            Path("/opt/homebrew/bin/ffmpeg"),
            Path("/usr/local/bin/ffmpeg"),
            Path("/usr/bin/ffmpeg"),
        ])
        
        # Debug info
        print(f"sys.executable: {sys.executable}")
        print(f"sys._MEIPASS: {getattr(sys, '_MEIPASS', 'Not set (app bundle mode)')}")
        
        # Check each possible path
        for path in possible_paths:
            if path and path.exists():
                try:
                    # Make sure it's executable
                    os.chmod(path, 0o755)
                    
                    # Test if it's actually ffmpeg
                    result = subprocess.run([str(path), '-version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and 'ffmpeg version' in result.stdout.lower():
                        print(f"✅ Found working ffmpeg at: {path}")
                        return str(path)
                    else:
                        print(f"❌ File exists but not working ffmpeg: {path}")
                except Exception as e:
                    print(f"⚠️  Error testing {path}: {e}")
            elif path:
                print(f"❌ Not found: {path}")
        
        # Try system PATH as final fallback
        system_ffmpeg = shutil.which('ffmpeg')
        if system_ffmpeg:
            print(f"✅ Found ffmpeg in system PATH: {system_ffmpeg}")
            return system_ffmpeg
        
        print("❌ FFmpeg not found anywhere!")
        return None
    
    @staticmethod
    def setup_librosa_ffmpeg():
        """Configure librosa to use our ffmpeg"""
        ffmpeg_path = AudioProcessor.find_ffmpeg()
        if ffmpeg_path:
            # Set environment variable for librosa/audioread
            os.environ['FFMPEG_BINARY'] = ffmpeg_path
            print(f"🔧 Set FFMPEG_BINARY to: {ffmpeg_path}")
            return ffmpeg_path
        return None
    
    @staticmethod
    def extract_audio_with_ffmpeg(video_path: str, output_path: str = None) -> str:
        """Extract audio using ffmpeg directly"""
        ffmpeg_path = AudioProcessor.find_ffmpeg()
        if not ffmpeg_path:
            raise RuntimeError("FFmpeg not found. Please install ffmpeg or check the app bundle.")
        
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        try:
            # Extract audio using ffmpeg
            cmd = [
                ffmpeg_path,
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                '-loglevel', 'error',  # Reduce ffmpeg output
                output_path
            ]
            
            print(f"🎵 Extracting audio: {' '.join(cmd[:3])}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed (code {result.returncode}): {result.stderr}")
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError(f"FFmpeg produced no output file")
            
            print(f"✅ Audio extracted successfully: {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise RuntimeError("FFmpeg extraction timed out (5 minutes)")
        except Exception as e:
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise RuntimeError(f"Audio extraction failed: {str(e)}")

    @staticmethod
    def test_ffmpeg():
        """Test ffmpeg functionality"""
        ffmpeg_path = AudioProcessor.find_ffmpeg()
        if not ffmpeg_path:
            return False, "FFmpeg not found"
        
        try:
            result = subprocess.run([ffmpeg_path, '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                return True, f"FFmpeg working: {version_line}"
            else:
                return False, f"FFmpeg test failed: {result.stderr}"
        except Exception as e:
            return False, f"FFmpeg test error: {str(e)}"

# Test with your problematic file
test_file = "M0-01_Course-Trailer.mp4"  # Use full path if needed

print("=== AUDIO EXTRACTION TEST ===")
print(f"Testing file: {test_file}")

# Test ffmpeg detection
ffmpeg_path = AudioProcessor.find_ffmpeg()
print(f"FFmpeg found: {ffmpeg_path}")

if ffmpeg_path:
    try:
        # Test extraction
        output_file = AudioProcessor.extract_audio_with_ffmpeg(test_file)
        print(f"✅ Extraction successful: {output_file}")
        print(f"Output size: {os.path.getsize(output_file)} bytes")
        
        # Clean up
        os.unlink(output_file)
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
else:
    print("❌ No ffmpeg found")
