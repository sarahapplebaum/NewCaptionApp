import sys
import os
import time
import gc
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Union

# Lazy imports for heavy libraries
# torch, faster_whisper imported inside functions

from src.utils import (
    ProcessingStats,
    SubtitleFormatter,
    VocabularyCorrector,
    merge_split_compound_words,
    safe_float,
    DURATION_PATTERN
)

logger = logging.getLogger(__name__)

class FasterWhisperModelManager:
    """Singleton model manager for faster-whisper"""
    _instance = None
    _model = None
    _current_model_id = None
    _current_device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def get_optimal_device_config():
        """Determine optimal device configuration"""
        # Lazy import torch
        try:
            import torch
        except ImportError:
            logger.error("PyTorch not found")
            return {'device': 'cpu', 'compute_type': 'int8'}

        config = {}
        
        if torch.cuda.is_available():
            config['device'] = "cuda"
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                config['compute_type'] = "int8" if gpu_memory < 6 else "float16"
                logger.info(f"🚀 CUDA GPU: {torch.cuda.get_device_name()} ({gpu_memory:.1f}GB)")
            except:
                config['compute_type'] = "int8"
        else:
            config['device'] = "cpu"
            config['compute_type'] = "int8"
            config['cpu_threads'] = min(8, os.cpu_count() or 4)
            logger.info(f"🖥️  CPU (using {config.get('cpu_threads')} threads)")
        
        return config
    
    def load_model_optimized(self, model_id: str = "large-v3") -> bool:
        """Load faster-whisper model"""
        try:
            # Lazy import faster-whisper
            from faster_whisper import WhisperModel
            import torch # Ensure torch is available
            
            model_name = model_id.replace("openai/whisper-", "").replace("openai/", "")
            config = self.get_optimal_device_config()
            
            # Check if already loaded
            if (self._current_model_id == model_name and 
                self._current_device == config['device'] and
                self._model is not None):
                logger.info(f"✅ Model already loaded: {model_name}")
                return True
            
            self.clear_model()
            
            logger.info(f"🔄 Loading faster-whisper model: {model_name}")
            start_time = time.time()
            
            model_kwargs = {
                'device': config['device'],
                'compute_type': config['compute_type'],
            }
            
            if config['device'] == 'cpu':
                model_kwargs['cpu_threads'] = config.get('cpu_threads', 4)
            
            self._model = WhisperModel(model_name, **model_kwargs)
            self._current_model_id = model_name
            self._current_device = config['device']
            
            logger.info(f"✅ Model loaded in {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.clear_model()
            return False
    
    def clear_model(self):
        """Clear model and free memory"""
        if self._model is not None:
            del self._model
            self._model = None
        
        self._current_model_id = None
        self._current_device = None
        
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def get_model(self):
        return self._model
    
    @contextmanager
    def model_context(self, model_id: str):
        """Context manager for model usage"""
        try:
            if not self.load_model_optimized(model_id):
                # raise RuntimeError(f"Failed to load model: {model_id}") # Don't raise, just yield None
                 yield None
            else:
                 yield self._model
        finally:
            pass  # Keep model loaded for batch processing

class AudioProcessor:
    """Audio extraction and processing"""
    
    _ffmpeg_cache = {}
    _audio_cache = {}
    
    @staticmethod
    def find_ffmpeg():
        """Find FFmpeg executable with cross-platform support"""
        import platform
        
        if 'ffmpeg_path' in AudioProcessor._ffmpeg_cache:
            cached = AudioProcessor._ffmpeg_cache['ffmpeg_path']
            if cached and os.path.exists(cached):
                return cached
        
        paths = []
        is_windows = platform.system() == 'Windows'
        ffmpeg_name = 'ffmpeg.exe' if is_windows else 'ffmpeg'
        
        # Priority 1: Bundled FFmpeg (PyInstaller)
        if hasattr(sys, '_MEIPASS'):
            bundle_path = Path(sys._MEIPASS)
            paths.extend([
                bundle_path / ffmpeg_name,
                bundle_path / 'ffmpeg' / ffmpeg_name,
                bundle_path / 'bin' / ffmpeg_name,
            ])
        
        # Priority 2: Application directory (for portable installs)
        if getattr(sys, 'frozen', False):
            app_dir = Path(sys.executable).parent
        else:
            # We are in src/core.py, so parent.parent is project root
            # But let's act relative to current working dir or typical locations
            app_dir = Path(os.getcwd()) 
            
        paths.extend([
            app_dir / ffmpeg_name,
            app_dir / 'ffmpeg' / ffmpeg_name,
            app_dir / 'bin' / ffmpeg_name,
        ])
        
        # Priority 3: Platform-specific common locations
        if is_windows:
            paths.extend([
                Path('C:/ffmpeg/bin/ffmpeg.exe'),
                Path('C:/Program Files/ffmpeg/bin/ffmpeg.exe'),
                Path('C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'),
            ])
            local_app_data = os.environ.get('LOCALAPPDATA', '')
            if local_app_data:
                paths.extend([
                    Path(local_app_data) / 'Microsoft' / 'WinGet' / 'Links' / 'ffmpeg.exe',
                    Path(local_app_data) / 'Programs' / 'ffmpeg' / 'bin' / 'ffmpeg.exe',
                ])
            choco_path = os.environ.get('ChocolateyInstall', 'C:\\ProgramData\\chocolatey')
            paths.append(Path(choco_path) / 'bin' / 'ffmpeg.exe')
        else:
            paths.extend([
                Path('/opt/homebrew/bin/ffmpeg'),
                Path('/usr/local/bin/ffmpeg'),
                Path('/usr/bin/ffmpeg'),
                Path('/snap/bin/ffmpeg'),
            ])
        
        for path in paths:
            if path.exists():
                try:
                    # Use shell=True on Windows to handle .exe properly is sometimes needed but we specifically avoid it if possible
                    creationflags = subprocess.CREATE_NO_WINDOW if is_windows else 0
                    result = subprocess.run(
                        [str(path), '-version'], 
                        capture_output=True, 
                        text=True, 
                        timeout=10,
                        creationflags=creationflags
                    )
                    if result.returncode == 0:
                        AudioProcessor._ffmpeg_cache['ffmpeg_path'] = str(path)
                        logger.info(f"✅ FFmpeg found: {path}")
                        return str(path)
                except Exception as e:
                    logger.debug(f"FFmpeg test failed for {path}: {e}")
                    continue
        
        system_ffmpeg = shutil.which('ffmpeg')
        if system_ffmpeg:
             AudioProcessor._ffmpeg_cache['ffmpeg_path'] = system_ffmpeg
             return system_ffmpeg
             
        logger.error("❌ FFmpeg not found. Please install FFmpeg.")
        return None

    @staticmethod
    def extract_audio(video_path: str, output_path: str = None) -> Tuple[str, float]:
        """Extract audio from video"""
        try:
            cache_key = f"{video_path}:{os.path.getmtime(video_path)}"
            if cache_key in AudioProcessor._audio_cache:
                cached_path, duration = AudioProcessor._audio_cache[cache_key]
                if os.path.exists(cached_path):
                    return cached_path, duration
        except:
            pass
        
        ffmpeg_path = AudioProcessor.find_ffmpeg()
        if not ffmpeg_path:
            raise RuntimeError("FFmpeg not found")
        
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        try:
            cmd = [
                ffmpeg_path, '-i', video_path, '-vn',
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-y', '-loglevel', 'error', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            duration = AudioProcessor.get_audio_duration(output_path)
            
            try:
                cache_key = f"{video_path}:{os.path.getmtime(video_path)}"
                AudioProcessor._audio_cache[cache_key] = (output_path, duration)
            except:
                pass
            
            return output_path, duration
            
        except Exception as e:
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except:
                    pass
            raise RuntimeError(f"Audio extraction failed: {str(e)}")
    
    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """Get audio duration using FFmpeg"""
        ffmpeg_path = AudioProcessor.find_ffmpeg()
        if ffmpeg_path:
            try:
                result = subprocess.run([ffmpeg_path, '-i', audio_path], 
                                      capture_output=True, text=True, timeout=10)
                # Output is on stderr often for ffmpeg -i
                match = DURATION_PATTERN.search(result.stderr)
                if match:
                    h, m, s, ms = map(int, match.groups())
                    return h * 3600 + m * 60 + s + ms / 100
            except Exception as e:
                logger.warning(f"Failed to get duration: {e}")
                pass
        return 0.0
    
    @staticmethod
    def is_audio_file(file_path: str) -> bool:
        return Path(file_path).suffix.lower() in {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma'}

class TranscriptionProcessor:
    """Transcription processing with faster-whisper"""
    
    def __init__(self, vocabulary_corrector: Optional[VocabularyCorrector] = None):
        self.model_manager = FasterWhisperModelManager()
        self.temp_files = []
        self.vocabulary_corrector = vocabulary_corrector
    
    def transcribe(self, audio_path: str, model, return_timestamps: bool, 
                  max_chars: int, file_stats: ProcessingStats,
                  initial_prompt: str = None) -> Tuple[str, List[Dict]]:
        
        transcription_start = time.time()
        
        try:
            transcribe_kwargs = {}
            if return_timestamps:
                transcribe_kwargs['word_timestamps'] = True
            
            if initial_prompt and initial_prompt.strip():
                transcribe_kwargs['initial_prompt'] = initial_prompt.strip()
                logger.info(f"📚 Using custom context prompt")
            elif self.vocabulary_corrector:
                vocab_prompt = self.vocabulary_corrector.get_initial_prompt()
                if vocab_prompt:
                    transcribe_kwargs['initial_prompt'] = vocab_prompt
                    logger.info("📚 Using vocabulary guidance")
            
            logger.info(f"🎙️ Transcribing...")
            segments_iter, info = model.transcribe(audio_path, **transcribe_kwargs)
            transcribed_segments = list(segments_iter)
            
            file_stats.transcription_time = time.time() - transcription_start
            
            text = " ".join([segment.text.strip() for segment in transcribed_segments if segment.text.strip()])
            segments = []
            
            if return_timestamps and transcribed_segments:
                words = []
                for segment in transcribed_segments:
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            words.append({
                                'word': word.word,
                                'start': safe_float(word.start, 0.0),
                                'end': safe_float(word.end, safe_float(word.start, 0.0) + 0.5)
                            })
                    else:
                         # Fallback logic
                        segment_text = segment.text.strip()
                        if segment_text:
                            segment_words = segment_text.split()
                            segment_duration = safe_float(segment.end, 0.0) - safe_float(segment.start, 0.0)
                            word_duration = segment_duration / max(len(segment_words), 1)
                            current_time = safe_float(segment.start, 0.0)
                            for word in segment_words:
                                words.append({
                                    'word': word,
                                    'start': current_time,
                                    'end': current_time + word_duration
                                })
                                current_time += word_duration

                if words and self.vocabulary_corrector:
                    logger.info("📚 Applying vocabulary corrections...")
                    words = self.vocabulary_corrector.correct_word_list(words)
                    if self.vocabulary_corrector.correction_log:
                        self.vocabulary_corrector.log_corrections_verbose()
                        logger.info(self.vocabulary_corrector.get_correction_summary())
                
                if words:
                    words = merge_split_compound_words(words)
                    segments = SubtitleFormatter.create_optimized_segments(words, max_chars)
                    for segment in segments:
                        if segment.get('text'):
                            segment['text'] = segment['text'].replace(' -', '-')
            
            logger.info(f"✅ Transcription complete: {len(text)} chars, {len(segments)} segments")
            return text, segments
            
        except Exception as e:
            file_stats.transcription_time = time.time() - transcription_start
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def extract_audio(self, video_path: str, file_stats: ProcessingStats) -> Tuple[str, float]:
        if AudioProcessor.is_audio_file(video_path):
            duration = AudioProcessor.get_audio_duration(video_path)
            return video_path, duration
        
        extraction_start = time.time()
        audio_path, duration = AudioProcessor.extract_audio(video_path)
        file_stats.audio_extraction_time = time.time() - extraction_start
        
        if audio_path != video_path:
            self.temp_files.append(audio_path)
        
        return audio_path, duration
    
    def cleanup_temp_files(self):
        for temp_file in self.temp_files[:]:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    self.temp_files.remove(temp_file)
            except Exception as e:
                logger.warning(f"⚠️ Could not cleanup {temp_file}: {e}")

def process_single_video_cli(video_path: str, output_folder: str, model_id: str = "small", 
                           max_chars_per_line: int = 42, max_chars_per_segment: int = 84,
                           generate_timestamps: bool = True, context_prompt: str = None,
                           vocab_csv: str = None, vocab_sensitivity: float = 0.85,
                           vocab_fallback: bool = True) -> bool:
    """Process a single video file from command line"""
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        vocabulary_corrector = None
        if vocab_csv and os.path.exists(vocab_csv):
            try:
                vocabulary_corrector = VocabularyCorrector(
                    csv_path=vocab_csv,
                    similarity_threshold=vocab_sensitivity,
                    enable_fallback=vocab_fallback
                )
                logger.info(f"📚 Vocabulary correction enabled with {len(vocabulary_corrector.terms)} terms")
            except Exception as e:
                logger.warning(f"⚠️ Could not load vocabulary file: {e}")
        
        model_manager = FasterWhisperModelManager()
        processor = TranscriptionProcessor(vocabulary_corrector=vocabulary_corrector)
        stats = ProcessingStats()
        stats.start_time = time.time()
        
        filename = os.path.basename(video_path)
        logger.info(f"🎬 Processing: {filename}")
        
        if context_prompt:
             logger.info(f"📚 Context prompt provided")

        if not os.path.exists(video_path):
            logger.error(f"❌ File not found: {video_path}")
            return False
        
        with model_manager.model_context(model_id) as model:
            if model is None:
                logger.error("❌ Failed to load AI model")
                return False
            
            logger.info("🎵 Extracting audio...")
            audio_path, duration = processor.extract_audio(video_path, stats)
            stats.audio_duration = duration
            logger.info(f"✅ Audio extracted: {duration:.2f}s")
            
            logger.info("🎙️ Transcribing...")
            text, segments = processor.transcribe(
                audio_path, model, generate_timestamps, max_chars_per_segment, stats, context_prompt
            )
            
            stats.characters_transcribed = len(text or "")
            stats.end_time = time.time()
            
            stem = Path(video_path).stem
            txt_path = Path(output_folder) / f"{stem}.txt"
            with open(txt_path, 'w', encoding='utf-8-sig') as f:
                f.write(text)
            logger.info(f"💾 Saved transcript: {txt_path}")
            
            if segments and generate_timestamps:
                vtt_path = Path(output_folder) / f"{stem}.vtt"
                vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line)
                with open(vtt_path, 'w', encoding='utf-8-sig') as f:
                    f.write(vtt_content)
                logger.info(f"💾 Saved VTT: {vtt_path}")
                
                srt_path = Path(output_folder) / f"{stem}.srt"
                srt_content = SubtitleFormatter.create_srt(segments, max_chars_per_line)
                with open(srt_path, 'w', encoding='utf-8-sig') as f:
                    f.write(srt_content)
                logger.info(f"💾 Saved SRT: {srt_path}")
            
            processor.cleanup_temp_files()
            
            logger.info(f"📊 Processing stats:")
            logger.info(f"   Total time: {stats.total_time:.2f}s")
            logger.info(f"   Audio duration: {stats.audio_duration:.2f}s")
            if stats.total_time > 0:
                logger.info(f"   Speed: {stats.audio_duration / stats.total_time:.1f}x realtime")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return False
