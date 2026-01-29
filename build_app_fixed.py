#!/usr/bin/env python3
"""
Fixed build script for Video Captioner application with proper macOS file type associations
"""

import os
import sys
import shutil
import subprocess
import platform

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'PyQt5',
        'torch',
        'faster_whisper',
        'librosa'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    # Check for PyInstaller separately
    try:
        result = subprocess.run([sys.executable, '-m', 'PyInstaller', '--version'], 
                                capture_output=True, text=True)
        if result.returncode != 0:
            missing.append('pyinstaller')
    except:
        missing.append('pyinstaller')
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Please install them with: pip install -r requirements.txt pyinstaller")
        return False
    
    print("✅ All Python dependencies found")
    
    # Check for FFmpeg
    ffmpeg_found = False
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        ffmpeg_found = True
        print(f"✅ FFmpeg found at: {ffmpeg_path}")
    else:
        print("⚠️  FFmpeg not found in PATH")
        print("The app will still build but will require FFmpeg on the target system")
    
    return True

def clean_build():
    """Clean previous build artifacts"""
    print("\nCleaning previous builds...")
    
    dirs_to_remove = ['build', 'dist', '__pycache__']
    files_to_remove = ['*.pyc', '*.pyo']
    
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}/")
    
    # Remove temp spec files
    for spec_file in ['captioner_compact_temp.spec', 'captioner_compact.spec.bak']:
        if os.path.exists(spec_file):
            os.remove(spec_file)
    
    print("✅ Clean complete")

def build_app():
    """Build the application with PyInstaller using fixed spec file"""
    print("\nBuilding application with fixed file type associations...")
    
    # Check if fixed spec file exists
    if not os.path.exists('captioner_compact_fixed.spec'):
        print("❌ captioner_compact_fixed.spec not found!")
        print("Make sure the fixed spec file is in the current directory.")
        return False
    
    # Check if entitlements file exists for macOS
    if platform.system() == 'Darwin' and not os.path.exists('entitlements.plist'):
        print("❌ entitlements.plist not found!")
        print("Make sure the entitlements file is in the current directory.")
        return False
    
    # Base PyInstaller command using the fixed spec file
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        'captioner_compact_fixed.spec',
        '--clean',
        '--noconfirm'
    ]
    
    # Add platform-specific options
    system = platform.system()
    if system == 'Darwin':  # macOS
        print("🍎 Building for macOS with proper file type associations...")
    elif system == 'Windows':
        print("🪟 Building for Windows...")
    else:  # Linux
        print("🐧 Building for Linux...")
    
    # Run PyInstaller
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Build completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False

def fix_macos_permissions():
    """Additional fixes for macOS app bundle"""
    if platform.system() != 'Darwin':
        return
    
    print("\nApplying macOS-specific fixes...")
    
    app_path = 'dist/VideoCaptioner.app'
    if not os.path.exists(app_path):
        print("❌ App bundle not found")
        return
    
    # Make the app executable
    try:
        subprocess.run(['chmod', '+x', f'{app_path}/Contents/MacOS/VideoCaptioner'], check=True)
        print("✅ Made app executable")
    except:
        print("⚠️  Could not set executable permissions")
    
    # Clear extended attributes that might cause issues
    try:
        subprocess.run(['xattr', '-cr', app_path], check=True)
        print("✅ Cleared extended attributes")
    except:
        print("⚠️  Could not clear extended attributes")

def create_distribution():
    """Create distribution package with instructions"""
    print("\nCreating distribution package...")
    
    system = platform.system()
    dist_dir = 'dist'
    
    if not os.path.exists(dist_dir):
        print("❌ No dist directory found. Build may have failed.")
        return
    
    # Create detailed README
    readme_content = """# Video Captioner - Installation Instructions

## IMPORTANT: macOS File Selection Fix

This version includes fixes for the file selection issue where all files appeared greyed out in Finder.

## Requirements
- FFmpeg must be installed on your system
  - macOS: `brew install ffmpeg`
  - Windows: Download from https://ffmpeg.org/download.html
  - Linux: `sudo apt-get install ffmpeg`

## Running the Application

### macOS
1. Open the VideoCaptioner.app
2. If you see a security warning:
   - Right-click the app and select "Open"
   - Click "Open" in the dialog that appears
3. You should now be able to select video files in the file picker

### Troubleshooting macOS File Selection

If files are still greyed out:
1. Open Terminal
2. Navigate to the app location: `cd /path/to/VideoCaptioner.app`
3. Run: `xattr -cr VideoCaptioner.app`
4. Run: `chmod +x VideoCaptioner.app/Contents/MacOS/VideoCaptioner`

### Windows
1. Run VideoCaptioner.exe
2. Windows Defender may show a warning on first run - click "More info" then "Run anyway"

### Linux
1. Make the file executable: `chmod +x VideoCaptioner`
2. Run: `./VideoCaptioner`

## File Type Support

The application supports the following file types:
- Video: MP4, MOV, AVI, MKV, WEBM, MPG, MPEG, WMV, M4V, FLV, 3GP
- Audio: MP3, WAV, M4A, FLAC, AAC, OGG, WMA

## Performance Tips
- Use smaller models (tiny, base, small) for faster processing
- GPU acceleration requires CUDA-capable NVIDIA GPU
- The first run will download AI models (requires internet connection)
"""
    
    readme_path = os.path.join(dist_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # Create a test checklist
    test_checklist = """# Testing Checklist for VideoCaptioner

## File Selection Test
1. [ ] Open the app
2. [ ] Click "Select Files"
3. [ ] Verify that video files (MP4, MOV, etc.) are NOT greyed out
4. [ ] Select a video file successfully
5. [ ] Process the video and verify output

## Supported File Types to Test
- [ ] MP4 file
- [ ] MOV file
- [ ] AVI file
- [ ] MKV file
- [ ] MP3 audio file
- [ ] WAV audio file

## If Files Are Still Greyed Out
This indicates the Info.plist file type associations didn't apply correctly.
Try rebuilding with: python build_app_fixed.py
"""
    
    checklist_path = os.path.join(dist_dir, 'TESTING_CHECKLIST.txt')
    with open(checklist_path, 'w') as f:
        f.write(test_checklist)
    
    print(f"✅ Distribution files created in {dist_dir}/")
    
    # List created files
    print("\nCreated files:")
    for item in os.listdir(dist_dir):
        item_path = os.path.join(dist_dir, item)
        if os.path.isfile(item_path):
            size_mb = os.path.getsize(item_path) / (1024 * 1024)
            print(f"  - {item} ({size_mb:.1f} MB)")
        else:
            print(f"  - {item}/")

def main():
    """Main build process"""
    print("🔨 Video Captioner Build Script (Fixed for macOS)")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Clean previous builds
    clean_build()
    
    # Build the app
    if not build_app():
        sys.exit(1)
    
    # Apply macOS-specific fixes
    fix_macos_permissions()
    
    # Create distribution
    create_distribution()
    
    print("\n✅ Build process complete!")
    print("\n📋 IMPORTANT: Test the app to ensure file selection works properly.")
    print("If files are still greyed out in the file picker, please let me know.")
    print("\nThe built app is in the 'dist' directory.")

if __name__ == "__main__":
    main()
