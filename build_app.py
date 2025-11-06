#!/usr/bin/env python3
"""
Build script for Video Captioner application
This script handles the PyInstaller build process with proper configuration
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
    
    # Remove .spec file if using custom build
    if os.path.exists('captioner_compact_temp.spec'):
        os.remove('captioner_compact_temp.spec')
    
    print("✅ Clean complete")

def build_app():
    """Build the application with PyInstaller"""
    print("\nBuilding application...")
    
    # Determine platform-specific options
    system = platform.system()
    
    # Base PyInstaller command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        'captioner_compact.spec',
        '--clean',
        '--noconfirm'
    ]
    
    # Add platform-specific options
    if system == 'Darwin':  # macOS
        print("🍎 Building for macOS...")
        # macOS specific options are handled in the spec file
    elif system == 'Windows':
        print("🪟 Building for Windows...")
        # Windows might need additional options
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

def create_distribution():
    """Create distribution package"""
    print("\nCreating distribution package...")
    
    system = platform.system()
    dist_dir = 'dist'
    
    if not os.path.exists(dist_dir):
        print("❌ No dist directory found. Build may have failed.")
        return
    
    # Create README for distribution
    readme_content = """# Video Captioner - Installation Instructions

## Requirements
- FFmpeg must be installed on your system
  - macOS: `brew install ffmpeg`
  - Windows: Download from https://ffmpeg.org/download.html
  - Linux: `sudo apt-get install ffmpeg`

## Running the Application

### macOS
1. Open the VideoCaptioner.app
2. If you see a security warning, right-click the app and select "Open"

### Windows
1. Run VideoCaptioner.exe
2. Windows Defender may show a warning on first run - click "More info" then "Run anyway"

### Linux
1. Make the file executable: `chmod +x VideoCaptioner`
2. Run: `./VideoCaptioner`

## Troubleshooting

### FFmpeg not found
Make sure FFmpeg is installed and in your system PATH.

### Model download on first run
The app will download AI models on first use. This requires an internet connection and may take a few minutes.

### Performance
- Use smaller models (tiny, base, small) for faster processing
- GPU acceleration requires CUDA-capable NVIDIA GPU
"""
    
    readme_path = os.path.join(dist_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
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
    print("🔨 Video Captioner Build Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Clean previous builds
    clean_build()
    
    # Build the app
    if not build_app():
        sys.exit(1)
    
    # Create distribution
    create_distribution()
    
    print("\n✅ Build process complete!")
    print("Check the 'dist' directory for the built application.")

if __name__ == "__main__":
    main()
