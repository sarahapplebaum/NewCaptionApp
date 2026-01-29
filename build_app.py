#!/usr/bin/env python3
"""
Build script for Video Captioner application
Cross-platform build with FFmpeg bundling and GPU support

Usage:
    python build_app.py              # Build with auto-detected settings
    python build_app.py --clean      # Clean previous builds first
    python build_app.py --download-ffmpeg  # Download FFmpeg before building
    python build_app.py --cpu-only   # Build CPU-only version (smaller size)
"""

import os
import sys
import shutil
import subprocess
import platform
import argparse

# ========================================
# CONFIGURATION
# ========================================

SPEC_FILE = "captioner_compact.spec"
MAIN_SCRIPT = "captioner_compact.py"
APP_NAME = "VideoCaptioner"

# ========================================
# HELPER FUNCTIONS
# ========================================

def print_header(message: str):
    """Print a formatted header"""
    print()
    print("=" * 60)
    print(f"  {message}")
    print("=" * 60)


def print_step(message: str):
    """Print a step message"""
    print(f"\n➡️  {message}")


def print_success(message: str):
    """Print a success message"""
    print(f"✅ {message}")


def print_warning(message: str):
    """Print a warning message"""
    print(f"⚠️  {message}")


def print_error(message: str):
    """Print an error message"""
    print(f"❌ {message}")


def run_command(cmd: list, description: str = None, check: bool = True) -> bool:
    """Run a command and handle output"""
    if description:
        print_step(description)
    
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=False,  # Show output in real-time
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print_error(f"Command not found: {cmd[0]}")
        return False


def check_python_dependencies() -> bool:
    """Check if all required Python packages are installed"""
    print_step("Checking Python dependencies...")
    
    required_packages = [
        ('PyQt5', 'PyQt5'),
        ('torch', 'torch'),
        ('faster_whisper', 'faster-whisper'),
        ('librosa', 'librosa'),
    ]
    
    missing = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"   ✓ {package_name}")
        except ImportError:
            print(f"   ✗ {package_name}")
            missing.append(package_name)
    
    # Check PyInstaller
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'PyInstaller', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"   ✓ pyinstaller ({version})")
        else:
            missing.append('pyinstaller')
            print("   ✗ pyinstaller")
    except:
        missing.append('pyinstaller')
        print("   ✗ pyinstaller")
    
    if missing:
        print_error(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt pyinstaller")
        return False
    
    print_success("All Python dependencies found")
    return True


def check_ffmpeg() -> str:
    """Check if FFmpeg is available and return its path"""
    print_step("Checking FFmpeg...")
    
    is_windows = platform.system() == 'Windows'
    ffmpeg_name = 'ffmpeg.exe' if is_windows else 'ffmpeg'
    
    # Check project directory first
    project_ffmpeg = os.path.join(os.path.dirname(os.path.abspath(__file__)), ffmpeg_name)
    if os.path.exists(project_ffmpeg):
        print_success(f"FFmpeg found in project: {project_ffmpeg}")
        return project_ffmpeg
    
    # Check system PATH
    system_ffmpeg = shutil.which('ffmpeg')
    if system_ffmpeg:
        print_success(f"FFmpeg found in PATH: {system_ffmpeg}")
        return system_ffmpeg
    
    print_warning("FFmpeg not found")
    print("   Run: python scripts/download_ffmpeg.py")
    print("   Or install manually:")
    if is_windows:
        print("   - winget install ffmpeg")
        print("   - choco install ffmpeg")
    elif platform.system() == 'Darwin':
        print("   - brew install ffmpeg")
    else:
        print("   - sudo apt install ffmpeg")
    
    return None


def check_gpu_support() -> dict:
    """Check GPU support"""
    print_step("Checking GPU support...")
    
    info = {
        'cuda_available': False,
        'cuda_version': None,
        'gpu_name': None,
        'mps_available': False,
    }
    
    try:
        import torch
        
        # Check CUDA (NVIDIA)
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            print_success(f"CUDA GPU: {info['gpu_name']} (CUDA {info['cuda_version']})")
        
        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['mps_available'] = True
            print_success("Apple Silicon GPU (MPS) available")
        
        else:
            print_warning("No GPU acceleration available, will use CPU")
    
    except ImportError:
        print_warning("PyTorch not installed, cannot check GPU support")
    
    return info


def clean_build():
    """Clean previous build artifacts"""
    print_step("Cleaning previous builds...")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"   Removed: {dir_name}/")
            except Exception as e:
                print_warning(f"Could not remove {dir_name}: {e}")
    
    # Clean .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                try:
                    os.remove(os.path.join(root, file))
                except:
                    pass
    
    print_success("Clean complete")


def download_ffmpeg():
    """Download FFmpeg for bundling"""
    print_step("Downloading FFmpeg...")
    
    script_path = os.path.join('scripts', 'download_ffmpeg.py')
    
    if not os.path.exists(script_path):
        print_error(f"Download script not found: {script_path}")
        return False
    
    return run_command(
        [sys.executable, script_path],
        description=None
    )


def build_application(cpu_only: bool = False):
    """Build the application with PyInstaller"""
    print_step(f"Building application ({'CPU-only' if cpu_only else 'with GPU support'})...")
    
    if not os.path.exists(SPEC_FILE):
        print_error(f"Spec file not found: {SPEC_FILE}")
        return False
    
    # Build command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        SPEC_FILE,
        '--clean',
        '--noconfirm',
    ]
    
    success = run_command(cmd)
    
    if success:
        print_success("Build completed successfully!")
        
        # List output
        dist_dir = 'dist'
        if os.path.exists(dist_dir):
            print("\nBuild output:")
            for item in os.listdir(dist_dir):
                item_path = os.path.join(dist_dir, item)
                if os.path.isfile(item_path):
                    size_mb = os.path.getsize(item_path) / (1024 * 1024)
                    print(f"   📦 {item} ({size_mb:.1f} MB)")
                else:
                    print(f"   📁 {item}/")
    
    return success


def create_distribution_readme():
    """Create a README for the distribution"""
    print_step("Creating distribution README...")
    
    dist_dir = 'dist'
    if not os.path.exists(dist_dir):
        return
    
    system = platform.system()
    
    readme_content = f"""# Video Captioner - Installation Instructions

## System Information
- Built on: {system}
- Python version: {sys.version.split()[0]}

## Requirements

### FFmpeg
FFmpeg must be installed on your system for video processing.

- **macOS**: `brew install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html or use `winget install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

### GPU Acceleration (Optional)
- **NVIDIA GPU**: Requires CUDA-capable GPU and latest drivers
- **Apple Silicon**: Uses Metal/MPS automatically

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

## First Run
- The app will download AI models on first use (~1-2GB)
- This requires an internet connection
- Subsequent runs will use the cached models

## Troubleshooting

### FFmpeg not found
Ensure FFmpeg is installed and in your system PATH.

### Performance Issues
- Use smaller models (tiny, base, small) for faster processing
- GPU acceleration significantly improves performance

### Model Download Fails
- Check your internet connection
- The models are downloaded from Hugging Face

## For More Information
Visit: https://github.com/sarahapplebaum/NewCaptionApp
"""
    
    readme_path = os.path.join(dist_dir, 'README.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print_success(f"Created: {readme_path}")


def main():
    """Main build process"""
    parser = argparse.ArgumentParser(
        description="Build Video Captioner application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_app.py                    # Standard build
  python build_app.py --clean            # Clean and build
  python build_app.py --download-ffmpeg  # Download FFmpeg first
  python build_app.py --cpu-only         # CPU-only build (smaller)
        """
    )
    
    parser.add_argument('--clean', action='store_true',
                       help='Clean previous build artifacts')
    parser.add_argument('--download-ffmpeg', action='store_true',
                       help='Download FFmpeg before building')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Build CPU-only version (smaller size)')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency checks')
    
    args = parser.parse_args()
    
    # Print header
    print_header(f"Video Captioner Build Script")
    print(f"Platform: {platform.system()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")
    
    # Clean if requested
    if args.clean:
        clean_build()
    
    # Check dependencies
    if not args.skip_checks:
        if not check_python_dependencies():
            sys.exit(1)
        
        check_gpu_support()
    
    # Download FFmpeg if requested
    if args.download_ffmpeg:
        if not download_ffmpeg():
            print_warning("FFmpeg download failed, continuing anyway...")
    
    # Check FFmpeg
    ffmpeg_path = check_ffmpeg()
    if not ffmpeg_path:
        print_warning("FFmpeg not found. The app will require FFmpeg on the target system.")
    
    # Build the application
    if not build_application(cpu_only=args.cpu_only):
        print_error("Build failed!")
        sys.exit(1)
    
    # Create distribution README
    create_distribution_readme()
    
    # Print summary
    print_header("Build Complete!")
    print(f"""
Next steps:
1. Check the 'dist' directory for the built application
2. Test the application before distribution
3. To create a release, push a version tag:
   git tag v1.0.0
   git push origin v1.0.0

This will trigger the GitHub Actions workflow to build
for Windows (CUDA + CPU) and macOS automatically.
""")


if __name__ == "__main__":
    main()
