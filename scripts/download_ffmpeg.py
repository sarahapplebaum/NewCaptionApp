#!/usr/bin/env python3
"""
FFmpeg Download Script for CI/CD Builds

This script downloads platform-specific static FFmpeg binaries for bundling with the application.
Supports Windows, macOS, and Linux.

Usage:
    python scripts/download_ffmpeg.py

The FFmpeg binary will be downloaded to the project root directory.
"""

import os
import sys
import platform
import urllib.request
import zipfile
import tarfile
import shutil
import hashlib
import io
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ========================================
# CONFIGURATION
# ========================================

# FFmpeg download URLs (static builds)
FFMPEG_URLS = {
    'Windows': {
        'url': 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip',
        'binary_path': 'ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe',
        'output_name': 'ffmpeg.exe',
    },
    'Darwin': {  # macOS
        'url': 'https://evermeet.cx/ffmpeg/getrelease/zip',
        'binary_path': 'ffmpeg',
        'output_name': 'ffmpeg',
    },
    'Linux': {
        'url': 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz',
        'binary_path': 'ffmpeg-master-latest-linux64-gpl/bin/ffmpeg',
        'output_name': 'ffmpeg',
    },
}

# Alternative URLs in case primary fails
FFMPEG_FALLBACK_URLS = {
    'Windows': 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
    'Darwin': 'https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip',
    'Linux': 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz',
}

# ========================================
# HELPER FUNCTIONS
# ========================================

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.resolve()


def download_file(url: str, dest_path: Path, description: str = "file") -> bool:
    """Download a file with progress indication"""
    print(f"[DOWNLOAD] Downloading {description}...")
    print(f"   URL: {url}")
    
    try:
        # Create request with user agent
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; FFmpegDownloader/1.0)'}
        )
        
        with urllib.request.urlopen(request, timeout=300) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192
            
            with open(dest_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end='', flush=True)
            
            print()  # New line after progress
        
        print(f"[OK] Downloaded to: {dest_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def extract_zip(archive_path: Path, extract_to: Path) -> bool:
    """Extract a ZIP archive"""
    print("[EXTRACT] Extracting ZIP archive...")
    try:
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_to)
        print(f"[OK] Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False


def extract_tar(archive_path: Path, extract_to: Path) -> bool:
    """Extract a TAR archive (including .tar.xz)"""
    print("[EXTRACT] Extracting TAR archive...")
    try:
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(extract_to)
        print(f"[OK] Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False


def find_ffmpeg_in_directory(directory: Path, expected_name: str) -> Path | None:
    """Find FFmpeg binary in extracted directory"""
    # Try exact path first
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower() == expected_name.lower() or file.lower() == 'ffmpeg' or file.lower() == 'ffmpeg.exe':
                return Path(root) / file
    return None


def make_executable(file_path: Path):
    """Make file executable on Unix systems"""
    if platform.system() != 'Windows':
        os.chmod(file_path, 0o755)
        print(f"[OK] Made executable: {file_path}")


# ========================================
# MAIN DOWNLOAD FUNCTION
# ========================================

def download_ffmpeg(output_dir: Path = None) -> bool:
    """
    Download FFmpeg for the current platform.
    
    Args:
        output_dir: Directory to place the FFmpeg binary (defaults to project root)
    
    Returns:
        True if successful, False otherwise
    """
    current_platform = platform.system()
    
    if current_platform not in FFMPEG_URLS:
        print(f"[ERROR] Unsupported platform: {current_platform}")
        return False
    
    config = FFMPEG_URLS[current_platform]
    
    if output_dir is None:
        output_dir = get_project_root()
    
    output_path = output_dir / config['output_name']
    
    # Check if FFmpeg already exists
    if output_path.exists():
        print(f"[INFO] FFmpeg already exists at: {output_path}")
        # Verify it works
        try:
            import subprocess
            result = subprocess.run([str(output_path), '-version'], capture_output=True, timeout=10)
            if result.returncode == 0:
                print("[OK] Existing FFmpeg is valid")
                return True
            else:
                print("[WARN] Existing FFmpeg is invalid, re-downloading...")
        except Exception:
            print("[WARN] Could not verify existing FFmpeg, re-downloading...")
    
    # Create temp directory for download
    temp_dir = output_dir / '.ffmpeg_temp'
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Determine archive extension
        url = config['url']
        if url.endswith('.zip'):
            archive_name = 'ffmpeg.zip'
            extract_func = extract_zip
        elif url.endswith('.tar.xz') or url.endswith('.tar.gz'):
            archive_name = 'ffmpeg.tar.xz'
            extract_func = extract_tar
        else:
            # Assume zip for unknown
            archive_name = 'ffmpeg.zip'
            extract_func = extract_zip
        
        archive_path = temp_dir / archive_name
        
        # Download
        success = download_file(url, archive_path, f"FFmpeg for {current_platform}")
        
        if not success:
            # Try fallback URL
            if current_platform in FFMPEG_FALLBACK_URLS:
                print("[WARN] Trying fallback URL...")
                fallback_url = FFMPEG_FALLBACK_URLS[current_platform]
                success = download_file(fallback_url, archive_path, f"FFmpeg (fallback)")
        
        if not success:
            return False
        
        # Extract
        extract_dir = temp_dir / 'extracted'
        extract_dir.mkdir(exist_ok=True)
        
        if not extract_func(archive_path, extract_dir):
            return False
        
        # Find FFmpeg binary
        ffmpeg_binary = find_ffmpeg_in_directory(extract_dir, config['output_name'])
        
        if ffmpeg_binary is None:
            # For macOS evermeet builds, the file might be extracted directly
            potential_path = extract_dir / 'ffmpeg'
            if potential_path.exists():
                ffmpeg_binary = potential_path
        
        if ffmpeg_binary is None:
            print("[ERROR] Could not find FFmpeg binary in extracted files")
            print(f"   Searched in: {extract_dir}")
            # List contents for debugging
            for item in extract_dir.rglob('*'):
                print(f"      {item}")
            return False
        
        # Copy to output location
        print(f"[COPY] Copying FFmpeg to: {output_path}")
        shutil.copy2(ffmpeg_binary, output_path)
        
        # Make executable on Unix
        make_executable(output_path)
        
        # Verify the binary works
        import subprocess
        try:
            result = subprocess.run([str(output_path), '-version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0] if result.stdout else 'Unknown version'
                print(f"[OK] FFmpeg verified: {version_line}")
            else:
                print("[WARN] FFmpeg returned non-zero exit code")
        except Exception as e:
            print(f"[WARN] Could not verify FFmpeg: {e}")
        
        return True
        
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            print("[CLEANUP] Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)


# ========================================
# CLI INTERFACE
# ========================================

def main():
    """Main entry point"""
    print("=" * 60)
    print("FFmpeg Download Script for Video Captioner")
    print("=" * 60)
    print(f"Platform: {platform.system()} ({platform.machine()})")
    print(f"Python: {sys.version}")
    print()
    
    # Parse optional output directory argument
    output_dir = None
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1]).resolve()
        print(f"Output directory: {output_dir}")
    else:
        print(f"Output directory: {get_project_root()} (project root)")
    
    print()
    
    success = download_ffmpeg(output_dir)
    
    print()
    print("=" * 60)
    if success:
        print("[SUCCESS] FFmpeg download completed successfully!")
        print("   The FFmpeg binary has been placed in the project root.")
        print("   It will be automatically bundled when building with PyInstaller.")
    else:
        print("[FAILED] FFmpeg download failed!")
        print("   Please download FFmpeg manually and place it in the project root.")
        print()
        print("   Manual download options:")
        print("   - Windows: https://www.gyan.dev/ffmpeg/builds/")
        print("   - macOS: https://evermeet.cx/ffmpeg/")
        print("   - Linux: https://johnvansickle.com/ffmpeg/")
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
