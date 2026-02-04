#!/usr/bin/env python3
"""
Optimized Build Script for Video Captioner
Builds CPU, CUDA (Windows), and macOS versions with minimal file size
"""

import subprocess
import sys
import os
import shutil
import platform

def run_command(cmd, description):
    """Run command and handle errors"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(e.stderr)
        return False

def clean_build():
    """Clean previous builds"""
    dirs_to_remove = ['build', 'dist', '__pycache__']
    for d in dirs_to_remove:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"✓ Cleaned: {d}")

def build_windows_cpu():
    """Build Windows CPU-only version"""
    print("\n🔨 Building Windows CPU version...")
    
    # Uninstall CUDA PyTorch if present
    run_command('pip uninstall -y torch torchvision torchaudio', 'Removing existing PyTorch')
    
    # Install CPU-only PyTorch
    run_command(
        'pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu',
        'Installing PyTorch CPU'
    )
    
    # Build with PyInstaller
    success = run_command(
        'pyinstaller --clean captioner_compact.spec',
        'Building with PyInstaller'
    )
    
    if success and os.path.exists('dist/VideoCaptioner'):
        # Rename output
        if os.path.exists('dist/VideoCaptioner-Windows-CPU'):
            shutil.rmtree('dist/VideoCaptioner-Windows-CPU')
        shutil.move('dist/VideoCaptioner', 'dist/VideoCaptioner-Windows-CPU')
        print("✅ Windows CPU build complete: dist/VideoCaptioner-Windows-CPU")
        return True
    return False

def build_windows_cuda():
    """Build Windows CUDA version"""
    print("\n🔨 Building Windows CUDA version...")
    
    # Uninstall CPU PyTorch
    run_command('pip uninstall -y torch torchvision torchaudio', 'Removing CPU PyTorch')
    
    # Install CUDA PyTorch
    run_command(
        'pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu121',
        'Installing PyTorch CUDA 12.1'
    )
    
    # Build with PyInstaller
    success = run_command(
        'pyinstaller --clean captioner_compact.spec',
        'Building with PyInstaller'
    )
    
    if success and os.path.exists('dist/VideoCaptioner'):
        # Rename output
        if os.path.exists('dist/VideoCaptioner-Windows-CUDA'):
            shutil.rmtree('dist/VideoCaptioner-Windows-CUDA')
        shutil.move('dist/VideoCaptioner', 'dist/VideoCaptioner-Windows-CUDA')
        print("✅ Windows CUDA build complete: dist/VideoCaptioner-Windows-CUDA")
        return True
    return False

def build_macos():
    """Build macOS version"""
    print("\n🔨 Building macOS version...")
    
    # Ensure CPU PyTorch for macOS
    run_command('pip uninstall -y torch torchvision torchaudio', 'Removing existing PyTorch')
    run_command(
        'pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0',
        'Installing PyTorch for macOS'
    )
    
    # Build with PyInstaller
    success = run_command(
        'pyinstaller --clean captioner_compact.spec',
        'Building with PyInstaller'
    )
    
    if success and os.path.exists('dist/VideoCaptioner.app'):
        print("✅ macOS build complete: dist/VideoCaptioner.app")
        return True
    return False

def main():
    """Main build orchestration"""
    print("🚀 Video Captioner - Optimized Build System")
    print(f"Platform: {platform.system()}")
    
    # Clean old builds
    clean_build()
    
    # Determine what to build
    is_windows = platform.system() == 'Windows'
    is_macos = platform.system() == 'Darwin'
    
    if is_windows:
        print("\n📋 Building for Windows...")
        
        # Ask user which builds to create
        print("\nAvailable builds:")
        print("  1. CPU-only (smaller, universal)")
        print("  2. CUDA (GPU support)")
        print("  3. Both")
        
        choice = input("\nSelect (1/2/3) [default: 1]: ").strip() or "1"
        
        if choice in ["1", "3"]:
            if not build_windows_cpu():
                print("❌ CPU build failed")
                return 1
        
        if choice in ["2", "3"]:
            if not build_windows_cuda():
                print("❌ CUDA build failed")
                return 1
    
    elif is_macos:
        print("\n📋 Building for macOS...")
        if not build_macos():
            print("❌ macOS build failed")
            return 1
    
    else:
        print("❌ Unsupported platform")
        return 1
    
    print("\n✅ Build process complete!")
    print(f"\n📦 Output location: {os.path.abspath('dist')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
