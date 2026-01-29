# Video Captioner - Build & Distribution Instructions

This document explains how to build and distribute the Video Captioner application for Windows, macOS, and Linux.

## Table of Contents
- [Quick Start](#quick-start)
- [Automated Builds (Recommended)](#automated-builds-recommended)
- [Manual Local Build](#manual-local-build)
- [Build Options](#build-options)
- [FFmpeg Bundling](#ffmpeg-bundling)
- [GPU Support](#gpu-support)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Option 1: Use GitHub Actions (Recommended)
The easiest way to build for Windows is to use the automated GitHub Actions workflow:

```bash
# Create a version tag
git tag v1.0.0
git push origin v1.0.0
```

This automatically builds:
- ✅ Windows x64 with NVIDIA CUDA support
- ✅ Windows x64 CPU-only (smaller download)
- ✅ macOS (Intel + Apple Silicon)

### Option 2: Local Build (macOS)
```bash
# Install dependencies
pip install -r requirements.txt pyinstaller

# Download FFmpeg for bundling
python scripts/download_ffmpeg.py

# Build the application
python build_app.py --clean
```

---

## Automated Builds (Recommended)

### GitHub Actions Workflow
The repository includes a GitHub Actions workflow that automatically builds the application when you push a version tag.

#### Trigger a Build

**Via Version Tag (creates a release):**
```bash
git tag v1.0.0
git push origin v1.0.0
```

**Manual Trigger (for testing):**
1. Go to the GitHub repository
2. Click "Actions" tab
3. Select "Build Application" workflow
4. Click "Run workflow"

#### Build Outputs
| Platform | GPU Support | Approximate Size |
|----------|-------------|-----------------|
| Windows x64 CUDA | ✅ NVIDIA | ~2.5 GB |
| Windows x64 CPU | ❌ | ~400 MB |
| macOS | ✅ Metal/MPS | ~500 MB |

#### Downloading Builds
- For tagged releases: Check the "Releases" page
- For manual builds: Check the "Actions" tab → Select run → Download artifacts

---

## Manual Local Build

### Prerequisites

#### All Platforms
```bash
# Install Python 3.11
python --version  # Should be 3.11.x

# Install dependencies
pip install -r requirements.txt
pip install pyinstaller
```

#### macOS
```bash
# Install FFmpeg
brew install ffmpeg
```

#### Windows
```powershell
# Install FFmpeg (choose one)
winget install ffmpeg
# or
choco install ffmpeg
```

#### Linux
```bash
# Install FFmpeg
sudo apt install ffmpeg
```

### Build Commands

#### Standard Build
```bash
python build_app.py
```

#### Clean Build
```bash
python build_app.py --clean
```

#### With FFmpeg Download
```bash
python build_app.py --download-ffmpeg --clean
```

#### CPU-Only Build (Smaller Size)
```bash
python build_app.py --cpu-only
```

### Build Output
The built application will be in the `dist/` directory:
- **macOS**: `dist/VideoCaptioner.app`
- **Windows**: `dist/VideoCaptioner.exe`
- **Linux**: `dist/VideoCaptioner`

---

## Build Options

### PyInstaller Spec File
The build configuration is in `captioner_compact.spec`. Key features:

```python
# Platform detection
PLATFORM = platform.system()  # 'Windows', 'Darwin', 'Linux'

# Dependencies collected automatically
- faster_whisper (Whisper AI model)
- ctranslate2 (CUDA support)
- torch (PyTorch with GPU support)
- librosa (audio processing)
- PyQt5 (GUI framework)

# FFmpeg bundling
- Automatically finds and bundles FFmpeg
- Checks project directory, common paths, and system PATH
```

### Build Script Options
```bash
python build_app.py --help

Options:
  --clean           Clean previous build artifacts
  --download-ffmpeg Download FFmpeg before building
  --cpu-only        Build CPU-only version (smaller size)
  --skip-checks     Skip dependency checks
```

---

## FFmpeg Bundling

### Automatic Download
```bash
python scripts/download_ffmpeg.py
```

This downloads a static FFmpeg build for your platform and places it in the project root.

### Manual Bundling
Place the FFmpeg binary in the project root:
- **Windows**: `ffmpeg.exe`
- **macOS/Linux**: `ffmpeg`

The spec file will automatically detect and bundle it.

### Download Sources
| Platform | Source |
|----------|--------|
| Windows | [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) |
| macOS | [evermeet.cx/ffmpeg](https://evermeet.cx/ffmpeg/) |
| Linux | [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) |

---

## GPU Support

### NVIDIA CUDA (Windows/Linux)
The Windows CUDA build includes:
- PyTorch with CUDA 12.1 support
- cuDNN libraries
- ctranslate2 CUDA backend

**Requirements on target system:**
- NVIDIA GPU with CUDA support
- Latest NVIDIA drivers (CUDA runtime is bundled)

### Apple Silicon (macOS)
- Uses Metal Performance Shaders (MPS)
- Automatically detected and enabled
- No additional drivers needed

### CPU Fallback
- All builds include CPU fallback
- Automatically used when no GPU is available

---

## Cross-Platform Building

### From macOS to Windows
**You cannot directly cross-compile.** Use GitHub Actions instead:

1. Push your code to GitHub
2. Create a version tag or trigger manual workflow
3. Download the Windows build from Artifacts/Releases

### From Windows to macOS
Same approach - use GitHub Actions for cross-platform builds.

---

## Distribution Checklist

### Before Release
- [ ] Test the application on target platform
- [ ] Verify FFmpeg is working
- [ ] Test GPU acceleration (if applicable)
- [ ] Check model download works
- [ ] Verify all features function correctly

### Release Package Contents
```
VideoCaptioner-Windows-x64-CUDA/
├── VideoCaptioner.exe
├── ffmpeg.exe (if bundled)
└── README.txt

VideoCaptioner-macOS/
├── VideoCaptioner.app/
└── README.txt
```

### Code Signing (Optional)
- **Windows**: Purchase code signing certificate (~$100-400/year)
- **macOS**: Requires Apple Developer account ($99/year)

Without code signing:
- Windows shows SmartScreen warning
- macOS shows Gatekeeper warning (right-click → Open)

---

## Troubleshooting

### Build Fails with Missing Module
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt pyinstaller

# Check for hidden imports in spec file
```

### FFmpeg Not Found in Build
```bash
# Download FFmpeg first
python scripts/download_ffmpeg.py

# Verify it's in project root
ls ffmpeg*  # Should show ffmpeg or ffmpeg.exe
```

### CUDA Not Working
```bash
# Verify CUDA PyTorch is installed
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Build Too Large
```bash
# Build CPU-only version
python build_app.py --cpu-only

# This removes CUDA libraries (~2GB smaller)
```

### GitHub Actions Fails
1. Check the workflow logs in the Actions tab
2. Verify `requirements.txt` is up to date
3. Check if FFmpeg download URL is still valid

---

## File Structure

```
NewCaptionApp/
├── .github/
│   └── workflows/
│       └── build.yml           # GitHub Actions workflow
├── scripts/
│   └── download_ffmpeg.py      # FFmpeg download helper
├── build_app.py                # Local build script
├── captioner_compact.spec      # PyInstaller configuration
├── captioner_compact.py        # Main application
├── requirements.txt            # Python dependencies
├── BUILD_INSTRUCTIONS.md       # This file
└── README.md                   # User documentation
```

---

## Version History

### v1.0.0
- Initial cross-platform build support
- Windows CUDA and CPU builds
- macOS Intel and Apple Silicon support
- Automatic FFmpeg bundling
- GitHub Actions CI/CD
