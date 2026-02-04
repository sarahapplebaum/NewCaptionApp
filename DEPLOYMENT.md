# Video Captioner - Deployment Guide

This guide covers building and deploying the Video Captioner application for Windows, Windows CUDA, and macOS.

## 🚀 Quick Start

### GitHub Actions (Recommended)

**Automatic builds are triggered when you push a version tag:**

```bash
# Create a new version tag
git tag v1.1.0

# Push the tag to GitHub
git push origin v1.1.0
```

This will:
1. Build Windows CPU version
2. Build Windows CUDA version  
3. Build macOS version with DMG installer
4. Create a GitHub Release with all artifacts

### Manual Workflow Trigger

You can also manually trigger builds from the GitHub Actions tab:
1. Go to **Actions** → **Build Application**
2. Click **Run workflow**
3. Select branch and build type
4. Click **Run workflow**

---

## 🏗️ Local Building

### Prerequisites

**All Platforms:**
- Python 3.11
- Git

**Windows:**
- Visual C++ Redistributable 2019+
- FFmpeg (will be downloaded automatically)

**macOS:**
- Xcode Command Line Tools: `xcode-select --install`
- Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- FFmpeg: `brew install ffmpeg`

### Local Build Script

We provide an optimized build script that handles everything:

```bash
# Windows
python build_optimized.py

# macOS
python3 build_optimized.py
```

**Interactive menu:**
```
🚀 Video Captioner - Optimized Build System
Platform: Windows

📋 Building for Windows...

Available builds:
  1. CPU-only (smaller, universal)
  2. CUDA (GPU support)
  3. Both

Select (1/2/3) [default: 1]:
```

The script will:
1. Clean previous builds
2. Install the correct PyTorch version
3. Build with PyInstaller
4. Package the application
5. Create output in `dist/` folder

---

## 📦 Build Outputs

### Windows CPU (`VideoCaptioner-Windows-CPU/`)
- **Size:** ~800MB
- **PyTorch:** 2.8.0 CPU-only
- **Contents:** One-folder bundle with all DLLs
- **Includes:** FFmpeg bundled

### Windows CUDA (`VideoCaptioner-Windows-CUDA/`)
- **Size:** ~2.5GB (includes CUDA libraries)
- **PyTorch:** 2.8.0 with CUDA 12.1
- **Contents:** One-folder bundle with CUDA DLLs
- **Includes:** FFmpeg bundled
- **Note:** Falls back to CPU if no GPU detected

### macOS (`VideoCaptioner.app`)
- **Size:** ~700MB
- **PyTorch:** 2.8.0 (optimized for both Apple Silicon and Intel)
- **Format:** App bundle + DMG installer
- **Includes:** FFmpeg bundled

---

## 🔧 Advanced: Manual PyInstaller Build

### Windows CPU

```bash
# Install dependencies
pip install -r requirements.txt
pip install pyinstaller

# Install CPU-only PyTorch 2.8.0
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Download FFmpeg
python scripts/download_ffmpeg.py

# Build
pyinstaller captioner_compact.spec --clean
```

### Windows CUDA

```bash
# Install dependencies
pip install -r requirements.txt
pip install pyinstaller

# Install CUDA PyTorch 2.8.0
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# Download FFmpeg
python scripts/download_ffmpeg.py

# Build
pyinstaller captioner_compact.spec --clean
```

### macOS

```bash
# Install dependencies
pip install -r requirements.txt
pip install pyinstaller

# Install FFmpeg
brew install ffmpeg

# Install PyTorch 2.8.0
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# Build
pyinstaller captioner_compact.spec --clean

# Create DMG (optional)
hdiutil create -volname "VideoCaptioner" \
               -srcfolder dist/VideoCaptioner.app \
               -ov -format UDZO \
               VideoCaptioner.dmg
```

---

## 🏷️ Version Tagging Strategy

### Semantic Versioning

Use semantic versioning: `v{MAJOR}.{MINOR}.{PATCH}`

```bash
# Bug fix release
git tag v1.0.1
git push origin v1.0.1

# Minor feature release
git tag v1.1.0
git push origin v1.1.0

# Major release
git tag v2.0.0
git push origin v2.0.0
```

### Pre-release Tags

For beta/test releases:

```bash
git tag v1.1.0-beta.1
git push origin v1.1.0-beta.1
```

### Release Notes

GitHub will automatically generate release notes, but you can customize them:
1. Go to the release on GitHub
2. Click "Edit release"
3. Add custom release notes
4. Update the description

---

## 📋 Deployment Checklist

### Before Release

- [ ] Test application locally on Windows
- [ ] Test application locally on macOS (if available)
- [ ] Verify vocabulary correction works
- [ ] Test batch processing
- [ ] Verify FFmpeg bundling works
- [ ] Update version in README.md
- [ ] Update CHANGELOG.md (if exists)

### Create Release

```bash
# Pull latest changes
git pull origin main

# Create and push tag
git tag v1.X.X
git push origin v1.X.X
```

### After Release

- [ ] Monitor GitHub Actions for build completion (~12 minutes)
- [ ] Download and test each platform build
- [ ] Verify Windows CPU build works
- [ ] Verify Windows CUDA build works  
- [ ] Verify macOS build works
- [ ] Update release notes if needed
- [ ] Announce release (if applicable)

---

## 🐛 Troubleshooting

### GitHub Actions Build Fails

**Common issues:**

1. **Python dependency error**
   - Check `requirements.txt` is up to date
   - Verify PyTorch version compatibility

2. **FFmpeg not found**
   - Check `scripts/download_ffmpeg.py` works
   - Verify FFmpeg download URL is valid

3. **PyInstaller failure**
   - Check `captioner_compact.spec` for errors
   - Verify hidden imports are correct

**View logs:**
1. Go to Actions → Failed workflow
2. Click on the failed job
3. Expand the failed step to see detailed logs

### Local Build Issues

**Windows: "torch DLL failed to load"**
```bash
# Reinstall Visual C++ Redistributable
choco install vcredist140 -y --force

# Clear cache and rebuild
rmdir /s /q build dist
pip uninstall -y torch torchvision torchaudio
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pyinstaller captioner_compact.spec --clean
```

**macOS: "Cannot be opened because the developer cannot be verified"**
```bash
# Remove quarantine attribute
xattr -cr dist/VideoCaptioner.app
```

**Build size too large**
```bash
# Use CPU-only version instead of CUDA
# Remove unnecessary files from spec file
# Consider using UPX compression (not recommended for PyTorch)
```

---

## 📊 Build Artifacts

### Artifact Retention

- **GitHub Actions:** 90 days
- **Releases:** Permanent (until manually deleted)

### Download Artifacts

**From GitHub Actions:**
1. Go to Actions → Completed workflow
2. Scroll to "Artifacts" section
3. Download the platform-specific ZIP

**From Releases:**
1. Go to Releases
2. Click on the version
3. Download from "Assets" section

---

## 🔐 Security Notes

### Code Signing

**Windows:**
- Builds are not code-signed
- Users will see Windows SmartScreen warnings
- To sign: Obtain a code signing certificate and use `signtool`

**macOS:**
- Builds are not notarized
- Users must right-click → Open on first launch
- To notarize: Enroll in Apple Developer Program ($99/year)

### Antivirus False Positives

PyInstaller executables may trigger antivirus warnings:
- **Solution:** Submit builds to antivirus vendors for whitelisting
- **Alternative:** Use installer (e.g., Inno Setup for Windows, DMG for macOS)

---

## 📝 Additional Resources

- **PyInstaller Docs:** https://pyinstaller.org/
- **GitHub Actions Docs:** https://docs.github.com/actions
- **PyTorch Install:** https://pytorch.org/get-started/locally/
- **FFmpeg:** https://ffmpeg.org/

---

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review GitHub Actions logs
3. Check `%TEMP%\videocaptioner_debug.log` (Windows) or `/tmp/videocaptioner_debug.log` (macOS)
4. Open an issue on GitHub with:
   - Platform and version
   - Error message
   - Debug log contents
   - Steps to reproduce

---

**Last Updated:** 2026-02-04  
**Version:** 1.1.0
