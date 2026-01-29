# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Video Captioner - Cross-Platform Build
# Supports Windows, macOS, and Linux with FFmpeg bundling and CUDA GPU support

import sys
import os
import platform
import shutil
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# ========================================
# PLATFORM DETECTION
# ========================================

PLATFORM = platform.system()  # 'Windows', 'Darwin', 'Linux'
IS_WINDOWS = PLATFORM == 'Windows'
IS_MACOS = PLATFORM == 'Darwin'
IS_LINUX = PLATFORM == 'Linux'

print(f"🔨 Building for: {PLATFORM}")

# ========================================
# COLLECT DEPENDENCIES
# ========================================

datas = []
binaries = []
hiddenimports = []

# Collect faster-whisper dependencies
print("📦 Collecting faster-whisper...")
faster_whisper_datas, faster_whisper_binaries, faster_whisper_hiddenimports = collect_all('faster_whisper')
datas += faster_whisper_datas
binaries += faster_whisper_binaries
hiddenimports += faster_whisper_hiddenimports

# Collect ctranslate2 dependencies (required by faster-whisper, includes CUDA support)
print("📦 Collecting ctranslate2 (with CUDA)...")
ctranslate2_datas, ctranslate2_binaries, ctranslate2_hiddenimports = collect_all('ctranslate2')
datas += ctranslate2_datas
binaries += ctranslate2_binaries
hiddenimports += ctranslate2_hiddenimports

# Collect torch dependencies (includes CUDA libraries)
print("📦 Collecting torch...")
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
datas += torch_datas
binaries += torch_binaries
hiddenimports += torch_hiddenimports

# Collect librosa dependencies
print("📦 Collecting librosa...")
librosa_datas, librosa_binaries, librosa_hiddenimports = collect_all('librosa')
datas += librosa_datas
binaries += librosa_binaries
hiddenimports += librosa_hiddenimports

# ========================================
# HIDDEN IMPORTS FOR CUDA/GPU SUPPORT
# ========================================

hiddenimports += [
    # Core dependencies
    'sklearn.utils._typedefs',
    'sklearn.neighbors._partition_nodes',
    'numba',
    'audioread',
    'soundfile',
    'tokenizers',
    'huggingface_hub',
    'onnxruntime',
    
    # CUDA/GPU support
    'torch.cuda',
    'torch.backends.cudnn',
    'torch.backends.cuda',
    'torch._C',
    'torch._C._cuda',
    
    # Additional ctranslate2 CUDA support
    'ctranslate2',
    
    # PyQt5 platform plugins
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
]

# ========================================
# FFMPEG BUNDLING
# ========================================

def find_ffmpeg():
    """Find FFmpeg binary for bundling"""
    ffmpeg_name = 'ffmpeg.exe' if IS_WINDOWS else 'ffmpeg'
    
    # Check for bundled FFmpeg in project directory
    project_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ffmpeg_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg', ffmpeg_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', ffmpeg_name),
    ]
    
    for path in project_paths:
        if os.path.exists(path):
            print(f"✅ FFmpeg found in project: {path}")
            return path
    
    # Platform-specific paths
    if IS_WINDOWS:
        paths = [
            'C:\\ffmpeg\\bin\\ffmpeg.exe',
            'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
            'C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe',
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'WinGet', 'Links', 'ffmpeg.exe'),
            os.path.join(os.environ.get('ChocolateyInstall', 'C:\\ProgramData\\chocolatey'), 'bin', 'ffmpeg.exe'),
        ]
    elif IS_MACOS:
        paths = [
            '/opt/homebrew/bin/ffmpeg',
            '/usr/local/bin/ffmpeg',
        ]
    else:  # Linux
        paths = [
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg',
            '/snap/bin/ffmpeg',
        ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"✅ FFmpeg found: {path}")
            return path
    
    # Try system PATH
    system_ffmpeg = shutil.which('ffmpeg')
    if system_ffmpeg:
        print(f"✅ FFmpeg found in PATH: {system_ffmpeg}")
        return system_ffmpeg
    
    print("⚠️ FFmpeg not found! App will require FFmpeg on target system.")
    return None

# Find and bundle FFmpeg
ffmpeg_path = find_ffmpeg()
if ffmpeg_path:
    # Bundle FFmpeg to the root of the package
    binaries.append((ffmpeg_path, '.'))
    print(f"📦 Bundling FFmpeg from: {ffmpeg_path}")

# ========================================
# ANALYSIS
# ========================================

a = Analysis(
    ['captioner_compact.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',
        'notebook',
        'ipython',
        'jupyter',
        'pytest',
        'sphinx',
        'docutils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ========================================
# BUNDLE CREATION
# ========================================

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Common EXE options
exe_options = {
    'name': 'VideoCaptioner',
    'debug': False,
    'bootloader_ignore_signals': False,
    'strip': False,
    'upx': True,
    'upx_exclude': [],
    'runtime_tmpdir': None,
    'console': False,  # GUI app, no console window
    'disable_windowed_traceback': False,
    'argv_emulation': False,
    'target_arch': None,
    'codesign_identity': None,
    'entitlements_file': None,
}

# Windows-specific options
if IS_WINDOWS:
    exe_options['icon'] = None  # Add .ico path if available
    # Add Windows manifest for high DPI support
    exe_options['manifest'] = None
    
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        **exe_options,
    )

# macOS-specific options
elif IS_MACOS:
    exe_options['icon'] = None  # Add .icns path if available
    
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        **exe_options,
    )
    
    # Create macOS app bundle
    app = BUNDLE(
        exe,
        name='VideoCaptioner.app',
        icon=None,  # Add .icns path if available
        bundle_identifier='com.videocaptioner.app',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSBackgroundOnly': 'False',
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
            'NSMicrophoneUsageDescription': 'Video Captioner needs microphone access for audio processing.',
            'LSMinimumSystemVersion': '10.15.0',
        },
    )

# Linux-specific options
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        **exe_options,
    )

print("✅ Build configuration complete!")
