# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Video Captioner

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# Collect all dependencies for complex packages
datas = []
binaries = []
hiddenimports = []

# Collect faster-whisper dependencies
faster_whisper_datas, faster_whisper_binaries, faster_whisper_hiddenimports = collect_all('faster_whisper')
datas += faster_whisper_datas
binaries += faster_whisper_binaries
hiddenimports += faster_whisper_hiddenimports

# Collect ctranslate2 dependencies (required by faster-whisper)
ctranslate2_datas, ctranslate2_binaries, ctranslate2_hiddenimports = collect_all('ctranslate2')
datas += ctranslate2_datas
binaries += ctranslate2_binaries
hiddenimports += ctranslate2_hiddenimports

# Collect torch dependencies
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
datas += torch_datas
binaries += torch_binaries
hiddenimports += torch_hiddenimports

# Collect librosa dependencies
librosa_datas, librosa_binaries, librosa_hiddenimports = collect_all('librosa')
datas += librosa_datas
binaries += librosa_binaries
hiddenimports += librosa_hiddenimports

# Add additional hidden imports
hiddenimports += [
    'sklearn.utils._typedefs',
    'sklearn.neighbors._partition_nodes',
    'numba',
    'audioread',
    'soundfile',
    'tokenizers',
    'huggingface_hub',
    'onnxruntime',
]

# Check if FFmpeg is available in the system
ffmpeg_path = None
for path in ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/usr/bin/ffmpeg']:
    if os.path.exists(path):
        ffmpeg_path = path
        break

if not ffmpeg_path:
    import shutil
    ffmpeg_path = shutil.which('ffmpeg')

# Add FFmpeg binary if found
if ffmpeg_path:
    binaries.append((ffmpeg_path, '.'))
    print(f"Including FFmpeg from: {ffmpeg_path}")
else:
    print("WARNING: FFmpeg not found! The app will require FFmpeg to be installed on the target system.")

a = Analysis(
    ['captioner_compact.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'notebook', 'ipython', 'jupyter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VideoCaptioner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want console output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file='entitlements.plist' if sys.platform == 'darwin' else None,
    icon=None,  # Add icon path here if you have one
)

# For macOS, create an app bundle with proper file type associations
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='VideoCaptioner.app',
        icon=None,  # Add icon path here if you have one
        bundle_identifier='com.videocaptioner.app',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSBackgroundOnly': 'False',
            'NSRequiresAquaSystemAppearance': 'False',  # Support dark mode
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
            'NSHumanReadableCopyright': 'Copyright © 2024',
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'Video Files',
                    'CFBundleTypeRole': 'Viewer',
                    'LSHandlerRank': 'Default',
                    'LSItemContentTypes': [
                        'public.movie',
                        'public.mpeg-4',
                        'public.avi',
                        'com.apple.quicktime-movie',
                        'public.3gpp',
                        'public.mpeg',
                        'com.microsoft.windows-media-wmv',
                        'public.mp4',
                    ],
                    'CFBundleTypeExtensions': [
                        'mp4', 'MP4',
                        'mov', 'MOV',
                        'avi', 'AVI',
                        'mkv', 'MKV',
                        'webm', 'WEBM',
                        'mpg', 'MPG',
                        'mpeg', 'MPEG',
                        'wmv', 'WMV',
                        'm4v', 'M4V',
                        'flv', 'FLV',
                        '3gp', '3GP',
                    ],
                },
                {
                    'CFBundleTypeName': 'Audio Files',
                    'CFBundleTypeRole': 'Viewer',
                    'LSHandlerRank': 'Default',
                    'LSItemContentTypes': [
                        'public.audio',
                        'public.mp3',
                        'public.mpeg-4-audio',
                        'com.apple.m4a-audio',
                        'com.microsoft.waveform-audio',
                        'org.xiph.flac',
                    ],
                    'CFBundleTypeExtensions': [
                        'mp3', 'MP3',
                        'wav', 'WAV',
                        'm4a', 'M4A',
                        'flac', 'FLAC',
                        'aac', 'AAC',
                        'ogg', 'OGG',
                        'wma', 'WMA',
                    ],
                },
            ],
            'UTImportedTypeDeclarations': [
                {
                    'UTTypeIdentifier': 'public.mkv',
                    'UTTypeReferenceURL': 'https://www.matroska.org/',
                    'UTTypeDescription': 'Matroska Video',
                    'UTTypeIconFile': '',
                    'UTTypeConformsTo': ['public.movie', 'public.audiovisual-content'],
                    'UTTypeTagSpecification': {
                        'public.filename-extension': ['mkv', 'MKV'],
                        'public.mime-type': ['video/x-matroska'],
                    },
                },
                {
                    'UTTypeIdentifier': 'public.webm',
                    'UTTypeReferenceURL': 'https://www.webmproject.org/',
                    'UTTypeDescription': 'WebM Video',
                    'UTTypeIconFile': '',
                    'UTTypeConformsTo': ['public.movie', 'public.audiovisual-content'],
                    'UTTypeTagSpecification': {
                        'public.filename-extension': ['webm', 'WEBM'],
                        'public.mime-type': ['video/webm'],
                    },
                },
            ],
        },
    )
