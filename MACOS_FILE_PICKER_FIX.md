# macOS File Picker Fix for VideoCaptioner

## Problem Description
When running VideoCaptioner as a standalone .app on macOS, users were unable to select video files in the file picker - all files appeared greyed out. This issue did not occur when running the app from VS Code.

## Root Cause
The issue was caused by missing file type associations in the macOS app bundle's Info.plist. macOS uses these associations to determine which file types an application can open. Without proper declarations, the system prevents file selection for security reasons.

## Solution Components

### 1. Fixed PyInstaller Spec File (`captioner_compact_fixed.spec`)
- Added comprehensive `CFBundleDocumentTypes` declarations for video and audio files
- Included `UTImportedTypeDeclarations` for non-standard formats (MKV, WebM)
- Specified all supported file extensions explicitly
- Added support for dark mode and high-resolution displays

### 2. Entitlements File (`entitlements.plist`)
- Added file access permissions for user-selected locations
- Included permissions for common folders (Downloads, Desktop, Documents, Movies)
- Added necessary permissions for PyTorch and JIT compilation
- Disabled library validation for third-party dependencies

### 3. Updated Build Script (`build_app_fixed.py`)
- Uses the fixed spec file instead of the original
- Applies additional permission fixes after building
- Clears extended attributes that might cause issues
- Creates testing documentation

## How to Build with the Fix

1. Ensure all files are in place:
   - `captioner_compact.py` (your main script)
   - `captioner_compact_fixed.spec` (fixed spec file)
   - `entitlements.plist` (permissions file)
   - `build_app_fixed.py` (build script)

2. Run the build script:
   ```bash
   python build_app_fixed.py
   ```

3. The fixed app will be created in the `dist` directory

## Testing the Fix

1. Open the built app from `dist/VideoCaptioner.app`
2. Click "Select Files"
3. Verify that video files (MP4, MOV, etc.) are selectable (not greyed out)
4. Select and process a video file

## Supported File Types

### Video Files
- MP4, MOV, AVI, MKV, WEBM, MPG, MPEG, WMV, M4V, FLV, 3GP

### Audio Files
- MP3, WAV, M4A, FLAC, AAC, OGG, WMA

## Troubleshooting

If files are still greyed out after building:

1. Clear quarantine attributes:
   ```bash
   xattr -cr /path/to/VideoCaptioner.app
   ```

2. Make the app executable:
   ```bash
   chmod +x /path/to/VideoCaptioner.app/Contents/MacOS/VideoCaptioner
   ```

3. If you're on macOS 10.15 or later, you may need to allow the app in System Preferences > Security & Privacy

## Technical Details

The fix works by:
1. Declaring supported file types using Apple's Uniform Type Identifiers (UTI)
2. Mapping file extensions to these UTIs
3. Providing proper entitlements for file system access
4. Ensuring the app bundle has correct permissions

## Future Improvements

To make the app even more robust, consider:
1. Code signing with a developer certificate
2. Notarization for easier distribution
3. Creating a DMG installer
4. Adding an app icon
