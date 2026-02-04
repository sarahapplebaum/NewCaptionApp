"""
PyInstaller Runtime Hook for PyTorch on Windows
Fixes DLL loading issues that cause WinError 1114 and silent crashes

This hook runs before the main application starts and configures
the DLL search paths to ensure PyTorch and CUDA libraries load correctly.
"""

import os
import sys
import logging

# Set up early logging
logging.basicConfig(
    level=logging.INFO,
    format='[RUNTIME HOOK] %(message)s'
)

def configure_dll_paths():
    """Configure DLL search paths for PyTorch on Windows"""
    if sys.platform != 'win32':
        return
    
    logging.info("Configuring Windows DLL paths for PyTorch...")
    
    # Determine base path based on PyInstaller mode
    base_path = None
    
    # Check if running in PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Get the directory containing the executable
        exe_dir = os.path.dirname(sys.executable)
        
        # Check for one-folder mode (_internal folder)
        internal_dir = os.path.join(exe_dir, '_internal')
        if os.path.exists(internal_dir):
            base_path = internal_dir
            logging.info(f"One-folder mode detected: {internal_dir}")
        # Check for one-file mode (sys._MEIPASS)
        elif hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
            logging.info(f"One-file mode detected: {base_path}")
        else:
            logging.warning("Frozen app but no _internal or _MEIPASS found")
            return
    else:
        logging.info("Not running in PyInstaller bundle, skipping DLL configuration")
        return
    
    logging.info(f"Base path for DLLs: {base_path}")
    
    # Critical DLL directories that need to be in search path
    dll_directories = [
        base_path,  # Root _internal directory
        os.path.join(base_path, 'torch', 'lib'),
        os.path.join(base_path, 'torch', 'bin'),
        os.path.join(base_path, 'numpy', '.libs'),
        os.path.join(base_path, 'numpy.libs'),
        os.path.join(base_path, 'ctranslate2'),
        os.path.join(base_path, 'ctranslate2', 'lib'),
    ]
    
    # Add DLL directories to search path (Python 3.8+)
    added_dirs = []
    if hasattr(os, 'add_dll_directory'):
        for dll_dir in dll_directories:
            if os.path.exists(dll_dir):
                try:
                    os.add_dll_directory(dll_dir)
                    added_dirs.append(dll_dir)
                    logging.info(f"  ✓ Added DLL directory: {dll_dir}")
                except Exception as e:
                    logging.warning(f"  ✗ Failed to add {dll_dir}: {e}")
            else:
                logging.debug(f"  - Directory not found: {dll_dir}")
    
    # Also update PATH environment variable (fallback for older methods)
    existing_path = os.environ.get('PATH', '')
    new_path_components = [d for d in dll_directories if os.path.exists(d)]
    
    if new_path_components:
        new_path = os.pathsep.join(new_path_components + [existing_path])
        os.environ['PATH'] = new_path
        logging.info(f"  ✓ Updated PATH with {len(new_path_components)} directories")
    
    # Set additional environment variables for library loading
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Allow duplicate OpenMP libraries
    
    logging.info("DLL path configuration complete")
    
    # List critical DLLs for debugging
    critical_dlls = [
        'c10.dll',
        'torch_cpu.dll',
        'torch_python.dll',
        'libiomp5md.dll',
        'ctranslate2.dll',
    ]
    
    logging.info("Checking for critical DLLs:")
    for dll_name in critical_dlls:
        found = False
        for dll_dir in dll_directories:
            dll_path = os.path.join(dll_dir, dll_name)
            if os.path.exists(dll_path):
                logging.info(f"  ✓ Found {dll_name} in {os.path.basename(dll_dir)}/")
                found = True
                break
        if not found:
            logging.warning(f"  ✗ Missing {dll_name}")

# Execute the configuration
try:
    configure_dll_paths()
except Exception as e:
    logging.error(f"Runtime hook failed: {e}")
    import traceback
    traceback.print_exc()
