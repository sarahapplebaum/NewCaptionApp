from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, get_module_file_attribute
import os

# Don't import torch - just specify what to include
datas = []
binaries = []

# Try to find torch directory without importing
try:
    torch_dir = os.path.dirname(get_module_file_attribute('torch'))
    
    # Add torch lib directory
    lib_dir = os.path.join(torch_dir, 'lib')
    if os.path.exists(lib_dir):
        for file in os.listdir(lib_dir):
            if file.endswith('.dll'):
                binaries.append((os.path.join(lib_dir, file), 'torch/lib'))
except:
    pass

hiddenimports = ['torch', 'torch.nn', 'torch.nn.functional', '_torch']
