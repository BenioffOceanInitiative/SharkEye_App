# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Collect all submodules of scipy
scipy_submodules = collect_submodules('scipy')

# Collect data files for scipy
scipy_data_files = collect_data_files('scipy')

# Get scipy path
import scipy
scipy_path = os.path.dirname(scipy.__file__)

# Collect Ultralytics submodules and data files
ultralytics_submodules = collect_submodules('ultralytics')
ultralytics_data_files = collect_data_files('ultralytics')

# Get Ultralytics path
import ultralytics
ultralytics_path = os.path.dirname(ultralytics.__file__)

# Update data files
data_files = [
    ('assets/images', 'assets/images'),
    ('assets/logo', 'assets/logo'),  # Add this line to include logo files
    ('model_weights/runs-detect-train-weights-best.pt', 'model_weights'),
    (scipy_path, 'scipy'),
    (ultralytics_path, 'ultralytics')
]

# Extend data files with scipy data files
data_files.extend(scipy_data_files)
data_files.extend(ultralytics_data_files)

# Define hidden imports
hidden_imports = [
    'ultralytics', 'torch', 'torchvision', 'lapx', 'PyQt6',
    'scipy',
    'scipy._lib.messagestream',
    'scipy._lib.array_api_compat',
    'scipy._lib.array_api_compat.numpy',
    'scipy._lib.array_api_compat.numpy.fft',
]

# Extend hidden imports with scipy submodules
hidden_imports.extend(scipy_submodules)
hidden_imports.extend(ultralytics_submodules)

if sys.platform.startswith('win'):
    icon_file = 'assets/logo/SharkEye.ico'
elif sys.platform.startswith('darwin'):
    icon_file = 'assets/logo/SharkEye.icns'
else:
    icon_file = None
a = Analysis(
    ['src/sharkeye_app.py'],
    pathex=[],
    binaries=[],
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SharkEye',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=icon_file
)

if sys.platform.startswith('darwin'):
    app = BUNDLE(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        name='SharkEye.app',
        icon=icon_file,
        bundle_identifier=None,
        info_plist={
            'NSHighResolutionCapable': 'True',
            'NSPrincipalClass': 'NSApplication',
            'NSAppleScriptEnabled': False,
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
            'LSBackgroundOnly': False,
            'CFBundleDocumentTypes': [],
            'NSCameraUsageDescription': 'This app requires camera access to process video files.',
            'NSPhotoLibraryUsageDescription': 'This app requires access to the photo library to process video files.',
        },
    )
else:
    # For Windows or other platforms
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='SharkEye'
    )
