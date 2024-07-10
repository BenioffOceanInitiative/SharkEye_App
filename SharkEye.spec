# -*- mode: python ; coding: utf-8 -*-

import os
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
    ('assets/images/logo-white.png', 'assets/images'),
    ('model_weights/train6-weights-best.pt', 'model_weights'),
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

a = Analysis(
    ['src/main.py'],
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
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SharkEye',
)

app = BUNDLE(
    coll,
    name='SharkEye.app',
    icon=None,
    bundle_identifier=None,
)