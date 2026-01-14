# -*- mode: python ; coding: utf-8 -*-

import os, sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
    get_package_paths,
)

block_cipher = None

# ---------- paths ----------
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

SRC   = HERE / "src"
ENTRY = SRC / "sharkeye_app.py"

# ---------- hidden imports ----------
hidden_imports  = []
hidden_imports += collect_submodules("PyQt6.QtCore")
hidden_imports += collect_submodules("PyQt6.QtGui")
hidden_imports += collect_submodules("PyQt6.QtWidgets")
hidden_imports += collect_submodules("PyQt6.QtSvgWidgets")
hidden_imports += collect_submodules("ultralytics")
hidden_imports += collect_submodules("segment_anything")
hidden_imports += collect_submodules("scipy")
hidden_imports += collect_submodules("tensorboard")

hidden_imports += [
    "torch.cuda",
    "torch.cuda.amp",
    "torch.backends.cudnn",
    "torch.backends.cuda",
    "torch._C",
]

# ---------- data files ----------
datas  = []
datas += collect_data_files("PyQt6.QtCore")
datas += collect_data_files("PyQt6.QtGui")
datas += collect_data_files("PyQt6.QtWidgets")
datas += collect_data_files("PyQt6.QtSvgWidgets")
datas += collect_data_files("ultralytics")
datas += collect_data_files("certifi")
datas += collect_data_files("tensorboard")

datas += [
    (str(HERE / "assets" / "images"), "assets/images"),
    (str(HERE / "assets" / "logo"), "assets/logo"),
    (str(HERE / "model_weights" / "runs-detect-train-weights-best.pt"), "model_weights"),
    (str(HERE / "model_weights" / "sam_vit_b_01ec64.pth"), "model_weights"),
]

# ---------- native binaries ----------
binaries  = []
binaries += collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("numpy")
binaries += collect_dynamic_libs("scipy")
binaries += collect_dynamic_libs("torch")

torch_base, _ = get_package_paths("torch")
for dll in Path(torch_base).rglob("nvrtc64_*.dll"):
    binaries.append((str(dll), "."))

# ---------- icon ----------
if sys.platform.startswith("win"):
    icon_file = str(HERE / "assets" / "logo" / "SharkEye.ico")
elif sys.platform.startswith("darwin"):
    icon_file = str(HERE / "assets" / "logo" / "SharkEye.icns")
else:
    icon_file = None

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[str(HERE)],
    runtime_hooks=[],
    excludes=[
        "PyQt6.QtBluetooth",
        "PyQt6.QtNfc",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SharkEye",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=icon_file,
)

if sys.platform.startswith("darwin"):
    app = BUNDLE(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        name="SharkEye.app",
        icon=icon_file,
        bundle_identifier="com.sharkeye.app",
        info_plist={
            "NSHighResolutionCapable": "True",
            "NSPrincipalClass": "NSApplication",
            "LSBackgroundOnly": False,
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1.0.0",
        },
    )
else:
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        name="SharkEye",
    )
