# -*- mode: python ; coding: utf-8 -*-

import os, sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
try:
    # PyInstaller >= 6.3
    from PyInstaller.utils.hooks import collect_dynamic_libs
except Exception:
    # Fallback name in some versions
    from PyInstaller.utils.hooks import collect_binaries as collect_dynamic_libs

block_cipher = None

# ---------- robust paths ----------
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
SRC = HERE / "src"                                   # your source dir
ENTRY = SRC / "sharkeye_app.py"                      # your main script

# ---------- hidden imports ----------
hidden_imports = []
hidden_imports += collect_submodules("PyQt6")
hidden_imports += collect_submodules("ultralytics")
hidden_imports += collect_submodules("segment_anything")
hidden_imports += collect_submodules("scipy")
# If you truly use these, uncomment; otherwise leave out to avoid build errors
# hidden_imports += ["lapx", "dask"]

# ---------- data files ----------
datas  = []
datas += collect_data_files("PyQt6")                 # Qt plugins/resources
datas += collect_data_files("ultralytics")
datas += collect_data_files("torch")
datas += collect_data_files("certifi")               # CA bundle for requests

# your app assets + model weights
datas += [
    (str(HERE / "assets" / "images"),       "assets/images"),
    (str(HERE / "assets" / "logo"),         "assets/logo"),
    (str(HERE / "model_weights" / "runs-detect-train-weights-best.pt"), "model_weights"),
    (str(HERE / "model_weights" / "sam_vit_b_01ec64.pth"),              "model_weights"),
]

# ---------- native binaries (DLLs/SOs) ----------
binaries  = []
binaries += collect_dynamic_libs("cv2")              # includes ffmpeg dlls
binaries += collect_dynamic_libs("numpy")
binaries += collect_dynamic_libs("scipy")
binaries += collect_dynamic_libs("torch")

# ---------- icon ----------
if sys.platform.startswith("win"):
    icon_file = str(HERE / "assets" / "logo" / "SharkEye.ico")
elif sys.platform.startswith("darwin"):
    icon_file = str(HERE / "assets" / "logo" / "SharkEye.icns")
else:
    icon_file = None

a = Analysis(
    [str(ENTRY)],
    pathex=[str(SRC)],                # make 'src' importable (utility, segmentation, etc.)
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[str(HERE)],
    hooksconfig={},
    runtime_hooks=[],                 # you can add a runtime hook file here if needed
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SharkEye",
    debug=False,                      # set True while diagnosing
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                        # <-- DO NOT UPX torch/Qt/scipy dlls
    console=True,                     # <-- True for debugging; set False for release
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
            "NSAppleScriptEnabled": False,
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1.0.0",
            "LSBackgroundOnly": False,
            "CFBundleDocumentTypes": [],
            "NSCameraUsageDescription": "This app requires camera access to process video files.",
            "NSPhotoLibraryUsageDescription": "This app requires access to the photo library to process video files.",
            "CFBundleIconFile": "SharkEye.icns",
            "CFBundleIconName": "SharkEye",
        },
    )
else:
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,                     # keep UPX off here too
        upx_exclude=[],
        name="SharkEye",
    )
