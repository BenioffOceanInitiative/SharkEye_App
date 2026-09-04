# -*- mode: python ; coding: utf-8 -*-

import os, sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata
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

# ---------- build version stamp ----------
# Write a version.json bundled at the app root so the installed app can identify
# itself (used for update checks against sharkeye-app-build/latest_version.json).
import json, platform, subprocess


def _detect_os_key():
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("darwin"):
        return "macos_silicon" if platform.machine().lower() in ("arm64", "aarch64") else "macos_intel"
    return sys.platform


def _git(args, default=""):
    try:
        return subprocess.run(
            ["git", *args], cwd=str(HERE),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return default


_commit = os.environ.get("GITHUB_SHA") or _git(["rev-parse", "HEAD"])
_committed_at = _git(["show", "-s", "--format=%cI", _commit]) if _commit else ""
_version_info = {
    "os": _detect_os_key(),
    "commit": _commit,
    "committed_at": _committed_at,
}
_version_path = HERE / "version.json"
_version_path.write_text(json.dumps(_version_info, indent=2) + "\n", encoding="utf-8")
print("SharkEye build version.json:", _version_info)

# ---------- hidden imports ----------
hidden_imports = []
hidden_imports += collect_submodules("PyQt6.QtCore")
hidden_imports += collect_submodules("PyQt6.QtGui")
hidden_imports += collect_submodules("PyQt6.QtWidgets")
hidden_imports += collect_submodules("PyQt6.QtSvgWidgets")
hidden_imports += collect_submodules("ultralytics")
hidden_imports += collect_submodules("segment_anything")
hidden_imports += collect_submodules("scipy")
hidden_imports += collect_submodules("av")           # PyAV: keyframe-scan decode path
# If you truly use these, uncomment; otherwise leave out to avoid build errors
# hidden_imports += ["lapx", "dask"]

# ---------- data files ----------
datas  = []
datas += collect_data_files("PyQt6.QtCore")
datas += collect_data_files("PyQt6.QtGui")
datas += collect_data_files("PyQt6.QtWidgets") 
datas += collect_data_files("PyQt6.QtSvgWidgets")                # Qt plugins/resources
datas += collect_data_files("ultralytics")
# torch ships C++ headers (include/) and a test suite that are useless at runtime;
# keep only the real runtime data files to trim ~tens of MB.
datas += [t for t in collect_data_files("torch")
          if "/include/" not in t[1].replace("\\", "/")
          and "/test/" not in t[1].replace("\\", "/")]
datas += collect_data_files("certifi")               # CA bundle for requests
datas += copy_metadata("imageio")

# your app assets + model weights
datas += [
    (str(HERE / "assets" / "images"),       "assets/images"),
    (str(HERE / "assets" / "logo"),         "assets/logo"),
    (str(HERE / "model_weights" / "runs-detect-train-weights-best.pt"), "model_weights"),
    (str(HERE / "model_weights" / "sam_vit_b_01ec64.pth"),              "model_weights"),
    (str(HERE / "docs"), "docs"),
    (str(_version_path), "."),                        # build identifier: os/commit/committed_at
]
_example_footage = HERE / "sample_data" / "example_footage.mp4"
if _example_footage.is_file():
    datas += [(str(_example_footage), "sample_data")]
# Legacy path kept for older layouts
_example_footage_data = HERE / "data" / "example_footage.mp4"
if _example_footage_data.is_file():
    datas += [(str(_example_footage_data), "data")]

# ---------- native binaries (DLLs/SOs) ----------
binaries  = []
binaries += collect_dynamic_libs("cv2")              # includes ffmpeg dlls
binaries += collect_dynamic_libs("numpy")
binaries += collect_dynamic_libs("scipy")
binaries += collect_dynamic_libs("torch")
# PyAV ships its own libav (libavcodec/format/util/...) dylibs. cv2 also bundles a
# libav build, so BOTH end up in the app — this is the dual-libav coexistence that
# prints the "AVFFrameReceiver implemented in both" objc warning at startup. It runs,
# but is the thing to verify in the built bundle before making keyframe sampling default.
binaries += collect_dynamic_libs("av")

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
    excludes=[
        # Confirmed unused in src/ — trims dead weight from the bundle.
        'tensorboard',
        # polars (~182 MB Rust runtime) is an ultralytics dep but is not used by our
        # inference-only path; verified YOLO detection + box access work without it.
        'polars', '_polars_runtime_32',
        # NOTE: torchvision is NOT excluded — ultralytics imports it at inference time
        # (autobackend.warmup + torchvision NMS in nms.py), so it must stay bundled.
        'dask',
        # Qt modules the app never touches (it only uses QtCore/QtGui/QtWidgets/QtSvg/QtSvgWidgets)
        'PyQt6.QtBluetooth', 'PyQt6.QtNfc',
        'PyQt6.QtQml', 'PyQt6.QtQuick', 'PyQt6.QtQuick3D',
        'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineWidgets',
        'PyQt6.QtMultimedia', 'PyQt6.QtMultimediaWidgets',
        'PyQt6.Qt3DCore', 'PyQt6.QtCharts', 'PyQt6.QtDataVisualization',
        # NOTE: QtDesigner is NOT excluded — PyQt6_SwitchControl imports it at load time.
        'PyQt6.QtTest', 'PyQt6.QtSql',
        'PyQt6.QtPositioning', 'PyQt6.QtSensors', 'PyQt6.QtSerialPort',
        'PyQt6.QtWebSockets', 'PyQt6.QtRemoteObjects',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# MinGW strip corrupts CUDA PE DLLs on Windows (WinError 998 loading nvrtc-*.dll).
# Keep strip for macOS dylib size; disable it on Windows.
_strip = not sys.platform.startswith("win")

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SharkEye",
    debug=False,                      # set True while diagnosing
    bootloader_ignore_signals=False,
    strip=_strip,                     # strip symbol tables from bundled dylibs (torch/Qt/scipy)
    upx=False,                        # <-- DO NOT UPX torch/Qt/scipy dlls
    console=False,                    # release build: no debug terminal window
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
        strip=_strip,                  # False on Windows — mingw strip corrupts CUDA PE DLLs
        upx=False,                     # keep UPX off here too
        upx_exclude=[],
        name="SharkEye",
    )