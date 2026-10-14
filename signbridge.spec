# PyInstaller spec for a standalone Windows build.
#
# Run it with:  pyinstaller --noconfirm signbridge.spec
# Output:       dist/SignBridge/SignBridge.exe
#
# This produces a folder, not a single file, on purpose. PySide6, MediaPipe and
# ONNX Runtime together are well over a gigabyte of shared libraries, and a
# one-file build has to unpack all of that into a temporary directory on every
# launch, which costs several seconds before the window appears. Shipping a
# folder and zipping it is faster to start and easier to reason about.
#
# The models and the voice are NOT bundled. They are downloaded once by
# `python -m signbridge.setup` into the user's own app data, so the build stays
# small and the assets can be updated without rebuilding the executable.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH)

datas = [("src/signbridge/ui/qml", "signbridge/ui/qml")]
hiddenimports = []

# MediaPipe and onnxruntime both ship data files and lazily imported modules
# that PyInstaller's static analysis misses. Without these the build succeeds
# and then fails at runtime on a module that was never collected.
for pkg in ("mediapipe", "onnxruntime"):
    found, binaries, _ = collect_all(pkg)
    datas += found
    hiddenimports += collect_submodules(pkg) + binaries

hiddenimports += collect_submodules("signbridge")
hiddenimports += ["piper", "sounddevice"]

analysis = Analysis(  # noqa: F821
    ["src/signbridge/__main__.py"],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=[
        # Training only. Pulling torch into a runtime build roughly doubles it
        # and nothing in the app imports it outside the training scripts.
        "torch",
        "torchvision",
        "matplotlib",
        "pandas",
        "IPython",
        "notebook",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SignBridge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(  # noqa: F821
    exe,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    strip=False,
    upx=False,
    name="SignBridge",
)