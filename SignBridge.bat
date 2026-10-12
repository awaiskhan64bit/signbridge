@echo off
REM Desktop launcher for SignBridge.
REM
REM Starts the app and, if no camera is attached, falls back to the built in
REM simulated signer so double clicking always shows something working.

setlocal
cd /d "C:\Users\awais\projects\signbridge"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo.
    echo SignBridge is not set up yet.
    echo Run scripts\setup_dev.bat first.
    echo.
    pause
    exit /b 1
)

REM Pick a source: prefer the camera, fall back to demo if none answers.
"%PY%" -c "import cv2,sys; cap=cv2.VideoCapture(0); ok=cap.isOpened(); cap.release(); sys.exit(0 if ok else 1)" 2>nul
if errorlevel 1 (
    "%PY%" -m signbridge --source simulated
) else (
    "%PY%" -m signbridge
)

if errorlevel 1 (
    echo.
    echo SignBridge exited with an error. The message is above.
    pause
)
