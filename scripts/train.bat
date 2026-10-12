@echo off
REM Train the WLASL sign classifier. CPU only, no GPU required.
REM Expect roughly ten to forty minutes depending on the machine.

setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment found. Run scripts\setup_dev.bat first.
    pause
    exit /b 1
)

if not exist "data\wlasl-index.json" (
    echo No dataset found. Run scripts\fetch_wlasl.bat first.
    pause
    exit /b 1
)

echo.
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
".venv\Scripts\python.exe" scripts\train.py %*

echo.
echo The trained model is in assets\models. Restart the app to pick it up.
pause
