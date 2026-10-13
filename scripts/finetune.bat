@echo off
REM Fine tune the model on clips recorded through SignBridge.
REM
REM Record some signs first using the "Teach it your signs" panel in the app,
REM then run this. Minutes, not the hour a full training run takes.

setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment found. Run scripts\setup_dev.bat first.
    pause
    exit /b 1
)

if not exist "assets\models\sign_classifier.onnx" (
    echo No base model found. Run scripts\train.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet 2>nul

echo.
echo Fine tuning on your recorded signs.
echo Add --with-wlasl to keep the full vocabulary from being forgotten.
echo.
".venv\Scripts\python.exe" scripts\finetune.py %*

echo.
echo Restart the app to use the fine tuned model.
pause
