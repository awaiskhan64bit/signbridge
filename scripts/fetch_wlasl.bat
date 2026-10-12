@echo off
REM Download the WLASL hand landmark corpus, about 430 MB. This is the only
REM large download in the project and it is not needed to run the app.
REM
REM It is resumable: re-running skips whatever is already on disk.

setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment found. Run scripts\setup_dev.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\fetch_wlasl.py
echo.
echo Now train the model with:  scripts\train.bat
pause
