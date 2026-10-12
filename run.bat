@echo off
REM Launch SignBridge. Double click this, or run it from a terminal.
REM It activates the virtual environment and starts the app in demo mode if no
REM camera is available.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment found. Run scripts\setup_dev.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m signbridge %*
endlocal
