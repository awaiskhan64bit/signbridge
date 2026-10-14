@echo off
REM ===========================================================================
REM  SignBridge one click setup for Windows.
REM
REM  Double click this file. It creates a virtual environment, installs the
REM  dependencies, downloads the models and voice, and starts the app. Run it
REM  again after any update and it will skip whatever is already done.
REM
REM  Needs internet for this script only. After it finishes the app makes no
REM  network calls at all.
REM ===========================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"
title SignBridge setup

set "PY=.venv\Scripts\python.exe"

echo.
echo   SignBridge setup
echo   ================
echo.

REM ---------------------------------------------------------------- Python
where python >nul 2>&1
if errorlevel 1 (
    echo   Python was not found on your PATH.
    echo.
    echo   Install Python 3.12 from python.org and tick "Add python.exe to PATH".
    echo   https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo   Found Python !PYVER!

REM 3.10 to 3.12 only. Anything else is refused up front rather than failing
REM later inside a wheel install with a message that says nothing useful.
python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 (
    echo.
    echo   SignBridge needs Python 3.10 to 3.12. This machine has !PYVER!.
    echo   Install 3.12 from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM ------------------------------------------------------------ environment
if not exist "!PY!" (
    echo.
    echo   Creating the virtual environment. This takes a moment.
    python -m venv .venv
    if errorlevel 1 (
        echo   Could not create the virtual environment.
        pause
        exit /b 1
    )
    "!PY!" -m pip install --upgrade pip --quiet
)

REM ------------------------------------------------------------ dependencies
REM A stamp file records the step so a re-run is quick. Without it every run
REM spends a minute resolving dependencies the user already has.
if not exist ".venv\.deps-ok" (
    echo.
    echo   Installing dependencies. This is the slow part, about 300 MB.
    echo.
    "!PY!" -m pip install --upgrade pip
    "!PY!" -m pip install -e ".[tts]"
    if errorlevel 1 (
        echo.
        echo   Dependency install failed. Scroll up for the reason.
        echo   If it mentions a network or proxy problem, that is the likely cause.
        pause
        exit /b 1
    )
    echo ok> ".venv\.deps-ok"
    echo.
    echo   Dependencies installed.
) else (
    echo   Dependencies already installed, skipping.
)

REM ------------------------------------------------------------------ assets
echo.
echo   Checking models and voice.
"!PY!" -m signbridge.setup
if errorlevel 1 (
    echo.
    echo   The asset download failed. Scroll up for the reason.
    echo   The app will still start, but it cannot see hands without the models.
    echo.
    pause
)

REM ------------------------------------------------------------------- start
REM The path is made absolute before handing it to `start`. `start` resolves a
REM relative path against its own working directory rather than this script's,
REM so passing ".venv\Scripts\python.exe" there fails with a message that points
REM at the wrong thing entirely.
for %%i in ("%PY%") do set "PYABS=%%~fi"

echo.
echo   Starting SignBridge.
echo.
start "SignBridge" "%PYABS%" -m signbridge %*
echo   The app is starting. This window can be closed.
timeout /t 3 >nul
endlocal