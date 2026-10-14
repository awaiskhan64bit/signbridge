@echo off
REM ===========================================================================
REM  SignBridge one click setup for Windows.
REM
REM  Double click this file. It will:
REM
REM    * install Python 3.12 if this machine has none
REM    * create a virtual environment and install every dependency
REM    * download the models and the voice
REM    * put a shortcut on your Desktop
REM    * start the app
REM
REM  Run it again after any update and it skips whatever is already done.
REM
REM  Needs internet for this script only. After it finishes the app makes no
REM  network calls at all.
REM
REM  Pass --no-python to refuse to install Python and only use what is already
REM  there, and --shortcut-only to just (re)create the Desktop shortcut.
REM ===========================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"
title SignBridge setup

set "PY=.venv\Scripts\python.exe"
set "WANT_PYTHON=1"
set "SHORTCUT_ONLY=0"

:parse
if "%~1"=="" goto afterparse
if /i "%~1"=="--no-python" (set "WANT_PYTHON=0"& shift& goto parse)
if /i "%~1"=="--shortcut-only" (set "SHORTCUT_ONLY=1"& shift& goto parse)
shift
goto parse
:afterparse

echo.
echo   SignBridge setup
echo   ================
echo.

REM ---------------------------------------------------------------- shortcut
if "%SHORTCUT_ONLY%"=="1" goto shortcut_only

REM ------------------------------------------------------------------- Python
REM Found by asking PATH first, then the places the installer actually puts
REM it. The second check matters because a machine that installed Python with
REM "just for me" has it on PATH only for that user, and a freshly installed one
REM will not be on PATH until the shell is restarted, which it cannot be from
REM inside this script.
set "PYEXE="
for %%p in (python.exe) do (
    for /f "delims=" %%i in ('where %%p 2^>nul') do if not defined PYEXE set "PYEXE=%%i"
)
if not defined PYEXE (
    for %%d in (
        "%LOCALAPPDATA%\Programs\Python\Python312"
        "%LOCALAPPDATA%\Programs\Python\Python313"
        "%ProgramFiles%\Python312"
        "%ProgramFiles%\Python313"
    ) do (
        if exist "%%~d\python.exe" if not defined PYEXE set "PYEXE=%%~d\python.exe"
    )
)

if defined PYEXE goto python_found

echo   Python was not found on this machine.
echo.
if "%WANT_PYTHON%"=="0" (
    echo   You asked not to install it. Install Python 3.12 from
    echo   https://www.python.org/downloads/ and run this again.
    echo   Tick "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

echo   Installing Python 3.12. This changes the machine, not just this app.
echo.
where winget >nul 2>&1
if errorlevel 1 (
    echo   winget is not available, so it cannot install Python automatically.
    echo   Install Python 3.12 from https://www.python.org/downloads/
    echo   and tick "Add python.exe to PATH", then run this again.
    echo.
    pause
    exit /b 1
)

echo   Running: winget install Python.Python.3.12
echo.
winget install --id Python.Python.3.12 -e --source winget ^
    --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo   The Python install did not succeed. Install it by hand from
    echo   https://www.python.org/downloads/ and run this again.
    echo.
    pause
    exit /b 1
)

REM The installer does not update the PATH of a shell that is already running,
REM so find it where it was just put.
for %%d in (
    "%LOCALAPPDATA%\Programs\Python\Python312"
    "%ProgramFiles%\Python312"
) do (
    if exist "%%~d\python.exe" if not defined PYEXE set "PYEXE=%%~d\python.exe"
)
if not defined PYEXE (
    echo.
    echo   Python installed but could not be located. Open a new terminal and
    echo   run this file again.
    echo.
    pause
    exit /b 1
)

:python_found
for /f "tokens=2" %%v in ('"%PYEXE%" --version 2^>^&1') do set "PYVER=%%v"
echo   Found Python !PYVER!

REM 3.10 to 3.12 only. Refused here rather than failing later inside a wheel
REM install with a message that says nothing useful.
"%PYEXE%" -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)"
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
    echo   Creating the virtual environment.
    "%PYEXE%" -m venv .venv
    if errorlevel 1 (
        echo   Could not create the virtual environment.
        pause
        exit /b 1
    )
    "!PY!" -m pip install --upgrade pip --quiet
)

REM ------------------------------------------------------------ dependencies
REM A stamp records the step so a re-run does not spend a minute resolving
REM dependencies that are already installed.
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

:shortcut_only
REM --------------------------------------------------------------- shortcut
for %%i in ("%PY%") do set "PYABS=%%~fi"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\make_shortcut.ps1" ^
    -Python "%PYABS%" -WorkDir "%CD%"
if errorlevel 1 (
    echo   Could not create the Desktop shortcut. The app still works.
) else (
    echo   Desktop shortcut ready.
)

REM ------------------------------------------------------------------- start
REM An absolute path, because start resolves a relative one against its own
REM working directory rather than this script's.
if "%SHORTCUT_ONLY%"=="1" goto done

echo.
echo   Starting SignBridge.
echo.
start "SignBridge" "%PYABS%" -m signbridge %*
echo   The app is starting. This window can be closed.

:done
timeout /t 3 >nul
endlocal