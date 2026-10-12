@echo off
REM One command setup on Windows: create the environment, install everything,
REM fetch the MediaPipe model and a Piper voice.
REM
REM This is the only step that touches the network.

setlocal
cd /d "%~dp0"

echo.
echo 1 of 4  creating the virtual environment
python -m venv .venv || goto :fail

echo.
echo 2 of 4  installing dependencies
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -e ".[tts]" --quiet || goto :fail

echo.
echo 3 of 4  fetching the hand model and a voice
".venv\Scripts\python.exe" -m signbridge.setup || goto :warn

echo.
echo 4 of 4  done
echo.
echo Start the app with:   run.bat
echo Try it without a camera:  run.bat --source simulated
echo Train the WLASL model:    scripts\fetch_wlasl.bat then scripts\train.bat
echo.
pause
exit /b 0

:warn
echo.
echo Setup finished with warnings above. The app will still start and will fall
echo back to Windows SAPI for speech. Re-run signbridge.setup to retry.
pause
exit /b 0

:fail
echo.
echo Setup failed. Check that Python 3.10 to 3.12 is on your PATH.
pause
exit /b 1
