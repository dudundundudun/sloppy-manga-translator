@echo off
title Miko-chan Manga Studio v1.0.8 - Server and GUI Launcher
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

echo ==================================================================
echo   Starting Miko-chan Manga Studio v1.0.8...
echo   Step 1: Starting local AI server on http://127.0.0.1:5000/
echo   Step 2: GUI window will open automatically when server is ready!
echo ==================================================================
echo.

python server.py --open-gui
if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error. Check messages above.
    pause
)
