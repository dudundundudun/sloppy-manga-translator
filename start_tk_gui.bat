@echo off
setlocal
title Miko-chan Manga Studio v6.1.0 - Tkinter Desktop Window
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

echo ==================================================================
echo   Starting Miko-chan Tkinter Desktop GUI v6.1.0...
echo ==================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

python "%~dp0gui_app.py"
pause
