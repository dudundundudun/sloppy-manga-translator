@echo off
title Miko-chan Manga Studio v1.0.0 - Tkinter Desktop Window
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

echo ==================================================================
echo   Starting Miko-chan Tkinter Desktop GUI v1.0.0...
echo ==================================================================
echo.

python gui_app.py
pause
