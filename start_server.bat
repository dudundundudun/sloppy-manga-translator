@echo off
title Manga Translator Local Server v1.0.0 (Port 5000)
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

echo ===================================================
echo   Starting Manga Translator Local Server v1.0.0...
echo   Miko-chan Web Studio: http://127.0.0.1:5000/
echo   Folder: %CD%
echo ===================================================

python server.py --open-gui
pause
