@echo off
setlocal
title Manga Translator Local Server v6.1.0 (Port 5000)
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
if not defined OLLAMA_FLASH_ATTENTION set OLLAMA_FLASH_ATTENTION=1
if not defined OLLAMA_KV_CACHE_TYPE set OLLAMA_KV_CACHE_TYPE=q8_0

echo ===================================================
echo   Starting Manga Translator Local Server v6.1.0...
echo   Miko-chan Web Studio: http://127.0.0.1:5000/
echo   Folder: %CD%
echo ===================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

python "%~dp0server.py" --open-gui
if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error. Check messages above.
)
pause
