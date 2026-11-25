@echo off
setlocal
title Miko-chan Manga Studio v6.1.0 - Server and GUI Launcher
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
if not defined OLLAMA_FLASH_ATTENTION set OLLAMA_FLASH_ATTENTION=1
if not defined OLLAMA_KV_CACHE_TYPE set OLLAMA_KV_CACHE_TYPE=q8_0

echo ==================================================================
echo   Starting Miko-chan Manga Studio v6.1.0...
echo   Step 1: Starting local AI server on http://127.0.0.1:5000/
echo   Step 2: GUI window will open automatically when server is ready!
echo ==================================================================
echo.

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
    pause
)
