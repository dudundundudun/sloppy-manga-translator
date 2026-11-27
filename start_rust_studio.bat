@echo off
cd /d "%~dp0"
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
title Miko-chan Manga Studio (Rust)
echo Starting Miko-chan Manga Studio (Rust + CUDA)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_rust_studio.ps1" -OpenBrowser %*
if errorlevel 1 (
    echo.
    echo [!] Failed to launch Rust Studio.
    pause
)
