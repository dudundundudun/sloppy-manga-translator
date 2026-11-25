@echo off
cd /d "%~dp0"
title Miko-chan Manga Studio (Rust)
echo Starting Miko-chan Manga Studio (Rust + CUDA)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_rust_studio.ps1" -OpenBrowser %*
if errorlevel 1 (
    echo.
    echo [!] Failed to launch Rust Studio.
    pause
)
