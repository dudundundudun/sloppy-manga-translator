@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Miko-chan Manga Studio - Portable Release Builder
echo ==================================================================
echo   Сборка портативных версий Miko-chan Manga Studio (Lite + Full)
echo ==================================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\package_release.ps1" -Mode All
if errorlevel 1 (
    echo.
    echo [!] Ошибка при сборке релиза.
    pause
    exit /b 1
)
echo.
pause
