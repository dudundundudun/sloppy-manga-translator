@echo off
title Miko-chan Manga Studio - GitHub Quick Sync
cd /d "%~dp0"

echo ==================================================================
echo   Miko-chan Manga Studio - GitHub Quick Backup
echo ==================================================================
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo [INFO] Git CLI is not in PATH. Please use GitHub Desktop app to Commit and Push.
    pause
    exit /b 1
)

git status
echo.
git add .
git commit -m "Update Miko-chan Manga Studio"
git push
echo.
echo [DONE] Changes pushed to GitHub!
pause
