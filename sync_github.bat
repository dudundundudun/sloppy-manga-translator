@echo off
title Miko-chan Manga Studio v6.1.0 - GitHub Quick Sync
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

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Not inside a git repository.
    pause
    exit /b 1
)

if exist "EN" del /f /q "EN"
git check-ignore -q config.json
if errorlevel 1 (
    echo [ERROR] config.json is NOT ignored by .gitignore! Aborting.
    pause
    exit /b 1
)
git check-ignore -q gui_settings.json
if errorlevel 1 (
    echo [ERROR] gui_settings.json is NOT ignored by .gitignore! Aborting.
    pause
    exit /b 1
)
git status --short
echo.
set "CONFIRM="
set /p "CONFIRM=Review git status above. Stage tracked changes and push? [y/N]: "
if /i not "%CONFIRM%"=="y" (
    echo [CANCELLED] Aborted by user.
    pause
    exit /b 0
)

git add -u
if errorlevel 1 (
    echo [ERROR] Failed to stage tracked changes.
    pause
    exit /b 1
)
git diff --cached --quiet
if not errorlevel 1 (
    echo [INFO] No staged tracked changes to commit.
) else (
    git commit -m "Update Miko-chan Manga Studio"
    if errorlevel 1 (
        echo [ERROR] git commit failed.
        pause
        exit /b 1
    )
)
git push
if errorlevel 1 (
    echo [ERROR] git push failed.
    pause
    exit /b 1
)
echo.
echo [DONE] Changes pushed to GitHub!
pause
