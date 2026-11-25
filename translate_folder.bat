@echo off
setlocal
title Local Manga Translator v6.1.0 - Batch Folder Translation (Auto-Resume)
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

set "FOLDER_ARG=%~1"
set "TARGET_LANG=en"
echo ================================================================
echo   Batch Manga Folder Translator v6.1.0 (with Auto-Resume)
echo   Press [Enter] for English (_EN), or type: ru (_RU) / romaji (_ROMAJI)
echo ================================================================
set "USER_LANG="
set /p "USER_LANG=Target language [en/ru/romaji] (default: en): "
if defined USER_LANG set "USER_LANG=%USER_LANG:"=%"
if /i "%USER_LANG%"=="en" set "TARGET_LANG=en"
if /i "%USER_LANG%"=="ru" set "TARGET_LANG=ru"
if /i "%USER_LANG%"=="romaji" set "TARGET_LANG=romaji"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)
if not exist "%~dp0cli_client.py" (
    echo [ERROR] cli_client.py not found in "%~dp0".
    pause
    exit /b 1
)

echo.
if defined FOLDER_ARG (
    echo Starting folder translation for "%FOLDER_ARG%"...
    python "%~dp0cli_client.py" folder "%TARGET_LANG%" "%FOLDER_ARG%"
) else (
    echo Opening Windows folder selection dialog...
    python "%~dp0cli_client.py" folder "%TARGET_LANG%"
)
if errorlevel 1 (
    echo.
    echo [ERROR] Folder translation failed. Make sure the local server is running.
    pause
    exit /b 1
)

echo.
pause
