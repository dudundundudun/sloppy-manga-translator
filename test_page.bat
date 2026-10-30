@echo off
title Local Manga Translator v1.0.8 - Single Page Translation
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

set "IMG_FILE=%~1"
if "%IMG_FILE%"=="" set "IMG_FILE=test.jpg"

if not exist "%IMG_FILE%" (
    echo [ERROR] File "%IMG_FILE%" not found in %CD%
    echo Put test.jpg into this folder or drag-and-drop any image onto test_page.bat!
    echo.
    pause
    exit /b 1
)

set "TARGET_LANG=en"
echo ================================================================
echo   Image: %IMG_FILE%
echo   Press [Enter] for English (default), or type: ru / romaji
echo ================================================================
set /p "USER_LANG=Target language [en/ru/romaji] (default: en): "
if /i "%USER_LANG%"=="ru" set "TARGET_LANG=ru"
if /i "%USER_LANG%"=="romaji" set "TARGET_LANG=romaji"

echo.
echo Sending "%IMG_FILE%" to local server (target_lang=%TARGET_LANG%)...
echo.
python cli_client.py test "%IMG_FILE%" "%TARGET_LANG%"

echo.
pause
