@echo off
title Local Manga Translator v1.0.0 - Batch Folder Translation (Auto-Resume)
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

set "TARGET_LANG=en"
echo ================================================================
echo   Batch Manga Folder Translator v1.0.0 (with Auto-Resume)
echo   Press [Enter] for English (_EN), or type: ru (_RU) / romaji (_ROMAJI)
echo ================================================================
set /p "USER_LANG=Target language [en/ru/romaji] (default: en): "
if /i "%USER_LANG%"=="ru" set "TARGET_LANG=ru"
if /i "%USER_LANG%"=="romaji" set "TARGET_LANG=romaji"

echo.
echo Opening Windows folder selection dialog...
python cli_client.py folder "%TARGET_LANG%"

echo.
pause
