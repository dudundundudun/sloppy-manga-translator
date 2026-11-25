@echo off
setlocal
title Miko-chan Manga Studio v6.1.0 - Register Launcher Protocol
cd /d "%~dp0"

echo Registering manga-translator:// protocol in Windows Registry...

reg add "HKCU\Software\Classes\manga-translator" /ve /d "URL:Manga Translator Protocol" /f
if errorlevel 1 goto :reg_error
reg add "HKCU\Software\Classes\manga-translator" /v "URL Protocol" /d "" /f
if errorlevel 1 goto :reg_error
reg add "HKCU\Software\Classes\manga-translator\shell\open\command" /ve /d "\"%~dp0start_server.bat\"" /f
if errorlevel 1 goto :reg_error

echo.
echo [OK] Protocol manga-translator:// registered successfully!
pause
exit /b 0

:reg_error
echo.
echo [ERROR] Failed to update Windows Registry.
pause
exit /b 1
