@echo off
cd /d "%~dp0"

echo Registering manga-translator:// protocol in Windows Registry...

reg add "HKCU\Software\Classes\manga-translator" /ve /d "URL:Manga Translator Protocol" /f
reg add "HKCU\Software\Classes\manga-translator" /v "URL Protocol" /d "" /f
reg add "HKCU\Software\Classes\manga-translator\shell\open\command" /ve /d "\"%~dp0start_server.bat\"" /f

echo.
echo [OK] Protocol manga-translator:// registered successfully!
pause
