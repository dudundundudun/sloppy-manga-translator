@echo off
title Miko-chan Manga Studio v6.1.0 - Install Korean and Chinese OCR Support (EasyOCR)
cd /d "%~dp0"

echo =======================================================================
echo   Installing Korean (Manhwa) and Chinese (Manhua) OCR dependencies
echo   Note: HayaiOCR v2.5 Nova already works natively on PyTorch/Transformers!
echo   (Using --no-deps to keep your PyTorch 2.10+cu130 and OpenCV intact!)
echo =======================================================================
echo.

python -m pip install --no-deps easyocr
if errorlevel 1 (
    echo [ERROR] Failed to install easyocr. Check Python/pip installation.
    pause
    exit /b 1
)
python -m pip install --no-deps scikit-image python-bidi pyclipper shapely ninja
if errorlevel 1 (
    echo [ERROR] Failed to install optional EasyOCR helper packages.
    pause
    exit /b 1
)

echo.
echo =======================================================================
echo   [OK] Korean (ko), Chinese (zh), and Auto-Detection (auto) are ready!
echo =======================================================================
echo.
pause
