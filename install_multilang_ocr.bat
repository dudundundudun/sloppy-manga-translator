@echo off
title Miko-chan Manga Studio v1.0.8 - Install Korean and Chinese OCR Support (EasyOCR)
cd /d "%~dp0"

echo =======================================================================
echo   Installing Korean (Manhwa) and Chinese (Manhua) OCR dependencies
echo   Note: HayaiOCR v2.5 Nova already works natively on PyTorch/Transformers!
echo   (Using --no-deps to keep your PyTorch 2.10+cu130 and OpenCV intact!)
echo =======================================================================
echo.

pip install easyocr --no-deps
pip install scipy scikit-image python-bidi pyclipper shapely ninja --no-deps

echo.
echo =======================================================================
echo   [OK] Korean (ko), Chinese (zh), and Auto-Detection (auto) are ready!
echo =======================================================================
echo.
pause
