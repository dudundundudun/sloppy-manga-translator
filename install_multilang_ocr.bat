@echo off
title Install Korean and Chinese OCR Support (EasyOCR)
cd /d "%~dp0"

echo =======================================================================
echo   Installing Korean (Manhwa) and Chinese (Manhua) OCR dependencies
echo   (Keeping your existing PyTorch 2.10+cu130 and OpenCV intact!)
echo =======================================================================
echo.

pip install easyocr --no-deps
pip install scipy scikit-image python-bidi pyclipper shapely ninja

echo.
echo =======================================================================
echo   [OK] Korean (ko), Chinese (zh), and Auto-Detection (auto) are ready!
echo =======================================================================
echo.
pause
