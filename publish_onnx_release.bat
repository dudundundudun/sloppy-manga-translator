@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Miko-chan Studio - Publish ONNX Models Pack (v1)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\publish_onnx_release.ps1"
pause
