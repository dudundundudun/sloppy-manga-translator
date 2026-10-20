@echo off
title Miko-chan Manga Studio v1.0.0 - Download Shisa and Top Models (with VPN 198.18.x.x Bypass)
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

echo ======================================================================
echo   Download Best Manga Translation Models (Auto-Bypass for VPN 198.18)
echo ======================================================================
echo   [1] Shisa V2.1 14B (Phi-4, Top-1 for Japanese Manga, ~8.3 GB) [DEFAULT]
echo   [2] Shisa V2.1 8B  (Qwen3, Fast Japanese/Russian, ~5.0 GB)
echo   [3] Shisa V2 12B Uncensored (Mistral-Nemo 18+ NSFW, ~7.0 GB)
echo   [4] Google Gemma 3 12B (Best literary Russian/English, ~7.3 GB)
echo   [5] Alibaba Qwen 2.5 14B (Best all-rounder JA/KO/ZH/Romaji, ~8.4 GB)
echo ======================================================================
set "CHOICE=1"
set /p "CHOICE=Select model to download and set as default [1-5] (Enter=1): "

set "MODEL_TAG=hf.co/mradermacher/shisa-v2.1-unphi4-14b-GGUF:Q4_K_M"
if "%CHOICE%"=="2" set "MODEL_TAG=hf.co/mradermacher/shisa-v2.1-qwen3-8b-GGUF:Q4_K_M"
if "%CHOICE%"=="3" set "MODEL_TAG=hf.co/yamatazen/Shisa-v2-Mistral-Nemo-12B-Abliterated-Q4_K_M-GGUF"
if "%CHOICE%"=="4" set "MODEL_TAG=gemma3:12b"
if "%CHOICE%"=="5" set "MODEL_TAG=qwen2.5:14b"

echo.
echo [1/1] Downloading and registering "%MODEL_TAG%" (with automatic Python GGUF bypass if Ollama blocks VPN redirect)...
python -c "import asyncio, sys; from translator import OllamaTranslator; t = OllamaTranslator(); res = asyncio.run(t.pull_ollama_model('%MODEL_TAG%', set_as_default=True)); sys.exit(0 if res.get('status') == 'completed' else 1)"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to download or register model. Make sure Ollama is running!
    pause
    exit /b 1
)

echo.
echo [DONE] Model is downloaded, imported into Ollama, and set as default!
pause
