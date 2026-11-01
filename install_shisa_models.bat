@echo off
title Miko-chan Manga Studio v5.3.0 - Download Shisa and Top Models (with VPN 198.18.x.x Bypass)
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

echo ==========================================================================
echo   Download Best Manga Translation and VLM-OCR Models (v5.3.0 VPN Bypass)
echo ==========================================================================
echo   [1]  Shisa V2.1 14B (Phi-4, Top-1 for Japanese Manga, ~8.3 GB) [DEFAULT]
echo   [2]  Shisa V2.1 8B  (Qwen3, Fast Japanese/Russian, ~5.0 GB)
echo   [3]  Shisa V2 12B Uncensored (Mistral-Nemo 18+ NSFW, ~7.0 GB)
echo   [4]  Sugoi 14B Ultra (Specialized JA->EN Uncensored, ~8.5 GB)
echo   [5]  TranslateGemma 12B (Google Specialized JA/KO/ZH->RU/EN, ~7.3 GB)
echo   [6]  Aya Expanse 8B (Fast Multilingual JA/KO/ZH->RU/EN, ~5.1 GB)
echo   [7]  Google Gemma 3 12B (Literary Russian/English, ~7.3 GB)
echo   [8]  Alibaba Qwen 3 14B (2026 Top All-rounder JA/KO/ZH + Romaji, ~9.0 GB)
echo   [9]  GLM-OCR 0.9B Vision (Sniper VLM-OCR for SFX and Covers, ~1.9 GB)
echo   [10] Qwen 3 VL 8B Vision (2026 Sniper VLM-OCR + Image-Aware Context, ~6.2 GB)
echo   [11] Gemma 3 27B QAT Vision (Ultra 16GB VRAM Translator + Vision, ~14.5 GB)
echo   [12] Qwen 3 30B-A3B MoE (Top-3 JP-TL-Bench 84.3% WR Translator, ~16.0 GB)
echo ==========================================================================
set "CHOICE=1"
set /p "CHOICE=Select model to download [1-12] (Enter=1): "

set "MODEL_TAG=hf.co/mradermacher/shisa-v2.1-unphi4-14b-GGUF:Q4_K_M"
set "SET_DEFAULT=True"
if "%CHOICE%"=="2" set "MODEL_TAG=hf.co/mradermacher/shisa-v2.1-qwen3-8b-GGUF:Q4_K_M"
if "%CHOICE%"=="3" set "MODEL_TAG=hf.co/yamatazen/Shisa-v2-Mistral-Nemo-12B-Abliterated-Q4_K_M-GGUF"
if "%CHOICE%"=="4" set "MODEL_TAG=hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M"
if "%CHOICE%"=="5" set "MODEL_TAG=translategemma:12b"
if "%CHOICE%"=="6" set "MODEL_TAG=aya-expanse:8b"
if "%CHOICE%"=="7" set "MODEL_TAG=gemma3:12b"
if "%CHOICE%"=="8" set "MODEL_TAG=qwen3:14b"
if "%CHOICE%"=="9" (
    set "MODEL_TAG=glm-ocr:latest"
    set "SET_DEFAULT=False"
)
if "%CHOICE%"=="10" set "MODEL_TAG=qwen3-vl:8b"
if "%CHOICE%"=="11" set "MODEL_TAG=gemma3:27b-it-qat"
if "%CHOICE%"=="12" set "MODEL_TAG=qwen3:30b-a3b"

echo.
echo [1/1] Downloading and registering "%MODEL_TAG%" (with automatic Python GGUF bypass if Ollama blocks VPN redirect)...
python -c "import asyncio, sys; from translator import OllamaTranslator; t = OllamaTranslator(); res = asyncio.run(t.pull_ollama_model('%MODEL_TAG%', set_as_default=%SET_DEFAULT%)); sys.exit(0 if res.get('status') == 'completed' else 1)"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to download or register model. Make sure Ollama is running!
    pause
    exit /b 1
)

echo.
echo [DONE] Model "%MODEL_TAG%" is downloaded and ready in Ollama!
pause
