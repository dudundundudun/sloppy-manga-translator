"""
Модуль перевода и транслитерации текста манги/манхвы/маньхуа через Ollama, HuggingFace GGUF и Облачные OpenAI-совместимые API (v1.0.8).

Изменения v1.0.6 (Шаг 4):
1. Глобальный глоссарий в `config.json` (`glossary: list[dict[str, str]]` с полями `src`, `dst`, `note`, `target_lang`)
   и локальный глоссарий главы в `.translation_state.json` (`manifest["glossary"]`).
2. Импорт и экспорт глоссария в форматах `.json`, `.tsv` и текстовом формате Sakura (`исходный_термин->перевод #примечание`).
3. Автоматическая фильтрация глоссария (`Matching`) в `OllamaTranslator._get_prompts`: перед отправкой пачки фраз
   отбираются только те термины, чей `src` встречается в `source_map_json` текущей пачки, и добавляются в системный промпт
   в блок `ОБЯЗАТЕЛЬНЫЙ ГЛОССАРИЙ ТЕРМИНОВ И ИМЕН` (100% консистентность имен без перегрузки контекста).

Изменения v1.0.0:
1. Автоматический обход ошибки `redirect target not allowed: us.aws.cdn.hf.co resolves to non-public 198.18.x.x`
   при скачивании моделей в Ollama (возникает из-за Fake-IP / TUN-режима VPN: Amnezia, Nekoray, Hiddify, Clash, Sing-box).
   Если `POST /api/pull` блокируется из-за `198.18.x.x`, переводчик автоматически скачивает `.gguf` файл напрямую
   через Python (`httpx` с поддержкой докачки `Range` и прогресс-баром в ГБ и %) и импортирует его локально
   через `ollama create -f Modelfile` без сетевых запросов со стороны Ollama!
2. Встроенный пресет и полная поддержка шлюза **ModuleAPI (`https://moduleapi.pro/v1`, продавец Morj666 на FunPay)**,
   приоритетная сортировка моделей DeepSeek (`deepseek-v4-pro`, `deepseek-v4-flash`, `deepseek-chat`) и автоматический
   перебор доступных моделей DeepSeek, если ключ ограничен конкретной моделью.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

logger = logging.getLogger(__name__)

# БЛОК 5: Системное правило Typesetting-Aware Translation для контроля длины фразы под геометрию бабла
TYPESETTING_AWARE_RULE_RU = (
    "5. Typesetting-Aware Translation (Контроль длины под верстку бабла): "
    "Если у блока указан мета-тег геометрии бабла (например, `[shape: narrow_vertical, max_chars: ~35]`), "
    "узкий бабл (`narrow_vertical`) или малый лимит `max_chars`, подбирай максимально емкие, живые и короткие "
    "разговорные формулировки без воды и длинных причастных оборотов, чтобы текст крупно и читаемо лег в бабл без мельчения шрифта."
)

TYPESETTING_AWARE_RULE_EN = (
    "5. Typesetting-Aware Translation (Bubble Geometry Budget): "
    "If a bubble includes a geometry tag such as `[shape: narrow_vertical, max_chars: ~35]`, "
    "a `narrow_vertical` shape, or a tight `max_chars` budget, use maximally concise, punchy, natural spoken phrasing "
    "without filler so the text fits at a large, readable font size without shrinking."
)

SYSTEM_PROMPT = (
    "Ты — профессиональный переводчик и редактор локализации манги (Miko-chan Manga Studio v3.5 Pro Scanlation Edition).\n"
    + TYPESETTING_AWARE_RULE_RU
)

SHAPE_META_TAG_REGEX = re.compile(r"\[\s*shape\s*:[^\]]*\]", flags=re.IGNORECASE)


def compute_bubble_geometry_budget(block: dict[str, Any]) -> dict[str, Any]:
    """
    БЛОК 5: Вычисляет геометрический бюджет бабла для контроля длины перевода:
    - `aspect_ratio = round(box_w / max(1, box_h), 2)`
    - `shape_hint`: `"narrow_vertical"` (< 0.65), `"round_oval"` (0.65..1.45), `"wide_horizontal"` (> 1.45)
    - `recommended_max_chars`: `int((box_w * box_h) / (18 * 18) * 1.3)`
    """
    raw_bbox = block.get("bbox") or block.get("clean_bbox") or [0, 0, 120, 120]
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        try:
            x1, y1, x2, y2 = [int(round(float(c))) for c in raw_bbox[:4]]
            box_w = max(1, abs(x2 - x1))
            box_h = max(1, abs(y2 - y1))
        except Exception:
            box_w, box_h = 120, 120
    else:
        box_w, box_h = 120, 120

    aspect_ratio = round(box_w / max(1, box_h), 2)
    if aspect_ratio < 0.65:
        shape_hint = "narrow_vertical"
    elif aspect_ratio <= 1.45:
        shape_hint = "round_oval"
    else:
        shape_hint = "wide_horizontal"

    recommended_max_chars = max(8, min(220, int((box_w * box_h) / (18 * 18) * 1.3)))
    meta_tag = f"[shape: {shape_hint}, max_chars: ~{recommended_max_chars}]"
    block["aspect_ratio"] = aspect_ratio
    block["shape_hint"] = shape_hint
    block["recommended_max_chars"] = recommended_max_chars
    return {
        "aspect_ratio": aspect_ratio,
        "shape_hint": shape_hint,
        "recommended_max_chars": recommended_max_chars,
        "meta_tag": meta_tag,
    }

# Каталог лучших локальных моделей для перевода манги (с прямыми GGUF-ссылками для авто-обхода блокировки Fake-IP 198.18.x.x в Ollama)
RECOMMENDED_LOCAL_MODELS: list[dict[str, str]] = [
    {
        "ollama_tag": "hf.co/mradermacher/shisa-v2.1-unphi4-14b-GGUF:Q4_K_M",
        "local_tag": "shisa-v2.1-14b:Q4_K_M",
        "hf_repo": "mradermacher/shisa-v2.1-unphi4-14b-GGUF",
        "hf_filename": "shisa-v2.1-unphi4-14b.Q4_K_M.gguf",
        "hf_quant": "Q4_K_M",
        "id": "ollama::hf.co/mradermacher/shisa-v2.1-unphi4-14b-GGUF:Q4_K_M",
        "title": "🌸 Shisa V2.1 14B (Phi-4) — Топ-1 для японской манги (~8.3 ГБ)",
        "short_name": "Shisa V2.1 14B",
        "size_gb": "8.3 ГБ",
        "langs": "JA ➔ EN / RU",
        "description": "Флагманская билингвальная модель Shisa V2.1 (14B на базе Phi-4). Идеально понимает японский сленг, именные суффиксы (-сан/-кун/-сэмпай), полутона и контекст манги.",
    },
    {
        "ollama_tag": "hf.co/mradermacher/shisa-v2.1-qwen3-8b-GGUF:Q4_K_M",
        "local_tag": "shisa-v2.1-8b:Q4_K_M",
        "hf_repo": "mradermacher/shisa-v2.1-qwen3-8b-GGUF",
        "hf_filename": "shisa-v2.1-qwen3-8b.Q4_K_M.gguf",
        "hf_quant": "Q4_K_M",
        "id": "ollama::hf.co/mradermacher/shisa-v2.1-qwen3-8b-GGUF:Q4_K_M",
        "title": "🌸 Shisa V2.1 8B (Qwen3) — Быстрая для манги JA➔EN/RU (~5.0 ГБ)",
        "short_name": "Shisa V2.1 8B",
        "size_gb": "5.0 ГБ",
        "langs": "JA ➔ EN / RU",
        "description": "Легкая и очень быстрая версия Shisa V2.1 на базе Qwen3-8B. Отлично переводит японскую мангу как на английский, так и на живой русский язык (требует всего ~5 ГБ VRAM).",
    },
    {
        "ollama_tag": "hf.co/yamatazen/Shisa-v2-Mistral-Nemo-12B-Abliterated-Q4_K_M-GGUF",
        "local_tag": "shisa-v2-12b-uncensored:Q4_K_M",
        "hf_repo": "yamatazen/Shisa-v2-Mistral-Nemo-12B-Abliterated-Q4_K_M-GGUF",
        "hf_filename": "shisa-v2-mistral-nemo-12b-abliterated-q4_k_m.gguf",
        "hf_quant": "Q4_K_M",
        "id": "ollama::hf.co/yamatazen/Shisa-v2-Mistral-Nemo-12B-Abliterated-Q4_K_M-GGUF",
        "title": "🔥 Shisa V2 12B Uncensored — Без цензуры для 18+ манги JA➔RU/EN (~7.0 ГБ)",
        "short_name": "Shisa V2 12B (18+)",
        "size_gb": "7.0 ГБ",
        "langs": "JA ➔ RU / EN (18+ без цензуры)",
        "description": "Версия Shisa V2 12B (Mistral-Nemo) с полностью отключенными фильтрами цензуры (Abliterated). Лучший выбор для взрослой 18+ манги и додзинси и для авто-подхвата при цензуре облачного API.",
    },
    {
        "ollama_tag": "hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M",
        "local_tag": "sugoi-14b-ultra:Q4_K_M",
        "hf_repo": "sugoitoolkit/Sugoi-14B-Ultra-GGUF",
        "hf_filename": "Sugoi-14B-Ultra-Q4_K_M.gguf",
        "hf_quant": "Q4_K_M",
        "id": "ollama::hf.co/sugoitoolkit/Sugoi-14B-Ultra-GGUF:Q4_K_M",
        "title": "⚡ Sugoi 14B Ultra — Узкий спец. JA➔EN без цензуры (~8.5 ГБ)",
        "short_name": "Sugoi 14B Ultra (JA➔EN)",
        "size_gb": "8.5 ГБ",
        "langs": "Только JA ➔ EN (18+ без цензуры)",
        "description": "Специализированная модель от создателя Sugoi Toolkit (mingshiba), обученная исключительно на диалогах японской манги, аниме и визуальных новелл без цензуры. Эталон для JA➔EN, но НЕ предназначена для перевода на русский язык.",
    },
    {
        "ollama_tag": "translategemma:12b",
        "local_tag": "translategemma:12b",
        "hf_repo": "mradermacher/translategemma-12b-it-GGUF",
        "hf_filename": "translategemma-12b-it.Q4_K_M.gguf",
        "hf_quant": "Q4_K_M",
        "id": "ollama::translategemma:12b",
        "title": "🌐 TranslateGemma 12B — Спец. переводчик Google JA/KO/ZH➔RU/EN (~7.3 ГБ)",
        "short_name": "TranslateGemma 12B",
        "size_gb": "7.3 ГБ",
        "langs": "JA / KO / ZH ➔ RU / EN",
        "description": "Специализированная модель перевода от Google на базе Gemma 3 12B (обучена для 55 языков). Выдает очень чистый, точный и литературный перевод манги, манхвы и маньхуа на русский и английский языки без галлюцинаций.",
    },
    {
        "ollama_tag": "aya-expanse:8b",
        "local_tag": "aya-expanse:8b",
        "hf_repo": "bartowski/aya-expanse-8b-GGUF",
        "hf_filename": "aya-expanse-8b-Q4_K_M.gguf",
        "hf_quant": "Q4_K_M",
        "id": "ollama::aya-expanse:8b",
        "title": "🌍 Aya Expanse 8B — Легкая мультиязычная JA/KO/ZH➔RU/EN (~5.1 ГБ)",
        "short_name": "Aya Expanse 8B",
        "size_gb": "5.1 ГБ",
        "langs": "JA / KO / ZH ➔ RU / EN",
        "description": "Быстрая мультиязычная модель от Cohere Labs (23 языка, включая японский, корейский, китайский и русский). Летает на видеокартах от 6–8 ГБ VRAM, почти не имеет цензуры и отлично передает живую разговорную речь.",
    },
    {
        "ollama_tag": "gemma3:12b",
        "local_tag": "gemma3:12b",
        "hf_repo": "unsloth/gemma-3-12b-it-GGUF",
        "hf_filename": "gemma-3-12b-it-Q4_K_M.gguf",
        "hf_quant": "Q4_K_M",
        "id": "ollama::gemma3:12b",
        "title": "💎 Gemma 3 12B — Живой литературный русский и английский (~7.3 ГБ)",
        "short_name": "Gemma 3 12B",
        "size_gb": "7.3 ГБ",
        "langs": "JA / KO / ZH ➔ RU / EN",
        "description": "Универсальная модель от Google с естественным и выразительным художественным слогом на русском и английском языках для повседневной и романтической манги.",
    },
    {
        "ollama_tag": "qwen2.5:14b",
        "local_tag": "qwen2.5:14b",
        "hf_repo": "tensorblock/Qwen2.5-14B-Instruct-GGUF",
        "hf_filename": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "hf_quant": "Q4_K_M",
        "id": "ollama::qwen2.5:14b",
        "title": "🐉 Qwen 2.5 14B — Универсал JA/KO/ZH + Ромадзи (~8.4 ГБ)",
        "short_name": "Qwen 2.5 14B",
        "size_gb": "8.4 ГБ",
        "langs": "JA / KO / ZH ➔ RU / EN + Ромадзи",
        "description": "Мощная азиатская модель от Alibaba для одновременной работы с японской мангой, корейской манхвой, китайской маньхуа (культивация/сянься) и точной транслитерацией кандзи в Ромадзи.",
    },
    {
        "ollama_tag": "glm-ocr:latest",
        "local_tag": "glm-ocr:latest",
        "hf_repo": "zai-org/GLM-OCR",
        "hf_filename": "",
        "hf_quant": "F16",
        "id": "ollama::glm-ocr:latest",
        "title": "👁 GLM-OCR 0.9B (Vision) — Снайперский OCR для сложных рукописных SFX и обложек (~1.9 ГБ)",
        "short_name": "GLM-OCR 0.9B (VLM-OCR)",
        "size_gb": "1.9 ГБ",
        "langs": "Vision OCR: JA / KO / ZH / EN (SFX и обложки)",
        "description": "Специализированная компактная Vision-OCR модель (0.9B, #1 на OmniDocBench V1.5) для снайперского чтения рукописных звуковых эффектов (SFX), изогнутых логотипов и цветных обложек при выделении «➕ Обвести» или по кнопке «👁 VLM-OCR».",
    },
    {
        "ollama_tag": "qwen2.5vl:7b",
        "local_tag": "qwen2.5vl:7b",
        "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
        "hf_filename": "",
        "hf_quant": "Q4_K_M",
        "id": "ollama::qwen2.5vl:7b",
        "title": "👁 Qwen 2.5 VL 7B (Vision) — Мультимодальный VLM-OCR для сложных шрифтов JA/KO/ZH (~5.0 ГБ)",
        "short_name": "Qwen 2.5 VL 7B (VLM-OCR)",
        "size_gb": "5.0 ГБ",
        "langs": "Vision OCR + Перевод: JA / KO / ZH",
        "description": "Мощная визуально-языковая модель Alibaba Qwen 2.5 VL 7B. Безошибочно читает стилизованные каллиграфические шрифты, декоративные плашки, вертикальные и наклонные SFX на японском, корейском и китайском языках.",
    },
]

CLOUD_API_PRESETS: list[dict[str, str]] = [
    {
        "id": "deepseek_flash",
        "name": "⚡ DeepSeek Flash (deepseek-v4-flash — быстрая и экономная)",
        "url": "https://moduleapi.pro/v1",
        "default_model": "deepseek-v4-flash",
    },
    {
        "id": "deepseek_pro",
        "name": "🧠 DeepSeek Pro (deepseek-v4-pro — максимальное качество)",
        "url": "https://moduleapi.pro/v1",
        "default_model": "deepseek-v4-pro",
    },
]

TRANSLATION_PROFILES: dict[str, dict[str, Any]] = {
    "eco_book": {
        "id": "eco_book",
        "title": "💎 Эконом + Книга (1–2 запроса)",
        "short_title": "💎 Эконом + Книга",
        "pages_per_batch": 25,
        "tail_merge_pages": 10,
        "max_parallel_streams": 1,
        "min_phrases_per_stream": 150,
        "parallel_trigger_phrases": 9999,
        "two_phase_book_mode": True,
        "local_only": False,
        "description": "Двухфазный режим: сначала локальный OCR пачки (до 25–35 стр.), затем перевод единым сценарием книги (по <=150 фраз без обрезки JSON и таймаутов).",
    },
    "balanced": {
        "id": "balanced",
        "title": "⚖️ Баланс (Без микро-потоков)",
        "short_title": "⚖️ Баланс",
        "pages_per_batch": 15,
        "tail_merge_pages": 5,
        "max_parallel_streams": 2,
        "min_phrases_per_stream": 90,
        "parallel_trigger_phrases": 120,
        "two_phase_book_mode": False,
        "local_only": False,
        "description": "Пачки по 15 страниц со слиянием хвостов (<= 5 стр.), параллельное дробление только при > 120 уникальных фраз (~2 запроса на 25 стр.).",
    },
    "turbo": {
        "id": "turbo",
        "title": "🚀 Турбо (Параллельный)",
        "short_title": "🚀 Турбо",
        "pages_per_batch": 12,
        "tail_merge_pages": 3,
        "max_parallel_streams": 3,
        "min_phrases_per_stream": 40,
        "parallel_trigger_phrases": 40,
        "two_phase_book_mode": False,
        "local_only": False,
        "description": "Быстрый показ первых страниц: пачки по 12 страниц со слиянием хвостов (<= 3 стр.), до 3 параллельных потоков (>= 40 фраз на поток).",
    },
    "local_only": {
        "id": "local_only",
        "title": "🏠 Локальный 100% (Ollama)",
        "short_title": "🏠 Локальный 100%",
        "pages_per_batch": 12,
        "tail_merge_pages": 4,
        "max_parallel_streams": 1,
        "min_phrases_per_stream": 12,
        "parallel_trigger_phrases": 9999,
        "two_phase_book_mode": False,
        "local_only": True,
        "description": "Полностью оффлайн-режим без запросов в облако. Автоматически выбирает локальную модель по языку оригинала (JA -> Sugoi/Shisa, ZH/KO -> Qwen/Gemma/Aya).",
    },
}

MODEL_LIST_CACHE_TTL_SEC = 300.0

CJK_REGEX = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\uff66-\uff9f]"
)
KANA_ANY_REGEX = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
CJK_SYMBOLS_REGEX = re.compile(r"[\u3000-\u303f\uff00-\uff65\uffa0-\uffef]")
THINK_REGEX = re.compile(r"<think>.*?(?:</think>|$)", flags=re.DOTALL | re.IGNORECASE)
KV_PAIR_REGEX = re.compile(r'"?#?((?:P\d+_\d+)|(?:\d+))"?\s*:\s*"((?:[^"\\]|\\.)*?)(?:"|$)', flags=re.IGNORECASE)

FULLWIDTH_PUNCT_MAP = str.maketrans(
    {
        "！": "!",
        "？": "?",
        "。": ".",
        "，": ",",
        "、": ", ",
        "・": " ",
        "：": ":",
        "；": ";",
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
        "【": "[",
        "】": "]",
        "《": '"',
        "》": '"',
        "〈": "<",
        "〉": ">",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "～": "~",
        "〜": "~",
        "…": "...",
        "‥": "..",
        "—": "-",
        "ー": "-",
        "　": " ",
    }
)

SOURCE_LANG_NAMES_EN = {
    "ja": "Japanese manga",
    "ko": "Korean manhwa (webtoon)",
    "zh": "Chinese manhua",
}

SOURCE_LANG_NAMES_RU = {
    "ja": "японской манги",
    "ko": "корейской манхвы (вебтуна)",
    "zh": "китайской маньхуа",
}

MAX_BUBBLES_PER_BATCH_LOCAL = 12
MAX_BUBBLES_PER_BATCH_CLOUD = 150
CLOUD_PARALLEL_CHUNK_SIZE = 42
MAX_CONCURRENT_CLOUD_REQUESTS = 3

REFUSAL_PATTERNS_REGEX = re.compile(
    r"(?i)(?:\bi\s*(?:can['\u2019]?t|cannot|am\s+unable\s+to|won['\u2019]?t|must\s+decline|refuse\s+to)\b.*"
    r"\b(?:translat|intensif|explicit|sexual|content|request|assist|fulfill|generate|rewrite|child|minor|nsfw|adult|inappropriate)\b"
    r"|\b(?:unable\s+to\s+translate|cannot\s+translate|can['\u2019]?t\s+translate|translation\s+(?:unavailable|refused|blocked|failed|omitted)|refused\s+to\s+translate)\b"
    r"|\b(?:as\s+an\s+ai|content\s+policy|safety\s+guidelines|explicit\s+(?:\w+\s+){0,3}content|incestuous\s+content|violates?\s+(?:our|the)\s+policy|content_filter|data_inspection_failed|censored\s+content)\b"
    r"|\b(?:content|dialogue|text|sound|sfx|line|speech|phrase|audio)\s+(?:omitted|redacted|censored|filtered|blocked|removed)\b"
    r"|\b(?:omitted|redacted|censored)\s+(?:due\s+to|for)\s+(?:safety|policy|explicit|nsfw|content)\b"
    r"|\b(?:я\s+не\s+могу\s+(?:перевести|выполнить|помочь|продолжить)|невозможно\s+перевести|не\s+удалось\s+перевести|перевод\s+(?:невозможен|недоступен|заблокирован|пропущен)|откровенн\w+\s+сексуальн\w+|наруша\w+\s+правил\w+|контент\s+(?:скрыт|удален|пропущен|заблокирован)|звук\s+пропущен)\b"
    r"|(?:抱歉.*(?:无法|不能).*(?:翻译|提供|生成|处理)|无法翻译|不能翻译|内容.*(?:违规|敏感|不当|省略|已省略)))"
)

BUBBLE_REFUSAL_VALUE_REGEX = re.compile(
    r"(?i)(?:\b(?:unable\s+to\s+translate|cannot\s+translate|can['\u2019]?t\s+translate|won['\u2019]?t\s+translate|"
    r"i\s+cannot\s+translate|i\s+can['\u2019]?t\s+translate|i\s+am\s+unable\s+to\s+translate|"
    r"translation\s+(?:is\s+|was\s+)?(?:unavailable|blocked|refused|failed|error|omitted)|"
    r"content\s+(?:is\s+|was\s+|has\s+been\s+)?(?:filtered|blocked|censored|policy|omitted|redacted|removed)|"
    r"safety\s+(?:filter|policy|guidelines)|inappropriate\s+content|explicit\s+(?:\w+\s+){0,3}(?:content|dialogue|text|line|language)|incestuous\s+(?:content|dialogue|text)|minor[\-\s]coded|"
    r"(?:sound|sfx|dialogue|text|line|speech|explicit|nsfw|sexual|part|section|phrase|audio)\s+(?:was\s+|is\s+|has\s+been\s+)?(?:omitted|redacted|censored|filtered|blocked|removed)|"
    r"(?:was|is|has\s+been)\s+(?:omitted|redacted|censored|filtered)\s+(?:due\s+to|for|because)|"
    r"omitted\s+due\s+to|not\s+able\s+to\s+translate|refuse\s+to\s+translate)\b|"
    r"[\[\(\<\*]\s*[^\]\)\>\*]{0,60}\b(?:omit(?:ted)?|censor(?:ed)?|redact(?:ed)?|filter(?:ed)?|block(?:ed)?|refus(?:ed)?|untranslat(?:able|ed)|explicit|incest\w*|nsfw|inappropriate|safety\s+policy)\b[^\]\)\>\*]{0,40}[\]\)\>\*]|"
    r"^\s*[\[\(\<\*\-]*\s*(?:unable\s+to\s+translate|cannot\s+translate|untranslatable|censored|filtered|blocked|nsfw|error|refused|omitted|redacted|sound\s+omitted|content\s+omitted|explicit\s+content\s+was\s+omitted|explicit\s+content\s+omitted|"
    r"не\s+удалось\s+перевести|невозможно\s+перевести|цензура|заблокировано|ошибка\s+перевода|пропущено|звук\s+пропущен|контент\s+пропущен)\s*[\]\)\>\*\-\!\.]*\s*$|"
    r"\b(?:не\s+удалось\s+перевести|невозможно\s+перевести|я\s+не\s+могу\s+перевести|перевод\s+заблокирован|контент\s+пропущен|звук\s+пропущен|откровенный\s+контент\s+пропущен)\b|"
    r"(?:无法翻译|不能翻译|内容违规|敏感内容|内容已省略|省略))"
)

CLOUD_CENSORSHIP_ERROR_REGEX = re.compile(
    r"(?i)(?:content_filter|content_policy|data_inspection_failed|inappropriate\s+content|"
    r"safety\s*(?:filter|system|policy|guidelines)|moderation|nsfw|explicit\s+sexual|"
    r"sensitive\s+content|violat(?:es|ion)\s+.*policy|unsafe\s+content)"
)

# Известные галлюцинации OCR/LLM (например, «Taxi driver, take me to the best restaurant!» от MangaOCR/Sugoi на шумовых кропах)
HALLUCINATION_TRANSLATION_REGEX = re.compile(
    r"(?i)(?:\btaxi\s*driver\b.*\b(?:best\s*restau?rant|take\s*me)\b|"
    r"\btake\s*me\s*to\s*the\s*best\s*restau?rant\b|"
    r"タクシー運転手|一番(?:うまい|美味しい)(?:店|レストラン)|"
    r"\bthank\s*you\s*for\s*watching\b|\bplease\s*subscribe\s*to\s*(?:my|the)\s*channel\b|"
    r"\bвод[иi]тель\s*такси\b.*\bресторан\b|\bотвези\s*меня\s*в\s*лучший\s*ресторан\b)"
)

SHORT_INTERJECTIONS_RU = {
    "あ": "Ах!",
    "あっ": "Ах!",
    "ああ": "А-ах...",
    "ああっ": "А-ах!",
    "あん": "Ахн!",
    "あんっ": "Ахн!",
    "ん": "Мм...",
    "んっ": "Мм!",
    "んん": "Мм-м...",
    "んんっ": "Мм-х!",
    "はぁ": "Ха-а...",
    "はぁっ": "Ха-а!",
    "はぁはぁ": "Ха-а... ха-а...",
    "ひゃ": "Кья!",
    "ひゃっ": "Кья!",
    "きゃ": "Кья!",
    "きゃっ": "Кья!",
    "え": "Э?",
    "えっ": "Э?!",
    "ええ": "Чего?!",
    "うそ": "Не может быть!",
    "だめ": "Нельзя!",
    "ダメ": "Не надо!",
    "やめて": "Прекрати!",
    "やめろ": "Перестань!",
    "いく": "Сейчас кончу!",
    "イク": "Кончаю!",
    "イッちゃう": "Сейчас кончу!",
    "気持ちいい": "Как хорошо...",
    "きもちいい": "Так приятно...",
    "すごい": "Невероятно!",
    "スゴイ": "Невероятно!",
    "なに": "Что?!",
    "ナニ": "Что?!",
    "何": "Что?!",
    "おい": "Эй!",
    "オイ": "Эй!",
    "ちょっと": "Постой!",
    "まて": "Постой!",
    "待て": "Постой!",
    "まって": "Подожди!",
    "待って": "Подожди!",
    "まじ": "Серьёзно?!",
    "マジ": "Серьёзно?!",
    "まじか": "Да ладно?!",
    "マジか": "Да ладно?!",
    "ほんと": "Правда?",
    "ホント": "Правда?",
    "本当": "Правда?",
    "ウソ": "Не может быть!",
    "嘘": "Ложь!",
    "いや": "Нет!",
    "イヤ": "Нет!",
    "嫌": "Не хочу!",
    "やだ": "Не хочу!",
    "ヤダ": "Не хочу!",
    "むり": "Невозможно!",
    "ムリ": "Невозможно!",
    "無理": "Невозможно!",
    "ばか": "Дурак!",
    "バカ": "Дурак!",
    "馬鹿": "Дурак!",
    "あほ": "Идиот!",
    "アホ": "Идиот!",
    "だれ": "Кто?!",
    "ダレ": "Кто?!",
    "誰": "Кто?!",
    "どこ": "Где?!",
    "ドコ": "Где?!",
    "ここ": "Здесь!",
    "ココ": "Здесь!",
    "なんで": "Почему?!",
    "どうして": "Зачем?!",
    "ごめん": "Прости!",
    "ゴメン": "Прости!",
    "ありがとう": "Спасибо!",
    "おねがい": "Пожалуйста!",
    "お願い": "Пожалуйста!",
    "やばい": "Плохо дело!",
    "ヤバい": "Жесть!",
    "ヤバッ": "Жесть!",
    "よし": "Отлично!",
    "いくぞ": "Погнали!",
    "もちろん": "Конечно!",
    "まさか": "Неужели?!",
    "なるほど": "Понятно...",
    "たしかに": "Точно!",
    "そうか": "Вот как...",
    "そうだね": "И правда.",
    "わかった": "Понял!",
    "分かった": "Понял!",
    "いいよ": "Хорошо!",
    "だいじょうぶ": "Всё в порядке!",
    "大丈夫": "Всё в порядке!",
    "かわいい": "Как мило!",
    "可愛い": "Как мило!",
    "いたい": "Больно!",
    "痛い": "Больно!",
    "いたっ": "Ай, больно!",
    "痛っ": "Ай!",
    "うるさい": "Заткнись!",
    "くそ": "Чёрт!",
    "クソ": "Чёрт!",
    "うん": "Угу!",
    "운": "Угу!",
    "はい": "Да!",
    "いいえ": "Нет!",
    "お": "О!",
    "おっ": "О!",
    "おお": "Ого!",
    "おおっ": "Ого!",
    "あぁ": "А-ах...",
    "あぁっ": "А-ах!",
    "ああん": "А-ахн!",
    "あふ": "Ахф...",
    "あふっ": "Ахф!",
    "う": "Ух...",
    "うっ": "Ух!",
    "うぅ": "У-ух...",
    "うぅっ": "У-ух!",
    "うう": "У-у...",
    "ううっ": "У-ух!",
    "んぅ": "Мм...",
    "んぅっ": "Ммх!",
    "んあ": "Мм-ах!",
    "んあっ": "Мм-ах!",
    "はあ": "Ха-а...",
    "はあっ": "Ха-а!",
    "はあはあ": "Ха-а... ха-а...",
    "はふ": "Хаф...",
    "はふっ": "Хаф!",
    "ふぅ": "Фух...",
    "ふぅっ": "Фух!",
    "ふう": "Фух...",
    "ふうっ": "Фух!",
    "ア": "Ах!",
    "アッ": "Ах!",
    "アア": "А-ах...",
    "アアッ": "А-ах!",
    "アン": "Ахн!",
    "アンッ": "Ахн!",
    "ウ": "Ух...",
    "ウッ": "Ух!",
    "ン": "Мм...",
    "ンッ": "Мм!",
    "ハァ": "Ха-а...",
    "ハァッ": "Ха-а!",
    "フゥ": "Фух...",
    "フゥッ": "Фух!",
    "わっ": "Вах!",
    "ワッ": "Вах!",
    "くっ": "Кх!",
    "クッ": "Кх!",
    "ちっ": "Ц!",
    "チッ": "Ц!",
    "へっ": "Хех!",
    "ヘッ": "Хех!",
    "へぇ": "Ого...",
    "ヘェ": "Ого...",
    "ほっ": "Фух...",
    "ホッ": "Фух...",
    "ちゅ": "Чмок!",
    "ちゅっ": "Чмок!",
    "チュ": "Чмок!",
    "チュッ": "Чмок!",
    "ぎゅ": "Сжим!",
    "ぎゅっ": "Сжим!",
    "ギュ": "Сжим!",
    "ギュッ": "Сжим!",
    "どき": "Тук-тук!",
    "どきっ": "Тук!",
    "ドキ": "Тук-тук!",
    "ドキッ": "Тук!",
    "びく": "Дёрг!",
    "びくっ": "Дёрг!",
    "ビク": "Дёрг!",
    "ビクッ": "Дёрг!",
    "くす": "Хи-хи...",
    "くすっ": "Хи-хи!",
    "クス": "Хи-хи...",
    "クスッ": "Хи-хи!",
    "ごく": "Глоть!",
    "ごくっ": "Глоть!",
    "ゴク": "Глоть!",
    "ゴクッ": "Глоть!",
}

SHORT_INTERJECTIONS_EN = {
    "あ": "Ah!",
    "あっ": "Ah!",
    "ああ": "Aah...",
    "ああっ": "Aah!",
    "あん": "Ahn!",
    "あんっ": "Ahn!",
    "ん": "Mm...",
    "んっ": "Mm!",
    "んん": "Mmm...",
    "んんっ": "Mmh!",
    "はぁ": "Haa...",
    "はぁっ": "Haa!",
    "はぁはぁ": "Haa... haa...",
    "ひゃ": "Hya!",
    "ひゃっ": "Hya!",
    "きゃ": "Kyaa!",
    "きゃっ": "Kyaa!",
    "え": "Eh?",
    "えっ": "Eh?!",
    "ええ": "What?!",
    "うそ": "No way!",
    "だめ": "No, don't!",
    "ダメ": "Stop, no!",
    "やめて": "Stop it!",
    "やめろ": "Stop!",
    "いく": "I'm coming!",
    "イク": "I'm coming!",
    "イッちゃう": "I'm gonna cum!",
    "気持ちいい": "Feels so good...",
    "きもちいい": "Feels amazing...",
    "すごい": "Amazing!",
    "スゴイ": "Amazing!",
    "なに": "What?!",
    "ナニ": "What?!",
    "何": "What?!",
    "おい": "Hey!",
    "オイ": "Hey!",
    "ちょっと": "Wait a sec!",
    "まて": "Wait!",
    "待て": "Wait!",
    "まって": "Hold on!",
    "待って": "Hold on!",
    "まじ": "Seriously?!",
    "マジ": "Seriously?!",
    "ほんと": "Really?",
    "ホント": "Really?",
    "本当": "Really?",
    "ウソ": "No way!",
    "嘘": "No way!",
    "いや": "No!",
    "イヤ": "No!",
    "嫌": "No way!",
    "やだ": "No way!",
    "ヤダ": "I don't want to!",
    "むり": "Impossible!",
    "ムリ": "No way!",
    "無理": "Impossible!",
    "ばか": "Idiot!",
    "バカ": "Idiot!",
    "馬鹿": "Idiot!",
    "あほ": "Moron!",
    "アホ": "Moron!",
    "だれ": "Who?!",
    "ダレ": "Who?!",
    "誰": "Who?!",
    "どこ": "Where?!",
    "ドコ": "Where?!",
    "ここ": "Here!",
    "ココ": "Right here!",
    "なんで": "Why?!",
    "どうして": "Why?!",
    "ごめん": "Sorry!",
    "ありがとう": "Thank you!",
    "おねがい": "Please!",
    "お願い": "Please!",
    "やばい": "Uh oh!",
    "ヤバい": "That's crazy!",
    "ヤバッ": "Crap!",
    "よし": "Alright!",
    "いくぞ": "Let's go!",
    "もちろん": "Of course!",
    "まさか": "No way?!",
    "なるほど": "I see...",
    "たしかに": "True!",
    "そうか": "I see...",
    "わかった": "Got it!",
    "分かった": "Got it!",
    "いいよ": "Sure!",
    "だいじょうぶ": "Are you okay?",
    "大丈夫": "It's fine!",
    "かわいい": "So cute!",
    "可愛い": "So cute!",
    "いたい": "Ouch!",
    "痛い": "It hurts!",
    "うるさい": "Shut up!",
    "くそ": "Damn it!",
    "クソ": "Damn it!",
    "うん": "Yeah!",
    "はい": "Yes!",
    "いいえ": "No!",
    "お": "Oh!",
    "おっ": "Oh!",
    "おお": "Ooh!",
    "おおっ": "Whoa!",
    "あぁ": "Aah...",
    "あぁっ": "Aah!",
    "ああん": "Aahn!",
    "あふ": "Ahf...",
    "あふっ": "Ahf!",
    "う": "Ugh...",
    "うっ": "Ugh!",
    "うぅ": "Uugh...",
    "うぅっ": "Uugh!",
    "うう": "Uuu...",
    "ううっ": "Uugh!",
    "んぅ": "Mmn...",
    "んぅっ": "Mmnh!",
    "んあ": "Mmah!",
    "んあっ": "Mmah!",
    "はあ": "Haa...",
    "はあっ": "Haa!",
    "はあはあ": "Haa... haa...",
    "はふ": "Haf...",
    "はふっ": "Haf!",
    "ふぅ": "Phew...",
    "ふぅっ": "Phew!",
    "ふう": "Phew...",
    "ふうっ": "Phew!",
    "ア": "Ah!",
    "アッ": "Ah!",
    "アア": "Aah...",
    "アアッ": "Aah!",
    "アン": "Ahn!",
    "アンッ": "Ahn!",
    "ウ": "Ugh...",
    "ウッ": "Ugh!",
    "ン": "Mm...",
    "ンッ": "Mm!",
    "ハァ": "Haa...",
    "ハァッ": "Haa!",
    "フゥ": "Phew...",
    "フゥッ": "Phew!",
    "わっ": "Wah!",
    "ワッ": "Wah!",
    "くっ": "Ghh!",
    "クッ": "Ghh!",
    "ちっ": "Tsk!",
    "チッ": "Tsk!",
    "へっ": "Heh!",
    "ヘッ": "Heh!",
    "へぇ": "Oh...",
    "ヘェ": "Oh...",
    "ほっ": "Phew...",
    "ホッ": "Phew...",
    "いたっ": "Ow!",
    "痛っ": "Ow!",
    "ちゅ": "Chu!",
    "ちゅっ": "Mwah!",
    "チュ": "Chu!",
    "チュッ": "Mwah!",
    "ぎゅ": "Squeeze!",
    "ぎゅっ": "Squeeze!",
    "ギュ": "Squeeze!",
    "ギュッ": "Squeeze!",
    "どき": "Badump!",
    "どきっ": "Thump!",
    "ドキ": "Badump!",
    "ドキッ": "Thump!",
    "びく": "Twitch!",
    "びくっ": "Flinch!",
    "ビク": "Twitch!",
    "ビクッ": "Flinch!",
    "くす": "Heh...",
    "くすっ": "Giggle!",
    "クス": "Heh...",
    "クスッ": "Giggle!",
    "ごく": "Gulp!",
    "ごくっ": "Gulp!",
    "ゴク": "Gulp!",
    "ゴクッ": "Gulp!",
}

SHORT_INTERJECTIONS_ROMAJI = {
    "あ": "A!",
    "あっ": "Ah!",
    "ああ": "Aah...",
    "ああっ": "Aah!",
    "あん": "An!",
    "あんっ": "Ahn!",
    "ん": "N...",
    "んっ": "Nn!",
    "んん": "Nnn...",
    "んんっ": "Nnh!",
    "はぁ": "Haa...",
    "はぁっ": "Haa!",
    "はぁはぁ": "Haa... haa...",
    "ひゃ": "Hya!",
    "ひゃっ": "Hyah!",
    "きゃ": "Kya!",
    "きゃっ": "Kyah!",
    "え": "E?",
    "えっ": "Eh?!",
    "ええ": "Ee?!",
    "うそ": "Uso!",
    "だめ": "Dame!",
    "ダメ": "Dame!",
    "やめて": "Yamete!",
    "やめろ": "Yamero!",
    "いく": "Iku!",
    "イク": "Iku!",
    "イッちゃう": "Icchau!",
    "気持ちいい": "Kimochiii...",
    "きもちいい": "Kimochiii...",
    "すごい": "Sugoi!",
    "なに": "Nani?!",
    "何": "Nani?!",
    "おい": "Oi!",
    "ちょっと": "Chotto!",
}

PURE_SIGH_KEYS_JA: set[str] = {
    "あ", "あっ", "ああ", "ああっ", "あん", "あんっ",
    "ん", "んっ", "んん", "んんっ",
    "はぁ", "はぁっ", "はぁはぁ",
    "ひゃ", "ひゃっ", "きゃ", "きゃっ",
    "う", "うっ", "ふぅ", "ふっ",
}

TRANSLATED_SIGH_REGEX = re.compile(
    r"^(?:а+|о+|у+|м+|н+|"
    r"а+х+|о+х+|у+х+|а+х*н+|м+х+|н+г*х*|"
    r"ха+|фу+х*|"
    r"a+|o+|u+|m+|n+|ah+|aah+|ahn+|ooh+|ha+|haa+|mm+h*|ngh+)$",
    flags=re.IGNORECASE,
)

# Частые японские слова на Ромадзи, которые модели иногда ошибочно выводят вместо английского/русского перевода
ROMAJI_LEAK_WORDS_SET: set[str] = {
    "yamete", "yamero", "dame", "kimochi", "kimochiii", "chotto", "matte", "matta",
    "hontou", "honto", "daijoubu", "gomen", "gomennasai", "arigatou", "sumimasen",
    "onegai", "onegaishimasu", "nani", "nande", "doushite", "kawaii", "sugoi", "sugee",
    "uso", "usodesho", "iya", "yada", "muri", "baka", "aho", "urusai", "kuso",
    "iku", "icchau", "itai", "oishii", "ureshii", "hazukashii", "abunai", "hentai",
    "masaka", "naruhodo", "tashika", "tashikani", "wakatta", "iiyo", "souka", "soudane",
    "doko", "dare", "koko", "soko", "asoko", "watashi", "atashi", "ore", "boku",
    "kimi", "anata", "kanojo", "kareshi", "nakadashi", "oppai", "chinko", "manko",
    "ecchi", "sukebe", "oniichan", "oneechan", "otousan", "okaasan", "ojisan", "obasan",
    "tasukete", "hayaku", "yappari", "zenshin", "zutto", "motto", "kitto", "yatto",
    "meccha", "totemo", "sugoku", "kowai", "atsui", "samui", "tsumetai", "warui",
    "yokatta", "subarashii", "kirei", "suteki", "saikou", "saitei", "daisuki", "suki",
    "kirai", "daikirai", "ohayou", "konnichiwa", "konbanwa", "oyasumi", "tadaima", "okaeri",
    "itadakimasu", "gochisousama", "otsukaresama", "shitsurei", "naruhodo", "mochiron",
}

# Прямая замена типичных фраз на Ромадзи на настоящий перевод (если ИИ упорно вернул транслит)
ROMAJI_PHRASE_FALLBACK_EN: dict[str, str] = {
    "yamete": "Stop it!",
    "yamete yo": "Stop it!",
    "yamero": "Stop!",
    "dame": "No, don't!",
    "dame yo": "No, you can't!",
    "dame da": "No way!",
    "kimochiii": "Feels so good...",
    "kimochi ii": "Feels so good...",
    "chotto": "Wait a sec!",
    "chotto matte": "Wait a second!",
    "matte": "Hold on!",
    "matte yo": "Wait up!",
    "hontou": "Really?",
    "hontou ni": "Really?",
    "honto": "Really?",
    "daijoubu": "Are you okay?",
    "daijoubu desu ka": "Are you alright?",
    "gomen": "Sorry!",
    "gomen ne": "I'm sorry!",
    "gomennasai": "I'm so sorry!",
    "arigatou": "Thank you!",
    "arigatou gozaimasu": "Thank you very much!",
    "sumimasen": "Excuse me!",
    "onegai": "Please!",
    "onegaishimasu": "Please!",
    "nani": "What?!",
    "nani kore": "What is this?!",
    "nani sore": "What's that?!",
    "nande": "Why?!",
    "doushite": "Why?!",
    "kawaii": "So cute!",
    "sugoi": "Amazing!",
    "sugee": "Awesome!",
    "uso": "No way!",
    "uso desho": "You're kidding, right?!",
    "iya": "No!",
    "yada": "No way!",
    "muri": "Impossible!",
    "baka": "Idiot!",
    "aho": "Moron!",
    "urusai": "Shut up!",
    "kuso": "Damn it!",
    "iku": "I'm coming!",
    "icchau": "I'm gonna cum!",
    "itai": "Ouch, it hurts!",
    "hazukashii": "So embarrassing...",
    "abunai": "Look out!",
    "hentai": "Pervert!",
    "masaka": "No way?!",
    "naruhodo": "I see...",
    "tashika ni": "That's true!",
    "wakatta": "Got it!",
    "ii yo": "Sure!",
    "iiyo": "Sure!",
    "sou ka": "I see...",
    "souka": "I see...",
    "sou da ne": "That's right.",
    "soudane": "That's right.",
    "doko": "Where?!",
    "dare": "Who?!",
    "koko": "Right here!",
    "tasukete": "Help me!",
    "hayaku": "Hurry up!",
    "yokatta": "Thank goodness!",
    "motto": "More!",
    "daisuki": "I love you!",
    "suki": "I like you!",
}

ROMAJI_PHRASE_FALLBACK_RU: dict[str, str] = {
    "yamete": "Прекрати!",
    "yamete yo": "Перестань же!",
    "yamero": "Остановись!",
    "dame": "Нельзя!",
    "dame yo": "Нет, не надо!",
    "dame da": "Нельзя!",
    "kimochiii": "Как хорошо...",
    "kimochi ii": "Так приятно...",
    "chotto": "Постой!",
    "chotto matte": "Подожди секунду!",
    "matte": "Подожди!",
    "matte yo": "Постой же!",
    "hontou": "Правда?",
    "hontou ni": "Серьёзно?",
    "honto": "Правда?",
    "daijoubu": "Всё в порядке?",
    "gomen": "Прости!",
    "gomen ne": "Прости меня!",
    "gomennasai": "Извини, пожалуйста!",
    "arigatou": "Спасибо!",
    "sumimasen": "Извините!",
    "onegai": "Пожалуйста!",
    "nani": "Что?!",
    "nani kore": "Что это?!",
    "nani sore": "Это ещё что?!",
    "nande": "Почему?!",
    "doushite": "Зачем?!",
    "kawaii": "Как мило!",
    "sugoi": "Невероятно!",
    "sugee": "Круто!",
    "uso": "Не может быть!",
    "uso desho": "Да ладно?!",
    "iya": "Нет!",
    "yada": "Не хочу!",
    "muri": "Невозможно!",
    "baka": "Дурак!",
    "aho": "Идиот!",
    "urusai": "Заткнись!",
    "kuso": "Чёрт!",
    "iku": "Сейчас кончу!",
    "icchau": "Сейчас кончу!",
    "itai": "Больно!",
    "hazukashii": "Как стыдно...",
    "abunai": "Осторожно!",
    "hentai": "Извращенец!",
    "masaka": "Неужели?!",
    "naruhodo": "Понятно...",
    "tashika ni": "И то верно!",
    "wakatta": "Понял!",
    "ii yo": "Хорошо!",
    "iiyo": "Хорошо!",
    "sou ka": "Вот как...",
    "souka": "Вот как...",
    "sou da ne": "И правда.",
    "soudane": "И правда.",
    "doko": "Где?!",
    "dare": "Кто?!",
    "koko": "Здесь!",
    "tasukete": "Помогите!",
    "hayaku": "Скорее!",
    "yokatta": "Слава богу!",
    "motto": "Ещё!",
    "daisuki": "Я люблю тебя!",
    "suki": "Ты мне нравишься!",
}


def is_translated_sigh_text(translated_text: str, original_text: str = "") -> bool:
    """
    Проверяет, является ли готовый перевод коротким вздохом/стоном («Ах!», «Мм...», «Ха-а...»).
    ВАЖНО: срабатывает ТОЛЬКО если исходный японский текст сам входит в узкий список чистых стонов `PURE_SIGH_KEYS_JA`!
    Никогда не удаляет осмысленные короткие реплики, вопросы (`?`/`？`) или кандзи.
    """
    if not translated_text:
        return True
    if "?" in (original_text or "") or "？" in (original_text or "") or "?" in translated_text:
        return False
    orig_clean = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\♡\♥\♪]+", "", original_text or "")
    if len(orig_clean) > 4:
        return False
    if orig_clean and orig_clean not in PURE_SIGH_KEYS_JA:
        return False

    cleaned = re.sub(
        r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\（\）\(\)\[\]\♡\♥\♪\★\☆\・\：\;\"\'\`]+",
        " ",
        translated_text.strip().lower(),
    ).strip()
    if not cleaned:
        return True

    tokens = [t for t in cleaned.split() if t]
    if not tokens or len(tokens) > 3:
        return False

    for tok in tokens:
        collapsed = re.sub(r"(.)\1{2,}", r"\1\1", tok)
        if not (TRANSLATED_SIGH_REGEX.match(tok) or TRANSLATED_SIGH_REGEX.match(collapsed)):
            return False
    return True

_DIGRAPH_ROMAJI_MAP: dict[str, str] = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho", "しぇ": "she",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho", "ちぇ": "che",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "じゃ": "ja",  "じゅ": "ju",  "じょ": "jo",  "じぇ": "je",
    "ぢゃ": "ja",  "ぢゅ": "ju",  "ぢょ": "jo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "ふぁ": "fa",  "ふぃ": "fi",  "ふぇ": "fe",  "ふぉ": "fo",
    "てぃ": "ti",  "でぃ": "di",  "とぅ": "tu",  "どぅ": "du",
    "うぃ": "wi",  "うぇ": "we",  "うぉ": "wo",
    "ゔぁ": "va",  "ゔぃ": "vi",  "ゔぇ": "ve",  "ゔぉ": "vo",
}

_MONOGRAPH_ROMAJI_MAP: dict[str, str] = {
    "あ": "a",  "い": "i",   "う": "u",   "え": "e",  "お": "o",
    "か": "ka", "き": "ki",  "く": "ku",  "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su",  "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni",  "ぬ": "nu",  "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi",  "ふ": "fu",  "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi",  "む": "mu",  "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu",  "よ": "yo",
    "ら": "ra", "り": "ri",  "る": "ru",  "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "wi",  "ゑ": "we",  "を": "o",  "ん": "n",
    "が": "ga", "ぎ": "gi",  "ぐ": "gu",  "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji",  "ず": "zu",  "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji",  "づ": "zu",  "で": "de", "ど": "do",
    "ば": "ba", "び": "bi",  "ぶ": "bu",  "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi",  "ぷ": "pu",  "ぺ": "pe", "ぽ": "po",
    "ゔ": "vu",
    "ぁ": "a",  "ぃ": "i",   "ぅ": "u",   "ぇ": "e",  "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu",  "ょ": "yo",  "ゎ": "wa",
}


def _katakana_to_hiragana(text: str) -> str:
    chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(ch)
    return "".join(chars)


def kana_to_romaji(text: str) -> str:
    if not text:
        return ""

    hira = _katakana_to_hiragana(text)
    out: list[str] = []
    i = 0
    n = len(hira)
    geminate_next = False

    while i < n:
        ch = hira[i]

        if ch == "っ":
            geminate_next = True
            i += 1
            continue

        if ch == "ー":
            if out and out[-1] and out[-1][-1] in "aeiou":
                out.append(out[-1][-1])
            else:
                out.append("-")
            i += 1
            continue

        syllable: str | None = None
        step = 1

        if i + 1 < n:
            pair = hira[i : i + 2]
            if pair in _DIGRAPH_ROMAJI_MAP:
                syllable = _DIGRAPH_ROMAJI_MAP[pair]
                step = 2

        if syllable is None and ch in _MONOGRAPH_ROMAJI_MAP:
            syllable = _MONOGRAPH_ROMAJI_MAP[ch]
            if ch == "ん" and i + 1 < n:
                nxt = hira[i + 1]
                nxt_rom = _MONOGRAPH_ROMAJI_MAP.get(nxt, "")
                if nxt_rom and nxt_rom[0] in "aeiouy":
                    syllable = "n'"

        if syllable is not None:
            if geminate_next:
                if syllable.startswith("ch"):
                    syllable = "t" + syllable
                elif syllable[0] not in "aeiou":
                    syllable = syllable[0] + syllable
                geminate_next = False
            out.append(syllable)
            i += step
        else:
            if geminate_next:
                if out and out[-1] and out[-1][-1] in "aeiou":
                    out.append("h")
                geminate_next = False
            out.append(ch)
            i += 1

    if geminate_next:
        if out and out[-1] and out[-1][-1] in "aeiou":
            out.append("h")

    res = "".join(out)
    return res[:1].upper() + res[1:] if res else ""



# Встроенный словарь частых японских слов и кандзи манги (для мгновенного локального чтения в Ромадзи даже без ИИ)
_MANGA_KANJI_TO_KANA_PAIRS: list[tuple[str, str]] = sorted(
    [
        # Местоимения и обращения
        ("私達", "わたしたち "), ("私たち", "わたしたち "), ("俺達", "おれたち "), ("俺たち", "おれたち "),
        ("僕達", "ぼくたち "), ("僕たち", "ぼくたち "), ("貴方", "あなた "), ("彼女", "かのじょ "),
        ("彼氏", "かれし "), ("先輩", " せんぱい "), ("先生", " せんせい "), ("お兄ちゃん", "おにいちゃん "),
        ("お姉ちゃん", "おねえちゃん "), ("兄貴", "あにき "), ("姉貴", "あねき "), ("皆様", "みなさま "),
        ("皆さん", "みなさん "), ("自分", "じぶん "), ("一人", "ひとり "), ("二人", "ふたり "),
        ("私", "わたし "), ("俺", "おれ "), ("僕", "ぼく "), ("君", "きみ "), ("彼", "かれ "),
        ("誰", "だれ "), ("何", "なに "),
        # Частые фразы и выражения манги
        ("ありがとう", "ありがとう "), ("すみません", "すみません "), ("ごめんなさい", "ごめんなさい "),
        ("ごめん", "ごめん "), ("お願い", "おねがい "), ("おはよう", "おはよう "), ("こんにちは", "こんにちは "),
        ("こんばんは", "こんばんは "), ("おやすみ", "おやすみ "), ("いただきます", "いただきます "),
        ("ごちそうさま", "ごちそうさま "), ("ただいま", "ただいま "), ("おかえり", "おかえり "),
        ("大丈夫", "だいじょうぶ "), ("本当", "ほんとう "), ("ほんと", "ほんと "), ("嘘", "うそ "),
        ("駄目", "だめ "), ("無理", "むり "), ("最高", "さいこう "), ("最低", "さいてい "),
        ("大好き", "だいすき "), ("大嫌い", "だいきらい "), ("好き", "すき "), ("嫌い", "きらい "),
        ("可愛い", "かわいい "), ("かわいい", "かわいい "), ("綺麗", "きれい "), ("素敵", "すてき "),
        ("気持ちいい", "きもちいい "), ("気持ち", "きもち "), ("気持", "きもち "), ("恥ずかしい", "はずかしい "),
        ("恥ずかし", "はずかし"), ("凄い", "すごい "), ("すごい", "すごい "), ("美味しい", "おいしい "),
        ("嬉しい", "うれしい "), ("悲しい", "かなしい "), ("楽しい", "たのしい "), ("寂しい", "さびしい "),
        ("怖い", "こわい "), ("痛い", "いたい "), ("熱い", "あつい "), ("暑い", "あつい "),
        ("寒い", "さむい "), ("冷たい", "つめたい "), ("早い", "はやい "), ("速い", "はやい "),
        ("遅い", "おそい "), ("強い", "つよい "), ("弱い", "よわい "), ("ا大きい", "おおきい "),
        ("大きい", "おおきい "), ("小さい", "ちいさい "), ("新しい", "あたらしい "), ("古い", "ふるい "),
        ("良い", "いい "), ("悪い", "わるい "), ("難しい", "むずかしい "), ("優しい", "やさしい "),
        ("危ない", "あぶない "), ("変態", "へんたい "), ("馬鹿", "ばか "), ("阿呆", "あほ "),
        # Время и место
        ("今日", "きょう "), ("明日", "あした "), ("昨日", "きのう "), ("毎日", "まいにち "),
        ("今夜", "こんや "), ("今朝", "けさ "), ("今週", "こんしゅう "), ("来週", "らいしゅう "),
        ("先週", "せんしゅう "), ("今月", "こんげつ "), ("来月", "らいげつ "), ("今年", "ことし "),
        ("来年", "らいねん "), ("去年", "きょねん "), ("時間", "じかん "), ("場所", "ばしょ "),
        ("学校", "がっこう "), ("教室", "きょうしつ "), ("部屋", "へや "), ("部活", "ぶかつ "),
        ("仕事", "しごと "), ("会社", "かいしゃ "), ("世界", "せかい "), ("人間", "にんげん "),
        ("友達", "ともだち "), ("友人", "ゆうじん "), ("家族", "かぞく "), ("両親", "りょうしん "),
        ("母親", "ははおや "), ("父親", "ちちおや "), ("お母さん", "おかあさん "), ("お父さん", "おとうさん "),
        ("B兄", "あに"), ("姉", "あね"), ("妹", "いもうと "), ("弟", "おとうと "),
        ("男の子", "おとこのこ "), ("女の子", "おんなのこ "), ("女性", "じょせい "), ("男性", "だんせい "),
        ("子供", "こども "), ("大人", "おとな "), ("名前", "なまえ "), ("意味", "いみ "),
        ("理由", "りゆう "), ("約束", "やくそく "), ("秘密", "ひみつ "), ("準備", "じゅんび "),
        ("連絡", "れんらく "), ("返事", "へんじ "), ("質問", "しつもん "), ("問題", "もんだい "),
        ("結果", "けっか "), ("最後", "さいご "), ("最初", "さいしょ "), ("全部", "ぜんぶ "),
        ("半分", "はんぶん "), ("一緒", "いっしょ "), ("絶対", "ぜったい "), ("全然", "ぜんぜん "),
        ("普通", "ふつう "), ("特別", "とくべつ "), ("無理", "むり "), ("我慢", "がまん "),
        # Частые глаголы с окуриганой и основы
        ("待って", "まって "), ("待た", "また"), ("待ち", "まち"), ("待つ", "まつ"),
        ("行って", "いって "), ("行こう", "いこう "), ("行こ", "いこ"), ("行く", "いく"), ("行き", "いき"), ("行か", "いか"),
        ("来て", "きて "), ("来い", "こい "), ("来る", "くる"), ("来な", "こな"), ("来た", "きた"),
        ("見て", "みて "), ("見る", "みる"), ("見た", "みた"), ("見せ", "みせ"), ("見え", "みえ"),
        ("聞いて", "きいて "), ("聞く", "きく"), ("聞い", "きい"), ("聞こ", "きこ"),
        ("言って", "いって "), ("言っ", "いっ"), ("言う", "いう"), ("言わ", "いわ"), ("言え", "いえ"),
        ("思って", "おもって "), ("思っ", "おもっ"), ("思う", "おもう"), ("思わ", "おもわ"),
        ("知って", "しって "), ("知ら", "しら"), ("知る", "しる"), ("知り", "しり"),
        ("分かって", "わかって "), ("分かっ", "わかっ"), ("分から", "わから"), ("分かる", "わかる"),
        ("食べて", "たべて "), ("食べる", "たべる"), ("食べ", "たべ"), ("飲んで", "のんで "), ("飲む", "のむ"),
        ("帰って", "かえって "), ("帰る", "かえる"), ("帰り", "かえり"), ("帰ろ", "かえろ"),
        ("入って", "はいって "), ("入っ", "はいっ"), ("入る", "はいる"), ("入れ", "いれ"),
        ("出して", "だして "), ("出す", "だす"), ("出し", "だし"), ("出て", "でて "), ("出る", "でる"), ("出ちゃ", "でちゃ"),
        ("止めて", "やめて "), ("止めろ", "やめろ "), ("止め", "やめ"),
        ("触って", "さわって "), ("触ら", "さわら"), ("触る", "さわる"), ("触り", "さわり"),
        ("感じて", "かんじて "), ("感じる", "かんじる"), ("感じ", "かんじ"),
        ("動いて", "うごいて "), ("動か", "うごか"), ("動く", "うごく"),
        ("寝て", "ねて "), ("寝る", "ねる"), ("起き", "おき"), ("泣い", "ない"), ("泣く", "なく"),
        ("笑っ", "わらっ"), ("笑う", "わらう"), ("怒っ", "おこっ"), ("怒る", "おこる"),
        ("忘れ", "わすれ"), ("覚え", "おぼえ"), ("考え", "かんがえ"), ("信じ", "しんじ"),
        ("助け", "たすけ"), ("守る", "まもる"), ("守り", "まもり"), ("변", ""),
        ("使っ", "つかっ"), ("使う", "つかう"), ("作っ", "つくっ"), ("作る", "つくる"),
        ("持っ", "もっ"), ("持つ", "もつ"), ("持ち", "もち"),
        ("買っ", "かっ"), ("買う", "かう"), ("売る", "うる"),
        ("読ん", "よん"), ("読む", "よむ"), ("書い", "かい"), ("書く", "かく"),
        ("선", ""), ("話", "はなし"), ("소", ""), ("声", "こえ "), ("身体", "からだ "), ("体", "からだ "),
        ("顔", "かお "), ("目", "め "), ("瞳", "ひとみ "), ("口", "くち "), ("唇", "くちびる "),
        ("手", "て "), ("指", "ゆび "), ("足", "あし "), ("脚", "あし "), ("胸", "むね "),
        ("お尻", "おしり "), ("尻", "しり "), ("お腹", "おなか "), ("頭", "あたま "), ("髪", "かみ "),
        ("心", "こころ "), ("愛", "あい "), ("恋", "こい "), ("夢", "ゆめ "), ("涙", "なみだ "),
        ("汗", "あせ "), ("血", "ち "), ("息", "いき "), ("奥", "おく "), ("中", "なか "),
        ("外", "そと "), ("上", "うえ "), ("下", "した "), ("前", "まえ "), ("後ろ", "うしろ "),
        ("隣", "となり "), ("横", "よこ "), ("側", "そば "), ("先", "さき "), ("次", "つぎ "),
        ("今", "いま "), (" مع", ""), ("少し", "すこし "), ("ちょっと", "ちょっと "),
        ("沢山", "たくさん "), ("一杯", "いっぱい "), ("一回", "いっかい "), ("もう一度", "もういちど "),
        ("一度", "いちど "), ("一番", "いちばん "), ("元気", "げんき "), ("本気", "ほんき "),
        ("気", "き "), ("事", "こと "), ("物", "もの "), ("人", "ひと "), ("方", "かた "),
        ("所", "ところ "), ("時", "とき "), ("日", "ひ "), ("月", "つき "), ("年", "とし "),
        ("水", "みず "), ("火", "ひ "), ("風", "かぜ "), ("空", "そら "), ("海", "うみ "),
        ("山", "やま "), ("川", "かわ "), ("道", "みち "), ("家", "いえ "), ("店", "みせ "),
        ("車", "くるま "), ("電車", "でんしゃ "), ("電話", "でんわ "), ("写真", "しゃしん "),
        ("服", "ふく "), ("下着", "したぎ "), ("水着", "みずぎ "), ("制服", "せいふく "),
    ],
    key=lambda item: len(item[0]),
    reverse=True,
)

# Пост-обработка пробелов перед японскими частицами в Ромадзи для удобного чтения по словам
_ROMAJI_PARTICLE_SPACING_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"([a-zA-Z])([!?.,~])"), r"§ "),
    (re.compile(r"\s+"), " "),
]

_PYKAKASI_CONV: Any | None = None
_PYKAKASI_CHECKED: bool = False


def _get_pykakasi_converter() -> Any | None:
    global _PYKAKASI_CONV, _PYKAKASI_CHECKED
    if _PYKAKASI_CHECKED:
        return _PYKAKASI_CONV
    _PYKAKASI_CHECKED = True
    try:
        import pykakasi  # type: ignore

        kks = pykakasi.kakasi()
        _PYKAKASI_CONV = kks
    except Exception:
        _PYKAKASI_CONV = None
    return _PYKAKASI_CONV


def japanese_to_romaji_local(text: str) -> str:
    """
    Быстрая локальная транслитерация японской строки (кандзи + хирагана + катакана) в Хэпбёрн-ромадзи:
    1. Проверяет словарь частых междометий.
    2. Если установлен pykakasi — использует его с разбивкой по словам.
    3. Иначе заменяет частые слова/кандзи манги по встроенному словарю на хирагану, выделяет частицы
       (wa, ga, o, ni, de, no, to, mo) и конвертирует через `kana_to_romaji`.
    """
    if not text or not text.strip():
        return ""

    raw = text.strip()
    norm_orig = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\♡\♥\♪]+", "", raw)
    if norm_orig in SHORT_INTERJECTIONS_ROMAJI:
        return SHORT_INTERJECTIONS_ROMAJI[norm_orig]

    kks = _get_pykakasi_converter()
    if kks is not None:
        try:
            items = kks.convert(raw)
            words = [str(it.get("hepburn") or it.get("orig") or "").strip() for it in items]
            joined = " ".join(w for w in words if w).translate(FULLWIDTH_PUNCT_MAP)
            joined = re.sub(r"\s+", " ", joined).strip()
            if joined:
                return joined[:1].upper() + joined[1:]
        except Exception:
            pass

    working = raw.translate(FULLWIDTH_PUNCT_MAP)

    # 1. Устойчивые связки с частицей は (читается как wa) и частые разговорные конструкции
    for pat, repl in (
        ("今日は", " きょう wa "),
        ("今は", " いま wa "),
        ("私は", " わたし wa "),
        ("俺は", " おれ wa "),
        ("僕は", " ぼく wa "),
        ("君は", " きみ wa "),
        ("あなたは", " あなた wa "),
        ("あいつは", " あいつ wa "),
        ("これは", " これ wa "),
        ("それは", " それ wa "),
        ("あれは", " あれ wa "),
        ("ここは", " ここ wa "),
        ("そこは", " そこ wa "),
        ("には", " ni wa "),
        ("では", " de wa "),
        ("とは", " to wa "),
        ("からは", " kara wa "),
        ("お兄ちゃん", " onii-chan "),
        ("お姉ちゃん", " onee-chan "),
        ("お兄さん", " onii-san "),
        ("お姉さん", " onee-san "),
        ("お母さん", " okaa-san "),
        ("お父さん", " otou-san "),
        ("おじさん", " oji-san "),
        ("おばさん", " oba-san "),
        ("ちゃん", "-chan "),
        ("さん", "-san "),
        ("くん", "-kun "),
        ("君", "-kun "),
        ("様", "-sama "),
        ("先生", " sensei "),
        ("先輩", " senpai "),
        ("なに", " nani "),
        ("なん", " nan "),
        ("どこ", " doko "),
        ("いつ", " itsu "),
        ("どう", " dou "),
        ("どうして", " doushite "),
        ("なんで", " nande "),
        ("なぜ", " naze "),
        ("ほんとうに", " hontou ni "),
        ("本当に", " hontou ni "),
        ("ちょっと", " chotto "),
        ("もっと", " motto "),
        ("ずっと", " zutto "),
        ("きっと", " kitto "),
        ("やっと", " yatto "),
        ("やっぱり", " yappari "),
        ("やはり", " yahari "),
        ("すごく", " sugoku "),
        ("とても", " totemo "),
        ("めっちゃ", " meccha "),
        ("やめて", " yamete "),
        ("やめろ", " yamero "),
        ("だめ", " dame "),
        ("ダメ", " dame "),
        ("いや", " iya "),
        ("イヤ", " iya "),
        ("うそ", " uso "),
        ("ウソ", " uso "),
        ("ありがとう", " arigatou "),
        ("ごめんね", " gomen ne "),
        ("ごめん", " gomen "),
        ("すみません", " sumimasen "),
        ("おねがい", " onegai "),
        ("お願い", " onegai "),
    ):
        if pat in working:
            working = working.replace(pat, repl)

    # 2. Замена кандзи-слов по словарю (окружаем маркером границы слова § для частиц)
    for kanji_word, kana_reading in _MANGA_KANJI_TO_KANA_PAIRS:
        if kanji_word and kanji_word in working:
            clean_k = kana_reading.strip()
            working = working.replace(kanji_word, f" {clean_k}§")

    # 3. Если сразу после кандзи-слова (§) идет падежная частица — отделяем её пробелом
    working = re.sub(r"§は(?![ぁ-ん])", " wa ", working)
    working = re.sub(r"§へ(?![ぁ-ん])", " e ", working)
    working = re.sub(r"§を", " o ", working)
    working = re.sub(r"§(が|に|で|と|も|の|から|まで|より|だけ|くらい|ぐらい|って)", r" \1 ", working)
    working = working.replace("§", " ")

    # Частица を всегда читается как отдельное "o"
    working = working.replace("を", " o ")

    # 4. Отделяем грамматические связки и финальные частицы
    for part_kana, part_rom in (
        ("ですか", " desu ka "),
        ("ですね", " desu ne "),
        ("ですよ", " desu yo "),
        ("でした", " deshita "),
        ("です", " desu "),
        ("ますか", " masu ka "),
        ("ますね", " masu ne "),
        ("ますよ", " masu yo "),
        ("ました", " mashita "),
        ("ません", " masen "),
        ("ます", " masu "),
        ("てるの", "teru no "),
        ("ているの", "te iru no "),
        ("てる", "teru "),
        ("ている", "te iru "),
        ("てた", "teta "),
        ("ていた", "te ita "),
        ("ないで", "naide "),
        ("なくて", "nakute "),
        ("ない", "nai "),
        ("だけど", " dakedo "),
        ("だから", " dakara "),
        ("なのに", " nanoni "),
        ("じゃない", " janai "),
        ("だよね", " dayo ne "),
        ("だよ", " dayo "),
        ("だね", " dane "),
        ("かな", " kana "),
        ("かしら", " kashira "),
        ("わよ", " wayo "),
        ("わね", " wane "),
        ("のよ", " no yo "),
        ("のね", " no ne "),
        ("よ!", " yo!"),
        ("よ?", " yo?"),
        ("ね!", " ne!"),
        ("ね?", " ne?"),
    ):
        if part_kana in working:
            working = working.replace(part_kana, part_rom)

    rom = kana_to_romaji(working)
    rom = re.sub(r"\s+-", "-", rom)
    rom = re.sub(r"\s+([!?.,~])", r"\1", rom)
    rom = re.sub(r"([!?.,~…]+)(?=[A-Za-z])", r"\1 ", rom)
    rom = re.sub(r"(\.\.\.)(?=[A-Za-z♡♥♪])", r"\1 ", rom)
    rom = re.sub(r"\s+", " ", rom).strip()
    return rom[:1].upper() + rom[1:] if rom else ""




def normalize_glossary_entries(entries: Any) -> list[dict[str, str]]:
    """
    Нормализует список записей глоссария к формату:
    `[{"src": str, "dst": str, "note": str, "target_lang": str}]`
    с дедупликацией по `(src, target_lang)`.
    """
    if not isinstance(entries, list):
        return []
    normalized: list[dict[str, str]] = []
    index_by_key: dict[tuple[str, str], int] = {}

    for item in entries:
        if not isinstance(item, dict):
            continue
        src = str(item.get("src") or item.get("source") or item.get("orig") or "").strip()
        dst = str(item.get("dst") or item.get("target") or item.get("trans") or item.get("translation") or "").strip()
        if not src or not dst:
            continue
        note = str(item.get("note") or item.get("comment") or item.get("info") or "").strip()
        t_lang = str(item.get("target_lang") or item.get("lang") or "all").strip().lower()
        if t_lang not in ("all", "ru", "en", "romaji"):
            t_lang = "all"

        dedup_key = (src, t_lang)
        entry_dict = {
            "src": src,
            "dst": dst,
            "note": note,
            "target_lang": t_lang,
        }
        if dedup_key in index_by_key:
            normalized[index_by_key[dedup_key]] = entry_dict
        else:
            index_by_key[dedup_key] = len(normalized)
            normalized.append(entry_dict)

    return normalized


def parse_glossary_text_or_json(
    raw_content: str,
    default_target_lang: str = "all",
) -> list[dict[str, str]]:
    """
    Парсит глоссарий из:
    - JSON массива/объекта (`.json`);
    - TSV таблицы (`src\tdst\tnote`);
    - текстового формата Sakura (`исходный_термин->перевод #примечание`, а также `=>`, `➔`, `=`).
    """
    text = (raw_content or "").strip()
    if not text:
        return []

    def_lang = (default_target_lang or "all").strip().lower()
    if def_lang not in ("all", "ru", "en", "romaji"):
        def_lang = "all"

    # 1. Пробуем JSON
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return normalize_glossary_entries(parsed)
            if isinstance(parsed, dict):
                if isinstance(parsed.get("glossary"), list):
                    return normalize_glossary_entries(parsed["glossary"])
                dict_entries: list[dict[str, str]] = []
                for k, v in parsed.items():
                    if isinstance(v, str) and k.strip() and v.strip():
                        dict_entries.append({"src": k.strip(), "dst": v.strip(), "note": "", "target_lang": def_lang})
                    elif isinstance(v, dict):
                        dict_entries.append({
                            "src": str(v.get("src") or k).strip(),
                            "dst": str(v.get("dst") or v.get("target") or "").strip(),
                            "note": str(v.get("note") or "").strip(),
                            "target_lang": str(v.get("target_lang") or def_lang).strip().lower(),
                        })
                return normalize_glossary_entries(dict_entries)
        except Exception:
            pass

    # 2. Построчный разбор Sakura (`src->dst #note`) и TSV (`src\tdst\tnote`)
    parsed_entries: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        # Если TSV (табуляция)
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) >= 2 and parts[0].lower() not in ("src", "source", "оригинал"):
                src_p = parts[0]
                dst_p = parts[1]
                note_p = parts[2] if len(parts) >= 3 else ""
                lang_p = parts[3].lower() if len(parts) >= 4 and parts[3] else def_lang
                if src_p and dst_p:
                    parsed_entries.append({
                        "src": src_p,
                        "dst": dst_p,
                        "note": note_p,
                        "target_lang": lang_p,
                    })
                continue

        # Отделяем комментарий `#примечание` в конце строки (формат Sakura)
        note_part = ""
        main_part = line
        if "#" in line:
            hash_idx = line.find("#")
            if hash_idx > 0:
                main_part = line[:hash_idx].strip()
                note_part = line[hash_idx + 1 :].strip()

        for sep in ("->", "=>", "➔", "→", "="):
            if sep in main_part:
                lhs, rhs = main_part.split(sep, 1)
                src_s = lhs.strip()
                dst_s = rhs.strip()
                if src_s and dst_s:
                    parsed_entries.append({
                        "src": src_s,
                        "dst": dst_s,
                        "note": note_part,
                        "target_lang": def_lang,
                    })
                break

    return normalize_glossary_entries(parsed_entries)


def export_glossary_to_sakura_text(entries: list[dict[str, str]]) -> str:
    """Экспортирует глоссарий в текстовый формат Sakura: `исходный_термин->перевод #примечание`."""
    norm = normalize_glossary_entries(entries)
    lines: list[str] = []
    for item in norm:
        src = item["src"]
        dst = item["dst"]
        note = item.get("note", "").strip()
        if note:
            lines.append(f"{src}->{dst} #{note}")
        else:
            lines.append(f"{src}->{dst}")
    return "\n".join(lines)


def export_glossary_to_tsv(entries: list[dict[str, str]]) -> str:
    """Экспортирует глоссарий в формат TSV (`src\tdst\tnote\ttarget_lang`)."""
    norm = normalize_glossary_entries(entries)
    lines: list[str] = ["src\tdst\tnote\ttarget_lang"]
    for item in norm:
        lines.append(
            f"{item['src']}\t{item['dst']}\t{item.get('note', '')}\t{item.get('target_lang', 'all')}"
        )
    return "\n".join(lines)


def merge_glossaries(
    global_glossary: list[dict[str, str]] | None,
    folder_glossary: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Объединяет глобальный глоссарий и локальный глоссарий главы (записи главы имеют приоритет)."""
    combined: list[dict[str, str]] = []
    if isinstance(global_glossary, list):
        combined.extend(global_glossary)
    if isinstance(folder_glossary, list):
        combined.extend(folder_glossary)
    return normalize_glossary_entries(combined)


def filter_matching_glossary_entries(
    entries: list[dict[str, str]] | None,
    text_blob: str,
    target_lang: str = "en",
) -> list[dict[str, str]]:
    """
    Отбирает только те записи глоссария (`Matching`), чей `src` реально встречается
    в `text_blob` (например, в `source_map_json` текущей пачки), и чей `target_lang`
    совпадает с целевым языком или равен `'all'`.
    """
    if not entries or not text_blob:
        return []
    t_lang = (target_lang or "en").strip().lower()
    norm_entries = normalize_glossary_entries(entries)

    matched: list[dict[str, str]] = []
    seen_src: set[str] = set()
    # Сначала проверяем точные совпадения по target_lang, затем 'all'
    for pass_exact in (True, False):
        for item in norm_entries:
            item_lang = item.get("target_lang", "all")
            if pass_exact and item_lang != t_lang:
                continue
            if not pass_exact and item_lang != "all":
                continue
            src = item["src"]
            if src in seen_src:
                continue
            if src in text_blob:
                seen_src.add(src)
                matched.append(item)
    return matched


class OllamaTranslator:
    """Универсальный клиент пакетного перевода манги/манхвы/маньхуа через Ollama, HuggingFace GGUF и Облачные API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        unsloth_url: str = "https://moduleapi.pro/v1",
        unsloth_api_key: str = "",
        default_model: str | None = None,
        timeout_sec: float = 90.0,
        num_ctx: int = 4096,
        config_path: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.ollama_url = self.base_url
        self.unsloth_url = self._normalize_api_base_url(unsloth_url)
        self.unsloth_api_key = unsloth_api_key or os.getenv("UNSLOTH_API_KEY", "")
        self.default_model = default_model or os.getenv(
            "OLLAMA_MODEL",
            "hf.co/mradermacher/shisa-v2.1-unphi4-14b-GGUF:Q4_K_M",
        )
        self.timeout_sec = timeout_sec
        self.num_ctx = num_ctx
        self.config_path = (
            Path(config_path) if config_path else Path(__file__).resolve().parent / "config.json"
        )
        self.gguf_dir = self.config_path.resolve().parent / "models" / "gguf"
        self._resolved_model: str | None = None
        self._cached_models: list[dict[str, str]] = []
        self._cache_timestamp: float = 0.0
        self._cached_cloud_probe: dict[str, tuple[float, tuple[str, list[str], bool]]] = {}
        self._cached_ollama_tags: tuple[float, list[str]] | None = None
        self._http_client: httpx.AsyncClient | None = None
        self.last_unsloth_status: dict[str, Any] = {
            "reachable": False,
            "url": self.unsloth_url,
            "detected_models": [],
            "error": None,
        }
        self.pull_state: dict[str, Any] = {
            "status": "idle",
            "model": "",
            "progress_pct": 0.0,
            "completed_gb": 0.0,
            "total_gb": 0.0,
            "message": "",
            "error": None,
        }
        self.auto_save_web_pages: bool = True
        self.web_save_dir: str = str(self.config_path.resolve().parent / "Saved_Manga")
        self.text_filter_mode: str = "bubbles_only"
        self.skip_sighs: bool = False
        self.skip_sfx: bool = True
        self.detect_free_text: bool = False
        self.min_text_length: int = 1
        self._filter_v109_migrated: bool = True
        self.min_font_size: int = 12
        self.max_font_size: int = 32
        self.uppercase: bool = False
        self.overflow_mode: str = "shrink"
        self.batch_pages: int = 15
        self.translation_profile: str = "balanced"
        self.local_fallback_model_multilang: str = "auto"
        self.ui_settings: dict[str, Any] = {
            "editor_dock_mode": "right",
            "inline_bubble_popup": True,
            "always_show_pins": True,
            "auto_scroll_to_bubble": True,
            "floating_auto_fade": True,
            "auto_collapse_sidebar": False,
            "compact_mascot": False,
            "default_zoom_mode": "fit_width",
            "right_panel_width": 440,
            "bottom_panel_height": 340,
            "left_sidebar_width": 360,
        }
        self.recent_context_lines: list[str] = []
        self.phrase_cache: dict[str, str] = {}
        self.glossary: list[dict[str, str]] = []
        self.stop_requested: bool = False
        self._last_call_was_refusal: bool = False
        self.last_refusal_message: str = ""
        self.last_local_fallback_model: str = ""
        self.diagnostics_stats: dict[str, Any] = {
            "session_started_at": int(time.time()),
            "total_llm_requests": 0,
            "total_ollama_requests": 0,
            "total_cloud_requests": 0,
            "parallel_batches_count": 0,
            "phrases_requested_total": 0,
            "phrases_sent_to_llm": 0,
            "phrases_saved_by_cache": 0,
            "refusals_count": 0,
            "local_censorship_fallbacks_count": 0,
            "sanitized_retries_count": 0,
            "micro_retries_count": 0,
            "network_retries_count": 0,
            "cloud_to_local_error_fallbacks_count": 0,
            "romaji_leaks_caught_count": 0,
            "cjk_leaks_caught_count": 0,
            "dict_interjections_used_count": 0,
            "sighs_skipped_count": 0,
            "untranslated_blocks_count": 0,
            "errors_count": 0,
            "total_llm_time_sec": 0.0,
        }
        self.recent_api_calls: list[dict[str, Any]] = []
        self.recent_translation_incidents: list[dict[str, Any]] = []
        self._load_saved_config()

    def reset_diagnostics(self) -> None:
        self.diagnostics_stats = {
            "session_started_at": int(time.time()),
            "total_llm_requests": 0,
            "total_ollama_requests": 0,
            "total_cloud_requests": 0,
            "parallel_batches_count": 0,
            "phrases_requested_total": 0,
            "phrases_sent_to_llm": 0,
            "phrases_saved_by_cache": 0,
            "refusals_count": 0,
            "local_censorship_fallbacks_count": 0,
            "sanitized_retries_count": 0,
            "micro_retries_count": 0,
            "network_retries_count": 0,
            "cloud_to_local_error_fallbacks_count": 0,
            "romaji_leaks_caught_count": 0,
            "cjk_leaks_caught_count": 0,
            "dict_interjections_used_count": 0,
            "sighs_skipped_count": 0,
            "untranslated_blocks_count": 0,
            "errors_count": 0,
            "total_llm_time_sec": 0.0,
        }
        self.recent_api_calls.clear()
        self.recent_translation_incidents.clear()

    def _record_incident(
        self,
        incident_type: str,
        model: str,
        summary_ru: str,
        details: str = "",
        affected_count: int = 0,
    ) -> None:
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": int(time.time()),
            "incident_type": incident_type,
            "model": model,
            "summary_ru": summary_ru,
            "details": (details or "")[:260],
            "affected_count": affected_count,
        }
        self.recent_translation_incidents.append(entry)
        if len(self.recent_translation_incidents) > 120:
            self.recent_translation_incidents = self.recent_translation_incidents[-120:]

    def _record_api_call(
        self,
        provider: str,
        model: str,
        elapsed_sec: float,
        returned_keys: int,
        status: str,
        note: str = "",
    ) -> None:
        self.diagnostics_stats["total_llm_requests"] += 1
        if provider == "ollama":
            self.diagnostics_stats["total_ollama_requests"] += 1
        else:
            self.diagnostics_stats["total_cloud_requests"] += 1
        self.diagnostics_stats["total_llm_time_sec"] = round(
            float(self.diagnostics_stats["total_llm_time_sec"]) + elapsed_sec, 3
        )
        if "REFUSAL" in status:
            self.diagnostics_stats["refusals_count"] += 1
        if "ERROR" in status:
            self.diagnostics_stats["errors_count"] += 1

        entry = {
            "time": time.strftime("%H:%M:%S"),
            "timestamp": int(time.time()),
            "provider": provider,
            "model": model,
            "elapsed_sec": round(elapsed_sec, 2),
            "returned_keys": returned_keys,
            "status": status,
            "note": (note or "")[:180],
        }
        self.recent_api_calls.append(entry)
        if len(self.recent_api_calls) > 60:
            self.recent_api_calls = self.recent_api_calls[-60:]

    def get_diagnostics_summary(self) -> dict[str, Any]:
        total_req = int(self.diagnostics_stats.get("total_llm_requests") or 0)
        tot_time = float(self.diagnostics_stats.get("total_llm_time_sec") or 0.0)
        avg_lat = round(tot_time / total_req, 2) if total_req > 0 else 0.0
        tot_phr = int(self.diagnostics_stats.get("phrases_requested_total") or 0)
        saved_phr = int(self.diagnostics_stats.get("phrases_saved_by_cache") or 0)
        cache_rate = round((saved_phr / tot_phr) * 100.0, 1) if tot_phr > 0 else 0.0
        masked_key = ""
        if self.unsloth_api_key:
            k = self.unsloth_api_key.strip()
            masked_key = f"{k[:5]}...{k[-4:]}" if len(k) > 10 else "***"
        return {
            **self.diagnostics_stats,
            "avg_llm_latency_sec": avg_lat,
            "cache_hit_rate_pct": cache_rate,
            "phrase_cache_size": len(self.phrase_cache),
            "glossary_size": len(self.glossary),
            "selected_model": self._resolved_model or "auto",
            "api_url": self.unsloth_url,
            "masked_api_key": masked_key,
            "text_filter_mode": self.text_filter_mode,
            "skip_sighs": self.skip_sighs,
            "skip_sfx": self.skip_sfx,
            "detect_free_text": self.detect_free_text,
            "min_text_length": self.min_text_length,
            "overflow_mode": self.overflow_mode,
            "batch_pages": self.batch_pages,
            "translation_profile": self.translation_profile,
            "local_fallback_model_multilang": self.local_fallback_model_multilang,
            "cloud_batch_limit": MAX_BUBBLES_PER_BATCH_CLOUD,
            "cloud_parallel_chunk_size": CLOUD_PARALLEL_CHUNK_SIZE,
            "last_local_fallback_model": self.last_local_fallback_model,
            "recent_api_calls": list(reversed(self.recent_api_calls[-30:])),
            "recent_translation_incidents": list(reversed(self.recent_translation_incidents[-50:])),
        }

    @staticmethod
    def _normalize_api_base_url(raw_url: str) -> str:
        u = (raw_url or "").strip().rstrip("/")
        if not u:
            return "http://127.0.0.1:8888"
        if not u.startswith(("http://", "https://")):
            is_local = u.startswith(("127.0.0.1", "localhost", "0.0.0.0", "192.168.", "10."))
            u = f"http://{u}" if is_local else f"https://{u}"
        for suffix in (
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/models",
            "/api/v1/models",
            "/api/models",
            "/v1",
        ):
            if u.endswith(suffix):
                u = u[: -len(suffix)].rstrip("/")
                break
        return u

    @staticmethod
    def _is_remote_cloud_url(url: str) -> bool:
        u = url.lower()
        return not any(loc in u for loc in ("127.0.0.1", "localhost", "0.0.0.0", "192.168."))

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_sec, connect=15.0),
                trust_env=False,
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
        self._http_client = None

    def _load_saved_config(self) -> None:
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                if data.get("unsloth_url"):
                    self.unsloth_url = self._normalize_api_base_url(str(data["unsloth_url"]))
                if "unsloth_api_key" in data:
                    self.unsloth_api_key = str(data["unsloth_api_key"]).strip()
                saved_model = str(data.get("selected_model") or "").strip()
                if saved_model.startswith("unsloth::ollama/"):
                    saved_model = "ollama::" + saved_model.split("unsloth::ollama/", 1)[1]
                if saved_model and saved_model != "auto":
                    self._resolved_model = saved_model
                if "auto_save_web_pages" in data and data["auto_save_web_pages"] is not None:
                    self.auto_save_web_pages = bool(data["auto_save_web_pages"])
                if data.get("web_save_dir"):
                    self.web_save_dir = str(data["web_save_dir"]).strip()
                if data.get("text_filter_mode"):
                    self.text_filter_mode = str(data["text_filter_mode"]).strip()
                if "skip_sighs" in data and data["skip_sighs"] is not None:
                    self.skip_sighs = bool(data["skip_sighs"])
                if "skip_sfx" in data and data["skip_sfx"] is not None:
                    self.skip_sfx = bool(data["skip_sfx"])
                if "detect_free_text" in data and data["detect_free_text"] is not None:
                    self.detect_free_text = bool(data["detect_free_text"])
                if "min_text_length" in data and data["min_text_length"] is not None:
                    self.min_text_length = max(1, min(20, int(data["min_text_length"])))
                if "min_font_size" in data and data["min_font_size"] is not None:
                    self.min_font_size = max(8, min(48, int(data["min_font_size"])))
                if "max_font_size" in data and data["max_font_size"] is not None:
                    self.max_font_size = max(12, min(72, int(data["max_font_size"])))
                if "uppercase" in data and data["uppercase"] is not None:
                    self.uppercase = bool(data["uppercase"])
                if data.get("overflow_mode") and str(data["overflow_mode"]).strip().lower() in ("shrink", "overflow", "expand_bubble"):
                    self.overflow_mode = str(data["overflow_mode"]).strip().lower()
                if "batch_pages" in data and data["batch_pages"] is not None:
                    self.batch_pages = max(1, min(50, int(data["batch_pages"])))
                if data.get("translation_profile") and str(data["translation_profile"]).strip().lower() in TRANSLATION_PROFILES:
                    self.translation_profile = str(data["translation_profile"]).strip().lower()
                if data.get("local_fallback_model_multilang"):
                    self.local_fallback_model_multilang = str(data["local_fallback_model_multilang"]).strip()
                if isinstance(data.get("ui_settings"), dict):
                    self.ui_settings.update(data["ui_settings"])
                if isinstance(data.get("glossary"), list):
                    self.glossary = normalize_glossary_entries(data["glossary"])
                # Автоматическая миграция старых настроек фильтрации (v1.0.9), чтобы реплики внутри бабблов не пропускались без ведома пользователя
                if not data.get("filter_v109_migrated"):
                    self.text_filter_mode = "bubbles_only"
                    self.skip_sighs = False
                    self.min_text_length = 1
                    self._filter_v109_migrated = True
                    try:
                        data["text_filter_mode"] = "bubbles_only"
                        data["skip_sighs"] = False
                        data["min_text_length"] = 1
                        data["filter_v109_migrated"] = True
                        self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("Не удалось прочитать config.json: %s", exc)

    def get_profile_config(self, profile_id: str | None = None) -> dict[str, Any]:
        pid = (profile_id or self.translation_profile or "balanced").strip().lower()
        return dict(TRANSLATION_PROFILES.get(pid, TRANSLATION_PROFILES["balanced"]))

    def save_config(
        self,
        unsloth_url: str | None = None,
        unsloth_api_key: str | None = None,
        selected_model: str | None = None,
        translation_profile: str | None = None,
        local_fallback_model_multilang: str | None = None,
        auto_save_web_pages: bool | None = None,
        web_save_dir: str | None = None,
        text_filter_mode: str | None = None,
        skip_sighs: bool | None = None,
        skip_sfx: bool | None = None,
        detect_free_text: bool | None = None,
        min_text_length: int | None = None,
        min_font_size: int | None = None,
        max_font_size: int | None = None,
        uppercase: bool | None = None,
        overflow_mode: str | None = None,
        batch_pages: int | None = None,
        ui_settings: dict[str, Any] | None = None,
        glossary: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        prev_url = self.unsloth_url
        prev_key = self.unsloth_api_key

        if unsloth_url is not None and unsloth_url.strip():
            self.unsloth_url = self._normalize_api_base_url(unsloth_url)

        if unsloth_api_key is not None:
            self.unsloth_api_key = unsloth_api_key.strip()

        if selected_model is not None:
            clean_sel = selected_model.strip()
            if clean_sel.startswith("unsloth::ollama/"):
                clean_sel = "ollama::" + clean_sel.split("unsloth::ollama/", 1)[1]
            self._resolved_model = None if clean_sel in ("", "auto") else clean_sel

        if translation_profile is not None:
            clean_prof = str(translation_profile).strip().lower()
            if clean_prof in TRANSLATION_PROFILES:
                self.translation_profile = clean_prof
                if batch_pages is None:
                    self.batch_pages = int(TRANSLATION_PROFILES[clean_prof]["pages_per_batch"])

        if local_fallback_model_multilang is not None and str(local_fallback_model_multilang).strip():
            self.local_fallback_model_multilang = str(local_fallback_model_multilang).strip()

        if auto_save_web_pages is not None:
            self.auto_save_web_pages = bool(auto_save_web_pages)

        if web_save_dir is not None:
            clean_dir = web_save_dir.strip()
            self.web_save_dir = (
                str(Path(clean_dir).resolve())
                if clean_dir
                else str(self.config_path.resolve().parent / "Saved_Manga")
            )

        if text_filter_mode is not None and text_filter_mode.strip():
            self.text_filter_mode = text_filter_mode.strip()

        if skip_sighs is not None:
            self.skip_sighs = bool(skip_sighs)

        if skip_sfx is not None:
            self.skip_sfx = bool(skip_sfx)

        if detect_free_text is not None:
            self.detect_free_text = bool(detect_free_text)

        if min_text_length is not None:
            self.min_text_length = max(1, min(20, int(min_text_length)))

        if min_font_size is not None:
            self.min_font_size = max(8, min(48, int(min_font_size)))

        if max_font_size is not None:
            self.max_font_size = max(12, min(72, int(max_font_size)))

        if uppercase is not None:
            self.uppercase = bool(uppercase)

        if overflow_mode is not None and str(overflow_mode).strip().lower() in ("shrink", "overflow", "expand_bubble"):
            self.overflow_mode = str(overflow_mode).strip().lower()

        if batch_pages is not None:
            self.batch_pages = max(1, min(50, int(batch_pages)))

        if isinstance(ui_settings, dict):
            self.ui_settings.update(ui_settings)

        if glossary is not None and isinstance(glossary, list):
            self.glossary = normalize_glossary_entries(glossary)

        # БЛОК 5: Сбрасывать кэш моделей ТОЛЬКО при изменении api_url или api_key
        if self.unsloth_url != prev_url or self.unsloth_api_key != prev_key:
            self._cache_timestamp = 0.0
            self._cached_cloud_probe.clear()

        cfg: dict[str, Any] = {
            "filter_v109_migrated": True,
            "unsloth_url": self.unsloth_url,
            "unsloth_api_key": self.unsloth_api_key,
            "selected_model": self._resolved_model or "auto",
            "translation_profile": self.translation_profile,
            "local_fallback_model_multilang": self.local_fallback_model_multilang,
            "auto_save_web_pages": self.auto_save_web_pages,
            "web_save_dir": self.web_save_dir,
            "text_filter_mode": self.text_filter_mode,
            "skip_sighs": self.skip_sighs,
            "skip_sfx": self.skip_sfx,
            "detect_free_text": self.detect_free_text,
            "min_text_length": self.min_text_length,
            "min_font_size": self.min_font_size,
            "max_font_size": self.max_font_size,
            "uppercase": self.uppercase,
            "overflow_mode": self.overflow_mode,
            "batch_pages": self.batch_pages,
            "ui_settings": self.ui_settings,
            "glossary": self.glossary,
        }
        try:
            self.config_path.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(
                "Настройки сохранены в config.json: selected_model=%s, api_url=%s, auto_save_web=%s, web_save_dir=%s",
                cfg["selected_model"],
                cfg["unsloth_url"],
                cfg["auto_save_web_pages"],
                cfg["web_save_dir"],
            )
        except Exception as exc:
            logger.warning("Не удалось сохранить config.json: %s", exc)
        return cfg

    def _unsloth_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "MikoMangaStudio/1.0.8",
        }
        if self.unsloth_api_key:
            headers["Authorization"] = f"Bearer {self.unsloth_api_key}"
        return headers

    @staticmethod
    def _list_ollama_cli_models() -> list[str]:
        try:
            proc = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=2.5,
            )
            if proc.returncode != 0:
                return []
            names: list[str] = []
            for line in proc.stdout.splitlines()[1:]:
                parts = line.strip().split()
                if parts:
                    names.append(parts[0])
            return names
        except Exception:
            return []

    @staticmethod
    def _extract_model_ids_from_payload(data: Any) -> list[str]:
        ids: list[str] = []

        def _add(val: Any) -> None:
            if isinstance(val, str):
                s = val.strip()
                if s and s not in ids:
                    ids.append(s)
            elif isinstance(val, dict):
                for key in ("id", "name", "model", "model_name", "path", "repo_id"):
                    v = val.get(key)
                    if isinstance(v, str) and v.strip():
                        s = v.strip()
                        if s not in ids:
                            ids.append(s)
                        break

        if isinstance(data, list):
            for item in data:
                _add(item)
        elif isinstance(data, dict):
            for list_key in ("data", "models", "items", "available_models", "loaded_models"):
                sub = data.get(list_key)
                if isinstance(sub, list):
                    for item in sub:
                        _add(item)
            for single_key in ("model", "model_name", "current_model", "active_model", "id"):
                if isinstance(data.get(single_key), str):
                    _add(data[single_key])

        return ids

    @staticmethod
    def _prioritize_cloud_models(model_ids: list[str]) -> list[str]:
        """Оставляет и сортирует только рабочие модели DeepSeek (Flash и Pro), убирая лишние модели шлюза."""
        deepseek_only = [m for m in model_ids if "deepseek" in m.lower()]
        pool = deepseek_only if deepseek_only else model_ids

        def _score(m_id: str) -> tuple[int, str]:
            low = m_id.lower()
            if "deepseek-v4-flash" in low:
                return (0, m_id)
            if "deepseek-v4-pro" in low:
                return (1, m_id)
            if "deepseek-chat" in low or "deepseek-v3" in low:
                return (2, m_id)
            if "deepseek" in low:
                return (3, m_id)
            return (10, m_id)

        return sorted(pool, key=_score)

    async def _probe_unsloth_url(self, u_url: str, force_refresh: bool = False) -> tuple[str, list[str], bool]:
        now = time.monotonic()
        cache_k = f"{u_url}|{self.unsloth_api_key}"
        if not force_refresh and cache_k in self._cached_cloud_probe:
            c_ts, c_val = self._cached_cloud_probe[cache_k]
            if (now - c_ts) < MODEL_LIST_CACHE_TTL_SEC:
                return c_val
        is_remote = self._is_remote_cloud_url(u_url)
        probe_timeout = httpx.Timeout(8.0, connect=5.0) if is_remote else httpx.Timeout(3.0, connect=0.8)
        client = self._get_http_client()
        port_alive = False
        found_ids: list[str] = []

        # Для облачных OpenAI-шлюзов (ModuleAPI, PositionAPI, DeepSeek) опрашиваем /v1/models без лишних 404/HTML-запросов
        paths_to_try = ("/v1/models",) if is_remote else ("/v1/models", "/models", "/api/v1/models", "/api/models")
        for path in paths_to_try:
            try:
                resp = await client.get(
                    f"{u_url}{path}",
                    headers=self._unsloth_headers(),
                    timeout=probe_timeout,
                )
                if resp.status_code < 500:
                    port_alive = True
                if resp.status_code in (401, 403) and is_remote:
                    # Шлюз отвечает, но /v1/models закрыт или требует ключ при chat/completions — возвращаем рабочие пресеты без спама 404
                    self._cached_cloud_probe[cache_k] = (time.monotonic(), (u_url, ["deepseek-v4-flash", "deepseek-v4-pro"], True)); return u_url, ["deepseek-v4-flash", "deepseek-v4-pro"], True
                if resp.status_code == 200:
                    ctype = (resp.headers.get("content-type") or "").lower()
                    if "text/html" in ctype and is_remote:
                        continue
                    try:
                        data = resp.json()
                        extracted = self._extract_model_ids_from_payload(data)
                        for m_id in extracted:
                            if m_id not in found_ids:
                                found_ids.append(m_id)
                        if found_ids:
                            if is_remote:
                                found_ids = self._prioritize_cloud_models(found_ids)
                            self._cached_cloud_probe[cache_k] = (time.monotonic(), (u_url, found_ids, True)); return u_url, found_ids, True
                        if is_remote and path == "/v1/models":
                            self._cached_cloud_probe[cache_k] = (time.monotonic(), (u_url, ["deepseek-v4-flash", "deepseek-v4-pro"], True)); return u_url, ["deepseek-v4-flash", "deepseek-v4-pro"], True
                    except Exception:
                        if is_remote and path == "/v1/models":
                            self._cached_cloud_probe[cache_k] = (time.monotonic(), (u_url, ["deepseek-v4-flash", "deepseek-v4-pro"], True)); return u_url, ["deepseek-v4-flash", "deepseek-v4-pro"], True
            except Exception:
                continue

        if not port_alive and not is_remote:
            try:
                resp = await client.get(f"{u_url}/", timeout=httpx.Timeout(1.5, connect=0.6))
                if resp.status_code < 500:
                    port_alive = True
            except Exception:
                pass

        res_tuple = (u_url, found_ids, port_alive)
        self._cached_cloud_probe[cache_k] = (time.monotonic(), res_tuple)
        return res_tuple

    @staticmethod
    def _resolve_gguf_fallback_info(clean_tag: str) -> tuple[str, str, str, str] | None:
        """
        Возвращает (hf_repo, hf_filename, hf_quant, local_ollama_tag) для прямого скачивания GGUF через Python,
        если Ollama блокирует редирект из-за Fake-IP VPN (198.18.x.x).
        """
        norm = clean_tag.strip()
        for rec in RECOMMENDED_LOCAL_MODELS:
            if norm.lower() in (rec["ollama_tag"].lower(), rec.get("local_tag", "").lower()):
                return (
                    rec["hf_repo"],
                    rec["hf_filename"],
                    rec.get("hf_quant", "Q4_K_M"),
                    rec.get("local_tag", rec["ollama_tag"]),
                )

        if norm.lower().startswith("hf.co/"):
            rest = norm[6:]
            if ":" in rest:
                repo_part, quant_part = rest.rsplit(":", 1)
            else:
                repo_part, quant_part = rest, "Q4_K_M"
            short_repo = repo_part.split("/")[-1]
            safe_local = re.sub(r"-gguf$", "", short_repo, flags=re.IGNORECASE).lower()
            local_tag = f"{safe_local}:{quant_part}"
            guessed_file = f"{ safe_local }.{quant_part}.gguf"
            return (repo_part, guessed_file, quant_part, local_tag)

        return None

    async def _pull_gguf_via_python_and_create(
        self,
        clean_tag: str,
        set_as_default: bool = True,
        reason: str = "",
    ) -> dict[str, Any]:
        """
        Обход ошибки Ollama `redirect target not allowed: ... resolves to non-public 198.18.x.x` (Fake-IP VPN).
        Скачивает `.gguf` напрямую средствами Python (`httpx` не блокирует 198.18.x.x) с поддержкой докачки
        и регистрирует модель локально через `ollama create <local_tag> -f Modelfile`.
        """
        info = self._resolve_gguf_fallback_info(clean_tag)
        if not info:
            raise RuntimeError(
                f"Ollama отклонил редирект ({reason}), а прямой GGUF-репозиторий для '{clean_tag}' не найден."
            )

        hf_repo, default_filename, hf_quant, local_tag = info
        self.gguf_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[GGUF Python Bypass] Обнаружена блокировка редиректа Fake-IP (198.18.x.x) в Ollama!\n"
            f"[GGUF Python Bypass] Включаю прямое скачивание GGUF через Python из репозитория {hf_repo}...",
            flush=True,
        )
        self.pull_state.update(
            {
                "status": "downloading",
                "model": local_tag,
                "message": f"Обход блокировки VPN (198.18.x.x): поиск GGUF в {hf_repo}...",
                "error": None,
            }
        )

        target_filename = default_filename
        api_url = f"https://huggingface.co/api/models/{hf_repo}"
        dl_headers = {"User-Agent": "MikoMangaStudio/1.0.8 (Python-GGUF-Bypass)"}
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            dl_headers["Authorization"] = f"Bearer {hf_token}"

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, connect=8.0),
                follow_redirects=True,
                trust_env=False,
                headers=dl_headers,
            ) as meta_client:
                meta_resp = await meta_client.get(api_url)
                if meta_resp.status_code == 200:
                    siblings = meta_resp.json().get("siblings", [])
                    gguf_files = [
                        str(s.get("rfilename"))
                        for s in siblings
                        if isinstance(s, dict) and str(s.get("rfilename", "")).lower().endswith(".gguf")
                    ]
                    if gguf_files:
                        exact = [f for f in gguf_files if f.lower() == default_filename.lower()]
                        if exact:
                            target_filename = exact[0]
                        else:
                            q_low = hf_quant.lower()
                            matched = [f for f in gguf_files if q_low in f.lower()]
                            target_filename = matched[0] if matched else gguf_files[0]
        except Exception as meta_exc:
            logger.debug("Не удалось уточнить список файлов через HF API (%s), используем %s", meta_exc, target_filename)

        final_gguf_path = self.gguf_dir / Path(target_filename).name
        part_gguf_path = self.gguf_dir / f"{Path(target_filename).name}.part"
        download_url = f"https://huggingface.co/{hf_repo}/resolve/main/{target_filename}"

        if not (final_gguf_path.exists() and final_gguf_path.stat().st_size > 100 * 1024 * 1024):
            resume_bytes = part_gguf_path.stat().st_size if part_gguf_path.exists() else 0
            req_headers = dict(dl_headers)
            if resume_bytes > 0:
                req_headers["Range"] = f"bytes={resume_bytes}-"
                print(
                    f"[GGUF Python Bypass] Продолжаю докачку файла {final_gguf_path.name} с {round(resume_bytes / (1024**3), 2)} ГБ...",
                    flush=True,
                )

            stream_timeout = httpx.Timeout(7200.0, connect=20.0, read=120.0)
            async with httpx.AsyncClient(
                timeout=stream_timeout,
                follow_redirects=True,
                trust_env=False,
            ) as dl_client:
                async with dl_client.stream("GET", download_url, headers=req_headers) as resp:
                    if resp.status_code == 416 and part_gguf_path.exists():
                        part_gguf_path.replace(final_gguf_path)
                    else:
                        if resp.status_code not in (200, 206):
                            resp.raise_for_status()

                        is_resumed = resp.status_code == 206 and resume_bytes > 0
                        content_len = int(resp.headers.get("Content-Length") or 0)
                        total_bytes = (resume_bytes + content_len) if is_resumed else content_len
                        completed_bytes = resume_bytes if is_resumed else 0
                        mode = "ab" if is_resumed else "wb"

                        last_ui_ts = 0.0
                        last_logged_pct = -5

                        with open(part_gguf_path, mode) as f_out:
                            async for chunk in resp.aiter_bytes(chunk_size=512 * 1024):
                                if not chunk:
                                    continue
                                f_out.write(chunk)
                                completed_bytes += len(chunk)
                                now = time.monotonic()
                                if total_bytes > 0 and (now - last_ui_ts >= 0.35 or completed_bytes >= total_bytes):
                                    last_ui_ts = now
                                    pct = round(min(99.5, (completed_bytes / total_bytes) * 100.0), 1)
                                    comp_gb = round(completed_bytes / (1024**3), 2)
                                    tot_gb = round(total_bytes / (1024**3), 2)
                                    self.pull_state.update(
                                        {
                                            "progress_pct": pct,
                                            "completed_gb": comp_gb,
                                            "total_gb": tot_gb,
                                            "message": f"Скачивание GGUF через Python (обход 198.18.x.x): {pct}% ({comp_gb} / {tot_gb} ГБ)",
                                        }
                                    )
                                    if int(pct) >= last_logged_pct + 5:
                                        last_logged_pct = int(pct)
                                        print(
                                            f"[GGUF Python Bypass] {final_gguf_path.name} -> {pct}% ({comp_gb}/{tot_gb} ГБ)",
                                            flush=True,
                                        )

                        part_gguf_path.replace(final_gguf_path)

        # Импортируем скачанный локальный .gguf файл в Ollama через `ollama create`
        self.pull_state.update(
            {
                "progress_pct": 99.8,
                "message": f"Импорт {final_gguf_path.name} в локальный Ollama (ollama create {local_tag})...",
            }
        )
        print(
            f"[GGUF Python Bypass] Файл скачан ({final_gguf_path}). Создаю модель '{local_tag}' в Ollama...",
            flush=True,
        )

        safe_modelfile_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", local_tag)
        modelfile_path = self.gguf_dir / f"Modelfile.{safe_modelfile_name}"
        gguf_abs_str = str(final_gguf_path.resolve()).replace("\\", "/")
        modelfile_content = (
            f'FROM "{gguf_abs_str}"\n'
            f"PARAMETER temperature 0.2\n"
            f"PARAMETER repeat_penalty 1.15\n"
            f"PARAMETER num_ctx {self.num_ctx}\n"
        )
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        proc = await asyncio.to_thread(
            subprocess.run,
            ["ollama", "create", local_tag, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=600,
        )
        if proc.returncode != 0:
            err_out = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Ошибка `ollama create {local_tag}`: {err_out}")

        full_id = f"ollama::{local_tag}"
        is_pure_ocr = "glm-ocr" in local_tag.lower()
        if set_as_default and not is_pure_ocr:
            self.save_config(selected_model=full_id)
        else:
            self._cache_timestamp = 0.0

        await self.list_available_models(force_refresh=True)
        self.pull_state.update(
            {
                "status": "completed",
                "model": local_tag,
                "progress_pct": 100.0,
                "message": f"Модель {local_tag} успешно скачана (обход VPN) и установлена по умолчанию!",
                "error": None,
            }
        )
        print(
            f"[GGUF Python Bypass] УСПЕХ! Модель '{local_tag}' импортирована в Ollama и выбрана по умолчанию.\n",
            flush=True,
        )
        return self.pull_state

    async def pull_ollama_model(
        self,
        model_name: str,
        set_as_default: bool = True,
    ) -> dict[str, Any]:
        """
        Скачивает модель через локальный сервер Ollama (`POST /api/pull`).
        Если Ollama выдает ошибку `redirect target not allowed: ... resolves to non-public 198.18.x.x`
        из-за Fake-IP режима VPN, автоматически переключается на прямое скачивание `.gguf` через Python
        с последующим локальным импортом через `ollama create`!
        """
        clean_tag = model_name.strip()
        if clean_tag.startswith("ollama::"):
            clean_tag = clean_tag.split("ollama::", 1)[1]
        if clean_tag.startswith("ollama/"):
            clean_tag = clean_tag.split("ollama/", 1)[1]

        self.pull_state = {
            "status": "downloading",
            "model": clean_tag,
            "progress_pct": 0.0,
            "completed_gb": 0.0,
            "total_gb": 0.0,
            "message": f"Подключение к Ollama для скачивания {clean_tag}...",
            "error": None,
        }
        print(f"\n[Ollama Pull] Начинаю скачивание модели: {clean_tag}", flush=True)

        # Мгновенная проверка Fake-IP (198.18.x.x / 198.19.x.x) от TUN/VPN, чтобы не ждать 65 секунд таймаута Ollama
        if self._resolve_gguf_fallback_info(clean_tag) is not None:
            fake_ip_detected = False
            resolved_ip = ""
            for test_host in ("us.aws.cdn.hf.co", "huggingface.co", "registry.ollama.ai"):
                try:
                    ip = socket.gethostbyname(test_host)
                    if ip.startswith(("198.18.", "198.19.", "10.", "127.")):
                        fake_ip_detected = True
                        resolved_ip = f"{test_host} -> {ip}"
                        break
                except Exception:
                    pass
            if fake_ip_detected:
                print(
                    f"[GGUF Python Bypass] Обнаружен Fake-IP DNS ({resolved_ip}). "
                    f"Пропускаю ожидание ошибки Ollama и сразу качаю GGUF напрямую через Python!",
                    flush=True,
                )
                try:
                    return await self._pull_gguf_via_python_and_create(
                        clean_tag=clean_tag,
                        set_as_default=set_as_default,
                        reason=f"Fake-IP DNS ({resolved_ip})",
                    )
                except Exception as direct_exc:
                    logger.exception("Ошибка прямого скачивания GGUF: %s", direct_exc)
                    self.pull_state.update(
                        {
                            "status": "error",
                            "message": f"Ошибка скачивания {clean_tag}: {direct_exc}",
                            "error": str(direct_exc),
                        }
                    )
                    return self.pull_state

        try:
            pull_timeout = httpx.Timeout(3600.0, connect=10.0)
            async with httpx.AsyncClient(timeout=pull_timeout, trust_env=False) as pull_client:
                async with pull_client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"model": clean_tag, "stream": True},
                ) as resp:
                    resp.raise_for_status()
                    last_logged_pct = -5
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if evt.get("error"):
                            raise RuntimeError(str(evt["error"]))

                        status_text = str(evt.get("status") or "Скачивание...")
                        total = int(evt.get("total") or 0)
                        completed = int(evt.get("completed") or 0)

                        if total > 0:
                            pct = round(min(100.0, (completed / total) * 100.0), 1)
                            comp_gb = round(completed / (1024**3), 2)
                            tot_gb = round(total / (1024**3), 2)
                            self.pull_state.update(
                                {
                                    "progress_pct": pct,
                                    "completed_gb": comp_gb,
                                    "total_gb": tot_gb,
                                    "message": f"{status_text}: {pct}% ({comp_gb} / {tot_gb} ГБ)",
                                }
                            )
                            if int(pct) >= last_logged_pct + 5:
                                last_logged_pct = int(pct)
                                print(
                                    f"[Ollama Pull] {clean_tag} -> {pct}% ({comp_gb}/{tot_gb} ГБ)",
                                    flush=True,
                                )
                        else:
                            self.pull_state["message"] = status_text

            full_id = f"ollama::{clean_tag}"
            is_pure_ocr = "glm-ocr" in clean_tag.lower()
            if set_as_default and not is_pure_ocr:
                self.save_config(selected_model=full_id)
            else:
                self._cache_timestamp = 0.0

            await self.list_available_models(force_refresh=True)
            done_msg = (
                f"Vision-OCR модель {clean_tag} успешно скачана и готова для VLM-OCR!"
                if is_pure_ocr
                else f"Модель {clean_tag} успешно скачана и установлена по умолчанию!"
            )
            self.pull_state.update(
                {
                    "status": "completed",
                    "progress_pct": 100.0,
                    "message": done_msg,
                }
            )
            print(f"[Ollama Pull] УСПЕХ! Модель {clean_tag} готова и выбрана по умолчанию.\n", flush=True)
            return self.pull_state

        except Exception as exc:
            err_str = str(exc)
            # Если Ollama заблокировал скачивание из-за Fake-IP VPN (198.18.x.x) или сбоя редиректа Hugging Face:
            if self._resolve_gguf_fallback_info(clean_tag) is not None:
                logger.warning(
                    "Стандартный `ollama pull %s` завершился ошибкой (%s). Пробую прямое скачивание GGUF через Python...",
                    clean_tag,
                    err_str,
                )
                try:
                    return await self._pull_gguf_via_python_and_create(
                        clean_tag=clean_tag,
                        set_as_default=set_as_default,
                        reason=err_str,
                    )
                except Exception as fallback_exc:
                    logger.exception(
                        "Ошибка резервного скачивания GGUF модели %s: %s",
                        clean_tag,
                        fallback_exc,
                    )
                    err_str = f"{fallback_exc} (исходная ошибка Ollama: {err_str})"

            logger.exception("Ошибка скачивания модели %s через Ollama: %s", clean_tag, err_str)
            self.pull_state.update(
                {
                    "status": "error",
                    "message": f"Ошибка скачивания {clean_tag}: {err_str}",
                    "error": err_str,
                }
            )
            return self.pull_state

    async def list_available_models(self, force_refresh: bool = False) -> list[dict[str, str]]:
        now = time.monotonic()
        if force_refresh:
            self._cached_cloud_probe.clear()
            self._cached_ollama_tags = None
        if not force_refresh and self._cached_models and (now - self._cache_timestamp) < MODEL_LIST_CACHE_TTL_SEC:
            return self._cached_models

        models_list: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        installed_ollama_tags: set[str] = set()
        client = self._get_http_client()

        # 1. Опрашиваем локальный Ollama (127.0.0.1:11434) с in-memory кэшем TTL = 300с
        ollama_names_cached: list[str] | None = None
        if not force_refresh and self._cached_ollama_tags is not None:
            o_ts, o_names = self._cached_ollama_tags
            if (now - o_ts) < MODEL_LIST_CACHE_TTL_SEC:
                ollama_names_cached = o_names

        if ollama_names_cached is not None:
            for m_name in ollama_names_cached:
                installed_ollama_tags.add(m_name)
                installed_ollama_tags.add(m_name.lower())
                full_id = f"ollama::{m_name}"
                if full_id not in seen_ids:
                    seen_ids.add(full_id)
                    badge = "🌸 [Shisa / Ollama]" if "shisa" in m_name.lower() else "🦙 [Ollama]"
                    models_list.append(
                        {
                            "id": full_id,
                            "name": f"{badge} {m_name} (Установлена ✅)",
                            "provider": "ollama",
                            "installed": "true",
                        }
                    )
        else:
            fetched_ollama_names: list[str] = []
            try:
                ollama_timeout = httpx.Timeout(2.5, connect=0.8)
                resp = await client.get(f"{self.base_url}/api/tags", timeout=ollama_timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("models", []):
                        m_name = item.get("name")
                        if m_name:
                            fetched_ollama_names.append(m_name)
                            installed_ollama_tags.add(m_name)
                            installed_ollama_tags.add(m_name.lower())
                            full_id = f"ollama::{m_name}"
                            if full_id not in seen_ids:
                                seen_ids.add(full_id)
                                badge = "🌸 [Shisa / Ollama]" if "shisa" in m_name.lower() else "🦙 [Ollama]"
                                models_list.append(
                                    {
                                        "id": full_id,
                                        "name": f"{badge} {m_name} (Установлена ✅)",
                                        "provider": "ollama",
                                        "installed": "true",
                                    }
                                )
                    self._cached_ollama_tags = (time.monotonic(), fetched_ollama_names)
            except Exception as exc:
                logger.debug("Ollama HTTP /api/tags недоступен: %s", exc)

        if not installed_ollama_tags:
            for cli_name in self._list_ollama_cli_models():
                installed_ollama_tags.add(cli_name)
                installed_ollama_tags.add(cli_name.lower())
                full_id = f"ollama::{cli_name}"
                if full_id not in seen_ids:
                    seen_ids.add(full_id)
                    badge = "🌸 [Shisa / Ollama]" if "shisa" in cli_name.lower() else "🦙 [Ollama]"
                    models_list.append(
                        {
                            "id": full_id,
                            "name": f"{badge} {cli_name} (Установлена ✅)",
                            "provider": "ollama",
                            "installed": "true",
                        }
                    )

        # 2. Проверяем внешний Облачный API (ModuleAPI / PositionAPI / ListAPI / DeepSeek / OpenRouter) или локальный Unsloth Studio
        is_remote_api = self._is_remote_cloud_url(self.unsloth_url)
        if is_remote_api:
            unsloth_urls_to_try = [self.unsloth_url]
        else:
            unsloth_urls_to_try = [self.unsloth_url]
            for fallback_url in ("http://127.0.0.1:8888", "http://127.0.0.1:8000", "http://127.0.0.1:1234"):
                if fallback_url not in unsloth_urls_to_try:
                    unsloth_urls_to_try.append(fallback_url)

        probe_results = await asyncio.gather(
            *(self._probe_unsloth_url(u, force_refresh=force_refresh) for u in unsloth_urls_to_try)
        )

        unsloth_reachable = False
        detected_unsloth_ids: list[str] = []

        for u_url, m_ids, is_alive in probe_results:
            if m_ids:
                self.unsloth_url = u_url
                unsloth_reachable = True
                icon_prefix = "☁️ [Cloud API]" if self._is_remote_cloud_url(u_url) else "🚀 [Unsloth]"
                for m_id in m_ids:
                    if not self._is_remote_cloud_url(u_url) and m_id.lower().startswith("ollama/"):
                        raw_ollama_name = m_id.split("/", 1)[1]
                        ollama_id = f"ollama::{raw_ollama_name}"
                        if ollama_id not in seen_ids:
                            seen_ids.add(ollama_id)
                            installed_ollama_tags.add(raw_ollama_name)
                            installed_ollama_tags.add(raw_ollama_name.lower())
                            models_list.insert(
                                0,
                                {
                                    "id": ollama_id,
                                    "name": f"🦙 [Ollama] {raw_ollama_name} (Установлена ✅)",
                                    "provider": "ollama",
                                    "installed": "true",
                                },
                            )
                        continue

                    detected_unsloth_ids.append(m_id)
                    full_id = f"unsloth::{m_id}"
                    if full_id not in seen_ids:
                        seen_ids.add(full_id)
                        models_list.append(
                            {
                                "id": full_id,
                                "name": f"{icon_prefix} {m_id}",
                                "provider": "unsloth",
                                "installed": "true",
                            }
                        )
                break
            elif is_alive and not unsloth_reachable:
                self.unsloth_url = u_url
                unsloth_reachable = True

        # Всегда гарантируем наличие двух основных пресетов DeepSeek (Flash и Pro) в списке при использовании внешнего API
        if is_remote_api:
            for cloud_m, label_m in (
                ("deepseek-v4-flash", "⚡ [DeepSeek API] deepseek-v4-flash (Flash — быстрая)"),
                ("deepseek-v4-pro", "🧠 [DeepSeek API] deepseek-v4-pro (Pro — макс. качество)"),
            ):
                cid = f"unsloth::{cloud_m}"
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    models_list.insert(
                        0 if cloud_m == "deepseek-v4-flash" else min(1, len(models_list)),
                        {
                            "id": cid,
                            "name": label_m,
                            "provider": "unsloth",
                            "installed": "true",
                        },
                    )

        # 3. Добавляем рекомендуемые локальные модели (если они еще не установлены в Ollama)
        for rec in RECOMMENDED_LOCAL_MODELS:
            rec_id = rec["id"]
            rec_tag = rec["ollama_tag"]
            loc_tag = rec.get("local_tag", "")
            already_in = (
                rec_tag.lower() in installed_ollama_tags
                or (loc_tag and loc_tag.lower() in installed_ollama_tags)
            )
            if rec_id not in seen_ids and not already_in:
                seen_ids.add(rec_id)
                models_list.append(
                    {
                        "id": rec_id,
                        "name": f"{rec['title']} [📥 Скачать в Ollama]",
                        "provider": "ollama",
                        "installed": "false",
                        "size_gb": rec.get("size_gb", ""),
                    }
                )

        self.last_unsloth_status = {
            "reachable": unsloth_reachable,
            "url": self.unsloth_url,
            "is_cloud": is_remote_api,
            "detected_models": detected_unsloth_ids,
        }

        if self._resolved_model and self._resolved_model not in ("", "auto") and self._resolved_model not in seen_ids:
            prov = "unsloth" if self._resolved_model.startswith("unsloth::") else "ollama"
            short = self._resolved_model.split("::", 1)[-1]
            icon = (
                "☁️ [Cloud API]"
                if (prov == "unsloth" and is_remote_api)
                else ("🚀 [Unsloth]" if prov == "unsloth" else "🦙 [Ollama]")
            )
            models_list.insert(
                0,
                {
                    "id": self._resolved_model,
                    "name": f"{icon} {short}",
                    "provider": prov,
                    "installed": "true",
                },
            )

        self._cached_models = models_list
        self._cache_timestamp = time.monotonic()
        return models_list

    @classmethod
    def _pick_best_default_model(
        cls,
        available: list[dict[str, str]],
        fallback_name: str,
        has_cloud_key: bool = False,
        is_cloud_url: bool = False,
    ) -> str:
        """
        Приоритет в режиме 'auto':
        1. Если пользователем прописан облачный ключ (ModuleAPI / PositionAPI / DeepSeek API) и внешний URL —
           выбираем лучшую модель DeepSeek (`deepseek-v4-pro` -> `deepseek-v4-flash` -> `deepseek-chat`) или первую облачную.
        2. Если в Ollama уже скачана любая модель семейства Shisa (`shisa-v2.1` / `shisa-v2`) — ставим Shisa!
        3. Далее: `gemma3` -> `qwen2.5:14b` -> `qwen3.5` -> `qwen2.5` -> любая установленная модель Ollama.
        """
        if has_cloud_key and is_cloud_url:
            cloud_models = [
                m for m in available if m["provider"] == "unsloth" and m["id"] != "unsloth::default"
            ]
            for pref in ("deepseek-v4-flash", "deepseek-chat", "deepseek-v4-pro", "deepseek"):
                for m in cloud_models:
                    if pref in m["id"].lower():
                        return m["id"]
            if cloud_models:
                return cloud_models[0]["id"]

        installed_ollama = [
            m
            for m in available
            if m["provider"] == "ollama"
            and str(m.get("installed")).lower() == "true"
            and "glm-ocr" not in m["id"].lower()
        ]

        for m in installed_ollama:
            if "shisa" in m["id"].lower():
                return m["id"]

        for m in installed_ollama:
            low_id = m["id"].lower()
            if any(k in low_id for k in ("translategemma", "gemma3", "qwen2.5:14b", "aya-expanse", "sugoi")):
                return m["id"]

        for m in installed_ollama:
            if "qwen3.5" in m["id"].lower():
                return m["id"]

        for m in installed_ollama:
            if "qwen" in m["id"].lower():
                return m["id"]

        if installed_ollama:
            return installed_ollama[0]["id"]

        for m in available:
            if m["provider"] == "unsloth" and m["id"] != "unsloth::default":
                return m["id"]

        return f"ollama::{fallback_name}"

    async def resolve_model(
        self,
        requested_model: str | None = None,
        source_lang: str = "ja",
        target_lang: str = "en",
        profile: str | None = None,
    ) -> str:
        eff_prof = (profile or self.translation_profile or "balanced").strip().lower()
        is_local_only = eff_prof == "local_only"

        req = (requested_model or "").strip()
        if req.startswith("unsloth::ollama/"):
            req = "ollama::" + req.split("unsloth::ollama/", 1)[1]

        # Если активен профиль local_only (Локальный 100% Ollama), никогда не возвращаем облачные модели unsloth::
        if is_local_only:
            if req and req != "auto" and req.startswith("ollama::"):
                if source_lang in ("zh", "ko") and "sugoi" in req.lower():
                    loc_ml = await self._resolve_local_fallback_model(target_lang=target_lang, source_lang=source_lang)
                    if loc_ml:
                        return loc_ml
                return req
            if self._resolved_model and self._resolved_model.startswith("ollama::"):
                if not (source_lang in ("zh", "ko") and "sugoi" in self._resolved_model.lower()):
                    return self._resolved_model
            loc_auto = await self._resolve_local_fallback_model(target_lang=target_lang, source_lang=source_lang)
            if loc_auto:
                return loc_auto
            return f"ollama::{self.default_model}"

        # Быстрый возврат без лишних сетевых HTTP-запросов к /v1/models и /api/tags во время пакетного перевода!
        if req and req != "auto" and "::" in req:
            return req

        if (not req or req == "auto") and self._resolved_model and self._resolved_model != "auto":
            if self._resolved_model.startswith("unsloth::ollama/"):
                self._resolved_model = "ollama::" + self._resolved_model.split("unsloth::ollama/", 1)[1]
            return self._resolved_model

        available = await self.list_available_models(force_refresh=False)

        if req and req != "auto":
            for m in available:
                if req.lower() in m["id"].lower():
                    return m["id"]
            if self._is_remote_cloud_url(self.unsloth_url) and self.unsloth_api_key:
                return f"unsloth::{req}"
            return f"ollama::{req}"

        picked = self._pick_best_default_model(
            available,
            self.default_model,
            has_cloud_key=bool(self.unsloth_api_key),
            is_cloud_url=self._is_remote_cloud_url(self.unsloth_url),
        )
        # Кэшируем выбранную модель при auto, чтобы не пересчитывать перед каждой страницей
        if not req or req == "auto":
            self._resolved_model = picked
        return picked

    @staticmethod
    def _normalize_key(raw_key: Any) -> str:
        s = str(raw_key).strip().lstrip("#")
        m_book = re.match(r"^(P\d+_\d+)$", s, flags=re.IGNORECASE)
        if m_book:
            return m_book.group(1).upper()
        digits = re.sub(r"\D+", "", s)
        return digits if digits else s

    @classmethod
    def _is_refusal_response(cls, raw_text: str) -> bool:
        if not raw_text:
            return False
        cleaned = THINK_REGEX.sub("", raw_text).strip()
        if not cleaned:
            return False
        if REFUSAL_PATTERNS_REGEX.search(cleaned) or BUBBLE_REFUSAL_VALUE_REGEX.search(cleaned):
            return True
        if "{" not in cleaned and ":" not in cleaned:
            low = cleaned.lower()
            if low.startswith(("i can't", "i cannot", "i’m sorry", "i'm sorry", "sorry,", "unable to translate", "cannot translate", "извините", "я не могу", "не удалось перевести")):
                return True
        return False

    @classmethod
    def _is_hallucinated_translation(cls, val_text: str, orig_text: str = "") -> bool:
        """
        Отсекает известные галлюцинации OCR/LLM («taxi driver take me to the best restaurant», ложный «Twitch» и т.п.),
        а также аномально длинные предложения, сгенерированные моделью из 1–3 шумовых символов.
        """
        if not val_text:
            return False
        s = str(val_text).strip()
        if not s:
            return False
        if HALLUCINATION_TRANSLATION_REGEX.search(s):
            return True
        norm_val_alpha = re.sub(r"[^a-zA-Zа-яА-ЯёЁ]+", "", s).lower()
        if orig_text:
            if HALLUCINATION_TRANSLATION_REGEX.search(orig_text):
                return True
            clean_orig = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\（\）\(\)\[\]\♡\♥\♪\★\☆\・\：\;\"\'\`]+", "", orig_text)
            # Защита от залипания ответа «Twitch» (когда модель копирует «Twitch» из контекста или шумового SFX на обычную реплику)
            if norm_val_alpha in ("twitch", "twitches", "twitching", "flinch"):
                if clean_orig and not any(sfx_k in clean_orig for sfx_k in ("ビク", "ピク", "ぴく", "びく", "ヒク", "ひく", "痙攣", "つる")):
                    return True
            words = [w for w in re.split(r"\s+", s) if w]
            # Если исходный OCR-текст — это всего 1–2 символа (часто шум штриховки), а модель выдала длинное предложение из 8+ слов (>48 симв.)
            if 0 < len(clean_orig) <= 2 and (len(s) > 48 or len(words) >= 8):
                return True
            if 0 < len(clean_orig) <= 4 and (len(s) > 85 or len(words) >= 14):
                return True
        return False

    @classmethod
    def _is_bubble_refusal_value(cls, val_text: str) -> bool:
        """
        Проверяет, является ли отдельное значение внутри JSON-ответа заглушкой цензуры/отказа API
        (например: 'Unable to translate', '[Unable to translate]', 'Cannot translate explicit content', 'Не удалось перевести')
        или известной фантомной галлюцинацией ('taxi driver take me to the best restaurant').
        """
        if not val_text:
            return False
        s = str(val_text).strip()
        if not s:
            return False
        if BUBBLE_REFUSAL_VALUE_REGEX.search(s) or REFUSAL_PATTERNS_REGEX.search(s) or HALLUCINATION_TRANSLATION_REGEX.search(s):
            return True
        low = re.sub(r"[\[\]\(\)\<\>\*\_\"\'\`\!\.\,\:\;\-\—]+", " ", s.lower()).strip()
        low = re.sub(r"\s+", " ", low)
        if low in (
            "unable to translate",
            "cannot translate",
            "can t translate",
            "cant translate",
            "untranslatable",
            "translation unavailable",
            "translation error",
            "translation blocked",
            "translation refused",
            "translation omitted",
            "content filtered",
            "content blocked",
            "content omitted",
            "explicit content omitted",
            "explicit incestuous content omitted",
            "explicit sexual content omitted",
            "sound omitted",
            "sfx omitted",
            "dialogue omitted",
            "text omitted",
            "omitted",
            "redacted",
            "censored",
            "filtered",
            "nsfw",
            "не удалось перевести",
            "невозможно перевести",
            "перевод невозможен",
            "перевод недоступен",
            "цензура",
            "пропущено",
            "звук пропущен",
            "контент пропущен",
        ):
            return True
        if any(kw in low for kw in ("omitted", "redacted", "censored")) and any(
            w in low
            for w in (
                "explicit",
                "incest",
                "sexual",
                "content",
                "sound",
                "sfx",
                "dialogue",
                "text",
                "line",
                "speech",
                "part",
                "section",
                "censored",
                "policy",
                "safety",
                "was",
                "is",
            )
        ):
            return True
        if low.startswith((
            "unable to translate",
            "cannot translate",
            "i cannot translate",
            "i can t translate",
            "i am unable to translate",
            "i m unable to translate",
            "sorry i cannot",
            "sorry i can t",
            "as an ai",
            "не удалось перевести",
            "невозможно перевести",
            "я не могу перевести",
        )):
            return True
        return False

    @classmethod
    def _parse_json_response(
        cls,
        raw_text: str,
        expected_keys: list[str] | None = None,
    ) -> dict[str, str]:
        if not raw_text:
            return {}

        cleaned = THINK_REGEX.sub("", raw_text).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        for candidate in (cleaned,):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return {
                        cls._normalize_key(k): str(v).strip()
                        for k, v in parsed.items()
                        if v is not None
                    }
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return {
                        cls._normalize_key(k): str(v).strip()
                        for k, v in parsed.items()
                        if v is not None
                    }
            except json.JSONDecodeError:
                pass

        recovered: dict[str, str] = {}
        # 1. Сначала пробуем построчный разбор (корректно обрабатывает неэкранированные кавычки внутри реплик и ключи P01_01!)
        for raw_line in cleaned.splitlines():
            line_s = raw_line.strip().rstrip(",")
            m_line = re.match(r"""^['"]?#?((?:P\d+_\d+)|(?:\d+))['"]?\s*:\s*['"](.*)['"]$""", line_s, flags=re.IGNORECASE)
            if m_line:
                norm_k = cls._normalize_key(m_line.group(1))
                val_s = m_line.group(2).replace('\"', '"').replace("\\n", " ").strip()
                if norm_k and val_s:
                    recovered[norm_k] = val_s

        # 2. Дополняем регулярным выражением для однострочных/обрезанных JSON
        for k_match, v_match in KV_PAIR_REGEX.findall(cleaned):
            norm_k = cls._normalize_key(k_match)
            if norm_k not in recovered:
                unescaped_v = v_match.replace('\"', '"').replace("\\n", " ").strip()
                if norm_k and unescaped_v:
                    recovered[norm_k] = unescaped_v

        if recovered:
            logger.warning(
                "JSON от LLM был поврежден/обрезан, но успешно восстановлено %d реплик.",
                len(recovered),
            )
            return recovered

        # БЛОК 4: Регулярка-экстрактор на случай, если модель ответила разговорным текстом для 1 короткой фразы/имени
        # (например: 「トウマ(32)」は **Touma (32)** です / means "Kumi" / translates to **Karo**)
        chatty_conversational_markers = (
            "it looks like you",
            "you've shared a json",
            "だけでは意図が分かりにくい",
            "何をしましょうか",
            "何を手伝いましょうか",
            "私に何を",
            "データですね",
        )
        low_clean = cleaned.lower()
        target_single_key = cls._normalize_key(expected_keys[0]) if (expected_keys and len(expected_keys) == 1) else None
        if not target_single_key:
            m_key_hint = re.search(r"""(?:key|ключ|キー)\s*[`"'「]?((?:P\d+_\d+)|(?:\d+))[`"'」]?""", cleaned, flags=re.IGNORECASE)
            if m_key_hint:
                target_single_key = cls._normalize_key(m_key_hint.group(1))

        if target_single_key:
            # 1) Извлекаем выделенный жирным перевод **...** (например: 「トウマ(32)」は **Touma (32)** です)
            bold_matches = [
                b.strip()
                for b in re.findall(r"\*\*([^*\n]{1,80})\*\*", cleaned)
                if b.strip() and not CJK_REGEX.search(b) and b.strip() != target_single_key
            ]
            if bold_matches:
                rescued_val = bold_matches[-1].strip('"\x27«»„“”')
                if rescued_val:
                    logger.info("Блок 4: Извлечен перевод '%s' из markdown **...** для ключа %s", rescued_val, target_single_key)
                    return {target_single_key: rescued_val}

            # 2) Извлекаем перевод после 'means / translates to / translation is / это / переводится как'
            m_means = re.search(
                r"""(?:means|translates\s+to|translation\s+is|romanized\s+as|is\s+translated\s+as|переводится\s+как|означает)\s*[:\-]?\s*[`"'«„“]([^`"'»”\n]{1,80})[`"'»”]""",
                cleaned,
                flags=re.IGNORECASE,
            )
            if m_means and not CJK_REGEX.search(m_means.group(1)):
                rescued_val = m_means.group(1).strip()
                if rescued_val and rescued_val != target_single_key:
                    logger.info("Блок 4: Извлечен перевод '%s' из кавычек объяснения для ключа %s", rescued_val, target_single_key)
                    return {target_single_key: rescued_val}

            # 3) Если модель вернула просто одну короткую строку перевода без JSON и без болтовни
            if (
                "\n" not in cleaned
                and len(cleaned) <= 64
                and not CJK_REGEX.search(cleaned)
                and not any(m in low_clean for m in chatty_conversational_markers)
                and not cls._is_refusal_response(cleaned)
            ):
                plain_val = cleaned.strip('"\x27«»„“”` ')
                if plain_val:
                    return {target_single_key: plain_val}

        logger.warning("Не удалось распарсить JSON от LLM. Сырой ответ: %s", cleaned[:240])
        return {}

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(CJK_REGEX.search(text))

    @classmethod
    def _is_romaji_leak(cls, candidate_text: str, orig_text: str = "", target_lang: str = "en") -> bool:
        """
        Определяет, выдала ли модель японский транслит (Ромадзи) вместо реального перевода на английский/русский язык.
        """
        t_lang = (target_lang or "en").lower().strip()
        if t_lang == "romaji" or not candidate_text:
            return False
        if t_lang == "ru":
            # Для русского языка отсутствие кириллицы при наличии латиницы — это 100% Ромадзи/английский
            return not bool(re.search(r"[А-Яа-яЁё]", candidate_text))

        # Для target_lang == "en":
        norm_cand = re.sub(r"[^a-zA-Z\s']", " ", candidate_text.lower()).strip()
        norm_cand = re.sub(r"\s+", " ", norm_cand)
        if not norm_cand:
            return False

        if norm_cand in ROMAJI_PHRASE_FALLBACK_EN or norm_cand.replace(" ", "") in ROMAJI_PHRASE_FALLBACK_EN:
            return True

        # ВАЖНО: Не сравниваем слепо `cand_compact == kana_to_romaji(orig_text)`, так как имена персонажей
        # (スズネ -> Suzune, タダシ -> Tadashi, カズ -> Kazu) и заимствования катаканой (ビデオ -> Video, ホテル -> Hotel)
        # при правильном переводе на английский совпадают с транслитерацией! Проверяем только по словарю японских слов `ROMAJI_LEAK_WORDS_SET`.
        words = [w.strip("'") for w in norm_cand.split() if w.strip("'")]
        if not words:
            return False

        # Игнорируем именные суффиксы (-san, -chan, -kun, -senpai, -sensei, -sama) при подсчете
        stripped_words: list[str] = []
        for w in words:
            w_base = re.sub(r"(?:san|chan|kun|senpai|sensei|sama)$", "", w).strip("-")
            if w_base:
                stripped_words.append(w_base)
        if not stripped_words:
            return False

        romaji_hits = sum(
            1 for w in stripped_words if w in ROMAJI_LEAK_WORDS_SET or re.sub(r"(.)\1+", r"\1", w) in ROMAJI_LEAK_WORDS_SET
        )
        if romaji_hits > 0 and romaji_hits >= max(1, (len(stripped_words) + 1) // 2):
            return True

        return False

    @classmethod
    def _rescue_romaji_or_interjection(cls, candidate_text: str, orig_text: str, target_lang: str) -> str:
        """
        Если модель вернула Ромадзи или пустую строку, пытается перевести фразу через словарь частых реплик манги.
        Никогда не возвращает Ромадзи для режимов 'en' и 'ru'!
        """
        t_lang = (target_lang or "en").lower().strip()
        norm_orig = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\♡\♥\♪]+", "", orig_text or "")
        interj_dict = (
            SHORT_INTERJECTIONS_RU
            if t_lang == "ru"
            else (SHORT_INTERJECTIONS_ROMAJI if t_lang == "romaji" else SHORT_INTERJECTIONS_EN)
        )
        if norm_orig in interj_dict:
            return interj_dict[norm_orig]

        if t_lang in ("en", "ru"):
            fb_map = ROMAJI_PHRASE_FALLBACK_RU if t_lang == "ru" else ROMAJI_PHRASE_FALLBACK_EN
            for probe in (candidate_text, japanese_to_romaji_local(orig_text), kana_to_romaji(orig_text)):
                if not probe:
                    continue
                norm_p = re.sub(r"[^a-zA-Z\s]", " ", probe.lower()).strip()
                norm_p = re.sub(r"\s+", " ", norm_p)
                if norm_p in fb_map:
                    return fb_map[norm_p]
                compact_p = norm_p.replace(" ", "")
                if compact_p in fb_map:
                    return fb_map[compact_p]
                # Пословное спасение одиночных японских слов из ROMAJI_PHRASE_FALLBACK
                words_p = norm_p.split()
                if 1 <= len(words_p) <= 3 and all(w in fb_map for w in words_p):
                    return " ".join(fb_map[w] for w in words_p)
        return ""

    @staticmethod
    def _clean_translated_text(text: str, target_lang: str = "en") -> str:
        if not text:
            return ""
        t_lang = (target_lang or "en").lower().strip()
        # ВАЖНО: Конвертируем кану в Ромадзи ТОЛЬКО если целевой язык — 'romaji'!
        # Раньше здесь было `in ("romaji", "en")`, из-за чего непереведенный японский текст превращался в Ромадзи!
        if t_lang == "romaji" and KANA_ANY_REGEX.search(text):
            text = kana_to_romaji(text)
        from renderer import normalize_manga_special_symbols
        text = SHAPE_META_TAG_REGEX.sub("", text).strip()
        normalized = normalize_manga_special_symbols(text).translate(FULLWIDTH_PUNCT_MAP)
        no_cjk = CJK_REGEX.sub("", normalized)
        no_sym = CJK_SYMBOLS_REGEX.sub("", no_cjk)
        return re.sub(r"\s+", " ", no_sym).strip()

    @staticmethod
    def _get_prompts(
        source_lang: str,
        target_lang: str,
        source_map_json: str,
        context_lines: list[str] | None = None,
        glossary_entries: list[dict[str, str]] | None = None,
        geometry_hints_map: dict[str, str] | None = None,
    ) -> tuple[str, str, str]:
        src_en = SOURCE_LANG_NAMES_EN.get(source_lang, "comic")
        src_ru = SOURCE_LANG_NAMES_RU.get(source_lang, "комикса")
        t_lang = target_lang.lower().strip()

        matched_glossary = filter_matching_glossary_entries(
            glossary_entries,
            source_map_json,
            target_lang=t_lang,
        )
        glossary_block_ru = ""
        glossary_block_en = ""
        if matched_glossary:
            g_lines: list[str] = []
            for g_item in matched_glossary:
                note_s = f" ({g_item['note']})" if g_item.get("note") else ""
                g_lines.append(f"- {g_item['src']} -> {g_item['dst']}{note_s}")
            joined_g = "\n".join(g_lines)
            glossary_block_ru = (
                "\nОБЯЗАТЕЛЬНЫЙ ГЛОССАРИЙ ТЕРМИНОВ И ИМЕН (используй СТРОГО этот перевод для указанных имен и терминов):\n"
                f"{joined_g}"
            )
            glossary_block_en = (
                "\nОБЯЗАТЕЛЬНЫЙ ГЛОССАРИЙ ТЕРМИНОВ И ИМЕН / MANDATORY GLOSSARY OF TERMS AND NAMES (strictly use these exact translations):\n"
                f"{joined_g}"
            )

        context_block_ru = glossary_block_ru
        context_block_en = glossary_block_en
        if context_lines:
            joined_ctx = "\n".join(f"- {line}" for line in context_lines[-8:])
            context_block_ru += (
                "\nPrevious Context (do not translate, for reference only — сохраняй связность диалога, пол персонажей и имена, не включай эти строки в ответ):\n"
                f"{joined_ctx}"
            )
            context_block_en += (
                "\nPrevious Context (do not translate, for reference only — keep dialogue continuity, pronouns, and character name spelling, do NOT include in output):\n"
                f"{joined_ctx}"
            )

        if geometry_hints_map:
            g_hint_lines = [f"- #{k}: {tag}" for k, tag in list(geometry_hints_map.items())[:60]]
            if g_hint_lines:
                joined_hints = "\n".join(g_hint_lines)
                context_block_ru += (
                    "\nГЕОМЕТРИЯ БАБЛОВ (Typesetting Budget — подбирай длину фразы под форму и лимит символов бабла, НЕ включай тег [shape: ...] в текст перевода):\n"
                    f"{joined_hints}"
                )
                context_block_en += (
                    "\nBUBBLE GEOMETRY BUDGET (fit translation length to bubble shape & max_chars, do NOT include [shape: ...] tags in output values):\n"
                    f"{joined_hints}"
                )

        # БЛОК 4: Проверяем размер пачки. Для малых пачек (1–5 блоков) добавляем жесткую анти-чат инструкцию
        try:
            parsed_cnt = len(json.loads(source_map_json))
        except Exception:
            parsed_cnt = 10
        is_small_batch = parsed_cnt <= 5
        small_batch_guard = (
            "\nTRANSLATE_MANGA_BUBBLES_JSON: You are a strict JSON translation API, NOT a conversational assistant. "
            "Even if the input contains only 1 short word, name, sound effect, or incomplete syllable, translate or romanize it "
            "into the target language and return ONLY a valid JSON object {\"id\": \"translation\"}. "
            "NEVER explain the meaning, NEVER ask questions, NEVER output markdown prose."
            if is_small_batch
            else ""
        )
        usr_prefix_tag = "TRANSLATE_MANGA_BUBBLES_JSON:\n" if is_small_batch else ""

        if t_lang == "romaji":
            sys_prompt = (
                "You are a Japanese linguist and Hepburn Romaji transliterator.\n"
                "Rules:\n"
                "1. Transliterate every JSON value into space-separated Hepburn Romaji using ONLY Latin letters (A-Z, a-z) and punctuation.\n"
                "2. Do NOT translate meaning into English. Output zero Kanji/Kana/CJK characters.\n"
                "3. Return ONLY a valid JSON object with the exact same string keys."
                f"{small_batch_guard}{context_block_en}"
            )
            usr_prompt = f"{usr_prefix_tag}Transliterate all JSON values into Hepburn Romaji (keep exact keys):\n{source_map_json}"
            retry_prefix = "TRANSLATE_MANGA_BUBBLES_JSON: Transliterate into Hepburn Romaji (Latin letters ONLY, exact same JSON keys):\n"
            return sys_prompt, usr_prompt, retry_prefix

        if t_lang == "ru":
            sys_prompt = (
                f"Ты — профессиональный переводчик и редактор локализации диалогов {src_ru} на русский язык.\n"
                "Правила:\n"
                "1. Имена, фамилии и домашние прозвища передавай фонетически (Поливанов/Хэпбёрн), НЕ переводи их буквально по смыслу иероглифов (七香/ななか -> Нанака, 七ちゃん -> Нана-тян, НЕ «Семь-тян»; 一 -> Ити/Кадзу; 双葉 -> Футаба; 三玖 -> Мику; 四葉 -> Ёцуба; 五月 -> Ицуки). Сохраняй суффиксы (-сан, -тян, -кун, -сэмпай, -сэнсэй).\n"
                "2. Переводи реплики и междометия связно, живо и лаконично под размер облачков.\n"
                "3. Запрещено оставлять японские/корейские/китайские символы или пропускать ключи JSON.\n"
                "4. Не выводи <think>. Верни ТОЛЬКО валидный JSON-объект с теми же ключами.\n"
                f"{TYPESETTING_AWARE_RULE_RU}"
                f"{small_batch_guard}{context_block_ru}"
            )
            usr_prompt = f"{usr_prefix_tag}Переведи все значения JSON с {src_ru} на русский язык (только кириллицей, сохрани ключи):\n{source_map_json}"
            retry_prefix = "TRANSLATE_MANGA_BUBBLES_JSON: Переведи на русский язык (только кириллицей, верни JSON с теми же ключами):\n"
            return sys_prompt, usr_prompt, retry_prefix

        sys_prompt = (
            f"You are a professional {src_en}-to-English localization translator.\n"
            "Rules:\n"
            "1. Translate the MEANING of every dialogue line, exclamation, and interjection into natural, idiomatic English. NEVER transliterate sentences or words into Japanese Romaji (e.g., write 'Stop it!', NEVER 'Yamete!'; write 'Wait!', NEVER 'Chotto matte!'; write 'Feels so good...', NEVER 'Kimochiii').\n"
            "2. Only personal character names should be romanized phonetically (e.g., 七香 -> Nanaka, 七ちゃん -> Nana-chan) with honorifics (-san, -chan, -kun, -senpai, -sensei).\n"
            "3. Output zero Japanese/Korean/Chinese characters, NEVER skip any JSON key, and NEVER output censorship/omission placeholders like '[Explicit content omitted]', '[Sound omitted]', or '[Censored]'.\n"
            "4. Do NOT output <think> tags. Return ONLY a valid JSON object with the exact same keys.\n"
            f"{TYPESETTING_AWARE_RULE_EN}"
            f"{small_batch_guard}{context_block_en}"
        )
        usr_prompt = f"{usr_prefix_tag}Translate all JSON values from {src_en} into natural English (translate meaning, do NOT output Romaji, keep exact keys):\n{source_map_json}"
        retry_prefix = "TRANSLATE_MANGA_BUBBLES_JSON: Translate meaning into natural English (NO Romaji, NO CJK, return JSON with exact same keys):\n"
        return sys_prompt, usr_prompt, retry_prefix

    @staticmethod
    def _sanitize_safety_triggers_for_cloud(text: str) -> str:
        """
        Заменяет слова-триггеры родства/возраста в исходном тексте при повторной попытке
        после отказа облачного фильтра безопасности (parent and child / incest),
        чтобы облачный шлюз перевел реплики без блокировки всей пачки.
        """
        if not text:
            return text
        replacements = (
            ("実の娘", "彼女"),
            ("義理の娘", "彼女"),
            ("お父さん", "あなた"),
            ("父さん", "あなた"),
            ("おとうさん", "あなた"),
            ("パパ", "ダーリン"),
            ("父親", "彼"),
            ("義父", "彼"),
            ("お母さん", "あなた"),
            ("母さん", "あなた"),
            ("ママ", "ハニー"),
            ("母親", "彼女"),
            ("お兄ちゃん", "あなた"),
            ("兄ちゃん", "あなた"),
            ("おにいちゃん", "あなた"),
            ("お兄さん", "あなた"),
            ("兄さん", "あなた"),
            ("兄貴", "あなた"),
            ("お姉ちゃん", "あなた"),
            ("姉ちゃん", "あなた"),
            ("お姉さん", "あなた"),
            ("姉さん", "あなた"),
            ("実の妹", "彼女"),
            ("妹", "彼女"),
            ("いもうと", "かのじょ"),
            ("弟", "彼"),
            ("娘", "彼女"),
            ("むすめ", "かのじょ"),
            ("息子", "彼"),
            ("親子", "二人"),
            ("近親", "恋人"),
            ("小学生", "学生"),
            ("中学生", "学生"),
            ("未成年", "大人"),
            ("子供", "女の子"),
            ("こども", "おんなのこ"),
        )
        out = text
        for old_w, new_w in replacements:
            if old_w in out:
                out = out.replace(old_w, new_w)
        return out

    @staticmethod
    def _get_minimal_retry_prompts(target_lang: str, retry_json: str) -> tuple[str, str]:
        """Сверхкомпактный промпт для гарантированного доперевода пропущенных кружков (Блок 4: с защитой от чат-ответов на коротких именах/словах)."""
        t_lang = target_lang.lower().strip()
        strict_guard = (
            "TRANSLATE_MANGA_BUBBLES_JSON: You are a strict JSON translation API, NOT a conversational assistant. "
            "Even if the input contains only 1 short word, name, sound effect, or incomplete syllable, translate or romanize it "
            "into the target language and return ONLY a valid JSON object {\"id\": \"translation\"}. "
            "NEVER explain the meaning, NEVER ask questions, NEVER output markdown prose."
        )
        if t_lang == "ru":
            return (
                f"{strict_guard}\nПереведи смысл каждого значения JSON на русский язык (ТОЛЬКО кириллицей, без латиницы и без Ромадзи, имена и прозвища передавай фонетически кириллицей). Не пропускай ни одного ключа. Верни ТОЛЬКО валидный JSON с теми же ключами.",
                f"TRANSLATE_MANGA_BUBBLES_JSON:\n{retry_json}",
            )
        if t_lang == "romaji":
            return (
                f"{strict_guard}\nTransliterate all JSON values into Hepburn Romaji (Latin letters ONLY). Return ONLY valid JSON with the exact same keys.",
                f"TRANSLATE_MANGA_BUBBLES_JSON:\n{retry_json}",
            )
        return (
            f"{strict_guard}\nTranslate the meaning of every JSON value into natural English (personal names phonetically, NEVER output Japanese Romaji like 'Yamete' or 'Matte'; translate every word into English). Return ONLY valid JSON with the exact same keys.",
            f"TRANSLATE_MANGA_BUBBLES_JSON:\n{retry_json}",
        )

    async def _resolve_local_fallback_model(
        self,
        target_lang: str = "en",
        source_lang: str = "ja",
    ) -> str | None:
        """
        БЛОК 3: Умный языковой роутинг локального фоллбэка (Language-Aware Local Fallback):
        - Использует `ollama::sugoi-14b-ultra` ТОЛЬКО если `source_lang == 'ja'` и `target_lang == 'en'`.
        - Если `source_lang in ('zh', 'ko')` (китайский или корейский), выбирает мультиязычную модель в Ollama
          (`qwen2.5`, `qwen3`, `gemma3`, `translategemma`, `aya-expanse`, `llama3`, `shisa` или заданную
          в настройках `local_fallback_model_multilang`).
        - Если в Ollama установлена только японская `sugoi`, а переводится китайский (`zh`) или корейский (`ko`) текст —
          НЕ вызывает `sugoi` впустую, а выводит предупреждение в лог и возвращает `None`.
        """
        installed_ollama = [
            m["id"]
            for m in (self._cached_models or [])
            if m.get("provider") == "ollama" and m.get("installed") == "true"
        ]
        if not installed_ollama:
            try:
                available = await self.list_available_models(force_refresh=False)
                installed_ollama = [
                    m["id"]
                    for m in available
                    if m.get("provider") == "ollama" and m.get("installed") == "true"
                ]
            except Exception as exc:
                logger.debug("Не удалось обновить список моделей для локального фоллбэка: %s", exc)

        if not installed_ollama:
            cli_tags = self._list_ollama_cli_models()
            installed_ollama = [f"ollama::{t}" for t in cli_tags if t]

        # Исключаем узкие Vision-OCR модели (glm-ocr) из текстового перевода
        installed_ollama = [
            m_id for m_id in installed_ollama if "glm-ocr" not in m_id.lower()
        ]

        s_lang = (source_lang or "ja").lower().strip()
        t_lang = (target_lang or "en").lower().strip()
        is_cjk_multilang = s_lang in ("zh", "ko")

        if installed_ollama:
            # Проверяем пользовательскую настройку local_fallback_model_multilang для zh/ko
            custom_ml = (self.local_fallback_model_multilang or "").strip()
            if is_cjk_multilang and custom_ml and custom_ml != "auto":
                for m_id in installed_ollama:
                    if custom_ml.lower() in m_id.lower() and "sugoi" not in m_id.lower():
                        return m_id

            if is_cjk_multilang:
                # Для китайской маньхуа (zh) и корейской манхвы (ko) НИКОГДА не выбираем японскую sugoi!
                multilang_candidates = [
                    m_id for m_id in installed_ollama if "sugoi" not in m_id.lower()
                ]
                if not multilang_candidates:
                    logger.warning(
                        "⚠️ [ЛОКАЛЬНЫЙ ФОЛЛБЭК] В Ollama установлена только японская модель Sugoi (%s), "
                        "которая не поддерживает язык оригинала '%s' (китайский/корейский). "
                        "Пропускаю холостой вызов Sugoi! Рекомендуется скачать мультиязычную модель (qwen2.5, gemma3, aya-expanse).",
                        installed_ollama[0],
                        s_lang,
                    )
                    return None
                priority_tuple = (
                    "qwen2.5",
                    "qwen3",
                    "qwen",
                    "translategemma",
                    "gemma3",
                    "aya-expanse",
                    "aya",
                    "llama3",
                    "uncensored",
                    "abliterated",
                    "shisa",
                )
                for priority_kw in priority_tuple:
                    for m_id in multilang_candidates:
                        if priority_kw in m_id.lower():
                            return m_id
                return multilang_candidates[0]

            if t_lang == "en":
                priority_tuple = (
                    "uncensored",
                    "abliterated",
                    "sugoi",
                    "shisa",
                    "translategemma",
                    "aya-expanse",
                    "gemma3",
                    "qwen2.5:14b",
                    "qwen3",
                    "qwen",
                )
            else:
                priority_tuple = (
                    "uncensored",
                    "abliterated",
                    "shisa",
                    "translategemma",
                    "aya-expanse",
                    "gemma3",
                    "qwen2.5:14b",
                    "qwen3",
                    "qwen",
                )
            for priority_kw in priority_tuple:
                for m_id in installed_ollama:
                    if priority_kw in m_id.lower():
                        return m_id
            for m_id in installed_ollama:
                if t_lang == "ru" and "sugoi" in m_id.lower():
                    continue
                return m_id
            if t_lang == "ru" and all("sugoi" in m.lower() for m in installed_ollama):
                logger.warning(
                    "⚠️ [ЛОКАЛЬНЫЙ ФОЛЛБЭК] В Ollama установлена только JA->EN модель Sugoi, не поддерживающая русский язык."
                )
                return None
            return installed_ollama[0]

        return None

    async def _translate_map_via_local_ollama(
        self,
        client: httpx.AsyncClient,
        source_map: dict[str, str],
        source_lang: str,
        target_lang: str,
        context_lines: list[str] | None = None,
        glossary_entries: list[dict[str, str]] | None = None,
        local_model_id: str | None = None,
    ) -> dict[str, str]:
        """
        Переводит переданный словарь реплик `source_map` через локальную модель Ollama
        (разбивая на компактные подгруппы по `MAX_BUBBLES_PER_BATCH_LOCAL = 12` для скорости и точности),
        не изменяя активную облачную модель по умолчанию.
        """
        if not source_map or self.stop_requested:
            return {}

        loc_model = local_model_id or await self._resolve_local_fallback_model(
            target_lang=target_lang,
            source_lang=source_lang,
        )
        if not loc_model:
            return {}
        if source_lang in ("zh", "ko") and "sugoi" in loc_model.lower():
            logger.warning(
                "⚠️ Пропуск вызова '%s' для языка '%s' (Sugoi поддерживает только японский язык JA->EN).",
                loc_model,
                source_lang,
            )
            return {}

        items_list = list(source_map.items())
        sub_size = MAX_BUBBLES_PER_BATCH_LOCAL
        sub_chunks = [
            dict(items_list[i : i + sub_size])
            for i in range(0, len(items_list), sub_size)
        ]

        combined_res: dict[str, str] = {}
        t_lang = (target_lang or "en").lower().strip()
        for sub_idx, sub_map in enumerate(sub_chunks, start=1):
            if self.stop_requested:
                break
            sub_json = json.dumps(sub_map, ensure_ascii=False, separators=(",", ":"))
            sys_p, usr_p, _ = self._get_prompts(
                source_lang=source_lang,
                target_lang=t_lang,
                source_map_json=sub_json,
                context_lines=context_lines,
                glossary_entries=glossary_entries,
            )
            try:
                loc_res = await self._call_ollama_direct(
                    client=client,
                    ollama_model_name=loc_model,
                    system_prompt=sys_p,
                    user_prompt=usr_p,
                    persist_as_default=False,
                )
                loc_refused = bool(loc_res.pop("__refusal__", None))
                if not loc_res or loc_refused:
                    min_sys, min_usr = self._get_minimal_retry_prompts(t_lang, sub_json)
                    loc_res = await self._call_ollama_direct(
                        client=client,
                        ollama_model_name=loc_model,
                        system_prompt=min_sys,
                        user_prompt=min_usr,
                        persist_as_default=False,
                    )
                    loc_res.pop("__refusal__", None)
                if loc_res:
                    combined_res.update(loc_res)
            except Exception as loc_exc:
                logger.warning(
                    "Ошибка при переводе цензурного куска (%d/%d) через локальную модель %s: %s",
                    sub_idx,
                    len(sub_chunks),
                    loc_model,
                    loc_exc,
                )
        return combined_res

    async def _call_sugoi_plain_lines(
        self,
        client: httpx.AsyncClient,
        clean_name: str,
        source_dict: dict[str, str],
    ) -> dict[str, str]:
        """
        Нативный построчный режим перевода специально для `Sugoi-14B-Ultra`:
        Модель `Sugoi-14B-Ultra` обучена на прямых парах «японская фраза -> английский перевод» без JSON-схемы
        и без списков контекста в system_prompt (из-за которых при `"format": "json"` она может копировать
        предыдущее слово из контекста вроде «Twitch»).
        """
        out_map: dict[str, str] = {}
        sugoi_sys = (
            "You are a professional Japanese-to-English manga translator. "
            "Translate the Japanese dialogue into natural, idiomatic English. "
            "Output ONLY the English translation without quotes, notes, or explanations."
        )
        for k, ja_line in source_dict.items():
            clean_ja = str(ja_line or "").strip()
            if not clean_ja:
                continue
            payload_line: dict[str, Any] = {
                "model": clean_name,
                "system": sugoi_sys,
                "prompt": f"Translate from Japanese to English:\n{clean_ja}",
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.1,
                    "repeat_penalty": 1.12,
                    "num_ctx": 2048,
                    "num_predict": 256,
                },
            }
            try:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload_line)
                if resp.status_code == 200:
                    raw_ans = str(resp.json().get("response") or "").strip()
                    cleaned_ans = THINK_REGEX.sub("", raw_ans).strip()
                    cleaned_ans = cleaned_ans.splitlines()[0].strip() if cleaned_ans else ""
                    cleaned_ans = cleaned_ans.strip('"\'«»„“”')
                    if cleaned_ans and not self._is_bubble_refusal_value(cleaned_ans) and not self._is_hallucinated_translation(cleaned_ans, clean_ja):
                        out_map[self._normalize_key(k)] = cleaned_ans
            except Exception as exc:
                logger.debug("Ошибка построчного вызова Sugoi для ключа %s: %s", k, exc)
        return out_map

    async def _call_ollama_direct(
        self,
        client: httpx.AsyncClient,
        ollama_model_name: str,
        system_prompt: str,
        user_prompt: str,
        persist_as_default: bool = True,
    ) -> dict[str, str]:
        self._last_call_was_refusal = False
        clean_name = ollama_model_name.strip()
        if clean_name.startswith("ollama::"):
            clean_name = clean_name.split("ollama::", 1)[1]
        if clean_name.startswith("ollama/"):
            clean_name = clean_name.split("ollama/", 1)[1]

        _t0_ol = time.perf_counter()
        is_sugoi = "sugoi" in clean_name.lower()
        extracted_input_dict: dict[str, str] = {}
        m_json_any = re.search(r"\{.*\}", user_prompt, flags=re.DOTALL)
        if m_json_any:
            try:
                raw_d = json.loads(m_json_any.group(0))
                if isinstance(raw_d, dict):
                    extracted_input_dict = {str(k): str(v) for k, v in raw_d.items() if v is not None}
            except Exception:
                pass
        expected_keys_list = list(extracted_input_dict.keys()) if extracted_input_dict else None
        if is_sugoi:
            # Защита Блока 3: проверяем ТОЛЬКО первую строку system_prompt, чтобы слова в диалогах/контексте не отменяли вызов Sugoi!
            first_sys_line = (system_prompt.splitlines()[0] if system_prompt else "").lower()
            is_non_ja_source = any(
                marker in first_sys_line
                for marker in (
                    "chinese manhua",
                    "korean manhwa",
                    "китайской маньхуа",
                    "корейской манхвы",
                )
            )
            if is_non_ja_source:
                logger.warning(
                    "⚠️ [БЛОК 3] Отменен вызов '%s' для не-японского текста (Sugoi обучена только на JA->EN).",
                    clean_name,
                )
                return {}
            # Для одиночных бабблов и малых пачек (<= 6 фраз) сразу используем нативный построчный режим Sugoi без JSON-грамматики!
            if 0 < len(extracted_input_dict) <= 6:
                direct_sugoi_map = await self._call_sugoi_plain_lines(client, clean_name, extracted_input_dict)
                if direct_sugoi_map:
                    _el_s = time.perf_counter() - _t0_ol
                    self._record_api_call("ollama", clean_name, _el_s, len(direct_sugoi_map), "OK")
                    return direct_sugoi_map

        # Для Sugoi в пакетном режиме убираем блок NAME GLOSSARY & CONTEXT со стрелками '=>', чтобы модель не копировала старые реплики
        eff_system_prompt = system_prompt
        if is_sugoi and "NAME GLOSSARY & CONTEXT" in eff_system_prompt:
            eff_system_prompt = eff_system_prompt.split("NAME GLOSSARY & CONTEXT", 1)[0].strip()

        payload: dict[str, Any] = {
            "model": clean_name,
            "system": eff_system_prompt,
            "prompt": user_prompt,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1 if is_sugoi else 0.2,
                "repeat_penalty": 1.10 if is_sugoi else 1.15,
                "num_ctx": self.num_ctx,
                "num_predict": 2048,
            },
        }
        response = await client.post(f"{self.base_url}/api/generate", json=payload)

        if response.status_code == 404:
            fb_info = self._resolve_gguf_fallback_info(clean_name)
            if fb_info:
                _, _, _, loc_tag = fb_info
                if loc_tag != clean_name:
                    payload["model"] = loc_tag
                    resp_loc = await client.post(f"{self.base_url}/api/generate", json=payload)
                    if resp_loc.status_code == 200:
                        if persist_as_default:
                            self.save_config(selected_model=f"ollama::{loc_tag}")
                        raw_output = resp_loc.json().get("response", "{}")
                        parsed_loc = self._parse_json_response(raw_output, expected_keys=expected_keys_list)
                        if parsed_loc:
                            return parsed_loc

            logger.info(
                "Модель '%s' еще не скачана в Ollama — запускаю автоматическое скачивание...",
                clean_name,
            )
            pull_res = await self.pull_ollama_model(clean_name, set_as_default=persist_as_default)
            if pull_res.get("status") == "completed":
                actual_tag = str(pull_res.get("model") or clean_name)
                payload["model"] = actual_tag
                response = await client.post(f"{self.base_url}/api/generate", json=payload)

        response.raise_for_status()
        raw_output = response.json().get("response", "{}")
        parsed = self._parse_json_response(raw_output, expected_keys=expected_keys_list)

        # Если GBNF-грамматика "format": "json" в Ollama заставила модель сразу закрыть "{}" (пустой ответ),
        # повторяем запрос без жесткого "format": "json"!
        if not parsed and not self._is_refusal_response(raw_output):
            payload_no_fmt = dict(payload)
            payload_no_fmt.pop("format", None)
            try:
                resp2 = await client.post(f"{self.base_url}/api/generate", json=payload_no_fmt)
                if resp2.status_code == 200:
                    raw_output2 = resp2.json().get("response", "{}")
                    parsed2 = self._parse_json_response(raw_output2, expected_keys=expected_keys_list)
                    if parsed2:
                        parsed = parsed2
                        raw_output = raw_output2
            except Exception:
                pass

        for rk in [
            k
            for k, v in list(parsed.items())
            if self._is_bubble_refusal_value(v)
            or self._is_hallucinated_translation(v, extracted_input_dict.get(k, ""))
        ]:
            parsed.pop(rk, None)
        if is_sugoi and extracted_input_dict:
            missing_sugoi = {
                k: v for k, v in extracted_input_dict.items() if self._normalize_key(k) not in parsed or not parsed.get(self._normalize_key(k))
            }
            if 0 < len(missing_sugoi) <= 8:
                rescued_sugoi = await self._call_sugoi_plain_lines(client, clean_name, missing_sugoi)
                parsed.update(rescued_sugoi)
        _el_ol = time.perf_counter() - _t0_ol
        if not parsed and self._is_refusal_response(raw_output):
            self._last_call_was_refusal = True
            self.last_refusal_message = raw_output.strip()[:240]
            self._record_api_call("ollama", clean_name, _el_ol, 0, "REFUSAL", self.last_refusal_message)
            return {"__refusal__": self.last_refusal_message}
        self._record_api_call("ollama", clean_name, _el_ol, len(parsed), "OK")
        return parsed

    async def _call_llm_json(
        self,
        client: httpx.AsyncClient,
        full_model_id: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, str]:
        self._last_call_was_refusal = False
        if full_model_id.startswith("unsloth::ollama/") and not self._is_remote_cloud_url(self.unsloth_url):
            real_ollama = full_model_id.split("unsloth::ollama/", 1)[1]
            logger.info("Перенаправление прокси-модели '%s' напрямую в Ollama (%s)...", full_model_id, real_ollama)
            return await self._call_ollama_direct(client, real_ollama, system_prompt, user_prompt)

        if full_model_id.startswith("unsloth::"):
            model_name = full_model_id.split("unsloth::", 1)[1].strip()
            is_cloud = self._is_remote_cloud_url(self.unsloth_url)

            if model_name in ("", "default", "auto"):
                if is_cloud:
                    # По умолчанию для облака всегда выбираем экономную deepseek-v4-flash (не дорогую pro!)
                    model_name = "deepseek-v4-flash"
                else:
                    _, detected_ids, _ = await self._probe_unsloth_url(self.unsloth_url)
                    non_ollama_ids = [x for x in detected_ids if not x.lower().startswith("ollama/")]
                    if non_ollama_ids:
                        model_name = non_ollama_ids[0]
                    elif detected_ids and detected_ids[0].lower().startswith("ollama/"):
                        real_ollama = detected_ids[0].split("/", 1)[1]
                        return await self._call_ollama_direct(client, real_ollama, system_prompt, user_prompt)
                    else:
                        model_name = "default"

            m_cloud_json = re.search(r"\{.*\}", user_prompt, flags=re.DOTALL)
            expected_keys_list: list[str] | None = None
            if m_cloud_json:
                try:
                    raw_cd = json.loads(m_cloud_json.group(0))
                    if isinstance(raw_cd, dict):
                        expected_keys_list = [str(k) for k in raw_cd.keys()]
                except Exception:
                    pass
            items_cnt = len(expected_keys_list) if expected_keys_list is not None else 20
            # Адаптивные таймауты с запасом (<=15 фраз -> 48с, 16..70 фраз -> 68с, >70 фраз -> 90с),
            # чтобы не обрывать почти готовый ответ облачного ИИ по ReadTimeout на 22-й секунде!
            if items_cnt <= 15:
                req_timeout = httpx.Timeout(48.0, connect=10.0)
            elif items_cnt <= 70:
                req_timeout = httpx.Timeout(68.0, connect=12.0)
            else:
                req_timeout = httpx.Timeout(max(85.0, float(self.timeout_sec)), connect=15.0)

            # Для внешних облачных API (ModuleAPI, DeepSeek, OpenRouter) используем строго выбранную модель
            # и НИКОГДА не переключаем deepseek-v4-flash на дорогую deepseek-v4-pro!
            if is_cloud:
                is_moduleapi = "moduleapi" in self.unsloth_url.lower()
                omit_resp_fmt = bool(getattr(self, "_cloud_omit_response_format", False)) or is_moduleapi
                payload_cloud: dict[str, Any] = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096 if (omit_resp_fmt or items_cnt <= 80) else 8192,
                    "stream": False,
                }
                if not omit_resp_fmt:
                    payload_cloud["response_format"] = {"type": "json_object"}
                last_cloud_err: Exception | None = None
                max_attempts = 2 if items_cnt <= 15 else 3
                for attempt in range(max_attempts):
                    if self.stop_requested:
                        return {}
                    _t0_cloud = time.perf_counter()
                    try:
                        response = await client.post(
                            f"{self.unsloth_url}/v1/chat/completions",
                            headers=self._unsloth_headers(),
                            json=payload_cloud,
                            timeout=req_timeout,
                        )
                        if response.status_code in (400, 403, 422, 451):
                            err_body = (response.text or "")[:500]
                            if CLOUD_CENSORSHIP_ERROR_REGEX.search(err_body) or self._is_refusal_response(err_body):
                                _el_cloud = time.perf_counter() - _t0_cloud
                                self._last_call_was_refusal = True
                                self.last_refusal_message = f"HTTP {response.status_code} Censorship: {err_body[:180]}"
                                self._record_api_call("cloud", model_name, _el_cloud, 0, "REFUSAL", self.last_refusal_message)
                                return {"__refusal__": self.last_refusal_message}
                            if response.status_code in (400, 422) and ("response_format" in payload_cloud or payload_cloud.get("max_tokens", 4096) > 4096):
                                self._cloud_omit_response_format = True
                                payload_cloud.pop("response_format", None)
                                payload_cloud["max_tokens"] = 4096
                                response = await client.post(
                                    f"{self.unsloth_url}/v1/chat/completions",
                                    headers=self._unsloth_headers(),
                                    json=payload_cloud,
                                    timeout=req_timeout,
                                )
                        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                            wait_s = 1.5 * (attempt + 1)
                            self.diagnostics_stats["network_retries_count"] = (
                                int(self.diagnostics_stats.get("network_retries_count") or 0) + 1
                            )
                            self._record_incident(
                                "CLOUD_HTTP_RETRY",
                                model_name,
                                f"Облачный шлюз вернул HTTP {response.status_code} (попытка {attempt + 1}/{max_attempts}), повтор через {wait_s:.1f}с",
                                details=(response.text or "")[:180],
                            )
                            logger.warning(
                                "Облачный шлюз (%s) вернул HTTP %s для '%s' (попытка %d/%d) — повтор через %.1fс...",
                                self.unsloth_url,
                                response.status_code,
                                model_name,
                                attempt + 1,
                                max_attempts,
                                wait_s,
                            )
                            await asyncio.sleep(wait_s)
                            continue
                        response.raise_for_status()
                        data = response.json()
                        choices = data.get("choices", [])
                        first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                        finish_reason = str(first_choice.get("finish_reason") or "").lower().strip()
                        msg_obj = first_choice.get("message", {}) if isinstance(first_choice.get("message"), dict) else {}
                        raw_output = str(msg_obj.get("content") or "{}")
                        _el_cloud = time.perf_counter() - _t0_cloud
                        parsed = self._parse_json_response(raw_output, expected_keys=expected_keys_list)
                        refusal_keys = [k for k, v in parsed.items() if self._is_bubble_refusal_value(v)]
                        for rk in refusal_keys:
                            parsed.pop(rk, None)
                        is_filter_stop = finish_reason in ("content_filter", "safety", "moderation", "sensitive")
                        if refusal_keys or (not parsed and (is_filter_stop or self._is_refusal_response(raw_output))):
                            self._last_call_was_refusal = True
                            self.last_refusal_message = (
                                f"Censorship in {len(refusal_keys)} bubbles (e.g. 'unable to translate')"
                                if refusal_keys
                                else (
                                    raw_output.strip()[:240]
                                    if raw_output.strip() and raw_output.strip() != "{}"
                                    else f"finish_reason={finish_reason or 'content_filter'}"
                                )
                            )
                            if not parsed:
                                self._record_api_call("cloud", model_name, _el_cloud, 0, "REFUSAL", self.last_refusal_message)
                                return {"__refusal__": self.last_refusal_message}
                            parsed["__partial_refusal__"] = self.last_refusal_message
                            self._record_api_call("cloud", model_name, _el_cloud, len(parsed) - 1, "PARTIAL_REFUSAL", self.last_refusal_message)
                            return parsed
                        self._record_api_call("cloud", model_name, _el_cloud, len(parsed), "OK")
                        return parsed
                    except (
                        httpx.ConnectError,
                        httpx.ConnectTimeout,
                        httpx.ReadTimeout,
                        httpx.WriteTimeout,
                        httpx.PoolTimeout,
                        httpx.RemoteProtocolError,
                        httpx.ReadError,
                        httpx.WriteError,
                    ) as net_exc:
                        last_cloud_err = net_exc
                        if attempt < max_attempts - 1:
                            wait_s = 1.5 * (attempt + 1)
                            self.diagnostics_stats["network_retries_count"] = (
                                int(self.diagnostics_stats.get("network_retries_count") or 0) + 1
                            )
                            self._record_incident(
                                "CLOUD_NETWORK_RETRY",
                                model_name,
                                f"Сетевой сбой {type(net_exc).__name__} (попытка {attempt + 1}/{max_attempts}), повтор через {wait_s:.1f}с",
                                details=str(net_exc),
                            )
                            logger.warning(
                                "Сбой соединения с '%s' (%s: %s, попытка %d/%d) — повтор через %.1fс (0 лишних токенов)...",
                                model_name,
                                type(net_exc).__name__,
                                net_exc,
                                attempt + 1,
                                max_attempts,
                                wait_s,
                            )
                            await asyncio.sleep(wait_s)
                            continue
                        break
                    except Exception as c_exc:
                        last_cloud_err = c_exc
                        break
                if last_cloud_err is not None:
                    self._record_api_call("cloud", model_name, 0.0, 0, "ERROR", str(last_cloud_err))
                    low_sys = (system_prompt + " " + user_prompt).lower()
                    hint_src = "zh" if ("chinese manhua" in low_sys or "китайской маньхуа" in low_sys) else ("ko" if ("korean manhwa" in low_sys or "корейской манхвы" in low_sys) else "ja")
                    hint_tgt = "ru" if "русск" in low_sys else "en"
                    fallback_ollama = await self._resolve_local_fallback_model(target_lang=hint_tgt, source_lang=hint_src)
                    if fallback_ollama:
                        self.last_local_fallback_model = fallback_ollama
                        self.diagnostics_stats["cloud_to_local_error_fallbacks_count"] = (
                            int(self.diagnostics_stats.get("cloud_to_local_error_fallbacks_count") or 0) + 1
                        )
                        self._record_incident(
                            "CLOUD_ERROR_LOCAL_FALLBACK",
                            model_name,
                            f"Облачный API недоступен ({type(last_cloud_err).__name__}), пачка перехвачена локальной моделью {fallback_ollama}",
                            details=str(last_cloud_err),
                        )
                        logger.warning(
                            "Облачный API (%s) недоступен (%s). Временный подхват пачки локальной моделью %s (затем возврат к API)!",
                            self.unsloth_url,
                            last_cloud_err,
                            fallback_ollama,
                        )
                        return await self._call_ollama_direct(
                            client,
                            fallback_ollama,
                            system_prompt,
                            user_prompt,
                            persist_as_default=False,
                        )
                    raise last_cloud_err

            payload_full: dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "stream": False,
                "enable_tools": False,
                "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"},
                "response_format": {"type": "json_object"},
            }
            try:
                response = await client.post(
                    f"{self.unsloth_url}/v1/chat/completions",
                    headers=self._unsloth_headers(),
                    json=payload_full,
                )

                if response.status_code in (400, 422):
                    payload_basic: dict[str, Any] = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.2,
                        "stream": False,
                    }
                    response = await client.post(
                        f"{self.unsloth_url}/v1/chat/completions",
                        headers=self._unsloth_headers(),
                        json=payload_basic,
                    )

                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                raw_output = choices[0].get("message", {}).get("content", "{}") if choices else "{}"
                parsed = self._parse_json_response(raw_output, expected_keys=expected_keys_list)
                if not parsed and self._is_refusal_response(raw_output):
                    self._last_call_was_refusal = True
                    self.last_refusal_message = raw_output.strip()[:240]
                    return {"__refusal__": self.last_refusal_message}
                return parsed
            except Exception as unsloth_exc:
                fallback_ollama = await self._resolve_local_fallback_model()
                if fallback_ollama:
                    logger.warning(
                        "Порт Unsloth (%s) вернул ошибку (%s). Временное переключение на локальную модель %s!",
                        self.unsloth_url,
                        unsloth_exc,
                        fallback_ollama,
                    )
                    return await self._call_ollama_direct(
                        client,
                        fallback_ollama,
                        system_prompt,
                        user_prompt,
                        persist_as_default=False,
                    )
                raise

        return await self._call_ollama_direct(client, full_model_id, system_prompt, user_prompt)

    async def _translate_batch_chunk(
        self,
        client: httpx.AsyncClient,
        chunk_blocks: list[dict[str, Any]],
        source_lang: str,
        target_lang: str,
        active_model: str,
        context_lines: list[str] | None = None,
        glossary_entries: list[dict[str, str]] | None = None,
        provenance_out: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, str]:
        if self.stop_requested or not chunk_blocks:
            return {}

        source_map: dict[str, str] = {
            str(block["id"]): block["original_text"] for block in chunk_blocks
        }
        geometry_hints_map: dict[str, str] = {}
        for block in chunk_blocks:
            b_key = str(block["id"])
            g_budget = compute_bubble_geometry_budget(block)
            if g_budget["shape_hint"] == "narrow_vertical" or g_budget["recommended_max_chars"] <= 55:
                geometry_hints_map[b_key] = g_budget["meta_tag"]

        # Компактный JSON без отступов экономит 150-300 входных токенов на каждой пачке
        source_map_json = json.dumps(source_map, ensure_ascii=False, separators=(",", ":"))
        eff_glossary = merge_glossaries(self.glossary, glossary_entries)
        system_prompt, user_prompt, _ = self._get_prompts(
            source_lang=source_lang,
            target_lang=target_lang,
            source_map_json=source_map_json,
            context_lines=context_lines,
            glossary_entries=eff_glossary,
            geometry_hints_map=geometry_hints_map if geometry_hints_map else None,
        )

        translated_map = await self._call_llm_json(
            client=client,
            full_model_id=active_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        full_refusal = bool(translated_map.pop("__refusal__", None))
        partial_refusal = bool(translated_map.pop("__partial_refusal__", None))
        # Дополнительно вычищаем любые бабблы, где модель написала "unable to translate" и т.п.
        inline_refusal_keys = [
            k for k, v in list(translated_map.items()) if self._is_bubble_refusal_value(v)
        ]
        for rk in inline_refusal_keys:
            translated_map.pop(rk, None)
        was_refusal = full_refusal or partial_refusal or bool(inline_refusal_keys)
        refusal_reason_snapshot = (
            self.last_refusal_message
            or ("API returned 'unable to translate' inside bubbles" if inline_refusal_keys else "")
        ) if was_refusal else ""

        t_lang = target_lang.lower().strip()
        if t_lang == "ru":
            interj_dict = SHORT_INTERJECTIONS_RU
        elif t_lang == "romaji":
            interj_dict = SHORT_INTERJECTIONS_ROMAJI
        else:
            interj_dict = SHORT_INTERJECTIONS_EN

        def _is_valid_target_translation(cand_str: str, orig_val: str = "") -> bool:
            if not cand_str or self._contains_cjk(cand_str):
                return False
            if self._is_bubble_refusal_value(cand_str) or self._is_hallucinated_translation(cand_str, orig_val):
                return False
            c = self._clean_translated_text(cand_str, target_lang=t_lang)
            if not c or self._is_bubble_refusal_value(c) or self._is_hallucinated_translation(c, orig_val):
                return False
            if orig_val and not self._contains_cjk(orig_val):
                return True
            if re.search(r"[0-9]", c) and not re.search(r"[A-Za-zА-Яа-яЁё]", c):
                return True
            if t_lang == "ru":
                return bool(re.search(r"[А-Яа-яЁё]", c))
            if t_lang == "en":
                if not re.search(r"[A-Za-z]", c):
                    return False
                if self._is_romaji_leak(c, orig_val, "en"):
                    return False
                return True
            return bool(re.search(r"[A-Za-zА-Яа-яЁё]", c))

        chunk_prov: dict[str, dict[str, str]] = {}
        for k_init, v_init in translated_map.items():
            if v_init and _is_valid_target_translation(v_init, source_map.get(k_init, "")):
                chunk_prov[k_init] = {
                    "source": "llm",
                    "note": f"Переведено основной моделью ({active_model})",
                }

        # СЦЕНАРИЙ 1: Если при работе через облачный API модель наткнулась на цензуру (was_refusal=True —
        # либо весь ответ заблокирован, либо в отдельных бабблах вернулось "unable to translate"),
        # МГНОВЕННО переключаемся на скачанную локальную модель Ollama, переводим цензурные бабблы через неё
        # (с сохранением глоссария и контекста) и затем автоматически возвращаемся к переводу следующих кусков через API!
        refused_sub_map = {
            k: orig_v
            for k, orig_v in source_map.items()
            if not _is_valid_target_translation(translated_map.get(k, ""), orig_v)
        }
        if was_refusal and refused_sub_map and not self.stop_requested:
            sample_phrases = ", ".join(f"«{v[:18]}»" for v in list(refused_sub_map.values())[:3])
            self._record_incident(
                "CLOUD_CENSORSHIP_REFUSAL",
                active_model,
                f"Облачная модель отклонила {len(refused_sub_map)}/{len(source_map)} фраз из-за цензуры/фильтра ({sample_phrases})",
                details=refusal_reason_snapshot,
                affected_count=len(refused_sub_map),
            )
            local_fallback_model = (
                await self._resolve_local_fallback_model(target_lang=t_lang, source_lang=source_lang)
                if active_model.startswith("unsloth::")
                else None
            )
            if local_fallback_model and active_model.startswith("unsloth::"):
                self.last_local_fallback_model = local_fallback_model
                self.diagnostics_stats["local_censorship_fallbacks_count"] = (
                    int(self.diagnostics_stats.get("local_censorship_fallbacks_count") or 0) + 1
                )
                logger.warning(
                    "🛡 [ОБХОД ЦЕНЗУРЫ API] Облачная модель '%s' отклонила %d/%d блоков из-за цензуры (%s)! "
                    "Мгновенно переключаюсь на локальную модель '%s' для перевода цензурных бабблов (затем вернусь обратно на API '%s')...",
                    active_model,
                    len(refused_sub_map),
                    len(source_map),
                    refusal_reason_snapshot[:80],
                    local_fallback_model,
                    active_model,
                )
                loc_chunk_map = await self._translate_map_via_local_ollama(
                    client=client,
                    source_map=refused_sub_map,
                    source_lang=source_lang,
                    target_lang=t_lang,
                    context_lines=context_lines,
                    glossary_entries=eff_glossary,
                    local_model_id=local_fallback_model,
                )
                if loc_chunk_map:
                    for k_loc, v_loc in loc_chunk_map.items():
                        if v_loc and not self._is_bubble_refusal_value(v_loc):
                            translated_map[k_loc] = v_loc
                            chunk_prov[k_loc] = {
                                "source": "llm_local_bypass",
                                "note": f"Переведено локальной моделью ({local_fallback_model}) после отказа цензуры API ({active_model})",
                            }
                    self._record_incident(
                        "LOCAL_CENSORSHIP_BYPASS_OK",
                        local_fallback_model,
                        f"Локальная модель успешно перевела цензурные бабблы ({len(loc_chunk_map)}/{len(refused_sub_map)} фраз) вместо {active_model}",
                        details=f"Причина отказа облака: {refusal_reason_snapshot}",
                        affected_count=len(loc_chunk_map),
                    )
                    logger.info(
                        "✅ [ОБХОД ЦЕНЗУРЫ API] Цензурные бабблы (%d/%d блоков) успешно переведены локальной моделью '%s'. "
                        "Возвращаюсь к переводу через API '%s'!",
                        len(loc_chunk_map),
                        len(refused_sub_map),
                        local_fallback_model,
                        active_model,
                    )

            # Резервный вариант (только если локальный Ollama вообще не запущен или не вернул перевод, и в пачке > 2 фраз):
            if not translated_map and not self.stop_requested and len(source_map) > 2:
                items_list = list(source_map.items())
                sub_size = max(10, (len(items_list) + 1) // 2) if len(items_list) <= 30 else max(12, (len(items_list) + 2) // 3)
                sub_chunks = [dict(items_list[i : i + sub_size]) for i in range(0, len(items_list), sub_size)]
                self._record_incident(
                    "SANITIZED_CLOUD_RETRY",
                    active_model,
                    f"Локальный Ollama недоступен для обхода цензуры — пачка ({len(source_map)} фраз) разбита на {len(sub_chunks)} подгруппы с санитизацией триггеров",
                    details=refusal_reason_snapshot,
                    affected_count=len(source_map),
                )
                logger.warning(
                    "Локальная модель недоступна для обхода цензуры — делю пакет (%d блоков) на %d подгруппы с санитизацией триггеров...",
                    len(source_map),
                    len(sub_chunks),
                )
                for sc_idx, sub_map in enumerate(sub_chunks, start=1):
                    if self.stop_requested:
                        break
                    self.diagnostics_stats["sanitized_retries_count"] += 1
                    sanitized_sub_map = {
                        k: self._sanitize_safety_triggers_for_cloud(v) for k, v in sub_map.items()
                    }
                    sub_json = json.dumps(sanitized_sub_map, ensure_ascii=False, separators=(",", ":"))
                    min_sys, min_usr = self._get_minimal_retry_prompts(t_lang, sub_json)
                    sub_res = await self._call_llm_json(
                        client=client,
                        full_model_id=active_model,
                        system_prompt=min_sys,
                        user_prompt=min_usr,
                    )
                    sub_res.pop("__refusal__", None)
                    sub_res.pop("__partial_refusal__", None)
                    for rk_san in [ks for ks, vs in list(sub_res.items()) if self._is_bubble_refusal_value(vs)]:
                        sub_res.pop(rk_san, None)
                    if sub_res:
                        translated_map.update(sub_res)
                        for k_san, v_san in sub_res.items():
                            if v_san and not self._is_bubble_refusal_value(v_san):
                                chunk_prov[k_san] = {
                                    "source": "llm_sanitized_retry",
                                    "note": f"Переведено через API ({active_model}) после санитизации триггеров цензуры",
                                }

        # Сначала мгновенно (за 0 мс и 0 токенов) спасаем короткие междометия и известные фразы из словаря Ромадзи,
        # чтобы не запускать лишний проход ИИ ради одиночных восклицаний вроде 'Yamete!' / 'Dame!'
        for k, orig_val in source_map.items():
            cur_v = translated_map.get(k, "")
            if not cur_v or not _is_valid_target_translation(cur_v, orig_val):
                has_q_inner = ("?" in orig_val) or ("？" in orig_val)
                norm_orig = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\♡\♥\♪\(\）\（\)]+", "", orig_val)
                if norm_orig in interj_dict and len(norm_orig) <= 5:
                    d_val = interj_dict[norm_orig]
                    if has_q_inner and not d_val.endswith("?"):
                        d_val = re.sub(r"[!\.…]+$", "", d_val) + "?"
                    translated_map[k] = d_val
                    self.diagnostics_stats["dict_interjections_used_count"] = (
                        int(self.diagnostics_stats.get("dict_interjections_used_count") or 0) + 1
                    )
                    chunk_prov[k] = {
                        "source": "interjection_dict",
                        "note": "Переведено по встроенному словарю коротких междометий",
                    }
                    continue
                rescued_pre = self._rescue_romaji_or_interjection(cur_v, orig_val, t_lang)
                if rescued_pre and _is_valid_target_translation(rescued_pre, orig_val):
                    translated_map[k] = rescued_pre
                    chunk_prov[k] = {
                        "source": "romaji_rescue_dict",
                        "note": "Мгновенный перевод частой фразы по встроенному словарю (0 запросов к ИИ)",
                    }
                    continue
                if t_lang == "romaji" and KANA_ANY_REGEX.search(orig_val) and not re.search(r"[\u4e00-\u9fff]", orig_val):
                    translated_map[k] = self._clean_translated_text(kana_to_romaji(orig_val), target_lang="romaji")
                    chunk_prov[k] = {
                        "source": "romaji_local",
                        "note": "Локальная транслитерация каны в Ромадзи",
                    }

        # СЦЕНАРИЙ 2: Гарантированный доперевод ВСЕХ пропущенных или ошибочно транслитерированных (Ромадзи) блоков!
        # Если доперевод через API натыкается на цензуру (или если это остатки после цензурного куска),
        # сразу переводим эти фразы через локальную модель Ollama и возвращаемся к API!
        bad_reason_by_k: dict[str, str] = {}
        if not self.stop_requested:
            bad_keys: list[str] = []
            for k in source_map:
                cand_v1 = translated_map.get(k, "")
                orig_v1 = source_map[k]
                if not _is_valid_target_translation(cand_v1, orig_v1) and bool(CJK_REGEX.search(orig_v1)):
                    bad_keys.append(k)
                    if not cand_v1 or not cand_v1.strip():
                        bad_reason_by_k[k] = "пропущен ключ / пустой ответ ИИ"
                    elif self._contains_cjk(cand_v1):
                        bad_reason_by_k[k] = f"остались CJK-символы в ответе («{cand_v1[:24]}»)"
                        self.diagnostics_stats["cjk_leaks_caught_count"] = (
                            int(self.diagnostics_stats.get("cjk_leaks_caught_count") or 0) + 1
                        )
                    elif t_lang == "ru" and not re.search(r"[А-Яа-яЁё]", cand_v1):
                        bad_reason_by_k[k] = f"ответ без кириллицы / транслит («{cand_v1[:24]}»)"
                        self.diagnostics_stats["romaji_leaks_caught_count"] = (
                            int(self.diagnostics_stats.get("romaji_leaks_caught_count") or 0) + 1
                        )
                    elif t_lang == "en" and self._is_romaji_leak(cand_v1, orig_v1, "en"):
                        bad_reason_by_k[k] = f"утечка Ромадзи вместо перевода («{cand_v1[:24]}»)"
                        self.diagnostics_stats["romaji_leaks_caught_count"] = (
                            int(self.diagnostics_stats.get("romaji_leaks_caught_count") or 0) + 1
                        )
                    else:
                        bad_reason_by_k[k] = "невалидный ответ ИИ"

            # Если отказа цензуры НЕ было (not was_refusal) и остался всего 1–2 мелких блока (например, вздох/шум/имя),
            # не тратим время и запросы на повторный вызов ИИ: пробуем спасти существующий ответ или локальный словарь!
            if bad_keys and not was_refusal and len(bad_keys) <= 2:
                still_bad: list[str] = []
                for bk_minor in bad_keys:
                    cand_minor = self._clean_translated_text(translated_map.get(bk_minor, ""), target_lang=t_lang)
                    orig_minor = source_map[bk_minor]
                    resc_minor = self._rescue_romaji_or_interjection(cand_minor, orig_minor, t_lang)
                    if resc_minor:
                        translated_map[bk_minor] = resc_minor
                        chunk_prov[bk_minor] = {
                            "source": "romaji_rescue_dict",
                            "note": "Спасено локальным словарем без повторного запроса к ИИ",
                        }
                    elif cand_minor and not self._contains_cjk(cand_minor) and not self._is_bubble_refusal_value(cand_minor):
                        translated_map[bk_minor] = cand_minor
                        chunk_prov[bk_minor] = {
                            "source": "llm",
                            "note": f"Сохранен ответ первого прохода ({active_model}) без лишнего ретрая",
                        }
                    else:
                        has_q_m = ("?" in orig_minor) or ("？" in orig_minor)
                        norm_m = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\♡\♥\♪]+", "", orig_minor)
                        if not has_q_m and norm_m in PURE_SIGH_KEYS_JA:
                            # Только чистый вздох из PURE_SIGH_KEYS_JA пропускаем без доперевода; короткие реплики (1-3 кандзи/каны) отправляем в локальный фоллбэк!
                            translated_map[bk_minor] = ""
                            chunk_prov[bk_minor] = {
                                "source": "skipped_sigh_orig",
                                "note": f"Короткий вздох «{orig_minor}» пропущен без повторного запроса к ИИ",
                            }
                        else:
                            still_bad.append(bk_minor)
                bad_keys = still_bad

            if bad_keys:
                sample_bad = "; ".join(
                    f"#{bk} «{source_map[bk][:16]}» ({bad_reason_by_k.get(bk, 'пропуск')})"
                    for bk in bad_keys[:3]
                )
                self._record_incident(
                    "MICRO_RETRY_TRIGGERED",
                    active_model,
                    f"Запущен микро-доперевод для {len(bad_keys)} пропущенных/невалидных фраз",
                    details=sample_bad,
                    affected_count=len(bad_keys),
                )
                retry_chunk_size = (
                    8
                    if not active_model.startswith("unsloth::")
                    else (14 if was_refusal else 80)
                )
                bad_sublists = [
                    bad_keys[i : i + retry_chunk_size]
                    for i in range(0, len(bad_keys), retry_chunk_size)
                ]
                logger.info(
                    "Доперевод %d пропущенных/невалидных блоков (%d подгрупп по <=%d шт., %s -> %s)...",
                    len(bad_keys),
                    len(bad_sublists),
                    retry_chunk_size,
                    source_lang,
                    target_lang,
                )
                for b_sub in bad_sublists:
                    if self.stop_requested:
                        break
                    self.diagnostics_stats["micro_retries_count"] += 1
                    raw_sub_source = {k: source_map[k] for k in b_sub}
                    retry_map: dict[str, str] = {}
                    used_local_for_sub = False
                    loc_model_used_name = ""

                    # БЛОК 3.2: Запрет облачного доперевода ради 1–2 мелких фраз (или если пачка уже ловила цензуру API)!
                    # Сразу передаем 1–2 непереведенных блока в локальный фоллбэк (с учетом языка source_lang) или словарь междометий,
                    # не делая новый HTTP-запрос к облачному API и исключая 90-секундные ReadTimeout!
                    skip_cloud_micro_retry = active_model.startswith("unsloth::") and (
                        was_refusal or len(b_sub) <= 2
                    )
                    if skip_cloud_micro_retry:
                        loc_model = await self._resolve_local_fallback_model(
                            target_lang=t_lang,
                            source_lang=source_lang,
                        )
                        if loc_model:
                            self.last_local_fallback_model = loc_model
                            loc_model_used_name = loc_model
                            used_local_for_sub = True
                            logger.info(
                                "🏠 [БЛОК 3] Пропуск повторного облачного запроса для %d мелких/цензурных фраз — "
                                "перевод через локальный фоллбэк '%s' (%s -> %s)...",
                                len(b_sub),
                                loc_model,
                                source_lang,
                                t_lang,
                            )
                            retry_map = await self._translate_map_via_local_ollama(
                                client=client,
                                source_map=raw_sub_source,
                                source_lang=source_lang,
                                target_lang=t_lang,
                                context_lines=context_lines,
                                glossary_entries=eff_glossary,
                                local_model_id=loc_model,
                            )
                        else:
                            logger.info(
                                "⚡ [БЛОК 3] Пропуск повторного облачного запроса для %d мелких фраз (подходящая локальная модель для '%s' не установлена).",
                                len(b_sub),
                                source_lang,
                            )

                    if not retry_map and not skip_cloud_micro_retry:
                        used_local_for_sub = False
                        retry_source = {
                            k: (
                                self._sanitize_safety_triggers_for_cloud(source_map[k])
                                if active_model.startswith("unsloth::")
                                else source_map[k]
                            )
                            for k in b_sub
                        }
                        retry_json = json.dumps(retry_source, ensure_ascii=False, separators=(",", ":"))
                        min_sys, min_usr = self._get_minimal_retry_prompts(t_lang, retry_json)
                        try:
                            retry_map = await self._call_llm_json(
                                client=client,
                                full_model_id=active_model,
                                system_prompt=min_sys,
                                user_prompt=min_usr,
                            )
                            retry_refused = bool(retry_map.pop("__refusal__", None)) or bool(retry_map.pop("__partial_refusal__", None))
                            for rk_r in [kr for kr, vr in list(retry_map.items()) if self._is_bubble_refusal_value(vr)]:
                                retry_map.pop(rk_r, None)
                                retry_refused = True
                            if retry_refused:
                                was_refusal = True
                            missing_after_cloud_retry = [
                                kr for kr in b_sub if not _is_valid_target_translation(retry_map.get(kr, ""), source_map[kr])
                            ]
                            if (retry_refused or not retry_map or missing_after_cloud_retry) and active_model.startswith("unsloth::"):
                                loc_model = await self._resolve_local_fallback_model(
                                    target_lang=t_lang,
                                    source_lang=source_lang,
                                )
                                if loc_model:
                                    self.last_local_fallback_model = loc_model
                                    loc_model_used_name = loc_model
                                    used_local_for_sub = True
                                    if retry_refused:
                                        self.diagnostics_stats["local_censorship_fallbacks_count"] = (
                                            int(self.diagnostics_stats.get("local_censorship_fallbacks_count") or 0) + 1
                                        )
                                    logger.warning(
                                        "🛡 [ОБХОД ЦЕНЗУРЫ API] Подгруппа из %d фраз отклонена API '%s' — "
                                        "мгновенно перевожу её локальной моделью '%s' и возвращаюсь к API!",
                                        len(b_sub),
                                        active_model,
                                        loc_model,
                                    )
                                    loc_target_source = {
                                        kr: source_map[kr]
                                        for kr in (missing_after_cloud_retry if missing_after_cloud_retry else b_sub)
                                    }
                                    loc_retry_res = await self._translate_map_via_local_ollama(
                                        client=client,
                                        source_map=loc_target_source,
                                        source_lang=source_lang,
                                        target_lang=t_lang,
                                        context_lines=context_lines,
                                        glossary_entries=eff_glossary,
                                        local_model_id=loc_model,
                                    )
                                    if loc_retry_res:
                                        for kr_l, vr_l in loc_retry_res.items():
                                            if vr_l and not self._is_bubble_refusal_value(vr_l):
                                                retry_map[kr_l] = vr_l
                        except Exception as retry_exc:
                            logger.warning("Подгруппа доперевода (%d блоков) не удалась: %s", len(b_sub), retry_exc)

                    for k in b_sub:
                        cand_r = retry_map.get(k, "")
                        if cand_r and _is_valid_target_translation(cand_r, source_map[k]):
                            translated_map[k] = cand_r
                            chunk_prov[k] = {
                                "source": "llm_local_bypass" if used_local_for_sub else "llm_micro_retry",
                                "note": (
                                    f"Допереведено локальной моделью ({loc_model_used_name}) после: {bad_reason_by_k.get(k, 'пропуск')}"
                                    if used_local_for_sub
                                    else f"Допереведено со 2-й попытки ИИ (после: {bad_reason_by_k.get(k, 'пропуск')})"
                                ),
                            }
                        elif cand_r:
                            rescued_r = self._rescue_romaji_or_interjection(cand_r, source_map[k], t_lang)
                            if rescued_r:
                                translated_map[k] = rescued_r
                                chunk_prov[k] = {
                                    "source": "romaji_rescue_dict",
                                    "note": f"Спасено словарем частых фраз манги (после: {bad_reason_by_k.get(k, 'пропуск')})",
                                }

        # Если была выбрана локальная модель Ollama (например, Sugoi), но какие-то фразы остались непереведенными
        # или отсеяны как галлюцинация («Twitch»), автоматически допереводим их через облачный API (если есть ключ)!
        still_missing_local = [
            k for k, orig_v in source_map.items() if not _is_valid_target_translation(translated_map.get(k, ""), orig_v)
        ]
        if (
            still_missing_local
            and len(still_missing_local) > 2
            and self.translation_profile != "local_only"
            and not active_model.startswith("unsloth::")
            and self._is_remote_cloud_url(self.unsloth_url)
            and bool(self.unsloth_api_key)
            and not self.stop_requested
        ):
            cloud_rescue_source = {
                k: self._sanitize_safety_triggers_for_cloud(source_map[k]) for k in still_missing_local
            }
            cloud_rescue_json = json.dumps(cloud_rescue_source, ensure_ascii=False, separators=(",", ":"))
            c_sys, c_usr = self._get_minimal_retry_prompts(t_lang, cloud_rescue_json)
            try:
                c_res = await self._call_llm_json(
                    client=client,
                    full_model_id="unsloth::deepseek-v4-flash",
                    system_prompt=c_sys,
                    user_prompt=c_usr,
                )
                c_res.pop("__refusal__", None)
                c_res.pop("__partial_refusal__", None)
                for k_c in still_missing_local:
                    cand_c = c_res.get(k_c, "")
                    if cand_c and _is_valid_target_translation(cand_c, source_map[k_c]):
                        translated_map[k_c] = cand_c
                        chunk_prov[k_c] = {
                            "source": "llm_micro_retry",
                            "note": "Допереведено через облачный API (deepseek-v4-flash) после сбоя локальной модели",
                        }
            except Exception as c_rescue_exc:
                logger.debug("Резервный доперевод через облачный API пропущен: %s", c_rescue_exc)

        final_map: dict[str, str] = {}
        for k, orig_val in source_map.items():
            raw_val = translated_map.get(k, "")
            if self._is_bubble_refusal_value(raw_val) or self._is_hallucinated_translation(raw_val, orig_val):
                raw_val = ""
            cleaned_val = self._clean_translated_text(raw_val, target_lang=t_lang)
            if self._is_bubble_refusal_value(cleaned_val) or self._is_hallucinated_translation(cleaned_val, orig_val):
                cleaned_val = ""
            if t_lang == "ru" and cleaned_val and not re.search(r"[А-Яа-яЁё]", cleaned_val):
                cleaned_val = ""
            elif t_lang == "en" and cleaned_val and self._is_romaji_leak(cleaned_val, orig_val, "en"):
                cleaned_val = ""

            if not cleaned_val:
                rescued_final = self._rescue_romaji_or_interjection(raw_val, orig_val, t_lang)
                if rescued_final:
                    cleaned_val = rescued_final
                    chunk_prov[k] = {
                        "source": "romaji_rescue_dict",
                        "note": "Восстановлено по встроенному словарю фраз манги",
                    }
                elif t_lang == "romaji":
                    cleaned_val = self._clean_translated_text(japanese_to_romaji_local(orig_val), target_lang="romaji")
                    chunk_prov[k] = {
                        "source": "romaji_local",
                        "note": "Локальная транслитерация в Хэпбёрн-ромадзи",
                    }
                else:
                    fail_reason = bad_reason_by_k.get(
                        k,
                        "ИИ-модель вернула пустой или невалидный ответ",
                    )
                    if was_refusal and refusal_reason_snapshot:
                        fail_reason += f" (Отказ API: {refusal_reason_snapshot[:80]})"
                    chunk_prov[k] = {
                        "source": "untranslated_error",
                        "note": f"Не удалось перевести: {fail_reason}",
                        "api_censored": "true" if was_refusal else "false",
                        "api_failed": "true",
                    }
                    self.diagnostics_stats["untranslated_blocks_count"] = (
                        int(self.diagnostics_stats.get("untranslated_blocks_count") or 0) + 1
                    )
                    self._record_incident(
                        "UNTRANSLATED_BLOCK",
                        active_model,
                        f"Блок #{k} «{orig_val[:28]}» остался без перевода",
                        details=fail_reason,
                        affected_count=1,
                    )

            final_map[k] = cleaned_val

        if provenance_out is not None:
            provenance_out.update(chunk_prov)

        return final_map

    async def translate_blocks(
        self,
        blocks: list[dict[str, Any]],
        source_lang: str = "ja",
        target_lang: str = "en",
        model: str | None = None,
        context_lines: list[str] | None = None,
        external_cache: dict[str, str] | None = None,
        use_cache: bool = True,
        skip_sighs: bool | None = None,
        glossary_entries: list[dict[str, str]] | None = None,
        profile: str | None = None,
    ) -> list[dict[str, Any]]:
        if not blocks or self.stop_requested:
            return []

        prof_cfg = self.get_profile_config(profile)
        active_model = await self.resolve_model(
            model,
            source_lang=source_lang,
            target_lang=target_lang,
            profile=prof_cfg["id"],
        )
        client = self._get_http_client()
        t_lang = target_lang.lower().strip()
        if t_lang == "ru":
            interj_dict = SHORT_INTERJECTIONS_RU
        elif t_lang == "romaji":
            interj_dict = SHORT_INTERJECTIONS_ROMAJI
        else:
            interj_dict = SHORT_INTERJECTIONS_EN

        if external_cache:
            for ext_k, ext_v in list(external_cache.items()):
                orig_k = ext_k.split("||", 1)[1] if "||" in str(ext_k) else ""
                if (
                    not isinstance(ext_v, str)
                    or self._is_bubble_refusal_value(ext_v)
                    or self._is_hallucinated_translation(ext_v, orig_k)
                ):
                    external_cache.pop(ext_k, None)
            if use_cache:
                self.phrase_cache.update(external_cache)

        for pk, pv in list(self.phrase_cache.items()):
            orig_pk = pk.split("||", 1)[1] if "||" in str(pk) else ""
            if (
                not isinstance(pv, str)
                or self._is_bubble_refusal_value(pv)
                or self._is_hallucinated_translation(pv, orig_pk)
            ):
                self.phrase_cache.pop(pk, None)

        if not use_cache:
            for blk_flush in blocks:
                orig_flush = (blk_flush.get("original_text") or "").strip()
                if orig_flush:
                    flush_key = f"{t_lang}||{orig_flush}"
                    self.phrase_cache.pop(flush_key, None)
                    if external_cache is not None:
                        external_cache.pop(flush_key, None)

        translated_map: dict[str, str] = {}
        block_prov: dict[str, dict[str, str]] = {}
        unique_to_translate: list[dict[str, Any]] = []
        seen_orig_in_batch: dict[str, str] = {}
        duplicate_links: dict[str, str] = {}
        cached_hits = 0

        eff_skip_sighs = self.skip_sighs if skip_sighs is None else bool(skip_sighs)

        for block in blocks:
            compute_bubble_geometry_budget(block)
            k = str(block["id"])
            if block.get("is_deleted"):
                translated_map[k] = ""
                block_prov[k] = {
                    "source": "deleted_by_user",
                    "note": "Блок удален пользователем в редакторе",
                }
                continue
            orig_raw = (block.get("original_text") or "").strip()
            if not orig_raw:
                translated_map[k] = ""
                block_prov[k] = {
                    "source": "empty_orig",
                    "note": "Пустой исходный текст",
                }
                continue

            # 0. Если в блоке вообще нет CJK-символов (например, числовой шум штриховки '7755', '111' или уже латиница/кириллица) —
            # никогда не отправляем его в ИИ и не тратим токены/запросы!
            if not self._contains_cjk(orig_raw):
                if not re.search(r"[A-Za-zА-Яа-яЁё]", orig_raw):
                    translated_map[k] = ""
                    cached_hits += 1
                    block_prov[k] = {
                        "source": "skipped_sigh_orig",
                        "note": f"Числовой/символьный артефакт без букв «{orig_raw}» пропущен (0 токенов)",
                    }
                else:
                    translated_map[k] = self._clean_translated_text(orig_raw, target_lang=t_lang)
                    cached_hits += 1
                    block_prov[k] = {
                        "source": "cache",
                        "note": "Текст уже без CJK-символов (оставлен без запроса к ИИ, 0 токенов)",
                    }
                continue

            # 1. Обработка коротких одиночных междометий/вздохов
            # ВАЖНО: Если в реплике есть вопросительный знак ('?' / '？'), это осмысленный вопрос («ん?» -> «Hm?», «はぁ?» -> «Huh?»),
            # его НИКОГДА нельзя пропускать как беззвучный вздох или превращать в утверждение с '!'
            has_question_mark = ("?" in orig_raw) or ("？" in orig_raw)
            norm_orig = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\♡\♥\♪\(\）\（\)]+", "", orig_raw)
            if eff_skip_sighs and not has_question_mark and norm_orig in PURE_SIGH_KEYS_JA:
                translated_map[k] = ""
                cached_hits += 1
                self.diagnostics_stats["sighs_skipped_count"] = (
                    int(self.diagnostics_stats.get("sighs_skipped_count") or 0) + 1
                )
                block_prov[k] = {
                    "source": "skipped_sigh_orig",
                    "note": f"Короткий вздох/стон «{orig_raw}» намеренно пропущен (skip_sighs=True)",
                }
                continue
            collapsed_orig = re.sub(r"(.)\1+", r"\1", norm_orig)
            matched_interj_key = norm_orig if (norm_orig in interj_dict and len(norm_orig) <= 5) else (
                collapsed_orig if (collapsed_orig in interj_dict and len(norm_orig) <= 7 and not re.search(r"[\u4e00-\u9fff]", norm_orig)) else ""
            )
            if matched_interj_key:
                dict_val = interj_dict[matched_interj_key]
                if has_question_mark:
                    # Корректная передача вопросительной интонации («ん?» -> «Hm?», «はぁ?» -> «Huh?», «うん?» -> «Yeah?»)
                    q_overrides_en = {"ん": "Hm?", "んっ": "Mm?!", "んん": "Mmm?", "はぁ": "Huh?", "え": "Eh?", "えっ": "Eh?!", "ええ": "What?!", "うん": "Yeah?", "はい": "Yes?", "お": "Oh?", "おっ": "Oh?"}
                    q_overrides_ru = {"ん": "Мм?", "んっ": "Мм?!", "んん": "Мм-м?", "はぁ": "Ась?", "え": "Э?", "えっ": "Э?!", "ええ": "Чего?!", "うん": "Угу?", "はい": "Да?", "お": "О?", "おっ": "О?"}
                    if t_lang == "en" and norm_orig in q_overrides_en:
                        dict_val = q_overrides_en[norm_orig]
                    elif t_lang == "ru" and norm_orig in q_overrides_ru:
                        dict_val = q_overrides_ru[norm_orig]
                    elif not dict_val.endswith("?"):
                        dict_val = re.sub(r"[!\.…]+$", "", dict_val) + "?"
                translated_map[k] = dict_val
                cached_hits += 1
                self.diagnostics_stats["dict_interjections_used_count"] = (
                    int(self.diagnostics_stats.get("dict_interjections_used_count") or 0) + 1
                )
                block_prov[k] = {
                    "source": "interjection_dict",
                    "note": "Мгновенный перевод по словарю коротких междометий",
                }
                continue

            # 2. Проверка локального кэша уже переведенных фраз (с защитой от старых записей с Ромадзи и отказами цензуры!)
            cache_key = f"{t_lang}||{orig_raw}"
            if use_cache and cache_key in self.phrase_cache and self.phrase_cache[cache_key]:
                cached_cand = self.phrase_cache[cache_key]
                is_bad_cached = (
                    self._contains_cjk(cached_cand)
                    or self._is_bubble_refusal_value(cached_cand)
                    or self._is_hallucinated_translation(cached_cand, orig_raw)
                    or (t_lang == "ru" and not re.search(r"[А-Яа-яЁё]", cached_cand))
                    or (t_lang == "en" and self._is_romaji_leak(cached_cand, orig_raw, "en"))
                )
                if is_bad_cached:
                    self.phrase_cache.pop(cache_key, None)
                    if external_cache is not None:
                        external_cache.pop(cache_key, None)
                else:
                    translated_map[k] = cached_cand
                    cached_hits += 1
                    block_prov[k] = {
                        "source": "cache",
                        "note": "Взято из локального кэша уже переведенных фраз (0 токенов)",
                    }
                    continue

            # 3. Дедупликация одинаковых реплик внутри многостраничной пачки (отправляем в API только 1 раз!)
            if orig_raw in seen_orig_in_batch:
                duplicate_links[k] = seen_orig_in_batch[orig_raw]
                cached_hits += 1
                continue

            seen_orig_in_batch[orig_raw] = k
            unique_to_translate.append(block)

        is_cloud_call = active_model.startswith("unsloth::") and self._is_remote_cloud_url(self.unsloth_url)
        batch_limit = MAX_BUBBLES_PER_BATCH_CLOUD if is_cloud_call else MAX_BUBBLES_PER_BATCH_LOCAL
        max_streams = int(prof_cfg.get("max_parallel_streams", 2)) if is_cloud_call else 1
        min_per_stream = int(prof_cfg.get("min_phrases_per_stream", 90))
        par_trigger = int(prof_cfg.get("parallel_trigger_phrases", 120))
        allow_parallel = (
            is_cloud_call
            and max_streams > 1
            and len(unique_to_translate) > par_trigger
            and len(unique_to_translate) >= min_per_stream * 2
        )

        self.diagnostics_stats["phrases_requested_total"] += len(blocks)
        self.diagnostics_stats["phrases_sent_to_llm"] += len(unique_to_translate)
        self.diagnostics_stats["phrases_saved_by_cache"] += cached_hits
        if allow_parallel:
            self.diagnostics_stats["parallel_batches_count"] += 1

        logger.info(
            "Перевод блоков [профиль=%s]: всего=%d, отправлено в ИИ=%d (сэкономлено из кэша/повторов=%d, лимит пачки=%d, модель=%s)",
            prof_cfg["id"],
            len(blocks),
            len(unique_to_translate),
            cached_hits,
            batch_limit,
            active_model,
        )

        running_context: list[str] = []
        if t_lang in ("en", "ru"):
            for cl in list(self.recent_context_lines) + list(context_lines or []):
                if not cl or "=>" not in cl:
                    continue
                lhs, rhs = [p.strip() for p in cl.split("=>", 1)]
                if self._is_bubble_refusal_value(rhs) or self._is_hallucinated_translation(rhs, lhs):
                    continue
                if t_lang == "ru" and not re.search(r"[А-Яа-яЁё]", rhs):
                    continue
                if t_lang == "en" and self._is_romaji_leak(rhs, lhs, "en"):
                    continue
                if cl not in running_context:
                    running_context.append(cl)
        running_context = running_context[-8:]

        # БЛОК 1: Управляемое параллельное дробление строго по выбранному профилю перевода (eco_book / balanced / turbo)
        if allow_parallel:
            num_parallel = min(
                max_streams,
                max(2, len(unique_to_translate) // max(1, min_per_stream)),
            )
            sub_chunk_size = (len(unique_to_translate) + num_parallel - 1) // num_parallel
            parallel_chunks = [
                unique_to_translate[i : i + sub_chunk_size]
                for i in range(0, len(unique_to_translate), sub_chunk_size)
            ]
            logger.info(
                "⚡ Параллельный облачный перевод: %d уникальных фраз разделены на %d одновременных потока (по ~%d фраз) для ускорения в %dx раза!",
                len(unique_to_translate),
                len(parallel_chunks),
                sub_chunk_size,
                len(parallel_chunks),
            )
            sem = asyncio.Semaphore(MAX_CONCURRENT_CLOUD_REQUESTS)
            ctx_snapshot = list(running_context) if running_context else None

            async def _run_sub(idx: int, sub_c: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
                if idx > 0:
                    await asyncio.sleep(idx * 0.35)
                sub_prov: dict[str, dict[str, str]] = {}
                async with sem:
                    if self.stop_requested:
                        return {}, sub_prov
                    try:
                        sub_map = await self._translate_batch_chunk(
                            client=client,
                            chunk_blocks=sub_c,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            active_model=active_model,
                            context_lines=ctx_snapshot,
                            glossary_entries=glossary_entries,
                            provenance_out=sub_prov,
                        )
                        return sub_map, sub_prov
                    except Exception as sub_exc:
                        self._record_incident(
                            "PARALLEL_STREAM_ERROR",
                            active_model,
                            f"Сбой в параллельном потоке #{idx + 1}/{len(parallel_chunks)} ({len(sub_c)} фраз)",
                            details=str(sub_exc),
                            affected_count=len(sub_c),
                        )
                        logger.error(
                            "Ошибка в параллельном потоке #%d/%d (%d фраз): %s — соседние потоки и кэш сохранены!",
                            idx + 1,
                            len(parallel_chunks),
                            len(sub_c),
                            sub_exc,
                        )
                        for b_err in sub_c:
                            sub_prov[str(b_err["id"])] = {
                                "source": "untranslated_error",
                                "note": f"Ошибка параллельного потока #{idx + 1}: {sub_exc}",
                            }
                        return {}, sub_prov

            sub_results = await asyncio.gather(*(_run_sub(i, sc) for i, sc in enumerate(parallel_chunks)))
            for sc, (chunk_map, sub_prov) in zip(parallel_chunks, sub_results):
                translated_map.update(chunk_map)
                block_prov.update(sub_prov)
                for b in sc:
                    k = str(b["id"])
                    orig_raw = (b.get("original_text") or "").strip()
                    t_val = chunk_map.get(k, "")
                    if t_val and orig_raw and not self._is_bubble_refusal_value(t_val):
                        self.phrase_cache[f"{t_lang}||{orig_raw}"] = t_val
                        if len(orig_raw) >= 4 and len(t_val) >= 4:
                            pair_str = f"{orig_raw} => {t_val}"
                            if pair_str not in running_context:
                                running_context.append(pair_str)
            running_context = running_context[-8:]
        else:
            total_uniq = len(unique_to_translate)
            if total_uniq > batch_limit:
                num_seq_chunks = (total_uniq + batch_limit - 1) // batch_limit
                seq_chunk_size = (total_uniq + num_seq_chunks - 1) // num_seq_chunks
                seq_chunks = [
                    unique_to_translate[i : i + seq_chunk_size]
                    for i in range(0, total_uniq, seq_chunk_size)
                ]
                logger.info(
                    "📦 Равномерная балансировка большой пачки (%d фраз, лимит=%d) -> %d сбалансированных чанков (%s)",
                    total_uniq,
                    batch_limit,
                    len(seq_chunks),
                    ", ".join(f"{len(c)} фраз" for c in seq_chunks),
                )
            else:
                seq_chunks = [unique_to_translate] if unique_to_translate else []

            for chunk in seq_chunks:
                if self.stop_requested:
                    break
                sub_prov: dict[str, dict[str, str]] = {}
                try:
                    chunk_map = await self._translate_batch_chunk(
                        client=client,
                        chunk_blocks=chunk,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        active_model=active_model,
                        context_lines=running_context if running_context else None,
                        glossary_entries=glossary_entries,
                        provenance_out=sub_prov,
                    )
                except Exception as chunk_exc:
                    self._record_incident(
                        "CHUNK_TRANSLATION_ERROR",
                        active_model,
                        f"Сбой перевода подгруппы ({len(chunk)} фраз)",
                        details=str(chunk_exc),
                        affected_count=len(chunk),
                    )
                    logger.error(
                        "Ошибка перевода подгруппы (%d фраз): %s — переводы из кэша и словаря сохранены!",
                        len(chunk),
                        chunk_exc,
                    )
                    chunk_map = {}
                    for b_err in chunk:
                        sub_prov[str(b_err["id"])] = {
                            "source": "untranslated_error",
                            "note": f"Ошибка подгруппы: {chunk_exc}",
                        }
                translated_map.update(chunk_map)
                block_prov.update(sub_prov)

                for b in chunk:
                    k = str(b["id"])
                    orig_raw = (b.get("original_text") or "").strip()
                    t_val = chunk_map.get(k, "")
                    if t_val and orig_raw and not self._is_bubble_refusal_value(t_val):
                        self.phrase_cache[f"{t_lang}||{orig_raw}"] = t_val
                        if len(orig_raw) >= 4 and len(t_val) >= 4:
                            pair_str = f"{orig_raw} => {t_val}"
                            if pair_str not in running_context:
                                running_context.append(pair_str)
                running_context = running_context[-8:]

        # Заполняем дубликаты внутри пачки
        for dup_k, master_k in duplicate_links.items():
            dup_val = translated_map.get(master_k, "")
            translated_map[dup_k] = dup_val
            if dup_val:
                block_prov[dup_k] = {
                    "source": "batch_dedup",
                    "note": f"Повтор идентичной фразы внутри пачки (из блока #{master_k}, 0 токенов)",
                }
            else:
                master_prov = block_prov.get(master_k, {})
                block_prov[dup_k] = {
                    "source": master_prov.get("source", "untranslated_error"),
                    "note": master_prov.get("note", "Не переведено (дубликат непереведенной фразы)"),
                    "api_censored": master_prov.get("api_censored", "false"),
                    "api_failed": master_prov.get("api_failed", "true"),
                }

        if t_lang in ("en", "ru"):
            self.recent_context_lines = list(running_context[-8:])
        if external_cache is not None:
            external_cache.update(self.phrase_cache)

        enriched_blocks: list[dict[str, Any]] = []
        for block in blocks:
            key = str(block["id"])
            orig_raw = (block.get("original_text") or "").strip()
            translated_text = translated_map.get(key, "").strip()
            prov_info = block_prov.get(key, {})
            t_source = str(prov_info.get("source") or ("llm" if translated_text else "untranslated_error"))
            t_note = str(prov_info.get("note") or "")

            if eff_skip_sighs and translated_text and is_translated_sigh_text(translated_text, orig_raw):
                logger.debug(
                    "Перевод '%s' (ориг. '%s') пропущен как короткий вздох (skip_sighs=True).",
                    translated_text,
                    orig_raw,
                )
                self.diagnostics_stats["sighs_skipped_count"] = (
                    int(self.diagnostics_stats.get("sighs_skipped_count") or 0) + 1
                )
                t_source = "skipped_sigh_trans"
                t_note = f"Перевод «{translated_text}» (ориг. «{orig_raw}») скрыт как короткий вздох (skip_sighs=True)"
                translated_text = ""

            block_copy = dict(block)
            block_copy["translated_text"] = translated_text
            block_copy["translation_source"] = t_source
            block_copy["skip_reason"] = "" if translated_text else t_note
            block_copy["api_censored"] = bool(prov_info.get("api_censored") == "true") if not translated_text else False
            block_copy["api_failed"] = bool(prov_info.get("api_failed") == "true" or t_source == "untranslated_error") if not translated_text else False
            if t_lang == "romaji":
                block_copy["romaji_text"] = translated_text
            else:
                cached_rom = self.phrase_cache.get(f"romaji||{orig_raw}", "")
                block_copy["romaji_text"] = (
                    cached_rom
                    or str(block.get("romaji_text") or "").strip()
                    or japanese_to_romaji_local(orig_raw)
                )
            enriched_blocks.append(block_copy)

        return enriched_blocks

    async def transliterate_blocks_to_romaji(
        self,
        blocks: list[dict[str, Any]],
        use_llm: bool = True,
        model: str | None = None,
        external_cache: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Заполняет поле `romaji_text` для каждого блока страницы.
        Если `use_llm=True` и в блоках есть кандзи, не найденные в кэше, запрашивает точное контекстное
        Хэпбёрн-ромадзи у активной ИИ-модели с мгновенным фоллбэком на `japanese_to_romaji_local`.
        """
        if not blocks:
            return []

        if external_cache:
            self.phrase_cache.update(external_cache)

        blocks_copy = [dict(b) for b in blocks]
        need_llm_blocks: list[dict[str, Any]] = []

        for b in blocks_copy:
            orig_raw = (b.get("original_text") or "").strip()
            if not orig_raw:
                b["romaji_text"] = ""
                continue
            cached_rom = self.phrase_cache.get(f"romaji||{orig_raw}", "")
            if cached_rom and not self._contains_cjk(cached_rom):
                b["romaji_text"] = cached_rom
                continue

            local_rom = japanese_to_romaji_local(orig_raw)
            b["romaji_text"] = local_rom
            # Если в строке остались нераспознанные кандзи или запрошен точный разбор через ИИ
            if use_llm and (self._contains_cjk(local_rom) or re.search(r"[\u4e00-\u9fff]", orig_raw)):
                need_llm_blocks.append(b)

        if use_llm and need_llm_blocks:
            try:
                romaji_enriched = await self.translate_blocks(
                    blocks=need_llm_blocks,
                    source_lang="ja",
                    target_lang="romaji",
                    model=model,
                    external_cache=external_cache,
                    use_cache=True,
                )
                rom_by_id = {
                    int(rb["id"]): str(rb.get("translated_text") or "").strip()
                    for rb in romaji_enriched
                }
                for b in blocks_copy:
                    bid = int(b["id"])
                    if bid in rom_by_id and rom_by_id[bid]:
                        b["romaji_text"] = rom_by_id[bid]
            except Exception as exc:
                logger.warning("Фоллбэк на локальный Ромадзи (ИИ недоступен: %s)", exc)

        # Финальная очистка на случай остаточных символов
        for b in blocks_copy:
            orig_raw = (b.get("original_text") or "").strip()
            cur_rom = str(b.get("romaji_text") or "").strip()
            if not cur_rom:
                cur_rom = japanese_to_romaji_local(orig_raw)
            b["romaji_text"] = cur_rom

        if external_cache is not None:
            external_cache.update(self.phrase_cache)

        return blocks_copy

    async def proofread_chapter_book_script(
        self,
        book_map: dict[str, dict[str, str]],
        source_lang: str = "ja",
        target_lang: str = "en",
        model: str | None = None,
        glossary_entries: list[dict[str, str]] | None = None,
    ) -> dict[str, str]:
        """
        БЛОК 2.3: Литературная вычитка всей главы за 1 запрос (✨ Вычитать стиль всей главы за 1 запрос).
        Принимает словарь `{"P01_01": {"orig": "...", "trans": "..."}, ...}`, проверяет и улучшает связность диалогов,
        род местоимений, именные суффиксы и стиль реплик по всей главе целиком и возвращает `{"P01_01": "улучшенный перевод"}`.
        """
        if not book_map or self.stop_requested:
            return {}

        active_model = await self.resolve_model(
            model,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        client = self._get_http_client()
        t_lang = (target_lang or "en").lower().strip()
        src_ru = SOURCE_LANG_NAMES_RU.get(source_lang, "манги")
        src_en = SOURCE_LANG_NAMES_EN.get(source_lang, "manga")

        compact_payload = {
            k: f"{v.get('orig', '')} => {v.get('trans', '')}"
            for k, v in book_map.items()
        }
        payload_json = json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))
        eff_glossary = merge_glossaries(self.glossary, glossary_entries)
        matched_glossary = filter_matching_glossary_entries(eff_glossary, payload_json, target_lang=t_lang)
        glossary_str = ""
        if matched_glossary:
            g_lines = [f"- {g['src']} -> {g['dst']}" + (f" ({g['note']})" if g.get("note") else "") for g in matched_glossary]
            glossary_str = "\nMANDATORY GLOSSARY:\n" + "\n".join(g_lines)

        if t_lang == "ru":
            sys_p = (
                f"Ты — главный литературный редактор перевода {src_ru} на русский язык.\n"
                "Тебе передан полный сценарий главы в формате JSON: {\"P01_01\": \"оригинал => текущий перевод\"}.\n"
                "Твоя задача:\n"
                "1. Вычитать весь сценарий главы как единую книгу: согласовать пол говорящих персонажей, местоимения, обращения (ты/Вы), имена и живой разговорный слог.\n"
                "2. Исправить любые неточности, обрывки или буквальные кальки, сохранив лаконичность под размер бабблов.\n"
                "3. Вернуть ТОЛЬКО валидный JSON-объект {\"P01_01\": \"отредактированный русский перевод\", ...} со всеми ключами (без исходного текста и без стрелок =>)."
                f"{glossary_str}"
            )
            usr_p = f"Вычитай и улучши перевод всей главы на русский язык (верни JSON с теми же ключами Pxx_yy):\n{payload_json}"
        else:
            sys_p = (
                f"You are a senior literary localization editor for {src_en}-to-English translation.\n"
                "You receive the full chapter script as a JSON object: {\"P01_01\": \"original => current translation\"}.\n"
                "Rules:\n"
                "1. Proofread and polish the entire chapter as a cohesive book: ensure consistent character voice, gender/pronouns, name spelling, and natural, idiomatic comic dialogue.\n"
                "2. Fix any awkward phrasing, mistranslations, or Romaji leaks while keeping lines concise for speech bubbles.\n"
                "3. Return ONLY a valid JSON object {\"P01_01\": \"polished English translation\", ...} with the exact same keys (do NOT include the original CJK text or '=>' arrows)."
                f"{glossary_str}"
            )
            usr_p = f"Proofread and polish the full chapter translation into natural English (return JSON with exact keys Pxx_yy):\n{payload_json}"

        res_map = await self._call_llm_json(
            client=client,
            full_model_id=active_model,
            system_prompt=sys_p,
            user_prompt=usr_p,
        )
        res_map.pop("__refusal__", None)
        res_map.pop("__partial_refusal__", None)

        cleaned_out: dict[str, str] = {}
        for k, v in res_map.items():
            norm_k = self._normalize_key(k)
            val_s = str(v or "").strip()
            if "=>" in val_s:
                val_s = val_s.split("=>")[-1].strip()
            val_s = self._clean_translated_text(val_s, target_lang=t_lang)
            if val_s and not self._is_bubble_refusal_value(val_s):
                cleaned_out[norm_k] = val_s
        return cleaned_out

    async def translate_blocks_batch(
        self,
        blocks: list[dict[str, Any]],
        source_lang: str = "ja",
        target_lang: str = "en",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Алиас пакетного перевода блоков с расчетом геометрического бюджета баблов (Блок 5)."""
        return await self.translate_blocks(
            blocks=blocks,
            source_lang=source_lang,
            target_lang=target_lang,
            **kwargs,
        )

    async def translate_page(
        self,
        blocks: list[dict[str, Any]],
        source_lang: str = "ja",
        target_lang: str = "en",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Алиас перевода страницы с расчетом геометрического бюджета баблов (Блок 5)."""
        return await self.translate_blocks(
            blocks=blocks,
            source_lang=source_lang,
            target_lang=target_lang,
            **kwargs,
        )

    @staticmethod
    def _compress_text_heuristic(text: str, target_lang: str = "ru", target_max_chars: int = 35) -> str:
        """Локальный эвристический компрессор разговорной реплики на случай недоступности LLM."""
        s = (text or "").strip()
        if not s:
            return s
        t_lang = (target_lang or "ru").lower().strip()
        if t_lang == "ru":
            replacements = [
                (r"\bпожалуйста\b,?\s*", ""),
                (r"\bна самом деле\b,?\s*", ""),
                (r"\bв конце концов\b,?\s*", ""),
                (r"\bкак бы то ни было\b,?\s*", ""),
                (r"\bчестно говоря\b,?\s*", ""),
                (r"\bпочему бы тебе не\b", "может,"),
                (r"\bчто ты такое говоришь\b", "что ты несешь"),
                (r"\bя понятия не имею\b", "без понятия"),
                (r"\bне может этого быть\b", "не может быть"),
                (r"\bсовершенно верно\b", "точно"),
                (r"\bв данный момент\b", "сейчас"),
                (r"\bпрямо сейчас\b", "сейчас"),
                (r"\bтолько что\b", "сейчас"),
                (r"\bиз-за того, что\b", "потому что"),
                (r"\bдля того чтобы\b", "чтобы"),
                (r"\bв том случае, если\b", "если"),
                (r"\bнесмотря на то, что\b", "хотя"),
                (r"\bочень сильно\b", "очень"),
            ]
        else:
            replacements = [
                (r"\bas a matter of fact\b,?\s*", ""),
                (r"\bto tell you the truth\b,?\s*", ""),
                (r"\bin order to\b", "to"),
                (r"\bat this point in time\b", "now"),
                (r"\bright now\b", "now"),
                (r"\bdo not\b", "don't"),
                (r"\bcannot\b", "can't"),
                (r"\bwill not\b", "won't"),
                (r"\bshould not\b", "shouldn't"),
                (r"\bcould not\b", "couldn't"),
                (r"\bwould not\b", "wouldn't"),
                (r"\bI am\b", "I'm"),
                (r"\byou are\b", "you're"),
                (r"\bwe are\b", "we're"),
                (r"\bthey are\b", "they're"),
                (r"\bwhat is\b", "what's"),
                (r"\bthat is\b", "that's"),
                (r"\bthere is\b", "there's"),
                (r"\bgoing to\b", "gonna"),
                (r"\bwant to\b", "wanna"),
                (r"\bgot to\b", "gotta"),
            ]
        out = s
        for pat, rep in replacements:
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)
        out = re.sub(r"\s+", " ", out).strip()
        out = re.sub(r"\s+([,.!?;:])", r"\1", out)
        if out and s and s[0].isupper():
            out = out[0].upper() + out[1:]
        return out or s

    async def shorten_text_for_bubble(
        self,
        text: str,
        orig_text: str = "",
        target_lang: str = "ru",
        bbox: list[int] | None = None,
        model: str | None = None,
    ) -> str:
        """
        БЛОК 6.3: Кнопка «✂️ Сократить под бабл (AI)»:
        Сокращает длину переведенной фразы на 25–35% с сохранением смысла и тона персонажа,
        чтобы текст крупно и свободно поместился в узкий бабл.
        """
        clean_cur = (text or "").strip()
        if not clean_cur:
            return ""
        t_lang = (target_lang or "ru").lower().strip()
        dummy_blk: dict[str, Any] = {"bbox": bbox or [0, 0, 100, 180]}
        budget = compute_bubble_geometry_budget(dummy_blk)
        target_len = max(6, int(round(len(clean_cur) * 0.68)))
        eff_max_chars = min(target_len, int(budget["recommended_max_chars"]))

        active_model = await self.resolve_model(model, source_lang="ja", target_lang=t_lang)
        client = self._get_http_client()

        if t_lang == "ru":
            sys_p = (
                "Ты — профессиональный редактор и тайпсеттер манги.\n"
                f"Сократи длину русской реплики на 25–35% (желательно до ~{eff_max_chars} символов) "
                f"для узкого бабла `{budget['meta_tag']}`, полностью сохранив смысл, эмоцию и тон персонажа.\n"
                "Убери лишние местоимения, вводные слова и длинные обороты, замени на живую короткую разговорную фразу.\n"
                "Верни ТОЛЬКО валидный JSON-объект {\"1\": \"сокращенная фраза\"} без пояснений."
            )
            orig_part = f" (оригинал: {orig_text})" if orig_text else ""
            usr_p = f"Сократи фразу на 25-35% под узкий бабл{orig_part}:\n" + json.dumps({"1": clean_cur}, ensure_ascii=False)
        else:
            sys_p = (
                "You are a professional manga scanlation editor and typesetter.\n"
                f"Shorten the English dialogue line by 25–35% (aim for ~{eff_max_chars} chars) "
                f"to fit a narrow bubble `{budget['meta_tag']}` while preserving the exact meaning, emotion, and character voice.\n"
                "Use natural contractions and punchy phrasing. Return ONLY a valid JSON object {\"1\": \"shortened line\"}."
            )
            orig_part = f" (original: {orig_text})" if orig_text else ""
            usr_p = f"Shorten by 25-35% for a narrow speech bubble{orig_part}:\n" + json.dumps({"1": clean_cur}, ensure_ascii=False)

        try:
            res_map = await self._call_llm_json(
                client=client,
                full_model_id=active_model,
                system_prompt=sys_p,
                user_prompt=usr_p,
            )
            res_map.pop("__refusal__", None)
            res_map.pop("__partial_refusal__", None)
            cand = str(res_map.get("1") or "").strip()
            if not cand and res_map:
                first_v = next(iter(res_map.values()), "")
                cand = str(first_v or "").strip()
            cand = self._clean_translated_text(cand, target_lang=t_lang)
            if cand and not self._is_bubble_refusal_value(cand):
                return cand
        except Exception as exc:
            logger.warning("Фоллбэк на эвристическое сжатие фразы (ИИ недоступен: %s)", exc)

        return self._compress_text_heuristic(clean_cur, target_lang=t_lang, target_max_chars=eff_max_chars)
