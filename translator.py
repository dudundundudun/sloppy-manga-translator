"""
Модуль перевода и транслитерации текста манги/манхвы/маньхуа через Ollama, HuggingFace GGUF и Облачные OpenAI-совместимые API (v1.0.0).

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
        "description": "Специализированная японско-английская модель Shisa V2.1 (14B). Идеально понимает сленг, именную вежливость (-сан/-кун/-сэмпай) и контекст манги.",
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
        "description": "Легкая и очень быстрая версия Shisa V2.1 на базе Qwen3-8B. Отлично переводит и на английский, и на русский язык.",
    },
    {
        "ollama_tag": "hf.co/yamatazen/Shisa-v2-Mistral-Nemo-12B-Abliterated-Q4_K_M-GGUF",
        "local_tag": "shisa-v2-12b-uncensored:Q4_K_M",
        "hf_repo": "yamatazen/Shisa-v2-Mistral-Nemo-12B-Abliterated-Q4_K_M-GGUF",
        "hf_filename": "shisa-v2-mistral-nemo-12b-abliterated-q4_k_m.gguf",
        "hf_quant": "Q4_K_M",
        "id": "ollama::hf.co/yamatazen/Shisa-v2-Mistral-Nemo-12B-Abliterated-Q4_K_M-GGUF",
        "title": "🔥 Shisa V2 12B Uncensored — Без цензуры для 18+ манги (~7.0 ГБ)",
        "short_name": "Shisa V2 12B (18+)",
        "size_gb": "7.0 ГБ",
        "description": "Версия Shisa V2 12B со снятыми фильтрами цензуры (Abliterated). Лучший выбор для взрослой 18+ манги и додзинси.",
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
        "description": "Модель от Google с самым естественным и живым художественным слогом на русском и английском языках.",
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
        "description": "Мощная модель для одновременной работы с японской мангой, корейской манхвой, китайской маньхуа и транслитерацией в Ромадзи.",
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

CJK_REGEX = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\uff66-\uff9f]"
)
KANA_ANY_REGEX = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
CJK_SYMBOLS_REGEX = re.compile(r"[\u3000-\u303f\uff00-\uff65\uffa0-\uffef]")
THINK_REGEX = re.compile(r"<think>.*?(?:</think>|$)", flags=re.DOTALL | re.IGNORECASE)
KV_PAIR_REGEX = re.compile(r'"?#?(\d+)"?\s*:\s*"((?:[^"\\]|\\.)*?)(?:"|$)')

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

MAX_BUBBLES_PER_BATCH_LOCAL = 24
MAX_BUBBLES_PER_BATCH_CLOUD = 260
CLOUD_PARALLEL_CHUNK_SIZE = 42
MAX_CONCURRENT_CLOUD_REQUESTS = 3

REFUSAL_PATTERNS_REGEX = re.compile(
    r"(?i)(?:\bi\s*(?:can['\u2019]?t|cannot|am\s+unable\s+to|won['\u2019]?t|must\s+decline)\b.*"
    r"\b(?:translat|intensif|explicit|sexual|content|request|assist|fulfill|generate|rewrite|child|minor)\b"
    r"|\b(?:as\s+an\s+ai|content\s+policy|safety\s+guidelines|explicit\s+sexual\s+content)\b"
    r"|\b(?:я\s+не\s+могу\s+(?:перевести|выполнить|помочь)|откровенн\w+\s+сексуальн\w+)\b)"
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
    "うそ": "Не может быть!",
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
    "え", "えっ", "お", "おっ", "おお", "う", "うっ", "ふぅ", "ふっ",
}

TRANSLATED_SIGH_REGEX = re.compile(
    r"^(?:а+|о+|э+|у+|ы+|и+|м+|н+|"
    r"а+х+|о+х+|у+х+|э+х+|ы+х+|и+х+|а+х*н+|м+х+|н+г*х*|"
    r"ха+|фу+х*|уф+|ой+|ай+|кья+|хи+я*|хья+|хм+|пф+|тц+|"
    r"a+|o+|e+|u+|m+|n+|ah+|aah+|ahn+|oh+|ooh+|uh+|eh+|ha+|haa+|mm+h*|hm+|"
    r"phew+|ugh+|gah+|ngh+|kya+|hya+|eek+|huh+|wah+)$",
    flags=re.IGNORECASE,
)


def is_translated_sigh_text(translated_text: str, original_text: str = "") -> bool:
    """
    Проверяет, является ли готовый перевод коротким вздохом/стоном/охом («А», «О», «Ах!», «Ох...», «Мм...», «Ха-а...»).
    Используется при включенной опции `skip_sighs`, чтобы не замазывать такие облачка на картинке.
    """
    if not translated_text:
        return True
    orig_clean = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\♡\♥\♪]+", "", original_text or "")
    # Если оригинал длинный (полноценное предложение), не отбрасываем его
    if len(orig_clean) > 6:
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
        self.text_filter_mode: str = "dialogue_only"
        self.skip_sighs: bool = True
        self.skip_sfx: bool = True
        self.detect_free_text: bool = False
        self.min_text_length: int = 2
        self.min_font_size: int = 12
        self.max_font_size: int = 32
        self.uppercase: bool = True
        self.batch_pages: int = 12
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
        self.stop_requested: bool = False
        self._last_call_was_refusal: bool = False
        self.last_refusal_message: str = ""
        self._load_saved_config()

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
                timeout=httpx.Timeout(self.timeout_sec, connect=10.0),
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
                if "batch_pages" in data and data["batch_pages"] is not None:
                    self.batch_pages = max(1, min(25, int(data["batch_pages"])))
                if isinstance(data.get("ui_settings"), dict):
                    self.ui_settings.update(data["ui_settings"])
            except Exception as exc:
                logger.warning("Не удалось прочитать config.json: %s", exc)

    def save_config(
        self,
        unsloth_url: str | None = None,
        unsloth_api_key: str | None = None,
        selected_model: str | None = None,
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
        batch_pages: int | None = None,
        ui_settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if unsloth_url is not None and unsloth_url.strip():
            self.unsloth_url = self._normalize_api_base_url(unsloth_url)

        if unsloth_api_key is not None:
            self.unsloth_api_key = unsloth_api_key.strip()

        if selected_model is not None:
            clean_sel = selected_model.strip()
            if clean_sel.startswith("unsloth::ollama/"):
                clean_sel = "ollama::" + clean_sel.split("unsloth::ollama/", 1)[1]
            self._resolved_model = None if clean_sel in ("", "auto") else clean_sel

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

        if batch_pages is not None:
            self.batch_pages = max(1, min(25, int(batch_pages)))

        if isinstance(ui_settings, dict):
            self.ui_settings.update(ui_settings)

        self._cache_timestamp = 0.0

        cfg: dict[str, Any] = {
            "unsloth_url": self.unsloth_url,
            "unsloth_api_key": self.unsloth_api_key,
            "selected_model": self._resolved_model or "auto",
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
            "batch_pages": self.batch_pages,
            "ui_settings": self.ui_settings,
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
            "User-Agent": "MikoMangaStudio/1.0.0",
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

    async def _probe_unsloth_url(self, u_url: str) -> tuple[str, list[str], bool]:
        is_remote = self._is_remote_cloud_url(u_url)
        probe_timeout = httpx.Timeout(8.0, connect=5.0) if is_remote else httpx.Timeout(3.0, connect=0.8)
        client = self._get_http_client()
        port_alive = False
        found_ids: list[str] = []

        # Для облачных OpenAI-шлюзов (ModuleAPI, PositionAPI, DeepSeek) сначала опрашиваем /v1/models
        paths_to_try = ("/v1/models", "/models", "/api/v1/models", "/api/models")
        for path in paths_to_try:
            try:
                resp = await client.get(
                    f"{u_url}{path}",
                    headers=self._unsloth_headers(),
                    timeout=probe_timeout,
                )
                if resp.status_code < 500:
                    port_alive = True
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        extracted = self._extract_model_ids_from_payload(data)
                        for m_id in extracted:
                            if m_id not in found_ids:
                                found_ids.append(m_id)
                        if found_ids:
                            if is_remote:
                                found_ids = self._prioritize_cloud_models(found_ids)
                            return u_url, found_ids, True
                        if is_remote and path == "/v1/models":
                            return u_url, ["deepseek-v4-flash", "deepseek-v4-pro"], True
                    except Exception:
                        if is_remote and path == "/v1/models":
                            return u_url, ["deepseek-v4-flash", "deepseek-v4-pro"], True
            except Exception:
                continue

        if not port_alive and not is_remote:
            try:
                resp = await client.get(f"{u_url}/", timeout=httpx.Timeout(1.5, connect=0.6))
                if resp.status_code < 500:
                    port_alive = True
            except Exception:
                pass

        return u_url, found_ids, port_alive

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
        dl_headers = {"User-Agent": "MikoMangaStudio/1.0.0 (Python-GGUF-Bypass)"}
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
        if set_as_default:
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
            if set_as_default:
                self.save_config(selected_model=full_id)
            else:
                self._cache_timestamp = 0.0

            await self.list_available_models(force_refresh=True)
            self.pull_state.update(
                {
                    "status": "completed",
                    "progress_pct": 100.0,
                    "message": f"Модель {clean_tag} успешно скачана и установлена по умолчанию!",
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
        if not force_refresh and self._cached_models and (now - self._cache_timestamp) < 15.0:
            return self._cached_models

        models_list: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        installed_ollama_tags: set[str] = set()
        client = self._get_http_client()

        # 1. Опрашиваем локальный Ollama (127.0.0.1:11434)
        try:
            ollama_timeout = httpx.Timeout(2.5, connect=0.8)
            resp = await client.get(f"{self.base_url}/api/tags", timeout=ollama_timeout)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("models", []):
                    m_name = item.get("name")
                    if m_name:
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
            *(self._probe_unsloth_url(u) for u in unsloth_urls_to_try)
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
            m for m in available if m["provider"] == "ollama" and m.get("installed") == "true"
        ]

        for m in installed_ollama:
            if "shisa" in m["id"].lower():
                return m["id"]

        for m in installed_ollama:
            if "gemma3" in m["id"].lower() or "qwen2.5:14b" in m["id"].lower():
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

    async def resolve_model(self, requested_model: str | None = None) -> str:
        req = (requested_model or "").strip()
        if req.startswith("unsloth::ollama/"):
            req = "ollama::" + req.split("unsloth::ollama/", 1)[1]

        # Быстрый возврат без лишних сетевых HTTP-запросов к /v1/models и /api/tags во время пакетного перевода!
        if req and req != "auto" and "::" in req:
            return req

        if (not req or req == "auto") and self._resolved_model and self._resolved_model != "auto":
            if self._resolved_model.startswith("unsloth::ollama/"):
                self._resolved_model = "ollama::" + self._resolved_model.split("unsloth::ollama/", 1)[1]
            return self._resolved_model

        available = await self.list_available_models()

        if req and req != "auto":
            for m in available:
                if req.lower() in m["id"].lower():
                    return m["id"]
            if self._is_remote_cloud_url(self.unsloth_url) and self.unsloth_api_key:
                return f"unsloth::{req}"
            return f"ollama::{req}"

        return self._pick_best_default_model(
            available,
            self.default_model,
            has_cloud_key=bool(self.unsloth_api_key),
            is_cloud_url=self._is_remote_cloud_url(self.unsloth_url),
        )

    @staticmethod
    def _normalize_key(raw_key: Any) -> str:
        s = str(raw_key).strip()
        digits = re.sub(r"\D+", "", s)
        return digits if digits else s

    @classmethod
    def _is_refusal_response(cls, raw_text: str) -> bool:
        if not raw_text:
            return False
        cleaned = THINK_REGEX.sub("", raw_text).strip()
        if not cleaned:
            return False
        if REFUSAL_PATTERNS_REGEX.search(cleaned):
            return True
        if "{" not in cleaned and ":" not in cleaned:
            low = cleaned.lower()
            if low.startswith(("i can't", "i cannot", "i’m sorry", "i'm sorry", "sorry,", "извините", "я не могу")):
                return True
        return False

    @classmethod
    def _parse_json_response(cls, raw_text: str) -> dict[str, str]:
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
        for k_match, v_match in KV_PAIR_REGEX.findall(cleaned):
            norm_k = cls._normalize_key(k_match)
            unescaped_v = v_match.replace('\"', '"').replace("\\n", " ").strip()
            if norm_k and unescaped_v:
                recovered[norm_k] = unescaped_v

        if recovered:
            logger.warning(
                "JSON от LLM был поврежден/обрезан, но успешно восстановлено %d реплик.",
                len(recovered),
            )
            return recovered

        logger.warning("Не удалось распарсить JSON от LLM. Сырой ответ: %s", cleaned[:240])
        return {}

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(CJK_REGEX.search(text))

    @staticmethod
    def _clean_translated_text(text: str, target_lang: str = "en") -> str:
        if not text:
            return ""
        if KANA_ANY_REGEX.search(text) and target_lang.lower().strip() in ("romaji", "en"):
            text = kana_to_romaji(text)
        normalized = text.translate(FULLWIDTH_PUNCT_MAP)
        no_cjk = CJK_REGEX.sub("", normalized)
        no_sym = CJK_SYMBOLS_REGEX.sub("", no_cjk)
        return re.sub(r"\s+", " ", no_sym).strip()

    @staticmethod
    def _get_prompts(
        source_lang: str,
        target_lang: str,
        source_map_json: str,
        context_lines: list[str] | None = None,
    ) -> tuple[str, str, str]:
        src_en = SOURCE_LANG_NAMES_EN.get(source_lang, "comic")
        src_ru = SOURCE_LANG_NAMES_RU.get(source_lang, "комикса")
        t_lang = target_lang.lower().strip()

        context_block_ru = ""
        context_block_en = ""
        if context_lines:
            joined_ctx = "\n".join(f"- {line}" for line in context_lines[-6:])
            context_block_ru = (
                "\nГЛОССАРИЙ И КОНТЕКСТ (сохраняй такое же написание имен, не включай эти строки в ответ):\n"
                f"{joined_ctx}"
            )
            context_block_en = (
                "\nNAME GLOSSARY & CONTEXT (keep identical character name spelling, do NOT include in output):\n"
                f"{joined_ctx}"
            )

        if t_lang == "romaji":
            sys_prompt = (
                "You are a Japanese linguist and Hepburn Romaji transliterator.\n"
                "Rules:\n"
                "1. Transliterate every JSON value into space-separated Hepburn Romaji using ONLY Latin letters (A-Z, a-z) and punctuation.\n"
                "2. Do NOT translate meaning into English. Output zero Kanji/Kana/CJK characters.\n"
                "3. Return ONLY a valid JSON object with the exact same string keys."
                f"{context_block_en}"
            )
            usr_prompt = f"Transliterate all JSON values into Hepburn Romaji (keep exact keys):\n{source_map_json}"
            retry_prefix = "Transliterate into Hepburn Romaji (Latin letters ONLY, exact same JSON keys):\n"
            return sys_prompt, usr_prompt, retry_prefix

        if t_lang == "ru":
            sys_prompt = (
                f"Ты — профессиональный переводчик и редактор локализации диалогов {src_ru} на русский язык.\n"
                "Правила:\n"
                "1. Имена, фамилии и домашние прозвища передавай фонетически (Поливанов/Хэпбёрн), НЕ переводи их буквально по смыслу иероглифов (七香/ななか -> Нанака, 七ちゃん -> Нана-тян, НЕ «Семь-тян»; 一 -> Ити/Кадзу; 双葉 -> Футаба; 三玖 -> Мику; 四葉 -> Ёцуба; 五月 -> Ицуки). Сохраняй суффиксы (-сан, -тян, -кун, -сэмпай, -сэнсэй).\n"
                "2. Переводи реплики и междометия связно, живо и лаконично под размер облачков.\n"
                "3. Запрещено оставлять японские/корейские/китайские символы или пропускать ключи JSON.\n"
                "4. Не выводи <think>. Верни ТОЛЬКО валидный JSON-объект с теми же ключами."
                f"{context_block_ru}"
            )
            usr_prompt = f"Переведи все значения JSON с {src_ru} на русский язык (только кириллицей, сохрани ключи):\n{source_map_json}"
            retry_prefix = "Переведи на русский язык (только кириллицей, верни JSON с теми же ключами):\n"
            return sys_prompt, usr_prompt, retry_prefix

        sys_prompt = (
            f"You are a professional {src_en}-to-English localization translator.\n"
            "Rules:\n"
            "1. Romanize personal names and nicknames phonetically (Hepburn) and NEVER translate name Kanji literally (e.g., 七香 -> Nanaka, 七ちゃん -> Nana-chan, NEVER 'Seven-chan'; 一 -> Ichi/Kazu; 双葉 -> Futaba; 三玖 -> Miku; 四葉 -> Yotsuba; 五月 -> Itsuki). Keep honorifics (-san, -chan, -kun, -senpai, -sensei).\n"
            "2. Translate every line and interjection into natural, concise, idiomatic English that fits speech bubbles.\n"
            "3. Output zero Japanese/Korean/Chinese characters and never skip any JSON key.\n"
            "4. Do NOT output <think> tags. Return ONLY a valid JSON object with the exact same keys."
            f"{context_block_en}"
        )
        usr_prompt = f"Localize all JSON values from {src_en} into natural English (keep exact keys, no CJK):\n{source_map_json}"
        retry_prefix = "Translate into natural English with zero CJK characters (return JSON with exact same keys):\n"
        return sys_prompt, usr_prompt, retry_prefix

    @staticmethod
    def _get_minimal_retry_prompts(target_lang: str, retry_json: str) -> tuple[str, str]:
        """Сверхкомпактный промпт (~25 токенов) для доперевода 1-10 пропущенных кружков без повторной отправки огромного системного промпта."""
        t_lang = target_lang.lower().strip()
        if t_lang == "ru":
            return (
                "Переведи значения JSON с японского/корейского/китайского на русский язык (кириллицей, имена фонетически). Верни ТОЛЬКО валидный JSON с теми же ключами.",
                retry_json,
            )
        if t_lang == "romaji":
            return (
                "Transliterate all JSON values into Hepburn Romaji (Latin letters ONLY). Return ONLY valid JSON with the exact same keys.",
                retry_json,
            )
        return (
            "Provide literal English translations for each JSON value (romanize names phonetically). Return ONLY valid JSON with the exact same keys.",
            retry_json,
        )

    async def _call_ollama_direct(
        self,
        client: httpx.AsyncClient,
        ollama_model_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, str]:
        self._last_call_was_refusal = False
        clean_name = ollama_model_name.strip()
        if clean_name.startswith("ollama::"):
            clean_name = clean_name.split("ollama::", 1)[1]
        if clean_name.startswith("ollama/"):
            clean_name = clean_name.split("ollama/", 1)[1]

        payload = {
            "model": clean_name,
            "system": system_prompt,
            "prompt": user_prompt,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "repeat_penalty": 1.15,
                "num_ctx": self.num_ctx,
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
                        self.save_config(selected_model=f"ollama::{loc_tag}")
                        raw_output = resp_loc.json().get("response", "{}")
                        return self._parse_json_response(raw_output)

            logger.info(
                "Модель '%s' еще не скачана в Ollama — запускаю автоматическое скачивание...",
                clean_name,
            )
            pull_res = await self.pull_ollama_model(clean_name, set_as_default=True)
            if pull_res.get("status") == "completed":
                actual_tag = str(pull_res.get("model") or clean_name)
                payload["model"] = actual_tag
                response = await client.post(f"{self.base_url}/api/generate", json=payload)

        response.raise_for_status()
        raw_output = response.json().get("response", "{}")
        parsed = self._parse_json_response(raw_output)
        if not parsed and self._is_refusal_response(raw_output):
            self._last_call_was_refusal = True
            self.last_refusal_message = raw_output.strip()[:240]
            return {"__refusal__": self.last_refusal_message}
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

            # Для внешних облачных API (ModuleAPI, DeepSeek, OpenRouter) используем строго выбранную модель
            # и НИКОГДА не переключаем deepseek-v4-flash на дорогую deepseek-v4-pro!
            if is_cloud:
                payload_cloud: dict[str, Any] = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "stream": False,
                }
                last_cloud_err: Exception | None = None
                for attempt in range(2):
                    if self.stop_requested:
                        return {}
                    try:
                        response = await client.post(
                            f"{self.unsloth_url}/v1/chat/completions",
                            headers=self._unsloth_headers(),
                            json=payload_cloud,
                        )
                        if response.status_code in (502, 503, 504) and attempt == 0:
                            logger.warning(
                                "Облачный шлюз (%s) вернул %s для '%s' — повторная попытка через 1.5с...",
                                self.unsloth_url,
                                response.status_code,
                                model_name,
                            )
                            await asyncio.sleep(1.5)
                            continue
                        response.raise_for_status()
                        data = response.json()
                        choices = data.get("choices", [])
                        msg_obj = choices[0].get("message", {}) if choices else {}
                        raw_output = str(msg_obj.get("content") or "{}")
                        parsed = self._parse_json_response(raw_output)
                        if not parsed and self._is_refusal_response(raw_output):
                            self._last_call_was_refusal = True
                            self.last_refusal_message = raw_output.strip()[:240]
                            return {"__refusal__": self.last_refusal_message}
                        return parsed
                    except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as net_exc:
                        last_cloud_err = net_exc
                        if attempt == 0:
                            logger.warning(
                                "Таймаут/сбой сети при обращении к '%s' (%s) — повторная попытка на той же модели...",
                                model_name,
                                net_exc,
                            )
                            await asyncio.sleep(1.2)
                            continue
                        break
                    except Exception as c_exc:
                        last_cloud_err = c_exc
                        break
                if last_cloud_err is not None:
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
                parsed = self._parse_json_response(raw_output)
                if not parsed and self._is_refusal_response(raw_output):
                    self._last_call_was_refusal = True
                    self.last_refusal_message = raw_output.strip()[:240]
                    return {"__refusal__": self.last_refusal_message}
                return parsed
            except Exception as unsloth_exc:
                available = await self.list_available_models()
                ollama_candidates = [
                    m["id"] for m in available if m["provider"] == "ollama" and m.get("installed") == "true"
                ]
                if ollama_candidates:
                    fallback_ollama = ollama_candidates[0]
                    logger.warning(
                        "Порт Unsloth (%s) вернул ошибку (%s). Автоматическое переключение на %s!",
                        self.unsloth_url,
                        unsloth_exc,
                        fallback_ollama,
                    )
                    self._resolved_model = fallback_ollama
                    return await self._call_ollama_direct(client, fallback_ollama, system_prompt, user_prompt)
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
    ) -> dict[str, str]:
        if self.stop_requested or not chunk_blocks:
            return {}

        source_map: dict[str, str] = {
            str(block["id"]): block["original_text"] for block in chunk_blocks
        }
        # Компактный JSON без отступов экономит 150-300 входных токенов на каждой пачке
        source_map_json = json.dumps(source_map, ensure_ascii=False, separators=(",", ":"))
        system_prompt, user_prompt, _ = self._get_prompts(
            source_lang=source_lang,
            target_lang=target_lang,
            source_map_json=source_map_json,
            context_lines=context_lines,
        )

        translated_map = await self._call_llm_json(
            client=client,
            full_model_id=active_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        was_refusal = bool(translated_map.pop("__refusal__", None))

        t_lang = target_lang.lower().strip()
        if t_lang == "ru":
            interj_dict = SHORT_INTERJECTIONS_RU
        elif t_lang == "romaji":
            interj_dict = SHORT_INTERJECTIONS_ROMAJI
        else:
            interj_dict = SHORT_INTERJECTIONS_EN

        def _is_valid_target_translation(cand_str: str) -> bool:
            c = self._clean_translated_text(cand_str, target_lang=t_lang)
            if not c:
                return False
            if t_lang == "ru":
                return bool(re.search(r"[А-Яа-яЁё]", c))
            return bool(re.search(r"[A-Za-zА-Яа-яЁё]", c))

        # СЦЕНАРИЙ 1: Если облачный API вернул отказ цензуры (например, "I can't translate..."),
        # КАТЕГОРИЧЕСКИ НЕ запускаем поштучный доперевод 85 кружков!
        # Вместо этого пробуем 1 раз с нейтральным лингвистическим промптом БЕЗ context_lines (которые часто содержат триггер-слова с прошлых страниц).
        if was_refusal and not translated_map and not self.stop_requested:
            items_list = list(source_map.items())
            sub_size = max(15, (len(items_list) + 1) // 2) if len(items_list) <= 50 else max(15, (len(items_list) + 2) // 3)
            sub_chunks = [dict(items_list[i : i + sub_size]) for i in range(0, len(items_list), sub_size)]
            logger.warning(
                "Облачный API отклонил пакет (%d блоков) из-за фильтра цензуры провайдера. "
                "Делю пакет на %d нейтральных подгруппы без контекста прошлых страниц (никаких поштучных запросов!)...",
                len(source_map),
                len(sub_chunks),
            )
            local_ollama = [
                m["id"]
                for m in (self._cached_models or [])
                if m.get("provider") == "ollama" and m.get("installed") == "true"
            ]
            for sc_idx, sub_map in enumerate(sub_chunks, start=1):
                if self.stop_requested:
                    break
                sub_json = json.dumps(sub_map, ensure_ascii=False, separators=(",", ":"))
                min_sys, min_usr = self._get_minimal_retry_prompts(t_lang, sub_json)
                sub_res = await self._call_llm_json(
                    client=client,
                    full_model_id=active_model,
                    system_prompt=min_sys,
                    user_prompt=min_usr,
                )
                sub_refused = bool(sub_res.pop("__refusal__", None))
                if sub_res:
                    translated_map.update(sub_res)
                elif sub_refused:
                    if local_ollama and active_model.startswith("unsloth::"):
                        logger.warning(
                            "Подгруппа %d/%d отклонена облачной цензурой — перевожу её локально через %s (0 токенов)!",
                            sc_idx,
                            len(sub_chunks),
                            local_ollama[0],
                        )
                        try:
                            loc_res = await self._call_ollama_direct(
                                client=client,
                                ollama_model_name=local_ollama[0],
                                system_prompt=min_sys,
                                user_prompt=min_usr,
                            )
                            loc_res.pop("__refusal__", None)
                            if loc_res:
                                translated_map.update(loc_res)
                        except Exception:
                            pass
                    else:
                        logger.warning(
                            "Подгруппа %d/%d (%d фраз) заблокирована цензурой облачного шлюза (%s). "
                            "Повторные запросы для этой подгруппы остановлены для экономии ваших токенов!",
                            sc_idx,
                            len(sub_chunks),
                            len(sub_map),
                            active_model,
                        )

        # Сначала подставляем известные междометия из локального словаря для всех пустых ключей (0 токенов)
        for k, orig_val in source_map.items():
            cur_v = translated_map.get(k, "")
            if not cur_v or not _is_valid_target_translation(cur_v) or self._contains_cjk(cur_v):
                norm_orig = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\♡\♥\♪]+", "", orig_val)
                if norm_orig in interj_dict:
                    translated_map[k] = interj_dict[norm_orig]
                elif t_lang == "romaji" and KANA_ANY_REGEX.search(orig_val) and not re.search(r"[\u4e00-\u9fff]", orig_val):
                    translated_map[k] = self._clean_translated_text(kana_to_romaji(orig_val), target_lang="romaji")

        # СЦЕНАРИЙ 2: Если это НЕ отказ цензуры, но модель случайно пропустила несколько ключей (например, 1-15 шт. из 110),
        # отправляем МАКСИМУМ 1 сверхлегкий микро-запрос (~25 токенов промпта) для пропущенных ключей.
        # Никаких поштучных запросов в цикле!
        if not was_refusal and not self.stop_requested:
            bad_keys = [
                k
                for k in source_map
                if (
                    not translated_map.get(k)
                    or self._contains_cjk(translated_map.get(k, ""))
                    or not _is_valid_target_translation(translated_map.get(k, ""))
                )
                and bool(CJK_REGEX.search(source_map[k]))
            ]
            if bad_keys and len(bad_keys) <= max(15, len(source_map) // 2):
                logger.info(
                    "Доперевод %d пропущенных кружков %s одним легким микро-запросом (%s -> %s)...",
                    len(bad_keys),
                    bad_keys,
                    source_lang,
                    target_lang,
                )
                retry_source = {k: source_map[k] for k in bad_keys}
                retry_json = json.dumps(retry_source, ensure_ascii=False, separators=(",", ":"))
                min_sys, min_usr = self._get_minimal_retry_prompts(t_lang, retry_json)
                try:
                    retry_map = await self._call_llm_json(
                        client=client,
                        full_model_id=active_model,
                        system_prompt=min_sys,
                        user_prompt=min_usr,
                    )
                    retry_map.pop("__refusal__", None)
                    for k in bad_keys:
                        if retry_map.get(k) and _is_valid_target_translation(retry_map.get(k, "")):
                            translated_map[k] = retry_map[k]
                except Exception as retry_exc:
                    logger.warning("Микро-доперевод пропущенных блоков не удался: %s", retry_exc)

        final_map: dict[str, str] = {}
        for k, orig_val in source_map.items():
            raw_val = translated_map.get(k, "")
            cleaned_val = self._clean_translated_text(raw_val, target_lang=t_lang)
            if t_lang == "ru" and cleaned_val and not re.search(r"[А-Яа-яЁё]", cleaned_val):
                cleaned_val = ""

            if not cleaned_val:
                norm_orig = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\♡\♥\♪]+", "", orig_val)
                if norm_orig in interj_dict:
                    cleaned_val = interj_dict[norm_orig]
                elif t_lang == "romaji":
                    cleaned_val = self._clean_translated_text(japanese_to_romaji_local(orig_val), target_lang="romaji")

            final_map[k] = cleaned_val

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
    ) -> list[dict[str, Any]]:
        if not blocks or self.stop_requested:
            return []

        active_model = await self.resolve_model(model)
        client = self._get_http_client()
        t_lang = target_lang.lower().strip()
        if t_lang == "ru":
            interj_dict = SHORT_INTERJECTIONS_RU
        elif t_lang == "romaji":
            interj_dict = SHORT_INTERJECTIONS_ROMAJI
        else:
            interj_dict = SHORT_INTERJECTIONS_EN

        if external_cache:
            self.phrase_cache.update(external_cache)

        translated_map: dict[str, str] = {}
        unique_to_translate: list[dict[str, Any]] = []
        seen_orig_in_batch: dict[str, str] = {}
        duplicate_links: dict[str, str] = {}
        cached_hits = 0

        eff_skip_sighs = self.skip_sighs if skip_sighs is None else bool(skip_sighs)

        for block in blocks:
            k = str(block["id"])
            if block.get("is_deleted"):
                translated_map[k] = ""
                continue
            orig_raw = (block.get("original_text") or "").strip()
            if not orig_raw:
                translated_map[k] = ""
                continue

            # 1. Обработка коротких одиночных междометий/вздохов
            norm_orig = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\♡\♥\♪]+", "", orig_raw)
            if eff_skip_sighs and norm_orig in PURE_SIGH_KEYS_JA:
                translated_map[k] = ""
                cached_hits += 1
                continue
            if norm_orig in interj_dict and len(norm_orig) <= 4:
                translated_map[k] = interj_dict[norm_orig]
                cached_hits += 1
                continue

            # 2. Проверка локального кэша уже переведенных фраз (0 токенов при перепереводе страниц и повторах!)
            cache_key = f"{t_lang}||{orig_raw}"
            if use_cache and cache_key in self.phrase_cache and self.phrase_cache[cache_key]:
                translated_map[k] = self.phrase_cache[cache_key]
                cached_hits += 1
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

        logger.info(
            "Перевод блоков: всего=%d, отправлено в ИИ=%d (сэкономлено из кэша/повторов=%d, лимит пачки=%d, модель=%s)",
            len(blocks),
            len(unique_to_translate),
            cached_hits,
            batch_limit,
            active_model,
        )

        running_context: list[str] = list(self.recent_context_lines)
        if context_lines:
            for cl in context_lines:
                if cl and cl not in running_context:
                    running_context.append(cl)
        running_context = running_context[-6:]

        # Для облачного API при большом числе фраз (>42 блоков, например с 12 страниц)
        # делим их на 2–3 параллельных подзапроса по ~40 фраз и отправляем ОДНОВРЕМЕННО через asyncio.gather!
        # Это сокращает время генерации ответа с 50–55 секунд до 14–18 секунд и исключает пропуски ключей в конце длинного JSON.
        if is_cloud_call and len(unique_to_translate) > CLOUD_PARALLEL_CHUNK_SIZE:
            num_parallel = min(
                MAX_CONCURRENT_CLOUD_REQUESTS,
                max(2, (len(unique_to_translate) + CLOUD_PARALLEL_CHUNK_SIZE - 1) // CLOUD_PARALLEL_CHUNK_SIZE),
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

            async def _run_sub(sub_c: list[dict[str, Any]]) -> dict[str, str]:
                async with sem:
                    if self.stop_requested:
                        return {}
                    return await self._translate_batch_chunk(
                        client=client,
                        chunk_blocks=sub_c,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        active_model=active_model,
                        context_lines=ctx_snapshot,
                    )

            sub_results = await asyncio.gather(*(_run_sub(sc) for sc in parallel_chunks))
            for sc, chunk_map in zip(parallel_chunks, sub_results):
                translated_map.update(chunk_map)
                for b in sc:
                    k = str(b["id"])
                    orig_raw = (b.get("original_text") or "").strip()
                    t_val = chunk_map.get(k, "")
                    if t_val and orig_raw:
                        self.phrase_cache[f"{t_lang}||{orig_raw}"] = t_val
                        if len(orig_raw) >= 4 and len(t_val) >= 4:
                            pair_str = f"{orig_raw} => {t_val}"
                            if pair_str not in running_context:
                                running_context.append(pair_str)
            running_context = running_context[-6:]
        else:
            for start_idx in range(0, len(unique_to_translate), batch_limit):
                if self.stop_requested:
                    break
                chunk = unique_to_translate[start_idx : start_idx + batch_limit]
                chunk_map = await self._translate_batch_chunk(
                    client=client,
                    chunk_blocks=chunk,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    active_model=active_model,
                    context_lines=running_context if running_context else None,
                )
                translated_map.update(chunk_map)

                for b in chunk:
                    k = str(b["id"])
                    orig_raw = (b.get("original_text") or "").strip()
                    t_val = chunk_map.get(k, "")
                    if t_val and orig_raw:
                        self.phrase_cache[f"{t_lang}||{orig_raw}"] = t_val
                        # В контекст добавляем только содержательные реплики (не короткие вздохи), чтобы экономить токены
                        if len(orig_raw) >= 4 and len(t_val) >= 4:
                            pair_str = f"{orig_raw} => {t_val}"
                            if pair_str not in running_context:
                                running_context.append(pair_str)
                running_context = running_context[-6:]

        # Заполняем дубликаты внутри пачки
        for dup_k, master_k in duplicate_links.items():
            translated_map[dup_k] = translated_map.get(master_k, "")

        self.recent_context_lines = list(running_context[-6:])
        if external_cache is not None:
            external_cache.update(self.phrase_cache)

        enriched_blocks: list[dict[str, Any]] = []
        for block in blocks:
            key = str(block["id"])
            orig_raw = (block.get("original_text") or "").strip()
            translated_text = translated_map.get(key, "").strip()
            if eff_skip_sighs and translated_text and is_translated_sigh_text(translated_text, orig_raw):
                logger.debug(
                    "Перевод '%s' (ориг. '%s') пропущен как короткий вздох (skip_sighs=True).",
                    translated_text,
                    orig_raw,
                )
                translated_text = ""
            block_copy = dict(block)
            block_copy["translated_text"] = translated_text
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
