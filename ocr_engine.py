"""
Мультиязычный модуль детекции бабблов, свободных надписей и распознавания текста (v1.0.7 — Шаги 1 + 2 + 3 + 4).

Изменения v1.0.7:
- Шаг 1 (Двухконтурная детекция и точный кроп текста):
  1. 3-классовый детектор `ogkalu/comic-text-and-bubble-detector` (RT-DETR-v2 r50vd) в тандеме с `YOLOv8m` и OpenCV:
     `0: bubble` (контур облачка), `1: text_bubble` (точный внутренний бокс текста `text_bbox`), `2: text_free` (свободный текст).
  2. Умное сопоставление `_associate_bubbles_and_text_boxes` с авто-разделением сдвоенных бабблов (`_try_split_bubble_by_text_boxes`)
     и спасением «сиротских» `text_bubble`.
  3. Плотная подгонка кропа по чернилам букв (`_compute_tight_ink_bbox` + `_extract_bubble_interior_mask`).
- Шаг 2 (Гибридный японский OCR 2026 года и умный препроцессинг):
  1. Неавторегрессионный распознаватель `muscgab/manga-ocr-nar-small` (~35 мс на страницу, CER 1.73%) в каскаде с
     `MangaOCR FP16` (`max_new_tokens=300`) для длинных монологов (`>= 56` символов).
  2. Авто-инверсия белого текста на темном фоне и очистка серого скринтона (`_preprocess_crop_np_for_ocr`) +
     защита пропорций узких/широких кропов (`_pad_to_balanced_aspect_for_vit`).
- Шаг 3 (Кардинальное улучшение корейского `ko` и китайского `zh` OCR + построчная сегментация + автоопределение языка):
  1. Подключена нативная мультиязычная PyTorch-модель 2026 года `JustANormalTinkerer/hayai-ocr-v2.5-nova`
     (`SigLIP2 NaFlex` + 12-слойный GQA декодер, поддержка `JA`, `KO`, `ZH`, `EN` с сохранением пропорций `max_num_patches=384/512`,
     `repetition_penalty=1.0`, нормализацией `NFKC` и удалением пробелов между CJK-символами) без установки новых C++/ONNX пакетов.
  2. Поддержка опционального `RapidOCR` (`rapidocr_onnxruntime`) и `EasyOCR` как быстрых резервных/дополнительных движков.
  3. Умная построчная сегментация многострочных горизонтальных бабблов (`_split_horizontal_lines_for_ocr`) по профилю
     проекции чернил (`row_ink_profile`) и строгая вертикально-горизонтальная сортировка строк (`_sort_ocr_line_items_top_to_bottom`),
     гарантирующая чтение 2–6 строк вебтунов строго сверху вниз без потери коротких реплик и знаков препинания.
  4. Улучшенное автоопределение языка `detect_page_language` (`auto`) на базе мультиязычного `HayaiOCR v2.5 Nova`,
     `manga-ocr-nar-small`, `EasyOCR`/`RapidOCR` и анализа ориентации штрихов.
- Шаг 4 (Опциональный «снайперский» VLM-OCR для ручного выделения `➕ Обвести`, сложных рукописных SFX и цветных обложек):
  1. Метод `recognize_crop_via_vlm` и подготовка цветного LANCZOS-кропа `_prepare_crop_for_vlm_ocr` для чтения сложных
     рукописных/каллиграфических шрифтов, звуковых спецэффектов (SFX) и цветных логотипов через локальные Vision-модели
     в Ollama (`glm-ocr`, `qwen2.5vl`, `qwen3-vl`, `gemma3`, `minicpm-v`, `llava`) или облачный Vision-совместимый API.
  2. Автоматический и принудительный (`use_vlm_ocr=True`) вызов VLM-OCR в `recognize_manual_region` (при обведении мышкой
     `➕ Обвести` или по нажатию кнопки `👁 VLM-OCR` в попапе/редакторе).
"""

from __future__ import annotations

import base64
import io
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from manga_ocr import MangaOcr
from PIL import Image, ImageOps
from ultralytics import YOLO

logger = logging.getLogger(__name__)

HF_YOLO_REPO = "ogkalu/comic-speech-bubble-detector-yolov8m"
HF_YOLO_FILENAME = "comic-speech-bubble-detector.pt"

HF_RTDETR_REPO = "ogkalu/comic-text-and-bubble-detector"
HF_RTDETR_SUBDIR = "comic-text-and-bubble-detector"
HF_RTDETR_PT_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")

HF_JA_NAR_REPO = "muscgab/manga-ocr-nar-small"
HF_JA_NAR_SUBDIR = "manga-ocr-nar-small"
HF_JA_NAR_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.py",
    "*.toml",
    "manga_ocr_nar_small/*",
    "manga_ocr_nar_small/**/*",
    "mangaocr_nar_small/*",
    "mangaocr_nar_small/**/*",
)

HF_HAYAI_REPO = "JustANormalTinkerer/hayai-ocr-v2.5-nova"
HF_HAYAI_SUBDIR = "hayai-ocr-v2.5-nova"
HF_HAYAI_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.py",
    "*.txt",
    "*.model",
)

HF_HAYAI_PROCESSOR_REPO = "google/siglip2-base-patch16-naflex"
HF_HAYAI_PROCESSOR_SUBDIR = "siglip2-base-patch16-naflex"
HF_HAYAI_PROCESSOR_ALLOW_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
)

# Приоритет локальных Vision-моделей в Ollama для снайперского чтения сложных SFX и рукописных шрифтов (Шаг 4)
VLM_OCR_MODEL_PRIORITIES: tuple[str, ...] = (
    "glm-ocr",
    "qwen3-vl",
    "qwen2.5vl",
    "qwen2-vl",
    "minicpm-v",
    "gemma3",
    "llama3.2-vision",
    "llava",
    "pixtral",
    "internvl",
)

VLM_OCR_PREFIX_STRIP_REGEX = re.compile(
    r"^(?:the\s+text\s+(?:in\s+the\s+image\s+)?is|transcribed\s+text|transcription|ocr\s+result|result|output|answer|ocr|text|"
    r"текст\s+на\s+изображении|распознанный\s+текст|текст)\s*[:：\-—]+\s*",
    flags=re.IGNORECASE,
)

HANGUL_REGEX = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
KANA_REGEX = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
KATAKANA_ONLY_REGEX = re.compile(r"^[\u30a0-\u30ff]+$")
HANZI_REGEX = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
CJK_SPACE_STRIP_REGEX = re.compile(
    r"([\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf\uac00-\ud7af\u1100-\u11ff\u3130-\u318f])\s+(?=[\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf\uac00-\ud7af\u1100-\u11ff\u3130-\u318f])"
)
MEANINGFUL_CHAR_REGEX = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7afA-Za-zА-Яа-яЁё0-9]"
)
PUNCT_ONLY_REGEX = re.compile(
    r"^[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\（\）\(\)\[\]\♡\♥\♪\★\☆\・\：\;\"\'\`]+$"
)
STRIP_PUNCT_REGEX = re.compile(
    r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\（\）\(\)\[\]\♡\♥\♪\★\☆\・\：\;\"\'\`]+"
)
REPEATED_NOISE_REGEX = re.compile(r"^(.)\1{3,}$")
REDUPLICATED_SFX_REGEX = re.compile(r"^(.{1,3})\1+$")

# Известные галлюцинации декодеров MangaOCR / manga-ocr-nar-small / HayaiOCR на пустых/шумовых кропах фона и штриховки
# (например, заученная из синтетического корпуса фраза «タクシー運転手さん一番うまい店に連れてって» -> «Taxi driver, take me to the best restaurant!»)
KNOWN_OCR_HALLUCINATIONS_REGEX = re.compile(
    r"(?i)(?:タクシー運転手|運転手さん.*(?:うまい店|美味しい店|レストラン|連れて)|一番(?:うまい|美味しい)(?:店|レストラン).*連れて|"
    r"taxi\s*driver.*(?:best\s*restau?rant|take\s*me)|take\s*me\s*to\s*the\s*best\s*restau?rant|"
    r"ご視聴ありがとうございました|チャンネル登録.*お願い|高評価.*チャンネル登録)"
)

# Словарь ТОЛЬКО чистых вздохов, стонов и охов (без диалоговых слов типа いや / やだ / おい / うん!)
SIGH_EXACT_SET: set[str] = {
    # Только чистые бессловесные вздохи и стоны (хирагана/катакана) — без диалоговых реакций вроде えっ/くっ/ちっ/わっ!
    "あ", "あっ", "ああ", "ああっ", "あぁ", "あぁっ", "あん", "あんっ", "ああん", "あف", "あふっ",
    "う", "うっ", "うぅ", "うぅっ", "うう", "ううっ",
    "ん", "んっ", "んん", "んんっ", "んぅ", "んぅっ", "んあ", "んあっ",
    "はぁ", "はぁっ", "はぁはぁ", "はあ", "はあっ", "はあはあ", "はふ", "はふっ",
    "ふぅ", "ふぅっ", "ふう", "ふうっ",
    "ア", "アッ", "アア", "アアッ", "アァ", "アァッ", "アン", "アンッ",
    "ウ", "ウッ", "ウゥ", "ウゥッ",
    "ン", "ンッ", "ンン", "ンンッ", "ンゥ",
    "ハァ", "ハァッ", "ハァハァ", "ハア", "ハアッ",
    "フゥ", "フゥッ", "フウ",
    # Корейские чистые вздохи
    "아", "아앗", "아아", "우", "우웃", "으", "으음", "음", "하아", "하앗", "후우", "후웃",
    # Китайские чистые вздохи
    "啊", "啊啊", "嗯", "嗯嗯", "唔", "呼",
    # Русские и английские чистые вздохи
    "а", "о", "у", "м", "мм", "ммм", "ах", "аах", "ахн", "ох", "оох", "хаа",
    "a", "o", "u", "m", "mm", "mmm", "mmh", "ah", "aah", "ahn", "ooh", "haa",
}

# Слова из гласных и короткие реакции, которые являются осмысленными репликами диалога, а НЕ вздохами
DIALOGUE_VOWEL_WORDS_JA: set[str] = {
    "おい", "オイ", "いい", "イイ", "いいえ", "イイエ", "うん", "ウン",
    "ええ", "エエ", "え", "えっ", "えぇ", "エ", "エッ", "エェ",
    "お", "おっ", "おお", "オ", "オッ", "オオ",
    "おう", "オウ", "あい", "アイ", "うえ", "ウエ",
    "あおい", "アオイ", "おおい", "オオイ", "いう", "イウ", "あう", "アウ",
    "わっ", "ワッ", "くっ", "クッ", "ぐっ", "グッ", "ちっ", "チッ", "ほっ", "ホッ", "へっ", "ヘッ",
}

# Регулярное выражение только для комбинаций чистых вздохов/стонов («あ…あっ», «はぁ…はぁ…», «んっ…あ»)
SIGH_COMBO_REGEX = re.compile(
    r"^(?:[あぁアァ]+[っッー]*|[うぅウゥ]+[っッー]*|[んン]+[っッー]*|"
    r"はぁ|はあ|ハァ|ハア|ふぅ|ふう|フゥ|フウ|"
    r"あん|アン|あっ|アッ|うっ|ウッ|んっ|ンッ|あふ|アフ)+$"
)

# Словарь частых звуковых спецэффектов манги и манхвы (SFX / ономатопея)
SFX_EXACT_SET: set[str] = {
    "ドキ", "ドキッ", "ドキドキ", "ドクン", "ドクッ", "ビク", "ビクッ", "ビクン", "ビクビク",
    "ギュ", "ギュッ", "ギュウ", "ゾク", "ゾクッ", "ゾクゾク", "パン", "パアン", "バン", "バアン",
    "ドン", "ドオン", "ドーン", "ゴゴ", "ゴゴゴ", "ゴゴゴゴ", "ザワ", "ザワザワ", "シーン", "シィン",
    "ガタ", "ガタッ", "ガタガタ", "パッ", "スッ", "チュ", "チュッ", "チュウ", "ジュワ", "ジュワッ",
    "ヌル", "ヌルッ", "ヌルヌル", "ピチャ", "ピチャッ", "ズキ", "ズキッ", "ズキズキ", "ワク", "ワクワク",
    "ニコ", "ニコッ", "ニコニコ", "チラ", "チラッ", "チラチラ", "フワ", "フワッ", "フワフワ",
    "ボッ", "カッ", "ギシ", "ギシッ", "ギシギシ", "ミシ", "ミシッ", "バタン", "ガラ", "ガラッ", "ガラガラ",
    "スタ", "スタスタ", "トコ", "トコトコ", "ペコ", "ペコッ", "クス", "クスッ", "クスクス", "ゴクリ", "ゴクッ",
    "ブル", "ブルッ", "ブルブル", "ガク", "ガクガク", "ガシャン", "ドサ", "ドサッ", "バサ", "バサッ",
    "サッ", "シュ", "シュッ", "グッ", "ピタ", "ピタッ", "クル", "クルッ", "パチ", "パチッ", "パチパチ",
    "ズッ", "ブッ", "プッ", "ブチュ", "ジュル", "ジュルッ", "レロ", "レロレロ", "ペロ", "ペロッ",
    "モミ", "モミモミ", "ズブ", "ズブッ", "ズプ", "ズプッ", "グチュ", "グチュッ", "パンッ", "パチン",
    "ピ질", "두근", "두근두근", "움찔", "덜컥", "쾅", "퍽", "짝", "스륵", "부들", "부들부들",
    "바르르", "꿀꺽", "주륵", "반짝", "휘잉", "철컥", "쿵", "쿵쿵", "탁", "휙", "짝짝",
    "ぷる", "ぷるぷる", "びく", "びくっ", "びくん", "どき", "どきっ", "どきどき", "ざわ", "ざわざわ",
    "ぴく", "ぴくっ", "ぎゅ", "ぎゅっ", "ちゅ", "ちゅっ", "ぱん", "どん", "ばん", "ずき", "ずきっ",
    "ぞく", "ぞくっ", "ぬる", "ぬるっ", "じゅわ", "じゅわっ", "じゅる", "れろ", "ぺろ", "もみ", "ぐちゅ",
}


class MangaOCREngine:
    """
    Мультиязычный движок v1.0.7 (Шаги 1 + 2 + 3 + 4):
    - Двухконтурная детекция (`RT-DETR-v2` 3-class `bubble` / `text_bubble` / `text_free` + `YOLOv8m` + OpenCV)
    - Плотная подгонка кропа по чернилам (`_compute_tight_ink_bbox`)
    - Японский OCR (`muscgab/manga-ocr-nar-small` + `MangaOCR FP16`)
    - Мультиязычный OCR для корейской манхвы (`ko`) и китайской маньхуа (`zh`):
      `JustANormalTinkerer/hayai-ocr-v2.5-nova` (`SigLIP2 NaFlex`) + построчная сегментация (`_split_horizontal_lines_for_ocr`)
      + опциональные `RapidOCR` / `EasyOCR`.
    - Снайперский VLM-OCR (`recognize_crop_via_vlm`) через локальные Vision-модели Ollama (`glm-ocr`, `qwen2.5vl`,
      `qwen3-vl`, `gemma3`, `minicpm-v`, `llava`) или облачный Vision API для сложных рукописных SFX и обложек.
    """

    def __init__(
        self,
        models_dir: str | Path = "models",
        device: str | None = None,
        conf_threshold: float = 0.22,
        iou_threshold: float = 0.45,
        imgsz: int = 1280,
        use_rtdetr: bool = True,
        use_ja_nar: bool = True,
        use_hayai_ocr: bool = True,
        ollama_base_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.use_rtdetr = bool(use_rtdetr)
        self.use_ja_nar = bool(use_ja_nar)
        self.use_hayai_ocr = bool(use_hayai_ocr)
        self.ollama_base_url = ollama_base_url.rstrip("/")

        logger.info("Инициализация MangaOCREngine (v1.0.7 Step 1+2+3+4) на устройстве: %s", self.device)
        self.detector = self._load_detector()
        self.ja_recognizer = self._load_manga_ocr()

        self._ja_nar_recognizer: Any | None = None
        self._ja_nar_available: bool | None = None if self.use_ja_nar else False

        self._hayai_model: Any | None = None
        self._hayai_tokenizer: Any | None = None
        self._hayai_processor: Any | None = None
        self._hayai_available: bool | None = None if self.use_hayai_ocr else False

        self._rtdetr_model: Any | None = None
        self._rtdetr_processor: Any | None = None
        self._rtdetr_available: bool | None = None if self.use_rtdetr else False
        self._last_rtdetr_free_boxes: list[dict[str, Any]] = []

        self._rapidocr_reader: Any | None = None
        self._rapidocr_available: bool | None = None

        self._ko_reader: Any | None = None
        self._zh_reader: Any | None = None
        self._craft_reader: Any | None = None
        self._easyocr_available: bool | None = None

        self._last_vlm_model_used: str = ""
        self._vlm_ocr_calls_count: int = 0

        self._page_ocr_cache: dict[tuple[int, int, int, int, str], str] = {}
        self._last_detected_lang: str = "ja"
        self.last_page_diagnostics: dict[str, Any] = {}
        self.last_detection_stats: dict[str, Any] = {
            "detector_backend": "YOLOv8m + TightInkCrop",
            "rtdetr_active": False,
            "hayai_active": False,
            "rapidocr_active": False,
            "vlm_ocr_backend": "Auto (Ollama GLM-OCR / Qwen2.5-VL / Gemma3 / Cloud Vision)",
            "vlm_ocr_calls": 0,
            "yolo_bubbles": 0,
            "rtdetr_bubbles": 0,
            "rtdetr_text_bubbles": 0,
            "rtdetr_free_texts": 0,
            "split_multi_bubbles": 0,
            "rescued_orphan_text_bubbles": 0,
        }

        # Пробуем сразу инициализировать 3-классовый детектор RT-DETR-v2 (ogkalu/comic-text-and-bubble-detector)
        if self.use_rtdetr:
            self._get_rtdetr_detector()

        # Пробуем инициализировать скоростной высокоточный японский OCR 2026 года (muscgab/manga-ocr-nar-small)
        if self.use_ja_nar:
            self._get_ja_nar_recognizer()

        # Если мультиязычная модель HayaiOCR v2.5 Nova уже скачана на диск, подгружаем её сразу,
        # иначе она скачается автоматически по первому запросу ko / zh / auto без задержки старта для чистого ja
        if self.use_hayai_ocr and self._is_hayai_cached_locally():
            self._get_hayai_recognizer()

        self.last_detection_stats["ja_ocr_backend"] = self.get_ja_ocr_backend_name()
        self.last_detection_stats["ko_zh_ocr_backend"] = self.get_multilang_ocr_backend_name()
        self.last_detection_stats["vlm_ocr_backend"] = self.get_vlm_ocr_backend_name()
        logger.info(
            "Базовые модели загружены (Детектор: %s | OCR JA: %s | OCR KO/ZH: %s | VLM-OCR: %s).",
            self.get_detector_backend_name(),
            self.get_ja_ocr_backend_name(),
            self.get_multilang_ocr_backend_name(),
            self.get_vlm_ocr_backend_name(),
        )

    def get_detector_backend_name(self) -> str:
        if self._rtdetr_available and self._rtdetr_model is not None:
            return "Hybrid RT-DETR-v2 (3-class: bubble + text_bubble + text_free) + YOLOv8m + TightInkCrop"
        return "YOLOv8m + OpenCV Contours + TightInkCrop"

    def get_ja_ocr_backend_name(self) -> str:
        if self._ja_nar_available and self._ja_nar_recognizer is not None:
            return "Hybrid manga-ocr-nar-small (v2026, CER 1.73%) + MangaOCR FP16 (long-bubble fallback)"
        return "MangaOCR FP16 + SmartPreprocess (Auto-Invert & AspectGuard)"

    def get_multilang_ocr_backend_name(self) -> str:
        parts: list[str] = []
        if self._hayai_available and self._hayai_model is not None:
            parts.append("HayaiOCR v2.5 Nova (SigLIP2 NaFlex JA/KO/ZH/EN)")
        elif self.use_hayai_ocr and self._hayai_available is not False:
            parts.append("HayaiOCR v2.5 Nova (Auto-Load JA/KO/ZH/EN)")
        if self._check_rapidocr():
            parts.append("RapidOCR")
        if self._check_easyocr():
            parts.append("EasyOCR")
        parts.append("SmartLineSplit")
        return " + ".join(parts)

    def get_vlm_ocr_backend_name(self) -> str:
        if self._last_vlm_model_used:
            return f"Sniper VLM-OCR ({self._last_vlm_model_used}, вызовов: {self._vlm_ocr_calls_count})"
        return "Sniper VLM-OCR (Ollama GLM-OCR / Qwen2.5-VL / Gemma3 / Cloud Vision)"

    def _is_hayai_cached_locally(self) -> bool:
        local_dir = self.models_dir / HF_HAYAI_SUBDIR
        return (local_dir / "model.safetensors").exists() and (local_dir / "config.json").exists()

    def _get_hayai_recognizer(self) -> tuple[Any, Any, Any] | None:
        """
        Загружает мультиязычную OCR-модель 2026 года `JustANormalTinkerer/hayai-ocr-v2.5-nova` (~150M параметров,
        SigLIP2 NaFlex + 12-слойный GQA декодер с 4x DSCProjector).
        Поддерживает корейский (`KO`), китайский (`ZH`), японский (`JA`) и английский (`EN`),
        сохраняя нативные пропорции баббла (`max_num_patches=384/512`) и работая напрямую через `transformers`
        без установки сторонних C++/ONNX пакетов.
        """
        if not self.use_hayai_ocr or self._hayai_available is False:
            return None
        if (
            self._hayai_model is not None
            and self._hayai_tokenizer is not None
            and self._hayai_processor is not None
        ):
            return self._hayai_model, self._hayai_tokenizer, self._hayai_processor

        local_dir = self.models_dir / HF_HAYAI_SUBDIR
        proc_dir = self.models_dir / HF_HAYAI_PROCESSOR_SUBDIR
        local_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)

        try:
            weights_file = local_dir / "model.safetensors"
            config_file = local_dir / "config.json"
            if not weights_file.exists() or not config_file.exists():
                logger.info(
                    "Скачивание мультиязычной OCR-модели (KO/ZH/JA/EN) %s...",
                    HF_HAYAI_REPO,
                )
                snapshot_download(
                    repo_id=HF_HAYAI_REPO,
                    local_dir=str(local_dir),
                    allow_patterns=list(HF_HAYAI_ALLOW_PATTERNS),
                )

            import transformers
            from transformers import AutoModel, AutoProcessor, PreTrainedTokenizerFast

            # Загружаем быстрый токенизатор HayaiOCR
            try:
                tokenizer = PreTrainedTokenizerFast.from_pretrained(str(local_dir))
            except Exception:
                auto_tok_cls = getattr(transformers, "AutoTokenizer", None)
                if auto_tok_cls is None:
                    raise
                tokenizer = auto_tok_cls.from_pretrained(str(local_dir), trust_remote_code=True)

            # Загружаем процессор SigLIP2 NaFlex (скачиваем только легкие JSON-конфиги без весов SigLIP2!)
            processor = None
            try:
                if (local_dir / "preprocessor_config.json").exists():
                    processor = AutoProcessor.from_pretrained(str(local_dir))
            except Exception:
                processor = None

            if processor is None:
                if not (proc_dir / "preprocessor_config.json").exists():
                    logger.info(
                        "Скачивание конфигурации процессора %s...",
                        HF_HAYAI_PROCESSOR_REPO,
                    )
                    snapshot_download(
                        repo_id=HF_HAYAI_PROCESSOR_REPO,
                        local_dir=str(proc_dir),
                        allow_patterns=list(HF_HAYAI_PROCESSOR_ALLOW_PATTERNS),
                    )
                try:
                    processor = AutoProcessor.from_pretrained(str(proc_dir))
                except Exception:
                    img_proc_cls = getattr(transformers, "AutoImageProcessor", None)
                    if img_proc_cls is not None:
                        processor = img_proc_cls.from_pretrained(str(proc_dir))
                    else:
                        processor = AutoProcessor.from_pretrained(HF_HAYAI_PROCESSOR_REPO)

            model = AutoModel.from_pretrained(str(local_dir), trust_remote_code=True)
            model.to(self.device)
            model.eval()

            self._hayai_model = model
            self._hayai_tokenizer = tokenizer
            self._hayai_processor = processor
            self._hayai_available = True
            self.last_detection_stats["hayai_active"] = True
            self.last_detection_stats["ko_zh_ocr_backend"] = self.get_multilang_ocr_backend_name()
            logger.info(
                "Мультиязычная модель %s (KO/ZH/JA/EN) успешно подключена на %s!",
                HF_HAYAI_REPO,
                self.device,
            )
            return self._hayai_model, self._hayai_tokenizer, self._hayai_processor
        except Exception as exc:
            logger.warning(
                "Модель %s недоступна (%s). Используем резервный конвейер SmartLineSplit + RapidOCR/EasyOCR.",
                HF_HAYAI_REPO,
                exc,
            )
            self._hayai_available = False
            self._hayai_model = None
            self._hayai_tokenizer = None
            self._hayai_processor = None
            self.last_detection_stats["hayai_active"] = False
            self.last_detection_stats["ko_zh_ocr_backend"] = self.get_multilang_ocr_backend_name()
            return None

    @staticmethod
    def _normalize_hayai_ocr_text(text: str, lang: str = "auto") -> str:
        """
        Нормализация выхода `hayai-ocr-v2.5-nova` согласно спецификации модели:
        1. Каноническая декомпозиция/композиция Unicode `NFKC`.
        2. Замена переводов строк и табуляций на пробел.
        3. Удаление пробелов между CJK-символами (кандзи, кана, ханьцзы, хангыль) через lookahead-регулярку,
           чтобы цепочки из 3+ символов склеивались за один проход без остаточных пробелов.
        """
        if not text:
            return ""
        norm = unicodedata.normalize("NFKC", str(text))
        norm = re.sub(r"[\r\n\t]+", " ", norm)
        norm = CJK_SPACE_STRIP_REGEX.sub(r"\1", norm)
        return re.sub(r"\s+", " ", norm).strip()

    @classmethod
    def _prepare_crop_for_vlm_ocr(
        cls,
        rgb_image: Image.Image,
        bbox: list[int] | tuple[int, int, int, int],
        pad: int = 12,
    ) -> tuple[Image.Image, str]:
        """
        Готовит высококачественный цветной кроп для снайперского VLM-OCR (Шаг 4):
        - Сохраняет оригинальный цветной контур букв (важно для градиентных SFX, цветных обложек и полупрозрачных надписей).
        - Плавно масштабирует мелкие кропы (`max(w, h) < 180`) через `LANCZOS`, чтобы мультимодальный энкодер
          четко различал мелкие черточки, фуригану и дакутэн.
        - Возвращает `(pil_crop, base64_png_str)`.
        """
        img_w, img_h = rgb_image.size
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        cx1 = max(0, x1 - 6)
        cy1 = max(0, y1 - 6)
        cx2 = min(img_w, x2 + 6)
        cy2 = min(img_h, y2 + 6)
        crop = rgb_image.crop((cx1, cy1, cx2, cy2)).convert("RGB")
        cw, ch = crop.size
        max_dim = max(cw, ch)
        if 0 < max_dim < 180:
            scale = min(2.5, max(1.5, 240.0 / float(max_dim)))
            new_w = max(16, int(round(cw * scale)))
            new_h = max(16, int(round(ch * scale)))
            resample_filter = getattr(Image, "Resampling", Image).LANCZOS
            crop = crop.resize((new_w, new_h), resample=resample_filter)
        elif max_dim > 1024:
            scale = 1024.0 / float(max_dim)
            new_w = max(16, int(round(cw * scale)))
            new_h = max(16, int(round(ch * scale)))
            resample_filter = getattr(Image, "Resampling", Image).LANCZOS
            crop = crop.resize((new_w, new_h), resample=resample_filter)

        if pad > 0:
            crop = ImageOps.expand(crop, border=pad, fill=(255, 255, 255))

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        b64_png = base64.b64encode(buf.getvalue()).decode("ascii")
        return crop, b64_png

    @classmethod
    def _clean_vlm_ocr_output(cls, raw_text: str, fallback_lang: str = "ja") -> tuple[str, str]:
        """
        Очищает ответ VLM-OCR от `<think>...</think>`, markdown-блоков, кавычек и пояснений,
        нормализует через `NFKC` и определяет язык распознанного текста (`ja`, `ko`, `zh`).
        """
        if not raw_text:
            return "", fallback_lang
        cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", str(raw_text), flags=re.DOTALL | re.IGNORECASE).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        cleaned = VLM_OCR_PREFIX_STRIP_REGEX.sub("", cleaned).strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            if len(cleaned) >= 2:
                cleaned = cleaned[1:-1].strip()
        norm = cls._normalize_hayai_ocr_text(cleaned)
        if not norm or PUNCT_ONLY_REGEX.match(norm):
            return "", fallback_lang

        det_lang = fallback_lang if fallback_lang in ("ja", "ko", "zh") else "ja"
        if HANGUL_REGEX.search(norm):
            det_lang = "ko"
        elif KANA_REGEX.search(norm):
            det_lang = "ja"
        elif HANZI_REGEX.search(norm):
            det_lang = "zh" if fallback_lang == "zh" else ("ja" if fallback_lang == "ja" else "zh")
        return norm, det_lang

    @staticmethod
    def _pick_best_ollama_vlm_model(installed_models: list[str], preferred_model: str | None = None) -> str | None:
        """
        Выбирает лучшую установленную в Ollama локальную Vision-модель для VLM-OCR:
        1. Если `preferred_model` явно указан и подходит — возвращает его.
        2. Иначе ищет по приоритету `VLM_OCR_MODEL_PRIORITIES`:
           `glm-ocr` -> `qwen3-vl` -> `qwen2.5vl` -> `qwen2-vl` -> `minicpm-v` -> `gemma3` -> `llama3.2-vision` -> `llava`.
        """
        if preferred_model and preferred_model.strip() and preferred_model.strip() != "auto":
            clean_pref = preferred_model.strip()
            if clean_pref.startswith("ollama::"):
                clean_pref = clean_pref.split("ollama::", 1)[1]
            for m in installed_models:
                if clean_pref.lower() == m.lower() or clean_pref.lower() in m.lower():
                    return m
            if any(k in clean_pref.lower() for k in VLM_OCR_MODEL_PRIORITIES):
                return clean_pref

        for prio in VLM_OCR_MODEL_PRIORITIES:
            for m in installed_models:
                if prio in m.lower():
                    return m
        return None

    def recognize_crop_via_vlm(
        self,
        rgb_image: Image.Image,
        bbox: list[int] | tuple[int, int, int, int],
        lang: str = "auto",
        vlm_model: str | None = None,
        ollama_base_url: str | None = None,
        cloud_api_url: str | None = None,
        cloud_api_key: str | None = None,
        cloud_vlm_model: str | None = None,
    ) -> tuple[str, str, str]:
        """
        Снайперское распознавание сложного кропа (рукописный SFX, каллиграфия, цветная обложка) через Vision-модель (Шаг 4):
        1. Сначала проверяет локальные Vision-модели в Ollama (`glm-ocr`, `qwen2.5vl`, `qwen3-vl`, `gemma3`, `minicpm-v`, `llava`)
           с обязательным `trust_env=False` (обход VPN Fake-IP `198.18.x.x`).
        2. Если в Ollama нет Vision-модели, пробует облачный Vision-совместимый API (`/v1/chat/completions` с `image_url`),
           если переданы `cloud_api_url` и `cloud_api_key`.
        Возвращает `(recognized_text, detected_lang, used_vlm_model_name)`.
        """
        req_lang = (lang or "auto").lower().strip()
        fb_lang = req_lang if req_lang in ("ja", "ko", "zh") else (self._last_detected_lang or "ja")
        _, b64_png = self._prepare_crop_for_vlm_ocr(rgb_image, bbox, pad=12)

        lang_hint = {
            "ja": "Japanese manga (Kanji, Hiragana, Katakana, or sound effects / SFX)",
            "ko": "Korean manhwa / webtoon (Hangul dialogue or SFX)",
            "zh": "Chinese manhua (Hanzi dialogue or SFX)",
        }.get(fb_lang, "Japanese, Korean, or Chinese comic/manga")

        vlm_prompt = (
            f"Transcribe ONLY the exact {lang_hint} text visible in this image crop "
            "(including stylized sound effects / SFX, handwritten text, or decorative logos). "
            "Do NOT translate, do NOT romanize, do NOT add explanations or quotes. "
            "Output ONLY the raw transcribed characters."
        )

        base_ol_url = (ollama_base_url or self.ollama_base_url or "http://127.0.0.1:11434").rstrip("/")

        # 1. Пробуем локальную Vision-модель в Ollama (с 30-секундным кэшем списка /api/tags)
        try:
            now_mono = time.monotonic()
            cached_ts = getattr(self, "_ollama_vlm_tags_ts", 0.0)
            installed_names: list[str] = list(getattr(self, "_ollama_vlm_tags_cache", []) or [])
            with httpx.Client(timeout=httpx.Timeout(25.0, connect=2.0), trust_env=False) as client:
                if (now_mono - cached_ts) > 30.0 or not hasattr(self, "_ollama_vlm_tags_cache"):
                    installed_names = []
                    try:
                        tags_resp = client.get(f"{base_ol_url}/api/tags")
                        if tags_resp.status_code == 200:
                            for item in tags_resp.json().get("models", []):
                                m_name = str(item.get("name") or "").strip()
                                if m_name:
                                    installed_names.append(m_name)
                        self._ollama_vlm_tags_cache = installed_names
                        self._ollama_vlm_tags_ts = now_mono
                    except Exception:
                        pass

                chosen_ol_vlm = self._pick_best_ollama_vlm_model(installed_names, preferred_model=vlm_model)
                if chosen_ol_vlm:
                    payload_ol: dict[str, Any] = {
                        "model": chosen_ol_vlm,
                        "prompt": vlm_prompt,
                        "images": [b64_png],
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 160,
                        },
                    }
                    gen_resp = client.post(f"{base_ol_url}/api/generate", json=payload_ol)
                    if gen_resp.status_code == 200:
                        raw_out = str(gen_resp.json().get("response") or "")
                        text_out, det_lang = self._clean_vlm_ocr_output(raw_out, fallback_lang=fb_lang)
                        if text_out and not KNOWN_OCR_HALLUCINATIONS_REGEX.search(text_out):
                            self._last_vlm_model_used = f"ollama::{chosen_ol_vlm}"
                            self._vlm_ocr_calls_count += 1
                            self.last_detection_stats["vlm_ocr_calls"] = self._vlm_ocr_calls_count
                            self.last_detection_stats["vlm_ocr_backend"] = self.get_vlm_ocr_backend_name()
                            logger.info(
                                "🎯 [VLM-OCR Ollama] Модель %s распознала текст: '%s' (lang=%s)",
                                chosen_ol_vlm,
                                text_out,
                                det_lang,
                            )
                            return text_out, det_lang, self._last_vlm_model_used
        except Exception as ol_exc:
            logger.debug("Локальный VLM-OCR через Ollama пропущен: %s", ol_exc)

        # 2. Пробуем облачный OpenAI-совместимый Vision API (с ретраями при сетевых таймаутах)
        if cloud_api_url and cloud_api_key:
            c_url = cloud_api_url.rstrip("/")
            for sfx in ("/v1/chat/completions", "/chat/completions", "/v1"):
                if c_url.endswith(sfx):
                    c_url = c_url[: -len(sfx)].rstrip("/")
                    break
            candidate_cloud_models: list[str] = []
            if cloud_vlm_model and cloud_vlm_model.strip() and cloud_vlm_model.strip() != "auto":
                cm = cloud_vlm_model.strip().split("::", 1)[-1]
                candidate_cloud_models.append(cm)
            for def_cm in ("deepseek-v4-flash", "deepseek-v4-pro", "gpt-4o-mini", "qwen-vl-max"):
                if def_cm not in candidate_cloud_models:
                    candidate_cloud_models.append(def_cm)

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {cloud_api_key.strip()}",
                "User-Agent": "MikoMangaStudio/1.0.8",
            }
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), trust_env=False) as client:
                for c_mod in candidate_cloud_models[:2]:
                    payload_cloud = {
                        "model": c_mod,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": vlm_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                                    },
                                ],
                            }
                        ],
                        "temperature": 0.0,
                        "max_tokens": 160,
                        "stream": False,
                    }
                    for attempt in range(2):
                        try:
                            c_resp = client.post(
                                f"{c_url}/v1/chat/completions",
                                headers=headers,
                                json=payload_cloud,
                            )
                            if c_resp.status_code == 200:
                                choices = c_resp.json().get("choices") or []
                                if choices and isinstance(choices[0], dict):
                                    raw_c = str((choices[0].get("message") or {}).get("content") or "")
                                    text_out, det_lang = self._clean_vlm_ocr_output(raw_c, fallback_lang=fb_lang)
                                    if text_out and not KNOWN_OCR_HALLUCINATIONS_REGEX.search(text_out):
                                        self._last_vlm_model_used = f"cloud::{c_mod}"
                                        self._vlm_ocr_calls_count += 1
                                        self.last_detection_stats["vlm_ocr_calls"] = self._vlm_ocr_calls_count
                                        self.last_detection_stats["vlm_ocr_backend"] = self.get_vlm_ocr_backend_name()
                                        logger.info(
                                            "🎯 [VLM-OCR Cloud] Модель %s распознала текст: '%s' (lang=%s)",
                                            c_mod,
                                            text_out,
                                            det_lang,
                                        )
                                        return text_out, det_lang, self._last_vlm_model_used
                                break
                            elif c_resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                                time.sleep(0.8)
                                continue
                            else:
                                break
                        except Exception as c_exc:
                            if attempt == 0:
                                logger.debug("Повтор VLM-OCR Cloud (%s) после сетевого сбоя: %s", c_mod, c_exc)
                                time.sleep(0.6)
                                continue
                            logger.warning("Сбой облачного VLM-OCR (%s): %s", c_mod, c_exc)

        return "", fb_lang, ""

    def _recognize_crops_batch_hayai(
        self,
        crops_pil: list[Image.Image],
        max_num_patches: int | None = None,
        max_new_tokens: int = 128,
        batch_size: int = 8,
        lang: str = "auto",
    ) -> list[str]:
        """
        Пакетное и поштучное распознавание кропов через `JustANormalTinkerer/hayai-ocr-v2.5-nova`
        с параметром `repetition_penalty=1.0` и нормализацией `NFKC`.
        """
        if not crops_pil:
            return []
        hayai_triple = self._get_hayai_recognizer()
        if hayai_triple is None:
            return [""] * len(crops_pil)

        model, tokenizer, processor = hayai_triple
        rgb_crops = [c.convert("RGB") for c in crops_pil]
        outputs_norm: list[str] = []

        def _choose_patch_budget(imgs: list[Image.Image]) -> int:
            if max_num_patches is not None:
                return int(max_num_patches)
            for im in imgs:
                w, h = im.size
                aspect = max(w, h) / max(1.0, float(min(w, h)))
                if (w * h) >= 42000 or aspect >= 2.6:
                    return 512
            return 384

        def _run_single_crop(img: Image.Image) -> str:
            patch_budget = _choose_patch_budget([img])
            inputs = processor(images=[img], max_num_patches=patch_budget, return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            gen_kwargs: dict[str, Any] = {
                "pixel_values": inputs["pixel_values"],
                "tokenizer": tokenizer,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": 1.0,
            }
            if "pixel_attention_mask" in inputs:
                gen_kwargs["pixel_attention_mask"] = inputs["pixel_attention_mask"]
            if "spatial_shapes" in inputs:
                gen_kwargs["spatial_shapes"] = inputs["spatial_shapes"]

            raw_out = model.generate(**gen_kwargs)
            if isinstance(raw_out, str):
                raw_txt = raw_out
            elif isinstance(raw_out, (list, tuple)) and len(raw_out) > 0:
                first_el = raw_out[0]
                if isinstance(first_el, str):
                    raw_txt = first_el
                else:
                    raw_txt = tokenizer.decode(first_el, skip_special_tokens=True)
            elif hasattr(raw_out, "shape"):
                decoded = tokenizer.batch_decode(raw_out, skip_special_tokens=True)
                raw_txt = decoded[0] if decoded else ""
            else:
                raw_txt = str(raw_out)
            return self._normalize_hayai_ocr_text(raw_txt, lang=lang)

        with torch.inference_mode():
            for start_i in range(0, len(rgb_crops), max(1, batch_size)):
                sub_crops = rgb_crops[start_i : start_i + max(1, batch_size)]
                if len(sub_crops) == 1:
                    try:
                        outputs_norm.append(_run_single_crop(sub_crops[0]))
                    except Exception as single_exc:
                        logger.debug("HayaiOCR single crop error: %s", single_exc)
                        outputs_norm.append("")
                    continue

                # Пробуем батч-прогон; если кастомный generate() модели требует B=1, бесшовно переходим на поштучный цикл
                batch_ok = False
                try:
                    patch_budget = _choose_patch_budget(sub_crops)
                    inputs = processor(images=sub_crops, max_num_patches=patch_budget, return_tensors="pt")
                    inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                    gen_kwargs = {
                        "pixel_values": inputs["pixel_values"],
                        "tokenizer": tokenizer,
                        "max_new_tokens": max_new_tokens,
                        "repetition_penalty": 1.0,
                    }
                    if "pixel_attention_mask" in inputs:
                        gen_kwargs["pixel_attention_mask"] = inputs["pixel_attention_mask"]
                    if "spatial_shapes" in inputs:
                        gen_kwargs["spatial_shapes"] = inputs["spatial_shapes"]

                    raw_batch = model.generate(**gen_kwargs)
                    if isinstance(raw_batch, (list, tuple)) and len(raw_batch) == len(sub_crops):
                        for item in raw_batch:
                            if isinstance(item, str):
                                outputs_norm.append(self._normalize_hayai_ocr_text(item, lang=lang))
                            else:
                                dec = tokenizer.decode(item, skip_special_tokens=True)
                                outputs_norm.append(self._normalize_hayai_ocr_text(dec, lang=lang))
                        batch_ok = True
                    elif hasattr(raw_batch, "shape") and int(raw_batch.shape[0]) == len(sub_crops):
                        decoded_list = tokenizer.batch_decode(raw_batch, skip_special_tokens=True)
                        for dec in decoded_list:
                            outputs_norm.append(self._normalize_hayai_ocr_text(dec, lang=lang))
                        batch_ok = True
                except Exception:
                    batch_ok = False

                if not batch_ok:
                    for single_im in sub_crops:
                        try:
                            outputs_norm.append(_run_single_crop(single_im))
                        except Exception as single_exc:
                            logger.debug("HayaiOCR fallback single crop error: %s", single_exc)
                            outputs_norm.append("")

        return outputs_norm

    def _get_ja_nar_recognizer(self) -> Any | None:
        """
        Загружает неавторегрессионную японскую OCR-модель 2026 года `muscgab/manga-ocr-nar-small` (54M параметров,
        CER 1.73% против 4.84% у оригинального MangaOCR, точность на рукописном/стилизованном тексте 80.8% против 48.2%,
        скорость ~35 мс на всю страницу с сохранением пропорций кропа 324 патча 14x14).
        Скачивает только веса PyTorch (`model.safetensors`, 217 МБ) и встроенный пакет `manga_ocr_nar_small/`,
        пропуская папки `onnx/` и `coreml/`. Не требует `pip install`.
        """
        if not self.use_ja_nar or self._ja_nar_available is False:
            return None
        if self._ja_nar_recognizer is not None:
            return self._ja_nar_recognizer

        local_dir = self.models_dir / HF_JA_NAR_SUBDIR
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            weights_file = local_dir / "model.safetensors"
            pkg_init = local_dir / "manga_ocr_nar_small" / "__init__.py"
            if not weights_file.exists() or not pkg_init.exists():
                logger.info(
                    "Скачивание японской OCR-модели нового поколения %s (~217 МБ)...",
                    HF_JA_NAR_REPO,
                )
                snapshot_download(
                    repo_id=HF_JA_NAR_REPO,
                    local_dir=str(local_dir),
                    allow_patterns=list(HF_JA_NAR_ALLOW_PATTERNS),
                )

            local_dir_str = str(local_dir.resolve())
            if local_dir_str not in sys.path:
                sys.path.insert(0, local_dir_str)

            try:
                from manga_ocr_nar_small import MangaOcrNarSmall  # type: ignore[import-not-found]
            except ImportError:
                from mangaocr_nar_small import MangaOcrNarSmall  # type: ignore[import-not-found]

            nar_inst = None
            last_err: Exception | None = None
            backend_candidates = ("cuda", "auto", None) if self.device.startswith("cuda") else ("auto", None)
            for b_arg in backend_candidates:
                try:
                    if b_arg is None:
                        nar_inst = MangaOcrNarSmall()
                    else:
                        nar_inst = MangaOcrNarSmall(b_arg)
                    break
                except Exception as exc:
                    last_err = exc

            if nar_inst is None:
                raise RuntimeError(f"Не удалось создать экземпляр MangaOcrNarSmall ({last_err})")

            self._ja_nar_recognizer = nar_inst
            self._ja_nar_available = True
            self.last_detection_stats["ja_ocr_backend"] = self.get_ja_ocr_backend_name()
            logger.info(
                "Модель %s успешно подключена (гибридный режим с MangaOCR для длинных бабблов)!",
                HF_JA_NAR_REPO,
            )
            return self._ja_nar_recognizer
        except Exception as exc:
            logger.warning(
                "Модель %s недоступна (%s). Используем улучшенный MangaOCR FP16 + SmartPreprocess.",
                HF_JA_NAR_REPO,
                exc,
            )
            self._ja_nar_available = False
            self._ja_nar_recognizer = None
            return None

    def _load_detector(self) -> YOLO:
        local_weights = self.models_dir / HF_YOLO_FILENAME
        if not local_weights.exists():
            logger.info("Скачивание весов детектора %s/%s...", HF_YOLO_REPO, HF_YOLO_FILENAME)
            downloaded_path = hf_hub_download(
                repo_id=HF_YOLO_REPO,
                filename=HF_YOLO_FILENAME,
                local_dir=str(self.models_dir),
            )
            local_weights = Path(downloaded_path)

        detector = YOLO(str(local_weights))
        detector.to(self.device)
        return detector

    def _get_rtdetr_detector(self) -> tuple[Any, Any] | None:
        """
        Загружает 3-классовый детектор `ogkalu/comic-text-and-bubble-detector` (RT-DETR-v2 r50vd):
        - 0: `bubble` (внешний контур облачка речи)
        - 1: `text_bubble` (точный бокс текста внутри облачка)
        - 2: `text_free` (свободный текст и плашки вне облачков)
        Скачивает только 3 файла PyTorch (~172 МБ), пропуская тяжелые дубли `.onnx`.
        При любой ошибке сети или старой версии `transformers` мягко переключается на YOLOv8m + TightInkCrop.
        """
        if not self.use_rtdetr or self._rtdetr_available is False:
            return None
        if self._rtdetr_model is not None and self._rtdetr_processor is not None:
            return self._rtdetr_model, self._rtdetr_processor

        local_dir = self.models_dir / HF_RTDETR_SUBDIR
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            for fname in HF_RTDETR_PT_FILES:
                fpath = local_dir / fname
                if not fpath.exists():
                    logger.info(
                        "Скачивание компонента двухконтурного детектора %s/%s...",
                        HF_RTDETR_REPO,
                        fname,
                    )
                    hf_hub_download(
                        repo_id=HF_RTDETR_REPO,
                        filename=fname,
                        local_dir=str(local_dir),
                    )

            import transformers

            proc_cls = getattr(transformers, "AutoImageProcessor", None) or getattr(
                transformers, "RTDetrImageProcessor", None
            )
            if proc_cls is None:
                raise RuntimeError("В установленной версии transformers отсутствует AutoImageProcessor.")

            processor = proc_cls.from_pretrained(str(local_dir))

            model = None
            last_err: Exception | None = None
            for cls_name in (
                "AutoModelForObjectDetection",
                "RTDetrV2ForObjectDetection",
                "RTDetrForObjectDetection",
            ):
                model_cls = getattr(transformers, cls_name, None)
                if model_cls is None:
                    continue
                try:
                    model = model_cls.from_pretrained(str(local_dir))
                    break
                except Exception as exc:
                    last_err = exc

            if model is None:
                raise RuntimeError(f"Не удалось инициализировать RT-DETR-v2 ({last_err})")

            model.to(self.device)
            model.eval()

            self._rtdetr_model = model
            self._rtdetr_processor = processor
            self._rtdetr_available = True
            self.last_detection_stats["rtdetr_active"] = True
            self.last_detection_stats["detector_backend"] = self.get_detector_backend_name()
            logger.info(
                "Двухконтурный детектор RT-DETR-v2 (%s) успешно загружен на %s!",
                HF_RTDETR_REPO,
                self.device,
            )
            return self._rtdetr_model, self._rtdetr_processor
        except Exception as exc:
            logger.warning(
                "Двухконтурный детектор RT-DETR-v2 недоступен (%s). Работаем на связке YOLOv8m + TightInkCrop.",
                exc,
            )
            self._rtdetr_available = False
            self._rtdetr_model = None
            self._rtdetr_processor = None
            return None

    def _load_manga_ocr(self) -> MangaOcr:
        force_cpu = not self.device.startswith("cuda")
        return MangaOcr(force_cpu=force_cpu)

    def _check_rapidocr(self) -> bool:
        if self._rapidocr_available is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found] # noqa: F401

                self._rapidocr_available = True
            except ImportError:
                self._rapidocr_available = False
        return bool(self._rapidocr_available)

    def _get_rapidocr_reader(self) -> Any | None:
        if not self._check_rapidocr():
            return None
        if self._rapidocr_reader is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]

                self._rapidocr_reader = RapidOCR()
                self.last_detection_stats["rapidocr_active"] = True
                self.last_detection_stats["ko_zh_ocr_backend"] = self.get_multilang_ocr_backend_name()
            except Exception as exc:
                logger.debug("Не удалось инициализировать RapidOCR: %s", exc)
                self._rapidocr_available = False
                return None
        return self._rapidocr_reader

    def _check_easyocr(self) -> bool:
        if self._easyocr_available is None:
            try:
                import easyocr  # noqa: F401

                self._easyocr_available = True
            except ImportError:
                self._easyocr_available = False
        return bool(self._easyocr_available)

    def _check_multilang_ocr(self) -> bool:
        """
        Проверяет доступность хотя бы одного мультиязычного OCR-движка для корейского (`ko`) и китайского (`zh`):
        `HayaiOCR v2.5 Nova` (работает на базе штатного `transformers`), `RapidOCR` или `EasyOCR`.
        """
        if self.use_hayai_ocr and self._hayai_available is not False:
            return True
        if self._check_rapidocr() or self._check_easyocr():
            return True
        return False

    def _get_ko_reader(self) -> Any | None:
        if not self._check_easyocr():
            return None
        if self._ko_reader is None:
            import easyocr

            logger.info("Загрузка модели EasyOCR для корейского языка (ko)...")
            self._ko_reader = easyocr.Reader(
                ["ko", "en"],
                gpu=self.device.startswith("cuda"),
                model_storage_directory=str(self.models_dir / "easyocr"),
                verbose=False,
            )
        return self._ko_reader

    def _get_zh_reader(self) -> Any | None:
        if not self._check_easyocr():
            return None
        if self._zh_reader is None:
            import easyocr

            logger.info("Загрузка модели EasyOCR для китайского языка (ch_sim)...")
            self._zh_reader = easyocr.Reader(
                ["ch_sim", "en"],
                gpu=self.device.startswith("cuda"),
                model_storage_directory=str(self.models_dir / "easyocr"),
                verbose=False,
            )
        return self._zh_reader

    def _get_any_craft_reader(self) -> Any | None:
        """
        Возвращает любой уже загруженный экземпляр EasyOCR Reader для вызова `.detect()` (CRAFT),
        либо инициализирует только детектор CRAFT (`recognizer=False`), не нагружая VRAM лишним распознавателем.
        """
        if self._ko_reader is not None:
            return self._ko_reader
        if self._zh_reader is not None:
            return self._zh_reader
        if self._craft_reader is not None:
            return self._craft_reader
        if not self._check_easyocr():
            return None
        try:
            import easyocr

            logger.info("Инициализация легкого CRAFT-детектора EasyOCR (recognizer=False) для поиска текста вне бабблов...")
            self._craft_reader = easyocr.Reader(
                ["en"],
                gpu=self.device.startswith("cuda"),
                model_storage_directory=str(self.models_dir / "easyocr"),
                detector=True,
                recognizer=False,
                verbose=False,
            )
            return self._craft_reader
        except Exception as exc:
            logger.debug("Не удалось создать detector-only EasyOCR (%s), используем стандартный Reader.", exc)
            try:
                import easyocr

                self._craft_reader = easyocr.Reader(
                    ["en"],
                    gpu=self.device.startswith("cuda"),
                    model_storage_directory=str(self.models_dir / "easyocr"),
                    verbose=False,
                )
                return self._craft_reader
            except Exception:
                return None

    @staticmethod
    def _get_ocr_bbox(det: dict[str, Any]) -> list[int]:
        """
        Возвращает точные координаты области текста (`text_bbox`) для подачи в OCR,
        если они вычислены двухконтурным детектором или алгоритмом TightInkCrop,
        иначе возвращает общий `bbox` баббла.
        """
        tb = det.get("text_bbox")
        if isinstance(tb, (list, tuple)) and len(tb) >= 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in tb[:4]]
            if (x2 - x1) >= 8 and (y2 - y1) >= 8:
                return [x1, y1, x2, y2]
        b = det.get("bbox") or [0, 0, 0, 0]
        return [int(round(float(v))) for v in b[:4]]

    @staticmethod
    def _preprocess_crop_np_for_ocr(crop_rgb_np: np.ndarray) -> np.ndarray:
        """
        Умный препроцессинг кропа текста перед добавлением белых полей и подачей в OCR (Шаг 2):
        1. Автоматически распознает **белый/светлый текст на черном или темном фоне** (темные облачка мыслей,
           черные плашки рассказчика, инвертированные драматические бабблы), проверяя, что и медиана кропа `p50 < 115`,
           и периметр кропа темный (`border_median < 128`, чтобы не спутать с жирной черной буквой на белом фоне!),
           и мягко инвертирует его в четкий черный текст на белом фоне.
        2. Если скан бледный или внутри баббла серый скринтон (`p95 - p5 < 140` или фон серый `175 <= p80 < 235`),
           вытягивает контраст, делая фон чистым белым, а тонкие штрихи (`ー`, `…`, `っ`, дакутэн) — четкими.
        """
        if crop_rgb_np.size == 0:
            return crop_rgb_np
        h, w = crop_rgb_np.shape[:2]
        if h < 10 or w < 10:
            return crop_rgb_np

        gray = cv2.cvtColor(crop_rgb_np, cv2.COLOR_RGB2GRAY)
        p5, p20, p50, p80, p95 = [float(v) for v in np.percentile(gray, [5, 20, 50, 80, 95])]

        # Собираем пиксели внешней рамки кропа (толщиной 2px), чтобы точно знать цвет фона вокруг букв
        top_b = gray[:2, :].ravel()
        bot_b = gray[max(0, h - 2) :, :].ravel()
        left_b = gray[:, :2].ravel()
        right_b = gray[:, max(0, w - 2) :].ravel()
        border_pixels = np.concatenate([top_b, bot_b, left_b, right_b])
        border_median = float(np.median(border_pixels)) if border_pixels.size > 0 else p50

        # 1. Белый текст на темном/черном фоне -> инвертируем в черный текст на белом фоне
        if p50 < 115.0 and border_median < 128.0 and (p95 - p20) >= 35.0:
            inv_gray = 255 - gray
            inv_p5, inv_p85 = [float(v) for v in np.percentile(inv_gray, [5, 85])]
            denom = max(25.0, inv_p85 - inv_p5)
            stretched = np.clip((inv_gray.astype(np.float32) - inv_p5) * (255.0 / denom), 0.0, 255.0).astype(np.uint8)
            return cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)

        # 2. Бледный скан или серый скринтон на фоне баббла -> осветляем фон до белого и усиливаем штрихи
        if border_median >= 145.0 and ((p95 - p5) < 140.0 or (170.0 <= p80 < 236.0)):
            hi_ref = max(p80, border_median, p5 + 30.0)
            lo_ref = p5
            denom = max(25.0, hi_ref - lo_ref)
            stretched = np.clip((gray.astype(np.float32) - lo_ref) * (255.0 / denom), 0.0, 255.0).astype(np.uint8)
            return cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)

        return crop_rgb_np

    @staticmethod
    def _pad_to_balanced_aspect_for_vit(
        pil_crop: Image.Image,
        max_aspect_ratio: float = 2.0,
    ) -> Image.Image:
        """
        Защищает классический `MangaOCR` (ViT 224x224) от расплющивания букв на узких вертикальных столбиках
        (`h > w * 2.0`) и длинных горизонтальных строках (`w > h * 2.0`), добавляя белые поля по бокам
        или сверху/снизу до комфортного соотношения сторон `<= 2.0 : 1`.
        """
        w, h = pil_crop.size
        if w < 8 or h < 8:
            return pil_crop

        if h > w * max_aspect_ratio:
            target_w = int(round(h / max_aspect_ratio))
            pad_total = max(0, target_w - w)
            if pad_total >= 4:
                pad_l = pad_total // 2
                pad_r = pad_total - pad_l
                return ImageOps.expand(pil_crop, border=(pad_l, 0, pad_r, 0), fill=(255, 255, 255))
        elif w > h * max_aspect_ratio:
            target_h = int(round(w / max_aspect_ratio))
            pad_total = max(0, target_h - h)
            if pad_total >= 4:
                pad_t = pad_total // 2
                pad_b = pad_total - pad_t
                return ImageOps.expand(pil_crop, border=(0, pad_t, 0, pad_b), fill=(255, 255, 255))

        return pil_crop

    @classmethod
    def _crop_with_padding(
        cls,
        rgb_image: Image.Image,
        bbox: tuple[int, int, int, int] | list[int],
        pad: int = 8,
        smart_preprocess: bool = True,
    ) -> Image.Image:
        img_w, img_h = rgb_image.size
        x1, y1, x2, y2 = bbox
        cx1 = max(0, x1 - 4)
        cy1 = max(0, y1 - 4)
        cx2 = min(img_w, x2 + 4)
        cy2 = min(img_h, y2 + 4)
        crop = rgb_image.crop((cx1, cy1, cx2, cy2))
        if smart_preprocess:
            crop_np = np.array(crop)
            proc_np = cls._preprocess_crop_np_for_ocr(crop_np)
            if proc_np is not crop_np:
                crop = Image.fromarray(proc_np, mode="RGB")
        if pad > 0:
            crop = ImageOps.expand(crop, border=pad, fill=(255, 255, 255))
        return crop

    @classmethod
    def _split_horizontal_lines_for_ocr(
        cls,
        crop_rgb_np: np.ndarray,
        min_line_height: int = 10,
        max_lines: int = 8,
    ) -> list[tuple[int, int, np.ndarray]]:
        """
        Умная построчная сегментация многострочных горизонтальных бабблов для корейской манхвы (`ko`)
        и китайской маньхуа (`zh`) по профилю горизонтальной проекции чернил (`row_ink_profile`) (Шаг 3):
        - Находит горизонтальные строки текста (2..8 строк) и их межстрочные просветы.
        - Отсекает случайные вертикальные линии боковых рамок баббла, чтобы они не склеивали строки.
        - Объединяет микро-зазоры внутри одной строки (например, между верхней и нижней частью корейского слога `그`, `호`, `수`).
        - Разрезает слишком высокие слипшиеся полосы по локальному минимуму профиля чернил.
        - Возвращает список кортежей `(line_y1, line_y2, line_crop_rgb_np)` строго сверху вниз (`line_y1` по возрастанию).
        """
        if crop_rgb_np.size == 0:
            return []
        h, w = crop_rgb_np.shape[:2]
        if h < 26 or w < 18:
            return [(0, h, crop_rgb_np)]

        proc_np = cls._preprocess_crop_np_for_ocr(crop_rgb_np)
        gray = cv2.cvtColor(proc_np, cv2.COLOR_RGB2GRAY)

        pad_x = max(1, int(w * 0.04))
        inner_w = max(2, w - 2 * pad_x)
        inner_gray = gray[:, pad_x : pad_x + inner_w]

        p75 = float(np.percentile(inner_gray, 75))
        median_val = float(np.median(inner_gray))
        if p75 >= 200.0:
            ink_thresh = max(155, int(p75 - 28))
            binary = ((inner_gray < ink_thresh) * 255).astype(np.uint8)
        elif median_val >= 127.0:
            _, binary = cv2.threshold(inner_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(inner_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Удаляем длинные тонкие вертикальные линии по краям, которые могут проходить сквозь все строки сверху вниз
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lbl in range(1, num_labels):
            cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            if ch >= int(h * 0.72) and cw <= max(5, int(inner_w * 0.08)):
                binary[labels == lbl] = 0

        row_ink_profile = np.mean(binary > 0, axis=1).astype(np.float32)
        if row_ink_profile.size < 8:
            return [(0, h, proc_np)]

        smooth_profile = np.convolve(
            row_ink_profile,
            np.array([0.25, 0.50, 0.25], dtype=np.float32),
            mode="same",
        )
        peak_val = float(np.percentile(smooth_profile, 85))
        if peak_val < 0.012:
            return [(0, h, proc_np)]

        active_thresh = max(0.010, min(0.042, peak_val * 0.18))
        is_ink_row = smooth_profile >= active_thresh

        raw_bands: list[list[int]] = []
        in_band = False
        start_y = 0
        for y_idx, val in enumerate(is_ink_row.tolist()):
            if val and not in_band:
                in_band = True
                start_y = y_idx
            elif not val and in_band:
                in_band = False
                raw_bands.append([start_y, y_idx])
        if in_band:
            raw_bands.append([start_y, h])

        if not raw_bands:
            return [(0, h, proc_np)]

        # Склеиваем микро-разрывы внутри одной строки (например, между верхней и нижней половиной корейского слога)
        min_gap = max(2, int(round(h * 0.016)))
        merged_bands: list[list[int]] = [raw_bands[0]]
        for b_start, b_end in raw_bands[1:]:
            prev_s, prev_e = merged_bands[-1]
            gap = b_start - prev_e
            prev_h = prev_e - prev_s
            cur_h = b_end - b_start
            if gap <= min_gap or (gap <= 5 and (prev_h < max(7, min_line_height // 2) or cur_h < max(7, min_line_height // 2))):
                merged_bands[-1][1] = b_end
            else:
                merged_bands.append([b_start, b_end])

        # Если какая-то полоса получилась в 2+ раза выше типичной строки, проверяем локальный минимум внутри нее
        final_bands: list[ tuple[int, int] ] = []
        band_heights = [b[1] - b[0] for b in merged_bands if (b[1] - b[0]) >= min_line_height]
        ref_line_h = float(np.median(band_heights)) if band_heights else float(max(16, min_line_height * 2))

        for b_start, b_end in merged_bands:
            bh = b_end - b_start
            if bh < max(8, min_line_height - 2):
                continue
            if bh >= max(36, int(round(ref_line_h * 1.95))):
                margin_v = max(min_line_height, int(bh * 0.24))
                search_s = b_start + margin_v
                search_e = b_end - margin_v
                if search_e > search_s + 2:
                    sub_prof = smooth_profile[search_s:search_e]
                    min_rel = int(np.argmin(sub_prof))
                    valley_y = search_s + min_rel
                    valley_val = float(smooth_profile[valley_y])
                    top_peak = float(np.max(smooth_profile[b_start:valley_y])) if valley_y > b_start else peak_val
                    bot_peak = float(np.max(smooth_profile[valley_y:b_end])) if b_end > valley_y else peak_val
                    if valley_val <= min(top_peak, bot_peak) * 0.30:
                        final_bands.append((b_start, valley_y))
                        final_bands.append((valley_y, b_end))
                        continue
            final_bands.append((b_start, b_end))

        # Оставляем только содержательные строки
        valid_bands: list[tuple[int, int]] = []
        for by1, by2 in final_bands:
            if (by2 - by1) < max(8, min_line_height - 2):
                continue
            if int(np.count_nonzero(binary[by1:by2, :])) < 10:
                continue
            valid_bands.append((by1, by2))

        if len(valid_bands) <= 1 or len(valid_bands) > max_lines:
            return [(0, h, proc_np)]

        line_crops: list[tuple[int, int, np.ndarray]] = []
        for idx_b, (ly1, ly2) in enumerate(valid_bands):
            lh = ly2 - ly1
            pad_y = max(3, min(6, lh // 5))
            upper_bound = valid_bands[idx_b - 1][1] if idx_b > 0 else 0
            lower_bound = valid_bands[idx_b + 1][0] if (idx_b + 1) < len(valid_bands) else h
            cy1 = max(upper_bound, max(0, ly1 - pad_y))
            cy2 = min(lower_bound, min(h, ly2 + pad_y))
            if (cy2 - cy1) < 8:
                cy1, cy2 = ly1, ly2

            col_ink = np.where(np.any(binary[ly1:ly2, :] > 0, axis=0))[0]
            if col_ink.size > 0:
                cx1 = max(0, pad_x + int(col_ink[0]) - 6)
                cx2 = min(w, pad_x + int(col_ink[-1]) + 7)
            else:
                cx1, cx2 = 0, w

            if (cx2 - cx1) < 10:
                cx1, cx2 = 0, w

            sub_rgb = proc_np[cy1:cy2, cx1:cx2]
            bordered = cv2.copyMakeBorder(
                sub_rgb,
                4,
                4,
                6,
                6,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            line_crops.append((cy1, cy2, bordered))

        return line_crops if len(line_crops) >= 2 else [(0, h, proc_np)]

    @staticmethod
    def _sort_ocr_line_items_top_to_bottom(
        items: list[tuple[float, float, float, str, float]],
    ) -> list[tuple[str, float]]:
        """
        Группирует и сортирует фрагменты построчного OCR (`EasyOCR` / `RapidOCR`)
        строго сверху вниз по строкам, а внутри одной горизонтальной строки — слева направо.
        Каждый элемент `items`: `(y_min, y_max, x_center, text, conf)`.
        """
        if not items:
            return []
        if len(items) == 1:
            return [(items[0][3], items[0][4])]

        sorted_by_y = sorted(items, key=lambda it: (it[0] + it[1]) / 2.0)
        rows: list[list[tuple[float, float, float, str, float]]] = []

        for item in sorted_by_y:
            y1, y2, _, _, _ = item
            ih = max(1.0, y2 - y1)
            icy = (y1 + y2) / 2.0

            placed = False
            for row in rows:
                ry1 = min(r[0] for r in row)
                ry2 = max(r[1] for r in row)
                rh = max(1.0, ry2 - ry1)
                rcy = sum((r[0] + r[1]) / 2.0 for r in row) / len(row)

                overlap = min(y2, ry2) - max(y1, ry1)
                if overlap >= min(ih, rh) * 0.40 or abs(icy - rcy) <= max(ih, rh) * 0.42:
                    row.append(item)
                    placed = True
                    break
            if not placed:
                rows.append([item])

        rows.sort(key=lambda r: sum((x[0] + x[1]) / 2.0 for x in r) / len(r))
        ordered: list[tuple[str, float]] = []
        for row in rows:
            row.sort(key=lambda x: x[2])
            for _, _, _, txt, conf in row:
                ordered.append((txt, conf))
        return ordered

    @staticmethod
    def _extract_poly_bounds(poly: Any) -> tuple[float, float, float]:
        """
        Извлекает `(y_min, y_max, x_center)` из полигона координат `EasyOCR` или `RapidOCR`.
        """
        try:
            pts = np.asarray(poly, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] >= 2:
                xs = pts[:, 0]
                ys = pts[:, 1]
                return float(np.min(ys)), float(np.max(ys)), float(np.mean(xs))
        except Exception:
            pass
        return 0.0, 10.0, 0.0

    @staticmethod
    def _extract_bubble_interior_mask(roi_gray: np.ndarray, bg_level: float) -> np.ndarray:
        """
        Выделяет внутреннюю светлую область речевого облачка внутри темного контура баббла,
        отсекая как сам контур баббла, так и 4 внешних угла прямоугольного бокса за пределами овала.
        """
        rh, rw = roi_gray.shape[:2]
        wall_thresh = max(145, int(bg_level - 55))
        light_pixels = (roi_gray >= wall_thresh).astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(light_pixels, connectivity=4)
        if num_labels <= 1:
            return light_pixels

        cx_mid, cy_mid = rw / 2.0, rh / 2.0
        best_label = -1
        best_score = -1.0

        c_x1, c_x2 = int(rw * 0.20), max(int(rw * 0.20) + 1, int(rw * 0.80))
        c_y1, c_y2 = int(rh * 0.16), max(int(rh * 0.16) + 1, int(rh * 0.84))
        center_slice = labels[c_y1:c_y2, c_x1:c_x2]

        selected_labels: list[int] = []
        for lbl in range(1, num_labels):
            area = float(stats[lbl, cv2.CC_STAT_AREA])
            center_hits = float(np.count_nonzero(center_slice == lbl))
            if area < (rh * rw * 0.04) and center_hits < (center_slice.size * 0.03):
                continue
            dist = ((centroids[lbl][0] - cx_mid) ** 2 + ((centroids[lbl][1] - cy_mid) ** 2)) ** 0.5
            lx = stats[lbl, cv2.CC_STAT_LEFT]
            ly = stats[lbl, cv2.CC_STAT_TOP]
            lw = stats[lbl, cv2.CC_STAT_WIDTH]
            lh = stats[lbl, cv2.CC_STAT_HEIGHT]
            edge_touches = sum([lx <= 0, ly <= 0, (lx + lw) >= rw, (ly + lh) >= rh])
            corner_penalty = 0.0
            if edge_touches >= 2 and center_hits < area * 0.25:
                corner_penalty = area * 0.75

            score = center_hits * 3.5 + area - dist * 4.0 - corner_penalty
            if score > best_score:
                best_score = score
                best_label = lbl
            if center_hits >= max(8.0, center_slice.size * 0.06) and edge_touches <= 2:
                selected_labels.append(lbl)

        if best_label == -1:
            return light_pixels

        interior = np.zeros_like(light_pixels)
        interior[labels == best_label] = 255
        for lbl in selected_labels:
            interior[labels == lbl] = 255

        inv = cv2.bitwise_not(interior)
        n_inv, lbl_inv, st_inv, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        for idx in range(1, n_inv):
            lx = st_inv[idx, cv2.CC_STAT_LEFT]
            ly = st_inv[idx, cv2.CC_STAT_TOP]
            lw = st_inv[idx, cv2.CC_STAT_WIDTH]
            lh = st_inv[idx, cv2.CC_STAT_HEIGHT]
            touches_edge = lx <= 0 or ly <= 0 or (lx + lw) >= rw or (ly + lh) >= rh
            if not touches_edge:
                interior[lbl_inv == idx] = 255

        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.erode(interior, erode_k, iterations=1)

    @classmethod
    def _compute_tight_ink_bbox(
        cls,
        img_np: np.ndarray,
        bubble_bbox: list[int] | tuple[int, int, int, int],
        hint_text_bbox: list[int] | tuple[int, int, int, int] | None = None,
        is_free_text: bool = False,
    ) -> list[int]:
        """
        Вычисляет плотный внутренний прямоугольник реальных штрихов букв (`text_bbox`) внутри `bubble_bbox`,
        отсекая внешние черные контуры самого речевого облачка, рамки фреймов и пустые белые поля.
        Если передан `hint_text_bbox` (от класса `1: text_bubble` из RT-DETR-v2), объединяет его с реальными
        штрихами букв внутри баббла, чтобы ни один знак препинания или фуригана не обрезался.
        """
        img_h, img_w = img_np.shape[:2]
        bx1, by1, bx2, by2 = [int(round(float(c))) for c in bubble_bbox[:4]]
        bx1 = max(0, min(bx1, img_w - 2))
        by1 = max(0, min(by1, img_h - 2))
        bx2 = max(bx1 + 2, min(bx2, img_w))
        by2 = max(by1 + 2, min(by2, img_h))
        bw = bx2 - bx1
        bh = by2 - by1

        valid_hint: list[int] | None = None
        if isinstance(hint_text_bbox, (list, tuple)) and len(hint_text_bbox) >= 4:
            hx1, hy1, hx2, hy2 = [int(round(float(c))) for c in hint_text_bbox[:4]]
            hx1 = max(max(0, bx1 - 4), min(hx1, img_w - 2))
            hy1 = max(max(0, by1 - 4), min(hy1, img_h - 2))
            hx2 = max(hx1 + 2, min(hx2, min(img_w, bx2 + 4)))
            hy2 = max(hy1 + 2, min(hy2, min(img_h, by2 + 4)))
            if (hx2 - hx1) >= 10 and (hy2 - hy1) >= 10:
                valid_hint = [hx1, hy1, hx2, hy2]

        if bw < 22 or bh < 22:
            return valid_hint if valid_hint is not None else [bx1, by1, bx2, by2]

        if is_free_text:
            return valid_hint if valid_hint is not None else [bx1, by1, bx2, by2]

        # Отступаем на 3% от самого края бокса баббла, чтобы не захватывать внешнюю рамку фрейма
        pad_in_x = max(1, int(bw * 0.03))
        pad_in_y = max(1, int(bh * 0.03))
        ix1 = min(bx2 - 2, bx1 + pad_in_x)
        iy1 = min(by2 - 2, by1 + pad_in_y)
        ix2 = max(ix1 + 2, bx2 - pad_in_x)
        iy2 = max(iy1 + 2, by2 - pad_in_y)

        roi_rgb = img_np[iy1:iy2, ix1:ix2]
        rh, rw = roi_rgb.shape[:2]
        if rh < 12 or rw < 12:
            return valid_hint if valid_hint is not None else [bx1, by1, bx2, by2]

        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        p75 = float(np.percentile(roi_gray, 75))
        median_val = float(np.median(roi_gray))

        if p75 >= 210.0:
            interior_mask = cls._extract_bubble_interior_mask(roi_gray, p75)
            ink_thresh = max(165, int(p75 - 18))
            binary = ((roi_gray < ink_thresh) & (interior_mask > 0)).astype(np.uint8) * 255
        elif median_val >= 127.0:
            _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_cx, min_cy = rw, rh
        max_cx, max_cy = 0, 0
        kept_ink_px = 0

        for lbl in range(1, num_labels):
            cx = int(stats[lbl, cv2.CC_STAT_LEFT])
            cy = int(stats[lbl, cv2.CC_STAT_TOP])
            cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            carea = int(stats[lbl, cv2.CC_STAT_AREA])

            if carea < 3:
                continue

            touches_l = cx <= 1
            touches_t = cy <= 1
            touches_r = (cx + cw) >= (rw - 1)
            touches_b = (cy + ch) >= (rh - 1)
            edge_touches = sum([touches_l, touches_t, touches_r, touches_b])

            # Отсекаем контур самого баббла и длинные линии рамки, касающиеся границ бокса
            if edge_touches >= 2 and (cw > rw * 0.60 or ch > rh * 0.60):
                continue
            if edge_touches >= 1 and (cw > rw * 0.72 or ch > rh * 0.72):
                continue
            if edge_touches >= 1 and (cw <= 4 or ch <= 4) and (cw > rw * 0.35 or ch > rh * 0.35):
                continue

            # Если компонент лежит в самом углу кропа и касается границы — это дуга овального баббла
            if edge_touches >= 1:
                comp_cx = cx + cw / 2.0
                comp_cy = cy + ch / 2.0
                in_corner_x = comp_cx < rw * 0.16 or comp_cx > rw * 0.84
                in_corner_y = comp_cy < rh * 0.16 or comp_cy > rh * 0.84
                if in_corner_x and in_corner_y:
                    continue

            min_cx = min(min_cx, cx)
            min_cy = min(min_cy, cy)
            max_cx = max(max_cx, cx + cw)
            max_cy = max(max_cy, cy + ch)
            kept_ink_px += carea

        if kept_ink_px < 12 or max_cx <= min_cx or max_cy <= min_cy:
            return valid_hint if valid_hint is not None else [bx1, by1, bx2, by2]

        # Переводим координаты чернил обратно в систему страницы с небольшим запасом 4px вокруг штрихов
        ink_x1 = max(bx1, ix1 + min_cx - 4)
        ink_y1 = max(by1, iy1 + min_cy - 4)
        ink_x2 = min(bx2, ix1 + max_cx + 4)
        ink_y2 = min(by2, iy1 + max_cy + 4)

        if (ink_x2 - ink_x1) < 10 or (ink_y2 - ink_y1) < 10:
            return valid_hint if valid_hint is not None else [bx1, by1, bx2, by2]

        if valid_hint is not None:
            hx1, hy1, hx2, hy2 = valid_hint
            # Проверяем пересечение `hint_text_bbox` (от RT-DETR-v2) и найденных чернил
            inter_x1 = max(hx1, ink_x1)
            inter_y1 = max(hy1, ink_y1)
            inter_x2 = min(hx2, ink_x2)
            inter_y2 = min(hy2, ink_y2)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                # Объединяем `hint_text_bbox` и чернила (но не шире внутренней области баббла),
                # чтобы гарантированно не потерять крайние кавычки/точки и при этом не взять стенки баббла
                ux1 = max(bx1, min(hx1, ink_x1))
                uy1 = max(by1, min(hy1, ink_y1))
                ux2 = min(bx2, max(hx2, ink_x2))
                uy2 = min(by2, max(hy2, ink_y2))
                return [ux1, uy1, ux2, uy2]
            return valid_hint

        return [ink_x1, ink_y1, ink_x2, ink_y2]

    @staticmethod
    def _has_meaningful_ink(crop_rgb: np.ndarray, is_free_text: bool = False) -> tuple[bool, int]:
        h, w = crop_rgb.shape[:2]
        if h < 12 or w < 12:
            return False, 0

        margin_factor = 0.06 if is_free_text else 0.08
        pad_x = max(1, int(w * margin_factor))
        pad_y = max(1, int(h * margin_factor))
        roi = crop_rgb[pad_y : max(pad_y + 2, h - pad_y), pad_x : max(pad_x + 2, w - pad_x)]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        p5, p95 = np.percentile(gray, [5, 95])
        min_contrast = 42.0 if is_free_text else 18.0
        if (p95 - p5) < min_contrast:
            return False, 0

        if float(np.median(gray)) >= 127.0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ink_pixels = int(np.count_nonzero(binary))
        total_pixels = max(1, binary.size)
        ink_ratio = ink_pixels / total_pixels

        min_ratio = 0.018 if is_free_text else 0.0025
        max_ratio = 0.65 if is_free_text else 0.84
        min_ink_px = 14 if is_free_text else 9
        if ink_pixels < min_ink_px or ink_ratio < min_ratio or ink_ratio > max_ratio:
            return False, ink_pixels

        return True, ink_pixels

    @staticmethod
    def _is_valid_free_text_crop(crop_rgb: np.ndarray) -> bool:
        h, w = crop_rgb.shape[:2]
        if h < 14 or w < 14:
            return False
        aspect = w / max(1.0, float(h))
        if aspect > 12.0 or aspect < 0.06:
            return False

        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        p10, p50, p90 = np.percentile(gray, [10, 50, 90])
        if (p90 - p10) < 48.0:
            return False

        if p50 >= 120.0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        char_like_comps = 0
        long_line_comps = 0

        for idx in range(1, num_labels):
            cw = stats[idx, cv2.CC_STAT_WIDTH]
            ch = stats[idx, cv2.CC_STAT_HEIGHT]
            carea = stats[idx, cv2.CC_STAT_AREA]
            if carea < 5:
                continue
            if (cw >= w * 0.92 and ch <= 5) or (ch >= h * 0.92 and cw <= 5):
                long_line_comps += 1
                continue
            if 3 <= cw <= w * 0.92 and 3 <= ch <= h * 0.92:
                char_like_comps += 1

        if char_like_comps < 2 or long_line_comps > char_like_comps:
            return False
        return True

    @staticmethod
    def _has_real_text_glyph_structure(crop_rgb: np.ndarray, is_free_text: bool = False) -> bool:
        """
        Проверяет, что внутри кропа действительно присутствуют дискретные текстовые глифы (буквы/иероглифы),
        а не просто линии рисунка, волосы, складки одежды, рамки фреймов или однородный скринтон.
        Защищает декодеры MangaOCR / NAR от подачи бессловесных кусков арта, вызывающих галлюцинации.
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return False
        h, w = crop_rgb.shape[:2]
        if h < 12 or w < 12:
            return False

        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb
        p10, p50, p90 = [float(v) for v in np.percentile(gray, [10, 50, 90])]
        if (p90 - p10) < (42.0 if is_free_text else 28.0):
            return False

        if p50 >= 120.0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return False

        char_like_comps = 0
        long_wall_comps = 0
        tiny_specks = 0
        total_ink_area = max(1, int(np.count_nonzero(binary)))
        largest_comp_area = 0

        for idx in range(1, num_labels):
            cw = int(stats[idx, cv2.CC_STAT_WIDTH])
            ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
            carea = int(stats[idx, cv2.CC_STAT_AREA])
            if carea > largest_comp_area:
                largest_comp_area = carea
            if carea < 4:
                tiny_specks += 1
                continue
            # Длинные линии рамок/волос/контуров, пересекающие почти весь кроп
            if (cw >= w * 0.90 and ch <= max(5, int(h * 0.18))) or (ch >= h * 0.90 and cw <= max(5, int(w * 0.18))):
                long_wall_comps += 1
                continue
            if cw >= w * 0.94 and ch >= h * 0.94:
                long_wall_comps += 1
                continue
            if 2 <= cw <= w * 0.88 and 3 <= ch <= h * 0.88:
                char_like_comps += 1

        # Если один гигантский кусок заливки/тени занимает >82% всех чернил и нет отдельных букв — это рисунок, а не текст
        if largest_comp_area > total_ink_area * 0.82 and char_like_comps <= 1:
            return False
        # Если мелких точек скринтона в десятки раз больше, чем осмысленных штрихов — это скринтон
        if tiny_specks > max(28, char_like_comps * 9) and char_like_comps < 4:
            return False

        min_req_comps = 2 if is_free_text else 1
        if char_like_comps < min_req_comps:
            return False
        if is_free_text and long_wall_comps > char_like_comps:
            return False
        return True

    @staticmethod
    def _is_hallucinated_ocr_text(
        text: str,
        ink_pixels: int,
        is_free_text: bool = False,
        crop_shape: tuple[int, int] | None = None,
    ) -> bool:
        clean = text.strip()
        if not clean:
            return True
        if PUNCT_ONLY_REGEX.match(clean):
            return True
        if not MEANINGFUL_CHAR_REGEX.search(clean):
            return True
        if KNOWN_OCR_HALLUCINATIONS_REGEX.search(clean):
            logger.warning("Отсеяна известная галлюцинация OCR на фоновом/шумовом блоке: '%s'", clean)
            return True

        norm = STRIP_PUNCT_REGEX.sub("", clean)
        if not norm:
            return True
        if REPEATED_NOISE_REGEX.match(norm):
            return True

        # Проверка физической плотности чернил на символ: длинная фраза из 8+ символов не может возникнуть из пары точек
        char_len = len(norm)
        if char_len >= 8 and ink_pixels < char_len * 11:
            return True
        if is_free_text and ink_pixels < 42 and char_len > 3:
            return True
        if not is_free_text and ink_pixels < 22 and char_len > 4:
            return True

        if crop_shape is not None and len(crop_shape) >= 2:
            ch, cw = int(crop_shape[0]), int(crop_shape[1])
            crop_area = max(1, ch * cw)
            # В реальном баббле 1 символ занимает минимум ~45-50 кв. пикселей; если в крошечном боксе «прочитано» 12+ символов — это галлюцинация декодера
            if char_len >= 7 and crop_area < char_len * 45:
                return True

        if is_free_text:
            if char_len <= 1 and ink_pixels < 150:
                return True
            if ink_pixels < 95 and char_len >= 7:
                return True

        return False

    @classmethod
    def is_sigh_or_interjection(cls, text: str) -> bool:
        """
        Определяет, является ли распознанный текст коротким вздохом, стоном, охом или одиночным междометием
        (например: «あ», «あっ», «お», «んっ», «はぁ», «きゃ», «а», «о», «ах», «ох», «мм» и т.п.).
        """
        if not text:
            return True
        if "?" in text or "？" in text:
            return False
        core = STRIP_PUNCT_REGEX.sub("", text.strip()).lower()
        if not core:
            return True
        if core in DIALOGUE_VOWEL_WORDS_JA:
            return False
        if core in SIGH_EXACT_SET:
            return True

        # Схлопываем повторы одинаковых гласных/символов (например «ああああっ» -> «あっ», «おおお» -> «お»)
        collapsed = re.sub(r"(.)\1+", r"\1", core)
        if collapsed in SIGH_EXACT_SET and core not in DIALOGUE_VOWEL_WORDS_JA:
            return True

        # Если строка состоит только из вздохов/стонов (до 7 символов без кандзи)
        if len(core) <= 7 and not HANZI_REGEX.search(core):
            if SIGH_COMBO_REGEX.match(core) or SIGH_COMBO_REGEX.match(collapsed):
                return True

        # Одиночный символ хираганы/катаканы/хангыля/латиницы/кириллицы (не кандзи!) — почти всегда звук/вздох
        if len(core) == 1 and not HANZI_REGEX.search(core):
            return True

        return False

    @classmethod
    def is_sfx_or_onomatopoeia(cls, text: str, is_free_text: bool = False) -> bool:
        """
        Определяет, является ли блок звуковым спецэффектом (SFX / ономатопеей).
        ВАЖНО: Внутри кружков/бабблов (`is_free_text=False`) короткие слова катаканой (マジ, ダメ, ウソ, バカ, イヤ, ナニ, ココ)
        являются живыми репликами персонажей, поэтому внутри бабблов отсекаются ТОЛЬКО явные звуки из `SFX_EXACT_SET`!
        """
        if not text:
            return True
        core = STRIP_PUNCT_REGEX.sub("", text.strip())
        if not core:
            return True
        if core in SFX_EXACT_SET or core.lower() in SFX_EXACT_SET:
            return True

        collapsed = re.sub(r"(.)\1{2,}", r"\1", core)
        if collapsed in SFX_EXACT_SET:
            return True

        # Для текста ВНЕ бабблов (на фоне) отсекаем звуковую ономатопею катаканой и повторы слогов
        if is_free_text and not HANZI_REGEX.search(core):
            if len(core) <= 8 and REDUPLICATED_SFX_REGEX.match(core):
                return True
            if KATAKANA_ONLY_REGEX.match(core) and len(core) <= 6:
                return True

        return False

    @classmethod
    def classify_filter_reason(
        cls,
        text: str,
        is_free_text: bool = False,
        text_filter_mode: str = "dialogue_only",
        skip_sighs: bool = True,
        skip_sfx: bool = True,
        min_text_length: int = 2,
    ) -> tuple[str, str] | None:
        """
        Возвращает `(reason_code, reason_label_ru)`, если блок должен быть отсеян фильтром объема текста,
        либо `None`, если блок должен быть сохранен для перевода.
        """
        core = STRIP_PUNCT_REGEX.sub("", (text or "").strip())
        if not core:
            return ("empty_punct", "Только знаки препинания / пустая строка")

        mode = (text_filter_mode or "dialogue_only").lower().strip()
        eff_skip_sighs = bool(skip_sighs) or mode in ("dialogue_only", "bubbles_no_sighs", "long_only")
        eff_skip_sfx = bool(skip_sfx) or mode in ("dialogue_only", "bubbles_no_sighs", "bubbles_only", "long_only")
        eff_min_len = max(1, int(min_text_length or 1))
        if mode == "long_only":
            eff_min_len = max(eff_min_len, 4)
        elif mode == "all" and not skip_sighs and not skip_sfx and min_text_length <= 1:
            eff_min_len = 1

        if mode in ("dialogue_only", "bubbles_only", "long_only") and is_free_text:
            return ("free_text_disabled", f"Надпись вне баббла (отключено в профиле {mode})")

        # Одиночный кандзи внутри баббла (например 何 / 誰 / 嘘 / 嫌 / 悪) — это полноценное слово, никогда не отбрасываем при min_len <= 2!
        has_kanji = bool(HANZI_REGEX.search(core))
        effective_char_len = len(core) + (1 if (has_kanji and len(core) == 1 and eff_min_len <= 2) else 0)
        if effective_char_len < eff_min_len:
            return ("min_length", f"Короткий текст ({len(core)} симв. < порога {eff_min_len})")

        if eff_skip_sighs and cls.is_sigh_or_interjection(text):
            return ("sigh", "Короткий вздох/стон (skip_sighs=True)")

        if eff_skip_sfx and cls.is_sfx_or_onomatopoeia(text, is_free_text=is_free_text):
            return ("sfx", "Звуковой спецэффект / SFX (skip_sfx=True)")

        return None

    @classmethod
    def should_filter_out_block(
        cls,
        text: str,
        is_free_text: bool = False,
        text_filter_mode: str = "dialogue_only",
        skip_sighs: bool = True,
        skip_sfx: bool = True,
        min_text_length: int = 2,
    ) -> bool:
        """
        Проверяет, нужно ли пропустить (не замазывать и не переводить) данный текстовый блок
        согласно настройкам объема перевода.
        """
        return (
            cls.classify_filter_reason(
                text=text,
                is_free_text=is_free_text,
                text_filter_mode=text_filter_mode,
                skip_sighs=skip_sighs,
                skip_sfx=skip_sfx,
                min_text_length=min_text_length,
            )
            is not None
        )

    @staticmethod
    def _is_vertical_text_crop(crop_rgb: np.ndarray) -> bool:
        h, w = crop_rgb.shape[:2]
        if h < 20 or w < 20:
            return h > w

        pad_x = max(2, int(w * 0.12))
        pad_y = max(2, int(h * 0.12))
        roi = crop_rgb[pad_y : max(pad_y + 2, h - pad_y), pad_x : max(pad_x + 2, w - pad_x)]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        if float(np.median(gray)) >= 127.0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        norm = binary.astype(np.float32) / 255.0
        col_profile_var = float(np.var(np.mean(norm, axis=0)))
        row_profile_var = float(np.var(np.mean(norm, axis=1)))

        if h > w * 1.35 and col_profile_var >= row_profile_var * 0.9:
            return True
        return col_profile_var > row_profile_var * 1.25

    @classmethod
    def _read_easyocr_crop_with_stats(
        cls,
        reader: Any,
        crop_np: np.ndarray,
        target_regex: re.Pattern[str],
        min_conf: float = 0.15,
    ) -> tuple[str, int, float]:
        """
        Читает кроп через `EasyOCR` с двумя улучшениями Шага 3:
        1. Все найденные полигоны сортируются строго сверху вниз по строкам и слева направо внутри строки
           через `_sort_ocr_line_items_top_to_bottom` (исправляет перепутанные строки в вебтунах).
        2. Если в баббле 2+ горизонтальных строк по `_split_horizontal_lines_for_ocr`, а прямой вызов `EasyOCR`
           потерял строки или дал меньше целевых символов, выполняет построчное чтение сверху вниз!
        """
        def _run_on_array(arr: np.ndarray) -> tuple[str, int, float, int]:
            res = reader.readtext(arr, detail=1, paragraph=False)
            raw_items: list[tuple[float, float, float, str, float]] = []
            for item in res:
                if len(item) < 3:
                    continue
                poly, raw_txt, conf = item[0], str(item[1]).strip(), float(item[2])
                if not raw_txt or conf < min_conf:
                    continue
                y_min, y_max, x_center = cls._extract_poly_bounds(poly)
                raw_items.append((y_min, y_max, x_center, raw_txt, conf))

            ordered = cls._sort_ocr_line_items_top_to_bottom(raw_items)
            texts: list[str] = []
            total_chars = 0
            weighted_conf_sum = 0.0
            for raw_txt, conf in ordered:
                texts.append(raw_txt)
                matched_chars = len(target_regex.findall(raw_txt))
                if matched_chars > 0:
                    total_chars += matched_chars
                    weighted_conf_sum += conf * matched_chars

            joined_text = " ".join(texts).strip()
            avg_conf = (weighted_conf_sum / total_chars) if total_chars > 0 else 0.0
            return joined_text, total_chars, avg_conf, len(ordered)

        direct_text, direct_chars, direct_conf, direct_boxes = _run_on_array(crop_np)

        # Проверяем, не многострочный ли это баббл, в котором EasyOCR пропустил часть строк
        line_splits = cls._split_horizontal_lines_for_ocr(crop_np)
        if len(line_splits) >= 2 and (direct_boxes < len(line_splits) or direct_chars < len(line_splits) * 2 or direct_conf < 0.58):
            line_texts: list[str] = []
            line_total_chars = 0
            line_weighted_conf = 0.0
            for _, _, line_rgb in line_splits:
                l_txt, l_chars, l_conf, _ = _run_on_array(line_rgb)
                if l_txt:
                    line_texts.append(l_txt)
                    line_total_chars += l_chars
                    line_weighted_conf += l_conf * l_chars

            if line_texts:
                split_joined = " ".join(line_texts).strip()
                split_avg_conf = (line_weighted_conf / line_total_chars) if line_total_chars > 0 else 0.0
                if line_total_chars > direct_chars or (
                    line_total_chars == direct_chars and split_avg_conf > direct_conf + 0.05
                ):
                    return split_joined, line_total_chars, split_avg_conf

        return direct_text, direct_chars, direct_conf

    def _read_rapidocr_crop_with_stats(
        self,
        crop_np: np.ndarray,
        target_regex: re.Pattern[str],
        min_conf: float = 0.18,
    ) -> tuple[str, int, float]:
        """
        Опциональное чтение кропа через `RapidOCR` (`rapidocr_onnxruntime`), если он установлен в системе,
        с вертикально-горизонтальной сортировкой строк и построчным дочитыванием.
        """
        rapid = self._get_rapidocr_reader()
        if rapid is None:
            return "", 0, 0.0

        def _run_rapid(arr: np.ndarray) -> tuple[str, int, float, int]:
            try:
                res, _ = rapid(arr)
            except Exception:
                return "", 0, 0.0, 0
            if not res:
                return "", 0, 0.0, 0
            raw_items: list[tuple[float, float, float, str, float]] = []
            for item in res:
                if not item or len(item) < 3:
                    continue
                poly, raw_txt, conf = item[0], str(item[1]).strip(), float(item[2])
                if not raw_txt or conf < min_conf:
                    continue
                y_min, y_max, x_center = self._extract_poly_bounds(poly)
                raw_items.append((y_min, y_max, x_center, raw_txt, conf))

            ordered = self._sort_ocr_line_items_top_to_bottom(raw_items)
            texts: list[str] = []
            total_chars = 0
            weighted_conf_sum = 0.0
            for raw_txt, conf in ordered:
                texts.append(raw_txt)
                matched_chars = len(target_regex.findall(raw_txt))
                if matched_chars > 0:
                    total_chars += matched_chars
                    weighted_conf_sum += conf * matched_chars
            joined = " ".join(texts).strip()
            avg_c = (weighted_conf_sum / total_chars) if total_chars > 0 else 0.0
            return joined, total_chars, avg_c, len(ordered)

        direct_t, direct_c, direct_conf, direct_n = _run_rapid(crop_np)
        line_splits = self._split_horizontal_lines_for_ocr(crop_np)
        if len(line_splits) >= 2 and (direct_n < len(line_splits) or direct_c < len(line_splits) * 2):
            l_texts: list[str] = []
            l_chars = 0
            l_wconf = 0.0
            for _, _, line_rgb in line_splits:
                lt, lc, lcf, _ = _run_rapid(line_rgb)
                if lt:
                    l_texts.append(lt)
                    l_chars += lc
                    l_wconf += lcf * lc
            if l_texts and l_chars > direct_c:
                return " ".join(l_texts).strip(), l_chars, (l_wconf / l_chars if l_chars > 0 else 0.0)
        return direct_t, direct_c, direct_conf

    def _recognize_single_ko_zh_crop(
        self,
        padded_crop_pil: Image.Image,
        raw_crop_np: np.ndarray,
        lang: str,
        is_free_text: bool = False,
        precomputed_hayai_text: str | None = None,
    ) -> str:
        """
        Гибридное чтение одного корейского (`ko`) или китайского (`zh`) баббла (Шаг 3):
        1. Основной проход через `JustANormalTinkerer/hayai-ocr-v2.5-nova` (`SigLIP2 NaFlex`).
        2. Если баббл высокий многострочный (`>= 3` строк по `_split_horizontal_lines_for_ocr`), проверяет,
           не потерял ли одиночный проход часть строк, и при необходимости дочитывает построчно сверху вниз.
        3. Если в результате нет целевых символов (`HANGUL_REGEX` / `HANZI_REGEX`) или `HayaiOCR` недоступен,
           выполняет построчное чтение через `RapidOCR` / `EasyOCR`.
        """
        target_regex = HANGUL_REGEX if lang == "ko" else HANZI_REGEX
        padded_np = np.array(padded_crop_pil)
        min_conf = 0.28 if is_free_text else 0.14

        hayai_text = precomputed_hayai_text
        if hayai_text is None and self.use_hayai_ocr and self._hayai_available is not False:
            batch_res = self._recognize_crops_batch_hayai([padded_crop_pil], lang=lang)
            hayai_text = batch_res[0].strip() if batch_res else ""
        else:
            hayai_text = (hayai_text or "").strip()

        hayai_target_chars = len(target_regex.findall(hayai_text)) if hayai_text else 0

        # Если баббл высокий и содержит >= 3 горизонтальных строк, проверяем, не выгоднее ли построчное чтение HayaiOCR
        line_splits = self._split_horizontal_lines_for_ocr(raw_crop_np)
        if (
            len(line_splits) >= 3
            and self.use_hayai_ocr
            and self._hayai_available is not False
            and hayai_target_chars < len(line_splits) * 2
        ):
            line_pils = [Image.fromarray(l_rgb, mode="RGB") for _, _, l_rgb in line_splits]
            line_hayai_outs = self._recognize_crops_batch_hayai(line_pils, max_num_patches=256, lang=lang)
            valid_lines = [t.strip() for t in line_hayai_outs if t and t.strip()]
            if valid_lines:
                joined_lines = self._normalize_hayai_ocr_text(" ".join(valid_lines), lang=lang)
                if len(target_regex.findall(joined_lines)) > hayai_target_chars:
                    hayai_text = joined_lines
                    hayai_target_chars = len(target_regex.findall(hayai_text))

        if hayai_text and hayai_target_chars >= 1:
            return hayai_text

        # Резервный проход 1: RapidOCR (если установлен)
        if self._check_rapidocr():
            r_txt, r_chars, _ = self._read_rapidocr_crop_with_stats(padded_np, target_regex, min_conf=min_conf)
            if r_txt and r_chars >= 1:
                return self._normalize_hayai_ocr_text(r_txt, lang=lang)

        # Резервный проход 2: EasyOCR с построчной сегментацией и сортировкой сверху вниз
        reader = self._get_ko_reader() if lang == "ko" else self._get_zh_reader()
        if reader is not None:
            e_txt, e_chars, _ = self._read_easyocr_crop_with_stats(
                reader,
                padded_np,
                target_regex,
                min_conf=min_conf,
            )
            if e_txt and (e_chars >= 1 or not hayai_text):
                return self._normalize_hayai_ocr_text(e_txt, lang=lang)
            if not is_free_text:
                lines = reader.readtext(padded_np, detail=0, paragraph=True)
                raw_fallback = " ".join(str(x).strip() for x in lines if str(x).strip())
                if raw_fallback:
                    return self._normalize_hayai_ocr_text(raw_fallback, lang=lang)

        if hayai_text and not is_free_text:
            return hayai_text

        if not self._check_multilang_ocr():
            lang_label = "корейской манхвы" if lang == "ko" else "китайской маньхуа"
            raise RuntimeError(
                f"Для перевода {lang_label} требуется модель HayaiOCR v2.5 Nova или установленный EasyOCR (install_multilang_ocr.bat)."
            )
        return ""

    def detect_page_language(
        self,
        rgb_image: Image.Image,
        raw_detections: list[dict[str, Any]],
    ) -> str:
        """
        Улучшенное автоопределение языка страницы (`auto` -> `ja` / `ko` / `zh`) (Шаг 3):
        1. Применяет `_preprocess_crop_np_for_ocr` (через `_crop_with_padding`) и определяет ориентацию строк
           (`vertical_votes` vs `horizontal_votes`).
        2. Быстро проверяет японский через уже загруженный в VRAM `manga-ocr-nar-small` / `MangaOCR`:
           если текст вертикальный и содержит уверенную долю каны, мгновенно возвращает `ja` (~35 мс).
        3. Если текст горизонтальный (характерно для корейской манхвы `ko` и китайской маньхуа `zh`) или доля каны низкая:
           использует мультиязычную модель `HayaiOCR v2.5 Nova` (которая одновременно распознает `JA`, `KO`, `ZH`, `EN`
           без галлюцинаций чужого алфавита!) и/или `EasyOCR`/`RapidOCR`, сохраняя результаты в `_page_ocr_cache`.
        """
        if not raw_detections:
            return "ja"

        img_area = max(1, rgb_image.width * rgb_image.height)
        normal_boxes = [
            d
            for d in raw_detections
            if not d.get("is_free_text", False)
            and 0.001 <= ((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])) / img_area <= 0.25
        ]
        pool = normal_boxes if len(normal_boxes) >= 2 else raw_detections

        top_boxes = sorted(
            pool,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )[:6]

        crops_data: list[tuple[tuple[int, int, int, int], Image.Image, np.ndarray, bool]] = []
        vertical_votes = 0
        horizontal_votes = 0

        for det in top_boxes:
            x1, y1, x2, y2 = det["bbox"]
            bbox_key = (x1, y1, x2, y2)
            ocr_coords = self._get_ocr_bbox(det)
            raw_crop_np = np.array(rgb_image.crop(tuple(ocr_coords)))
            has_ink, _ = self._has_meaningful_ink(raw_crop_np, is_free_text=bool(det.get("is_free_text", False)))
            if not has_ink:
                continue
            padded_pil = self._crop_with_padding(rgb_image, ocr_coords, pad=8, smart_preprocess=True)
            padded_np = np.array(padded_pil)
            is_vert = self._is_vertical_text_crop( np.array(rgb_image.crop(tuple(ocr_coords))) )
            if is_vert:
                vertical_votes += 1
            else:
                horizontal_votes += 1
            crops_data.append((bbox_key, padded_pil, padded_np, is_vert))

        if not crops_data:
            return self._last_detected_lang

        kana_chars = 0
        ja_hanzi_chars = 0
        ja_evaluated = False

        def _evaluate_ja() -> tuple[int, int]:
            nonlocal kana_chars, ja_hanzi_chars, ja_evaluated
            if ja_evaluated:
                return kana_chars, ja_hanzi_chars
            ja_evaluated = True
            try:
                batch_pils = [item[1] for item in crops_data]
                ja_texts = self._recognize_crops_batch_ja(batch_pils)
                for (bbox_key, _, _, _), ja_txt in zip(crops_data, ja_texts):
                    self._page_ocr_cache[(*bbox_key, "ja")] = ja_txt
                    kana_chars += len(KANA_REGEX.findall(ja_txt))
                    ja_hanzi_chars += len(HANZI_REGEX.findall(ja_txt))
            except Exception:
                pass
            return kana_chars, ja_hanzi_chars

        # 1. Быстрая проверка японского (классическая японская манга и обложки с хираганой/катаканой — ~35 мс, без лишней подгрузки EasyOCR!)
        _evaluate_ja()
        ja_total = kana_chars + ja_hanzi_chars
        kana_ratio = kana_chars / max(1, ja_total)
        if vertical_votes > horizontal_votes and kana_chars >= 2 and kana_ratio >= 0.20:
            self._last_detected_lang = "ja"
            return "ja"
        if self._ja_nar_available and self._hayai_model is None and kana_chars >= 3 and kana_ratio >= 0.25:
            self._last_detected_lang = "ja"
            return "ja"
        if not self._check_multilang_ocr():
            self._last_detected_lang = "ja"
            return "ja"

        # 2. Если на странице преобладают горизонтальные бабблы или мало каны, проверяем через мультиязычный HayaiOCR v2.5 Nova
        # (если он уже загружен или если EasyOCR не установлен)
        if self.use_hayai_ocr and self._hayai_available is not False and (
            self._hayai_model is not None or not self._check_easyocr()
        ):
            try:
                batch_pils = [item[1] for item in crops_data]
                hayai_texts = self._recognize_crops_batch_hayai(batch_pils, max_num_patches=256, lang="auto")
                h_hangul = 0
                h_kana = 0
                h_hanzi = 0
                for (bbox_key, _, _, _), h_txt in zip(crops_data, hayai_texts):
                    if not h_txt:
                        continue
                    cnt_ko = len(HANGUL_REGEX.findall(h_txt))
                    cnt_ja = len(KANA_REGEX.findall(h_txt))
                    cnt_zh = len(HANZI_REGEX.findall(h_txt))
                    h_hangul += cnt_ko
                    h_kana += cnt_ja
                    h_hanzi += cnt_zh
                    if cnt_ko > 0:
                        self._page_ocr_cache[(*bbox_key, "ko")] = h_txt
                    if cnt_zh > 0 and cnt_ja == 0:
                        self._page_ocr_cache[(*bbox_key, "zh")] = h_txt

                if h_hangul >= 2 and h_hangul >= h_kana:
                    self._last_detected_lang = "ko"
                    return "ko"
                if h_kana >= 2 and (h_kana / max(1, h_kana + h_hanzi)) >= 0.18:
                    self._last_detected_lang = "ja"
                    return "ja"
                if h_hanzi >= 2 and h_kana == 0 and h_hangul == 0:
                    self._last_detected_lang = "zh"
                    return "zh"
            except Exception as exc:
                logger.debug("HayaiOCR language detection skipped: %s", exc)

        # 3. Если горизонтальная доля каны очень высокая и нет EasyOCR-признаков
        if kana_chars >= 4 and kana_ratio >= 0.34 and vertical_votes >= horizontal_votes:
            self._last_detected_lang = "ja"
            return "ja"

        # 4. Проверка через EasyOCR (ko / zh) при его наличии
        hangul_chars = 0
        ko_score = 0.0
        ko_weighted_sum = 0.0

        ko_reader = self._get_ko_reader()
        if ko_reader is not None:
            for bbox_key, _, padded_np, _ in crops_data:
                try:
                    ko_txt, h_cnt, h_conf = self._read_easyocr_crop_with_stats(
                        ko_reader, padded_np, HANGUL_REGEX
                    )
                    if ko_txt:
                        self._page_ocr_cache[(*bbox_key, "ko")] = ko_txt
                    if h_cnt > 0:
                        hangul_chars += h_cnt
                        ko_weighted_sum += h_conf * h_cnt
                        ko_score = max(ko_score, h_conf)
                except Exception:
                    pass
            if hangul_chars > 0:
                ko_score = max(ko_score, ko_weighted_sum / hangul_chars)

        if hangul_chars >= 2 and ko_score >= 0.52:
            self._last_detected_lang = "ko"
            return "ko"

        hanzi_chars = 0
        zh_score = 0.0
        zh_weighted_sum = 0.0

        zh_reader = self._get_zh_reader()
        if zh_reader is not None:
            for bbox_key, _, padded_np, _ in crops_data:
                try:
                    zh_txt, c_cnt, c_conf = self._read_easyocr_crop_with_stats(
                        zh_reader, padded_np, HANZI_REGEX
                    )
                    if zh_txt:
                        self._page_ocr_cache[(*bbox_key, "zh")] = zh_txt
                    if c_cnt > 0:
                        hanzi_chars += c_cnt
                        zh_weighted_sum += c_conf * c_cnt
                        zh_score = max(zh_score, c_conf)
                except Exception:
                    pass
            if hanzi_chars > 0:
                zh_avg = zh_weighted_sum / hanzi_chars
                zh_score = (zh_score * 0.5) + (zh_avg * 0.5)

        if hangul_chars >= 2 and ko_score >= 0.46 and ko_score >= (zh_score - 0.05):
            self._last_detected_lang = "ko"
            return "ko"

        if kana_chars >= 2 and kana_ratio >= 0.22:
            if vertical_votes >= horizontal_votes or zh_score < 0.74:
                self._last_detected_lang = "ja"
                return "ja"

        if (kana_chars + ja_hanzi_chars + hanzi_chars) <= 3 and vertical_votes >= horizontal_votes:
            return self._last_detected_lang

        if hanzi_chars >= 2 and zh_score >= 0.60:
            if kana_ratio < 0.20 or (horizontal_votes > vertical_votes and zh_score >= 0.65):
                self._last_detected_lang = "zh"
                return "zh"

        if vertical_votes > horizontal_votes and kana_chars >= 1:
            self._last_detected_lang = "ja"
            return "ja"

        if hanzi_chars >= 3 and zh_score >= 0.55 and kana_chars == 0:
            self._last_detected_lang = "zh"
            return "zh"

        return self._last_detected_lang

    @staticmethod
    def _merge_and_nms_detections(
        detections: list[dict[str, Any]],
        iou_threshold: float = 0.52,
        iomin_threshold: float = 0.62,
    ) -> list[dict[str, Any]]:
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(
            detections,
            key=lambda d: ((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
            * (0.5 + float(d["confidence"])),
            reverse=True,
        )

        merged: list[dict[str, Any]] = []
        for cand in sorted_dets:
            cx1, cy1, cx2, cy2 = cand["bbox"]
            c_area = max(1.0, float((cx2 - cx1) * (cy2 - cy1)))
            c_conf = float(cand["confidence"])
            c_free = bool(cand.get("is_free_text", False))

            absorbed = False
            for existing in merged:
                ex1, ey1, ex2, ey2 = existing["bbox"]
                e_area = max(1.0, float((ex2 - ex1) * (ey2 - ey1)))

                ix1 = max(cx1, ex1)
                iy1 = max(cy1, ey1)
                ix2 = min(cx2, ex2)
                iy2 = min(cy2, ey2)

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = float((ix2 - ix1) * (iy2 - iy1))
                    union_area = c_area + e_area - inter_area
                    iou = inter_area / max(1.0, union_area)
                    iomin = inter_area / max(1.0, min(c_area, e_area))

                    if iou >= iou_threshold or iomin >= iomin_threshold:
                        existing["bbox"] = [
                            min(cx1, ex1),
                            min(cy1, ey1),
                            max(cx2, ex2),
                            max(cy2, ey2),
                        ]
                        existing["confidence"] = round(max(float(existing["confidence"]), c_conf), 4)
                        existing["is_free_text"] = bool(existing.get("is_free_text", False) and c_free)
                        absorbed = True
                        break

            if not absorbed:
                merged_item: dict[str, Any] = {
                    "bbox": [cx1, cy1, cx2, cy2],
                    "confidence": round(c_conf, 4),
                    "is_free_text": c_free,
                }
                if cand.get("text_bbox"):
                    merged_item["text_bbox"] = list(cand["text_bbox"])
                merged.append(merged_item)

        return merged

    @staticmethod
    def _split_double_bubbles(
        img_np: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Разделяет только широкие сдвоенные по горизонтали бабблы (когда два соседних облачка
        слились в один широкий бокс `bw >= bh * 1.25` и между ними есть перемычка/граница).
        """
        refined: list[dict[str, Any]] = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            is_free = bool(det.get("is_free_text", False))

            if is_free or bw < 160 or bw < bh * 1.25:
                refined.append(det)
                continue

            pad_x = max(6, int(bw * 0.12))
            pad_y = max(6, int(bh * 0.10))
            roi = img_np[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
            rh, rw = roi.shape[:2]
            if rh < 36 or rw < 60:
                refined.append(det)
                continue

            full_crop_gray = cv2.cvtColor(img_np[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
            white_col_ratio = np.mean(full_crop_gray >= 230, axis=0)
            mid_s = int(bw * 0.35)
            mid_e = int(bw * 0.65)
            if mid_e <= mid_s + 8:
                refined.append(det)
                continue

            min_white_col = int(np.argmin(white_col_ratio[mid_s:mid_e])) + mid_s
            left_white = float(np.max(white_col_ratio[: max(1, min_white_col - 8)]))
            right_white = float(np.max(white_col_ratio[min(bw - 1, min_white_col + 8) :]))
            pinch_white = float(white_col_ratio[min_white_col])

            if pinch_white <= min(left_white, right_white) * 0.62 and (min_white_col >= 55) and (bw - min_white_col >= 55):
                split_x = x1 + min_white_col
                refined.append({"bbox": [x1, y1, split_x, y2], "confidence": det["confidence"], "is_free_text": is_free})
                refined.append({"bbox": [split_x, y1, x2, y2], "confidence": det["confidence"], "is_free_text": is_free})
            else:
                refined.append(det)

        return refined

    def _run_rtdetr_on_slice(
        self,
        slice_np: np.ndarray,
        y_start: int,
        img_w: int,
        img_h: int,
        active_conf: float,
        min_box_size: int = 14,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Прогоняет 3-классовый детектор `ogkalu/comic-text-and-bubble-detector` (RT-DETR-v2) на одном слайсе страницы.
        Возвращает `(bubbles_cls0, text_bubbles_cls1, free_texts_cls2)`.
        """
        pair = self._get_rtdetr_detector()
        if pair is None:
            return [], [], []

        model, processor = pair
        bubbles_cls0: list[dict[str, Any]] = []
        text_bubbles_cls1: list[dict[str, Any]] = []
        free_texts_cls2: list[dict[str, Any]] = []

        try:
            sh, sw = slice_np.shape[:2]
            pil_slice = Image.fromarray(slice_np)
            inputs = processor(images=pil_slice, return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            low_thresh = max(0.18, min(active_conf, 0.25))
            with torch.inference_mode():
                outputs = model(**inputs)
                target_sizes = torch.tensor([[sh, sw]], device=self.device)
                results = processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=low_thresh,
                )

            if not results or len(results) == 0:
                return [], [], []

            res0 = results[0]
            scores_list = res0["scores"].detach().cpu().numpy().tolist()
            labels_list = res0["labels"].detach().cpu().numpy().tolist()
            boxes_list = res0["boxes"].detach().cpu().numpy().tolist()

            for score, label_id, box in zip(scores_list, labels_list, boxes_list):
                conf_val = float(score)
                lbl = int(label_id)
                x1, y1, x2, y2 = [int(round(float(c))) for c in box[:4]]
                y1 += y_start
                y2 += y_start

                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(x1 + 1, min(x2, img_w))
                y2 = max(y1 + 1, min(y2, img_h))

                bw = x2 - x1
                bh = y2 - y1
                if bw < min_box_size or bh < min_box_size:
                    continue

                if lbl == 0:
                    # Class 0: bubble
                    if conf_val >= max(0.24, active_conf):
                        bubbles_cls0.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(conf_val, 4),
                                "is_free_text": False,
                                "_source": "rtdetr_bubble",
                            }
                        )
                elif lbl == 1:
                    # Class 1: text_bubble (точный контур текста внутри баббла)
                    if conf_val >= max(0.28, active_conf):
                        text_bubbles_cls1.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(conf_val, 4),
                                "is_free_text": False,
                                "_source": "rtdetr_text_bubble",
                            }
                        )
                elif lbl == 2:
                    # Class 2: text_free (текст вне бабблов на фоне)
                    if conf_val >= max(0.32, active_conf + 0.06):
                        free_texts_cls2.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(conf_val, 4),
                                "is_free_text": True,
                                "_source": "rtdetr_text_free",
                            }
                        )
        except Exception as exc:
            logger.debug("RT-DETR-v2 ошибка на слайсе (%s), продолжаем с YOLOv8m.", exc)

        return bubbles_cls0, text_bubbles_cls1, free_texts_cls2

    @classmethod
    def _try_split_bubble_by_text_boxes(
        cls,
        img_np: np.ndarray,
        bubble_det: dict[str, Any],
        matched_tbs: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """
        Если внутри одного родительского `bubble` найдено 2 отдельных неперекрывающихся `text_bubble` (от RT-DETR-v2),
        проверяет, разделены ли они перемычкой/границей баббла или широким пустым зазором.
        Если да — аккуратно разрезает родительский `bubble` на 2 независимых баббла с собственными `bbox` и `text_bbox`.
        """
        if len(matched_tbs) != 2:
            return None

        bx1, by1, bx2, by2 = bubble_det["bbox"]
        bw = bx2 - bx1
        bh = by2 - by1
        if bw < 72 and bh < 72:
            return None

        tb1 = matched_tbs[0]["bbox"]
        tb2 = matched_tbs[1]["bbox"]

        ix1 = max(tb1[0], tb2[0])
        iy1 = max(tb1[1], tb2[1])
        ix2 = min(tb1[2], tb2[2])
        iy2 = min(tb1[3], tb2[3])
        if ix2 > ix1 and iy2 > iy1:
            inter_a = float((ix2 - ix1) * (iy2 - iy1))
            min_a = float(min((tb1[2] - tb1[0]) * (tb1[3] - tb1[1]), (tb2[2] - tb2[0]) * (tb2[3] - tb2[1])))
            if inter_a / max(1.0, min_a) >= 0.22:
                return None

        x_gap = max(tb1[0], tb2[0]) - min(tb1[2], tb2[2])
        y_gap = max(tb1[1], tb2[1]) - min(tb1[3], tb2[3])

        crop_gray = cv2.cvtColor(img_np[by1:by2, bx1:bx2], cv2.COLOR_RGB2GRAY)

        # Вариант 1: Горизонтальный раздел (два облачка слева и справа)
        if x_gap >= 8 and x_gap >= y_gap:
            left_tb, right_tb = (tb1, tb2) if tb1[0] <= tb2[0] else (tb2, tb1)
            gap_s = max(bx1 + 12, left_tb[2]) - bx1
            gap_e = min(bx2 - 12, right_tb[0]) - bx1
            if gap_e > gap_s + 2:
                white_cols = np.mean(crop_gray >= 225, axis=0)
                min_rel = int(np.argmin(white_cols[gap_s:gap_e])) + gap_s
                pinch_white = float(white_cols[min_rel])
                base_white = float(np.percentile(white_cols, 75))
                if pinch_white <= base_white * 0.76 or x_gap >= max(24, int(bw * 0.16)):
                    split_x = bx1 + min_rel
                    if (split_x - bx1) >= 28 and (bx2 - split_x) >= 28:
                        b_left = [bx1, by1, split_x, by2]
                        b_right = [split_x, by1, bx2, by2]
                        return [
                            {
                                "bbox": b_left,
                                "text_bbox": cls._compute_tight_ink_bbox(img_np, b_left, left_tb, is_free_text=False),
                                "confidence": round(max(float(bubble_det["confidence"]), float(matched_tbs[0]["confidence"])), 4),
                                "is_free_text": False,
                            },
                            {
                                "bbox": b_right,
                                "text_bbox": cls._compute_tight_ink_bbox(img_np, b_right, right_tb, is_free_text=False),
                                "confidence": round(max(float(bubble_det["confidence"]), float(matched_tbs[1]["confidence"])), 4),
                                "is_free_text": False,
                            },
                        ]

        # Вариант 2: Вертикальный раздел (два облачка сверху и снизу — часто в манхве/вебтунах)
        if y_gap >= 10 and y_gap > x_gap:
            top_tb, bot_tb = (tb1, tb2) if tb1[1] <= tb2[1] else (tb2, tb1)
            gap_s = max(by1 + 12, top_tb[3]) - by1
            gap_e = min(by2 - 12, bot_tb[1]) - by1
            if gap_e > gap_s + 2:
                white_rows = np.mean(crop_gray >= 225, axis=1)
                min_rel = int(np.argmin(white_rows[gap_s:gap_e])) + gap_s
                pinch_white = float(white_rows[min_rel])
                base_white = float(np.percentile(white_rows, 75))
                if pinch_white <= base_white * 0.76 or y_gap >= max(26, int(bh * 0.18)):
                    split_y = by1 + min_rel
                    if (split_y - by1) >= 26 and (by2 - split_y) >= 26:
                        b_top = [bx1, by1, bx2, split_y]
                        b_bot = [bx1, split_y, bx2, by2]
                        return [
                            {
                                "bbox": b_top,
                                "text_bbox": cls._compute_tight_ink_bbox(img_np, b_top, top_tb, is_free_text=False),
                                "confidence": round(max(float(bubble_det["confidence"]), float(matched_tbs[0]["confidence"])), 4),
                                "is_free_text": False,
                            },
                            {
                                "bbox": b_bot,
                                "text_bbox": cls._compute_tight_ink_bbox(img_np, b_bot, bot_tb, is_free_text=False),
                                "confidence": round(max(float(bubble_det["confidence"]), float(matched_tbs[1]["confidence"])), 4),
                                "is_free_text": False,
                            },
                        ]

        return None

    def _associate_bubbles_and_text_boxes(
        self,
        img_np: np.ndarray,
        bubbles: list[dict[str, Any]],
        text_bubbles: list[dict[str, Any]],
        min_box_size: int = 16,
    ) -> list[dict[str, Any]]:
        """
        Сопоставляет контуры облачков `bubbles` (от YOLOv8m + RT-DETR-v2 class 0 + OpenCV)
        и точные текстовые боксы `text_bubbles` (от RT-DETR-v2 class 1).
        Для каждого итогового блока гарантирует наличие как `bbox` (контур баббла для верстки),
        так и точного `text_bbox` (плотный контур букв для безшумного OCR и защиты краев при очистке).
        """
        img_h, img_w = img_np.shape[:2]

        # Дедуплицируем список `text_bubbles`, но с более строгим порогом, чтобы не сливать соседние реплики
        dedup_tbs = self._merge_and_nms_detections(
            text_bubbles,
            iou_threshold=0.48,
            iomin_threshold=0.72,
        )

        bubble_to_tbs: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(len(bubbles))}
        orphan_tbs: list[dict[str, Any]] = []

        for tb in dedup_tbs:
            tx1, ty1, tx2, ty2 = tb["bbox"]
            t_area = max(1.0, float((tx2 - tx1) * (ty2 - ty1)))
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0

            best_b_idx = -1
            best_iomin = 0.0

            for b_idx, b_det in enumerate(bubbles):
                bx1, by1, bx2, by2 = b_det["bbox"]
                ix1 = max(tx1, bx1)
                iy1 = max(ty1, by1)
                ix2 = min(tx2, bx2)
                iy2 = min(ty2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter_area = float((ix2 - ix1) * (iy2 - iy1))
                iomin = inter_area / t_area
                center_inside = (bx1 <= tcx <= bx2) and (by1 <= tcy <= by2)

                if (iomin >= 0.42 or (center_inside and iomin >= 0.28)) and iomin > best_iomin:
                    best_iomin = iomin
                    best_b_idx = b_idx

            if best_b_idx >= 0:
                bubble_to_tbs[best_b_idx].append(tb)
            else:
                orphan_tbs.append(tb)

        final_detections: list[dict[str, Any]] = []
        split_count = 0
        rescued_count = 0

        for b_idx, b_det in enumerate(bubbles):
            matched = bubble_to_tbs.get(b_idx) or []
            bx1, by1, bx2, by2 = b_det["bbox"]
            is_free = bool(b_det.get("is_free_text", False))

            if len(matched) == 0:
                tight_tb = self._compute_tight_ink_bbox(img_np, [bx1, by1, bx2, by2], None, is_free_text=is_free)
                b_copy = dict(b_det)
                b_copy["text_bbox"] = tight_tb
                final_detections.append(b_copy)
            elif len(matched) == 1:
                tb_coords = matched[0]["bbox"]
                tight_tb = self._compute_tight_ink_bbox(
                    img_np,
                    [bx1, by1, bx2, by2],
                    tb_coords,
                    is_free_text=is_free,
                )
                b_copy = dict(b_det)
                b_copy["text_bbox"] = tight_tb
                b_copy["confidence"] = round(max(float(b_det["confidence"]), float(matched[0]["confidence"])), 4)
                final_detections.append(b_copy)
            else:
                split_parts = self._try_split_bubble_by_text_boxes(img_np, b_det, matched)
                if split_parts:
                    split_count += 1
                    final_detections.extend(split_parts)
                else:
                    ux1 = min(m["bbox"][0] for m in matched)
                    uy1 = min(m["bbox"][1] for m in matched)
                    ux2 = max(m["bbox"][2] for m in matched)
                    uy2 = max(m["bbox"][3] for m in matched)
                    tight_tb = self._compute_tight_ink_bbox(
                        img_np,
                        [bx1, by1, bx2, by2],
                        [ux1, uy1, ux2, uy2],
                        is_free_text=is_free,
                    )
                    b_copy = dict(b_det)
                    b_copy["text_bbox"] = tight_tb
                    b_copy["confidence"] = round(
                        max([float(b_det["confidence"])] + [float(m["confidence"]) for m in matched]),
                        4,
                    )
                    final_detections.append(b_copy)

        # Обрабатываем «сиротские» `text_bubble` (облачка без рамок, мысленные облачка или плашки, пропущенные YOLOv8m)
        for otb in orphan_tbs:
            otb_conf = float(otb.get("confidence", 0.0))
            # Если ни YOLOv8m, ни RT-DETR class 0 не увидели вокруг баббла, требуем высокую уверенность текстового бокса
            if otb_conf < 0.42:
                continue

            tx1, ty1, tx2, ty2 = otb["bbox"]
            tw = tx2 - tx1
            th = ty2 - ty1
            if tw < min_box_size or th < min_box_size:
                continue

            crop_rgb = img_np[ty1:ty2, tx1:tx2]
            has_ink, _ = self._has_meaningful_ink(crop_rgb, is_free_text=False)
            if not has_ink or not self._has_real_text_glyph_structure(crop_rgb, is_free_text=True):
                continue

            # Расширяем `text_bubble` на 8% (4..14 px), чтобы получить комфортный `bbox` для верстки перевода
            pad_x = max(4, min(14, int(round(tw * 0.08))))
            pad_y = max(4, min(14, int(round(th * 0.08))))
            ex1 = max(0, tx1 - pad_x)
            ey1 = max(0, ty1 - pad_y)
            ex2 = min(img_w, tx2 + pad_x)
            ey2 = min(img_h, ty2 + pad_y)

            # Проверим, не пересекается ли уже сильно с каким-то из итоговых бабблов
            overlaps_existing = False
            o_area = max(1.0, float((ex2 - ex1) * (ey2 - ey1)))
            for ex_det in final_detections:
                fx1, fy1, fx2, fy2 = ex_det["bbox"]
                ix1 = max(ex1, fx1)
                iy1 = max(ey1, fy1)
                ix2 = min(ex2, fx2)
                iy2 = min(ey2, fy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    if inter / o_area >= 0.35:
                        overlaps_existing = True
                        break
            if overlaps_existing:
                continue

            # Строгая проверка: действительно ли сиротский text_bubble находится внутри однородного светлого облачка,
            # а не на фоне рисунка/штриховки/одежды, где просто есть 25% белых пикселей!
            exp_crop = img_np[ey1:ey2, ex1:ex2]
            exp_gray = cv2.cvtColor(exp_crop, cv2.COLOR_RGB2GRAY) if exp_crop.ndim == 3 else exp_crop
            white_pixel_ratio = float(np.mean(exp_gray >= 228)) if exp_gray.size > 0 else 0.0
            non_ink_mask = exp_gray >= 175
            bg_std = float(np.std(exp_gray[non_ink_mask])) if int(np.count_nonzero(non_ink_mask)) >= 16 else 99.0
            top_b = exp_gray[:3, :].ravel()
            bot_b = exp_gray[max(0, exp_gray.shape[0] - 3) :, :].ravel()
            left_b = exp_gray[:, :3].ravel()
            right_b = exp_gray[:, max(0, exp_gray.shape[1] - 3) :].ravel()
            perimeter_px = np.concatenate([top_b, bot_b, left_b, right_b])
            perimeter_median = float(np.median(perimeter_px)) if perimeter_px.size > 0 else 0.0
            perimeter_white_ratio = float(np.mean(perimeter_px >= 224)) if perimeter_px.size > 0 else 0.0

            is_clean_bubble_bg = (
                white_pixel_ratio >= 0.58
                and bg_std <= 11.5
                and perimeter_median >= 230.0
                and perimeter_white_ratio >= 0.62
            )
            is_free_orphan = not is_clean_bubble_bg

            rescued_count += 1
            final_detections.append(
                {
                    "bbox": [ex1, ey1, ex2, ey2],
                    "text_bbox": self._compute_tight_ink_bbox(img_np, [ex1, ey1, ex2, ey2], [tx1, ty1, tx2, ty2], is_free_text=is_free_orphan),
                    "confidence": round(otb_conf, 4),
                    "is_free_text": is_free_orphan,
                    "_source": "rtdetr_rescued_text_bubble",
                }
            )

        self.last_detection_stats["split_multi_bubbles"] = split_count
        self.last_detection_stats["rescued_orphan_text_bubbles"] = rescued_count
        return final_detections

    def _detect_bubbles(
        self,
        img_np: np.ndarray,
        min_box_size: int = 16,
        conf_override: float | None = None,
    ) -> list[dict[str, Any]]:
        img_h, img_w = img_np.shape[:2]
        yolo_detections: list[dict[str, Any]] = []
        rtdetr_bubbles: list[dict[str, Any]] = []
        rtdetr_text_bubbles: list[dict[str, Any]] = []
        rtdetr_free_texts: list[dict[str, Any]] = []

        active_conf = conf_override if conf_override is not None else self.conf_threshold

        slices: list[tuple[int, int]] = []
        if img_h > 1400 and (img_h / max(1, img_w)) > 2.5:
            win_h = min(img_h, max(1280, int(round(img_w * 1.6))))
            overlap_h = int(round(win_h * 0.28))
            step_y = max(400, win_h - overlap_h)

            y_cur = 0
            while y_cur < img_h:
                y_end = min(img_h, y_cur + win_h)
                if (y_end - y_cur) < min(win_h, 600) and y_cur > 0:
                    y_cur = max(0, img_h - win_h)
                    y_end = img_h
                    slices.append((y_cur, y_end))
                    break
                slices.append((y_cur, y_end))
                if y_end >= img_h:
                    break
                y_cur += step_y
        else:
            slices.append((0, img_h))

        with torch.inference_mode():
            for y_start, y_end in slices:
                slice_np = img_np[y_start:y_end, :]

                # 1. Прогон YOLOv8m (speech bubble detector)
                results = self.detector.predict(
                    source=slice_np,
                    conf=active_conf,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    xyxy_list = boxes.xyxy.cpu().numpy().tolist()
                    conf_list = boxes.conf.cpu().numpy().tolist()

                    for bbox, conf in zip(xyxy_list, conf_list):
                        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
                        y1 += y_start
                        y2 += y_start

                        x1 = max(0, min(x1, img_w - 1))
                        y1 = max(0, min(y1, img_h - 1))
                        x2 = max(x1 + 1, min(x2, img_w))
                        y2 = max(y1 + 1, min(y2, img_h))

                        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                            continue

                        yolo_detections.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(float(conf), 4),
                                "is_free_text": False,
                                "_source": "yolov8",
                            }
                        )

                # 2. Прогон двухконтурного 3-классового детектора RT-DETR-v2 (bubble / text_bubble / text_free)
                if self.use_rtdetr:
                    rt_b, rt_tb, rt_free = self._run_rtdetr_on_slice(
                        slice_np,
                        y_start=y_start,
                        img_w=img_w,
                        img_h=img_h,
                        active_conf=active_conf,
                        min_box_size=max(12, min_box_size - 2),
                    )
                    if rt_b:
                        rtdetr_bubbles.extend(rt_b)
                    if rt_tb:
                        rtdetr_text_bubbles.extend(rt_tb)
                    if rt_free:
                        rtdetr_free_texts.extend(rt_free)

        self._last_rtdetr_free_boxes = rtdetr_free_texts
        self.last_detection_stats.update(
            {
                "detector_backend": self.get_detector_backend_name(),
                "rtdetr_active": bool(self._rtdetr_available and self._rtdetr_model is not None),
                "yolo_bubbles": len(yolo_detections),
                "rtdetr_bubbles": len(rtdetr_bubbles),
                "rtdetr_text_bubbles": len(rtdetr_text_bubbles),
                "rtdetr_free_texts": len(rtdetr_free_texts),
            }
        )

        raw_detections = yolo_detections + rtdetr_bubbles

        # 3. Резервный контурный поиск OpenCV запускаем ТОЛЬКО если нейросетевые детекторы (YOLOv8m + RT-DETR-v2)
        # не нашли бабблов на странице (или если RT-DETR-v2 отключен), чтобы не превращать светлые детали одежды/фона в ложные бабблы
        if len(raw_detections) == 0 or not (self._rtdetr_available and self._rtdetr_model is not None):
            cv_bubbles = self._detect_missed_white_bubbles(img_np, raw_detections, min_box_size=min_box_size)
            if cv_bubbles:
                raw_detections.extend(cv_bubbles)

        if raw_detections:
            raw_detections = self._merge_and_nms_detections(
                raw_detections,
                iou_threshold=0.52,
                iomin_threshold=0.62,
            )
            raw_detections = self._split_double_bubbles(img_np, raw_detections)

        # 4. Сопоставляем найденные контуры бабблов с точными внутренними боксами текста (`text_bubble` + TightInkCrop)
        raw_detections = self._associate_bubbles_and_text_boxes(
            img_np,
            raw_detections,
            rtdetr_text_bubbles,
            min_box_size=min_box_size,
        )

        return raw_detections

    def _detect_missed_white_bubbles(
        self,
        img_np: np.ndarray,
        existing_bubbles: list[dict[str, Any]],
        min_box_size: int = 24,
    ) -> list[dict[str, Any]]:
        """
        Находит замкнутые белые/светлые кружки и овалы speech bubbles с черным текстом внутри,
        если нейросеть YOLOv8 пропустила кружок (например, идеально круглые, облачные или мелкие бабблы).
        """
        img_h, img_w = img_np.shape[:2]
        img_area = max(1, img_w * img_h)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Белые внутренности бабблов
        _, white_mask = cv2.threshold(gray, 232, 255, cv2.THRESH_BINARY)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_white = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        contours, _ = cv2.findContours(closed_white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        added: list[dict[str, Any]] = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < max(26, min_box_size) or h < max(26, min_box_size):
                continue
            box_area = float(w * h)
            if box_area < 900 or box_area > img_area * 0.18:
                continue
            aspect = w / max(1.0, float(h))
            if aspect < 0.22 or aspect > 3.8:
                continue

            # Не должен касаться самых краев страницы (фон/поля страницы)
            if x <= 2 or y <= 2 or (x + w) >= (img_w - 2) or (y + h) >= (img_h - 2):
                continue

            already_covered = False
            for ex in existing_bubbles:
                ex1, ey1, ex2, ey2 = ex["bbox"]
                ix1, iy1 = max(x, ex1), max(y, ey1)
                ix2, iy2 = min(x + w, ex2), min(y + h, ey2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    if inter / min(box_area, float((ex2 - ex1) * (ey2 - ey1))) >= 0.30:
                        already_covered = True
                        break
            if already_covered:
                continue

            cnt_area = float(cv2.contourArea(cnt))
            if cnt_area < box_area * 0.48:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = max(1.0, float(cv2.contourArea(hull)))
            solidity = cnt_area / hull_area
            if solidity < 0.76:
                continue

            pad_in_x = max(3, int(w * 0.16))
            pad_in_y = max(3, int(h * 0.16))
            inner_gray = gray[y + pad_in_y : y + h - pad_in_y, x + pad_in_x : x + w - pad_in_x]
            if inner_gray.size < 64:
                continue

            if float(np.percentile(inner_gray, 75)) < 235.0:
                continue
            dark_ink_px = int(np.count_nonzero(inner_gray < 150))
            inner_ratio = dark_ink_px / float(inner_gray.size)
            if dark_ink_px < 18 or inner_ratio < 0.015 or inner_ratio > 0.45:
                continue

            _, inner_bin = cv2.threshold(inner_gray, 165, 255, cv2.THRESH_BINARY_INV)
            n_lbl, _, st_lbl, _ = cv2.connectedComponentsWithStats(inner_bin, connectivity=8)
            char_comps = 0
            ih, iw = inner_gray.shape[:2]
            for idx_c in range(1, n_lbl):
                cw = st_lbl[idx_c, cv2.CC_STAT_WIDTH]
                ch = st_lbl[idx_c, cv2.CC_STAT_HEIGHT]
                ca = st_lbl[idx_c, cv2.CC_STAT_AREA]
                if ca >= 5 and 2 <= cw <= iw * 0.85 and 2 <= ch <= ih * 0.85:
                    char_comps += 1
            if char_comps < 2:
                continue

            added.append(
                {
                    "bbox": [x, y, x + w, y + h],
                    "confidence": 0.35,
                    "is_free_text": False,
                }
            )

        if len(added) > 10:
            added.sort(key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]), reverse=True)
            added = added[:10]
        return added

    @staticmethod
    def _group_free_text_boxes(
        raw_boxes: list[list[int]],
        img_w: int,
        img_h: int,
    ) -> list[list[int]]:
        if not raw_boxes:
            return []

        boxes = [list(b) for b in raw_boxes]
        changed = True
        while changed:
            changed = False
            new_boxes: list[list[int]] = []
            used = [False] * len(boxes)

            for i in range(len(boxes)):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2 = boxes[i]
                aw = max(1, ax2 - ax1)
                ah = max(1, ay2 - ay1)

                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    bx1, by1, bx2, by2 = boxes[j]
                    bw = max(1, bx2 - bx1)
                    bh = max(1, by2 - by1)

                    ix1 = max(ax1, bx1)
                    iy1 = max(ay1, by1)
                    ix2 = min(ax2, bx2)
                    iy2 = min(ay2, by2)
                    x_overlap = max(0, ix2 - ix1)
                    y_overlap = max(0, iy2 - iy1)

                    x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
                    y_gap = max(0, max(ay1, by1) - min(ay2, by2))

                    min_w = min(aw, bw)
                    min_h = min(ah, bh)

                    intersects = x_overlap > 0 and y_overlap > 0
                    vert_aligned = x_overlap >= min_w * 0.45 and y_gap <= max(16, int(min_h * 0.65))
                    adj_columns = (
                        y_overlap >= min_h * 0.55
                        and x_gap <= max(14, int(min_w * 0.55))
                        and (max(ax2, bx2) - min(ax1, bx1)) <= int(img_w * 0.36)
                    )

                    if intersects or vert_aligned or adj_columns:
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        aw = max(1, ax2 - ax1)
                        ah = max(1, ay2 - ay1)
                        used[j] = True
                        changed = True

                new_boxes.append([ax1, ay1, ax2, ay2])
            boxes = new_boxes

        return [
            [
                max(0, b[0] - 2),
                max(0, b[1] - 2),
                min(img_w, b[2] + 2),
                min(img_h, b[3] + 2),
            ]
            for b in boxes
        ]

    @staticmethod
    def _detect_free_text_cv2_fallback(
        masked_img_np: np.ndarray,
        existing_bubbles: list[dict[str, Any]],
    ) -> list[list[int]]:
        """
        Резервный морфологический детектор высококонтрастных надписей (вызывается только если EasyOCR не установлен).
        """
        img_h, img_w = masked_img_np.shape[:2]
        gray = cv2.cvtColor(masked_img_np, cv2.COLOR_RGB2GRAY)

        k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_small)
        _, thresh = cv2.threshold(blackhat, 72, 255, cv2.THRESH_BINARY)

        exclude = np.zeros((img_h, img_w), dtype=np.uint8)
        for b in existing_bubbles:
            x1, y1, x2, y2 = b["bbox"]
            cv2.rectangle(
                exclude,
                (max(0, x1 - 12), max(0, y1 - 12)),
                (min(img_w, x2 + 12), min(img_h, y2 + 12)),
                255,
                -1,
            )
        thresh[exclude > 0] = 0

        k_group = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_group)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        candidates: list[list[int]] = []
        img_area = max(1, img_w * img_h)

        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = w * h

            if w < 22 or h < 22:
                continue
            if area < 600 or area > img_area * 0.10:
                continue
            if w > img_w * 0.60 and h < 28:
                continue
            if h > img_h * 0.60 and w < 28:
                continue

            roi_gray = gray[y : y + h, x : x + w]
            if float(np.percentile(roi_gray, 80)) < 200.0:
                continue

            candidates.append([x, y, x + w, y + h])

        return candidates

    def _detect_free_text_regions(
        self,
        img_np: np.ndarray,
        bubble_detections: list[dict[str, Any]],
        min_box_size: int = 18,
    ) -> list[dict[str, Any]]:
        img_h, img_w = img_np.shape[:2]
        masked_np = img_np.copy()

        for det in bubble_detections:
            bx1, by1, bx2, by2 = det["bbox"]
            mx1 = max(0, bx1 - 4)
            my1 = max(0, by1 - 4)
            mx2 = min(img_w, bx2 + 4)
            my2 = min(img_h, by2 + 4)
            masked_np[my1:my2, mx1:mx2] = 255

        raw_free_boxes: list[list[int]] = []

        # 1. Сначала берем высокоточные боксы класса `2: text_free` от RT-DETR-v2 (если он активен),
        # так как RT-DETR-v2 обучен на вертикальном и горизонтальном тексте манги/манхвы вне бабблов!
        if self._last_rtdetr_free_boxes:
            dedup_rt_free = self._merge_and_nms_detections(
                self._last_rtdetr_free_boxes,
                iou_threshold=0.45,
                iomin_threshold=0.65,
            )
            for rf in dedup_rt_free:
                rx1, ry1, rx2, ry2 = rf["bbox"]
                raw_free_boxes.append([rx1, ry1, rx2, ry2])

        # 2. Дополняем через CRAFT (или OpenCV fallback), если RT-DETR-v2 не загружен или нашел мало блоков
        craft_reader = self._get_any_craft_reader() if len(raw_free_boxes) < 6 else None
        if craft_reader is not None:
            slices: list[tuple[int, int]] = []
            if img_h > 1600 and (img_h / max(1, img_w)) > 2.2:
                win_h = min(img_h, max(1280, int(round(img_w * 1.5))))
                step_y = max(500, int(win_h * 0.78))
                y_cur = 0
                while y_cur < img_h:
                    y_end = min(img_h, y_cur + win_h)
                    slices.append((y_cur, y_end))
                    if y_end >= img_h:
                        break
                    y_cur += step_y
            else:
                slices.append((0, img_h))

            for y_start, y_end in slices:
                slice_np = masked_np[y_start:y_end, :]
                try:
                    horiz_list, free_list = craft_reader.detect(
                        slice_np,
                        min_size=16,
                        text_threshold=0.62,
                        low_text=0.38,
                        link_threshold=0.35,
                        width_ths=0.5,
                    )
                    if horiz_list and len(horiz_list) > 0:
                        for box in horiz_list[0]:
                            if len(box) >= 4:
                                x1, x2, y1, y2 = int(box[0]), int(box[1]), int(box[2]) + y_start, int(box[3]) + y_start
                                raw_free_boxes.append([x1, y1, x2, y2])
                    if free_list and len(free_list) > 0:
                        for poly in free_list[0]:
                            xs = [int(pt[0]) for pt in poly]
                            ys = [int(pt[1]) + y_start for pt in poly]
                            raw_free_boxes.append([min(xs), min(ys), max(xs), max(ys)])
                except Exception as exc:
                    logger.debug("CRAFT detect вне бабблов пропущен: %s", exc)
        elif not raw_free_boxes:
            try:
                cv2_boxes = self._detect_free_text_cv2_fallback(masked_np, bubble_detections)
                raw_free_boxes.extend(cv2_boxes)
            except Exception as exc:
                logger.debug("OpenCV free-text fallback ошибка: %s", exc)

        if not raw_free_boxes:
            return []

        grouped_boxes = self._group_free_text_boxes(raw_free_boxes, img_w, img_h)
        free_detections: list[dict[str, Any]] = []

        for gx1, gy1, gx2, gy2 in grouped_boxes:
            gx1 = max(0, min(gx1, img_w - 1))
            gy1 = max(0, min(gy1, img_h - 1))
            gx2 = max(gx1 + 1, min(gx2, img_w))
            gy2 = max(gy1 + 1, min(gy2, img_h))

            gw = gx2 - gx1
            gh = gy2 - gy1
            if gw < min_box_size or gh < min_box_size:
                continue

            g_area = float(gw * gh)
            overlaps_bubble = False
            for b in bubble_detections:
                bx1, by1, bx2, by2 = b["bbox"]
                ix1 = max(gx1, bx1)
                iy1 = max(gy1, by1)
                ix2 = min(gx2, bx2)
                iy2 = min(gy2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    if inter / max(1.0, g_area) >= 0.35:
                        overlaps_bubble = True
                        break
            if overlaps_bubble:
                continue

            crop_np = img_np[gy1:gy2, gx1:gx2]
            has_ink, ink_px = self._has_meaningful_ink(crop_np, is_free_text=True)
            if not has_ink or not self._is_valid_free_text_crop(crop_np):
                continue

            free_detections.append(
                {
                    "bbox": [gx1, gy1, gx2, gy2],
                    "text_bbox": [gx1, gy1, gx2, gy2],
                    "confidence": 0.45,
                    "is_free_text": True,
                    "_ink_score": ink_px,
                }
            )

        if len(free_detections) > 12:
            free_detections.sort(key=lambda d: int(d.get("_ink_score", 0)), reverse=True)
            free_detections = free_detections[:12]

        for d in free_detections:
            d.pop("_ink_score", None)

        return free_detections

    @staticmethod
    def _sort_boxes_reading_order(
        detections: list[dict[str, Any]],
        rtl: bool = True,
        row_overlap_ratio: float = 0.35,
    ) -> list[dict[str, Any]]:
        if not detections:
            return []

        sorted_by_y = sorted(detections, key=lambda d: (d["bbox"][1] + d["bbox"][3]) / 2.0)
        rows: list[list[dict[str, Any]]] = []

        for det in sorted_by_y:
            x1, y1, x2, y2 = det["bbox"]
            box_h = max(1, y2 - y1)
            box_cy = (y1 + y2) / 2.0

            placed = False
            for row in rows:
                row_y1 = min(item["bbox"][1] for item in row)
                row_y2 = max(item["bbox"][3] for item in row)
                row_h = max(1, row_y2 - row_y1)
                row_cy = sum((item["bbox"][1] + item["bbox"][3]) / 2.0 for item in row) / len(row)

                overlap = min(y2, row_y2) - max(y1, row_y1)
                if overlap > min(box_h, row_h) * row_overlap_ratio and abs(box_cy - row_cy) < max(box_h, row_h) * 0.75:
                    row.append(det)
                    placed = True
                    break

            if not placed:
                rows.append([det])

        ordered: list[dict[str, Any]] = []
        for row in rows:
            row.sort(
                key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2.0,
                reverse=rtl,
            )
            ordered.extend(row)

        return ordered

    def _recognize_crop(
        self,
        padded_crop_pil: Image.Image,
        raw_crop_np: np.ndarray,
        lang: str,
        is_free_text: bool = False,
    ) -> str:
        has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free_text)
        if not has_ink:
            return ""

        if lang in ("ko", "zh"):
            ko_zh_text = self._recognize_single_ko_zh_crop(
                padded_crop_pil,
                raw_crop_np,
                lang=lang,
                is_free_text=is_free_text,
            )
            if self._is_hallucinated_ocr_text(ko_zh_text, ink_pixels, is_free_text=is_free_text):
                return ""
            return ko_zh_text

        ja_list = self._recognize_crops_batch_ja([padded_crop_pil])
        ja_text = ja_list[0].strip() if ja_list else ""

        if self._is_hallucinated_ocr_text(ja_text, ink_pixels, is_free_text=is_free_text):
            return ""
        return ja_text

    def _run_classic_manga_ocr_batch(
        self,
        crops_pil: list[Image.Image],
        batch_size: int = 16,
    ) -> list[str]:
        """
        Пакетное распознавание через классический авторегрессионный `MangaOcr` (до 300 символов)
        с автоматическим выравниванием пропорций `_pad_to_balanced_aspect_for_vit` для узких/широких кропов.
        """
        if not crops_pil:
            return []
        balanced_crops = [self._pad_to_balanced_aspect_for_vit(c, max_aspect_ratio=2.0) for c in crops_pil]
        if len(balanced_crops) == 1:
            with torch.inference_mode():
                return [self.ja_recognizer(balanced_crops[0]).strip()]

        model = getattr(self.ja_recognizer, "model", None)
        tokenizer = getattr(self.ja_recognizer, "tokenizer", None)
        processor = getattr(self.ja_recognizer, "processor", None) or getattr(
            self.ja_recognizer, "feature_extractor", None
        )
        post_proc = getattr(self.ja_recognizer, "_post_process", None) or getattr(
            self.ja_recognizer, "post_process", None
        )

        if model is None or tokenizer is None or processor is None:
            with torch.inference_mode():
                return [self.ja_recognizer(c).strip() for c in balanced_crops]

        try:
            results_out: list[str] = []
            model_device = next(model.parameters()).device
            use_cuda_fp16 = model_device.type == "cuda"

            with torch.inference_mode():
                for start_i in range(0, len(balanced_crops), batch_size):
                    sub_crops = balanced_crops[start_i : start_i + batch_size]
                    gray_rgb_list = [c.convert("L").convert("RGB") for c in sub_crops]
                    pixel_values = processor(gray_rgb_list, return_tensors="pt").pixel_values.to(
                        model_device, non_blocking=True
                    )
                    if use_cuda_fp16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            token_ids = model.generate(pixel_values, max_new_tokens=300)
                    else:
                        token_ids = model.generate(pixel_values, max_new_tokens=300)

                    decoded_list = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
                    for raw_str in decoded_list:
                        if callable(post_proc):
                            cleaned = str(post_proc(raw_str)).strip()
                        else:
                            cleaned = re.sub(r"\s+", "", str(raw_str)).strip()
                        results_out.append(cleaned)

            if len(results_out) == len(balanced_crops):
                return results_out
        except Exception as batch_exc:
            logger.debug("Пакетный режим MangaOCR недоступен (%s), используем поштучный режим.", batch_exc)

        with torch.inference_mode():
            return [self.ja_recognizer(c).strip() for c in balanced_crops]

    def _recognize_crops_batch_ja(
        self,
        crops_pil: list[Image.Image],
        batch_size: int = 16,
    ) -> list[str]:
        """
        Гибридное пакетное GPU-распознавание японских бабблов (Шаг 2):
        1. Основной проход выполняется через `muscgab/manga-ocr-nar-small` (CER 1.73%, EM 91.09%, сохранение пропорций
           кропа 324 патча 14x14, чтение всей пачки за ~35 мс).
        2. Если какой-то баббл содержит длинный монолог (`len(text) >= 56`, близко к потолку 64 символов у NAR-декодера)
           или вернул пустую строку, только этот баббл автоматически дочитывается через авторегрессионный `MangaOcr`
           (`max_new_tokens=300`) с защитой пропорций `_pad_to_balanced_aspect_for_vit`!
        """
        if not crops_pil:
            return []

        nar_model = self._get_ja_nar_recognizer()
        if nar_model is not None:
            try:
                rgb_crops = [c.convert("RGB") for c in crops_pil]
                with torch.inference_mode():
                    if len(rgb_crops) == 1:
                        raw_res = nar_model(rgb_crops[0])
                        nar_texts = [str(raw_res).strip()] if isinstance(raw_res, str) else [str(x).strip() for x in raw_res]
                    else:
                        raw_res = nar_model(rgb_crops)
                        if isinstance(raw_res, str):
                            nar_texts = [raw_res.strip()]
                        else:
                            nar_texts = [str(x).strip() for x in raw_res]

                if len(nar_texts) == len(crops_pil):
                    # ВАЖНО: Дочитываем через классический авторегрессионный MangaOCR ТОЛЬКО длинные монологи (len >= 54),
                    # которые уперлись в потолок 64 токенов NAR-декодера!
                    # Если manga-ocr-nar-small вернул пустую строку ("") на кропе — значит внутри нет японских букв,
                    # и принудительный вызов классического MangaOCR на пустом/фоновом кропе приводил к галлюцинации
                    # заученных фраз («タクシー運転手さん一番うまい店に連れてって» -> «Taxi driver, take me to the best restaurant»).
                    long_indices = [
                        idx for idx, txt in enumerate(nar_texts) if len(txt) >= 54
                    ]
                    if long_indices:
                        sub_crops = [crops_pil[idx] for idx in long_indices]
                        fallback_texts = self._run_classic_manga_ocr_batch(sub_crops, batch_size=batch_size)
                        for idx, fb_txt in zip(long_indices, fallback_texts):
                            cur_nar = nar_texts[idx]
                            if len(cur_nar) >= 54 and len(fb_txt) > len(cur_nar) and not KNOWN_OCR_HALLUCINATIONS_REGEX.search(fb_txt):
                                nar_texts[idx] = fb_txt
                    return nar_texts
            except Exception as nar_exc:
                logger.debug("Ошибка пакетного вызова manga-ocr-nar-small (%s), переключаемся на MangaOCR.", nar_exc)

        return self._run_classic_manga_ocr_batch(crops_pil, batch_size=batch_size)

    def _recognize_crops_batch_ko_zh(
        self,
        crops_pil: list[Image.Image],
        raw_crops_np: list[np.ndarray],
        is_free_list: list[bool],
        lang: str,
    ) -> list[str]:
        """
        Пакетное GPU-распознавание бабблов для корейской манхвы (`ko`) и китайской маньхуа (`zh`) (Шаг 3):
        1. Выполняет быстрый батч-прогон всех кропов страницы через `HayaiOCR v2.5 Nova` (`SigLIP2 NaFlex`).
        2. Для каждого кропа проверяет полноту чтения (включая построчную сегментацию `_split_horizontal_lines_for_ocr`
           для высоких многострочных бабблов и фоллбэк на `RapidOCR` / `EasyOCR`).
        """
        if not crops_pil:
            return []

        hayai_batch_texts: list[str | None] = [None] * len(crops_pil)
        if self.use_hayai_ocr and self._hayai_available is not False:
            hayai_outs = self._recognize_crops_batch_hayai(crops_pil, lang=lang)
            if len(hayai_outs) == len(crops_pil):
                hayai_batch_texts = [t for t in hayai_outs]

        final_texts: list[str] = []
        for padded_pil, raw_np, is_f, pre_hayai in zip(crops_pil, raw_crops_np, is_free_list, hayai_batch_texts):
            txt = self._recognize_single_ko_zh_crop(
                padded_pil,
                raw_np,
                lang=lang,
                is_free_text=is_f,
                precomputed_hayai_text=pre_hayai,
            )
            final_texts.append(txt)
        return final_texts

    def process_image(
        self,
        image: Image.Image,
        lang: str = "auto",
        min_box_size: int = 16,
        detect_free_text: bool = False,
        text_filter_mode: str = "dialogue_only",
        skip_sighs: bool = True,
        skip_sfx: bool = True,
        min_text_length: int = 2,
    ) -> tuple[list[dict[str, Any]], str]:
        req_lang = (lang or "auto").lower().strip()
        if req_lang not in ("auto", "ja", "ko", "zh"):
            req_lang = "auto"

        mode = (text_filter_mode or "dialogue_only").lower().strip()
        # В режимах "dialogue_only", "bubbles_only" и "long_only" свободный текст на фоне принудительно отключен,
        # чтобы не замазывать рисунки и спецэффекты вне бабблов.
        effective_free_text = bool(detect_free_text) and mode not in ("dialogue_only", "bubbles_only", "long_only")
        if mode in ("all", "bubbles_no_sighs"):
            effective_free_text = True if mode == "all" else bool(detect_free_text)

        self._page_ocr_cache.clear()

        rgb_image = image.convert("RGB")
        img_np = np.array(rgb_image)

        if mode == "long_only":
            yolo_conf = max(0.25, self.conf_threshold)
            effective_min_box = max(min_box_size, 24)
        else:
            yolo_conf = max(0.22, self.conf_threshold)
            effective_min_box = max(18, min_box_size)

        raw_detections = self._detect_bubbles(
            img_np,
            min_box_size=effective_min_box,
            conf_override=yolo_conf,
        )

        if effective_free_text:
            free_detections = self._detect_free_text_regions(
                img_np,
                raw_detections,
                min_box_size=max(18, effective_min_box),
            )
            if free_detections:
                logger.info(
                    "Найдено %d дополнительных текстовых областей вне speech bubbles.",
                    len(free_detections),
                )
                raw_detections.extend(free_detections)

        if not raw_detections:
            det_lang_empty = self._last_detected_lang if req_lang == "auto" else req_lang
            self.last_page_diagnostics = {
                "raw_candidates_count": 0,
                "yolo_bubbles": int(self.last_detection_stats.get("yolo_bubbles", 0)),
                "rtdetr_bubbles": int(self.last_detection_stats.get("rtdetr_bubbles", 0)),
                "rtdetr_text_bubbles": int(self.last_detection_stats.get("rtdetr_text_bubbles", 0)),
                "rtdetr_free_texts": int(self.last_detection_stats.get("rtdetr_free_texts", 0)),
                "split_multi_bubbles": int(self.last_detection_stats.get("split_multi_bubbles", 0)),
                "rescued_orphan_text_bubbles": int(self.last_detection_stats.get("rescued_orphan_text_bubbles", 0)),
                "no_ink_count": 0,
                "hallucinated_count": 0,
                "empty_ocr_count": 0,
                "filtered_count": 0,
                "filtered_items": [],
                "restored_by_empty_guard_count": 0,
                "kept_blocks_count": 0,
                "free_text_kept_count": 0,
                "detected_lang": det_lang_empty,
                "zero_reason_code": "no_bubbles_detected",
                "zero_reason_ru": "На странице не обнаружено речевых облачков или текстовых рамок (обложка, чистый арт или страница без диалогов)",
            }
            return [], det_lang_empty

        resolved_lang = (
            self.detect_page_language(rgb_image, raw_detections)
            if req_lang == "auto"
            else req_lang
        )
        if req_lang in ("ja", "ko", "zh"):
            self._last_detected_lang = req_lang

        sorted_detections = self._sort_boxes_reading_order(
            raw_detections,
            rtl=(resolved_lang == "ja"),
        )

        output_blocks: list[dict[str, Any]] = []
        filtered_bubble_SamePage_fallback: list[dict[str, Any]] = []
        filtered_items_diag: list[dict[str, Any]] = []
        block_id = 1
        skipped_by_filter = 0
        no_ink_count = 0
        hallucinated_count = 0
        empty_ocr_count = 0
        restored_by_empty_guard_count = 0

        # Пакетное GPU-распознавание всех бабблов страницы (для ja через manga-ocr-nar-small/MangaOCR,
        # для ko/zh через HayaiOCR v2.5 Nova + SmartLineSplit + RapidOCR/EasyOCR) строго по `_get_ocr_bbox(det)`!
        batch_candidates_crops: list[Image.Image] = []
        batch_candidates_raw_np: list[np.ndarray] = []
        batch_candidates_meta: list[tuple[tuple[int, int, int, int, str], int, bool]] = []
        rejected_hallucination_keys: set[tuple[int, int, int, int, str]] = set()
        for det in sorted_detections:
            x1, y1, x2, y2 = det["bbox"]
            is_free = bool(det.get("is_free_text", False))
            cache_key = (x1, y1, x2, y2, resolved_lang)
            if cache_key in self._page_ocr_cache:
                continue
            ocr_coords = self._get_ocr_bbox(det)
            ox1, oy1, ox2, oy2 = ocr_coords
            raw_crop_np = img_np[oy1:oy2, ox1:ox2]
            has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free)
            has_glyphs = has_ink and self._has_real_text_glyph_structure(raw_crop_np, is_free_text=is_free)
            if (not has_ink or not has_glyphs) and ocr_coords != [x1, y1, x2, y2]:
                # Фоллбэк на полный bbox баббла, если text_bbox оказался слишком узким
                raw_crop_np = img_np[y1:y2, x1:x2]
                has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free)
                has_glyphs = has_ink and self._has_real_text_glyph_structure(raw_crop_np, is_free_text=is_free)
                ocr_coords = [x1, y1, x2, y2]
            if not has_ink or not has_glyphs:
                self._page_ocr_cache[cache_key] = ""
                rejected_hallucination_keys.add(cache_key)
                continue
            padded_crop = self._crop_with_padding(rgb_image, ocr_coords, pad=8)
            batch_candidates_crops.append(padded_crop)
            batch_candidates_raw_np.append(raw_crop_np)
            batch_candidates_meta.append((cache_key, ink_pixels, is_free))

        if batch_candidates_crops:
            if resolved_lang == "ja":
                batch_texts = self._recognize_crops_batch_ja(batch_candidates_crops)
            else:
                batch_texts = self._recognize_crops_batch_ko_zh(
                    batch_candidates_crops,
                    batch_candidates_raw_np,
                    [m[2] for m in batch_candidates_meta],
                    lang=resolved_lang,
                )
            for (c_key, ink_px, is_f), raw_np_item, b_txt in zip(batch_candidates_meta, batch_candidates_raw_np, batch_texts):
                if self._is_hallucinated_ocr_text(b_txt, ink_px, is_free_text=is_f, crop_shape=raw_np_item.shape[:2]):
                    if b_txt and b_txt.strip():
                        hallucinated_count += 1
                    self._page_ocr_cache[c_key] = ""
                    rejected_hallucination_keys.add(c_key)
                else:
                    self._page_ocr_cache[c_key] = b_txt

        for det in sorted_detections:
            x1, y1, x2, y2 = det["bbox"]
            ocr_coords = self._get_ocr_bbox(det)
            ox1, oy1, ox2, oy2 = ocr_coords
            is_free = bool(det.get("is_free_text", False))
            cache_key = (x1, y1, x2, y2, resolved_lang)

            raw_crop_np = img_np[oy1:oy2, ox1:ox2]
            has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free)
            if not has_ink and ocr_coords != [x1, y1, x2, y2]:
                raw_crop_np = img_np[y1:y2, x1:x2]
                has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free)

            if cache_key in self._page_ocr_cache:
                recognized_text = self._page_ocr_cache[cache_key]
                if (
                    not has_ink
                    or cache_key in rejected_hallucination_keys
                    or self._is_hallucinated_ocr_text(recognized_text, ink_pixels, is_free_text=is_free, crop_shape=raw_crop_np.shape[:2])
                ):
                    recognized_text = ""
            else:
                if not has_ink or not self._has_real_text_glyph_structure(raw_crop_np, is_free_text=is_free):
                    recognized_text = ""
                else:
                    padded_crop = self._crop_with_padding(rgb_image, ocr_coords, pad=8)
                    recognized_text = self._recognize_crop(
                        padded_crop,
                        raw_crop_np,
                        resolved_lang,
                        is_free_text=is_free,
                    )

            if not recognized_text:
                if not has_ink:
                    no_ink_count += 1
                else:
                    empty_ocr_count += 1
                continue

            filter_reason = self.classify_filter_reason(
                recognized_text,
                is_free_text=is_free,
                text_filter_mode=mode,
                skip_sighs=skip_sighs,
                skip_sfx=skip_sfx,
                min_text_length=min_text_length,
            )
            if filter_reason is not None:
                r_code, r_label = filter_reason
                skipped_by_filter += 1
                filtered_items_diag.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "text": recognized_text,
                        "reason_code": r_code,
                        "reason_ru": r_label,
                        "is_free_text": is_free,
                    }
                )
                if not is_free and not self.is_sfx_or_onomatopoeia(recognized_text, is_free_text=False):
                    filtered_bubble_SamePage_fallback.append(
                        {
                            "id": 0,
                            "bbox": [x1, y1, x2, y2],
                            "clean_bbox": [x1, y1, x2, y2],
                            "text_bbox": ocr_coords,
                            "confidence": det["confidence"],
                            "original_text": recognized_text,
                            "is_free_text": False,
                            "is_deleted": False,
                        }
                    )
                logger.debug(
                    "Блок [%d,%d,%d,%d] '%s' пропущен фильтром объема текста (%s, mode=%s)",
                    x1,
                    y1,
                    x2,
                    y2,
                    recognized_text,
                    r_label,
                    mode,
                )
                continue

            output_blocks.append(
                {
                    "id": block_id,
                    "bbox": [x1, y1, x2, y2],
                    "clean_bbox": [x1, y1, x2, y2],
                    "text_bbox": ocr_coords,
                    "confidence": det["confidence"],
                    "original_text": recognized_text,
                    "is_free_text": is_free,
                    "is_deleted": False,
                }
            )
            block_id += 1

        # Защита от пустых страниц: срабатывает ТОЛЬКО если пользователь НЕ включал пропуск вздохов (skip_sighs=False),
        # а фильтр min_text_length отсеял единственную короткую реплику на странице (не более 2 реальных бабблов).
        # Никогда не возвращает отфильтрованные вздохи/шум при skip_sighs=True!
        if filtered_bubble_SamePage_fallback and not skip_sighs and mode not in ("long_only", "dialogue_only", "bubbles_no_sighs"):
            kept_bubbles = sum(1 for b in output_blocks if not b.get("is_free_text"))
            if kept_bubbles == 0 and len(filtered_bubble_SamePage_fallback) <= 2:
                restored_by_empty_guard_count = len(filtered_bubble_SamePage_fallback)
                restored_bboxes = {tuple(b["bbox"]) for b in filtered_bubble_SamePage_fallback}
                filtered_items_diag = [
                    it for it in filtered_items_diag if tuple(it["bbox"]) not in restored_bboxes
                ]
                combined_all = output_blocks + filtered_bubble_SamePage_fallback
                combined_all = self._sort_boxes_reading_order(combined_all, rtl=(resolved_lang == "ja"))
                output_blocks = []
                for idx_new, b_item in enumerate(combined_all, start=1):
                    b_item["id"] = idx_new
                    output_blocks.append(b_item)
                skipped_by_filter = max(0, skipped_by_filter - len(filtered_bubble_SamePage_fallback))

        if skipped_by_filter > 0:
            logger.info(
                "Фильтр объема текста (mode=%s): пропущено %d мелких вздохов/спецэффектов, оставлено для перевода %d блоков.",
                mode,
                skipped_by_filter,
                len(output_blocks),
            )

        zero_reason_code = ""
        zero_reason_ru = ""
        if len(output_blocks) == 0:
            if len(filtered_items_diag) > 0:
                samples = ", ".join(f"«{it.get('text', '')}» ({it.get('reason_ru', '')})" for it in filtered_items_diag[:5])
                zero_reason_code = "all_filtered_out"
                zero_reason_ru = f"Весь распознанный текст ({len(filtered_items_diag)} шт.) отсеян фильтром настроек: {samples}"
            elif no_ink_count >= len(raw_detections):
                zero_reason_code = "no_ink_in_bubbles"
                zero_reason_ru = f"Детектор нашел контуры ({len(raw_detections)} шт.), но внутри нет читаемого текста/чернил"
            else:
                zero_reason_code = "empty_ocr_or_noise"
                zero_reason_ru = f"Детектор нашел области ({len(raw_detections)} шт.), но OCR отсеял их как шум/рисунок без букв (без чернил: {no_ink_count}, шум: {hallucinated_count})"

        self.last_page_diagnostics = {
            "raw_candidates_count": len(raw_detections),
            "yolo_bubbles": int(self.last_detection_stats.get("yolo_bubbles", 0)),
            "rtdetr_bubbles": int(self.last_detection_stats.get("rtdetr_bubbles", 0)),
            "rtdetr_text_bubbles": int(self.last_detection_stats.get("rtdetr_text_bubbles", 0)),
            "rtdetr_free_texts": int(self.last_detection_stats.get("rtdetr_free_texts", 0)),
            "split_multi_bubbles": int(self.last_detection_stats.get("split_multi_bubbles", 0)),
            "rescued_orphan_text_bubbles": int(self.last_detection_stats.get("rescued_orphan_text_bubbles", 0)),
            "no_ink_count": no_ink_count,
            "hallucinated_count": hallucinated_count,
            "empty_ocr_count": empty_ocr_count,
            "filtered_count": len(filtered_items_diag),
            "filtered_items": filtered_items_diag[:35],
            "restored_by_empty_guard_count": restored_by_empty_guard_count,
            "kept_blocks_count": len(output_blocks),
            "free_text_kept_count": sum(1 for b in output_blocks if b.get("is_free_text")),
            "detected_lang": resolved_lang,
            "zero_reason_code": zero_reason_code,
            "zero_reason_ru": zero_reason_ru,
        }

        self.last_detection_stats["ko_zh_ocr_backend"] = self.get_multilang_ocr_backend_name()
        self._page_ocr_cache.clear()
        return output_blocks, resolved_lang

    def recognize_manual_region(
        self,
        image: Image.Image,
        bbox: list[int],
        lang: str = "auto",
        use_vlm_ocr: bool = False,
        vlm_model: str | None = None,
        ollama_base_url: str | None = None,
        cloud_api_url: str | None = None,
        cloud_api_key: str | None = None,
    ) -> tuple[str, str, bool]:
        """
        Принудительное распознавание текста внутри выделенной пользователем мышкой области `bbox = [x1, y1, x2, y2]`
        без отсева по фильтрам вздохов/SFX/контраста (Шаги 1–4):
        - Если `use_vlm_ocr=True`, сразу выполняет снайперское чтение через Vision-модель (`recognize_crop_via_vlm`).
        - Иначе сначала использует быстрый каскад (`_compute_tight_ink_bbox` + `manga-ocr-nar-small` / `HayaiOCR v2.5 Nova`
          / `EasyOCR`), а если стандартный OCR не справился или вернул пустую строку — автоматически вызывает
          снайперский `recognize_crop_via_vlm`!
        Возвращает `(recognized_text, resolved_lang, is_free_text)`.
        """
        rgb_image = image.convert("RGB")
        img_w, img_h = rgb_image.size
        img_np = np.array(rgb_image)
        x1, y1, x2, y2 = [int(round(c)) for c in bbox[:4]]
        x1 = max(0, min(x1, img_w - 2))
        y1 = max(0, min(y1, img_h - 2))
        x2 = max(x1 + 4, min(x2, img_w))
        y2 = max(y1 + 4, min(y2, img_h))

        raw_crop_np = img_np[y1:y2, x1:x2]
        gray_crop = cv2.cvtColor(raw_crop_np, cv2.COLOR_RGB2GRAY)
        p70 = float(np.percentile(gray_crop, 70))
        # Если фон внутри выделенной области светлый (>= 205), считаем это кружком/бабблом для чистой заливки
        is_free_text = bool(p70 < 205.0)

        req_lang = (lang or "auto").lower().strip()
        if req_lang not in ("ja", "ko", "zh"):
            resolved_lang = self._last_detected_lang or "ja"
        else:
            resolved_lang = req_lang

        # 1. Если пользователь явно запросил снайперский VLM-OCR (`use_vlm_ocr=True`)
        if use_vlm_ocr:
            try:
                vlm_txt, vlm_lang, _ = self.recognize_crop_via_vlm(
                    rgb_image=rgb_image,
                    bbox=[x1, y1, x2, y2],
                    lang=req_lang,
                    vlm_model=vlm_model,
                    ollama_base_url=ollama_base_url,
                    cloud_api_url=cloud_api_url,
                    cloud_api_key=cloud_api_key,
                )
                if vlm_txt and not PUNCT_ONLY_REGEX.match(vlm_txt):
                    return vlm_txt, vlm_lang, is_free_text
            except Exception as vlm_exc:
                logger.warning("Сбой VLM-OCR при ручном выделении, переход к локальному OCR: %s", vlm_exc)

        # ВАЖНО: При ручном выделении («➕ Обвести») пользователь сам обводит нужный текст!
        # Нельзя полагаться только на `_compute_tight_ink_bbox`, так как буквы часто касаются границ
        # пользовательской рамки и отбрасываются как «стенки баббла», оставляя одну точку/дакутэн в центре,
        # которую декодер читает как «ビクッ» / «ピクッ» (-> «Twitch»)!
        full_padded = self._crop_with_padding(rgb_image, [x1, y1, x2, y2], pad=12)
        full_padded_np = np.array(full_padded)
        tight_coords = self._compute_tight_ink_bbox(img_np, [x1, y1, x2, y2], None, is_free_text=is_free_text)
        tx1, ty1, tx2, ty2 = tight_coords
        tight_area = max(1, (tx2 - tx1) * (ty2 - ty1))
        full_area = max(1, (x2 - x1) * (y2 - y1))
        # Используем tight_coords только если он сохранил >= 45% площади выделения пользователя
        use_tight = tight_coords != [x1, y1, x2, y2] and (tight_area >= full_area * 0.45)
        tight_padded = self._crop_with_padding(rgb_image, tight_coords, pad=10) if use_tight else full_padded
        tight_raw_np = img_np[ty1:ty2, tx1:tx2] if use_tight else raw_crop_np
        _, full_ink_px = self._has_meaningful_ink(raw_crop_np, is_free_text=False)

        recognized = ""
        if resolved_lang in ("ko", "zh"):
            try:
                recognized = self._recognize_single_ko_zh_crop(
                    full_padded,
                    raw_crop_np,
                    lang=resolved_lang,
                    is_free_text=False,
                )
            except Exception:
                recognized = ""
            if (not recognized or PUNCT_ONLY_REGEX.match(recognized)) and use_tight:
                try:
                    recognized = self._recognize_single_ko_zh_crop(
                        tight_padded,
                        tight_raw_np,
                        lang=resolved_lang,
                        is_free_text=False,
                    )
                except Exception:
                    pass
        else:
            # Для одиночного выделенного пользователем баббла запускаем и классический MangaOCR (с выравниванием пропорций ViT),
            # и manga-ocr-nar-small на полном кропе `full_padded`, выбирая наиболее полное и осмысленное чтение!
            ja_candidates: list[str] = []
            try:
                classic_outs = self._run_classic_manga_ocr_batch([full_padded], batch_size=1)
                if classic_outs and classic_outs[0].strip():
                    ja_candidates.append(classic_outs[0].strip())
            except Exception:
                pass
            try:
                nar_outs = self._recognize_crops_batch_ja([full_padded])
                if nar_outs and nar_outs[0].strip() and nar_outs[0].strip() not in ja_candidates:
                    ja_candidates.append(nar_outs[0].strip())
            except Exception:
                pass
            if use_tight:
                try:
                    classic_tight = self._run_classic_manga_ocr_batch([tight_padded], batch_size=1)
                    if classic_tight and classic_tight[0].strip() and classic_tight[0].strip() not in ja_candidates:
                        ja_candidates.append(classic_tight[0].strip())
                except Exception:
                    pass

            def _score_ja_manual_cand(cand_s: str) -> float:
                c = cand_s.strip()
                if not c or PUNCT_ONLY_REGEX.match(c) or KNOWN_OCR_HALLUCINATIONS_REGEX.search(c):
                    return -999.0
                core = STRIP_PUNCT_REGEX.sub("", c)
                if not core or REPEATED_NOISE_REGEX.match(core):
                    return -999.0
                score = float(len(core)) * 2.5
                kanji_cnt = len(HANZI_REGEX.findall(core))
                kana_cnt = len(KANA_REGEX.findall(core))
                score += kanji_cnt * 4.5 + kana_cnt * 1.5
                # Если локальный декодер выдал одиночный короткий SFX вроде «ビクッ» / «ピクッ» («Twitch»)
                # или вздох при большом количестве чернил в баббле — сильно штрафуем такой кандидат!
                if self.is_sfx_or_onomatopoeia(c, is_free_text=True) or c.strip("〜~！!？?…・ ") in ("ビクッ", "ピクッ", "ぴくっ", "びくっ", "ビク", "ピク"):
                    if full_ink_px >= 110 or full_area >= 1800:
                        score -= 25.0
                    else:
                        score -= 6.0
                elif self.is_sigh_or_interjection(c) and len(core) <= 3 and full_ink_px >= 160:
                    score -= 10.0
                return score

            scored_ja = sorted(
                [( _score_ja_manual_cand(cand), cand ) for cand in ja_candidates],
                key=lambda item: item[0],
                reverse=True,
            )
            if scored_ja and scored_ja[0][0] > -100.0:
                recognized = scored_ja[0][1]

            # Если текст пустой или на другом языке, пробуем HayaiOCR v2.5 Nova или EasyOCR
            if not recognized or PUNCT_ONLY_REGEX.match(recognized):
                if self.use_hayai_ocr and self._hayai_available is not False:
                    h_outs = self._recognize_crops_batch_hayai([full_padded], max_num_patches=512, lang="auto")
                    cand_h = h_outs[0].strip() if h_outs else ""
                    if cand_h and not PUNCT_ONLY_REGEX.match(cand_h):
                        recognized = cand_h
                        if HANGUL_REGEX.search(recognized):
                            resolved_lang = "ko"
                        elif HANZI_REGEX.search(recognized) and not KANA_REGEX.search(recognized):
                            resolved_lang = "zh"
                if (not recognized or PUNCT_ONLY_REGEX.match(recognized)) and self._check_easyocr():
                    for fallback_reader in (self._get_ko_reader(), self._get_zh_reader()):
                        if fallback_reader is not None:
                            lines = fallback_reader.readtext(full_padded_np, detail=0, paragraph=True)
                            cand = " ".join(str(x).strip() for x in lines if str(x).strip())
                            if cand and not PUNCT_ONLY_REGEX.match(cand):
                                recognized = cand
                                break

        # Если с первого раза текст слишком бледный, пробуем повысить контраст кропа
        if not recognized or PUNCT_ONLY_REGEX.match(recognized):
            try:
                contrast_pil = ImageOps.autocontrast(full_padded, cutoff=2)
                with torch.inference_mode():
                    cand2 = self.ja_recognizer(self._pad_to_balanced_aspect_for_vit(contrast_pil)).strip()
                if cand2 and not PUNCT_ONLY_REGEX.match(cand2) and not KNOWN_OCR_HALLUCINATIONS_REGEX.search(cand2):
                    recognized = cand2
            except Exception:
                pass

        # Автоматический снайперский фоллбэк на VLM-OCR (Шаг 4):
        # Срабатывает не только когда стандартный OCR вернул пустую строку, но и когда он ошибочно прочитал
        # крупный баббл с текстом (`full_ink_px >= 110`) как короткий шумовой SFX «ビクッ»/«ピクッ» («Twitch»)!
        rec_core = STRIP_PUNCT_REGEX.sub("", recognized or "")
        is_suspicious_sfx_on_rich_bubble = bool(
            recognized
            and full_ink_px >= 110
            and len(rec_core) <= 4
            and (
                self.is_sfx_or_onomatopoeia(recognized, is_free_text=True)
                or rec_core in ("ビクッ", "ピクッ", "ぴくっ", "びくっ", "ビク", "ピク", "ヒクッ", "ひくっ")
            )
        )
        if not use_vlm_ocr and (not recognized or PUNCT_ONLY_REGEX.match(recognized) or is_suspicious_sfx_on_rich_bubble):
            try:
                vlm_txt, vlm_lang, _ = self.recognize_crop_via_vlm(
                    rgb_image=rgb_image,
                    bbox=[x1, y1, x2, y2],
                    lang=req_lang,
                    vlm_model=vlm_model,
                    ollama_base_url=ollama_base_url,
                    cloud_api_url=cloud_api_url,
                    cloud_api_key=cloud_api_key,
                )
                if vlm_txt and not PUNCT_ONLY_REGEX.match(vlm_txt):
                    recognized = vlm_txt
                    resolved_lang = vlm_lang
            except Exception as vlm_fb_exc:
                logger.debug("Фоллбэк VLM-OCR пропущен: %s", vlm_fb_exc)

        return recognized, resolved_lang, is_free_text