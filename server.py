"""
FastAPI-сервер для локального переводчика манги, манхвы и маньхуа (v1.0.6).

Изменения v1.0.6:
- Расширение схемы `TextBlock` полями профессиональной верстки и локальной ретуши маски:
  `font_family`, `text_color`, `stroke_color`, `stroke_width`, `align`, `angle`,
  `style_preset`, `mask_dilate`, `custom_mask_strokes` (с полной обратной совместимостью).
- Возврат списка доступных шрифтов `available_fonts` в `GET /health` из папки `fonts/` и системных шрифтов.
- Учет `mask_dilate` и `custom_mask_strokes` при расчете `clean_state_hash`.

Изменения v1.0.0:
1. Эндпоинты `POST /pull_model` и `GET /pull_status` для автоматического скачивания лучших моделей
   (Shisa V2.1 14B, Shisa V2.1 8B, Shisa V2 12B Uncensored, Gemma 3 12B, Qwen 2.5 14B) напрямую через Ollama
   с отображением прогресса в процентах и гигабайтах и установкой по умолчанию в 1 клик (без запуска Unsloth Studio!).
2. Полная поддержка внешних OpenAI-совместимых API-ключей (FunPay `positionapi.top/v1`, `list-api.listapi.workers.dev/v1`,
   TokenWave, DeepSeek Official API, OpenRouter) с сохранением `unsloth_api_key` и `unsloth_url` в `config.json`.
3. Поддержка независимых координат `clean_bbox` (исходное место замазки текста) и `bbox` (интерактивно перемещаемая
   и масштабируемая мышкой рамка перевода) + возврат `rendered_font_size`.
"""

from __future__ import annotations

import asyncio
import copy
import base64
import binascii
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import cv2
import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from pydantic import BaseModel, Field

from cleaner import BubbleCleaner
from ocr_engine import MangaOCREngine
from renderer import MangaRenderer
from translator import (
    CLOUD_API_PRESETS,
    RECOMMENDED_LOCAL_MODELS,
    TRANSLATION_PROFILES,
    OllamaTranslator,
    export_glossary_to_sakura_text,
    export_glossary_to_tsv,
    japanese_to_romaji_local,
    merge_glossaries,
    normalize_glossary_entries,
    is_translated_sigh_text,
    parse_glossary_text_or_json,
)

from collections import OrderedDict, deque
import hashlib
from logging.handlers import RotatingFileHandler
import platform
from urllib.parse import quote

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp_debug"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MAIN_LOG_FILE = LOGS_DIR / "miko_studio.log"
DIAG_EVENTS_FILE = LOGS_DIR / "diagnostics_events.jsonl"
UX_TELEMETRY_JSON_FILE = LOGS_DIR / "ux_telemetry.json"
UX_EVENTS_JSONL_FILE = LOGS_DIR / "ux_events.jsonl"
UX_REPORT_TXT_FILE = LOGS_DIR / "UX_USAGE_REPORT.txt"
SERVER_STARTED_AT = time.time()

_recent_log_entries: deque[dict[str, Any]] = deque(maxlen=400)
_recent_warn_err_entries: deque[dict[str, Any]] = deque(maxlen=80)
_recent_job_profiles: deque[dict[str, Any]] = deque(maxlen=50)


class RingBufferLogHandler(logging.Handler):
    """Перехватывает все логи сервера в кольцевой буфер ОЗУ для живой диагностики в интерфейсе и отчетах."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            entry = {
                "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "formatted": msg,
            }
            _recent_log_entries.append(entry)
            if record.levelno >= logging.WARNING:
                _recent_warn_err_entries.append(entry)
        except Exception:
            pass


_log_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
_root_logger = logging.getLogger()
try:
    _file_handler = RotatingFileHandler(
        MAIN_LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    _file_handler.setFormatter(_log_formatter)
    _root_logger.addHandler(_file_handler)
except Exception:
    pass

_ring_handler = RingBufferLogHandler()
_ring_handler.setFormatter(_log_formatter)
_root_logger.addHandler(_ring_handler)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logger = logging.getLogger("manga_server")


def _get_vram_allocated_mb() -> float:
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated(0) / (1024**2), 1)
    except Exception:
        pass
    return 0.0


def _append_diagnostic_event(event_type: str, data: dict[str, Any]) -> None:
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": int(time.time()),
        "event_type": event_type,
        "vram_mb": _get_vram_allocated_mb(),
        **data,
    }
    _recent_job_profiles.append(record)
    try:
        with open(DIAG_EVENTS_FILE, "a", encoding="utf-8") as f_ev:
            f_ev.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VALID_SOURCE_LANGS = {"auto", "ja", "ko", "zh"}
VALID_TARGET_LANGS = {"en", "ru", "romaji"}
OUTPUT_FOLDER_SUFFIXES = ("_EN", "_RU", "_ROMAJI", "temp_debug")
MANIFEST_FILENAME = ".translation_state.json"
REPORT_FILENAME = "TRANSLATION_REPORT.txt"

ocr_engine: MangaOCREngine | None = None
translator_engine: OllamaTranslator | None = None
cleaner_engine: BubbleCleaner | None = None
renderer_engine: MangaRenderer | None = None

gpu_lock = asyncio.Lock()

folder_job_state: dict[str, Any] = {
    "status": "idle",
    "source_dir": None,
    "output_dir": None,
    "lang": "auto",
    "detected_lang": "ja",
    "target_lang": "en",
    "total": 0,
    "current": 0,
    "skipped_existing": 0,
    "translated_in_run": 0,
    "current_file": "",
    "elapsed_sec": 0.0,
    "error": None,
}
_folder_task: asyncio.Task[None] | None = None
_pull_task: asyncio.Task[Any] | None = None
_folder_stop_requested: bool = False
_AUTO_OPEN_GUI: bool = False


def _ensure_block_clean_bbox(block: dict[str, Any]) -> dict[str, Any]:
    """Гарантирует наличие исходного `clean_bbox` и безопасных дефолтов полей стилизации/ретуши (Шаг 3)."""
    bbox_raw = block.get("bbox") or [0, 0, 0, 0]
    bbox = [int(round(float(x))) for x in bbox_raw[:4]] if len(bbox_raw) >= 4 else [0, 0, 0, 0]
    block["bbox"] = bbox

    clean_raw = block.get("clean_bbox") or block.get("original_bbox")
    if isinstance(clean_raw, (list, tuple)) and len(clean_raw) >= 4:
        clean_bbox = [int(round(float(x))) for x in clean_raw[:4]]
    else:
        clean_bbox = list(bbox)
    block["clean_bbox"] = clean_bbox
    tb_raw = block.get("text_bbox")
    if isinstance(tb_raw, (list, tuple)) and len(tb_raw) >= 4:
        try:
            block["text_bbox"] = [int(round(float(x))) for x in tb_raw[:4]]
        except Exception:
            block["text_bbox"] = None
    else:
        block.setdefault("text_bbox", None)
    block["bbox_modified"] = bool(block.get("bbox_modified", False)) or (clean_bbox != bbox)
    cw = clean_bbox[2] - clean_bbox[0]
    ch = clean_bbox[3] - clean_bbox[1]
    nw = bbox[2] - bbox[0]
    nh = bbox[3] - bbox[1]
    block["bbox_resized"] = bool(block.get("bbox_resized", False)) or (abs(nw - cw) >= 3 or abs(nh - ch) >= 3)

    # Поля профессиональной верстки и ретуши маски (Шаг 3) с обратной совместимостью
    block.setdefault("font_family", None)
    block.setdefault("text_color", None)
    block.setdefault("stroke_color", None)
    if block.get("stroke_width") is not None:
        try:
            block["stroke_width"] = max(0, min(20, int(block["stroke_width"])))
        except (ValueError, TypeError):
            block["stroke_width"] = None
    else:
        block["stroke_width"] = None

    align_val = str(block.get("align") or "center").strip().lower()
    block["align"] = align_val if align_val in ("center", "left", "right") else "center"

    try:
        angle_val = float(block.get("angle") or 0.0)
        block["angle"] = max(-180.0, min(180.0, round(angle_val, 1)))
    except (ValueError, TypeError):
        block["angle"] = 0.0

    preset_val = str(block.get("style_preset") or "dialogue").strip().lower()
    block["style_preset"] = (
        preset_val
        if preset_val in ("dialogue", "thought", "shout", "whisper", "narration", "sfx")
        else "dialogue"
    )

    bg_style_val = str(block.get("bg_style") or "none").strip().lower()
    if bg_style_val in ("text_box", "pill", "box", "text", "lines"):
        block["bg_style"] = "text_box"
    elif bg_style_val in ("bubble", "ellipse", "oval", "capsule"):
        block["bg_style"] = "bubble"
    elif bg_style_val in ("full_box", "rect", "frame", "full"):
        block["bg_style"] = "full_box"
    else:
        block["bg_style"] = "none"

    block.setdefault("bg_color", None)
    if block.get("bg_padding") is not None:
        try:
            block["bg_padding"] = max(-24, min(48, int(block["bg_padding"])))
        except (ValueError, TypeError):
            block["bg_padding"] = None
    else:
        block["bg_padding"] = None

    if block.get("mask_dilate") is not None:
        try:
            md_val = int(block["mask_dilate"])
            block["mask_dilate"] = max(1, min(28, md_val)) if md_val > 0 else None
        except (ValueError, TypeError):
            block["mask_dilate"] = None
    else:
        block["mask_dilate"] = None

    raw_strokes = block.get("custom_mask_strokes")
    if isinstance(raw_strokes, list):
        norm_strokes: list[dict[str, Any]] = []
        for st in raw_strokes:
            if not isinstance(st, dict):
                continue
            try:
                sx = int(round(float(st.get("x", 0))))
                sy = int(round(float(st.get("y", 0))))
                sr = max(1, min(80, int(round(float(st.get("r", 10))))))
                smode = "erase" if str(st.get("mode") or "add").strip().lower() == "erase" else "add"
                norm_strokes.append({"x": sx, "y": sy, "r": sr, "mode": smode})
            except (ValueError, TypeError):
                continue
        block["custom_mask_strokes"] = norm_strokes if norm_strokes else None
    else:
        block["custom_mask_strokes"] = None

    block.setdefault("translation_source", None)
    block.setdefault("skip_reason", None)

    return block


_CLEANED_PAGE_CACHE_MAX = 8
_cleaned_page_lru_cache: OrderedDict[tuple[str, str], Image.Image] = OrderedDict()
_cleaned_page_cache_lock = threading.Lock()


def _is_block_active_for_cleaning(block: dict[str, Any]) -> bool:
    if not isinstance(block, dict):
        return False
    if bool(block.get("is_deleted", False)):
        return False
    if "translated_text" in block and not str(block.get("translated_text") or "").strip():
        return False
    return True


def _compute_clean_state_hash(
    src_file: Path | str | None,
    blocks: list[dict[str, Any]],
    dilate_size: int | None = None,
) -> str:
    """
    Вычисляет детерминированный хэш состояния замазки страницы `clean_state_hash`
    на основе пути исходного файла `src_file`, его `mtime`, глобального `dilate_size`
    и отсортированного списка кортежей `(tuple(clean_bbox), bool(is_free_text), int(block.get("mask_dilate") or 0))`
    для всех активных неудаленных блоков.
    """
    eff_dilate = int(
        dilate_size
        if dilate_size is not None
        else (cleaner_engine.dilate_size if cleaner_engine is not None else 7)
    )
    if cleaner_engine is not None and hasattr(cleaner_engine, "compute_clean_state_hash"):
        return cleaner_engine.compute_clean_state_hash(blocks=blocks, src_file=src_file)

    src_key = ""
    src_mtime = 0
    if src_file:
        try:
            p = Path(src_file).resolve()
            src_key = str(p)
            if p.exists():
                src_mtime = p.stat().st_mtime_ns
        except Exception:
            src_key = str(src_file)

    active_tuples: list[tuple[tuple[int, int, int, int], bool, int, tuple[tuple[int, int, int, str], ...]]] = []
    for b in blocks:
        if not _is_block_active_for_cleaning(b):
            continue
        raw_cb = b.get("clean_bbox") or b.get("original_bbox") or b.get("bbox") or [0, 0, 0, 0]
        cb = [int(round(float(c))) for c in raw_cb[:4]] if len(raw_cb) >= 4 else [0, 0, 0, 0]
        clean_tuple = (cb[0], cb[1], cb[2], cb[3])
        is_free = bool(b.get("is_free_text", False))
        mask_dilate_val = int(b.get("mask_dilate") or 0)
        raw_st = b.get("custom_mask_strokes") or []
        st_list: list[tuple[int, int, int, str]] = []
        if isinstance(raw_st, list):
            for st in raw_st:
                if isinstance(st, dict):
                    try:
                        st_list.append((
                            int(round(float(st.get("x", 0)))),
                            int(round(float(st.get("y", 0)))),
                            max(1, int(round(float(st.get("r", 8))))),
                            "erase" if str(st.get("mode") or "add").lower() == "erase" else "add",
                        ))
                    except (ValueError, TypeError):
                        pass
        active_tuples.append((clean_tuple, is_free, mask_dilate_val, tuple(st_list)))

    active_tuples.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[1], item[2], item[3]))
    payload = {
        "src_file": src_key,
        "mtime": int(src_mtime),
        "dilate_size": eff_dilate,
        "active_clean_blocks": [
            [list(coords), is_free, mdil, [list(s) for s in st_tup]]
            for coords, is_free, mdil, st_tup in active_tuples
        ],
    }
    raw_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:24]


def _get_cached_cleaned_image(
    src_file: Path | str | None,
    clean_state_hash: str,
) -> Image.Image | None:
    if not src_file or not clean_state_hash:
        return None
    try:
        src_key = str(Path(src_file).resolve())
    except Exception:
        src_key = str(src_file)

    cache_key = (src_key, clean_state_hash)
    with _cleaned_page_cache_lock:
        cached = _cleaned_page_lru_cache.get(cache_key)
        if cached is not None:
            _cleaned_page_lru_cache.move_to_end(cache_key)
            return cached.copy()

    if cleaner_engine is not None and hasattr(cleaner_engine, "get_cached_cleaned_image"):
        eng_cached = cleaner_engine.get_cached_cleaned_image(src_key, clean_state_hash)
        if eng_cached is not None:
            with _cleaned_page_cache_lock:
                _cleaned_page_lru_cache[cache_key] = eng_cached.copy()
                _cleaned_page_lru_cache.move_to_end(cache_key)
                while len(_cleaned_page_lru_cache) > _CLEANED_PAGE_CACHE_MAX:
                    _cleaned_page_lru_cache.popitem(last=False)
            return eng_cached
    return None


def _store_cached_cleaned_image(
    src_file: Path | str | None,
    clean_state_hash: str,
    cleaned_image: Image.Image,
    save_to_disk: bool = True,
) -> None:
    if not src_file or not clean_state_hash:
        return
    try:
        src_key = str(Path(src_file).resolve())
    except Exception:
        src_key = str(src_file)

    cache_key = (src_key, clean_state_hash)
    rgb_copy = cleaned_image.convert("RGB").copy()
    with _cleaned_page_cache_lock:
        _cleaned_page_lru_cache[cache_key] = rgb_copy
        _cleaned_page_lru_cache.move_to_end(cache_key)
        while len(_cleaned_page_lru_cache) > _CLEANED_PAGE_CACHE_MAX:
            _cleaned_page_lru_cache.popitem(last=False)

    if cleaner_engine is not None and hasattr(cleaner_engine, "put_cached_cleaned_image"):
        cleaner_engine.put_cached_cleaned_image(
            src_key,
            clean_state_hash,
            rgb_copy,
            save_to_disk=save_to_disk,
        )


async def _get_or_clean_page_image(
    pil_image: Image.Image,
    active_blocks: list[dict[str, Any]],
    src_file: Path | str | None = None,
    save_to_disk: bool = True,
) -> tuple[Image.Image, Image.Image | None, bool, str]:
    """
    Возвращает `(cleaned_image, mask_image, cache_hit, clean_state_hash)`.
    - Если `clean_state_hash` для `src_file` совпадает с записью в LRU-кэше (8 последних страниц)
      или в `temp_debug/clean_cache/`, возвращает готовый `cleaned_image` за 0 мс.
    - Если изменились только ручные мазки кисти (`custom_mask_strokes`), а базовая очистка (`base_hash`)
      уже есть в кэше, мгновенно (< 2 мс) накладывает мазки поверх базового слоя без вызова SimpleLama.
    """
    assert cleaner_engine is not None
    clean_state_hash = _compute_clean_state_hash(src_file, active_blocks)
    if src_file:
        cached_clean = _get_cached_cleaned_image(src_file, clean_state_hash)
        if cached_clean is not None:
            return cached_clean, None, True, clean_state_hash

        if hasattr(cleaner_engine, "compute_base_clean_hash") and hasattr(cleaner_engine, "_apply_custom_strokes_fast"):
            base_hash = cleaner_engine.compute_base_clean_hash(active_blocks, src_file=src_file)
            if base_hash != clean_state_hash:
                base_cached = _get_cached_cleaned_image(src_file, base_hash)
                if base_cached is not None:
                    def _apply_strokes_on_cached() -> Image.Image:
                        base_np = np.array(base_cached.convert("RGB"))
                        raw_np = np.array(pil_image.convert("RGB"))
                        empty_m = np.zeros(raw_np.shape[:2], dtype=np.uint8)
                        out_np, _ = cleaner_engine._apply_custom_strokes_fast(base_np, raw_np, empty_m, active_blocks)
                        return Image.fromarray(out_np, mode="RGB")

                    fast_cleaned = await asyncio.to_thread(_apply_strokes_on_cached)
                    _store_cached_cleaned_image(
                        src_file,
                        clean_state_hash,
                        fast_cleaned,
                        save_to_disk=False,
                    )
                    return fast_cleaned, None, True, clean_state_hash

    async with gpu_lock:
        cleaned_image, mask_image = await asyncio.to_thread(
            cleaner_engine.clean_image, pil_image, active_blocks, src_file
        )
    if src_file:
        has_strokes = any(bool(b.get("custom_mask_strokes")) for b in active_blocks if isinstance(b, dict))
        _store_cached_cleaned_image(
            src_file,
            clean_state_hash,
            cleaned_image,
            save_to_disk=(save_to_disk and not has_strokes),
        )
    return cleaned_image, mask_image, False, clean_state_hash


def launch_studio_window(url: str = "http://127.0.0.1:5000/") -> None:
    print(f"\n>>> Открываю окно интерфейса Miko-chan Manga Studio ({url})...", flush=True)
    if os.name == "nt":
        candidates = [
            shutil.which("msedge"),
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            shutil.which("chrome"),
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        for exe in candidates:
            if exe and os.path.exists(exe):
                try:
                    subprocess.Popen([exe, f"--app={url}", "--window-size=1480,900"])
                    return
                except Exception:
                    continue
    try:
        webbrowser.open(url)
    except Exception as exc:
        logger.warning("Не удалось автоматически открыть окно браузера: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_engine, translator_engine, cleaner_engine, renderer_engine
    print("\n==================================================================", flush=True)
    print("  [1/4] Инициализация детектора YOLOv8 и MangaOCR (GPU/CPU)...", flush=True)
    ocr_engine = MangaOCREngine(models_dir=BASE_DIR / "models")

    print("  [2/4] Инициализация нейросетевой очистки SimpleLama...", flush=True)
    cleaner_engine = BubbleCleaner(use_lama=True, dilate_size=7, cache_max_pages=8, disk_cache_dir=TEMP_DIR / "clean_cache")

    print("  [3/4] Инициализация шрифтового движка и клиента Ollama/Cloud API...", flush=True)
    renderer_engine = MangaRenderer(
        fonts_dir=BASE_DIR / "fonts",
        overflow_mode=getattr(translator_engine, "overflow_mode", "shrink") if translator_engine else "shrink",
    )
    translator_engine = OllamaTranslator(config_path=BASE_DIR / "config.json")
    resolved_model = await translator_engine.resolve_model()

    print("  [4/4] СЕРВЕР ГОТОВ К РАБОТЕ! (v1.0.6)", flush=True)
    print(f"        Активная модель перевода : {resolved_model}", flush=True)
    print("        Адрес локальной студии   : http://127.0.0.1:5000/", flush=True)
    print("==================================================================\n", flush=True)

    if _AUTO_OPEN_GUI:
        threading.Timer(0.8, launch_studio_window, args=("http://127.0.0.1:5000/",)).start()

    yield
    logger.info("Остановка сервера и очистка ресурсов...")
    if translator_engine is not None:
        await translator_engine.aclose()
    ocr_engine = None
    translator_engine = None
    cleaner_engine = None
    renderer_engine = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Local Manga/Manhwa/Manhua Translator API",
    description="Локальный пайплайн перевода манги (JA), манхвы (KO) и маньхуа (ZH) с поддержкой Shisa V2.1, Ромадзи и облачных API",
    version="1.0.6",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    image: str | None = Field(default=None, description="Base64 изображение")
    image_base64: str | None = Field(default=None, description="Алиас для Base64 изображения")
    lang: str = Field(default="auto", description="Язык оригинала: 'auto', 'ja', 'ko' или 'zh'")
    target_lang: str = Field(default="en", description="Целевой язык: 'en', 'ru' или 'romaji'")
    ollama_model: str | None = Field(default=None)
    model: str | None = Field(default=None)
    uppercase: bool = Field(default=True)
    min_font_size: int = Field(default=12, description="Минимальный размер шрифта (px)")
    max_font_size: int = Field(default=32, description="Максимальный размер шрифта (px)")
    overflow_mode: str | None = Field(
        default=None,
        description="Режим вписывания длинного текста: 'shrink' (сжимать до мин. шрифта), 'overflow' (Вариант Б: выход за рамки баббла), 'expand_bubble' (Вариант В: авто-расширение/перерисовка баббла)",
    )
    detect_free_text: bool | None = Field(
        default=None,
        description="Переводить также текст вне speech bubbles (надписи на фоне, мысли, заголовки)",
    )
    text_filter_mode: str | None = Field(
        default=None,
        description="Режим объема перевода: 'dialogue_only', 'bubbles_no_sighs', 'bubbles_only', 'long_only', 'all'",
    )
    skip_sighs: bool | None = Field(
        default=None,
        description="Не переводить и не замазывать короткие вздохи/охи (а, о, ах, ох, мм и т.п.)",
    )
    skip_sfx: bool | None = Field(
        default=None,
        description="Не переводить и не замазывать звуковые спецэффекты (SFX / ономатопею)",
    )
    min_text_length: int | None = Field(
        default=None,
        description="Минимальная длина текста в символах для перевода",
    )
    return_image: bool = Field(default=True)
    image_format: str = Field(default="webp", description="Формат возвращаемой картинки: 'webp', 'png' или 'jpeg'")
    save_debug: bool = Field(default=False)
    auto_save: bool | None = Field(
        default=None,
        description="Автоматически сохранять переведенную страницу на ПК (по умолчанию True из настроек сервера)",
    )
    save_dir: str | None = Field(
        default=None,
        description="Папка на ПК для сохранения переведенной манги из браузера (по умолчанию Saved_Manga)",
    )
    chapter_title: str | None = Field(
        default=None,
        description="Название манги/главы (из заголовка вкладки браузера) для создания подпапки",
    )
    page_url: str | None = Field(default=None, description="URL страницы сайта, где читается манга")
    image_url: str | None = Field(default=None, description="URL исходной картинки манги")
    page_index: int | None = Field(default=None, description="Порядковый номер картинки на веб-странице (1..N)")
    force_retranslate: bool = Field(
        default=False,
        description="Принудительно перевести все блоки заново через ИИ (игнорируя и обновляя старый кэш фраз)",
    )


class TextBlock(BaseModel):
    id: int
    bbox: list[int]
    clean_bbox: list[int] | None = Field(
        default=None,
        description="Исходные координаты оригинального текста для инпейнтинга (не смещаются при перетаскивании рамки перевода)",
    )
    text_bbox: list[int] | None = Field(
        default=None,
        description="Точный внутренний контур букв внутри баббла (от двухконтурного детектора RT-DETR-v2 / TightInkCrop)",
    )
    bbox_modified: bool = Field(
        default=False,
        description="Если True, пользователь вручную передвинул или изменил размер рамки перевода мышкой",
    )
    bbox_resized: bool = Field(
        default=False,
        description="Если True, размер рамки был изменен пользователем (ресайз, объединение Merge или разделение Split)",
    )
    confidence: float = 1.0
    original_text: str
    romaji_text: str = ""
    translated_text: str = ""
    is_free_text: bool = False
    is_deleted: bool = Field(
        default=False,
        description="Если True, перевод этого блока удален пользователем — на картинке сохраняется оригинальный текст/рисунок",
    )
    font_size: int | None = Field(default=None, description="Индивидуальный размер шрифта для этого блока (px), если задан вручную")
    rendered_font_size: int | None = Field(default=None, description="Фактический кегль шрифта (px), рассчитанный при верстке")
    font_family: str | None = Field(
        default=None,
        description="Имя файла шрифта из папки fonts/ или системный идентификатор шрифта для данного блока",
    )
    text_color: str | None = Field(
        default=None,
        description="Цвет текста в формате HEX (#RRGGBB). Если None — авто-контраст по яркости фона",
    )
    stroke_color: str | None = Field(
        default=None,
        description="Цвет обводки в формате HEX (#RRGGBB). Если None — авто-контраст",
    )
    stroke_width: int | None = Field(
        default=None,
        description="Толщина обводки текста (0..16 px). Если None — авто-расчет от кегля",
    )
    align: str = Field(
        default="center",
        description="Выравнивание текста: 'center', 'left' или 'right'",
    )
    angle: float = Field(
        default=0.0,
        description="Угол поворота текста в градусах (-45..+45)",
    )
    style_preset: str = Field(
        default="dialogue",
        description="Пресет оформления блока: 'dialogue', 'thought', 'shout', 'whisper', 'narration', 'sfx'",
    )
    bg_style: str = Field(
        default="none",
        description="Режим подложки под текстом: 'none' (выкл), 'text_box' (плашка по тексту), 'bubble' (овал внутри баббла), 'full_box' (вся рамка)",
    )
    bg_color: str | None = Field(
        default=None,
        description="Цвет подложки под текстом в формате HEX (#RRGGBB). Если None — авто (#FFFFFF для светлых бабблов)",
    )
    bg_padding: int | None = Field(
        default=None,
        description="Дополнительный отступ/расширение подложки текста в пикселях (-24..+48)",
    )
    overflow_mode: str | None = Field(
        default=None,
        description="Индивидуальный режим переполнения для блока: 'shrink', 'overflow' или 'expand_bubble'",
    )
    is_overflowing: bool | None = Field(
        default=None,
        description="True, если текст не поместился в исходную рамку при минимальном размере шрифта",
    )
    mask_dilate: int | None = Field(
        default=None,
        description="Индивидуальная толщина маски очистки для этого блока (1..28 px). Если None — авто",
    )
    custom_mask_strokes: list[dict[str, Any]] | None = Field(
        default=None,
        description="Ручные мазки кисти/ластика замазки: [{'x': int, 'y': int, 'r': int, 'mode': 'add'|'erase'}]",
    )
    translation_source: str | None = Field(
        default=None,
        description="Источник перевода блока (llm, llm_local_bypass, llm_micro_retry, cache, batch_dedup, interjection_dict, skipped_sigh_orig, skipped_sigh_trans, untranslated_error)",
    )
    skip_reason: str | None = Field(
        default=None,
        description="Причина пропуска или отсутствия перевода для данного блока",
    )
    api_censored: bool = Field(
        default=False,
        description="True, если блок был отклонен облачным API из-за цензуры (не отправлять повторно в облако)",
    )
    api_failed: bool = Field(
        default=False,
        description="True, если блок вернулся пустым после доперевода через облачный API",
    )


class TranslateResponse(BaseModel):
    status: str = "ok"
    lang: str
    detected_lang: str
    target_lang: str
    ollama_model: str
    inpaint_backend: str
    image_size: list[int]
    processing_time_sec: float
    blocks_count: int
    free_text_blocks_count: int = 0
    saved_output_path: str | None = None
    debug_image_path: str | None = None
    debug_mask_path: str | None = None
    debug_cleaned_path: str | None = None
    debug_final_path: str | None = None
    debug_json_path: str | None = None
    translated_image: str | None = None
    blocks: list[TextBlock]


class FolderTranslateRequest(BaseModel):
    folder_path: str | None = Field(default=None)
    lang: str = Field(default="auto", description="'auto', 'ja', 'ko' или 'zh'")
    target_lang: str = Field(default="en", description="'en', 'ru' или 'romaji'")
    ollama_model: str | None = Field(default=None)
    model: str | None = Field(default=None)
    uppercase: bool = Field(default=True)
    min_font_size: int = Field(default=12)
    max_font_size: int = Field(default=32)
    overflow_mode: str | None = Field(default=None)
    detect_free_text: bool | None = Field(default=None)
    text_filter_mode: str | None = Field(default=None)
    skip_sighs: bool | None = Field(default=None)
    skip_sfx: bool | None = Field(default=None)
    min_text_length: int | None = Field(default=None)
    skip_existing: bool = Field(default=True)
    force_retranslate: bool = Field(default=False)
    batch_pages: int = Field(default=15, description="Сколько страниц сканировать через OCR и отправлять в ИИ одним пакетом (экономия токенов)")
    translation_profile: str | None = Field(
        default=None,
        description="Профиль перевода: 'eco_book', 'balanced', 'turbo', 'local_only'",
    )
    selected_files: list[str] | None = Field(default=None)
    open_when_done: bool = Field(default=True)


class FolderInspectRequest(BaseModel):
    folder_path: str
    target_lang: str = Field(default="en")


class PageFileTranslateRequest(BaseModel):
    file_path: str = Field(description="Путь к исходному файлу страницы манги на диске")
    source_dir: str | None = Field(default=None)
    output_path: str | None = Field(default=None)
    lang: str = Field(default="auto")
    target_lang: str = Field(default="en")
    ollama_model: str | None = Field(default=None)
    model: str | None = Field(default=None)
    uppercase: bool = Field(default=True)
    min_font_size: int = Field(default=12)
    max_font_size: int = Field(default=32)
    overflow_mode: str | None = Field(default=None)
    detect_free_text: bool | None = Field(default=None)
    text_filter_mode: str | None = Field(default=None)
    skip_sighs: bool | None = Field(default=None)
    skip_sfx: bool | None = Field(default=None)
    min_text_length: int | None = Field(default=None)
    force_retranslate: bool = Field(
        default=False,
        description="Принудительно перевести все блоки страницы заново через ИИ (без старого кэша)",
    )
    return_image: bool = Field(default=True)
    image_format: str = Field(default="webp")


class ServerSettingsRequest(BaseModel):
    unsloth_url: str | None = Field(default=None)
    unsloth_api_key: str | None = Field(default=None)
    selected_model: str | None = Field(default=None)
    translation_profile: str | None = Field(default=None)
    local_fallback_model_multilang: str | None = Field(default=None)
    auto_save_web_pages: bool | None = Field(default=None)
    web_save_dir: str | None = Field(default=None)
    text_filter_mode: str | None = Field(default=None)
    skip_sighs: bool | None = Field(default=None)
    skip_sfx: bool | None = Field(default=None)
    detect_free_text: bool | None = Field(default=None)
    min_text_length: int | None = Field(default=None)
    min_font_size: int | None = Field(default=None)
    max_font_size: int | None = Field(default=None)
    uppercase: bool | None = Field(default=None)
    overflow_mode: str | None = Field(default=None)
    batch_pages: int | None = Field(default=None)
    ui_settings: dict[str, Any] | None = Field(default=None)


class PullModelRequest(BaseModel):
    model: str = Field(
        default="hf.co/mradermacher/shisa-v2.1-unphi4-14b-GGUF:Q4_K_M",
        description="Тег модели в Ollama или GGUF-репозиторий hf.co/...",
    )
    set_as_default: bool = Field(default=True)


class PageBlocksRequest(BaseModel):
    source_path: str = Field(description="Путь к исходному файлу страницы манги")
    source_dir: str | None = Field(default=None)
    dest_path: str | None = Field(default=None)
    lang: str = Field(default="auto")
    target_lang: str = Field(default="en")
    detect_free_text: bool | None = Field(default=None)
    text_filter_mode: str | None = Field(default=None)
    skip_sighs: bool | None = Field(default=None)
    skip_sfx: bool | None = Field(default=None)
    min_text_length: int | None = Field(default=None)
    force_rescan: bool = Field(default=False, description="Принудительно запустить OCR заново")
    use_llm_romaji: bool = Field(default=False, description="Уточнить чтение кандзи в Ромадзи через активную ИИ-модель")
    ollama_model: str | None = Field(default=None)


class PageRomajiRequest(BaseModel):
    source_path: str | None = Field(default=None)
    source_dir: str | None = Field(default=None)
    target_lang: str = Field(default="en")
    use_llm: bool = Field(default=True, description="Использовать ИИ для контекстного чтения кандзи в Ромадзи")
    ollama_model: str | None = Field(default=None)
    blocks: list[TextBlock]


class SaveEditedPageRequest(BaseModel):
    source_path: str = Field(description="Путь к исходному файлу страницы манги")
    dest_path: str | None = Field(default=None, description="Путь сохранения перерисованной страницы")
    source_dir: str | None = Field(default=None)
    target_lang: str = Field(default="en")
    uppercase: bool = Field(default=True)
    min_font_size: int = Field(default=12)
    max_font_size: int = Field(default=32)
    overflow_mode: str | None = Field(default=None)
    overflow_mode: str | None = Field(default=None)
    return_image: bool = Field(default=True)
    prefer_url: bool = Field(
        default=True,
        description="Если True и dest_path существует, возвращает легковесный URL /local_image вместо огромного Base64",
    )
    image_format: str = Field(default="webp")
    blocks: list[TextBlock]


class ManualRegionTranslateRequest(BaseModel):
    source_path: str = Field(description="Путь к исходному файлу страницы манги")
    dest_path: str | None = Field(default=None)
    source_dir: str | None = Field(default=None)
    bbox: list[int] = Field(description="Координаты обведенной области [x1, y1, x2, y2]")
    lang: str = Field(default="auto")
    target_lang: str = Field(default="en")
    ollama_model: str | None = Field(default=None)
    use_vlm_ocr: bool = Field(
        default=False,
        description="Принудительно использовать снайперский VLM-OCR (Vision-модель) для чтения сложного рукописного текста/SFX/обложки",
    )
    vlm_model: str | None = Field(default=None, description="Явное имя Vision-модели для VLM-OCR (опционально)")
    uppercase: bool = Field(default=True)
    min_font_size: int = Field(default=12)
    max_font_size: int = Field(default=32)
    overflow_mode: str | None = Field(default=None)
    overflow_mode: str | None = Field(default=None)
    return_image: bool = Field(default=True)
    image_format: str = Field(default="webp")
    custom_original_text: str | None = Field(default=None)
    custom_translated_text: str | None = Field(default=None)
    blocks: list[TextBlock] = Field(default_factory=list)


class RetranslateSingleBlockRequest(BaseModel):
    source_path: str = Field(description="Путь к исходному файлу страницы манги")
    dest_path: str | None = Field(default=None)
    source_dir: str | None = Field(default=None)
    block_id: int = Field(description="ID блока, который нужно принудительно перевести через ИИ")
    lang: str = Field(default="auto")
    target_lang: str = Field(default="en")
    ollama_model: str | None = Field(default=None)
    use_vlm_ocr: bool = Field(
        default=False,
        description="Если True, заново распознает оригинал блока через снайперский VLM-OCR (Vision-модель) перед переводом",
    )
    vlm_model: str | None = Field(default=None, description="Явное имя Vision-модели для VLM-OCR (опционально)")
    uppercase: bool = Field(default=True)
    min_font_size: int = Field(default=12)
    max_font_size: int = Field(default=32)
    overflow_mode: str | None = Field(default=None)
    overflow_mode: str | None = Field(default=None)
    return_image: bool = Field(default=True)
    image_format: str = Field(default="webp")
    blocks: list[TextBlock] = Field(default_factory=list)


def _natural_sort_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path.relative_to(path.anchor)))
    ]


def _normalize_rel_path(rel: Path | str) -> str:
    return str(rel).replace("\\", "/").strip("/")


def _unwrap_output_folder_if_needed(folder: Path) -> Path:
    """Если пользователь случайно выбрал уже переведенную папку `<Имя>_RU`, возвращает чистую исходную `<Имя>`, если она есть рядом."""
    name = folder.name
    for sfx in OUTPUT_FOLDER_SUFFIXES:
        if name.upper().endswith(sfx.upper()) and len(name) > len(sfx):
            orig_candidate = folder.parent / name[: -len(sfx)]
            if orig_candidate.exists() and orig_candidate.is_dir():
                return orig_candidate
    return folder


def get_output_dir_for_folder(source_dir: Path, target_lang: str) -> Path:
    clean_src = _unwrap_output_folder_if_needed(source_dir)
    return clean_src.parent / f"{clean_src.name}_{target_lang.upper().strip()}"


def collect_images_in_folder(folder: Path) -> list[Path]:
    images = [
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        and p.name != "mascot.png"
        and not p.parent.name.endswith(OUTPUT_FOLDER_SUFFIXES)
    ]
    images.sort(key=_natural_sort_key)
    return images


def _load_folder_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("pages", {})
                data.setdefault("rolling_context", [])
                data.setdefault("phrase_cache", {})
                data.setdefault("glossary", [])
                data.setdefault("run_history", [])
                data.setdefault("batches_log", [])
                return data
        except Exception as exc:
            logger.warning("Не удалось прочитать %s: %s", manifest_path, exc)
    return {
        "pages": {},
        "rolling_context": [],
        "phrase_cache": {},
        "glossary": [],
        "run_history": [],
        "batches_log": [],
    }


_manifest_write_lock = threading.Lock()


def _save_folder_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    try:
        with _manifest_write_lock:
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = output_dir / MANIFEST_FILENAME
            tmp_path = output_dir / f"{MANIFEST_FILENAME}.tmp"
            tmp_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, manifest_path)
    except Exception as exc:
        logger.warning("Не удалось сохранить %s: %s", MANIFEST_FILENAME, exc)


CHAPTER_BOOK_MD_FILENAME = "_chapter_book.md"
CHAPTER_SCRIPT_JSON_FILENAME = "_chapter_script_bilingual.json"
CHAPTER_SCRIPT_TXT_FILENAME = "_chapter_script_bilingual.txt"


def _sort_page_blocks_reading_order(blocks: list[dict[str, Any]], lang: str = "ja") -> list[dict[str, Any]]:
    """
    БЛОК 2.1: Упорядочивает текстовые блоки страницы в естественном порядке чтения комикса:
    - Для японской манги (ja): справа налево, сверху вниз (rtl=True).
    - Для китайской маньхуа (zh) и корейской манхвы (ko): слева направо, сверху вниз (rtl=False).
    """
    if not blocks:
        return []
    rtl = (lang or "ja").lower().strip() == "ja"
    valid_boxes: list[dict[str, Any]] = []
    for b in blocks:
        bbox = b.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            valid_boxes.append(b)
        else:
            b_copy = dict(b)
            b_copy["bbox"] = [0, 0, 100, 100]
            valid_boxes.append(b_copy)
    ordered = MangaOCREngine._sort_boxes_reading_order(valid_boxes, rtl=rtl)
    for idx_b, blk in enumerate(ordered, start=1):
        blk["id"] = idx_b
    return ordered


def _build_smart_page_batches(
    items: list[Any],
    pages_per_batch: int,
    tail_merge_pages: int,
) -> list[list[Any]]:
    """
    БЛОК 1: Разбивает список страниц на пачки по `pages_per_batch` с умным слиянием коротких хвостов:
    если в последней пачке остается `<= tail_merge_pages` страниц, они присоединяются к предыдущей пачке
    (например, при 25 стр. и пачке 15 стр. получается 1 пачка 15 + 10 = 25 или 2 сбалансированные пачки 12 + 13,
    без микро-пачек по 1 странице!).
    """
    if not items:
        return []
    n = len(items)
    p_batch = max(1, int(pages_per_batch or 15))
    t_merge = max(0, int(tail_merge_pages or 0))
    if n <= p_batch:
        return [items]
    # Если для режима balanced (p_batch=15, t_merge=5) 25 страниц = 15 + 10 (или если 16..20 стр., делим пополам 10+10 / 12+13)
    raw_batches = [items[i : i + p_batch] for i in range(0, n, p_batch)]
    if len(raw_batches) >= 2 and len(raw_batches[-1]) <= t_merge:
        tail = raw_batches.pop()
        combined_last = raw_batches[-1] + tail
        # Если объединенная пачка стала слишком большой (> p_batch + 3) и всего было 2 пачки, делим ровно пополам (например, 25 -> 12 + 13)
        if len(raw_batches) == 1 and len(combined_last) > p_batch:
            mid = len(combined_last) // 2
            raw_batches = [combined_last[:mid], combined_last[mid:]]
        else:
            raw_batches[-1] = combined_last
    return raw_batches


def _generate_and_save_chapter_book_files(
    output_dir: Path,
    manifest: dict[str, Any] | None = None,
    source_dir: Path | None = None,
    target_lang: str | None = None,
) -> dict[str, Any]:
    """
    БЛОК 2.2: Автоматически формирует и сохраняет в папку перевода файлы «Книги перевода»:
    1. `_chapter_book.md` — чистая книга для чтения и ИИ (`## Страница 001 (filename)`, далее реплики по порядку чтения).
    2. `_chapter_script_bilingual.json` и `_chapter_script_bilingual.txt` — двуязычный скрипт с привязкой к бабблам
       (`page_name`, `page_index`, `block_id`, `book_key`, `bbox`, `original_text`, `translated_text`).
    """
    try:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    if manifest is None:
        manifest = _load_folder_manifest(output_dir)

    if source_dir is None:
        saved_src = manifest.get("source_dir")
        if saved_src and Path(str(saved_src)).exists():
            source_dir = Path(str(saved_src)).resolve()
        else:
            source_dir = _unwrap_output_folder_if_needed(output_dir)
    else:
        source_dir = _unwrap_output_folder_if_needed(Path(source_dir).resolve())

    t_lang = (target_lang or manifest.get("target_lang") or "en").lower().strip()
    pages_meta: dict[str, Any] = manifest.get("pages", {}) if isinstance(manifest.get("pages"), dict) else {}

    all_src_images: list[Path] = []
    if source_dir and source_dir.exists() and source_dir.is_dir():
        try:
            all_src_images = collect_images_in_folder(source_dir)
        except Exception:
            all_src_images = []

    rel_paths_ordered: list[str] = []
    if all_src_images:
        for img_p in all_src_images:
            try:
                rel_paths_ordered.append(_normalize_rel_path(img_p.relative_to(source_dir)))
            except Exception:
                rel_paths_ordered.append(img_p.name)
    for k_rel in sorted(pages_meta.keys()):
        if k_rel not in rel_paths_ordered:
            rel_paths_ordered.append(k_rel)

    chapter_title = source_dir.name if source_dir else output_dir.name
    md_lines: list[str] = [
        f"# 📖 Книга главы: {chapter_title} ({t_lang.upper()})",
        "",
    ]
    bilingual_entries: list[dict[str, Any]] = []
    bilingual_txt_lines: list[str] = [
        f"# Двуязычный сценарий главы: {chapter_title} ({t_lang.upper()})",
        "# Формат: [Pxx_yy] (page_name | #block_id) Оригинал => Перевод",
        "",
    ]

    for p_idx, rel_norm in enumerate(rel_paths_ordered, start=1):
        p_meta = pages_meta.get(rel_norm, {}) if isinstance(pages_meta.get(rel_norm), dict) else {}
        p_blocks = p_meta.get("blocks", []) if isinstance(p_meta.get("blocks"), list) else []
        md_lines.append(f"## Страница {p_idx:03d} ({rel_norm})")
        active_lines_on_page = 0
        for b_idx, b in enumerate(p_blocks, start=1):
            if not isinstance(b, dict):
                continue
            bid = int(b.get("id") or b_idx)
            orig_t = str(b.get("original_text") or "").strip()
            trans_t = str(b.get("translated_text") or "").strip()
            is_del = bool(b.get("is_deleted", False))
            if is_del and not trans_t:
                continue
            if not orig_t and not trans_t:
                continue
            book_key = f"P{p_idx:02d}_{bid:02d}"
            display_line = trans_t if trans_t else orig_t
            if display_line:
                md_lines.append(f"- **[{book_key}]** {display_line}")
                active_lines_on_page += 1
            entry = {
                "book_key": book_key,
                "page_index": p_idx,
                "page_name": rel_norm,
                "block_id": bid,
                "bbox": list(b.get("bbox") or [0, 0, 0, 0]),
                "original_text": orig_t,
                "translated_text": trans_t,
            }
            bilingual_entries.append(entry)
            bilingual_txt_lines.append(
                f"[{book_key}] ({rel_norm} | #{bid}) {orig_t} => {trans_t}"
            )
        if active_lines_on_page == 0:
            md_lines.append("*(Без диалогов)*")
        md_lines.append("")

    book_md_text = "\n".join(md_lines).strip() + "\n"
    bilingual_txt_text = "\n".join(bilingual_txt_lines).strip() + "\n"
    bilingual_json_payload = {
        "chapter_title": chapter_title,
        "source_dir": str(source_dir) if source_dir else "",
        "output_dir": str(output_dir),
        "target_lang": t_lang,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_pages": len(rel_paths_ordered),
        "total_blocks": len(bilingual_entries),
        "entries": bilingual_entries,
    }

    md_path = output_dir / CHAPTER_BOOK_MD_FILENAME
    json_path = output_dir / CHAPTER_SCRIPT_JSON_FILENAME
    txt_path = output_dir / CHAPTER_SCRIPT_TXT_FILENAME
    try:
        md_path.write_text(book_md_text, encoding="utf-8")
        json_path.write_text(json.dumps(bilingual_json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        txt_path.write_text(bilingual_txt_text, encoding="utf-8")
    except Exception as exc:
        logger.warning("Не удалось сохранить файлы Книги главы в %s: %s", output_dir, exc)

    return {
        "book_md_path": str(md_path),
        "script_json_path": str(json_path),
        "script_txt_path": str(txt_path),
        "book_md": book_md_text,
        "bilingual_txt": bilingual_txt_text,
        "bilingual_json": bilingual_json_payload,
    }


def _generate_and_save_folder_report(
    output_dir: Path,
    manifest: dict[str, Any] | None = None,
    source_dir: Path | None = None,
    target_lang: str | None = None,
    run_status_label: str = "",
) -> str:
    """
    Формирует и сохраняет в папку переведенной манги (`<output_dir>/TRANSLATION_REPORT.txt`)
    подробный диагностический отчет и бенчмарк-журнал перевода главы:
    1. Паспорт главы и полный стек использованных нейросетей (Детектор, OCR JA/KO/ZH, VLM-OCR,
       основная и резервная ИИ-модель, движок очистки LaMa/OpenCV, GPU/VRAM).
    2. Активные параметры фильтрации и верстки (для удобного сравнения между запусками).
    3. Сводная воронка страниц и текстовых блоков (сколько найдено, отсеяно фильтром, переведено ИИ,
       перехвачено локальной моделью, взято из кэша, пропущено как вздох или не переведено из-за ошибки).
    4. Таблица истории запусков (бенчмарк разных моделей и параметров на этой папке).
    5. Журнал работы облачной и локальной ИИ-модели (отказы цензуры content_filter, сетевые ретраи,
       утечки Ромадзи/CJK и микро-допереводы).
    6. Детальный разбор пропусков: какие страницы остались без перевода (0 блоков) и почему,
       какие отдельные блоки не были переведены или были отсеяны фильтром (с оригинальным текстом).
    7. Чек-лист контроля качества верстки (тесные бабблы с минимальным кеглем, низкий confidence)
       и автоматические рекомендации по улучшению параметров.
    8. Постраничная сводная таблица по всем страницам папки и хронология пачек.
    """
    try:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    if manifest is None:
        manifest = _load_folder_manifest(output_dir)

    if source_dir is None:
        saved_src = manifest.get("source_dir")
        if saved_src and Path(str(saved_src)).exists():
            source_dir = Path(str(saved_src)).resolve()
        else:
            source_dir = _unwrap_output_folder_if_needed(output_dir)
    else:
        source_dir = _unwrap_output_folder_if_needed(Path(source_dir).resolve())

    t_lang = (target_lang or manifest.get("target_lang") or "ru").lower().strip()
    if output_dir.name.upper().endswith("_RU"):
        t_lang = "ru"
    elif output_dir.name.upper().endswith("_EN"):
        t_lang = "en"
    elif output_dir.name.upper().endswith("_ROMAJI"):
        t_lang = "romaji"

    pages_meta: dict[str, Any] = manifest.get("pages", {}) if isinstance(manifest.get("pages"), dict) else {}
    run_history: list[dict[str, Any]] = manifest.get("run_history", []) if isinstance(manifest.get("run_history"), list) else []
    batches_log: list[dict[str, Any]] = manifest.get("batches_log", []) if isinstance(manifest.get("batches_log"), list) else []
    folder_glossary = normalize_glossary_entries(manifest.get("glossary", []))
    global_glossary = normalize_glossary_entries(translator_engine.glossary if translator_engine else [])
    merged_glossary = merge_glossaries(global_glossary, folder_glossary)

    all_src_images: list[Path] = []
    if source_dir and source_dir.exists() and source_dir.is_dir():
        try:
            all_src_images = collect_images_in_folder(source_dir)
        except Exception:
            all_src_images = []

    rel_paths_ordered: list[str] = []
    if all_src_images:
        for img_p in all_src_images:
            try:
                rel_paths_ordered.append(_normalize_rel_path(img_p.relative_to(source_dir)))
            except Exception:
                rel_paths_ordered.append(img_p.name)
    for k_rel in sorted(pages_meta.keys()):
        if k_rel not in rel_paths_ordered:
            rel_paths_ordered.append(k_rel)

    total_pages_in_folder = len(rel_paths_ordered)
    translated_pages_cnt = 0
    pages_with_blocks_cnt = 0
    zero_block_pages: list[dict[str, Any]] = []
    error_pages: list[dict[str, Any]] = []

    tot_raw_candidates = 0
    tot_yolo_bubbles = 0
    tot_rtdetr_bubbles = 0
    tot_rtdetr_text_bubbles = 0
    tot_split_multi = 0
    tot_rescued_orphan = 0
    tot_no_ink = 0
    tot_hallucinated = 0
    tot_empty_ocr = 0
    tot_filtered_ocr = 0
    tot_restored_empty_guard = 0
    filter_reasons_counter: dict[str, int] = {
        "sigh": 0,
        "sfx": 0,
        "min_length": 0,
        "free_text_disabled": 0,
        "empty_punct": 0,
    }
    all_filtered_ocr_records: list[dict[str, Any]] = []

    tot_kept_blocks = 0
    tot_bubble_blocks = 0
    tot_free_blocks = 0
    tot_translated_ok = 0
    source_counter: dict[str, int] = {
        "llm": 0,
        "llm_local_bypass": 0,
        "llm_sanitized_retry": 0,
        "llm_micro_retry": 0,
        "cache": 0,
        "batch_dedup": 0,
        "interjection_dict": 0,
        "romaji_rescue_dict": 0,
        "romaji_local": 0,
        "manual_edit": 0,
        "skipped_sigh_orig": 0,
        "skipped_sigh_trans": 0,
        "deleted_by_user": 0,
        "untranslated_error": 0,
    }

    untranslated_error_blocks: list[dict[str, Any]] = []
    skipped_post_ocr_blocks: list[dict[str, Any]] = []
    cramped_min_font_blocks: list[dict[str, Any]] = []
    low_conf_blocks: list[dict[str, Any]] = []
    per_page_rows: list[dict[str, Any]] = []
    detected_langs_freq: dict[str, int] = {}
    models_used_set: list[str] = []

    default_min_fs = int(translator_engine.min_font_size if translator_engine else 12)

    for idx_p, rel_norm in enumerate(rel_paths_ordered, start=1):
        p_meta = pages_meta.get(rel_norm, {}) if isinstance(pages_meta.get(rel_norm), dict) else {}
        dest_p = output_dir / rel_norm
        is_done = is_page_already_translated(dest_p, page_meta=p_meta if p_meta else None, target_lang=t_lang)
        if is_done:
            translated_pages_cnt += 1

        p_det_lang = str(p_meta.get("detected_lang") or "ja")
        if p_det_lang:
            detected_langs_freq[p_det_lang] = detected_langs_freq.get(p_det_lang, 0) + 1
        p_model = str(p_meta.get("model") or "").strip()
        if p_model and p_model not in models_used_set:
            models_used_set.append(p_model)

        p_min_fs = int(p_meta.get("min_font_size") or default_min_fs)
        ocr_diag = p_meta.get("ocr_diag", {}) if isinstance(p_meta.get("ocr_diag"), dict) else {}
        ocr_err = str(p_meta.get("ocr_error") or "").strip()
        if ocr_err:
            error_pages.append({"index": idx_p, "rel_path": rel_norm, "error": ocr_err})

        p_blocks = p_meta.get("blocks", []) if isinstance(p_meta.get("blocks"), list) else []
        raw_cand = int(ocr_diag.get("raw_candidates_count", len(p_blocks)))
        tot_raw_candidates += raw_cand
        tot_yolo_bubbles += int(ocr_diag.get("yolo_bubbles", 0))
        tot_rtdetr_bubbles += int(ocr_diag.get("rtdetr_bubbles", 0))
        tot_rtdetr_text_bubbles += int(ocr_diag.get("rtdetr_text_bubbles", 0))
        tot_split_multi += int(ocr_diag.get("split_multi_bubbles", 0))
        tot_rescued_orphan += int(ocr_diag.get("rescued_orphan_text_bubbles", 0))
        tot_no_ink += int(ocr_diag.get("no_ink_count", 0))
        tot_hallucinated += int(ocr_diag.get("hallucinated_count", 0))
        tot_empty_ocr += int(ocr_diag.get("empty_ocr_count", 0))
        tot_restored_empty_guard += int(ocr_diag.get("restored_by_empty_guard_count", 0))

        filt_items = ocr_diag.get("filtered_items", []) if isinstance(ocr_diag.get("filtered_items"), list) else []
        p_filt_cnt = int(ocr_diag.get("filtered_count", len(filt_items)))
        tot_filtered_ocr += p_filt_cnt
        for fi in filt_items:
            if not isinstance(fi, dict):
                continue
            rc = str(fi.get("reason_code") or "sigh")
            filter_reasons_counter[rc] = filter_reasons_counter.get(rc, 0) + 1
            all_filtered_ocr_records.append(
                {
                    "page_index": idx_p,
                    "rel_path": rel_norm,
                    "bbox": fi.get("bbox"),
                    "text": str(fi.get("text") or ""),
                    "reason_code": rc,
                    "reason_ru": str(fi.get("reason_ru") or rc),
                }
            )

        p_kept = len(p_blocks)
        tot_kept_blocks += p_kept
        p_trans_ok = 0
        p_skipped_sigh = 0
        p_untranslated_err = 0
        p_fonts: list[int] = []
        p_sources_short: list[str] = []

        for b in p_blocks:
            if not isinstance(b, dict):
                continue
            bid = int(b.get("id") or 1)
            is_free = bool(b.get("is_free_text", False))
            if is_free:
                tot_free_blocks += 1
            else:
                tot_bubble_blocks += 1

            orig_t = str(b.get("original_text") or "").strip()
            trans_t = str(b.get("translated_text") or "").strip()
            is_del = bool(b.get("is_deleted", False))
            tsrc = str(b.get("translation_source") or "").strip()
            s_reason = str(b.get("skip_reason") or "").strip()
            conf_val = float(b.get("confidence") or 1.0)
            r_fs = b.get("font_size") or b.get("rendered_font_size")

            if conf_val < 0.35:
                low_conf_blocks.append(
                    {
                        "page_index": idx_p,
                        "rel_path": rel_norm,
                        "block_id": bid,
                        "confidence": round(conf_val, 2),
                        "original_text": orig_t,
                        "translated_text": trans_t,
                    }
                )

            if trans_t and not is_del:
                p_trans_ok += 1
                tot_translated_ok += 1
                if not tsrc or tsrc in ("untranslated_error", "skipped_sigh_orig", "skipped_sigh_trans", "deleted_by_user"):
                    tsrc = "llm"
                source_counter[tsrc] = source_counter.get(tsrc, 0) + 1
                if tsrc not in p_sources_short:
                    p_sources_short.append(tsrc)
                if r_fs is not None:
                    try:
                        fs_int = int(r_fs)
                        p_fonts.append(fs_int)
                        if fs_int <= p_min_fs and len(trans_t) >= 18:
                            cramped_min_font_blocks.append(
                                {
                                    "page_index": idx_p,
                                    "rel_path": rel_norm,
                                    "block_id": bid,
                                    "font_size": fs_int,
                                    "min_font_size": p_min_fs,
                                    "bbox": b.get("bbox"),
                                    "original_text": orig_t,
                                    "translated_text": trans_t,
                                }
                            )
                    except (ValueError, TypeError):
                        pass
            else:
                if is_del and tsrc not in ("skipped_sigh_orig", "skipped_sigh_trans"):
                    tsrc = tsrc if tsrc in ("skipped_sigh_orig", "skipped_sigh_trans") else "deleted_by_user"
                if not tsrc:
                    if ocr_engine and ocr_engine.is_sigh_or_interjection(orig_t):
                        tsrc = "skipped_sigh_orig"
                        s_reason = f"Короткий вздох «{orig_t}» (skip_sighs=True)"
                    else:
                        tsrc = "untranslated_error"
                        s_reason = "Пустой перевод (пропуск ИИ или обрыв генерации)"

                source_counter[tsrc] = source_counter.get(tsrc, 0) + 1
                if tsrc in ("skipped_sigh_orig", "skipped_sigh_trans", "deleted_by_user"):
                    p_skipped_sigh += 1
                    skipped_post_ocr_blocks.append(
                        {
                            "page_index": idx_p,
                            "rel_path": rel_norm,
                            "block_id": bid,
                            "bbox": b.get("bbox"),
                            "original_text": orig_t,
                            "source": tsrc,
                            "reason": s_reason or "Намеренно пропущен (вздох / удален пользователем)",
                        }
                    )
                else:
                    p_untranslated_err += 1
                    untranslated_error_blocks.append(
                        {
                            "page_index": idx_p,
                            "rel_path": rel_norm,
                            "block_id": bid,
                            "bbox": b.get("bbox"),
                            "original_text": orig_t,
                            "reason": s_reason or "Модель не вернула валидный перевод",
                        }
                    )

        if p_trans_ok > 0:
            pages_with_blocks_cnt += 1
        else:
            z_reason = str(ocr_diag.get("zero_reason_ru") or "").strip()
            if ocr_err:
                z_reason = f"Ошибка OCR на странице: {ocr_err}"
            elif not p_meta and not is_done:
                z_reason = "Страница еще не обрабатывалась переводчиком (ожидает очереди)"
            elif not z_reason:
                if p_kept > 0 and p_skipped_sigh == p_kept:
                    z_reason = f"Все распознанные реплики ({p_kept} шт.) оказались короткими вздохами и были скрыты (skip_sighs=True)"
                elif p_kept > 0 and p_untranslated_err > 0:
                    z_reason = f"Найдено {p_kept} блоков, но ИИ не вернул для них перевод (см. Раздел 6Б)"
                elif p_filt_cnt > 0:
                    z_reason = f"Весь найденный текст ({p_filt_cnt} шт.) отфильтрован как вздохи/SFX"
                else:
                    z_reason = "На странице не обнаружено речевых облачков с текстом (обложка, иллюстрация или страница без слов)"
            zero_block_pages.append(
                {
                    "index": idx_p,
                    "rel_path": rel_norm,
                    "is_translated": is_done,
                    "raw_candidates": raw_cand,
                    "filtered_count": p_filt_cnt,
                    "kept_blocks": p_kept,
                    "reason": z_reason,
                }
            )

        if ocr_err:
            status_cell = "❌ ОШИБКА"
        elif not p_meta and not is_done:
            status_cell = "⏳ ЖДЕТ"
        elif p_untranslated_err > 0:
            status_cell = f"⚠️ ПРОПУСК ({p_untranslated_err})"
        elif p_trans_ok == 0:
            status_cell = "⚪ 0 БЛОКОВ"
        else:
            status_cell = "✅ ОК"

        fs_cell = f"{min(p_fonts)}..{max(p_fonts)}" if p_fonts else "—"
        note_parts: list[str] = []
        if "llm_local_bypass" in p_sources_short:
            note_parts.append("🛡Обход цензуры (Ollama)")
        if "llm_micro_retry" in p_sources_short:
            note_parts.append("🔄Доперевод 2-й проход")
        if p_filt_cnt > 0:
            note_parts.append(f"Фильтр отсеял: {p_filt_cnt}")
        if p_skipped_sigh > 0:
            note_parts.append(f"Вздохи/удал.: {p_skipped_sigh}")
        if p_trans_ok == 0 and zero_block_pages:
            note_parts.append(zero_block_pages[-1]["reason"][:58])

        per_page_rows.append(
            {
                "index": idx_p,
                "rel_path": rel_norm,
                "status": status_cell,
                "lang": p_det_lang.upper(),
                "raw_cand": raw_cand,
                "filtered": p_filt_cnt,
                "kept": p_kept,
                "trans_ok": p_trans_ok,
                "untrans_err": p_untranslated_err,
                "fs_range": fs_cell,
                "note": "; ".join(note_parts) if note_parts else "Штатно",
            }
        )

    d_payload = _build_diagnostics_payload()
    gpu_info = d_payload.get("gpu", {})
    tr_info = d_payload.get("translator", {})
    srv_info = d_payload.get("server", {})

    dominant_lang = (
        max(detected_langs_freq.items(), key=lambda kv: kv[1])[0]
        if detected_langs_freq
        else str(folder_job_state.get("detected_lang") or "ja")
    )
    lang_names_full = {
        "ja": "JA (Японская манга)",
        "ko": "KO (Корейская манхва / вебтун)",
        "zh": "ZH (Китайская маньхуа)",
    }
    target_names_full = {
        "ru": "RU (Русский язык)",
        "en": "EN (Английский язык)",
        "romaji": "ROMAJI (Транслитерация Хэпбёрн)",
    }

    active_llm = (
        models_used_set[-1]
        if models_used_set
        else str(tr_info.get("selected_model") or "auto")
    )
    local_fb_llm = str(
        tr_info.get("last_local_fallback_model")
        or (translator_engine.default_model if translator_engine else "shisa-v2.1-14b")
    )
    report_path = output_dir / REPORT_FILENAME

    eff_free, eff_mode, eff_skip_sighs, eff_skip_sfx, eff_min_len = _resolve_filter_params()
    cur_min_fs = int(translator_engine.min_font_size if translator_engine else 12)
    cur_max_fs = int(translator_engine.max_font_size if translator_engine else 32)
    cur_upper = bool(translator_engine.uppercase if translator_engine else True)
    cur_overflow = str(getattr(translator_engine, "overflow_mode", "shrink") if translator_engine else "shrink")
    cur_batch_p = int(translator_engine.batch_pages if translator_engine else 12)

    last_run = run_history[-1] if run_history else {}
    status_header = (
        run_status_label
        or str(last_run.get("status_label") or "")
        or f"Готово {translated_pages_cnt} из {total_pages_in_folder} стр."
    )

    overflow_names_ru = {
        "shrink": "Стандарт (внутри рамки до мин. шрифта)",
        "overflow": "Вариант Б — Выход текста за рамки баббла (с обводкой)",
        "expand_bubble": "Вариант В — Авто-перерисовка (расширение) баббла под текст",
    }
    lines: list[str] = [
        "========================================================================================",
        "  📑 ДИАГНОСТИЧЕСКИЙ ОТЧЕТ И ЖУРНАЛ ПЕРЕВОДА ГЛАВЫ — MIKO-CHAN MANGA STUDIO v1.0.7",
        "========================================================================================",
        f"• Исходная папка манги     : {source_dir}",
        f"• Папка готового перевода  : {output_dir}",
        f"• Файл этого отчета        : {report_path}",
        f"• Дата обновления отчета   : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"• Статус перевода главы    : {status_header}",
        "",
        "----------------------------------------------------------------------------------------",
        "  🤖 1. ЗАДЕЙСТВОВАННЫЕ МОДЕЛИ, НЕЙРОСЕТИ И АППАРАТНОЕ УСКОРЕНИЕ",
        "----------------------------------------------------------------------------------------",
        f"• Основная ИИ-модель (LLM)     : {active_llm}",
        f"• Шлюз облачного API           : {tr_info.get('api_url', '—')} (ключ: {tr_info.get('masked_api_key') or 'локальный режим'})",
        f"• Локальная модель (фоллбэк)   : {local_fb_llm} (авто-обход цензуры API и сбоев сети)",
        f"• Детектор бабблов и рамок     : {gpu_info.get('detector_backend', 'Hybrid RT-DETR-v2 + YOLOv8m')}",
        f"• OCR (Японский JA)            : {gpu_info.get('ja_ocr_backend', 'Hybrid manga-ocr-nar-small + MangaOCR FP16')}",
        f"• OCR (Корейский KO / Кит. ZH) : {gpu_info.get('ko_zh_ocr_backend', 'HayaiOCR v2.5 Nova + SmartLineSplit')}",
        f"• Снайперский VLM-OCR (SFX)    : {gpu_info.get('vlm_ocr_backend', 'Sniper VLM-OCR')}",
        f"• Нейросетевая очистка (Inpaint): {gpu_info.get('inpaint_backend', 'SimpleLama GPU')} + CleanCache (0 мс при правках)",
        f"• Видеокарта и окружение       : {gpu_info.get('device_name', 'CPU')} (VRAM: {gpu_info.get('vram_allocated_mb', 0)} / {gpu_info.get('vram_total_mb', 0)} МБ) | PyTorch {gpu_info.get('torch_version')} | Python {srv_info.get('python_version')}",
        "",
        "----------------------------------------------------------------------------------------",
        "  ⚙️ 2. ПАРАМЕТРЫ ПЕРЕВОДА, ФИЛЬТРАЦИИ И ВЕРСТКИ (ДЛЯ СРАВНЕНИЯ ЗАПУСКОВ)",
        "----------------------------------------------------------------------------------------",
        f"• Направление перевода         : {lang_names_full.get(dominant_lang, dominant_lang)} ➔ {target_names_full.get(t_lang, t_lang.upper())}",
        f"• Профиль фильтрации текста    : {eff_mode}",
        f"• Пропуск коротких вздохов     : {'ВКЛ (skip_sighs=True)' if eff_skip_sighs else 'ВЫКЛ (переводить все вздохи)'}",
        f"• Пропуск звуковых эффектов    : {'ВКЛ (skip_sfx=True)' if eff_skip_sfx else 'ВЫКЛ (переводить SFX)'}",
        f"• Текст вне бабблов (на фоне)  : {'ВКЛ (detect_free_text=True)' if eff_free else 'ВЫКЛ (защита рисунка фона от замазки)'}",
        f"• Мин. длина текста для OCR    : {eff_min_len} симв. (min_text_length={eff_min_len})",
        f"• Размер шрифта и регистр      : {cur_min_fs} px .. {cur_max_fs} px | Заглавные буквы (UPPERCASE): {'ВКЛ' if cur_upper else 'ВЫКЛ'}",
        f"• Режим узких бабблов          : {overflow_names_ru.get(cur_overflow, cur_overflow)} (overflow_mode={cur_overflow})",
        f"• Пакетный режим (batch_pages) : {cur_batch_p} стр. в пачке | Лимит облака: {tr_info.get('cloud_batch_limit', 260)} фраз (параллельный чанк: {tr_info.get('cloud_parallel_chunk_size', 42)} фраз)",
        f"• Активный глоссарий           : {len(merged_glossary)} терминов (Глобальный: {len(global_glossary)}, Локальный главы: {len(folder_glossary)})",
    ]
    if merged_glossary:
        g_preview = ", ".join(f"{g['src']}->{g['dst']}" for g in merged_glossary[:10])
        if len(merged_glossary) > 10:
            g_preview += f" ... (+еще {len(merged_glossary) - 10})"
        lines.append(f"  └─ Термины глоссария         : {g_preview}")

    pct_ok = round((tot_translated_ok / max(1, tot_kept_blocks)) * 100.0, 1) if tot_kept_blocks > 0 else 100.0
    intentional_skips = (
        source_counter.get("skipped_sigh_orig", 0)
        + source_counter.get("skipped_sigh_trans", 0)
        + source_counter.get("deleted_by_user", 0)
    )
    untranslated_errs_cnt = len(untranslated_error_blocks)

    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            "  📊 3. СВОДНАЯ ВОРОНКА СТРАНИЦ, ТЕКСТОВЫХ БЛОКОВ И ТАЙМИНГОВ ГЛАВЫ",
            "----------------------------------------------------------------------------------------",
            "[Страницы главы]",
            f"  • Всего страниц в папке            : {total_pages_in_folder} стр.",
            f"  • Переведено и сохранено на диск   : {translated_pages_cnt} стр.",
            f"  • Страниц с переведенным текстом   : {pages_with_blocks_cnt} стр.",
            f"  • Страниц без перевода (0 блоков)  : {len(zero_block_pages)} стр. (подробные причины см. в Разделе 6А)",
            f"  • Страниц со сбоем чтения (OCR)    : {len(error_pages)} стр.",
            "",
            "[Сквозная воронка текстовых блоков от детекции до рендера]",
            f"  1) Найдено контуров детекторами    : {tot_raw_candidates} шт. (YOLOv8m: {tot_yolo_bubbles}, RT-DETR-v2 бабблы: {tot_rtdetr_bubbles}, точные text_bbox: {tot_rtdetr_text_bubbles}; разделено сдвоенных: {tot_split_multi}, спасено безрамочных: {tot_rescued_orphan})",
            f"  2) Отброшено пустых / без чернил   : {tot_no_ink + tot_hallucinated + tot_empty_ocr} шт. (нет букв внутри рамки: {tot_no_ink}, отсеяно анти-галлюцинацией/шум: {tot_hallucinated + tot_empty_ocr})",
            f"  3) Отсеяно фильтром настроек OCR   : {tot_filtered_ocr} шт. (вздохи skip_sighs: {filter_reasons_counter.get('sigh', 0)}, спецэффекты skip_sfx: {filter_reasons_counter.get('sfx', 0)}, короткие <{eff_min_len} симв.: {filter_reasons_counter.get('min_length', 0)}, текст на фоне: {filter_reasons_counter.get('free_text_disabled', 0)})",
            f"  4) Возвращено защитой пустых стр.  : {tot_restored_empty_guard} шт. (короткие бабблы сохранены, чтобы страница не осталась без перевода)",
            f"  5) Принято блоков к переводу       : {tot_kept_blocks} шт. (облачка диалогов: {tot_bubble_blocks}, надписи на фоне: {tot_free_blocks})",
            f"  6) ✅ УСПЕШНО ПЕРЕВЕДЕНО НА СТР.   : {tot_translated_ok} из {tot_kept_blocks} шт. ({pct_ok}%)",
            f"       ├─ Основной ИИ-моделью с 1-й попытки    : {source_counter.get('llm', 0)} шт.",
            f"       ├─ Локальной моделью (обход цензуры API): {source_counter.get('llm_local_bypass', 0)} шт.",
            f"       ├─ Доперевод ИИ со 2-й попытки (ретрай) : {source_counter.get('llm_micro_retry', 0) + source_counter.get('llm_sanitized_retry', 0)} шт.",
            f"       ├─ Из кэша фраз и повторов (0 токенов)  : {source_counter.get('cache', 0) + source_counter.get('batch_dedup', 0)} шт.",
            f"       └─ По встроенному словарю междометий    : {source_counter.get('interjection_dict', 0) + source_counter.get('romaji_rescue_dict', 0) + source_counter.get('romaji_local', 0)} шт.",
            f"  7) ⚪ Намеренно оставлено без пер. : {intentional_skips} шт. (вздохи в оригинале: {source_counter.get('skipped_sigh_orig', 0)}, вздохи после перевода: {source_counter.get('skipped_sigh_trans', 0)}, удалено вручную: {source_counter.get('deleted_by_user', 0)})",
            f"  8) ❌ НЕ ПЕРЕВЕДЕНО (сбой / пропуск): {untranslated_errs_cnt} шт.",
        ]
    )

    if last_run:
        r_tot = float(last_run.get("total_sec") or 0.0)
        r_ocr = float(last_run.get("ocr_sec") or 0.0)
        r_llm = float(last_run.get("llm_sec") or 0.0)
        r_ren = float(last_run.get("render_sec") or 0.0)
        r_spp = float(last_run.get("sec_per_page") or 0.0)
        nz = max(0.01, (r_ocr + r_llm + r_ren) if (r_ocr + r_llm + r_ren) > 0 else r_tot)
        lines.extend(
            [
                "",
                "[Тайминги последнего пакетного прогона]",
                f"  • Общее время прогона              : {r_tot:.2f} сек (средняя скорость: {r_spp:.2f} сек/стр.)",
                f"  • Этап 1 — Детекция и OCR (GPU)    : {r_ocr:.2f} сек ({int(round(r_ocr / nz * 100))}%)",
                f"  • Этап 2 — ИИ-перевод (LLM/API)    : {r_llm:.2f} сек ({int(round(r_llm / nz * 100))}%)",
                f"  • Этап 3 — Очистка LaMa и Верстка  : {r_ren:.2f} сек ({int(round(r_ren / nz * 100))}%)",
            ]
        )

    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            "  🏆 4. ИСТОРИЯ ЗАПУСКОВ НА ЭТОЙ ПАПКЕ (СРАВНЕНИЕ МОДЕЛЕЙ И ПАРАМЕТРОВ)",
            "----------------------------------------------------------------------------------------",
        ]
    )
    if not run_history:
        lines.append("  (Записи пакетных прогонов появятся после запуска перевода папки; текущие данные собраны по страницам)")
    else:
        lines.append(
            "  №  | Дата и время        | Модель ИИ                        | Фильтр          | Пакет | Стр. | Перев.бл | Фильтр. | Ошиб.бл | Отказ API | Скор.   | Тайминг (OCR / ИИ / Рендер)"
        )
        lines.append("  " + "-" * 166)
        for r_idx, rh in enumerate(run_history[-15:], start=1):
            m_short = str(rh.get("model") or "auto")
            if len(m_short) > 32:
                m_short = m_short[:29] + "..."
            f_short = str(rh.get("filter_mode") or "dialogue_only")[:15]
            lines.append(
                f"  {r_idx:<2d} | {str(rh.get('time', '—')):<19s} | {m_short:<32s} | {f_short:<15s} | "
                f"{int(rh.get('batch_pages', 12)):>3d}c. | {int(rh.get('pages_processed', 0)):>4d} | "
                f"{int(rh.get('blocks_translated', 0)):>8d} | {int(rh.get('blocks_filtered', 0)):>7d} | "
                f"{int(rh.get('blocks_untranslated', 0)):>7d} | {int(rh.get('refusals_count', 0)):>9d} | "
                f"{float(rh.get('sec_per_page', 0.0)):>5.2f}с | "
                f"OCR={float(rh.get('ocr_sec', 0.0)):.1f}с, ИИ={float(rh.get('llm_sec', 0.0)):.1f}с, Ренд={float(rh.get('render_sec', 0.0)):.1f}с (Всего {float(rh.get('total_sec', 0.0)):.1f}с)"
            )

    incidents = tr_info.get("recent_translation_incidents") or []
    api_calls = tr_info.get("recent_api_calls") or []
    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            "  ☁️ 5. ЖУРНАЛ РАБОТЫ ИИ: КОГДА И ПОЧЕМУ НЕ СРАБОТАЛА ОБЛАЧНАЯ МОДЕЛЬ / ДОПЕРЕВОДЫ",
            "----------------------------------------------------------------------------------------",
            f"• Всего запросов к ИИ за сессию      : {tr_info.get('total_llm_requests', 0)} (Облако: {tr_info.get('total_cloud_requests', 0)}, Локальный Ollama: {tr_info.get('total_ollama_requests', 0)}, Параллельных пачек: {tr_info.get('parallel_batches_count', 0)})",
            f"• Среднее время ответа ИИ            : {tr_info.get('avg_llm_latency_sec', 0.0)} сек (суммарно в ИИ: {tr_info.get('total_llm_time_sec', 0.0)} сек)",
            f"• Экономия токенов за счет кэша      : {tr_info.get('phrases_saved_by_cache', 0)} из {tr_info.get('phrases_requested_total', 0)} фраз ({tr_info.get('cache_hit_rate_pct', 0.0)}%) взято без вызовов API",
            f"• Отказов цензуры облачного API      : {tr_info.get('refusals_count', 0)} раз (перехвачено локальной моделью Ollama: {tr_info.get('local_censorship_fallbacks_count', 0)}, санитизировано триггеров: {tr_info.get('sanitized_retries_count', 0)})",
            f"• Сетевых повторов (HTTP 429/502/Таймаут): {tr_info.get('network_retries_count', 0)} | Аварийных переключений на Ollama: {tr_info.get('cloud_to_local_error_fallbacks_count', 0)}",
            f"• Поймано и исправлено брака ИИ      : утечек Ромадзи/латиницы={tr_info.get('romaji_leaks_caught_count', 0)}, остаточных иероглифов CJK={tr_info.get('cjk_leaks_caught_count', 0)}, микро-допереводов={tr_info.get('micro_retries_count', 0)}",
            "",
        ]
    )

    if not incidents:
        lines.append("  ✅ Инцидентов ИИ не зафиксировано: все запросы к модели прошли с 1-й попытки без цензуры и сетевых сбоев.")
    else:
        lines.append(f"  [Подробный журнал инцидентов и авто-восстановлений ИИ ({len(incidents)} событий)]:")
        for inc in incidents[:30]:
            det_s = f" | Детали: {inc.get('details')}" if inc.get("details") else ""
            lines.append(
                f"  • [{inc.get('time')}] [{inc.get('incident_type')}] Модель: {inc.get('model')} — {inc.get('summary_ru')}{det_s}"
            )

    if api_calls:
        lines.append("")
        lines.append(f"  [Последние прямые вызовы LLM API ({min(15, len(api_calls))} из {len(api_calls)})]:")
        for ac in api_calls[:15]:
            note_s = f" | Причина/Примечание: {ac.get('note')}" if ac.get("note") else ""
            lines.append(
                f"  • [{ac.get('time')}] {ac.get('provider')}::{ac.get('model')} | Статус: {ac.get('status')} | "
                f"Время: {ac.get('elapsed_sec')}с | Ключей получено: {ac.get('returned_keys')}{note_s}"
            )

    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            "  🔍 6. РАЗБОР ПРОПУСКОВ: ПОЧЕМУ НЕ БЫЛИ ПЕРЕВЕДЕНЫ СТРАНИЦЫ ИЛИ ОТДЕЛЬНЫЙ ТЕКСТ",
            "----------------------------------------------------------------------------------------",
            f"  6А. Страницы без переведенного текста (0 переведенных блоков: {len(zero_block_pages)} стр.):",
        ]
    )
    if not zero_block_pages:
        lines.append("      ✅ На всех страницах главы есть переведенные диалоги (нет пустых страниц).")
    else:
        for zp in zero_block_pages:
            lines.append(
                f"      • Стр. #{zp['index']:03d} ({zp['rel_path']}) — Кандидатов детектора: {zp['raw_candidates']}, "
                f"Отсеяно фильтром: {zp['filtered_count']} ➔ Причина: {zp['reason']}"
            )

    lines.append("")
    lines.append(
        f"  6Б. Текстовые блоки, которые НЕ удалось перевести из-за ошибки/пропуска ИИ ({len(untranslated_error_blocks)} шт.):"
    )
    if not untranslated_error_blocks:
        lines.append("      ✅ 0 потерянных блоков! Все принятые к переводу реплики успешно переведены.")
    else:
        for ub in untranslated_error_blocks[:50]:
            lines.append(
                f"      • ❌ Стр. #{ub['page_index']:03d} ({ub['rel_path']}), Кружок #{ub['block_id']} {ub.get('bbox')}: "
                f"Оригинал «{ub['original_text']}» — Причина: {ub['reason']}"
            )

    lines.append("")
    lines.append(
        f"  6В. Блоки, намеренно оставленные в оригинале после OCR (короткие вздохи / удаленные вручную: {len(skipped_post_ocr_blocks)} шт.):"
    )
    if not skipped_post_ocr_blocks:
        lines.append("      (Таких блоков нет)")
    else:
        for sb in skipped_post_ocr_blocks[:40]:
            lines.append(
                f"      • ⚪ Стр. #{sb['page_index']:03d} ({sb['rel_path']}), Кружок #{sb['block_id']}: "
                f"Оригинал «{sb['original_text']}» — {sb['reason']}"
            )

    lines.append("")
    lines.append(
        f"  6Г. Распознанный текст, отсеянный фильтром на этапе OCR (вздохи, SFX, короткие символы: {len(all_filtered_ocr_records)} шт.):"
    )
    if not all_filtered_ocr_records:
        lines.append("      (На этапе OCR фильтр не отсеял ни одной текстовой области)")
    else:
        for fr in all_filtered_ocr_records[:60]:
            lines.append(
                f"      • 🛡 Стр. #{fr['page_index']:03d} ({fr['rel_path']}) {fr.get('bbox')}: "
                f"«{fr['text']}» — Отсеяно: {fr['reason_ru']}"
            )
        if len(all_filtered_ocr_records) > 60:
            lines.append(f"      ... и еще {len(all_filtered_ocr_records) - 60} отфильтрованных мелких звуков/вздохов.")

    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            "  💡 7. КОНТРОЛЬ КАЧЕСТВА ВЕРСТКИ (УЗКИЕ БАББЛЫ) И АВТО-РЕКОМЕНДАЦИИ",
            "----------------------------------------------------------------------------------------",
            f"  7А. Кружки с минимальным размером шрифта (кегль уперся в {cur_min_fs} px — возможна теснота текста: {len(cramped_min_font_blocks)} шт.):",
        ]
    )
    if not cramped_min_font_blocks:
        lines.append("      ✅ Все реплики комфортно поместились в бабблы без сжатия до минимального предела шрифта.")
    else:
        for cb_item in cramped_min_font_blocks[:25]:
            lines.append(
                f"      • 🔎 Стр. #{cb_item['page_index']:03d} ({cb_item['rel_path']}), Кружок #{cb_item['block_id']} "
                f"(шрифт {cb_item['font_size']}px): «{cb_item['translated_text'][:55]}»"
            )

    lines.append("")
    lines.append(f"  7Б. Области с низкой уверенностью детектора (< 0.35: {len(low_conf_blocks)} шт.):")
    if not low_conf_blocks:
        lines.append("      ✅ Все текстовые рамки распознаны с высокой уверенностью детектора (>= 0.35).")
    else:
        for lcb in low_conf_blocks[:20]:
            lines.append(
                f"      • Стр. #{lcb['page_index']:03d} ({lcb['rel_path']}), Кружок #{lcb['block_id']} "
                f"(conf={lcb['confidence']}): «{lcb['original_text'][:30]}» ➔ «{lcb['translated_text'][:35]}»"
            )

    lines.append("")
    lines.append("  7В. Автоматические рекомендации по оптимизации параметров для этой главы:")
    recs: list[str] = []
    if untranslated_errs_cnt > 0:
        recs.append(
            f"⚠️ Обнаружено {untranslated_errs_cnt} непереведенных блоков (см. Раздел 6Б). Откройте указанные страницы в веб-студии и нажмите «🔄 Перевести ИИ» в карточке нужного кружка."
        )
    if int(tr_info.get("refusals_count") or 0) > 0:
        if int(tr_info.get("local_censorship_fallbacks_count") or 0) > 0:
            recs.append(
                f"🛡 Облачный API отклонил {tr_info.get('refusals_count')} пачек из-за цензуры, но локальная модель ({local_fb_llm}) успешно перехватила и перевела их. Для полностью бесцензурного локального перевода 18+ глав рекомендуем выбрать модель «🔥 Shisa V2 12B Uncensored»."
            )
        else:
            recs.append(
                f"⚠️ Облачный API отклонял запросы из-за цензуры ({tr_info.get('refusals_count')} раз), а локальная модель в Ollama не была запущена. Скачайте во вкладке Настройки модель «🔥 Shisa V2 12B Uncensored» для мгновенного авто-обхода цензуры!"
            )
    if len(cramped_min_font_blocks) >= 4:
        recs.append(
            f"🔤 В {len(cramped_min_font_blocks)} узких бабблах шрифт уменьшился до порога {cur_min_fs}px. Вы можете снизить «Минимальный размер шрифта» до 10px в настройках или слегка расширить рамки указанных кружков мышкой."
        )
    if filter_reasons_counter.get("sigh", 0) >= 8:
        recs.append(
            f"💬 Фильтр skip_sighs=True бережно пропустил {filter_reasons_counter.get('sigh', 0)} коротких вздохов/охов без замазки рисунка. Если вы хотите переводить даже короткие стоны («Ах!», «Мм!»), переключите профиль фильтра на «🗨 Все облачка диалогов (bubbles_only)»."
        )
    if not recs:
        recs.append(
            "✨ Текущие параметры фильтрации, шрифта и выбранная модель отработали оптимально — корректировка настроек не требуется!"
        )
    for r_line in recs:
        lines.append(f"      • {r_line}")

    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            f"  📄 8. ПОСТРАНИЧНАЯ СВОДНАЯ ТАБЛИЦА ({len(per_page_rows)} СТРАНИЦ)",
            "----------------------------------------------------------------------------------------",
            "  #    | Файл страницы              | Статус          | Яз. | Канд. | Фильтр | Блоков | Перев. | Ошиб. | Шрифт (px) | Примечания по странице",
            "  " + "-" * 138,
        ]
    )
    for row in per_page_rows:
        fname_disp = row["rel_path"]
        if len(fname_disp) > 26:
            fname_disp = "..." + fname_disp[-23:]
        lines.append(
            f"  #{row['index']:03d} | {fname_disp:<26s} | {row['status']:<15s} | {row['lang']:<3s} | "
            f"{row['raw_cand']:>5d} | {row['filtered']:>6d} | {row['kept']:>6d} | "
            f"{row['trans_ok']:>6d} | {row['untrans_err']:>5d} | {row['fs_range']:<10s} | {row['note']}"
        )

    if batches_log:
        lines.extend(
            [
                "",
                "----------------------------------------------------------------------------------------",
                f"  📦 9. ХРОНОЛОГИЯ ПАКЕТНОЙ ОБРАБОТКИ ({len(batches_log)} ПАЧЕК)",
                "----------------------------------------------------------------------------------------",
            ]
        )
        for b_item in batches_log[-30:]:
            lines.append(
                f"  • [{b_item.get('time', '—')}] {b_item.get('label')} | Файлы: {b_item.get('files_range', '—')} | "
                f"Блоков: {b_item.get('blocks_count', 0)} (перев.: {b_item.get('translated_count', 0)}, отфильтр.: {b_item.get('filtered_count', 0)}) | "
                f"OCR={b_item.get('ocr_sec', 0)}с, ИИ={b_item.get('llm_sec', 0)}с, Рендер={b_item.get('render_sec', 0)}с "
                f"=> ВСЕГО={b_item.get('total_sec', 0)}с ({b_item.get('sec_per_page', 0)}с/стр) | Модель: {b_item.get('model', active_llm)}"
            )

    lines.extend(
        [
            "",
            "========================================================================================",
            "  КОНЕЦ ДИАГНОСТИЧЕСКОГО ОТЧЕТА ГЛАВЫ (АВТОСОХРАНЕНО В ПАПКУ ПЕРЕВОДА)",
            "========================================================================================",
            "",
        ]
    )

    report_text = "\n".join(lines)
    try:
        report_path.write_text(report_text, encoding="utf-8")
    except Exception as exc:
        logger.warning("Не удалось записать диагностический отчет %s: %s", report_path, exc)
    try:
        _generate_and_save_chapter_book_files(
            output_dir=output_dir,
            manifest=manifest,
            source_dir=source_dir,
            target_lang=t_lang,
        )
    except Exception as book_exc:
        logger.warning("Ошибка генерации файлов книги главы: %s", book_exc)
    return report_text


def is_page_already_translated(
    dest_path: Path,
    page_meta: dict[str, Any] | None = None,
    target_lang: str = "en",
) -> bool:
    try:
        if not (dest_path.exists() and dest_path.is_file() and dest_path.stat().st_size > 128):
            return False
    except OSError:
        return False

    if isinstance(page_meta, dict):
        if bool(page_meta.get("manually_edited")):
            return True
        if "blocks" in page_meta and isinstance(page_meta.get("blocks"), list):
            saved_blocks = page_meta["blocks"]
            if not saved_blocks:
                return True
            active_blocks = [
                b
                for b in saved_blocks
                if isinstance(b, dict)
                and not bool(b.get("is_deleted", False))
                and not str(b.get("skip_reason") or "").strip()
                and str(b.get("translation_source") or "") != "skipped_sigh"
                and bool(str(b.get("translated_text") or "").strip())
            ]
            if not active_blocks:
                has_only_errors = any(
                    isinstance(b, dict)
                    and not bool(b.get("is_deleted", False))
                    and str(b.get("translation_source") or "") == "untranslated_error"
                    for b in saved_blocks
                )
                return not has_only_errors

            valid_translated_count = 0
            for b in active_blocks:
                tr_txt = str(b.get("translated_text") or "").strip()
                if not tr_txt:
                    continue
                if OllamaTranslator._contains_cjk(tr_txt):
                    continue
                if OllamaTranslator._is_bubble_refusal_value(tr_txt):
                    continue
                valid_translated_count += 1

            if valid_translated_count == 0:
                return False

    return True


def inspect_folder_pages(source_dir: Path, target_lang: str = "en") -> dict[str, Any]:
    t_lang = (target_lang or "en").lower().strip()
    if t_lang not in VALID_TARGET_LANGS:
        t_lang = "en"

    source_dir = _unwrap_output_folder_if_needed(source_dir)
    output_dir = get_output_dir_for_folder(source_dir, t_lang)
    images = collect_images_in_folder(source_dir)
    manifest = _load_folder_manifest(output_dir)
    pages_meta = manifest.get("pages", {}) if isinstance(manifest.get("pages"), dict) else {}

    pages_info: list[dict[str, Any]] = []
    translated_count = 0

    for idx, img_path in enumerate(images, start=1):
        rel = img_path.relative_to(source_dir)
        rel_norm = _normalize_rel_path(rel)
        dest_path = output_dir / rel
        try:
            dest_exists = bool(dest_path.exists() and dest_path.is_file() and dest_path.stat().st_size > 128)
        except OSError:
            dest_exists = False
        meta = pages_meta.get(rel_norm, {}) if isinstance(pages_meta.get(rel_norm), dict) else {}
        done = is_page_already_translated(dest_path, page_meta=meta, target_lang=t_lang)
        if done:
            translated_count += 1
        has_saved_blocks_key = "blocks" in meta and isinstance(meta.get("blocks"), list)
        saved_blocks_raw = meta.get("blocks") if has_saved_blocks_key else []
        phrase_cache_dict = manifest.get("phrase_cache") if isinstance(manifest.get("phrase_cache"), dict) else {}
        enriched_saved_blocks: list[dict[str, Any]] = []
        for sb in saved_blocks_raw:
            if not isinstance(sb, dict):
                continue
            sb_copy = _ensure_block_clean_bbox(dict(sb))
            tr_s = str(sb_copy.get("translated_text") or "").strip()
            if bool(sb_copy.get("is_deleted", False)) or not tr_s:
                sb_copy["is_deleted"] = True
                sb_copy["translated_text"] = ""
            else:
                sb_copy["is_deleted"] = False
                sb_copy["translated_text"] = tr_s
            orig_s = str(sb_copy.get("original_text") or "").strip()
            if orig_s and not str(sb_copy.get("romaji_text") or "").strip():
                sb_copy["romaji_text"] = (
                    str(phrase_cache_dict.get(f"romaji||{orig_s}") or "").strip()
                    or japanese_to_romaji_local(orig_s)
                )
            enriched_saved_blocks.append(sb_copy)

        pages_info.append(
            {
                "index": idx,
                "rel_path": rel_norm,
                "filename": img_path.name,
                "source_path": str(img_path),
                "dest_path": str(dest_path),
                "dest_exists": dest_exists,
                "is_translated": done,
                "blocks_count": meta.get("blocks_count", len(enriched_saved_blocks) if enriched_saved_blocks else None),
                "free_text_count": meta.get("free_text_count", 0),
                "detected_lang": meta.get("detected_lang"),
                "updated_at": meta.get("updated_at"),
                "has_saved_blocks": has_saved_blocks_key,
                "blocks": enriched_saved_blocks,
            }
        )

    folder_glossary = normalize_glossary_entries(manifest.get("glossary", []))
    return {
        "status": "ok",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "target_lang": t_lang,
        "total_pages": len(images),
        "translated_pages": translated_count,
        "remaining_pages": max(0, len(images) - translated_count),
        "folder_glossary": folder_glossary,
        "pages": pages_info,
    }


def open_native_folder_picker() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title="Выберите папку с мангой / манхвой / маньхуа для перевода"
        )
        root.destroy()
        return selected if selected else None
    except Exception as exc:
        logger.error("Не удалось открыть диалог выбора папки Tkinter: %s", exc)
        return None


def open_native_file_picker() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Выберите страницу манги для перевода",
            filetypes=[
                ("Изображения манги", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("Все файлы", "*.*"),
            ],
        )
        root.destroy()
        return selected if selected else None
    except Exception as exc:
        logger.error("Не удалось открыть диалог выбора файла Tkinter: %s", exc)
        return None


def _pil_to_rgb_clean(img: Image.Image) -> Image.Image:
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.split()[-1]
        bg.paste(img.convert("RGB"), mask=alpha)
        return bg
    return img.convert("RGB")


def decode_base64_image(base64_str: str) -> Image.Image:
    cleaned = base64_str.strip()
    if "," in cleaned and cleaned.startswith("data:"):
        cleaned = cleaned.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(cleaned, validate=False)
        raw_img = Image.open(io.BytesIO(image_bytes))
        return _pil_to_rgb_clean(raw_img)
    except (binascii.Error, OSError, ValueError) as exc:
        logger.warning("Ошибка декодирования изображения (400): %s", exc)
        raise HTTPException(
            status_code=400, detail=f"Некорректный Base64 или формат картинки: {exc}"
        ) from exc


def encode_pil_to_data_url(image: Image.Image, fmt: str = "WEBP", quality: int = 92) -> str:
    buf = io.BytesIO()
    norm_fmt = (fmt or "WEBP").upper().strip()
    if norm_fmt in ("JPG", "JPEG"):
        image.save(buf, format="JPEG", quality=95, optimize=True)
        mime = "image/jpeg"
    elif norm_fmt == "PNG":
        image.save(buf, format="PNG")
        mime = "image/png"
    else:
        image.save(buf, format="WEBP", quality=quality, method=4)
        mime = "image/webp"

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def save_debug_artifacts(
    raw_image: Image.Image,
    mask_image: Image.Image,
    cleaned_image: Image.Image,
    final_image: Image.Image,
    blocks: list[dict[str, Any]],
) -> tuple[str, str, str, str, str]:
    req_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    boxes_path = TEMP_DIR / f"{req_id}_detected.png"
    mask_path = TEMP_DIR / f"{req_id}_mask.png"
    cleaned_path = TEMP_DIR / f"{req_id}_cleaned.png"
    final_path = TEMP_DIR / f"{req_id}_final.png"
    json_path = TEMP_DIR / f"{req_id}_translation.json"

    vis_bgr = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
    for block in blocks:
        x1, y1, x2, y2 = block["bbox"]
        is_free = bool(block.get("is_free_text", False))
        color = (255, 140, 0) if is_free else (0, 200, 0)
        tag = "TXT" if is_free else "BBL"
        label = f"#{block['id']}[{tag}] ({block['confidence']:.2f})"
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis_bgr,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(boxes_path), vis_bgr)
    mask_image.save(mask_path)
    cleaned_image.save(cleaned_path)
    final_image.save(final_path)
    json_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")

    return (
        str(boxes_path.relative_to(BASE_DIR)),
        str(mask_path.relative_to(BASE_DIR)),
        str(cleaned_path.relative_to(BASE_DIR)),
        str(final_path.relative_to(BASE_DIR)),
        str(json_path.relative_to(BASE_DIR)),
    )


def _resolve_filter_params(
    detect_free_text: bool | None = None,
    text_filter_mode: str | None = None,
    skip_sighs: bool | None = None,
    skip_sfx: bool | None = None,
    min_text_length: int | None = None,
) -> tuple[bool, str, bool, bool, int]:
    t_eng = translator_engine
    eff_mode = (
        text_filter_mode.strip()
        if (text_filter_mode and text_filter_mode.strip())
        else (t_eng.text_filter_mode if t_eng else "dialogue_only")
    )
    eff_skip_sighs = (
        bool(skip_sighs)
        if skip_sighs is not None
        else (t_eng.skip_sighs if t_eng else True)
    )
    eff_skip_sfx = (
        bool(skip_sfx)
        if skip_sfx is not None
        else (t_eng.skip_sfx if t_eng else True)
    )
    eff_free = (
        bool(detect_free_text)
        if detect_free_text is not None
        else (t_eng.detect_free_text if t_eng else False)
    )
    eff_min_len = (
        max(1, int(min_text_length))
        if min_text_length is not None
        else (t_eng.min_text_length if t_eng else 2)
    )
    return eff_free, eff_mode, eff_skip_sighs, eff_skip_sfx, eff_min_len


def _run_ocr_compat(
    engine: MangaOCREngine,
    pil_image: Image.Image,
    lang: str,
    detect_free_text: bool = False,
    text_filter_mode: str = "dialogue_only",
    skip_sighs: bool = True,
    skip_sfx: bool = True,
    min_text_length: int = 2,
) -> tuple[list[dict[str, Any]], str]:
    try:
        res = engine.process_image(
            pil_image,
            lang=lang,
            detect_free_text=detect_free_text,
            text_filter_mode=text_filter_mode,
            skip_sighs=skip_sighs,
            skip_sfx=skip_sfx,
            min_text_length=min_text_length,
        )
    except TypeError:
        res = engine.process_image(pil_image, lang=lang, detect_free_text=detect_free_text)
    except (ValueError, RuntimeError) as exc:
        if lang != "ja" and "install_multilang_ocr" not in str(exc):
            logger.warning(
                "ocr_engine не принял lang='%s' (%s). Повтор с lang='ja'.",
                lang,
                exc,
            )
            res = engine.process_image(
                pil_image,
                lang="ja",
                detect_free_text=detect_free_text,
                text_filter_mode=text_filter_mode,
                skip_sighs=skip_sighs,
                skip_sfx=skip_sfx,
                min_text_length=min_text_length,
            )
        else:
            raise

    if isinstance(res, tuple) and len(res) == 2:
        return res[0], str(res[1])
    return list(res), "ja"


async def process_single_pil_image(
    pil_image: Image.Image,
    lang: str,
    target_lang: str,
    ollama_model: str | None = None,
    uppercase: bool = True,
    min_font_size: int = 12,
    max_font_size: int = 32,
    overflow_mode: str | None = None,
    detect_free_text: bool | None = None,
    text_filter_mode: str | None = None,
    skip_sighs: bool | None = None,
    skip_sfx: bool | None = None,
    min_text_length: int | None = None,
    context_lines: list[str] | None = None,
    external_cache: dict[str, str] | None = None,
    src_file: Path | str | None = None,
    glossary_entries: list[dict[str, str]] | None = None,
    use_cache: bool = True,
) -> tuple[Image.Image, Image.Image, Image.Image, list[dict[str, Any]], str, str]:
    assert ocr_engine is not None
    assert translator_engine is not None
    assert cleaner_engine is not None
    assert renderer_engine is not None

    eff_free, eff_mode, eff_skip_sighs, eff_skip_sfx, eff_min_len = _resolve_filter_params(
        detect_free_text=detect_free_text,
        text_filter_mode=text_filter_mode,
        skip_sighs=skip_sighs,
        skip_sfx=skip_sfx,
        min_text_length=min_text_length,
    )

    _t0_all = time.perf_counter()
    _t0_ocr = time.perf_counter()
    async with gpu_lock:
        blocks, detected_lang = await asyncio.to_thread(
            _run_ocr_compat,
            ocr_engine,
            pil_image,
            lang,
            eff_free,
            eff_mode,
            eff_skip_sighs,
            eff_skip_sfx,
            eff_min_len,
        )
    _ocr_sec = round(time.perf_counter() - _t0_ocr, 2)

    translator_engine.stop_requested = False
    active_model = await translator_engine.resolve_model(ollama_model)

    if not blocks:
        empty_mask = Image.new("L", pil_image.size, 0)
        return pil_image.copy(), pil_image.copy(), empty_mask, [], active_model, detected_lang

    _t0_llm = time.perf_counter()
    blocks = await translator_engine.translate_blocks(
        blocks=blocks,
        source_lang=detected_lang,
        target_lang=target_lang,
        model=active_model,
        context_lines=context_lines,
        external_cache=external_cache,
        use_cache=use_cache,
        skip_sighs=eff_skip_sighs,
        glossary_entries=glossary_entries,
    )
    _llm_sec = round(time.perf_counter() - _t0_llm, 2)

    active_render_blocks: list[dict[str, Any]] = []
    for idx_b, blk in enumerate(blocks, start=1):
        blk["id"] = idx_b
        clean_tr_blk = str(blk.get("translated_text") or "").strip()
        if bool(blk.get("is_deleted", False)) or not clean_tr_blk:
            blk["is_deleted"] = True
            blk["translated_text"] = ""
        else:
            blk["is_deleted"] = False
            blk["translated_text"] = clean_tr_blk
            active_render_blocks.append(blk)
        _ensure_block_clean_bbox(blk)

    if not active_render_blocks:
        empty_mask = Image.new("L", pil_image.size, 0)
        return pil_image.copy(), pil_image.copy(), empty_mask, blocks, active_model, detected_lang

    _t0_render = time.perf_counter()
    cleaned_image, mask_image, _clean_hit, _clean_hash = await _get_or_clean_page_image(
        pil_image,
        active_render_blocks,
        src_file=src_file,
        save_to_disk=True,
    )
    if mask_image is None:
        mask_image = Image.new("L", pil_image.size, 0)

    eff_overflow = overflow_mode or (getattr(translator_engine, "overflow_mode", "shrink") if translator_engine else "shrink")
    final_image = await asyncio.to_thread(
        renderer_engine.render_blocks,
        cleaned_image,
        active_render_blocks,
        uppercase,
        max_font_size,
        min_font_size,
        None,
        eff_overflow,
    )
    _render_sec = round(time.perf_counter() - _t0_render, 2)
    _total_sec = round(time.perf_counter() - _t0_all, 2)

    logger.info(
        "📊 [ДИАГНОСТИКА СТРАНИЦЫ] Блоков: %d | Тайминг: OCR=%.2fс, ИИ=%.2fс, Очистка+Верстка=%.2fс | Всего=%.2fс | Модель=%s",
        len(blocks),
        _ocr_sec,
        _llm_sec,
        _render_sec,
        _total_sec,
        active_model,
    )
    _append_diagnostic_event(
        "single_page",
        {
            "label": "1 стр.",
            "pages_count": 1,
            "blocks_count": len(blocks),
            "detected_lang": detected_lang,
            "target_lang": target_lang,
            "model": active_model,
            "ocr_sec": _ocr_sec,
            "llm_sec": _llm_sec,
            "render_sec": _render_sec,
            "total_sec": _total_sec,
            "sec_per_page": _total_sec,
        },
    )

    return final_image, cleaned_image, mask_image, blocks, active_model, detected_lang


async def _run_folder_translation_job(
    source_dir: Path,
    output_dir: Path,
    image_paths: list[Path],
    lang: str,
    target_lang: str,
    ollama_model: str | None,
    uppercase: bool,
    min_font_size: int,
    max_font_size: int,
    detect_free_text: bool | None,
    skip_existing: bool,
    force_retranslate: bool,
    open_when_done: bool,
    batch_pages: int = 15,
    translation_profile: str | None = None,
    text_filter_mode: str | None = None,
    skip_sighs: bool | None = None,
    skip_sfx: bool | None = None,
    min_text_length: int | None = None,
    overflow_mode: str | None = None,
) -> None:
    global _folder_stop_requested
    assert ocr_engine is not None
    assert translator_engine is not None
    assert cleaner_engine is not None
    assert renderer_engine is not None

    _folder_stop_requested = False
    translator_engine.stop_requested = False
    start_ts = time.perf_counter()
    prof_cfg = translator_engine.get_profile_config(translation_profile)
    active_profile_id = str(prof_cfg["id"])
    pages_per_batch = (
        int(prof_cfg["pages_per_batch"])
        if active_profile_id in ("eco_book", "balanced", "turbo")
        else max(1, min(50, int(batch_pages or prof_cfg["pages_per_batch"])))
    )
    tail_merge_pages = int(prof_cfg.get("tail_merge_pages", 5))

    manifest = _load_folder_manifest(output_dir)
    manifest["source_dir"] = str(source_dir)
    manifest["target_lang"] = target_lang
    if not isinstance(manifest.get("pages"), dict):
        manifest["pages"] = {}
    if not isinstance(manifest.get("phrase_cache"), dict):
        manifest["phrase_cache"] = {}
    if not isinstance(manifest.get("run_history"), list):
        manifest["run_history"] = []
    if not isinstance(manifest.get("batches_log"), list):
        manifest["batches_log"] = []
    t_lang_norm = (target_lang or "en").lower().strip()
    raw_rolling = list(manifest.get("rolling_context") or [])[-12:]
    rolling_context: list[str] = []
    for rc_line in raw_rolling:
        if isinstance(rc_line, str) and "=>" in rc_line:
            lhs, rhs = [p.strip() for p in rc_line.split("=>", 1)]
            if t_lang_norm == "ru" and not re.search(r"[А-Яа-яЁё]", rhs):
                continue
            if t_lang_norm == "en" and OllamaTranslator._is_romaji_leak(rhs, lhs, "en"):
                continue
            rolling_context.append(rc_line)

    raw_phrase_cache: dict[str, str] = dict(manifest.get("phrase_cache") or {})
    phrase_cache: dict[str, str] = {}
    for ck, cv in raw_phrase_cache.items():
        if not isinstance(ck, str) or not isinstance(cv, str) or not cv.strip():
            continue
        if OllamaTranslator._is_bubble_refusal_value(cv):
            continue
        if ck.startswith("ru||") and not re.search(r"[А-Яа-яЁё]", cv):
            continue
        if ck.startswith("en||"):
            orig_part = ck.split("||", 1)[1] if "||" in ck else ""
            if OllamaTranslator._is_romaji_leak(cv, orig_part, "en"):
                continue
        phrase_cache[ck] = cv
    if force_retranslate:
        phrase_cache.clear()
        translator_engine.phrase_cache.clear()

    to_process: list[tuple[int, Path, Path, str]] = []
    skipped_count = 0
    pages_meta_dict = manifest.get("pages", {}) if isinstance(manifest.get("pages"), dict) else {}

    for idx, img_path in enumerate(image_paths, start=1):
        rel = img_path.relative_to(source_dir)
        rel_norm = _normalize_rel_path(rel)
        dest_path = output_dir / rel
        p_meta = pages_meta_dict.get(rel_norm) if isinstance(pages_meta_dict.get(rel_norm), dict) else None
        if skip_existing and not force_retranslate and is_page_already_translated(dest_path, page_meta=p_meta, target_lang=t_lang_norm):
            skipped_count += 1
        else:
            to_process.append((idx, img_path, dest_path, rel_norm))

    folder_job_state.update(
        {
            "status": "running",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "lang": lang,
            "target_lang": target_lang,
            "total": len(image_paths),
            "current": skipped_count,
            "skipped_existing": skipped_count,
            "translated_in_run": 0,
            "current_file": (
                f"Пропущено уже готовых: {skipped_count} стр." if skipped_count > 0 else "Подготовка..."
            ),
            "elapsed_sec": 0.0,
            "error": None,
        }
    )

    if not to_process:
        folder_job_state["status"] = "completed"
        folder_job_state["current"] = len(image_paths)
        folder_job_state["current_file"] = f"Все страницы ({len(image_paths)}) уже переведены!"
        return

    try:
        done_in_run = 0
        run_ocr_sec_total = 0.0
        run_llm_sec_total = 0.0
        run_render_sec_total = 0.0
        run_blocks_translated = 0
        run_blocks_filtered = 0
        run_blocks_untranslated = 0
        refusals_before_run = int(translator_engine.diagnostics_stats.get("refusals_count") or 0)
        chapter_detected_lang: str | None = lang if lang in ("ja", "ko", "zh") else None
        active_mod = await translator_engine.resolve_model(
            ollama_model,
            source_lang=chapter_detected_lang or "ja",
            target_lang=target_lang,
            profile=active_profile_id,
        )
        eff_free, eff_mode, eff_skip_sighs, eff_skip_sfx, eff_min_len = _resolve_filter_params(
            detect_free_text=detect_free_text,
            text_filter_mode=text_filter_mode,
            skip_sighs=skip_sighs,
            skip_sfx=skip_sfx,
            min_text_length=min_text_length,
        )

        def _load_pil_sync(p_path: Path) -> Image.Image:
            with Image.open(p_path) as raw_im:
                return _pil_to_rgb_clean(raw_im)

        # БЛОК 1: Разбиваем страницы на пачки с умным слиянием хвостов (без пачек по 1 стр.)
        smart_batches = _build_smart_page_batches(
            to_process,
            pages_per_batch=pages_per_batch,
            tail_merge_pages=tail_merge_pages,
        )
        logger.info(
            "🚀 Пакетный перевод папки [профиль=%s (%s)]: всего к переводу %d стр. -> %d пачек (%s)",
            active_profile_id,
            prof_cfg.get("title", active_profile_id),
            len(to_process),
            len(smart_batches),
            ", ".join(f"{len(b)} стр." for b in smart_batches),
        )

        for _batch_num, page_batch in enumerate(smart_batches, start=1):
            if _folder_stop_requested:
                folder_job_state["status"] = "stopped"
                folder_job_state["current_file"] = f"Пауза ({folder_job_state['current']}/{len(image_paths)} готово)"
                _save_folder_manifest(output_dir, manifest)
                return

            _batch_t0 = time.perf_counter()
            scanned_pages: list[dict[str, Any]] = []
            combined_blocks: list[dict[str, Any]] = []
            batch_detected_lang = chapter_detected_lang or "ja"
            global_block_id = 1

            # Параллельно предзагружаем изображения пачки с диска в ОЗУ (без ожидания внутри GPU-цикла)
            _ocr_t0 = time.perf_counter()
            preloaded_pils = await asyncio.gather(
                *(asyncio.to_thread(_load_pil_sync, item[1]) for item in page_batch),
                return_exceptions=True,
            )

            # Шаг 1: Быстрое локальное OCR-сканирование пачки страниц на видеокарте (с батчингом MangaOCR!)
            for p_idx, ((orig_idx, img_path, dest_path, rel_norm), pre_pil) in enumerate(
                zip(page_batch, preloaded_pils), start=1
            ):
                if _folder_stop_requested:
                    break
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if active_profile_id == "eco_book":
                    folder_job_state["current_file"] = f"🔍 [Фаза 1: OCR страницы {orig_idx} из {len(image_paths)}] {rel_norm}"
                else:
                    folder_job_state["current_file"] = (
                        f"🔍 [1/3 OCR пачки {p_idx}/{len(page_batch)}] {rel_norm}"
                        if len(page_batch) > 1
                        else rel_norm
                    )
                folder_job_state["elapsed_sec"] = round(time.perf_counter() - start_ts, 1)

                try:
                    if isinstance(pre_pil, Exception):
                        raise pre_pil
                    pil_img: Image.Image = pre_pil
                    scan_lang = chapter_detected_lang if chapter_detected_lang else lang
                    async with gpu_lock:
                        p_blocks, det_lang = await asyncio.to_thread(
                            _run_ocr_compat,
                            ocr_engine,
                            pil_img,
                            scan_lang,
                            eff_free,
                            eff_mode,
                            eff_skip_sighs,
                            eff_skip_sfx,
                            eff_min_len,
                        )
                        p_ocr_diag = dict(getattr(ocr_engine, "last_page_diagnostics", {}) or {})
                    if chapter_detected_lang is None and det_lang in ("ja", "ko", "zh"):
                        chapter_detected_lang = det_lang
                        if active_profile_id == "local_only":
                            active_mod = await translator_engine.resolve_model(
                                ollama_model,
                                source_lang=det_lang,
                                target_lang=target_lang,
                                profile=active_profile_id,
                            )
                    batch_detected_lang = det_lang
                    folder_job_state["detected_lang"] = det_lang

                    # БЛОК 2.1: Упорядочиваем блоки каждой страницы в естественном порядке чтения комикса (ja -> RTL, zh/ko -> LTR)
                    p_blocks = _sort_page_blocks_reading_order(p_blocks, lang=det_lang)

                    page_global_ids: list[str | int] = []
                    for b_ord_idx, b in enumerate(p_blocks, start=1):
                        _ensure_block_clean_bbox(b)
                        if active_profile_id == "eco_book":
                            gid: str | int = f"P{orig_idx:02d}_{b_ord_idx:02d}"
                        else:
                            gid = global_block_id
                            global_block_id += 1
                        page_global_ids.append(gid)
                        b_copy = dict(b)
                        b_copy["id"] = gid
                        combined_blocks.append(b_copy)

                    scanned_pages.append(
                        {
                            "img_path": img_path,
                            "dest_path": dest_path,
                            "rel_norm": rel_norm,
                            "pil_img": pil_img,
                            "blocks": p_blocks,
                            "global_ids": page_global_ids,
                            "detected_lang": det_lang,
                            "ocr_diag": p_ocr_diag,
                            "ok": True,
                        }
                    )
                except Exception as ocr_exc:
                    logger.exception("Ошибка OCR при чтении файла %s: %s", img_path, ocr_exc)
                    shutil.copy2(img_path, dest_path)
                    manifest["pages"][rel_norm] = {
                        "blocks_count": 0,
                        "free_text_count": 0,
                        "detected_lang": batch_detected_lang,
                        "model": active_mod,
                        "ocr_error": str(ocr_exc),
                        "updated_at": int(time.time()),
                        "blocks": [],
                    }
                    scanned_pages.append(
                        {
                            "img_path": img_path,
                            "dest_path": dest_path,
                            "rel_norm": rel_norm,
                            "ocr_error": str(ocr_exc),
                            "ok": False,
                        }
                    )

            if _folder_stop_requested:
                folder_job_state["status"] = "stopped"
                folder_job_state["current_file"] = f"Пауза ({folder_job_state['current']}/{len(image_paths)} готово)"
                _save_folder_manifest(output_dir, manifest)
                return

            _batch_ocr_sec = round(time.perf_counter() - _ocr_t0, 2)

            # Шаг 2: ИИ-перевод пачки (в профиле eco_book — единым сценарием книги P01_01..P25_09 за 1 запрос!)
            _llm_t0 = time.perf_counter()
            translated_by_gid: dict[str, str] = {}
            source_by_gid: dict[str, str] = {}
            skip_reason_by_gid: dict[str, str] = {}
            api_censored_by_gid: dict[str, bool] = {}
            api_failed_by_gid: dict[str, bool] = {}
            if combined_blocks:
                folder_job_state["current_file"] = (
                    f"📖 [Фаза 2: Перевод книги главы единым сценарием: {len(scanned_pages)} стр. / {len(combined_blocks)} фраз]..."
                    if active_profile_id == "eco_book"
                    else f"☁️ [2/3 ИИ-перевод пачки #{_batch_num}: {len(scanned_pages)} стр. / {len(combined_blocks)} фраз]..."
                )
                folder_job_state["elapsed_sec"] = round(time.perf_counter() - start_ts, 1)
                try:
                    folder_glossary_entries = normalize_glossary_entries(manifest.get("glossary", []))
                    enriched_combined = await translator_engine.translate_blocks(
                        blocks=combined_blocks,
                        source_lang=batch_detected_lang,
                        target_lang=target_lang,
                        model=active_mod,
                        context_lines=rolling_context if rolling_context else None,
                        external_cache=phrase_cache,
                        use_cache=(not force_retranslate),
                        skip_sighs=eff_skip_sighs,
                        glossary_entries=folder_glossary_entries,
                        profile=active_profile_id,
                    )
                    for eb in enriched_combined:
                        gid_eb = str(eb["id"])
                        translated_by_gid[gid_eb] = str(eb.get("translated_text") or "").strip()
                        source_by_gid[gid_eb] = str(eb.get("translation_source") or "")
                        skip_reason_by_gid[gid_eb] = str(eb.get("skip_reason") or "")
                        api_censored_by_gid[gid_eb] = bool(eb.get("api_censored", False))
                        api_failed_by_gid[gid_eb] = bool(eb.get("api_failed", False))
                except Exception as llm_exc:
                    logger.exception("Ошибка пакетного ИИ-перевода пачки страниц: %s", llm_exc)

            if _folder_stop_requested:
                folder_job_state["status"] = "stopped"
                folder_job_state["current_file"] = f"Пауза ({folder_job_state['current']}/{len(image_paths)} готово)"
                _save_folder_manifest(output_dir, manifest)
                _generate_and_save_folder_report(
                    output_dir, manifest, source_dir=source_dir, target_lang=target_lang, run_status_label="⏸ Остановлено на паузе"
                )
                return

            _batch_llm_sec = round(time.perf_counter() - _llm_t0, 2)

            # БЛОК 3.1 и 3.2: Проверяем непереведенные бабблы пачки:
            # НИКОГДА не отправляем повторно в облачный API блоки с флагом api_censored или api_failed!
            # Такие блоки (или пачки <= 2 фраз) допереводим ТОЛЬКО локальной моделью (с учетом source_lang) или оставляем как SFX/оригинал.
            missing_combined: list[dict[str, Any]] = []
            for cb in combined_blocks:
                gid_c = str(cb["id"])
                cur_tr = translated_by_gid.get(gid_c, "").strip()
                cur_src = source_by_gid.get(gid_c, "")
                if (
                    not cur_tr
                    and str(cb.get("original_text") or "").strip()
                    and cur_src not in ("skipped_sigh_orig", "skipped_sigh_trans", "deleted_by_user")
                ):
                    cb_copy = dict(cb)
                    cb_copy["api_censored"] = api_censored_by_gid.get(gid_c, False)
                    cb_copy["api_failed"] = api_failed_by_gid.get(gid_c, False)
                    missing_combined.append(cb_copy)

            if missing_combined and not _folder_stop_requested:
                # Определяем, нужно ли переводить остатки строго через локальную модель (без облака)
                any_censored_or_failed = any(
                    bool(b.get("api_censored")) or bool(b.get("api_failed")) for b in missing_combined
                )
                force_local_rescue = (
                    active_mod.startswith("unsloth::")
                    and (any_censored_or_failed or len(missing_combined) <= 2)
                )
                rescue_model: str | None = active_mod
                if force_local_rescue:
                    rescue_model = await translator_engine._resolve_local_fallback_model(
                        target_lang=target_lang,
                        source_lang=batch_detected_lang,
                    )
                    if not rescue_model:
                        logger.info(
                            "🛡 [БЛОК 3] Пропуск повторной отправки %d отцензуренных/пропущенных бабблов в облако "
                            "(подходящая локальная модель для '%s' не установлена).",
                            len(missing_combined),
                            batch_detected_lang,
                        )
                if rescue_model:
                    logger.info(
                        "Доперевод %d оставшихся пустыми бабблов в пачке #%d через '%s' (skip_sighs=False)...",
                        len(missing_combined),
                        _batch_num,
                        rescue_model,
                    )
                    try:
                        rescued_list = await translator_engine.translate_blocks(
                            blocks=missing_combined,
                            source_lang=batch_detected_lang,
                            target_lang=target_lang,
                            model=rescue_model,
                            context_lines=rolling_context if rolling_context else None,
                            external_cache=phrase_cache,
                            use_cache=False,
                            skip_sighs=False,
                            profile="local_only" if force_local_rescue else active_profile_id,
                        )
                        for rb in rescued_list:
                            gid_r = str(rb["id"])
                            r_txt = str(rb.get("translated_text") or "").strip()
                            if r_txt:
                                translated_by_gid[gid_r] = r_txt
                                source_by_gid[gid_r] = str(rb.get("translation_source") or "llm_micro_retry")
                                skip_reason_by_gid[gid_r] = ""
                            elif rb.get("skip_reason"):
                                skip_reason_by_gid[gid_r] = str(rb.get("skip_reason"))
                    except Exception as rescue_exc:
                        logger.warning("Доперевод оставшихся блоков пачки не удался: %s", rescue_exc)

            # Шаг 3: Быстрая параллельная очистка бабблов, верстка и сохранение всех страниц пачки!
            _render_t0 = time.perf_counter()
            for sp in scanned_pages:
                if not sp.get("ok"):
                    continue
                p_blocks = sp["blocks"]
                gids = sp["global_ids"]
                for b, gid in zip(p_blocks, gids):
                    gid_s = str(gid)
                    t_txt = str(translated_by_gid.get(gid_s, "") or "").strip()
                    b["translated_text"] = t_txt
                    b["is_deleted"] = not bool(t_txt)
                    b["translation_source"] = source_by_gid.get(gid_s) or ("llm" if t_txt else "untranslated_error")
                    b["skip_reason"] = "" if t_txt else skip_reason_by_gid.get(gid_s, "")
                    b["api_censored"] = bool(api_censored_by_gid.get(gid_s, False)) if not t_txt else False
                    b["api_failed"] = bool(api_failed_by_gid.get(gid_s, False)) if not t_txt else False
                    _ensure_block_clean_bbox(b)
                    orig_t = (b.get("original_text") or "").strip()
                    if orig_t:
                        b["romaji_text"] = (
                            t_txt
                            if target_lang == "romaji"
                            else (
                                phrase_cache.get(f"romaji||{orig_t}")
                                or str(b.get("romaji_text") or "").strip()
                                or japanese_to_romaji_local(orig_t)
                            )
                        )
                    if orig_t and t_txt:
                        pair_s = f"{orig_t} => {t_txt}"
                        if pair_s not in rolling_context:
                            rolling_context.append(pair_s)
                rolling_context = rolling_context[-12:]

            render_sem = asyncio.Semaphore(4)
            uses_gpu_lama = cleaner_engine.lama_model is not None

            def _clean_render_save_sync(
                pil_img_in: Image.Image,
                blocks_in: list[dict[str, Any]],
                src_p: Path,
                dst_p: Path,
            ) -> None:
                active_in = [
                    b for b in (blocks_in or [])
                    if not bool(b.get("is_deleted")) and str(b.get("translated_text") or "").strip()
                ]
                if not active_in:
                    shutil.copy2(src_p, dst_p)
                    return
                clean_hash = _compute_clean_state_hash(src_p, active_in)
                cached_im = _get_cached_cleaned_image(src_p, clean_hash)
                if cached_im is not None:
                    cleaned_im = cached_im
                else:
                    cleaned_im, _ = cleaner_engine.clean_image(pil_img_in, active_in)
                    _store_cached_cleaned_image(src_p, clean_hash, cleaned_im, save_to_disk=True)
                eff_overflow = overflow_mode or (getattr(translator_engine, "overflow_mode", "shrink") if translator_engine else "shrink")
                final_im = renderer_engine.render_blocks(
                    cleaned_im,
                    active_in,
                    uppercase,
                    max_font_size,
                    min_font_size,
                    None,
                    eff_overflow,
                )
                ext = dst_p.suffix.lower()
                if ext == ".webp":
                    final_im.save(dst_p, format="WEBP", quality=92, method=2)
                elif ext in (".jpg", ".jpeg"):
                    final_im.save(dst_p, format="JPEG", quality=95, optimize=True)
                else:
                    final_im.save(dst_p)

            async def _render_page_task(sp_item: dict[str, Any]) -> None:
                nonlocal done_in_run
                if _folder_stop_requested:
                    return
                if not sp_item.get("ok"):
                    done_in_run += 1
                    folder_job_state["translated_in_run"] = done_in_run
                    folder_job_state["current"] = skipped_count + done_in_run
                    return

                img_path = sp_item["img_path"]
                dest_path = sp_item["dest_path"]
                rel_norm = sp_item["rel_norm"]
                pil_img = sp_item["pil_img"]
                p_blocks = sp_item["blocks"]
                det_lang = sp_item["detected_lang"]

                folder_job_state["current_file"] = f"🎨 [3/3 Очистка и верстка] {rel_norm}"
                try:
                    if uses_gpu_lama and p_blocks:
                        async with gpu_lock:
                            await asyncio.to_thread(_clean_render_save_sync, pil_img, p_blocks, img_path, dest_path)
                    else:
                        async with render_sem:
                            await asyncio.to_thread(_clean_render_save_sync, pil_img, p_blocks, img_path, dest_path)

                    manifest["pages"][rel_norm] = {
                        "blocks_count": len(p_blocks),
                        "free_text_count": sum(1 for b in p_blocks if b.get("is_free_text")),
                        "detected_lang": det_lang,
                        "model": active_mod,
                        "uppercase": uppercase,
                        "min_font_size": min_font_size,
                        "max_font_size": max_font_size,
                        "overflow_mode": overflow_mode or (getattr(translator_engine, "overflow_mode", "shrink") if translator_engine else "shrink"),
                        "ocr_diag": sp_item.get("ocr_diag", {}),
                        "updated_at": int(time.time()),
                        "blocks": p_blocks,
                    }
                    manifest["rolling_context"] = rolling_context
                    manifest["phrase_cache"] = phrase_cache
                    if len(scanned_pages) <= 6 or (done_in_run + 1) % 6 == 0:
                        _save_folder_manifest(output_dir, manifest)
                except Exception as render_exc:
                    logger.exception("Ошибка при сохранении страницы %s: %s", img_path, render_exc)
                    shutil.copy2(img_path, dest_path)
                finally:
                    sp_item["pil_img"] = None

                done_in_run += 1
                folder_job_state["translated_in_run"] = done_in_run
                folder_job_state["current"] = skipped_count + done_in_run
                folder_job_state["elapsed_sec"] = round(time.perf_counter() - start_ts, 1)

            if uses_gpu_lama:
                for sp in scanned_pages:
                    if _folder_stop_requested:
                        break
                    await _render_page_task(sp)
            else:
                await asyncio.gather(*(_render_page_task(sp) for sp in scanned_pages))

            _batch_render_sec = round(time.perf_counter() - _render_t0, 2)
            _batch_total_sec = round(time.perf_counter() - _batch_t0, 2)
            _pages_cnt = max(1, len(page_batch))
            _sec_per_page = round(_batch_total_sec / _pages_cnt, 2)
            _tot_non_zero = max(0.01, _batch_total_sec)
            _ocr_pct = int(round((_batch_ocr_sec / _tot_non_zero) * 100))
            _llm_pct = int(round((_batch_llm_sec / _tot_non_zero) * 100))
            _ren_pct = max(0, 100 - _ocr_pct - _llm_pct)

            logger.info(
                "📊 [ДИАГНОСТИКА ПАЧКИ #%d] Стр: %d | Блоков: %d | Тайминг: OCR=%.2fс (%d%%), ИИ=%.2fс (%d%%), Очистка+Верстка=%.2fс (%d%%) | Всего=%.2fс (%.2fс/стр) | VRAM=%.0f МБ",
                _batch_num,
                len(page_batch),
                len(combined_blocks),
                _batch_ocr_sec,
                _ocr_pct,
                _batch_llm_sec,
                _llm_pct,
                _batch_render_sec,
                _ren_pct,
                _batch_total_sec,
                _sec_per_page,
                _get_vram_allocated_mb(),
            )
            _append_diagnostic_event(
                "folder_batch",
                {
                    "label": f"Пачка #{_batch_num} ({len(page_batch)} стр.)",
                    "pages_count": len(page_batch),
                    "blocks_count": len(combined_blocks),
                    "detected_lang": batch_detected_lang,
                    "target_lang": target_lang,
                    "model": active_mod,
                    "ocr_sec": _batch_ocr_sec,
                    "llm_sec": _batch_llm_sec,
                    "render_sec": _batch_render_sec,
                    "total_sec": _batch_total_sec,
                    "sec_per_page": _sec_per_page,
                },
            )

            batch_trans_cnt = sum(1 for v in translated_by_gid.values() if v and v.strip())
            batch_filt_cnt = sum(
                int((sp.get("ocr_diag") or {}).get("filtered_count") or 0)
                for sp in scanned_pages
                if sp.get("ok")
            )
            batch_untrans_cnt = sum(
                1
                for cb in combined_blocks
                if not translated_by_gid.get(str(cb["id"]), "").strip()
                and source_by_gid.get(str(cb["id"]), "") not in ("skipped_sigh_orig", "skipped_sigh_trans", "deleted_by_user")
            )
            run_ocr_sec_total = round(run_ocr_sec_total + _batch_ocr_sec, 2)
            run_llm_sec_total = round(run_llm_sec_total + _batch_llm_sec, 2)
            run_render_sec_total = round(run_render_sec_total + _batch_render_sec, 2)
            run_blocks_translated += batch_trans_cnt
            run_blocks_filtered += batch_filt_cnt
            run_blocks_untranslated += batch_untrans_cnt

            first_rel = page_batch[0][3] if page_batch else ""
            last_rel = page_batch[-1][3] if page_batch else ""
            files_range_str = f"{first_rel} .. {last_rel}" if first_rel != last_rel else first_rel
            manifest["batches_log"].append(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "label": f"Пачка #{_batch_num} ({len(page_batch)} стр.)",
                    "files_range": files_range_str,
                    "pages_count": len(page_batch),
                    "blocks_count": len(combined_blocks),
                    "translated_count": batch_trans_cnt,
                    "filtered_count": batch_filt_cnt,
                    "untranslated_count": batch_untrans_cnt,
                    "detected_lang": batch_detected_lang,
                    "target_lang": target_lang,
                    "model": active_mod,
                    "ocr_sec": _batch_ocr_sec,
                    "llm_sec": _batch_llm_sec,
                    "render_sec": _batch_render_sec,
                    "total_sec": _batch_total_sec,
                    "sec_per_page": _sec_per_page,
                }
            )
            manifest["batches_log"] = manifest["batches_log"][-60:]
            manifest["rolling_context"] = rolling_context
            manifest["phrase_cache"] = phrase_cache
            _save_folder_manifest(output_dir, manifest)

            if _folder_stop_requested:
                folder_job_state["status"] = "stopped"
                folder_job_state["current_file"] = f"Пауза ({folder_job_state['current']}/{len(image_paths)} готово)"
                _generate_and_save_folder_report(
                    output_dir,
                    manifest,
                    source_dir=source_dir,
                    target_lang=target_lang,
                    run_status_label=f"⏸ Остановлено на паузе ({folder_job_state['current']}/{len(image_paths)} стр. готово)",
                )
                return

        total_elapsed = round(time.perf_counter() - start_ts, 2)
        folder_job_state["status"] = "completed"
        folder_job_state["elapsed_sec"] = total_elapsed
        folder_job_state["current_file"] = "Готово!"

        refusals_in_run = max(
            0, int(translator_engine.diagnostics_stats.get("refusals_count") or 0) - refusals_before_run
        )
        manifest["run_history"].append(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": int(time.time()),
                "status_label": f"✅ Завершен успешно (переведено в прогоне: {done_in_run} стр., было готово: {skipped_count} стр.)",
                "model": active_mod,
                "fallback_model": translator_engine.last_local_fallback_model or "",
                "filter_mode": eff_mode,
                "skip_sighs": eff_skip_sighs,
                "skip_sfx": eff_skip_sfx,
                "detect_free_text": eff_free,
                "min_text_length": eff_min_len,
                "batch_pages": pages_per_batch,
                "pages_processed": done_in_run,
                "pages_skipped_existing": skipped_count,
                "blocks_translated": run_blocks_translated,
                "blocks_filtered": run_blocks_filtered,
                "blocks_untranslated": run_blocks_untranslated,
                "refusals_count": refusals_in_run,
                "ocr_sec": run_ocr_sec_total,
                "llm_sec": run_llm_sec_total,
                "render_sec": run_render_sec_total,
                "total_sec": total_elapsed,
                "sec_per_page": round(total_elapsed / max(1, done_in_run), 2),
            }
        )
        manifest["run_history"] = manifest["run_history"][-25:]
        _save_folder_manifest(output_dir, manifest)
        _generate_and_save_folder_report(
            output_dir,
            manifest,
            source_dir=source_dir,
            target_lang=target_lang,
            run_status_label=f"✅ Завершен успешно за {total_elapsed}с (переведено: {done_in_run} стр., ранее готово: {skipped_count} стр.)",
        )

        if open_when_done and os.name == "nt" and output_dir.exists():
            try:
                os.startfile(str(output_dir))
            except Exception:
                pass

    except asyncio.CancelledError:
        folder_job_state["status"] = "stopped"
        folder_job_state["current_file"] = "Пауза"
        _save_folder_manifest(output_dir, manifest)
        _generate_and_save_folder_report(
            output_dir, manifest, source_dir=source_dir, target_lang=target_lang, run_status_label="⏸ Остановлено пользователем"
        )
    except Exception as exc:
        logger.exception("Критическая ошибка пакетного перевода папки")
        folder_job_state["status"] = "error"
        folder_job_state["error"] = str(exc)
        _generate_and_save_folder_report(
            output_dir, manifest, source_dir=source_dir, target_lang=target_lang, run_status_label=f"❌ Ошибка: {exc}"
        )


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def studio_web_ui() -> HTMLResponse:
    html_path = BASE_DIR / "web_studio.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Manga Translator v1.0.6 Server Running</h1>")


@app.get("/mascot.png")
async def get_mascot_image():
    mascot_path = BASE_DIR / "mascot.png"
    if mascot_path.exists():
        return FileResponse(str(mascot_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="mascot.png not found")


@app.get("/health")
async def health_check() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_info: dict[str, Any] = {"cuda_available": cuda_available}

    if cuda_available:
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        gpu_info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
        gpu_info["vram_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**2), 2
        )

    models = (
        await translator_engine.list_available_models(force_refresh=False)
        if translator_engine is not None
        else []
    )
    active_model = (
        await translator_engine.resolve_model() if translator_engine is not None else "uninitialized"
    )
    inpaint_backend = (
        "SimpleLama (LaMa GPU)"
        if (cleaner_engine is not None and cleaner_engine.lama_model is not None)
        else "OpenCV INPAINT_TELEA"
    )
    has_multilang_engine = bool(
        ocr_engine and (hasattr(ocr_engine, "_check_multilang_ocr") or hasattr(ocr_engine, "_check_easyocr"))
    )
    easyocr_ok = bool(
        ocr_engine._check_multilang_ocr()
        if (ocr_engine and hasattr(ocr_engine, "_check_multilang_ocr"))
        else (ocr_engine._check_easyocr() if (ocr_engine and hasattr(ocr_engine, "_check_easyocr")) else False)
    )

    return {
        "status": "ready" if ocr_engine is not None else "initializing",
        "version": "1.0.6",
        "gpu": gpu_info,
        "ollama_model": active_model,
        "active_model": active_model,
        "selected_model": translator_engine._resolved_model or "auto" if translator_engine else "auto",
        "models": models,
        "recommended_models": RECOMMENDED_LOCAL_MODELS,
        "cloud_presets": CLOUD_API_PRESETS,
        "unsloth_url": translator_engine.unsloth_url if translator_engine else "http://127.0.0.1:8888",
        "unsloth_api_key": translator_engine.unsloth_api_key if translator_engine else "",
        "unsloth_status": translator_engine.last_unsloth_status if translator_engine else {},
        "has_unsloth_key": bool(translator_engine and translator_engine.unsloth_api_key),
        "auto_save_web_pages": translator_engine.auto_save_web_pages if translator_engine else True,
        "web_save_dir": translator_engine.web_save_dir if translator_engine else str(BASE_DIR / "Saved_Manga"),
        "text_filter_mode": translator_engine.text_filter_mode if translator_engine else "dialogue_only",
        "skip_sighs": translator_engine.skip_sighs if translator_engine else True,
        "skip_sfx": translator_engine.skip_sfx if translator_engine else True,
        "detect_free_text": translator_engine.detect_free_text if translator_engine else False,
        "min_text_length": translator_engine.min_text_length if translator_engine else 2,
        "min_font_size": translator_engine.min_font_size if translator_engine else 12,
        "max_font_size": translator_engine.max_font_size if translator_engine else 32,
        "uppercase": translator_engine.uppercase if translator_engine else True,
        "overflow_mode": getattr(translator_engine, "overflow_mode", "shrink") if translator_engine else "shrink",
        "batch_pages": translator_engine.batch_pages if translator_engine else 15,
        "translation_profile": getattr(translator_engine, "translation_profile", "balanced") if translator_engine else "balanced",
        "translation_profiles": list(TRANSLATION_PROFILES.values()),
        "local_fallback_model_multilang": getattr(translator_engine, "local_fallback_model_multilang", "auto") if translator_engine else "auto",
        "ui_settings": translator_engine.ui_settings if translator_engine else {},
        "glossary": translator_engine.glossary if translator_engine else [],
        "glossary_count": len(translator_engine.glossary) if translator_engine else 0,
        "pull_state": translator_engine.pull_state if translator_engine else {},
        "supported_source_langs": ["auto", "ja", "ko", "zh"],
        "default_target_lang": "en",
        "supported_target_langs": ["en", "ru", "romaji"],
        "easyocr_installed": easyocr_ok,
        "ocr_engine_updated": has_multilang_engine,
        "inpaint_backend": inpaint_backend,
        "font": renderer_engine.font_path if renderer_engine is not None else None,
        "available_fonts": (
            renderer_engine.list_available_fonts()
            if (renderer_engine is not None and hasattr(renderer_engine, "list_available_fonts"))
            else []
        ),
        "folder_job": folder_job_state,
    }


@app.get("/models")
async def list_models_endpoint(
    refresh: bool = Query(default=False, description="Сбросить 15-секундный кэш и опросить порты заново"),
) -> dict[str, Any]:
    if translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")
    models = await translator_engine.list_available_models(force_refresh=refresh)
    active = await translator_engine.resolve_model()
    return {
        "status": "ok",
        "active_model": active,
        "selected_model": translator_engine._resolved_model or "auto",
        "unsloth_url": translator_engine.unsloth_url,
        "unsloth_api_key": translator_engine.unsloth_api_key,
        "unsloth_status": translator_engine.last_unsloth_status,
        "has_unsloth_key": bool(translator_engine.unsloth_api_key),
        "recommended_models": RECOMMENDED_LOCAL_MODELS,
        "cloud_presets": CLOUD_API_PRESETS,
        "pull_state": translator_engine.pull_state,
        "models": models,
    }


@app.post("/pull_model")
async def pull_model_endpoint(payload: PullModelRequest) -> dict[str, Any]:
    global _pull_task
    if translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")
    if translator_engine.pull_state.get("status") == "downloading":
        return {
            "status": "already_downloading",
            "pull_state": translator_engine.pull_state,
        }

    _pull_task = asyncio.create_task(
        translator_engine.pull_ollama_model(
            model_name=payload.model,
            set_as_default=payload.set_as_default,
        )
    )
    return {
        "status": "started",
        "model": payload.model,
        "pull_state": translator_engine.pull_state,
    }


@app.get("/pull_status")
async def get_pull_status_endpoint() -> dict[str, Any]:
    if translator_engine is None:
        return {"status": "idle"}
    return translator_engine.pull_state


@app.post("/settings")
async def update_settings_endpoint(payload: ServerSettingsRequest) -> dict[str, Any]:
    if translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")
    url_or_key_changed = (
        (payload.unsloth_url is not None and translator_engine._normalize_api_base_url(payload.unsloth_url) != translator_engine.unsloth_url)
        or (payload.unsloth_api_key is not None and payload.unsloth_api_key.strip() != translator_engine.unsloth_api_key)
    )
    saved = translator_engine.save_config(
        unsloth_url=payload.unsloth_url,
        unsloth_api_key=payload.unsloth_api_key,
        selected_model=payload.selected_model,
        translation_profile=payload.translation_profile,
        local_fallback_model_multilang=payload.local_fallback_model_multilang,
        auto_save_web_pages=payload.auto_save_web_pages,
        web_save_dir=payload.web_save_dir,
        text_filter_mode=payload.text_filter_mode,
        skip_sighs=payload.skip_sighs,
        skip_sfx=payload.skip_sfx,
        detect_free_text=payload.detect_free_text,
        min_text_length=payload.min_text_length,
        min_font_size=payload.min_font_size,
        max_font_size=payload.max_font_size,
        uppercase=payload.uppercase,
        overflow_mode=payload.overflow_mode,
        batch_pages=payload.batch_pages,
        ui_settings=payload.ui_settings,
    )
    if renderer_engine is not None and payload.overflow_mode:
        renderer_engine.overflow_mode = renderer_engine._normalize_overflow_mode(payload.overflow_mode)
    # БЛОК 5: Не сбрасываем 5-минутный кэш моделей при обычном сохранении настроек (только если изменились api_url / api_key)
    models = await translator_engine.list_available_models(force_refresh=url_or_key_changed)
    active = await translator_engine.resolve_model()
    return {
        "status": "ok",
        "active_model": active,
        "selected_model": saved["selected_model"],
        "translation_profile": saved.get("translation_profile", "balanced"),
        "local_fallback_model_multilang": saved.get("local_fallback_model_multilang", "auto"),
        "unsloth_url": saved["unsloth_url"],
        "unsloth_api_key": saved["unsloth_api_key"],
        "unsloth_status": translator_engine.last_unsloth_status,
        "has_unsloth_key": bool(saved["unsloth_api_key"]),
        "auto_save_web_pages": saved["auto_save_web_pages"],
        "web_save_dir": saved["web_save_dir"],
        "text_filter_mode": saved["text_filter_mode"],
        "skip_sighs": saved["skip_sighs"],
        "skip_sfx": saved["skip_sfx"],
        "detect_free_text": saved["detect_free_text"],
        "min_text_length": saved["min_text_length"],
        "min_font_size": saved["min_font_size"],
        "max_font_size": saved["max_font_size"],
        "uppercase": saved["uppercase"],
        "overflow_mode": saved.get("overflow_mode", "shrink"),
        "batch_pages": saved["batch_pages"],
        "ui_settings": saved["ui_settings"],
        "models": models,
    }


@app.post("/pick_web_save_folder")
async def pick_web_save_folder_endpoint() -> dict[str, Any]:
    """Открывает диалог выбора папки Windows для автосохранения манги из расширения Chrome."""
    if translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")
    folder_str = await asyncio.to_thread(open_native_folder_picker)
    if not folder_str:
        return {"status": "cancelled", "web_save_dir": translator_engine.web_save_dir}
    chosen_dir = Path(folder_str).resolve()
    chosen_dir.mkdir(parents=True, exist_ok=True)
    saved = translator_engine.save_config(web_save_dir=str(chosen_dir))
    return {
        "status": "ok",
        "web_save_dir": saved["web_save_dir"],
        "auto_save_web_pages": saved["auto_save_web_pages"],
    }


@app.post("/open_web_save_folder")
async def open_web_save_folder_endpoint(path: str | None = Query(default=None)) -> dict[str, Any]:
    """Создает (если еще нет) и открывает в Проводнике Windows папку автосохранения манги из браузера."""
    raw_dir = path or (translator_engine.web_save_dir if translator_engine else str(BASE_DIR / "Saved_Manga"))
    target = Path(raw_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt" and hasattr(os, "startfile"):
        try:
            os.startfile(str(target))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok", "web_save_dir": str(target)}


@app.post("/shutdown")
async def shutdown_server() -> dict[str, str]:
    loop = asyncio.get_running_loop()
    loop.call_later(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    return {"status": "shutting_down"}


@app.post("/pick_folder")
async def pick_folder_endpoint(target_lang: str = Query(default="en")) -> dict[str, Any]:
    folder_str = await asyncio.to_thread(open_native_folder_picker)
    if not folder_str:
        return {"status": "cancelled"}
    source_dir = Path(folder_str).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Папка не найдена: {source_dir}")
    return inspect_folder_pages(source_dir, target_lang=target_lang)


@app.post("/pick_file")
async def pick_file_endpoint() -> dict[str, Any]:
    file_str = await asyncio.to_thread(open_native_file_picker)
    if not file_str:
        return {"status": "cancelled"}
    file_path = Path(file_str).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Файл не найден: {file_path}")
    return {
        "status": "ok",
        "file_path": str(file_path),
        "filename": file_path.name,
        "parent_dir": str(file_path.parent),
    }


@app.post("/open_folder")
async def open_folder_in_explorer_endpoint(path: str = Query(...)) -> dict[str, str]:
    target = Path(path).resolve()
    if target.exists() and os.name == "nt" and hasattr(os, "startfile"):
        try:
            os.startfile(str(target))
            return {"status": "ok"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise HTTPException(status_code=404, detail="Папка еще не создана")


@app.post("/folder_inspect")
async def folder_inspect_endpoint(payload: FolderInspectRequest) -> dict[str, Any]:
    source_dir = Path(payload.folder_path).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Папка не найдена: {source_dir}")
    return inspect_folder_pages(source_dir, target_lang=payload.target_lang)


@app.get("/local_image")
async def get_local_image_endpoint(path: str = Query(..., description="Абсолютный путь к изображению на ПК")):
    img_path = Path(path).resolve()
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {img_path}")
    if img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла.")
    return FileResponse(str(img_path))


def _sanitize_folder_name(raw_name: str, fallback: str = "Web_Manga") -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", str(raw_name or "")).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._-")
    for sfx in ("_EN", "_RU", "_ROMAJI", "_en", "_ru", "_romaji"):
        if cleaned.endswith(sfx) and len(cleaned) > len(sfx):
            cleaned = cleaned[: -len(sfx)].rstrip(" ._-")
    if len(cleaned) > 68:
        cleaned = cleaned[:68].rstrip(" ._-")
    return cleaned or fallback


def _save_web_translated_page(
    raw_image: Image.Image,
    final_image: Image.Image,
    blocks: list[dict[str, Any]],
    target_lang: str,
    detected_lang: str,
    active_model: str,
    uppercase: bool,
    min_font_size: int,
    max_font_size: int,
    custom_save_dir: str | None = None,
    chapter_title: str | None = None,
    page_url: str | None = None,
    image_url: str | None = None,
    page_index: int | None = None,
    cleaned_image: Image.Image | None = None,
) -> str:
    """
    Сохраняет переведенную из браузера (Chrome Extension) страницу манги на компьютер:
    - Чистый оригинал сохраняется в `<save_dir>/<Название_главы>/<файл>.png`
    - Переведенная страница и `.translation_state.json` (с блоками и Ромадзи) сохраняются в
      `<save_dir>/<Название_главы>_<ЯЗЫК>/<файл>.png`, чтобы главу можно было в любой момент
      открыть в GUI-студии и отредактировать любой баббл вручную!
    """
    base_dir = Path(custom_save_dir.strip()).resolve() if (custom_save_dir and custom_save_dir.strip()) else (BASE_DIR / "Saved_Manga")
    base_dir.mkdir(parents=True, exist_ok=True)

    raw_chapter = (chapter_title or "").strip()
    if not raw_chapter and page_url:
        m_host = re.sub(r"^https?://", "", page_url.strip()).split("/")[0]
        raw_chapter = f"Web_{m_host}"
    if not raw_chapter:
        raw_chapter = time.strftime("Web_Manga_%Y-%m-%d")

    safe_chapter = _sanitize_folder_name(raw_chapter, fallback="Web_Manga")
    source_chapter_dir = base_dir / safe_chapter
    output_chapter_dir = get_output_dir_for_folder(source_chapter_dir, target_lang)

    source_chapter_dir.mkdir(parents=True, exist_ok=True)
    output_chapter_dir.mkdir(parents=True, exist_ok=True)

    url_stem = ""
    if image_url and not image_url.startswith("data:") and not image_url.startswith("blob:"):
        url_path = image_url.split("?", 1)[0].split("#", 1)[0]
        cand_name = Path(url_path).stem
        cand_clean = re.sub(r"[^a-zA-Z0-9_\-]+", "_", cand_name).strip("_")
        if cand_clean and len(cand_clean) <= 32:
            url_stem = cand_clean

    if page_index is not None and int(page_index) > 0:
        idx_prefix = f"{int(page_index):03d}"
        if url_stem and not url_stem.isdigit():
            filename = f"{idx_prefix}_{url_stem}.png"
        else:
            filename = f"{idx_prefix}.png"
    elif url_stem:
        filename = f"{url_stem}.png"
    else:
        existing_cnt = len(list(output_chapter_dir.glob("*.png")))
        filename = f"{existing_cnt + 1:03d}_{int(time.time()) % 100000}.png"

    orig_file_path = source_chapter_dir / filename
    dest_file_path = output_chapter_dir / filename

    if not orig_file_path.exists():
        raw_image.save(orig_file_path, format="PNG")
    final_image.save(dest_file_path, format="PNG")
    if cleaned_image is not None and blocks:
        try:
            c_hash = _compute_clean_state_hash(orig_file_path, blocks)
            _store_cached_cleaned_image(orig_file_path, c_hash, cleaned_image, save_to_disk=True)
        except Exception:
            pass

    manifest = _load_folder_manifest(output_chapter_dir)
    if not isinstance(manifest.get("pages"), dict):
        manifest["pages"] = {}
    if not isinstance(manifest.get("phrase_cache"), dict):
        manifest["phrase_cache"] = {}

    t_lang = (target_lang or "en").lower().strip()
    for b in blocks:
        _ensure_block_clean_bbox(b)
        orig_t = str(b.get("original_text") or "").strip()
        trans_t = str(b.get("translated_text") or "").strip()
        rom_t = str(b.get("romaji_text") or "").strip()
        if orig_t and trans_t:
            manifest["phrase_cache"][f"{t_lang}||{orig_t}"] = trans_t
        if orig_t and rom_t:
            manifest["phrase_cache"][f"romaji||{orig_t}"] = rom_t

    manifest["source_dir"] = str(source_chapter_dir)
    manifest["target_lang"] = target_lang
    manifest["pages"][filename] = {
        "blocks_count": len(blocks),
        "free_text_count": sum(1 for b in blocks if b.get("is_free_text")),
        "detected_lang": detected_lang,
        "model": active_model,
        "uppercase": uppercase,
        "min_font_size": min_font_size,
        "max_font_size": max_font_size,
        "ocr_diag": dict(getattr(ocr_engine, "last_page_diagnostics", {}) or {}) if ocr_engine else {},
        "page_url": page_url,
        "updated_at": int(time.time()),
        "blocks": blocks,
    }
    _save_folder_manifest(output_chapter_dir, manifest)
    _generate_and_save_folder_report(
        output_chapter_dir, manifest, source_dir=source_chapter_dir, target_lang=target_lang
    )
    logger.info("Веб-страница манги автоматически сохранена на ПК: %s", dest_file_path)
    return str(dest_file_path)


@app.post("/translate", response_model=TranslateResponse)
async def translate_manga_page(payload: TranslateRequest) -> TranslateResponse:
    if (
        ocr_engine is None
        or translator_engine is None
        or cleaner_engine is None
        or renderer_engine is None
    ):
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    raw_b64 = payload.image or payload.image_base64
    if not raw_b64:
        raise HTTPException(
            status_code=400, detail="Поле 'image' (или 'image_base64') обязательно."
        )

    req_lang = payload.lang.lower().strip()
    if req_lang not in VALID_SOURCE_LANGS:
        req_lang = "auto"

    target_lang = payload.target_lang.lower().strip()
    if target_lang not in VALID_TARGET_LANGS:
        raise HTTPException(
            status_code=400,
            detail="Используйте target_lang: 'en', 'ru' или 'romaji'.",
        )

    start_ts = time.perf_counter()
    pil_image = decode_base64_image(raw_b64)

    try:
        final_image, cleaned_image, mask_image, blocks, active_model, detected_lang = (
            await process_single_pil_image(
                pil_image=pil_image,
                lang=req_lang,
                target_lang=target_lang,
                ollama_model=(payload.ollama_model or payload.model),
                uppercase=payload.uppercase,
                min_font_size=payload.min_font_size,
                max_font_size=payload.max_font_size,
                overflow_mode=payload.overflow_mode,
                detect_free_text=payload.detect_free_text,
                text_filter_mode=payload.text_filter_mode,
                skip_sighs=payload.skip_sighs,
                skip_sfx=payload.skip_sfx,
                min_text_length=payload.min_text_length,
                use_cache=(not payload.force_retranslate),
            )
        )
    except RuntimeError as exc:
        logger.error("Ошибка выполнения (400): %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        logger.error("Ошибка LLM API (%s): %s", exc.response.status_code, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка LLM API ({exc.response.status_code}): проверьте ключ API или статус модели в Ollama.",
        ) from exc
    except httpx.ConnectError as exc:
        logger.error("Не удалось подключиться к LLM: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Не удалось подключиться к Ollama / внешнему API. Проверьте запущен ли Ollama или правильность URL API.",
        ) from exc
    except Exception as exc:
        logger.exception("Ошибка при переводе страницы")
        raise HTTPException(status_code=500, detail=f"Ошибка пайплайна: {exc}") from exc

    debug_img_path = debug_mask_path = debug_cleaned_path = debug_final_path = debug_json_path = None
    if payload.save_debug:
        (
            debug_img_path,
            debug_mask_path,
            debug_cleaned_path,
            debug_final_path,
            debug_json_path,
        ) = save_debug_artifacts(pil_image, mask_image, cleaned_image, final_image, blocks)

    # Автосохранение переведенной страницы из расширения Chrome (по умолчанию включено!)
    saved_web_path: str | None = None
    should_auto_save = (
        payload.auto_save
        if payload.auto_save is not None
        else bool(translator_engine.auto_save_web_pages)
    )
    if should_auto_save and blocks:
        try:
            saved_web_path = await asyncio.to_thread(
                _save_web_translated_page,
                raw_image=pil_image,
                final_image=final_image,
                blocks=blocks,
                target_lang=target_lang,
                detected_lang=detected_lang,
                active_model=active_model,
                uppercase=payload.uppercase,
                min_font_size=payload.min_font_size,
                max_font_size=payload.max_font_size,
                custom_save_dir=payload.save_dir or translator_engine.web_save_dir,
                chapter_title=payload.chapter_title,
                page_url=payload.page_url,
                image_url=payload.image_url,
                page_index=payload.page_index,
                cleaned_image=cleaned_image,
            )
        except Exception as save_exc:
            logger.warning("Не удалось автосохранить веб-страницу на диск: %s", save_exc)

    translated_b64 = (
        encode_pil_to_data_url(final_image, fmt=payload.image_format)
        if payload.return_image
        else None
    )
    inpaint_backend = "SimpleLama" if cleaner_engine.lama_model is not None else "OpenCV_TELEA"
    elapsed = round(time.perf_counter() - start_ts, 3)
    free_cnt = sum(1 for b in blocks if b.get("is_free_text"))

    return TranslateResponse(
        status="ok",
        lang=req_lang,
        detected_lang=detected_lang,
        target_lang=target_lang,
        ollama_model=active_model,
        inpaint_backend=inpaint_backend,
        image_size=[pil_image.width, pil_image.height],
        processing_time_sec=elapsed,
        blocks_count=len(blocks),
        free_text_blocks_count=free_cnt,
        saved_output_path=saved_web_path,
        debug_image_path=debug_img_path,
        debug_mask_path=debug_mask_path,
        debug_cleaned_path=debug_cleaned_path,
        debug_final_path=debug_final_path,
        debug_json_path=debug_json_path,
        translated_image=translated_b64,
        blocks=[TextBlock(**b) for b in blocks],
    )


@app.post("/translate_page_file", response_model=TranslateResponse)
async def translate_page_file_endpoint(payload: PageFileTranslateRequest) -> TranslateResponse:
    if (
        ocr_engine is None
        or translator_engine is None
        or cleaner_engine is None
        or renderer_engine is None
    ):
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    src_file = Path(payload.file_path).resolve()
    if not src_file.exists() or not src_file.is_file():
        raise HTTPException(status_code=400, detail=f"Исходный файл не найден: {src_file}")

    req_lang = payload.lang.lower().strip()
    if req_lang not in VALID_SOURCE_LANGS:
        req_lang = "auto"

    target_lang = payload.target_lang.lower().strip()
    if target_lang not in VALID_TARGET_LANGS:
        raise HTTPException(status_code=400, detail="Используйте target_lang: 'en', 'ru' или 'romaji'.")

    source_dir: Path | None = None
    output_dir: Path | None = None
    rel_norm: str | None = None

    if payload.output_path:
        dest_path = Path(payload.output_path).resolve()
    elif payload.source_dir:
        source_dir = Path(payload.source_dir).resolve()
        output_dir = get_output_dir_for_folder(source_dir, target_lang)
        try:
            rel = src_file.relative_to(source_dir)
        except ValueError:
            rel = Path(src_file.name)
        rel_norm = _normalize_rel_path(rel)
        dest_path = output_dir / rel
    else:
        source_dir = src_file.parent
        output_dir = get_output_dir_for_folder(source_dir, target_lang)
        rel_norm = src_file.name
        dest_path = output_dir / src_file.name

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    start_ts = time.perf_counter()

    with Image.open(src_file) as raw_im:
        pil_image = _pil_to_rgb_clean(raw_im)

    context_lines: list[str] | None = None
    phrase_cache: dict[str, str] = {}
    manifest: dict[str, Any] | None = None
    already_had_translation = dest_path.exists()
    if output_dir is not None:
        manifest = _load_folder_manifest(output_dir)
        ctx = manifest.get("rolling_context")
        if isinstance(ctx, list) and ctx:
            context_lines = [str(x) for x in ctx[-12:]]
        if isinstance(manifest.get("phrase_cache"), dict):
            phrase_cache = {
                k: v
                for k, v in manifest["phrase_cache"].items()
                if isinstance(v, str) and not OllamaTranslator._is_bubble_refusal_value(v)
            }
        if rel_norm and isinstance(manifest.get("pages"), dict) and rel_norm in manifest["pages"]:
            already_had_translation = True
            prev_page_blocks = (manifest["pages"].get(rel_norm) or {}).get("blocks") or []
            if isinstance(prev_page_blocks, list):
                for pb in prev_page_blocks:
                    if isinstance(pb, dict):
                        pb_orig = str(pb.get("original_text") or "").strip()
                        if pb_orig:
                            ck_evict = f"{target_lang}||{pb_orig}"
                            phrase_cache.pop(ck_evict, None)
                            translator_engine.phrase_cache.pop(ck_evict, None)

    is_retranslate = bool(payload.force_retranslate or already_had_translation)

    final_image, _, _, blocks, active_model, detected_lang = await process_single_pil_image(
        pil_image=pil_image,
        lang=req_lang,
        target_lang=target_lang,
        ollama_model=(payload.ollama_model or payload.model),
        uppercase=payload.uppercase,
        min_font_size=payload.min_font_size,
        max_font_size=payload.max_font_size,
        overflow_mode=payload.overflow_mode,
        detect_free_text=payload.detect_free_text,
        text_filter_mode=payload.text_filter_mode,
        skip_sighs=payload.skip_sighs,
        skip_sfx=payload.skip_sfx,
        min_text_length=payload.min_text_length,
        context_lines=context_lines,
        external_cache=phrase_cache,
        src_file=src_file,
        glossary_entries=normalize_glossary_entries(manifest.get("glossary", [])) if manifest else None,
        use_cache=(not is_retranslate),
    )

    if not blocks:
        shutil.copy2(src_file, dest_path)
    else:
        save_kwargs: dict[str, Any] = {}
        if dest_path.suffix.lower() in (".jpg", ".jpeg", ".webp"):
            save_kwargs["quality"] = 95
        final_image.save(dest_path, **save_kwargs)

    free_cnt = sum(1 for b in blocks if b.get("is_free_text"))
    if output_dir is not None and manifest is not None and rel_norm:
        if not isinstance(manifest.get("pages"), dict):
            manifest["pages"] = {}
        manifest["source_dir"] = str(source_dir) if source_dir else ""
        manifest["target_lang"] = target_lang
        manifest["pages"][rel_norm] = {
            "blocks_count": len(blocks),
            "free_text_count": free_cnt,
            "detected_lang": detected_lang,
            "model": active_model,
            "uppercase": payload.uppercase,
            "min_font_size": payload.min_font_size,
            "max_font_size": payload.max_font_size,
            "ocr_diag": dict(getattr(ocr_engine, "last_page_diagnostics", {}) or {}) if ocr_engine else {},
            "updated_at": int(time.time()),
            "blocks": blocks,
        }
        if translator_engine is not None and translator_engine.recent_context_lines:
            manifest["rolling_context"] = list(translator_engine.recent_context_lines[-12:])
        manifest["phrase_cache"] = phrase_cache
        _save_folder_manifest(output_dir, manifest)
        await asyncio.to_thread(
            _generate_and_save_folder_report,
            output_dir,
            manifest,
            source_dir,
            target_lang,
        )

    translated_b64 = (
        encode_pil_to_data_url(final_image, fmt=payload.image_format)
        if payload.return_image
        else None
    )
    inpaint_backend = "SimpleLama" if cleaner_engine.lama_model is not None else "OpenCV_TELEA"
    elapsed = round(time.perf_counter() - start_ts, 3)

    return TranslateResponse(
        status="ok",
        lang=req_lang,
        detected_lang=detected_lang,
        target_lang=target_lang,
        ollama_model=active_model,
        inpaint_backend=inpaint_backend,
        image_size=[pil_image.width, pil_image.height],
        processing_time_sec=elapsed,
        blocks_count=len(blocks),
        free_text_blocks_count=free_cnt,
        saved_output_path=str(dest_path),
        translated_image=translated_b64,
        blocks=[TextBlock(**b) for b in blocks],
    )


def _resolve_page_paths(
    source_path_str: str,
    source_dir_str: str | None,
    dest_path_str: str | None,
    target_lang: str,
) -> tuple[Path, Path, Path, Path, str]:
    src_file = Path(source_path_str).resolve()
    t_lang = (target_lang or "en").lower().strip()
    if t_lang not in VALID_TARGET_LANGS:
        t_lang = "en"

    if source_dir_str:
        source_dir = _unwrap_output_folder_if_needed(Path(source_dir_str).resolve())
    else:
        source_dir = _unwrap_output_folder_if_needed(src_file.parent)

    # Если исходный файл был выбран прямо из папки перевода (<Папка>_RU/001.png),
    # а в чистой папке <Папка>/001.png лежит оригинал — берем чистый оригинал!
    if src_file.parent != source_dir and (source_dir / src_file.name).exists():
        src_file = (source_dir / src_file.name).resolve()

    output_dir = get_output_dir_for_folder(source_dir, t_lang)
    try:
        rel = src_file.relative_to(source_dir)
    except ValueError:
        rel = Path(src_file.name)
    rel_norm = _normalize_rel_path(rel)

    if dest_path_str:
        dest_path = Path(dest_path_str).resolve()
    else:
        dest_path = output_dir / rel

    return src_file, source_dir, output_dir, dest_path, rel_norm


@app.post("/page_blocks")
async def get_or_scan_page_blocks_endpoint(payload: PageBlocksRequest) -> dict[str, Any]:
    """
    Возвращает список текстовых блоков страницы (оригинал, ромадзи, перевод, координаты) для ручного редактирования.
    - Если блоки уже сохранены в `.translation_state.json`, возвращает их мгновенно.
    - Если страница переводилась старой версией или еще не сканировалась, выполняет быстрый OCR исходника,
      подтягивает готовые переводы из `phrase_cache` (0 токенов!) и генерирует Ромадзи.
    """
    if ocr_engine is None or translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    src_file, source_dir, output_dir, dest_path, rel_norm = _resolve_page_paths(
        payload.source_path,
        payload.source_dir,
        payload.dest_path,
        payload.target_lang,
    )
    if not src_file.exists() or not src_file.is_file():
        raise HTTPException(status_code=400, detail=f"Исходный файл не найден: {src_file}")

    t_lang = (payload.target_lang or "en").lower().strip()
    if t_lang not in VALID_TARGET_LANGS:
        t_lang = "en"

    req_lang = (payload.lang or "auto").lower().strip()
    if req_lang not in VALID_SOURCE_LANGS:
        req_lang = "auto"

    manifest = _load_folder_manifest(output_dir)
    if not isinstance(manifest.get("pages"), dict):
        manifest["pages"] = {}
    if not isinstance(manifest.get("phrase_cache"), dict):
        manifest["phrase_cache"] = {}
    phrase_cache: dict[str, str] = dict(manifest["phrase_cache"])

    page_meta = manifest["pages"].get(rel_norm, {})
    if not isinstance(page_meta, dict):
        page_meta = {}

    saved_blocks = page_meta.get("blocks")
    det_lang = str(page_meta.get("detected_lang") or "ja")

    if isinstance(saved_blocks, list) and not payload.force_rescan:
        blocks = []
        for b in saved_blocks:
            if not isinstance(b, dict):
                continue
            b_copy = _ensure_block_clean_bbox(dict(b))
            tr_val = str(b_copy.get("translated_text") or "").strip()
            if b_copy.get("is_deleted") or not tr_val:
                b_copy["is_deleted"] = True
                b_copy["translated_text"] = ""
            else:
                b_copy["is_deleted"] = False
                b_copy["translated_text"] = tr_val
            blocks.append(b_copy)
    else:
        eff_free, eff_mode, eff_skip_sighs, eff_skip_sfx, eff_min_len = _resolve_filter_params(
            detect_free_text=payload.detect_free_text,
            text_filter_mode=payload.text_filter_mode,
            skip_sighs=payload.skip_sighs,
            skip_sfx=payload.skip_sfx,
            min_text_length=payload.min_text_length,
        )
        with Image.open(src_file) as raw_im:
            pil_image = _pil_to_rgb_clean(raw_im)
        async with gpu_lock:
            scanned_blocks, det_lang = await asyncio.to_thread(
                _run_ocr_compat,
                ocr_engine,
                pil_image,
                req_lang,
                eff_free,
                eff_mode,
                eff_skip_sighs,
                eff_skip_sfx,
                eff_min_len,
            )
        blocks = []
        for b in scanned_blocks:
            b_copy = _ensure_block_clean_bbox(dict(b))
            orig_t = str(b_copy.get("original_text") or "").strip()
            cached_trans = (
                phrase_cache.get(f"{t_lang}||{orig_t}")
                or translator_engine.phrase_cache.get(f"{t_lang}||{orig_t}")
                or ""
            )
            b_copy["translated_text"] = str(b_copy.get("translated_text") or cached_trans).strip()
            b_copy["is_deleted"] = False
            cached_rom = (
                phrase_cache.get(f"romaji||{orig_t}")
                or translator_engine.phrase_cache.get(f"romaji||{orig_t}")
                or ""
            )
            b_copy["romaji_text"] = cached_rom or japanese_to_romaji_local(orig_t)
            blocks.append(b_copy)

    # Убеждаемся, что у каждого блока заполнен romaji_text и clean_bbox
    for b in blocks:
        _ensure_block_clean_bbox(b)
        orig_t = str(b.get("original_text") or "").strip()
        if orig_t and not str(b.get("romaji_text") or "").strip():
            cached_rom = (
                phrase_cache.get(f"romaji||{orig_t}")
                or translator_engine.phrase_cache.get(f"romaji||{orig_t}")
                or ""
            )
            b["romaji_text"] = cached_rom or japanese_to_romaji_local(orig_t)

    if payload.use_llm_romaji and blocks:
        blocks = await translator_engine.transliterate_blocks_to_romaji(
            blocks=blocks,
            use_llm=True,
            model=payload.ollama_model,
            external_cache=phrase_cache,
        )

    page_meta.update(
        {
            "blocks_count": len(blocks),
            "free_text_count": sum(1 for b in blocks if b.get("is_free_text")),
            "detected_lang": det_lang,
            "updated_at": int(time.time()),
            "blocks": blocks,
        }
    )
    manifest["pages"][rel_norm] = page_meta
    manifest["phrase_cache"] = phrase_cache
    _save_folder_manifest(output_dir, manifest)

    return {
        "status": "ok",
        "source_path": str(src_file),
        "dest_path": str(dest_path),
        "rel_path": rel_norm,
        "detected_lang": det_lang,
        "blocks_count": len(blocks),
        "blocks": [TextBlock(**b) for b in blocks],
    }


@app.post("/page_romaji")
async def generate_page_romaji_endpoint(payload: PageRomajiRequest) -> dict[str, Any]:
    """
    Генерирует или уточняет чтение Ромадзи (`romaji_text`) для всех блоков выбранной страницы
    (локальный словарь + контекстное чтение кандзи через ИИ при `use_llm=True`).
    """
    if translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    blocks_dicts = [_ensure_block_clean_bbox(b.model_dump() if hasattr(b, "model_dump") else b.dict()) for b in payload.blocks]
    phrase_cache: dict[str, str] = {}
    output_dir: Path | None = None
    rel_norm: str | None = None
    manifest: dict[str, Any] | None = None

    if payload.source_path:
        _, _, output_dir, _, rel_norm = _resolve_page_paths(
            payload.source_path,
            payload.source_dir,
            None,
            payload.target_lang,
        )
        manifest = _load_folder_manifest(output_dir)
        if isinstance(manifest.get("phrase_cache"), dict):
            phrase_cache = dict(manifest["phrase_cache"])

    updated_blocks = await translator_engine.transliterate_blocks_to_romaji(
        blocks=blocks_dicts,
        use_llm=payload.use_llm,
        model=payload.ollama_model,
        external_cache=phrase_cache,
    )

    if output_dir is not None and manifest is not None and rel_norm:
        if not isinstance(manifest.get("pages"), dict):
            manifest["pages"] = {}
        page_meta = manifest["pages"].get(rel_norm, {})
        if not isinstance(page_meta, dict):
            page_meta = {}
        page_meta["blocks"] = updated_blocks
        page_meta["updated_at"] = int(time.time())
        manifest["pages"][rel_norm] = page_meta
        manifest["phrase_cache"] = phrase_cache
        _save_folder_manifest(output_dir, manifest)

    return {
        "status": "ok",
        "blocks_count": len(updated_blocks),
        "blocks": [TextBlock(**b) for b in updated_blocks],
    }


@app.post("/save_edited_page")
async def save_edited_page_endpoint(payload: SaveEditedPageRequest) -> dict[str, Any]:
    """
    Принимает отредактированные пользователем текстовые блоки страницы (включая новые координаты/размер рамки `bbox`,
    сохраняя очистку оригинала строго на `clean_bbox`), заново очищает бабблы на чистом оригинале,
    верстает новый текст и сохраняет обновленную картинку в папку перевода и в `.translation_state.json`.
    Не требует повторного вызова OCR или LLM (0 токенов API!).
    """
    if cleaner_engine is None or renderer_engine is None or translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    src_file, source_dir, output_dir, dest_path, rel_norm = _resolve_page_paths(
        payload.source_path,
        payload.source_dir,
        payload.dest_path,
        payload.target_lang,
    )
    if not src_file.exists() or not src_file.is_file():
        raise HTTPException(status_code=400, detail=f"Исходный файл страницы не найден: {src_file}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    start_ts = time.perf_counter()

    with Image.open(src_file) as raw_im:
        pil_image = _pil_to_rgb_clean(raw_im)

    blocks_dicts: list[dict[str, Any]] = []
    active_blocks: list[dict[str, Any]] = []
    for b in payload.blocks:
        bd = _ensure_block_clean_bbox(b.model_dump() if hasattr(b, "model_dump") else b.dict())
        orig_t = str(bd.get("original_text") or "").strip()
        if orig_t and not str(bd.get("romaji_text") or "").strip():
            bd["romaji_text"] = japanese_to_romaji_local(orig_t)
        clean_trans = str(bd.get("translated_text") or "").strip()
        if bool(bd.get("is_deleted")) or not clean_trans:
            bd["is_deleted"] = True
            bd["translated_text"] = ""
        else:
            bd["is_deleted"] = False
            bd["translated_text"] = clean_trans
            active_blocks.append(bd)
        blocks_dicts.append(bd)

    clean_cache_hit = False
    clean_state_hash = ""
    if not active_blocks:
        final_image = pil_image.copy()
        clean_cache_hit = True
    else:
        cleaned_image, _, clean_cache_hit, clean_state_hash = await _get_or_clean_page_image(
            pil_image,
            active_blocks,
            src_file=src_file,
            save_to_disk=False,
        )
        eff_overflow = payload.overflow_mode or (getattr(translator_engine, "overflow_mode", "shrink") if translator_engine else "shrink")
        final_image = await asyncio.to_thread(
            renderer_engine.render_blocks,
            cleaned_image,
            active_blocks,
            payload.uppercase,
            payload.max_font_size,
            payload.min_font_size,
            None,
            eff_overflow,
        )

    save_kwargs: dict[str, Any] = {}
    if dest_path.suffix.lower() in (".jpg", ".jpeg", ".webp"):
        save_kwargs["quality"] = 95
    elif dest_path.suffix.lower() == ".png":
        save_kwargs["compress_level"] = 1
    final_image.save(dest_path, **save_kwargs)

    t_lang = (payload.target_lang or "en").lower().strip()
    manifest = _load_folder_manifest(output_dir)
    if not isinstance(manifest.get("pages"), dict):
        manifest["pages"] = {}
    if not isinstance(manifest.get("phrase_cache"), dict):
        manifest["phrase_cache"] = {}

    for bd in blocks_dicts:
        orig_t = str(bd.get("original_text") or "").strip()
        trans_t = str(bd.get("translated_text") or "").strip()
        rom_t = str(bd.get("romaji_text") or "").strip()
        if orig_t and trans_t:
            manifest["phrase_cache"][f"{t_lang}||{orig_t}"] = trans_t
            translator_engine.phrase_cache[f"{t_lang}||{orig_t}"] = trans_t
        if orig_t and rom_t:
            manifest["phrase_cache"][f"romaji||{orig_t}"] = rom_t
            translator_engine.phrase_cache[f"romaji||{orig_t}"] = rom_t

    prev_meta = manifest["pages"].get(rel_norm, {})
    if not isinstance(prev_meta, dict):
        prev_meta = {}
    prev_meta.update(
        {
            "manually_edited": True,
            "blocks_count": len(blocks_dicts),
            "free_text_count": sum(1 for b in blocks_dicts if b.get("is_free_text")),
            "uppercase": payload.uppercase,
            "min_font_size": payload.min_font_size,
            "max_font_size": payload.max_font_size,
            "updated_at": int(time.time()),
            "blocks": blocks_dicts,
        }
    )
    manifest["pages"][rel_norm] = prev_meta
    _save_folder_manifest(output_dir, manifest)
    threading.Thread(
        target=_generate_and_save_folder_report,
        args=(output_dir, copy.deepcopy(manifest), source_dir, t_lang),
        daemon=True,
    ).start()

    translated_image_url: str | None = (
        f"/local_image?path={quote(str(dest_path))}&t={int(time.time() * 1000)}"
        if dest_path.exists()
        else None
    )
    if payload.return_image:
        if payload.prefer_url and translated_image_url:
            translated_b64 = None
        else:
            translated_b64 = encode_pil_to_data_url(final_image, fmt=payload.image_format)
    else:
        translated_b64 = None

    elapsed = round(time.perf_counter() - start_ts, 3)
    logger.info(
        "💾 [SAVE EDITED PAGE] %s | Время=%.3fс | Кэш очистки=%s | Блоков=%d",
        rel_norm,
        elapsed,
        "HIT (0мс SimpleLama)" if clean_cache_hit else "MISS (SimpleLama)",
        len(blocks_dicts),
    )

    return {
        "status": "ok",
        "saved_output_path": str(dest_path),
        "processing_time_sec": elapsed,
        "clean_cache_hit": clean_cache_hit,
        "clean_state_hash": clean_state_hash,
        "blocks_count": len(blocks_dicts),
        "translated_image_url": translated_image_url,
        "translated_image": translated_b64,
        "blocks": [TextBlock(**b) for b in blocks_dicts],
    }


@app.post("/translate_manual_region")
async def translate_manual_region_endpoint(payload: ManualRegionTranslateRequest) -> dict[str, Any]:
    """
    Принудительно распознает (OCR без фильтров) и переводит (через активную ИИ-модель) область `bbox = [x1, y1, x2, y2]`,
    которую пользователь обвел мышкой на странице в редакторе, добавляет новый блок к существующим,
    перерисовывает страницу и сохраняет результат.
    """
    if (
        ocr_engine is None
        or translator_engine is None
        or cleaner_engine is None
        or renderer_engine is None
    ):
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    if not payload.bbox or len(payload.bbox) < 4:
        raise HTTPException(status_code=400, detail="Некорректные координаты обведенной области bbox.")

    src_file, source_dir, output_dir, dest_path, rel_norm = _resolve_page_paths(
        payload.source_path,
        payload.source_dir,
        payload.dest_path,
        payload.target_lang,
    )
    if not src_file.exists() or not src_file.is_file():
        raise HTTPException(status_code=400, detail=f"Исходный файл не найден: {src_file}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    start_ts = time.perf_counter()

    with Image.open(src_file) as raw_im:
        pil_image = _pil_to_rgb_clean(raw_im)

    img_w, img_h = pil_image.size
    x1, y1, x2, y2 = [int(round(c)) for c in payload.bbox[:4]]
    x1 = max(0, min(x1, img_w - 4))
    y1 = max(0, min(y1, img_h - 4))
    x2 = max(x1 + 8, min(x2, img_w))
    y2 = max(y1 + 8, min(y2, img_h))
    clamped_bbox = [x1, y1, x2, y2]

    async with gpu_lock:
        recognized_text, det_lang, is_free_text = await asyncio.to_thread(
            ocr_engine.recognize_manual_region,
            pil_image,
            clamped_bbox,
            payload.lang,
            payload.use_vlm_ocr,
            payload.vlm_model,
            getattr(translator_engine, "base_url", "http://127.0.0.1:11434") if translator_engine else "http://127.0.0.1:11434",
            translator_engine.unsloth_url if translator_engine else None,
            translator_engine.unsloth_api_key if translator_engine else None,
        )

    orig_text = (payload.custom_original_text or recognized_text or "").strip()
    existing_blocks: list[dict[str, Any]] = [
        _ensure_block_clean_bbox(b.model_dump() if hasattr(b, "model_dump") else b.dict()) for b in payload.blocks
    ]
    next_id = max([int(b.get("id", 0)) for b in existing_blocks], default=0) + 1

    new_block: dict[str, Any] = {
        "id": next_id,
        "bbox": list(clamped_bbox),
        "clean_bbox": list(clamped_bbox),
        "bbox_modified": False,
        "confidence": 1.0,
        "original_text": orig_text or "[Выделенная область]",
        "romaji_text": japanese_to_romaji_local(orig_text) if orig_text else "",
        "translated_text": (payload.custom_translated_text or "").strip(),
        "is_free_text": is_free_text,
        "is_deleted": False,
        "font_size": None,
    }

    manifest = _load_folder_manifest(output_dir)
    if not isinstance(manifest.get("pages"), dict):
        manifest["pages"] = {}
    if not isinstance(manifest.get("phrase_cache"), dict):
        manifest["phrase_cache"] = {}
    phrase_cache: dict[str, str] = dict(manifest["phrase_cache"])
    rolling_ctx = manifest.get("rolling_context")
    context_lines = [
        str(x)
        for x in (rolling_ctx[-10:] if isinstance(rolling_ctx, list) else [])
        if str(x) and not str(x).lower().strip().endswith("=> twitch")
    ] or None

    if not new_block["translated_text"] and orig_text:
        active_mod = await translator_engine.resolve_model(payload.ollama_model)
        # Принудительно переводим выделенный кружок без пропуска вздохов и без старого кэша
        translated_list = await translator_engine.translate_blocks(
            blocks=[new_block],
            source_lang=det_lang,
            target_lang=payload.target_lang,
            model=active_mod,
            context_lines=context_lines,
            external_cache=phrase_cache,
            use_cache=False,
            skip_sighs=False,
        )
        if translated_list:
            new_block = _ensure_block_clean_bbox(dict(translated_list[0]))
            new_block["is_deleted"] = False
        if not str(new_block.get("translated_text") or "").strip():
            alt_mod = None
            if not active_mod.startswith("unsloth::") and translator_engine.unsloth_api_key:
                alt_mod = "unsloth::deepseek-v4-flash"
            elif active_mod.startswith("unsloth::"):
                alt_mod = await translator_engine._resolve_local_fallback_model(target_lang=payload.target_lang, source_lang=det_lang)
            if alt_mod and alt_mod != active_mod:
                alt_list = await translator_engine.translate_blocks(
                    blocks=[new_block],
                    source_lang=det_lang,
                    target_lang=payload.target_lang,
                    model=alt_mod,
                    context_lines=context_lines,
                    external_cache=phrase_cache,
                    use_cache=False,
                    skip_sighs=False,
                )
                if alt_list and str(alt_list[0].get("translated_text") or "").strip():
                    new_block = _ensure_block_clean_bbox(dict(alt_list[0]))
                    new_block["is_deleted"] = False

    existing_blocks.append(new_block)

    active_blocks: list[dict[str, Any]] = []
    for bd in existing_blocks:
        clean_tr = str(bd.get("translated_text") or "").strip()
        if not bool(bd.get("is_deleted")) and clean_tr:
            bd["is_deleted"] = False
            bd["translated_text"] = clean_tr
            active_blocks.append(bd)

    if not active_blocks:
        final_image = pil_image.copy()
    else:
        cleaned_image, _, _, _ = await _get_or_clean_page_image(
            pil_image,
            active_blocks,
            src_file=src_file,
            save_to_disk=True,
        )
        eff_overflow = payload.overflow_mode or (getattr(translator_engine, "overflow_mode", "shrink") if translator_engine else "shrink")
        final_image = await asyncio.to_thread(
            renderer_engine.render_blocks,
            cleaned_image,
            active_blocks,
            payload.uppercase,
            payload.max_font_size,
            payload.min_font_size,
            None,
            eff_overflow,
        )

    save_kwargs: dict[str, Any] = {}
    if dest_path.suffix.lower() in (".jpg", ".jpeg", ".webp"):
        save_kwargs["quality"] = 95
    final_image.save(dest_path, **save_kwargs)

    t_lang = (payload.target_lang or "en").lower().strip()
    if orig_text and new_block.get("translated_text"):
        manifest["phrase_cache"][f"{t_lang}||{orig_text}"] = new_block["translated_text"]
        translator_engine.phrase_cache[f"{t_lang}||{orig_text}"] = new_block["translated_text"]

    prev_meta = manifest["pages"].get(rel_norm, {})
    if not isinstance(prev_meta, dict):
        prev_meta = {}
    prev_meta.update(
        {
            "manually_edited": True,
            "blocks_count": len(existing_blocks),
            "free_text_count": sum(1 for b in existing_blocks if b.get("is_free_text")),
            "detected_lang": det_lang,
            "uppercase": payload.uppercase,
            "min_font_size": payload.min_font_size,
            "max_font_size": payload.max_font_size,
            "updated_at": int(time.time()),
            "blocks": existing_blocks,
        }
    )
    manifest["pages"][rel_norm] = prev_meta
    _save_folder_manifest(output_dir, manifest)
    await asyncio.to_thread(
        _generate_and_save_folder_report,
        output_dir,
        manifest,
        source_dir,
        t_lang,
    )

    translated_b64 = (
        encode_pil_to_data_url(final_image, fmt=payload.image_format)
        if payload.return_image
        else None
    )
    elapsed = round(time.perf_counter() - start_ts, 3)

    return {
        "status": "ok",
        "saved_output_path": str(dest_path),
        "processing_time_sec": elapsed,
        "new_block": TextBlock(**new_block),
        "blocks_count": len(existing_blocks),
        "translated_image": translated_b64,
        "blocks": [TextBlock(**b) for b in existing_blocks],
    }


@app.post("/retranslate_single_block")
async def retranslate_single_block_endpoint(payload: RetranslateSingleBlockRequest) -> dict[str, Any]:
    t_lang = (payload.target_lang or "en").lower().strip()
    """
    Принудительно переводит один конкретный блок `#block_id` через активную ИИ-модель
    (с отключенным пропуском вздохов и сбросом кэша для этой фразы) и сразу перерисовывает страницу.
    """
    if (
        ocr_engine is None
        or translator_engine is None
        or cleaner_engine is None
        or renderer_engine is None
    ):
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    src_file, source_dir, output_dir, dest_path, rel_norm = _resolve_page_paths(
        payload.source_path,
        payload.source_dir,
        payload.dest_path,
        payload.target_lang,
    )
    if not src_file.exists() or not src_file.is_file():
        raise HTTPException(status_code=400, detail=f"Исходный файл не найден: {src_file}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    start_ts = time.perf_counter()

    with Image.open(src_file) as raw_im:
        pil_image = _pil_to_rgb_clean(raw_im)

    blocks_dicts: list[dict[str, Any]] = [
        _ensure_block_clean_bbox(b.model_dump() if hasattr(b, "model_dump") else b.dict()) for b in payload.blocks
    ]
    target_blk = next((b for b in blocks_dicts if int(b.get("id", -1)) == int(payload.block_id)), None)
    if target_blk is None:
        raise HTTPException(status_code=404, detail=f"Блок #{payload.block_id} не найден.")

    orig_t = str(target_blk.get("original_text") or "").strip()
    old_tr_val = str(target_blk.get("translated_text") or "").strip()
    det_lang = payload.lang if payload.lang in ("ja", "ko", "zh") else "ja"
    norm_old_tr = re.sub(r"[^a-zA-Zа-яА-ЯёЁ]+", "", old_tr_val).lower()
    norm_orig_core = re.sub(r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』]+", "", orig_t)
    # Если блок ранее был ошибочно прочитан как короткий SFX («ビクッ»/«ピクッ» -> «Twitch») или пуст,
    # то даже при нажатии на обычную кнопку «🔄 ИИ» перечитываем ПОЛНЫЙ bbox баббла (с подключением VLM-OCR)!
    suspect_ocr = (
        not orig_t
        or orig_t == "[Выделенная область]"
        or norm_old_tr in ("twitch", "twitches", "twitching")
        or norm_orig_core in ("ビクッ", "ピクッ", "ぴくっ", "びくっ", "ビク", "ピク")
    )
    if payload.use_vlm_ocr or suspect_ocr:
        # Всегда берем полный bbox баббла, а не урезанный text_bbox, который мог обрезать края иероглифов!
        ocr_coords = target_blk["bbox"]
        force_vlm = bool(payload.use_vlm_ocr or norm_old_tr in ("twitch", "twitches", "twitching"))
        async with gpu_lock:
            rec_t, det_lang, _ = await asyncio.to_thread(
                ocr_engine.recognize_manual_region,
                pil_image,
                ocr_coords,
                payload.lang,
                force_vlm,
                payload.vlm_model,
                getattr(translator_engine, "base_url", "http://127.0.0.1:11434") if translator_engine else "http://127.0.0.1:11434",
                translator_engine.unsloth_url if translator_engine else None,
                translator_engine.unsloth_api_key if translator_engine else None,
            )
        if rec_t and rec_t != "[Выделенная область]":
            orig_t = rec_t
            target_blk["original_text"] = orig_t
            target_blk["romaji_text"] = japanese_to_romaji_local(orig_t)

    manifest = _load_folder_manifest(output_dir)
    if not isinstance(manifest.get("pages"), dict):
        manifest["pages"] = {}
    if not isinstance(manifest.get("phrase_cache"), dict):
        manifest["phrase_cache"] = {}
    rolling_ctx = manifest.get("rolling_context")
    context_lines = [
        str(x)
        for x in (rolling_ctx[-10:] if isinstance(rolling_ctx, list) else [])
        if str(x) and not str(x).lower().strip().endswith("=> twitch")
    ] or None
    folder_glossary_entries = normalize_glossary_entries(manifest.get("glossary", []))

    if orig_t and orig_t != "[Выделенная область]":
        ck_single = f"{t_lang}||{orig_t}"
        manifest["phrase_cache"].pop(ck_single, None)
        translator_engine.phrase_cache.pop(ck_single, None)
        active_mod = await translator_engine.resolve_model(payload.ollama_model)
        blk_for_llm = dict(target_blk)
        blk_for_llm["translated_text"] = ""
        blk_for_llm["is_deleted"] = False
        translated_one = await translator_engine.translate_blocks(
            blocks=[blk_for_llm],
            source_lang=det_lang,
            target_lang=payload.target_lang,
            model=active_mod,
            context_lines=context_lines,
            glossary_entries=folder_glossary_entries,
            use_cache=False,
            skip_sighs=False,
        )
        new_tr_val = str(translated_one[0].get("translated_text") or "").strip() if translated_one else ""
        # Если выбранная модель вернула пустую строку или снова выдала старый ошибочный «Twitch»,
        # автоматически пробуем альтернативный провайдер (Cloud DeepSeek <-> Local Ollama)!
        if not new_tr_val or (norm_old_tr == "twitch" and new_tr_val.strip().lower() == "twitch"):
            alt_mod = None
            if not active_mod.startswith("unsloth::") and translator_engine.unsloth_api_key:
                alt_mod = "unsloth::deepseek-v4-flash"
            elif active_mod.startswith("unsloth::"):
                alt_mod = await translator_engine._resolve_local_fallback_model(target_lang=t_lang, source_lang=det_lang)
            if alt_mod and alt_mod != active_mod:
                translated_alt = await translator_engine.translate_blocks(
                    blocks=[blk_for_llm],
                    source_lang=det_lang,
                    target_lang=payload.target_lang,
                    model=alt_mod,
                    context_lines=context_lines,
                    glossary_entries=folder_glossary_entries,
                    use_cache=False,
                    skip_sighs=False,
                )
                if translated_alt:
                    alt_val = str(translated_alt[0].get("translated_text") or "").strip()
                    if alt_val:
                        translated_one = translated_alt
                        new_tr_val = alt_val

        if new_tr_val:
            target_blk["translated_text"] = new_tr_val
            target_blk["romaji_text"] = translated_one[0].get("romaji_text") or target_blk.get("romaji_text") or ""
            target_blk["translation_source"] = translated_one[0].get("translation_source") or "llm"
            target_blk["skip_reason"] = ""
            target_blk["is_deleted"] = False
            manifest["phrase_cache"][ck_single] = new_tr_val
            translator_engine.phrase_cache[ck_single] = new_tr_val

    active_blocks = [
        b for b in blocks_dicts if not bool(b.get("is_deleted")) and str(b.get("translated_text") or "").strip()
    ]
    if not active_blocks:
        final_image = pil_image.copy()
    else:
        cleaned_image, _, _, _ = await _get_or_clean_page_image(
            pil_image,
            active_blocks,
            src_file=src_file,
            save_to_disk=True,
        )
        eff_overflow = payload.overflow_mode or (getattr(translator_engine, "overflow_mode", "shrink") if translator_engine else "shrink")
        final_image = await asyncio.to_thread(
            renderer_engine.render_blocks,
            cleaned_image,
            active_blocks,
            payload.uppercase,
            payload.max_font_size,
            payload.min_font_size,
            None,
            eff_overflow,
        )

    save_kwargs: dict[str, Any] = {}
    if dest_path.suffix.lower() in (".jpg", ".jpeg", ".webp"):
        save_kwargs["quality"] = 95
    final_image.save(dest_path, **save_kwargs)

    prev_meta = manifest["pages"].get(rel_norm, {})
    if not isinstance(prev_meta, dict):
        prev_meta = {}
    prev_meta["manually_edited"] = True
    prev_meta["blocks"] = blocks_dicts
    prev_meta["updated_at"] = int(time.time())
    manifest["pages"][rel_norm] = prev_meta
    _save_folder_manifest(output_dir, manifest)
    await asyncio.to_thread(
        _generate_and_save_folder_report,
        output_dir,
        manifest,
        source_dir,
        t_lang,
    )

    translated_b64 = (
        encode_pil_to_data_url(final_image, fmt=payload.image_format)
        if payload.return_image
        else None
    )
    elapsed = round(time.perf_counter() - start_ts, 3)

    return {
        "status": "ok",
        "saved_output_path": str(dest_path),
        "processing_time_sec": elapsed,
        "updated_block": TextBlock(**target_blk),
        "translated_image": translated_b64,
        "blocks": [TextBlock(**b) for b in blocks_dicts],
    }


@app.post("/translate_folder")
async def translate_folder_endpoint(payload: FolderTranslateRequest) -> dict[str, Any]:
    global _folder_task

    if folder_job_state.get("status") == "running":
        raise HTTPException(status_code=409, detail="Перевод папки уже выполняется.")

    req_lang = payload.lang.lower().strip()
    if req_lang not in VALID_SOURCE_LANGS:
        req_lang = "auto"

    target_lang = payload.target_lang.lower().strip()
    if target_lang not in VALID_TARGET_LANGS:
        raise HTTPException(
            status_code=400,
            detail="target_lang должен быть 'en', 'ru' или 'romaji'.",
        )

    folder_str = payload.folder_path
    if not folder_str:
        folder_str = await asyncio.to_thread(open_native_folder_picker)
        if not folder_str:
            return {"status": "cancelled", "message": "Выбор папки отменен."}

    source_dir = Path(folder_str).resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Папка не найдена: {source_dir}")

    all_images = collect_images_in_folder(source_dir)
    if not all_images:
        raise HTTPException(status_code=400, detail=f"В папке '{source_dir.name}' нет картинок.")

    if payload.selected_files:
        wanted = {_normalize_rel_path(x) for x in payload.selected_files if str(x).strip()}
        images = [
            p
            for p in all_images
            if _normalize_rel_path(p.relative_to(source_dir)) in wanted or p.name in wanted
        ]
        if not images:
            raise HTTPException(status_code=400, detail="Выбранные страницы не найдены в папке.")
    else:
        images = all_images

    output_dir = get_output_dir_for_folder(source_dir, target_lang)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_preview = _load_folder_manifest(output_dir)
    pages_meta_preview = manifest_preview.get("pages", {}) if isinstance(manifest_preview.get("pages"), dict) else {}
    already_done = sum(
        1
        for p in images
        if payload.skip_existing
        and not payload.force_retranslate
        and is_page_already_translated(
            output_dir / p.relative_to(source_dir),
            page_meta=pages_meta_preview.get(_normalize_rel_path(p.relative_to(source_dir))),
            target_lang=target_lang,
        )
    )

    _folder_task = asyncio.create_task(
        _run_folder_translation_job(
            source_dir=source_dir,
            output_dir=output_dir,
            image_paths=images,
            lang=req_lang,
            target_lang=target_lang,
            ollama_model=(payload.ollama_model or payload.model),
            uppercase=payload.uppercase,
            min_font_size=payload.min_font_size,
            max_font_size=payload.max_font_size,
            detect_free_text=payload.detect_free_text,
            overflow_mode=payload.overflow_mode,
            skip_existing=payload.skip_existing,
            force_retranslate=payload.force_retranslate,
            open_when_done=payload.open_when_done,
            batch_pages=payload.batch_pages,
            translation_profile=payload.translation_profile,
            text_filter_mode=payload.text_filter_mode,
            skip_sighs=payload.skip_sighs,
            skip_sfx=payload.skip_sfx,
            min_text_length=payload.min_text_length,
        )
    )

    return {
        "status": "started",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "total_images": len(images),
        "already_translated": already_done,
        "remaining_images": max(0, len(images) - already_done),
        "lang": req_lang,
        "target_lang": target_lang,
    }


@app.post("/folder_stop")
async def stop_folder_translation_endpoint() -> dict[str, Any]:
    global _folder_stop_requested
    if folder_job_state.get("status") != "running":
        return {"status": "idle", "message": "Пакетный перевод сейчас не выполняется."}
    _folder_stop_requested = True
    if translator_engine is not None:
        translator_engine.stop_requested = True
    folder_job_state["status"] = "stopping"
    folder_job_state["current_file"] = "Пауза после текущей страницы..."
    return {"status": "stopping", "message": "Запрос на остановку отправлен. Прогресс сохранен."}


@app.get("/folder_status")
async def get_folder_status() -> dict[str, Any]:
    return folder_job_state


def _build_diagnostics_payload() -> dict[str, Any]:
    cuda_avail = torch.cuda.is_available()
    gpu_data: dict[str, Any] = {
        "cuda_available": cuda_avail,
        "torch_version": str(torch.__version__),
    }
    if cuda_avail:
        try:
            gpu_data["device_name"] = torch.cuda.get_device_name(0)
            gpu_data["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2), 1)
            gpu_data["vram_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024**2), 1)
            gpu_data["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1)
        except Exception:
            pass

    inpaint_backend = (
        "SimpleLama (LaMa GPU)"
        if (cleaner_engine is not None and cleaner_engine.lama_model is not None)
        else "Hybrid Pure White Fill + OpenCV INPAINT_TELEA"
    )
    easyocr_ok = bool(ocr_engine and hasattr(ocr_engine, "_check_easyocr") and ocr_engine._check_easyocr())
    log_size_kb = round(MAIN_LOG_FILE.stat().st_size / 1024.0, 1) if MAIN_LOG_FILE.exists() else 0.0

    trans_diag = translator_engine.get_diagnostics_summary() if translator_engine is not None else {}

    return {
        "status": "ok",
        "server": {
            "version": "1.0.6",
            "pid": os.getpid(),
            "uptime_sec": round(time.time() - SERVER_STARTED_AT, 1),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(SERVER_STARTED_AT)),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "base_dir": str(BASE_DIR),
            "log_file_path": str(MAIN_LOG_FILE),
            "events_file_path": str(DIAG_EVENTS_FILE),
            "log_file_size_kb": log_size_kb,
        },
        "gpu": {
            **gpu_data,
            "inpaint_backend": inpaint_backend,
            "easyocr_installed": easyocr_ok,
            "detector_backend": (
                ocr_engine.get_detector_backend_name()
                if (ocr_engine is not None and hasattr(ocr_engine, "get_detector_backend_name"))
                else "YOLOv8m"
            ),
            "ja_ocr_backend": (
                ocr_engine.get_ja_ocr_backend_name()
                if (ocr_engine is not None and hasattr(ocr_engine, "get_ja_ocr_backend_name"))
                else "MangaOCR FP16"
            ),
            "ko_zh_ocr_backend": (
                ocr_engine.get_multilang_ocr_backend_name()
                if (ocr_engine is not None and hasattr(ocr_engine, "get_multilang_ocr_backend_name"))
                else "HayaiOCR v2.5 Nova + SmartLineSplit"
            ),
            "vlm_ocr_backend": (
                ocr_engine.get_vlm_ocr_backend_name()
                if (ocr_engine is not None and hasattr(ocr_engine, "get_vlm_ocr_backend_name"))
                else "Sniper VLM-OCR (Ollama Vision / Cloud Vision)"
            ),
            "last_detection_stats": (
                dict(ocr_engine.last_detection_stats)
                if (ocr_engine is not None and hasattr(ocr_engine, "last_detection_stats"))
                else {}
            ),
        },
        "translator": trans_diag,
        "ux_telemetry": {
            "total_events": int(_load_ux_store().get("total_events") or 0),
            "report_file_path": str(UX_REPORT_TXT_FILE),
            "top_actions": [
                {"label": a.get("label"), "count": a.get("count"), "zone": a.get("zone")}
                for a in sorted(
                    (_load_ux_store().get("actions") or {}).values(),
                    key=lambda x: int(x.get("count") or 0),
                    reverse=True,
                )[:8]
            ],
        },
        "folder_job": folder_job_state,
        "recent_jobs": list(reversed(list(_recent_job_profiles)[-25:])),
        "recent_warnings_errors": list(reversed(list(_recent_warn_err_entries)[-40:])),
        "recent_logs": [e["formatted"] for e in list(_recent_log_entries)[-150:]],
    }




class GlossarySaveRequest(BaseModel):
    global_glossary: list[dict[str, Any]] | None = Field(
        default=None,
        description="Полный список записей глобального глоссария (config.json)",
    )
    folder_glossary: list[dict[str, Any]] | None = Field(
        default=None,
        description="Полный список записей локального глоссария выбранной главы (.translation_state.json)",
    )
    folder_path: str | None = Field(
        default=None,
        description="Путь к исходной папке главы для сохранения локального глоссария",
    )
    target_lang: str = Field(default="en")
    import_text: str | None = Field(
        default=None,
        description="Сырой текст для импорта (.json, .tsv или формат Sakura `термин->перевод #примечание`)",
    )
    import_scope: str = Field(
        default="both",
        description="Куда импортировать записи из import_text: 'global', 'folder' или 'both'",
    )
    merge_import: bool = Field(
        default=True,
        description="Если True — объединить импортируемые записи с существующими; если False — заменить",
    )


class FolderFindReplaceRequest(BaseModel):
    folder_path: str = Field(description="Путь к папке главы манги")
    target_lang: str = Field(default="en")
    find_text: str = Field(description="Искомая подстрока (например, имя персонажа или термин)")
    replace_text: str = Field(default="", description="Текст замены")
    match_case: bool = Field(default=False, description="Учитывать ли регистр символов при поиске и замене")
    search_in: str = Field(
        default="translated",
        description="Где искать: 'translated' (в переведенном тексте) или 'original' (в японском/корейском оригинале)",
    )
    dry_run: bool = Field(
        default=False,
        description="Если True — только найти и вернуть список всех совпадений без изменения файлов",
    )
    uppercase: bool = Field(default=True)
    min_font_size: int = Field(default=12)
    max_font_size: int = Field(default=32)
    overflow_mode: str | None = Field(default=None)


@app.get("/glossary")
async def get_glossary_endpoint(
    folder_path: str | None = Query(default=None),
    target_lang: str = Query(default="en"),
) -> dict[str, Any]:
    """Возвращает глобальный глоссарий (из config.json), локальный глоссарий папки и их объединение + экспорт в Sakura/TSV."""
    if translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    global_list = normalize_glossary_entries(translator_engine.glossary)
    folder_list: list[dict[str, str]] = []
    resolved_output_dir: str | None = None

    if folder_path and folder_path.strip():
        src_dir = _unwrap_output_folder_if_needed(Path(folder_path.strip()).resolve())
        if src_dir.exists() and src_dir.is_dir():
            out_dir = get_output_dir_for_folder(src_dir, target_lang)
            resolved_output_dir = str(out_dir)
            manifest = _load_folder_manifest(out_dir)
            folder_list = normalize_glossary_entries(manifest.get("glossary", []))

    merged = merge_glossaries(global_list, folder_list)
    return {
        "status": "ok",
        "global_glossary": global_list,
        "folder_glossary": folder_list,
        "merged_glossary": merged,
        "output_dir": resolved_output_dir,
        "sakura_text": export_glossary_to_sakura_text(merged),
        "tsv_text": export_glossary_to_tsv(merged),
    }


@app.post("/glossary")
async def save_glossary_endpoint(payload: GlossarySaveRequest) -> dict[str, Any]:
    """Сохраняет или импортирует глобальный (`config.json`) и папочный (`.translation_state.json`) глоссарий."""
    if translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    t_lang = (payload.target_lang or "en").strip().lower()
    global_list = (
        normalize_glossary_entries(payload.global_glossary)
        if payload.global_glossary is not None
        else normalize_glossary_entries(translator_engine.glossary)
    )

    folder_list: list[dict[str, str]] = []
    out_dir: Path | None = None
    manifest: dict[str, Any] | None = None

    if payload.folder_path and payload.folder_path.strip():
        src_dir = _unwrap_output_folder_if_needed(Path(payload.folder_path.strip()).resolve())
        if src_dir.exists() and src_dir.is_dir():
            out_dir = get_output_dir_for_folder(src_dir, t_lang)
            manifest = _load_folder_manifest(out_dir)
            folder_list = (
                normalize_glossary_entries(payload.folder_glossary)
                if payload.folder_glossary is not None
                else normalize_glossary_entries(manifest.get("glossary", []))
            )
    elif payload.folder_glossary is not None:
        folder_list = normalize_glossary_entries(payload.folder_glossary)

    imported_count = 0
    if payload.import_text and payload.import_text.strip():
        imported_items = parse_glossary_text_or_json(payload.import_text, default_target_lang="all")
        imported_count = len(imported_items)
        scope = (payload.import_scope or "both").strip().lower()
        if scope in ("global", "both"):
            global_list = (
                merge_glossaries(global_list, imported_items)
                if payload.merge_import
                else normalize_glossary_entries(imported_items)
            )
        if scope in ("folder", "both") and out_dir is not None:
            folder_list = (
                merge_glossaries(folder_list, imported_items)
                if payload.merge_import
                else normalize_glossary_entries(imported_items)
            )

    translator_engine.save_config(glossary=global_list)

    if out_dir is not None and manifest is not None:
        manifest["glossary"] = folder_list
        _save_folder_manifest(out_dir, manifest)

    merged = merge_glossaries(global_list, folder_list)
    return {
        "status": "ok",
        "imported_count": imported_count,
        "global_glossary": global_list,
        "folder_glossary": folder_list,
        "merged_glossary": merged,
        "sakura_text": export_glossary_to_sakura_text(merged),
        "tsv_text": export_glossary_to_tsv(merged),
    }


@app.post("/folder_find_replace")
async def folder_find_replace_endpoint(payload: FolderFindReplaceRequest) -> dict[str, Any]:
    """
    Глобальный Поиск и Замена (`Ctrl+F` / `Ctrl+H`) по всем переведенным страницам папки без расхода токенов ИИ.
    - При `dry_run=True` мгновенно сканирует `.translation_state.json` и возвращает список всех совпадений.
    - При `dry_run=False` заменяет текст во всех совпавших блоках, обновляет `phrase_cache` и перерисовывает
      только затронутые страницы через кэш очищенного слоя `cleaned_image` (0 мс на SimpleLama) и `renderer_engine.render_blocks`.
    """
    if cleaner_engine is None or renderer_engine is None or translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    find_q = (payload.find_text or "").strip()
    if not find_q:
        raise HTTPException(status_code=400, detail="Поле 'find_text' не может быть пустым.")

    raw_folder = (payload.folder_path or "").strip()
    if not raw_folder:
        raise HTTPException(status_code=400, detail="Сначала выберите папку с главой манги.")

    source_dir = _unwrap_output_folder_if_needed(Path(raw_folder).resolve())
    if not source_dir.exists() or not source_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Папка не найдена: {source_dir}")

    t_lang = (payload.target_lang or "en").strip().lower()
    if t_lang not in VALID_TARGET_LANGS:
        t_lang = "en"

    output_dir = get_output_dir_for_folder(source_dir, t_lang)
    manifest = _load_folder_manifest(output_dir)
    pages_meta = manifest.get("pages")
    if not isinstance(pages_meta, dict) or not pages_meta:
        return {
            "status": "ok",
            "dry_run": payload.dry_run,
            "matches_count": 0,
            "affected_pages_count": 0,
            "rerendered_pages_count": 0,
            "processing_time_sec": 0.0,
            "matches": [],
        }

    if not isinstance(manifest.get("phrase_cache"), dict):
        manifest["phrase_cache"] = {}

    start_ts = time.perf_counter()
    flags = 0 if payload.match_case else re.IGNORECASE
    pattern = re.compile(re.escape(find_q), flags=flags)
    search_in = "original" if str(payload.search_in).lower().strip() == "original" else "translated"
    repl_str = payload.replace_text if payload.replace_text is not None else ""

    all_images = collect_images_in_folder(source_dir)
    page_index_by_rel: dict[str, int] = {}
    src_path_by_rel: dict[str, Path] = {}
    for idx, img_p in enumerate(all_images, start=1):
        rel_n = _normalize_rel_path(img_p.relative_to(source_dir))
        page_index_by_rel[rel_n] = idx
        src_path_by_rel[rel_n] = img_p

    matches: list[dict[str, Any]] = []
    affected_rel_paths: list[str] = []

    for rel_norm, p_meta in pages_meta.items():
        if not isinstance(p_meta, dict):
            continue
        blocks_list = p_meta.get("blocks")
        if not isinstance(blocks_list, list):
            continue

        page_modified = False
        for b in blocks_list:
            if not isinstance(b, dict):
                continue
            _ensure_block_clean_bbox(b)
            orig_txt = str(b.get("original_text") or "")
            trans_txt = str(b.get("translated_text") or "")

            haystack = orig_txt if search_in == "original" else trans_txt
            if not haystack or not pattern.search(haystack):
                continue

            if search_in == "translated":
                new_trans_txt = pattern.sub(lambda _: repl_str, trans_txt)
            else:
                # Если ищем по японскому/корейскому оригиналу, а заменяем перевод:
                if pattern.search(trans_txt):
                    new_trans_txt = pattern.sub(lambda _: repl_str, trans_txt)
                elif repl_str:
                    new_trans_txt = repl_str
                else:
                    new_trans_txt = trans_txt

            matches.append(
                {
                    "page_rel_path": rel_norm,
                    "page_index": page_index_by_rel.get(rel_norm, 0),
                    "block_id": int(b.get("id") or 1),
                    "original_text": orig_txt,
                    "before_text": trans_txt,
                    "after_text": new_trans_txt,
                }
            )

            if not payload.dry_run and new_trans_txt != trans_txt:
                b["translated_text"] = new_trans_txt
                b["is_deleted"] = not bool(new_trans_txt.strip())
                if orig_txt.strip() and new_trans_txt.strip():
                    manifest["phrase_cache"][f"{t_lang}||{orig_txt.strip()}"] = new_trans_txt.strip()
                    translator_engine.phrase_cache[f"{t_lang}||{orig_txt.strip()}"] = new_trans_txt.strip()
                page_modified = True

        if page_modified:
            p_meta["updated_at"] = int(time.time())
            affected_rel_paths.append(rel_norm)

    rerendered_count = 0
    if not payload.dry_run and affected_rel_paths:
        _save_folder_manifest(output_dir, manifest)

        for rel_norm in affected_rel_paths:
            src_file = src_path_by_rel.get(rel_norm) or (source_dir / rel_norm)
            if not src_file.exists() or not src_file.is_file():
                continue
            dest_path = output_dir / rel_norm
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            p_meta = pages_meta[rel_norm]
            p_blocks = [
                _ensure_block_clean_bbox(b)
                for b in (p_meta.get("blocks") or [])
                if isinstance(b, dict)
            ]
            active_blocks = [
                b for b in p_blocks if not bool(b.get("is_deleted")) and str(b.get("translated_text") or "").strip()
            ]
            with Image.open(src_file) as raw_im:
                pil_image = _pil_to_rgb_clean(raw_im)

            if not active_blocks:
                final_image = pil_image.copy()
            else:
                cleaned_image, _, _, _ = await _get_or_clean_page_image(
                    pil_image,
                    active_blocks,
                    src_file=src_file,
                    save_to_disk=True,
                )
                use_upper = bool(p_meta.get("uppercase", payload.uppercase))
                min_fs = int(p_meta.get("min_font_size") or payload.min_font_size or 12)
                max_fs = int(p_meta.get("max_font_size") or payload.max_font_size or 32)
                eff_overflow = payload.overflow_mode or str(p_meta.get("overflow_mode") or (getattr(translator_engine, "overflow_mode", "shrink") if translator_engine else "shrink"))
                final_image = await asyncio.to_thread(
                    renderer_engine.render_blocks,
                    cleaned_image,
                    active_blocks,
                    use_upper,
                    max_fs,
                    min_fs,
                    None,
                    eff_overflow,
                )

            save_kwargs: dict[str, Any] = {}
            if dest_path.suffix.lower() in (".jpg", ".jpeg", ".webp"):
                save_kwargs["quality"] = 95
            final_image.save(dest_path, **save_kwargs)
            p_meta["blocks"] = p_blocks
            rerendered_count += 1

        _save_folder_manifest(output_dir, manifest)

    elapsed = round(time.perf_counter() - start_ts, 3)
    affected_pages_unique = len({m["page_rel_path"] for m in matches})
    return {
        "status": "ok",
        "dry_run": payload.dry_run,
        "matches_count": len(matches),
        "affected_pages_count": affected_pages_unique,
        "rerendered_pages_count": rerendered_count,
        "processing_time_sec": elapsed,
        "matches": matches,
    }


class FolderCleanNoiseRequest(BaseModel):
    source_dir: str = Field(..., description="Путь к исходной папке главы")
    target_lang: str = Field(default="en", description="Целевой язык: en, ru, romaji")
    remove_free_text: bool = Field(default=True, description="Удалить все рамки вне бабблов (is_free_text)")
    remove_sighs: bool = Field(default=True, description="Удалить короткие вздохи и междометия")
    remove_sfx: bool = Field(default=True, description="Удалить звуковые спецэффекты (SFX)")
    remove_empty: bool = Field(default=True, description="Удалить пустые/удаленные рамки")


@app.post("/folder_clean_noise")
async def folder_clean_noise_endpoint(payload: FolderCleanNoiseRequest) -> dict[str, Any]:
    """
    Пакетно очищает все уже переведенные страницы папки от лишних рамок (надписей на фоне вне бабблов,
    коротких вздохов, SFX и пустых блоков) и мгновенно перерисовывает затронутые страницы (0 токенов ИИ!).
    """
    if cleaner_engine is None or renderer_engine is None or translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    source_dir = _unwrap_output_folder_if_needed(Path(payload.source_dir).resolve())
    if not source_dir.exists() or not source_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Папка не найдена: {source_dir}")

    t_lang = (payload.target_lang or "en").lower().strip()
    output_dir = get_output_dir_for_folder(source_dir, t_lang)
    manifest = _load_folder_manifest(output_dir)
    pages_meta = manifest.get("pages")
    if not isinstance(pages_meta, dict) or not pages_meta:
        return {
            "status": "ok",
            "affected_pages": 0,
            "removed_blocks": 0,
            "message": "В этой папке еще нет сохраненных переведенных страниц.",
        }

    affected_pages = 0
    total_removed_blocks = 0

    for rel_norm, p_meta in list(pages_meta.items()):
        if not isinstance(p_meta, dict):
            continue
        blocks = p_meta.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            continue

        kept_blocks: list[dict[str, Any]] = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            is_del = bool(b.get("is_deleted")) or not str(b.get("translated_text") or "").strip()
            if payload.remove_empty and is_del:
                continue
            is_free = bool(b.get("is_free_text", False))
            if payload.remove_free_text and is_free:
                continue
            orig_t = str(b.get("original_text") or "").strip()
            trans_t = str(b.get("translated_text") or "").strip()
            if payload.remove_sighs and (
                MangaOCREngine.is_sigh_or_interjection(orig_t)
                or is_translated_sigh_text(trans_t, orig_t)
            ):
                continue
            if payload.remove_sfx and MangaOCREngine.is_sfx_or_onomatopoeia(orig_t, is_free_text=is_free):
                continue
            kept_blocks.append(b)

        removed_on_page = len(blocks) - len(kept_blocks)
        if removed_on_page <= 0:
            continue

        total_removed_blocks += removed_on_page
        affected_pages += 1

        src_file = (source_dir / rel_norm).resolve()
        dest_file = (output_dir / rel_norm).resolve()
        if src_file.exists() and src_file.is_file():
            try:
                with Image.open(src_file) as raw_im:
                    pil_image = _pil_to_rgb_clean(raw_im)
                active_blocks = [
                    _ensure_block_clean_bbox(dict(kb))
                    for kb in kept_blocks
                    if not bool(kb.get("is_deleted")) and str(kb.get("translated_text") or "").strip()
                ]
                if not active_blocks:
                    final_image = pil_image.copy()
                else:
                    cleaned_image, _, _, _ = await _get_or_clean_page_image(
                        pil_image,
                        active_blocks,
                        src_file=src_file,
                        save_to_disk=False,
                    )
                    eff_overflow = getattr(translator_engine, "overflow_mode", "shrink") or "shrink"
                    final_image = await asyncio.to_thread(
                        renderer_engine.render_blocks,
                        cleaned_image,
                        active_blocks,
                        bool(p_meta.get("uppercase", translator_engine.uppercase)),
                        int(p_meta.get("max_font_size", translator_engine.max_font_size)),
                        int(p_meta.get("min_font_size", translator_engine.min_font_size)),
                        None,
                        eff_overflow,
                    )
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                save_kwargs: dict[str, Any] = {}
                if dest_file.suffix.lower() in (".jpg", ".jpeg", ".webp"):
                    save_kwargs["quality"] = 95
                elif dest_file.suffix.lower() == ".png":
                    save_kwargs["compress_level"] = 1
                final_image.save(dest_file, **save_kwargs)
            except Exception as exc:
                logger.warning("Не удалось перерисовать страницу %s при пакетной очистке шума: %s", rel_norm, exc)

        p_meta["blocks"] = kept_blocks
        p_meta["blocks_count"] = len(kept_blocks)
        p_meta["free_text_count"] = sum(1 for kb in kept_blocks if kb.get("is_free_text"))
        p_meta["updated_at"] = int(time.time())
        pages_meta[rel_norm] = p_meta

    if affected_pages > 0:
        _save_folder_manifest(output_dir, manifest)
        threading.Thread(
            target=_generate_and_save_folder_report,
            args=(output_dir, copy.deepcopy(manifest), source_dir, t_lang),
            daemon=True,
        ).start()

    return {
        "status": "ok",
        "affected_pages": affected_pages,
        "removed_blocks": total_removed_blocks,
    }


@app.get("/folder_report")
async def get_folder_report_endpoint(
    folder_path: str | None = Query(default=None, description="Путь к исходной или переведенной папке главы"),
    target_lang: str = Query(default="en", description="Целевой язык: en, ru или romaji"),
) -> dict[str, Any]:
    """
    Генерирует (или обновляет на диске) и возвращает содержимое диагностического отчета главы
    `<output_dir>/TRANSLATION_REPORT.txt`.
    """
    t_lang = (target_lang or "en").strip().lower()
    if t_lang not in VALID_TARGET_LANGS:
        t_lang = "en"

    raw_folder = (folder_path or folder_job_state.get("source_dir") or folder_job_state.get("output_dir") or "").strip()
    if not raw_folder:
        raise HTTPException(
            status_code=400,
            detail="Сначала выберите папку с мангой слева, чтобы сформировать отчет по главе.",
        )

    cand_dir = Path(raw_folder).resolve()
    if not cand_dir.exists() or not cand_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Папка не найдена: {cand_dir}")

    src_dir = _unwrap_output_folder_if_needed(cand_dir)
    out_dir = get_output_dir_for_folder(src_dir, t_lang)
    manifest = _load_folder_manifest(out_dir)
    report_text = await asyncio.to_thread(
        _generate_and_save_folder_report,
        out_dir,
        manifest,
        src_dir,
        t_lang,
    )
    report_file = out_dir / REPORT_FILENAME
    return {
        "status": "ok",
        "source_dir": str(src_dir),
        "output_dir": str(out_dir),
        "report_path": str(report_file),
        "report_text": report_text,
    }


@app.post("/open_folder_report")
async def open_folder_report_endpoint(
    folder_path: str | None = Query(default=None),
    target_lang: str = Query(default="en"),
) -> dict[str, str]:
    """
    Обновляет `<output_dir>/TRANSLATION_REPORT.txt` и открывает его в стандартном текстовом редакторе (Блокноте) Windows.
    """
    res = await get_folder_report_endpoint(folder_path=folder_path, target_lang=target_lang)
    report_file = Path(res["report_path"])
    if os.name == "nt" and hasattr(os, "startfile") and report_file.exists():
        try:
            os.startfile(str(report_file))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "status": "ok",
        "report_path": str(report_file),
        "output_dir": res["output_dir"],
    }




class ChapterBookImportRequest(BaseModel):
    folder_path: str = Field(..., description="Путь к папке главы манги")
    target_lang: str = Field(default="en", description="Целевой язык: en, ru, romaji")
    book_text: str = Field(..., description="Отредактированный текст книги (_chapter_book.md, _chapter_script_bilingual.txt или JSON)")
    uppercase: bool = Field(default=True)
    min_font_size: int = Field(default=12)
    max_font_size: int = Field(default=32)
    overflow_mode: str | None = Field(default=None)


class ChapterBookProofreadRequest(BaseModel):
    folder_path: str = Field(..., description="Путь к папке главы манги")
    target_lang: str = Field(default="en", description="Целевой язык: en, ru, romaji")
    ollama_model: str | None = Field(default=None)
    uppercase: bool = Field(default=True)
    min_font_size: int = Field(default=12)
    max_font_size: int = Field(default=32)
    overflow_mode: str | None = Field(default=None)


async def _rerender_modified_book_pages(
    source_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    affected_rel_paths: list[str],
    uppercase: bool = True,
    min_font_size: int = 12,
    max_font_size: int = 32,
    overflow_mode: str | None = None,
) -> int:
    """
    БЛОК 2.3: Мгновенно перерисовывает затронутые страницы главы, используя уже готовый кэш очистки
    (SimpleLama / _get_or_clean_page_image), БЕЗ повторного запуска OCR и без лишних запросов к API.
    """
    if cleaner_engine is None or renderer_engine is None:
        return 0
    pages_meta = manifest.get("pages", {})
    rerendered = 0
    for rel_norm in affected_rel_paths:
        src_file = (source_dir / rel_norm).resolve()
        if not src_file.exists() or not src_file.is_file():
            continue
        dest_path = output_dir / rel_norm
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        p_meta = pages_meta.get(rel_norm, {})
        if not isinstance(p_meta, dict):
            continue
        p_blocks = [
            _ensure_block_clean_bbox(b)
            for b in (p_meta.get("blocks") or [])
            if isinstance(b, dict)
        ]
        active_blocks = [
            b for b in p_blocks if not bool(b.get("is_deleted")) and str(b.get("translated_text") or "").strip()
        ]
        with Image.open(src_file) as raw_im:
            pil_image = _pil_to_rgb_clean(raw_im)
        if not active_blocks:
            final_image = pil_image.copy()
        else:
            cleaned_image, _, _, _ = await _get_or_clean_page_image(
                pil_image,
                active_blocks,
                src_file=src_file,
                save_to_disk=True,
            )
            use_upper = bool(p_meta.get("uppercase", uppercase))
            min_fs = int(p_meta.get("min_font_size") or min_font_size or 12)
            max_fs = int(p_meta.get("max_font_size") or max_font_size or 32)
            eff_overflow = overflow_mode or str(
                p_meta.get("overflow_mode")
                or (getattr(translator_engine, "overflow_mode", "shrink") if translator_engine else "shrink")
            )
            final_image = await asyncio.to_thread(
                renderer_engine.render_blocks,
                cleaned_image,
                active_blocks,
                use_upper,
                max_fs,
                min_fs,
                None,
                eff_overflow,
            )
        save_kwargs: dict[str, Any] = {}
        if dest_path.suffix.lower() in (".jpg", ".jpeg", ".webp"):
            save_kwargs["quality"] = 95
        final_image.save(dest_path, **save_kwargs)
        p_meta["blocks"] = p_blocks
        rerendered += 1
    return rerendered


def _parse_imported_book_text(raw_text: str) -> dict[str, str]:
    """
    Разбирает отредактированный пользователем текст книги (`_chapter_book.md`, `_chapter_script_bilingual.txt`,
    `_chapter_script_bilingual.json` или построчный Markdown со страницами `## Страница 001`)
    и возвращает словарь `{"P01_01": "новый перевод", ...}`.
    """
    text_s = (raw_text or "").strip()
    if not text_s:
        return {}

    # 1. Пробуем JSON (_chapter_script_bilingual.json или прямой словарь {"P01_01": "..."})
    if text_s.startswith("{") or text_s.startswith("["):
        try:
            data = json.loads(text_s)
            if isinstance(data, dict):
                if isinstance(data.get("entries"), list):
                    out_j: dict[str, str] = {}
                    for item in data["entries"]:
                        if isinstance(item, dict) and item.get("book_key"):
                            out_j[str(item["book_key"]).upper()] = str(item.get("translated_text") or "").strip()
                    if out_j:
                        return out_j
                return {
                    str(k).strip().upper(): str(v).strip()
                    for k, v in data.items()
                    if re.match(r"^P\d+_\d+$", str(k).strip(), flags=re.IGNORECASE)
                }
        except Exception:
            pass

    updates: dict[str, str] = {}
    current_page_idx = 0
    current_block_seq = 0

    for raw_line in text_s.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Заголовок страницы: ## Страница 001 ...
        m_page = re.match(r"^#+\s*(?:Страница|Page)\s*0*(\d+)", line, flags=re.IGNORECASE)
        if m_page:
            current_page_idx = int(m_page.group(1))
            current_block_seq = 0
            continue
        if line.startswith("#") or line.startswith("*("):
            continue

        # Явный ключ [P01_01] или **[P01_01]**
        m_key_line = re.match(
            r"^(?:[-*•]\s*)?(?:\*\*)?\[(P\d+_\d+)\](?:\*\*)?\s*(?:\([^)]*\)\s*)?(.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if m_key_line:
            b_key = m_key_line.group(1).upper()
            rest_val = m_key_line.group(2).strip()
            if "=>" in rest_val:
                rest_val = rest_val.split("=>", 1)[1].strip()
            updates[b_key] = rest_val
            continue

        # Если строка внутри секции `## Страница N` без ключа `[Pxx_yy]`, сопоставляем по порядку реплик на странице
        if current_page_idx > 0:
            clean_bullet = re.sub(r"^[-*•]\s*", "", line).strip()
            if clean_bullet:
                current_block_seq += 1
                b_key = f"P{current_page_idx:02d}_{current_block_seq:02d}"
                if "=>" in clean_bullet:
                    clean_bullet = clean_bullet.split("=>", 1)[1].strip()
                updates[b_key] = clean_bullet

    return updates


@app.get("/chapter_book")
async def get_chapter_book_endpoint(
    folder_path: str | None = Query(default=None),
    target_lang: str = Query(default="en"),
) -> dict[str, Any]:
    """
    БЛОК 2.2 / 2.3: Генерирует (или обновляет) и возвращает файлы «Книги перевода» главы:
    `_chapter_book.md` и `_chapter_script_bilingual.json` / `.txt`.
    """
    t_lang = (target_lang or "en").strip().lower()
    if t_lang not in VALID_TARGET_LANGS:
        t_lang = "en"

    raw_folder = (folder_path or folder_job_state.get("source_dir") or folder_job_state.get("output_dir") or "").strip()
    if not raw_folder:
        raise HTTPException(status_code=400, detail="Сначала выберите папку с главой манги слева.")

    cand_dir = Path(raw_folder).resolve()
    if not cand_dir.exists() or not cand_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Папка не найдена: {cand_dir}")

    src_dir = _unwrap_output_folder_if_needed(cand_dir)
    out_dir = get_output_dir_for_folder(src_dir, t_lang)
    manifest = _load_folder_manifest(out_dir)
    book_data = await asyncio.to_thread(
        _generate_and_save_chapter_book_files,
        out_dir,
        manifest,
        src_dir,
        t_lang,
    )
    return {
        "status": "ok",
        "source_dir": str(src_dir),
        "output_dir": str(out_dir),
        **book_data,
    }


@app.post("/chapter_book/import")
async def import_chapter_book_endpoint(payload: ChapterBookImportRequest) -> dict[str, Any]:
    """
    БЛОК 2.3: Импортирует отредактированный пользователем текст «Книги главы» (`_chapter_book.md` или двуязычный скрипт),
    обновляет переводы в бабблах и мгновенно перерисовывает только затронутые страницы из кэша очистки (0 запросов к OCR и API!).
    """
    if cleaner_engine is None or renderer_engine is None or translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    raw_folder = (payload.folder_path or "").strip()
    if not raw_folder:
        raise HTTPException(status_code=400, detail="Сначала выберите папку с главой манги.")

    source_dir = _unwrap_output_folder_if_needed(Path(raw_folder).resolve())
    if not source_dir.exists() or not source_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Папка не найдена: {source_dir}")

    t_lang = (payload.target_lang or "en").strip().lower()
    output_dir = get_output_dir_for_folder(source_dir, t_lang)
    manifest = _load_folder_manifest(output_dir)
    pages_meta = manifest.get("pages")
    if not isinstance(pages_meta, dict) or not pages_meta:
        raise HTTPException(status_code=400, detail="В этой папке еще нет переведенных страниц для обновления книги.")

    updates_by_key = _parse_imported_book_text(payload.book_text)
    if not updates_by_key:
        raise HTTPException(
            status_code=400,
            detail="Не удалось распознать строки книги. Используйте метки [P01_01] или заголовки '## Страница 001'.",
        )

    start_ts = time.perf_counter()
    all_src_images = collect_images_in_folder(source_dir)
    rel_paths_ordered = [_normalize_rel_path(p.relative_to(source_dir)) for p in all_src_images]
    for k_rel in sorted(pages_meta.keys()):
        if k_rel not in rel_paths_ordered:
            rel_paths_ordered.append(k_rel)

    if not isinstance(manifest.get("phrase_cache"), dict):
        manifest["phrase_cache"] = {}

    updated_blocks_count = 0
    affected_rel_paths: list[str] = []

    for p_idx, rel_norm in enumerate(rel_paths_ordered, start=1):
        p_meta = pages_meta.get(rel_norm)
        if not isinstance(p_meta, dict):
            continue
        p_blocks = p_meta.get("blocks")
        if not isinstance(p_blocks, list):
            continue
        page_changed = False
        for b_idx, b in enumerate(p_blocks, start=1):
            if not isinstance(b, dict):
                continue
            bid = int(b.get("id") or b_idx)
            b_key_1 = f"P{p_idx:02d}_{bid:02d}"
            b_key_2 = f"P{p_idx:02d}_{b_idx:02d}"
            new_tr = updates_by_key.get(b_key_1)
            if new_tr is None:
                new_tr = updates_by_key.get(b_key_2)
            if new_tr is None:
                continue
            new_tr_clean = new_tr.strip()
            old_tr_clean = str(b.get("translated_text") or "").strip()
            if new_tr_clean != old_tr_clean:
                b["translated_text"] = new_tr_clean
                b["is_deleted"] = not bool(new_tr_clean)
                b["translation_source"] = "manual_edit"
                orig_t = str(b.get("original_text") or "").strip()
                if orig_t and new_tr_clean:
                    manifest["phrase_cache"][f"{t_lang}||{orig_t}"] = new_tr_clean
                    translator_engine.phrase_cache[f"{t_lang}||{orig_t}"] = new_tr_clean
                updated_blocks_count += 1
                page_changed = True
        if page_changed:
            p_meta["manually_edited"] = True
            p_meta["updated_at"] = int(time.time())
            affected_rel_paths.append(rel_norm)

    rerendered_pages = 0
    if affected_rel_paths:
        _save_folder_manifest(output_dir, manifest)
        rerendered_pages = await _rerender_modified_book_pages(
            source_dir=source_dir,
            output_dir=output_dir,
            manifest=manifest,
            affected_rel_paths=affected_rel_paths,
            uppercase=payload.uppercase,
            min_font_size=payload.min_font_size,
            max_font_size=payload.max_font_size,
            overflow_mode=payload.overflow_mode,
        )
        _save_folder_manifest(output_dir, manifest)

    book_data = await asyncio.to_thread(
        _generate_and_save_chapter_book_files,
        output_dir,
        manifest,
        source_dir,
        t_lang,
    )
    elapsed = round(time.perf_counter() - start_ts, 3)
    return {
        "status": "ok",
        "updated_blocks_count": updated_blocks_count,
        "affected_pages_count": len(affected_rel_paths),
        "rerendered_pages_count": rerendered_pages,
        "processing_time_sec": elapsed,
        **book_data,
    }


@app.post("/chapter_book/proofread")
async def proofread_chapter_book_endpoint(payload: ChapterBookProofreadRequest) -> dict[str, Any]:
    """
    БЛОК 2.3: «✨ Вычитать стиль всей главы за 1 запрос»:
    Отправляет весь сценарий главы одним JSON-пакетом в ИИ для согласования стиля, пола персонажей и местоимений,
    обновляет переводы в бабблах и мгновенно перерисовывает затронутые страницы из кэша очистки без повторного OCR.
    """
    if cleaner_engine is None or renderer_engine is None or translator_engine is None:
        raise HTTPException(status_code=503, detail="Сервер еще инициализируется.")

    raw_folder = (payload.folder_path or "").strip()
    if not raw_folder:
        raise HTTPException(status_code=400, detail="Сначала выберите папку с главой манги.")

    source_dir = _unwrap_output_folder_if_needed(Path(raw_folder).resolve())
    if not source_dir.exists() or not source_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Папка не найдена: {source_dir}")

    t_lang = (payload.target_lang or "en").strip().lower()
    output_dir = get_output_dir_for_folder(source_dir, t_lang)
    manifest = _load_folder_manifest(output_dir)
    pages_meta = manifest.get("pages")
    if not isinstance(pages_meta, dict) or not pages_meta:
        raise HTTPException(status_code=400, detail="Сначала переведите страницы главы перед литературной вычиткой.")

    start_ts = time.perf_counter()
    all_src_images = collect_images_in_folder(source_dir)
    rel_paths_ordered = [_normalize_rel_path(p.relative_to(source_dir)) for p in all_src_images]
    for k_rel in sorted(pages_meta.keys()):
        if k_rel not in rel_paths_ordered:
            rel_paths_ordered.append(k_rel)

    book_script_map: dict[str, dict[str, str]] = {}
    detected_lang = "ja"
    for p_idx, rel_norm in enumerate(rel_paths_ordered, start=1):
        p_meta = pages_meta.get(rel_norm)
        if not isinstance(p_meta, dict):
            continue
        if p_meta.get("detected_lang") in ("ja", "ko", "zh"):
            detected_lang = str(p_meta["detected_lang"])
        for b_idx, b in enumerate(p_meta.get("blocks") or [], start=1):
            if not isinstance(b, dict) or bool(b.get("is_deleted")):
                continue
            orig_t = str(b.get("original_text") or "").strip()
            trans_t = str(b.get("translated_text") or "").strip()
            if not orig_t and not trans_t:
                continue
            bid = int(b.get("id") or b_idx)
            b_key = f"P{p_idx:02d}_{bid:02d}"
            book_script_map[b_key] = {"orig": orig_t, "trans": trans_t}

    if not book_script_map:
        raise HTTPException(status_code=400, detail="В главе нет переведенных реплик для вычитки.")

    folder_glossary = normalize_glossary_entries(manifest.get("glossary", []))
    polished_map = await translator_engine.proofread_chapter_book_script(
        book_map=book_script_map,
        source_lang=detected_lang,
        target_lang=t_lang,
        model=payload.ollama_model,
        glossary_entries=folder_glossary,
    )

    if not isinstance(manifest.get("phrase_cache"), dict):
        manifest["phrase_cache"] = {}

    updated_blocks_count = 0
    affected_rel_paths: list[str] = []
    for p_idx, rel_norm in enumerate(rel_paths_ordered, start=1):
        p_meta = pages_meta.get(rel_norm)
        if not isinstance(p_meta, dict):
            continue
        page_changed = False
        for b_idx, b in enumerate(p_meta.get("blocks") or [], start=1):
            if not isinstance(b, dict):
                continue
            bid = int(b.get("id") or b_idx)
            b_key = f"P{p_idx:02d}_{bid:02d}"
            new_tr = polished_map.get(b_key, "").strip()
            old_tr = str(b.get("translated_text") or "").strip()
            if new_tr and new_tr != old_tr:
                b["translated_text"] = new_tr
                b["translation_source"] = "llm_book_proofread"
                orig_t = str(b.get("original_text") or "").strip()
                if orig_t:
                    manifest["phrase_cache"][f"{t_lang}||{orig_t}"] = new_tr
                    translator_engine.phrase_cache[f"{t_lang}||{orig_t}"] = new_tr
                updated_blocks_count += 1
                page_changed = True
        if page_changed:
            p_meta["updated_at"] = int(time.time())
            affected_rel_paths.append(rel_norm)

    rerendered_pages = 0
    if affected_rel_paths:
        _save_folder_manifest(output_dir, manifest)
        rerendered_pages = await _rerender_modified_book_pages(
            source_dir=source_dir,
            output_dir=output_dir,
            manifest=manifest,
            affected_rel_paths=affected_rel_paths,
            uppercase=payload.uppercase,
            min_font_size=payload.min_font_size,
            max_font_size=payload.max_font_size,
            overflow_mode=payload.overflow_mode,
        )
        _save_folder_manifest(output_dir, manifest)

    book_data = await asyncio.to_thread(
        _generate_and_save_chapter_book_files,
        output_dir,
        manifest,
        source_dir,
        t_lang,
    )
    elapsed = round(time.perf_counter() - start_ts, 2)
    return {
        "status": "ok",
        "updated_blocks_count": updated_blocks_count,
        "affected_pages_count": len(affected_rel_paths),
        "rerendered_pages_count": rerendered_pages,
        "processing_time_sec": elapsed,
        **book_data,
    }


@app.get("/diagnostics")
async def get_diagnostics_endpoint() -> dict[str, Any]:
    """Возвращает полный диагностический слепок сервера, GPU/VRAM, таймингов этапов, запросов к ИИ и логов."""
    return _build_diagnostics_payload()


@app.get("/diagnostics/report")
async def get_diagnostics_report_endpoint() -> dict[str, str]:
    """Генерирует готовый к отправке в чат диагностический Markdown-отчет (API-ключ замаскирован)."""
    d = _build_diagnostics_payload()
    srv = d["server"]
    gpu = d["gpu"]
    tr = d["translator"]
    fj = d["folder_job"]

    overflow_names_ru = {
        "shrink": "Стандарт (внутри рамки до мин. шрифта)",
        "overflow": "Вариант Б — Выход текста за рамки баббла (с обводкой)",
        "expand_bubble": "Вариант В — Авто-перерисовка (расширение) баббла под текст",
    }
    lines: list[str] = [
        "=== 🩺 ДИАГНОСТИЧЕСКИЙ ОТЧЕТ MIKO-CHAN MANGA STUDIO v1.0.6 ===",
        f"• Время отчета: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"• Сервер: v{srv.get('version')} | PID={srv.get('pid')} | Uptime={srv.get('uptime_sec')}с | Старт={srv.get('started_at')}",
        f"• Окружение: Python {srv.get('python_version')} | PyTorch {gpu.get('torch_version')} | {srv.get('platform')}",
        f"• Папка проекта: {srv.get('base_dir')}",
        f"• Лог-файл: {srv.get('log_file_path')} ({srv.get('log_file_size_kb')} КБ)",
        f"• Видеокарта: {gpu.get('device_name', 'CPU')} | VRAM занято: {gpu.get('vram_allocated_mb', 0)} / {gpu.get('vram_total_mb', 0)} МБ (резерв: {gpu.get('vram_reserved_mb', 0)} МБ)",
        f"• Детектор текста/бабблов: {gpu.get('detector_backend', 'YOLOv8m')}",
        f"• Движок OCR (Японский JA): {gpu.get('ja_ocr_backend', 'MangaOCR FP16')}",
        f"• Движок OCR (Корейский KO / Китайский ZH): {gpu.get('ko_zh_ocr_backend', 'HayaiOCR v2.5 Nova + SmartLineSplit')}",
        f"• Снайперский VLM-OCR (SFX / Обложки): {gpu.get('vlm_ocr_backend', 'Sniper VLM-OCR (Ollama Vision / Cloud Vision)')}",
        f"• Очистка бабблов: {gpu.get('inpaint_backend')} | Мультиязычный OCR активен: {gpu.get('easyocr_installed')}",
        "",
        "--- 🤖 НАСТРОЙКИ И СТАТИСТИКА ИИ / КЭША ---",
        f"• Активная модель: {tr.get('selected_model')} | API URL: {tr.get('api_url')} | Ключ: {tr.get('masked_api_key') or 'нет'}",
        f"• Лимит облачной пачки: {tr.get('cloud_batch_limit')} | Параллельный чанк: {tr.get('cloud_parallel_chunk_size')} | Страниц в пачке: {tr.get('batch_pages')}",
        f"• Фильтр текста: mode={tr.get('text_filter_mode')}, skip_sighs={tr.get('skip_sighs')}, skip_sfx={tr.get('skip_sfx')}, free_text={tr.get('detect_free_text')}",
        f"• Запросов к ИИ за сессию: всего={tr.get('total_llm_requests')} (Cloud={tr.get('total_cloud_requests')}, Ollama={tr.get('total_ollama_requests')}, параллельных пачек={tr.get('parallel_batches_count')})",
        f"• Среднее время ответа ИИ: {tr.get('avg_llm_latency_sec')}с | Суммарное время ИИ: {tr.get('total_llm_time_sec')}с",
        f"• Экономия кэша: {tr.get('phrases_saved_by_cache')} из {tr.get('phrases_requested_total')} фраз ({tr.get('cache_hit_rate_pct')}%) | Отправлено в ИИ: {tr.get('phrases_sent_to_llm')} | Размер кэша: {tr.get('phrase_cache_size')}",
        f"• Отказов цензуры API: {tr.get('refusals_count')} | Переведено локальной моделью (обход цензуры): {tr.get('local_censorship_fallbacks_count', 0)} | Санитизатор: {tr.get('sanitized_retries_count')} | Микро-допереводов: {tr.get('micro_retries_count')} | Ошибок сети: {tr.get('errors_count')}",
        f"• Статус перевода папки: status={fj.get('status')}, прогресс={fj.get('current')}/{fj.get('total')}, файл={fj.get('current_file')}, время={fj.get('elapsed_sec')}с",
    ]

    recent_jobs = d.get("recent_jobs") or []
    lines.append("")
    lines.append(f"--- ⏱ ТАЙМИНГИ ПОСЛЕДНИХ ПЕРЕВОДОВ ({len(recent_jobs)} шт.) ---")
    if not recent_jobs:
        lines.append("  (переводов в этой сессии еще не было)")
    else:
        for jb in recent_jobs[:12]:
            lines.append(
                f"  [{jb.get('time')}] {jb.get('label')} | Блоков: {jb.get('blocks_count')} | "
                f"OCR={jb.get('ocr_sec')}с, ИИ={jb.get('llm_sec')}с, Рендер={jb.get('render_sec')}с "
                f"=> ВСЕГО={jb.get('total_sec')}с ({jb.get('sec_per_page')}с/стр) | Модель: {jb.get('model')}"
            )

    api_calls = tr.get("recent_api_calls") or []
    lines.append("")
    lines.append(f"--- ☁️ ПОСЛЕДНИЕ ВЫЗОВЫ API ИИ ({len(api_calls)} шт.) ---")
    if not api_calls:
        lines.append("  (вызовов API еще не было)")
    else:
        for ac in api_calls[:15]:
            note_s = f" | {ac.get('note')}" if ac.get("note") else ""
            lines.append(
                f"  [{ac.get('time')}] {ac.get('provider')}::{ac.get('model')} | Статус: {ac.get('status')} | "
                f"Время: {ac.get('elapsed_sec')}с | Получено ключей: {ac.get('returned_keys')}{note_s}"
            )

    warns = d.get("recent_warnings_errors") or []
    lines.append("")
    lines.append(f"--- ⚠️ ПОСЛЕДНИЕ ПРЕДУПРЕЖДЕНИЯ И ОШИБКИ ({len(warns)} шт.) ---")
    if not warns:
        lines.append("  (ошибок и предупреждений нет)")
    else:
        for w in warns[:20]:
            lines.append(f"  {w.get('formatted')}")

    ux_d = d.get("ux_telemetry") or {}
    ux_top = ux_d.get("top_actions") or []
    lines.append("")
    lines.append(f"--- 📊 ТОП НАЖИМАЕМЫХ КНОПОК UX-ТЕЛЕМЕТРИИ (всего действий: {ux_d.get('total_events', 0)}) ---")
    if not ux_top:
        lines.append("  (действий UX пока не зафиксировано)")
    else:
        for ua in ux_top:
            lines.append(f"  • {ua.get('count')} раз — {ua.get('label')} ({ua.get('zone')})")

    tail_logs = (d.get("recent_logs") or [])[-50:]
    lines.append("")
    lines.append(f"--- 📜 ПОСЛЕДНИЕ {len(tail_logs)} СТРОК ЛОГА СЕРВЕРА ---")
    lines.extend(tail_logs)
    lines.append("=== КОНЕЦ ДИАГНОСТИЧЕСКОГО ОТЧЕТА ===")

    return {"status": "ok", "report_text": "\n".join(lines)}


@app.get("/diagnostics/download_log")
async def download_log_file_endpoint():
    """Скачивание полного файла лога logs/miko_studio.log."""
    if not MAIN_LOG_FILE.exists():
        MAIN_LOG_FILE.write_text("Лог-файл инициализирован.\n", encoding="utf-8")
    return FileResponse(
        str(MAIN_LOG_FILE),
        media_type="text/plain; charset=utf-8",
        filename="miko_studio.log",
    )


@app.post("/diagnostics/open_logs")
async def open_logs_folder_endpoint() -> dict[str, str]:
    """Открывает папку logs/ в Проводнике Windows."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt" and hasattr(os, "startfile"):
        try:
            os.startfile(str(LOGS_DIR))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok", "logs_dir": str(LOGS_DIR), "log_file": str(MAIN_LOG_FILE)}


@app.post("/diagnostics/clear")
async def clear_diagnostics_endpoint() -> dict[str, str]:
    """Очищает оперативные счетчики диагностики и буфер логов в памяти."""
    _recent_log_entries.clear()
    _recent_warn_err_entries.clear()
    _recent_job_profiles.clear()
    if translator_engine is not None:
        translator_engine.reset_diagnostics()
    logger.info("Диагностические счетчики и буфер логов очищены пользователем.")
    return {"status": "ok"}





# ==============================================================================
#  📊 МОДУЛЬ UX-ТЕЛЕМЕТРИИ И АНАЛИТИКИ ИСПОЛЬЗОВАНИЯ ИНТЕРФЕЙСА (v1.0.7)
# ==============================================================================

_ux_telemetry_lock = threading.Lock()

UX_ZONE_LABELS_RU: dict[str, str] = {
    "topbar": "Верхняя панель (Topbar: Языки / Настройки / Вид)",
    "left_sidebar": "Левая панель (Папка / Список страниц / Старт)",
    "viewer_toolbar": "Панель над страницей (Навигация / Режим вида / Масштаб / Инструменты)",
    "viewer_stage_pin": "Прямо на странице манги (Кружки #N / Перетаскивание / Ресайз / Колесико)",
    "inline_popup": "Всплывающий попап у кружка #N (Быстрое редактирование на холсте)",
    "editor_drawer_header": "Шапка панели редактора (Сбоку/Снизу/Окно / Сохранить / Отмена)",
    "editor_drawer_subbar": "Подпанель редактора (Обвести / Подложка на все / Удаление рамок / Ромадзи)",
    "editor_drawer_card": "Карточки списка блоков в редакторе (Текст / Стиль / Шрифт / Поворот)",
    "floating_brush_bar": "Плавающая панель живой кисти ретуши (Замазка ЛКМ / Оригинал ПКМ / Радиус)",
    "settings_modal": "Окно настроек, Глоссария, Поиска Ctrl+F и Диагностики",
    "keyboard_hotkey": "Горячие клавиши клавиатуры (Хоткеи)",
    "other": "Прочие элементы интерфейса",
}

# Список основных функций интерфейса для отслеживания того, чем пользователь пользуется, а какие кнопки не нажимает вовсе
KNOWN_UI_FEATURES_CATALOG: dict[str, tuple[str, str]] = {
    "act:save_page": ("💾 Сохранить страницу (кнопка или Ctrl+S)", "Редактирование и сохранение"),
    "act:undo": ("↩ Отменить действие (кнопка или Ctrl+Z)", "Редактирование и сохранение"),
    "act:redo": ("↪ Повторить действие (кнопка или Ctrl+Y)", "Редактирование и сохранение"),
    "act:nav_prev_next": ("◀ / ▶ Переключение страниц (кнопки или A/D/Стрелки)", "Навигация и просмотр"),
    "act:view_mode_switch": ("🔀 Переключение режима До и После / Перевод / Оригинал", "Навигация и просмотр"),
    "act:hold_compare_space": ("👁 Зажатие Space / V для мгновенной сверки с оригиналом", "Навигация и просмотр"),
    "act:zoom_controls": ("🔍 Изменение масштаба страницы (Шир. / Выс. / ➖ / ➕ / Ctrl+Колесико)", "Навигация и просмотр"),
    "act:pin_click": ("📌 Клик по кружку #N на странице", "Работа с рамками на странице"),
    "act:pin_move": ("✋ Перетаскивание рамки #N мышкой", "Работа с рамками на странице"),
    "act:pin_resize": ("📐 Изменение размера рамки #N за края/уголки", "Работа с рамками на странице"),
    "act:pin_wheel_font": ("🖱 Изменение кегля шрифта колесиком мыши прямо на рамке #N", "Работа с рамками на странице"),
    "act:manual_draw_region": ("➕ Обвести пропущенный кружок/текст мышкой", "Работа с рамками на странице"),
    "act:vlm_ocr_sniper": ("👁 Снайперский VLM-OCR для сложных рукописных SFX и обложек", "Работа с рамками на странице"),
    "act:merge_blocks": ("🔗 Объединение нескольких рамок (Shift+Клик + M)", "Работа с рамками на странице"),
    "act:split_block": ("✂️ Разделение одной рамки на два блока", "Работа с рамками на странице"),
    "act:delete_block_frame": ("🗑 Полное удаление рамки #N с картинки", "Работа с рамками на странице"),
    "act:clear_block_text": ("🧹 Очистка только текста внутри рамки", "Работа с рамками на странице"),
    "act:retranslate_single_block": ("🔄 Повторный ИИ-перевод одного кружка #N", "Работа с рамками на странице"),
    "act:bg_style_backing": ("🛡 Подложка под текстом (Овал / По тексту / Рамка)", "Стилизация и верстка текста"),
    "act:page_all_bg": ("🛡 Подложка сразу на все рамки страницы", "Стилизация и верстка текста"),
    "act:style_preset": ("🎨 Пресеты стиля (Диалог / Мысли / Крик / Шепот / Нарратив / SFX)", "Стилизация и верстка текста"),
    "act:font_family_change": ("🔤 Смена гарнитуры шрифта для блока", "Стилизация и верстка текста"),
    "act:text_align_change": ("↔ Выравнивание текста (По левому / По центру / По правому)", "Стилизация и верстка текста"),
    "act:text_color_stroke": ("🎨 Ручной выбор цвета текста или обводки", "Стилизация и верстка текста"),
    "act:text_angle_rotate": ("🔄 Поворот текста ползунком угла (-45°..+45°)", "Стилизация и верстка текста"),
    "act:special_symbols_insert": ("♥★♪ Вставка спецсимволов манги в 1 клик", "Стилизация и верстка текста"),
    "act:live_mask_brush": ("🖌 Живая 60 FPS кисть ретуши замазки (ЛКМ замазать / ПКМ оригинал)", "Ретушь и замазка оригинала"),
    "act:mask_dilate_slider": ("🖌 Ползунок толщины авто-замазки (dilate)", "Ретушь и замазка оригинала"),
    "act:romaji_toggle_or_ai": ("🔤 Просмотр или ИИ-генерация транскрипции Ромадзи", "Глоссарий, Ромадзи и Поиск"),
    "act:glossary_manage": ("📖 Добавление/импорт терминов и имен в Глоссарий", "Глоссарий, Ромадзи и Поиск"),
    "act:folder_find_replace": ("🔎 Глобальный Поиск и Замена по всей главе (Ctrl+F / Ctrl+H)", "Глоссарий, Ромадзи и Поиск"),
    "act:dock_mode_switch": ("📐 Переключение расположения редактора (Сбоку / Снизу / Окно)", "Навигация и просмотр"),
    "act:toggle_left_sidebar": ("◀ Сворачивание левой панели (Широкий экран / клавиша [ )", "Навигация и просмотр"),
}

# Сопоставление кнопок мыши и их быстрых клавиш (для подсказок, где пользователь часто тянется мышкой вместо 1 нажатия клавиши)
HOTKEY_ALTERNATIVES_MAP: dict[str, tuple[str, str]] = {
    "btn-save-page-edits": ("Ctrl + S (или Ctrl + Enter в поле текста)", "Сохранение страницы"),
    "btn-undo-action": ("Ctrl + Z", "Отмена последнего действия"),
    "btn-redo-action": ("Ctrl + Y (или Ctrl + Shift + Z)", "Повтор отмененного действия"),
    "btn-prev-page": ("Клавиша A или Стрелка влево (←)", "Переход на предыдущую страницу"),
    "btn-next-page": ("Клавиша D или Стрелка вправо (→)", "Переход на следующую страницу"),
    "btn-toggle-sidebar": ("Клавиша [", "Свернуть / развернуть левую панель"),
    "btn-wide-mode": ("Клавиша [", "Широкий экран (скрыть левую панель)"),
    "btn-top-mask-brush": ("Клавиша B", "Включить / выключить живую кисть маски"),
    "popup-mask-brush-btn": ("Клавиша B", "Включить / выключить живую кисть маски"),
    "btn-merge-blocks": ("Клавиша M (после Shift+Клика по рамкам)", "Объединить выделенные рамки"),
    "popup-merge-btn": ("Клавиша M (после Shift+Клика по рамкам)", "Объединить выделенные рамки"),
    "popup-del-btn": ("Клавиша Delete (при выбранном кружке)", "Полное удаление рамки #N с картинки"),
}


def _new_empty_ux_store() -> dict[str, Any]:
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "schema_version": 1,
        "created_at": now_str,
        "updated_at": now_str,
        "total_events": 0,
        "total_mouse_clicks": 0,
        "total_hotkey_presses": 0,
        "total_canvas_manipulations": 0,
        "total_text_edits": 0,
        "active_usage_sec": 0.0,
        "actions": {},
        "zones": {},
        "hotkeys": {},
        "missed_hotkeys": {},
        "canvas_ops": {},
        "features_used": {},
        "transitions": {},
        "dwell_sec": {
            "dock_right": 0.0,
            "dock_bottom": 0.0,
            "dock_floating": 0.0,
            "view_split": 0.0,
            "view_translated": 0.0,
            "view_original": 0.0,
            "edit_in_popup": 0.0,
            "edit_in_drawer": 0.0,
        },
        "last_action_key": "",
        "last_action_label": "",
        "last_action_ts": 0.0,
        "recent_events": [],
    }


def _load_ux_store() -> dict[str, Any]:
    if UX_TELEMETRY_JSON_FILE.exists():
        try:
            data = json.loads(UX_TELEMETRY_JSON_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                base = _new_empty_ux_store()
                for k, v in base.items():
                    if k not in data:
                        data[k] = v
                if isinstance(data.get("dwell_sec"), dict):
                    for dk, dv in base["dwell_sec"].items():
                        data["dwell_sec"].setdefault(dk, dv)
                return data
        except Exception as exc:
            logger.warning("Не удалось прочитать ux_telemetry.json: %s", exc)
    return _new_empty_ux_store()


def _save_ux_store(store: dict[str, Any]) -> None:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        store["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp_path = UX_TELEMETRY_JSON_FILE.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, UX_TELEMETRY_JSON_FILE)
    except Exception as exc:
        logger.warning("Не удалось сохранить ux_telemetry.json: %s", exc)


class UxEventItem(BaseModel):
    event_type: str = Field(
        default="button_click",
        description="Тип события: button_click, control_change, hotkey_use, canvas_action, text_edit, dwell_tick",
    )
    action_id: str = Field(default="", description="Уникальный ключ элемента или действия")
    label: str = Field(default="", description="Понятное название нажатой кнопки или действия")
    zone: str = Field(default="other", description="Зона интерфейса, где произошло действие")
    feature_key: str | None = Field(default=None, description="Ключ из KNOWN_UI_FEATURES_CATALOG")
    hotkey: str | None = Field(default=None, description="Нажатая комбинация клавиш, если это hotkey_use")
    meta: dict[str, Any] | None = Field(default=None, description="Дополнительные параметры (режим докинга, вид и т.д.)")
    timestamp_ms: int | None = Field(default=None)


class UxEventsBatchRequest(BaseModel):
    events: list[UxEventItem] = Field(default_factory=list)
    active_dock_mode: str | None = Field(default=None)
    active_view_mode: str | None = Field(default=None)
    elapsed_active_sec: float = Field(default=0.0, description="Активное время работы пользователя за батч (сек)")


def _ingest_ux_events_batch(payload: UxEventsBatchRequest) -> dict[str, Any]:
    with _ux_telemetry_lock:
        store = _load_ux_store()
        now_ts = time.time()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        elapsed_sec = max(0.0, min(120.0, float(payload.elapsed_active_sec or 0.0)))
        if elapsed_sec > 0:
            store["active_usage_sec"] = round(float(store.get("active_usage_sec") or 0.0) + elapsed_sec, 1)
            dock_m = str(payload.active_dock_mode or "").strip().lower()
            if dock_m in ("right", "bottom", "floating"):
                dk = f"dock_{dock_m}"
                store["dwell_sec"][dk] = round(float(store["dwell_sec"].get(dk) or 0.0) + elapsed_sec, 1)
            view_m = str(payload.active_view_mode or "").strip().lower()
            if view_m in ("split", "translated", "original"):
                vk = f"view_{view_m}"
                store["dwell_sec"][vk] = round(float(store["dwell_sec"].get(vk) or 0.0) + elapsed_sec, 1)

        jsonl_lines: list[str] = []

        for ev in payload.events:
            ev_type = (ev.event_type or "button_click").strip().lower()
            act_id = (ev.action_id or "").strip() or "unknown_action"
            label = (ev.label or act_id).strip()[:110]
            zone = (ev.zone or "other").strip().lower()
            if zone not in UX_ZONE_LABELS_RU:
                zone = "other"
            feat_key = (ev.feature_key or "").strip()

            if ev_type == "dwell_tick":
                meta_d = ev.meta or {}
                add_s = max(0.0, min(60.0, float(meta_d.get("sec") or 0.0)))
                target_d = str(meta_d.get("target") or "")
                if add_s > 0 and target_d in store["dwell_sec"]:
                    store["dwell_sec"][target_d] = round(float(store["dwell_sec"].get(target_d) or 0.0) + add_s, 1)
                continue

            store["total_events"] = int(store.get("total_events") or 0) + 1
            store["zones"][zone] = int(store["zones"].get(zone) or 0) + 1

            if ev_type in ("button_click", "control_change"):
                store["total_mouse_clicks"] = int(store.get("total_mouse_clicks") or 0) + 1
                if act_id in HOTKEY_ALTERNATIVES_MAP:
                    hk_combo, hk_desc = HOTKEY_ALTERNATIVES_MAP[act_id]
                    mh_entry = store["missed_hotkeys"].get(act_id) or {
                        "action_id": act_id,
                        "label": label,
                        "hotkey": hk_combo,
                        "description": hk_desc,
                        "mouse_clicks": 0,
                    }
                    mh_entry["mouse_clicks"] = int(mh_entry.get("mouse_clicks") or 0) + 1
                    mh_entry["label"] = label
                    store["missed_hotkeys"][act_id] = mh_entry
            elif ev_type == "hotkey_use":
                store["total_hotkey_presses"] = int(store.get("total_hotkey_presses") or 0) + 1
                hk_name = (ev.hotkey or act_id).strip()
                hk_entry = store["hotkeys"].get(hk_name) or {
                    "hotkey": hk_name,
                    "label": label,
                    "count": 0,
                }
                hk_entry["count"] = int(hk_entry.get("count") or 0) + 1
                hk_entry["label"] = label
                store["hotkeys"][hk_name] = hk_entry
            elif ev_type == "canvas_action":
                store["total_canvas_manipulations"] = int(store.get("total_canvas_manipulations") or 0) + 1
                c_entry = store["canvas_ops"].get(act_id) or {
                    "action_id": act_id,
                    "label": label,
                    "count": 0,
                }
                c_entry["count"] = int(c_entry.get("count") or 0) + 1
                c_entry["label"] = label
                store["canvas_ops"][act_id] = c_entry
            elif ev_type == "text_edit":
                store["total_text_edits"] = int(store.get("total_text_edits") or 0) + 1
                if zone == "inline_popup":
                    store["dwell_sec"]["edit_in_popup"] = round(
                        float(store["dwell_sec"].get("edit_in_popup") or 0.0) + 1.0, 1
                    )
                elif zone in ("editor_drawer_card", "editor_drawer_header"):
                    store["dwell_sec"]["edit_in_drawer"] = round(
                        float(store["dwell_sec"].get("edit_in_drawer") or 0.0) + 1.0, 1
                    )

            if feat_key:
                store["features_used"][feat_key] = int(store["features_used"].get(feat_key) or 0) + 1

            # Записываем в общий рейтинг действий
            act_entry = store["actions"].get(act_id) or {
                "action_id": act_id,
                "label": label,
                "zone": zone,
                "event_type": ev_type,
                "count": 0,
                "last_used": now_str,
            }
            act_entry["count"] = int(act_entry.get("count") or 0) + 1
            act_entry["label"] = label
            act_entry["zone"] = zone
            act_entry["last_used"] = now_str
            store["actions"][act_id] = act_entry

            # Записываем цепочки переходов (Bigram: Действие А -> Действие Б), если между ними прошло <= 45 секунд
            prev_key = str(store.get("last_action_key") or "")
            prev_lbl = str(store.get("last_action_label") or "")
            prev_ts = float(store.get("last_action_ts") or 0.0)
            if prev_key and prev_key != act_id and (now_ts - prev_ts) <= 45.0:
                pair_key = f"{prev_key}>>>{act_id}"
                tr_entry = store["transitions"].get(pair_key) or {
                    "from_id": prev_key,
                    "from_label": prev_lbl,
                    "to_id": act_id,
                    "to_label": label,
                    "count": 0,
                }
                tr_entry["count"] = int(tr_entry.get("count") or 0) + 1
                tr_entry["from_label"] = prev_lbl
                tr_entry["to_label"] = label
                store["transitions"][pair_key] = tr_entry

            store["last_action_key"] = act_id
            store["last_action_label"] = label
            store["last_action_ts"] = now_ts

            rec_item = {
                "time": time.strftime("%H:%M:%S"),
                "type": ev_type,
                "action_id": act_id,
                "label": label,
                "zone": zone,
            }
            store["recent_events"].append(rec_item)
            jsonl_lines.append(
                json.dumps(
                    {
                        "timestamp": int(now_ts),
                        **rec_item,
                        "feature_key": feat_key or None,
                        "meta": ev.meta or None,
                    },
                    ensure_ascii=False,
                )
            )

        if len(store["recent_events"]) > 180:
            store["recent_events"] = store["recent_events"][-180:]

        _save_ux_store(store)

        if jsonl_lines:
            try:
                with open(UX_EVENTS_JSONL_FILE, "a", encoding="utf-8") as f_jl:
                    f_jl.write("\n".join(jsonl_lines) + "\n")
            except Exception:
                pass

        return store


def _generate_ux_usage_report(store: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """
    Формирует подробный аналитический отчет по использованию интерфейса `logs/UX_USAGE_REPORT.txt`
    и возвращает `(report_text, summary_dict)`.
    """
    if store is None:
        with _ux_telemetry_lock:
            store = _load_ux_store()

    total_ev = int(store.get("total_events") or 0)
    mouse_clicks = int(store.get("total_mouse_clicks") or 0)
    hotkey_presses = int(store.get("total_hotkey_presses") or 0)
    canvas_ops_cnt = int(store.get("total_canvas_manipulations") or 0)
    text_edits_cnt = int(store.get("total_text_edits") or 0)
    active_sec = float(store.get("active_usage_sec") or 0.0)
    active_min = round(active_sec / 60.0, 1)

    actions_dict: dict[str, dict[str, Any]] = store.get("actions") or {}
    sorted_actions = sorted(
        actions_dict.values(),
        key=lambda x: int(x.get("count") or 0),
        reverse=True,
    )

    zones_dict: dict[str, int] = store.get("zones") or {}
    sorted_zones = sorted(zones_dict.items(), key=lambda kv: int(kv[1]), reverse=True)

    hotkeys_dict: dict[str, dict[str, Any]] = store.get("hotkeys") or {}
    sorted_hotkeys = sorted(
        hotkeys_dict.values(),
        key=lambda x: int(x.get("count") or 0),
        reverse=True,
    )

    missed_hk_dict: dict[str, dict[str, Any]] = store.get("missed_hotkeys") or {}
    sorted_missed_hk = sorted(
        missed_hk_dict.values(),
        key=lambda x: int(x.get("mouse_clicks") or 0),
        reverse=True,
    )

    canvas_dict: dict[str, dict[str, Any]] = store.get("canvas_ops") or {}
    sorted_canvas = sorted(
        canvas_dict.values(),
        key=lambda x: int(x.get("count") or 0),
        reverse=True,
    )

    transitions_dict: dict[str, dict[str, Any]] = store.get("transitions") or {}
    sorted_transitions = sorted(
        transitions_dict.values(),
        key=lambda x: int(x.get("count") or 0),
        reverse=True,
    )

    features_used: dict[str, int] = store.get("features_used") or {}
    unused_features: list[tuple[str, str, str]] = []
    used_features_ranked: list[tuple[str, str, str, int]] = []
    for f_key, (f_title, f_cat) in KNOWN_UI_FEATURES_CATALOG.items():
        cnt = int(features_used.get(f_key) or 0)
        if cnt <= 0:
            unused_features.append((f_key, f_title, f_cat))
        else:
            used_features_ranked.append((f_key, f_title, f_cat, cnt))
    used_features_ranked.sort(key=lambda x: x[3], reverse=True)

    dwell = store.get("dwell_sec") or {}
    popup_edits = int(dwell.get("edit_in_popup") or 0)
    drawer_edits = int(dwell.get("edit_in_drawer") or 0)

    # Формируем умные автоматические рекомендации по редизайну UI/UX на основе реальных данных пользователя
    ux_recommendations: list[str] = []
    if sorted_actions:
        top5_names = ", ".join(f"«{a['label']}» ({a['count']}x)" for a in sorted_actions[:5])
        ux_recommendations.append(
            f"🔥 Главные кандидаты на закрепление в самой быстрой доступности (Топ-5 ваших действий): {top5_names}."
        )

    if popup_edits > 0 or drawer_edits > 0:
        if popup_edits >= drawer_edits * 1.3:
            ux_recommendations.append(
                f"💬 Вы редактируете текст преимущественно через компактный Всплывающий попап прямо у кружка #N ({popup_edits} правок против {drawer_edits} в боковой панели). Имеет смысл сделать кнопки в попапе крупнее и вынести туда ваши любимые пресеты в 1 клик, а большую боковую панель открывать только по запросу."
            )
        elif drawer_edits >= popup_edits * 1.3:
            ux_recommendations.append(
                f"📋 Вы чаще редактируете перевод в основной панели списка блоков ({drawer_edits} правок против {popup_edits} во всплывающем попапе). Можно оптимизировать карточку блока в списке: оставить на виду только те кнопки, которые вы реально нажимаете, и убрать лишние строки."
            )

    if sorted_transitions and int(sorted_transitions[0].get("count") or 0) >= 2:
        t0 = sorted_transitions[0]
        ux_recommendations.append(
            f"⚡ Обнаружена частая связка действий: «{t0['from_label']}» ➔ «{t0['to_label']}» ({t0['count']} раз подряд). Эту пару действий можно объединить в одну комбинированную кнопку-макрос в 1 клик!"
        )

    if sorted_missed_hk and int(sorted_missed_hk[0].get("mouse_clicks") or 0) >= 3:
        m0 = sorted_missed_hk[0]
        ux_recommendations.append(
            f"⌨️ Потенциал ускорения: вы нажали мышкой «{m0['label']}» {m0['mouse_clicks']} раз(а). Можно либо вынести эту кнопку ближе к центру холста/попапа, либо использовать горячую клавишу {m0['hotkey']}."
        )

    settings_clicks = int(zones_dict.get("settings_modal") or 0)
    if total_ev > 10 and (settings_clicks / max(1, total_ev)) >= 0.22:
        ux_recommendations.append(
            f"⚙️ Вы довольно часто заходите в модальное окно Настроек ({settings_clicks} действий, {round(settings_clicks / max(1, total_ev) * 100, 1)}% всех кликов). Те переключатели, ради которых вы открываете настройки, стоит вынести прямо на верхнюю панель!"
        )

    if len(unused_features) >= 5 and total_ev >= 15:
        ux_recommendations.append(
            f"🧹 Из {len(KNOWN_UI_FEATURES_CATALOG)} инструментов интерфейса вы пока ни разу не использовали {len(unused_features)} шт. (см. Раздел 6). Их можно спрятать под компактное меню «⋯ Еще», чтобы разгрузить интерфейс и сделать основные кнопки крупнее!"
        )

    if not ux_recommendations:
        ux_recommendations.append(
            "💡 Поработайте в программе в своем обычном ритме (переведите несколько страниц, поправьте бабблы, шрифты или маску), а затем скопируйте этот отчет в наш чат — и мы перестроим интерфейс идеально под вашу руку!"
        )

    mouse_vs_key_total = max(1, mouse_clicks + hotkey_presses)
    mouse_pct = round((mouse_clicks / mouse_vs_key_total) * 100.0, 1)
    key_pct = round((hotkey_presses / mouse_vs_key_total) * 100.0, 1)

    overflow_names_ru = {
        "shrink": "Стандарт (внутри рамки до мин. шрифта)",
        "overflow": "Вариант Б — Выход текста за рамки баббла (с обводкой)",
        "expand_bubble": "Вариант В — Авто-перерисовка (расширение) баббла под текст",
    }
    lines: list[str] = [
        "========================================================================================",
        "  📊 ОТЧЕТ UX-АНАЛИТИКИ И ТЕЛЕМЕТРИИ ИНТЕРФЕЙСА — MIKO-CHAN MANGA STUDIO v1.0.7",
        "========================================================================================",
        f"• Файл отчета              : {UX_REPORT_TXT_FILE}",
        f"• База накопленной телеметрии: {UX_TELEMETRY_JSON_FILE}",
        f"• Статистика собирается с  : {store.get('created_at', '—')} (обновлено: {store.get('updated_at', '—')})",
        f"• Активное время работы    : {active_min} мин ({active_sec:.0f} сек)",
        f"• Всего зафиксировано действ.: {total_ev} (Кликов мышью: {mouse_clicks} [{mouse_pct}%] | Хоткеев клавиатуры: {hotkey_presses} [{key_pct}%] | Манипуляций с рамками/холстом: {canvas_ops_cnt} | Правок текста: {text_edits_cnt})",
        "",
        "----------------------------------------------------------------------------------------",
        "  🏆 1. ТОП-25 САМЫХ ЧАСТЫХ КНОПОК И ДЕЙСТВИЙ (ЧТО НУЖНО ДЕРЖАТЬ ПОД РУКОЙ В 1 КЛИК)",
        "----------------------------------------------------------------------------------------",
    ]

    if not sorted_actions:
        lines.append("  (Пока нет записанных кликов — понажимайте кнопки в интерфейсе, и здесь появится рейтинг)")
    else:
        lines.append(
            "  #   | Нажатий | Доля   | Тип            | Зона интерфейса                | Кнопка / Действие (и ID элемента)"
        )
        lines.append("  " + "-" * 132)
        for idx, act in enumerate(sorted_actions[:25], start=1):
            cnt = int(act.get("count") or 0)
            pct = round((cnt / max(1, total_ev)) * 100.0, 1)
            ev_t = str(act.get("event_type") or "click")
            z_short = str(act.get("zone") or "other")[:30]
            lbl = str(act.get("label") or act.get("action_id") or "—")
            aid = str(act.get("action_id") or "")
            lines.append(
                f"  #{idx:02d} | {cnt:>7d} | {pct:>5.1f}% | {ev_t:<14s} | {z_short:<30s} | {lbl} [{aid}]"
            )

    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            "  🗺 2. РАСПРЕДЕЛЕНИЕ КЛИКОВ ПО ЗОНАМ ЭКРАНА (ГДЕ ВЫ ПРОВОДИТЕ БОЛЬШЕ ВСЕГО ВРЕМЕНИ)",
            "----------------------------------------------------------------------------------------",
        ]
    )
    if not sorted_zones:
        lines.append("  (Нет данных по зонам экрана)")
    else:
        for z_key, z_cnt in sorted_zones:
            z_pct = round((int(z_cnt) / max(1, total_ev)) * 100.0, 1)
            z_title = UX_ZONE_LABELS_RU.get(z_key, z_key)
            bar_len = max(1, int(round(z_pct / 4.0))) if z_cnt > 0 else 0
            bar_str = "█" * bar_len
            lines.append(f"  • {z_cnt:>5d} ({z_pct:>5.1f}%) [{bar_str:<25s}] — {z_title}")

    lines.extend(
        [
            "",
            "[Предпочтения по режиму просмотра и расположению редактора]",
            f"  • Расположение редактора : ➡ Сбоку={dwell.get('dock_right', 0):.0f}с | ⬇ Снизу={dwell.get('dock_bottom', 0):.0f}с | 🪟 Плавающее окно={dwell.get('dock_floating', 0):.0f}с",
            f"  • Режим просмотра манги  : 🔀 До и После={dwell.get('view_split', 0):.0f}с | 🈯 Перевод={dwell.get('view_translated', 0):.0f}с | 🇯🇵 Оригинал={dwell.get('view_original', 0):.0f}с",
            f"  • Где вы правите текст   : 💬 Всплывающий попап у кружка={popup_edits} раз | 📋 Карточки в боковой панели={drawer_edits} раз",
        ]
    )

    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            "  🎯 3. РАБОТА С РАМКАМИ #N НА ХОЛСТЕ, ШРИФТАМИ, ПОДЛОЖКОЙ И КИСТЬЮ МАСКИ",
            "----------------------------------------------------------------------------------------",
        ]
    )
    if not sorted_canvas:
        lines.append("  (Манипуляций с рамками #N, колесиком шрифта или кистью на холсте пока не зафиксировано)")
    else:
        for c_item in sorted_canvas[:20]:
            lines.append(
                f"  • {int(c_item.get('count') or 0):>5d} раз — {c_item.get('label')} [{c_item.get('action_id')}]"
            )

    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            "  🔗 4. ТОП-15 СВЯЗОК ДЕЙСТВИЙ ПОДРЯД («ДЕЙСТВИЕ А ➔ ДЕЙСТВИЕ Б» — КАНДИДАТЫ НА МАКРО-КНОПКУ)",
            "----------------------------------------------------------------------------------------",
        ]
    )
    if not sorted_transitions:
        lines.append("  (Цепочки последовательных действий появятся после нескольких операций в программе)")
    else:
        for tr_idx, tr in enumerate(sorted_transitions[:15], start=1):
            lines.append(
                f"  #{tr_idx:02d} ({int(tr.get('count') or 0):>3d} раз подряд): «{tr.get('from_label')}»  ➔  «{tr.get('to_label')}»"
            )

    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            "  ⌨️ 5. МЫШКА ПРОТИВ ГОРЯЧИХ КЛАВИШ (ГДЕ МОЖНО СЭКОНОМИТЬ ЛИШНИЕ КЛИКИ)",
            "----------------------------------------------------------------------------------------",
            "  5А. Используемые вами горячие клавиши клавиатуры:",
        ]
    )
    if not sorted_hotkeys:
        lines.append("      (Нажатий горячих клавиш пока не зафиксировано)")
    else:
        for hk in sorted_hotkeys[:15]:
            lines.append(f"      • {int(hk.get('count') or 0):>4d} раз — [{hk.get('hotkey')}] {hk.get('label')}")

    lines.append("")
    lines.append("  5Б. Кнопки, которые вы нажимаете мышкой, хотя для них есть быстрая клавиша (или их стоит вынести ближе):")
    if not sorted_missed_hk:
        lines.append("      (Лишних кликов мышкой по кнопкам с хоткеями не обнаружено)")
    else:
        for mhk in sorted_missed_hk[:12]:
            lines.append(
                f"      • Нажато мышкой {int(mhk.get('mouse_clicks') or 0):>4d} раз: «{mhk.get('label')}» — есть клавиша [{mhk.get('hotkey')}] ({mhk.get('description')})"
            )

    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            f"  🧹 6. КАРТА ВОСТРЕБОВАННОСТИ ФУНКЦИЙ (ИСПОЛЬЗУЕТСЯ: {len(used_features_ranked)} / НЕ ИСПОЛЬЗУЕТСЯ: {len(unused_features)})",
            "----------------------------------------------------------------------------------------",
            "  6А. Активно используемые вами возможности:",
        ]
    )
    if not used_features_ranked:
        lines.append("      (Список заполнится по мере работы в программе)")
    else:
        for f_key, f_title, f_cat, f_cnt in used_features_ranked:
            lines.append(f"      • ✅ {f_cnt:>4d} раз | [{f_cat}] {f_title}")

    lines.append("")
    lines.append("  6Б. Функции и кнопки, которыми вы НИ РАЗУ не пользовались (кандидаты на скрытие в меню «Еще»):")
    if not unused_features:
        lines.append("      🎉 Вы опробовали абсолютно все инструменты студии!")
    else:
        for f_key, f_title, f_cat in unused_features:
            lines.append(f"      • ⚪ 0 раз   | [{f_cat}] {f_title}")

    lines.extend(
        [
            "",
            "----------------------------------------------------------------------------------------",
            "  💡 7. АВТОМАТИЧЕСКИЕ ВЫВОДЫ И РЕКОМЕНДАЦИИ ДЛЯ ОПТИМИЗАЦИИ UI / UX",
            "----------------------------------------------------------------------------------------",
        ]
    )
    for rec_line in ux_recommendations:
        lines.append(f"  • {rec_line}")

    recent_ev = store.get("recent_events") or []
    if recent_ev:
        lines.extend(
            [
                "",
                "----------------------------------------------------------------------------------------",
                f"  🕒 8. ХРОНОЛОГИЯ ПОСЛЕДНИХ ДЕЙСТВИЙ ПОЛЬЗОВАТЕЛЯ (ПОСЛЕДНИЕ {min(35, len(recent_ev))} СОБЫТИЙ)",
                "----------------------------------------------------------------------------------------",
            ]
        )
        for rev in recent_ev[-35:]:
            lines.append(
                f"  [{rev.get('time')}] ({rev.get('zone')}) {rev.get('label')} [{rev.get('action_id')}]"
            )

    lines.extend(
        [
            "",
            "========================================================================================",
            "  КОНЕЦ ОТЧЕТА UX-АНАЛИТИКИ (СКОПИРУЙТЕ ЕГО В ЧАТ ДЛЯ ПЕРСОНАЛЬНОЙ НАСТРОЙКИ ИНТЕРФЕЙСА)",
            "========================================================================================",
            "",
        ]
    )

    report_text = "\n".join(lines)
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        UX_REPORT_TXT_FILE.write_text(report_text, encoding="utf-8")
    except Exception as exc:
        logger.warning("Не удалось записать UX_USAGE_REPORT.txt: %s", exc)

    summary = {
        "total_events": total_ev,
        "total_mouse_clicks": mouse_clicks,
        "total_hotkey_presses": hotkey_presses,
        "total_canvas_manipulations": canvas_ops_cnt,
        "total_text_edits": text_edits_cnt,
        "active_usage_sec": active_sec,
        "top_actions": sorted_actions[:10],
        "top_transitions": sorted_transitions[:8],
        "zones": dict(sorted_zones),
        "used_features_count": len(used_features_ranked),
        "unused_features_count": len(unused_features),
        "recommendations": ux_recommendations,
        "report_path": str(UX_REPORT_TXT_FILE),
    }
    return report_text, summary


@app.post("/ux_telemetry/record")
async def record_ux_telemetry_endpoint(payload: UxEventsBatchRequest) -> dict[str, Any]:
    """Принимает пачку событий взаимодействия пользователя с интерфейсом и сохраняет статистику в logs/ux_telemetry.json."""
    store = await asyncio.to_thread(_ingest_ux_events_batch, payload)
    return {
        "status": "ok",
        "total_events": int(store.get("total_events") or 0),
        "total_mouse_clicks": int(store.get("total_mouse_clicks") or 0),
        "total_hotkey_presses": int(store.get("total_hotkey_presses") or 0),
    }


@app.get("/ux_telemetry/report")
async def get_ux_telemetry_report_endpoint() -> dict[str, Any]:
    """Формирует, сохраняет в logs/UX_USAGE_REPORT.txt и возвращает полный отчет UX-аналитики кликов и действий пользователя."""
    report_text, summary = await asyncio.to_thread(_generate_ux_usage_report)
    return {
        "status": "ok",
        "report_path": str(UX_REPORT_TXT_FILE),
        "report_text": report_text,
        "summary": summary,
    }


@app.post("/ux_telemetry/open_report")
async def open_ux_telemetry_report_endpoint() -> dict[str, str]:
    """Обновляет файл logs/UX_USAGE_REPORT.txt и открывает его в Блокноте Windows."""
    report_text, _ = await asyncio.to_thread(_generate_ux_usage_report)
    if os.name == "nt" and hasattr(os, "startfile") and UX_REPORT_TXT_FILE.exists():
        try:
            os.startfile(str(UX_REPORT_TXT_FILE))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "status": "ok",
        "report_path": str(UX_REPORT_TXT_FILE),
    }


@app.post("/ux_telemetry/clear")
async def clear_ux_telemetry_endpoint() -> dict[str, str]:
    """Сбрасывает накопленную статистику кликов UX-телеметрии для начала чистого замера."""
    with _ux_telemetry_lock:
        fresh = _new_empty_ux_store()
        _save_ux_store(fresh)
        try:
            if UX_EVENTS_JSONL_FILE.exists():
                UX_EVENTS_JSONL_FILE.unlink()
        except Exception:
            pass
    await asyncio.to_thread(_generate_ux_usage_report, fresh)
    logger.info("Статистика UX-телеметрии кликов и действий сброшена пользователем.")
    return {"status": "ok"}


def _is_server_already_running(url: str = "http://127.0.0.1:5000/health") -> bool:
    try:
        with httpx.Client(trust_env=False, timeout=2.5) as client:
            r = client.get(url)
            if r.status_code == 200:
                data = r.json()
                running_ver = str(data.get("version") or "")
                if running_ver != "1.0.6":
                    print(
                        f"[UPDATE] На порту 5000 запущена старая версия сервера (v{running_ver}). "
                        f"Останавливаю старый процесс для запуска v1.0.6...",
                        flush=True,
                    )
                    try:
                        client.post("http://127.0.0.1:5000/shutdown", timeout=2.0)
                    except Exception:
                        pass
                    time.sleep(1.5)
                    return False
                return True
    except Exception:
        return False
    return False


if __name__ == "__main__":
    if "--open-gui" in sys.argv:
        _AUTO_OPEN_GUI = True

    if _is_server_already_running():
        print("[OK] Локальный сервер (127.0.0.1:5000) уже запущен!", flush=True)
        if _AUTO_OPEN_GUI:
            launch_studio_window("http://127.0.0.1:5000/")
        sys.exit(0)

    print("Запуск локального сервера Manga Translator v1.0.6 на 127.0.0.1:5000...", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=False)