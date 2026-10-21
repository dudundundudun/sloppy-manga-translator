"""
FastAPI-сервер для локального переводчика манги, манхвы и маньхуа (v1.0.0).

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
from translator import CLOUD_API_PRESETS, RECOMMENDED_LOCAL_MODELS, OllamaTranslator, japanese_to_romaji_local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("manga_server")

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp_debug"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VALID_SOURCE_LANGS = {"auto", "ja", "ko", "zh"}
VALID_TARGET_LANGS = {"en", "ru", "romaji"}
OUTPUT_FOLDER_SUFFIXES = ("_EN", "_RU", "_ROMAJI", "temp_debug")
MANIFEST_FILENAME = ".translation_state.json"

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
    """Гарантирует наличие исходного `clean_bbox` у блока, чтобы при сдвиге рамки инпейнтинг оставался на месте."""
    bbox_raw = block.get("bbox") or [0, 0, 0, 0]
    bbox = [int(round(float(x))) for x in bbox_raw[:4]] if len(bbox_raw) >= 4 else [0, 0, 0, 0]
    block["bbox"] = bbox

    clean_raw = block.get("clean_bbox") or block.get("original_bbox")
    if isinstance(clean_raw, (list, tuple)) and len(clean_raw) >= 4:
        clean_bbox = [int(round(float(x))) for x in clean_raw[:4]]
    else:
        clean_bbox = list(bbox)
    block["clean_bbox"] = clean_bbox
    block["bbox_modified"] = bool(block.get("bbox_modified", False)) or (clean_bbox != bbox)
    return block


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
    cleaner_engine = BubbleCleaner(use_lama=True, dilate_size=7)

    print("  [3/4] Инициализация шрифтового движка и клиента Ollama/Cloud API...", flush=True)
    renderer_engine = MangaRenderer(fonts_dir=BASE_DIR / "fonts")
    translator_engine = OllamaTranslator(config_path=BASE_DIR / "config.json")
    resolved_model = await translator_engine.resolve_model()

    print("  [4/4] СЕРВЕР ГОТОВ К РАБОТЕ! (v1.0.0)", flush=True)
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
    version="1.0.0",
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


class TextBlock(BaseModel):
    id: int
    bbox: list[int]
    clean_bbox: list[int] | None = Field(
        default=None,
        description="Исходные координаты оригинального текста для инпейнтинга (не смещаются при перетаскивании рамки перевода)",
    )
    bbox_modified: bool = Field(
        default=False,
        description="Если True, пользователь вручную передвинул или изменил размер рамки перевода мышкой",
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
    detect_free_text: bool | None = Field(default=None)
    text_filter_mode: str | None = Field(default=None)
    skip_sighs: bool | None = Field(default=None)
    skip_sfx: bool | None = Field(default=None)
    min_text_length: int | None = Field(default=None)
    skip_existing: bool = Field(default=True)
    force_retranslate: bool = Field(default=False)
    batch_pages: int = Field(default=12, description="Сколько страниц сканировать через OCR и отправлять в ИИ одним пакетом (экономия токенов)")
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
    detect_free_text: bool | None = Field(default=None)
    text_filter_mode: str | None = Field(default=None)
    skip_sighs: bool | None = Field(default=None)
    skip_sfx: bool | None = Field(default=None)
    min_text_length: int | None = Field(default=None)
    return_image: bool = Field(default=True)
    image_format: str = Field(default="webp")


class ServerSettingsRequest(BaseModel):
    unsloth_url: str | None = Field(default=None)
    unsloth_api_key: str | None = Field(default=None)
    selected_model: str | None = Field(default=None)
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
    return_image: bool = Field(default=True)
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
    uppercase: bool = Field(default=True)
    min_font_size: int = Field(default=12)
    max_font_size: int = Field(default=32)
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
    uppercase: bool = Field(default=True)
    min_font_size: int = Field(default=12)
    max_font_size: int = Field(default=32)
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
                return data
        except Exception as exc:
            logger.warning("Не удалось прочитать %s: %s", manifest_path, exc)
    return {"pages": {}, "rolling_context": [], "phrase_cache": {}}


def _save_folder_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Не удалось сохранить %s: %s", MANIFEST_FILENAME, exc)


def is_page_already_translated(dest_path: Path) -> bool:
    try:
        return dest_path.exists() and dest_path.is_file() and dest_path.stat().st_size > 128
    except OSError:
        return False


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
        done = is_page_already_translated(dest_path)
        if done:
            translated_count += 1

        meta = pages_meta.get(rel_norm, {}) if isinstance(pages_meta.get(rel_norm), dict) else {}
        has_saved_blocks_key = "blocks" in meta and isinstance(meta.get("blocks"), list)
        saved_blocks_raw = meta.get("blocks") if has_saved_blocks_key else []
        phrase_cache_dict = manifest.get("phrase_cache") if isinstance(manifest.get("phrase_cache"), dict) else {}
        enriched_saved_blocks: list[dict[str, Any]] = []
        for sb in saved_blocks_raw:
            if not isinstance(sb, dict):
                continue
            sb_copy = _ensure_block_clean_bbox(dict(sb))
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
                "is_translated": done,
                "blocks_count": meta.get("blocks_count", len(enriched_saved_blocks) if enriched_saved_blocks else None),
                "free_text_count": meta.get("free_text_count", 0),
                "detected_lang": meta.get("detected_lang"),
                "updated_at": meta.get("updated_at"),
                "has_saved_blocks": has_saved_blocks_key,
                "blocks": enriched_saved_blocks,
            }
        )

    return {
        "status": "ok",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "target_lang": t_lang,
        "total_pages": len(images),
        "translated_pages": translated_count,
        "remaining_pages": max(0, len(images) - translated_count),
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
    detect_free_text: bool | None = None,
    text_filter_mode: str | None = None,
    skip_sighs: bool | None = None,
    skip_sfx: bool | None = None,
    min_text_length: int | None = None,
    context_lines: list[str] | None = None,
    external_cache: dict[str, str] | None = None,
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

    translator_engine.stop_requested = False
    active_model = await translator_engine.resolve_model(ollama_model)

    if not blocks:
        empty_mask = Image.new("L", pil_image.size, 0)
        return pil_image.copy(), pil_image.copy(), empty_mask, [], active_model, detected_lang

    blocks = await translator_engine.translate_blocks(
        blocks=blocks,
        source_lang=detected_lang,
        target_lang=target_lang,
        model=active_model,
        context_lines=context_lines,
        external_cache=external_cache,
        skip_sighs=eff_skip_sighs,
    )

    # Оставляем только блоки с непустым переводом, чтобы не замазывать пропущенные вздохи
    blocks = [b for b in blocks if str(b.get("translated_text") or "").strip()]
    for idx_b, blk in enumerate(blocks, start=1):
        blk["id"] = idx_b
        blk["is_deleted"] = False
        _ensure_block_clean_bbox(blk)

    if not blocks:
        empty_mask = Image.new("L", pil_image.size, 0)
        return pil_image.copy(), pil_image.copy(), empty_mask, [], active_model, detected_lang

    async with gpu_lock:
        cleaned_image, mask_image = await asyncio.to_thread(
            cleaner_engine.clean_image, pil_image, blocks
        )

    final_image = await asyncio.to_thread(
        renderer_engine.render_blocks,
        cleaned_image,
        blocks,
        uppercase,
        max_font_size,
        min_font_size,
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
    batch_pages: int = 12,
    text_filter_mode: str | None = None,
    skip_sighs: bool | None = None,
    skip_sfx: bool | None = None,
    min_text_length: int | None = None,
) -> None:
    global _folder_stop_requested
    assert ocr_engine is not None
    assert translator_engine is not None
    assert cleaner_engine is not None
    assert renderer_engine is not None

    _folder_stop_requested = False
    translator_engine.stop_requested = False
    start_ts = time.perf_counter()
    pages_per_batch = max(1, min(25, int(batch_pages or 12)))

    manifest = _load_folder_manifest(output_dir)
    if not isinstance(manifest.get("pages"), dict):
        manifest["pages"] = {}
    if not isinstance(manifest.get("phrase_cache"), dict):
        manifest["phrase_cache"] = {}
    rolling_context: list[str] = list(manifest.get("rolling_context") or [])[-12:]
    phrase_cache: dict[str, str] = dict(manifest.get("phrase_cache") or {})

    to_process: list[tuple[int, Path, Path, str]] = []
    skipped_count = 0

    for idx, img_path in enumerate(image_paths, start=1):
        rel = img_path.relative_to(source_dir)
        rel_norm = _normalize_rel_path(rel)
        dest_path = output_dir / rel
        if skip_existing and not force_retranslate and is_page_already_translated(dest_path):
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
        active_mod = await translator_engine.resolve_model(ollama_model)
        chapter_detected_lang: str | None = lang if lang in ("ja", "ko", "zh") else None
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

        # Обрабатываем страницы пачками по pages_per_batch (по умолчанию 12 страниц)
        for batch_start in range(0, len(to_process), pages_per_batch):
            if _folder_stop_requested:
                folder_job_state["status"] = "stopped"
                folder_job_state["current_file"] = f"Пауза ({folder_job_state['current']}/{len(image_paths)} готово)"
                _save_folder_manifest(output_dir, manifest)
                return

            page_batch = to_process[batch_start : batch_start + pages_per_batch]
            scanned_pages: list[dict[str, Any]] = []
            combined_blocks: list[dict[str, Any]] = []
            batch_detected_lang = chapter_detected_lang or "ja"
            global_block_id = 1

            # Параллельно предзагружаем изображения пачки с диска в ОЗУ (без ожидания внутри GPU-цикла)
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
                    if chapter_detected_lang is None and det_lang in ("ja", "ko", "zh"):
                        chapter_detected_lang = det_lang
                    batch_detected_lang = det_lang
                    folder_job_state["detected_lang"] = det_lang

                    page_global_ids: list[int] = []
                    for b in p_blocks:
                        _ensure_block_clean_bbox(b)
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
                            "ok": True,
                        }
                    )
                except Exception as ocr_exc:
                    logger.exception("Ошибка OCR при чтении файла %s: %s", img_path, ocr_exc)
                    shutil.copy2(img_path, dest_path)
                    scanned_pages.append({"img_path": img_path, "dest_path": dest_path, "rel_norm": rel_norm, "ok": False})

            if _folder_stop_requested:
                folder_job_state["status"] = "stopped"
                folder_job_state["current_file"] = f"Пауза ({folder_job_state['current']}/{len(image_paths)} готово)"
                _save_folder_manifest(output_dir, manifest)
                return

            # Шаг 2: Параллельный ИИ-перевод пачки (автоматически делится на 2-3 параллельных потока в translator.py!)
            translated_by_gid: dict[int, str] = {}
            if combined_blocks:
                folder_job_state["current_file"] = (
                    f"☁️ [2/3 ИИ-перевод пачки: {len(scanned_pages)} стр. / {len(combined_blocks)} фраз]..."
                )
                folder_job_state["elapsed_sec"] = round(time.perf_counter() - start_ts, 1)
                try:
                    enriched_combined = await translator_engine.translate_blocks(
                        blocks=combined_blocks,
                        source_lang=batch_detected_lang,
                        target_lang=target_lang,
                        model=active_mod,
                        context_lines=rolling_context if rolling_context else None,
                        external_cache=phrase_cache,
                        skip_sighs=eff_skip_sighs,
                    )
                    for eb in enriched_combined:
                        translated_by_gid[int(eb["id"])] = str(eb.get("translated_text") or "").strip()
                except Exception as llm_exc:
                    logger.exception("Ошибка пакетного ИИ-перевода пачки страниц: %s", llm_exc)

            if _folder_stop_requested:
                folder_job_state["status"] = "stopped"
                folder_job_state["current_file"] = f"Пауза ({folder_job_state['current']}/{len(image_paths)} готово)"
                _save_folder_manifest(output_dir, manifest)
                return

            # Шаг 3: Быстрая параллельная очистка бабблов, верстка и сохранение всех страниц пачки!
            for sp in scanned_pages:
                if not sp.get("ok"):
                    continue
                p_blocks = sp["blocks"]
                gids = sp["global_ids"]
                for b, gid in zip(p_blocks, gids):
                    t_txt = translated_by_gid.get(gid, "")
                    b["translated_text"] = t_txt
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
                if not blocks_in:
                    shutil.copy2(src_p, dst_p)
                    return
                cleaned_im, _ = cleaner_engine.clean_image(pil_img_in, blocks_in)
                final_im = renderer_engine.render_blocks(
                    cleaned_im,
                    blocks_in,
                    uppercase,
                    max_font_size,
                    min_font_size,
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
                        "updated_at": int(time.time()),
                        "blocks": p_blocks,
                    }
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

            manifest["rolling_context"] = rolling_context
            manifest["phrase_cache"] = phrase_cache
            _save_folder_manifest(output_dir, manifest)

            if _folder_stop_requested:
                folder_job_state["status"] = "stopped"
                folder_job_state["current_file"] = f"Пауза ({folder_job_state['current']}/{len(image_paths)} готово)"
                return

        total_elapsed = round(time.perf_counter() - start_ts, 2)
        folder_job_state["status"] = "completed"
        folder_job_state["elapsed_sec"] = total_elapsed
        folder_job_state["current_file"] = "Готово!"

        if open_when_done and os.name == "nt" and output_dir.exists():
            try:
                os.startfile(str(output_dir))
            except Exception:
                pass

    except asyncio.CancelledError:
        folder_job_state["status"] = "stopped"
        folder_job_state["current_file"] = "Пауза"
        _save_folder_manifest(output_dir, manifest)
    except Exception as exc:
        logger.exception("Критическая ошибка пакетного перевода папки")
        folder_job_state["status"] = "error"
        folder_job_state["error"] = str(exc)


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def studio_web_ui() -> HTMLResponse:
    html_path = BASE_DIR / "web_studio.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Manga Translator v1.0.0 Server Running</h1>")


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
    has_multilang_engine = bool(ocr_engine and hasattr(ocr_engine, "_check_easyocr"))
    easyocr_ok = bool(ocr_engine._check_easyocr()) if has_multilang_engine else False

    return {
        "status": "ready" if ocr_engine is not None else "initializing",
        "version": "1.0.0",
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
        "batch_pages": translator_engine.batch_pages if translator_engine else 12,
        "ui_settings": translator_engine.ui_settings if translator_engine else {},
        "pull_state": translator_engine.pull_state if translator_engine else {},
        "supported_source_langs": ["auto", "ja", "ko", "zh"],
        "default_target_lang": "en",
        "supported_target_langs": ["en", "ru", "romaji"],
        "easyocr_installed": easyocr_ok,
        "ocr_engine_updated": has_multilang_engine,
        "inpaint_backend": inpaint_backend,
        "font": renderer_engine.font_path if renderer_engine is not None else None,
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
    saved = translator_engine.save_config(
        unsloth_url=payload.unsloth_url,
        unsloth_api_key=payload.unsloth_api_key,
        selected_model=payload.selected_model,
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
        batch_pages=payload.batch_pages,
        ui_settings=payload.ui_settings,
    )
    models = await translator_engine.list_available_models(force_refresh=True)
    active = await translator_engine.resolve_model()
    return {
        "status": "ok",
        "active_model": active,
        "selected_model": saved["selected_model"],
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

    manifest["pages"][filename] = {
        "blocks_count": len(blocks),
        "free_text_count": sum(1 for b in blocks if b.get("is_free_text")),
        "detected_lang": detected_lang,
        "model": active_model,
        "uppercase": uppercase,
        "min_font_size": min_font_size,
        "max_font_size": max_font_size,
        "page_url": page_url,
        "updated_at": int(time.time()),
        "blocks": blocks,
    }
    _save_folder_manifest(output_chapter_dir, manifest)
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
                detect_free_text=payload.detect_free_text,
                text_filter_mode=payload.text_filter_mode,
                skip_sighs=payload.skip_sighs,
                skip_sfx=payload.skip_sfx,
                min_text_length=payload.min_text_length,
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
    if output_dir is not None:
        manifest = _load_folder_manifest(output_dir)
        ctx = manifest.get("rolling_context")
        if isinstance(ctx, list) and ctx:
            context_lines = [str(x) for x in ctx[-12:]]
        if isinstance(manifest.get("phrase_cache"), dict):
            phrase_cache = dict(manifest["phrase_cache"])

    final_image, _, _, blocks, active_model, detected_lang = await process_single_pil_image(
        pil_image=pil_image,
        lang=req_lang,
        target_lang=target_lang,
        ollama_model=(payload.ollama_model or payload.model),
        uppercase=payload.uppercase,
        min_font_size=payload.min_font_size,
        max_font_size=payload.max_font_size,
        detect_free_text=payload.detect_free_text,
        text_filter_mode=payload.text_filter_mode,
        skip_sighs=payload.skip_sighs,
        skip_sfx=payload.skip_sfx,
        min_text_length=payload.min_text_length,
        context_lines=context_lines,
        external_cache=phrase_cache,
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
        manifest["pages"][rel_norm] = {
            "blocks_count": len(blocks),
            "free_text_count": free_cnt,
            "detected_lang": detected_lang,
            "model": active_model,
            "uppercase": payload.uppercase,
            "min_font_size": payload.min_font_size,
            "max_font_size": payload.max_font_size,
            "updated_at": int(time.time()),
            "blocks": blocks,
        }
        if translator_engine is not None and translator_engine.recent_context_lines:
            manifest["rolling_context"] = list(translator_engine.recent_context_lines[-12:])
        manifest["phrase_cache"] = phrase_cache
        _save_folder_manifest(output_dir, manifest)

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
            if b_copy.get("is_deleted"):
                b_copy["is_deleted"] = True
                b_copy["translated_text"] = ""
            else:
                b_copy["is_deleted"] = False
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

    if not active_blocks:
        final_image = pil_image.copy()
    else:
        async with gpu_lock:
            cleaned_image, _ = await asyncio.to_thread(
                cleaner_engine.clean_image, pil_image, active_blocks
            )
        final_image = await asyncio.to_thread(
            renderer_engine.render_blocks,
            cleaned_image,
            active_blocks,
            payload.uppercase,
            payload.max_font_size,
            payload.min_font_size,
        )

    save_kwargs: dict[str, Any] = {}
    if dest_path.suffix.lower() in (".jpg", ".jpeg", ".webp"):
        save_kwargs["quality"] = 95
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
        "blocks_count": len(blocks_dicts),
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
    context_lines = [str(x) for x in rolling_ctx[-10:]] if isinstance(rolling_ctx, list) and rolling_ctx else None

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
        async with gpu_lock:
            cleaned_image, _ = await asyncio.to_thread(
                cleaner_engine.clean_image, pil_image, active_blocks
            )
        final_image = await asyncio.to_thread(
            renderer_engine.render_blocks,
            cleaned_image,
            active_blocks,
            payload.uppercase,
            payload.max_font_size,
            payload.min_font_size,
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
    det_lang = payload.lang if payload.lang in ("ja", "ko", "zh") else "ja"
    if not orig_t or orig_t == "[Выделенная область]":
        ocr_coords = target_blk.get("clean_bbox") or target_blk["bbox"]
        async with gpu_lock:
            rec_t, det_lang, _ = await asyncio.to_thread(
                ocr_engine.recognize_manual_region,
                pil_image,
                ocr_coords,
                payload.lang,
            )
        if rec_t:
            orig_t = rec_t
            target_blk["original_text"] = orig_t

    if orig_t:
        active_mod = await translator_engine.resolve_model(payload.ollama_model)
        translated_one = await translator_engine.translate_blocks(
            blocks=[target_blk],
            source_lang=det_lang,
            target_lang=payload.target_lang,
            model=active_mod,
            use_cache=False,
            skip_sighs=False,
        )
        if translated_one and translated_one[0].get("translated_text"):
            target_blk["translated_text"] = translated_one[0]["translated_text"]
            target_blk["romaji_text"] = translated_one[0].get("romaji_text") or target_blk.get("romaji_text") or ""
            target_blk["is_deleted"] = False

    active_blocks = [
        b for b in blocks_dicts if not bool(b.get("is_deleted")) and str(b.get("translated_text") or "").strip()
    ]
    if not active_blocks:
        final_image = pil_image.copy()
    else:
        async with gpu_lock:
            cleaned_image, _ = await asyncio.to_thread(
                cleaner_engine.clean_image, pil_image, active_blocks
            )
        final_image = await asyncio.to_thread(
            renderer_engine.render_blocks,
            cleaned_image,
            active_blocks,
            payload.uppercase,
            payload.max_font_size,
            payload.min_font_size,
        )

    save_kwargs: dict[str, Any] = {}
    if dest_path.suffix.lower() in (".jpg", ".jpeg", ".webp"):
        save_kwargs["quality"] = 95
    final_image.save(dest_path, **save_kwargs)

    manifest = _load_folder_manifest(output_dir)
    if not isinstance(manifest.get("pages"), dict):
        manifest["pages"] = {}
    prev_meta = manifest["pages"].get(rel_norm, {})
    if not isinstance(prev_meta, dict):
        prev_meta = {}
    prev_meta["blocks"] = blocks_dicts
    prev_meta["updated_at"] = int(time.time())
    manifest["pages"][rel_norm] = prev_meta
    _save_folder_manifest(output_dir, manifest)

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

    already_done = sum(
        1
        for p in images
        if payload.skip_existing
        and not payload.force_retranslate
        and is_page_already_translated(output_dir / p.relative_to(source_dir))
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
            skip_existing=payload.skip_existing,
            force_retranslate=payload.force_retranslate,
            open_when_done=payload.open_when_done,
            batch_pages=payload.batch_pages,
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


def _is_server_already_running(url: str = "http://127.0.0.1:5000/health") -> bool:
    try:
        with httpx.Client(trust_env=False, timeout=2.5) as client:
            r = client.get(url)
            if r.status_code == 200:
                data = r.json()
                running_ver = str(data.get("version") or "")
                if running_ver != "1.0.0":
                    print(
                        f"[UPDATE] На порту 5000 запущена старая версия сервера (v{running_ver}). "
                        f"Останавливаю старый процесс для запуска v1.0.0...",
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

    print("Запуск локального сервера Manga Translator v1.0.0 на 127.0.0.1:5000...", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=False)
