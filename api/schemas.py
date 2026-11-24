"""
Pydantic-схемы запросов и ответов FastAPI-сервера Miko-chan Manga Studio.
Вынесены из server.py в рамках модульной декомпозиции (Шаг 6 аудита).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    image: str | None = Field(default=None, description="Base64 изображение")
    image_base64: str | None = Field(default=None, description="Алиас для Base64 изображения")
    lang: str = Field(default="auto", description="Язык оригинала: 'auto', 'ja', 'ko' или 'zh'")
    target_lang: str = Field(default="en", description="Целевой язык: 'en', 'ru' или 'romaji'")
    ollama_model: str | None = Field(default=None)
    model: str | None = Field(default=None)
    uppercase: bool = Field(default=False)
    min_font_size: int = Field(default=12, ge=6, le=200, description="Минимальный размер шрифта (px)")
    max_font_size: int = Field(default=32, ge=8, le=300, description="Максимальный размер шрифта (px)")
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
        ge=1,
        le=50,
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
    page_index: int | None = Field(default=None, ge=1, le=100000, description="Порядковый номер картинки на веб-странице (1..N)")
    force_retranslate: bool = Field(
        default=False,
        description="Принудительно перевести все блоки заново через ИИ (игнорируя и обновляя старый кэш фраз)",
    )
    pipeline_mode: str | None = Field(
        default=None,
        description="Режим конвейера: 'fast' (Быстрый), 'balanced' (Оптимальный), 'accuracy' (Макс. точность)",
    )
    use_image_context: bool | None = Field(
        default=None,
        description="Передавать уменьшенное превью страницы с номерами бабблов #1..#N в мультимодальную модель (Image-Aware VLM Context)",
    )


class TextBlock(BaseModel):
    id: int
    bbox: list[int] = Field(..., min_length=4, max_length=4)
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
    is_merged_bubble: bool = Field(
        default=False,
        description="Если True, блок создан объединением нескольких соседних рамок (M)",
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
        ge=0,
        le=16,
        description="Толщина обводки текста (0..16 px). Если None — авто-расчет от кегля",
    )
    align: str = Field(
        default="center",
        description="Выравнивание текста: 'center', 'left' или 'right'",
    )
    angle: float = Field(
        default=0.0,
        ge=-45.0,
        le=45.0,
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
        ge=-24,
        le=48,
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
    font_too_small_warning: bool | None = Field(
        default=None,
        description="True, если выбранный кегль < 72% от базового диалогового кегля страницы (Блок 4.1)",
    )
    page_base_font_size: int | None = Field(
        default=None,
        description="Базовый гармонизированный размер шрифта диалога на странице (px)",
    )
    safe_bbox: list[int] | None = Field(
        default=None,
        description="Безопасная внутренняя зона бабла с учетом отступов [sx1, sy1, sx2, sy2]",
    )
    bubble_bbox: list[int] | None = Field(
        default=None,
        description="Границы сегментированной полости бабла [bx1, by1, bx2, by2]",
    )
    lir_bbox: list[int] | None = Field(
        default=None,
        description="Максимальный вписанный прямоугольник LIR [lx1, ly1, lx2, ly2] внутри полигона баббла (v5.4.0)",
    )
    decomp_mode: str | None = Field(
        default=None,
        description="Режим декомпозиции полигона баббла (single_convex, split_dumbbell, lir_clamped и т.д., v5.4.0)",
    )
    optical_center: list[int] | None = Field(
        default=None,
        description="Оптический центр бабла [cx, cy] по Pole of Inaccessibility",
    )
    aspect_ratio: float | None = Field(
        default=None,
        description="Отношение ширины к высоте бабла",
    )
    shape_hint: str | None = Field(
        default=None,
        description="Форма бабла: narrow_vertical, round_oval, wide_horizontal",
    )
    recommended_max_chars: int | None = Field(
        default=None,
        description="Рекомендуемый лимит символов для комфортной верстки в бабле",
    )
    speaker_id: str | None = Field(
        default=None,
        description="Идентификатор говорящего персонажа на странице (Speaker_A, Speaker_B, Monologue_Box, SFX_Free)",
    )
    speaker_tag: str | None = Field(
        default=None,
        description="Тег роли и говорящего для LLM ([Bubble: dialogue, speaker=A] или [Box: narration/monologue])",
    )
    box_role: str | None = Field(
        default=None,
        description="Топологическая роль рамки: dialogue, monologue, sfx",
    )
    tail_vector: list[float] | None = Field(
        default=None,
        description="Единичный вектор направления хвостика баббла [tail_dx, tail_dy]",
    )
    condense_ratio: float | None = Field(
        default=None,
        ge=0.5,
        le=1.0,
        description="Коэффициент горизонтального анаморфного сжатия строк (0.88..1.00) при полиграфическом тайпсеттинге",
    )
    mask_dilate: int | None = Field(
        default=None,
        ge=1,
        le=28,
        description="Индивидуальная толщина маски очистки для этого блока (1..28 px). Если None — авто",
    )
    custom_mask_strokes: list[dict[str, Any]] | None = Field(
        default=None,
        max_length=2000,
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
    uppercase: bool = Field(default=False)
    min_font_size: int = Field(default=12, ge=6, le=200)
    max_font_size: int = Field(default=32, ge=8, le=300)
    overflow_mode: str | None = Field(default=None)
    detect_free_text: bool | None = Field(default=None)
    text_filter_mode: str | None = Field(default=None)
    skip_sighs: bool | None = Field(default=None)
    skip_sfx: bool | None = Field(default=None)
    min_text_length: int | None = Field(default=None, ge=1, le=50)
    skip_existing: bool = Field(default=True)
    force_retranslate: bool = Field(default=False)
    batch_pages: int = Field(default=15, ge=1, le=50, description="Сколько страниц сканировать через OCR и отправлять в ИИ одним пакетом (экономия токенов)")
    translation_profile: str | None = Field(
        default=None,
        description="Профиль перевода: 'eco_book', 'balanced', 'turbo', 'local_only'",
    )
    pipeline_mode: str | None = Field(
        default=None,
        description="Режим работы конвейера: 'fast', 'balanced', 'accuracy'",
    )
    use_image_context: bool | None = Field(
        default=None,
        description="Включить передачу визуального контекста страницы в VLM при переводе",
    )
    anti_censorship_prompt: bool | None = Field(
        default=None,
        description="Если True, добавляет в самое начало системного промпта для DeepSeek усиленный анти-цензурный блок",
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
    uppercase: bool = Field(default=False)
    min_font_size: int = Field(default=12, ge=6, le=200)
    max_font_size: int = Field(default=32, ge=8, le=300)
    overflow_mode: str | None = Field(default=None)
    detect_free_text: bool | None = Field(default=None)
    text_filter_mode: str | None = Field(default=None)
    skip_sighs: bool | None = Field(default=None)
    skip_sfx: bool | None = Field(default=None)
    min_text_length: int | None = Field(default=None, ge=1, le=50)
    force_retranslate: bool = Field(
        default=False,
        description="Принудительно перевести все блоки страницы заново через ИИ (без старого кэша)",
    )
    return_image: bool = Field(default=True)
    image_format: str = Field(default="webp")
    pipeline_mode: str | None = Field(default=None)
    use_image_context: bool | None = Field(default=None)


class DeleteModelRequest(BaseModel):
    model_id: str = Field(description="ID или тег модели (например, 'ollama::qwen3:14b' или 'yolo26_seg')")
    category: str = Field(default="auto", description="'ollama', 'local_weights' или 'auto'")
    delete_gguf_cache: bool = Field(default=True, description="Удалять ли также скачанный .gguf файл из models/gguf/")


class ModelStackActionRequest(BaseModel):
    action: str = Field(
        default="apply_and_set_default",
        description="'apply_only', 'apply_and_set_default', 'save_preset', 'delete_preset'",
    )
    stack: dict[str, str] | None = Field(default=None)
    preset_id: str | None = Field(default=None)
    preset_name: str | None = Field(default=None)
    preset_description: str | None = Field(default=None)


class ServerSettingsRequest(BaseModel):
    unsloth_url: str | None = Field(default=None)
    default_model_stack: dict[str, str] | None = Field(default=None)
    unsloth_api_key: str | None = Field(default=None)
    clear_api_key: bool = Field(default=False)
    selected_model: str | None = Field(default=None)
    translation_profile: str | None = Field(default=None)
    pipeline_mode: str | None = Field(default=None)
    use_image_context: bool | None = Field(default=None)
    local_fallback_model_multilang: str | None = Field(default=None)
    anti_censorship_prompt: bool | None = Field(default=None)
    anti_censorship_custom_prompt: str | None = Field(default=None)
    preflight_nsfw_split: bool | None = Field(default=None)
    cloud_only_no_local_fallback: bool | None = Field(default=None)
    ultra_proofread_mode: str | None = Field(default=None)
    auto_save_web_pages: bool | None = Field(default=None)
    web_save_dir: str | None = Field(default=None)
    text_filter_mode: str | None = Field(default=None)
    skip_sighs: bool | None = Field(default=None)
    skip_sfx: bool | None = Field(default=None)
    detect_free_text: bool | None = Field(default=None)
    min_text_length: int | None = Field(default=None, ge=1, le=50)
    min_font_size: int | None = Field(default=None, ge=6, le=200)
    max_font_size: int | None = Field(default=None, ge=8, le=300)
    uppercase: bool | None = Field(default=None)
    overflow_mode: str | None = Field(default=None)
    batch_pages: int | None = Field(default=None, ge=1, le=50)
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
    min_text_length: int | None = Field(default=None, ge=1, le=50)
    force_rescan: bool = Field(default=False, description="Принудительно запустить OCR заново")
    use_llm_romaji: bool = Field(default=False, description="Уточнить чтение кандзи в Ромадзи через активную ИИ-модель")
    ollama_model: str | None = Field(default=None)


class PageRomajiRequest(BaseModel):
    source_path: str | None = Field(default=None)
    source_dir: str | None = Field(default=None)
    target_lang: str = Field(default="en")
    use_llm: bool = Field(default=True, description="Использовать ИИ для контекстного чтения кандзи в Ромадзи")
    ollama_model: str | None = Field(default=None)
    blocks: list[TextBlock] = Field(..., max_length=500)


class SaveEditedPageRequest(BaseModel):
    source_path: str = Field(description="Путь к исходному файлу страницы манги")
    dest_path: str | None = Field(default=None, description="Путь сохранения перерисованной страницы")
    source_dir: str | None = Field(default=None)
    target_lang: str = Field(default="en")
    uppercase: bool = Field(default=False)
    min_font_size: int = Field(default=12, ge=6, le=200)
    max_font_size: int = Field(default=32, ge=8, le=300)
    overflow_mode: str | None = Field(default=None)
    return_image: bool = Field(default=True)
    prefer_url: bool = Field(
        default=True,
        description="Если True и dest_path существует, возвращает легковесный URL /local_image вместо огромного Base64",
    )
    image_format: str = Field(default="webp")
    blocks: list[TextBlock] = Field(..., max_length=500)


class ManualRegionTranslateRequest(BaseModel):
    source_path: str = Field(description="Путь к исходному файлу страницы манги")
    dest_path: str | None = Field(default=None)
    source_dir: str | None = Field(default=None)
    bbox: list[int] = Field(..., min_length=4, max_length=4, description="Координаты обведенной области [x1, y1, x2, y2]")
    lang: str = Field(default="auto")
    target_lang: str = Field(default="en")
    ollama_model: str | None = Field(default=None)
    use_vlm_ocr: bool = Field(
        default=False,
        description="Принудительно использовать снайперский VLM-OCR (Vision-модель) для чтения сложного рукописного текста/SFX/обложки",
    )
    vlm_model: str | None = Field(default=None, description="Явное имя Vision-модели для VLM-OCR (опционально)")
    uppercase: bool = Field(default=False)
    min_font_size: int = Field(default=12, ge=6, le=200)
    max_font_size: int = Field(default=32, ge=8, le=300)
    overflow_mode: str | None = Field(default=None)
    return_image: bool = Field(default=True)
    image_format: str = Field(default="webp")
    custom_original_text: str | None = Field(default=None)
    custom_translated_text: str | None = Field(default=None)
    blocks: list[TextBlock] = Field(default_factory=list, max_length=500)


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
    uppercase: bool = Field(default=False)
    min_font_size: int = Field(default=12, ge=6, le=200)
    max_font_size: int = Field(default=32, ge=8, le=300)
    overflow_mode: str | None = Field(default=None)
    return_image: bool = Field(default=True)
    image_format: str = Field(default="webp")
    blocks: list[TextBlock] = Field(default_factory=list, max_length=500)


class DiamondWrapBlockRequest(BaseModel):
    block_id: int
    text: str | None = Field(default=None)
    bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    style_preset: str = Field(default="dialogue")
    font_family: str | None = Field(default=None)
    font_size: int | None = Field(default=None, ge=6, le=300)
    uppercase: bool = Field(default=False)
    source_path: str | None = Field(default=None)
    dest_path: str | None = Field(default=None)
    source_dir: str | None = Field(default=None)
    target_lang: str = Field(default="ru")
    min_font_size: int = Field(default=12, ge=6, le=200)
    max_font_size: int = Field(default=32, ge=8, le=300)
    overflow_mode: str | None = Field(default=None)
    return_image: bool = Field(default=True)
    prefer_url: bool = Field(default=True)
    image_format: str = Field(default="webp")
    blocks: list[TextBlock] = Field(default_factory=list, max_length=500)


class ShortenBlockTextRequest(BaseModel):
    block_id: int
    source_path: str | None = Field(default=None)
    dest_path: str | None = Field(default=None)
    source_dir: str | None = Field(default=None)
    target_lang: str = Field(default="ru")
    ollama_model: str | None = Field(default=None)
    uppercase: bool = Field(default=False)
    min_font_size: int = Field(default=12, ge=6, le=200)
    max_font_size: int = Field(default=32, ge=8, le=300)
    overflow_mode: str | None = Field(default=None)
    return_image: bool = Field(default=True)
    prefer_url: bool = Field(default=True)
    image_format: str = Field(default="webp")
    blocks: list[TextBlock] = Field(default_factory=list)


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
    uppercase: bool = Field(default=False)
    min_font_size: int = Field(default=12, ge=6, le=200)
    max_font_size: int = Field(default=32, ge=8, le=300)
    overflow_mode: str | None = Field(default=None)


class FolderCleanNoiseRequest(BaseModel):
    source_dir: str = Field(..., description="Путь к исходной папке главы")
    target_lang: str = Field(default="en", description="Целевой язык: en, ru, romaji")
    remove_free_text: bool = Field(default=True, description="Удалить все рамки вне бабблов (is_free_text)")
    remove_sighs: bool = Field(default=True, description="Удалить короткие вздохи и междометия")
    remove_sfx: bool = Field(default=True, description="Удалить звуковые спецэффекты (SFX)")
    remove_empty: bool = Field(default=True, description="Удалить пустые/удаленные рамки")


class ChapterBookImportRequest(BaseModel):
    folder_path: str = Field(..., description="Путь к папке главы манги")
    target_lang: str = Field(default="en", description="Целевой язык: en, ru, romaji")
    book_text: str = Field(..., max_length=5_000_000, description="Отредактированный текст книги (_chapter_book.md, _chapter_script_bilingual.txt или JSON)")
    uppercase: bool = Field(default=False)
    min_font_size: int = Field(default=12, ge=6, le=200)
    max_font_size: int = Field(default=32, ge=8, le=300)
    overflow_mode: str | None = Field(default=None)


class ChapterBookProofreadRequest(BaseModel):
    folder_path: str = Field(..., description="Путь к папке главы манги")
    target_lang: str = Field(default="en", description="Целевой язык: en, ru, romaji")
    ollama_model: str | None = Field(default=None)
    uppercase: bool = Field(default=False)
    min_font_size: int = Field(default=12, ge=6, le=200)
    max_font_size: int = Field(default=32, ge=8, le=300)
    overflow_mode: str | None = Field(default=None)


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
    events: list[UxEventItem] = Field(default_factory=list, max_length=500)
    active_dock_mode: str | None = Field(default=None)
    active_view_mode: str | None = Field(default=None)
    elapsed_active_sec: float = Field(default=0.0, description="Активное время работы пользователя за батч (сек)")
