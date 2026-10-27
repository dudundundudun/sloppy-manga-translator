"""
Модуль верстки и отрисовки переведенного текста в бабблах и на фоне — Typesetting (Miko-chan Manga Studio v3.5 Pro Scanlation Typesetting Edition).

Архитектура обновления v3.5 (Pro Scanlation Typesetting Edition):
1. Двухуровневая матрица профессиональных сканлейт-шрифтов (RU / EN) без `Comic Sans MS` (`comicbd.ttf`):
   - Приоритетный поиск профильных сканлейт-шрифтов в `fonts/` (`CC Wild Words`, `Anime Ace v05`, `Manga Temple`,
     `Chinacat`, `Badaboom BB`, `Komika Axis`) и качественный системный фоллбэк (`Segoe UI Semibold`, `Trebuchet MS`,
     `Segoe Print`, `Impact`, `Georgia Bold`).
   - Фильтр комиксной типографики `_format_comic_typography()` и правило `Crossbar-I` (`_apply_crossbar_i_rule()` по стандарту Blambot).
2. Сегментация реальных границ бабла и `Distance Transform` (`_analyze_bubble_geometry`):
   - Поиск внутренней маски бабла (`Bubble Interior Mask`) через морфологическое замыкание Canny/Sobel и `cv2.floodFill`
     с проверкой замкнутости (`Leakage Guard`).
   - Отсечение хвостика бабла (`cv2.MORPH_OPEN`) и поиск оптического центра (`Pole of Inaccessibility` через `cv2.distanceTransform`
     + вертикальная оптическая поправка `-2.2%`).
   - Безопасные отступы от стенок бабла (`safe_mask` 8–11% от малой оси) и извлечение профиля горизонтальных хорд `chord_width(y)`.
3. 2D-DP алгоритм «алмазной» верстки (`_wrap_text_contour_dp`):
   - Укладка строк по профилю суперэллипса Ламе ($p = 1.85$) и реальным хордам `safe_mask`.
   - Запрет форм «песочных часов» и «перевернутой пирамиды», жесткая склейка висячих предлогов/союзов/частиц,
     минимизация штрафа аспектной гармонии (`Block Aspect Ratio Penalty`).
   - Раздельная политика переноса слов RU vs EN (умный слоговой перенос для русского от 9/11 букв с сохранением приставок
     и запретом лесенки дефисов; отключение переносов для английского в обычных баблах).
4. Двухпроходная гармонизация кегля на уровне страницы (`Page-Level Font Size Consistency`) и плотный комиксный leading:
   - Проход 1: расчет `page_base_fs` как взвешенной медианы (вес $\\sqrt{\\text{len(text)}}$) по всем диалоговым баблам страницы.
   - Проход 2: ограничение раздувания коротких реплик (1–3 слова, $\\le 14$ симв.) до `1.15 * page_base_fs`,
     детектор мелкого кегля `font_too_small_warning` (`< 72%` от `page_base_fs`), плотный интерлиньяж для UPPERCASE (`8%`)
     с защитой диакритики `Ё`/`Й`.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cleaner import analyze_bubble_geometry

logger = logging.getLogger(__name__)


# Рекомендуемые сканлейт-шрифты для ненавязчивой подсказки в UI при пустой папке fonts/
RECOMMENDED_SCANLATION_FONTS: list[dict[str, str]] = [
    {"name": "CC Wild Words (RU/EN)", "role": "dialogue", "desc": "Золотой стандарт диалогов манги (кириллица + латиница)"},
    {"name": "Anime Ace v05 / v02", "role": "dialogue", "desc": "Классический шрифт сканлейта для баблов речи"},
    {"name": "Manga Temple", "role": "dialogue", "desc": "Мягкий комиксный гротеск для диалогов и нарратива"},
    {"name": "Chinacat", "role": "thought", "desc": "Уютный шрифт для мыслей, дневников и закадрового текста"},
    {"name": "v_BadaboomBB / Komika Axis", "role": "shout", "desc": "Акцидентный шрифт для криков и звуковых эффектов (SFX)"},
]

# Профессиональная матрица шрифтов (RU / EN) по ролям баблов (Блок 1.1 — БЕЗ Comic Sans MS!)
DEFAULT_FONT_PATHS: list[str] = [
    "v_CCWildWordsLower.ttf",
    "v_CCWildWords.ttf",
    "ccwildwords.ttf",
    "wildwords.ttf",
    "AnimeAcev05.ttf",
    "animeace2_reg.ttf",
    "animeace.ttf",
    "MangaTemple.ttf",
    "mangat.ttf",
    "Chinacat.ttf",
    "C:/Windows/Fonts/seguisb.ttf",
    "C:/Windows/Fonts/trebucbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
]

SYSTEM_FONT_PRESETS: dict[str, list[str]] = {
    # Основные роли баблов по стандарту v3.5
    "dialogue": [
        "v_CCWildWordsLower.ttf",
        "v_CCWildWords.ttf",
        "CCWildWords.ttf",
        "ccwildwords.ttf",
        "wildwords.ttf",
        "Anime Ace v05.ttf",
        "AnimeAcev05.ttf",
        "v_AnimeAce2.ttf",
        "animeace2_reg.ttf",
        "animeace.ttf",
        "Manga Temple.ttf",
        "MangaTemple.ttf",
        "mangat.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/trebucbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ],
    "thought": [
        "v_CCWildWordsItalic.ttf",
        "CCWildWords-Italic.ttf",
        "Anime Ace v05 Italic.ttf",
        "AnimeAcev05-Italic.ttf",
        "Chinacat.ttf",
        "chinacat.ttf",
        "Augie.ttf",
        "augie.ttf",
        "Felt Regular.ttf",
        "felt.ttf",
        "C:/Windows/Fonts/segoeprb.ttf",
        "C:/Windows/Fonts/trebucbi.ttf",
        "C:/Windows/Fonts/segoeuii.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "shout": [
        "v_CCWildWordsBoldItalic.ttf",
        "CCWildWords-BoldItalic.ttf",
        "v_BadaboomBB.ttf",
        "BadaboomBB_Reg.ttf",
        "badaboom.ttf",
        "Komika Axis.ttf",
        "KomikaAxis.ttf",
        "v_SequentialistBB.ttf",
        "Manganya.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/trebucbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "whisper": [
        "v_CCWildWords.ttf",
        "CCWildWords.ttf",
        "ccwildwords.ttf",
        "Anime Ace v05.ttf",
        "AnimeAcev05.ttf",
        "animeace2_reg.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/trebuc.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    "narration": [
        "Chinacat.ttf",
        "chinacat.ttf",
        "Manga Temple.ttf",
        "MangaTemple.ttf",
        "mangat.ttf",
        "Gill Sans MT.ttf",
        "gillsans.ttf",
        "C:/Windows/Fonts/georgiab.ttf",
        "C:/Windows/Fonts/cambriab.ttf",
        "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    ],
    "sfx": [
        "v_BadaboomBB.ttf",
        "BadaboomBB_Reg.ttf",
        "TrashHand.ttf",
        "Mistral.ttf",
        "Komika Title.ttf",
        "KomikaTitle.ttf",
        "Brush Script.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    # Идентификаторы семейств для селектора шрифтов в UI (без Comic Sans MS!)
    "WildWords": [
        "v_CCWildWordsLower.ttf",
        "v_CCWildWords.ttf",
        "ccwildwords.ttf",
        "wildwords.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/trebucbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "AnimeAce": [
        "Anime Ace v05.ttf",
        "AnimeAcev05.ttf",
        "v_AnimeAce2.ttf",
        "animeace2_reg.ttf",
        "animeace.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/trebucbd.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
    "MangaTemple": [
        "Manga Temple.ttf",
        "MangaTemple.ttf",
        "mangat.ttf",
        "Chinacat.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/trebucbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "Chinacat": [
        "Chinacat.ttf",
        "chinacat.ttf",
        "Augie.ttf",
        "C:/Windows/Fonts/segoeprb.ttf",
        "C:/Windows/Fonts/trebucbi.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    ],
    "Segoe": [
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/trebucbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "Trebuchet": [
        "C:/Windows/Fonts/trebucbd.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "Impact": [
        "v_BadaboomBB.ttf",
        "Komika Axis.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/trebucbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "Serif": [
        "C:/Windows/Fonts/georgiab.ttf",
        "C:/Windows/Fonts/cambriab.ttf",
        "C:/Windows/Fonts/georgia.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    ],
}

FALLBACK_SYMBOL_FONT_CANDIDATES: list[str] = [
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/seguisb.ttf",
    "C:/Windows/Fonts/trebucbd.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# Ключевые подстроки названий профессиональных комикс-шрифтов с поддержкой глифы Slash-I ('i')
COMIC_CROSSBAR_FONT_KEYWORDS: tuple[str, ...] = (
    "wildwords",
    "wild_words",
    "animeace",
    "anime_ace",
    "mangatemple",
    "manga_temple",
    "mangat",
    "blambot",
    "komika",
    "badaboom",
    "sequentialist",
    "manganya",
    "chinacat",
    "augie",
    "felt",
)

# Символы, которые всегда рисуются 4x-сглаженным векторным рендером (чтобы никогда не превращаться в квадратики □
# и иметь полноценную комиксную обводку stroke_width на любом шрифте)
VECTOR_MANGA_SYMBOLS: set[str] = {"♥", "♡", "★", "☆", "✦", "♪", "♫", "◆", "◇"}

# Невидимые модификаторы вариаций и zero-width символы, которые в Pillow рисуются как пустой квадрат □
INVISIBLE_TOFU_CHARS_RE = re.compile(r"[\ufe00-\ufe0f\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\ufffc\ufffd]")

# Таблица приведения эмодзи, редких сердечек, звездочек, нот и широкой пунктуации к поддерживаемым символам
SPECIAL_SYMBOLS_TRANSLATE_MAP = str.maketrans(
    {
        # Сердечки закрашенные -> ♥
        "❤": "♥",
        "❥": "♥",
        "❣": "♥",
        "❦": "♥",
        "❧": "♥",
        "💕": "♥",
        "💖": "♥",
        "💗": "♥",
        "💓": "♥",
        "💞": "♥",
        "💘": "♥",
        "💝": "♥",
        "💟": "♥",
        "🖤": "♥",
        "💙": "♥",
        "💚": "♥",
        "💛": "♥",
        "💜": "♥",
        "🧡": "♥",
        "🤎": "♥",
        # Сердечки контурные -> ♡
        "🤍": "♡",
        "♤": "♡",
        # Звездочки закрашенные -> ★
        "⭐": "★",
        "🌟": "★",
        "🌠": "★",
        "✪": "★",
        "✫": "★",
        "✬": "★",
        "✭": "★",
        "✮": "★",
        "✯": "★",
        "★": "★",
        # Звездочки контурные -> ☆
        "✩": "☆",
        "✰": "☆",
        "☆": "☆",
        # Искры / блеск -> ✦
        "✨": "✦",
        "✧": "✦",
        "❇": "✦",
        "❈": "✦",
        "✴": "✦",
        "✵": "✦",
        "✶": "✦",
        "✷": "✦",
        "✸": "✦",
        "✹": "✦",
        # Ноты -> ♪ / ♫
        "♩": "♪",
        "🎵": "♪",
        "♬": "♫",
        "🎶": "♫",
        "♭": "♪",
        "♯": "♪",
        # Ромбы / геометрические акценты
        "♦": "◆",
        "♢": "◇",
        "❖": "◆",
        # Тильды, тире и пунктуация, часто дающая квадратики в комиксных шрифтах
        "〜": "~",
        "～": "~",
        "∼": "~",
        "˜": "~",
        "﹏": "~",
        "―": "—",
        "─": "—",
        "━": "—",
        "‒": "-",
        "‐": "-",
        "‑": "-",
        "−": "-",
        "・": "·",
        "∙": "·",
        "•": "·",
        "※": "*",
        "＊": "*",
        "✱": "*",
        "✲": "*",
        "❓": "?",
        "❔": "?",
        "❗": "!",
        "❕": "!",
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
        "【": "[",
        "】": "]",
        "《": '"',
        "》": '"',
        "（": "(",
        "）": ")",
        "！": "!",
        "？": "?",
        "　": " ",
    }
)


def normalize_manga_special_symbols(text: str) -> str:
    """
    Очищает переведенный или редактируемый текст от невидимых селекторов (`\\uFE0F`, вызывающих квадратики `□`)
    и приводит все варианты сердечек, звездочек, нот и спецпунктуации к каноническим символам манги.
    """
    if not text:
        return ""
    s = INVISIBLE_TOFU_CHARS_RE.sub("", str(text))
    s = (
        s.replace("‼", "!!")
        .replace("⁉", "!?")
        .replace("⁈", "?!")
        .replace("…", "...")
        .replace("⋯", "...")
        .replace("‥", "..")
    )
    s = s.translate(SPECIAL_SYMBOLS_TRANSLATE_MAP)
    return s


def _apply_crossbar_i_rule(text: str) -> str:
    """
    Правило Blambot Crossbar-I:
    Оставляет заглавную 'I' (с засечками сверху и снизу) только в личном местоимении I,
    контракциях I'M, I'LL, I'D, I'VE и римских цифрах/инициалах с точками (F.B.I., II, III, IV),
    а внутри всех остальных слов (TiME, FiGHT, WiTH) заменяет на строчную 'i' (Slash-I).
    """
    def _repl(match: re.Match) -> str:
        word = match.group(0)
        if word in {"I", "I'M", "I'LL", "I'D", "I'VE", "I’M", "I’LL", "I’D", "I’VE"}:
            return word
        if re.fullmatch(r"(?:[A-Z]\.)+", word) or re.fullmatch(r"[IVX]+", word):
            return word
        return word.replace("I", "i")

    return re.sub(r"[A-Za-z'’\.]+", _repl, text.upper())


class MangaRenderer:
    """Вписывает переведенный текст в очищенные бабблы и свободные области страницы манги (v3.5 Pro Scanlation Edition)."""

    def __init__(
        self,
        fonts_dir: str | Path = "fonts",
        max_font_size: int = 32,
        min_font_size: int = 12,
        box_padding_ratio: float = 0.10,
        line_spacing: int = 2,
        uppercase_comic_style: bool = True,
        overflow_mode: str = "shrink",
        crossbar_i_mode: bool = True,
    ) -> None:
        self.fonts_dir = Path(fonts_dir)
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.box_padding_ratio = box_padding_ratio
        self.line_spacing = line_spacing
        self.uppercase_comic_style = uppercase_comic_style
        self.overflow_mode = self._normalize_overflow_mode(overflow_mode)
        self.crossbar_i_mode = crossbar_i_mode

        self.font_path = self._resolve_font_path()
        self.fallback_font_paths: list[str] = [
            p for p in FALLBACK_SYMBOL_FONT_CANDIDATES if Path(p).exists()
        ]
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self._notdef_sig_cache: dict[int, set[tuple[tuple[int, int], bytes]]] = {}
        self._glyph_support_cache: dict[tuple[int, str], bool] = {}
        self._symbol_icon_cache: dict[tuple[str, int, tuple[int, int, int], tuple[int, int, int], int], Image.Image] = {}
        logger.info("Используемый шрифт для тайпсеттинга: %s", self.font_path or "Pillow Default")

    def has_custom_scanlation_fonts(self) -> bool:
        """Возвращает True, если в папке `fonts/` установлены пользовательские `.ttf`/`.otf` шрифты."""
        if not self.fonts_dir.exists():
            return False
        custom_fonts = list(self.fonts_dir.glob("*.ttf")) + list(self.fonts_dir.glob("*.otf"))
        return len(custom_fonts) > 0

    def get_font_status_info(self) -> dict[str, Any]:
        """Возвращает диагностику шрифтовой системы и список рекомендуемых сканлейт-шрифтов для UI."""
        has_custom = self.has_custom_scanlation_fonts()
        return {
            "has_custom_fonts": has_custom,
            "active_default_font": self.font_path,
            "fonts_dir": str(self.fonts_dir.resolve()) if self.fonts_dir.exists() else str(self.fonts_dir),
            "recommended_fonts": RECOMMENDED_SCANLATION_FONTS,
            "hint_message": (
                ""
                if has_custom
                else "Папка fonts/ пуста — используется системный гротеск (Segoe UI Semibold / Trebuchet MS). "
                     "Для эталонного сканлейта рекомендуем положить в папку fonts/ шрифты: "
                     "CC Wild Words, Anime Ace v05, Manga Temple, Chinacat."
            ),
        }

    def _resolve_font_path(self) -> str | None:
        """
        Двухуровневый поиск базового шрифта диалогов:
        1) Приоритетный поиск эталонных сканлейт-шрифтов в `fonts/` (`CC Wild Words`, `Anime Ace v05`, `Manga Temple`).
        2) Любой `.ttf`/`.otf` в `fonts/`.
        3) Безопасный системный гротеск (`Segoe UI Semibold`, `Trebuchet MS`, `DejaVuSans-Bold`) — НИКОГДА не `Comic Sans MS`!
        """
        if self.fonts_dir.exists():
            custom_fonts = sorted(
                list(self.fonts_dir.glob("*.ttf")) + list(self.fonts_dir.glob("*.otf"))
            )
            if custom_fonts:
                for pref_name in ("wildwords", "animeace", "anime_ace", "mangatemple", "manga_temple", "chinacat"):
                    for f_path in custom_fonts:
                        if pref_name in f_path.stem.lower().replace(" ", "_"):
                            return str(f_path)
                return str(custom_fonts[0])

        for candidate in DEFAULT_FONT_PATHS:
            local_cand = self.fonts_dir / candidate
            if local_cand.exists() and local_cand.is_file():
                return str(local_cand)
            if Path(candidate).exists() and Path(candidate).is_file():
                return candidate

        return None

    def list_available_fonts(self) -> list[dict[str, str]]:
        """Возвращает список доступных шрифтовых пресетов и пользовательских `.ttf`/`.otf` файлов из папки `fonts/` (без Comic Sans)."""
        items: list[dict[str, str]] = [
            {"id": "default", "name": "✨ По умолчанию (Авто по роли бабла)"},
            {"id": "WildWords", "name": "🌸 CC Wild Words / Комикс (Диалог)"},
            {"id": "AnimeAce", "name": "🎌 Anime Ace v05 / Сканлейт"},
            {"id": "MangaTemple", "name": "🏮 Manga Temple (Диалог / Нарратив)"},
            {"id": "Chinacat", "name": "💭 Chinacat / Segoe Print (Мысли)"},
            {"id": "Segoe", "name": "📘 Segoe UI Semibold / Trebuchet MS (Чистый гротеск)"},
            {"id": "Impact", "name": "💥 Badaboom BB / Impact (Крик / SFX)"},
            {"id": "Serif", "name": "📖 Georgia Bold / Cambria (Книжный нарратив)"},
        ]
        seen_ids = {x["id"].lower() for x in items}
        if self.fonts_dir.exists():
            custom_files = sorted(
                list(self.fonts_dir.glob("*.ttf")) + list(self.fonts_dir.glob("*.otf"))
            )
            for f_path in custom_files:
                fid = f_path.name
                if fid.lower() in seen_ids:
                    continue
                seen_ids.add(fid.lower())
                items.append({"id": fid, "name": f"📁 {f_path.stem} ({f_path.suffix.lower()})"})
        return items

    def _resolve_font_family_path(
        self,
        font_family: str | None = None,
        style_preset: str = "dialogue",
    ) -> str | None:
        fam = (font_family or "").strip()
        preset = (style_preset or "dialogue").lower().strip()
        if preset not in ("dialogue", "thought", "shout", "whisper", "narration", "sfx"):
            preset = "dialogue"

        # Защита от устаревших ссылок на Comic Sans в сохраненных проектах
        if fam.lower() in ("comic", "comicbd.ttf", "comic.ttf", "comic sans", "comic sans ms"):
            fam = "Segoe"

        if fam and fam.lower() not in ("default", "auto", "none"):
            local_cand = self.fonts_dir / fam
            if local_cand.exists() and local_cand.is_file():
                return str(local_cand)
            if self.fonts_dir.exists():
                for f_p in list(self.fonts_dir.glob("*.ttf")) + list(self.fonts_dir.glob("*.otf")):
                    if f_p.name.lower() == fam.lower() or f_p.stem.lower() == fam.lower():
                        return str(f_p)

            for preset_key, cand_list in SYSTEM_FONT_PRESETS.items():
                if preset_key.lower() == fam.lower():
                    for cand in cand_list:
                        local_p = self.fonts_dir / cand
                        if local_p.exists() and local_p.is_file():
                            return str(local_p)
                        if Path(cand).exists() and Path(cand).is_file():
                            return cand

            if Path(fam).exists() and Path(fam).is_file():
                return str(Path(fam))

        # Авто-подбор шрифта по матрице style_preset (Блок 1.1)
        preset_candidates = SYSTEM_FONT_PRESETS.get(preset, SYSTEM_FONT_PRESETS["dialogue"])
        for cand in preset_candidates:
            local_p = self.fonts_dir / cand
            if local_p.exists() and local_p.is_file():
                return str(local_p)
            if self.fonts_dir.exists():
                cand_stem = Path(cand).stem.lower().replace(" ", "").replace("_", "").replace("-", "")
                for f_p in list(self.fonts_dir.glob("*.ttf")) + list(self.fonts_dir.glob("*.otf")):
                    f_norm = f_p.stem.lower().replace(" ", "").replace("_", "").replace("-", "")
                    if cand_stem and (f_norm == cand_stem or cand_stem in f_norm):
                        return str(f_p)
            if Path(cand).exists() and Path(cand).is_file():
                return cand

        return self.font_path

    def _format_comic_typography(
        self,
        text: str,
        font_path: str | None = None,
        use_upper: bool = True,
    ) -> str:
        """
        Блок 1.2: Фильтр комиксной пунктуации и правило Crossbar-I (по стандарту Blambot):
        1. Нормализует многоточия (`…` -> `...`) и тире (` - ` / `—` -> `--` для EN, неразрывное `—` для RU).
        2. При `use_upper=True` применяет правило Crossbar-I для английского текста (`I` только в `I`, `I'M`, `I'LL`, `I'D`, `I'VE`,
           инициалах с точками и римских цифрах; внутри остальных слов — строчная `i`).
        """
        if not text:
            return ""
        s = normalize_manga_special_symbols(text)
        # Нормализация многоточий в три отдельные точки с нормальным трекингом
        s = re.sub(r"\.{3,}", "...", s)

        has_cyrillic = bool(re.search(r"[А-Яа-яЁё]", s))
        has_latin = bool(re.search(r"[A-Za-z]", s))

        # Обработка тире по языковой традиции комиксов (Blambot для EN vs русская типографика)
        if has_latin and not has_cyrillic:
            s = re.sub(r"\s+(?:—|–|-)\s+", " -- ", s)
            s = s.replace("—", "--")
        elif has_cyrillic:
            s = re.sub(r"\s+-\s+", " — ", s)

        if not use_upper:
            return s

        # Применяем правило Crossbar-I для латиницы в режиме UPPERCASE
        if has_latin and self.crossbar_i_mode:
            return _apply_crossbar_i_rule(s)

        return s.upper()

    @staticmethod
    def _apply_crossbar_i_rule(text: str) -> str:
        """Статический доступ к правилу Blambot Crossbar-I."""
        return _apply_crossbar_i_rule(text)

    def _get_font(
        self,
        size: int,
        font_family_path: str | None = None,
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        size = max(6, int(size))
        eff_path = font_family_path if font_family_path is not None else (self.font_path or "")
        cache_key = (str(eff_path), size)
        if cache_key not in self._font_cache:
            if eff_path:
                try:
                    self._font_cache[cache_key] = ImageFont.truetype(eff_path, size=size)
                except OSError:
                    if self.font_path and self.font_path != eff_path:
                        try:
                            self._font_cache[cache_key] = ImageFont.truetype(self.font_path, size=size)
                        except OSError:
                            self._font_cache[cache_key] = ImageFont.load_default()
                    else:
                        self._font_cache[cache_key] = ImageFont.load_default()
            else:
                self._font_cache[cache_key] = ImageFont.load_default()
        return self._font_cache[cache_key]

    def _is_glyph_missing(
        self,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        ch: str,
    ) -> bool:
        if not ch or ch.isspace():
            return False
        if ch in VECTOR_MANGA_SYMBOLS:
            return True

        font_id = id(font)
        cache_key = (font_id, ch)
        if cache_key in self._glyph_support_cache:
            return not self._glyph_support_cache[cache_key]

        if font_id not in self._notdef_sig_cache:
            sigs: set[tuple[tuple[int, int], bytes]] = set()
            for probe in ("\ufffd", "\ufffe", "\uffff", "\U0010ffff"):
                try:
                    m = font.getmask(probe)
                    b = bytes(m)
                    if b:
                        sigs.add((m.size, b))
                except Exception:
                    pass
            self._notdef_sig_cache[font_id] = sigs

        notdef_sigs = self._notdef_sig_cache[font_id]
        try:
            mask = font.getmask(ch)
            raw = bytes(mask)
            if not raw:
                supported = False
            elif (mask.size, raw) in notdef_sigs:
                supported = False
            else:
                supported = True
        except Exception:
            supported = False

        self._glyph_support_cache[cache_key] = supported
        return not supported

    def _get_fallback_font_for_char(
        self,
        ch: str,
        size: int,
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont | None:
        for fb_path in self.fallback_font_paths:
            fb_font = self._get_font(size, font_family_path=fb_path)
            if not self._is_glyph_missing(fb_font, ch):
                return fb_font
        return None

    def _needs_special_rendering(
        self,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        text: str,
    ) -> bool:
        for ch in text:
            if ch == "\n" or ch.isspace():
                continue
            if ch in VECTOR_MANGA_SYMBOLS or self._is_glyph_missing(font, ch):
                return True
        return False

    def _get_vector_symbol_icon(
        self,
        sym: str,
        font_size: int,
        text_fill: tuple[int, int, int],
        stroke_fill: tuple[int, int, int],
        stroke_w: int,
    ) -> Image.Image:
        fs = max(8, int(font_size))
        sw_clamped = max(0, min(16, int(stroke_w)))
        cache_key = (sym, fs, text_fill, stroke_fill, sw_clamped)
        cached = self._symbol_icon_cache.get(cache_key)
        if cached is not None:
            return cached

        ss = 4  # 4x supersampling
        box = max(10, int(round(fs * 0.92)))
        pad = max(3, sw_clamped + 2)
        w = (box + pad * 2) * ss
        h = (box + pad * 2) * ss
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        cx = w / 2.0
        cy = h / 2.0
        r = (box * ss) * 0.44
        sw = max(0, sw_clamped * ss)
        t_col = (*text_fill, 255)
        s_col = (*stroke_fill, 255)
        hollow_ring_w = max(2 * ss, int(round(box * ss * 0.14)))

        def _draw_poly(pts: list[tuple[float, float]], fill_c: tuple[int, int, int, int] | None, out_c: tuple[int, int, int, int], out_w: int, inner_w: int = 0) -> None:
            if out_w > 0:
                total_w = out_w * 2 + inner_w
                for idx in range(len(pts)):
                    p1 = pts[idx]
                    p2 = pts[(idx + 1) % len(pts)]
                    draw.line([p1, p2], fill=out_c, width=total_w)
                    rad = total_w // 2
                    draw.ellipse([p1[0] - rad, p1[1] - rad, p1[0] + rad, p1[1] + rad], fill=out_c)
            if fill_c is not None:
                draw.polygon(pts, fill=fill_c)
            if inner_w > 0:
                for idx in range(len(pts)):
                    p1 = pts[idx]
                    p2 = pts[(idx + 1) % len(pts)]
                    draw.line([p1, p2], fill=t_col, width=inner_w)
                    rad = inner_w // 2
                    draw.ellipse([p1[0] - rad, p1[1] - rad, p1[0] + rad, p1[1] + rad], fill=t_col)

        if sym in ("♥", "♡"):
            pts: list[tuple[float, float]] = []
            steps = 56
            for i in range(steps):
                t = (2.0 * math.pi * i) / steps
                x = 16.0 * (math.sin(t) ** 3)
                y = -(13.0 * math.cos(t) - 5.0 * math.cos(2 * t) - 2.0 * math.cos(3 * t) - math.cos(4 * t))
                px = cx + (x / 16.5) * r
                py = cy + (y / 16.5) * r + (r * 0.06)
                pts.append((px, py))
            if sym == "♥":
                _draw_poly(pts, fill_c=t_col, out_c=s_col, out_w=sw, inner_w=max(2, ss))
            else:
                _draw_poly(pts, fill_c=s_col if sw > 0 else (255, 255, 255, 255), out_c=s_col, out_w=sw, inner_w=hollow_ring_w)

        elif sym in ("★", "☆"):
            pts = []
            inner_r = r * 0.42
            for i in range(10):
                ang = -math.pi / 2.0 + i * (math.pi / 5.0)
                curr_r = r if i % 2 == 0 else inner_r
                pts.append((cx + curr_r * math.cos(ang), cy + curr_r * math.sin(ang) + r * 0.04))
            if sym == "★":
                _draw_poly(pts, fill_c=t_col, out_c=s_col, out_w=sw, inner_w=max(2, ss))
            else:
                _draw_poly(pts, fill_c=s_col if sw > 0 else (255, 255, 255, 255), out_c=s_col, out_w=sw, inner_w=hollow_ring_w)

        elif sym == "✦":
            pts = []
            inner_r = r * 0.28
            for i in range(8):
                ang = -math.pi / 2.0 + i * (math.pi / 4.0)
                curr_r = r * 1.04 if i % 2 == 0 else inner_r
                pts.append((cx + curr_r * math.cos(ang), cy + curr_r * math.sin(ang)))
            _draw_poly(pts, fill_c=t_col, out_c=s_col, out_w=sw, inner_w=max(2, ss))

        elif sym in ("◆", "◇"):
            pts = [
                (cx, cy - r * 0.96),
                (cx + r * 0.76, cy),
                (cx, cy + r * 0.96),
                (cx - r * 0.76, cy),
            ]
            if sym == "◆":
                _draw_poly(pts, fill_c=t_col, out_c=s_col, out_w=sw, inner_w=max(2, ss))
            else:
                _draw_poly(pts, fill_c=s_col if sw > 0 else (255, 255, 255, 255), out_c=s_col, out_w=sw, inner_w=hollow_ring_w)

        elif sym == "♪":
            stem_w = max(2 * ss, int(round(r * 0.24)))
            head_rx = r * 0.42
            head_ry = r * 0.30
            hx = cx - r * 0.16
            hy = cy + r * 0.46
            sx = hx + head_rx * 0.74
            sy_top = cy - r * 0.76
            flag_end = (sx + r * 0.54, sy_top + r * 0.46)
            if sw > 0:
                draw.ellipse([hx - head_rx - sw, hy - head_ry - sw, hx + head_rx + sw, hy + head_ry + sw], fill=s_col)
                draw.line([(sx, hy), (sx, sy_top)], fill=s_col, width=stem_w + sw * 2)
                draw.line([(sx, sy_top), flag_end], fill=s_col, width=stem_w + sw * 2)
            draw.ellipse([hx - head_rx, hy - head_ry, hx + head_rx, hy + head_ry], fill=t_col)
            draw.line([(sx, hy), (sx, sy_top)], fill=t_col, width=stem_w)
            draw.line([(sx, sy_top), flag_end], fill=t_col, width=stem_w)

        elif sym == "♫":
            stem_w = max(2 * ss, int(round(r * 0.20)))
            head_rx = r * 0.32
            head_ry = r * 0.24
            h1x, h1y = cx - r * 0.40, cy + r * 0.50
            h2x, h2y = cx + r * 0.22, cy + r * 0.34
            s1x, s1_top = h1x + head_rx * 0.72, cy - r * 0.60
            s2x, s2_top = h2x + head_rx * 0.72, cy - r * 0.76
            if sw > 0:
                for hx, hy, sx, st in ((h1x, h1y, s1x, s1_top), (h2x, h2y, s2x, s2_top)):
                    draw.ellipse([hx - head_rx - sw, hy - head_ry - sw, hx + head_rx + sw, hy + head_ry + sw], fill=s_col)
                    draw.line([(sx, hy), (sx, st)], fill=s_col, width=stem_w + sw * 2)
                draw.line([(s1x, s1_top), (s2x, s2_top)], fill=s_col, width=stem_w + sw * 2 + ss)
            for hx, hy, sx, st in ((h1x, h1y, s1x, s1_top), (h2x, h2y, s2x, s2_top)):
                draw.ellipse([hx - head_rx, hy - head_ry, hx + head_rx, hy + head_ry], fill=t_col)
                draw.line([(sx, hy), (sx, st)], fill=t_col, width=stem_w)
            draw.line([(s1x, s1_top), (s2x, s2_top)], fill=t_col, width=stem_w + ss)

        target_size = (box + pad * 2, box + pad * 2)
        icon = img.resize(target_size, resample=Image.LANCZOS)
        if len(self._symbol_icon_cache) > 512:
            self._symbol_icon_cache.clear()
        self._symbol_icon_cache[cache_key] = icon
        return icon

    @staticmethod
    def _get_font_pixel_size(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
        return int(getattr(font, "size", 18) or 18)

    def _measure_line_width(
        self,
        draw: ImageDraw.ImageDraw,
        line: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> float:
        if not line:
            return 0.0
        if not self._needs_special_rendering(font, line):
            return float(draw.textlength(line, font=font))

        fs = self._get_font_pixel_size(font)
        sym_advance = float(max(8, int(round(fs * 0.88))))
        total_w = 0.0
        buf = ""
        for ch in line:
            if ch in VECTOR_MANGA_SYMBOLS:
                if buf:
                    total_w += float(draw.textlength(buf, font=font))
                    buf = ""
                total_w += sym_advance
            elif self._is_glyph_missing(font, ch):
                if buf:
                    total_w += float(draw.textlength(buf, font=font))
                    buf = ""
                fb_font = self._get_fallback_font_for_char(ch, fs)
                if fb_font is not None:
                    total_w += float(draw.textlength(ch, font=fb_font))
            else:
                buf += ch
        if buf:
            total_w += float(draw.textlength(buf, font=font))
        return total_w

    _LINE_METRIC_CACHE: dict[tuple[int, int, bool, bool], tuple[int, int, int, int]] = {}

    @classmethod
    def _get_font_line_metrics(
        cls,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None,
        font_size: int,
        use_upper: bool = True,
        has_tall_cyr: bool = True,
    ) -> tuple[int, int, int, int]:
        """
        Возвращает `(bA_y1, ref_y0, ref_y1, metric_deficit)` для точной компенсации
        внутренней формулы Pillow `_multiline_spacing = textbbox("A")[3] + 2*stroke_w + spacing`.
        В кириллических и комиксных шрифтах (`comicbd.ttf`, `DejaVuSans`, `WildWords`, `AnimeAce`)
        полный размах букв с диакритикой (`Ё`, `Й`) и нижними выносными элементами (`Д`, `Щ`, `Ц`, `р`, `у`)
        превышает `textbbox("A")[3]` на 3..8 пикселей (`metric_deficit`), из-за чего без компенсации
        соседние строки налезают друг на друга.
        """
        fs = max(6, int(font_size))
        if font is None:
            est_def = max(2, int(round(fs * (0.18 if not use_upper else (0.14 if has_tall_cyr else 0.10)))))
            return fs, 0, fs + est_def, est_def

        cache_key = (id(font), fs, bool(use_upper), bool(has_tall_cyr))
        cached = cls._LINE_METRIC_CACHE.get(cache_key)
        if cached is not None:
            return cached

        dummy_img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
        dummy_draw = ImageDraw.Draw(dummy_img)
        b_a = dummy_draw.textbbox((0, 0), "A", font=font, stroke_width=0)
        ba_y1 = int(b_a[3])

        if use_upper:
            ref_sample = "АБВДЁЙЦЩABCDEFGHIJKLMNOPQRSTUVWXYZ?!" if has_tall_cyr else "АБВДЦЩABCDEFGHIJKLMNOPQRSTUVWXYZ?!"
        else:
            ref_sample = "АБВДЁЙЦЩабвгдеёжзийклмнопрстуфхцчшщъыьэюяABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz?!"

        b_ref = dummy_draw.textbbox((0, 0), ref_sample, font=font, stroke_width=0)
        ref_y0 = int(b_ref[1])
        ref_y1 = int(b_ref[3])
        ref_h = max(fs, ref_y1 - ref_y0)
        metric_deficit = max(0, ref_h - ba_y1)

        res = (ba_y1, ref_y0, ref_y1, metric_deficit)
        if len(cls._LINE_METRIC_CACHE) > 1024:
            cls._LINE_METRIC_CACHE.clear()
        cls._LINE_METRIC_CACHE[cache_key] = res
        return res

    @staticmethod
    def _draw_multiline_text_clean(
        draw_obj: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        fill: tuple[int, ...],
        spacing: int,
        align: str = "center",
        stroke_width: int = 0,
        stroke_fill: tuple[int, ...] | None = None,
    ) -> None:
        """
        Двухпроходная отрисовка многострочного текста с обводкой (по стандарту Photoshop / Clip Studio):
        1. Сначала рисует единый слой обводки (`stroke_fill`) для ВСЕХ строк сразу.
        2. Затем поверх рисует чистую заливку букв (`fill`, `stroke_width=0`) для всех строк.
        Это гарантирует, что белая обводка нижней строки НИКОГДА не закрасит и не «откусит»
        нижние штрихи букв (`Д`, `Щ`, `Ц`, `р`, `у`) верхней строки!
        """
        if stroke_width > 0 and stroke_fill is not None:
            draw_obj.multiline_text(
                xy,
                text,
                font=font,
                fill=stroke_fill,
                spacing=spacing,
                align=align,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
        draw_obj.multiline_text(
            xy,
            text,
            font=font,
            fill=fill,
            spacing=spacing,
            align=align,
            stroke_width=0,
        )

    def _render_rich_text_layer(
        self,
        wrapped_text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        spacing: int,
        align: str,
        text_fill: tuple[int, int, int],
        stroke_fill: tuple[int, int, int],
        stroke_w: int,
    ) -> Image.Image:
        fs = self._get_font_pixel_size(font)
        dummy_img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        dummy_draw = ImageDraw.Draw(dummy_img)

        lines = wrapped_text.split("\n")
        line_widths = [self._measure_line_width(dummy_draw, ln, font) for ln in lines]
        max_line_w = max(line_widths, default=float(fs))

        has_tall_cyr = any(ch in wrapped_text for ch in ("Ё", "Й", "ё", "й"))
        ba_y1, ref_y0, ref_y1, _ = self._get_font_line_metrics(
            font=font,
            font_size=fs,
            use_upper=False,
            has_tall_cyr=has_tall_cyr,
        )
        glyph_h = max(fs, ref_y1 - ref_y0)
        # Шаг между строками в точности равен шагу Pillow _multiline_spacing, но не меньше полной высоты глифов + обводки + 2px
        step_y = max(ba_y1 + 2 * stroke_w + spacing, glyph_h + 2 * stroke_w + max(2, int(round(fs * 0.12))))
        total_h = max(glyph_h + 2 * stroke_w, (len(lines) - 1) * step_y + glyph_h + 2 * stroke_w)

        pad = max(14, stroke_w * 2 + 10)
        layer_w = max(16, int(math.ceil(max_line_w + pad * 2)))
        layer_h = max(16, int(math.ceil(total_h + pad * 2)))
        layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
        l_draw = ImageDraw.Draw(layer)

        sym_advance = float(max(8, int(round(fs * 0.88))))

        # Проход 1: рисуем обводку (stroke) всех текстовых сегментов всех строк, чтобы обводка нижней строки не перекрывала буквы верхней
        # Проход 2: рисуем чистую заливку (fill) и векторные иконки поверх обводки
        for pass_idx in (0, 1):
            if pass_idx == 0 and stroke_w <= 0:
                continue
            for idx, (ln, lw) in enumerate(zip(lines, line_widths)):
                if not ln:
                    continue
                if align == "left":
                    cur_x = float(pad)
                elif align == "right":
                    cur_x = float(pad) + (max_line_w - lw)
                else:
                    cur_x = float(pad) + (max_line_w - lw) / 2.0

                # Единая базовая координата строки без прыгающих сдвигов по tb[1]
                cur_y = float(pad - min(0, ref_y0) + idx * step_y)
                run_buf = ""

                def _flush_run(x_pos: float, text_run: str) -> float:
                    if not text_run:
                        return x_pos
                    if pass_idx == 0:
                        l_draw.text(
                            (x_pos, cur_y),
                            text_run,
                            font=font,
                            fill=(*stroke_fill, 255),
                            stroke_width=stroke_w,
                            stroke_fill=(*stroke_fill, 255),
                        )
                    else:
                        l_draw.text(
                            (x_pos, cur_y),
                            text_run,
                            font=font,
                            fill=(*text_fill, 255),
                            stroke_width=0,
                        )
                    return x_pos + float(l_draw.textlength(text_run, font=font))

                for ch in ln:
                    if ch in VECTOR_MANGA_SYMBOLS:
                        cur_x = _flush_run(cur_x, run_buf)
                        run_buf = ""
                        if pass_idx == 1:
                            icon = self._get_vector_symbol_icon(ch, fs, text_fill, stroke_fill, stroke_w)
                            iw, ih = icon.size
                            ix = int(round(cur_x + (sym_advance - iw) / 2.0))
                            iy = int(round(cur_y + ref_y0 + (glyph_h - ih) / 2.0))
                            layer.alpha_composite(icon, dest=(max(0, ix), max(0, iy)))
                            l_draw = ImageDraw.Draw(layer)
                        cur_x += sym_advance
                    elif self._is_glyph_missing(font, ch):
                        cur_x = _flush_run(cur_x, run_buf)
                        run_buf = ""
                        fb_font = self._get_fallback_font_for_char(ch, fs)
                        if fb_font is not None:
                            if pass_idx == 0:
                                l_draw.text(
                                    (cur_x, cur_y),
                                    ch,
                                    font=fb_font,
                                    fill=(*stroke_fill, 255),
                                    stroke_width=stroke_w,
                                    stroke_fill=(*stroke_fill, 255),
                                )
                            else:
                                l_draw.text(
                                    (cur_x, cur_y),
                                    ch,
                                    font=fb_font,
                                    fill=(*text_fill, 255),
                                    stroke_width=0,
                                )
                            cur_x += float(l_draw.textlength(ch, font=fb_font))
                    else:
                        run_buf += ch

                cur_x = _flush_run(cur_x, run_buf)

        alpha_bbox = layer.getbbox()
        if alpha_bbox is not None:
            return layer.crop(alpha_bbox)
        return layer

    @classmethod
    def _get_line_spacing(
        cls,
        font_size: int,
        style_preset: str = "dialogue",
        use_upper: bool = True,
        text: str = "",
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
    ) -> int:
        """
        Блок 4.2: Профессиональный Leading (межстрочный интервал) с точной компенсацией метрик шрифта:
        - Компенсирует разницу (`metric_deficit`) между `textbbox("A")[3]` в Pillow и реальной высотой
          кириллических/латинских глифов с диакритикой (`Ё`, `Й`) и выносными элементами (`Д`, `Щ`, `Ц`, `р`, `у`),
          чтобы строки НИКОГДА не наезжали друг на друга даже в `comicbd.ttf` или кастомных `.ttf`.
        - Поверх `metric_deficit` добавляет чистый оптический зазор (`visual_gap`):
          для UPPERCASE диалогов `0.13 * font_size` (`shout` — `0.10 * font_size`, `thought` — `0.15 * font_size`),
          для смешанного регистра `0.18..0.20 * font_size`.
        """
        preset = (style_preset or "dialogue").lower().strip()
        has_tall_cyr = bool(text and any(ch in text for ch in ("Ё", "Й", "ё", "й")))
        _, _, _, metric_deficit = cls._get_font_line_metrics(
            font=font,
            font_size=font_size,
            use_upper=use_upper,
            has_tall_cyr=has_tall_cyr,
        )

        if use_upper:
            if preset == "shout":
                visual_gap = max(2, int(round(font_size * 0.10)))
            elif preset == "thought":
                visual_gap = max(3, int(round(font_size * 0.15)))
            else:
                visual_gap = max(2, int(round(font_size * 0.13)))
        else:
            if preset == "thought":
                visual_gap = max(3, int(round(font_size * 0.20)))
            else:
                visual_gap = max(3, int(round(font_size * 0.18)))

        if has_tall_cyr:
            visual_gap += max(1, int(round(font_size * 0.04)))

        return int(metric_deficit + visual_gap)

    @staticmethod
    def _parse_hex_color(color_str: str | None) -> tuple[int, int, int] | None:
        if not color_str:
            return None
        s = str(color_str).strip().lower()
        if not s or s in ("auto", "default", "none"):
            return None
        if s.startswith("#"):
            s = s[1:]
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) == 6:
            try:
                return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
            except ValueError:
                return None
        return None

    @staticmethod
    def _normalize_overflow_mode(raw_mode: Any) -> str:
        s = str(raw_mode or "shrink").strip().lower()
        if s in ("overflow", "out", "outside", "variant_b", "b"):
            return "overflow"
        if s in ("expand_bubble", "redraw_bubble", "expand", "redraw", "variant_c", "c", "v"):
            return "expand_bubble"
        return "shrink"

    @staticmethod
    def _normalize_bg_style(raw_style: Any) -> str:
        s = str(raw_style or "none").strip().lower()
        if s in ("text_box", "pill", "box", "text", "lines"):
            return "text_box"
        if s in ("bubble", "ellipse", "oval", "capsule"):
            return "bubble"
        if s in ("full_box", "rect", "frame", "full"):
            return "full_box"
        return "none"

    # Гласные и спецбуквы для правильного слогового и морфемного переноса (Блок 3.3: RU vs EN)
    _VOWELS_RU = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
    _VOWELS_EN = set("aeiouyAEIOUY")
    _VOWELS_ALL = _VOWELS_RU | _VOWELS_EN
    _NO_START_NEXT_RU = set("йъьыЙЪЬЫ")
    _MODIFIERS_RU = set("ъьЪЬ")

    # Русские приставки для приоритетного морфемного слогораздела (Грамота.ру)
    _RU_PREFIXES = (
        "пре", "при", "под", "над", "пред", "про", "раз", "рас", "роз", "рос",
        "без", "бес", "воз", "вос", "вз", "вс", "из", "ис", "низ", "нис", "через", "чрез",
        "меж", "сверх", "контр", "супер", "гипер", "интер", "транс", "анти", "псевдо",
        "пере", "недо", "подо", "надо", "обо", "ото", "разо",
    )

    # Английские неделимые диграфы и морфемы
    _EN_UNBREAKABLE_DIGRAPHS = {
        "aa", "ae", "ai", "ao", "au", "ay", "ea", "ee", "ei", "eo", "eu", "ey",
        "ia", "ie", "io", "iu", "oa", "oe", "oi", "oo", "ou", "oy", "ua", "ue", "ui", "uo",
        "sh", "ch", "th", "ph", "wh", "gh", "ck", "ng", "qu", "gu",
    }
    _EN_PREFIXES = ("under", "inter", "trans", "super", "hyper", "over", "anti", "auto", "semi", "non", "dis", "mis", "pre", "pro", "sub", "out", "un")
    _EN_SUFFIXES = ("ingly", "ement", "ation", "ition", "ously", "ively", "fully", "lessly", "ness", "ment", "tion", "sion", "able", "ible", "ably", "ibly", "less", "ship", "ward", "wise", "ing", "ful", "ous", "ive")

    # Однобуквенные и двухбуквенные предлоги, союзы, артикли и местоимения (RU + EN), которые жестко привязываются к следующему слову (Блок 3.2 п.4)
    _SHORT_WORDS_BIND_NEXT = {
        "в", "к", "с", "у", "о", "а", "и",
        "не", "ни", "на", "но", "по", "за", "из", "от", "до", "об", "ко", "со", "во", "про", "без", "для", "под", "над", "при",
        "i", "a", "an", "the", "to", "in", "of", "on", "at", "or", "by",
    }
    _HANGING_SHORT_WORDS = {
        "в", "к", "с", "у", "о", "а", "и", "я",
        "не", "ни", "на", "но", "по", "за", "из", "от", "до", "ко", "со", "во", "об", "про", "без", "для", "под", "над", "при",
        "i", "a", "an", "the", "to", "in", "on", "of", "at", "by", "or", "so", "no", "my", "we", "he", "it", "is", "be", "do", "as", "if", "up", "us", "am", "for", "and", "but", "not", "his", "her", "our", "its", "who",
    }
    # Частицы и знаки, которые жестко склеиваются с предыдущим словом (Блок 3.2 п.4)
    _ENCLITIC_PARTICLES = {"бы", "б", "ли", "ль", "же", "ж", "то", "ка", "—", "--", "-", "–", "!", "?", "?!", "!?", "...", "…", "~", "♥", "♡", "★", "☆", "♪", "♫"}

    @classmethod
    def _find_syllable_split_indices(
        cls,
        core_word: str,
        min_part_len: int = 3,
        is_narrow_bubble: bool = False,
    ) -> list[int]:
        """
        Блок 3.3: Раздельная политика переноса слов (RU vs EN):
        - Для русского языка (RU): порог длины слова от 9 букв (в узких баблах `safe_w < 140px`)
          и от 11 букв (в обычных баблах), минимум 3 буквы до и после дефиса, соблюдение правил Грамота.ру
          (запрет отрыва одной гласной, запрет переноса `й`, `ъ`, `ь`, `ы`, разделение удвоенных согласных посередине,
          сохранение русских приставок `пре-`, `под-`, `раз-`, `бес-`).
        - Для английского языка (EN): перенос только для слов >= 11 букв строго по морфемным границам
          (`un-`, `under-`, `inter-`, `-tion`, `-ment`, `-ing`, `-able`).
        """
        if "'" in core_word or "’" in core_word:
            return []

        n = len(core_word)
        low_word = core_word.lower()
        is_cyrillic = bool(re.search(r"[а-яА-ЯёЁ]", core_word))
        is_latin = bool(re.search(r"[a-zA-Z]", core_word)) and not is_cyrillic

        if is_latin:
            min_word_len = 11
            eff_min_part = max(3, min_part_len)
        else:
            min_word_len = 9 if is_narrow_bubble else 11
            eff_min_part = max(3, min_part_len)

        if n < max(min_word_len, eff_min_part * 2):
            return []

        if is_latin:
            morpheme_splits: list[int] = []
            for pfx in cls._EN_PREFIXES:
                if low_word.startswith(pfx) and eff_min_part <= len(pfx) <= n - eff_min_part:
                    idx_p = len(pfx)
                    if any(ch in cls._VOWELS_EN for ch in core_word[idx_p:]):
                        morpheme_splits.append(idx_p)
            for sfx in cls._EN_SUFFIXES:
                idx_s = n - len(sfx)
                if low_word.endswith(sfx) and eff_min_part <= idx_s <= n - eff_min_part:
                    if any(ch in cls._VOWELS_EN for ch in core_word[:idx_s]):
                        morpheme_splits.append(idx_s)
            return sorted(set(morpheme_splits))

        # Русский язык: морфемные приставки + фонетический слогораздел Liang-Knuth / Грамота.ру
        ru_prefix_indices: set[int] = set()
        for pfx in cls._RU_PREFIXES:
            if low_word.startswith(pfx) and eff_min_part <= len(pfx) <= n - eff_min_part:
                nxt_ch = low_word[len(pfx)]
                if nxt_ch not in cls._NO_START_NEXT_RU:
                    ru_prefix_indices.add(len(pfx))

        valid_splits: list[int] = []
        for i in range(eff_min_part, n - eff_min_part + 1):
            left = core_word[:i]
            right = core_word[i:]
            left_low = low_word[:i]
            right_low = low_word[i:]

            if not any(ch in cls._VOWELS_ALL for ch in left):
                continue
            if not any(ch in cls._VOWELS_ALL for ch in right):
                continue

            if right[0] in cls._NO_START_NEXT_RU:
                continue

            if i in ru_prefix_indices:
                valid_splits.append(i)
                continue

            # Удвоенные согласные делятся строго посередине (клас-сный, груп-па)
            if len(right) >= 2 and right_low[0] == right_low[1] and right[0] not in cls._VOWELS_ALL:
                continue
            if len(left) >= 2 and left_low[-1] == left_low[-2] and left[-1] not in cls._VOWELS_ALL:
                continue

            prev_ch = left[-1]
            next_ch = right[0]
            is_prev_vowel = prev_ch in cls._VOWELS_ALL
            is_next_vowel = next_ch in cls._VOWELS_ALL

            # Не рвем зияния гласных
            if is_prev_vowel and is_next_vowel:
                continue

            # Не отрываем согласную от следующей гласной (кроме разреза после ъ/ь или на границе приставки)
            if not is_prev_vowel and is_next_vowel and prev_ch not in cls._MODIFIERS_RU:
                continue

            valid_splits.append(i)

        return valid_splits

    def _split_long_word(
        self,
        self_draw: ImageDraw.ImageDraw,
        word: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
        safe_w: int = 160,
        safe_h: int = 160,
    ) -> list[str]:
        """
        Блок 3.3: Слоговой/морфемный перенос длинных слов с раздельной политикой RU vs EN.
        """
        draw = self_draw
        if self._measure_line_width(draw, word, font) <= max_width * 0.52:
            return [word]

        if "-" in word and not word.startswith("-") and not word.endswith("-") and "--" not in word:
            hyph_parts = word.split("-")
            if len(hyph_parts) == 2 and len(hyph_parts[0]) >= 3 and len(hyph_parts[1]) >= 3:
                p1 = hyph_parts[0] + "-"
                p2 = hyph_parts[1]
                if self._measure_line_width(draw, p1, font) <= max_width * 1.18:
                    return [p1, p2]

        if "'" in word or "’" in word:
            return [word]

        m = re.match(r"^([^A-Za-zА-Яа-яЁё0-9]*)([A-Za-zА-Яа-яЁё0-9]+)(.*)$", word)
        if not m:
            return [word]

        lead_punct, core, trail_punct = m.group(1), m.group(2), m.group(3)
        is_cyrillic = bool(re.search(r"[а-яА-ЯёЁ]", core))
        is_latin = bool(re.search(r"[a-zA-Z]", core)) and not is_cyrillic

        # Для английского языка переносы разрешены только в экстремально узких баблах (safe_w / safe_h < 0.48) и для слов >= 11 букв
        if is_latin:
            if (safe_w / max(1.0, float(safe_h))) >= 0.48 or len(core) < 11:
                return [word]

        is_narrow_bubble = safe_w < 140
        min_ru_len = 9 if is_narrow_bubble else 11
        if is_cyrillic and len(core) < min_ru_len:
            return [word]

        split_candidates = self._find_syllable_split_indices(
            core,
            min_part_len=3,
            is_narrow_bubble=is_narrow_bubble,
        )
        if not split_candidates:
            return [word]

        mid = len(core) / 2.0
        fitting_splits: list[tuple[float, int]] = []
        for idx in split_candidates:
            part1 = f"{lead_punct}{core[:idx]}-"
            part2 = f"{core[idx:]}{trail_punct}"
            w1 = self._measure_line_width(draw, part1, font)
            w2 = self._measure_line_width(draw, part2, font)
            if w1 <= max_width * 1.15 and w2 <= max_width * 1.18:
                fitting_splits.append((abs(idx - mid), idx))

        if fitting_splits:
            fitting_splits.sort(key=lambda item: item[0])
            best_idx = fitting_splits[0][1]
            return [f"{lead_punct}{core[:best_idx]}-", f"{core[best_idx:]}{trail_punct}"]

        return [word]

    @classmethod
    def _preprocess_tokens_no_orphans(cls, raw_words: list[str]) -> list[str]:
        """
        Блок 3.2 п.4: Запрет висячих предлогов, союзов и частиц (No Orphans / Widows):
        - Частицы (`бы`, `ли`, `же`, `то`, `ка`, `—`, `--`, `?!`, `...`) жестко склеиваются с предыдущим словом.
        - Однобуквенные и двухбуквенные предлоги/союзы/артикли (`в`, `к`, `с`, `у`, `о`, `а`, `и`, `не`, `ни`, `на`, `но`,
          `по`, `за`, `из`, `от`, `до`, `I`, `a`, `an`, `the`, `to`, `in`, `of`, `on`, `at`, `or`, `by`)
          склеиваются со следующим словом.
        """
        if not raw_words:
            return []
        step1: list[str] = []
        for tok in raw_words:
            clean_low = re.sub(r"[^a-zA-Zа-яА-ЯёЁ]", "", tok).lower()
            if step1 and (
                tok in cls._ENCLITIC_PARTICLES
                or (clean_low in {"бы", "б", "ли", "ль", "же", "ж", "то", "ка"} and len(clean_low) <= 2)
            ):
                step1[-1] = f"{step1[-1]} {tok}"
            else:
                step1.append(tok)

        merged: list[str] = []
        i = 0
        while i < len(step1):
            tok = step1[i]
            clean_low = re.sub(r"^[^a-zA-Zа-яА-ЯёЁ]+|[^a-zA-Zа-яА-ЯёЁ]+$", "", tok).lower()
            has_trailing_punct = bool(tok and tok[-1] in ".,!?;:…»)")
            if (
                i + 1 < len(step1)
                and not has_trailing_punct
                and clean_low in cls._SHORT_WORDS_BIND_NEXT
            ):
                next_clean = re.sub(r"[^a-zA-Zа-яА-ЯёЁ]", "", step1[i + 1])
                if len(next_clean) <= 9:
                    merged.append(f"{tok} {step1[i + 1]}")
                    i += 2
                    continue
            merged.append(tok)
            i += 1
        return merged

    @classmethod
    def _is_hanging_short_word(cls, token: str) -> bool:
        last_piece = token.rsplit(" ", 1)[-1]
        if last_piece and last_piece[-1] in ".,!?;:…»)":
            return False
        clean = re.sub(r"^[^a-zA-Zа-яА-ЯёЁ']+|[^a-zA-Zа-яА-ЯёЁ']+$", "", last_piece).lower()
        return clean in cls._HANGING_SHORT_WORDS

    @staticmethod
    def _lame_superellipse_factor(k: int, total_lines: int, p: float = 1.85) -> float:
        """
        Блок 3.1: Вычисляет целевой коэффициент ширины строки k из total_lines по кривой суперэллипса Ламе (p = 1.85):
        Для K=3 -> [0.78, 1.00, 0.78]
        Для K=4 -> [0.72, 0.96, 0.96, 0.72]
        Для K=5 -> [0.68, 0.91, 1.00, 0.91, 0.68]
        """
        if total_lines <= 1:
            return 1.0
        if total_lines == 2:
            return 0.90
        t_k = ((2.0 * (k + 0.5)) / float(total_lines) - 1.0) * 0.85
        t_abs = min(0.88, abs(t_k))
        val = max(0.55, (1.0 - (t_abs ** p)) ** (1.0 / p))
        return float(val)

    def _wrap_text_contour_dp(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
        target_h: int | None = None,
        allow_hyphenation: bool = False,
        elliptical_shaping: bool = True,
        chord_profile: list[int] | None = None,
    ) -> str:
        """
        Блок 3: 2D-DP алгоритм «алмазной/овальной» верстки (`Diamond / Contour Shaping`):
        Укладывает слова по кривой суперэллипса Ламе ($p = 1.85$) и реальным хордам `safe_mask` бабла,
        исключая формы «песочных часов», «перевернутой пирамиды», «лесенки дефисов» и висячие предлоги.
        """
        raw_words = text.split()
        if not raw_words:
            return ""

        tokens = self._preprocess_tokens_no_orphans(raw_words)
        if not tokens:
            return ""

        safe_w = max(16, int(max_width))
        safe_h = max(16, int(target_h if target_h and target_h > 0 else max_width))
        has_cyrillic = bool(re.search(r"[А-Яа-яЁё]", text))

        # Раздельная политика переноса слов (Блок 3.3):
        # В русском языке умный перенос длинных слов активен в узких/средних баблах;
        # в английском — только при allow_hyphenation=True И экстремально узком бабле (safe_w / safe_h < 0.48).
        eff_allow_hyph = allow_hyphenation or (has_cyrillic and safe_w < 165)
        if not has_cyrillic and (safe_w / max(1.0, float(safe_h))) >= 0.48:
            eff_allow_hyph = False

        # Если вся фраза короткая (1–2 токена) и легко помещается на 1 строку в широком/круглом бабле — не дробим
        full_joined = " ".join(tokens)
        full_w = self._measure_line_width(draw, full_joined, font)
        if len(tokens) <= 2 and full_w <= safe_w * (0.92 if elliptical_shaping else 1.02) and safe_w >= safe_h * 0.62:
            return full_joined

        words: list[str] = []
        is_soft_hyph: list[bool] = []
        for w in tokens:
            w_len = self._measure_line_width(draw, w, font)
            if eff_allow_hyph and " " not in w and (w_len > safe_w * 0.56 or len(w) >= (9 if safe_w < 140 else 11)):
                parts = self._split_long_word(draw, w, font, safe_w, safe_w=safe_w, safe_h=safe_h)
                if len(parts) == 2 and parts[0].endswith("-"):
                    words.append(parts[0])
                    is_soft_hyph.append(True)
                    words.append(parts[1])
                    is_soft_hyph.append(False)
                else:
                    for pt in parts:
                        words.append(pt)
                        is_soft_hyph.append(False)
            else:
                words.append(w)
                is_soft_hyph.append(False)

        n = len(words)
        if n == 1:
            return words[0]

        word_widths = [self._measure_line_width(draw, w, font) for w in words]
        # Ширина первой половины слова без дефиса, если обе половины остаются на одной строке
        stem_widths = [
            self._measure_line_width(draw, w[:-1], font) if (sh and w.endswith("-")) else ww
            for w, ww, sh in zip(words, word_widths, is_soft_hyph)
        ]
        space_w = max(2.0, self._measure_line_width(draw, " ", font))
        fs = self._get_font_pixel_size(font)
        ba_y1, ref_y0, ref_y1, m_def = self._get_font_line_metrics(font=font, font_size=fs, use_upper=True, has_tall_cyr=True)
        line_h_est = max(float(max(ref_y1 - ref_y0, ba_y1 + m_def)) + float(fs) * 0.22, float(fs) * 1.28, 12.0)

        def _join_line_tokens(p_idx: int, i_idx: int) -> str:
            out_parts: list[str] = []
            idx = p_idx
            while idx < i_idx:
                if is_soft_hyph[idx] and (idx + 1) < i_idx and words[idx].endswith("-"):
                    # Обе части мягкого переноса попали на одну строку — склеиваем обратно в целое слово без дефиса!
                    out_parts.append(words[idx][:-1] + words[idx + 1])
                    idx += 2
                else:
                    out_parts.append(words[idx])
                    idx += 1
            return " ".join(out_parts)

        def _line_width(p_idx: int, i_idx: int) -> float:
            total = 0.0
            items_cnt = 0
            idx = p_idx
            while idx < i_idx:
                if is_soft_hyph[idx] and (idx + 1) < i_idx:
                    total += stem_widths[idx] + word_widths[idx + 1]
                    idx += 2
                else:
                    total += word_widths[idx]
                    idx += 1
                items_cnt += 1
            return total + space_w * max(0, items_cnt - 1)

        total_text_w = sum(word_widths) + space_w * max(0, n - 1)
        max_single_w = max(word_widths, default=float(safe_w))
        w_base = max(float(safe_w), max_single_w)

        # Оценка оптимального числа строк по емкости и аспектному соотношению бабла (формула manga2eng / TypeR)
        bubble_aspect = max(0.28, min(3.5, float(safe_w) / max(1.0, float(safe_h))))
        min_phys_lines = max(1, int(math.ceil(total_text_w / max(1.0, w_base * (0.82 if elliptical_shaping else 0.94)))))
        if elliptical_shaping and total_text_w > w_base * 0.78 and min_phys_lines == 1 and n >= 2:
            min_phys_lines = 2
        ideal_block_w = math.sqrt(max(1.0, (total_text_w / 0.82) * line_h_est * bubble_aspect))
        ideal_block_w = min(w_base * 0.95, max(w_base * 0.55, ideal_block_w))
        k_est = max(min_phys_lines, int(round(total_text_w / max(1.0, ideal_block_w * 0.84))))

        k_min = max(1, min(min_phys_lines, k_est - 1))
        k_max = min(n, max(k_est + 2, min_phys_lines + 3))

        def _get_line_cap_and_target(k_idx: int, total_k: int) -> tuple[float, float]:
            lame_factor = self._lame_superellipse_factor(k_idx, total_k, p=1.85) if elliptical_shaping else 1.0
            w_target = w_base * lame_factor
            if chord_profile and len(chord_profile) >= 4 and elliptical_shaping:
                ch_len = len(chord_profile)
                y_start = int(round((k_idx / float(total_k)) * (ch_len - 1)))
                y_end = int(round(((k_idx + 1) / float(total_k)) * (ch_len - 1)))
                y_start = max(0, min(ch_len - 1, y_start))
                y_end = max(y_start + 1, min(ch_len, y_end + 1))
                chord_min = float(min(chord_profile[y_start:y_end]))
                w_cap = max(max_single_w, min(w_base * 1.04, max(chord_min, w_base * lame_factor * 1.06)))
                w_target = min(w_target, w_cap * 0.95)
            else:
                w_cap = max(max_single_w, w_base * (min(1.02, lame_factor * 1.22) if elliptical_shaping else 1.02))
            return w_cap, w_target

        INF = 1e18
        best_overall_cost = INF
        best_overall_lines: list[str] | None = None

        # Предрасчет матриц ширины и текста всех допустимых отрезков words[p:i]
        seg_lw = [[0.0] * (n + 1) for _ in range(n + 1)]
        for p_i in range(n):
            for end_i in range(p_i + 1, n + 1):
                seg_lw[p_i][end_i] = _line_width(p_i, end_i)

        for K in range(k_min, k_max + 1):
            # 2D-DP с сохранением индекса начала предыдущей строки (k, p, i), чтобы точно контролировать выпуклость
            # dp_states[k][i] -> список (cost, p_start, lw, prev_q) для строки k = words[p_start:i]
            dp_states: list[list[list[tuple[float, int, float, int]]]] = [
                [[] for _ in range(n + 1)] for _ in range(K + 1)
            ]
            dp_states[0][0] = [(0.0, 0, 0.0, -1)]

            for k in range(1, K + 1):
                w_cap, w_target = _get_line_cap_and_target(k - 1, K)
                is_last_line = (k == K)
                mid_idx = (K - 1) / 2.0
                for i in range(k, n - (K - k) + 1):
                    if is_last_line and i != n:
                        continue
                    if is_last_line and is_soft_hyph[i - 1]:
                        continue
                    last_tok = words[i - 1]
                    hanging_pen = 260.0 if (not is_last_line and self._is_hanging_short_word(last_tok)) else 0.0
                    hyph_pen = 28.0 if (not is_last_line and is_soft_hyph[i - 1]) else 0.0

                    candidates_for_i: list[tuple[float, int, float, int]] = []
                    for p in range(k - 1, i):
                        prev_list = dp_states[k - 1][p]
                        if not prev_list:
                            continue
                        lw = seg_lw[p][i]
                        # 1. Hard Overflow Penalty
                        if i > p + 1 and lw > w_cap:
                            continue

                        base_step = hanging_pen + hyph_pen
                        if lw > w_cap:
                            base_step += ((lw - w_cap) / w_base) ** 2 * 900.0

                        # 2. Contour Shaping Penalty (суперэллипс Ламе p=1.85)
                        base_step += ((lw - w_target) / w_base) ** 2 * 100.0

                        # 4. Запрет коротких «сирот» на первой и последней строке
                        if k == 1 and K >= 3 and lw < 0.34 * w_base:
                            base_step += 200.0
                        if is_last_line and K >= 3 and lw < 0.35 * w_base:
                            base_step += 220.0

                        # Запрет «лесенки дефисов» (две соседние строки подряд заканчиваются дефисом)
                        if k >= 2 and p > 0 and words[p - 1].endswith("-") and last_tok.endswith("-"):
                            base_step += 500.0

                        best_c_for_p = INF
                        best_q_for_p = -1
                        for prev_c, prev_q, prev_lw, _ in prev_list:
                            conv_pen = 0.0
                            if k >= 2 and elliptical_shaping and K >= 3:
                                if (k - 1) <= mid_idx:
                                    if lw > prev_lw * 1.05:
                                        conv_pen -= 8.0
                                    elif lw < prev_lw * 0.92:
                                        conv_pen += 180.0
                                if (k - 1) > mid_idx and lw > prev_lw * 1.08:
                                    conv_pen += 180.0
                                if k >= 3 and prev_q >= 0:
                                    prev2_lw = seg_lw[prev_q][p]
                                    if prev_lw < prev2_lw * 0.92 and prev_lw < lw * 0.92:
                                        conv_pen += 180.0
                            tot = prev_c + base_step + conv_pen
                            if tot < best_c_for_p:
                                best_c_for_p = tot
                                best_q_for_p = prev_q

                        if best_c_for_p < INF:
                            candidates_for_i.append((best_c_for_p, p, lw, best_q_for_p))

                    if candidates_for_i:
                        candidates_for_i.sort(key=lambda item: item[0])
                        # Оставляем топ-4 лучших состояний по стоимости с разной шириной строки lw
                        dp_states[k][i] = candidates_for_i[:4]

            final_states = dp_states[K][n]
            if not final_states:
                continue

            best_final_c, best_p, best_lw, _ = final_states[0]
            # Реконструируем строки для кандидата K
            lines_k: list[str] = []
            widths_k: list[float] = []
            curr_i = n
            curr_p = best_p
            valid_recon = True
            for k_step in range(K, 0, -1):
                if curr_p < 0:
                    valid_recon = False
                    break
                lines_k.append(_join_line_tokens(curr_p, curr_i))
                widths_k.append(seg_lw[curr_p][curr_i])
                if k_step > 1:
                    # Находим состояние на шаге k_step, из которого мы пришли
                    st_list = dp_states[k_step][curr_i]
                    matched = next((st for st in st_list if st[1] == curr_p), st_list[0] if st_list else None)
                    next_p = matched[3] if matched is not None else -1
                    curr_i = curr_p
                    curr_p = next_p
            if not valid_recon:
                continue
            lines_k.reverse()
            widths_k.reverse()

            w_block = max(widths_k, default=1.0)
            h_block = max(line_h_est, K * line_h_est)
            block_aspect = max(0.15, w_block / max(1.0, h_block))
            aspect_penalty = (math.log(block_aspect / bubble_aspect) ** 2) * 90.0
            height_overflow_penalty = ((h_block - safe_h) / max(1.0, float(safe_h))) ** 2 * 450.0 if h_block > safe_h else 0.0

            total_k_cost = best_final_c + aspect_penalty + height_overflow_penalty + (K * 14.0)
            if total_k_cost < best_overall_cost:
                best_overall_cost = total_k_cost
                best_overall_lines = lines_k

        if best_overall_lines:
            return "\n".join(best_overall_lines)

        # Надёжный фоллбэк, если при сверхкрупном кегле ни один K не прошел жесткий порог w_cap
        fallback_lines: list[str] = []
        cur_line: list[str] = []
        cur_w = 0.0
        for w, ww in zip(words, word_widths):
            add_w = ww if not cur_line else (space_w + ww)
            if cur_line and (cur_w + add_w) > w_base:
                fallback_lines.append(" ".join(cur_line))
                cur_line = [w]
                cur_w = ww
            else:
                cur_line.append(w)
                cur_w += add_w
        if cur_line:
            fallback_lines.append(" ".join(cur_line))
        return "\n".join(fallback_lines)

    def _wrap_text_pixel(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
        allow_hyphenation: bool = False,
        elliptical_shaping: bool = True,
        target_h: int | None = None,
        chord_profile: list[int] | None = None,
    ) -> str:
        """Совместимая обертка над 2D-DP алгоритмом алмазной верстки `_wrap_text_contour_dp`."""
        return self._wrap_text_contour_dp(
            draw=draw,
            text=text,
            font=font,
            max_width=max_width,
            target_h=target_h,
            allow_hyphenation=allow_hyphenation,
            elliptical_shaping=elliptical_shaping,
            chord_profile=chord_profile,
        )

    def format_diamond_lines(
        self,
        text: str,
        bbox: list[int] | tuple[int, int, int, int],
        style_preset: str = "dialogue",
        font_family: str | None = None,
        font_size: int | None = None,
        uppercase: bool | None = None,
    ) -> str:
        """
        Блок 6.3: Принудительно рассчитывает алмазную разбивку строк (`\\n`) по профилю суперэллипса
        для кнопки «💎 Алмазная разбивка» в UI инспектора блока.
        """
        clean_raw = normalize_manga_special_symbols(re.sub(r"\s+", " ", (text or "").strip()))
        if not clean_raw:
            return ""
        use_upper = self.uppercase_comic_style if uppercase is None else bool(uppercase)
        preset = (style_preset or "dialogue").lower().strip()
        font_path = self._resolve_font_family_path(font_family=font_family, style_preset=preset)
        formatted_text = self._format_comic_typography(clean_raw, font_path=font_path, use_upper=use_upper)

        x1, y1, x2, y2 = [int(round(float(c))) for c in bbox[:4]]
        bw = max(24, x2 - x1)
        bh = max(24, y2 - y1)
        margin_px = max(7, int(min(bw, bh) * (0.08 if preset == "shout" else 0.11)))
        safe_w = max(16, bw - 2 * margin_px)
        safe_h = max(16, bh - 2 * margin_px)

        dummy_img = Image.new("RGB", (16, 16), (255, 255, 255))
        dummy_draw = ImageDraw.Draw(dummy_img)
        eff_fs = int(font_size) if (font_size and int(font_size) >= 8) else max(14, min(26, int(min(safe_w, safe_h) * 0.22)))
        font = self._get_font(eff_fs, font_family_path=font_path)

        wrapped = self._wrap_text_contour_dp(
            draw=dummy_draw,
            text=formatted_text,
            font=font,
            max_width=safe_w,
            target_h=safe_h,
            allow_hyphenation=True,
            elliptical_shaping=(preset not in ("narration", "sfx")),
            chord_profile=None,
        )
        return wrapped

    def _fits_in_box(
        self,
        draw: ImageDraw.ImageDraw,
        wrapped_text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        spacing: int,
        stroke_w: int,
        target_w: int,
        target_h: int,
        elliptical: bool = True,
        align: str = "center",
        width_tolerance_ratio: float = 1.0,
        chord_profile: list[int] | None = None,
    ) -> bool:
        lines = wrapped_text.split("\n")
        if self._needs_special_rendering(font, wrapped_text):
            fs = self._get_font_pixel_size(font)
            line_widths = [self._measure_line_width(draw, ln, font) + 2 * stroke_w for ln in lines]
            total_w = max(line_widths, default=0.0)
            ba_y1, ref_y0, ref_y1, _ = self._get_font_line_metrics(font=font, font_size=fs, use_upper=False, has_tall_cyr=True)
            glyph_h = max(fs, ref_y1 - ref_y0)
            step_y = max(ba_y1 + 2 * stroke_w + spacing, glyph_h + 2 * stroke_w + max(2, int(round(fs * 0.12))))
            total_h = (len(lines) - 1) * step_y + glyph_h + 2 * stroke_w
        else:
            bbox = draw.multiline_textbbox(
                (0, 0),
                wrapped_text,
                font=font,
                spacing=spacing,
                align=align,
                stroke_width=stroke_w,
            )
            total_w = bbox[2] - bbox[0]
            total_h = bbox[3] - bbox[1]

        allowed_w = target_w * max(1.0, float(width_tolerance_ratio))
        if total_w > allowed_w or total_h > target_h:
            return False

        if not elliptical or len(lines) <= 1:
            return True

        n_lines = len(lines)
        for idx, line in enumerate(lines):
            line_w = self._measure_line_width(draw, line, font) + 2 * stroke_w
            if chord_profile and len(chord_profile) >= 4:
                ch_len = len(chord_profile)
                y_mid = int(round(((idx + 0.5) / float(n_lines)) * (ch_len - 1)))
                y_mid = max(0, min(ch_len - 1, y_mid))
                chord_w = float(chord_profile[y_mid]) * max(1.0, float(width_tolerance_ratio))
            else:
                lame_f = self._lame_superellipse_factor(idx, n_lines, p=1.85)
                chord_w = allowed_w * min(1.04, lame_f * 1.15)
            if line.strip().count(" ") == 0:
                if line_w > allowed_w * 1.02:
                    return False
            elif line_w > max(chord_w, allowed_w * 0.78):
                return False

        return True

    @staticmethod
    def _is_bbox_modified(block: dict[str, Any]) -> bool:
        if bool(block.get("bbox_modified", False)):
            return True
        clean_b = block.get("clean_bbox") or block.get("original_bbox")
        cur_b = block.get("bbox")
        if (
            isinstance(clean_b, (list, tuple))
            and isinstance(cur_b, (list, tuple))
            and len(clean_b) >= 4
            and len(cur_b) >= 4
        ):
            return [int(round(float(x))) for x in clean_b[:4]] != [int(round(float(x))) for x in cur_b[:4]]
        return False

    @staticmethod
    def _is_bbox_resized(block: dict[str, Any]) -> bool:
        if bool(block.get("bbox_resized", False)):
            return True
        clean_b = block.get("clean_bbox") or block.get("original_bbox")
        cur_b = block.get("bbox")
        if (
            isinstance(clean_b, (list, tuple))
            and isinstance(cur_b, (list, tuple))
            and len(clean_b) >= 4
            and len(cur_b) >= 4
        ):
            cw = int(round(float(clean_b[2]))) - int(round(float(clean_b[0])))
            ch = int(round(float(clean_b[3]))) - int(round(float(clean_b[1])))
            nw = int(round(float(cur_b[2]))) - int(round(float(cur_b[0])))
            nh = int(round(float(cur_b[3]))) - int(round(float(cur_b[1])))
            if abs(nw - cw) >= 3 or abs(nh - ch) >= 3:
                return True
        return bool(block.get("bbox_modified", False))

    @staticmethod
    def _compute_stroke_width(
        font_size: int,
        is_free_text: bool,
        style_preset: str,
        stroke_width_override: int | None,
    ) -> int:
        if stroke_width_override is not None:
            return max(0, min(16, int(stroke_width_override)))
        preset = (style_preset or "dialogue").lower().strip()
        if preset == "sfx":
            return max(4, font_size // 6)
        if preset == "shout":
            return max(3, font_size // 7)
        if preset == "whisper":
            return max(1, font_size // 12)
        if is_free_text:
            return max(3, font_size // 8)
        return max(2, font_size // 10)

    def _analyze_bubble_geometry(
        self,
        cleaned_img_np: np.ndarray,
        text_bbox: list[int] | tuple[int, int, int, int],
        img_w: int,
        img_h: int,
        style_preset: str = "dialogue",
        is_free_text: bool = False,
        other_bboxes: list[list[int] | tuple[int, int, int, int]] | None = None,
    ) -> dict[str, Any]:
        """Блок 2: Вызывает сегментатор реальных границ бабла и Distance Transform из cleaner.py."""
        return analyze_bubble_geometry(
            cleaned_img_np=cleaned_img_np,
            text_bbox=text_bbox,
            img_w=img_w,
            img_h=img_h,
            style_preset=style_preset,
            is_free_text=is_free_text,
            other_bboxes=other_bboxes,
        )

    def _fit_text_to_box(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        target_w: int,
        target_h: int,
        min_font_limit: int,
        max_font_limit: int,
        is_free_text: bool = False,
        stroke_width_override: int | None = None,
        force_rectangular: bool = False,
        unconstrained_size: bool = False,
        font_family_path: str | None = None,
        style_preset: str = "dialogue",
        align: str = "center",
        overflow_mode: str = "shrink",
        max_expand_w: int | None = None,
        use_upper: bool = True,
        chord_profile: list[int] | None = None,
        preserve_explicit_newlines: bool = False,
    ) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int, int, bool]:
        preset = (style_preset or "dialogue").lower().strip()
        if preset in ("narration", "sfx"):
            force_rectangular = True

        min_lim = max(6 if unconstrained_size else 8, int(min_font_limit))
        max_lim = max(min_lim, int(max_font_limit))
        use_ellipse = not (is_free_text or force_rectangular)

        if unconstrained_size:
            start_size = min(
                max_lim,
                max(min_lim, int(min(target_w, target_h) * 0.88)),
            )
        else:
            ratio = 0.45 if (is_free_text or preset in ("shout", "sfx")) else 0.40
            start_size = min(
                max_lim,
                max(min_lim, int(min(target_w, target_h) * ratio)),
            )

        has_explicit_nl = preserve_explicit_newlines and ("\n" in text)

        def _evaluate_font_size(
            font_size: int,
            allow_hyphenation: bool,
            width_slack: float = 1.0,
        ) -> tuple[bool, str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int]:
            font = self._get_font(font_size, font_family_path=font_family_path)
            stroke_w = self._compute_stroke_width(
                font_size=font_size,
                is_free_text=is_free_text,
                style_preset=preset,
                stroke_width_override=stroke_width_override,
            )
            spacing = self._get_line_spacing(font_size, style_preset=preset, use_upper=use_upper, text=text, font=font)
            eff_wrap_w = max(12, int(round(target_w * width_slack)))
            if max_expand_w and eff_wrap_w > max_expand_w:
                eff_wrap_w = max(target_w, max_expand_w)
            eff_tol = eff_wrap_w / max(1.0, float(target_w))

            if has_explicit_nl:
                explicit_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                wrapped = "\n".join(explicit_lines)
            else:
                wrapped = self._wrap_text_contour_dp(
                    draw=draw,
                    text=text,
                    font=font,
                    max_width=eff_wrap_w,
                    target_h=target_h,
                    allow_hyphenation=allow_hyphenation,
                    elliptical_shaping=use_ellipse,
                    chord_profile=chord_profile if width_slack <= 1.05 else None,
                )
            fits = self._fits_in_box(
                draw,
                wrapped,
                font,
                spacing,
                stroke_w,
                target_w,
                target_h,
                elliptical=use_ellipse and not has_explicit_nl,
                align=align,
                width_tolerance_ratio=eff_tol,
                chord_profile=chord_profile if width_slack <= 1.05 else None,
            )
            return fits, wrapped, font, stroke_w, spacing

        def _binary_search_best_fit(
            low_bound: int,
            high_bound: int,
            allow_hyphenation: bool,
            width_slack: float = 1.0,
        ) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int, int] | None:
            if high_bound < low_bound:
                return None

            fits_hi, wrap_hi, font_hi, stroke_hi, space_hi = _evaluate_font_size(high_bound, allow_hyphenation, width_slack)
            if fits_hi:
                return wrap_hi, font_hi, stroke_hi, space_hi, high_bound

            low = low_bound
            high = high_bound - 1
            best: tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int, int] | None = None

            while low <= high:
                mid = (low + high) // 2
                fits, wrapped, font, stroke_w, spacing = _evaluate_font_size(mid, allow_hyphenation, width_slack)
                if fits:
                    best = (wrapped, font, stroke_w, spacing, mid)
                    low = mid + 1
                else:
                    high = mid - 1

            if best is not None:
                best_fs = best[4]
                for probe_fs in range(min(high_bound - 1, best_fs + 2), best_fs, -1):
                    p_fits, p_wrap, p_font, p_stroke, p_space = _evaluate_font_size(probe_fs, allow_hyphenation, width_slack)
                    if p_fits:
                        return p_wrap, p_font, p_stroke, p_space, probe_fs
            return best

        # Проход 1: бинарный поиск без переносов слов строго внутри безопасной зоны бабла (safe_mask)
        best_no_hyph = _binary_search_best_fit(min_lim, start_size, allow_hyphenation=False, width_slack=1.0)
        if best_no_hyph is not None:
            return (*best_no_hyph, True)

        # Проход 2: умный слоговой/морфемный перенос (для RU от 9/11 букв, для EN в узких баблах) строго в границах бабла
        best_with_hyph = _binary_search_best_fit(min_lim, start_size, allow_hyphenation=True, width_slack=1.04)
        if best_with_hyph is not None:
            return (*best_with_hyph, True)

        # Проход 3: умеренный допуск (+12%..+18% только если бабл не был замкнут или текст экстремально длинный)
        if overflow_mode == "shrink":
            hyph_start = max(min_lim, min(start_size, max(22 if unconstrained_size else 18, min_lim)))
            best_slack_hyph = _binary_search_best_fit(min_lim, hyph_start, allow_hyphenation=True, width_slack=1.14)
            if best_slack_hyph is not None:
                return (*best_slack_hyph, True)

        min_font = self._get_font(min_lim, font_family_path=font_family_path)
        min_spacing = self._get_line_spacing(min_lim, style_preset=preset, use_upper=use_upper, text=text, font=min_font)
        min_stroke = self._compute_stroke_width(
            font_size=min_lim,
            is_free_text=is_free_text,
            style_preset=preset,
            stroke_width_override=stroke_width_override,
        )
        if overflow_mode in ("overflow", "expand_bubble"):
            full_line_px = max(1.0, self._measure_line_width(draw, text, min_font))
            line_h_est = max(min_lim + 2, int(min_lim * 1.14))
            ideal_w = int(math.sqrt(full_line_px * line_h_est * 1.35))
            upper_w = max_expand_w if (max_expand_w and max_expand_w > target_w) else max(target_w * 2, 260)
            wrap_limit = max(target_w, min(upper_w, max(ideal_w, int(target_w * 1.35), int(min_lim * 7.2))))
            wrapped = self._wrap_text_contour_dp(
                draw,
                text,
                min_font,
                wrap_limit,
                target_h=target_h,
                allow_hyphenation=True,
                elliptical_shaping=use_ellipse,
            )
            return wrapped, min_font, min_stroke, min_spacing, min_lim, False

        wrap_limit = max(target_w, int(target_w * 1.12))
        if max_expand_w and wrap_limit > max_expand_w:
            wrap_limit = max(target_w, max_expand_w)
        wrapped = self._wrap_text_contour_dp(
            draw,
            text,
            min_font,
            wrap_limit,
            target_h=target_h,
            allow_hyphenation=True,
            elliptical_shaping=use_ellipse,
        )
        return wrapped, min_font, min_stroke, min_spacing, min_lim, False

    @staticmethod
    def _compute_perceptual_luminance(roi: np.ndarray) -> float:
        if roi.size == 0:
            return 255.0
        if roi.ndim == 3 and roi.shape[2] >= 3:
            r = roi[:, :, 0].astype(np.float32)
            g = roi[:, :, 1].astype(np.float32)
            b = roi[:, :, 2].astype(np.float32)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            return float(np.median(lum))
        return float(np.median(roi))

    @staticmethod
    def _expand_free_text_width(
        bbox: list[int],
        all_blocks: list[dict[str, Any]],
        img_w: int,
    ) -> tuple[int, int]:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw >= bh * 0.85 and bw >= 95:
            return x1, x2

        desired_w = min(img_w - 8, max(bw, int(bh * 0.90), 115))
        cx = (x1 + x2) // 2
        nx1 = max(4, cx - desired_w // 2)
        nx2 = min(img_w - 4, cx + desired_w // 2)

        for other in all_blocks:
            ox1, oy1, ox2, oy2 = other["bbox"]
            if [ox1, oy1, ox2, oy2] == [x1, y1, x2, y2]:
                continue
            if min(y2, oy2) > max(y1, oy1):
                if ox2 <= x1 and nx1 < ox2 + 4:
                    nx1 = ox2 + 4
                if ox1 >= x2 and nx2 > ox1 - 4:
                    nx2 = ox1 - 4

        if nx2 - nx1 < bw:
            return x1, x2
        return nx1, nx2

    @staticmethod
    def _compute_weighted_median(values_and_weights: list[tuple[int, float]], default_val: int = 20) -> int:
        """Вычисляет взвешенную медиану кегля шрифта страницы (с весом sqrt(len(text)))."""
        if not values_and_weights:
            return default_val
        sorted_vw = sorted(values_and_weights, key=lambda item: item[0])
        total_w = sum(max(0.1, w) for _, w in sorted_vw)
        half_w = total_w / 2.0
        acc = 0.0
        for val, w in sorted_vw:
            acc += max(0.1, w)
            if acc >= half_w:
                return int(round(val))
        return int(round(sorted_vw[-1][0]))

    def render_blocks(
        self,
        cleaned_image: Image.Image,
        blocks: list[dict[str, Any]],
        uppercase: bool | None = None,
        max_font_size: int | None = None,
        min_font_size: int | None = None,
        stroke_width_override: int | None = None,
        overflow_mode: str | None = None,
    ) -> Image.Image:
        """
        Двухпроходный конвейер тайпсеттинга страницы манги (v3.5 Pro Scanlation Edition):
        - Проход 1: Геометрическая сегментация внутренней полости каждого бабла (`_analyze_bubble_geometry`),
          поиск индивидуального максимального кегля `raw_best_fs[i]` и расчет базового кегля страницы `page_base_fs`
          (взвешенная медиана с весом $\\sqrt{\\text{len(text)}}$).
        - Проход 2: Гармонизация кегля между баблами страницы, 2D-DP алмазная верстка, детекция `font_too_small_warning`
          и отрисовка текста в оптическом центре бабла (`Pole of Inaccessibility` + оптическое смещение).
        """
        canvas = cleaned_image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        img_np = np.array(canvas)
        img_w, img_h = canvas.size

        use_upper = self.uppercase_comic_style if uppercase is None else uppercase
        font_min = int(min_font_size) if (min_font_size and min_font_size >= 6) else self.min_font_size
        font_cap = int(max_font_size) if (max_font_size and max_font_size >= 8) else self.max_font_size
        if font_min > font_cap:
            font_min, font_cap = font_cap, font_min
        global_overflow_mode = self._normalize_overflow_mode(overflow_mode if overflow_mode is not None else self.overflow_mode)

        # ==================== ПРОХОД 1: АНАЛИЗ ГЕОМЕТРИИ И ЕМКОСТИ БАБЛОВ СТРАНИЦЫ ====================
        prepared_items: list[dict[str, Any]] = []
        dialogue_fs_weights: list[tuple[int, float]] = []

        all_active_bboxes: list[list[int]] = []
        for b_cand in blocks:
            if bool(b_cand.get("is_deleted", False)):
                continue
            if not (b_cand.get("translated_text") or "").strip():
                continue
            rb = b_cand.get("bbox")
            if isinstance(rb, (list, tuple)) and len(rb) >= 4:
                all_active_bboxes.append([int(round(float(c))) for c in rb[:4]])

        for block in blocks:
            block["font_too_small_warning"] = False
            if bool(block.get("is_deleted", False)):
                continue
            raw_text = normalize_manga_special_symbols((block.get("translated_text") or "").strip())
            if not raw_text:
                continue

            style_preset = str(block.get("style_preset") or "dialogue").lower().strip()
            if style_preset not in ("dialogue", "thought", "shout", "whisper", "narration", "sfx"):
                style_preset = "dialogue"

            align = str(block.get("align") or "center").lower().strip()
            if align not in ("center", "left", "right"):
                align = "center"

            try:
                angle = float(block.get("angle") or 0.0)
                angle = max(-45.0, min(45.0, angle))
            except (ValueError, TypeError):
                angle = 0.0

            bg_style = self._normalize_bg_style(block.get("bg_style"))
            try:
                bg_pad_adj = int(block.get("bg_padding") or 0) if block.get("bg_padding") is not None else 0
                bg_pad_adj = max(-24, min(48, bg_pad_adj))
            except (ValueError, TypeError):
                bg_pad_adj = 0

            font_family_path = self._resolve_font_family_path(
                font_family=block.get("font_family"),
                style_preset=style_preset,
            )

            text = self._format_comic_typography(raw_text, font_path=font_family_path, use_upper=use_upper)
            is_free_text = bool(block.get("is_free_text", False)) or (style_preset == "sfx")
            is_modified = self._is_bbox_modified(block)
            is_merged_bubble = bool(block.get("is_merged_bubble", False))
            is_resized = self._is_bbox_resized(block) and not is_merged_bubble
            force_rect = (is_modified and not is_merged_bubble) or (style_preset in ("narration", "sfx"))

            raw_bbox = block.get("bbox") or [0, 0, 0, 0]
            x1, y1, x2, y2 = [int(round(float(c))) for c in raw_bbox[:4]]
            x1 = max(0, min(x1, img_w - 4))
            y1 = max(0, min(y1, img_h - 4))
            x2 = max(x1 + 4, min(x2, img_w))
            y2 = max(y1 + 4, min(y2, img_h))

            if is_free_text and not is_modified and style_preset != "narration":
                rx1, rx2 = self._expand_free_text_width([x1, y1, x2, y2], blocks, img_w)
            else:
                rx1, rx2 = x1, x2

            box_w = rx2 - rx1
            box_h = y2 - y1
            if box_w < 10 or box_h < 10:
                continue

            other_boxes_for_blk = [
                ob for ob in all_active_bboxes if ob != [x1, y1, x2, y2]
            ]

            # Сегментация реальных границ бабла и поиск Pole of Inaccessibility (Блок 2)
            if (not is_modified or is_merged_bubble) and not is_free_text:
                geom = self._analyze_bubble_geometry(
                    cleaned_img_np=img_np,
                    text_bbox=[rx1, y1, rx2, y2],
                    img_w=img_w,
                    img_h=img_h,
                    style_preset=style_preset,
                    is_free_text=is_free_text,
                    other_bboxes=other_boxes_for_blk,
                )
            else:
                pad_x = max(2, int(box_w * 0.04)) if force_rect else max(6, int(min(box_w, box_h) * 0.10))
                pad_y = max(2, int(box_h * 0.04)) if force_rect else max(6, int(min(box_w, box_h) * 0.10))
                sx1 = max(0, min(rx2 - 4, rx1 + pad_x))
                sy1 = max(0, min(y2 - 4, y1 + pad_y))
                sx2 = max(sx1 + 4, min(img_w, rx2 - pad_x))
                sy2 = max(sy1 + 4, min(img_h, y2 - pad_y))
                geom = {
                    "is_closed_bubble": False,
                    "bubble_bbox": [rx1, y1, rx2, y2],
                    "safe_bbox": [sx1, sy1, sx2, sy2],
                    "safe_w": max(10, sx2 - sx1),
                    "safe_h": max(10, sy2 - sy1),
                    "optical_center": [
                        int(round((rx1 + rx2) / 2.0)),
                        int(round((y1 + y2) / 2.0 - box_h * (0.0 if is_modified else 0.022))),
                    ],
                    "margin_px": min(pad_x, pad_y),
                    "chord_profile": None,
                }

            safe_w = max(10, int(geom["safe_w"]))
            safe_h = max(10, int(geom["safe_h"]))
            sx1, sy1, sx2, sy2 = geom["safe_bbox"]
            opt_cx, opt_cy = geom["optical_center"]
            chord_profile = geom.get("chord_profile")

            # Сохраняем геометрию в блок для отображения оверлея безопасной зоны в UI (Блок 6.1)
            block["safe_bbox"] = [int(sx1), int(sy1), int(sx2), int(sy2)]
            block["bubble_bbox"] = [int(c) for c in geom["bubble_bbox"]]
            block["optical_center"] = [int(opt_cx), int(opt_cy)]

            block_font_override = block.get("font_size")
            has_manual_fs = isinstance(block_font_override, (int, float)) and int(block_font_override) >= 6
            unconstrained_size = False
            if has_manual_fs:
                b_min = b_max = max(6, min(160, int(block_font_override)))
            elif is_resized:
                b_min, b_max = 6, min(72, max(font_cap + 8, int(round(font_cap * 1.65))))
                unconstrained_size = True
            elif style_preset == "shout":
                b_min = font_min
                b_max = min(140, max(font_cap + 6, int(round(font_cap * 1.25))))
            elif style_preset == "whisper":
                b_min = max(8, font_min - 2)
                b_max = max(b_min + 2, int(round(font_cap * 0.80)))
            else:
                b_min, b_max = font_min, font_cap

            block_stroke_override = block.get("stroke_width")
            if isinstance(block_stroke_override, (int, float)) and int(block_stroke_override) >= 0:
                eff_stroke_override: int | None = max(0, min(12, int(block_stroke_override)))
            else:
                eff_stroke_override = stroke_width_override

            block_overflow_mode = (
                self._normalize_overflow_mode(block.get("overflow_mode"))
                if block.get("overflow_mode")
                else global_overflow_mode
            )
            max_exp_w = max(safe_w, min(img_w - 16, max(int(box_w * 2.0), 260)))

            # Первичный расчет индивидуального кегля raw_best_fs
            raw_wrap, raw_font, raw_stroke, raw_spacing, raw_fs, raw_fit_ok = self._fit_text_to_box(
                draw=draw,
                text=text,
                target_w=safe_w,
                target_h=safe_h,
                min_font_limit=b_min,
                max_font_limit=b_max,
                is_free_text=is_free_text,
                stroke_width_override=eff_stroke_override,
                force_rectangular=force_rect,
                unconstrained_size=unconstrained_size,
                font_family_path=font_family_path,
                style_preset=style_preset,
                align=align,
                overflow_mode=block_overflow_mode,
                max_expand_w=max_exp_w,
                use_upper=use_upper,
                chord_profile=chord_profile,
                preserve_explicit_newlines=("\n" in raw_text),
            )

            clean_chars_len = len(re.sub(r"\s+", "", raw_text))
            if (
                style_preset in ("dialogue", "thought")
                and not has_manual_fs
                and not is_resized
                and not is_free_text
                and clean_chars_len >= 2
            ):
                weight = math.sqrt(float(max(2, clean_chars_len)))
                dialogue_fs_weights.append((raw_fs, weight))

            prepared_items.append(
                {
                    "block": block,
                    "raw_text": raw_text,
                    "text": text,
                    "style_preset": style_preset,
                    "align": align,
                    "angle": angle,
                    "bg_style": bg_style,
                    "bg_pad_adj": bg_pad_adj,
                    "font_family_path": font_family_path,
                    "is_free_text": is_free_text,
                    "is_modified": is_modified,
                    "is_resized": is_resized,
                    "force_rect": force_rect,
                    "has_manual_fs": has_manual_fs,
                    "unconstrained_size": unconstrained_size,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "rx1": rx1,
                    "rx2": rx2,
                    "box_w": box_w,
                    "box_h": box_h,
                    "safe_w": safe_w,
                    "safe_h": safe_h,
                    "sx1": sx1,
                    "sy1": sy1,
                    "sx2": sx2,
                    "sy2": sy2,
                    "opt_cx": opt_cx,
                    "opt_cy": opt_cy,
                    "chord_profile": chord_profile,
                    "b_min": b_min,
                    "b_max": b_max,
                    "eff_stroke_override": eff_stroke_override,
                    "block_overflow_mode": block_overflow_mode,
                    "max_exp_w": max_exp_w,
                    "raw_wrap": raw_wrap,
                    "raw_font": raw_font,
                    "raw_stroke": raw_stroke,
                    "raw_spacing": raw_spacing,
                    "raw_fs": raw_fs,
                    "raw_fit_ok": raw_fit_ok,
                }
            )

        page_base_fs = self._compute_weighted_median(
            dialogue_fs_weights,
            default_val=max(font_min + 2, min(font_cap, 20)),
        )

        # ==================== ПРОХОД 2: ГАРМОНИЗАЦИЯ КЕГЛЯ И ПОДГОТОВКА СЛОЕВ ====================
        def _measure_item_layer(it: dict[str, Any]) -> None:
            w_txt = it["wrapped_text"]
            f_obj = it["font"]
            sp_val = it["dyn_spacing"]
            al_val = it["align"]
            sw_val = it["stroke_w"]
            tf_val = it["text_fill"]
            sf_val = it["stroke_fill"]

            prelim_b = draw.multiline_textbbox(
                (0, 0),
                w_txt,
                font=f_obj,
                spacing=sp_val,
                align=al_val,
                stroke_width=sw_val,
            )
            p_tw = prelim_b[2] - prelim_b[0]
            p_th = prelim_b[3] - prelim_b[1]
            is_ovf = (not it["fit_ok"]) or (p_tw > it["safe_w"] + 4) or (p_th > it["safe_h"] + 4)
            it["is_overflowing_box"] = bool(is_ovf)
            it["block"]["is_overflowing"] = bool(is_ovf)

            if is_ovf and it["eff_stroke_override"] is None:
                sw_val = max(sw_val, 4 if it["block_overflow_mode"] == "overflow" else 3)
                it["stroke_w"] = sw_val

            needs_rich = self._needs_special_rendering(f_obj, w_txt)
            if needs_rich:
                r_layer = self._render_rich_text_layer(
                    wrapped_text=w_txt,
                    font=f_obj,
                    spacing=sp_val,
                    align=al_val,
                    text_fill=tf_val,
                    stroke_fill=sf_val,
                    stroke_w=sw_val,
                )
                tw_v, th_v = r_layer.size
                tb_v = (0, 0, tw_v, th_v)
            else:
                r_layer = None
                tb_v = draw.multiline_textbbox(
                    (0, 0),
                    w_txt,
                    font=f_obj,
                    spacing=sp_val,
                    align=al_val,
                    stroke_width=sw_val,
                )
                tw_v = tb_v[2] - tb_v[0]
                th_v = tb_v[3] - tb_v[1]

            it["rich_layer"] = r_layer
            it["t_bbox"] = tb_v
            it["tw"] = tw_v
            it["th"] = th_v

        for item in prepared_items:
            block = item["block"]
            raw_text = item["raw_text"]
            text = item["text"]
            style_preset = item["style_preset"]
            align = item["align"]
            bg_style = item["bg_style"]
            font_family_path = item["font_family_path"]
            is_free_text = item["is_free_text"]
            is_resized = item["is_resized"]
            force_rect = item["force_rect"]
            has_manual_fs = item["has_manual_fs"]
            unconstrained_size = item["unconstrained_size"]
            x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
            safe_w, safe_h = item["safe_w"], item["safe_h"]
            chord_profile = item["chord_profile"]
            b_min, b_max = item["b_min"], item["b_max"]
            eff_stroke_override = item["eff_stroke_override"]
            block_overflow_mode = item["block_overflow_mode"]
            max_exp_w = item["max_exp_w"]

            wrapped_text = item["raw_wrap"]
            font = item["raw_font"]
            stroke_w = item["raw_stroke"]
            dyn_spacing = item["raw_spacing"]
            chosen_fs = item["raw_fs"]
            fit_ok = item["raw_fit_ok"]

            harmonized_min = b_min
            harmonized_max = b_max
            if not has_manual_fs and not is_resized and not is_free_text:
                words_cnt = len(raw_text.split())
                chars_cnt = len(raw_text.strip())

                if style_preset == "dialogue" and words_cnt <= 3 and chars_cnt <= 14:
                    harmonized_max = min(b_max, max(b_min, int(round(page_base_fs * 1.15))))
                elif style_preset == "whisper":
                    harmonized_max = min(b_max, max(b_min, int(round(page_base_fs * 0.80))))
                elif style_preset == "shout":
                    harmonized_min = max(b_min, min(b_max, int(round(page_base_fs * 1.15))))
                    harmonized_max = max(harmonized_min, min(140, int(round(b_max * 1.25))))

                if chosen_fs > harmonized_max or chosen_fs < harmonized_min:
                    wrapped_text, font, stroke_w, dyn_spacing, chosen_fs, fit_ok = self._fit_text_to_box(
                        draw=draw,
                        text=text,
                        target_w=safe_w,
                        target_h=safe_h,
                        min_font_limit=harmonized_min,
                        max_font_limit=harmonized_max,
                        is_free_text=is_free_text,
                        stroke_width_override=eff_stroke_override,
                        force_rectangular=force_rect,
                        unconstrained_size=unconstrained_size,
                        font_family_path=font_family_path,
                        style_preset=style_preset,
                        align=align,
                        overflow_mode=block_overflow_mode,
                        max_expand_w=max_exp_w,
                        use_upper=use_upper,
                        chord_profile=chord_profile,
                        preserve_explicit_newlines=("\n" in raw_text),
                    )

            item["harmonized_min"] = harmonized_min
            item["harmonized_max"] = harmonized_max
            item["wrapped_text"] = wrapped_text
            item["font"] = font
            item["stroke_w"] = stroke_w
            item["dyn_spacing"] = dyn_spacing
            item["chosen_fs"] = chosen_fs
            item["fit_ok"] = fit_ok

            block["rendered_font_size"] = chosen_fs
            block["page_base_font_size"] = page_base_fs
            block["font_too_small_warning"] = bool(
                style_preset in ("dialogue", "thought")
                and not is_free_text
                and chosen_fs < page_base_fs * 0.72
            )

            cx1 = max(0, min(x2 - 1, x1 + 2))
            cy1 = max(0, min(y2 - 1, y1 + 2))
            cx2 = min(img_w, max(cx1 + 1, x2 - 2))
            cy2 = min(img_h, max(cy1 + 1, y2 - 2))
            roi = img_np[cy1:cy2, cx1:cx2]
            median_lum = self._compute_perceptual_luminance(roi)

            custom_bg_rgb = self._parse_hex_color(block.get("bg_color"))
            if bg_style != "none":
                if custom_bg_rgb is not None:
                    bg_fill = custom_bg_rgb
                else:
                    bg_fill = (255, 255, 255) if median_lum >= 85.0 else (18, 18, 18)
                bg_lum = 0.299 * bg_fill[0] + 0.587 * bg_fill[1] + 0.114 * bg_fill[2]
                effective_lum = bg_lum
            else:
                bg_fill = (255, 255, 255)
                effective_lum = median_lum

            if effective_lum >= 115.0:
                if style_preset == "whisper":
                    default_text_fill = (74, 74, 74)
                elif style_preset == "thought":
                    default_text_fill = (34, 34, 34)
                else:
                    default_text_fill = (15, 15, 15)
                default_stroke_fill = (255, 255, 255)
            else:
                default_text_fill = (250, 250, 250)
                default_stroke_fill = (10, 10, 10)

            custom_text_rgb = self._parse_hex_color(block.get("text_color"))
            custom_stroke_rgb = self._parse_hex_color(block.get("stroke_color"))
            item["text_fill"] = custom_text_rgb if custom_text_rgb is not None else default_text_fill
            item["stroke_fill"] = custom_stroke_rgb if custom_stroke_rgb is not None else default_stroke_fill
            item["bg_fill"] = bg_fill
            item["center_x"] = float(item["opt_cx"])
            item["center_y"] = float(item["opt_cy"])

            _measure_item_layer(item)

        # ==================== ПРОХОД 2.5: ЗАЩИТА ОТ СТОЛКНОВЕНИЯ СОСЕДНИХ/СДВОЕННЫХ БАБЛОВ ====================
        # Если два соседних бабла расположены вплотную (или сдвоены) и их реальные текстовые слои [tx1, ty1, tx2, ty2]
        # пересекаются по вертикали или горизонтали, раздвигаем их в пределах своих рамок и при необходимости
        # поджимаем высоту/кегль, чтобы строчки одного бабла НИКОГДА не наезжали на строчки соседнего бабла!
        if len(prepared_items) >= 2:
            min_block_gap = 5.0
            for _iter in range(3):
                any_collision = False
                for i_idx in range(len(prepared_items)):
                    for j_idx in range(i_idx + 1, len(prepared_items)):
                        it_a = prepared_items[i_idx]
                        it_b = prepared_items[j_idx]

                        ax1 = it_a["center_x"] - it_a["tw"] / 2.0
                        ay1 = it_a["center_y"] - it_a["th"] / 2.0
                        ax2 = it_a["center_x"] + it_a["tw"] / 2.0
                        ay2 = it_a["center_y"] + it_a["th"] / 2.0

                        bx1 = it_b["center_x"] - it_b["tw"] / 2.0
                        by1 = it_b["center_y"] - it_b["th"] / 2.0
                        bx2 = it_b["center_x"] + it_b["tw"] / 2.0
                        by2 = it_b["center_y"] + it_b["th"] / 2.0

                        x_overlap = min(ax2, bx2) - max(ax1, bx1)
                        y_overlap = min(ay2, by2) - max(ay1, by1)
                        if x_overlap <= -min_block_gap or y_overlap <= -min_block_gap:
                            continue

                        # Два текстовых блока пересекаются или стоят ближе чем на min_block_gap!
                        any_collision = True
                        dx_centers = abs(it_a["center_x"] - it_b["center_x"])
                        dy_centers = abs(it_a["center_y"] - it_b["center_y"])

                        if dy_centers >= dx_centers * 0.65:
                            # Вертикальное соседство (один бабл над другим)
                            up_it, dn_it = (it_a, it_b) if (
                                it_a["center_y"] < it_b["center_y"]
                                or (it_a["center_y"] == it_b["center_y"] and it_a["y1"] <= it_b["y1"])
                            ) else (it_b, it_a)

                            needed_shift = (y_overlap + min_block_gap) / 2.0
                            up_min_cy = max(float(up_it["th"]) / 2.0 + 2.0, float(up_it["y1"]) + float(up_it["th"]) / 2.0 + 3.0)
                            dn_max_cy = min(float(img_h) - float(dn_it["th"]) / 2.0 - 2.0, float(dn_it["y2"]) - float(dn_it["th"]) / 2.0 - 3.0)

                            new_up_cy = max(up_min_cy, up_it["center_y"] - needed_shift)
                            new_dn_cy = min(dn_max_cy, dn_it["center_y"] + needed_shift)
                            up_it["center_y"] = new_up_cy
                            dn_it["center_y"] = new_dn_cy

                            # Если после раздвигания в пределах рамок слои все еще налезают друг на друга —
                            # переверстываем их с жестким потолком высоты до линии раздела!
                            rem_overlap = (new_up_cy + up_it["th"] / 2.0 + min_block_gap) - (new_dn_cy - dn_it["th"] / 2.0)
                            if rem_overlap > 0.5 and _iter == 0:
                                split_line_y = (
                                    (float(up_it["y1"] + up_it["y2"]) / 2.0)
                                    + (float(dn_it["y1"] + dn_it["y2"]) / 2.0)
                                ) / 2.0
                                for sub_it, is_upper in ((up_it, True), (dn_it, False)):
                                    if sub_it["has_manual_fs"]:
                                        continue
                                    if is_upper:
                                        avail_h = max(18, int(round(split_line_y - float(sub_it["y1"]) - 6.0)))
                                    else:
                                        avail_h = max(18, int(round(float(sub_it["y2"]) - split_line_y - 6.0)))
                                    if avail_h < sub_it["th"]:
                                        sub_it["safe_h"] = min(sub_it["safe_h"], avail_h)
                                        w_t, f_o, sw_o, sp_o, fs_o, fk_o = self._fit_text_to_box(
                                            draw=draw,
                                            text=sub_it["text"],
                                            target_w=sub_it["safe_w"],
                                            target_h=sub_it["safe_h"],
                                            min_font_limit=max(8, sub_it["b_min"] - 2),
                                            max_font_limit=min(sub_it["harmonized_max"], sub_it["chosen_fs"]),
                                            is_free_text=sub_it["is_free_text"],
                                            stroke_width_override=sub_it["eff_stroke_override"],
                                            force_rectangular=sub_it["force_rect"],
                                            unconstrained_size=False,
                                            font_family_path=sub_it["font_family_path"],
                                            style_preset=sub_it["style_preset"],
                                            align=sub_it["align"],
                                            overflow_mode="shrink",
                                            max_expand_w=sub_it["safe_w"],
                                            use_upper=use_upper,
                                            chord_profile=None,
                                            preserve_explicit_newlines=("\n" in sub_it["raw_text"]),
                                        )
                                        sub_it["wrapped_text"] = w_t
                                        sub_it["font"] = f_o
                                        sub_it["stroke_w"] = sw_o
                                        sub_it["dyn_spacing"] = sp_o
                                        sub_it["chosen_fs"] = fs_o
                                        sub_it["fit_ok"] = fk_o
                                        sub_it["block"]["rendered_font_size"] = fs_o
                                        _measure_item_layer(sub_it)
                                        if is_upper:
                                            sub_it["center_y"] = max(
                                                float(sub_it["y1"]) + sub_it["th"] / 2.0 + 3.0,
                                                min(sub_it["center_y"], split_line_y - sub_it["th"] / 2.0 - 2.5),
                                            )
                                        else:
                                            sub_it["center_y"] = min(
                                                float(sub_it["y2"]) - sub_it["th"] / 2.0 - 3.0,
                                                max(sub_it["center_y"], split_line_y + sub_it["th"] / 2.0 + 2.5),
                                            )
                        else:
                            # Горизонтальное соседство (два бабла бок о бок)
                            left_it, right_it = (it_a, it_b) if (
                                it_a["center_x"] < it_b["center_x"]
                                or (it_a["center_x"] == it_b["center_x"] and it_a["x1"] <= it_b["x1"])
                            ) else (it_b, it_a)

                            needed_shift_x = (x_overlap + min_block_gap) / 2.0
                            left_min_cx = max(float(left_it["tw"]) / 2.0 + 2.0, float(left_it["x1"]) + float(left_it["tw"]) / 2.0 + 3.0)
                            right_max_cx = min(float(img_w) - float(right_it["tw"]) / 2.0 - 2.0, float(right_it["x2"]) - float(right_it["tw"]) / 2.0 - 3.0)
                            left_it["center_x"] = max(left_min_cx, left_it["center_x"] - needed_shift_x)
                            right_it["center_x"] = min(right_max_cx, right_it["center_x"] + needed_shift_x)

                if not any_collision:
                    break

        # ==================== ПРОХОД 3: ФИНАЛЬНАЯ ОТРИСОВКА НА ХОЛСТЕ ====================
        for item in prepared_items:
            block = item["block"]
            style_preset = item["style_preset"]
            align = item["align"]
            angle = item["angle"]
            bg_style = item["bg_style"]
            bg_pad_adj = item["bg_pad_adj"]
            is_free_text = item["is_free_text"]
            x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
            rx1, rx2 = item["rx1"], item["rx2"]
            box_w, box_h = item["box_w"], item["box_h"]
            sx1, sy1, sx2, sy2 = item["sx1"], item["sy1"], item["sx2"], item["sy2"]
            block_overflow_mode = item["block_overflow_mode"]

            wrapped_text = item["wrapped_text"]
            font = item["font"]
            stroke_w = item["stroke_w"]
            dyn_spacing = item["dyn_spacing"]
            chosen_fs = item["chosen_fs"]
            is_overflowing_box = item["is_overflowing_box"]
            text_fill = item["text_fill"]
            stroke_fill = item["stroke_fill"]
            bg_fill = item["bg_fill"]
            rich_layer = item["rich_layer"]
            t_bbox = item["t_bbox"]
            tw = item["tw"]
            th = item["th"]
            center_x = item["center_x"]
            center_y = item["center_y"]
            block["optical_center"] = [int(round(center_x)), int(round(center_y))]

            if block_overflow_mode == "expand_bubble" and is_overflowing_box and not is_free_text and bg_style == "none":
                pad_exp_x = max(10, int(chosen_fs * 0.65) + bg_pad_adj)
                pad_exp_y = max(8, int(chosen_fs * 0.55) + bg_pad_adj)
                bx1 = max(2, int(round(min(rx1, center_x - tw / 2.0 - pad_exp_x))))
                bx2 = min(img_w - 2, int(round(max(rx2, center_x + tw / 2.0 + pad_exp_x))))
                by1 = max(2, int(round(min(y1, center_y - th / 2.0 - pad_exp_y))))
                by2 = min(img_h - 2, int(round(max(y2, center_y + th / 2.0 + pad_exp_y))))
                rad = max(12, int(min(bx2 - bx1, by2 - by1) * (0.18 if style_preset == "narration" else 0.44)))
                border_col = (25, 25, 25) if bg_fill[0] >= 128 else (230, 230, 230)
                draw.rounded_rectangle(
                    [bx1, by1, bx2, by2],
                    radius=rad,
                    fill=bg_fill,
                    outline=border_col,
                    width=2,
                )

            if bg_style == "bubble":
                in_x = max(1, int(box_w * 0.09) - bg_pad_adj)
                in_y = max(1, int(box_h * 0.05) - bg_pad_adj)
                bx1 = max(0, min(img_w - 2, rx1 + in_x))
                by1 = max(0, min(img_h - 2, y1 + in_y))
                bx2 = max(bx1 + 4, min(img_w, rx2 - in_x))
                by2 = max(by1 + 4, min(img_h, y2 - in_y))
                bx1 = max(0, min(bx1, int(round(center_x - tw / 2.0 - 4))))
                bx2 = min(img_w, max(bx2, int(round(center_x + tw / 2.0 + 4))))
                by1 = max(0, min(by1, int(round(center_y - th / 2.0 - 4))))
                by2 = min(img_h, max(by2, int(round(center_y + th / 2.0 + 4))))
                rad = max(8, int(min(bx2 - bx1, by2 - by1) * 0.48))
                draw.rounded_rectangle([bx1, by1, bx2, by2], radius=rad, fill=bg_fill)
            elif bg_style == "full_box":
                bx1 = max(0, rx1 - bg_pad_adj)
                by1 = max(0, y1 - bg_pad_adj)
                bx2 = min(img_w, max(bx1 + 4, rx2 + bg_pad_adj))
                by2 = min(img_h, max(by1 + 4, y2 + bg_pad_adj))
                rad = max(2, min(14, min(bx2 - bx1, by2 - by1) // 6))
                draw.rounded_rectangle([bx1, by1, bx2, by2], radius=rad, fill=bg_fill)

            if abs(angle) >= 0.5 or rich_layer is not None:
                pad_rot = max(14, stroke_w * 2 + 10 + max(0, bg_pad_adj))
                layer_w = max(16, int(math.ceil(tw + 2 * pad_rot)))
                layer_h = max(16, int(math.ceil(th + 2 * pad_rot)))
                txt_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
                txt_draw = ImageDraw.Draw(txt_layer)

                if bg_style == "text_box":
                    p_bg = max(2, 7 + bg_pad_adj)
                    rx_1 = max(0, int(pad_rot - p_bg))
                    ry_1 = max(0, int(pad_rot - p_bg))
                    rx_2 = min(layer_w, int(pad_rot + tw + p_bg))
                    ry_2 = min(layer_h, int(pad_rot + th + p_bg))
                    rad = max(4, min(16, min(rx_2 - rx_1, ry_2 - ry_1) // 4))
                    txt_draw.rounded_rectangle(
                        [rx_1, ry_1, rx_2, ry_2],
                        radius=rad,
                        fill=(*bg_fill, 255),
                    )

                if rich_layer is not None:
                    txt_layer.alpha_composite(rich_layer, dest=(pad_rot, pad_rot))
                else:
                    local_x = pad_rot - t_bbox[0]
                    local_y = pad_rot - t_bbox[1]
                    self._draw_multiline_text_clean(
                        txt_draw,
                        (local_x, local_y),
                        wrapped_text,
                        font=font,
                        fill=(*text_fill, 255),
                        spacing=dyn_spacing,
                        align=align,
                        stroke_width=stroke_w,
                        stroke_fill=(*stroke_fill, 255) if stroke_w > 0 else None,
                    )

                if abs(angle) >= 0.5:
                    rot_layer = txt_layer.rotate(angle, resample=Image.BICUBIC, expand=True)
                else:
                    rot_layer = txt_layer
                rw_rot, rh_rot = rot_layer.size
                if align == "left" and abs(angle) < 0.5:
                    paste_x = int(round(sx1 - pad_rot))
                elif align == "right" and abs(angle) < 0.5:
                    paste_x = int(round(sx2 - tw - pad_rot))
                else:
                    paste_x = int(round(center_x - rw_rot / 2.0))
                paste_y = int(round(center_y - rh_rot / 2.0))
                paste_x = max(-rw_rot // 2, min(paste_x, img_w - rw_rot // 2))
                paste_y = max(-rh_rot // 2, min(paste_y, img_h - rh_rot // 2))
                canvas.paste(rot_layer, (paste_x, paste_y), rot_layer)
                draw = ImageDraw.Draw(canvas)
            else:
                if align == "left":
                    draw_x = float(sx1) - t_bbox[0]
                elif align == "right":
                    draw_x = float(sx2) - tw - t_bbox[0]
                else:
                    draw_x = center_x - tw / 2.0 - t_bbox[0]
                draw_y = center_y - th / 2.0 - t_bbox[1]

                draw_x = max(2.0, min(draw_x, float(img_w - tw - 2)))
                draw_y = max(2.0, min(draw_y, float(img_h - th - 2)))

                if bg_style == "text_box":
                    p_bg = max(2, 7 + bg_pad_adj)
                    tx1 = max(0, int(round(draw_x + t_bbox[0] - p_bg)))
                    ty1 = max(0, int(round(draw_y + t_bbox[1] - p_bg)))
                    tx2 = min(img_w, int(round(draw_x + t_bbox[2] + p_bg)))
                    ty2 = min(img_h, int(round(draw_y + t_bbox[3] + p_bg)))
                    if tx2 > tx1 + 2 and ty2 > ty1 + 2:
                        rad = max(4, min(16, min(tx2 - tx1, ty2 - ty1) // 4))
                        draw.rounded_rectangle([tx1, ty1, tx2, ty2], radius=rad, fill=bg_fill)

                self._draw_multiline_text_clean(
                    draw,
                    (draw_x, draw_y),
                    wrapped_text,
                    font=font,
                    fill=text_fill,
                    spacing=dyn_spacing,
                    align=align,
                    stroke_width=stroke_w,
                    stroke_fill=stroke_fill if stroke_w > 0 else None,
                )

        return canvas
