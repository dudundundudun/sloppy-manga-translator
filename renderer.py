"""
Модуль верстки и отрисовки переведенного текста в бабблах и на фоне — Typesetting (Шаг 4, v1.0.6).

Изменения v1.0.6:
1. Индивидуальные шрифты (`font_family`), кэшируемые по ключу `(font_family_path, size)`,
   и сканирование всех доступных `.ttf`/`.otf` из `fonts/` + системных пресетов (`list_available_fonts`).
2. Поддержка пресетов стилей бабблов (`style_preset`):
   - `"dialogue"` (Диалог): классический комиксный стиль по центру;
   - `"thought"` (Мысли): курсивный/мягкий шрифт, темно-серый текст `#222222`, увеличенный межстрочный интервал;
   - `"shout"` (Крик): жирный акцидентный кегль, увеличенная контрастная обводка;
   - `"whisper"` (Шепот): уменьшенный аккуратный кегль и тонкая обводка;
   - `"narration"` (Рассказчик / Квадратная рамка): прямоугольная верстка (`force_rectangular=True`) без эллиптического сжатия;
   - `"sfx"` (Звук / Фон): крупная контрастная обводка `stroke_width >= 4`.
3. Поддержка индивидуальных цветов (`text_color`, `stroke_color`), толщины обводки (`stroke_width`),
   выравнивания (`align`: `"center"`, `"left"`, `"right"`) и поворота текста (`angle`: `-45°..+45°`
   через рендер на прозрачном `RGBA`-слое с `Image.rotate(angle, resample=Image.BICUBIC, expand=True)`).
4. Поддержка подложки под текстом (`bg_style`: `"none"`, `"text_box"`, `"bubble"`, `"full_box"`,
   `bg_color`, `bg_padding`), чтобы полностью перекрывать просвечивающий японский текст или сложный фон
   без повторного вызова нейросети инпейнтинга.
5. Защита от «квадратиков» (`.notdef` tofu `□`) на месте сердечек, звездочек, нот и спецсимволов:
   - Автоматическая нормализация всех вариантов эмодзи/Unicode сердечек (`❤`, `💕`, `💖`, `♥`, `♡`),
     звездочек (`★`, `☆`, `✨`, `✦`, `⭐`), нот (`♪`, `♫`, `🎵`, `🎶`), волнистых тильд (`〜`, `～`)
     и удаление невидимых селекторов (`\uFE0F`, `\uFE0E`, `\u200D`).
   - Векторная 4x-сглаженная отрисовка символов манги (`♥`, `♡`, `★`, `☆`, `✦`, `♪`, `♫`, `◆`, `◇`)
     с точным цветом текста и обводкой блока (работает идеально на любом шрифте, даже если в `.ttf` нет глифов).
   - Посимвольный детектор отсутствующих глифов (`_is_glyph_missing`) с автоматическим переключением
     на системный символьный шрифт (`Segoe UI Symbol`, `Segoe UI`, `Arial`, `DejaVuSans`) и пропуском `.notdef`.
6. Сохранен бинарный поиск оптимального размера шрифта из v1.0.6-step1.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


SYSTEM_FONT_PRESETS: dict[str, list[str]] = {
    "WildWords": [
        "ccwildwords.ttf",
        "wildwords.ttf",
        "C:/Windows/Fonts/comicbd.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "AnimeAce": [
        "animeace2_reg.ttf",
        "animeace.ttf",
        "C:/Windows/Fonts/comicbd.ttf",
        "C:/Windows/Fonts/comic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
    "Comic": [
        "C:/Windows/Fonts/comicbd.ttf",
        "C:/Windows/Fonts/comic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "ComicItalic": [
        "C:/Windows/Fonts/comici.ttf",
        "C:/Windows/Fonts/comic.ttf",
        "C:/Windows/Fonts/segoeuii.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "Segoe": [
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "Impact": [
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/trebucbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "Serif": [
        "C:/Windows/Fonts/georgiab.ttf",
        "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    ],
}

FALLBACK_SYMBOL_FONT_CANDIDATES: list[str] = [
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/seguisb.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# Символы, которые всегда рисуются 4x-сглаженным векторным рендером (чтобы никогда не превращаться в квадратики □
# и иметь полноценную комиксную обводку stroke_width на любом шрифте, включая Comic Sans и WildWords)
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
    # Многосимвольные замены
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


class MangaRenderer:
    """Вписывает переведенный текст в очищенные бабблы и свободные области страницы манги."""

    def __init__(
        self,
        fonts_dir: str | Path = "fonts",
        max_font_size: int = 32,
        min_font_size: int = 12,
        box_padding_ratio: float = 0.05,
        line_spacing: int = 3,
        uppercase_comic_style: bool = True,
        overflow_mode: str = "shrink",
    ) -> None:
        self.fonts_dir = Path(fonts_dir)
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.box_padding_ratio = box_padding_ratio
        self.line_spacing = line_spacing
        self.uppercase_comic_style = uppercase_comic_style
        self.overflow_mode = self._normalize_overflow_mode(overflow_mode)

        self.font_path = self._resolve_font_path()
        self.fallback_font_paths: list[str] = [
            p for p in FALLBACK_SYMBOL_FONT_CANDIDATES if Path(p).exists()
        ]
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self._notdef_sig_cache: dict[int, set[tuple[tuple[int, int], bytes]]] = {}
        self._glyph_support_cache: dict[tuple[int, str], bool] = {}
        self._symbol_icon_cache: dict[tuple[str, int, tuple[int, int, int], tuple[int, int, int], int], Image.Image] = {}
        logger.info("Используемый шрифт для тайпсеттинга: %s", self.font_path or "Pillow Default")

    def _resolve_font_path(self) -> str | None:
        if self.fonts_dir.exists():
            custom_fonts = sorted(
                list(self.fonts_dir.glob("*.ttf")) + list(self.fonts_dir.glob("*.otf"))
            )
            if custom_fonts:
                return str(custom_fonts[0])

        candidates = [
            "C:/Windows/Fonts/comicbd.ttf",
            "C:/Windows/Fonts/comic.ttf",
            "C:/Windows/Fonts/seguisb.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate

        return None

    def list_available_fonts(self) -> list[dict[str, str]]:
        """Возвращает список доступных шрифтовых пресетов и пользовательских `.ttf`/`.otf` файлов из папки `fonts/`."""
        items: list[dict[str, str]] = [
            {"id": "default", "name": "✨ По умолчанию (Авто)"},
            {"id": "Comic", "name": "💬 Comic Sans Bold (Манга)"},
            {"id": "WildWords", "name": "🌸 WildWords / Комикс"},
            {"id": "AnimeAce", "name": "🎌 Anime Ace / Сканлейт"},
            {"id": "Segoe", "name": "📘 Segoe UI Semibold (Чистый)"},
            {"id": "Impact", "name": "💥 Impact / Акцидентный (Крик/SFX)"},
            {"id": "Serif", "name": "📖 Georgia / Serif (Книжный/Мысли)"},
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

        if fam and fam.lower() not in ("default", "auto", "none"):
            # 1. Проверяем файл в папке fonts/
            local_cand = self.fonts_dir / fam
            if local_cand.exists() and local_cand.is_file():
                return str(local_cand)
            if self.fonts_dir.exists():
                for f_p in list(self.fonts_dir.glob("*.ttf")) + list(self.fonts_dir.glob("*.otf")):
                    if f_p.name.lower() == fam.lower() or f_p.stem.lower() == fam.lower():
                        return str(f_p)

            # 2. Проверяем системный пресет
            for preset_key, cand_list in SYSTEM_FONT_PRESETS.items():
                if preset_key.lower() == fam.lower():
                    for cand in cand_list:
                        local_p = self.fonts_dir / cand
                        if local_p.exists():
                            return str(local_p)
                        if Path(cand).exists():
                            return cand

            # 3. Прямой путь
            if Path(fam).exists() and Path(fam).is_file():
                return str(Path(fam))

        # Авто-подбор шрифта по style_preset, если font_family не задан вручную
        if preset == "thought":
            for cand in SYSTEM_FONT_PRESETS["ComicItalic"] + SYSTEM_FONT_PRESETS["Serif"]:
                if Path(cand).exists():
                    return cand
        elif preset in ("shout", "sfx"):
            for cand in SYSTEM_FONT_PRESETS["Impact"] + SYSTEM_FONT_PRESETS["Comic"]:
                if Path(cand).exists():
                    return cand
        elif preset == "narration":
            for cand in SYSTEM_FONT_PRESETS["Segoe"] + SYSTEM_FONT_PRESETS["Serif"]:
                if Path(cand).exists():
                    return cand

        return self.font_path

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
        """
        Проверяет, отсутствует ли символ `ch` в шрифте `font` (будет ли он отрисован как квадратик `.notdef` `□`).
        Результат кэшируется в `self._glyph_support_cache`.
        """
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
        """Находит первый системный шрифт, в котором реально присутствует глиф для символа `ch` (без квадратика)."""
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
        """
        Рендерит 4x-сглаженную векторную иконку для символов манги (`♥`, `♡`, `★`, `☆`, `✦`, `♪`, `♫`, `◆`, `◇`)
        с точным цветом текста `text_fill` и контурной обводкой `stroke_fill` (`stroke_w`).
        """
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
        """
        Рендерит многострочный блок текста с векторными спецсимволами (`♥`, `♡`, `★`, `☆`, `✦`, `♪`, `♫`)
        и fallback-глифами на прозрачном `RGBA`-слое.
        """
        fs = self._get_font_pixel_size(font)
        dummy_img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        dummy_draw = ImageDraw.Draw(dummy_img)

        lines = wrapped_text.split("\n")
        line_widths = [self._measure_line_width(dummy_draw, ln, font) for ln in lines]
        max_line_w = max(line_widths, default=float(fs))

        ref_bbox = dummy_draw.textbbox((0, 0), "AyБр", font=font, stroke_width=stroke_w)
        line_h = max(int(round(fs * 1.12)), (ref_bbox[3] - ref_bbox[1]) - stroke_w)
        total_h = max(line_h, len(lines) * line_h + max(0, len(lines) - 1) * spacing)

        pad = max(12, stroke_w * 2 + 8)
        layer_w = max(16, int(math.ceil(max_line_w + pad * 2)))
        layer_h = max(16, int(math.ceil(total_h + pad * 2)))
        layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
        l_draw = ImageDraw.Draw(layer)

        sym_advance = float(max(8, int(round(fs * 0.88))))

        for idx, (ln, lw) in enumerate(zip(lines, line_widths)):
            if not ln:
                continue
            if align == "left":
                cur_x = float(pad)
            elif align == "right":
                cur_x = float(pad) + (max_line_w - lw)
            else:
                cur_x = float(pad) + (max_line_w - lw) / 2.0

            cur_y = float(pad + idx * (line_h + spacing))

            # Разбиваем строку на сегменты обычного текста, векторных символов и fallback-символов
            run_buf = ""

            def _flush_run(x_pos: float, text_run: str) -> float:
                if not text_run:
                    return x_pos
                tb = l_draw.textbbox((0, 0), text_run, font=font)
                l_draw.text(
                    (x_pos, cur_y - tb[1] * 0.15),
                    text_run,
                    font=font,
                    fill=(*text_fill, 255),
                    stroke_width=stroke_w,
                    stroke_fill=(*stroke_fill, 255) if stroke_w > 0 else None,
                )
                return x_pos + float(l_draw.textlength(text_run, font=font))

            for ch in ln:
                if ch in VECTOR_MANGA_SYMBOLS:
                    cur_x = _flush_run(cur_x, run_buf)
                    run_buf = ""
                    icon = self._get_vector_symbol_icon(ch, fs, text_fill, stroke_fill, stroke_w)
                    iw, ih = icon.size
                    ix = int(round(cur_x + (sym_advance - iw) / 2.0))
                    iy = int(round(cur_y + (line_h - ih) / 2.0))
                    layer.alpha_composite(icon, dest=(max(0, ix), max(0, iy)))
                    l_draw = ImageDraw.Draw(layer)
                    cur_x += sym_advance
                elif self._is_glyph_missing(font, ch):
                    cur_x = _flush_run(cur_x, run_buf)
                    run_buf = ""
                    fb_font = self._get_fallback_font_for_char(ch, fs)
                    if fb_font is not None:
                        l_draw.text(
                            (cur_x, cur_y),
                            ch,
                            font=fb_font,
                            fill=(*text_fill, 255),
                            stroke_width=stroke_w,
                            stroke_fill=(*stroke_fill, 255) if stroke_w > 0 else None,
                        )
                        cur_x += float(l_draw.textlength(ch, font=fb_font))
                else:
                    run_buf += ch

            cur_x = _flush_run(cur_x, run_buf)

        alpha_bbox = layer.getbbox()
        if alpha_bbox is not None:
            return layer.crop(alpha_bbox)
        return layer

    @staticmethod
    def _get_line_spacing(font_size: int, style_preset: str = "dialogue") -> int:
        preset = (style_preset or "dialogue").lower().strip()
        if preset == "thought":
            return max(3, int(round(font_size * 0.22)))
        if preset == "shout":
            return max(2, int(round(font_size * 0.12)))
        return max(2, int(round(font_size * 0.15)))

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

    # Гласные и спецбуквы для правильного слогового и морфемного переноса (Русский + Английский + Ромадзи)
    _VOWELS_RU = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
    _VOWELS_EN = set("aeiouyAEIOUY")
    _VOWELS_ALL = _VOWELS_RU | _VOWELS_EN
    _NO_START_NEXT_RU = set("йъьыЙЪЬЫ")
    _MODIFIERS_RU = set("ъьЪЬ")

    # Английские неделимые буквосочетания (диграфы гласных и согласных), которые нельзя разрывать посередине
    _EN_UNBREAKABLE_DIGRAPHS = {
        "aa", "ae", "ai", "ao", "au", "ay", "ea", "ee", "ei", "eo", "eu", "ey",
        "ia", "ie", "io", "iu", "oa", "oe", "oi", "oo", "ou", "oy", "ua", "ue", "ui", "uo",
        "sh", "ch", "th", "ph", "wh", "gh", "ck", "ng", "qu", "gu",
    }
    # Английские приставки и суффиксы для естественного морфемного переноса длинных слов
    _EN_PREFIXES = ("under", "inter", "trans", "super", "hyper", "over", "anti", "auto", "semi", "non", "dis", "mis", "pre", "pro", "sub", "out", "un", "re", "in", "im", "il", "ir", "de", "ex")
    _EN_SUFFIXES = ("ingly", "ement", "ation", "ition", "ously", "ively", "fully", "lessly", "ness", "ment", "tion", "sion", "able", "ible", "ably", "ibly", "less", "ship", "ward", "wise", "ing", "ful", "ous", "ive", "est")

    # Однобуквенные предлоги, союзы, артикли и местоимения (RU + EN), которые привязываются к следующему слову
    _SINGLE_LETTER_BIND_NEXT = {"в", "к", "с", "у", "о", "а", "и", "a", "i"}
    # Короткие предлоги, артикли, союзы и вспомогательные слова (RU + EN), которые не должны «висеть» на конце строки
    _HANGING_SHORT_WORDS = {
        "в", "к", "с", "у", "о", "а", "и", "я",
        "не", "ни", "на", "но", "по", "за", "из", "от", "до", "ко", "со", "во", "об", "про", "без", "для", "под", "над", "при",
        "i", "a", "an", "the", "to", "in", "on", "of", "at", "by", "or", "so", "no", "my", "we", "he", "it", "is", "be", "do", "as", "if", "up", "us", "am", "for", "and", "but", "not", "his", "her", "our", "its", "who",
    }
    # Частицы и символы, которые нельзя отрывать от предыдущего слова на новую строку
    _ENCLITIC_PARTICLES = {"бы", "б", "ли", "ль", "же", "ж", "то", "ка", "—", "-", "–", "!", "?", "?!", "!?", "...", "…", "~", "♥", "♡", "★", "☆", "♪", "♫"}

    @classmethod
    def _find_syllable_split_indices(cls, core_word: str, min_part_len: int = 4) -> list[int]:
        """
        Возвращает список допустимых индексов разреза `i` (`core_word[:i] + "-"`, `core_word[i:]`)
        по правилам как русской, так и английской орфографии/типографики:
        - Переносятся только сверхдлинные слова (от 12+ букв), при этом минимальная длина обеих частей —
          не менее `min_part_len` букв (минимум 4 буквы до и после дефиса, никаких 1–3 букв!).
        - В английских словах никогда не рвутся апострофные сокращения (`don't`, `everything's`),
          гласные и согласные диграфы (`ea`, `ie`, `ou`, `oo`, `sh`, `ch`, `th`, `ph`, `ck`, `qu`),
          и учитываются морфемные границы приставок/суффиксов (`un-`, `under-`, `-ing`, `-tion`, `-ably`, `-ment`).
        - В русских словах соблюдаются правила слогораздела (Грамота.ру: запрет отрыва `й`, `ъ`, `ь`, `ы`,
          разделение двойных согласных посередине).
        """
        if "'" in core_word or "’" in core_word:
            return []

        n = len(core_word)
        if n < max(13, min_part_len * 2 + 2):
            return []

        low_word = core_word.lower()
        is_latin = bool(re.search(r"[a-zA-Z]", core_word))

        # Для английских слов сначала проверяем естественные морфемные границы (суффиксы и длинные приставки)
        morpheme_indices: set[int] = set()
        if is_latin:
            for pfx in cls._EN_PREFIXES:
                if low_word.startswith(pfx) and min_part_len <= len(pfx) <= n - min_part_len:
                    morpheme_indices.add(len(pfx))
            for sfx in cls._EN_SUFFIXES:
                idx_s = n - len(sfx)
                if low_word.endswith(sfx) and min_part_len <= idx_s <= n - min_part_len:
                    # Если перед суффиксом удвоенная согласная (напр. STOP-PING, RUN-NING), разрез между ними
                    morpheme_indices.add(idx_s)

        valid_splits: list[int] = []
        for i in range(min_part_len, n - min_part_len + 1):
            left = core_word[:i]
            right = core_word[i:]
            left_low = low_word[:i]
            right_low = low_word[i:]

            # В обеих частях обязательно должна быть хотя бы одна настоящая гласная
            if not any(ch in cls._VOWELS_ALL for ch in left):
                continue
            if not any(ch in cls._VOWELS_ALL for ch in right):
                continue
            # В английском немое конечное 'e' (напр. -ve, -ce, -le, -se, -te) не образует слог само по себе
            if is_latin and sum(1 for ch in right_low if ch in cls._VOWELS_EN) == 1 and right_low.endswith("e"):
                continue

            if right[0] in cls._NO_START_NEXT_RU:
                continue

            # Для английского не разрываем неделимые диграфы (ea, ie, ou, oo, th, sh, ch, ph, ck, ng, qu)
            if is_latin:
                boundary_pair = low_word[i - 1 : i + 1]
                if boundary_pair in cls._EN_UNBREAKABLE_DIGRAPHS:
                    continue
                if i in morpheme_indices:
                    valid_splits.append(i)
                    continue

            # Нельзя разбивать удвоенную согласную иначе как посередине
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

            # Не отрываем согласную от следующей гласной (разрез должен быть либо Г-СВ, либо С-С)
            if not is_prev_vowel and is_next_vowel and prev_ch not in cls._MODIFIERS_RU:
                continue

            valid_splits.append(i)

        return valid_splits

    def _split_long_word(
        self,
        draw: ImageDraw.ImageDraw,
        word: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
    ) -> list[str]:
        """
        Аккуратно переносит только сверхдлинные русские и английские слова (от 12+ букв) строго по слогам/морфемам
        и максимум на 2 части, гарантируя минимум 4 буквы до и после дефиса.
        """
        if self._measure_line_width(draw, word, font) <= max_width:
            return [word]

        # Если в слове уже есть дефис (например, «КТО-НИБУДЬ», «SELF-CONTROL», «WELL-KNOWN»), бьем по нему
        if "-" in word and not word.startswith("-") and not word.endswith("-"):
            hyph_parts = word.split("-")
            if len(hyph_parts) == 2 and len(hyph_parts[0]) >= 3 and len(hyph_parts[1]) >= 3:
                p1 = hyph_parts[0] + "-"
                p2 = hyph_parts[1]
                if self._measure_line_width(draw, p1, font) <= max_width * 1.25:
                    return [p1, p2]

        # Не разбиваем английские слова с апострофами (DON'T, EVERYTHING'S, SHOULDN'T)
        if "'" in word or "’" in word:
            return [word]

        m = re.match(r"^([^A-Za-zА-Яа-яЁё0-9]*)([A-Za-zА-Яа-яЁё0-9]+)(.*)$", word)
        if not m:
            return [word]

        lead_punct, core, trail_punct = m.group(1), m.group(2), m.group(3)
        # Слова короче 12 букв (и русские, и английские) НИКОГДА не рвем дефисами!
        if len(core) < 13:
            return [word]

        split_candidates = self._find_syllable_split_indices(core, min_part_len=4)
        if not split_candidates:
            return [word]

        mid = len(core) / 2.0
        fitting_splits: list[tuple[float, int]] = []
        for idx in split_candidates:
            part1 = f"{lead_punct}{core[:idx]}-"
            part2 = f"{core[idx:]}{trail_punct}"
            w1 = self._measure_line_width(draw, part1, font)
            w2 = self._measure_line_width(draw, part2, font)
            if w1 <= max_width * 1.22 and w2 <= max_width * 1.25:
                fitting_splits.append((abs(idx - mid), idx))

        if fitting_splits:
            fitting_splits.sort(key=lambda item: item[0])
            best_idx = fitting_splits[0][1]
            return [f"{lead_punct}{core[:best_idx]}-", f"{core[best_idx:]}{trail_punct}"]

        return [word]

    @classmethod
    def _preprocess_tokens_no_orphans(cls, raw_words: list[str]) -> list[str]:
        """
        1. Присоединяет одиночные знаки препинания и частицы (`бы`, `ли`, `же`, `—`, `♥`) к предыдущему слову.
        2. Присоединяет однобуквенные предлоги, союзы, артикли и местоимения (`в`, `к`, `с`, `у`, `о`, `а`, `и`, `a`, `i`)
           к следующему слову (если следующее слово не сверхдлинное <= 8 букв), чтобы они никогда не оставались
           в одиночку на строке и не висели оторванными в конце строки.
        """
        if not raw_words:
            return []
        step1: list[str] = []
        for tok in raw_words:
            clean_low = re.sub(r"[^a-zA-Zа-яА-ЯёЁ]", "", tok).lower()
            if step1 and (tok in cls._ENCLITIC_PARTICLES or (clean_low in {"бы", "б", "ли", "ль", "же", "ж"} and len(clean_low) <= 2)):
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
                and clean_low in cls._SINGLE_LETTER_BIND_NEXT
                and len(clean_low) == 1
            ):
                next_clean = re.sub(r"[^a-zA-Zа-яА-ЯёЁ]", "", step1[i + 1])
                if len(next_clean) <= 8:
                    merged.append(f"{tok} {step1[i + 1]}")
                    i += 2
                    continue
            merged.append(tok)
            i += 1
        return merged

    @classmethod
    def _is_hanging_short_word(cls, token: str) -> bool:
        """Проверяет, является ли последнее слово строки коротким предлогом/артиклем/союзом (RU + EN), который плохо смотрится в конце строки."""
        last_piece = token.rsplit(" ", 1)[-1]
        if last_piece and last_piece[-1] in ".,!?;:…»)":
            return False
        clean = re.sub(r"^[^a-zA-Zа-яА-ЯёЁ']+|[^a-zA-Zа-яА-ЯёЁ']+$", "", last_piece).lower()
        return clean in cls._HANGING_SHORT_WORDS

    def _wrap_text_pixel(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
        allow_hyphenation: bool = False,
        elliptical_shaping: bool = True,
    ) -> str:
        raw_words = text.split()
        if not raw_words:
            return ""

        tokens = self._preprocess_tokens_no_orphans(raw_words)
        if not tokens:
            return ""

        edge_width = max(16, int(max_width * 0.85)) if elliptical_shaping else max_width

        # Если вся фраза короткая (1–2 токена) и помещается на 1 строку — не разбиваем
        full_joined = " ".join(tokens)
        if len(tokens) <= 2 and self._measure_line_width(draw, full_joined, font) <= max_width * 1.08:
            return full_joined

        words: list[str] = []
        for w in tokens:
            w_len = self._measure_line_width(draw, w, font)
            if allow_hyphenation and w_len > max_width * 1.15 and " " not in w:
                words.extend(self._split_long_word(draw, w, font, max_width))
            else:
                words.append(w)

        n = len(words)
        if n == 1:
            return words[0]

        word_widths = [self._measure_line_width(draw, w, font) for w in words]
        space_w = max(2.0, self._measure_line_width(draw, " ", font))

        def _line_width(i: int, j: int) -> float:
            return sum(word_widths[i:j]) + space_w * max(0, j - i - 1)

        total_text_w = sum(word_widths) + space_w * max(0, n - 1)
        max_single_w = max(word_widths, default=float(max_width))
        eff_max_w = max(float(max_width), max_single_w)

        est_lines = max(1, int(math.ceil(total_text_w / max(1.0, eff_max_w * (0.84 if elliptical_shaping else 0.92)))))
        if elliptical_shaping and total_text_w > edge_width and est_lines == 1:
            est_lines = 2
        ideal_line_w = min(eff_max_w * 0.92, max(eff_max_w * 0.58, total_text_w / max(1, est_lines)))

        INF = 1e18
        dp_cost = [INF] * (n + 1)
        dp_next = [n] * (n + 1)
        dp_cost[n] = 0.0

        for i in range(n - 1, -1, -1):
            is_first = (i == 0)
            line_cap = (eff_max_w * 0.88 if (elliptical_shaping and is_first and n >= 3) else eff_max_w)
            for j in range(i + 1, n + 1):
                lw = _line_width(i, j)
                if j > i + 1 and lw > eff_max_w * 1.08:
                    break

                is_last = (j == n)
                cost = 0.0

                if elliptical_shaping and (is_first or is_last) and n >= 3 and lw > line_cap:
                    over = lw - line_cap
                    cost += (over * over) * 3.0

                target_for_line = ideal_line_w * (0.86 if (elliptical_shaping and (is_first or is_last) and n >= 3) else 1.0)
                diff = target_for_line - lw
                if is_last:
                    if j - i == 1 and n >= 3 and lw < ideal_line_w * 0.42:
                        cost += (ideal_line_w * 0.55 - lw) ** 2 * 1.8
                    else:
                        cost += (diff * diff) * 0.35
                else:
                    cost += (diff * diff) * 1.0

                if not is_last and self._is_hanging_short_word(words[j - 1]):
                    cost += (eff_max_w * 0.85) ** 2

                # Дополнительный штраф, если на строке остается одно-единственное однобуквенное слово («В», «А», «I», «A»)
                if j - i == 1 and len(re.sub(r"[^a-zA-Zа-яА-ЯёЁ]", "", words[i])) == 1:
                    cost += (eff_max_w * 1.5) ** 2

                if j > i + 1:
                    for k in range(i, j - 1):
                        if words[k].endswith("-"):
                            cost += (eff_max_w * 2.0) ** 2

                cost += (eff_max_w * 0.28) ** 2

                total_c = cost + dp_cost[j]
                if total_c < dp_cost[i]:
                    dp_cost[i] = total_c
                    dp_next[i] = j

        lines: list[str] = []
        curr = 0
        while curr < n:
            nxt = dp_next[curr]
            if nxt <= curr:
                nxt = curr + 1
            line_tokens = words[curr:nxt]
            lines.append(" ".join(line_tokens))
            curr = nxt

        return "\n".join(lines)

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
    ) -> bool:
        lines = wrapped_text.split("\n")
        if self._needs_special_rendering(font, wrapped_text):
            fs = self._get_font_pixel_size(font)
            line_widths = [self._measure_line_width(draw, ln, font) + 2 * stroke_w for ln in lines]
            total_w = max(line_widths, default=0.0)
            line_h = max(int(round(fs * 1.12)), fs + stroke_w)
            total_h = len(lines) * line_h + max(0, len(lines) - 1) * spacing + 2 * stroke_w
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

        if not elliptical:
            return True

        if len(lines) <= 1:
            return True

        rx = (allowed_w / 2.0) * 1.14
        ry = (target_h / 2.0) * 1.16
        if rx <= 1.0 or ry <= 1.0:
            return True

        line_h_avg = total_h / max(1, len(lines))
        top_y = -total_h / 2.0

        for idx, line in enumerate(lines):
            cy = top_y + (idx + 0.5) * line_h_avg
            norm_y = min(0.92, abs(cy) / ry)
            chord_w = 2.0 * rx * math.sqrt(max(0.24, 1.0 - (norm_y**2)))
            line_w = self._measure_line_width(draw, line, font) + 2 * stroke_w
            if line.strip().count(" ") <= 1:
                if line_w > allowed_w:
                    return False
            elif line_w > max(chord_w, allowed_w * 0.85):
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
            ratio = 0.45 if (is_free_text or preset in ("shout", "sfx")) else 0.38
            start_size = min(
                max_lim,
                max(min_lim, int(min(target_w, target_h) * ratio)),
            )

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
            spacing = self._get_line_spacing(font_size, style_preset=preset)
            eff_wrap_w = max(12, int(round(target_w * width_slack)))
            if max_expand_w and eff_wrap_w > max_expand_w:
                eff_wrap_w = max(target_w, max_expand_w)
            eff_tol = eff_wrap_w / max(1.0, float(target_w))
            wrapped = self._wrap_text_pixel(
                draw,
                text,
                font,
                eff_wrap_w,
                allow_hyphenation=allow_hyphenation,
                elliptical_shaping=use_ellipse,
            )
            fits = self._fits_in_box(
                draw,
                wrapped,
                font,
                spacing,
                stroke_w,
                target_w,
                target_h,
                elliptical=use_ellipse,
                align=align,
                width_tolerance_ratio=eff_tol,
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

        # Проход 1: бинарный поиск без переносов слов (allow_hyphenation=False, строго в пределах баббла)
        best_no_hyph = _binary_search_best_fit(min_lim, start_size, allow_hyphenation=False, width_slack=1.08)
        if best_no_hyph is not None:
            return (*best_no_hyph, True)

        # Проход 2: разрешаем легкий выход длинных целых слов за поля узкого баббла по горизонтали (до +50% ширины),
        # чтобы слова до 12 букв (и русские, и английские: «UNBELIEVABLY», «ПОЖАЛУЙСТА!») вообще не дробились дефисами!
        slack_cap = max(min_lim, min(start_size, max(24 if unconstrained_size else 19, min_lim)))
        best_slack_no_hyph = _binary_search_best_fit(min_lim, slack_cap, allow_hyphenation=False, width_slack=1.58)
        if best_slack_no_hyph is not None:
            return (*best_slack_no_hyph, True)

        # Проход 3: аккуратный слоговой/морфемный перенос только для сверхдлинных слов (от 12+ букв, минимум по 4 буквы с каждой стороны)
        if overflow_mode == "shrink":
            hyph_start = max(min_lim, min(start_size, max(24 if unconstrained_size else 18, min_lim)))
            best_with_hyph = _binary_search_best_fit(min_lim, hyph_start, allow_hyphenation=True, width_slack=1.35)
            if best_with_hyph is not None:
                return (*best_with_hyph, True)

        min_font = self._get_font(min_lim, font_family_path=font_family_path)
        min_spacing = self._get_line_spacing(min_lim, style_preset=preset)
        min_stroke = self._compute_stroke_width(
            font_size=min_lim,
            is_free_text=is_free_text,
            style_preset=preset,
            stroke_width_override=stroke_width_override,
        )
        if overflow_mode in ("overflow", "expand_bubble"):
            # Рассчитываем комфортную ширину переноса строк, чтобы длинный текст не превращался в узкую «колбасу»
            full_line_px = max(1.0, self._measure_line_width(draw, text, min_font))
            line_h_est = max(min_lim + 2, int(min_lim * 1.18))
            ideal_w = int(math.sqrt(full_line_px * line_h_est * 1.35))
            upper_w = max_expand_w if (max_expand_w and max_expand_w > target_w) else max(target_w * 2, 260)
            wrap_limit = max(target_w, min(upper_w, max(ideal_w, int(target_w * 1.45), int(min_lim * 7.5))))
            wrapped = self._wrap_text_pixel(
                draw,
                text,
                min_font,
                wrap_limit,
                allow_hyphenation=False,
                elliptical_shaping=False,
            )
            return wrapped, min_font, min_stroke, min_spacing, min_lim, False

        wrap_limit = max(int(target_w * 1.58), int(min_lim * 6.8))
        if max_expand_w and wrap_limit > max_expand_w:
            wrap_limit = max(target_w, max_expand_w)
        wrapped = self._wrap_text_pixel(
            draw,
            text,
            min_font,
            wrap_limit,
            allow_hyphenation=False,
            elliptical_shaping=False,
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

        for block in blocks:
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

            text = raw_text.upper() if use_upper else raw_text
            is_free_text = bool(block.get("is_free_text", False)) or (style_preset == "sfx")
            is_modified = self._is_bbox_modified(block)
            is_resized = self._is_bbox_resized(block)
            force_rect = is_modified or (style_preset in ("narration", "sfx"))

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

            if is_free_text or force_rect:
                pad_x = max(2, int(box_w * 0.04)) if force_rect else 2
                pad_y = max(2, int(box_h * 0.04)) if force_rect else 2
            else:
                pad_x = max(4, int(box_w * self.box_padding_ratio))
                pad_y = max(4, int(box_h * self.box_padding_ratio))

            safe_w = max(10, box_w - 2 * pad_x)
            safe_h = max(10, box_h - 2 * pad_y)

            block_font_override = block.get("font_size")
            unconstrained_size = False
            if isinstance(block_font_override, (int, float)) and int(block_font_override) >= 6:
                b_min = b_max = max(6, min(160, int(block_font_override)))
            elif is_resized:
                b_min, b_max = 6, 140
                unconstrained_size = True
            elif style_preset == "shout":
                b_min = font_min
                b_max = min(140, max(font_cap + 6, int(font_cap * 1.22)))
            elif style_preset == "whisper":
                b_min = max(8, font_min - 2)
                b_max = max(b_min + 2, int(font_cap * 0.82))
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
            max_exp_w = max(safe_w, min(img_w - 16, max(int(box_w * 2.2), 280)))

            wrapped_text, font, stroke_w, dyn_spacing, chosen_fs, fit_ok = self._fit_text_to_box(
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
            )
            block["rendered_font_size"] = chosen_fs

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
                    # По умолчанию для бабблов манги подложка чисто-белая (#FFFFFF), если фон не явно темный
                    bg_fill = (255, 255, 255) if median_lum >= 85.0 else (18, 18, 18)
                bg_lum = 0.299 * bg_fill[0] + 0.587 * bg_fill[1] + 0.114 * bg_fill[2]
                effective_lum = bg_lum
            else:
                bg_fill = (255, 255, 255)
                effective_lum = median_lum

            if effective_lum >= 115.0:
                default_text_fill = (34, 34, 34) if style_preset == "thought" else (15, 15, 15)
                default_stroke_fill = (255, 255, 255)
            else:
                default_text_fill = (250, 250, 250)
                default_stroke_fill = (10, 10, 10)

            custom_text_rgb = self._parse_hex_color(block.get("text_color"))
            custom_stroke_rgb = self._parse_hex_color(block.get("stroke_color"))
            text_fill = custom_text_rgb if custom_text_rgb is not None else default_text_fill
            stroke_fill = custom_stroke_rgb if custom_stroke_rgb is not None else default_stroke_fill

            # Первичный замер габаритов текста для проверки выхода за рамки баббла
            prelim_bbox = draw.multiline_textbbox(
                (0, 0),
                wrapped_text,
                font=font,
                spacing=dyn_spacing,
                align=align,
                stroke_width=stroke_w,
            )
            prelim_tw = prelim_bbox[2] - prelim_bbox[0]
            prelim_th = prelim_bbox[3] - prelim_bbox[1]
            is_overflowing_box = (not fit_ok) or (prelim_tw > safe_w + 2) or (prelim_th > safe_h + 2)
            block["is_overflowing"] = bool(is_overflowing_box)

            # Вариант Б (overflow): если текст выходит за рамки баббла и не задана ручная обводка,
            # усиливаем контрастную обводку (минимум 4px), чтобы буквы идеально читались поверх контура баббла!
            if is_overflowing_box and eff_stroke_override is None:
                stroke_w = max(stroke_w, 4 if block_overflow_mode == "overflow" else 3)

            needs_rich_symbols = self._needs_special_rendering(font, wrapped_text)
            rich_layer: Image.Image | None = None
            if needs_rich_symbols:
                rich_layer = self._render_rich_text_layer(
                    wrapped_text=wrapped_text,
                    font=font,
                    spacing=dyn_spacing,
                    align=align,
                    text_fill=text_fill,
                    stroke_fill=stroke_fill,
                    stroke_w=stroke_w,
                )
                tw, th = rich_layer.size
                t_bbox = (0, 0, tw, th)
            else:
                t_bbox = draw.multiline_textbbox(
                    (0, 0),
                    wrapped_text,
                    font=font,
                    spacing=dyn_spacing,
                    align=align,
                    stroke_width=stroke_w,
                )
                tw = t_bbox[2] - t_bbox[0]
                th = t_bbox[3] - t_bbox[1]

            # Вариант В (expand_bubble): если текст не помещается в исходный баббл (и это не свободный SFX на фоне),
            # автоматически перерисовываем/расширяем капсулу баббла с аккуратным контуром под размер текста!
            if block_overflow_mode == "expand_bubble" and is_overflowing_box and not is_free_text and bg_style == "none":
                cx_box = (rx1 + rx2) / 2.0
                cy_box = (y1 + y2) / 2.0
                pad_exp_x = max(10, int(chosen_fs * 0.65) + bg_pad_adj)
                pad_exp_y = max(8, int(chosen_fs * 0.55) + bg_pad_adj)
                bx1 = max(2, int(round(min(rx1, cx_box - tw / 2.0 - pad_exp_x))))
                bx2 = min(img_w - 2, int(round(max(rx2, cx_box + tw / 2.0 + pad_exp_x))))
                by1 = max(2, int(round(min(y1, cy_box - th / 2.0 - pad_exp_y))))
                by2 = min(img_h - 2, int(round(max(y2, cy_box + th / 2.0 + pad_exp_y))))
                rad = max(12, int(min(bx2 - bx1, by2 - by1) * (0.18 if style_preset == "narration" else 0.44)))
                border_col = (25, 25, 25) if bg_fill[0] >= 128 else (230, 230, 230)
                draw.rounded_rectangle(
                    [bx1, by1, bx2, by2],
                    radius=rad,
                    fill=bg_fill,
                    outline=border_col,
                    width=2,
                )

            # Отрисовка подложки для режимов "bubble" (Овал баббла) и "full_box" (Вся рамка)
            if bg_style == "bubble":
                in_x = max(1, int(box_w * 0.09) - bg_pad_adj)
                in_y = max(1, int(box_h * 0.05) - bg_pad_adj)
                bx1 = max(0, min(img_w - 2, rx1 + in_x))
                by1 = max(0, min(img_h - 2, y1 + in_y))
                bx2 = max(bx1 + 4, min(img_w, rx2 - in_x))
                by2 = max(by1 + 4, min(img_h, y2 - in_y))
                cx_box = (rx1 + rx2) / 2.0
                cy_box = (y1 + y2) / 2.0
                bx1 = max(0, min(bx1, int(round(cx_box - tw / 2.0 - 4))))
                bx2 = min(img_w, max(bx2, int(round(cx_box + tw / 2.0 + 4))))
                by1 = max(0, min(by1, int(round(cy_box - th / 2.0 - 4))))
                by2 = min(img_h, max(by2, int(round(cy_box + th / 2.0 + 4))))
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
                    txt_draw.multiline_text(
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
                    paste_x = int(round(rx1 + pad_x - pad_rot))
                elif align == "right" and abs(angle) < 0.5:
                    paste_x = int(round(rx2 - pad_x - tw - pad_rot))
                else:
                    paste_x = int(round(rx1 + (box_w - rw_rot) / 2.0))
                paste_y = int(round(y1 + (box_h - rh_rot) / 2.0))
                paste_x = max(-rw_rot // 2, min(paste_x, img_w - rw_rot // 2))
                paste_y = max(-rh_rot // 2, min(paste_y, img_h - rh_rot // 2))
                canvas.paste(rot_layer, (paste_x, paste_y), rot_layer)
                draw = ImageDraw.Draw(canvas)
            else:
                if align == "left":
                    draw_x = rx1 + pad_x - t_bbox[0]
                elif align == "right":
                    draw_x = rx2 - pad_x - tw - t_bbox[0]
                else:
                    draw_x = rx1 + (box_w - tw) / 2.0 - t_bbox[0]
                draw_y = y1 + (box_h - th) / 2.0 - t_bbox[1]

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

                draw.multiline_text(
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