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
        box_padding_ratio: float = 0.10,
        line_spacing: int = 3,
        uppercase_comic_style: bool = True,
    ) -> None:
        self.fonts_dir = Path(fonts_dir)
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.box_padding_ratio = box_padding_ratio
        self.line_spacing = line_spacing
        self.uppercase_comic_style = uppercase_comic_style

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
    def _normalize_bg_style(raw_style: Any) -> str:
        s = str(raw_style or "none").strip().lower()
        if s in ("text_box", "pill", "box", "text", "lines"):
            return "text_box"
        if s in ("bubble", "ellipse", "oval", "capsule"):
            return "bubble"
        if s in ("full_box", "rect", "frame", "full"):
            return "full_box"
        return "none"

    def _split_long_word(
        self,
        draw: ImageDraw.ImageDraw,
        word: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
    ) -> list[str]:
        if self._measure_line_width(draw, word, font) <= max_width or len(word) <= 4:
            return [word]

        parts: list[str] = []
        current = ""
        for idx, ch in enumerate(word):
            candidate = current + ch
            is_last = idx == len(word) - 1
            test_str = candidate if is_last else candidate + "-"
            if self._measure_line_width(draw, test_str, font) <= max_width or len(current) < 2:
                current = candidate
            else:
                parts.append(current + "-")
                current = ch
        if current:
            parts.append(current)
        return parts

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

        edge_width = max(12, int(max_width * 0.78)) if elliptical_shaping else max_width

        if len(raw_words) <= 3 and self._measure_line_width(draw, text, font) <= edge_width:
            return text

        words: list[str] = []
        for w in raw_words:
            w_len = self._measure_line_width(draw, w, font)
            if allow_hyphenation and w_len > edge_width:
                words.extend(self._split_long_word(draw, w, font, edge_width))
            elif allow_hyphenation and w_len > max_width:
                words.extend(self._split_long_word(draw, w, font, max_width))
            else:
                words.append(w)

        lines: list[str] = []
        current_line = words[0]

        for word in words[1:]:
            line_limit = edge_width if (elliptical_shaping and len(lines) == 0) else max_width
            test_line = f"{current_line} {word}"
            if self._measure_line_width(draw, test_line, font) <= line_limit:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)

        if elliptical_shaping and len(lines) >= 3:
            last_line = lines[-1]
            if self._measure_line_width(draw, last_line, font) > edge_width:
                last_words = last_line.split()
                if len(last_words) >= 2:
                    sub_lines: list[str] = []
                    cur = last_words[0]
                    for lw in last_words[1:]:
                        cand = f"{cur} {lw}"
                        if self._measure_line_width(draw, cand, font) <= edge_width:
                            cur = cand
                        else:
                            sub_lines.append(cur)
                            cur = lw
                    sub_lines.append(cur)
                    lines = lines[:-1] + sub_lines

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

        if total_w > target_w or total_h > target_h:
            return False

        if not elliptical:
            return True

        if len(lines) <= 1:
            return True

        rx = (target_w / 2.0) * 1.08
        ry = (target_h / 2.0) * 1.12
        if rx <= 1.0 or ry <= 1.0:
            return True

        line_h_avg = total_h / max(1, len(lines))
        top_y = -total_h / 2.0

        for idx, line in enumerate(lines):
            cy = top_y + (idx + 0.5) * line_h_avg
            norm_y = min(0.95, abs(cy) / ry)
            chord_w = 2.0 * rx * math.sqrt(max(0.15, 1.0 - (norm_y**2)))
            line_w = self._measure_line_width(draw, line, font) + 2 * stroke_w
            if line_w > chord_w:
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
    ) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int, int]:
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
        ) -> tuple[bool, str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int]:
            font = self._get_font(font_size, font_family_path=font_family_path)
            stroke_w = self._compute_stroke_width(
                font_size=font_size,
                is_free_text=is_free_text,
                style_preset=preset,
                stroke_width_override=stroke_width_override,
            )
            spacing = self._get_line_spacing(font_size, style_preset=preset)
            wrapped = self._wrap_text_pixel(
                draw,
                text,
                font,
                target_w,
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
            )
            return fits, wrapped, font, stroke_w, spacing

        def _binary_search_best_fit(
            low_bound: int,
            high_bound: int,
            allow_hyphenation: bool,
        ) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int, int] | None:
            if high_bound < low_bound:
                return None

            fits_hi, wrap_hi, font_hi, stroke_hi, space_hi = _evaluate_font_size(high_bound, allow_hyphenation)
            if fits_hi:
                return wrap_hi, font_hi, stroke_hi, space_hi, high_bound

            low = low_bound
            high = high_bound - 1
            best: tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int, int] | None = None

            while low <= high:
                mid = (low + high) // 2
                fits, wrapped, font, stroke_w, spacing = _evaluate_font_size(mid, allow_hyphenation)
                if fits:
                    best = (wrapped, font, stroke_w, spacing, mid)
                    low = mid + 1
                else:
                    high = mid - 1

            if best is not None:
                best_fs = best[4]
                for probe_fs in range(min(high_bound - 1, best_fs + 2), best_fs, -1):
                    p_fits, p_wrap, p_font, p_stroke, p_space = _evaluate_font_size(probe_fs, allow_hyphenation)
                    if p_fits:
                        return p_wrap, p_font, p_stroke, p_space, probe_fs
            return best

        # Проход 1: бинарный поиск без переносов слов (allow_hyphenation=False)
        best_no_hyph = _binary_search_best_fit(min_lim, start_size, allow_hyphenation=False)
        if best_no_hyph is not None:
            return best_no_hyph

        # Проход 2: бинарный поиск с разрешенными переносами длинных слов (allow_hyphenation=True)
        hyph_start = max(min_lim, min(start_size, max(28 if unconstrained_size else 22, min_lim)))
        best_with_hyph = _binary_search_best_fit(min_lim, hyph_start, allow_hyphenation=True)
        if best_with_hyph is not None:
            return best_with_hyph

        min_font = self._get_font(min_lim, font_family_path=font_family_path)
        min_spacing = self._get_line_spacing(min_lim, style_preset=preset)
        min_stroke = self._compute_stroke_width(
            font_size=min_lim,
            is_free_text=is_free_text,
            style_preset=preset,
            stroke_width_override=stroke_width_override,
        )
        wrap_limit = max(target_w, int(min_lim * 2.5)) if min_lim == max_lim else target_w
        wrapped = self._wrap_text_pixel(
            draw,
            text,
            min_font,
            wrap_limit,
            allow_hyphenation=(min_lim != max_lim),
            elliptical_shaping=False,
        )
        return wrapped, min_font, min_stroke, min_spacing, min_lim

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

            wrapped_text, font, stroke_w, dyn_spacing, chosen_fs = self._fit_text_to_box(
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