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
4. Сохранен бинарный поиск оптимального размера шрифта из v1.0.6-step1.
"""

from __future__ import annotations

import logging
import math
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
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
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
    def _split_long_word(
        draw: ImageDraw.ImageDraw,
        word: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
    ) -> list[str]:
        if draw.textlength(word, font=font) <= max_width or len(word) <= 4:
            return [word]

        parts: list[str] = []
        current = ""
        for idx, ch in enumerate(word):
            candidate = current + ch
            is_last = idx == len(word) - 1
            test_str = candidate if is_last else candidate + "-"
            if draw.textlength(test_str, font=font) <= max_width or len(current) < 2:
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

        if len(raw_words) <= 3 and draw.textlength(text, font=font) <= edge_width:
            return text

        words: list[str] = []
        for w in raw_words:
            if allow_hyphenation and draw.textlength(w, font=font) > edge_width:
                words.extend(self._split_long_word(draw, w, font, edge_width))
            elif allow_hyphenation and draw.textlength(w, font=font) > max_width:
                words.extend(self._split_long_word(draw, w, font, max_width))
            else:
                words.append(w)

        lines: list[str] = []
        current_line = words[0]

        for word in words[1:]:
            line_limit = edge_width if (elliptical_shaping and len(lines) == 0) else max_width
            test_line = f"{current_line} {word}"
            if draw.textlength(test_line, font=font) <= line_limit:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)

        if elliptical_shaping and len(lines) >= 3:
            last_line = lines[-1]
            if draw.textlength(last_line, font=font) > edge_width:
                last_words = last_line.split()
                if len(last_words) >= 2:
                    sub_lines: list[str] = []
                    cur = last_words[0]
                    for lw in last_words[1:]:
                        cand = f"{cur} {lw}"
                        if draw.textlength(cand, font=font) <= edge_width:
                            cur = cand
                        else:
                            sub_lines.append(cur)
                            cur = lw
                    sub_lines.append(cur)
                    lines = lines[:-1] + sub_lines

        return "\n".join(lines)

    @staticmethod
    def _fits_in_box(
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

        lines = wrapped_text.split("\n")
        if len(lines) <= 1:
            return True

        rx = (target_w / 2.0) * 1.08
        ry = (target_h / 2.0) * 1.12
        if rx <= 1.0 or ry <= 1.0:
            return True

        line_h = total_h / max(1, len(lines))
        top_y = -total_h / 2.0

        for idx, line in enumerate(lines):
            cy = top_y + (idx + 0.5) * line_h
            norm_y = min(0.95, abs(cy) / ry)
            chord_w = 2.0 * rx * math.sqrt(max(0.15, 1.0 - (norm_y**2)))
            line_w = draw.textlength(line, font=font) + 2 * stroke_w
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
            raw_text = (block.get("translated_text") or "").strip()
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

            if median_lum >= 115.0:
                default_text_fill = (34, 34, 34) if style_preset == "thought" else (15, 15, 15)
                default_stroke_fill = (255, 255, 255)
            else:
                default_text_fill = (250, 250, 250)
                default_stroke_fill = (10, 10, 10)

            custom_text_rgb = self._parse_hex_color(block.get("text_color"))
            custom_stroke_rgb = self._parse_hex_color(block.get("stroke_color"))
            text_fill = custom_text_rgb if custom_text_rgb is not None else default_text_fill
            stroke_fill = custom_stroke_rgb if custom_stroke_rgb is not None else default_stroke_fill

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

            if abs(angle) >= 0.5:
                pad_rot = max(12, stroke_w * 2 + 8)
                layer_w = max(16, int(math.ceil(tw + 2 * pad_rot)))
                layer_h = max(16, int(math.ceil(th + 2 * pad_rot)))
                txt_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
                txt_draw = ImageDraw.Draw(txt_layer)

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

                rot_layer = txt_layer.rotate(angle, resample=Image.BICUBIC, expand=True)
                rw_rot, rh_rot = rot_layer.size
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