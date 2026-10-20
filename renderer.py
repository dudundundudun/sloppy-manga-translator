"""
Модуль верстки и отрисовки переведенного текста в бабблах и на фоне — Typesetting (Шаг 4, v1.0.0).

Изменения v1.0.0:
1. Пропуск блоков с `is_deleted=True` (когда пользователь удалил перевод в редакторе, чтобы оставить оригинал).
2. Поддержка выбора минимального (`min_font_size`) и максимального (`max_font_size`) размера шрифта.
3. Автоматическая подгонка размера текста под новую площадь рамки (`bbox_modified`), когда пользователь
   растягивает или сжимает квадрат мышкой в редакторе (с обходом глобального потолка `max_font_size`),
   а также поддержка быстрой прокрутки кегля колесиком мыши (`font_size`).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


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
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
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

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        size = max(6, int(size))
        if size not in self._font_cache:
            if self.font_path:
                try:
                    self._font_cache[size] = ImageFont.truetype(self.font_path, size=size)
                except OSError:
                    self._font_cache[size] = ImageFont.load_default()
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    @staticmethod
    def _get_line_spacing(font_size: int) -> int:
        return max(2, int(round(font_size * 0.15)))

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
    ) -> bool:
        bbox = draw.multiline_textbbox(
            (0, 0),
            wrapped_text,
            font=font,
            spacing=spacing,
            align="center",
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
    ) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int, int]:
        min_lim = max(6 if unconstrained_size else 8, int(min_font_limit))
        max_lim = max(min_lim, int(max_font_limit))
        use_ellipse = not (is_free_text or force_rectangular)

        if unconstrained_size:
            start_size = min(
                max_lim,
                max(min_lim, int(min(target_w, target_h) * 0.88)),
            )
        else:
            start_size = min(
                max_lim,
                max(min_lim, int(min(target_w, target_h) * (0.45 if is_free_text else 0.38))),
            )

        for font_size in range(start_size, min_lim - 1, -1):
            font = self._get_font(font_size)
            stroke_w = (
                stroke_width_override
                if stroke_width_override is not None
                else (max(3, font_size // 8) if is_free_text else max(2, font_size // 10))
            )
            spacing = self._get_line_spacing(font_size)
            wrapped = self._wrap_text_pixel(
                draw,
                text,
                font,
                target_w,
                allow_hyphenation=False,
                elliptical_shaping=use_ellipse,
            )
            if self._fits_in_box(draw, wrapped, font, spacing, stroke_w, target_w, target_h, elliptical=use_ellipse):
                return wrapped, font, stroke_w, spacing, font_size

        hyph_start = max(min_lim, min(start_size, max(28 if unconstrained_size else 22, min_lim)))
        for font_size in range(hyph_start, min_lim - 1, -1):
            font = self._get_font(font_size)
            stroke_w = (
                stroke_width_override
                if stroke_width_override is not None
                else (max(3, font_size // 8) if is_free_text else max(2, font_size // 10))
            )
            spacing = self._get_line_spacing(font_size)
            wrapped = self._wrap_text_pixel(
                draw,
                text,
                font,
                target_w,
                allow_hyphenation=True,
                elliptical_shaping=use_ellipse,
            )
            if self._fits_in_box(draw, wrapped, font, spacing, stroke_w, target_w, target_h, elliptical=use_ellipse):
                return wrapped, font, stroke_w, spacing, font_size

        min_font = self._get_font(min_lim)
        min_spacing = self._get_line_spacing(min_lim)
        min_stroke = (
            stroke_width_override
            if stroke_width_override is not None
            else (max(3, min_lim // 8) if is_free_text else max(2, min_lim // 10))
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

            text = raw_text.upper() if use_upper else raw_text
            is_free_text = bool(block.get("is_free_text", False))
            is_modified = self._is_bbox_modified(block)
            is_resized = self._is_bbox_resized(block)

            raw_bbox = block.get("bbox") or [0, 0, 0, 0]
            x1, y1, x2, y2 = [int(round(float(c))) for c in raw_bbox[:4]]
            x1 = max(0, min(x1, img_w - 4))
            y1 = max(0, min(y1, img_h - 4))
            x2 = max(x1 + 4, min(x2, img_w))
            y2 = max(y1 + 4, min(y2, img_h))

            if is_free_text and not is_modified:
                rx1, rx2 = self._expand_free_text_width([x1, y1, x2, y2], blocks, img_w)
            else:
                rx1, rx2 = x1, x2

            box_w = rx2 - rx1
            box_h = y2 - y1
            if box_w < 10 or box_h < 10:
                continue

            if is_free_text or is_modified:
                pad_x = max(2, int(box_w * 0.04)) if is_modified else 2
                pad_y = max(2, int(box_h * 0.04)) if is_modified else 2
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
                # Если пользователь растянул/сжал рамку мышкой — кегль свободно подстраивается под новую площадь
                b_min, b_max = 6, 140
                unconstrained_size = True
            else:
                b_min, b_max = font_min, font_cap

            wrapped_text, font, stroke_w, dyn_spacing, chosen_fs = self._fit_text_to_box(
                draw=draw,
                text=text,
                target_w=safe_w,
                target_h=safe_h,
                min_font_limit=b_min,
                max_font_limit=b_max,
                is_free_text=is_free_text,
                stroke_width_override=stroke_width_override,
                force_rectangular=is_modified,
                unconstrained_size=unconstrained_size,
            )
            block["rendered_font_size"] = chosen_fs

            cx1 = max(0, min(x2 - 1, x1 + 2))
            cy1 = max(0, min(y2 - 1, y1 + 2))
            cx2 = min(img_w, max(cx1 + 1, x2 - 2))
            cy2 = min(img_h, max(cy1 + 1, y2 - 2))
            roi = img_np[cy1:cy2, cx1:cx2]
            median_lum = self._compute_perceptual_luminance(roi)

            if median_lum >= 115.0:
                text_fill = (15, 15, 15)
                stroke_fill = (255, 255, 255)
            else:
                text_fill = (250, 250, 250)
                stroke_fill = (10, 10, 10)

            t_bbox = draw.multiline_textbbox(
                (0, 0),
                wrapped_text,
                font=font,
                spacing=dyn_spacing,
                align="center",
                stroke_width=stroke_w,
            )
            tw = t_bbox[2] - t_bbox[0]
            th = t_bbox[3] - t_bbox[1]

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
                align="center",
                stroke_width=stroke_w,
                stroke_fill=stroke_fill,
            )

        return canvas
