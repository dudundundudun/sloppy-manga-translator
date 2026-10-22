"""
Модуль очистки бабблов и свободных надписей от оригинального текста — Inpainting (v1.0.6).

Изменения v1.0.6:
1. Индивидуальный радиус замазки `mask_dilate` (1..28 px) для каждого блока в `build_text_mask`:
   создает эллиптическое ядро `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))` специально для блока,
   позволяя в 1 движение ползунка убрать торчащие края иероглифов или уменьшить замазку для сохранения рисунка.
2. Двухуровневое кэширование очистки (`_block_crop_cache` + `_base_page_cache`):
   - Кэширует результат инпейнтинга каждого отдельного блока `(img_fingerprint, clean_bbox, is_free_text, mask_dilate)`,
     благодаря чему изменение ползунка `Замазка (dilate)` у одного блока пересчитывает только этот один кроп за ~15–35 мс,
     не затрагивая остальные бабблы на странице.
   - Кэширует базовый автоматически очищенный слой страницы до применения ручных мазков кисти.
3. Мгновенная (< 2 мс) обработка ручной кисти и ластика замазки `custom_mask_strokes`
   (`[{"x": int, "y": int, "r": int, "mode": "add"|"erase"}]`) поверх базового очищенного слоя без повторного вызова SimpleLama:
   - Режим `"add"` (ЛКМ — Замазать текст) мгновенно закрашивает указанную область локальным цветом баббла
     (или быстрым `cv2.INPAINT_TELEA` на сложном/темном фоне) с мягким сглаживанием краев.
   - Режим `"erase"` (ПКМ — Вернуть оригинал) мгновенно восстанавливает пиксели исходного рисунка из `raw_image`
     с мягкой растушевкой.
4. Улучшенное выделение внутренней области узких вертикальных бабблов (`_extract_bubble_interior_mask`)
   и защита от пропуска вертикальных колонок кандзи.
5. Сохранены все оптимизации v1.0.6-step1 (LRU-кэш `cleaned_image` на 8 страниц + дисковый кэш в `temp_debug/`).
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import logging
from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BubbleCleaner:
    """Очищает текстовые бабблы и надписи на странице по координатам Bounding Boxes с сохранением арта и LRU-кэшем."""

    def __init__(
        self,
        use_lama: bool = True,
        dilate_size: int = 9,
        box_margin_ratio: float = 0.03,
        crop_padding: int = 20,
        cache_max_pages: int = 8,
        disk_cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        if "dilate_Size" in kwargs and kwargs["dilate_Size"] is not None:
            dilate_size = int(kwargs["dilate_Size"])

        self.dilate_size = max(1, int(dilate_size))
        self.box_margin_ratio = box_margin_ratio
        self.crop_padding = max(8, int(crop_padding))
        self.cache_max_pages = max(1, int(cache_max_pages))
        self._cache_lock = threading.Lock()
        self._cleaned_lru_cache: OrderedDict[tuple[str, str], Image.Image] = OrderedDict()
        # Кэш базовой очистки страницы (без ручных мазков кисти) и кэш отдельных кропов блоков
        self._base_page_cache: OrderedDict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._block_crop_cache: OrderedDict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._block_crop_cache_max = 256
        self._base_page_cache_max = 10

        if disk_cache_dir is not None:
            self.disk_cache_dir: Path | None = Path(disk_cache_dir).resolve()
            try:
                self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                self.disk_cache_dir = None
        else:
            self.disk_cache_dir = None

        self.lama_model: Any | None = None

        if use_lama:
            try:
                from simple_lama_inpainting import SimpleLama

                logger.info("Инициализация модели нейросетевого инпейнтинга SimpleLama...")
                self.lama_model = SimpleLama()
                logger.info("SimpleLama успешно загружена.")
            except Exception as exc:
                logger.warning(
                    "Пакет simple-lama-inpainting не установлен или недоступен (%s). "
                    "Используется гибридный алгоритм очистки (Pure White Bubble Fill + OpenCV INPAINT_TELEA).",
                    exc,
                )
                self.lama_model = None

    @staticmethod
    def _normalize_src_key(src_file: str | Path | None) -> str:
        if not src_file:
            return ""
        try:
            return str(Path(src_file).resolve())
        except Exception:
            return str(src_file)

    @staticmethod
    def _normalize_block_strokes(block: dict[str, Any]) -> list[tuple[int, int, int, str]]:
        raw_strokes = block.get("custom_mask_strokes")
        if not isinstance(raw_strokes, list) or not raw_strokes:
            return []
        norm: list[tuple[int, int, int, str]] = []
        for s in raw_strokes:
            if not isinstance(s, dict):
                continue
            try:
                sx = int(round(float(s.get("x", 0))))
                sy = int(round(float(s.get("y", 0))))
                sr = max(1, min(80, int(round(float(s.get("r", 8))))))
                smode = "erase" if str(s.get("mode") or "add").lower().strip() == "erase" else "add"
                norm.append((sx, sy, sr, smode))
            except Exception:
                continue
        return norm

    @staticmethod
    def _compute_image_fingerprint(img_np: np.ndarray) -> str:
        """Быстрый отпечаток массива изображения (~0.1 мс) для ключей кэша кропов и базового слоя."""
        h, w = img_np.shape[:2]
        step_y = max(1, h // 24)
        step_x = max(1, w // 24)
        sample = img_np[::step_y, ::step_x]
        hasher = hashlib.blake2b(digest_size=10)
        hasher.update(f"{w}x{h}:".encode("ascii"))
        hasher.update(sample.tobytes())
        return hasher.hexdigest()

    def compute_base_clean_hash(
        self,
        blocks: list[dict[str, Any]],
        src_file: str | Path | None = None,
        src_mtime: float | int | None = None,
    ) -> str:
        """
        Вычисляет хэш базовой автоматической очистки страницы (без `custom_mask_strokes`),
        чтобы при рисовании кистью замазки не пересчитывать инпейнтинг бабблов с нуля.
        """
        src_key = self._normalize_src_key(src_file)
        if src_mtime is None and src_key:
            try:
                p = Path(src_key)
                if p.exists():
                    src_mtime = p.stat().st_mtime_ns
            except OSError:
                src_mtime = 0

        active_tuples: list[tuple[tuple[int, int, int, int], bool, int]] = []
        for block in blocks:
            if not isinstance(block, dict) or self._is_block_skipped(block):
                continue
            cb = self._get_clean_bbox(block)
            clean_tuple = (int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3]))
            is_free = bool(block.get("is_free_text", False))
            mask_dilate_val = int(block.get("mask_dilate") or 0)
            active_tuples.append((clean_tuple, is_free, mask_dilate_val))

        active_tuples.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[1], item[2]))
        payload = {
            "src_file": src_key,
            "mtime": int(src_mtime or 0),
            "dilate_size": int(self.dilate_size),
            "base_clean_blocks": [
                [list(coords), is_free, mdil]
                for coords, is_free, mdil in active_tuples
            ],
        }
        raw_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:24]

    def compute_clean_state_hash(
        self,
        blocks: list[dict[str, Any]],
        src_file: str | Path | None = None,
        src_mtime: float | int | None = None,
    ) -> str:
        """
        Вычисляет детерминированный хэш состояния замазки страницы на основе пути исходного файла `src_file`,
        его `mtime`, глобального `dilate_size` и отсортированного списка кортежей
        `(tuple(clean_bbox), bool(is_free_text), int(block.get("mask_dilate") or 0))` (+ `custom_mask_strokes`)
        для всех активных неудаленных блоков.
        """
        src_key = self._normalize_src_key(src_file)
        if src_mtime is None and src_key:
            try:
                p = Path(src_key)
                if p.exists():
                    src_mtime = p.stat().st_mtime_ns
            except OSError:
                src_mtime = 0

        active_tuples: list[tuple[tuple[int, int, int, int], bool, int, tuple[tuple[int, int, int, str], ...]]] = []
        for block in blocks:
            if not isinstance(block, dict) or self._is_block_skipped(block):
                continue
            cb = self._get_clean_bbox(block)
            clean_tuple = (int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3]))
            is_free = bool(block.get("is_free_text", False))
            mask_dilate_val = int(block.get("mask_dilate") or 0)
            strokes_tuple = tuple(self._normalize_block_strokes(block))
            active_tuples.append((clean_tuple, is_free, mask_dilate_val, strokes_tuple))

        active_tuples.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[1], item[2], item[3]))

        payload = {
            "src_file": src_key,
            "mtime": int(src_mtime or 0),
            "dilate_size": int(self.dilate_size),
            "active_clean_blocks": [
                [list(coords), is_free, mdil, [list(st) for st in strokes]]
                for coords, is_free, mdil, strokes in active_tuples
            ],
        }
        raw_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:24]

    def _get_disk_cache_path(self, src_key: str, clean_state_hash: str) -> Path | None:
        if self.disk_cache_dir is None or not clean_state_hash:
            return None
        stem = Path(src_key).stem if src_key else "page"
        safe_stem = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in stem)[:32] or "page"
        return self.disk_cache_dir / f"clean_{safe_stem}_{clean_state_hash}.png"

    def get_cached_cleaned_image(
        self,
        src_file: str | Path | None,
        clean_state_hash: str,
    ) -> Image.Image | None:
        """Возвращает копию очищенного слоя `cleaned_image` из LRU-кэша в памяти (или из `temp_debug/`), если хэш совпадает."""
        src_key = self._normalize_src_key(src_file)
        if not src_key or not clean_state_hash:
            return None
        cache_key = (src_key, clean_state_hash)
        with self._cache_lock:
            cached = self._cleaned_lru_cache.get(cache_key)
            if cached is not None:
                self._cleaned_lru_cache.move_to_end(cache_key)
                return cached.copy()

        disk_path = self._get_disk_cache_path(src_key, clean_state_hash)
        if disk_path is not None and disk_path.exists():
            try:
                with Image.open(disk_path) as im:
                    loaded = im.convert("RGB").copy()
                with self._cache_lock:
                    self._cleaned_lru_cache[cache_key] = loaded.copy()
                    self._cleaned_lru_cache.move_to_end(cache_key)
                    while len(self._cleaned_lru_cache) > self.cache_max_pages:
                        self._cleaned_lru_cache.popitem(last=False)
                return loaded
            except Exception as exc:
                logger.debug("Не удалось прочитать кэш очищенного слоя с диска (%s): %s", disk_path, exc)
        return None

    def put_cached_cleaned_image(
        self,
        src_file: str | Path | None,
        clean_state_hash: str,
        cleaned_image: Image.Image,
        save_to_disk: bool = False,
    ) -> None:
        """Сохраняет очищенный слой `cleaned_image` в LRU-кэш в памяти (до 8 страниц) и опционально в `temp_debug/`."""
        src_key = self._normalize_src_key(src_file)
        if not src_key or not clean_state_hash:
            return
        cache_key = (src_key, clean_state_hash)
        rgb_copy = cleaned_image.convert("RGB").copy()
        with self._cache_lock:
            self._cleaned_lru_cache[cache_key] = rgb_copy
            self._cleaned_lru_cache.move_to_end(cache_key)
            while len(self._cleaned_lru_cache) > self.cache_max_pages:
                self._cleaned_lru_cache.popitem(last=False)

        if save_to_disk and self.disk_cache_dir is not None:
            disk_path = self._get_disk_cache_path(src_key, clean_state_hash)
            if disk_path is not None and not disk_path.exists():
                try:
                    rgb_copy.save(disk_path, format="PNG", compress_level=1)
                    cache_files = sorted(
                        self.disk_cache_dir.glob("clean_*.png"),
                        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
                    )
                    for old_f in cache_files[:-16]:
                        try:
                            old_f.unlink(missing_ok=True)
                        except OSError:
                            pass
                except Exception as exc:
                    logger.debug("Не удалось сохранить кэш очищенного слоя на диск: %s", exc)

    @staticmethod
    def _extract_bubble_interior_mask(roi_gray: np.ndarray, bg_level: float) -> np.ndarray:
        """
        Находит внутреннюю светлую область speech bubble (внутри черного контура облачка),
        отсекая внешние углы бокса за пределами рамки баббла.
        Поддерживает узкие вертикальные бабблы, где плотный столбик кандзи делит светлый фон на части.
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
        interior_safe = cv2.erode(interior, erode_k, iterations=1)
        return interior_safe

    @staticmethod
    def _is_block_skipped(block: dict[str, Any]) -> bool:
        if bool(block.get("is_deleted", False)):
            return True
        if "translated_text" in block and not str(block.get("translated_text") or "").strip():
            return True
        return False

    @staticmethod
    def _get_clean_bbox(block: dict[str, Any]) -> list[int]:
        """
        Возвращает исходные координаты распознанного текста (clean_bbox / original_bbox),
        чтобы при перетаскивании или ресайзе рамки перевода в редакторе замазка оригинала не смещалась.
        """
        raw = block.get("clean_bbox") or block.get("original_bbox") or block.get("bbox") or [0, 0, 0, 0]
        if isinstance(raw, (list, tuple)) and len(raw) >= 4:
            return [int(round(float(c))) for c in raw[:4]]
        return [0, 0, 0, 0]

    def _get_effective_clean_bounds(
        self,
        block: dict[str, Any],
        img_w: int,
        img_h: int,
    ) -> tuple[int, int, int, int]:
        """
        Возвращает область базовой очистки блока по исходному `clean_bbox`.
        """
        x1, y1, x2, y2 = self._get_clean_bbox(block)
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        return x1, y1, x2, y2

    def build_text_mask(
        self,
        image_rgb: np.ndarray,
        blocks: list[dict[str, Any]],
        include_custom_strokes: bool = True,
    ) -> np.ndarray:
        """
        Строит точную бинарную маску (uint8, 0 или 255) по пикселям текста и его антиалиасинга
        внутри каждого активного Bounding Box с учетом индивидуального `mask_dilate` и (опционально)
        ручных мазков кисти/ластика `custom_mask_strokes`.
        """
        img_h, img_w = image_rgb.shape[:2]
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        gray_full = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        bubble_dilate = max(9, self.dilate_size)
        kernel_bubble = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (bubble_dilate, bubble_dilate),
        )
        free_dilate = max(9, self.dilate_size + 2)
        kernel_free = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (free_dilate, free_dilate),
        )

        for block in blocks:
            if self._is_block_skipped(block):
                continue
            x1, y1, x2, y2 = self._get_clean_bbox(block)
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))
            bw = x2 - x1
            bh = y2 - y1
            if bw < 10 or bh < 10:
                continue

            is_free_text = bool(block.get("is_free_text", False))
            raw_dilate = block.get("mask_dilate")
            custom_dilate = int(raw_dilate) if isinstance(raw_dilate, (int, float)) and int(raw_dilate) > 0 else 0
            if custom_dilate > 0:
                d_val = max(1, min(28, custom_dilate))
                block_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_val, d_val))
            else:
                block_kernel = None

            if is_free_text:
                ix1 = max(0, x1 - 4)
                iy1 = max(0, y1 - 4)
                ix2 = min(img_w, x2 + 4)
                iy2 = min(img_h, y2 + 4)
            else:
                pad_x = max(1, int(bw * self.box_margin_ratio))
                pad_y = max(1, int(bh * self.box_margin_ratio))
                ix1 = min(x2 - 1, x1 + pad_x)
                iy1 = min(y2 - 1, y1 + pad_y)
                ix2 = max(ix1 + 1, x2 - pad_x)
                iy2 = max(iy1 + 1, y2 - pad_y)

            roi_gray = gray_full[iy1:iy2, ix1:ix2]
            rh, rw = roi_gray.shape[:2]
            if rh < 6 or rw < 6:
                continue

            p75 = float(np.percentile(roi_gray, 75))
            median_val = float(np.median(roi_gray))

            if not is_free_text and p75 >= 215.0:
                interior_mask = self._extract_bubble_interior_mask(roi_gray, p75)
                ink_thresh = max(165, int(p75 - 14))
                binary = ((roi_gray < ink_thresh) & (interior_mask > 0)).astype(np.uint8) * 255

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                clean_roi_mask = np.zeros_like(binary)

                for label_idx in range(1, num_labels):
                    comp_x = stats[label_idx, cv2.CC_STAT_LEFT]
                    comp_y = stats[label_idx, cv2.CC_STAT_TOP]
                    comp_w = stats[label_idx, cv2.CC_STAT_WIDTH]
                    comp_h = stats[label_idx, cv2.CC_STAT_HEIGHT]
                    comp_area = stats[label_idx, cv2.CC_STAT_AREA]

                    if comp_area < 2:
                        continue

                    touches_left = comp_x <= 0
                    touches_top = comp_y <= 0
                    touches_right = (comp_x + comp_w) >= rw
                    touches_bottom = (comp_y + comp_h) >= rh
                    border_touches = sum([touches_left, touches_top, touches_right, touches_bottom])

                    if border_touches >= 2 and (comp_w > rw * 0.75 or comp_h > rh * 0.75):
                        continue
                    if border_touches == 1 and (comp_w > rw * 0.82 or comp_h > rh * 0.82):
                        continue

                    clean_roi_mask[labels == label_idx] = 255

                min_valid_ink_px = max(18, int(rw * rh * 0.012))
                if np.count_nonzero(clean_roi_mask) >= min_valid_ink_px:
                    eff_bub_kernel = block_kernel if block_kernel is not None else kernel_bubble
                    dilated_roi = cv2.dilate(clean_roi_mask, eff_bub_kernel, iterations=1)
                    if block_kernel is None or custom_dilate <= 12:
                        dilated_roi = cv2.bitwise_and(dilated_roi, interior_mask)
                    full_mask[iy1:iy2, ix1:ix2] = np.maximum(full_mask[iy1:iy2, ix1:ix2], dilated_roi)
                    continue

            if median_val >= 127.0:
                _, binary = cv2.threshold(
                    roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
            else:
                _, binary = cv2.threshold(
                    roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            clean_roi_mask = np.zeros_like(binary)

            for label_idx in range(1, num_labels):
                comp_x = stats[label_idx, cv2.CC_STAT_LEFT]
                comp_y = stats[label_idx, cv2.CC_STAT_TOP]
                comp_w = stats[label_idx, cv2.CC_STAT_WIDTH]
                comp_h = stats[label_idx, cv2.CC_STAT_HEIGHT]
                comp_area = stats[label_idx, cv2.CC_STAT_AREA]

                if comp_area < 3:
                    continue

                touches_left = comp_x <= 0
                touches_top = comp_y <= 0
                touches_right = (comp_x + comp_w) >= rw
                touches_bottom = (comp_y + comp_h) >= rh
                border_touches = sum([touches_left, touches_top, touches_right, touches_bottom])

                if is_free_text:
                    if border_touches >= 3 or (
                        border_touches >= 2 and (comp_w > rw * 0.88 or comp_h > rh * 0.88)
                    ):
                        continue
                else:
                    if border_touches >= 2 or (
                        border_touches == 1 and (comp_w > rw * 0.72 or comp_h > rh * 0.72)
                    ):
                        continue

                clean_roi_mask[labels == label_idx] = 255

            if np.count_nonzero(clean_roi_mask) == 0:
                margin_ratio = 0.08 if is_free_text else 0.14
                inner_px = max(1, int(rw * margin_ratio))
                inner_py = max(1, int(rh * margin_ratio))
                if rh > 2 * inner_py and rw > 2 * inner_px:
                    clean_roi_mask[inner_py : rh - inner_py, inner_px : rw - inner_px] = binary[
                        inner_py : rh - inner_py, inner_px : rw - inner_px
                    ]

            if np.count_nonzero(clean_roi_mask) == 0:
                continue

            active_kernel = (
                block_kernel
                if block_kernel is not None
                else (kernel_free if is_free_text else kernel_bubble)
            )
            dilated_roi = cv2.dilate(clean_roi_mask, active_kernel, iterations=1)
            full_mask[iy1:iy2, ix1:ix2] = np.maximum(full_mask[iy1:iy2, ix1:ix2], dilated_roi)

        # Защищаем области явно пропущенных/удаленных блоков от случайного захвата соседними рамками
        for block in blocks:
            if not self._is_block_skipped(block):
                continue
            dx1, dy1, dx2, dy2 = self._get_clean_bbox(block)
            dx1 = max(0, min(dx1, img_w - 1))
            dy1 = max(0, min(dy1, img_h - 1))
            dx2 = max(dx1 + 1, min(dx2, img_w))
            dy2 = max(dy1 + 1, min(dy2, img_h))
            full_mask[dy1:dy2, dx1:dx2] = 0

        if include_custom_strokes:
            for block in blocks:
                if self._is_block_skipped(block):
                    continue
                for sx, sy, sr, smode in self._normalize_block_strokes(block):
                    cx = max(0, min(img_w - 1, sx))
                    cy = max(0, min(img_h - 1, sy))
                    radius = max(1, min(80, sr))
                    if smode == "erase":
                        cv2.circle(full_mask, (cx, cy), radius, 0, -1)
                    else:
                        cv2.circle(full_mask, (cx, cy), radius, 255, -1)

        return full_mask

    def _inpaint_crop(
        self,
        crop_rgb_np: np.ndarray,
        crop_mask_np: np.ndarray,
        is_free_text: bool = False,
    ) -> np.ndarray:
        ch, cw = crop_rgb_np.shape[:2]

        if not is_free_text:
            ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            ring_mask = cv2.dilate(crop_mask_np, ring_k, iterations=1) - crop_mask_np
            ring_pixels = crop_rgb_np[ring_mask > 0]
            if len(ring_pixels) >= 16:
                ring_gray = np.mean(ring_pixels, axis=1)
                p70 = float(np.percentile(ring_gray, 70))
                if p70 >= 228.0:
                    bright_pixels = ring_pixels[ring_gray >= p70 - 12.0]
                    if len(bright_pixels) >= 8:
                        fill_color = np.median(bright_pixels, axis=0).round().astype(np.uint8)
                        if float(np.min(fill_color)) >= 242.0:
                            fill_color = np.array([255, 255, 255], dtype=np.uint8)
                        out = crop_rgb_np.copy()
                        out[crop_mask_np > 0] = fill_color
                        return out

        if self.lama_model is not None:
            crop_pil = Image.fromarray(crop_rgb_np, mode="RGB")
            mask_pil = Image.fromarray(crop_mask_np, mode="L")
            inpainted_pil = self.lama_model(crop_pil, mask_pil).convert("RGB")
            if inpainted_pil.size != (cw, ch):
                inpainted_pil = inpainted_pil.crop((0, 0, cw, ch))
            return np.array(inpainted_pil)

        bgr_crop = cv2.cvtColor(crop_rgb_np, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(
            bgr_crop,
            crop_mask_np,
            inpaintRadius=6,
            flags=cv2.INPAINT_TELEA,
        )
        return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    def _apply_custom_strokes_fast(
        self,
        base_cleaned_np: np.ndarray,
        raw_img_np: np.ndarray,
        base_mask_np: np.ndarray,
        blocks: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Мгновенно (< 2 мс) накладывает ручные мазки кисти/ластика (`custom_mask_strokes`)
        поверх базового очищенного слоя `base_cleaned_np` без перезапуска тяжелой модели инпейнтинга:
        - `mode == "add"` (ЛКМ / Замазать текст): соединяет соседние точки мазка в плавную линию и закрашивает
          локальным цветом баббла (или быстрым `INPAINT_TELEA` на сложном/темном фоне) с мягким сглаживанием краев.
        - `mode == "erase"` (ПКМ / Вернуть оригинал): восстанавливает пиксели оригинального изображения `raw_img_np`
          с мягкой растушевкой 3x3.
        """
        img_h, img_w = base_cleaned_np.shape[:2]
        add_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        erase_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        final_mask = base_mask_np.copy()
        has_any = False

        for block in blocks:
            if self._is_block_skipped(block):
                continue
            strokes = self._normalize_block_strokes(block)
            if not strokes:
                continue
            has_any = True
            prev_pt: tuple[int, int, int, str] | None = None
            for sx, sy, sr, smode in strokes:
                cx = max(0, min(img_w - 1, sx))
                cy = max(0, min(img_h - 1, sy))
                radius = max(1, min(80, sr))
                target_m = erase_mask if smode == "erase" else add_mask
                opposite_m = add_mask if smode == "erase" else erase_mask

                cv2.circle(target_m, (cx, cy), radius, 255, -1)
                cv2.circle(opposite_m, (cx, cy), radius, 0, -1)
                cv2.circle(final_mask, (cx, cy), radius, 0 if smode == "erase" else 255, -1)

                # Если соседние точки одного режима находятся близко — соединяем их плавной капсулой без разрывов
                if prev_pt is not None and prev_pt[3] == smode:
                    px, py, pr, _ = prev_pt
                    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    avg_r = max(1, int(round((radius + pr) / 2.0)))
                    if 0 < dist <= avg_r * 2.8:
                        cv2.line(target_m, (px, py), (cx, cy), 255, thickness=avg_r * 2)
                        cv2.line(opposite_m, (px, py), (cx, cy), 0, thickness=avg_r * 2)
                        cv2.line(
                            final_mask,
                            (px, py),
                            (cx, cy),
                            0 if smode == "erase" else 255,
                            thickness=avg_r * 2,
                        )
                prev_pt = (cx, cy, radius, smode)

        if not has_any:
            return base_cleaned_np, final_mask

        out_np = base_cleaned_np.copy()

        # 1. Применяем мазки замазки текста (mode == "add")
        if np.count_nonzero(add_mask) > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(add_mask, connectivity=8)
            ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            for lbl in range(1, num_labels):
                lx = int(stats[lbl, cv2.CC_STAT_LEFT])
                ly = int(stats[lbl, cv2.CC_STAT_TOP])
                lw = int(stats[lbl, cv2.CC_STAT_WIDTH])
                lh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
                pad = 12
                cx1 = max(0, lx - pad)
                cy1 = max(0, ly - pad)
                cx2 = min(img_w, lx + lw + pad)
                cy2 = min(img_h, ly + lh + pad)

                comp_mask = ((labels[cy1:cy2, cx1:cx2] == lbl).astype(np.uint8)) * 255
                if np.count_nonzero(comp_mask) == 0:
                    continue

                crop_rgb = out_np[cy1:cy2, cx1:cx2]
                ring_mask = cv2.dilate(comp_mask, ring_k, iterations=1) - comp_mask
                ring_pixels = crop_rgb[ring_mask > 0]

                filled_crop: np.ndarray | None = None
                if len(ring_pixels) >= 6:
                    ring_gray = np.mean(ring_pixels, axis=1)
                    p75 = float(np.percentile(ring_gray, 75))
                    if p75 >= 195.0:
                        bright_pixels = ring_pixels[ring_gray >= max(180.0, p75 - 16.0)]
                        if len(bright_pixels) >= 4:
                            fill_color = np.median(bright_pixels, axis=0).round().astype(np.uint8)
                            if float(np.min(fill_color)) >= 236.0:
                                fill_color = np.array([255, 255, 255], dtype=np.uint8)
                            filled_crop = crop_rgb.copy()
                            filled_crop[comp_mask > 0] = fill_color

                if filled_crop is None:
                    bgr_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                    inp_bgr = cv2.inpaint(bgr_crop, comp_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                    filled_crop = cv2.cvtColor(inp_bgr, cv2.COLOR_BGR2RGB)

                alpha = cv2.GaussianBlur(comp_mask.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
                out_np[cy1:cy2, cx1:cx2] = (
                    filled_crop * alpha + crop_rgb * (1.0 - alpha)
                ).round().astype(np.uint8)

        # 2. Применяем мазки восстановления оригинала (mode == "erase")
        if np.count_nonzero(erase_mask) > 0:
            ys, xs = np.where(erase_mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                ex1 = max(0, int(np.min(xs)) - 4)
                ey1 = max(0, int(np.min(ys)) - 4)
                ex2 = min(img_w, int(np.max(xs)) + 5)
                ey2 = min(img_h, int(np.max(ys)) + 5)
                sub_erase = erase_mask[ey1:ey2, ex1:ex2]
                alpha_er = cv2.GaussianBlur(sub_erase.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
                orig_sub = raw_img_np[ey1:ey2, ex1:ex2]
                cur_sub = out_np[ey1:ey2, ex1:ex2]
                out_np[ey1:ey2, ex1:ex2] = (
                    orig_sub * alpha_er + cur_sub * (1.0 - alpha_er)
                ).round().astype(np.uint8)

        return out_np, final_mask

    def clean_image(
        self,
        image: Image.Image,
        blocks: list[dict[str, Any]],
        src_file: str | Path | None = None,
    ) -> tuple[Image.Image, Image.Image]:
        """
        Удаляет оригинальный текст со страницы манги:
        1) Выполняет (или берет из кэша за 0 мс) базовую очистку бабблов по `clean_bbox` + `mask_dilate`
           с поблочным кэшированием кропов (`_block_crop_cache`).
        2) Мгновенно (< 2 мс) накладывает ручные мазки кисти/ластика (`custom_mask_strokes`).
        """
        rgb_image = image.convert("RGB")
        orig_w, orig_h = rgb_image.size
        img_np = np.array(rgb_image)

        img_fp = self._compute_image_fingerprint(img_np)
        base_hash = self.compute_base_clean_hash(blocks, src_file=src_file)
        base_cache_key = (img_fp, base_hash)

        with self._cache_lock:
            cached_base = self._base_page_cache.get(base_cache_key)
            if cached_base is not None:
                self._base_page_cache.move_to_end(base_cache_key)
                base_composited_np = cached_base[0].copy()
                base_mask_np = cached_base[1].copy()
            else:
                base_composited_np = None
                base_mask_np = None

        if base_composited_np is None or base_mask_np is None:
            base_mask_np = self.build_text_mask(img_np, blocks, include_custom_strokes=False)
            if np.count_nonzero(base_mask_np) == 0:
                base_composited_np = img_np.copy()
            else:
                base_composited_np = img_np.copy()
                pad = self.crop_padding

                for block in blocks:
                    if self._is_block_skipped(block):
                        continue
                    x1, y1, x2, y2 = self._get_effective_clean_bounds(block, orig_w, orig_h)
                    is_free_text = bool(block.get("is_free_text", False))
                    mdil = int(block.get("mask_dilate") or 0)

                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(orig_w, x2 + pad)
                    cy2 = min(orig_h, y2 + pad)

                    if (cx2 - cx1) < 8 or (cy2 - cy1) < 8:
                        continue

                    margin_extra = 4 if is_free_text else 1
                    bx1 = max(cx1, x1 - margin_extra)
                    by1 = max(cy1, y1 - margin_extra)
                    bx2 = min(cx2, x2 + margin_extra)
                    by2 = min(cy2, y2 + margin_extra)

                    crop_mask = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
                    crop_mask[by1 - cy1 : by2 - cy1, bx1 - cx1 : bx2 - cx1] = base_mask_np[by1:by2, bx1:bx2]
                    if np.count_nonzero(crop_mask) == 0:
                        continue

                    block_crop_key = (
                        img_fp,
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        bool(is_free_text),
                        int(mdil),
                        int(self.dilate_size),
                    )
                    with self._cache_lock:
                        cached_crop_pair = self._block_crop_cache.get(block_crop_key)
                        if cached_crop_pair is not None:
                            self._block_crop_cache.move_to_end(block_crop_key)

                    crop_rgb = base_composited_np[cy1:cy2, cx1:cx2]
                    if (
                        cached_crop_pair is not None
                        and cached_crop_pair[0].shape == crop_rgb.shape
                        and cached_crop_pair[1].shape == crop_mask.shape
                    ):
                        inpainted_crop, crop_mask = cached_crop_pair[0], cached_crop_pair[1]
                    else:
                        inpainted_crop = self._inpaint_crop(crop_rgb, crop_mask, is_free_text=is_free_text)
                        with self._cache_lock:
                            self._block_crop_cache[block_crop_key] = (inpainted_crop.copy(), crop_mask.copy())
                            while len(self._block_crop_cache) > self._block_crop_cache_max:
                                self._block_crop_cache.popitem(last=False)

                    if not is_free_text:
                        base_composited_np[cy1:cy2, cx1:cx2] = np.where(
                            crop_mask[..., None] > 0, inpainted_crop, crop_rgb
                        )
                    else:
                        alpha = cv2.GaussianBlur(crop_mask.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
                        blended_crop = (inpainted_crop * alpha + crop_rgb * (1.0 - alpha)).round().astype(np.uint8)
                        base_composited_np[cy1:cy2, cx1:cx2] = blended_crop

            with self._cache_lock:
                self._base_page_cache[base_cache_key] = (base_composited_np.copy(), base_mask_np.copy())
                while len(self._base_page_cache) > self._base_page_cache_max:
                    self._base_page_cache.popitem(last=False)

        final_composited_np, final_mask_np = self._apply_custom_strokes_fast(
            base_composited_np,
            img_np,
            base_mask_np,
            blocks,
        )
        return Image.fromarray(final_composited_np, mode="RGB"), Image.fromarray(final_mask_np, mode="L")