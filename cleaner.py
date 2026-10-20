"""
Модуль очистки бабблов и свободных надписей от оригинального текста — Inpainting (Шаг 3, v1.0.0).

Изменения v1.0.0:
1. Поддержка флага `is_deleted` (удаленный пользователем в редакторе перевод):
   если у блока `is_deleted=True` или пустой `translated_text`, этот блок полностью исключается из маски
   очистки, и на странице сохраняется нетронутый оригинальный рисунок и оригинальный текст!
2. Идеально чистая очистка белых и однотонных speech bubbles без серых разводов и «призраков» от букв.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class BubbleCleaner:
    """Очищает текстовые бабблы и надписи на странице по координатам Bounding Boxes с сохранением арта."""

    def __init__(
        self,
        use_lama: bool = True,
        dilate_size: int = 9,
        box_margin_ratio: float = 0.03,
        crop_padding: int = 20,
        **kwargs: Any,
    ) -> None:
        if "dilate_Size" in kwargs and kwargs["dilate_Size"] is not None:
            dilate_size = int(kwargs["dilate_Size"])

        self.dilate_size = max(1, int(dilate_size))
        self.box_margin_ratio = box_margin_ratio
        self.crop_padding = max(8, int(crop_padding))
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
    def _extract_bubble_interior_mask(roi_gray: np.ndarray, bg_level: float) -> np.ndarray:
        """
        Находит внутреннюю светлую область speech bubble (внутри черного контура облачка),
        отсекая внешние углы бокса за пределами рамки баббла.
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

        c_x1, c_x2 = int(rw * 0.22), max(int(rw * 0.22) + 1, int(rw * 0.78))
        c_y1, c_y2 = int(rh * 0.22), max(int(rh * 0.22) + 1, int(rh * 0.78))
        center_slice = labels[c_y1:c_y2, c_x1:c_x2]

        for lbl in range(1, num_labels):
            area = float(stats[lbl, cv2.CC_STAT_AREA])
            if area < (rh * rw * 0.08):
                continue
            center_hits = float(np.count_nonzero(center_slice == lbl))
            dist = ((centroids[lbl][0] - cx_mid) ** 2 + (centroids[lbl][1] - cy_mid) ** 2) ** 0.5
            score = center_hits * 3.0 + area - dist * 4.0
            if score > best_score:
                best_score = score
                best_label = lbl

        if best_label == -1:
            return light_pixels

        interior = np.zeros_like(light_pixels)
        interior[labels == best_label] = 255

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

    def build_text_mask(
        self,
        image_rgb: np.ndarray,
        blocks: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Строит точную бинарную маску (uint8, 0 или 255) по пикселям текста и его антиалиасинга
        внутри каждого активного Bounding Box. Удаленные в редакторе блоки полностью пропускаются.
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
            x1, y1, x2, y2 = block["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            if bw < 10 or bh < 10:
                continue

            is_free_text = bool(block.get("is_free_text", False))

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

                if np.count_nonzero(clean_roi_mask) > 0:
                    dilated_roi = cv2.dilate(clean_roi_mask, kernel_bubble, iterations=1)
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

            active_kernel = kernel_free if is_free_text else kernel_bubble
            dilated_roi = cv2.dilate(clean_roi_mask, active_kernel, iterations=1)
            full_mask[iy1:iy2, ix1:ix2] = np.maximum(full_mask[iy1:iy2, ix1:ix2], dilated_roi)

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

    def clean_image(
        self,
        image: Image.Image,
        blocks: list[dict[str, Any]],
    ) -> tuple[Image.Image, Image.Image]:
        """
        Удаляет оригинальный текст со страницы манги, обрабатывая кропы вокруг активных бабблов и надписей.
        """
        rgb_image = image.convert("RGB")
        orig_w, orig_h = rgb_image.size
        img_np = np.array(rgb_image)

        mask_np = self.build_text_mask(img_np, blocks)
        mask_pil = Image.fromarray(mask_np, mode="L")

        if np.count_nonzero(mask_np) == 0:
            return rgb_image.copy(), mask_pil

        composited_np = img_np.copy()
        pad = self.crop_padding

        for block in blocks:
            if self._is_block_skipped(block):
                continue
            x1, y1, x2, y2 = block["bbox"]
            is_free_text = bool(block.get("is_free_text", False))

            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(orig_w, x2 + pad)
            cy2 = min(orig_h, y2 + pad)

            if (cx2 - cx1) < 8 or (cy2 - cy1) < 8:
                continue

            crop_mask = mask_np[cy1:cy2, cx1:cx2]
            if np.count_nonzero(crop_mask) == 0:
                continue

            crop_rgb = composited_np[cy1:cy2, cx1:cx2]
            inpainted_crop = self._inpaint_crop(crop_rgb, crop_mask, is_free_text=is_free_text)

            if not is_free_text:
                composited_np[cy1:cy2, cx1:cx2] = np.where(
                    crop_mask[..., None] > 0, inpainted_crop, crop_rgb
                )
            else:
                alpha = cv2.GaussianBlur(crop_mask.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
                blended_crop = (inpainted_crop * alpha + crop_rgb * (1.0 - alpha)).round().astype(np.uint8)
                composited_np[cy1:cy2, cx1:cx2] = blended_crop

        return Image.fromarray(composited_np, mode="RGB"), mask_pil
