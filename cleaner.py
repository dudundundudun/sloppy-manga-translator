"""
Модуль очистки бабблов и свободных надписей от оригинального текста — Inpainting (v4.0 SOTA Deep Research Edition).

Архитектура v4.0 (на основе отчёта Deep Research + полная обратная совместимость с v3.5–v3.7):
1. Попиксельная сегментация штрихов, белого ореола (Geodesic Halo Extraction) и топологическая защита:
   - Классификация связных компонент через локальную ширину штриха (Stroke Width Transform / Ridge Distance Transform),
     изопериметрический коэффициент округлости (IQ = 4*pi*A / P^2) и выпуклую плотность (Solidity = A / A_hull).
   - Гарантированная защита тонких полых дуг колючих/разомкнутых бабблов (`hollow_arc_ring`, IQ < 0.08, fill_r <= 0.18)
     и массивного внутреннего чиби-арта (`chibi_art_mask` / `obstacle_mask`) от стирания.
   - Геодезическое выделение белого/контрастного ореола (`_extract_text_halo`) и многоштриховых краевых иероглифов
     (知, 露, 吧, 衣服干, 藤, 要, 人在, 没, 如果愿意的话, 東海) строго в пределах допустимой текстовой зоны.
2. Итеративное градиентное выращивание маски по алгоритму PanelCleaner (`_grow_and_evaluate_mask_pcleaner`):
   - Выращивание кандидатов маски от `min_mask_thickness = 4` с шагом `mask_growth_step_pixels = 2` (до 11 шагов)
     с замером стандартного отклонения цвета `sigma` на однопиксельном внешнем контуре (после нормализации околобелых
     пикселей >= 240 к 255) и автоматической остановкой перед касанием черной стенки баббла.
3. Двухзонная очистка с гарантией Zero Bleed (`mask_inside_bubble` vs `mask_outside_bubble`) и Smart Inpaint Router:
   - Жёсткое разделение маски текста контуром полости баббла на внутреннюю (`M_inside`) и внешнюю (`M_outside`).
   - Маршрутизация `M_inside` по `sigma_edge`:
     * Уровень 1 (`sigma <= 0.25`): чистая аналитическая заливка медианным цветом баббла.
     * Уровень 2 (`0.25 < sigma <= 15.0`): аналитическая заливка + локальный NLM-денойзинг (`fastNlMeansDenoisingColored`)
       5-пиксельного контурного пояса для подавления JPEG-артефактов.
     * Уровень 3 (`sigma > 15.0`): нейросетевой инпейнтинг LaMa (`big-lama.pt` / `AnimeMangaInpainting`) или `cv2.INPAINT_NS`.
   - Очистка `M_outside` (выступающие за баббл символы с белым ореолом) с барьерным экранированием контекста
     (`Context Shielding`): временная заливка прилегающей белой полости баббла медианным цветом внешнего фона перед
     вызовом LaMa / `INPAINT_NS`, что на 100% исключает вытекание белого цвета баббла на волосы, стены и скринтоны.
4. Точная геометрия полости баббла (`analyze_bubble_geometry`):
   - Разделение сдвоенных бабблов («восьмёрок» без перегородки) через маркерный водораздел (`cv2.watershed`) на
     инвертированном преобразовании расстояний (`-DT`) только при реальном перетекании пробного `floodFill` в ядро соседа.
   - Поиск оптимального центра текстового блока через двумерную морфологическую эрозию / интегральную свёртку
     (`cv2.integral` + `cv2.distanceTransform` на карте допустимых центров `F(tw, th)`) вместо сырого `Pole of Inaccessibility`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Callable, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", message=".*Failed to find CUDA.*", category=UserWarning)
warnings.filterwarnings("ignore", module="triton.*")

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

LAMA_WEIGHTS_URLS: tuple[str, ...] = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
    "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt",
)


@dataclass
class CleanerConfig:
    """Конфигурация SOTA-алгоритма выращивания маски и маршрутизации инпейнтинга (по стандарту PanelCleaner)."""

    # Параметры итеративного роста маски (PanelCleaner default.conf)
    min_mask_thickness: int = 4
    mask_growth_step_pixels: int = 2
    mask_growth_steps: int = 11
    off_white_max_threshold: int = 240
    mask_max_std_dev: float = 15.0
    mask_improvement_threshold: float = 0.1
    # Параметры Smart Inpaint Router & Denoiser
    noise_min_std_dev: float = 0.25
    noise_outline_size: int = 5
    noise_fade_radius: int = 2
    filter_strength: float = 10.0
    # Параметры LaMa Inpainting
    inpainting_min_std_dev: float = 15.0
    min_inpainting_radius: int = 5
    inpainting_radius_multiplier: float = 0.1
    max_inpainting_radius: int = 20
    inpainting_isolation_radius: int = 5


class BuiltInSimpleLama:
    """
    Встроенный нейросетевой движок очистки LaMa (big-lama.pt, TorchScript JIT),
    работающий напрямую через PyTorch (CUDA/CPU) без необходимости установки пакета `simple-lama-inpainting`.
    Автоматически находит или скачивает `big-lama.pt` (~196 МБ) в `models/lama/big-lama.pt` при первом запуске.
    """

    def __init__(
        self,
        device: str | Any | None = None,
        models_dir: str | Path | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch (torch) не установлен в окружении.")

        if device is None:
            resolved_dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev_str = str(device)
            resolved_dev = dev_str if (dev_str == "cpu" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(resolved_dev)

        base_models_dir = (
            Path(models_dir).resolve()
            if models_dir is not None
            else (Path(__file__).resolve().parent / "models")
        )
        lama_dir = base_models_dir / "lama"
        lama_dir.mkdir(parents=True, exist_ok=True)
        target_weights = lama_dir / "big-lama.pt"

        torch_home = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch")))
        candidate_paths = [
            target_weights,
            lama_dir / "lama_large_512px.ckpt",
            base_models_dir / "big-lama.pt",
            base_models_dir / "lama_large_512px.ckpt",
            torch_home / "hub" / "checkpoints" / "big-lama.pt",
            Path.home() / ".cache" / "simple_lama_inpainting" / "big-lama.pt",
        ]

        weights_path: Path | None = None
        for cand in candidate_paths:
            try:
                if cand.exists() and cand.is_file() and cand.stat().st_size >= 100_000_000:
                    weights_path = cand
                    break
            except OSError:
                continue

        if weights_path is None:
            weights_path = self._download_weights(target_weights)

        self.weights_path = weights_path
        self.model = torch.jit.load(str(self.weights_path), map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _download_weights(dest_path: Path) -> Path:
        if torch is None:
            raise RuntimeError("PyTorch (torch) не установлен.")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(".pt.tmp")
        last_err: Exception | None = None
        for url in LAMA_WEIGHTS_URLS:
            try:
                logger.info(
                    "Скачивание нейросетевой модели очистки LaMa (big-lama.pt, ~196 МБ) в %s...",
                    dest_path,
                )
                torch.hub.download_url_to_file(url, str(tmp_path), progress=True)
                if tmp_path.exists() and tmp_path.stat().st_size >= 100_000_000:
                    os.replace(tmp_path, dest_path)
                    return dest_path
            except Exception as exc:
                last_err = exc
                logger.warning("Не удалось скачать big-lama.pt с %s: %s", url, exc)
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
        raise RuntimeError(f"Не удалось автоматически скачать big-lama.pt: {last_err}")

    def __call__(
        self,
        image: Image.Image | np.ndarray,
        mask: Image.Image | np.ndarray,
    ) -> Image.Image:
        if torch is None:
            raise RuntimeError("PyTorch (torch) не установлен.")
        if isinstance(image, Image.Image):
            img_rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        else:
            img_rgb = image.astype(np.float32) / 255.0
            if img_rgb.ndim == 2:
                img_rgb = np.stack([img_rgb] * 3, axis=-1)
            elif img_rgb.shape[2] == 4:
                img_rgb = img_rgb[:, :, :3]

        if isinstance(mask, Image.Image):
            mask_gray = np.array(mask.convert("L"), dtype=np.float32) / 255.0
        else:
            mask_gray = mask.astype(np.float32)
            if mask_gray.ndim == 3:
                mask_gray = mask_gray[:, :, 0]
            if mask_gray.max() > 1.0:
                mask_gray = mask_gray / 255.0

        orig_h, orig_w = img_rgb.shape[:2]
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        mask_chw = mask_gray[np.newaxis, :, :]

        pad_h = (8 - (orig_h % 8)) % 8
        pad_w = (8 - (orig_w % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            img_chw = np.pad(img_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")
            mask_chw = np.pad(mask_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")

        img_t = torch.from_numpy(np.ascontiguousarray(img_chw)).unsqueeze(0).to(self.device)
        mask_t = (torch.from_numpy(np.ascontiguousarray(mask_chw)).unsqueeze(0).to(self.device) > 0.05).float()

        with torch.inference_mode():
            inpainted_t = self.model(img_t, mask_t)
            out_hwc = inpainted_t[0].permute(1, 2, 0).detach().cpu().numpy()

        out_cropped = np.clip(out_hwc[:orig_h, :orig_w, :] * 255.0, 0.0, 255.0).round().astype(np.uint8)
        return Image.fromarray(out_cropped, mode="RGB")


class SotaMangaCleaner:
    """
    SOTA-модуль очистки текста манги (cleaner.py) с защитой контуров бабблов,
    сохранением внутреннего чиби-арта, извлечением белого ореола (halo),
    итеративным выращиванием маски по стандарту PanelCleaner и гарантией Zero Bleed
    (Context Shielding) при выходе текста за рамку баббла.
    """

    def __init__(
        self,
        config: Optional[CleanerConfig] = None,
        lama_inpainter: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.cfg = config or CleanerConfig()
        self.lama_inpainter = lama_inpainter

    def clean_region(
        self,
        image_rgb: np.ndarray,
        text_bbox: Tuple[int, int, int, int],
        bubble_bbox: Optional[Tuple[int, int, int, int]] = None,
        neural_stroke_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Очищает один текстовый блок/баббл.
        Возвращает:
          - cleaned_image_rgb: очищенное RGB-изображение (H, W, 3)
          - cavity_mask_full: полноразмерная маска внутренней полости баббла (H, W) uint8
          - obstacle_mask_full: маска сохранённого внутреннего чиби-арта (H, W) uint8
        """
        H, W = image_rgb.shape[:2]
        tx1, ty1, tx2, ty2 = [int(round(v)) for v in text_bbox]
        if bubble_bbox is not None:
            bx1, by1, bx2, by2 = [int(round(v)) for v in bubble_bbox]
        else:
            bx1, by1, bx2, by2 = tx1, ty1, tx2, ty2

        pad = 36
        rx1 = max(0, min(tx1, bx1) - pad)
        ry1 = max(0, min(ty1, by1) - pad)
        rx2 = min(W, max(tx2, bx2) + pad)
        ry2 = min(H, max(ty2, by2) + pad)

        roi_rgb = image_rgb[ry1:ry2, rx1:rx2].copy()
        rh, rw = roi_rgb.shape[:2]

        l_tb = (max(0, tx1 - rx1), max(0, ty1 - ry1), min(rw, tx2 - rx1), min(rh, ty2 - ry1))
        l_bb = (max(0, bx1 - rx1), max(0, by1 - ry1), min(rw, bx2 - rx1), min(rh, by2 - ry1))

        if bubble_bbox is not None:
            cavity_roi, wall_roi = self._extract_bubble_cavity_and_wall(roi_rgb, l_bb, l_tb)
        else:
            cavity_roi = np.zeros((rh, rw), dtype=np.uint8)
            wall_roi = np.zeros((rh, rw), dtype=np.uint8)

        local_neural = (
            neural_stroke_mask[ry1:ry2, rx1:rx2] if neural_stroke_mask is not None else None
        )
        stroke_core, obstacle_roi, median_sw = self._segment_strokes_and_obstacles(
            roi_rgb, l_tb, l_bb, cavity_roi, wall_roi, local_neural
        )

        halo_mask = self._extract_text_halo(roi_rgb, stroke_core, cavity_roi, median_sw)
        raw_text_mask = cv2.bitwise_or(stroke_core, halo_mask)

        protected_mask = cv2.bitwise_or(wall_roi, obstacle_roi)
        raw_text_mask = cv2.bitwise_and(raw_text_mask, cv2.bitwise_not(protected_mask))

        inside_seed = cv2.bitwise_and(raw_text_mask, cavity_roi)
        outside_seed = cv2.bitwise_and(
            raw_text_mask, cv2.bitwise_not(cv2.bitwise_or(cavity_roi, wall_roi))
        )

        cleaned_roi = roi_rgb.copy()

        if np.any(inside_seed > 0):
            opt_inside_mask, best_sigma, fill_color = self._grow_and_evaluate_mask_pcleaner(
                cleaned_roi, inside_seed, cavity_roi, protected_mask
            )
            cleaned_roi = self._route_inside_inpaint(
                cleaned_roi, opt_inside_mask, cavity_roi, best_sigma, fill_color
            )

        if np.any(outside_seed > 0):
            cleaned_roi = self._inpaint_outside_zero_bleed(
                cleaned_roi, outside_seed, cavity_roi, wall_roi, median_sw
            )

        out_rgb = image_rgb.copy()
        out_rgb[ry1:ry2, rx1:rx2] = cleaned_roi

        cavity_full = np.zeros((H, W), dtype=np.uint8)
        obstacle_full = np.zeros((H, W), dtype=np.uint8)
        cavity_full[ry1:ry2, rx1:rx2] = cavity_roi
        obstacle_full[ry1:ry2, rx1:rx2] = obstacle_roi
        return out_rgb, cavity_full, obstacle_full

    def _extract_bubble_cavity_and_wall(
        self,
        roi_rgb: np.ndarray,
        l_bb: Tuple[int, int, int, int],
        l_tb: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Извлекает внутреннюю полость баббла и его чёрную стенку (с защитой от разрывов)."""
        rh, rw = roi_rgb.shape[:2]
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        bx1, by1, bx2, by2 = l_bb
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)

        tx1, ty1, tx2, ty2 = l_tb
        tb_crop = gray[ty1:ty2, tx1:tx2]
        bg_bright = float(np.percentile(tb_crop, 85)) if tb_crop.size > 0 else 245.0
        is_light_bubble = bg_bright >= 128.0

        if is_light_bubble:
            _, dark_bin = cv2.threshold(
                gray, int(min(200, bg_bright - 35)), 255, cv2.THRESH_BINARY_INV
            )
        else:
            _, dark_bin = cv2.threshold(
                gray, int(max(55, bg_bright + 35)), 255, cv2.THRESH_BINARY
            )

        num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(dark_bin, connectivity=8)
        wall_candidates = np.zeros_like(dark_bin)
        for lbl in range(1, num_cc):
            cw, ch = stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]
            c_area = stats[lbl, cv2.CC_STAT_AREA]
            if cw > 0.50 * bw or ch > 0.50 * bh or c_area > 0.35 * (bw + bh) * 3:
                wall_candidates[labels == lbl] = 255

        gap_k = int(np.clip(min(bw, bh) * 0.05, 3, 15)) | 1
        closed_wall = cv2.morphologyEx(
            wall_candidates,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_k, gap_k)),
        )

        barrier = closed_wall.copy()
        cv2.rectangle(
            barrier,
            (max(0, bx1), max(0, by1)),
            (min(rw - 1, max(bx1 + 1, bx2 - 1)), min(rh - 1, max(by1 + 1, by2 - 1))),
            255,
            2,
        )

        inv_barrier = cv2.bitwise_not(barrier)
        ff_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
        ff_mask[1:-1, 1:-1] = barrier

        seed_x = max(0, min(rw - 1, (tx1 + tx2) // 2))
        seed_y = max(0, min(rh - 1, (ty1 + ty2) // 2))
        sy_list, sx_list = np.where(inv_barrier[ty1:ty2, tx1:tx2] > 0)
        if len(sx_list) > 0:
            dists = (sx_list + tx1 - seed_x) ** 2 + (sy_list + ty1 - seed_y) ** 2
            idx = int(np.argmin(dists))
            seed_pt = (int(sx_list[idx] + tx1), int(sy_list[idx] + ty1))
        else:
            seed_pt = (seed_x, seed_y)

        flood_canvas = np.zeros((rh, rw), dtype=np.uint8)
        cv2.floodFill(
            flood_canvas,
            ff_mask,
            seed_pt,
            255,
            loDiff=0,
            upDiff=0,
            flags=8 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY,
        )
        cavity_roi = ff_mask[1:-1, 1:-1].copy()
        cavity_roi[barrier > 0] = 0

        cnts, _ = cv2.findContours(cavity_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(cnts) > 0:
            largest_cnt = max(cnts, key=cv2.contourArea)
            cavity_filled = np.zeros_like(cavity_roi)
            cv2.drawContours(cavity_filled, [largest_cnt], -1, 255, -1)
        else:
            cavity_filled = np.zeros_like(cavity_roi)
            cv2.ellipse(
                cavity_filled,
                ((bx1 + bx2) // 2, (by1 + by2) // 2),
                (max(2, bw // 2 - 2), max(2, bh // 2 - 2)),
                0,
                0,
                360,
                255,
                -1,
            )

        cavity_border_band = cv2.subtract(
            cv2.dilate(cavity_filled, np.ones((7, 7), np.uint8)),
            cv2.erode(cavity_filled, np.ones((3, 3), np.uint8)),
        )
        true_wall = cv2.bitwise_and(dark_bin, cv2.bitwise_or(wall_candidates, cavity_border_band))
        cavity_clean = cv2.bitwise_and(cavity_filled, cv2.bitwise_not(true_wall))
        return cavity_clean, true_wall

    def _segment_strokes_and_obstacles(
        self,
        roi_rgb: np.ndarray,
        l_tb: Tuple[int, int, int, int],
        l_bb: Tuple[int, int, int, int],
        cavity_roi: np.ndarray,
        wall_roi: np.ndarray,
        local_neural: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Отделяет штрихи текста (включая касающиеся стенки иероглифы) от стенки баббла
        и от внутреннего чиби-арта по топологическим признакам (SWT, IQ = 4*pi*A/P^2, Solidity).
        """
        rh, rw = roi_rgb.shape[:2]
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        tx1, ty1, tx2, ty2 = l_tb
        tw_box, th_box = max(1, tx2 - tx1), max(1, ty2 - ty1)

        if local_neural is not None and np.any(local_neural > 127):
            candidate_bin = (local_neural > 127).astype(np.uint8) * 255
        else:
            k_bh = int(np.clip(min(tw_box, th_box) // 6, 5, 25)) | 1
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bh, k_bh))
            bhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
            that = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
            tb_bhat = bhat[ty1:ty2, tx1:tx2]
            tb_that = that[ty1:ty2, tx1:tx2]
            if tb_bhat.size > 0 and tb_that.size > 0:
                polarity_map = (
                    bhat if np.percentile(tb_bhat, 90) >= np.percentile(tb_that, 90) else that
                )
            else:
                polarity_map = bhat
            _, candidate_bin = cv2.threshold(
                polarity_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        candidate_bin = cv2.bitwise_and(candidate_bin, cv2.bitwise_not(wall_roi))

        tb_search = np.zeros((rh, rw), dtype=np.uint8)
        pad_tb = 14
        cv2.rectangle(
            tb_search,
            (max(0, tx1 - pad_tb), max(0, ty1 - pad_tb)),
            (min(rw, tx2 + pad_tb), min(rh, ty2 + pad_tb)),
            255,
            -1,
        )

        active_zone = cv2.bitwise_or(tb_search, cavity_roi)
        cc_input = cv2.bitwise_and(candidate_bin, active_zone)
        num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(cc_input, connectivity=8)

        stroke_mask = np.zeros((rh, rw), dtype=np.uint8)
        obstacle_mask = np.zeros((rh, rw), dtype=np.uint8)
        stroke_widths: List[float] = []

        for lbl in range(1, num_cc):
            cx_l, cy_l, cw, ch, area = stats[lbl]
            if area < 4 or (cw > 0.65 * tw_box and ch > 0.65 * th_box):
                continue
            if cx_l >= tx1 and cy_l >= ty1 and (cx_l + cw) <= tx2 and (cy_l + ch) <= ty2:
                comp = (labels == lbl).astype(np.uint8)
                dt = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
                max_sw = float(np.max(dt) * 2.0)
                if 1.5 <= max_sw <= 18.0:
                    stroke_widths.append(max_sw)

        median_sw = float(np.median(stroke_widths)) if len(stroke_widths) > 0 else 4.5

        for lbl in range(1, num_cc):
            cx_l, cy_l, cw, ch, area = stats[lbl]
            if area < 3:
                continue
            comp = (labels == lbl).astype(np.uint8) * 255
            in_tb_ratio = float(np.sum((comp > 0) & (tb_search > 0))) / float(area)

            dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            sw_max = float(np.max(dt) * 2.0)

            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            perim = float(cv2.arcLength(cnts[0], True)) if len(cnts) > 0 else 1.0
            iq = (4.0 * np.pi * area) / (perim * perim + 1e-5)
            hull_area = (
                float(cv2.contourArea(cv2.convexHull(cnts[0]))) if len(cnts) > 0 else float(area)
            )
            solidity = float(area) / max(1.0, hull_area)

            if in_tb_ratio < 0.25:
                mid_y = min(rh - 1, max(0, cy_l + ch // 2))
                mid_x = min(rw - 1, max(0, cx_l + cw // 2))
                if area >= 25 and cavity_roi[mid_y, mid_x] > 0:
                    obstacle_mask[labels == lbl] = 255
                continue

            is_massive_blob = (sw_max > max(16.0, 2.6 * median_sw)) and (iq > 0.72) and (area > 180)
            is_long_frame_line = (
                (max(cw, ch) > 0.85 * max(tw_box, th_box)) and (iq < 0.11) and (solidity < 0.15)
            )

            if is_massive_blob or is_long_frame_line:
                obstacle_mask[labels == lbl] = 255
            else:
                stroke_mask[labels == lbl] = 255

        return stroke_mask, obstacle_mask, median_sw

    def _extract_text_halo(
        self,
        roi_rgb: np.ndarray,
        stroke_core: np.ndarray,
        cavity_roi: np.ndarray,
        median_sw: float,
    ) -> np.ndarray:
        """
        Извлекает белую/контрастную обводку (halo) вокруг штрихов текста,
        выходящих за рамку баббла поверх тёмных волос или скринтона.
        """
        if not np.any(stroke_core > 0):
            return np.zeros_like(stroke_core)

        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        r_halo = int(np.clip(round(2.2 * median_sw), 5, 14))
        se_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_halo + 1, 2 * r_halo + 1))

        search_ring = cv2.subtract(cv2.dilate(stroke_core, se_halo), stroke_core)
        outside_ring = cv2.bitwise_and(search_ring, cv2.bitwise_not(cavity_roi))
        if not np.any(outside_ring > 0):
            return np.zeros_like(stroke_core)

        core_med = float(np.median(gray[stroke_core > 0]))
        if core_med < 128.0:
            halo_cand = ((gray >= 200) & (outside_ring > 0)).astype(np.uint8) * 255
        else:
            halo_cand = ((gray <= 55) & (outside_ring > 0)).astype(np.uint8) * 255

        touch_core = cv2.dilate(stroke_core, np.ones((3, 3), np.uint8))
        num_cc, labels, _, _ = cv2.connectedComponentsWithStats(halo_cand, connectivity=8)
        halo_verified = np.zeros_like(stroke_core)
        for lbl in range(1, num_cc):
            comp = labels == lbl
            if np.any(comp & (touch_core > 0)):
                halo_verified[comp] = 255

        return halo_verified

    def _grow_and_evaluate_mask_pcleaner(
        self,
        roi_rgb: np.ndarray,
        inside_seed: np.ndarray,
        cavity_roi: np.ndarray,
        protected_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Реализует алгоритм PanelCleaner (pcleaner/masker.py):
        итеративное выращивание маски от min_mask_thickness с проверкой std вдоль
        однопиксельного внешнего контура и остановкой перед стенкой баббла.
        """
        allowed_zone = cv2.bitwise_and(cavity_roi, cv2.bitwise_not(protected_mask))
        norm_rgb = roi_rgb.copy()
        off_white_mask = np.min(norm_rgb, axis=2) >= self.cfg.off_white_max_threshold
        norm_rgb[off_white_mask] = 255
        norm_gray = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        best_mask = inside_seed.copy()
        best_sigma = 999.0
        best_color = np.array([255, 255, 255], dtype=np.uint8)

        radii = [
            self.cfg.min_mask_thickness + i * self.cfg.mask_growth_step_pixels
            for i in range(self.cfg.mask_growth_steps)
        ]
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for r in radii:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            grown = cv2.dilate(inside_seed, se)
            grown_clamped = cv2.bitwise_and(grown, allowed_zone)

            outer_ring = cv2.subtract(cv2.dilate(grown_clamped, k3), grown_clamped)
            raw_ring = cv2.subtract(cv2.dilate(grown, k3), grown)
            ring_to_eval = cv2.bitwise_and(outer_ring, allowed_zone)
            if np.sum(ring_to_eval > 0) < 8:
                ring_to_eval = raw_ring

            ring_vals = norm_gray[ring_to_eval > 0]
            if ring_vals.size == 0:
                continue

            sigma = float(np.std(ring_vals))
            if sigma < best_sigma - self.cfg.mask_improvement_threshold:
                best_sigma = sigma
                best_mask = grown_clamped
                best_color = np.median(roi_rgb[ring_to_eval > 0], axis=0).astype(np.uint8)
                if np.all(best_color >= self.cfg.off_white_max_threshold):
                    best_color = np.array([255, 255, 255], dtype=np.uint8)
                if sigma == 0.0:
                    break
            elif sigma > best_sigma + 5.0 and best_sigma <= self.cfg.mask_max_std_dev:
                break

        return best_mask, best_sigma, best_color

    def _route_inside_inpaint(
        self,
        roi_rgb: np.ndarray,
        mask_inside: np.ndarray,
        cavity_roi: np.ndarray,
        sigma: float,
        fill_color: np.ndarray,
    ) -> np.ndarray:
        """
        Трёхуровневый Smart Inpaint Router для внутренней области баббла:
        1) sigma <= 0.25 -> Чистая аналитическая заливка цветом
        2) 0.25 < sigma <= 15.0 -> Заливка + NLM Edge Denoiser (пояс 5px)
        3) sigma > 15.0 -> Вызов Big-LaMa (или Navier-Stokes при отсутствии весов)
        """
        out = roi_rgb.copy()
        if sigma <= self.cfg.noise_min_std_dev:
            out[mask_inside > 0] = fill_color
            return out

        if sigma <= self.cfg.mask_max_std_dev:
            out[mask_inside > 0] = fill_color
            belt_r = self.cfg.noise_outline_size
            se_belt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * belt_r + 1, 2 * belt_r + 1))
            belt_mask = cv2.subtract(cv2.dilate(mask_inside, se_belt), mask_inside)
            belt_mask = cv2.bitwise_and(belt_mask, cavity_roi)
            if np.any(belt_mask > 0):
                denoised = cv2.fastNlMeansDenoisingColored(
                    out,
                    None,
                    h=self.cfg.filter_strength,
                    hColor=self.cfg.filter_strength,
                    templateWindowSize=7,
                    searchWindowSize=21,
                )
                alpha = cv2.GaussianBlur(belt_mask.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
                out = (
                    out.astype(np.float32) * (1.0 - alpha) + denoised.astype(np.float32) * alpha
                ).round().astype(np.uint8)
            return out

        extra_r = int(
            np.clip(
                self.cfg.min_inpainting_radius + self.cfg.inpainting_radius_multiplier * sigma,
                self.cfg.min_inpainting_radius,
                self.cfg.max_inpainting_radius,
            )
        )
        se_inp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * extra_r + 1, 2 * extra_r + 1))
        inp_mask = cv2.bitwise_and(cv2.dilate(mask_inside, se_inp), cavity_roi)

        if self.lama_inpainter is not None:
            inpainted = self.lama_inpainter(out, inp_mask)
        else:
            bgr_in = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            bgr_out = cv2.inpaint(bgr_in, inp_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
            inpainted = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

        out[inp_mask > 0] = inpainted[inp_mask > 0]
        return out

    def _inpaint_outside_zero_bleed(
        self,
        roi_rgb: np.ndarray,
        outside_seed: np.ndarray,
        cavity_roi: np.ndarray,
        wall_roi: np.ndarray,
        median_sw: float,
    ) -> np.ndarray:
        """
        Инпейнтинг текста и белого ореола (halo), выступающих ЗА пределы баббла
        поверх волос/одежды/фона с гарантией Zero Bleed (Context Shielding).
        """
        out = roi_rgb.copy()
        r_dil = int(np.clip(round(0.5 * median_sw) + 3, 4, 10))
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_dil + 1, 2 * r_dil + 1))
        bubble_total = cv2.bitwise_or(cavity_roi, wall_roi)

        # Дополнительно захватываем примыкающий белый ореол вокруг внешней маски (если он остался на границе)
        gray_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        halo_ring = cv2.subtract(
            cv2.dilate(outside_seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))),
            outside_seed,
        )
        adj_white_halo = ((gray_out >= 200) & (halo_ring > 0) & (bubble_total == 0)).astype(np.uint8) * 255
        expanded_seed = cv2.bitwise_or(outside_seed, adj_white_halo)

        mask_outside = cv2.bitwise_and(
            cv2.dilate(expanded_seed, se), cv2.bitwise_not(bubble_total)
        )
        if not np.any(mask_outside > 0):
            return out

        shielded_input = out.copy()
        ext_ring = cv2.subtract(
            cv2.dilate(mask_outside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))),
            cv2.bitwise_or(mask_outside, bubble_total),
        )
        if np.any(ext_ring > 0):
            bg_med = np.median(out[ext_ring > 0], axis=0).astype(np.uint8)
        else:
            bg_med = np.array([60, 60, 60], dtype=np.uint8)

        near_bubble = cv2.bitwise_and(
            bubble_total,
            cv2.dilate(mask_outside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))),
        )
        shielded_input[near_bubble > 0] = bg_med

        if self.lama_inpainter is not None:
            inpainted = self.lama_inpainter(shielded_input, mask_outside)
        else:
            bgr_in = cv2.cvtColor(shielded_input, cv2.COLOR_RGB2BGR)
            bgr_out = cv2.inpaint(bgr_in, mask_outside, inpaintRadius=5, flags=cv2.INPAINT_NS)
            inpainted = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

        dt_in = cv2.distanceTransform(mask_outside, cv2.DIST_L2, 3)
        alpha = np.clip(dt_in / 2.0, 0.0, 1.0)[..., None]
        blended = (
            out.astype(np.float32) * (1.0 - alpha) + inpainted.astype(np.float32) * alpha
        ).round().astype(np.uint8)
        out[mask_outside > 0] = blended[mask_outside > 0]
        return out


def split_conjoined_double_bubble_watershed(
    cavity_mask: np.ndarray,
    target_box_roi: tuple[int, int, int, int],
    other_boxes_roi: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Разделяет сдвоенные бабблы («восьмёрки» без черной перегородки) по узкому горлышку
    через маркерный водораздел (cv2.watershed) на инвертированном преобразовании расстояний (-DT).
    Сохраняет 100% полезной площади чаши в отличие от грубого вычитания bbox.
    """
    if not other_boxes_roi:
        return cavity_mask

    rh, rw = cavity_mask.shape[:2]
    dist = cv2.distanceTransform(cavity_mask, cv2.DIST_L2, 5)
    markers = np.zeros((rh, rw), dtype=np.int32)
    markers[cavity_mask == 0] = 1

    tx1, ty1, tx2, ty2 = target_box_roi
    tcx = max(0, min(rw - 1, (tx1 + tx2) // 2))
    tcy = max(0, min(rh - 1, (ty1 + ty2) // 2))
    tw_c = max(3, (tx2 - tx1) // 4)
    th_c = max(3, (ty2 - ty1) // 4)
    markers[
        max(0, tcy - th_c) : min(rh, tcy + th_c + 1),
        max(0, tcx - tw_c) : min(rw, tcx + tw_c + 1),
    ] = 2

    valid_other = 0
    for idx, (ox1, oy1, ox2, oy2) in enumerate(other_boxes_roi, start=3):
        ocx = max(0, min(rw - 1, (ox1 + ox2) // 2))
        ocy = max(0, min(rh - 1, (oy1 + oy2) // 2))
        ow_c = max(3, (ox2 - ox1) // 4)
        oh_c = max(3, (oy2 - oy1) // 4)
        sub_m = cavity_mask[
            max(0, ocy - oh_c) : min(rh, ocy + oh_c + 1),
            max(0, ocx - ow_c) : min(rw, ocx + ow_c + 1),
        ]
        if np.any(sub_m > 0):
            markers[
                max(0, ocy - oh_c) : min(rh, ocy + oh_c + 1),
                max(0, ocx - ow_c) : min(rw, ocx + ow_c + 1),
            ] = idx
            valid_other += 1

    if valid_other == 0:
        return cavity_mask

    markers[
        max(0, tcy - max(2, th_c // 2)) : min(rh, tcy + max(2, th_c // 2) + 1),
        max(0, tcx - max(2, tw_c // 2)) : min(rw, tcx + max(2, tw_c // 2) + 1),
    ] = 2

    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    relief_3c = cv2.cvtColor(255 - dist_norm, cv2.COLOR_GRAY2BGR)
    cv2.watershed(relief_3c, markers)

    isolated = np.where((markers == 2) & (cavity_mask > 0), 255, 0).astype(np.uint8)
    if np.count_nonzero(isolated) < 32:
        return cavity_mask
    return isolated


def find_optimal_box_center_integral(
    safe_mask_roi: np.ndarray,
    integral_img: np.ndarray,
    tw: int | float,
    th: int | float,
    centroid_y: float,
    centroid_x: float | None = None,
) -> tuple[int, int, float] | None:
    """
    Математически точная замена сырого Pole of Inaccessibility:
    Находит множество допустимых положений прямоугольника (tw, th) за O(1) через cv2.integral
    и выбирает центр с максимальным запасом расстояния до стенок баббла через distanceTransform
    с мягкой регуляризацией к центроиду полости.
    """
    if safe_mask_roi is None or integral_img is None:
        return None
    rh, rw = int(safe_mask_roi.shape[0]), int(safe_mask_roi.shape[1])
    tw_i = int(round(float(tw)))
    th_i = int(round(float(th)))
    if tw_i <= 0 or th_i <= 0 or tw_i >= rw or th_i >= rh:
        return None

    rect_sums = (
        integral_img[th_i:, tw_i:]
        - integral_img[:-th_i, tw_i:]
        - integral_img[th_i:, :-tw_i]
        + integral_img[:-th_i, :-tw_i]
    )
    valid_tl = (rect_sums == (tw_i * th_i)).astype(np.uint8) * 255
    if not np.any(valid_tl > 0):
        return None

    feasible_centers = np.zeros((rh, rw), dtype=np.uint8)
    y_off, x_off = int(th_i // 2), int(tw_i // 2)
    vh, vw = int(valid_tl.shape[0]), int(valid_tl.shape[1])
    feasible_centers[
        y_off : y_off + vh, x_off : x_off + vw
    ] = valid_tl

    margin_map = cv2.distanceTransform(feasible_centers, cv2.DIST_L2, 5)
    y_grid = np.arange(rh, dtype=np.float32)[:, None]
    gravity_pen = 1.2 * np.abs(y_grid - float(centroid_y)) / max(1.0, float(rh))
    if centroid_x is not None:
        x_grid = np.arange(rw, dtype=np.float32)[None, :]
        gravity_pen = gravity_pen + 0.6 * np.abs(x_grid - float(centroid_x)) / max(1.0, float(rw))

    score_map = np.where(feasible_centers > 0, margin_map - gravity_pen, -1e6)
    best_idx = int(np.argmax(score_map))
    opt_cy, opt_cx = divmod(best_idx, rw)
    best_margin = float(margin_map[opt_cy, opt_cx])
    return int(opt_cx), int(opt_cy), best_margin


def analyze_bubble_geometry(
    cleaned_img_np: np.ndarray,
    text_bbox: list[int] | tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    style_preset: str = "dialogue",
    is_free_text: bool = False,
    other_bboxes: list[list[int] | tuple[int, int, int, int]] | None = None,
    target_tw: int | None = None,
    target_th: int | None = None,
) -> dict[str, Any]:
    """
    Сегментирует реальную внутреннюю полость бабла вокруг `text_bbox` на очищенном изображении `cleaned_img_np`,
    отсекает узкий хвостик бабла через морфологическое открытие, разделяет сдвоенные баблы-«восьмёрки»
    через маркерный водораздел на `-DT` + барьер Вороного ТОЛЬКО при реальном перетекании пробного floodFill в ядро соседа,
    вычисляет `safe_mask`, интегральное изображение `integral_img`, точный профиль хорд (`chord_profile`, `chord_left_roi`,
    `chord_right_roi`) и находит оптимальный центр текстового блока через 2D Integral Image Feasibility + регуляризованный центроид.
    """
    x1, y1, x2, y2 = [int(round(float(c))) for c in text_bbox[:4]]
    x1 = max(0, min(x1, img_w - 4))
    y1 = max(0, min(y1, img_h - 4))
    x2 = max(x1 + 4, min(x2, img_w))
    y2 = max(y1 + 4, min(y2, img_h))
    box_w = max(4, x2 - x1)
    box_h = max(4, y2 - y1)
    preset = (style_preset or "dialogue").lower().strip()

    def _clamp_against_neighbors(
        sx1_in: int,
        sy1_in: int,
        sx2_in: int,
        sy2_in: int,
    ) -> tuple[int, int, int, int]:
        if not other_bboxes:
            return sx1_in, sy1_in, sx2_in, sy2_in
        cx_self = (x1 + x2) / 2.0
        cy_self = (y1 + y2) / 2.0
        nsx1, nsy1, nsx2, nsy2 = sx1_in, sy1_in, sx2_in, sy2_in
        for ob in other_bboxes:
            if not isinstance(ob, (list, tuple)) or len(ob) < 4:
                continue
            ox1, oy1, ox2, oy2 = [int(round(float(c))) for c in ob[:4]]
            if [ox1, oy1, ox2, oy2] == [x1, y1, x2, y2]:
                continue
            ix1 = max(nsx1, ox1)
            iy1 = max(nsy1, oy1)
            ix2 = min(nsx2, ox2)
            iy2 = min(nsy2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            cx_other = (ox1 + ox2) / 2.0
            cy_other = (oy1 + oy2) / 2.0
            dx = abs(cx_self - cx_other)
            dy = abs(cy_self - cy_other)
            if dy >= dx * 0.75:
                if cy_self < cy_other:
                    mid_y = int(round((min(y2, oy2) + max(y1, oy1)) / 2.0)) if y2 > oy1 else oy1
                    nsy2 = min(nsy2, max(nsy1 + 12, min(y2, max(oy1 - 3, mid_y - 2))))
                else:
                    mid_y = int(round((min(y2, oy2) + max(y1, oy1)) / 2.0)) if oy2 > y1 else oy2
                    nsy1 = max(nsy1, min(nsy2 - 12, max(y1, min(oy2 + 3, mid_y + 2))))
            else:
                if cx_self < cx_other:
                    mid_x = int(round((min(x2, ox2) + max(x1, ox1)) / 2.0)) if x2 > ox1 else ox1
                    nsx2 = min(nsx2, max(nsx1 + 12, min(x2, max(ox1 - 3, mid_x - 2))))
                else:
                    mid_x = int(round((min(x2, ox2) + max(x1, ox1)) / 2.0)) if ox2 > x1 else ox2
                    nsx1 = max(nsx1, min(nsx2 - 12, max(x1, min(ox2 + 3, mid_x + 2))))
        return nsx1, nsy1, nsx2, nsy2

    def _build_fallback_geometry() -> dict[str, Any]:
        if is_free_text or preset in ("narration", "sfx"):
            pad_x = max(2, int(box_w * 0.04)) if preset == "narration" else 2
            pad_y = max(2, int(box_h * 0.04)) if preset == "narration" else 2
        else:
            margin_ratio = 0.08 if preset == "shout" else 0.11
            pad_x = max(5, int(min(box_w, box_h) * margin_ratio * 0.85))
            pad_y = max(5, int(min(box_w, box_h) * margin_ratio * 0.85))
        sx1 = max(0, min(x2 - 4, x1 + pad_x))
        sy1 = max(0, min(y2 - 4, y1 + pad_y))
        sx2 = max(sx1 + 4, min(img_w, x2 - pad_x))
        sy2 = max(sy1 + 4, min(img_h, y2 - pad_y))
        sx1, sy1, sx2, sy2 = _clamp_against_neighbors(sx1, sy1, sx2, sy2)
        sw = max(4, sx2 - sx1)
        sh = max(4, sy2 - sy1)
        cx_opt = int(round((sx1 + sx2) / 2.0))
        cy_opt = int(round((sy1 + sy2) / 2.0 - sh * 0.022))
        cy_opt = max(sy1 + 2, min(sy2 - 2, cy_opt))
        return {
            "is_closed_bubble": False,
            "bubble_bbox": [x1, y1, x2, y2],
            "safe_bbox": [sx1, sy1, sx2, sy2],
            "safe_w": sw,
            "safe_h": sh,
            "optical_center": [cx_opt, cy_opt],
            "margin_px": max(4, min(pad_x, pad_y)),
            "chord_profile": None,
            "roi_offset": (x1, y1),
            "safe_mask_roi": None,
            "integral_img": None,
            "chord_left_roi": None,
            "chord_right_roi": None,
            "centroid_y_roi": float(sh) / 2.0,
            "centroid_x_roi": float(sw) / 2.0,
        }

    if is_free_text or preset == "sfx" or cleaned_img_np.size == 0:
        return _build_fallback_geometry()

    is_narrow_col = box_w < box_h * 0.35
    expand_x = max(72 if is_narrow_col else 36, int(box_w * (1.35 if is_narrow_col else 0.55)))
    expand_y = max(40, int(box_h * 0.48))
    rx1 = max(0, x1 - expand_x)
    ry1 = max(0, y1 - expand_y)
    rx2 = min(img_w, x2 + expand_x)
    ry2 = min(img_h, y2 + expand_y)
    rw = rx2 - rx1
    rh = ry2 - ry1
    if rw < 16 or rh < 16:
        return _build_fallback_geometry()

    roi = cleaned_img_np[ry1:ry2, rx1:rx2]
    if roi.ndim == 3 and roi.shape[2] >= 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        roi_gray = roi.astype(np.uint8)

    tb_x1 = max(0, x1 - rx1)
    tb_y1 = max(0, y1 - ry1)
    tb_x2 = min(rw, x2 - rx1)
    tb_y2 = min(rh, y2 - ry1)
    center_crop = roi_gray[tb_y1:tb_y2, tb_x1:tb_x2]
    if center_crop.size == 0:
        return _build_fallback_geometry()

    median_lum = float(np.median(center_crop))
    if median_lum >= 170.0:
        lum_mask = (roi_gray >= 210).astype(np.uint8) * 255
    elif median_lum <= 75.0:
        lum_mask = (roi_gray <= 65).astype(np.uint8) * 255
    else:
        return _build_fallback_geometry()

    edges = cv2.Canny(roi_gray, 50, 140)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close, iterations=1)
    cavity_candidates = cv2.bitwise_and(lum_mask, cv2.bitwise_not(closed_edges))

    core_pad_x = max(1, int((tb_x2 - tb_x1) * 0.18))
    core_pad_y = max(1, int((tb_y2 - tb_y1) * 0.18))
    seed_x1 = min(tb_x2 - 1, tb_x1 + core_pad_x)
    seed_y1 = min(tb_y2 - 1, tb_y1 + core_pad_y)
    seed_x2 = max(seed_x1 + 1, tb_x2 - core_pad_x)
    seed_y2 = max(seed_y1 + 1, tb_y2 - core_pad_y)

    seed_cx = (tb_x1 + tb_x2) // 2
    seed_cy = (tb_y1 + tb_y2) // 2
    if cavity_candidates[seed_cy, seed_cx] == 0:
        cavity_candidates[seed_y1:seed_y2, seed_x1:seed_x2] = lum_mask[seed_y1:seed_y2, seed_x1:seed_x2]

    max_leak_x = max(48, int(box_w * 0.85)) if is_narrow_col else 2
    max_leak_y = max(6, int(box_h * 0.04))
    lim_x1 = max(0, tb_x1 - max_leak_x)
    lim_y1 = max(0, tb_y1 - max_leak_y)
    lim_x2 = min(rw, tb_x2 + max_leak_x)
    lim_y2 = min(rh, tb_y2 + max_leak_y)
    if lim_x1 > 0:
        cavity_candidates[:, :lim_x1] = 0
    if lim_x2 < rw:
        cavity_candidates[:, lim_x2:] = 0
    if lim_y1 > 0:
        cavity_candidates[:lim_y1, :] = 0
    if lim_y2 < rh:
        cavity_candidates[lim_y2:, :] = 0

    voronoi_cut_mask = np.zeros((rh, rw), dtype=bool)
    ff_probe = cavity_candidates.copy()
    ff_probe_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
    if ff_probe[seed_cy, seed_cx] > 0:
        cv2.floodFill(ff_probe, ff_probe_mask, (int(seed_cx), int(seed_cy)), 128)
        unsplit_fill = ff_probe_mask[1:-1, 1:-1] > 0
    else:
        return _build_fallback_geometry()

    leaking_neighbor_boxes_roi: list[tuple[int, int, int, int]] = []
    if other_bboxes:
        cx_self_roi = (tb_x1 + tb_x2) / 2.0
        cy_self_roi = (tb_y1 + tb_y2) / 2.0
        yy_grid, xx_grid = np.ogrid[:rh, :rw]
        for ob in other_bboxes:
            if not isinstance(ob, (list, tuple)) or len(ob) < 4:
                continue
            ox1, oy1, ox2, oy2 = [int(round(float(c))) for c in ob[:4]]
            if [ox1, oy1, ox2, oy2] == [x1, y1, x2, y2]:
                continue
            ob_rx1 = max(0, ox1 - rx1)
            ob_ry1 = max(0, oy1 - ry1)
            ob_rx2 = min(rw, ox2 - rx1)
            ob_ry2 = min(rh, oy2 - ry1)
            if ob_rx2 > ob_rx1 and ob_ry2 > ob_ry1:
                in_pad_x = max(1, int((ox2 - ox1) * 0.15))
                in_pad_y = max(1, int((oy2 - oy1) * 0.15))
                core_ox1 = max(0, (ox1 + in_pad_x) - rx1)
                core_oy1 = max(0, (oy1 + in_pad_y) - ry1)
                core_ox2 = min(rw, (ox2 - in_pad_x) - rx1)
                core_oy2 = min(rh, (oy2 - in_pad_y) - ry1)
                if core_ox2 <= core_ox1 or core_oy2 <= core_oy1:
                    continue
                if not np.any(unsplit_fill[core_oy1:core_oy2, core_ox1:core_ox2]):
                    continue

                leaking_neighbor_boxes_roi.append((ob_rx1, ob_ry1, ob_rx2, ob_ry2))
                cx_oth_roi = ((ox1 + ox2) / 2.0) - rx1
                cy_oth_roi = ((oy1 + oy2) / 2.0) - ry1
                vx = cx_oth_roi - cx_self_roi
                vy = cy_oth_roi - cy_self_roi
                v_norm_sq = vx * vx + vy * vy
                if v_norm_sq >= 64.0:
                    mid_x = (cx_self_roi + cx_oth_roi) / 2.0
                    mid_y = (cy_self_roi + cy_oth_roi) / 2.0
                    if abs(vy) >= abs(vx) * 1.15:
                        if cy_self_roi < cy_oth_roi and tb_y2 <= ob_ry1 + 8:
                            mid_y = (tb_y2 + ob_ry1) / 2.0
                        elif cy_self_roi > cy_oth_roi and ob_ry2 <= tb_y1 + 8:
                            mid_y = (ob_ry2 + tb_y1) / 2.0
                        half_plane = ((yy_grid - mid_y) * vy + (xx_grid * 0.0)) >= 0.0
                    else:
                        if cx_self_roi < cx_oth_roi and tb_x2 <= ob_rx1 + 8:
                            mid_x = (tb_x2 + ob_rx1) / 2.0
                        elif cx_self_roi > cx_oth_roi and ob_rx2 <= tb_x1 + 8:
                            mid_x = (ob_rx2 + tb_x1) / 2.0
                        half_plane = ((xx_grid - mid_x) * vx + (yy_grid * 0.0)) >= 0.0
                    half_plane[seed_y1:seed_y2, seed_x1:seed_x2] = False
                    voronoi_cut_mask |= half_plane

    if leaking_neighbor_boxes_roi:
        unsplit_u8 = unsplit_fill.astype(np.uint8) * 255
        ws_split = split_conjoined_double_bubble_watershed(
            unsplit_u8,
            (tb_x1, tb_y1, tb_x2, tb_y2),
            leaking_neighbor_boxes_roi,
        )
        ws_split[voronoi_cut_mask] = 0
        if np.count_nonzero(ws_split) >= 32 and ws_split[seed_cy, seed_cx] > 0:
            ff_work = ws_split.copy()
            ff_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
            cv2.floodFill(ff_work, ff_mask, (int(seed_cx), int(seed_cy)), 128)
            bubble_mask = (ff_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
        else:
            cavity_candidates[voronoi_cut_mask] = 0
            ff_work = cavity_candidates.copy()
            ff_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
            if ff_work[seed_cy, seed_cx] > 0:
                cv2.floodFill(ff_work, ff_mask, (int(seed_cx), int(seed_cy)), 128)
                bubble_mask = (ff_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
            else:
                bubble_mask = unsplit_u8
    else:
        bubble_mask = unsplit_fill.astype(np.uint8) * 255

    k_dent = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_CLOSE, k_dent, iterations=1)
    inv_bub = cv2.bitwise_not(bubble_mask)
    n_inv, lbl_inv, st_inv, _ = cv2.connectedComponentsWithStats(inv_bub, connectivity=8)
    for idx_i in range(1, n_inv):
        ix = st_inv[idx_i, cv2.CC_STAT_LEFT]
        iy = st_inv[idx_i, cv2.CC_STAT_TOP]
        iw = st_inv[idx_i, cv2.CC_STAT_WIDTH]
        ih = st_inv[idx_i, cv2.CC_STAT_HEIGHT]
        if ix > 0 and iy > 0 and (ix + iw) < rw and (iy + ih) < rh:
            bubble_mask[lbl_inv == idx_i] = 255

    if median_lum >= 170.0:
        rem_dark = ((roi_gray < 160) & (bubble_mask > 0)).astype(np.uint8) * 255
        if np.any(rem_dark > 0):
            n_ob, lbl_ob, st_ob, _ = cv2.connectedComponentsWithStats(rem_dark, connectivity=8)
            obs_mask = np.zeros_like(bubble_mask)
            for idx_o in range(1, n_ob):
                oa = st_ob[idx_o, cv2.CC_STAT_AREA]
                if oa >= 45:
                    obs_mask[lbl_ob == idx_o] = 255
            if np.any(obs_mask > 0):
                obs_dil = cv2.dilate(
                    obs_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                )
                cand_sub = cv2.bitwise_and(bubble_mask, cv2.bitwise_not(obs_dil))
                if np.count_nonzero(cand_sub) >= 0.55 * np.count_nonzero(bubble_mask):
                    bubble_mask = cand_sub

    area_bubble = float(np.count_nonzero(bubble_mask))
    area_text = float(max(1, box_w * box_h))
    eff_ref_area = float(max(area_text, min(rw * rh * 0.45, max(box_w, int(box_h * 0.75)) * box_h)))
    top_touch = int(np.count_nonzero(bubble_mask[0, :] > 0))
    bot_touch = int(np.count_nonzero(bubble_mask[rh - 1, :] > 0))
    left_touch = int(np.count_nonzero(bubble_mask[:, 0] > 0))
    right_touch = int(np.count_nonzero(bubble_mask[:, rw - 1] > 0))
    perimeter_touch_ratio = (top_touch + bot_touch + left_touch + right_touch) / float(max(1, 2 * (rw + rh)))

    max_allowed_area = 3.8 * eff_ref_area if perimeter_touch_ratio <= 0.02 else 3.8 * area_text
    if perimeter_touch_ratio > 0.25 or area_bubble > max_allowed_area or area_bubble < 0.45 * area_text:
        return _build_fallback_geometry()

    ys_b, xs_b = np.where(bubble_mask > 0)
    if len(xs_b) < 16:
        return _build_fallback_geometry()
    raw_bw = int(np.max(xs_b) - np.min(xs_b) + 1)
    raw_bh = int(np.max(ys_b) - np.min(ys_b) + 1)

    open_r = max(5, min(raw_bw, raw_bh) // 11)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1))
    core_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_OPEN, k_open, iterations=1)
    if np.count_nonzero(core_mask) < 0.45 * area_bubble:
        core_mask = bubble_mask.copy()

    n_c, lbl_c, st_c, _ = cv2.connectedComponentsWithStats(core_mask, connectivity=8)
    if n_c > 2:
        seed_lbl = int(lbl_c[min(rh - 1, max(0, seed_cy)), min(rw - 1, max(0, seed_cx))])
        if seed_lbl >= 1 and st_c[seed_lbl, cv2.CC_STAT_AREA] >= 0.25 * area_text:
            best_lbl = seed_lbl
        else:
            best_lbl = 1 + int(np.argmax(st_c[1:, cv2.CC_STAT_AREA]))
        core_mask = np.where(lbl_c == best_lbl, 255, 0).astype(np.uint8)

    ys_c, xs_c = np.where(core_mask > 0)
    if len(xs_c) < 16:
        return _build_fallback_geometry()

    bx1_roi, bx2_roi = int(np.min(xs_c)), int(np.max(xs_c) + 1)
    by1_roi, by2_roi = int(np.min(ys_c)), int(np.max(ys_c) + 1)
    bubble_w = max(8, bx2_roi - bx1_roi)
    bubble_h = max(8, by2_roi - by1_roi)

    dist_map = cv2.distanceTransform(core_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
    if max_val <= 1.0:
        return _build_fallback_geometry()

    pole_mask = dist_map >= (0.68 * float(max_val))
    pole_ys, pole_xs = np.where(pole_mask)
    if len(pole_xs) > 0:
        weights = dist_map[pole_ys, pole_xs].astype(np.float64)
        w_sum = float(np.sum(weights))
        cx_roi = float(np.sum(pole_xs * weights) / w_sum) if w_sum > 0 else float(max_loc[0])
        cy_roi = float(np.sum(pole_ys * weights) / w_sum) if w_sum > 0 else float(max_loc[1])
    else:
        cx_roi, cy_roi = float(max_loc[0]), float(max_loc[1])

    if preset == "shout":
        margin_px = max(5, int(min(bubble_w, bubble_h) * 0.068))
    elif preset == "narration":
        margin_px = max(6, int(min(bubble_w, bubble_h) * 0.078))
    else:
        margin_px = max(5, int(min(bubble_w, bubble_h) * 0.084))

    eff_margin = min(float(margin_px), float(max_val) * 0.44)
    if np.any(voronoi_cut_mask):
        unsplit_cavity = cv2.bitwise_or(
            core_mask, ((voronoi_cut_mask) & (lum_mask > 0)).astype(np.uint8) * 255
        )
        dist_wall = cv2.distanceTransform(unsplit_cavity, cv2.DIST_L2, 5)
        vor_margin = max(2.5, eff_margin * 0.32)
        safe_mask = ((dist_wall >= eff_margin) & (dist_map >= vor_margin)).astype(np.uint8) * 255
    else:
        safe_mask = (dist_map >= eff_margin).astype(np.uint8) * 255

    ys_s, xs_s = np.where(safe_mask > 0)
    if len(xs_s) < 12:
        safe_mask = core_mask
        ys_s, xs_s = ys_c, xs_c

    sx1_roi, sx2_roi = int(np.min(xs_s)), int(np.max(xs_s) + 1)
    sy1_roi, sy2_roi = int(np.min(ys_s)), int(np.max(ys_s) + 1)

    bub_x1_abs = rx1 + bx1_roi
    bub_y1_abs = ry1 + by1_roi
    bub_x2_abs = rx1 + bx2_roi
    bub_y2_abs = ry1 + by2_roi
    if not is_narrow_col:
        bub_x1_abs = max(x1, bub_x1_abs)
        bub_x2_abs = min(x2, bub_x2_abs)
    bub_y1_abs = max(y1, bub_y1_abs)
    bub_y2_abs = min(y2, bub_y2_abs)

    sx1_abs = max(0, rx1 + sx1_roi)
    sy1_abs = max(0, ry1 + sy1_roi)
    sx2_abs = min(img_w, rx1 + sx2_roi)
    sy2_abs = min(img_h, ry1 + sy2_roi)

    sx1_abs = max(sx1_abs, bub_x1_abs + 4)
    sy1_abs = max(sy1_abs, bub_y1_abs + 4)
    sx2_abs = min(sx2_abs, max(sx1_abs + 12, bub_x2_abs - 4))
    sy2_abs = min(sy2_abs, max(sy1_abs + 12, bub_y2_abs - 4))

    safe_w = max(12, sx2_abs - sx1_abs)
    safe_h = max(12, sy2_abs - sy1_abs)

    mask_01 = (safe_mask > 0).astype(np.uint8)
    integral_img = cv2.integral(mask_01, sdepth=cv2.CV_32S)
    centroid_x_roi = float(np.mean(xs_s))
    centroid_y_roi = float(np.mean(ys_s))

    safe_mid_x = (sx1_abs + sx2_abs) / 2.0
    safe_mid_y = (sy1_abs + sy2_abs) / 2.0
    mask_cent_x = float(rx1) + centroid_x_roi
    mask_cent_y = float(ry1) + centroid_y_roi
    base_cx = 0.55 * safe_mid_x + 0.45 * mask_cent_x
    base_cy = 0.55 * safe_mid_y + 0.45 * mask_cent_y

    est_tw = int(target_tw) if (target_tw and target_tw > 8) else max(10, int(round(safe_w * 0.68)))
    est_th = int(target_th) if (target_th and target_th > 8) else max(10, int(round(safe_h * 0.58)))
    integral_res = find_optimal_box_center_integral(
        safe_mask_roi=safe_mask,
        integral_img=integral_img,
        tw=min(rw - 2, est_tw),
        th=min(rh - 2, est_th),
        centroid_y=centroid_y_roi,
        centroid_x=centroid_x_roi,
    )
    if integral_res is not None:
        feas_abs_x = float(rx1 + integral_res[0])
        feas_abs_y = float(ry1 + integral_res[1])
    else:
        feas_abs_x = float(rx1) + cx_roi
        feas_abs_y = float(ry1) + cy_roi

    raw_opt_cx = 0.72 * base_cx + 0.28 * feas_abs_x
    raw_opt_cy = 0.72 * base_cy + 0.28 * feas_abs_y - safe_h * 0.012

    max_drift_x = min(12.0, safe_w * 0.08)
    max_drift_y = min(14.0, safe_h * 0.08)
    optical_cx = int(round(max(safe_mid_x - max_drift_x, min(safe_mid_x + max_drift_x, raw_opt_cx))))
    optical_cy = int(round(max(safe_mid_y - max_drift_y, min(safe_mid_y + max_drift_y, raw_opt_cy))))
    optical_cx = max(sx1_abs + 4, min(sx2_abs - 4, optical_cx))
    optical_cy = max(sy1_abs + 4, min(sy2_abs - 4, optical_cy))
    opt_cx_roi = max(0, min(rw - 1, optical_cx - rx1))

    chord_profile: list[int] = []
    chord_left_roi = np.full(rh, rw, dtype=np.int32)
    chord_right_roi = np.zeros(rh, dtype=np.int32)

    for y_r in range(rh):
        row_xs = np.where(safe_mask[y_r, :] > 0)[0]
        if len(row_xs) >= 2:
            splits = np.where(np.diff(row_xs) > 2)[0]
            run_starts = np.insert(row_xs[splits + 1], 0, row_xs[0])
            run_ends = np.append(row_xs[splits], row_xs[-1])
            best_s, best_e = int(run_starts[0]), int(run_ends[0])
            best_dist = 1e9
            best_w = 0
            for r_s, r_e in zip(run_starts, run_ends):
                dist_c = 0.0 if (r_s <= opt_cx_roi <= r_e) else float(min(abs(opt_cx_roi - r_s), abs(opt_cx_roi - r_e)))
                run_w = int(r_e - r_s + 1)
                if dist_c < best_dist or (dist_c == best_dist and run_w > best_w):
                    best_dist = dist_c
                    best_w = run_w
                    best_s, best_e = int(r_s), int(r_e)
            chord_left_roi[y_r] = best_s
            chord_right_roi[y_r] = best_e

    for y_abs in range(sy1_abs, sy2_abs):
        y_r = max(0, min(rh - 1, y_abs - ry1))
        r_s = int(chord_left_roi[y_r])
        r_e = int(chord_right_roi[y_r])
        if r_e > r_s:
            run_w = r_e - r_s + 1
            if r_s <= opt_cx_roi <= r_e:
                sym_w = 2 * int(min(opt_cx_roi - r_s, r_e - opt_cx_roi)) + 1
                best_cw = int(round(0.30 * sym_w + 0.70 * run_w))
            else:
                best_cw = run_w
            cw_y = max(12, min(safe_w, best_cw))
        else:
            rel_t = abs(2.0 * (y_abs - sy1_abs + 0.5) / max(1.0, float(safe_h)) - 1.0)
            rel_t = min(0.96, rel_t)
            cw_y = max(12, int(round(safe_w * ((1.0 - (rel_t ** 1.85)) ** (1.0 / 1.85)))))
        chord_profile.append(cw_y)

    return {
        "is_closed_bubble": True,
        "bubble_bbox": [bub_x1_abs, bub_y1_abs, bub_x2_abs, bub_y2_abs],
        "safe_bbox": [sx1_abs, sy1_abs, sx2_abs, sy2_abs],
        "safe_w": safe_w,
        "safe_h": safe_h,
        "optical_center": [optical_cx, optical_cy],
        "margin_px": int(round(eff_margin)),
        "chord_profile": chord_profile,
        "roi_offset": (rx1, ry1),
        "safe_mask_roi": safe_mask,
        "integral_img": integral_img,
        "chord_left_roi": chord_left_roi,
        "chord_right_roi": chord_right_roi,
        "centroid_y_roi": centroid_y_roi,
        "centroid_x_roi": centroid_x_roi,
    }


class BubbleCleaner:
    """Очищает текстовые бабблы и надписи на странице по координатам Bounding Boxes с сохранением арта и LRU-кэшем."""

    def __init__(
        self,
        use_lama: bool = True,
        dilate_size: int = 11,
        box_margin_ratio: float = 0.03,
        crop_padding: int = 48,
        cache_max_pages: int = 8,
        disk_cache_dir: str | Path | None = None,
        models_dir: str | Path | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        if "dilate_Size" in kwargs and kwargs["dilate_Size"] is not None:
            dilate_size = int(kwargs["dilate_Size"])

        self.dilate_size = max(1, int(dilate_size))
        self.box_margin_ratio = box_margin_ratio
        self.crop_padding = max(16, int(crop_padding))
        self.cache_max_pages = max(1, int(cache_max_pages))
        self.models_dir = (
            Path(models_dir).resolve()
            if models_dir is not None
            else (Path(__file__).resolve().parent / "models")
        )
        has_cuda = bool(torch is not None and torch.cuda.is_available())
        self.device = device or ("cuda" if has_cuda else "cpu")
        self._cache_lock = threading.Lock()
        self._cleaned_lru_cache: OrderedDict[tuple[str, str], Image.Image] = OrderedDict()
        self._base_page_cache: OrderedDict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._block_crop_cache: OrderedDict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._block_crop_cache_max = 256
        self._base_page_cache_max = 10
        self.cleaner_cfg = CleanerConfig()

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
            except Exception:
                try:
                    self.lama_model = BuiltInSimpleLama(device=self.device, models_dir=self.models_dir)
                    logger.info(
                        "Встроенная нейросетевая модель очистки LaMa (big-lama.pt) успешно подключена на %s!",
                        self.device,
                    )
                except Exception as builtin_exc:
                    logger.warning(
                        "Нейросетевая модель LaMa недоступна (%s). "
                        "Используется резервный алгоритм очистки (Smart Inpaint Router + OpenCV INPAINT_NS).",
                        builtin_exc,
                    )
                    self.lama_model = None

        self.sota_cleaner = SotaMangaCleaner(
            config=self.cleaner_cfg,
            lama_inpainter=self._invoke_lama_np if self.lama_model is not None else None,
        )

    def _invoke_lama_np(self, img_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """Обёртка вызова `self.lama_model` на numpy-массивах `(H, W, 3)` и `(H, W)`."""
        ch, cw = img_rgb.shape[:2]
        if self.lama_model is not None:
            try:
                crop_pil = Image.fromarray(img_rgb, mode="RGB")
                mask_pil = Image.fromarray(mask_u8, mode="L")
                inpainted_pil = self.lama_model(crop_pil, mask_pil).convert("RGB")
                if inpainted_pil.size != (cw, ch):
                    inpainted_pil = inpainted_pil.crop((0, 0, cw, ch))
                return np.array(inpainted_pil)
            except Exception as exc:
                logger.debug("Ошибка вызова LaMa на патче (%s), переключаемся на OpenCV INPAINT_NS.", exc)
        bgr_in = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        bgr_out = cv2.inpaint(bgr_in, mask_u8, inpaintRadius=5, flags=cv2.INPAINT_NS)
        return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    @staticmethod
    def analyze_bubble_geometry(
        cleaned_img_np: np.ndarray,
        text_bbox: list[int] | tuple[int, int, int, int],
        img_w: int,
        img_h: int,
        style_preset: str = "dialogue",
        is_free_text: bool = False,
        other_bboxes: list[list[int] | tuple[int, int, int, int]] | None = None,
    ) -> dict[str, Any]:
        """Публичный метод сегментации внутренней полости бабла и расчета оптического центра."""
        return analyze_bubble_geometry(
            cleaned_img_np=cleaned_img_np,
            text_bbox=text_bbox,
            img_w=img_w,
            img_h=img_h,
            style_preset=style_preset,
            is_free_text=is_free_text,
            other_bboxes=other_bboxes,
        )

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
            "clean_ver": "v4.0-sota",
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
            "clean_ver": "v4.0-sota",
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
        отсекая внешние углы бокса за пределами рамки баббла, заполняет внутренние отверстия
        иероглифов ДО морфологического замыкания и замыкает только компактные внутренние вмятины
        от тёмных иероглифов, касающихся стенок баббла (知, 露, 藤, 没, 要, 東海),
        НИКОГДА не перешагивая через черную стенку баббла на внешний белый ореол (halo)!
        """
        rh, rw = roi_gray.shape[:2]
        wall_thresh = max(150, int(bg_level - 52))
        light_pixels = (roi_gray >= wall_thresh).astype(np.uint8) * 255
        dark_wall_bin = cv2.bitwise_not(light_pixels)

        # Выделяем длинные барьерные линии стенки баббла, через которые MORPH_CLOSE не имеет права перешагивать
        num_d, lbl_d, st_d, _ = cv2.connectedComponentsWithStats(dark_wall_bin, connectivity=8)
        wall_barrier = np.zeros_like(dark_wall_bin)
        for ld in range(1, num_d):
            dw = st_d[ld, cv2.CC_STAT_WIDTH]
            dh = st_d[ld, cv2.CC_STAT_HEIGHT]
            if dw > rw * 0.45 or dh > rh * 0.45:
                wall_barrier[lbl_d == ld] = 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(light_pixels, connectivity=4)
        if num_labels <= 1:
            return light_pixels

        cx_mid, cy_mid = rw / 2.0, rh / 2.0
        best_label = -1
        best_score = -1.0

        c_x1, c_x2 = int(rw * 0.22), max(int(rw * 0.22) + 1, int(rw * 0.78))
        c_y1, c_y2 = int(rh * 0.18), max(int(rh * 0.18) + 1, int(rh * 0.82))
        center_slice = labels[c_y1:c_y2, c_x1:c_x2]

        selected_labels: list[int] = []
        for lbl in range(1, num_labels):
            area = float(stats[lbl, cv2.CC_STAT_AREA])
            center_hits = float(np.count_nonzero(center_slice == lbl))
            if area < (rh * rw * 0.03) and center_hits < (center_slice.size * 0.03):
                continue
            dist = ((centroids[lbl][0] - cx_mid) ** 2 + ((centroids[lbl][1] - cy_mid) ** 2)) ** 0.5
            lx = stats[lbl, cv2.CC_STAT_LEFT]
            ly = stats[lbl, cv2.CC_STAT_TOP]
            lw = stats[lbl, cv2.CC_STAT_WIDTH]
            lh = stats[lbl, cv2.CC_STAT_HEIGHT]
            edge_touches = sum([lx <= 0, ly <= 0, (lx + lw) >= rw, (ly + lh) >= rh])
            corner_penalty = area * 0.85 if (edge_touches >= 2 and center_hits < area * 0.28) else 0.0

            score = center_hits * 4.0 + area - dist * 4.0 - corner_penalty
            if score > best_score:
                best_score = score
                best_label = lbl
            if edge_touches == 0 and center_hits >= max(10.0, center_slice.size * 0.08):
                selected_labels.append(lbl)

        if best_label == -1:
            return light_pixels

        interior = np.zeros_like(light_pixels)
        interior[labels == best_label] = 255
        for lbl in selected_labels:
            interior[labels == lbl] = 255

        # 1. Заполняем внутренние замкнутые отверстия внутри иероглифов ДО морфологического замыкания
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

        # 2. Замыкаем только компактные внутренние вмятины от тёмных иероглифов, касающихся стенок баббла,
        # не присоединяя внешние белые ореолы (light_pixels != best_label) за черной стенкой!
        close_d = max(15, min(27, int(round(min(rw, rh) * 0.26)) | 1))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_d, close_d))
        padded = cv2.copyMakeBorder(interior, close_d, close_d, close_d, close_d, cv2.BORDER_CONSTANT, value=0)
        closed_pad = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, k_close, iterations=1)
        closed_int = closed_pad[close_d : close_d + rh, close_d : close_d + rw]

        inv2 = cv2.bitwise_not(closed_int)
        n_inv2, lbl_inv2, st_inv2, _ = cv2.connectedComponentsWithStats(inv2, connectivity=8)
        for idx2 in range(1, n_inv2):
            lx2 = st_inv2[idx2, cv2.CC_STAT_LEFT]
            ly2 = st_inv2[idx2, cv2.CC_STAT_TOP]
            lw2 = st_inv2[idx2, cv2.CC_STAT_WIDTH]
            lh2 = st_inv2[idx2, cv2.CC_STAT_HEIGHT]
            if lx2 > 0 and ly2 > 0 and (lx2 + lw2) < rw and (ly2 + lh2) < rh:
                closed_int[lbl_inv2 == idx2] = 255

        added = cv2.subtract(closed_int, interior)
        # Исключаем внешние светлые компоненты (ореолы вне баббла), которые не входили в interior initially
        ext_light = cv2.subtract(light_pixels, interior)
        added = cv2.bitwise_and(added, cv2.bitwise_not(ext_light))

        n_add, lbl_add, st_add, _ = cv2.connectedComponentsWithStats(added, connectivity=8)
        for idx_a in range(1, n_add):
            aw = st_add[idx_a, cv2.CC_STAT_WIDTH]
            ah = st_add[idx_a, cv2.CC_STAT_HEIGHT]
            a_area = st_add[idx_a, cv2.CC_STAT_AREA]
            fill_r = float(a_area) / max(1.0, float(aw * ah))
            is_compact_dent = (aw <= rw * 0.46 and ah <= rh * 0.46 and a_area <= rw * rh * 0.16)
            is_dense_char_dent = (fill_r >= 0.26 and aw <= rw * 0.70 and ah <= rh * 0.60 and a_area <= rw * rh * 0.25)
            if is_compact_dent or is_dense_char_dent:
                interior[lbl_add == idx_a] = 255

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
        raw = block.get("clean_bbox") or block.get("original_bbox") or block.get("bbox") or [0, 0, 0, 0]
        if isinstance(raw, (list, tuple)) and len(raw) >= 4:
            bx1, by1, bx2, by2 = [int(round(float(c))) for c in raw[:4]]
        else:
            return [0, 0, 0, 0]

        if not bool(block.get("bbox_modified", False)):
            raw_tb = block.get("text_bbox")
            if isinstance(raw_tb, (list, tuple)) and len(raw_tb) >= 4:
                try:
                    tx1, ty1, tx2, ty2 = [int(round(float(c))) for c in raw_tb[:4]]
                    if (tx2 - tx1) >= 8 and (ty2 - ty1) >= 8:
                        bx1 = min(bx1, tx1)
                        by1 = min(by1, ty1)
                        bx2 = max(bx2, tx2)
                        by2 = max(by2, ty2)
                except Exception:
                    pass
        return [bx1, by1, bx2, by2]

    def _get_effective_clean_bounds(
        self,
        block: dict[str, Any],
        img_w: int,
        img_h: int,
    ) -> tuple[int, int, int, int]:
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
        Строит точную бинарную маску (uint8, 0 или 255) по пикселям текста и его белого ореола (`halo`)
        внутри каждого активного Bounding Box с топологической защитой контуров бабблов (IQ, Solidity)
        и внутреннего чиби-арта (SWT / Ridge Distance Transform) и градиентным контролем по PanelCleaner.
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

            has_valid_tb = False
            tb_box = [x1, y1, x2, y2]
            if not bool(block.get("bbox_modified", False)):
                raw_tb = block.get("text_bbox")
                if isinstance(raw_tb, (list, tuple)) and len(raw_tb) >= 4:
                    try:
                        tx1, ty1, tx2, ty2 = [int(round(float(c))) for c in raw_tb[:4]]
                        if (tx2 - tx1) >= 8 and (ty2 - ty1) >= 8 and [tx1, ty1, tx2, ty2] != [x1, y1, x2, y2]:
                            has_valid_tb = True
                            tb_box = [tx1, ty1, tx2, ty2]
                    except Exception:
                        pass

            if is_free_text:
                ix1 = max(0, x1 - 6)
                iy1 = max(0, y1 - 6)
                ix2 = min(img_w, x2 + 6)
                iy2 = min(img_h, y2 + 6)
            elif has_valid_tb:
                ix1 = max(0, min(x1, tb_box[0] - 3))
                iy1 = max(0, min(y1, tb_box[1] - 3))
                ix2 = min(img_w, max(x2, tb_box[2] + 3))
                iy2 = min(img_h, max(y2, tb_box[3] + 3))
            else:
                scale_f = max(1.0, min(2.8, float(min(bw, bh)) / 64.0))
                pad_out = min(max(24, int(round(22 * scale_f))), max(10, int(min(bw, bh) * 0.22)))
                ix1 = max(0, x1 - pad_out)
                iy1 = max(0, y1 - pad_out)
                ix2 = min(img_w, x2 + pad_out)
                iy2 = min(img_h, y2 + pad_out)

            roi_rgb = image_rgb[iy1:iy2, ix1:ix2]
            roi_gray = gray_full[iy1:iy2, ix1:ix2]
            rh, rw = roi_gray.shape[:2]
            if rh < 6 or rw < 6:
                continue

            cx1_r = max(0, x1 - ix1)
            cy1_r = max(0, y1 - iy1)
            cx2_r = min(rw, x2 - ix1)
            cy2_r = min(rh, y2 - iy1)
            core_gray = roi_gray[cy1_r:cy2_r, cx1_r:cx2_r]
            if core_gray.size < 16:
                core_gray = roi_gray

            p75 = float(np.percentile(core_gray, 75))
            median_val = float(np.median(core_gray))

            if not is_free_text and p75 >= 210.0:
                core_int_safe = self._extract_bubble_interior_mask(core_gray, p75)
                interior_mask = np.zeros((rh, rw), dtype=np.uint8)
                interior_mask[cy1_r:cy2_r, cx1_r:cx2_r] = core_int_safe

                core_w = max(8, cx2_r - cx1_r)
                core_h = max(8, cy2_r - cy1_r)
                scale_f = max(1.0, min(2.8, float(min(core_w, core_h)) / 64.0))
                scale_area = scale_f * scale_f

                text_zone_mask = np.full((rh, rw), 255, dtype=np.uint8)
                hollow_arc_ring = np.zeros((rh, rw), dtype=np.uint8)
                if has_valid_tb:
                    pad_tb = 2
                    tz_x1 = max(0, tb_box[0] - ix1 - pad_tb)
                    tz_y1 = max(0, tb_box[1] - iy1 - pad_tb)
                    tz_x2 = min(rw, tb_box[2] - ix1 + pad_tb)
                    tz_y2 = min(rh, tb_box[3] - iy1 + pad_tb)
                    if (tz_x2 - tz_x1) >= 8 and (tz_y2 - tz_y1) >= 8:
                        text_zone_mask.fill(0)
                        text_zone_mask[tz_y1:tz_y2, tz_x1:tz_x2] = 255

                    strict_dark = (roi_gray < 155).astype(np.uint8) * 255
                    nl_s, lbl_s, st_s, _ = cv2.connectedComponentsWithStats(strict_dark, connectivity=8)
                    hollow_arc_mask = np.zeros_like(strict_dark)
                    for l_s in range(1, nl_s):
                        sx_s, sy_s, sw_s, sh_s, sa_s = st_s[l_s]
                        fill_r = float(sa_s) / max(1.0, float(sw_s * sh_s))
                        if (sw_s > core_w * 0.48 or sh_s > core_h * 0.48) and fill_r <= 0.18:
                            hollow_arc_mask[lbl_s == l_s] = 255
                    if np.count_nonzero(hollow_arc_mask) > 0:
                        inner_tb_core = np.zeros_like(strict_dark)
                        ic_x1 = max(0, tb_box[0] - ix1 + 8)
                        ic_y1 = max(0, tb_box[1] - iy1 + 8)
                        ic_x2 = min(rw, tb_box[2] - ix1 - 8)
                        ic_y2 = min(rh, tb_box[3] - iy1 - 8)
                        if ic_x2 > ic_x1 and ic_y2 > ic_y1:
                            inner_tb_core[ic_y1:ic_y2, ic_x1:ic_x2] = 255
                        hollow_arc_ring = cv2.bitwise_and(hollow_arc_mask, cv2.bitwise_not(inner_tb_core))
                        if np.count_nonzero(hollow_arc_ring) > 0:
                            arc_dil = cv2.dilate(
                                hollow_arc_ring, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                            )
                            interior_mask[arc_dil > 0] = 0
                            text_zone_mask[arc_dil > 0] = 0

                ink_thresh = max(165, int(p75 - 14))
                dark_all = (roi_gray < ink_thresh).astype(np.uint8) * 255
                binary = cv2.bitwise_and(dark_all, cv2.bitwise_and(interior_mask, text_zone_mask))

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                clean_roi_mask = np.zeros_like(binary)
                chibi_art_mask = np.zeros_like(binary)
                dt_dark = cv2.distanceTransform((roi_gray < 175).astype(np.uint8), cv2.DIST_L2, 3)
                stroke_widths_est: list[float] = []

                for label_idx in range(1, num_labels):
                    comp_x = stats[label_idx, cv2.CC_STAT_LEFT]
                    comp_y = stats[label_idx, cv2.CC_STAT_TOP]
                    comp_w = stats[label_idx, cv2.CC_STAT_WIDTH]
                    comp_h = stats[label_idx, cv2.CC_STAT_HEIGHT]
                    comp_area = stats[label_idx, cv2.CC_STAT_AREA]

                    if comp_area < 2:
                        continue

                    comp_dt = dt_dark[labels == label_idx]
                    max_dt_val = float(np.max(comp_dt)) if comp_dt.size > 0 else 1.5

                    if comp_area >= int(380 * scale_area):
                        if (
                            comp_dt.size > 0
                            and max_dt_val >= max(9.5, 5.2 * scale_f)
                            and float(np.percentile(comp_dt, 90)) >= max(5.2, 2.8 * scale_f)
                        ):
                            chibi_art_mask[labels == label_idx] = 255
                            continue

                    if not has_valid_tb:
                        touches_left = comp_x <= cx1_r + 1
                        touches_top = comp_y <= cy1_r + 1
                        touches_right = (comp_x + comp_w) >= cx2_r - 1
                        touches_bottom = (comp_y + comp_h) >= cy2_r - 1
                        border_touches = sum([touches_left, touches_top, touches_right, touches_bottom])

                        if border_touches >= 2 and (comp_w > core_w * 0.78 or comp_h > core_h * 0.78):
                            continue
                        if border_touches == 1 and (comp_w > core_w * 0.85 or comp_h > core_h * 0.85):
                            continue

                    clean_roi_mask[labels == label_idx] = 255
                    if 1.0 <= max_dt_val * 2.0 <= 16.0:
                        stroke_widths_est.append(max_dt_val * 2.0)

                median_sw = float(np.median(stroke_widths_est)) if stroke_widths_est else (4.2 * scale_f)

                if np.count_nonzero(chibi_art_mask) > 0:
                    chibi_d = max(9, int(round(12 * scale_f)) | 1)
                    chibi_dil = cv2.dilate(
                        chibi_art_mask,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (chibi_d, chibi_d)),
                    )
                    clean_roi_mask[chibi_dil > 0] = 0
                    text_zone_mask[chibi_dil > 0] = 0

                pure_white = (roi_gray >= max(220, int(p75 - 30))).astype(np.uint8) * 255
                sharp_dark = (roi_gray < min(155, int(p75 - 75))).astype(np.uint8) * 255
                dist_to_pw = cv2.distanceTransform(cv2.bitwise_not(pure_white), cv2.DIST_L2, 3)
                thin_sharp_dark = ((sharp_dark > 0) & (dist_to_pw <= 5.5 * scale_f)).astype(np.uint8) * 255

                gap_span = max(6, int(round(6 * scale_f)))
                k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_span, 1))
                k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_span))
                pw_gap_h = cv2.bitwise_and(
                    cv2.subtract(cv2.morphologyEx(thin_sharp_dark, cv2.MORPH_CLOSE, k_h), thin_sharp_dark),
                    pure_white,
                )
                pw_gap_v = cv2.bitwise_and(
                    cv2.subtract(cv2.morphologyEx(thin_sharp_dark, cv2.MORPH_CLOSE, k_v), thin_sharp_dark),
                    pure_white,
                )

                open_d = max(5, int(round(5 * scale_f)) | 1)
                open_pw = cv2.morphologyEx(
                    pure_white,
                    cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_d, open_d)),
                )
                narrow_pw = cv2.subtract(pure_white, open_pw)
                pw_gap_h = cv2.bitwise_and(pw_gap_h, narrow_pw)
                pw_gap_v = cv2.bitwise_and(pw_gap_v, narrow_pw)

                near_d = max(9, int(round(9 * scale_f)) | 1)
                k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_d, near_d))
                char_seeds = cv2.bitwise_and(
                    cv2.bitwise_or(pw_gap_h, pw_gap_v),
                    cv2.bitwise_and(cv2.dilate(pw_gap_h, k_near), cv2.dilate(pw_gap_v, k_near)),
                )

                dist_from_int = cv2.distanceTransform(cv2.bitwise_not(interior_mask), cv2.DIST_L2, 3)
                char_seeds = cv2.bitwise_and(char_seeds, (dist_from_int <= 22.0 * scale_f).astype(np.uint8) * 255)
                char_seeds = cv2.bitwise_and(char_seeds, text_zone_mask)

                cl_d = max(11, int(round(11 * scale_f)) | 1)
                char_clusters = cv2.morphologyEx(
                    char_seeds,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cl_d, cl_d)),
                )
                nc, lbl_c, st_c, _ = cv2.connectedComponentsWithStats(char_clusters, connectivity=8)
                valid_char_zones = np.zeros_like(char_clusters)
                for idx_c in range(1, nc):
                    bx_c, by_c, bw_c, bh_c, area_c = st_c[idx_c]
                    scnt = int(np.count_nonzero(char_seeds[lbl_c == idx_c]))
                    in_int = int(np.count_nonzero(interior_mask[lbl_c == idx_c]))
                    out_int = area_c - in_int
                    if (
                        (in_int >= int(22 * scale_area) and out_int >= int(14 * scale_area) and scnt >= int(20 * scale_area))
                        or (has_valid_tb and out_int >= int(12 * scale_area) and scnt >= int(10 * scale_area))
                    ):
                        ext_only = ((lbl_c == idx_c) & (interior_mask == 0)).astype(np.uint8) * 255
                        ne, lbl_e, st_e, _ = cv2.connectedComponentsWithStats(ext_only, connectivity=8)
                        for idx_e in range(1, ne):
                            ex, ey, ew, eh, earea = st_e[idx_e]
                            eseeds = int(np.count_nonzero(char_seeds[lbl_e == idx_e]))
                            if (
                                earea >= int(14 * scale_area)
                                and eseeds >= int(6 * scale_area)
                                and ew >= int(4 * scale_f)
                                and eh >= int(5 * scale_f)
                            ):
                                valid_char_zones[lbl_e == idx_e] = 255

                zd_d = max(9, int(round(9 * scale_f)) | 1)
                char_zone_dil = cv2.bitwise_and(
                    cv2.dilate(
                        valid_char_zones,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (zd_d, zd_d)),
                        iterations=1,
                    ),
                    text_zone_mask,
                )
                border_char_ink = cv2.bitwise_and(
                    dark_all,
                    cv2.bitwise_and(char_zone_dil, (dist_to_pw <= 6.5 * scale_f).astype(np.uint8) * 255),
                )

                # Геодезическое выделение белого ореола (halo) вокруг подтвержденных внешних штрихов
                if np.count_nonzero(border_char_ink) > 0:
                    halo_ext = self.sota_cleaner._extract_text_halo(
                        roi_rgb, border_char_ink, interior_mask, median_sw
                    )
                    halo_ext = cv2.bitwise_and(halo_ext, text_zone_mask)
                    border_char_ink = cv2.bitwise_or(border_char_ink, halo_ext)

                min_valid_ink_px = max(14, int(core_w * core_h * 0.008))
                if np.count_nonzero(clean_roi_mask) + np.count_nonzero(border_char_ink) >= min_valid_ink_px:
                    scaled_bub_d = max(bubble_dilate, int(round(9 * scale_f)) | 1)
                    eff_bub_kernel = (
                        block_kernel
                        if block_kernel is not None
                        else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scaled_bub_d, scaled_bub_d))
                    )
                    dilated_int = cv2.dilate(clean_roi_mask, eff_bub_kernel, iterations=1)
                    if block_kernel is None or custom_dilate <= 12:
                        allowed_int = cv2.bitwise_and(interior_mask, text_zone_mask)
                        protected_local = cv2.bitwise_or(hollow_arc_ring, chibi_art_mask)
                        pcleaner_mask, _, _ = self.sota_cleaner._grow_and_evaluate_mask_pcleaner(
                            roi_rgb, clean_roi_mask, allowed_int, protected_local
                        )
                        dilated_int = cv2.bitwise_and(
                            cv2.bitwise_or(dilated_int, pcleaner_mask), allowed_int
                        )

                    if np.count_nonzero(border_char_ink) > 0:
                        dilated_border_chars = cv2.bitwise_and(
                            cv2.dilate(border_char_ink, eff_bub_kernel, iterations=1),
                            text_zone_mask,
                        )
                        dilated_roi = cv2.bitwise_or(dilated_int, dilated_border_chars)
                    else:
                        dilated_roi = dilated_int

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
                        border_touches == 1 and (comp_w > rw * 0.78 or comp_h > rh * 0.78)
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

            if is_free_text:
                free_halo = self.sota_cleaner._extract_text_halo(
                    roi_rgb, clean_roi_mask, np.zeros_like(clean_roi_mask), 4.5
                )
                clean_roi_mask = cv2.bitwise_or(clean_roi_mask, free_halo)

            active_kernel = (
                block_kernel
                if block_kernel is not None
                else (kernel_free if is_free_text else kernel_bubble)
            )
            dilated_roi = cv2.dilate(clean_roi_mask, active_kernel, iterations=1)
            full_mask[iy1:iy2, ix1:ix2] = np.maximum(full_mask[iy1:iy2, ix1:ix2], dilated_roi)

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
        inner_box_rel: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """
        Двухзонный SOTA инпейнтинг кропа блока (Zero Bleed Guarantee + Smart Inpaint Router):
        - Разделяет `crop_mask_np` на `mask_inside` (внутри светлой полости баббла) и `mask_outside`
          (выступающие за контур баббла иероглифы с белым ореолом поверх волос/стен/скринтона).
        - `mask_inside` очищается через трёхуровневый маршрутизатор (чистая заливка при sigma <= 0.25,
          заливка + 5px пояс `fastNlMeansDenoisingColored` при 0.25 < sigma <= 15.0, LaMa/NS при sigma > 15.0).
        - `mask_outside` очищается отдельно с барьерным экранированием (`Context Shielding`) белой полости баббла,
          чтобы белый цвет баббла НИКОГДА не вытекал наружу на волосы, одежду или стены!
        """
        ch, cw = crop_rgb_np.shape[:2]
        if np.count_nonzero(crop_mask_np) == 0:
            return crop_rgb_np.copy()

        if not is_free_text:
            crop_gray = cv2.cvtColor(crop_rgb_np, cv2.COLOR_RGB2GRAY)
            if inner_box_rel is not None:
                bx1_r, by1_r, bx2_r, by2_r = inner_box_rel
                bx1_r = max(0, min(cw - 2, bx1_r))
                by1_r = max(0, min(ch - 2, by1_r))
                bx2_r = max(bx1_r + 2, min(cw, bx2_r))
                by2_r = max(by1_r + 2, min(ch, by2_r))
                core_sub = crop_gray[by1_r:by2_r, bx1_r:bx2_r]
            else:
                bx1_r, by1_r, bx2_r, by2_r = 0, 0, cw, ch
                core_sub = crop_gray

            p75_core = float(np.percentile(core_sub, 75)) if core_sub.size > 0 else 0.0
            if p75_core >= 210.0:
                cavity_sub = self._extract_bubble_interior_mask(core_sub, p75_core)
                cavity_sub = cv2.dilate(
                    cavity_sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
                )
                cavity_mask = np.zeros((ch, cw), dtype=np.uint8)
                cavity_mask[by1_r:by2_r, bx1_r:bx2_r] = cavity_sub
                wall_mask = ((crop_gray < max(150, int(p75_core - 55))) & (cavity_mask == 0)).astype(np.uint8) * 255

                mask_inside = cv2.bitwise_and(crop_mask_np, cavity_mask)
                mask_outside = cv2.bitwise_and(crop_mask_np, cv2.bitwise_not(cavity_mask))

                out = crop_rgb_np.copy()
                if np.any(mask_inside > 0):
                    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    ring_in = cv2.bitwise_and(
                        cv2.subtract(cv2.dilate(mask_inside, k3), mask_inside),
                        cavity_mask,
                    )
                    if np.count_nonzero(ring_in) < 8:
                        ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                        ring_in = cv2.bitwise_and(
                            cv2.subtract(cv2.dilate(mask_inside, ring_k), mask_inside),
                            cavity_mask,
                        )
                    ring_pixels = out[ring_in > 0] if np.any(ring_in > 0) else out[cavity_mask > 0]
                    if len(ring_pixels) >= 8:
                        norm_ring = ring_pixels.copy()
                        norm_ring[np.min(norm_ring, axis=1) >= self.cleaner_cfg.off_white_max_threshold] = 255
                        ring_gray_norm = np.mean(norm_ring, axis=1)
                        sigma_edge = float(np.std(ring_gray_norm))

                        raw_ring_gray = np.mean(ring_pixels, axis=1)
                        p50 = float(np.percentile(raw_ring_gray, 50))
                        p80 = float(np.percentile(raw_ring_gray, 80))
                        bright_px = ring_pixels[raw_ring_gray >= max(225.0, p50)]
                        if len(bright_px) >= 4:
                            fill_color = np.median(bright_px, axis=0).round().astype(np.uint8)
                        else:
                            fill_color = np.median(ring_pixels, axis=0).round().astype(np.uint8)
                        if float(np.min(fill_color)) >= self.cleaner_cfg.off_white_max_threshold:
                            fill_color = np.array([255, 255, 255], dtype=np.uint8)

                        if p50 >= 234.0 and p80 >= 245.0 and sigma_edge > self.cleaner_cfg.mask_max_std_dev:
                            sigma_edge = 8.0

                        out = self.sota_cleaner._route_inside_inpaint(
                            out, mask_inside, cavity_mask, sigma_edge, fill_color
                        )

                if np.any(mask_outside > 0):
                    out = self.sota_cleaner._inpaint_outside_zero_bleed(
                        out, mask_outside, cavity_mask, wall_mask, median_sw=4.5
                    )
                return out

            ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            ring_mask = cv2.dilate(crop_mask_np, ring_k, iterations=1) - crop_mask_np
            ring_pixels = crop_rgb_np[ring_mask > 0]
            if len(ring_pixels) >= 16:
                ring_gray = np.mean(ring_pixels, axis=1)
                p15 = float(np.percentile(ring_gray, 15))
                p50 = float(np.percentile(ring_gray, 50))
                ring_std = float(np.std(ring_gray))
                if p15 >= 232.0 and p50 >= 242.0 and ring_std <= 16.0:
                    bright_pixels = ring_pixels[ring_gray >= p15]
                    if len(bright_pixels) >= 8:
                        fill_color = np.median(bright_pixels, axis=0).round().astype(np.uint8)
                        if float(np.min(fill_color)) >= 240.0:
                            fill_color = np.array([255, 255, 255], dtype=np.uint8)
                        out = crop_rgb_np.copy()
                        out[crop_mask_np > 0] = fill_color
                        return out

        if self.lama_model is not None:
            try:
                lama_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                lama_mask_np = cv2.dilate(crop_mask_np, lama_k, iterations=1)
                crop_pil = Image.fromarray(crop_rgb_np, mode="RGB")
                mask_pil = Image.fromarray(lama_mask_np, mode="L")
                inpainted_pil = self.lama_model(crop_pil, mask_pil).convert("RGB")
                if inpainted_pil.size != (cw, ch):
                    inpainted_pil = inpainted_pil.crop((0, 0, cw, ch))
                return np.array(inpainted_pil)
            except Exception as lama_exc:
                logger.debug("Ошибка вызова LaMa на кропе (%s), переключаемся на OpenCV inpaint.", lama_exc)

        bgr_crop = cv2.cvtColor(crop_rgb_np, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(
            bgr_crop,
            crop_mask_np,
            inpaintRadius=7,
            flags=cv2.INPAINT_NS,
        )
        return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    def _apply_custom_strokes_fast(
        self,
        base_cleaned_np: np.ndarray,
        raw_img_np: np.ndarray,
        base_mask_np: np.ndarray,
        blocks: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
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
                if len(ring_pixels) >= 8:
                    ring_gray = np.mean(ring_pixels, axis=1)
                    p15 = float(np.percentile(ring_gray, 15))
                    p50 = float(np.percentile(ring_gray, 50))
                    ring_std = float(np.std(ring_gray))
                    if p15 >= 228.0 and p50 >= 238.0 and ring_std <= 18.0:
                        bright_pixels = ring_pixels[ring_gray >= p15]
                        if len(bright_pixels) >= 4:
                            fill_color = np.median(bright_pixels, axis=0).round().astype(np.uint8)
                            if float(np.min(fill_color)) >= 238.0:
                                fill_color = np.array([255, 255, 255], dtype=np.uint8)
                            filled_crop = crop_rgb.copy()
                            filled_crop[comp_mask > 0] = fill_color

                if filled_crop is None:
                    filled_crop = self._inpaint_crop(crop_rgb, comp_mask, is_free_text=True)

                alpha = cv2.GaussianBlur(comp_mask.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
                out_np[cy1:cy2, cx1:cx2] = (
                    filled_crop * alpha + crop_rgb * (1.0 - alpha)
                ).round().astype(np.uint8)

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
           с поблочным кэшированием кропов (`_block_crop_cache`) и двухзонной гарантией Zero Bleed.
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
            pad = self.crop_padding
            uncached_blocks: list[dict[str, Any]] = []
            skipped_blocks: list[dict[str, Any]] = []
            active_block_specs: list[
                tuple[
                    dict[str, Any],
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    bool,
                    tuple[Any, ...],
                    tuple[np.ndarray, np.ndarray] | None,
                ]
            ] = []

            for block in blocks:
                if self._is_block_skipped(block):
                    skipped_blocks.append(block)
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
                block_crop_key = (
                    img_fp,
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                    bool(is_free_text),
                    int(mdil),
                    int(self.dilate_size),
                    "v4.0-sota",
                )
                with self._cache_lock:
                    cached_crop_pair = self._block_crop_cache.get(block_crop_key)
                    if cached_crop_pair is not None:
                        self._block_crop_cache.move_to_end(block_crop_key)
                if (
                    cached_crop_pair is None
                    or cached_crop_pair[0].shape != (cy2 - cy1, cx2 - cx1, 3)
                    or cached_crop_pair[1].shape != (cy2 - cy1, cx2 - cx1)
                ):
                    uncached_blocks.append(block)
                    cached_crop_pair = None
                active_block_specs.append(
                    (block, x1, y1, x2, y2, cx1, cy1, cx2, cy2, is_free_text, block_crop_key, cached_crop_pair)
                )

            if not active_block_specs:
                base_mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)
                base_composited_np = img_np.copy()
            else:
                if uncached_blocks:
                    base_mask_np = self.build_text_mask(
                        img_np,
                        uncached_blocks + skipped_blocks,
                        include_custom_strokes=False,
                    )
                else:
                    base_mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)

                base_composited_np = img_np.copy()

                for (
                    block,
                    x1,
                    y1,
                    x2,
                    y2,
                    cx1,
                    cy1,
                    cx2,
                    cy2,
                    is_free_text,
                    block_crop_key,
                    cached_crop_pair,
                ) in active_block_specs:
                    crop_rgb = base_composited_np[cy1:cy2, cx1:cx2]
                    if cached_crop_pair is not None:
                        inpainted_crop, crop_mask = cached_crop_pair[0], cached_crop_pair[1]
                        base_mask_np[cy1:cy2, cx1:cx2] = np.maximum(
                            base_mask_np[cy1:cy2, cx1:cx2], crop_mask
                        )
                    else:
                        margin_extra = (
                            8 if is_free_text else min(pad - 2, max(12, int(min(x2 - x1, y2 - y1) * 0.22)))
                        )
                        bx1 = max(cx1, x1 - margin_extra)
                        by1 = max(cy1, y1 - margin_extra)
                        bx2 = min(cx2, x2 + margin_extra)
                        by2 = min(cy2, y2 + margin_extra)

                        crop_mask = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
                        crop_mask[by1 - cy1 : by2 - cy1, bx1 - cx1 : bx2 - cx1] = base_mask_np[by1:by2, bx1:bx2]
                        if np.count_nonzero(crop_mask) == 0:
                            continue

                        inpainted_crop = self._inpaint_crop(
                            crop_rgb,
                            crop_mask,
                            is_free_text=is_free_text,
                            inner_box_rel=(x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1),
                        )
                        with self._cache_lock:
                            self._block_crop_cache[block_crop_key] = (
                                inpainted_crop.copy(),
                                crop_mask.copy(),
                            )
                            while len(self._block_crop_cache) > self._block_crop_cache_max:
                                self._block_crop_cache.popitem(last=False)

                    core_f = (crop_mask > 0).astype(np.float32)
                    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    dil_f = cv2.dilate(crop_mask, dil_k, iterations=1).astype(np.float32) / 255.0
                    blur_f = cv2.GaussianBlur(dil_f, (5, 5), 0)
                    alpha = np.maximum(core_f, blur_f)[..., None]
                    blended_crop = (
                        inpainted_crop * alpha + crop_rgb * (1.0 - alpha)
                    ).round().astype(np.uint8)
                    base_composited_np[cy1:cy2, cx1:cx2] = blended_crop

            with self._cache_lock:
                self._base_page_cache[base_cache_key] = (
                    base_composited_np.copy(),
                    base_mask_np.copy(),
                )
                while len(self._base_page_cache) > self._base_page_cache_max:
                    self._base_page_cache.popitem(last=False)

        final_composited_np, final_mask_np = self._apply_custom_strokes_fast(
            base_composited_np,
            img_np,
            base_mask_np,
            blocks,
        )
        return Image.fromarray(final_composited_np, mode="RGB"), Image.fromarray(final_mask_np, mode="L")