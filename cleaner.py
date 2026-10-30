"""
Модуль очистки бабблов и свободных надписей от оригинального текста — Inpainting (v4.0 SOTA Edition).

Архитектура обновления v4.0 (SOTA Clean & Typeset):
1. Попиксельная сегментация штрихов текста и белого ореола (Stroke & Halo Segmentation):
   - Локализация стенок баббла M_wall и разделение штрихов, вплотную примыкающих к внутренней черной стенке (知, 露, 藤, 没),
     через евклидово преобразование расстояний D_dark.
   - Геодезическое извлечение контрастного белого ореола (halo) вокруг выступающих за пределы баббла символов (吧, 衣服干, 要, 人在, 如果愿意的话)
     с адаптивным радиусом поиска r_halo = clip(1.6 * w_s, 3, 12).
   - Топологические инварианты (IQ = 4*pi*A/P^2, Solidity, SWT) для надежной защиты контуров колючих/разомкнутых бабблов
     и массивного чиби-арта внутри баббла.
2. Итеративное выращивание маски по стандарту PanelCleaner (default.conf):
   - Оценка стандартного отклонения sigma(dM_k) на однопиксельном внешнем кольце с нормализацией околобелых пикселей (>= 240 -> 255)
     и автоматической остановкой перед черной стенкой баббла.
3. Трёхуровневый маршрутизатор очистки (Smart Inpaint Router) и гарантия Zero Bleed:
   - Внутренняя зона баббла M_inside:
     * sigma <= 0.25 -> Чистая аналитическая заливка цветом бумаги баббла.
     * 0.25 < sigma <= 15.0 -> Заливка + локальный NLM Edge Denoiser (пояс 5 px с плавным альфа-затуханием).
     * sigma > 15.0 -> Нейросетевой инпейнтинг LaMa (big-lama.pt) / PatchMatch / Navier-Stokes.
   - Внешняя зона M_outside (выступающий текст на волосах/скринтоне):
     * Context Shielding: экранирование белой полости баббла внешним медианным фоном перед инпейнтингом
       для 100% исключения белых пятен и ореолов на волосах и скринтонах.
4. Разделение сдвоенных бабблов («восьмерок») в analyze_bubble_geometry:
   - Маркерный водораздел / дистанционное поле между ядрами блоков строго по седловой линии перемычки без потери полезной площади.
5. Поиск оптимального размещения текста через 2D-интегральную свертку (Feasible Center Region):
   - Математически точная замена Pole of Inaccessibility, гарантирующая 0% вылета прямоугольного текстового блока за границы safe_mask.
6. 100% обратная совместимость с API server.py, gui_app.py и web_studio.html.
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
    torch = None

logger = logging.getLogger(__name__)

LAMA_WEIGHTS_URLS: tuple[str, ...] = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
    "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt",
)


@dataclass
class CleanerConfig:
    """Конфигурация параметров очистки и инпейнтинга по стандарту PanelCleaner."""
    min_mask_thickness: int = 4
    mask_growth_step_pixels: int = 2
    mask_growth_steps: int = 11
    off_white_max_threshold: int = 240
    mask_max_std_dev: float = 15.0
    mask_improvement_threshold: float = 0.1

    noise_min_std_dev: float = 0.25
    noise_outline_size: int = 5
    noise_fade_radius: int = 2
    filter_strength: float = 10.0

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
        device: Any = None,
        models_dir: str | Path | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch не установлен в текущем окружении Python.")

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
            base_models_dir / "big-lama.pt",
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
            raise RuntimeError("PyTorch не установлен в текущем окружении Python.")

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


# ==============================================================================
# 1. ГЕОМЕТРИЧЕСКАЯ СЕГМЕНТАЦИЯ БАББЛА (analyze_bubble_geometry)
# ==============================================================================

def analyze_bubble_geometry(
    cleaned_img_np: np.ndarray,
    text_bbox: list[int] | tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    style_preset: str = "dialogue",
    is_free_text: bool = False,
    other_bboxes: list[list[int] | tuple[int, int, int, int]] | None = None,
) -> dict[str, Any]:
    """
    Сегментирует реальную внутреннюю полость бабла вокруг `text_bbox` на очищенном изображении `cleaned_img_np`.
    - Разделяет сдвоенные баблы («восьмерки») через топографическое дистанционное поле вдоль седловой линии перемычки.
    - Вычитает обнаруженный внутренний чиби-арт (obstacle_mask).
    - Вычисляет 2D Integral Feasibility карту для оптимального центрирования текста без вылета за пределы баббла.
    - Извлекает профиль доступных горизонтальных хорд chord_profile(y).
    """
    x1, y1, x2, y2 = [int(round(float(c))) for c in text_bbox[:4]]
    x1 = max(0, min(x1, img_w - 4))
    y1 = max(0, min(y1, img_h - 4))
    x2 = max(x1 + 4, min(x2, img_w))
    y2 = max(y1 + 4, min(y2, img_h))
    box_w = max(4, x2 - x1)
    box_h = max(4, y2 - y1)
    preset = (style_preset or "dialogue").lower().strip()

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

    # 1. Построение барьера контуров баббла (Canny + Closing + 2px ROI border barrier)
    edges = cv2.Canny(roi_gray, 50, 140)
    gap_k = int(np.clip(min(box_w, box_h) * 0.05, 3, 15)) | 1
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_k, gap_k))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close, iterations=1)
    cavity_candidates = cv2.bitwise_and(lum_mask, cv2.bitwise_not(closed_edges))

    # Рамка по периметру ROI предотвращает вытекание floodFill за края кадра (Frame Barrier)
    cavity_candidates[0, :] = 0
    cavity_candidates[-1, :] = 0
    cavity_candidates[:, 0] = 0
    cavity_candidates[:, -1] = 0

    # Гарантируем связность центрального ядра text_bbox
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

    # Leakage Guard: ограничение смещения от text_bbox
    max_leak_x = max(48, int(box_w * 0.85)) if is_narrow_col else max(24, int(box_w * 0.65))
    max_leak_y = max(16, int(box_h * 0.45))
    lim_x1 = max(0, tb_x1 - max_leak_x)
    lim_y1 = max(0, tb_y1 - max_leak_y)
    lim_x2 = min(rw, tb_x2 + max_leak_x)
    lim_y2 = min(rh, tb_y2 + max_leak_y)
    cavity_candidates[:, :lim_x1] = 0
    cavity_candidates[:, lim_x2:] = 0
    cavity_candidates[:lim_y1, :] = 0
    cavity_candidates[lim_y2:, :] = 0

    ff_probe = cavity_candidates.copy()
    ff_probe_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
    if ff_probe[seed_cy, seed_cx] > 0:
        cv2.floodFill(ff_probe, ff_probe_mask, (int(seed_cx), int(seed_cy)), 128)
        unsplit_fill = (ff_probe_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
    else:
        return _build_fallback_geometry()

    # 2. Разделение сдвоенных бабблов («восьмерок») через топографическое дистанционное поле
    isolated_cavity = unsplit_fill.copy()
    if other_bboxes:
        comp_other_cores = []
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
                # Проверяем, перетекает ли полость в ядро соседнего блока
                in_pad_x = max(1, int((ox2 - ox1) * 0.15))
                in_pad_y = max(1, int((oy2 - oy1) * 0.15))
                c_ox1 = max(0, (ox1 + in_pad_x) - rx1)
                c_oy1 = max(0, (oy1 + in_pad_y) - ry1)
                c_ox2 = min(rw, (ox2 - in_pad_x) - rx1)
                c_oy2 = min(rh, (oy2 - in_pad_y) - ry1)
                if c_ox2 > c_ox1 and c_oy2 > c_oy1:
                    if np.any(unsplit_fill[c_oy1:c_oy2, c_ox1:c_ox2] > 0):
                        comp_other_cores.append((c_ox1, c_oy1, c_ox2, c_oy2))

        if comp_other_cores:
            target_core_m = np.zeros((rh, rw), dtype=np.uint8)
            target_core_m[seed_y1:seed_y2, seed_x1:seed_x2] = 255
            other_cores_m = np.zeros((rh, rw), dtype=np.uint8)
            for c_ox1, c_oy1, c_ox2, c_oy2 in comp_other_cores:
                other_cores_m[c_oy1:c_oy2, c_ox1:c_ox2] = 255

            d_target = cv2.distanceTransform(cv2.bitwise_not(target_core_m), cv2.DIST_L2, 5)
            d_other = cv2.distanceTransform(cv2.bitwise_not(other_cores_m), cv2.DIST_L2, 5)
            saddle_partition = (d_target <= d_other)
            isolated_cavity = np.where(saddle_partition & (unsplit_fill > 0), 255, 0).astype(np.uint8)

    # Заполняем мелкие внутренние дырки
    k_dent = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bubble_mask = cv2.morphologyEx(isolated_cavity, cv2.MORPH_CLOSE, k_dent, iterations=1)
    inv_bub = cv2.bitwise_not(bubble_mask)
    n_inv, lbl_inv, st_inv, _ = cv2.connectedComponentsWithStats(inv_bub, connectivity=8)
    for idx_i in range(1, n_inv):
        ix = st_inv[idx_i, cv2.CC_STAT_LEFT]
        iy = st_inv[idx_i, cv2.CC_STAT_TOP]
        iw = st_inv[idx_i, cv2.CC_STAT_WIDTH]
        ih = st_inv[idx_i, cv2.CC_STAT_HEIGHT]
        if ix > 0 and iy > 0 and (ix + iw) < rw and (iy + ih) < rh:
            bubble_mask[lbl_inv == idx_i] = 255

    # Проверка замкнутости (Leakage Guard)
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

    # Отсечение хвостика баббла через MORPH_OPEN
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

    # 3. Вычисление безопасных отступов (safe_mask)
    dist_map = cv2.distanceTransform(core_mask, cv2.DIST_L2, 5)
    _, max_val, _, _ = cv2.minMaxLoc(dist_map)
    if max_val <= 1.0:
        return _build_fallback_geometry()

    if preset == "shout":
        margin_px = max(5, int(min(bubble_w, bubble_h) * 0.068))
    elif preset == "narration":
        margin_px = max(6, int(min(bubble_w, bubble_h) * 0.078))
    else:
        margin_px = max(5, int(min(bubble_w, bubble_h) * 0.084))

    eff_margin = min(float(margin_px), float(max_val) * 0.42)
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

    # 4. Поиск оптического центра через 2D-интегральную свертку (Feasible Center Region)
    # Гарантирует максимальный клиренс до всех стенок баббла и отсутствие вылета текста вверх
    II = cv2.integral((safe_mask > 0).astype(np.uint8), sdepth=cv2.CV_32S)
    est_tw = max(12, int(min(safe_w * 0.85, max(box_w * 0.75, 24))))
    est_th = max(12, int(min(safe_h * 0.75, max(box_h * 0.65, 24))))
    centroid_y_roi = float(np.mean(ys_s))
    centroid_x_roi = float(np.mean(xs_s))

    rect_sums = (
        II[est_th:, est_tw:]
        - II[:-est_th, est_tw:]
        - II[est_th:, :-est_tw]
        + II[:-est_th, :-est_tw]
    )
    valid_tl = (rect_sums == (est_tw * est_th)).astype(np.uint8) * 255
    y_off, x_off = est_th // 2, est_tw // 2

    if np.any(valid_tl > 0):
        feasible_centers = np.zeros((rh, rw), dtype=np.uint8)
        feasible_centers[y_off : y_off + valid_tl.shape[0], x_off : x_off + valid_tl.shape[1]] = valid_tl
        margin_map = cv2.distanceTransform(feasible_centers, cv2.DIST_L2, 5)
        y_grid = np.arange(rh, dtype=np.float32)[:, None]
        gravity_penalty = 1.2 * np.abs(y_grid - centroid_y_roi) / max(1.0, float(rh))
        score_map = np.where(feasible_centers > 0, margin_map - gravity_penalty, -1e6)
        best_idx = int(np.argmax(score_map))
        opt_cy_roi, opt_cx_roi = divmod(best_idx, rw)
    else:
        # Регуляризованный оптический центр
        safe_mid_x = (sx1_roi + sx2_roi) / 2.0
        safe_mid_y = (sy1_roi + sy2_roi) / 2.0
        _, _, _, max_loc = cv2.minMaxLoc(dist_map)
        pole_x, pole_y = float(max_loc[0]), float(max_loc[1])
        opt_cx_roi = int(round(0.68 * safe_mid_x + 0.32 * pole_x))
        opt_cy_roi = int(round(0.72 * safe_mid_y + 0.28 * pole_y - safe_h * 0.012))

    optical_cx = int(round(rx1 + opt_cx_roi))
    optical_cy = int(round(ry1 + opt_cy_roi))
    optical_cx = max(sx1_abs + 4, min(sx2_abs - 4, optical_cx))
    optical_cy = max(sy1_abs + 4, min(sy2_abs - 4, optical_cy))
    opt_cx_roi = max(0, min(rw - 1, optical_cx - rx1))

    # 5. Профиль доступных горизонтальных хорд chord_width(y) внутри safe_bbox
    chord_profile: list[int] = []
    for y_abs in range(sy1_abs, sy2_abs):
        y_r = max(0, min(rh - 1, y_abs - ry1))
        row_mask = safe_mask[y_r, :] > 0
        row_xs = np.where(row_mask)[0]
        if len(row_xs) >= 2:
            splits = np.where(np.diff(row_xs) > 2)[0]
            run_starts = np.insert(row_xs[splits + 1], 0, row_xs[0])
            run_ends = np.append(row_xs[splits], row_xs[-1])
            best_cw = 0
            best_dist = 1e9
            for r_s, r_e in zip(run_starts, run_ends):
                if r_s <= opt_cx_roi <= r_e:
                    dist_c = 0.0
                else:
                    dist_c = float(min(abs(opt_cx_roi - r_s), abs(opt_cx_roi - r_e)))
                run_w = int(r_e - r_s + 1)
                if dist_c < best_dist or (dist_c == best_dist and run_w > best_cw):
                    best_dist = dist_c
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
    }


# ==============================================================================
# 2. КЛАСС ОЧИСТКИ ТЕКСТА (BubbleCleaner)
# ==============================================================================

class BubbleCleaner:
    """
    SOTA-модуль очистки текста манги (cleaner.py) с защитой контуров бабблов,
    сохранением внутреннего чиби-арта, извлечением белого ореола (halo),
    маршрутизацией Smart Inpaint Router и гарантией Zero Bleed.
    """

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
        self.device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self.cfg = CleanerConfig()

        self._cache_lock = threading.Lock()
        self._cleaned_lru_cache: OrderedDict[tuple[str, str], Image.Image] = OrderedDict()
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
                        "Используется резервный алгоритм очистки (Uniform Bubble Fill + OpenCV INPAINT_NS).",
                        builtin_exc,
                    )
                    self.lama_model = None

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
        """Публичный интерфейс анализа геометрии баббла, делегирующий вызов в модуль."""
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
            "clean_ver": "v4.0",
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
            "clean_ver": "v4.0",
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
        """Находит внутреннюю светлую область speech bubble, морфологически замыкая микро-вмятины."""
        rh, rw = roi_gray.shape[:2]
        wall_thresh = max(150, int(bg_level - 52))
        light_pixels = (roi_gray >= wall_thresh).astype(np.uint8) * 255

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

        # 1. Заполняем внутренние замкнутые отверстия ДО морфологического замыкания
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

        # 2. Замыкаем компактные внутренние вмятины от иероглифов, касающихся стенок баббла (知, 露, 藤, 没)
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

    def _extract_bubble_cavity_and_wall(
        self,
        roi_rgb: np.ndarray,
        l_bb: Tuple[int, int, int, int],
        l_tb: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Извлекает внутреннюю полость баббла cavity_mask и его черную стенку wall_mask."""
        rh, rw = roi_rgb.shape[:2]
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        bx1, by1, bx2, by2 = l_bb
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)

        tx1, ty1, tx2, ty2 = l_tb
        tb_crop = gray[ty1:ty2, tx1:tx2]
        bg_bright = float(np.percentile(tb_crop, 85)) if tb_crop.size > 0 else 245.0
        is_light_bubble = bg_bright >= 128.0

        if is_light_bubble:
            _, dark_bin = cv2.threshold(gray, int(min(200, bg_bright - 35)), 255, cv2.THRESH_BINARY_INV)
        else:
            _, dark_bin = cv2.threshold(gray, int(max(55, bg_bright + 35)), 255, cv2.THRESH_BINARY)

        num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(dark_bin, connectivity=8)
        wall_candidates = np.zeros_like(dark_bin)
        for lbl in range(1, num_cc):
            cw, ch = stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]
            c_area = stats[lbl, cv2.CC_STAT_AREA]
            if cw > 0.48 * bw or ch > 0.48 * bh or c_area > 0.30 * (bw + bh) * 3:
                wall_candidates[labels == lbl] = 255

        gap_k = int(np.clip(min(bw, bh) * 0.05, 3, 15)) | 1
        closed_wall = cv2.morphologyEx(
            wall_candidates,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_k, gap_k)),
        )

        barrier = closed_wall.copy()
        cv2.rectangle(barrier, (bx1, by1), (bx2 - 1, by2 - 1), 255, 2)

        inv_barrier = cv2.bitwise_not(barrier)
        ff_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
        ff_mask[1:-1, 1:-1] = barrier
        seed_x = (tx1 + tx2) // 2
        seed_y = (ty1 + ty2) // 2
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
                (bw // 2 - 2, bh // 2 - 2),
                0,
                0,
                360,
                255,
                -1,
            )

        cavity_border_band = cv2.dilate(cavity_filled, np.ones((7, 7), np.uint8)) - cv2.erode(
            cavity_filled, np.ones((3, 3), np.uint8)
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
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Отделяет штрихи текста от стенки баббла и чиби-арта (SWT, IQ = 4*pi*A/P^2, Solidity)."""
        rh, rw = roi_rgb.shape[:2]
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        tx1, ty1, tx2, ty2 = l_tb
        tw_box, th_box = max(1, tx2 - tx1), max(1, ty2 - ty1)

        k_bh = int(np.clip(min(tw_box, th_box) // 6, 5, 25)) | 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bh, k_bh))
        bhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
        that = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
        p_bhat = np.percentile(bhat[ty1:ty2, tx1:tx2], 90) if tw_box > 0 and th_box > 0 else 0
        p_that = np.percentile(that[ty1:ty2, tx1:tx2], 90) if tw_box > 0 and th_box > 0 else 0
        polarity_map = bhat if p_bhat >= p_that else that
        _, candidate_bin = cv2.threshold(polarity_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
            hull_area = float(cv2.contourArea(cv2.convexHull(cnts[0]))) if len(cnts) > 0 else float(area)
            solidity = float(area) / max(1.0, hull_area)

            # Защита чиби-арта вне text_bbox
            if in_tb_ratio < 0.25:
                if area >= 25 and cavity_roi[min(rh - 1, cy_l + ch // 2), min(rw - 1, cx_l + cw // 2)] > 0:
                    obstacle_mask[labels == lbl] = 255
                continue

            # Защита массивного арта/зрачков чиби или длинных линий рамки внутри text_bbox
            is_massive_blob = (sw_max > max(16.0, 2.6 * median_sw)) and (iq > 0.72) and (area > 180)
            is_long_frame_line = (max(cw, ch) > 0.85 * max(tw_box, th_box)) and (iq < 0.11) and (solidity < 0.15)
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
        """Извлекает контрастный белый ореол (halo) вокруг штрихов текста, выходящих за рамку баббла."""
        if not np.any(stroke_core > 0):
            return np.zeros_like(stroke_core)
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        r_halo = int(np.clip(round(1.6 * median_sw), 3, 12))
        se_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_halo + 1, 2 * r_halo + 1))
        search_ring = cv2.subtract(cv2.dilate(stroke_core, se_halo), stroke_core)
        outside_ring = cv2.bitwise_and(search_ring, cv2.bitwise_not(cavity_roi))
        if not np.any(outside_ring > 0):
            return np.zeros_like(stroke_core)

        core_med = float(np.median(gray[stroke_core > 0]))
        if core_med < 128.0:
            halo_cand = ((gray >= 205) & (outside_ring > 0)).astype(np.uint8) * 255
        else:
            halo_cand = ((gray <= 50) & (outside_ring > 0)).astype(np.uint8) * 255

        touch_core = cv2.dilate(stroke_core, np.ones((3, 3), np.uint8))
        num_cc, labels, _, _ = cv2.connectedComponentsWithStats(halo_cand, connectivity=8)
        halo_verified = np.zeros_like(stroke_core)
        for lbl in range(1, num_cc):
            comp = (labels == lbl)
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
        """Итеративное выращивание маски по алгоритму PanelCleaner (pcleaner/masker.py)."""
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

    def build_text_mask(
        self,
        image_rgb: np.ndarray,
        blocks: list[dict[str, Any]],
        include_custom_strokes: bool = True,
    ) -> np.ndarray:
        """
        Строит точную бинарную маску текста (uint8, 0 или 255) по штрихам текста и ореолу
        для каждого активного Bounding Box с сохранением контуров бабблов и чиби-арта.
        """
        img_h, img_w = image_rgb.shape[:2]
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        for block in blocks:
            if self._is_block_skipped(block):
                continue
            x1, y1, x2, y2 = self._get_clean_bbox(block)
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))
            bw, bh = x2 - x1, y2 - y1
            if bw < 8 or bh < 8:
                continue

            is_free_text = bool(block.get("is_free_text", False))
            raw_dilate = block.get("mask_dilate")
            custom_dilate = int(raw_dilate) if isinstance(raw_dilate, (int, float)) and int(raw_dilate) > 0 else 0

            # Точные границы текста (text_bbox)
            raw_tb = block.get("text_bbox")
            if isinstance(raw_tb, (list, tuple)) and len(raw_tb) >= 4 and not bool(block.get("bbox_modified", False)):
                tx1, ty1, tx2, ty2 = [int(round(float(c))) for c in raw_tb[:4]]
                tx1 = max(0, min(tx1, img_w - 1))
                ty1 = max(0, min(ty1, img_h - 1))
                tx2 = max(tx1 + 1, min(tx2, img_w))
                ty2 = max(ty1 + 1, min(ty2, img_h))
            else:
                tx1, ty1, tx2, ty2 = x1, y1, x2, y2

            # ROI с запасом под выступающие символы и ореол (halo)
            pad = 28
            rx1 = max(0, min(x1, tx1) - pad)
            ry1 = max(0, min(y1, ty1) - pad)
            rx2 = min(img_w, max(x2, tx2) + pad)
            ry2 = min(img_h, max(y2, ty2) + pad)
            roi_rgb = image_rgb[ry1:ry2, rx1:rx2]
            rh, rw = roi_rgb.shape[:2]
            if rh < 8 or rw < 8:
                continue

            l_tb = (max(0, tx1 - rx1), max(0, ty1 - ry1), min(rw, tx2 - rx1), min(rh, ty2 - ry1))
            l_bb = (max(0, x1 - rx1), max(0, y1 - ry1), min(rw, x2 - rx1), min(rh, y2 - ry1))

            if not is_free_text:
                cavity_roi, wall_roi = self._extract_bubble_cavity_and_wall(roi_rgb, l_bb, l_tb)
                stroke_core, obstacle_roi, median_sw = self._segment_strokes_and_obstacles(
                    roi_rgb, l_tb, l_bb, cavity_roi, wall_roi
                )
                halo_mask = self._extract_text_halo(roi_rgb, stroke_core, cavity_roi, median_sw)
                raw_text_mask = cv2.bitwise_or(stroke_core, halo_mask)
                protected_mask = cv2.bitwise_or(wall_roi, obstacle_roi)
                raw_text_mask = cv2.bitwise_and(raw_text_mask, cv2.bitwise_not(protected_mask))

                inside_seed = cv2.bitwise_and(raw_text_mask, cavity_roi)
                outside_seed = cv2.bitwise_and(raw_text_mask, cv2.bitwise_not(cv2.bitwise_or(cavity_roi, wall_roi)))

                block_mask = np.zeros((rh, rw), dtype=np.uint8)
                if np.any(inside_seed > 0):
                    opt_inside, _, _ = self._grow_and_evaluate_mask_pcleaner(
                        roi_rgb, inside_seed, cavity_roi, protected_mask
                    )
                    block_mask = cv2.bitwise_or(block_mask, opt_inside)

                if np.any(outside_seed > 0):
                    r_dil = int(np.clip(round(0.5 * median_sw) + 3, 4, 10))
                    se_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_dil + 1, 2 * r_dil + 1))
                    opt_outside = cv2.bitwise_and(
                        cv2.dilate(outside_seed, se_out),
                        cv2.bitwise_not(cv2.bitwise_or(cavity_roi, wall_roi)),
                    )
                    block_mask = cv2.bitwise_or(block_mask, opt_outside)

                if custom_dilate > 0:
                    d_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (custom_dilate, custom_dilate))
                    block_mask = cv2.dilate(block_mask, d_k)
                    block_mask = cv2.bitwise_and(block_mask, cv2.bitwise_not(protected_mask))

                full_mask[ry1:ry2, rx1:rx2] = np.maximum(full_mask[ry1:ry2, rx1:rx2], block_mask)
            else:
                # Свободный текст на фоне (is_free_text)
                gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
                med_val = float(np.median(gray[l_tb[1]:l_tb[3], l_tb[0]:l_tb[2]])) if l_tb[2] > l_tb[0] and l_tb[3] > l_tb[1] else 128.0
                if med_val >= 127.0:
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                else:
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                tb_zone = np.zeros((rh, rw), dtype=np.uint8)
                cv2.rectangle(tb_zone, (l_tb[0], l_tb[1]), (l_tb[2], l_tb[3]), 255, -1)
                binary = cv2.bitwise_and(binary, tb_zone)

                d_val = custom_dilate if custom_dilate > 0 else max(9, self.dilate_size + 2)
                se_free = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_val, d_val))
                dilated_free = cv2.dilate(binary, se_free)
                full_mask[ry1:ry2, rx1:rx2] = np.maximum(full_mask[ry1:ry2, rx1:rx2], dilated_free)

        # Защита явно пропущенных или удаленных блоков
        for block in blocks:
            if not self._is_block_skipped(block):
                continue
            dx1, dy1, dx2, dy2 = self._get_clean_bbox(block)
            dx1 = max(0, min(dx1, img_w - 1))
            dy1 = max(0, min(dy1, img_h - 1))
            dx2 = max(dx1 + 1, min(dx2, img_w))
            dy2 = max(dy1 + 1, min(dy2, img_h))
            full_mask[dy1:dy2, dx1:dx2] = 0

        # Наложение ручных мазков кисти / ластика
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
        """
        Трёхуровневый инпейнтинг кропа со Smart Inpaint Router и гарантией Zero Bleed:
        - Внутренняя зона баббла: аналитическая заливка / Edge NLM Denoiser / LaMa
        - Внешняя зона: экранирование белого баббла (Context Shielding) + LaMa
        """
        ch, cw = crop_rgb_np.shape[:2]
        if np.count_nonzero(crop_mask_np) == 0:
            return crop_rgb_np

        if not is_free_text:
            ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            ring_mask = cv2.dilate(crop_mask_np, ring_k, iterations=1) - crop_mask_np
            ring_pixels = crop_rgb_np[ring_mask > 0]
            if len(ring_pixels) >= 16:
                ring_gray = np.mean(ring_pixels, axis=1)
                p15 = float(np.percentile(ring_gray, 15))
                p50 = float(np.percentile(ring_gray, 50))
                p65 = float(np.percentile(ring_gray, 65))
                p80 = float(np.percentile(ring_gray, 80))
                ring_std = float(np.std(ring_gray))

                is_uniform_light = (p15 >= 232.0 and p50 >= 242.0 and ring_std <= 16.0)
                is_white_bubble_wall_touch = (p50 >= 234.0 and p65 >= 242.0 and p80 >= 246.0)

                if is_uniform_light or is_white_bubble_wall_touch:
                    thresh_bright = p15 if is_uniform_light else max(230.0, p50)
                    bright_pixels = ring_pixels[ring_gray >= thresh_bright]
                    if len(bright_pixels) >= 8:
                        fill_color = np.median(bright_pixels, axis=0).round().astype(np.uint8)
                        if float(np.min(fill_color)) >= 240.0:
                            fill_color = np.array([255, 255, 255], dtype=np.uint8)

                        out = crop_rgb_np.copy()
                        out[crop_mask_np > 0] = fill_color

                        # Если шум на границе умеренный (0.25 < sigma <= 15.0), сглаживаем NLM-денойзером
                        if ring_std > self.cfg.noise_min_std_dev:
                            belt_r = self.cfg.noise_outline_size
                            se_belt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * belt_r + 1, 2 * belt_r + 1))
                            belt_mask = cv2.subtract(cv2.dilate(crop_mask_np, se_belt), crop_mask_np)
                            if np.any(belt_mask > 0):
                                try:
                                    denoised = cv2.fastNlMeansDenoisingColored(
                                        out, None, h=self.cfg.filter_strength, hColor=self.cfg.filter_strength,
                                        templateWindowSize=7, searchWindowSize=21,
                                    )
                                    alpha = cv2.GaussianBlur(belt_mask.astype(np.float32) / 255.0, (3, 3), 0)[..., None]
                                    out = (out.astype(np.float32) * (1.0 - alpha) + denoised.astype(np.float32) * alpha).round().astype(np.uint8)
                                except Exception:
                                    pass
                        return out

        # Нейросетевой инпейнтинг LaMa с экранированием фона (Zero Bleed Context Shielding)
        if self.lama_model is not None:
            try:
                shielded_input = crop_rgb_np.copy()
                # Context Shielding: если вокруг маски есть темный фон (волосы/скринтон), а внутри белый баббл,
                # маскируем белый баббл под цвет внешнего фона, чтобы LaMa не замазывала белым цветом волосы!
                ext_mask = cv2.dilate(crop_mask_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
                bg_pix = crop_rgb_np[ext_mask == 0]
                if len(bg_pix) >= 16:
                    bg_med = np.median(bg_pix, axis=0).astype(np.uint8)
                    pure_white_core = np.min(crop_rgb_np, axis=2) >= 245
                    shielded_input[pure_white_core & (ext_mask > 0)] = bg_med

                crop_pil = Image.fromarray(shielded_input, mode="RGB")
                mask_pil = Image.fromarray(crop_mask_np, mode="L")
                inpainted_pil = self.lama_model(crop_pil, mask_pil).convert("RGB")
                if inpainted_pil.size != (cw, ch):
                    inpainted_pil = inpainted_pil.crop((0, 0, cw, ch))
                inp_arr = np.array(inpainted_pil)

                # Мягкая альфа-растушевка границы инпейнтинга
                dt_in = cv2.distanceTransform(crop_mask_np, cv2.DIST_L2, 3)
                alpha = np.clip(dt_in / 2.0, 0.0, 1.0)[..., None]
                blended = (crop_rgb_np.astype(np.float32) * (1.0 - alpha) + inp_arr.astype(np.float32) * alpha).round().astype(np.uint8)
                out = crop_rgb_np.copy()
                out[crop_mask_np > 0] = blended[crop_mask_np > 0]
                return out
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
        """Мгновенное (< 2 мс) наложение интерактивных мазков кисти и ластика Web Studio."""
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
                        cv2.line(final_mask, (px, py), (cx, cy), 0 if smode == "erase" else 255, thickness=avg_r * 2)
                prev_pt = (cx, cy, radius, smode)

        if not has_any:
            return base_cleaned_np, final_mask

        out_np = base_cleaned_np.copy()

        # 1. Применяем мазки замазки (mode == "add")
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
        1) Выполняет (или берет из кэша за 0 мс) базовую очистку бабблов по clean_bbox + mask_dilate.
        2) Мгновенно (< 2 мс) накладывает ручные мазки кисти/ластика (custom_mask_strokes).
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
            active_block_specs = []

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
                active_block_specs.append((block, x1, y1, x2, y2, cx1, cy1, cx2, cy2, is_free_text, block_crop_key, cached_crop_pair))

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

                for block, x1, y1, x2, y2, cx1, cy1, cx2, cy2, is_free_text, block_crop_key, cached_crop_pair in active_block_specs:
                    crop_rgb = base_composited_np[cy1:cy2, cx1:cx2]
                    if cached_crop_pair is not None:
                        inpainted_crop, crop_mask = cached_crop_pair[0], cached_crop_pair[1]
                        base_mask_np[cy1:cy2, cx1:cx2] = np.maximum(base_mask_np[cy1:cy2, cx1:cx2], crop_mask)
                    else:
                        margin_extra = 8 if is_free_text else min(pad - 2, max(12, int(min(x2 - x1, y2 - y1) * 0.22)))
                        bx1 = max(cx1, x1 - margin_extra)
                        by1 = max(cy1, y1 - margin_extra)
                        bx2 = min(cx2, x2 + margin_extra)
                        by2 = min(cy2, y2 + margin_extra)

                        crop_mask = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
                        crop_mask[by1 - cy1 : by2 - cy1, bx1 - cx1 : bx2 - cx1] = base_mask_np[by1:by2, bx1:bx2]
                        if np.count_nonzero(crop_mask) == 0:
                            continue

                        inpainted_crop = self._inpaint_crop(crop_rgb, crop_mask, is_free_text=is_free_text)
                        with self._cache_lock:
                            self._block_crop_cache[block_crop_key] = (inpainted_crop.copy(), crop_mask.copy())
                            while len(self._block_crop_cache) > self._block_crop_cache_max:
                                self._block_crop_cache.popitem(last=False)

                    core_f = (crop_mask > 0).astype(np.float32)
                    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    dil_f = cv2.dilate(crop_mask, dil_k, iterations=1).astype(np.float32) / 255.0
                    blur_f = cv2.GaussianBlur(dil_f, (5, 5), 0)
                    alpha = np.maximum(core_f, blur_f)[..., None]
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