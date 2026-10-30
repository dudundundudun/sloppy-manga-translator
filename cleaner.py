"""
Модуль очистки бабблов и свободных надписей от оригинального текста — Inpainting (v3.5 Pro Scanlation Typesetting Edition).

Изменения v3.5:
1. Индивидуальный радиус замазки `mask_dilate` (1..28 px) для каждого блока в `build_text_mask`:
   создает эллиптическое ядро `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))` специально для блока,
   позволяя в 1 движение ползунка убрать торчащие края иероглифов или уменьшить замазку для сохранения рисунка.
2. Двухуровневое кэширование очистки (`_block_crop_cache` + `_base_page_cache`):
   - Кэширует результат инпейнтинга каждого отдельного блока `(img_fingerprint, clean_bbox, is_free_text, mask_dilate)`,
     благодаря чему изменение ползунка `Замазка (dilate)` у одного блока пересчитывает только этот один кроп за ~15–35 мс,
     не затрагивая остальные бабблы на странице.
   - Кэширует базовый автоматически очищенный слой страницы до применения ручных мазков кисти.
3. Мгновенная (< 2 мс) обработка ручной кисти и ластика замазки `custom_mask_strokes`
   (`[{"x": int, "y": int, "r": int, "mode": "add"|"erase"}]`) поверх базового очищенного слоя без повторного вызова SimpleLama.
4. Геометрическая сегментация реальных границ бабла (`analyze_bubble_geometry`):
   - Морфологическое замыкание контура (`cv2.morphologyEx` 3x3 на карте границ Canny/Sobel) + `cv2.floodFill`
     с защитой от протечек (`Leakage Guard`: касание краев ROI <= 25% периметра, площадь <= 3.8x `text_bbox`).
   - Отсечение хвостика бабла (`speech tail`) и шипов крика через `cv2.MORPH_OPEN`.
   - Поиск оптического центра бабла (`Pole of Inaccessibility`) через `cv2.distanceTransform(core_mask, cv2.DIST_L2, 5)`
     с вертикальной оптической поправкой (`-2.2%` высоты бабла) и безопасными отступами (`safe_mask` 8–11%).
5. Сохранены все оптимизации v1.0.8 (LRU-кэш `cleaned_image` на 8 страниц + дисковый кэш в `temp_debug/`).
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any
import warnings

warnings.filterwarnings("ignore", message=".*Failed to find CUDA.*", category=UserWarning)
warnings.filterwarnings("ignore", module="triton.*")

import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

LAMA_WEIGHTS_URLS: tuple[str, ...] = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
    "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt",
)


class BuiltInSimpleLama:
    """
    Встроенный нейросетевой движок очистки LaMa (big-lama.pt, TorchScript JIT),
    работающий напрямую через PyTorch (CUDA/CPU) без необходимости установки пакета `simple-lama-inpainting`.
    Автоматически находит или скачивает `big-lama.pt` (~196 МБ) в `models/lama/big-lama.pt` при первом запуске.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        models_dir: str | Path | None = None,
    ) -> None:
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
    Сегментирует реальную внутреннюю полость бабла вокруг `text_bbox` на очищенном изображении `cleaned_img_np`,
    отсекает узкий хвостик бабла через морфологическое открытие, изолирует соседние/сдвоенные баблы через
    барьеры Вороного (`other_bboxes`), находит оптический центр (`Pole of Inaccessibility`) через `cv2.distanceTransform`
    с оптической поправкой `-2.2%` по вертикали и вычисляет `safe_mask` и профиль хорд без схлопывания отступов.
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
            # Проверяем пересечение с расширенной безопасной зоной
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
            # Разделяем по доминирующей оси смещения между центрами соседних баблов
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
        }

    if is_free_text or preset == "sfx" or cleaned_img_np.size == 0:
        return _build_fallback_geometry()

    # Если text_bbox — это узкий японский столбик внутри широкого кружка (box_w < box_h * 0.55),
    # разрешаем расширение по X шире; если это уже полноценный бокс бабла — расширяем умеренно,
    # чтобы не захватывать соседние баблы
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

    # Морфологическое замыкание мелких разрывов контура бабла (Canny + Sobel)
    edges = cv2.Canny(roi_gray, 50, 140)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close, iterations=1)
    cavity_candidates = cv2.bitwise_and(lum_mask, cv2.bitwise_not(closed_edges))

    # Гарантируем, что центральное ядро text_bbox (уже очищенное от иероглифов) связно
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

    # Ограничиваем максимальный выход floodFill за пределы рамки детектора бабла (чтобы через стертые краевые
    # иероглифы вроде 吧 или 没有 заливка не вытекала в соседний экран телевизора или соседний кадр панели)
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

    # Изоляция соседних/сдвоенных баблов: ставим барьер Вороного по смежным/сдвоенным блокам (other_bboxes)
    # ТОЛЬКО если неразрезанный floodFill реально перетекает через открытую белую перемычку в ядро соседнего блока!
    voronoi_cut_mask = np.zeros((rh, rw), dtype=bool)
    ff_probe = cavity_candidates.copy()
    ff_probe_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
    if ff_probe[seed_cy, seed_cx] > 0:
        cv2.floodFill(ff_probe, ff_probe_mask, (int(seed_cx), int(seed_cy)), 128)
        unsplit_fill = ff_probe_mask[1:-1, 1:-1] > 0
    else:
        return _build_fallback_geometry()

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
                # Проверяем, перетекает ли полость текущего бабла в ядро соседнего блока (сдвоенный бабл без черной перегородки)
                if not np.any(unsplit_fill[core_oy1:core_oy2, core_ox1:core_ox2]):
                    continue

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
                    cavity_candidates[half_plane] = 0

    if np.any(voronoi_cut_mask):
        ff_work = cavity_candidates.copy()
        ff_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
        if ff_work[seed_cy, seed_cx] > 0:
            cv2.floodFill(ff_work, ff_mask, (int(seed_cx), int(seed_cy)), 128)
            bubble_mask = (ff_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
        else:
            bubble_mask = unsplit_fill.astype(np.uint8) * 255
    else:
        bubble_mask = unsplit_fill.astype(np.uint8) * 255

    # Заполняем мелкие внутренние дырки и сглаживаем микро-вмятины вдоль стенок бабла
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

    # 2.2. Отсечение хвостика бабла (speech tail) и узких перемычек через MORPH_OPEN
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

    # Если после MORPH_OPEN маска распалась на несколько долей (например, сдвоенный бабл-восьмерка),
    # выбираем ту компоненту, в которой лежит центр исходного text_bbox (seed_cx, seed_cy), либо крупнейшую!
    n_c, lbl_c, st_c, centroids_c = cv2.connectedComponentsWithStats(core_mask, connectivity=8)
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

    # Вычисляем карту расстояний (Pole of Inaccessibility)
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

    # 2.3. Комфортные сканлейт-отступы от стенок бабла (достаточный воздух до черного контура)
    if preset == "shout":
        margin_px = max(5, int(min(bubble_w, bubble_h) * 0.068))
    elif preset == "narration":
        margin_px = max(6, int(min(bubble_w, bubble_h) * 0.078))
    else:
        margin_px = max(5, int(min(bubble_w, bubble_h) * 0.084))

    eff_margin = min(float(margin_px), float(max_val) * 0.44)
    if np.any(voronoi_cut_mask):
        unsplit_cavity = cv2.bitwise_or(core_mask, ((voronoi_cut_mask) & (lum_mask > 0)).astype(np.uint8) * 255)
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

    # Сбалансированный оптический центр: 72% центроид/середина безопасной зоны + 28% Pole of Inaccessibility,
    # чтобы в каплевидных или обрезанных сверху баблах текст никогда не прижимался к верхней стенке!
    safe_mid_x = (sx1_abs + sx2_abs) / 2.0
    safe_mid_y = (sy1_abs + sy2_abs) / 2.0
    mask_cent_x = float(rx1) + float(np.mean(xs_s))
    mask_cent_y = float(ry1) + float(np.mean(ys_s))
    base_cx = 0.55 * safe_mid_x + 0.45 * mask_cent_x
    base_cy = 0.55 * safe_mid_y + 0.45 * mask_cent_y
    pole_abs_x = float(rx1) + cx_roi
    pole_abs_y = float(ry1) + cy_roi
    raw_opt_cx = 0.72 * base_cx + 0.28 * pole_abs_x
    raw_opt_cy = 0.72 * base_cy + 0.28 * pole_abs_y - safe_h * 0.012

    max_drift_x = min(12.0, safe_w * 0.08)
    max_drift_y = min(14.0, safe_h * 0.08)
    optical_cx = int(round(max(safe_mid_x - max_drift_x, min(safe_mid_x + max_drift_x, raw_opt_cx))))
    optical_cy = int(round(max(safe_mid_y - max_drift_y, min(safe_mid_y + max_drift_y, raw_opt_cy))))
    optical_cx = max(sx1_abs + 4, min(sx2_abs - 4, optical_cx))
    optical_cy = max(sy1_abs + 4, min(sy2_abs - 4, optical_cy))
    opt_cx_roi = max(0, min(rw - 1, optical_cx - rx1))

    # Профиль доступных горизонтальных хорд chord_width(y) для каждой строки внутри safe_bbox
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    ) -> dict[str, Any]:
        """Публичный метод сегментации внутренней полости бабла и расчета оптического центра."""
        return analyze_bubble_geometry(
            cleaned_img_np=cleaned_img_np,
            text_bbox=text_bbox,
            img_w=img_w,
            img_h=img_h,
            style_preset=style_preset,
            is_free_text=is_free_text,
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
            "clean_ver": "v3.7",
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
            "clean_ver": "v3.7",
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
        отсекая внешние углы бокса за пределами рамки баббла, и морфологически замыкает
        внутренние вмятины от иероглифов, касающихся стенок баббла (например, 知, 露, 藤, 没).
        """
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
            # Включаем дополнительные светлые компоненты только если они лежат внутри центрального ядра баббла
            # (не захватывая внешние белые окна/подушки за пределами черного контура баббла!)
            if edge_touches == 0 and center_hits >= max(10.0, center_slice.size * 0.08):
                selected_labels.append(lbl)

        if best_label == -1:
            return light_pixels

        interior = np.zeros_like(light_pixels)
        interior[labels == best_label] = 255
        for lbl in selected_labels:
            interior[labels == lbl] = 255

        # 1. Заполняем внутренние замкнутые отверстия внутри иероглифов ДО морфологического замыкания,
        # чтобы контур самого баббла, касающийся краев ROI, никогда не превращался во «внутреннюю дырку»!
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

        # 2. Замыкаем только компактные внутренние вмятины от иероглифов, касающихся стенок баббла (知, 露, 藤, 衣服干, 没),
        # не затрагивая длинные дуги контура баббла!
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
        """
        Возвращает исходные координаты распознанного текста (clean_bbox / original_bbox),
        а если рамка не двигалась вручную и имеется `text_bbox` (например, от RT-DETR-v2 class 1),
        объединяет `clean_bbox` с `text_bbox`, чтобы не обрезать выступающие за контур баббла столбцы иероглифов.
        """
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
        """
        Возвращает область базовой очистки блока по исходному `clean_bbox` (с объединением `text_bbox`).
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
        внутри каждого активного Bounding Box (а также по многоштриховым иероглифам с белым ореолом,
        пересекающим или выступающим за границу баббла, таким как 知, 露, 吧, 衣服干, 藤, 要, 人在, 没, 如果愿意的话).
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
                # При наличии точного text_bbox от детектора никогда не выходим за пределы баббла [x1, y1, x2, y2],
                # чтобы не задевать волосы, стены комнаты, рамки фреймов и карту погоды снаружи баббла!
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
                if has_valid_tb:
                    pad_tb = 2
                    tz_x1 = max(0, tb_box[0] - ix1 - pad_tb)
                    tz_y1 = max(0, tb_box[1] - iy1 - pad_tb)
                    tz_x2 = min(rw, tb_box[2] - ix1 + pad_tb)
                    tz_y2 = min(rh, tb_box[3] - iy1 + pad_tb)
                    if (tz_x2 - tz_x1) >= 8 and (tz_y2 - tz_y1) >= 8:
                        text_zone_mask.fill(0)
                        text_zone_mask[tz_y1:tz_y2, tz_x1:tz_x2] = 255

                    # Защита контуров «колючих»/разомкнутых бабблов (spiky bubbles, например 08.webp #11, 09.webp #1):
                    # тонкие полые дуги (fill_r <= 0.18) за пределами внутреннего ядра text_bbox не входят в маску текста
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
                            arc_dil = cv2.dilate(hollow_arc_ring, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                            interior_mask[arc_dil > 0] = 0
                            text_zone_mask[arc_dil > 0] = 0

                ink_thresh = max(165, int(p75 - 14))
                dark_all = (roi_gray < ink_thresh).astype(np.uint8) * 255
                binary = cv2.bitwise_and(dark_all, cv2.bitwise_and(interior_mask, text_zone_mask))

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                clean_roi_mask = np.zeros_like(binary)
                chibi_art_mask = np.zeros_like(binary)
                dt_dark = cv2.distanceTransform((roi_gray < 175).astype(np.uint8), cv2.DIST_L2, 3)

                for label_idx in range(1, num_labels):
                    comp_x = stats[label_idx, cv2.CC_STAT_LEFT]
                    comp_y = stats[label_idx, cv2.CC_STAT_TOP]
                    comp_w = stats[label_idx, cv2.CC_STAT_WIDTH]
                    comp_h = stats[label_idx, cv2.CC_STAT_HEIGHT]
                    comp_area = stats[label_idx, cv2.CC_STAT_AREA]

                    if comp_area < 2:
                        continue

                    # Защита внутреннего арта/чиби-головок внутри баббла (например, 08.webp #3):
                    # массивные сплошные темные пятна имеют радиус distanceTransform >= 5.2 * scale_f (у текста <= 4.5)
                    if comp_area >= int(380 * scale_area):
                        comp_dt = dt_dark[labels == label_idx]
                        if (
                            comp_dt.size > 0
                            and float(np.max(comp_dt)) >= max(9.5, 5.2 * scale_f)
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

                if np.count_nonzero(chibi_art_mask) > 0:
                    chibi_d = max(9, int(round(12 * scale_f)) | 1)
                    chibi_dil = cv2.dilate(
                        chibi_art_mask,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (chibi_d, chibi_d)),
                    )
                    clean_roi_mask[chibi_dil > 0] = 0
                    text_zone_mask[chibi_dil > 0] = 0

                # 2. Масштабно-инвариантная детекция краевых/пересекающих контур баббла иероглифов с белым ореолом
                # или внутренними просветами (таких как 知, 露, 吧, 衣服干, 藤, 要, 人在, 没, 如果愿意的话):
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
                open_pw = cv2.morphologyEx(pure_white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_d, open_d)))
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
                    if in_int >= int(22 * scale_area) and out_int >= int(14 * scale_area) and scnt >= int(20 * scale_area):
                        ext_only = ((lbl_c == idx_c) & (interior_mask == 0)).astype(np.uint8) * 255
                        ne, lbl_e, st_e, _ = cv2.connectedComponentsWithStats(ext_only, connectivity=8)
                        for idx_e in range(1, ne):
                            ex, ey, ew, eh, earea = st_e[idx_e]
                            eseeds = int(np.count_nonzero(char_seeds[lbl_e == idx_e]))
                            if (
                                earea >= int(20 * scale_area)
                                and eseeds >= int(8 * scale_area)
                                and ew >= int(5 * scale_f)
                                and eh >= int(6 * scale_f)
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

                min_valid_ink_px = max(14, int(core_w * core_h * 0.008))
                if np.count_nonzero(clean_roi_mask) + np.count_nonzero(border_char_ink) >= min_valid_ink_px:
                    scaled_bub_d = max(bubble_dilate, int(round(9 * scale_f)) | 1)
                    eff_bub_kernel = block_kernel if block_kernel is not None else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scaled_bub_d, scaled_bub_d))
                    dilated_int = cv2.dilate(clean_roi_mask, eff_bub_kernel, iterations=1)
                    if block_kernel is None or custom_dilate <= 12:
                        dilated_int = cv2.bitwise_and(dilated_int, cv2.bitwise_and(interior_mask, text_zone_mask))

                    if np.count_nonzero(border_char_ink) > 0:
                        dilated_border_chars = cv2.bitwise_and(
                            cv2.dilate(border_char_ink, eff_bub_kernel, iterations=1),
                            char_zone_dil,
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

        # Для речевых бабблов со светлым фоном заливаем чистым цветом бумаги баббла даже тогда,
        # когда краевые иероглифы касались черной стенки баббла или темных волос (из-за чего p15 падал ниже 232,
        # а вызов LaMa/INPAINT_NS размазывал черную стенку внутрь белого баббла).
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
                is_white_bubble_with_wall_touch = (p50 >= 234.0 and p65 >= 242.0 and p80 >= 246.0)
                if is_uniform_light or is_white_bubble_with_wall_touch:
                    thresh_bright = p15 if is_uniform_light else max(230.0, p50)
                    bright_pixels = ring_pixels[ring_gray >= thresh_bright]
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
            pad = self.crop_padding
            uncached_blocks: list[dict[str, Any]] = []
            skipped_blocks: list[dict[str, Any]] = []
            active_block_specs: list[tuple[dict[str, Any], int, int, int, int, int, int, int, int, bool, tuple[Any, ...], tuple[np.ndarray, np.ndarray] | None]] = []

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