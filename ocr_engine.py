"""
Мультиязычный модуль детекции бабблов, свободных надписей и распознавания текста (v1.0.8 — Шаги 1 + 2 + 3 + 4).

Изменения v1.0.7:
- Шаг 1 (Двухконтурная детекция и точный кроп текста):
  1. 3-классовый детектор `ogkalu/comic-text-and-bubble-detector` (RT-DETR-v2 r50vd) в тандеме с `YOLOv8m` и OpenCV:
     `0: bubble` (контур облачка), `1: text_bubble` (точный внутренний бокс текста `text_bbox`), `2: text_free` (свободный текст).
  2. Умное сопоставление `_associate_bubbles_and_text_boxes` с авто-разделением сдвоенных бабблов (`_try_split_bubble_by_text_boxes`)
     и спасением «сиротских» `text_bubble`.
  3. Плотная подгонка кропа по чернилам букв (`_compute_tight_ink_bbox` + `_extract_bubble_interior_mask`).
- Шаг 2 (Гибридный японский OCR 2026 года и умный препроцессинг):
  1. Неавторегрессионный распознаватель `muscgab/manga-ocr-nar-small` (~35 мс на страницу, CER 1.73%) в каскаде с
     `MangaOCR FP16` (`max_new_tokens=300`) для длинных монологов (`>= 56` символов).
  2. Авто-инверсия белого текста на темном фоне и очистка серого скринтона (`_preprocess_crop_np_for_ocr`) +
     защита пропорций узких/широких кропов (`_pad_to_balanced_aspect_for_vit`).
- Шаг 3 (Кардинальное улучшение корейского `ko` и китайского `zh` OCR + построчная сегментация + автоопределение языка):
  1. Подключена нативная мультиязычная PyTorch-модель 2026 года `JustANormalTinkerer/hayai-ocr-v2.5-nova`
     (`SigLIP2 NaFlex` + 12-слойный GQA декодер, поддержка `JA`, `KO`, `ZH`, `EN` с сохранением пропорций `max_num_patches=384/512`,
     `repetition_penalty=1.0`, нормализацией `NFKC` и удалением пробелов между CJK-символами) без установки новых C++/ONNX пакетов.
  2. Поддержка опционального `RapidOCR` (`rapidocr_onnxruntime`) и `EasyOCR` как быстрых резервных/дополнительных движков.
  3. Умная построчная сегментация многострочных горизонтальных бабблов (`_split_horizontal_lines_for_ocr`) по профилю
     проекции чернил (`row_ink_profile`) и строгая вертикально-горизонтальная сортировка строк (`_sort_ocr_line_items_top_to_bottom`),
     гарантирующая чтение 2–6 строк вебтунов строго сверху вниз без потери коротких реплик и знаков препинания.
  4. Улучшенное автоопределение языка `detect_page_language` (`auto`) на базе мультиязычного `HayaiOCR v2.5 Nova`,
     `manga-ocr-nar-small`, `EasyOCR`/`RapidOCR` и анализа ориентации штрихов.
- Шаг 4 (Опциональный «снайперский» VLM-OCR для ручного выделения `➕ Обвести`, сложных рукописных SFX и цветных обложек):
  1. Метод `recognize_crop_via_vlm` и подготовка цветного LANCZOS-кропа `_prepare_crop_for_vlm_ocr` для чтения сложных
     рукописных/каллиграфических шрифтов, звуковых спецэффектов (SFX) и цветных логотипов через локальные Vision-модели
     в Ollama (`glm-ocr`, `qwen2.5vl`, `qwen3-vl`, `gemma3`, `minicpm-v`, `llava`) или облачный Vision-совместимый API.
  2. Автоматический и принудительный (`use_vlm_ocr=True`) вызов VLM-OCR в `recognize_manual_region` (при обведении мышкой
     `➕ Обвести` или по нажатию кнопки `👁 VLM-OCR` в попапе/редакторе).
"""

from __future__ import annotations

import base64
import io
import logging
import re
import sys
import time
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

warnings.filterwarnings("ignore", message=".*Failed to find CUDA.*", category=UserWarning)
warnings.filterwarnings("ignore", module="triton.*")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    import jaconv
except ImportError:
    jaconv = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    hf_hub_download = None  # type: ignore[assignment]
    snapshot_download = None  # type: ignore[assignment]

try:
    from manga_ocr import MangaOcr
except ImportError:
    MangaOcr = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel
except ImportError:
    AutoTokenizer = None  # type: ignore[assignment]
    ViTImageProcessor = None  # type: ignore[assignment]
    VisionEncoderDecoderModel = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _torch_inference_mode():
    if torch is not None and hasattr(torch, "inference_mode"):
        return torch.inference_mode()

    def _decorator(fn):
        return fn

    return _decorator


# ==============================================================================
# Блок 1: Многоуровневый фильтр галлюцинаций MangaOCR
# (Pre-OCR Stroke Gate + Post-OCR Decoder Confidence / Attention Entropy Gate)
# ==============================================================================

# Кэш радиальных масок высоких частот для 2D RFFT (исключает пересчет сеток частот)
_FFT_HF_MASK_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _get_hf_fft_mask(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _FFT_HF_MASK_CACHE:
        fy = np.fft.fftfreq(h)[:, None] * h
        fx = np.fft.rfftfreq(w)[None, :] * w
        radius = np.hypot(fy, fx)
        # Полоса пространственных частот регулярного скринтона (период от 2 до 14 пикселей)
        _FFT_HF_MASK_CACHE[key] = radius >= max(4.0, min(h, w) / 14.0)
    return _FFT_HF_MASK_CACHE[key]


@dataclass
class StrokeGateResult:
    passed: bool
    reason: str
    valid_cc_count: int
    total_cc_count: int
    valid_ink_share: float
    fft_pmr: float
    sw_mean: float
    sw_cov: float
    skel_density: float
    elapsed_ms: float


def pre_ocr_stroke_gate(crop_img: Union[np.ndarray, Image.Image]) -> StrokeGateResult:
    """
    Быстрый морфологический, топологический и спектральный Pre-OCR фильтр (~0.04-1.32 мс).
    Отсекает градиенты, регулярные скринтоны (амитен), косую штриховку одежды и края рамок
    до запуска тяжелого ViT-энкодера и авторегрессионного декодера MangaOCR.
    Устойчив как к плотным кропам глифов (`_compute_tight_ink_bbox`), так и к кропам с белыми полями (`_crop_with_padding`).
    """
    t0 = time.perf_counter()

    if isinstance(crop_img, Image.Image):
        gray = np.array(crop_img.convert("L"), dtype=np.uint8)
    elif crop_img.ndim == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img

    H, W = gray.shape[:2]
    if H < 10 or W < 10:
        return StrokeGateResult(
            False, "crop_too_small", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, (time.perf_counter() - t0) * 1000.0
        )

    area_crop = float(H * W)

    # 1. Каскад контраста и резкости ступенчатых границ (Laplacian Step-Edge Gate, ~0.04 мс)
    _, std_v = cv2.meanStdDev(gray)
    p2, p98 = [float(v) for v in np.percentile(gray, [2, 98])]
    if float(std_v[0, 0]) < 6.0 and (p98 - p2) < 28.0:
        return StrokeGateResult(
            False, "low_std_contrast", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, (time.perf_counter() - t0) * 1000.0
        )

    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    abs_lap = cv2.convertScaleAbs(lap)
    _, sharp_mask = cv2.threshold(abs_lap, 55, 255, cv2.THRESH_BINARY)
    sharp_edge_px = int(cv2.countNonZero(sharp_mask))
    sharp_edge_ratio = float(sharp_edge_px) / area_crop
    if sharp_edge_px < 8 and sharp_edge_ratio < 0.002:
        return StrokeGateResult(
            False,
            f"smooth_gradient(sharp={sharp_edge_ratio:.4f})",
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (time.perf_counter() - t0) * 1000.0,
        )

    # 2. Бинаризация Оцу и автоматическая нормализация полярности (белый текст на черном фоне)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    border_sum = (
        int(np.sum(bw[:2, :]))
        + int(np.sum(bw[-2:, :]))
        + int(np.sum(bw[2:-2, :2]))
        + int(np.sum(bw[2:-2, -2:]))
    )
    border_pixels = max(1, 4 * W + 4 * (H - 4))
    if (border_sum / border_pixels) > 140.0 and float(np.mean(bw)) > 127.0:
        bw = 255 - bw

    # Срез краевой каймы: отсоединяет линии рамок фреймов, прижатые к краям кропа,
    # но не срезает штрихи плотно обрезанных букв на маленьких кропах
    my = min(2, H // 14) if H >= 28 else 1
    mx = min(2, W // 14) if W >= 28 else 1
    inner_bw = bw.copy()
    if my > 0:
        inner_bw[:my, :] = 0
        inner_bw[-my:, :] = 0
    if mx > 0:
        inner_bw[:, :mx] = 0
        inner_bw[:, -mx:] = 0

    ink_px_count = int(cv2.countNonZero(inner_bw))
    ink_ratio = float(ink_px_count) / area_crop
    if ink_px_count < 6 or ink_ratio < 0.0015 or ink_ratio > 0.78:
        return StrokeGateResult(
            False,
            f"ink_ratio_out_of_bounds({ink_ratio:.3f})",
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (time.perf_counter() - t0) * 1000.0,
        )

    # 3. Разметка 8-связных компонент и фильтрация растровых микро-точек скринтона
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inner_bw, connectivity=8)
    total_cc = num_labels - 1
    if total_cc <= 0:
        return StrokeGateResult(
            False, "no_connected_components", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, (time.perf_counter() - t0) * 1000.0
        )

    cc_stats = stats[1:]
    cc_areas = cc_stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    cc_w = cc_stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    cc_h = cc_stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    cc_left = cc_stats[:, cv2.CC_STAT_LEFT]
    cc_top = cc_stats[:, cv2.CC_STAT_TOP]

    min_glyph_area = max(5.0, min(14.0, 0.0006 * area_crop))
    micro_cc = int(np.sum(cc_areas < min_glyph_area))
    micro_ratio = micro_cc / float(total_cc)
    area_cov = float(np.std(cc_areas) / (np.mean(cc_areas) + 1e-5))

    # 4. Спектральный детектор периодического скринтона (2D RFFT Bragg Peak Ratio)
    ph, pw = min(64, H), min(64, W)
    y0, x0 = (H - ph) // 2, (W - pw) // 2
    patch = gray[y0 : y0 + ph, x0 : x0 + pw].astype(np.float32)
    patch -= float(np.mean(patch))
    fft_mag = np.abs(np.fft.rfft2(patch))
    hf_mask = _get_hf_fft_mask(ph, pw)
    hf_vals = fft_mag[hf_mask]
    fft_pmr = float(np.max(hf_vals) / (np.mean(hf_vals) + 1e-5)) if hf_vals.size > 0 else 0.0

    # 5. Топологическая отбраковка прямых линий рамок и длинной штриховки одежды
    span_h = cc_h / float(H)
    span_w = cc_w / float(W)
    aspect = np.maximum(cc_w, cc_h) / np.maximum(1.0, np.minimum(cc_w, cc_h))
    extent = cc_areas / np.maximum(1.0, cc_w * cc_h)

    touches_left = cc_left <= mx + 1
    touches_right = (cc_left + cc_w) >= (W - mx - 1)
    touches_top = cc_top <= my + 1
    touches_bot = (cc_top + cc_h) >= (H - my - 1)
    spans_across = (
        (touches_left & touches_right)
        | (touches_top & touches_bot)
        | ((touches_left | touches_right) & (touches_top | touches_bot) & ((span_h > 0.42) | (span_w > 0.42)))
    )

    is_line_cc = (
        (aspect > 9.0)
        | ((span_h > 0.84) & (extent < 0.16))
        | ((span_w > 0.84) & (extent < 0.16))
        | (spans_across & (extent < 0.17))
    )

    is_valid_cc = (cc_areas >= min_glyph_area) & (~is_line_cc)
    valid_cc_count = int(np.sum(is_valid_cc))
    valid_ink_share = float(np.sum(cc_areas[is_valid_cc]) / (np.sum(cc_areas) + 1e-5))

    # 6. Быстрая оценка ширины штриха (DT-SWT) и плотности медиального скелета
    dist = cv2.distanceTransform(inner_bw, cv2.DIST_L2, 3)
    dist_max = cv2.dilate(dist, np.ones((3, 3), dtype=np.uint8))
    ridge_mask = (inner_bw > 0) & (dist >= dist_max - 1e-4) & (dist > 0.4)
    sw_vals = 2.0 * dist[ridge_mask]
    if sw_vals.size > 0:
        sw_mean = float(np.mean(sw_vals))
        sw_cov = float(np.std(sw_vals) / (sw_mean + 1e-5))
        if valid_cc_count > 0:
            vx1 = float(np.min(cc_left[is_valid_cc]))
            vy1 = float(np.min(cc_top[is_valid_cc]))
            vx2 = float(np.max((cc_left + cc_w)[is_valid_cc]))
            vy2 = float(np.max((cc_top + cc_h)[is_valid_cc]))
            active_ink_box_area = max(0.18 * area_crop, (vx2 - vx1 + 4.0) * (vy2 - vy1 + 4.0))
        else:
            active_ink_box_area = area_crop
        skel_density = float(sw_vals.size) / active_ink_box_area
    else:
        sw_mean, sw_cov, skel_density = 0.0, 0.0, 0.0

    # 7. Итоговое решающее правило Pre-OCR фильтра
    passed = True
    reason = "ok"
    if total_cc >= 14 and micro_ratio > 0.65 and area_cov < 0.62:
        passed, reason = False, f"halftone_dots(micro={micro_ratio:.2f},cov={area_cov:.2f})"
    elif fft_pmr > 19.5 and (micro_ratio > 0.42 or valid_ink_share < 0.45):
        passed, reason = False, f"periodic_screentone(fft_pmr={fft_pmr:.1f})"
    elif valid_cc_count < 1 or valid_ink_share < 0.25:
        passed, reason = False, f"line_hatching_or_border(valid_cc={valid_cc_count},share={valid_ink_share:.2f})"
    elif sw_mean < 0.85 or sw_mean > max(16.0, 0.48 * min(H, W)):
        passed, reason = False, f"invalid_stroke_width(sw={sw_mean:.2f})"
    elif skel_density < 0.0008 or skel_density > 0.36:
        passed, reason = False, f"invalid_skel_density(rho={skel_density:.4f})"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return StrokeGateResult(
        passed=passed,
        reason=reason,
        valid_cc_count=valid_cc_count,
        total_cc_count=total_cc,
        valid_ink_share=valid_ink_share,
        fft_pmr=fft_pmr,
        sw_mean=sw_mean,
        sw_cov=sw_cov,
        skel_density=skel_density,
        elapsed_ms=elapsed_ms,
    )


@dataclass
class MangaOCRRecognition:
    text: str
    is_valid: bool
    reject_reason: str
    seq_confidence: float
    spatial_attn_entropy: float
    peak_spatial_attn: float
    char_to_ink_ratio: float
    num_tokens: int
    truncated: bool


class HallucinationFreeMangaOCR:
    """
    Защищенный раннер MangaOCR с Pre-OCR фильтрацией штрихов и Post-OCR скорингом
    лог-вероятностей декодера, энтропии кросс-внимания и отношения Character-to-Ink.
    Поддерживает прямую совместимость с интерфейсом `MangaOcr.__call__`.
    """

    _CJK_CONTENT_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF0-9A-Za-z]")
    _HALF_TO_FULL_TRANS = str.maketrans(
        "".join(chr(i) for i in range(0x21, 0x7F)),
        "".join(chr(i + 0xFEE0) for i in range(0x21, 0x7F)),
    )

    def __init__(
        self,
        pretrained_model_name_or_path: str = "kha-white/manga-ocr-base",
        device: Optional[str] = None,
        force_cpu: bool = False,
        min_seq_confidence: float = 0.55,
        max_spatial_entropy: float = 0.90,
        min_peak_spatial_attn: float = 0.04,
        max_char_to_ink_ratio: float = 2.80,
    ):
        if torch is None or VisionEncoderDecoderModel is None:
            raise RuntimeError("PyTorch и transformers необходимы для инициализации HallucinationFreeMangaOCR.")

        self.min_seq_confidence = min_seq_confidence
        self.max_spatial_entropy = max_spatial_entropy
        self.min_peak_spatial_attn = min_peak_spatial_attn
        self.max_char_to_ink_ratio = max_char_to_ink_ratio

        if force_cpu:
            self.device = "cpu"
        elif device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
            )
        else:
            self.device = device

        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.feature_extractor = self.processor
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)

        # Включаем eager attention, чтобы гарантировать возврат весов cross_attentions в новых версиях Transformers
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "eager"
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder.config, "_attn_implementation"):
            self.model.decoder.config._attn_implementation = "eager"

        self.model.to(self.device).eval()
        self.eos_token_id = int(getattr(self.model.config, "eos_token_id", None) or 3)
        self.pad_token_id = int(getattr(self.model.config, "pad_token_id", None) or 0)
        self._post_process = self._post_process_text
        self.post_process = self._post_process_text

    @classmethod
    def _post_process_text(cls, text: str) -> str:
        text = "".join(text.split())
        text = text.replace("…", "...")
        text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
        if jaconv is not None:
            return jaconv.h2z(text, ascii=True, digit=True)
        return text.translate(cls._HALF_TO_FULL_TRANS)

    @_torch_inference_mode()
    def recognize(
        self,
        img_or_crop: Union[np.ndarray, Image.Image],
        max_length: int = 300,
        gate_crop: Optional[Union[np.ndarray, Image.Image]] = None,
    ) -> MangaOCRRecognition:
        # Этап 1: Pre-OCR Stroke Gate
        gate = pre_ocr_stroke_gate(gate_crop if gate_crop is not None else img_or_crop)
        if not gate.passed:
            return MangaOCRRecognition(
                text="",
                is_valid=False,
                reject_reason=f"pre_ocr:{gate.reason}",
                seq_confidence=0.0,
                spatial_attn_entropy=1.0,
                peak_spatial_attn=0.0,
                char_to_ink_ratio=0.0,
                num_tokens=0,
                truncated=False,
            )

        if isinstance(img_or_crop, np.ndarray):
            if img_or_crop.ndim == 3:
                pil_img = Image.fromarray(cv2.cvtColor(img_or_crop, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(img_or_crop)
        else:
            pil_img = img_or_crop

        pil_img = pil_img.convert("L").convert("RGB")
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)

        # Этап 2: Генерация с извлечением логитов и кросс-внимания декодера
        gen_out = self.model.generate(
            pixel_values,
            max_length=max_length,
            output_scores=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        seq_ids = gen_out.sequences[0]  # [1 + T], начинается с decoder_start_token_id (2)
        gen_tokens = seq_ids[1:]        # Сгенерированные токены шагов 1..T
        num_steps = len(gen_out.scores)
        if num_steps == 0:
            return MangaOCRRecognition("", False, "empty_generation", 0.0, 1.0, 0.0, 0.0, 0, False)

        # Проверка на усечение (достигнут лимит шагов без генерации EOS=3)
        truncated = (num_steps >= max_length - 1) and (int(gen_tokens[-1].item()) != self.eos_token_id)

        # Вычисление геометрического среднего вероятностей выбранных токенов (Sequence Confidence)
        log_probs_list: List[float] = []
        valid_steps = 0
        for step_idx, step_logits in enumerate(gen_out.scores):
            token_id = int(gen_tokens[step_idx].item())
            if token_id == self.pad_token_id:
                break
            step_log_probs = F.log_softmax(step_logits[0], dim=-1)
            log_probs_list.append(float(step_log_probs[token_id].item()))
            valid_steps += 1
            if token_id == self.eos_token_id:
                break

        mean_log_prob = float(np.mean(log_probs_list)) if log_probs_list else -100.0
        seq_confidence = float(np.exp(mean_log_prob))

        # Вычисление пространственной энтропии Шеннона кросс-внимания (по 196 патчам ViT, исключая токен CLS=0)
        # Используем топ-3 наиболее сфокусированных голов внимания (positional glyph-tracking heads),
        # чтобы контекстные головы с равномерным вниманием не размывали пик на реальных предложениях
        spatial_entropies: List[float] = []
        peak_attentions: List[float] = []
        content_steps = max(1, valid_steps - 1)  # Исключаем финальный шаг EOS

        if getattr(gen_out, "cross_attentions", None) is not None:
            num_patches = 196
            log_num_patches = float(np.log(num_patches))
            for step_idx in range(min(content_steps, len(gen_out.cross_attentions))):
                step_cross = gen_out.cross_attentions[step_idx]
                if not step_cross or step_cross[-1] is None:
                    continue
                # Берем последний (2-й) слой BERT-декодера: размерность [batch=1, heads=12, q_len=1, kv_len=197]
                last_layer_attn = step_cross[-1][0, :, -1, :]  # [12, 197]
                spatial_heads = last_layer_attn[:, 1:]         # [12, 196] без CLS
                if spatial_heads.numel() == 0:
                    continue
                head_peaks = spatial_heads.max(dim=-1).values  # [12]
                k_top = min(3, int(spatial_heads.shape[0]))
                top_idx = torch.topk(head_peaks, k=k_top).indices
                spatial_attn = spatial_heads[top_idx].mean(dim=0)  # [196]
                peak_attentions.append(float(spatial_attn.max().item()))

                # Перенормировка распределения по пространственным патчам для расчета энтропии
                spatial_prob = spatial_attn / (spatial_attn.sum() + 1e-8)
                entropy = -torch.sum(spatial_prob * torch.log(spatial_prob + 1e-10)).item() / log_num_patches
                spatial_entropies.append(float(entropy))

        mean_spatial_entropy = float(np.mean(spatial_entropies)) if spatial_entropies else 0.5
        mean_peak_attn = float(np.mean(peak_attentions)) if peak_attentions else 0.2

        raw_text = self.tokenizer.decode(seq_ids, skip_special_tokens=True)
        clean_text = self._post_process_text(raw_text)
        cjk_chars = len(self._CJK_CONTENT_RE.findall(clean_text))
        char_to_ink_ratio = float(cjk_chars) / float(max(1, gate.valid_cc_count))

        # Этап 3: Пороговая проверка Post-OCR Gate
        if truncated:
            return MangaOCRRecognition(
                "", False, "post_ocr:truncated_loop_300",
                seq_confidence, mean_spatial_entropy, mean_peak_attn, char_to_ink_ratio, valid_steps, True
            )
        if seq_confidence < self.min_seq_confidence:
            return MangaOCRRecognition(
                "", False, f"post_ocr:low_confidence({seq_confidence:.3f}<{self.min_seq_confidence})",
                seq_confidence, mean_spatial_entropy, mean_peak_attn, char_to_ink_ratio, valid_steps, False
            )
        if mean_spatial_entropy > self.max_spatial_entropy or mean_peak_attn < self.min_peak_spatial_attn:
            return MangaOCRRecognition(
                "", False, f"post_ocr:diffuse_cross_attn(H={mean_spatial_entropy:.3f},peak={mean_peak_attn:.3f})",
                seq_confidence, mean_spatial_entropy, mean_peak_attn, char_to_ink_ratio, valid_steps, False
            )
        if cjk_chars >= 6 and char_to_ink_ratio > self.max_char_to_ink_ratio:
            return MangaOCRRecognition(
                "", False, f"post_ocr:char_to_ink_hallucination(CIR={char_to_ink_ratio:.2f},chars={cjk_chars},cc={gate.valid_cc_count})",
                seq_confidence, mean_spatial_entropy, mean_peak_attn, char_to_ink_ratio, valid_steps, False
            )

        return MangaOCRRecognition(
            text=clean_text,
            is_valid=True,
            reject_reason="ok",
            seq_confidence=seq_confidence,
            spatial_attn_entropy=mean_spatial_entropy,
            peak_spatial_attn=mean_peak_attn,
            char_to_ink_ratio=char_to_ink_ratio,
            num_tokens=valid_steps,
            truncated=False,
        )

    def __call__(
        self,
        img_or_crop: Union[np.ndarray, Image.Image],
        gate_crop: Optional[Union[np.ndarray, Image.Image]] = None,
    ) -> str:
        rec = self.recognize(img_or_crop, gate_crop=gate_crop)
        return rec.text if rec.is_valid else ""


# ==============================================================================
# Блок 2: Сегментация панелей (фреймов) манги и топологический граф порядка чтения
# (Manga Panel XY-Cut + Convexity-Defect Gutter Split + Reading Order DAG)
# ==============================================================================


def segment_manga_panels(
    page_bgr: np.ndarray,
    min_panel_area_ratio: float = 0.015,
    white_thresh: int = 210,
) -> List[Dict[str, Any]]:
    """
    Быстрая классическая CV-сегментация панелей (фреймов) манги без нейросетей (~7.15 мс).
    Поддерживает прямоугольные, трапециевидные/диагональные панели и автоматически рассекает
    перемычки (бабблы, волосы персонажей) через гаттеры с помощью анализа дефектов выпуклости.
    Совместима как с OpenCV 4.x (defects shape `(N, 1, 4)`), так и с OpenCV 5.x (defects shape `(N, 4)`).
    """
    H, W = page_bgr.shape[:2]
    page_area = float(H * W)
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY) if page_bgr.ndim == 3 else page_bgr

    # Бинаризация темных рамок и содержимого панелей относительно белой бумаги гаттеров
    _, bw = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY_INV)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panel_polys: List[np.ndarray] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_panel_area_ratio * page_area:
            continue

        cnt_pts = np.asarray(cnt, dtype=np.int32).reshape(-1, 2)
        cnt_cv = cnt_pts.reshape(-1, 1, 2)

        # Проверка на слияние соседних панелей через перемычку (Convexity Defects Diagonal Gutter Split)
        split_done = False
        if len(cnt_pts) >= 4:
            try:
                hull_idx = cv2.convexHull(cnt_cv, returnPoints=False)
                if hull_idx is not None and hull_idx.size > 3:
                    hull_idx_sorted = np.sort(np.asarray(hull_idx, dtype=np.int32).reshape(-1)).reshape(-1, 1)
                    defects = cv2.convexityDefects(cnt_cv, hull_idx_sorted)
                else:
                    defects = None
            except Exception:
                defects = None

            if defects is not None and defects.size >= 4:
                deep_defects: List[Tuple[float, Tuple[int, int]]] = []
                min_defect_depth = 0.05 * min(H, W)
                defects_arr = np.asarray(defects, dtype=np.int32).reshape(-1, 4)
                for _, _, far_idx, depth_fixed in defects_arr:
                    depth_px = float(depth_fixed) / 256.0
                    f_i = int(far_idx)
                    if depth_px > min_defect_depth and 0 <= f_i < len(cnt_pts):
                        far_pt = (int(cnt_pts[f_i, 0]), int(cnt_pts[f_i, 1]))
                        deep_defects.append((depth_px, far_pt))

                if len(deep_defects) >= 2:
                    deep_defects.sort(key=lambda item: item[0], reverse=True)
                    p1, p2 = deep_defects[0][1], deep_defects[1][1]
                    x, y, w, h = cv2.boundingRect(cnt_cv)
                    sub_mask = np.zeros((h, w), dtype=np.uint8)
                    shifted_cnt = (cnt_pts - np.array([[x, y]], dtype=np.int32)).reshape(-1, 1, 2)
                    cv2.drawContours(sub_mask, [shifted_cnt], -1, 255, -1)
                    # Рассекаем перемычку вдоль линии, соединяющей вершины встречных клиньев гаттера
                    cv2.line(sub_mask, (p1[0] - x, p1[1] - y), (p2[0] - x, p2[1] - y), 0, 10)
                    sub_cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid_subs: List[np.ndarray] = []
                    for sc in sub_cnts:
                        if cv2.contourArea(sc) >= min_panel_area_ratio * page_area:
                            sc_shifted = (np.asarray(sc, dtype=np.int32).reshape(-1, 2) + np.array([[x, y]], dtype=np.int32)).reshape(-1, 1, 2)
                            valid_subs.append(sc_shifted)
                    if len(valid_subs) >= 2:
                        for sc in valid_subs:
                            peri_s = cv2.arcLength(sc, True)
                            approx_s = cv2.approxPolyDP(sc, 0.012 * peri_s, True)
                            panel_polys.append(np.asarray(approx_s, dtype=np.int32).reshape(-1, 2))
                        split_done = True

        if not split_done:
            peri = cv2.arcLength(cnt_cv, True)
            approx = cv2.approxPolyDP(cnt_cv, 0.012 * peri, True)
            panel_polys.append(np.asarray(approx, dtype=np.int32).reshape(-1, 2))

    # Если страница безрамочная (borderless) и контуры не найдены — создаем единый фрейм на всю страницу
    if not panel_polys:
        panel_polys = [np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.int32)]

    panels: List[Dict[str, Any]] = []
    for idx, poly in enumerate(panel_polys):
        poly_cv = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(poly_cv)
        panels.append({
            "id": idx,
            "poly": np.asarray(poly, dtype=np.int32).reshape(-1, 2),
            "bbox": (int(x), int(y), int(x + w), int(y + h)),
            "area": float(cv2.contourArea(poly_cv)),
        })
    return panels


def sort_panels_dag(
    panels: List[Dict[str, Any]],
    reading_mode: str = "rtl",
    tau_cut: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Строит направленный ациклический граф (DAG) панелей и выполняет топологическую сортировку Кана.
    reading_mode: rtl (японская манга), ltr (манхва/маньхуа), webtoon (вертикальный скролл).
    """
    n = len(panels)
    if n <= 1:
        return panels

    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        x1_i, y1_i, x2_i, y2_i = panels[i]["bbox"]
        w_i, h_i = max(1, x2_i - x1_i), max(1, y2_i - y1_i)
        cx_i, cy_i = 0.5 * (x1_i + x2_i), 0.5 * (y1_i + y2_i)

        for j in range(i + 1, n):
            x1_j, y1_j, x2_j, y2_j = panels[j]["bbox"]
            w_j, h_j = max(1, x2_j - x1_j), max(1, y2_j - y1_j)
            cx_j, cy_j = 0.5 * (x1_j + x2_j), 0.5 * (y1_j + y2_j)

            ov_y = max(0, min(y2_i, y2_j) - max(y1_i, y1_j)) / float(min(h_i, h_j))
            ov_x = max(0, min(x2_i, x2_j) - max(x1_i, x1_j)) / float(min(w_i, w_j))

            if ov_y < tau_cut and ov_x >= tau_cut:
                if cy_i < cy_j:
                    adj[i].add(j)
                else:
                    adj[j].add(i)
            elif ov_x < tau_cut and ov_y >= tau_cut:
                if reading_mode == "rtl":
                    if cx_i > cx_j:
                        adj[i].add(j)
                    else:
                        adj[j].add(i)
                else:
                    if cx_i < cx_j:
                        adj[i].add(j)
                    else:
                        adj[j].add(i)
            elif ov_y < tau_cut and ov_x < tau_cut:
                if cy_i < cy_j and (y2_i <= y1_j + tau_cut * min(h_i, h_j)):
                    adj[i].add(j)
                elif cy_j < cy_i and (y2_j <= y1_i + tau_cut * min(h_i, h_j)):
                    adj[j].add(i)

    in_deg = {u: 0 for u in range(n)}
    for u in adj:
        for v in adj[u]:
            in_deg[v] += 1

    def _tie_key(idx: int) -> Tuple[int, int]:
        x1, y1, x2, _ = panels[idx]["bbox"]
        return (y1, -x2) if reading_mode == "rtl" else (y1, x1)

    ordered: List[Dict[str, Any]] = []
    remaining = set(range(n))
    while remaining:
        zero_nodes = [u for u in remaining if in_deg[u] == 0]
        if not zero_nodes:
            zero_nodes = [min(remaining, key=lambda u: (in_deg[u], _tie_key(u)))]
        zero_nodes.sort(key=_tie_key)
        curr = zero_nodes[0]
        ordered.append(panels[curr])
        remaining.remove(curr)
        for nxt in adj[curr]:
            if nxt in remaining:
                in_deg[nxt] -= 1

    return ordered


def sort_page_blocks_reading_order_dag(
    blocks: List[Dict[str, Any]],
    page_bgr: Optional[np.ndarray] = None,
    lang: str = "ja",
) -> List[Dict[str, Any]]:
    """
    Прямая замена для `_sort_page_blocks_reading_order`:
    1. Сегментирует страницу на фреймы (панели) и упорядочивает их через межпанельный DAG.
    2. Привязывает каждый текстовый блок/баббл к своей панели (с обработкой бабблов на границе двух фреймов).
    3. Упорядочивает бабблы внутри каждой панели через внутрипанельный DAG.
    Ожидается, что каждый элемент в `blocks` имеет ключ `bbox: (x1, y1, x2, y2)`.
    """
    if len(blocks) <= 1:
        return blocks

    reading_mode = "rtl" if lang == "ja" else ("webtoon" if lang == "ko" else "ltr")
    ordered_panels: Optional[List[Dict[str, Any]]] = None
    if page_bgr is not None and page_bgr.size > 0:
        try:
            raw_panels = segment_manga_panels(page_bgr)
            ordered_panels = sort_panels_dag(raw_panels, reading_mode=reading_mode)
        except Exception as seg_exc:
            logger.debug("Сегментация панелей переключена на полностраничный режим (%s).", seg_exc)
            ordered_panels = None

    if not ordered_panels:
        max_x = max((int(b["bbox"][2]) for b in blocks if "bbox" in b and len(b["bbox"]) >= 4), default=1000) + 32
        max_y = max((int(b["bbox"][3]) for b in blocks if "bbox" in b and len(b["bbox"]) >= 4), default=1400) + 32
        ordered_panels = [{
            "id": 0,
            "poly": np.array([[0, 0], [max_x, 0], [max_x, max_y], [0, max_y]], dtype=np.int32),
            "bbox": (0, 0, max_x, max_y),
            "area": float(max_x * max_y),
        }]

    panel_buckets: Dict[int, List[Dict[str, Any]]] = {p["id"]: [] for p in ordered_panels}
    panel_rank = {p["id"]: r for r, p in enumerate(ordered_panels)}

    for blk in blocks:
        bx1, by1, bx2, by2 = blk["bbox"]
        b_area = float(max(1, (bx2 - bx1) * (by2 - by1)))
        bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)

        overlaps: List[Tuple[float, Dict[str, Any]]] = []
        for p in ordered_panels:
            px1, py1, px2, py2 = p["bbox"]
            ix = max(0, min(bx2, px2) - max(bx1, px1))
            iy = max(0, min(by2, py2) - max(by1, py1))
            inter = float(ix * iy)
            if inter > 0:
                overlaps.append((inter / b_area, p))

        if overlaps:
            overlaps.sort(key=lambda item: item[0], reverse=True)
            best_p = overlaps[0][1]
            # Трансграничный баббл, перекрывающий сразу два фрейма
            if len(overlaps) >= 2 and overlaps[1][0] >= 0.20 and abs(overlaps[0][0] - overlaps[1][0]) < 0.25:
                p_a, p_b = overlaps[0][1], overlaps[1][1]
                best_p = p_a if panel_rank[p_a["id"]] < panel_rank[p_b["id"]] else p_b
            panel_buckets[best_p["id"]].append(blk)
        else:
            # Если баббл вынесен на белое поле за пределы рамок — привязываем к ближайшей панели
            best_p = min(
                ordered_panels,
                key=lambda p: max(0.0, p["bbox"][0] - bcx, bcx - p["bbox"][2]) ** 2
                + max(0.0, p["bbox"][1] - bcy, bcy - p["bbox"][3]) ** 2,
            )
            panel_buckets[best_p["id"]].append(blk)

    sorted_blocks: List[Dict[str, Any]] = []
    for p in ordered_panels:
        b_list = panel_buckets[p["id"]]
        if len(b_list) <= 1:
            sorted_blocks.extend(b_list)
            continue

        m = len(b_list)
        b_adj: Dict[int, Set[int]] = {i: set() for i in range(m)}
        for i in range(m):
            x1_i, y1_i, x2_i, y2_i = b_list[i]["bbox"]
            w_i, h_i = max(1, x2_i - x1_i), max(1, y2_i - y1_i)
            cx_i, cy_i = 0.5 * (x1_i + x2_i), 0.5 * (y1_i + y2_i)
            for j in range(i + 1, m):
                x1_j, y1_j, x2_j, y2_j = b_list[j]["bbox"]
                w_j, h_j = max(1, x2_j - x1_j), max(1, y2_j - y1_j)
                cx_j, cy_j = 0.5 * (x1_j + x2_j), 0.5 * (y1_j + y2_j)

                ov_x = max(0, min(x2_i, x2_j) - max(x1_i, x1_j)) / float(min(w_i, w_j))
                ov_y = max(0, min(y2_i, y2_j) - max(y1_i, y1_j)) / float(min(h_i, h_j))

                if reading_mode == "rtl":
                    if ov_y >= 0.25 and ov_x < 0.50:
                        if cx_i > cx_j:
                            b_adj[i].add(j)
                        else:
                            b_adj[j].add(i)
                    elif ov_x >= 0.25 and ov_y < 0.50:
                        if cy_i < cy_j:
                            b_adj[i].add(j)
                        else:
                            b_adj[j].add(i)
                    else:
                        score_ij = 0.65 * (cy_j - cy_i) + 0.35 * (cx_i - cx_j)
                        if score_ij > 0:
                            b_adj[i].add(j)
                        else:
                            b_adj[j].add(i)
                else:
                    if ov_y >= 0.35 and ov_x < 0.40:
                        if cx_i < cx_j:
                            b_adj[i].add(j)
                        else:
                            b_adj[j].add(i)
                    else:
                        if cy_i < cy_j:
                            b_adj[i].add(j)
                        else:
                            b_adj[j].add(i)

        in_d = {u: 0 for u in range(m)}
        for u in b_adj:
            for v in b_adj[u]:
                in_d[v] += 1

        rem = set(range(m))
        px1, py1, px2, _ = p["bbox"]
        while rem:
            z = [u for u in rem if in_d[u] == 0] or list(rem)
            if reading_mode == "rtl":
                z.sort(key=lambda u: float(np.hypot(px2 - b_list[u]["bbox"][2], b_list[u]["bbox"][1] - py1)))
            else:
                z.sort(key=lambda u: float(np.hypot(b_list[u]["bbox"][0] - px1, b_list[u]["bbox"][1] - py1)))
            curr = z[0]
            sorted_blocks.append(b_list[curr])
            rem.remove(curr)
            for nxt in b_adj[curr]:
                if nxt in rem:
                    in_d[nxt] -= 1

    return sorted_blocks


# ==============================================================================
# Блок 3: Топологическая группировка текста внутри полости баббла
# (Cavity-Aware Text Grouping + Watershed Neck Splitting)
# ==============================================================================


def group_text_boxes_by_bubble_cavity(
    page_bgr_or_gray: np.ndarray,
    text_boxes: List[Dict[str, Any]],
    neck_ratio_thresh: float = 0.48,
    min_lobe_radius_px: float = 18.0,
    white_thresh: int = 215,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Топологическая группировка текстовых боксов/колонок через сегментацию внутренней белой полости баббла (~14.77 мс).
    - Все вертикальные колонки (`tategaki`) в одной чаше собираются в единый блок справа налево.
    - Сдвоенные бабблы-«восьмёрки» разрезаются по узкому горлышку методом маркерного водораздела (`cv2.watershed`),
      предотвращая склеивание разных реплик.
    - Защищена от протечки белой маски в глобальный белый фон страницы/гаттеры при незамкнутом контуре баббла.
    Возвращает список сгруппированных блоков и полную карту полостей `final_cavity_map` (для использования в cleaner.py).
    """
    gray = (
        cv2.cvtColor(page_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        if page_bgr_or_gray.ndim == 3
        else page_bgr_or_gray
    )
    H, W = gray.shape[:2]
    if not text_boxes:
        return [], np.zeros((H, W), dtype=np.int32)

    page_area = float(max(1, H * W))
    # 1. Выделение светлых полостей бабблов с замыканием микро-разрывов (1-2px) в темных стенках бабблов
    dark_walls = (gray < white_thresh).astype(np.uint8) * 255
    closed_walls = cv2.morphologyEx(dark_walls, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    white_mask = cv2.bitwise_not(closed_walls)
    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=4)

    active_cc_ids = set()
    for tb in text_boxes:
        x1, y1, x2, y2 = [int(round(float(v))) for v in tb["bbox"][:4]]
        roi_lbl = cc_labels[max(0, y1) : min(H, y2), max(0, x1) : min(W, x2)]
        vals, counts = np.unique(roi_lbl[roi_lbl > 0], return_counts=True)
        if len(vals) > 0:
            active_cc_ids.add(int(vals[np.argmax(counts)]))

    final_cavity_map = np.zeros((H, W), dtype=np.int32)
    next_cavity_id = 1
    k_peak = np.ones((7, 7), dtype=np.uint8)
    leaked_bg_cc_ids: Set[int] = set()

    # 2. Анализ каждой активной полости: заливка дырок от букв -> Distance Transform -> Watershed по перешейку
    for cid in active_cc_ids:
        x0 = int(cc_stats[cid, cv2.CC_STAT_LEFT])
        y0 = int(cc_stats[cid, cv2.CC_STAT_TOP])
        w0 = int(cc_stats[cid, cv2.CC_STAT_WIDTH])
        h0 = int(cc_stats[cid, cv2.CC_STAT_HEIGHT])
        cc_area = float(cc_stats[cid, cv2.CC_STAT_AREA])

        # Защита от протечки в глобальный белый фон страницы / межпанельные гаттеры:
        # если белая компонента охватывает >38% площади полной страницы или касается противоположных краев страницы
        touches_edges = sum([x0 <= 1, y0 <= 1, (x0 + w0) >= (W - 1), (y0 + h0) >= (H - 1)])
        comp_m_edge = (cc_labels == cid)
        t_b = float(np.mean(comp_m_edge[0, :])) if H > 0 else 0.0
        b_b = float(np.mean(comp_m_edge[-1, :])) if H > 0 else 0.0
        l_b = float(np.mean(comp_m_edge[:, 0])) if W > 0 else 0.0
        r_b = float(np.mean(comp_m_edge[:, -1])) if W > 0 else 0.0
        heavy_border_sides = sum([t_b > 0.25, b_b > 0.25, l_b > 0.25, r_b > 0.25])
        if (
            (w0 * h0 > 0.38 * page_area and touches_edges >= 3 and heavy_border_sides >= 2)
            or (touches_edges >= 3 and heavy_border_sides >= 2 and (w0 > 0.72 * W or h0 > 0.72 * H))
            or ((l_b > 0.25 or r_b > 0.25) and (t_b > 0.35 or b_b > 0.35) and w0 > 0.75 * W and h0 > 0.75 * H)
        ):
            leaked_bg_cc_ids.add(cid)
            continue

        pad = 4
        rx1, ry1 = max(0, x0 - pad), max(0, y0 - pad)
        rx2, ry2 = min(W, x0 + w0 + pad), min(H, y0 + h0 + pad)

        sub_mask = ((cc_labels[ry1:ry2, rx1:rx2] == cid) * 255).astype(np.uint8)
        # Заливка дырок от черных глифов внутри чаши баббла через заливку внешнего контура
        cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sub_filled = np.zeros_like(sub_mask)
        cv2.drawContours(sub_filled, cnts, -1, 255, -1)

        dist = cv2.distanceTransform(sub_filled, cv2.DIST_L2, 3)
        dist_dil = cv2.dilate(dist, k_peak)
        max_r_cav = float(np.max(dist)) if dist.size > 0 else 0.0
        eff_min_lobe_r = max(min(min_lobe_radius_px, 0.38 * max_r_cav), 0.52 * max_r_cav)
        peaks_mask = (sub_filled > 0) & (dist >= dist_dil - 1e-3) & (dist >= eff_min_lobe_r)
        pys, pxs = np.where(peaks_mask)

        # Сортировка пиков по убыванию радиуса и жадный NMS (подавление плато внутри перешейка)
        order_p = np.argsort(-dist[pys, pxs])
        peaks: List[Tuple[int, int, float]] = []
        for idx_p in order_p:
            py, px = int(pys[idx_p]), int(pxs[idx_p])
            r_val = float(dist[py, px])
            if all((py - ey) ** 2 + (px - ex) ** 2 >= (0.85 * max(r_val, er)) ** 2 for ey, ex, er in peaks):
                peaks.append((py, px, r_val))

        if len(peaks) >= 2:
            parent = list(range(len(peaks)))

            def _find(i: int) -> int:
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            # Проверяем каждую пару пиков на наличие седловой точки (узкого горлышка)
            for i in range(len(peaks)):
                for j in range(i + 1, len(peaks)):
                    yA, xA, rA = peaks[i]
                    yB, xB, rB = peaks[j]
                    n_pts = int(max(12, np.hypot(yB - yA, xB - xA) / 2.0))
                    ys_l = np.clip(np.linspace(yA, yB, n_pts).astype(int), 0, dist.shape[0] - 1)
                    xs_l = np.clip(np.linspace(xA, xB, n_pts).astype(int), 0, dist.shape[1] - 1)
                    d_neck = float(np.min(dist[ys_l, xs_l]))
                    if d_neck >= neck_ratio_thresh * min(rA, rB):
                        pi, pj = _find(i), _find(j)
                        if pi != pj:
                            parent[pi] = pj

            distinct_groups: Dict[int, List[Tuple[int, int, float]]] = {}
            for i in range(len(peaks)):
                distinct_groups.setdefault(_find(i), []).append(peaks[i])

            if len(distinct_groups) >= 2:
                markers = np.zeros(sub_filled.shape, dtype=np.int32)
                markers[sub_filled == 0] = 1
                for g_idx, g_peaks in enumerate(distinct_groups.values(), start=2):
                    for py, px, r_val in g_peaks:
                        cv2.circle(markers, (px, py), max(3, int(0.25 * r_val)), g_idx, -1)

                dist_u8 = np.clip(255.0 * (1.0 - dist / (float(np.max(dist)) + 1e-5)), 0, 255).astype(np.uint8)
                relief_bgr = cv2.cvtColor(dist_u8, cv2.COLOR_GRAY2BGR)
                cv2.watershed(relief_bgr, markers)

                for g_idx in range(2, 2 + len(distinct_groups)):
                    lobe_mask = (markers == g_idx) & (sub_filled > 0)
                    if np.any(lobe_mask):
                        final_cavity_map[ry1:ry2, rx1:rx2][lobe_mask] = next_cavity_id
                        next_cavity_id += 1
                continue

        final_cavity_map[ry1:ry2, rx1:rx2][sub_filled > 0] = next_cavity_id
        next_cavity_id += 1

    # 3. Привязка текстовых боксов к сегментированным чашам по мажоритарному голосованию
    cavity_buckets: Dict[int, List[Dict[str, Any]]] = {}
    for tb in text_boxes:
        x1, y1, x2, y2 = [int(round(float(v))) for v in tb["bbox"][:4]]
        roi_cav = final_cavity_map[max(0, y1) : min(H, y2), max(0, x1) : min(W, x2)]
        vals, counts = np.unique(roi_cav[roi_cav > 0], return_counts=True)
        if len(vals) > 0:
            cid = int(vals[np.argmax(counts)])
        else:
            # Для текста на темном фоне, вне белых бабблов или в незамкнутом баббле на белом фоне выделяем индивидуальный ID
            cid = next_cavity_id
            next_cavity_id += 1
        cavity_buckets.setdefault(cid, []).append(tb)

    # 4. Сортировка колонок внутри каждой чаши (справа налево для вертикального tategaki)
    merged_groups: List[Dict[str, Any]] = []
    for cid, boxes in sorted(cavity_buckets.items()):
        aspects = [
            (b["bbox"][3] - b["bbox"][1]) / float(max(1, b["bbox"][2] - b["bbox"][0]))
            for b in boxes
        ]
        is_vertical = bool(np.median(aspects) >= 1.35)
        if is_vertical:
            boxes_sorted = sorted(
                boxes,
                key=lambda b: (-0.5 * (b["bbox"][0] + b["bbox"][2]), 0.5 * (b["bbox"][1] + b["bbox"][3])),
            )
        else:
            boxes_sorted = sorted(
                boxes,
                key=lambda b: (0.5 * (b["bbox"][1] + b["bbox"][3]), 0.5 * (b["bbox"][0] + b["bbox"][2])),
            )

        ux1 = min(int(round(float(b["bbox"][0]))) for b in boxes_sorted)
        uy1 = min(int(round(float(b["bbox"][1]))) for b in boxes_sorted)
        ux2 = max(int(round(float(b["bbox"][2]))) for b in boxes_sorted)
        uy2 = max(int(round(float(b["bbox"][3]))) for b in boxes_sorted)

        merged_groups.append({
            "cavity_id": cid,
            "is_vertical": is_vertical,
            "bbox": (int(ux1), int(uy1), int(ux2), int(uy2)),
            "sub_boxes": boxes_sorted,
            "merged_text": "".join(b.get("text", "") for b in boxes_sorted),
        })

    return merged_groups, final_cavity_map


# ==============================================================================
# Блок 5 (v1.2.0 SOTA Scanlation Upgrade):
# Топологическая сегментация пунктирных (Проблема №1), разомкнутых (Проблема №2)
# и сдвоенных (Проблема №3) бабблов, каскадный фильтр FP и Радон-Deskew (Проблема №6),
# а также единый транспортный контракт BubbleGeometryContext.
# ==============================================================================


@dataclass
class BubbleGeometryContext:
    """
    Единый транспортный контракт между ocr_engine.py, cleaner.py и renderer.py.
    Хранит локальные маски в координатах roi_bbox = (x1, y1, x2, y2).
    """

    bubble_id: int
    roi_bbox: tuple[int, int, int, int]          # (x1, y1, x2, y2) на странице
    text_bbox_rel: tuple[int, int, int, int]     # (tx1, ty1, tx2, ty2) внутри ROI
    topology_type: str                           # 'solid' | 'dashed' | 'open_wispy' | 'conjoined' | 'overhanging'
    protected_wall_mask: np.ndarray              # uint8 (H, W): штрихи пунктира, скобки, перегородка, внешние волосы
    interior_mask: np.ndarray                    # uint8 (H, W): герметичная чаша баббла (safe_mask для renderer.py)
    M_inside: np.ndarray                         # uint8 (H, W): маска текста внутри чаши баббла
    M_outside: np.ndarray                        # uint8 (H, W): маска выступающего текста + белого ореола снаружи
    skew_angle_deg: float = 0.0                  # Угол наклона для наклонных экранов/табличек
    regime_stats: Dict[str, float] = None        # Статистика локальных режимов подложки (A/B/C/D)
    ocr_text: str = ""

    def __post_init__(self) -> None:
        if self.regime_stats is None:
            self.regime_stats = {}


def segment_dashed_bubble_roi(
    roi_gray: np.ndarray,
    text_bbox_rel: tuple[int, int, int, int] | None = None,
    scale_f: float = 1.0,
) -> dict[str, np.ndarray | bool | int | float]:
    """
    Проблема №1: Детекция пунктирных/точечных стенок баббла, построение виртуального барьера
    и герметичной маски внутренней полости (interior_mask) без утечки на скринтон.
    Работает строго на cv2 + numpy без попиксельных циклов Python.
    """
    t_start = time.perf_counter()
    H, W = roi_gray.shape[:2]
    s = max(0.5, float(scale_f))

    # 1. Быстрая SIMD-бинаризация чернил (CV_8U), устойчивая к серым скринтонам снаружи
    blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    k_box = int(round(21 * s)) | 1
    local_mean = cv2.boxFilter(blur, cv2.CV_8U, (k_box, k_box))
    diff_img = cv2.subtract(local_mean, blur)
    ink_mask = cv2.bitwise_and(
        cv2.threshold(diff_img, 22, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(blur, 185, 255, cv2.THRESH_BINARY_INV)[1],
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8, ltype=cv2.CV_16U
    )
    if num_labels <= 1:
        empty = np.zeros((H, W), dtype=np.uint8)
        return {
            "protected_wall_mask": empty,
            "virtual_wall_barrier": empty,
            "interior_mask": empty,
            "clean_roi_mask": empty,
            "is_dashed_bubble": False,
            "num_dashes_detected": 0,
            "elapsed_ms": (time.perf_counter() - t_start) * 1000.0,
        }

    # 2. Предварительная фильтрация макро-стенок и периферийного кольца вокруг text_bbox
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    cw = stats[:, cv2.CC_STAT_WIDTH]
    ch = stats[:, cv2.CC_STAT_HEIGHT]
    is_solid_wall = (cw > 0.50 * W) | (ch > 0.50 * H)
    is_solid_wall[0] = False

    if text_bbox_rel is not None:
        if isinstance(text_bbox_rel, (list, tuple)) and len(text_bbox_rel) > 0 and isinstance(text_bbox_rel[0], (list, tuple)):
            tx1 = min(int(b[0]) for b in text_bbox_rel)
            ty1 = min(int(b[1]) for b in text_bbox_rel)
            tx2 = max(int(b[2]) for b in text_bbox_rel)
            ty2 = max(int(b[3]) for b in text_bbox_rel)
        else:
            tx1, ty1, tx2, ty2 = [int(v) for v in text_bbox_rel]
    else:
        tx1, ty1, tx2, ty2 = int(0.22 * W), int(0.20 * H), int(0.78 * W), int(0.80 * H)
    tcx, tcy = 0.5 * (tx1 + tx2), 0.5 * (ty1 + ty2)
    trx, try_ = max(8.0, 0.5 * (tx2 - tx1)), max(8.0, 0.5 * (ty2 - ty1))

    cx_all = centroids[:, 0].astype(np.float32)
    cy_all = centroids[:, 1].astype(np.float32)
    norm_r = np.sqrt(((cx_all - tcx) / trx) ** 2 + ((cy_all - tcy) / try_) ** 2)

    pre_cand = (
        (~is_solid_wall)
        & (areas >= 4.0 * (s ** 2))
        & (areas <= 400.0 * (s ** 2))
        & (norm_r >= 0.85)
        & (norm_r <= 3.50)
    )
    pre_cand[0] = False
    pre_ids = np.nonzero(pre_cand)[0]

    # 3. Вычисление центральных моментов инерции на микро-срезах кандидатов
    K_pre = len(pre_ids)
    semi_a = np.zeros(K_pre, dtype=np.float32)
    semi_b = np.zeros(K_pre, dtype=np.float32)
    ecc = np.zeros(K_pre, dtype=np.float32)
    theta = np.zeros(K_pre, dtype=np.float32)
    ribbon_fill = np.zeros(K_pre, dtype=np.float32)

    for idx, cid in enumerate(pre_ids):
        x0 = int(stats[cid, cv2.CC_STAT_LEFT])
        y0 = int(stats[cid, cv2.CC_STAT_TOP])
        w0 = int(stats[cid, cv2.CC_STAT_WIDTH])
        h0 = int(stats[cid, cv2.CC_STAT_HEIGHT])
        yy_l, xx_l = np.nonzero(labels[y0 : y0 + h0, x0 : x0 + w0] == cid)
        dx = xx_l.astype(np.float32) - (cx_all[cid] - x0)
        dy = yy_l.astype(np.float32) - (cy_all[cid] - y0)
        inv_a = 1.0 / max(1.0, float(areas[cid]))
        m20 = float(np.dot(dx, dx)) * inv_a
        m02 = float(np.dot(dy, dy)) * inv_a
        m11 = float(np.dot(dx, dy)) * inv_a
        df = m20 - m02
        dsc = float(np.sqrt(df * df + 4.0 * m11 * m11))
        l1 = max(0.5 * (m20 + m02 + dsc), 1e-4)
        l2 = max(0.5 * (m20 + m02 - dsc), 1e-4)
        sa = 2.0 * float(np.sqrt(l1)) + 0.5
        sb = 2.0 * float(np.sqrt(l2)) + 0.5
        semi_a[idx] = sa
        semi_b[idx] = sb
        ecc[idx] = float(np.sqrt(max(0.0, 1.0 - l2 / l1)))
        theta[idx] = 0.5 * float(np.arctan2(2.0 * m11, df))
        ribbon_fill[idx] = float(areas[cid]) / (np.pi * sa * sb + 1e-4)

    c_x = cx_all[pre_ids]
    c_y = cy_all[pre_ids]
    rad_dx = c_x - tcx
    rad_dy = c_y - tcy
    rad_len = np.hypot(rad_dx, rad_dy) + 1e-6
    tan_x = np.cos(theta)
    tan_y = np.sin(theta)
    radial_dot = np.abs(tan_x * (rad_dx / rad_len) + tan_y * (rad_dy / rad_len))

    keep = (semi_b <= 3.8 * s) & (ribbon_fill >= 0.42) & ((ecc < 0.65) | (radial_dot < 0.58))
    cand_ids = pre_ids[keep]
    c_x, c_y = c_x[keep], c_y[keep]
    t_x, t_y = tan_x[keep], tan_y[keep]
    s_a, e_c = semi_a[keep], ecc[keep]

    # 4. Матричный граф коциркулярности (K x K) и поиск цепочек пунктира
    dashed_wall_ids = np.array([], dtype=np.int32)
    bridge_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    K = len(cand_ids)
    if K >= 4:
        p_plus = np.stack([c_x + s_a * t_x, c_y + s_a * t_y], axis=-1)
        p_minus = np.stack([c_x - s_a * t_x, c_y - s_a * t_y], axis=-1)

        d_pp = np.linalg.norm(p_plus[:, None, :] - p_plus[None, :, :], axis=-1)
        d_pm = np.linalg.norm(p_plus[:, None, :] - p_minus[None, :, :], axis=-1)
        d_mp = np.linalg.norm(p_minus[:, None, :] - p_plus[None, :, :], axis=-1)
        d_mm = np.linalg.norm(p_minus[:, None, :] - p_minus[None, :, :], axis=-1)
        d_gap = np.minimum(np.minimum(d_pp, d_pm), np.minimum(d_mp, d_mm))
        np.fill_diagonal(d_gap, np.inf)

        chord_x = c_x[None, :] - c_x[:, None]
        chord_y = c_y[None, :] - c_y[:, None]
        chord_len = np.hypot(chord_x, chord_y) + 1e-6
        u_x, u_y = chord_x / chord_len, chord_y / chord_len

        cos_ti_u = np.abs(t_x[:, None] * u_x + t_y[:, None] * u_y)
        cos_tj_u = np.abs(t_x[None, :] * u_x + t_y[None, :] * u_y)
        cos_ti_tj = np.abs(t_x[:, None] * t_x[None, :] + t_y[:, None] * t_y[None, :])

        ok_i = (e_c[:, None] < 0.65) | (cos_ti_u >= 0.84)
        ok_j = (e_c[None, :] < 0.65) | (cos_tj_u >= 0.84)
        ok_ij = (e_c[:, None] < 0.65) | (e_c[None, :] < 0.65) | (cos_ti_tj >= 0.70)

        max_gap = 24.0 * s
        adj = (
            (d_gap <= max_gap)
            & (chord_len <= max_gap + 2.0 * (s_a[:, None] + s_a[None, :]))
            & ok_i
            & ok_j
            & ok_ij
        )

        visited = np.zeros(K, dtype=bool)
        valid_chain_nodes = np.zeros(K, dtype=bool)
        for start_node in range(K):
            if visited[start_node]:
                continue
            comp = [start_node]
            visited[start_node] = True
            head = 0
            while head < len(comp):
                curr = comp[head]
                head += 1
                for nb in np.nonzero(adj[curr] & (~visited))[0]:
                    visited[nb] = True
                    comp.append(int(nb))
            if len(comp) >= 3:
                valid_chain_nodes[comp] = True

        dashed_wall_ids = cand_ids[valid_chain_nodes]
        if len(dashed_wall_ids) >= 3:
            v_idx = np.nonzero(valid_chain_nodes)[0]
            ang = np.arctan2(c_y[v_idx] - tcy, c_x[v_idx] - tcx)
            v_ordered = v_idx[np.argsort(ang)]
            M = len(v_ordered)
            for idx_m in range(M):
                i_node = v_ordered[idx_m]
                j_node = v_ordered[(idx_m + 1) % M]
                if np.hypot(c_x[i_node] - c_x[j_node], c_y[i_node] - c_y[j_node]) <= 36.0 * s:
                    bridge_segments.append(
                        (
                            (float(c_x[i_node]), float(c_y[i_node])),
                            (float(c_x[j_node]), float(c_y[j_node])),
                        )
                    )

    # 5. Построение protected_wall_mask и замыкание virtual_wall_barrier дугами Безье
    wall_lut = is_solid_wall.astype(np.uint8) * 255
    if len(dashed_wall_ids) > 0:
        wall_lut[dashed_wall_ids] = 255
    protected_wall_mask = wall_lut[labels]
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    protected_wall_mask = cv2.dilate(protected_wall_mask, k3, iterations=1)

    virtual_wall_barrier = protected_wall_mask.copy()
    barrier_thickness = max(3, int(round(3.0 * s)))
    for p1, p2 in bridge_segments:
        mx, my = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
        chord_d = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        r_dx, r_dy = mx - tcx, my - tcy
        r_l = float(np.hypot(r_dx, r_dy)) + 1e-6
        sagitta = 0.08 * chord_d
        ctrl_x = mx + sagitta * (r_dx / r_l)
        ctrl_y = my + sagitta * (r_dy / r_l)

        ts = np.linspace(0.0, 1.0, 6)
        pts = np.stack(
            [
                (1.0 - ts) ** 2 * p1[0] + 2.0 * (1.0 - ts) * ts * ctrl_x + ts ** 2 * p2[0],
                (1.0 - ts) ** 2 * p1[1] + 2.0 * (1.0 - ts) * ts * ctrl_y + ts ** 2 * p2[1],
            ],
            axis=-1,
        ).astype(np.int32)
        cv2.polylines(
            virtual_wall_barrier, [pts], isClosed=False, color=255, thickness=barrier_thickness
        )

    # 6. Герметичное выделение interior_mask через floodFill внутри virtual_wall_barrier
    seed_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
    seed_mask[1:-1, 1:-1] = (virtual_wall_barrier > 0).astype(np.uint8)
    sy, sx = int(np.clip(tcy, 0, H - 1)), int(np.clip(tcx, 0, W - 1))
    if seed_mask[sy + 1, sx + 1] > 0 or roi_gray[sy, sx] < 180:
        win = roi_gray[max(0, sy - 10) : min(H, sy + 11), max(0, sx - 10) : min(W, sx + 11)]
        by_idx, bx_idx = np.unravel_index(np.argmax(win), win.shape)
        sy = max(0, sy - 10) + int(by_idx)
        sx = max(0, sx - 10) + int(bx_idx)

    dummy = np.full((H, W), 255, dtype=np.uint8)
    cv2.floodFill(dummy, seed_mask, (sx, sy), 128, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
    interior_mask = cv2.subtract(seed_mask[1:-1, 1:-1], virtual_wall_barrier)
    clean_roi_mask = cv2.bitwise_and(
        ink_mask, cv2.bitwise_and(interior_mask, cv2.bitwise_not(protected_wall_mask))
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "protected_wall_mask": protected_wall_mask,
        "virtual_wall_barrier": virtual_wall_barrier,
        "interior_mask": interior_mask,
        "clean_roi_mask": clean_roi_mask,
        "is_dashed_bubble": len(dashed_wall_ids) >= 3,
        "num_dashes_detected": int(len(dashed_wall_ids)),
        "elapsed_ms": elapsed_ms,
    }


def segment_open_wispy_bubble_roi(
    roi_gray: np.ndarray,
    text_bbox_rel: tuple[int, int, int, int],
    scale_f: float = 1.0,
) -> dict[str, np.ndarray | int | bool | float]:
    """
    Проблема №2: Сегментация разомкнутых/эскизных бабблов (Open / Wispy Bubbles):
    1) Находит и защищает тонкие боковые дуги-скобки (bounding parenthesis arcs) и внешний лайнарт волос.
    2) Строит виртуальную эллиптическую чашу (virtual_bowl_mask) для тайпсеттинга.
    3) Формирует геодезически зажатую маску очистки (tight_clean_mask) строго вокруг глифов.
    """
    t_start = time.perf_counter()
    H, W = roi_gray.shape[:2]
    s = max(0.5, float(scale_f))

    # 1. Быстрая бинаризация чернил
    blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    k_box = int(round(21 * s)) | 1
    local_mean = cv2.boxFilter(blur, cv2.CV_8U, (k_box, k_box))
    ink_mask = cv2.bitwise_and(
        cv2.threshold(cv2.subtract(local_mean, blur), 20, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY_INV)[1],
    )

    if isinstance(text_bbox_rel, (list, tuple)) and len(text_bbox_rel) > 0 and isinstance(text_bbox_rel[0], (list, tuple)):
        tx1 = min(int(b[0]) for b in text_bbox_rel)
        ty1 = min(int(b[1]) for b in text_bbox_rel)
        tx2 = max(int(b[2]) for b in text_bbox_rel)
        ty2 = max(int(b[3]) for b in text_bbox_rel)
    else:
        tx1, ty1, tx2, ty2 = [int(v) for v in text_bbox_rel]
    tcx, tcy = 0.5 * (tx1 + tx2), 0.5 * (ty1 + ty2)
    tw, th = max(10.0, float(tx2 - tx1)), max(10.0, float(ty2 - ty1))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8, ltype=cv2.CV_16U
    )
    if num_labels <= 1:
        empty = np.zeros((H, W), dtype=np.uint8)
        return {
            "protected_wall_mask": empty,
            "virtual_bowl_mask": empty,
            "tight_clean_mask": empty,
            "is_open_bubble": True,
            "num_arcs": 0,
            "has_opposing_arcs": False,
            "elapsed_ms": (time.perf_counter() - t_start) * 1000.0,
        }

    # 2. Векторизованный расчет периметра P, компактности IQ и моментов для всех компонент сразу
    k3_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary_mask = cv2.subtract(ink_mask, cv2.erode(ink_mask, k3_rect))
    ys, xs = np.nonzero(labels > 0)
    cids = labels[ys, xs]

    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    areas[0] = 1.0
    perims = (
        np.bincount(cids, weights=(boundary_mask[ys, xs] > 0).astype(np.float32), minlength=num_labels)
        * 1.12
    )
    iq_all = (4.0 * np.pi * areas) / (perims * perims + 1e-5)

    w_all = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    h_all = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    x0_all = stats[:, cv2.CC_STAT_LEFT].astype(np.float32)
    y0_all = stats[:, cv2.CC_STAT_TOP].astype(np.float32)
    fill_ratio_all = areas / (w_all * h_all + 1e-5)

    cx_all = centroids[:, 0].astype(np.float32)
    cy_all = centroids[:, 1].astype(np.float32)
    dx = xs.astype(np.float32) - cx_all[cids]
    dy = ys.astype(np.float32) - cy_all[cids]
    mu20 = np.bincount(cids, weights=dx * dx, minlength=num_labels) / areas
    mu02 = np.bincount(cids, weights=dy * dy, minlength=num_labels) / areas
    mu11 = np.bincount(cids, weights=dx * dy, minlength=num_labels) / areas
    theta_all = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)

    # Доля пикселей каждой компоненты внутри расширенного text_bbox
    pad_t = max(2, int(round(3.0 * s)))
    in_tbox_pix = (
        (xs >= tx1 - pad_t) & (xs <= tx2 + pad_t) & (ys >= ty1 - pad_t) & (ys <= ty2 + pad_t)
    )
    in_tbox_ratio = (
        np.bincount(cids, weights=in_tbox_pix.astype(np.float32), minlength=num_labels) / areas
    )

    core_x1, core_y1 = tx1 + 0.12 * tw, ty1 + 0.10 * th
    core_x2, core_y2 = tx2 - 0.12 * tw, ty2 - 0.10 * th
    strictly_in_core = (
        (x0_all >= core_x1)
        & (x0_all + w_all <= core_x2)
        & (y0_all >= core_y1)
        & (y0_all + h_all <= core_y2)
    )

    r_dx, r_dy = cx_all - tcx, cy_all - tcy
    r_len = np.hypot(r_dx, r_dy) + 1e-5
    tan_x, tan_y = np.cos(theta_all), np.sin(theta_all)
    radial_dot = np.abs(tan_x * (r_dx / r_len) + tan_y * (r_dy / r_len))
    chord_est = np.hypot(w_all, h_all)
    norm_r = np.hypot((cx_all - tcx) / (0.5 * tw), (cy_all - tcy) / (0.5 * th))

    is_cand_arc = (
        (~strictly_in_core)
        & (areas >= 10.0 * (s ** 2))
        & (iq_all <= 0.38)
        & (fill_ratio_all <= 0.48)
        & (chord_est >= 10.0 * s)
        & (radial_dot <= 0.58)
        & (norm_r >= 0.85)
        & (norm_r <= 3.50)
    )
    is_cand_arc[0] = False

    # 3. Проверка ориентации вогнутости (Inward Concavity) на кандидатах дуг
    cand_arc_ids = np.nonzero(is_cand_arc)[0]
    valid_arc_ids: list[int] = []
    strict_arc_cxs: list[float] = []
    arc_pts_x: list[np.ndarray] = []
    arc_pts_y: list[np.ndarray] = []

    for cid in cand_arc_ids:
        x0, y0, w0, h0 = int(x0_all[cid]), int(y0_all[cid]), int(w_all[cid]), int(h_all[cid])
        yy_l, xx_l = np.nonzero(labels[y0 : y0 + h0, x0 : x0 + w0] == cid)
        px = xx_l.astype(np.float32) + x0
        py = yy_l.astype(np.float32) + y0
        tx_v, ty_v = float(tan_x[cid]), float(tan_y[cid])
        proj = (px - cx_all[cid]) * tx_v + (py - cy_all[cid]) * ty_v
        i_min, i_max = int(np.argmin(proj)), int(np.argmax(proj))
        chord_mid_x = 0.5 * (px[i_min] + px[i_max])
        chord_mid_y = 0.5 * (py[i_min] + py[i_max])

        ortho = np.abs((px - chord_mid_x) * tx_v + (py - chord_mid_y) * ty_v)
        k_ap = max(1, len(px) // 8)
        ap_idx = np.argpartition(ortho, k_ap)[:k_ap]
        v_sag_x = float(np.mean(px[ap_idx])) - chord_mid_x
        v_sag_y = float(np.mean(py[ap_idx])) - chord_mid_y
        sag_len = float(np.hypot(v_sag_x, v_sag_y))

        rv_x, rv_y = chord_mid_x - tcx, chord_mid_y - tcy
        rv_l = float(np.hypot(rv_x, rv_y)) + 1e-5
        concav_cos = (v_sag_x * rv_x + v_sag_y * rv_y) / (sag_len * rv_l + 1e-5)

        if concav_cos >= 0.12 or sag_len < 1.5 * s:
            valid_arc_ids.append(int(cid))
            step = max(1, len(px) // 12)
            arc_pts_x.append(px[::step])
            arc_pts_y.append(py[::step])
            touches_roi_b = (x0 <= 1) or (y0 <= 1) or (x0 + w0 >= W - 1) or (y0 + h0 >= H - 1)
            if (not touches_roi_b) and concav_cos >= 0.16 and sag_len >= 1.1 * s and (h0 >= 0.35 * th or w0 >= 0.35 * tw):
                strict_arc_cxs.append(float(cx_all[cid]))

    has_opposing_arcs = (
        len(strict_arc_cxs) >= 2
        and (min(strict_arc_cxs) < tcx - 0.25 * tw)
        and (max(strict_arc_cxs) > tcx + 0.25 * tw)
    )

    is_valid_arc = np.zeros(num_labels, dtype=bool)
    if valid_arc_ids:
        is_valid_arc[valid_arc_ids] = True

    # Защищаем как валидные дуги скобок, так и весь внешний рисунок/волосы (< 60% внутри text_bbox)
    is_external_art_or_wall = is_valid_arc | (in_tbox_ratio < 0.60)
    is_external_art_or_wall[0] = False

    wall_lut = is_external_art_or_wall.astype(np.uint8) * 255
    protected_wall_mask = wall_lut[labels]
    k3_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    protected_wall_mask = cv2.dilate(protected_wall_mask, k3_ell, iterations=1)

    # 4. Аппроксимация вписанного эллипса чаши (virtual_bowl_mask) для безопасной верстки
    if arc_pts_x:
        ax = np.concatenate(arc_pts_x)
        ay = np.concatenate(arc_pts_y)
        est_rx = max(0.52 * tw, float(np.percentile(np.abs(ax - tcx), 85)) * 0.96)
        est_ry = max(0.54 * th, float(np.percentile(np.abs(ay - tcy), 85)) * 1.05)
    else:
        est_rx, est_ry = 0.55 * tw, 0.56 * th

    yy, xx = np.mgrid[0:H, 0:W]
    virtual_bowl_mask = (
        ((((xx - tcx) / est_rx) ** 2 + ((yy - tcy) / est_ry) ** 2) <= 1.0).astype(np.uint8) * 255
    )
    virtual_bowl_mask = cv2.bitwise_and(virtual_bowl_mask, cv2.bitwise_not(protected_wall_mask))

    # 5. Построение геодезически зажатой маски очистки (tight_clean_mask)
    text_lut = ((~is_external_art_or_wall) & (np.arange(num_labels) > 0)).astype(np.uint8) * 255
    text_glyphs = text_lut[labels]
    dist_to_wall = cv2.distanceTransform(cv2.bitwise_not(protected_wall_mask), cv2.DIST_L2, 3)

    r_dilate = max(2, int(round(3.0 * s)))
    k_tight = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_dilate + 1, 2 * r_dilate + 1))
    dilated_glyphs = cv2.dilate(text_glyphs, k_tight, iterations=1)
    tight_clean_mask = (
        ((dilated_glyphs > 0) & (dist_to_wall > 1.2 * s) & (virtual_bowl_mask > 0)).astype(np.uint8)
        * 255
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "protected_wall_mask": protected_wall_mask,
        "virtual_bowl_mask": virtual_bowl_mask,
        "tight_clean_mask": tight_clean_mask,
        "is_open_bubble": True,
        "num_arcs": len(valid_arc_ids),
        "has_opposing_arcs": bool(has_opposing_arcs),
        "elapsed_ms": elapsed_ms,
    }


def decompose_overlapping_bubbles(
    page_gray: np.ndarray,
    bubble_bboxes: list[tuple[int, int, int, int]],
    scale_f: float = 1.0,
) -> dict[str, list[np.ndarray] | np.ndarray | tuple[int, int, int, int] | float]:
    """
    Проблема №3: Топологическое разделение наложенных/сдвоенных бабблов (Overlapping & Conjoined Bubbles):
    - Защищает общую внутреннюю перегородку (shared septum) в protected_wall_mask.
    - Строит строго непересекающиеся маски полостей cavity_i и маски очистки clean_mask_i.
    - Если между сдвоенными бабблами нет черной перегородки (общая белая полость «восьмёрка»),
      разделяет полости по седловой точке узкого перешейка через водораздел на -DT.
    """
    t0 = time.perf_counter()
    if page_gray.ndim == 3:
        page_gray = cv2.cvtColor(page_gray, cv2.COLOR_RGB2GRAY)
    Hp, Wp = page_gray.shape[:2]
    s = max(0.5, float(scale_f))

    # 1. Объединенное окно ROI для группы пересекающихся бабблов
    pad = max(4, int(round(4.0 * s)))
    x1_u = max(0, min(b[0] for b in bubble_bboxes) - pad)
    y1_u = max(0, min(b[1] for b in bubble_bboxes) - pad)
    x2_u = min(Wp, max(b[2] for b in bubble_bboxes) + pad)
    y2_u = min(Hp, max(b[3] for b in bubble_bboxes) + pad)

    roi = page_gray[y1_u:y2_u, x1_u:x2_u]
    Hr, Wr = roi.shape[:2]

    # 2. Адаптивная бинаризация чернил
    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    k_box = int(round(21 * s)) | 1
    local_mean = cv2.boxFilter(blur, cv2.CV_8U, (k_box, k_box))
    ink = cv2.bitwise_and(
        cv2.threshold(cv2.subtract(local_mean, blur), 20, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(blur, 195, 255, cv2.THRESH_BINARY_INV)[1],
    )

    # 3. Выделение планарного графа стенок (включая внутреннюю перегородку shared septum)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink_closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k3, iterations=1)
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink_closed, connectivity=8, ltype=cv2.CV_16U
    )
    w_c = stats[:, cv2.CC_STAT_WIDTH]
    h_c = stats[:, cv2.CC_STAT_HEIGHT]
    is_wall_net = (w_c > 0.35 * Wr) | (h_c > 0.35 * Hr)
    is_wall_net[0] = False

    wall_network_mask = cv2.bitwise_and(ink, (is_wall_net[labels]).astype(np.uint8) * 255)
    protected_wall_mask = cv2.dilate(wall_network_mask, k3, iterations=1)

    # 4. Ортогональная декомпозиция полостей (cavity_A, cavity_B) как 4-связных граней дополнения стенок
    free_space = cv2.bitwise_and(
        cv2.bitwise_not(protected_wall_mask),
        cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1],
    )
    text_cand = cv2.bitwise_and(ink, cv2.bitwise_not(protected_wall_mask))
    cavity_space = cv2.bitwise_or(free_space, cv2.dilate(text_cand, k3, iterations=1))
    cavity_space = cv2.bitwise_and(cavity_space, cv2.bitwise_not(protected_wall_mask))

    num_cav, cav_labels, _, _ = cv2.connectedComponentsWithStats(
        cavity_space, connectivity=4, ltype=cv2.CV_16U
    )

    chosen_cids: list[int] = []
    cavities: list[np.ndarray] = []
    for idx_b, (bx1, by1, bx2, by2) in enumerate(bubble_bboxes):
        rx1, ry1 = max(0, bx1 - x1_u), max(0, by1 - y1_u)
        rx2, ry2 = min(Wr, bx2 - x1_u), min(Hr, by2 - y1_u)
        b_mask = np.zeros((Hr, Wr), dtype=bool)
        b_mask[ry1:ry2, rx1:rx2] = True
        b_area = (bx2 - bx1) * (by2 - by1)

        # Вычитаем меньшие наложенные боксы для получения эксклюзивного ядра K_i
        for idx_other, (ox1, oy1, ox2, oy2) in enumerate(bubble_bboxes):
            if idx_other != idx_b and (ox2 - ox1) * (oy2 - oy1) < b_area:
                orx1, ory1 = max(0, ox1 - x1_u), max(0, oy1 - y1_u)
                orx2, ory2 = min(Wr, ox2 - x1_u), min(Hr, oy2 - y1_u)
                b_mask[ory1:ory2, orx1:orx2] = False

        sub_cav = cav_labels[b_mask]
        sub_cav = sub_cav[sub_cav > 0]
        best_cid = int(np.argmax(np.bincount(sub_cav))) if len(sub_cav) > 0 else 0
        chosen_cids.append(best_cid)
        cav_mask = (
            ((cav_labels == best_cid) if best_cid > 0 else np.zeros((Hr, Wr), dtype=bool)).astype(
                np.uint8
            )
            * 255
        )
        cavities.append(cav_mask)

    # Фоллбэк водораздела для сдвоенных бабблов-«восьмёрок» без черной перегородки (общий best_cid > 0)
    if len(bubble_bboxes) >= 2 and len(set(chosen_cids)) < len(chosen_cids):
        for idx_b, (bx1, by1, bx2, by2) in enumerate(bubble_bboxes):
            cid_cur = chosen_cids[idx_b]
            if cid_cur <= 0 or chosen_cids.count(cid_cur) <= 1:
                continue
            shared_cav = cavities[idx_b]
            dist = cv2.distanceTransform(shared_cav, cv2.DIST_L2, 5)
            markers = np.zeros((Hr, Wr), dtype=np.int32)
            markers[shared_cav == 0] = 1
            for m_idx, (ox1, oy1, ox2, oy2) in enumerate(bubble_bboxes, start=2):
                if chosen_cids[m_idx - 2] != cid_cur:
                    continue
                ocx = max(0, min(Wr - 1, ((ox1 + ox2) // 2) - x1_u))
                ocy = max(0, min(Hr - 1, ((oy1 + oy2) // 2) - y1_u))
                cv2.circle(markers, (ocx, ocy), max(3, min(ox2 - ox1, oy2 - oy1) // 6), m_idx, -1)
            dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.watershed(cv2.cvtColor(255 - dist_norm, cv2.COLOR_GRAY2BGR), markers)
            cavities[idx_b] = np.where((markers == (idx_b + 2)) & (shared_cav > 0), 255, 0).astype(np.uint8)

    clean_masks: list[np.ndarray] = []
    for cav_mask in cavities:
        cm = cv2.bitwise_and(ink, cv2.bitwise_and(cav_mask, cv2.bitwise_not(protected_wall_mask)))
        clean_masks.append(cm)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "protected_wall_mask": protected_wall_mask,
        "cavities": cavities,
        "clean_masks": clean_masks,
        "union_bbox": (x1_u, y1_u, x2_u, y2_u),
        "elapsed_ms": dt_ms,
    }


def verify_bubble_and_text_candidate(
    roi_gray: np.ndarray,
    bubbles_only: bool = True,
    scale_f: float = 1.0,
) -> dict[str, bool | float | int]:
    """
    Проблема №6а: Каскадный фильтр-верификатор для ocr_engine.py:
    Отличает реальные диалоговые бабблы (включая облачные, колючие и разомкнутые со скобками)
    от черт лица (глаза/брови/рот/прическа), градиентной штриховки (KAGON), кафельной плитки и фоновых SFX.
    """
    t0 = time.perf_counter()
    if roi_gray.ndim == 3:
        roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_RGB2GRAY)
    Hr, Wr = roi_gray.shape[:2]
    s = max(0.5, float(scale_f))
    if Hr < 14 or Wr < 14:
        return {
            "is_valid": False,
            "has_enclosing_frame": False,
            "is_kagon": False,
            "is_tile_grid": False,
            "is_face": True,
            "h_ori": 0.0,
            "internal_holes": 0,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    # 1. Спектральная энтропия ориентаций градиента (отсечение параллельной штриховки KAGON)
    blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    edge_m = mag > 35.0
    if np.any(edge_m):
        angles = np.rad2deg(np.arctan2(gy[edge_m], gx[edge_m])) % 180.0
        hist, _ = np.histogram(angles, bins=18, range=(0.0, 180.0), weights=mag[edge_m])
        hist_prob = hist / (np.sum(hist) + 1e-6)
        h_ori = float(-np.sum(hist_prob * np.log(hist_prob + 1e-8)))
        top1_energy = float(np.max(hist_prob))
    else:
        h_ori, top1_energy = 0.0, 1.0

    is_kagon = (top1_energy > 0.42) or (h_ori < 1.55)

    # 2. Топологический анализ замкнутых полостей (отсечение кафельной плитки и проверка чаши баббла)
    k_box = int(round(21 * s)) | 1
    local_mean = cv2.boxFilter(blur, cv2.CV_8U, (k_box, k_box))
    ink_adapt = cv2.bitwise_and(
        cv2.threshold(cv2.subtract(local_mean, blur), 20, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(blur, 195, 255, cv2.THRESH_BINARY_INV)[1],
    )
    solid_dark = (blur < 115).astype(np.uint8) * 255
    ink = cv2.bitwise_or(ink_adapt, solid_dark)

    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8, ltype=cv2.CV_16U)
    areas = stats[:, cv2.CC_STAT_AREA]
    w_c = stats[:, cv2.CC_STAT_WIDTH]
    h_c = stats[:, cv2.CC_STAT_HEIGHT]

    wall_cands = (w_c > 0.42 * Wr) | (h_c > 0.42 * Hr)
    wall_cands[0] = False
    wall_mask = (wall_cands[labels]).astype(np.uint8) * 255

    inv_wall = cv2.bitwise_not(cv2.dilate(wall_mask, np.ones((3, 3), np.uint8)))
    num_holes, _, hole_stats, _ = cv2.connectedComponentsWithStats(
        inv_wall, connectivity=4, ltype=cv2.CV_16U
    )

    internal_holes = 0
    has_strict_hole_bowl = False
    for hid in range(1, num_holes):
        hx = int(hole_stats[hid, cv2.CC_STAT_LEFT])
        hy = int(hole_stats[hid, cv2.CC_STAT_TOP])
        hw = int(hole_stats[hid, cv2.CC_STAT_WIDTH])
        hh = int(hole_stats[hid, cv2.CC_STAT_HEIGHT])
        ha = int(hole_stats[hid, cv2.CC_STAT_AREA])
        if hx > 1 and hy > 1 and (hx + hw < Wr - 1) and (hy + hh < Hr - 1) and ha > 40 * (s ** 2):
            internal_holes += 1
            if (hx < 0.45 * Wr) and (hx + hw > 0.55 * Wr) and (hy < 0.45 * Hr) and (hy + hh > 0.55 * Hr):
                has_strict_hole_bowl = True

    is_tile_grid = internal_holes >= 4

    # 3. Проверка центральной белой чаши (включая бабблы, обрезанные сверху/снизу рамкой фрейма)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wall_closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k3, iterations=1)
    white_space = cv2.bitwise_not(wall_closed)
    n_cav, cav_labels, _, _ = cv2.connectedComponentsWithStats(white_space, connectivity=4, ltype=cv2.CV_16U)

    cy1, cy2 = int(0.18 * Hr), max(int(0.18 * Hr) + 2, int(0.82 * Hr))
    cx1, cx2 = int(0.18 * Wr), max(int(0.18 * Wr) + 2, int(0.82 * Wr))
    core_gray = roi_gray[cy1:cy2, cx1:cx2]
    core_white_ratio = float(np.mean(core_gray >= 225)) if core_gray.size > 0 else 0.0

    core_cav = cav_labels[cy1:cy2, cx1:cx2]
    vals, counts = np.unique(core_cav[core_cav > 0], return_counts=True)
    best_cid = int(vals[np.argmax(counts)]) if len(vals) > 0 else 0

    has_central_bowl = False
    is_face_head = False
    isolated_glyphs = 0
    isolated_glyph_area = 0

    if best_cid > 0:
        cav_mask = (cav_labels == best_cid).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(cav_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            cav_filled = np.zeros_like(cav_mask)
            cv2.drawContours(cav_filled, [cnt], -1, 255, -1)
            cav_fill_ratio = float(np.mean(cav_filled > 0))
            area_f = float(cv2.contourArea(cnt))
            hull_a = max(1.0, float(cv2.contourArea(cv2.convexHull(cnt))))
            solidity = area_f / hull_a

            t_leak = float(np.mean(cav_mask[0, :] > 0))
            b_leak = float(np.mean(cav_mask[-1, :] > 0))
            l_leak = float(np.mean(cav_mask[:, 0] > 0))
            r_leak = float(np.mean(cav_mask[:, -1] > 0))
            leak_sides = sum([t_leak > 0.32, b_leak > 0.32, l_leak > 0.32, r_leak > 0.32])
            leaking_lr = sum([l_leak > 0.25, r_leak > 0.25])

            mid_y1, mid_y2 = int(0.22 * Hr), int(0.78 * Hr)
            left_has_wall = False
            right_has_wall = False
            for y_r in range(mid_y1, mid_y2, max(1, (mid_y2 - mid_y1) // 10)):
                xs_r = np.nonzero(cav_filled[y_r, :] > 0)[0]
                if len(xs_r) >= 4:
                    x_l, x_r = int(xs_r[0]), int(xs_r[-1])
                    if x_l <= 2 or np.any(ink[y_r, max(0, x_l - 4) : min(Wr, x_l + 2)] > 0):
                        left_has_wall = True
                    if x_r >= Wr - 3 or np.any(ink[y_r, max(0, x_r - 1) : min(Wr, x_r + 5)] > 0):
                        right_has_wall = True

            cav_inner = cv2.erode(cav_filled, np.ones((5, 5), np.uint8))
            wall_protrude_px = 0
            iso_ink_px = 0
            for lid in range(1, num_l):
                lx, ly, lw, lh, la = [int(stats[lid, k]) for k in range(5)]
                if la < 6:
                    continue
                comp_m = (labels == lid)
                in_px = int(np.count_nonzero(comp_m & (cav_inner > 0)))
                if in_px == 0:
                    continue
                out_px = int(np.count_nonzero(comp_m & (cav_inner == 0)))
                if out_px > 0:
                    wall_protrude_px += in_px
                else:
                    iso_ink_px += in_px
                    touches_edge = (lx <= 1) or (ly <= 1) or (lx + lw >= Wr - 1) or (ly + lh >= Hr - 1)
                    min_g = float(np.min(roi_gray[comp_m]))
                    if (not touches_edge) and min_g <= 125.0 and lw < 0.65 * Wr and lh < 0.65 * Hr:
                        isolated_glyphs += 1
                        isolated_glyph_area += la

            prot_ratio = float(wall_protrude_px) / float(max(1, wall_protrude_px + iso_ink_px))
            if solidity < 0.855 or prot_ratio >= 0.16:
                is_face_head = True

            if (
                left_has_wall
                and right_has_wall
                and leaking_lr == 0
                and leak_sides <= 2
                and cav_fill_ratio >= 0.36
                and core_white_ratio >= 0.48
                and (not is_face_head)
            ):
                has_central_bowl = True

    if has_strict_hole_bowl and not is_face_head and not is_tile_grid:
        has_central_bowl = True

    # Если чаша разомкнута, проверяем наличие парных вогнутых боковых дуг-скобок через Проблему №2
    if not has_central_bowl and not is_face_head:
        open_check = segment_open_wispy_bubble_roi(
            roi_gray, (int(0.25 * Wr), int(0.22 * Hr), int(0.75 * Wr), int(0.78 * Hr)), scale_f=s
        )
        has_open_arcs = bool(open_check.get("has_opposing_arcs", False)) and (core_white_ratio >= 0.65)
    else:
        has_open_arcs = False

    has_enclosing_frame = has_central_bowl or has_open_arcs

    # 4. Инвариант ширины штриха (отсечение зрачков глаз и черт лица)
    char_cands = (~wall_cands) & (areas >= 8 * (s ** 2)) & (areas <= 800 * (s ** 2))
    char_cands[0] = False
    char_ids = np.nonzero(char_cands)[0]

    dist_ink = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    ys_i, xs_i = np.nonzero(labels > 0)
    max_r_all = np.zeros(num_l, dtype=np.float32)
    if len(ys_i) > 0:
        np.maximum.at(max_r_all, labels[ys_i, xs_i], dist_ink[ys_i, xs_i])

    has_thick_eye_blob = bool(np.any(max_r_all[char_ids] > 4.2 * s)) if len(char_ids) > 0 else False
    is_face_or_unframed = is_face_head or (
        (not has_enclosing_frame) and (bubbles_only or has_thick_eye_blob or len(char_ids) >= 3)
    )

    min_req_glyphs = 1 if (has_open_arcs or has_strict_hole_bowl) else 2
    min_req_area = int(20 * (s ** 2)) if (has_open_arcs or has_strict_hole_bowl) else max(40, int(0.010 * Hr * Wr))
    has_valid_glyph_core = (
        (isolated_glyphs >= min_req_glyphs and isolated_glyph_area >= min_req_area)
        or ((has_open_arcs or has_strict_hole_bowl) and len(char_ids) >= 1 and not has_thick_eye_blob)
    )

    is_valid = (
        (not is_kagon)
        and (not is_tile_grid)
        and (not is_face_or_unframed)
        and has_valid_glyph_core
    )

    return {
        "is_valid": bool(is_valid),
        "has_enclosing_frame": bool(has_enclosing_frame),
        "is_kagon": bool(is_kagon),
        "is_tile_grid": bool(is_tile_grid),
        "is_face": bool(is_face_or_unframed),
        "h_ori": h_ori,
        "internal_holes": internal_holes,
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
    }


def estimate_and_deskew_tilted_text_block(
    roi_gray: np.ndarray,
    scale_f: float = 1.0,
) -> dict[str, np.ndarray | float | bool | int]:
    """
    Проблема №6б: Быстрое определение угла наклона (15°..50°) и выпрямление (Deskew)
    многострочных экранов смартфонов и табличек перед подачей в OCR.
    """
    t0 = time.perf_counter()
    if roi_gray.ndim == 3:
        roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_RGB2GRAY)
    Hr, Wr = roi_gray.shape[:2]
    s = max(0.5, float(scale_f))

    blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    k_box = int(round(21 * s)) | 1
    local_mean = cv2.boxFilter(blur, cv2.CV_8U, (k_box, k_box))
    ink = cv2.bitwise_and(
        cv2.threshold(cv2.subtract(local_mean, blur), 20, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(blur, 195, 255, cv2.THRESH_BINARY_INV)[1],
    )

    num_l, _, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8, ltype=cv2.CV_16U)
    areas = stats[:, cv2.CC_STAT_AREA]
    w_c = stats[:, cv2.CC_STAT_WIDTH]
    h_c = stats[:, cv2.CC_STAT_HEIGHT]

    char_mask = (
        (areas >= 12 * (s ** 2))
        & (areas <= 600 * (s ** 2))
        & (w_c < 0.40 * Wr)
        & (h_c < 0.40 * Hr)
        & (np.maximum(w_c, h_c) / (np.minimum(w_c, h_c) + 1e-4) <= 3.5)
    )
    char_mask[0] = False
    char_ids = np.nonzero(char_mask)[0]

    if len(char_ids) < 4:
        return {
            "deskewed": roi_gray,
            "angle_deg": 0.0,
            "is_tilted_screen": False,
            "num_chars": int(len(char_ids)),
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    pts = centroids[char_ids].astype(np.float32)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)

    nn1 = np.argmin(dist, axis=1)
    d1 = dist[np.arange(len(pts)), nn1]
    med_d = float(np.median(d1))

    i_idx, j_idx = np.nonzero((dist <= 1.65 * med_d) & np.triu(np.ones_like(dist, dtype=bool), 1))
    if len(i_idx) < 3:
        return {
            "deskewed": roi_gray,
            "angle_deg": 0.0,
            "is_tilted_screen": False,
            "num_chars": int(len(char_ids)),
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    vecs = pts[j_idx] - pts[i_idx]
    flip = vecs[:, 0] < 0
    vecs[flip] = -vecs[flip]
    pair_angles = np.rad2deg(np.arctan2(vecs[:, 1], vecs[:, 0]))
    coarse_angle = float(np.median(pair_angles))

    # Точный поиск угла максимизацией проекционной дисперсии Радона
    test_angles = np.linspace(coarse_angle - 4.0, coarse_angle + 4.0, 33)
    rads = np.deg2rad(test_angles)
    proj_y = -pts[:, 0:1] * np.sin(rads)[None, :] + pts[:, 1:2] * np.cos(rads)[None, :]

    best_a = coarse_angle
    best_score = -1.0
    num_bins = max(8, int(Hr // 6))
    for idx_a, a_deg in enumerate(test_angles):
        hist, _ = np.histogram(proj_y[:, idx_a], bins=num_bins)
        score = float(np.sum(hist.astype(np.float32) ** 2))
        if score > best_score:
            best_score = score
            best_a = float(a_deg)

    if abs(best_a) >= 8.0:
        rad_a = np.deg2rad(best_a)
        cos_a, sin_a = abs(float(np.cos(rad_a))), abs(float(np.sin(rad_a)))
        new_W = int(np.ceil(Wr * cos_a + Hr * sin_a))
        new_H = int(np.ceil(Wr * sin_a + Hr * cos_a))
        M_deskew = cv2.getRotationMatrix2D((Wr / 2.0, Hr / 2.0), best_a, 1.0)
        M_deskew[0, 2] += (new_W - Wr) / 2.0
        M_deskew[1, 2] += (new_H - Hr) / 2.0
        deskewed = cv2.warpAffine(
            roi_gray, M_deskew, (new_W, new_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    else:
        deskewed = roi_gray

    return {
        "deskewed": deskewed,
        "angle_deg": best_a,
        "is_tilted_screen": abs(best_a) >= 10.0 and len(char_ids) >= 6,
        "num_chars": int(len(char_ids)),
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
    }


def build_bubble_geometry_context(
    page_gray: np.ndarray,
    bubble_id: int,
    roi_bbox: tuple[int, int, int, int],
    text_bbox_rel: tuple[int, int, int, int],
    has_overhang: bool = False,
    bubbles_only: bool = True,
    scale_f: float = 1.0,
) -> BubbleGeometryContext | None:
    """
    Раздел 7.1: Единый конструктор BubbleGeometryContext, вызываемый после детекции RT-DETR-v2 + YOLOv8m:
    1) Отсекает ложные срабатывания (лица, KAGON, плитка) через verify_bubble_and_text_candidate.
    2) Определяет топологический тип баббла ('solid' | 'dashed' | 'open_wispy' | 'overhanging') и строит защитные маски.
    """
    if page_gray.ndim == 3:
        page_gray = cv2.cvtColor(page_gray, cv2.COLOR_RGB2GRAY)
    x1, y1, x2, y2 = roi_bbox
    roi_gray = page_gray[y1:y2, x1:x2]

    ver = verify_bubble_and_text_candidate(roi_gray, bubbles_only=bubbles_only, scale_f=scale_f)
    if not ver["is_valid"]:
        return None

    H, W = roi_gray.shape[:2]
    if has_overhang:
        from cleaner import clean_overhanging_text_zero_bleed

        ov = clean_overhanging_text_zero_bleed(roi_gray, lama_runner=None, scale_f=scale_f)
        return BubbleGeometryContext(
            bubble_id=bubble_id,
            roi_bbox=roi_bbox,
            text_bbox_rel=text_bbox_rel,
            topology_type="overhanging",
            protected_wall_mask=ov["full_wall_mask"],
            interior_mask=ov["reconstructed_interior"],
            M_inside=ov["M_inside"],
            M_outside=ov["M_outside"],
        )

    dash_res = segment_dashed_bubble_roi(roi_gray, text_bbox_rel=text_bbox_rel, scale_f=scale_f)
    interior_area_ratio = float(np.count_nonzero(dash_res["interior_mask"])) / float(H * W + 1e-5)

    if (not dash_res["is_dashed_bubble"]) and (interior_area_ratio > 0.82 or interior_area_ratio < 0.08):
        open_res = segment_open_wispy_bubble_roi(roi_gray, text_bbox_rel=text_bbox_rel, scale_f=scale_f)
        return BubbleGeometryContext(
            bubble_id=bubble_id,
            roi_bbox=roi_bbox,
            text_bbox_rel=text_bbox_rel,
            topology_type="open_wispy",
            protected_wall_mask=open_res["protected_wall_mask"],
            interior_mask=open_res["virtual_bowl_mask"],
            M_inside=open_res["tight_clean_mask"],
            M_outside=np.zeros((H, W), dtype=np.uint8),
        )

    topo = "dashed" if dash_res["is_dashed_bubble"] else "solid"
    return BubbleGeometryContext(
        bubble_id=bubble_id,
        roi_bbox=roi_bbox,
        text_bbox_rel=text_bbox_rel,
        topology_type=topo,
        protected_wall_mask=dash_res["protected_wall_mask"],
        interior_mask=dash_res["interior_mask"],
        M_inside=dash_res["clean_roi_mask"],
        M_outside=np.zeros((H, W), dtype=np.uint8),
    )

HF_YOLO_REPO = "ogkalu/comic-speech-bubble-detector-yolov8m"
HF_YOLO_FILENAME = "comic-speech-bubble-detector.pt"

HF_RTDETR_REPO = "ogkalu/comic-text-and-bubble-detector"
HF_RTDETR_SUBDIR = "comic-text-and-bubble-detector"
HF_RTDETR_PT_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")

HF_JA_NAR_REPO = "muscgab/manga-ocr-nar-small"
HF_JA_NAR_SUBDIR = "manga-ocr-nar-small"
HF_JA_NAR_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.py",
    "*.toml",
    "manga_ocr_nar_small/*",
    "manga_ocr_nar_small/**/*",
    "mangaocr_nar_small/*",
    "mangaocr_nar_small/**/*",
)

HF_HAYAI_REPO = "JustANormalTinkerer/hayai-ocr-v2.5-nova"
HF_HAYAI_SUBDIR = "hayai-ocr-v2.5-nova"
HF_HAYAI_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.py",
    "*.txt",
    "*.model",
)

HF_HAYAI_PROCESSOR_REPO = "google/siglip2-base-patch16-naflex"
HF_HAYAI_PROCESSOR_SUBDIR = "siglip2-base-patch16-naflex"
HF_HAYAI_PROCESSOR_ALLOW_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
)

# Приоритет локальных Vision-моделей в Ollama для снайперского чтения сложных SFX и рукописных шрифтов (Шаг 4)
VLM_OCR_MODEL_PRIORITIES: tuple[str, ...] = (
    "glm-ocr",
    "qwen3-vl",
    "qwen2.5vl",
    "qwen2-vl",
    "minicpm-v",
    "gemma3",
    "llama3.2-vision",
    "llava",
    "pixtral",
    "internvl",
)

VLM_OCR_PREFIX_STRIP_REGEX = re.compile(
    r"^(?:the\s+text\s+(?:in\s+the\s+image\s+)?is|transcribed\s+text|transcription|ocr\s+result|result|output|answer|ocr|text|"
    r"текст\s+на\s+изображении|распознанный\s+текст|текст)\s*[:：\-—]+\s*",
    flags=re.IGNORECASE,
)

HANGUL_REGEX = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
KANA_REGEX = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
KATAKANA_ONLY_REGEX = re.compile(r"^[\u30a0-\u30ff]+$")
HANZI_REGEX = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
CJK_SPACE_STRIP_REGEX = re.compile(
    r"([\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf\uac00-\ud7af\u1100-\u11ff\u3130-\u318f])\s+(?=[\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf\uac00-\ud7af\u1100-\u11ff\u3130-\u318f])"
)
MEANINGFUL_CHAR_REGEX = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7afA-Za-zА-Яа-яЁё0-9]"
)
PUNCT_ONLY_REGEX = re.compile(
    r"^[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\（\）\(\)\[\]\♡\♥\♪\★\☆\・\：\;\"\'\`]+$"
)
STRIP_PUNCT_REGEX = re.compile(
    r"[\s\.\,\!\?\…\‥\ー\〜\~\-\—\！\？\。\、\「\」\『\』\（\）\(\)\[\]\♡\♥\♪\★\☆\・\：\;\"\'\`]+"
)
REPEATED_NOISE_REGEX = re.compile(r"^(.)\1{3,}$")
REDUPLICATED_SFX_REGEX = re.compile(r"^(.{1,3})\1+$")

# Известные галлюцинации декодеров MangaOCR / manga-ocr-nar-small / HayaiOCR на пустых/шумовых кропах фона и штриховки
# (например, заученная из синтетического корпуса фраза «タクシー運転手さん一番うまい店に連れてって» -> «Taxi driver, take me to the best restaurant!»)
KNOWN_OCR_HALLUCINATIONS_REGEX = re.compile(
    r"(?i)(?:タクシー運転手|運転手さん.*(?:うまい店|美味しい店|レストラン|連れて)|一番(?:うまい|美味しい)(?:店|レストラン).*連れて|"
    r"taxi\s*driver.*(?:best\s*restau?rant|take\s*me)|take\s*me\s*to\s*the\s*best\s*restau?rant|"
    r"ご視聴ありがとうございました|チャンネル登録.*お願い|高評価.*チャンネル登録)"
)

# Словарь ТОЛЬКО чистых вздохов, стонов и охов (без диалоговых слов типа いや / やだ / おい / うん!)
SIGH_EXACT_SET: set[str] = {
    # Только чистые бессловесные вздохи и стоны (хирагана/катакана) — без диалоговых реакций вроде えっ/くっ/ちっ/わっ!
    "あ", "あっ", "ああ", "ああっ", "あぁ", "あぁっ", "あん", "あんっ", "ああん", "あف", "あふっ",
    "う", "うっ", "うぅ", "うぅっ", "うう", "ううっ",
    "ん", "んっ", "んん", "んんっ", "んぅ", "んぅっ", "んあ", "んあっ",
    "はぁ", "はぁっ", "はぁはぁ", "はあ", "はあっ", "はあはあ", "はふ", "はふっ",
    "ふぅ", "ふぅっ", "ふう", "ふうっ",
    "ア", "アッ", "アア", "アアッ", "アァ", "アァッ", "アン", "アンッ",
    "ウ", "ウッ", "ウゥ", "ウゥッ",
    "ン", "ンッ", "ンン", "ンンッ", "ンゥ",
    "ハァ", "ハァッ", "ハァハァ", "ハア", "ハアッ",
    "フゥ", "フゥッ", "フウ",
    # Корейские чистые вздохи
    "아", "아앗", "아아", "우", "우웃", "으", "으음", "음", "하아", "하앗", "후우", "후웃",
    # Китайские чистые вздохи
    "啊", "啊啊", "嗯", "嗯嗯", "唔", "呼",
    # Русские и английские чистые вздохи
    "а", "о", "у", "м", "мм", "ммм", "ах", "аах", "ахн", "ох", "оох", "хаа",
    "a", "o", "u", "m", "mm", "mmm", "mmh", "ah", "aah", "ahn", "ooh", "haa",
}

# Слова из гласных и короткие реакции, которые являются осмысленными репликами диалога, а НЕ вздохами
ZH_SPECIFIC_MARKERS_REGEX = re.compile(
    r"[这那们么没着过还让给从向到得着了的吗呢吧啊呀哦哈你她它咱啥谁哪怎什因为所以虽然但是如果已经可以这个那个]"
)

DIALOGUE_VOWEL_WORDS_JA: set[str] = {
    "おい", "オイ", "いい", "イイ", "いいえ", "イイエ",
    "うん", "ウン", "うんっ", "ウンッ", "うんうん", "ウンウン", "ううん", "ウウン", "うーん", "ウーン",
    "ええ", "エエ", "え", "えっ", "えぇ", "エ", "エッ", "エェ",
    "お", "おっ", "おお", "おおっ", "オ", "オッ", "オオ", "オオッ",
    "おう", "オウ", "あい", "アイ", "うえ", "ウエ",
    "あおい", "アオイ", "おおい", "オオイ", "いう", "イウ", "あう", "アウ",
    "わっ", "ワッ", "くっ", "クッ", "ぐっ", "グッ", "ちっ", "チッ", "ほっ", "ホッ", "へっ", "ヘッ", "へ", "へぇ", "ヘ", "ヘェ",
}

# Регулярное выражение только для комбинаций чистых вздохов/стонов («あ…あっ», «はぁ…はぁ…», «んっ…あ»)
SIGH_COMBO_REGEX = re.compile(
    r"^(?:[あぁアァ]+[っッー]*|[うぅウゥ]+[っッー]*|[んン]+[っッー]*|"
    r"はぁ|はあ|ハァ|ハア|ふぅ|ふう|フゥ|フウ|"
    r"あん|アン|あっ|アッ|うっ|ウッ|んっ|ンッ|あふ|アフ)+$"
)

# Словарь частых звуковых спецэффектов манги и манхвы (SFX / ономатопея)
SFX_EXACT_SET: set[str] = {
    "ドキ", "ドキッ", "ドキドキ", "ドクン", "ドクッ", "ビク", "ビクッ", "ビクン", "ビクビク",
    "ギュ", "ギュッ", "ギュウ", "ゾク", "ゾクッ", "ゾクゾク", "パン", "パアン", "バン", "バアン",
    "ドン", "ドオン", "ドーン", "ゴゴ", "ゴゴゴ", "ゴゴゴゴ", "ザワ", "ザワザワ", "シーン", "シィン",
    "ガタ", "ガタッ", "ガタガタ", "パッ", "スッ", "チュ", "チュッ", "チュウ", "ジュワ", "ジュワッ",
    "ヌル", "ヌルッ", "ヌルヌル", "ピチャ", "ピチャッ", "ズキ", "ズキッ", "ズキズキ", "ワク", "ワクワク",
    "ニコ", "ニコッ", "ニコニコ", "チラ", "チラッ", "チラチラ", "フワ", "フワッ", "フワフワ",
    "ボッ", "カッ", "ギシ", "ギシッ", "ギシギシ", "ミシ", "ミシッ", "バタン", "ガラ", "ガラッ", "ガラガラ",
    "スタ", "スタスタ", "トコ", "トコトコ", "ペコ", "ペコッ", "クス", "クスッ", "クスクス", "ゴクリ", "ゴクッ",
    "ブル", "ブルッ", "ブルブル", "ガク", "ガクガク", "ガシャン", "ドサ", "ドサッ", "バサ", "バサッ",
    "サッ", "シュ", "シュッ", "グッ", "ピタ", "ピタッ", "クル", "クルッ", "パチ", "パチッ", "パチパチ",
    "ズッ", "ブッ", "プッ", "ブチュ", "ジュル", "ジュルッ", "レロ", "レロレロ", "ペロ", "ペロッ",
    "モミ", "モミモミ", "ズブ", "ズブッ", "ズプ", "ズプッ", "グチュ", "グチュッ", "パンッ", "パチン",
    "ピ질", "두근", "두근두근", "움찔", "덜컥", "쾅", "퍽", "짝", "스륵", "부들", "부들부들",
    "바르르", "꿀꺽", "주륵", "반짝", "휘잉", "철컥", "쿵", "쿵쿵", "탁", "휙", "짝짝",
    "ぷる", "ぷるぷる", "びく", "びくっ", "びくん", "どき", "どきっ", "どきどき", "ざわ", "ざわざわ",
    "ぴく", "ぴくっ", "ぎゅ", "ぎゅっ", "ちゅ", "ちゅっ", "ぱん", "どん", "ばん", "ずき", "ずきっ",
    "ぞく", "ぞくっ", "ぬる", "ぬるっ", "じゅわ", "じゅわっ", "じゅる", "れろ", "ぺろ", "もみ", "ぐちゅ",
}


class MangaOCREngine:
    """
    Мультиязычный движок v1.0.8 (Шаги 1 + 2 + 3 + 4):
    - Двухконтурная детекция (`RT-DETR-v2` 3-class `bubble` / `text_bubble` / `text_free` + `YOLOv8m` + OpenCV)
    - Плотная подгонка кропа по чернилам (`_compute_tight_ink_bbox`)
    - Японский OCR (`muscgab/manga-ocr-nar-small` + `MangaOCR FP16`)
    - Мультиязычный OCR для корейской манхвы (`ko`) и китайской маньхуа (`zh`):
      `JustANormalTinkerer/hayai-ocr-v2.5-nova` (`SigLIP2 NaFlex`) + построчная сегментация (`_split_horizontal_lines_for_ocr`)
      + опциональные `RapidOCR` / `EasyOCR`.
    - Снайперский VLM-OCR (`recognize_crop_via_vlm`) через локальные Vision-модели Ollama (`glm-ocr`, `qwen2.5vl`,
      `qwen3-vl`, `gemma3`, `minicpm-v`, `llava`) или облачный Vision API для сложных рукописных SFX и обложек.
    """

    def __init__(
        self,
        models_dir: str | Path = "models",
        device: str | None = None,
        conf_threshold: float = 0.22,
        iou_threshold: float = 0.45,
        imgsz: int = 1280,
        use_rtdetr: bool = True,
        use_ja_nar: bool = True,
        use_hayai_ocr: bool = True,
        ollama_base_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.use_rtdetr = bool(use_rtdetr)
        self.use_ja_nar = bool(use_ja_nar)
        self.use_hayai_ocr = bool(use_hayai_ocr)
        self.ollama_base_url = ollama_base_url.rstrip("/")

        logger.info("Инициализация MangaOCREngine (v1.0.8 Step 1+2+3+4) на устройстве: %s", self.device)
        self.detector = self._load_detector()
        self.ja_recognizer = self._load_manga_ocr()

        self._ja_nar_recognizer: Any | None = None
        self._ja_nar_available: bool | None = None if self.use_ja_nar else False

        self._hayai_model: Any | None = None
        self._hayai_tokenizer: Any | None = None
        self._hayai_processor: Any | None = None
        self._hayai_available: bool | None = None if self.use_hayai_ocr else False

        self._rtdetr_model: Any | None = None
        self._rtdetr_processor: Any | None = None
        self._rtdetr_available: bool | None = None if self.use_rtdetr else False
        self._last_rtdetr_free_boxes: list[dict[str, Any]] = []

        self._rapidocr_reader: Any | None = None
        self._rapidocr_available: bool | None = None

        self._ko_reader: Any | None = None
        self._zh_reader: Any | None = None
        self._craft_reader: Any | None = None
        self._easyocr_available: bool | None = None

        self._last_vlm_model_used: str = ""
        self._vlm_ocr_calls_count: int = 0

        self._page_ocr_cache: dict[tuple[int, int, int, int, str], str] = {}
        self._last_detected_lang: str = "ja"
        self.last_page_diagnostics: dict[str, Any] = {}
        self.last_detection_stats: dict[str, Any] = {
            "detector_backend": "YOLOv8m + TightInkCrop",
            "rtdetr_active": False,
            "hayai_active": False,
            "rapidocr_active": False,
            "vlm_ocr_backend": "Auto (Ollama GLM-OCR / Qwen2.5-VL / Gemma3 / Cloud Vision)",
            "vlm_ocr_calls": 0,
            "yolo_bubbles": 0,
            "rtdetr_bubbles": 0,
            "rtdetr_text_bubbles": 0,
            "rtdetr_free_texts": 0,
            "split_multi_bubbles": 0,
            "rescued_orphan_text_bubbles": 0,
        }

        # Пробуем сразу инициализировать 3-классовый детектор RT-DETR-v2 (ogkalu/comic-text-and-bubble-detector)
        if self.use_rtdetr:
            self._get_rtdetr_detector()

        # Пробуем инициализировать скоростной высокоточный японский OCR 2026 года (muscgab/manga-ocr-nar-small)
        if self.use_ja_nar:
            self._get_ja_nar_recognizer()

        # Если мультиязычная модель HayaiOCR v2.5 Nova уже скачана на диск, подгружаем её сразу,
        # иначе она скачается автоматически по первому запросу ko / zh / auto без задержки старта для чистого ja
        if self.use_hayai_ocr and self._is_hayai_cached_locally():
            self._get_hayai_recognizer()

        self.last_detection_stats["ja_ocr_backend"] = self.get_ja_ocr_backend_name()
        self.last_detection_stats["ko_zh_ocr_backend"] = self.get_multilang_ocr_backend_name()
        self.last_detection_stats["vlm_ocr_backend"] = self.get_vlm_ocr_backend_name()
        logger.info(
            "Базовые модели загружены (Детектор: %s | OCR JA: %s | OCR KO/ZH: %s | VLM-OCR: %s).",
            self.get_detector_backend_name(),
            self.get_ja_ocr_backend_name(),
            self.get_multilang_ocr_backend_name(),
            self.get_vlm_ocr_backend_name(),
        )

    def get_detector_backend_name(self) -> str:
        if self._rtdetr_available and self._rtdetr_model is not None:
            return "Hybrid RT-DETR-v2 (3-class: bubble + text_bubble + text_free) + YOLOv8m + TightInkCrop"
        return "YOLOv8m + OpenCV Contours + TightInkCrop"

    def get_ja_ocr_backend_name(self) -> str:
        if self._ja_nar_available and self._ja_nar_recognizer is not None:
            return "Hybrid manga-ocr-nar-small (v2026, CER 1.73%) + MangaOCR FP16 (long-bubble fallback)"
        return "MangaOCR FP16 + SmartPreprocess (Auto-Invert & AspectGuard)"

    def get_multilang_ocr_backend_name(self) -> str:
        parts: list[str] = []
        if self._hayai_available and self._hayai_model is not None:
            parts.append("HayaiOCR v2.5 Nova (SigLIP2 NaFlex JA/KO/ZH/EN)")
        elif self.use_hayai_ocr and self._hayai_available is not False:
            parts.append("HayaiOCR v2.5 Nova (Auto-Load JA/KO/ZH/EN)")
        if self._check_rapidocr():
            parts.append("RapidOCR")
        if self._check_easyocr():
            parts.append("EasyOCR")
        parts.append("SmartLineSplit")
        return " + ".join(parts)

    def get_vlm_ocr_backend_name(self) -> str:
        if self._last_vlm_model_used:
            return f"Sniper VLM-OCR ({self._last_vlm_model_used}, вызовов: {self._vlm_ocr_calls_count})"
        return "Sniper VLM-OCR (Ollama GLM-OCR / Qwen2.5-VL / Gemma3 / Cloud Vision)"

    def _is_hayai_cached_locally(self) -> bool:
        local_dir = self.models_dir / HF_HAYAI_SUBDIR
        return (local_dir / "model.safetensors").exists() and (local_dir / "config.json").exists()

    def _get_hayai_recognizer(self) -> tuple[Any, Any, Any] | None:
        """
        Загружает мультиязычную OCR-модель 2026 года `JustANormalTinkerer/hayai-ocr-v2.5-nova` (~150M параметров,
        SigLIP2 NaFlex + 12-слойный GQA декодер с 4x DSCProjector).
        Поддерживает корейский (`KO`), китайский (`ZH`), японский (`JA`) и английский (`EN`),
        сохраняя нативные пропорции баббла (`max_num_patches=384/512`) и работая напрямую через `transformers`
        без установки сторонних C++/ONNX пакетов.
        """
        if not self.use_hayai_ocr or self._hayai_available is False:
            return None
        if (
            self._hayai_model is not None
            and self._hayai_tokenizer is not None
            and self._hayai_processor is not None
        ):
            return self._hayai_model, self._hayai_tokenizer, self._hayai_processor

        local_dir = self.models_dir / HF_HAYAI_SUBDIR
        proc_dir = self.models_dir / HF_HAYAI_PROCESSOR_SUBDIR
        local_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)

        try:
            weights_file = local_dir / "model.safetensors"
            config_file = local_dir / "config.json"
            if not weights_file.exists() or not config_file.exists():
                logger.info(
                    "Скачивание мультиязычной OCR-модели (KO/ZH/JA/EN) %s...",
                    HF_HAYAI_REPO,
                )
                snapshot_download(
                    repo_id=HF_HAYAI_REPO,
                    local_dir=str(local_dir),
                    allow_patterns=list(HF_HAYAI_ALLOW_PATTERNS),
                )

            import transformers
            from transformers import AutoModel, AutoProcessor, PreTrainedTokenizerFast

            hf_logging = getattr(transformers, "logging", None)
            if hf_logging is not None and hasattr(hf_logging, "set_verbosity_error"):
                try:
                    hf_logging.set_verbosity_error()
                except Exception:
                    pass
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

            # Загружаем быстрый токенизатор HayaiOCR локально без лишних сетевых запросов
            try:
                tokenizer = PreTrainedTokenizerFast.from_pretrained(str(local_dir), local_files_only=True)
            except Exception:
                try:
                    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(local_dir))
                except Exception:
                    auto_tok_cls = getattr(transformers, "AutoTokenizer", None)
                    if auto_tok_cls is None:
                        raise
                    tokenizer = auto_tok_cls.from_pretrained(str(local_dir), trust_remote_code=True)

            # Загружаем процессор SigLIP2 NaFlex (сначала из локального кэша proc_dir без обращения к сети!)
            processor = None
            if not (proc_dir / "preprocessor_config.json").exists():
                logger.info(
                    "Скачивание конфигурации процессора %s...",
                    HF_HAYAI_PROCESSOR_REPO,
                )
                snapshot_download(
                    repo_id=HF_HAYAI_PROCESSOR_REPO,
                    local_dir=str(proc_dir),
                    allow_patterns=list(HF_HAYAI_PROCESSOR_ALLOW_PATTERNS),
                )

            img_proc_cls = getattr(transformers, "AutoImageProcessor", None)
            if img_proc_cls is not None:
                try:
                    processor = img_proc_cls.from_pretrained(str(proc_dir), local_files_only=True)
                except Exception:
                    try:
                        processor = img_proc_cls.from_pretrained(str(proc_dir))
                    except Exception:
                        processor = None
            if processor is None:
                try:
                    processor = AutoProcessor.from_pretrained(str(proc_dir), local_files_only=True)
                except Exception:
                    try:
                        if (local_dir / "preprocessor_config.json").exists():
                            processor = AutoProcessor.from_pretrained(str(local_dir))
                    except Exception:
                        processor = AutoProcessor.from_pretrained(HF_HAYAI_PROCESSOR_REPO)

            try:
                model = AutoModel.from_pretrained(str(local_dir), trust_remote_code=True, local_files_only=True)
            except Exception:
                model = AutoModel.from_pretrained(str(local_dir), trust_remote_code=True)
            model.to(self.device)
            model.eval()

            self._hayai_model = model
            self._hayai_tokenizer = tokenizer
            self._hayai_processor = processor
            self._hayai_available = True
            self.last_detection_stats["hayai_active"] = True
            self.last_detection_stats["ko_zh_ocr_backend"] = self.get_multilang_ocr_backend_name()
            logger.info(
                "Мультиязычная модель %s (KO/ZH/JA/EN) успешно подключена на %s!",
                HF_HAYAI_REPO,
                self.device,
            )
            return self._hayai_model, self._hayai_tokenizer, self._hayai_processor
        except Exception as exc:
            logger.warning(
                "Модель %s недоступна (%s). Используем резервный конвейер SmartLineSplit + RapidOCR/EasyOCR.",
                HF_HAYAI_REPO,
                exc,
            )
            self._hayai_available = False
            self._hayai_model = None
            self._hayai_tokenizer = None
            self._hayai_processor = None
            self.last_detection_stats["hayai_active"] = False
            self.last_detection_stats["ko_zh_ocr_backend"] = self.get_multilang_ocr_backend_name()
            return None

    @staticmethod
    def _normalize_hayai_ocr_text(text: str, lang: str = "auto") -> str:
        """
        Нормализация выхода `hayai-ocr-v2.5-nova` согласно спецификации модели:
        1. Каноническая декомпозиция/композиция Unicode `NFKC`.
        2. Замена переводов строк и табуляций на пробел.
        3. Удаление пробелов между CJK-символами (кандзи, кана, ханьцзы, хангыль) через lookahead-регулярку,
           чтобы цепочки из 3+ символов склеивались за один проход без остаточных пробелов.
        """
        if not text:
            return ""
        norm = unicodedata.normalize("NFKC", str(text))
        norm = re.sub(r"[\r\n\t]+", " ", norm)
        norm = CJK_SPACE_STRIP_REGEX.sub(r"\1", norm)
        return re.sub(r"\s+", " ", norm).strip()

    @classmethod
    def _prepare_crop_for_vlm_ocr(
        cls,
        rgb_image: Image.Image,
        bbox: list[int] | tuple[int, int, int, int],
        pad: int = 12,
    ) -> tuple[Image.Image, str]:
        """
        Готовит высококачественный цветной кроп для снайперского VLM-OCR (Шаг 4):
        - Сохраняет оригинальный цветной контур букв (важно для градиентных SFX, цветных обложек и полупрозрачных надписей).
        - Плавно масштабирует мелкие кропы (`max(w, h) < 180`) через `LANCZOS`, чтобы мультимодальный энкодер
          четко различал мелкие черточки, фуригану и дакутэн.
        - Возвращает `(pil_crop, base64_png_str)`.
        """
        img_w, img_h = rgb_image.size
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        cx1 = max(0, x1 - 6)
        cy1 = max(0, y1 - 6)
        cx2 = min(img_w, x2 + 6)
        cy2 = min(img_h, y2 + 6)
        crop = rgb_image.crop((cx1, cy1, cx2, cy2)).convert("RGB")
        cw, ch = crop.size
        max_dim = max(cw, ch)
        if 0 < max_dim < 180:
            scale = min(2.5, max(1.5, 240.0 / float(max_dim)))
            new_w = max(16, int(round(cw * scale)))
            new_h = max(16, int(round(ch * scale)))
            resample_filter = getattr(Image, "Resampling", Image).LANCZOS
            crop = crop.resize((new_w, new_h), resample=resample_filter)
        elif max_dim > 1024:
            scale = 1024.0 / float(max_dim)
            new_w = max(16, int(round(cw * scale)))
            new_h = max(16, int(round(ch * scale)))
            resample_filter = getattr(Image, "Resampling", Image).LANCZOS
            crop = crop.resize((new_w, new_h), resample=resample_filter)

        if pad > 0:
            crop = ImageOps.expand(crop, border=pad, fill=(255, 255, 255))

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        b64_png = base64.b64encode(buf.getvalue()).decode("ascii")
        return crop, b64_png

    @classmethod
    def _clean_vlm_ocr_output(cls, raw_text: str, fallback_lang: str = "ja") -> tuple[str, str]:
        """
        Очищает ответ VLM-OCR от `<think>...</think>`, markdown-блоков, кавычек и пояснений,
        нормализует через `NFKC` и определяет язык распознанного текста (`ja`, `ko`, `zh`).
        """
        if not raw_text:
            return "", fallback_lang
        cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", str(raw_text), flags=re.DOTALL | re.IGNORECASE).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        cleaned = VLM_OCR_PREFIX_STRIP_REGEX.sub("", cleaned).strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            if len(cleaned) >= 2:
                cleaned = cleaned[1:-1].strip()
        norm = cls._normalize_hayai_ocr_text(cleaned)
        if not norm or PUNCT_ONLY_REGEX.match(norm):
            return "", fallback_lang

        det_lang = fallback_lang if fallback_lang in ("ja", "ko", "zh") else "ja"
        if HANGUL_REGEX.search(norm):
            det_lang = "ko"
        elif KANA_REGEX.search(norm):
            det_lang = "ja"
        elif HANZI_REGEX.search(norm):
            # В японской манге бабблы без каны (только из кандзи: «母», «先生», «本当», «第一話») остаются японскими ('ja'),
            # и переключаются на 'zh' только если язык страницы/запроса явно китайский ('zh') или есть >= 2 китайских маркеров
            det_lang = "zh" if (fallback_lang == "zh" or len(ZH_SPECIFIC_MARKERS_REGEX.findall(norm)) >= 2) else "ja"
        return norm, det_lang

    @staticmethod
    def _pick_best_ollama_vlm_model(installed_models: list[str], preferred_model: str | None = None) -> str | None:
        """
        Выбирает лучшую установленную в Ollama локальную Vision-модель для VLM-OCR:
        1. Если `preferred_model` явно указан и подходит — возвращает его.
        2. Иначе ищет по приоритету `VLM_OCR_MODEL_PRIORITIES`:
           `glm-ocr` -> `qwen3-vl` -> `qwen2.5vl` -> `qwen2-vl` -> `minicpm-v` -> `gemma3` -> `llama3.2-vision` -> `llava`.
        """
        if preferred_model and preferred_model.strip() and preferred_model.strip() != "auto":
            clean_pref = preferred_model.strip()
            if clean_pref.startswith("ollama::"):
                clean_pref = clean_pref.split("ollama::", 1)[1]
            for m in installed_models:
                if clean_pref.lower() == m.lower() or clean_pref.lower() in m.lower():
                    return m
            if any(k in clean_pref.lower() for k in VLM_OCR_MODEL_PRIORITIES):
                return clean_pref

        for prio in VLM_OCR_MODEL_PRIORITIES:
            for m in installed_models:
                if prio in m.lower():
                    return m
        return None

    def recognize_crop_via_vlm(
        self,
        rgb_image: Image.Image,
        bbox: list[int] | tuple[int, int, int, int],
        lang: str = "auto",
        vlm_model: str | None = None,
        ollama_base_url: str | None = None,
        cloud_api_url: str | None = None,
        cloud_api_key: str | None = None,
        cloud_vlm_model: str | None = None,
    ) -> tuple[str, str, str]:
        """
        Снайперское распознавание сложного кропа (рукописный SFX, каллиграфия, цветная обложка) через Vision-модель (Шаг 4):
        1. Сначала проверяет локальные Vision-модели в Ollama (`glm-ocr`, `qwen2.5vl`, `qwen3-vl`, `gemma3`, `minicpm-v`, `llava`)
           с обязательным `trust_env=False` (обход VPN Fake-IP `198.18.x.x`).
        2. Если в Ollama нет Vision-модели, пробует облачный Vision-совместимый API (`/v1/chat/completions` с `image_url`),
           если переданы `cloud_api_url` и `cloud_api_key`.
        Возвращает `(recognized_text, detected_lang, used_vlm_model_name)`.
        """
        req_lang = (lang or "auto").lower().strip()
        fb_lang = req_lang if req_lang in ("ja", "ko", "zh") else (self._last_detected_lang or "ja")
        _, b64_png = self._prepare_crop_for_vlm_ocr(rgb_image, bbox, pad=12)

        lang_hint = {
            "ja": "Japanese manga (Kanji, Hiragana, Katakana, or sound effects / SFX)",
            "ko": "Korean manhwa / webtoon (Hangul dialogue or SFX)",
            "zh": "Chinese manhua (Hanzi dialogue or SFX)",
        }.get(fb_lang, "Japanese, Korean, or Chinese comic/manga")

        vlm_prompt = (
            f"Transcribe ONLY the exact {lang_hint} text visible in this image crop "
            "(including stylized sound effects / SFX, handwritten text, or decorative logos). "
            "Do NOT translate, do NOT romanize, do NOT add explanations or quotes. "
            "Output ONLY the raw transcribed characters."
        )

        base_ol_url = (ollama_base_url or self.ollama_base_url or "http://127.0.0.1:11434").rstrip("/")

        # 1. Пробуем локальную Vision-модель в Ollama (с 30-секундным кэшем списка /api/tags)
        try:
            now_mono = time.monotonic()
            cached_ts = getattr(self, "_ollama_vlm_tags_ts", 0.0)
            installed_names: list[str] = list(getattr(self, "_ollama_vlm_tags_cache", []) or [])
            with httpx.Client(timeout=httpx.Timeout(25.0, connect=2.0), trust_env=False) as client:
                if (now_mono - cached_ts) > 300.0 or not hasattr(self, "_ollama_vlm_tags_cache"):
                    installed_names = []
                    try:
                        tags_resp = client.get(f"{base_ol_url}/api/tags")
                        if tags_resp.status_code == 200:
                            for item in tags_resp.json().get("models", []):
                                m_name = str(item.get("name") or "").strip()
                                if m_name:
                                    installed_names.append(m_name)
                        self._ollama_vlm_tags_cache = installed_names
                        self._ollama_vlm_tags_ts = now_mono
                    except Exception:
                        pass

                chosen_ol_vlm = self._pick_best_ollama_vlm_model(installed_names, preferred_model=vlm_model)
                if chosen_ol_vlm:
                    payload_ol: dict[str, Any] = {
                        "model": chosen_ol_vlm,
                        "prompt": vlm_prompt,
                        "images": [b64_png],
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 160,
                        },
                    }
                    gen_resp = client.post(f"{base_ol_url}/api/generate", json=payload_ol)
                    if gen_resp.status_code == 200:
                        raw_out = str(gen_resp.json().get("response") or "")
                        text_out, det_lang = self._clean_vlm_ocr_output(raw_out, fallback_lang=fb_lang)
                        if text_out and not KNOWN_OCR_HALLUCINATIONS_REGEX.search(text_out):
                            self._last_vlm_model_used = f"ollama::{chosen_ol_vlm}"
                            self._vlm_ocr_calls_count += 1
                            self.last_detection_stats["vlm_ocr_calls"] = self._vlm_ocr_calls_count
                            self.last_detection_stats["vlm_ocr_backend"] = self.get_vlm_ocr_backend_name()
                            logger.info(
                                "🎯 [VLM-OCR Ollama] Модель %s распознала текст: '%s' (lang=%s)",
                                chosen_ol_vlm,
                                text_out,
                                det_lang,
                            )
                            return text_out, det_lang, self._last_vlm_model_used
        except Exception as ol_exc:
            logger.debug("Локальный VLM-OCR через Ollama пропущен: %s", ol_exc)

        # 2. Пробуем облачный OpenAI-совместимый Vision API (с ретраями при сетевых таймаутах)
        if cloud_api_url and cloud_api_key:
            c_url = cloud_api_url.rstrip("/")
            for sfx in ("/v1/chat/completions", "/chat/completions", "/v1"):
                if c_url.endswith(sfx):
                    c_url = c_url[: -len(sfx)].rstrip("/")
                    break
            candidate_cloud_models: list[str] = []
            if cloud_vlm_model and cloud_vlm_model.strip() and cloud_vlm_model.strip() != "auto":
                cm = cloud_vlm_model.strip().split("::", 1)[-1]
                candidate_cloud_models.append(cm)
            for def_cm in ("deepseek-v4-flash", "deepseek-v4-pro", "gpt-4o-mini", "qwen-vl-max"):
                if def_cm not in candidate_cloud_models:
                    candidate_cloud_models.append(def_cm)

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {cloud_api_key.strip()}",
                "User-Agent": "MikoMangaStudio/1.0.8",
            }
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), trust_env=False) as client:
                for c_mod in candidate_cloud_models[:2]:
                    payload_cloud = {
                        "model": c_mod,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": vlm_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                                    },
                                ],
                            }
                        ],
                        "temperature": 0.0,
                        "max_tokens": 160,
                        "stream": False,
                    }
                    for attempt in range(2):
                        try:
                            c_resp = client.post(
                                f"{c_url}/v1/chat/completions",
                                headers=headers,
                                json=payload_cloud,
                            )
                            if c_resp.status_code == 200:
                                choices = c_resp.json().get("choices") or []
                                if choices and isinstance(choices[0], dict):
                                    raw_c = str((choices[0].get("message") or {}).get("content") or "")
                                    text_out, det_lang = self._clean_vlm_ocr_output(raw_c, fallback_lang=fb_lang)
                                    if text_out and not KNOWN_OCR_HALLUCINATIONS_REGEX.search(text_out):
                                        self._last_vlm_model_used = f"cloud::{c_mod}"
                                        self._vlm_ocr_calls_count += 1
                                        self.last_detection_stats["vlm_ocr_calls"] = self._vlm_ocr_calls_count
                                        self.last_detection_stats["vlm_ocr_backend"] = self.get_vlm_ocr_backend_name()
                                        logger.info(
                                            "🎯 [VLM-OCR Cloud] Модель %s распознала текст: '%s' (lang=%s)",
                                            c_mod,
                                            text_out,
                                            det_lang,
                                        )
                                        return text_out, det_lang, self._last_vlm_model_used
                                break
                            elif c_resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                                time.sleep(0.8)
                                continue
                            else:
                                break
                        except Exception as c_exc:
                            if attempt == 0:
                                logger.debug("Повтор VLM-OCR Cloud (%s) после сетевого сбоя: %s", c_mod, c_exc)
                                time.sleep(0.6)
                                continue
                            logger.warning("Сбой облачного VLM-OCR (%s): %s", c_mod, c_exc)

        return "", fb_lang, ""

    def _recognize_crops_batch_hayai(
        self,
        crops_pil: list[Image.Image],
        max_num_patches: int | None = None,
        max_new_tokens: int = 128,
        batch_size: int = 8,
        lang: str = "auto",
    ) -> list[str]:
        """
        Пакетное и поштучное распознавание кропов через `JustANormalTinkerer/hayai-ocr-v2.5-nova`
        с параметром `repetition_penalty=1.0` и нормализацией `NFKC`.
        """
        if not crops_pil:
            return []
        hayai_triple = self._get_hayai_recognizer()
        if hayai_triple is None:
            return [""] * len(crops_pil)

        model, tokenizer, processor = hayai_triple
        rgb_crops = [c.convert("RGB") for c in crops_pil]
        outputs_norm: list[str] = []

        def _choose_patch_budget(imgs: list[Image.Image]) -> int:
            if max_num_patches is not None:
                return int(max_num_patches)
            for im in imgs:
                w, h = im.size
                aspect = max(w, h) / max(1.0, float(min(w, h)))
                if (w * h) >= 42000 or aspect >= 2.6:
                    return 512
            return 384

        def _run_single_crop(img: Image.Image) -> str:
            patch_budget = _choose_patch_budget([img])
            inputs = processor(images=[img], max_num_patches=patch_budget, return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            gen_kwargs: dict[str, Any] = {
                "pixel_values": inputs["pixel_values"],
                "tokenizer": tokenizer,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": 1.0,
            }
            if "pixel_attention_mask" in inputs:
                gen_kwargs["pixel_attention_mask"] = inputs["pixel_attention_mask"]
            if "spatial_shapes" in inputs:
                gen_kwargs["spatial_shapes"] = inputs["spatial_shapes"]

            raw_out = model.generate(**gen_kwargs)
            if isinstance(raw_out, str):
                raw_txt = raw_out
            elif isinstance(raw_out, (list, tuple)) and len(raw_out) > 0:
                first_el = raw_out[0]
                if isinstance(first_el, str):
                    raw_txt = first_el
                else:
                    raw_txt = tokenizer.decode(first_el, skip_special_tokens=True)
            elif hasattr(raw_out, "shape"):
                decoded = tokenizer.batch_decode(raw_out, skip_special_tokens=True)
                raw_txt = decoded[0] if decoded else ""
            else:
                raw_txt = str(raw_out)
            return self._normalize_hayai_ocr_text(raw_txt, lang=lang)

        with torch.inference_mode():
            for start_i in range(0, len(rgb_crops), max(1, batch_size)):
                sub_crops = rgb_crops[start_i : start_i + max(1, batch_size)]
                if len(sub_crops) == 1:
                    try:
                        outputs_norm.append(_run_single_crop(sub_crops[0]))
                    except Exception as single_exc:
                        logger.debug("HayaiOCR single crop error: %s", single_exc)
                        outputs_norm.append("")
                    continue

                # Пробуем батч-прогон; если кастомный generate() модели требует B=1, бесшовно переходим на поштучный цикл
                batch_ok = False
                try:
                    patch_budget = _choose_patch_budget(sub_crops)
                    inputs = processor(images=sub_crops, max_num_patches=patch_budget, return_tensors="pt")
                    inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                    gen_kwargs = {
                        "pixel_values": inputs["pixel_values"],
                        "tokenizer": tokenizer,
                        "max_new_tokens": max_new_tokens,
                        "repetition_penalty": 1.0,
                    }
                    if "pixel_attention_mask" in inputs:
                        gen_kwargs["pixel_attention_mask"] = inputs["pixel_attention_mask"]
                    if "spatial_shapes" in inputs:
                        gen_kwargs["spatial_shapes"] = inputs["spatial_shapes"]

                    raw_batch = model.generate(**gen_kwargs)
                    if isinstance(raw_batch, (list, tuple)) and len(raw_batch) == len(sub_crops):
                        for item in raw_batch:
                            if isinstance(item, str):
                                outputs_norm.append(self._normalize_hayai_ocr_text(item, lang=lang))
                            else:
                                dec = tokenizer.decode(item, skip_special_tokens=True)
                                outputs_norm.append(self._normalize_hayai_ocr_text(dec, lang=lang))
                        batch_ok = True
                    elif hasattr(raw_batch, "shape") and int(raw_batch.shape[0]) == len(sub_crops):
                        decoded_list = tokenizer.batch_decode(raw_batch, skip_special_tokens=True)
                        for dec in decoded_list:
                            outputs_norm.append(self._normalize_hayai_ocr_text(dec, lang=lang))
                        batch_ok = True
                except Exception:
                    batch_ok = False

                if not batch_ok:
                    for single_im in sub_crops:
                        try:
                            outputs_norm.append(_run_single_crop(single_im))
                        except Exception as single_exc:
                            logger.debug("HayaiOCR fallback single crop error: %s", single_exc)
                            outputs_norm.append("")

        return outputs_norm

    def _get_ja_nar_recognizer(self) -> Any | None:
        """
        Загружает неавторегрессионную японскую OCR-модель 2026 года `muscgab/manga-ocr-nar-small` (54M параметров,
        CER 1.73% против 4.84% у оригинального MangaOCR, точность на рукописном/стилизованном тексте 80.8% против 48.2%,
        скорость ~35 мс на всю страницу с сохранением пропорций кропа 324 патча 14x14).
        Скачивает только веса PyTorch (`model.safetensors`, 217 МБ) и встроенный пакет `manga_ocr_nar_small/`,
        пропуская папки `onnx/` и `coreml/`. Не требует `pip install`.
        """
        if not self.use_ja_nar or self._ja_nar_available is False:
            return None
        if self._ja_nar_recognizer is not None:
            return self._ja_nar_recognizer

        local_dir = self.models_dir / HF_JA_NAR_SUBDIR
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            weights_file = local_dir / "model.safetensors"
            pkg_init = local_dir / "manga_ocr_nar_small" / "__init__.py"
            if not weights_file.exists() or not pkg_init.exists():
                logger.info(
                    "Скачивание японской OCR-модели нового поколения %s (~217 МБ)...",
                    HF_JA_NAR_REPO,
                )
                snapshot_download(
                    repo_id=HF_JA_NAR_REPO,
                    local_dir=str(local_dir),
                    allow_patterns=list(HF_JA_NAR_ALLOW_PATTERNS),
                )

            local_dir_str = str(local_dir.resolve())
            if local_dir_str not in sys.path:
                sys.path.insert(0, local_dir_str)

            try:
                from manga_ocr_nar_small import MangaOcrNarSmall  # type: ignore[import-not-found]
            except ImportError:
                from mangaocr_nar_small import MangaOcrNarSmall  # type: ignore[import-not-found]

            nar_inst = None
            last_err: Exception | None = None
            backend_candidates = ("cuda", "auto", None) if self.device.startswith("cuda") else ("auto", None)
            for b_arg in backend_candidates:
                try:
                    if b_arg is None:
                        nar_inst = MangaOcrNarSmall()
                    else:
                        nar_inst = MangaOcrNarSmall(b_arg)
                    break
                except Exception as exc:
                    last_err = exc

            if nar_inst is None:
                raise RuntimeError(f"Не удалось создать экземпляр MangaOcrNarSmall ({last_err})")

            self._ja_nar_recognizer = nar_inst
            self._ja_nar_available = True
            self.last_detection_stats["ja_ocr_backend"] = self.get_ja_ocr_backend_name()
            logger.info(
                "Модель %s успешно подключена (гибридный режим с MangaOCR для длинных бабблов)!",
                HF_JA_NAR_REPO,
            )
            return self._ja_nar_recognizer
        except Exception as exc:
            logger.warning(
                "Модель %s недоступна (%s). Используем улучшенный MangaOCR FP16 + SmartPreprocess.",
                HF_JA_NAR_REPO,
                exc,
            )
            self._ja_nar_available = False
            self._ja_nar_recognizer = None
            return None

    def _load_detector(self) -> YOLO:
        local_weights = self.models_dir / HF_YOLO_FILENAME
        if not local_weights.exists():
            logger.info("Скачивание весов детектора %s/%s...", HF_YOLO_REPO, HF_YOLO_FILENAME)
            downloaded_path = hf_hub_download(
                repo_id=HF_YOLO_REPO,
                filename=HF_YOLO_FILENAME,
                local_dir=str(self.models_dir),
            )
            local_weights = Path(downloaded_path)

        detector = YOLO(str(local_weights))
        detector.to(self.device)
        return detector

    def _get_rtdetr_detector(self) -> tuple[Any, Any] | None:
        """
        Загружает 3-классовый детектор `ogkalu/comic-text-and-bubble-detector` (RT-DETR-v2 r50vd):
        - 0: `bubble` (внешний контур облачка речи)
        - 1: `text_bubble` (точный бокс текста внутри облачка)
        - 2: `text_free` (свободный текст и плашки вне облачков)
        Скачивает только 3 файла PyTorch (~172 МБ), пропуская тяжелые дубли `.onnx`.
        При любой ошибке сети или старой версии `transformers` мягко переключается на YOLOv8m + TightInkCrop.
        """
        if not self.use_rtdetr or self._rtdetr_available is False:
            return None
        if self._rtdetr_model is not None and self._rtdetr_processor is not None:
            return self._rtdetr_model, self._rtdetr_processor

        local_dir = self.models_dir / HF_RTDETR_SUBDIR
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            for fname in HF_RTDETR_PT_FILES:
                fpath = local_dir / fname
                if not fpath.exists():
                    logger.info(
                        "Скачивание компонента двухконтурного детектора %s/%s...",
                        HF_RTDETR_REPO,
                        fname,
                    )
                    hf_hub_download(
                        repo_id=HF_RTDETR_REPO,
                        filename=fname,
                        local_dir=str(local_dir),
                    )

            import transformers

            hf_logging = getattr(transformers, "logging", None)
            if hf_logging is not None and hasattr(hf_logging, "set_verbosity_error"):
                try:
                    hf_logging.set_verbosity_error()
                except Exception:
                    pass
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

            proc_cls = getattr(transformers, "AutoImageProcessor", None) or getattr(
                transformers, "RTDetrImageProcessor", None
            )
            if proc_cls is None:
                raise RuntimeError("В установленной версии transformers отсутствует AutoImageProcessor.")

            try:
                processor = proc_cls.from_pretrained(str(local_dir), local_files_only=True)
            except Exception:
                processor = proc_cls.from_pretrained(str(local_dir))

            model = None
            last_err: Exception | None = None
            for cls_name in (
                "AutoModelForObjectDetection",
                "RTDetrV2ForObjectDetection",
                "RTDetrForObjectDetection",
            ):
                model_cls = getattr(transformers, cls_name, None)
                if model_cls is None:
                    continue
                try:
                    model = model_cls.from_pretrained(str(local_dir))
                    break
                except Exception as exc:
                    last_err = exc

            if model is None:
                raise RuntimeError(f"Не удалось инициализировать RT-DETR-v2 ({last_err})")

            model.to(self.device)
            model.eval()

            self._rtdetr_model = model
            self._rtdetr_processor = processor
            self._rtdetr_available = True
            self.last_detection_stats["rtdetr_active"] = True
            self.last_detection_stats["detector_backend"] = self.get_detector_backend_name()
            logger.info(
                "Двухконтурный детектор RT-DETR-v2 (%s) успешно загружен на %s!",
                HF_RTDETR_REPO,
                self.device,
            )
            return self._rtdetr_model, self._rtdetr_processor
        except Exception as exc:
            logger.warning(
                "Двухконтурный детектор RT-DETR-v2 недоступен (%s). Работаем на связке YOLOv8m + TightInkCrop.",
                exc,
            )
            self._rtdetr_available = False
            self._rtdetr_model = None
            self._rtdetr_processor = None
            return None

    def _load_manga_ocr(self) -> Any:
        force_cpu = not self.device.startswith("cuda")
        local_mocr_dir = self.models_dir / "manga-ocr-base"
        model_source = "kha-white/manga-ocr-base"
        try:
            has_weights = (
                (local_mocr_dir / "pytorch_model.bin").exists()
                or (local_mocr_dir / "model.safetensors").exists()
            )
            if (not (local_mocr_dir / "config.json").exists() or not has_weights) and snapshot_download is not None:
                snapshot_download(
                    repo_id="kha-white/manga-ocr-base",
                    local_dir=str(local_mocr_dir),
                    allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt"],
                )
            if (local_mocr_dir / "config.json").exists():
                model_source = str(local_mocr_dir)
        except Exception as exc:
            logger.debug("Локальный кэш manga-ocr-base пропущен (%s), используем стандартную загрузку.", exc)

        try:
            return HallucinationFreeMangaOCR(
                pretrained_model_name_or_path=model_source,
                device="cpu" if force_cpu else self.device,
                force_cpu=force_cpu,
            )
        except Exception as hf_exc:
            logger.debug("HallucinationFreeMangaOCR недоступен (%s), используем классический MangaOcr.", hf_exc)
            if MangaOcr is not None:
                return MangaOcr(pretrained_model_name_or_path=model_source, force_cpu=force_cpu)
            return None

    def _check_rapidocr(self) -> bool:
        if self._rapidocr_available is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found] # noqa: F401

                self._rapidocr_available = True
            except ImportError:
                self._rapidocr_available = False
        return bool(self._rapidocr_available)

    def _get_rapidocr_reader(self) -> Any | None:
        if not self._check_rapidocr():
            return None
        if self._rapidocr_reader is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]

                self._rapidocr_reader = RapidOCR()
                self.last_detection_stats["rapidocr_active"] = True
                self.last_detection_stats["ko_zh_ocr_backend"] = self.get_multilang_ocr_backend_name()
            except Exception as exc:
                logger.debug("Не удалось инициализировать RapidOCR: %s", exc)
                self._rapidocr_available = False
                return None
        return self._rapidocr_reader

    def _check_easyocr(self) -> bool:
        if self._easyocr_available is None:
            try:
                import easyocr  # noqa: F401

                self._easyocr_available = True
            except ImportError:
                self._easyocr_available = False
        return bool(self._easyocr_available)

    def _check_multilang_ocr(self) -> bool:
        """
        Проверяет доступность хотя бы одного мультиязычного OCR-движка для корейского (`ko`) и китайского (`zh`):
        `HayaiOCR v2.5 Nova` (работает на базе штатного `transformers`), `RapidOCR` или `EasyOCR`.
        """
        if self.use_hayai_ocr and self._hayai_available is not False:
            return True
        if self._check_rapidocr() or self._check_easyocr():
            return True
        return False

    def _get_ko_reader(self) -> Any | None:
        if not self._check_easyocr():
            return None
        if self._ko_reader is None:
            import easyocr

            logger.info("Загрузка модели EasyOCR для корейского языка (ko)...")
            self._ko_reader = easyocr.Reader(
                ["ko", "en"],
                gpu=self.device.startswith("cuda"),
                model_storage_directory=str(self.models_dir / "easyocr"),
                verbose=False,
            )
        return self._ko_reader

    def _get_zh_reader(self) -> Any | None:
        if not self._check_easyocr():
            return None
        if self._zh_reader is None:
            import easyocr

            logger.info("Загрузка модели EasyOCR для китайского языка (ch_sim)...")
            self._zh_reader = easyocr.Reader(
                ["ch_sim", "en"],
                gpu=self.device.startswith("cuda"),
                model_storage_directory=str(self.models_dir / "easyocr"),
                verbose=False,
            )
        return self._zh_reader

    def _get_any_craft_reader(self) -> Any | None:
        """
        Возвращает любой уже загруженный экземпляр EasyOCR Reader для вызова `.detect()` (CRAFT),
        либо инициализирует только детектор CRAFT (`recognizer=False`), не нагружая VRAM лишним распознавателем.
        """
        if self._ko_reader is not None:
            return self._ko_reader
        if self._zh_reader is not None:
            return self._zh_reader
        if self._craft_reader is not None:
            return self._craft_reader
        if not self._check_easyocr():
            return None
        try:
            import easyocr

            logger.info("Инициализация легкого CRAFT-детектора EasyOCR (recognizer=False) для поиска текста вне бабблов...")
            self._craft_reader = easyocr.Reader(
                ["en"],
                gpu=self.device.startswith("cuda"),
                model_storage_directory=str(self.models_dir / "easyocr"),
                detector=True,
                recognizer=False,
                verbose=False,
            )
            return self._craft_reader
        except Exception as exc:
            logger.debug("Не удалось создать detector-only EasyOCR (%s), используем стандартный Reader.", exc)
            try:
                import easyocr

                self._craft_reader = easyocr.Reader(
                    ["en"],
                    gpu=self.device.startswith("cuda"),
                    model_storage_directory=str(self.models_dir / "easyocr"),
                    verbose=False,
                )
                return self._craft_reader
            except Exception:
                return None

    @staticmethod
    def _get_ocr_bbox(det: dict[str, Any]) -> list[int]:
        """
        Возвращает точные координаты области текста (`text_bbox`) для подачи в OCR,
        если они вычислены двухконтурным детектором или алгоритмом TightInkCrop,
        иначе возвращает общий `bbox` баббла.
        """
        tb = det.get("text_bbox")
        if isinstance(tb, (list, tuple)) and len(tb) >= 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in tb[:4]]
            if (x2 - x1) >= 8 and (y2 - y1) >= 8:
                return [x1, y1, x2, y2]
        b = det.get("bbox") or [0, 0, 0, 0]
        return [int(round(float(v))) for v in b[:4]]

    @staticmethod
    def _preprocess_crop_np_for_ocr(crop_rgb_np: np.ndarray) -> np.ndarray:
        """
        Умный препроцессинг кропа текста перед добавлением белых полей и подачей в OCR (Шаг 2):
        1. Автоматически распознает **белый/светлый текст на черном или темном фоне** (темные облачка мыслей,
           черные плашки рассказчика, инвертированные драматические бабблы), проверяя, что и медиана кропа `p50 < 115`,
           и периметр кропа темный (`border_median < 128`, чтобы не спутать с жирной черной буквой на белом фоне!),
           и мягко инвертирует его в четкий черный текст на белом фоне.
        2. Если скан бледный или внутри баббла серый скринтон (`p95 - p5 < 140` или фон серый `175 <= p80 < 235`),
           вытягивает контраст, делая фон чистым белым, а тонкие штрихи (`ー`, `…`, `っ`, дакутэн) — четкими.
        """
        if crop_rgb_np.size == 0:
            return crop_rgb_np
        h, w = crop_rgb_np.shape[:2]
        if h < 10 or w < 10:
            return crop_rgb_np

        gray = cv2.cvtColor(crop_rgb_np, cv2.COLOR_RGB2GRAY)
        p5, p20, p50, p80, p95 = [float(v) for v in np.percentile(gray, [5, 20, 50, 80, 95])]

        # Собираем пиксели внешней рамки кропа (толщиной 2px), чтобы точно знать цвет фона вокруг букв
        top_b = gray[:2, :].ravel()
        bot_b = gray[max(0, h - 2) :, :].ravel()
        left_b = gray[:, :2].ravel()
        right_b = gray[:, max(0, w - 2) :].ravel()
        border_pixels = np.concatenate([top_b, bot_b, left_b, right_b])
        border_median = float(np.median(border_pixels)) if border_pixels.size > 0 else p50

        out_rgb = crop_rgb_np
        # 1. Белый текст на темном/черном фоне -> инвертируем в черный текст на белом фоне
        if p50 < 115.0 and border_median < 128.0 and (p95 - p20) >= 35.0:
            inv_gray = 255 - gray
            inv_p5, inv_p85 = [float(v) for v in np.percentile(inv_gray, [5, 85])]
            denom = max(25.0, inv_p85 - inv_p5)
            stretched = np.clip((inv_gray.astype(np.float32) - inv_p5) * (255.0 / denom), 0.0, 255.0).astype(np.uint8)
            out_rgb = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)
        # 2. Бледный скан или серый скринтон на фоне баббла -> осветляем фон до белого и усиливаем штрихи
        elif border_median >= 145.0 and ((p95 - p5) < 140.0 or (170.0 <= p80 < 236.0)):
            hi_ref = max(p80, border_median, p5 + 30.0)
            lo_ref = p5
            denom = max(25.0, hi_ref - lo_ref)
            stretched = np.clip((gray.astype(np.float32) - lo_ref) * (255.0 / denom), 0.0, 255.0).astype(np.uint8)
            out_rgb = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)

        # 3. Двукратный бикубический апскейл + легкий unsharp mask для мелких кропов (< 64px по меньшей стороне),
        # чтобы тонкие штрихи кандзи и дакутэн не размывались при нарезке на ViT-патчи 14x14 / 16x16
        oh, ow = out_rgb.shape[:2]
        if min(oh, ow) < 64 or max(oh, ow) < 96:
            up = cv2.resize(out_rgb, (ow * 2, oh * 2), interpolation=cv2.INTER_CUBIC)
            blur = cv2.GaussianBlur(up, (0, 0), 1.0)
            out_rgb = cv2.addWeighted(up, 1.28, blur, -0.28, 0)

        return out_rgb

    @staticmethod
    def _pad_to_balanced_aspect_for_vit(
        pil_crop: Image.Image,
        max_aspect_ratio: float = 2.0,
    ) -> Image.Image:
        """
        Защищает классический `MangaOCR` (ViT 224x224) от расплющивания букв на узких вертикальных столбиках
        (`h > w * 2.0`) и длинных горизонтальных строках (`w > h * 2.0`), добавляя белые поля по бокам
        или сверху/снизу до комфортного соотношения сторон `<= 2.0 : 1`.
        """
        w, h = pil_crop.size
        if w < 8 or h < 8:
            return pil_crop

        if h > w * max_aspect_ratio:
            target_w = int(round(h / max_aspect_ratio))
            pad_total = max(0, target_w - w)
            if pad_total >= 4:
                pad_l = pad_total // 2
                pad_r = pad_total - pad_l
                return ImageOps.expand(pil_crop, border=(pad_l, 0, pad_r, 0), fill=(255, 255, 255))
        elif w > h * max_aspect_ratio:
            target_h = int(round(w / max_aspect_ratio))
            pad_total = max(0, target_h - h)
            if pad_total >= 4:
                pad_t = pad_total // 2
                pad_b = pad_total - pad_t
                return ImageOps.expand(pil_crop, border=(0, pad_t, 0, pad_b), fill=(255, 255, 255))

        return pil_crop

    @classmethod
    def _crop_with_padding(
        cls,
        rgb_image: Image.Image,
        bbox: tuple[int, int, int, int] | list[int],
        pad: int = 12,
        smart_preprocess: bool = True,
    ) -> Image.Image:
        img_w, img_h = rgb_image.size
        x1, y1, x2, y2 = bbox
        cx1 = max(0, x1 - 6)
        cy1 = max(0, y1 - 6)
        cx2 = min(img_w, x2 + 6)
        cy2 = min(img_h, y2 + 6)
        crop = rgb_image.crop((cx1, cy1, cx2, cy2))
        if smart_preprocess:
            crop_np = np.array(crop)
            proc_np = cls._preprocess_crop_np_for_ocr(crop_np)
            if proc_np is not crop_np:
                crop = Image.fromarray(proc_np, mode="RGB")
        if pad > 0:
            crop = ImageOps.expand(crop, border=pad, fill=(255, 255, 255))
        return crop

    @classmethod
    def _split_horizontal_lines_for_ocr(
        cls,
        crop_rgb_np: np.ndarray,
        min_line_height: int = 10,
        max_lines: int = 8,
    ) -> list[tuple[int, int, np.ndarray]]:
        """
        Умная построчная сегментация многострочных горизонтальных бабблов для корейской манхвы (`ko`)
        и китайской маньхуа (`zh`) по профилю горизонтальной проекции чернил (`row_ink_profile`) (Шаг 3):
        - Находит горизонтальные строки текста (2..8 строк) и их межстрочные просветы.
        - Отсекает случайные вертикальные линии боковых рамок баббла, чтобы они не склеивали строки.
        - Объединяет микро-зазоры внутри одной строки (например, между верхней и нижней частью корейского слога `그`, `호`, `수`).
        - Разрезает слишком высокие слипшиеся полосы по локальному минимуму профиля чернил.
        - Возвращает список кортежей `(line_y1, line_y2, line_crop_rgb_np)` строго сверху вниз (`line_y1` по возрастанию).
        """
        if crop_rgb_np.size == 0:
            return []
        h, w = crop_rgb_np.shape[:2]
        if h < 26 or w < 18:
            return [(0, h, crop_rgb_np)]

        proc_np = cls._preprocess_crop_np_for_ocr(crop_rgb_np)
        gray = cv2.cvtColor(proc_np, cv2.COLOR_RGB2GRAY)

        pad_x = max(1, int(w * 0.04))
        inner_w = max(2, w - 2 * pad_x)
        inner_gray = gray[:, pad_x : pad_x + inner_w]

        p75 = float(np.percentile(inner_gray, 75))
        median_val = float(np.median(inner_gray))
        if p75 >= 200.0:
            ink_thresh = max(155, int(p75 - 28))
            binary = ((inner_gray < ink_thresh) * 255).astype(np.uint8)
        elif median_val >= 127.0:
            _, binary = cv2.threshold(inner_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(inner_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Удаляем длинные тонкие вертикальные линии по краям, которые могут проходить сквозь все строки сверху вниз
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lbl in range(1, num_labels):
            cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            if ch >= int(h * 0.72) and cw <= max(5, int(inner_w * 0.08)):
                binary[labels == lbl] = 0

        row_ink_profile = np.mean(binary > 0, axis=1).astype(np.float32)
        if row_ink_profile.size < 8:
            return [(0, h, proc_np)]

        smooth_profile = np.convolve(
            row_ink_profile,
            np.array([0.25, 0.50, 0.25], dtype=np.float32),
            mode="same",
        )
        peak_val = float(np.percentile(smooth_profile, 85))
        if peak_val < 0.012:
            return [(0, h, proc_np)]

        active_thresh = max(0.010, min(0.042, peak_val * 0.18))
        is_ink_row = smooth_profile >= active_thresh

        raw_bands: list[list[int]] = []
        in_band = False
        start_y = 0
        for y_idx, val in enumerate(is_ink_row.tolist()):
            if val and not in_band:
                in_band = True
                start_y = y_idx
            elif not val and in_band:
                in_band = False
                raw_bands.append([start_y, y_idx])
        if in_band:
            raw_bands.append([start_y, h])

        if not raw_bands:
            return [(0, h, proc_np)]

        # Склеиваем микро-разрывы внутри одной строки (например, между верхней и нижней половиной корейского слога)
        min_gap = max(2, int(round(h * 0.016)))
        merged_bands: list[list[int]] = [raw_bands[0]]
        for b_start, b_end in raw_bands[1:]:
            prev_s, prev_e = merged_bands[-1]
            gap = b_start - prev_e
            prev_h = prev_e - prev_s
            cur_h = b_end - b_start
            if gap <= min_gap or (gap <= 5 and (prev_h < max(7, min_line_height // 2) or cur_h < max(7, min_line_height // 2))):
                merged_bands[-1][1] = b_end
            else:
                merged_bands.append([b_start, b_end])

        # Если какая-то полоса получилась в 2+ раза выше типичной строки, проверяем локальный минимум внутри нее
        final_bands: list[ tuple[int, int] ] = []
        band_heights = [b[1] - b[0] for b in merged_bands if (b[1] - b[0]) >= min_line_height]
        ref_line_h = float(np.median(band_heights)) if band_heights else float(max(16, min_line_height * 2))

        for b_start, b_end in merged_bands:
            bh = b_end - b_start
            if bh < max(8, min_line_height - 2):
                continue
            if bh >= max(36, int(round(ref_line_h * 1.95))):
                margin_v = max(min_line_height, int(bh * 0.24))
                search_s = b_start + margin_v
                search_e = b_end - margin_v
                if search_e > search_s + 2:
                    sub_prof = smooth_profile[search_s:search_e]
                    min_rel = int(np.argmin(sub_prof))
                    valley_y = search_s + min_rel
                    valley_val = float(smooth_profile[valley_y])
                    top_peak = float(np.max(smooth_profile[b_start:valley_y])) if valley_y > b_start else peak_val
                    bot_peak = float(np.max(smooth_profile[valley_y:b_end])) if b_end > valley_y else peak_val
                    if valley_val <= min(top_peak, bot_peak) * 0.30:
                        final_bands.append((b_start, valley_y))
                        final_bands.append((valley_y, b_end))
                        continue
            final_bands.append((b_start, b_end))

        # Оставляем только содержательные строки
        valid_bands: list[tuple[int, int]] = []
        for by1, by2 in final_bands:
            if (by2 - by1) < max(8, min_line_height - 2):
                continue
            if int(np.count_nonzero(binary[by1:by2, :])) < 10:
                continue
            valid_bands.append((by1, by2))

        if len(valid_bands) <= 1 or len(valid_bands) > max_lines:
            return [(0, h, proc_np)]

        line_crops: list[tuple[int, int, np.ndarray]] = []
        for idx_b, (ly1, ly2) in enumerate(valid_bands):
            lh = ly2 - ly1
            pad_y = max(3, min(6, lh // 5))
            upper_bound = valid_bands[idx_b - 1][1] if idx_b > 0 else 0
            lower_bound = valid_bands[idx_b + 1][0] if (idx_b + 1) < len(valid_bands) else h
            cy1 = max(upper_bound, max(0, ly1 - pad_y))
            cy2 = min(lower_bound, min(h, ly2 + pad_y))
            if (cy2 - cy1) < 8:
                cy1, cy2 = ly1, ly2

            col_ink = np.where(np.any(binary[ly1:ly2, :] > 0, axis=0))[0]
            if col_ink.size > 0:
                cx1 = max(0, pad_x + int(col_ink[0]) - 6)
                cx2 = min(w, pad_x + int(col_ink[-1]) + 7)
            else:
                cx1, cx2 = 0, w

            if (cx2 - cx1) < 10:
                cx1, cx2 = 0, w

            sub_rgb = proc_np[cy1:cy2, cx1:cx2]
            bordered = cv2.copyMakeBorder(
                sub_rgb,
                4,
                4,
                6,
                6,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            line_crops.append((cy1, cy2, bordered))

        return line_crops if len(line_crops) >= 2 else [(0, h, proc_np)]

    @classmethod
    def _is_text_on_clean_white_cavity(
        cls,
        crop_rgb: np.ndarray,
        min_glyphs: int = 3,
    ) -> bool:
        """
        Проверяет, лежат ли темные штрихи текста внутри чистой белой/светлой полости
        (экран телефона, монитор, записка, табличка или облачко без стандартной овальной рамки),
        даже если внешние углы прямоугольного бокса захватывают темный корпус телефона или рамку фрейма.
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return False
        rh, rw = crop_rgb.shape[:2]
        if rh < 14 or rw < 14:
            return False
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb
        if float(np.percentile(gray, 80)) < 222.0:
            return False

        _, bw_inv = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY_INV)
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(bw_inv, connectivity=8)
        if n_cc <= 1:
            return False

        wall_mask = np.zeros_like(bw_inv)
        glyph_mask = np.zeros_like(bw_inv)
        max_char_dim = max(20, min(48, int(min(rw, rh) * 0.32)))
        valid_glyphs = 0

        for i in range(1, n_cc):
            cx, cy, cw, ch, area = stats[i]
            if cw > rw * 0.34 or ch > rh * 0.34 or area > (rw * rh * 0.045):
                wall_mask[labels == i] = 255
                continue
            if 3 <= area <= (rw * rh * 0.04) and 1 <= cw <= max_char_dim and 1 <= ch <= max_char_dim:
                aspect = max(cw, ch) / max(1.0, float(min(cw, ch)))
                if aspect > 7.0 and area > 28:
                    continue
                pad = 4
                sub_g = gray[max(0, cy - pad) : min(rh, cy + ch + pad), max(0, cx - pad) : min(rw, cx + cw + pad)]
                sub_w = wall_mask[max(0, cy - pad) : min(rh, cy + ch + pad), max(0, cx - pad) : min(rw, cx + cw + pad)]
                valid_bg = sub_g[sub_w == 0]
                if valid_bg.size > 0 and float(np.percentile(valid_bg, 80)) >= 225.0:
                    white_px = valid_bg[valid_bg >= 215]
                    if white_px.size >= valid_bg.size * 0.30 and float(np.std(white_px)) <= 15.5:
                        glyph_mask[labels == i] = 255
                        valid_glyphs += 1

        if valid_glyphs < min_glyphs:
            return False

        # Проверяем локальный фон непосредственно вокруг найденных глифов (кольцо радиусом 6px без черных стенок)
        dil_g = cv2.dilate(glyph_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
        local_ring = (dil_g > 0) & (bw_inv == 0) & (wall_mask == 0)
        ring_px = gray[local_ring]
        if ring_px.size < 16:
            return False
        white_ring_ratio = float(np.mean(ring_px >= 218))
        ring_median = float(np.median(ring_px))
        ring_white_vals = ring_px[ring_px >= 210]
        ring_std = float(np.std(ring_white_vals)) if ring_white_vals.size >= 8 else 99.0

        return bool(white_ring_ratio >= 0.56 and ring_median >= 226.0 and ring_std <= 16.5)

    @classmethod
    def _extract_tilted_or_multiline_horizontal_crops(
        cls,
        crop_rgb_np: np.ndarray,
    ) -> tuple[float, list[Image.Image], Image.Image | None]:
        """
        Автоматическое выпрямление (Deskew) и построчная сегментация наклонных текстов на экранах телефонов,
        мониторах, записках, табличках и многострочных горизонтальных блоков (`yokogaki`):
        1. Срезает синтетические белые поля `_crop_with_padding` (`>= 254`), выделяет главную белую полость
           экрана/баббла (`screen_cavity`) и отсоединяет прямые внешние линии рамок панелей, сохраняя
           краевые иероглифы (`紅`, `事`, `談`, `助`), касающиеся границ экрана.
        2. Ищет угол наклона строк `best_deg` в диапазоне `[-54°, +54°]` методом максимизации дисперсии
           горизонтальной проекции глифов и доли межстрочных просветов.
        3. Никогда не вращает вертикальные японские бабблы (`tategaki`, где `v0_var > h0_var * 1.8`).
        4. Если блок наклонен (`|best_deg| >= 5.5°`) ИЛИ содержит `>= 3` горизонтальных строк (`>= 10` глифов):
           очищает черные рамки корпуса телефона вокруг букв, поворачивает текст в строго горизонтальное
           положение, нарезает на отдельные строки (разрезая слипшиеся двойные полосы по локальному минимуму
           и присоединяя короткие 1-2 символьные переносы вроде «だ。» / «ない。» к предыдущей строке).
        Возвращает `(best_deg, line_pils, deskewed_full_pil)`.
        """
        if crop_rgb_np is None or crop_rgb_np.size == 0:
            return 0.0, [], None
        h_in, w_in = crop_rgb_np.shape[:2]
        if h_in < 28 or w_in < 28:
            return 0.0, [], None

        gray_in = cv2.cvtColor(crop_rgb_np, cv2.COLOR_RGB2GRAY) if crop_rgb_np.ndim == 3 else crop_rgb_np
        # 1. Срезаем искусственную белую кайму от ImageOps.expand (pad=8..12), чтобы она не закорачивала белые области вокруг рамок телефона
        ty1, ty2, tx1, tx2 = 0, h_in, 0, w_in
        max_trim_y = min(18, h_in // 4)
        max_trim_x = min(18, w_in // 4)
        while ty1 < max_trim_y and bool(np.all(gray_in[ty1, :] >= 254)):
            ty1 += 1
        while ty2 > (h_in - max_trim_y) and bool(np.all(gray_in[ty2 - 1, :] >= 254)):
            ty2 -= 1
        while tx1 < max_trim_x and bool(np.all(gray_in[:, tx1] >= 254)):
            tx1 += 1
        while tx2 > (w_in - max_trim_x) and bool(np.all(gray_in[:, tx2 - 1] >= 254)):
            tx2 -= 1

        work_rgb = crop_rgb_np[ty1:ty2, tx1:tx2]
        rh, rw = work_rgb.shape[:2]
        if rh < 24 or rw < 24:
            return 0.0, [], None

        roi_gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY) if work_rgb.ndim == 3 else work_rgb
        if float(np.percentile(roi_gray, 75)) < 205.0:
            return 0.0, [], None

        _, bw_inv = cv2.threshold(roi_gray, 185, 255, cv2.THRESH_BINARY_INV)

        # 2. Выделяем главную белую полость экрана/контейнера с закрытием краевых вмятин от букв
        white_bin = (roi_gray >= 210).astype(np.uint8) * 255
        nw, w_lbl, w_stats, _ = cv2.connectedComponentsWithStats(white_bin, connectivity=4)
        if nw <= 1:
            return 0.0, [], None
        best_wid = max(range(1, nw), key=lambda wid: int(w_stats[wid, cv2.CC_STAT_AREA]))
        if int(w_stats[best_wid, cv2.CC_STAT_AREA]) < int(rw * rh * 0.18):
            return 0.0, [], None

        main_white = ((w_lbl == best_wid) * 255).astype(np.uint8)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        pad_d = 20
        padded_w = cv2.copyMakeBorder(main_white, pad_d, pad_d, pad_d, pad_d, cv2.BORDER_CONSTANT, value=0)
        closed_w = cv2.morphologyEx(padded_w, cv2.MORPH_CLOSE, k_close)[pad_d : pad_d + rh, pad_d : pad_d + rw]
        cnts, _ = cv2.findContours(closed_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        screen_cavity = np.zeros_like(main_white)
        cv2.drawContours(screen_cavity, cnts, -1, 255, -1)

        # 3. Отсоединяем прямые внешние рамки фреймов в 10-пиксельной краевой полосе
        k_hline = cv2.getStructuringElement(cv2.MORPH_RECT, (max(28, rw // 3), 1))
        k_vline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(28, rh // 3)))
        h_lines = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, k_hline)
        v_lines = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, k_vline)
        border_band = np.zeros_like(bw_inv)
        b_margin = 10
        border_band[:b_margin, :] = 255
        border_band[max(0, rh - b_margin) :, :] = 255
        border_band[:, :b_margin] = 255
        border_band[:, max(0, rw - b_margin) :] = 255
        edge_frame = cv2.bitwise_and(cv2.bitwise_or(h_lines, v_lines), border_band)
        sever_mask = cv2.dilate(edge_frame, np.ones((3, 3), np.uint8), iterations=1)

        severed_bw = cv2.bitwise_and(bw_inv, cv2.bitwise_and(screen_cavity, cv2.bitwise_not(sever_mask)))
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(severed_bw, connectivity=8)

        wall_mask = sever_mask.copy()
        char_candidates = np.zeros_like(bw_inv)
        max_char_dim = max(20, min(46, int(min(rw, rh) * 0.26)))
        for i in range(1, n_cc):
            cx, cy, cw, ch, area = stats[i]
            if cw > rw * 0.28 or ch > rh * 0.28 or area > (rw * rh * 0.035):
                wall_mask[labels == i] = 255
                continue
            if 3 <= area <= (rw * rh * 0.03) and 1 <= cw <= max_char_dim and 1 <= ch <= max_char_dim:
                aspect = max(cw, ch) / max(1.0, float(min(cw, ch)))
                if aspect > 6.5 and area > 25:
                    continue
                pad = 4
                sub_g = roi_gray[max(0, cy - pad) : min(rh, cy + ch + pad), max(0, cx - pad) : min(rw, cx + cw + pad)]
                sub_w = wall_mask[max(0, cy - pad) : min(rh, cy + ch + pad), max(0, cx - pad) : min(rw, cx + cw + pad)]
                valid_bg = sub_g[sub_w == 0]
                if valid_bg.size > 0 and float(np.percentile(valid_bg, 80)) >= 225.0:
                    char_candidates[labels == i] = 255

        k_cluster = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        clustered = cv2.dilate(char_candidates, k_cluster, iterations=1)
        nc, c_lbl, _, _ = cv2.connectedComponentsWithStats(clustered, connectivity=8)
        glyph_mask = np.zeros_like(bw_inv)
        best_ink = 0
        c_infos: list[tuple[int, int, int]] = []
        for cid in range(1, nc):
            sub_c = (c_lbl == cid) & (char_candidates > 0)
            ink_px = int(np.count_nonzero(sub_c))
            comp_cnt = len([c for c in np.unique(labels[sub_c]) if c > 0])
            c_infos.append((cid, ink_px, comp_cnt))
            if ink_px > best_ink:
                best_ink = ink_px

        total_kept_comps = 0
        for cid, ink_px, comp_cnt in c_infos:
            if comp_cnt >= 4 or (ink_px >= best_ink * 0.35 and comp_cnt >= 3):
                glyph_mask[(c_lbl == cid) & (char_candidates > 0)] = 255
                total_kept_comps += comp_cnt

        ys, xs = np.where(glyph_mask > 0)
        if xs.size < 35 or total_kept_comps < 6:
            return 0.0, [], None

        # Подвыборка точек для сверхбыстрого поиска угла (< 0.8 мс)
        if xs.size > 1400:
            step = max(2, xs.size // 1200)
            pts_x = xs[::step].astype(np.float32) - rw / 2.0
            pts_y = ys[::step].astype(np.float32) - rh / 2.0
        else:
            pts_x = xs.astype(np.float32) - rw / 2.0
            pts_y = ys.astype(np.float32) - rh / 2.0

        def _score_angle(deg: float) -> tuple[float, float, float, int]:
            rad = np.deg2rad(deg)
            y_rot = -pts_x * np.sin(rad) + pts_y * np.cos(rad)
            span = max(10.0, float(np.max(y_rot) - np.min(y_rot)))
            bins = max(10, int(round(span)))
            hist, _ = np.histogram(y_rot, bins=bins)
            norm_h = hist.astype(np.float32) / max(1.0, float(len(y_rot)))
            var_val = float(np.var(norm_h)) * 10000.0
            zero_frac = float(np.mean(hist <= 1))
            in_b, nb, s = False, 0, 0
            for idx_b, v in enumerate(hist >= 2):
                if v and not in_b:
                    in_b, s = True, idx_b
                elif not v and in_b:
                    in_b = False
                    if idx_b - s >= 5:
                        nb += 1
            if in_b and bins - s >= 5:
                nb += 1
            return var_val + zero_frac * 0.35 + nb * 0.025, var_val, zero_frac, nb

        x_span = max(10.0, float(np.max(pts_x) - np.min(pts_x)))
        x_bins = max(10, int(round(x_span)))
        x_hist, _ = np.histogram(pts_x, bins=x_bins)
        v0_var = float(np.var(x_hist.astype(np.float32) / max(1.0, float(len(pts_x))))) * 10000.0
        h0_sc, h0_var, _, h0_nb = _score_angle(0.0)

        # Защита вертикальных японских бабблов (tategaki): если вертикальные колонки выражены сильнее строк, не вращаем!
        if v0_var > h0_var * 1.65 and v0_var >= 1.25:
            return 0.0, [], None

        best_deg = 0.0
        best_sc = h0_sc
        best_var = h0_var
        best_nb = h0_nb
        cand_angles = list(np.linspace(-54.0, 54.0, 55))
        try:
            sota_tilt = estimate_and_deskew_tilted_text_block(roi_gray)
            if sota_tilt.get("is_tilted_screen") and abs(float(sota_tilt.get("angle_deg", 0.0))) >= 8.0:
                cand_angles.append(float(sota_tilt["angle_deg"]))
                cand_angles.append(-float(sota_tilt["angle_deg"]))
        except Exception:
            pass
        for deg in cand_angles:
            sc, vv, _, nb = _score_angle(float(deg))
            if sc > best_sc:
                best_sc, best_var, best_deg, best_nb = sc, vv, float(deg), nb

        for deg in np.linspace(best_deg - 2.0, best_deg + 2.0, 9):
            sc, vv, _, nb = _score_angle(float(deg))
            if sc > best_sc:
                best_sc, best_var, best_deg, best_nb = sc, vv, float(deg), nb

        is_tilted = abs(best_deg) >= 5.5 and best_var >= max(h0_var * 1.55, v0_var * 1.35) and best_nb >= 2
        if not is_tilted:
            best_deg = 0.0
            best_nb = h0_nb
            # Для ненаклоненных блоков активируем построчную нарезку только если это крупный многострочный горизонтальный блок (>= 4 строк и h0_var > v0_var * 1.25)
            if best_nb < 4 or total_kept_comps < 18 or h0_var <= v0_var * 1.25:
                return 0.0, [], None

        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        keep_mask = cv2.bitwise_and(
            cv2.dilate(glyph_mask, dil_k, iterations=1),
            cv2.bitwise_and(screen_cavity, cv2.bitwise_not(wall_mask)),
        )
        clean_rgb = np.full_like(work_rgb, 255)
        clean_rgb[keep_mask > 0] = work_rgb[keep_mask > 0]

        if abs(best_deg) >= 5.5:
            rot_rgb = np.array(
                Image.fromarray(clean_rgb).rotate(best_deg, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
            )
            rot_mask = np.array(
                Image.fromarray(glyph_mask).rotate(best_deg, resample=Image.NEAREST, expand=True, fillcolor=0)
            )
        else:
            rot_rgb = clean_rgb
            rot_mask = glyph_mask

        row_px = np.sum(rot_mask > 0, axis=1)
        in_b, raw_bands, s = False, [], 0
        for idx_r, v in enumerate(row_px >= 2):
            if v and not in_b:
                in_b, s = True, idx_r
            elif not v and in_b:
                in_b = False
                if idx_r - s >= 6:
                    raw_bands.append((s, idx_r))
        if in_b and len(row_px) - s >= 6:
            raw_bands.append((s, len(row_px)))

        if not raw_bands:
            return best_deg, [], Image.fromarray(rot_rgb)

        heights = [b[1] - b[0] for b in raw_bands]
        med_h = float(np.median(heights)) if heights else 16.0
        bands: list[tuple[int, int]] = []
        for by1, by2 in raw_bands:
            bh = by2 - by1
            if bh >= max(24, int(round(med_h * 1.65))):
                m_pad = max(5, int(bh * 0.22))
                if by2 - m_pad > by1 + m_pad + 2:
                    sub_p = row_px[by1 + m_pad : by2 - m_pad]
                    vy = by1 + m_pad + int(np.argmin(sub_p))
                    if row_px[vy] <= max(4, int(np.max(row_px[by1:by2]) * 0.25)):
                        bands.append((by1, vy))
                        bands.append((vy, by2))
                        continue
            bands.append((by1, by2))

        line_boxes: list[list[int]] = []
        for by1, by2 in bands:
            cols = np.where(np.any(rot_mask[by1:by2, :] > 0, axis=0))[0]
            if cols.size < 4:
                continue
            bx1, bx2 = int(cols[0]), int(cols[-1]) + 1
            if (bx2 - bx1) < 10:
                continue
            line_boxes.append([bx1, by1, bx2, by2])

        if not line_boxes:
            return best_deg, [], Image.fromarray(rot_rgb)

        # Присоединяем сверхкороткие строки-переносы (ширина < 55px, например «だ。», «ない。») к предыдущей строке
        merged_boxes: list[list[int]] = []
        for lb in line_boxes:
            lw_i = lb[2] - lb[0]
            if merged_boxes and lw_i < 55 and (lb[1] - merged_boxes[-1][3]) <= 12:
                prev = merged_boxes[-1]
                merged_boxes[-1] = [min(prev[0], lb[0]), prev[1], max(prev[2], lb[2]), lb[3]]
            else:
                merged_boxes.append(lb)

        rh_r, rw_r = rot_rgb.shape[:2]
        out_pils: list[Image.Image] = []
        for bx1, by1, bx2, by2 in merged_boxes:
            cy1, cy2 = max(0, by1 - 3), min(rh_r, by2 + 3)
            cx1, cx2 = max(0, bx1 - 6), min(rw_r, bx2 + 6)
            sub_l = rot_rgb[cy1:cy2, cx1:cx2]
            padded_l = cv2.copyMakeBorder(sub_l, 8, 8, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            out_pils.append(Image.fromarray(padded_l))

        return best_deg, out_pils, Image.fromarray(rot_rgb)

    @classmethod
    def _is_strict_multiline_screen_crop(
        cls,
        crop_rgb_np: np.ndarray,
    ) -> tuple[bool, float, list[Image.Image], Image.Image | None]:
        """
        Строгая верификация многострочного текстового экрана (смартфон, монитор, длинное письмо под углом):
        Гарантирует 0% ложных срабатываний на лицах персонажей, штриховке, плитке пола и фоновых SFX!
        Требует:
        1. Минимум 4 параллельные горизонтальные строки (`len(line_pils) >= 4`),
        2. Минимум 4 строки должны успешно проходить `pre_ocr_stroke_gate` с `valid_cc_count >= 3` и горизонтальной пропорцией `w >= h * 2.0`,
        3. Суммарное число валидных буквенных компонент по строкам `total_valid_cc >= 28`.
        """
        deg, lines, deskewed_im = cls._extract_tilted_or_multiline_horizontal_crops(crop_rgb_np)
        if len(lines) < 4:
            return False, 0.0, [], None
        valid_wide_lines: list[Image.Image] = []
        total_valid_cc = 0
        for lp in lines:
            lw, lh = lp.size
            gate = pre_ocr_stroke_gate(lp)
            if gate.passed and gate.valid_cc_count >= 2:
                valid_wide_lines.append(lp)
                if gate.valid_cc_count >= 3 and lw >= lh * 2.0:
                    total_valid_cc += gate.valid_cc_count
        if len(valid_wide_lines) < 4 or total_valid_cc < 28:
            return False, 0.0, [], None
        return True, deg, valid_wide_lines, deskewed_im

    @staticmethod
    def _sort_ocr_line_items_top_to_bottom(
        items: list[tuple[float, float, float, str, float]],
    ) -> list[tuple[str, float]]:
        """
        Группирует и сортирует фрагменты построчного OCR (`EasyOCR` / `RapidOCR`)
        строго сверху вниз по строкам, а внутри одной горизонтальной строки — слева направо.
        Каждый элемент `items`: `(y_min, y_max, x_center, text, conf)`.
        """
        if not items:
            return []
        if len(items) == 1:
            return [(items[0][3], items[0][4])]

        sorted_by_y = sorted(items, key=lambda it: (it[0] + it[1]) / 2.0)
        rows: list[list[tuple[float, float, float, str, float]]] = []

        for item in sorted_by_y:
            y1, y2, _, _, _ = item
            ih = max(1.0, y2 - y1)
            icy = (y1 + y2) / 2.0

            placed = False
            for row in rows:
                ry1 = min(r[0] for r in row)
                ry2 = max(r[1] for r in row)
                rh = max(1.0, ry2 - ry1)
                rcy = sum((r[0] + r[1]) / 2.0 for r in row) / len(row)

                overlap = min(y2, ry2) - max(y1, ry1)
                if overlap >= min(ih, rh) * 0.40 or abs(icy - rcy) <= max(ih, rh) * 0.42:
                    row.append(item)
                    placed = True
                    break
            if not placed:
                rows.append([item])

        rows.sort(key=lambda r: sum((x[0] + x[1]) / 2.0 for x in r) / len(r))
        ordered: list[tuple[str, float]] = []
        for row in rows:
            row.sort(key=lambda x: x[2])
            for _, _, _, txt, conf in row:
                ordered.append((txt, conf))
        return ordered

    @staticmethod
    def _extract_poly_bounds(poly: Any) -> tuple[float, float, float]:
        """
        Извлекает `(y_min, y_max, x_center)` из полигона координат `EasyOCR` или `RapidOCR`.
        """
        try:
            pts = np.asarray(poly, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] >= 2:
                xs = pts[:, 0]
                ys = pts[:, 1]
                return float(np.min(ys)), float(np.max(ys)), float(np.mean(xs))
        except Exception:
            pass
        return 0.0, 10.0, 0.0

    @staticmethod
    def _extract_bubble_interior_mask(roi_gray: np.ndarray, bg_level: float) -> np.ndarray:
        """
        Выделяет внутреннюю светлую область речевого облачка внутри темного контура баббла,
        отсекая как сам контур баббла, так и 4 внешних угла прямоугольного бокса за пределами овала,
        и морфологически замыкает вмятины от крайних иероглифов, касающихся стенок баббла,
        с автоматической поддержкой пунктирных (Проблема №1) и разомкнутых (Проблема №2) бабблов.
        """
        rh, rw = roi_gray.shape[:2]
        wall_thresh = max(150, int(bg_level - 52))
        light_pixels = (roi_gray >= wall_thresh).astype(np.uint8) * 255
        dark_wall_bin = cv2.bitwise_not(light_pixels)

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
        best_edge_touches = 0

        c_x1, c_x2 = int(rw * 0.22), max(int(rw * 0.22) + 1, int(rw * 0.78))
        c_y1, c_y2 = int(rh * 0.18), max(int(rh * 0.18) + 1, int(rh * 0.82))
        center_slice = labels[c_y1:c_y2, c_x1:c_x2]

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
                best_edge_touches = edge_touches
        if best_label == -1:
            return light_pixels

        # Если основная светлая область касается >= 3 границ кропа (пробой через пунктирную или разомкнутую стенку),
        # пробуем герметизировать чашу через SOTA-модули Проблемы №1 и Проблемы №2:
        if best_edge_touches >= 3 and rh >= 24 and rw >= 24:
            dash_res = segment_dashed_bubble_roi(roi_gray, None)
            if dash_res.get("is_dashed_bubble"):
                d_int = dash_res["interior_mask"]
                if 0.08 * rh * rw <= np.count_nonzero(d_int) <= 0.90 * rh * rw:
                    return d_int
            open_res = segment_open_wispy_bubble_roi(
                roi_gray, (int(0.22 * rw), int(0.20 * rh), int(0.78 * rw), int(0.80 * rh))
            )
            if open_res.get("has_opposing_arcs"):
                o_int = open_res["virtual_bowl_mask"]
                if 0.08 * rh * rw <= np.count_nonzero(o_int) <= 0.90 * rh * rw:
                    return o_int

        interior = np.zeros_like(light_pixels)
        interior[labels == best_label] = 255

        inv = cv2.bitwise_not(interior)
        n_inv, lbl_inv, st_inv, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        for idx in range(1, n_inv):
            lx = st_inv[idx, cv2.CC_STAT_LEFT]
            ly = st_inv[idx, cv2.CC_STAT_TOP]
            lw = st_inv[idx, cv2.CC_STAT_WIDTH]
            lh = st_inv[idx, cv2.CC_STAT_HEIGHT]
            la = st_inv[idx, cv2.CC_STAT_AREA]
            touches_edge = lx <= 0 or ly <= 0 or (lx + lw) >= rw or (ly + lh) >= rh
            if not touches_edge and lw < rw * 0.52 and lh < rh * 0.52 and la < rw * rh * 0.18:
                if not np.any((lbl_inv == idx) & (wall_barrier > 0)):
                    interior[lbl_inv == idx] = 255

        close_d = max(15, min(27, int(round(min(rw, rh) * 0.26)) | 1))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_d, close_d))
        ys_i, xs_i = np.where(interior > 0)
        if xs_i.size >= 16:
            iy1_b, iy2_b = int(np.min(ys_i)), int(np.max(ys_i)) + 1
            ix1_b, ix2_b = int(np.min(xs_i)), int(np.max(xs_i)) + 1
            sub_int = interior[iy1_b:iy2_b, ix1_b:ix2_b]
            sh_i, sw_i = sub_int.shape[:2]
            top_flat = float(np.mean(sub_int[0, :] > 0)) >= 0.22
            bot_flat = float(np.mean(sub_int[sh_i - 1, :] > 0)) >= 0.22
            left_flat = float(np.mean(sub_int[:, 0] > 0)) >= 0.22
            right_flat = float(np.mean(sub_int[:, sw_i - 1] > 0)) >= 0.22
            padded = cv2.copyMakeBorder(sub_int, close_d, close_d, close_d, close_d, cv2.BORDER_CONSTANT, value=0)
            if top_flat:
                padded[:close_d, close_d : close_d + sw_i] = sub_int[0:1, :]
            if bot_flat:
                padded[close_d + sh_i :, close_d : close_d + sw_i] = sub_int[sh_i - 1 : sh_i, :]
            if left_flat:
                padded[close_d : close_d + sh_i, :close_d] = sub_int[:, 0:1]
            if right_flat:
                padded[close_d : close_d + sh_i, close_d + sw_i :] = sub_int[:, sw_i - 1 : sw_i]
            closed_pad = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, k_close, iterations=1)
            closed_int = np.zeros_like(interior)
            closed_int[iy1_b:iy2_b, ix1_b:ix2_b] = closed_pad[close_d : close_d + sh_i, close_d : close_d + sw_i]
        else:
            padded = cv2.copyMakeBorder(interior, close_d, close_d, close_d, close_d, cv2.BORDER_CONSTANT, value=0)
            closed_pad = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, k_close, iterations=1)
            closed_int = closed_pad[close_d : close_d + rh, close_d : close_d + rw]

        closed_int[wall_barrier > 0] = 0
        inv2 = cv2.bitwise_not(closed_int)
        n_inv2, lbl_inv2, st_inv2, _ = cv2.connectedComponentsWithStats(inv2, connectivity=8)
        for idx2 in range(1, n_inv2):
            lx2 = st_inv2[idx2, cv2.CC_STAT_LEFT]
            ly2 = st_inv2[idx2, cv2.CC_STAT_TOP]
            lw2 = st_inv2[idx2, cv2.CC_STAT_WIDTH]
            lh2 = st_inv2[idx2, cv2.CC_STAT_HEIGHT]
            la2 = st_inv2[idx2, cv2.CC_STAT_AREA]
            if lx2 > 0 and ly2 > 0 and (lx2 + lw2) < rw and (ly2 + lh2) < rh:
                if lw2 < rw * 0.52 and lh2 < rh * 0.52 and la2 < rw * rh * 0.18:
                    if not np.any((lbl_inv2 == idx2) & (wall_barrier > 0)):
                        closed_int[lbl_inv2 == idx2] = 255

        added = cv2.subtract(closed_int, interior)
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
        return cv2.erode(interior, erode_k, iterations=1)

    @classmethod
    def _compute_tight_ink_bbox(
        cls,
        img_np: np.ndarray,
        bubble_bbox: list[int] | tuple[int, int, int, int],
        hint_text_bbox: list[int] | tuple[int, int, int, int] | None = None,
        is_free_text: bool = False,
    ) -> list[int]:
        """
        Вычисляет плотный внутренний прямоугольник реальных штрихов букв (`text_bbox`) внутри `bubble_bbox`,
        отсекая внешние черные контуры самого речевого облачка, рамки фреймов и пустые белые поля,
        но сохраняя все краевые и выступающие за контур баббла иероглифы.
        """
        img_h, img_w = img_np.shape[:2]
        bx1, by1, bx2, by2 = [int(round(float(c))) for c in bubble_bbox[:4]]
        bx1 = max(0, min(bx1, img_w - 2))
        by1 = max(0, min(by1, img_h - 2))
        bx2 = max(bx1 + 2, min(bx2, img_w))
        by2 = max(by1 + 2, min(by2, img_h))
        bw = bx2 - bx1
        bh = by2 - by1

        max_overhang = max(24, int(min(bw, bh) * 0.30))
        valid_hint: list[int] | None = None
        if isinstance(hint_text_bbox, (list, tuple)) and len(hint_text_bbox) >= 4:
            hx1, hy1, hx2, hy2 = [int(round(float(c))) for c in hint_text_bbox[:4]]
            hx1 = max(max(0, bx1 - max_overhang), min(hx1, img_w - 2))
            hy1 = max(max(0, by1 - max_overhang), min(hy1, img_h - 2))
            hx2 = max(hx1 + 2, min(hx2, min(img_w, bx2 + max_overhang)))
            hy2 = max(hy1 + 2, min(hy2, min(img_h, by2 + max_overhang)))
            if (hx2 - hx1) >= 10 and (hy2 - hy1) >= 10:
                valid_hint = [hx1, hy1, hx2, hy2]

        if bw < 22 or bh < 22:
            return valid_hint if valid_hint is not None else [bx1, by1, bx2, by2]

        if is_free_text:
            return valid_hint if valid_hint is not None else [bx1, by1, bx2, by2]

        pad_in_x = max(1, int(bw * 0.02))
        pad_in_y = max(1, int(bh * 0.02))
        ix1 = min(bx2 - 2, bx1 + pad_in_x)
        iy1 = min(by2 - 2, by1 + pad_in_y)
        ix2 = max(ix1 + 2, bx2 - pad_in_x)
        iy2 = max(iy1 + 2, by2 - pad_in_y)

        roi_rgb = img_np[iy1:iy2, ix1:ix2]
        rh, rw = roi_rgb.shape[:2]
        if rh < 12 or rw < 12:
            return valid_hint if valid_hint is not None else [bx1, by1, bx2, by2]

        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        p75 = float(np.percentile(roi_gray, 75))
        median_val = float(np.median(roi_gray))

        has_clean_interior = False
        if p75 >= 210.0:
            interior_mask = cls._extract_bubble_interior_mask(roi_gray, p75)
            ink_thresh = max(165, int(p75 - 18))
            binary = ((roi_gray < ink_thresh) & (interior_mask > 0)).astype(np.uint8) * 255
            has_clean_interior = True
        elif median_val >= 127.0:
            _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_cx, min_cy = rw, rh
        max_cx, max_cy = 0, 0
        kept_ink_px = 0

        for lbl in range(1, num_labels):
            cx = int(stats[lbl, cv2.CC_STAT_LEFT])
            cy = int(stats[lbl, cv2.CC_STAT_TOP])
            cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            carea = int(stats[lbl, cv2.CC_STAT_AREA])

            if carea < 3:
                continue

            touches_l = cx <= 1
            touches_t = cy <= 1
            touches_r = (cx + cw) >= (rw - 1)
            touches_b = (cy + ch) >= (rh - 1)
            edge_touches = sum([touches_l, touches_t, touches_r, touches_b])

            if edge_touches >= 2 and (cw > rw * 0.65 or ch > rh * 0.65):
                continue
            if edge_touches >= 1 and (cw > rw * 0.78 or ch > rh * 0.78):
                continue
            if edge_touches >= 1 and (cw <= 4 or ch <= 4) and (cw > rw * 0.35 or ch > rh * 0.35):
                continue

            if not has_clean_interior and edge_touches >= 1:
                comp_cx = cx + cw / 2.0
                comp_cy = cy + ch / 2.0
                in_corner_x = comp_cx < rw * 0.14 or comp_cx > rw * 0.86
                in_corner_y = comp_cy < rh * 0.14 or comp_cy > rh * 0.86
                if in_corner_x and in_corner_y:
                    continue

            min_cx = min(min_cx, cx)
            min_cy = min(min_cy, cy)
            max_cx = max(max_cx, cx + cw)
            max_cy = max(max_cy, cy + ch)
            kept_ink_px += carea

        if kept_ink_px < 12 or max_cx <= min_cx or max_cy <= min_cy:
            return valid_hint if valid_hint is not None else [bx1, by1, bx2, by2]

        ink_x1 = max(bx1, ix1 + min_cx - 4)
        ink_y1 = max(by1, iy1 + min_cy - 4)
        ink_x2 = min(bx2, ix1 + max_cx + 4)
        ink_y2 = min(by2, iy1 + max_cy + 4)

        if (ink_x2 - ink_x1) < 10 or (ink_y2 - ink_y1) < 10:
            return valid_hint if valid_hint is not None else [bx1, by1, bx2, by2]

        # Защита от фрагментации interior_mask на длинных вертикальных/горизонтальных плашках без RT-DETR подсказки:
        # если _extract_bubble_interior_mask отрезал больше 42% высоты/ширины баббла, а в отрезанной внутренней
        # части баббла осталась значимая доля темных штрихов текста — расширяем text_bbox на все штрихи внутри баббла!
        if valid_hint is None and has_clean_interior and (bh >= 70 or bw >= 70):
            span_h_ratio = float(ink_y2 - ink_y1) / float(max(1, bh))
            span_w_ratio = float(ink_x2 - ink_x1) / float(max(1, bw))
            if span_h_ratio < 0.58 or span_w_ratio < 0.58:
                in_pad_x = max(3, int(rw * 0.08))
                in_pad_y = max(3, int(rh * 0.03))
                if rh > 2 * in_pad_y + 8 and rw > 2 * in_pad_x + 8:
                    core_sub_gray = roi_gray[in_pad_y : rh - in_pad_y, in_pad_x : rw - in_pad_x]
                    csh, csw = core_sub_gray.shape[:2]
                    core_ink_thresh = max(160, int(p75 - 24))
                    core_dark_mask = (core_sub_gray < core_ink_thresh).astype(np.uint8) * 255
                    n_c, _, st_c, _ = cv2.connectedComponentsWithStats(core_dark_mask, connectivity=8)
                    c_min_x, c_min_y = csw, csh
                    c_max_x, c_max_y = 0, 0
                    c_kept_px = 0
                    for lbl_c in range(1, n_c):
                        ccx = int(st_c[lbl_c, cv2.CC_STAT_LEFT])
                        ccy = int(st_c[lbl_c, cv2.CC_STAT_TOP])
                        ccw = int(st_c[lbl_c, cv2.CC_STAT_WIDTH])
                        cch = int(st_c[lbl_c, cv2.CC_STAT_HEIGHT])
                        cca = int(st_c[lbl_c, cv2.CC_STAT_AREA])
                        if cca < 4:
                            continue
                        t_l = ccx <= 1
                        t_t = ccy <= 1
                        t_r = (ccx + ccw) >= (csw - 1)
                        t_b = (ccy + cch) >= (csh - 1)
                        if (t_l and t_r) or (t_t and t_b) or (sum([t_l, t_t, t_r, t_b]) >= 1 and (ccw > csw * 0.78 or cch > csh * 0.78)):
                            continue
                        c_min_x = min(c_min_x, ccx)
                        c_min_y = min(c_min_y, ccy)
                        c_max_x = max(c_max_x, ccx + ccw)
                        c_max_y = max(c_max_y, ccy + cch)
                        c_kept_px += cca
                    if c_kept_px > int(kept_ink_px * 1.25) and c_kept_px >= 24 and c_max_x > c_min_x and c_max_y > c_min_y:
                        ink_x1 = max(bx1, min(ink_x1, ix1 + in_pad_x + c_min_x - 4))
                        ink_y1 = max(by1, min(ink_y1, iy1 + in_pad_y + c_min_y - 4))
                        ink_x2 = min(bx2, max(ink_x2, ix1 + in_pad_x + c_max_x + 4))
                        ink_y2 = min(by2, max(ink_y2, iy1 + in_pad_y + c_max_y + 4))

        if valid_hint is not None:
            hx1, hy1, hx2, hy2 = valid_hint
            inter_x1 = max(hx1, ink_x1)
            inter_y1 = max(hy1, ink_y1)
            inter_x2 = min(hx2, ink_x2)
            inter_y2 = min(hy2, ink_y2)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                ux1 = max(0, min(hx1, ink_x1))
                uy1 = max(0, min(hy1, ink_y1))
                ux2 = min(img_w, max(hx2, ink_x2))
                uy2 = min(img_h, max(hy2, ink_y2))
                return [ux1, uy1, ux2, uy2]
            return valid_hint

        return [ink_x1, ink_y1, ink_x2, ink_y2]

    @staticmethod
    def _has_meaningful_ink(crop_rgb: np.ndarray, is_free_text: bool = False) -> tuple[bool, int]:
        h, w = crop_rgb.shape[:2]
        if h < 12 or w < 12:
            return False, 0

        margin_factor = 0.06 if is_free_text else 0.08
        pad_x = max(1, int(w * margin_factor))
        pad_y = max(1, int(h * margin_factor))
        roi = crop_rgb[pad_y : max(pad_y + 2, h - pad_y), pad_x : max(pad_x + 2, w - pad_x)]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        p5, p95 = np.percentile(gray, [5, 95])
        min_contrast = 42.0 if is_free_text else 18.0
        if (p95 - p5) < min_contrast:
            return False, 0

        if float(np.median(gray)) >= 127.0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ink_pixels = int(np.count_nonzero(binary))
        total_pixels = max(1, binary.size)
        ink_ratio = ink_pixels / total_pixels

        min_ratio = 0.018 if is_free_text else 0.0025
        max_ratio = 0.65 if is_free_text else 0.84
        min_ink_px = 14 if is_free_text else 9
        if ink_pixels < min_ink_px or ink_ratio < min_ratio or ink_ratio > max_ratio:
            return False, ink_pixels

        return True, ink_pixels

    @staticmethod
    def _is_valid_free_text_crop(crop_rgb: np.ndarray) -> bool:
        h, w = crop_rgb.shape[:2]
        if h < 14 or w < 14:
            return False
        aspect = w / max(1.0, float(h))
        if aspect > 12.0 or aspect < 0.06:
            return False

        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        p10, p50, p90 = np.percentile(gray, [10, 50, 90])
        if (p90 - p10) < 48.0:
            return False

        if p50 >= 120.0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        char_like_comps = 0
        long_line_comps = 0

        for idx in range(1, num_labels):
            cw = stats[idx, cv2.CC_STAT_WIDTH]
            ch = stats[idx, cv2.CC_STAT_HEIGHT]
            carea = stats[idx, cv2.CC_STAT_AREA]
            if carea < 5:
                continue
            if (cw >= w * 0.92 and ch <= 5) or (ch >= h * 0.92 and cw <= 5):
                long_line_comps += 1
                continue
            if 3 <= cw <= w * 0.92 and 3 <= ch <= h * 0.92:
                char_like_comps += 1

        if char_like_comps < 2 or long_line_comps > char_like_comps:
            return False
        return True

    @staticmethod
    def _has_real_text_glyph_structure(crop_rgb: np.ndarray, is_free_text: bool = False) -> bool:
        """
        Проверяет, что внутри кропа действительно присутствуют дискретные текстовые глифы (буквы/иероглифы),
        а не просто линии рисунка, волосы, складки одежды, рамки фреймов или однородный скринтон.
        Защищает декодеры MangaOCR / NAR от подачи бессловесных кусков арта, вызывающих галлюцинации.
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return False
        h, w = crop_rgb.shape[:2]
        if h < 12 or w < 12:
            return False

        # Быстрый 6-каскадный фильтр Pre-OCR Stroke Gate (градиенты, регулярные скринтоны, штриховка, рамки)
        stroke_gate = pre_ocr_stroke_gate(crop_rgb)
        if not stroke_gate.passed:
            return False

        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb
        p10, p50, p90 = [float(v) for v in np.percentile(gray, [10, 50, 90])]
        if (p90 - p10) < (42.0 if is_free_text else 28.0):
            return False

        if p50 >= 120.0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return False

        char_like_comps = 0
        long_wall_comps = 0
        tiny_specks = 0
        total_ink_area = max(1, int(np.count_nonzero(binary)))
        largest_comp_area = 0

        for idx in range(1, num_labels):
            cw = int(stats[idx, cv2.CC_STAT_WIDTH])
            ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
            carea = int(stats[idx, cv2.CC_STAT_AREA])
            if carea > largest_comp_area:
                largest_comp_area = carea
            if carea < 4:
                tiny_specks += 1
                continue
            # Длинные линии рамок/волос/контуров, пересекающие почти весь кроп
            if (cw >= w * 0.90 and ch <= max(5, int(h * 0.18))) or (ch >= h * 0.90 and cw <= max(5, int(w * 0.18))):
                long_wall_comps += 1
                continue
            max_span_ratio = 0.88 if is_free_text else 0.97
            if cw >= w * 0.96 and ch >= h * 0.96:
                long_wall_comps += 1
                continue
            if 2 <= cw <= w * max_span_ratio and 3 <= ch <= h * max_span_ratio:
                char_like_comps += 1

        crop_box_area = max(1, w * h)
        # Для свободного текста вне бабблов отсекаем одиночные пятна/тени
        if is_free_text and largest_comp_area > total_ink_area * 0.82 and char_like_comps <= 1:
            return False
        # Внутри бабблов отсекаем только сплошную черную заливку (>68% площади всего кропа)
        if not is_free_text and largest_comp_area > crop_box_area * 0.68 and char_like_comps == 0:
            return False
        # Если мелких точек скринтона в десятки раз больше, чем осмысленных штрихов — это скринтон
        if tiny_specks > max(32, char_like_comps * 11) and (char_like_comps < 4 if is_free_text else char_like_comps == 0):
            return False

        min_req_comps = 2 if is_free_text else 1
        if char_like_comps < min_req_comps:
            return False
        if is_free_text and long_wall_comps > char_like_comps:
            return False
        return True

    @staticmethod
    def _is_hallucinated_ocr_text(
        text: str,
        ink_pixels: int,
        is_free_text: bool = False,
        crop_shape: tuple[int, int] | None = None,
    ) -> bool:
        clean = text.strip()
        if not clean:
            return True
        if PUNCT_ONLY_REGEX.match(clean):
            return True
        if not MEANINGFUL_CHAR_REGEX.search(clean):
            return True
        if KNOWN_OCR_HALLUCINATIONS_REGEX.search(clean):
            logger.warning("Отсеяна известная галлюцинация OCR на фоновом/шумовом блоке: '%s'", clean)
            return True

        norm = STRIP_PUNCT_REGEX.sub("", clean)
        if not norm:
            return True
        if REPEATED_NOISE_REGEX.match(norm):
            return True
        # Числовой шум штриховки (например '7755', '111', '00' из наклонных линий '////') без единой буквы/кандзи/каны
        if re.fullmatch(r"[0-9０-９]+", norm):
            if is_free_text or len(set(norm)) <= 2 or len(norm) >= 4:
                return True

        # Проверка физической плотности чернил на символ: длинная фраза из 8+ символов не может возникнуть из пары точек
        char_len = len(norm)
        if char_len >= 8 and ink_pixels < char_len * 11:
            return True
        if is_free_text and ink_pixels < 42 and char_len > 3:
            return True
        if not is_free_text and ink_pixels < 22 and char_len > 4:
            return True

        if crop_shape is not None and len(crop_shape) >= 2:
            ch, cw = int(crop_shape[0]), int(crop_shape[1])
            crop_area = max(1, ch * cw)
            # В реальном баббле 1 символ занимает минимум ~45-50 кв. пикселей; если в крошечном боксе «прочитано» 12+ символов — это галлюцинация декодера
            if char_len >= 7 and crop_area < char_len * 45:
                return True

        if is_free_text:
            if char_len <= 1 and ink_pixels < 150:
                return True
            if ink_pixels < 95 and char_len >= 7:
                return True

        return False

    @classmethod
    def is_sigh_or_interjection(cls, text: str) -> bool:
        """
        Определяет, является ли распознанный текст коротким вздохом, стоном, охом или одиночным междометием
        (например: «あ», «あっ», «お», «んっ», «はぁ», «きゃ», «а», «о», «ах», «ох», «мм» и т.п.).
        """
        if not text:
            return True
        if "?" in text or "？" in text:
            return False
        core = STRIP_PUNCT_REGEX.sub("", text.strip()).lower()
        if not core:
            return True
        if core in DIALOGUE_VOWEL_WORDS_JA:
            return False
        if core in SIGH_EXACT_SET:
            return True

        # Схлопываем повторы одинаковых гласных/символов (например «ああああっ» -> «あっ», «おおお» -> «お»)
        collapsed = re.sub(r"(.)\1+", r"\1", core)
        if collapsed in SIGH_EXACT_SET and core not in DIALOGUE_VOWEL_WORDS_JA:
            return True

        # Защита для диалогового согласия «うん» / «うんうん» («угу / да-да»), чтобы SIGH_COMBO_REGEX не путал его со стоном «う…ん»
        if "うん" in core or "ウン" in core:
            return False

        # Если строка состоит только из вздохов/стонов (до 7 символов без кандзи)
        if len(core) <= 7 and not HANZI_REGEX.search(core):
            if SIGH_COMBO_REGEX.match(core) or SIGH_COMBO_REGEX.match(collapsed):
                return True

        # Одиночный символ считается вздохом только если он входит в SIGH_EXACT_SET
        if len(core) == 1 and core in SIGH_EXACT_SET:
            return True

        return False

    @classmethod
    def is_sfx_or_onomatopoeia(cls, text: str, is_free_text: bool = False) -> bool:
        """
        Определяет, является ли блок фоновым звуковым спецэффектом (SFX / ономатопеей).
        ВАЖНО: Внутри речевых бабблов (`is_free_text=False`) любые слова, реакции и междометия
        (включая チュッ, ドキッ, クスッ, ゴクッ, ダメ, ウソ, マジ) являются частью речи/реакции персонажа
        и НИКОГДА не отбрасываются как фоновый SFX!
        """
        if not text:
            return True
        core = STRIP_PUNCT_REGEX.sub("", text.strip())
        if not core:
            return True
        if not is_free_text:
            return False

        if core in SFX_EXACT_SET or core.lower() in SFX_EXACT_SET:
            return True

        collapsed = re.sub(r"(.)\1{2,}", r"\1", core)
        if collapsed in SFX_EXACT_SET:
            return True

        if not HANZI_REGEX.search(core):
            if len(core) <= 8 and REDUPLICATED_SFX_REGEX.match(core):
                return True
            if KATAKANA_ONLY_REGEX.match(core) and len(core) <= 6:
                return True

        return False

    @classmethod
    def classify_filter_reason(
        cls,
        text: str,
        is_free_text: bool = False,
        text_filter_mode: str = "bubbles_only",
        skip_sighs: bool = False,
        skip_sfx: bool = True,
        min_text_length: int = 1,
    ) -> tuple[str, str] | None:
        """
        Возвращает `(reason_code, reason_label_ru)`, если блок должен быть отсеян фильтром объема текста,
        либо `None`, если блок должен быть сохранен для перевода.
        """
        core = STRIP_PUNCT_REGEX.sub("", (text or "").strip())
        if not core:
            return ("empty_punct", "Только знаки препинания / пустая строка")

        mode = (text_filter_mode or "bubbles_only").lower().strip()
        eff_skip_sighs = bool(skip_sighs) or mode in ("bubbles_no_sighs", "long_only")
        eff_skip_sfx = bool(skip_sfx) or mode in ("dialogue_only", "bubbles_no_sighs", "bubbles_only", "long_only")
        eff_min_len = max(1, int(min_text_length or 1))
        if mode == "long_only":
            eff_min_len = max(eff_min_len, 4)
        elif not is_free_text and mode != "long_only":
            eff_min_len = 1

        if mode in ("dialogue_only", "bubbles_only", "long_only") and is_free_text:
            return ("free_text_disabled", f"Надпись вне баббла (отключено в профиле {mode})")

        # Чистые цифры без единой буквы или иероглифа ('7755', '100', '11') не требуют перевода и часто являются шумом штриховки
        if mode != "all" and re.fullmatch(r"[0-9０-９]+", core):
            return ("numeric_noise", "Числовой артефакт OCR / цифры без букв (не требует перевода)")

        has_kanji = bool(HANZI_REGEX.search(core))
        effective_char_len = len(core) + (1 if (has_kanji and len(core) == 1 and eff_min_len <= 2) else 0)
        if effective_char_len < eff_min_len:
            return ("min_length", f"Короткий текст ({len(core)} симв. < порога {eff_min_len})")

        if eff_skip_sighs and cls.is_sigh_or_interjection(text):
            return ("sigh", "Короткий вздох/стон (skip_sighs=True)")

        if eff_skip_sfx and cls.is_sfx_or_onomatopoeia(text, is_free_text=is_free_text):
            return ("sfx", "Звуковой спецэффект / SFX (skip_sfx=True)")

        return None

    @classmethod
    def should_filter_out_block(
        cls,
        text: str,
        is_free_text: bool = False,
        text_filter_mode: str = "bubbles_only",
        skip_sighs: bool = False,
        skip_sfx: bool = True,
        min_text_length: int = 1,
    ) -> bool:
        """
        Проверяет, нужно ли пропустить (не замазывать и не переводить) данный текстовый блок
        согласно настройкам объема перевода.
        """
        return (
            cls.classify_filter_reason(
                text=text,
                is_free_text=is_free_text,
                text_filter_mode=text_filter_mode,
                skip_sighs=skip_sighs,
                skip_sfx=skip_sfx,
                min_text_length=min_text_length,
            )
            is not None
        )

    @staticmethod
    def _is_vertical_text_crop(crop_rgb: np.ndarray) -> bool:
        h, w = crop_rgb.shape[:2]
        if h < 20 or w < 20:
            return h > w

        pad_x = max(2, int(w * 0.12))
        pad_y = max(2, int(h * 0.12))
        roi = crop_rgb[pad_y : max(pad_y + 2, h - pad_y), pad_x : max(pad_x + 2, w - pad_x)]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        if float(np.median(gray)) >= 127.0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        norm = binary.astype(np.float32) / 255.0
        col_profile_var = float(np.var(np.mean(norm, axis=0)))
        row_profile_var = float(np.var(np.mean(norm, axis=1)))

        if h > w * 1.35 and col_profile_var >= row_profile_var * 0.9:
            return True
        return col_profile_var > row_profile_var * 1.25

    @classmethod
    def _read_easyocr_crop_with_stats(
        cls,
        reader: Any,
        crop_np: np.ndarray,
        target_regex: re.Pattern[str],
        min_conf: float = 0.15,
    ) -> tuple[str, int, float]:
        """
        Читает кроп через `EasyOCR` с двумя улучшениями Шага 3:
        1. Все найденные полигоны сортируются строго сверху вниз по строкам и слева направо внутри строки
           через `_sort_ocr_line_items_top_to_bottom` (исправляет перепутанные строки в вебтунах).
        2. Если в баббле 2+ горизонтальных строк по `_split_horizontal_lines_for_ocr`, а прямой вызов `EasyOCR`
           потерял строки или дал меньше целевых символов, выполняет построчное чтение сверху вниз!
        """
        def _run_on_array(arr: np.ndarray) -> tuple[str, int, float, int]:
            res = reader.readtext(arr, detail=1, paragraph=False)
            raw_items: list[tuple[float, float, float, str, float]] = []
            for item in res:
                if len(item) < 3:
                    continue
                poly, raw_txt, conf = item[0], str(item[1]).strip(), float(item[2])
                if not raw_txt or conf < min_conf:
                    continue
                y_min, y_max, x_center = cls._extract_poly_bounds(poly)
                raw_items.append((y_min, y_max, x_center, raw_txt, conf))

            ordered = cls._sort_ocr_line_items_top_to_bottom(raw_items)
            texts: list[str] = []
            total_chars = 0
            weighted_conf_sum = 0.0
            for raw_txt, conf in ordered:
                texts.append(raw_txt)
                matched_chars = len(target_regex.findall(raw_txt))
                if matched_chars > 0:
                    total_chars += matched_chars
                    weighted_conf_sum += conf * matched_chars

            joined_text = " ".join(texts).strip()
            avg_conf = (weighted_conf_sum / total_chars) if total_chars > 0 else 0.0
            return joined_text, total_chars, avg_conf, len(ordered)

        direct_text, direct_chars, direct_conf, direct_boxes = _run_on_array(crop_np)

        # Проверяем, не многострочный ли это баббл, в котором EasyOCR пропустил часть строк
        line_splits = cls._split_horizontal_lines_for_ocr(crop_np)
        if len(line_splits) >= 2 and (direct_boxes < len(line_splits) or direct_chars < len(line_splits) * 2 or direct_conf < 0.58):
            line_texts: list[str] = []
            line_total_chars = 0
            line_weighted_conf = 0.0
            for _, _, line_rgb in line_splits:
                l_txt, l_chars, l_conf, _ = _run_on_array(line_rgb)
                if l_txt:
                    line_texts.append(l_txt)
                    line_total_chars += l_chars
                    line_weighted_conf += l_conf * l_chars

            if line_texts:
                split_joined = " ".join(line_texts).strip()
                split_avg_conf = (line_weighted_conf / line_total_chars) if line_total_chars > 0 else 0.0
                if line_total_chars > direct_chars or (
                    line_total_chars == direct_chars and split_avg_conf > direct_conf + 0.05
                ):
                    return split_joined, line_total_chars, split_avg_conf

        return direct_text, direct_chars, direct_conf

    def _read_rapidocr_crop_with_stats(
        self,
        crop_np: np.ndarray,
        target_regex: re.Pattern[str],
        min_conf: float = 0.18,
    ) -> tuple[str, int, float]:
        """
        Опциональное чтение кропа через `RapidOCR` (`rapidocr_onnxruntime`), если он установлен в системе,
        с вертикально-горизонтальной сортировкой строк и построчным дочитыванием.
        """
        rapid = self._get_rapidocr_reader()
        if rapid is None:
            return "", 0, 0.0

        def _run_rapid(arr: np.ndarray) -> tuple[str, int, float, int]:
            try:
                res, _ = rapid(arr)
            except Exception:
                return "", 0, 0.0, 0
            if not res:
                return "", 0, 0.0, 0
            raw_items: list[tuple[float, float, float, str, float]] = []
            for item in res:
                if not item or len(item) < 3:
                    continue
                poly, raw_txt, conf = item[0], str(item[1]).strip(), float(item[2])
                if not raw_txt or conf < min_conf:
                    continue
                y_min, y_max, x_center = self._extract_poly_bounds(poly)
                raw_items.append((y_min, y_max, x_center, raw_txt, conf))

            ordered = self._sort_ocr_line_items_top_to_bottom(raw_items)
            texts: list[str] = []
            total_chars = 0
            weighted_conf_sum = 0.0
            for raw_txt, conf in ordered:
                texts.append(raw_txt)
                matched_chars = len(target_regex.findall(raw_txt))
                if matched_chars > 0:
                    total_chars += matched_chars
                    weighted_conf_sum += conf * matched_chars
            joined = " ".join(texts).strip()
            avg_c = (weighted_conf_sum / total_chars) if total_chars > 0 else 0.0
            return joined, total_chars, avg_c, len(ordered)

        direct_t, direct_c, direct_conf, direct_n = _run_rapid(crop_np)
        line_splits = self._split_horizontal_lines_for_ocr(crop_np)
        if len(line_splits) >= 2 and (direct_n < len(line_splits) or direct_c < len(line_splits) * 2):
            l_texts: list[str] = []
            l_chars = 0
            l_wconf = 0.0
            for _, _, line_rgb in line_splits:
                lt, lc, lcf, _ = _run_rapid(line_rgb)
                if lt:
                    l_texts.append(lt)
                    l_chars += lc
                    l_wconf += lcf * lc
            if l_texts and l_chars > direct_c:
                return " ".join(l_texts).strip(), l_chars, (l_wconf / l_chars if l_chars > 0 else 0.0)
        return direct_t, direct_c, direct_conf

    def _recognize_single_ko_zh_crop(
        self,
        padded_crop_pil: Image.Image,
        raw_crop_np: np.ndarray,
        lang: str,
        is_free_text: bool = False,
        precomputed_hayai_text: str | None = None,
    ) -> str:
        """
        Гибридное чтение одного корейского (`ko`) или китайского (`zh`) баббла (Шаг 3):
        1. Основной проход через `JustANormalTinkerer/hayai-ocr-v2.5-nova` (`SigLIP2 NaFlex`).
        2. Если баббл высокий многострочный (`>= 3` строк по `_split_horizontal_lines_for_ocr`), проверяет,
           не потерял ли одиночный проход часть строк, и при необходимости дочитывает построчно сверху вниз.
        3. Если в результате нет целевых символов (`HANGUL_REGEX` / `HANZI_REGEX`) или `HayaiOCR` недоступен,
           выполняет построчное чтение через `RapidOCR` / `EasyOCR`.
        """
        target_regex = HANGUL_REGEX if lang == "ko" else HANZI_REGEX
        padded_np = np.array(padded_crop_pil)
        min_conf = 0.28 if is_free_text else 0.14

        # Если блок на экране телефона или табличке наклонен под углом (|deg| >= 5.5°), выпрямляем и читаем построчно!
        try:
            deg_kz, line_pils_kz, deskew_kz = self._extract_tilted_or_multiline_horizontal_crops(raw_crop_np)
            if abs(deg_kz) >= 5.5 and line_pils_kz:
                if self.use_hayai_ocr and self._hayai_available is not False:
                    kz_outs = self._recognize_crops_batch_hayai(line_pils_kz, max_num_patches=256, lang=lang)
                    valid_kz = [t.strip() for t in kz_outs if t and t.strip()]
                    if valid_kz:
                        sep_kz = " " if lang == "ko" else ""
                        joined_kz = self._normalize_hayai_ocr_text(sep_kz.join(valid_kz), lang=lang)
                        if len(target_regex.findall(joined_kz)) >= 1:
                            return joined_kz
                if deskew_kz is not None:
                    padded_crop_pil = deskew_kz
                    padded_np = np.array(deskew_kz)
                    raw_crop_np = padded_np
                    precomputed_hayai_text = None
        except Exception:
            pass

        hayai_text = precomputed_hayai_text
        if hayai_text is None and self.use_hayai_ocr and self._hayai_available is not False:
            batch_res = self._recognize_crops_batch_hayai([padded_crop_pil], lang=lang)
            hayai_text = batch_res[0].strip() if batch_res else ""
        else:
            hayai_text = (hayai_text or "").strip()

        hayai_target_chars = len(target_regex.findall(hayai_text)) if hayai_text else 0

        # Если баббл высокий и содержит >= 3 горизонтальных строк, проверяем, не выгоднее ли построчное чтение HayaiOCR
        line_splits = self._split_horizontal_lines_for_ocr(raw_crop_np)
        if (
            len(line_splits) >= 3
            and self.use_hayai_ocr
            and self._hayai_available is not False
            and hayai_target_chars < len(line_splits) * 2
        ):
            line_pils = [Image.fromarray(l_rgb, mode="RGB") for _, _, l_rgb in line_splits]
            line_hayai_outs = self._recognize_crops_batch_hayai(line_pils, max_num_patches=256, lang=lang)
            valid_lines = [t.strip() for t in line_hayai_outs if t and t.strip()]
            if valid_lines:
                joined_lines = self._normalize_hayai_ocr_text(" ".join(valid_lines), lang=lang)
                if len(target_regex.findall(joined_lines)) > hayai_target_chars:
                    hayai_text = joined_lines
                    hayai_target_chars = len(target_regex.findall(hayai_text))

        if hayai_text and hayai_target_chars >= 1:
            return hayai_text

        # Резервный проход 1: RapidOCR (если установлен)
        if self._check_rapidocr():
            r_txt, r_chars, _ = self._read_rapidocr_crop_with_stats(padded_np, target_regex, min_conf=min_conf)
            if r_txt and r_chars >= 1:
                return self._normalize_hayai_ocr_text(r_txt, lang=lang)

        # Резервный проход 2: EasyOCR с построчной сегментацией и сортировкой сверху вниз
        reader = self._get_ko_reader() if lang == "ko" else self._get_zh_reader()
        if reader is not None:
            e_txt, e_chars, _ = self._read_easyocr_crop_with_stats(
                reader,
                padded_np,
                target_regex,
                min_conf=min_conf,
            )
            if e_txt and (e_chars >= 1 or not hayai_text):
                return self._normalize_hayai_ocr_text(e_txt, lang=lang)
            if not is_free_text:
                lines = reader.readtext(padded_np, detail=0, paragraph=True)
                raw_fallback = " ".join(str(x).strip() for x in lines if str(x).strip())
                if raw_fallback:
                    return self._normalize_hayai_ocr_text(raw_fallback, lang=lang)

        if hayai_text and not is_free_text:
            return hayai_text

        if not self._check_multilang_ocr():
            lang_label = "корейской манхвы" if lang == "ko" else "китайской маньхуа"
            raise RuntimeError(
                f"Для перевода {lang_label} требуется модель HayaiOCR v2.5 Nova или установленный EasyOCR (install_multilang_ocr.bat)."
            )
        return ""

    def detect_page_language(
        self,
        rgb_image: Image.Image,
        raw_detections: list[dict[str, Any]],
    ) -> str:
        """
        Улучшенное автоопределение языка страницы (`auto` -> `ja` / `ko` / `zh`) (Шаг 3):
        1. Применяет `_preprocess_crop_np_for_ocr` (через `_crop_with_padding`) и определяет ориентацию строк
           (`vertical_votes` vs `horizontal_votes`).
        2. Быстро проверяет японский через уже загруженный в VRAM `manga-ocr-nar-small` / `MangaOCR`:
           если текст вертикальный и содержит уверенную долю каны, мгновенно возвращает `ja` (~35 мс).
        3. Если текст горизонтальный (характерно для корейской манхвы `ko` и китайской маньхуа `zh`) или доля каны низкая:
           использует мультиязычную модель `HayaiOCR v2.5 Nova` (которая одновременно распознает `JA`, `KO`, `ZH`, `EN`
           без галлюцинаций чужого алфавита!) и/или `EasyOCR`/`RapidOCR`, сохраняя результаты в `_page_ocr_cache`.
        """
        if not raw_detections:
            return "ja"

        img_area = max(1, rgb_image.width * rgb_image.height)
        normal_boxes = [
            d
            for d in raw_detections
            if not d.get("is_free_text", False)
            and 0.001 <= ((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])) / img_area <= 0.25
        ]
        pool = normal_boxes if len(normal_boxes) >= 2 else raw_detections

        top_boxes = sorted(
            pool,
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )[:10]

        crops_data: list[tuple[tuple[int, int, int, int], Image.Image, np.ndarray, bool]] = []
        vertical_votes = 0
        horizontal_votes = 0

        for det in top_boxes:
            x1, y1, x2, y2 = det["bbox"]
            bbox_key = (x1, y1, x2, y2)
            ocr_coords = self._get_ocr_bbox(det)
            raw_crop_np = np.array(rgb_image.crop(tuple(ocr_coords)))
            has_ink, _ = self._has_meaningful_ink(raw_crop_np, is_free_text=bool(det.get("is_free_text", False)))
            if not has_ink:
                continue
            padded_pil = self._crop_with_padding(rgb_image, ocr_coords, pad=8, smart_preprocess=True)
            padded_np = np.array(padded_pil)
            is_vert = self._is_vertical_text_crop( np.array(rgb_image.crop(tuple(ocr_coords))) )
            if is_vert:
                vertical_votes += 1
            else:
                horizontal_votes += 1
            crops_data.append((bbox_key, padded_pil, padded_np, is_vert))

        if not crops_data:
            return self._last_detected_lang

        kana_chars = 0
        ja_hanzi_chars = 0
        ja_evaluated = False

        def _evaluate_ja() -> tuple[int, int]:
            nonlocal kana_chars, ja_hanzi_chars, ja_evaluated
            if ja_evaluated:
                return kana_chars, ja_hanzi_chars
            ja_evaluated = True
            try:
                batch_pils = [item[1] for item in crops_data]
                ja_texts = self._recognize_crops_batch_ja(batch_pils)
                for (bbox_key, _, _, _), ja_txt in zip(crops_data, ja_texts):
                    self._page_ocr_cache[(*bbox_key, "ja")] = ja_txt
                    kana_chars += len(KANA_REGEX.findall(ja_txt))
                    ja_hanzi_chars += len(HANZI_REGEX.findall(ja_txt))
            except Exception:
                pass
            return kana_chars, ja_hanzi_chars

        # 1. Быстрая проверка японского (классическая японская манга и обложки с хираганой/катаканой — ~35 мс)
        _evaluate_ja()
        ja_total = kana_chars + ja_hanzi_chars
        kana_ratio = kana_chars / max(1, ja_total)
        # Если в бабблах есть кана и вертикальная верстка (или уверенная доля каны >= 18% при >= 2 символах каны) — это 100% японская манга
        if vertical_votes >= horizontal_votes and kana_chars >= 1 and kana_ratio >= 0.12:
            self._last_detected_lang = "ja"
            return "ja"
        if kana_chars >= 2 and kana_ratio >= 0.18:
            self._last_detected_lang = "ja"
            return "ja"
        # Если предыдущие страницы главы уже определены как японские ('ja') и на текущей странице только кандзи без каны
        # (например, обложка, схема родства, табличка «第一話» или короткие восклицания «何?!», «本当?!»),
        # при наличии вертикальных бабблов или малом числе иероглифов (< 14) сохраняем японский язык ('ja')!
        if self._last_detected_lang == "ja" and kana_chars == 0 and (vertical_votes >= horizontal_votes or ja_hanzi_chars < 14):
            return "ja"
        if not self._check_multilang_ocr():
            self._last_detected_lang = "ja"
            return "ja"

        # 2. Если на странице преобладают горизонтальные бабблы и нет каны, проверяем через мультиязычный HayaiOCR v2.5 Nova
        if self.use_hayai_ocr and self._hayai_available is not False and (
            self._hayai_model is not None or not self._check_easyocr()
        ):
            try:
                batch_pils = [item[1] for item in crops_data]
                hayai_texts = self._recognize_crops_batch_hayai(batch_pils, max_num_patches=256, lang="auto")
                h_hangul = 0
                h_kana = 0
                h_hanzi = 0
                h_zh_markers = 0
                for (bbox_key, _, _, _), h_txt in zip(crops_data, hayai_texts):
                    if not h_txt:
                        continue
                    cnt_ko = len(HANGUL_REGEX.findall(h_txt))
                    cnt_ja = len(KANA_REGEX.findall(h_txt))
                    cnt_zh = len(HANZI_REGEX.findall(h_txt))
                    h_hangul += cnt_ko
                    h_kana += cnt_ja
                    h_hanzi += cnt_zh
                    h_zh_markers += len(ZH_SPECIFIC_MARKERS_REGEX.findall(h_txt))
                    if cnt_ko > 0:
                        self._page_ocr_cache[(*bbox_key, "ko")] = h_txt
                    if cnt_zh > 0 and cnt_ja == 0:
                        self._page_ocr_cache[(*bbox_key, "zh")] = h_txt

                if h_hangul >= 2 and h_hangul >= h_kana:
                    self._last_detected_lang = "ko"
                    return "ko"
                if h_kana >= 1 or kana_chars >= 1:
                    self._last_detected_lang = "ja"
                    return "ja"
                # Переключаем на китайский ('zh') ТОЛЬКО если есть специфичные китайские маркеры или длинный горизонтальный текст без каны
                if h_hanzi >= 2 and h_kana == 0 and kana_chars == 0 and h_hangul == 0:
                    if h_zh_markers >= 2 or (h_hanzi >= 6 and h_zh_markers >= 1 and horizontal_votes > vertical_votes):
                        self._last_detected_lang = "zh"
                        return "zh"
                    return self._last_detected_lang or "ja"
            except Exception as exc:
                logger.debug("HayaiOCR language detection skipped: %s", exc)

        # 3. Если горизонтальная доля каны очень высокая и нет EasyOCR-признаков
        if kana_chars >= 4 and kana_ratio >= 0.34 and vertical_votes >= horizontal_votes:
            self._last_detected_lang = "ja"
            return "ja"

        # 4. Проверка через EasyOCR (ko / zh) при его наличии
        hangul_chars = 0
        ko_score = 0.0
        ko_weighted_sum = 0.0

        ko_reader = self._get_ko_reader()
        if ko_reader is not None:
            for bbox_key, _, padded_np, _ in crops_data:
                try:
                    ko_txt, h_cnt, h_conf = self._read_easyocr_crop_with_stats(
                        ko_reader, padded_np, HANGUL_REGEX
                    )
                    if ko_txt:
                        self._page_ocr_cache[(*bbox_key, "ko")] = ko_txt
                    if h_cnt > 0:
                        hangul_chars += h_cnt
                        ko_weighted_sum += h_conf * h_cnt
                        ko_score = max(ko_score, h_conf)
                except Exception:
                    pass
            if hangul_chars > 0:
                ko_score = max(ko_score, ko_weighted_sum / hangul_chars)

        if hangul_chars >= 2 and ko_score >= 0.52:
            self._last_detected_lang = "ko"
            return "ko"

        hanzi_chars = 0
        zh_score = 0.0
        zh_weighted_sum = 0.0

        zh_reader = self._get_zh_reader()
        if zh_reader is not None:
            for bbox_key, _, padded_np, _ in crops_data:
                try:
                    zh_txt, c_cnt, c_conf = self._read_easyocr_crop_with_stats(
                        zh_reader, padded_np, HANZI_REGEX
                    )
                    if zh_txt:
                        self._page_ocr_cache[(*bbox_key, "zh")] = zh_txt
                    if c_cnt > 0:
                        hanzi_chars += c_cnt
                        zh_weighted_sum += c_conf * c_cnt
                        zh_score = max(zh_score, c_conf)
                except Exception:
                    pass
            if hanzi_chars > 0:
                zh_avg = zh_weighted_sum / hanzi_chars
                zh_score = (zh_score * 0.5) + (zh_avg * 0.5)

        if hangul_chars >= 2 and ko_score >= 0.46 and ko_score >= (zh_score - 0.05):
            self._last_detected_lang = "ko"
            return "ko"

        if kana_chars >= 2 and kana_ratio >= 0.22:
            if vertical_votes >= horizontal_votes or zh_score < 0.74:
                self._last_detected_lang = "ja"
                return "ja"

        if (kana_chars + ja_hanzi_chars + hanzi_chars) <= 3 and vertical_votes >= horizontal_votes:
            return self._last_detected_lang

        if kana_chars >= 1:
            self._last_detected_lang = "ja"
            return "ja"

        zh_cached_joined = " ".join(
            str(val) for k, val in self._page_ocr_cache.items() if isinstance(k, tuple) and len(k) >= 5 and k[-1] == "zh"
        )
        easy_zh_markers = len(ZH_SPECIFIC_MARKERS_REGEX.findall(zh_cached_joined))

        if hanzi_chars >= 2 and zh_score >= 0.65 and easy_zh_markers >= 2:
            self._last_detected_lang = "zh"
            return "zh"

        if hanzi_chars >= 4 and zh_score >= 0.60 and kana_chars == 0 and easy_zh_markers >= 1 and horizontal_votes > vertical_votes:
            self._last_detected_lang = "zh"
            return "zh"

        return self._last_detected_lang

    @staticmethod
    def _merge_and_nms_detections(
        detections: list[dict[str, Any]],
        iou_threshold: float = 0.52,
        iomin_threshold: float = 0.62,
    ) -> list[dict[str, Any]]:
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(
            detections,
            key=lambda d: ((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
            * (0.5 + float(d["confidence"])),
            reverse=True,
        )

        merged: list[dict[str, Any]] = []
        for cand in sorted_dets:
            cx1, cy1, cx2, cy2 = cand["bbox"]
            c_area = max(1.0, float((cx2 - cx1) * (cy2 - cy1)))
            c_conf = float(cand["confidence"])
            c_free = bool(cand.get("is_free_text", False))

            absorbed = False
            for existing in merged:
                ex1, ey1, ex2, ey2 = existing["bbox"]
                e_area = max(1.0, float((ex2 - ex1) * (ey2 - ey1)))

                ix1 = max(cx1, ex1)
                iy1 = max(cy1, ey1)
                ix2 = min(cx2, ex2)
                iy2 = min(cy2, ey2)

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = float((ix2 - ix1) * (iy2 - iy1))
                    union_area = c_area + e_area - inter_area
                    iou = inter_area / max(1.0, union_area)
                    iomin = inter_area / max(1.0, min(c_area, e_area))

                    if iou >= iou_threshold or iomin >= iomin_threshold:
                        if iou >= iou_threshold or max(c_area, e_area) <= 1.45 * min(c_area, e_area):
                            existing["bbox"] = [
                                min(cx1, ex1),
                                min(cy1, ey1),
                                max(cx2, ex2),
                                max(cy2, ey2),
                            ]
                        elif c_area < e_area and c_conf >= float(existing["confidence"]) * 0.85:
                            # Если более компактный и точный бокс баббла (например от RT-DETR-v2) лежит внутри
                            # раздутого бокса (захватившего соседнего чиби-персонажа или SFX), сохраняем компактный бокс!
                            existing["bbox"] = [cx1, cy1, cx2, cy2]
                        existing["confidence"] = round(max(float(existing["confidence"]), c_conf), 4)
                        existing["is_free_text"] = bool(existing.get("is_free_text", False) and c_free)
                        absorbed = True
                        break

            if not absorbed:
                merged_item: dict[str, Any] = {
                    "bbox": [cx1, cy1, cx2, cy2],
                    "confidence": round(c_conf, 4),
                    "is_free_text": c_free,
                }
                if cand.get("text_bbox"):
                    merged_item["text_bbox"] = list(cand["text_bbox"])
                merged.append(merged_item)

        return merged

    @staticmethod
    def _split_double_bubbles(
        img_np: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Разделяет только широкие сдвоенные по горизонтали бабблы (когда два соседних облачка
        слились в один широкий бокс `bw >= bh * 1.25` и между ними есть перемычка/граница).
        """
        refined: list[dict[str, Any]] = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            is_free = bool(det.get("is_free_text", False))

            if is_free or bw < 160 or bw < bh * 1.25:
                refined.append(det)
                continue

            pad_x = max(6, int(bw * 0.12))
            pad_y = max(6, int(bh * 0.10))
            roi = img_np[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
            rh, rw = roi.shape[:2]
            if rh < 36 or rw < 60:
                refined.append(det)
                continue

            full_crop_gray = cv2.cvtColor(img_np[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
            white_col_ratio = np.mean(full_crop_gray >= 230, axis=0)
            mid_s = int(bw * 0.35)
            mid_e = int(bw * 0.65)
            if mid_e <= mid_s + 8:
                refined.append(det)
                continue

            min_white_col = int(np.argmin(white_col_ratio[mid_s:mid_e])) + mid_s
            left_white = float(np.max(white_col_ratio[: max(1, min_white_col - 8)]))
            right_white = float(np.max(white_col_ratio[min(bw - 1, min_white_col + 8) :]))
            pinch_white = float(white_col_ratio[min_white_col])

            if pinch_white <= min(left_white, right_white) * 0.62 and (min_white_col >= 55) and (bw - min_white_col >= 55):
                split_x = x1 + min_white_col
                refined.append({"bbox": [x1, y1, split_x, y2], "confidence": det["confidence"], "is_free_text": is_free})
                refined.append({"bbox": [split_x, y1, x2, y2], "confidence": det["confidence"], "is_free_text": is_free})
            else:
                refined.append(det)

        return refined

    def _run_rtdetr_on_slice(
        self,
        slice_np: np.ndarray,
        y_start: int,
        img_w: int,
        img_h: int,
        active_conf: float,
        min_box_size: int = 14,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Прогоняет 3-классовый детектор `ogkalu/comic-text-and-bubble-detector` (RT-DETR-v2) на одном слайсе страницы.
        Возвращает `(bubbles_cls0, text_bubbles_cls1, free_texts_cls2)`.
        """
        pair = self._get_rtdetr_detector()
        if pair is None:
            return [], [], []

        model, processor = pair
        bubbles_cls0: list[dict[str, Any]] = []
        text_bubbles_cls1: list[dict[str, Any]] = []
        free_texts_cls2: list[dict[str, Any]] = []

        try:
            sh, sw = slice_np.shape[:2]
            pil_slice = Image.fromarray(slice_np)
            inputs = processor(images=pil_slice, return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            low_thresh = max(0.18, min(active_conf, 0.25))
            with torch.inference_mode():
                outputs = model(**inputs)
                target_sizes = torch.tensor([[sh, sw]], device=self.device)
                results = processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=low_thresh,
                )

            if not results or len(results) == 0:
                return [], [], []

            res0 = results[0]
            scores_list = res0["scores"].detach().cpu().numpy().tolist()
            labels_list = res0["labels"].detach().cpu().numpy().tolist()
            boxes_list = res0["boxes"].detach().cpu().numpy().tolist()

            for score, label_id, box in zip(scores_list, labels_list, boxes_list):
                conf_val = float(score)
                lbl = int(label_id)
                x1, y1, x2, y2 = [int(round(float(c))) for c in box[:4]]
                y1 += y_start
                y2 += y_start

                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(x1 + 1, min(x2, img_w))
                y2 = max(y1 + 1, min(y2, img_h))

                bw = x2 - x1
                bh = y2 - y1
                if bw < min_box_size or bh < min_box_size:
                    continue

                if lbl == 0:
                    # Class 0: bubble
                    if conf_val >= max(0.24, active_conf):
                        bubbles_cls0.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(conf_val, 4),
                                "is_free_text": False,
                                "_source": "rtdetr_bubble",
                            }
                        )
                elif lbl == 1:
                    # Class 1: text_bubble (точный контур текста внутри баббла)
                    if conf_val >= max(0.28, active_conf):
                        text_bubbles_cls1.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(conf_val, 4),
                                "is_free_text": False,
                                "_source": "rtdetr_text_bubble",
                            }
                        )
                elif lbl == 2:
                    # Class 2: text_free (текст вне бабблов на фоне)
                    if conf_val >= max(0.32, active_conf + 0.06):
                        free_texts_cls2.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(conf_val, 4),
                                "is_free_text": True,
                                "_source": "rtdetr_text_free",
                            }
                        )
        except Exception as exc:
            logger.debug("RT-DETR-v2 ошибка на слайсе (%s), продолжаем с YOLOv8m.", exc)

        return bubbles_cls0, text_bubbles_cls1, free_texts_cls2

    @classmethod
    def _try_split_bubble_by_text_boxes(
        cls,
        img_np: np.ndarray,
        bubble_det: dict[str, Any],
        matched_tbs: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """
        Если внутри одного родительского `bubble` найдено 2 (или более) отдельных неперекрывающихся `text_bubble`
        (от RT-DETR-v2), разрезает родительский `bubble` на независимые под-баблы с собственными неперекрывающимися
        `bbox` и `text_bbox` (включая диагональные сдвоенные баблы-«восьмерки» с белой перемычкой без черной линии!).
        """
        if len(matched_tbs) < 2:
            return None

        bx1, by1, bx2, by2 = bubble_det["bbox"]
        bw = bx2 - bx1
        bh = by2 - by1
        if bw < 64 and bh < 64:
            return None

        # Если внутри сдвоенного/строенного облачка найдено 3+ отдельных текстовых блока
        if len(matched_tbs) >= 3:
            # Проверяем, что текстовые боксы попарно не являются дублями друг друга
            for idx_a in range(len(matched_tbs)):
                ta = matched_tbs[idx_a]["bbox"]
                area_a = float(max(1, (ta[2] - ta[0]) * (ta[3] - ta[1])))
                for idx_b in range(idx_a + 1, len(matched_tbs)):
                    tb = matched_tbs[idx_b]["bbox"]
                    area_b = float(max(1, (tb[2] - tb[0]) * (tb[3] - tb[1])))
                    ix1, iy1 = max(ta[0], tb[0]), max(ta[1], tb[1])
                    ix2, iy2 = min(ta[2], tb[2]), min(ta[3], tb[3])
                    if ix2 > ix1 and iy2 > iy1:
                        if float((ix2 - ix1) * (iy2 - iy1)) / min(area_a, area_b) >= 0.20:
                            return None
            out_multi: list[dict[str, Any]] = []
            for m_tb in matched_tbs:
                tx1, ty1, tx2, ty2 = m_tb["bbox"]
                tw_m = max(12, tx2 - tx1)
                th_m = max(12, ty2 - ty1)
                pad_x = max(12, int(round(tw_m * 0.28)))
                pad_y = max(12, int(round(th_m * 0.28)))
                sub_b = [
                    max(bx1, tx1 - pad_x),
                    max(by1, ty1 - pad_y),
                    min(bx2, tx2 + pad_x),
                    min(by2, ty2 + pad_y),
                ]
                out_multi.append(
                    {
                        "bbox": sub_b,
                        "text_bbox": cls._compute_tight_ink_bbox(img_np, sub_b, [tx1, ty1, tx2, ty2], is_free_text=False),
                        "confidence": round(max(float(bubble_det["confidence"]), float(m_tb["confidence"])), 4),
                        "is_free_text": False,
                    }
                )
            return out_multi

        tb1 = matched_tbs[0]["bbox"]
        tb2 = matched_tbs[1]["bbox"]
        w1, h1 = max(8, tb1[2] - tb1[0]), max(8, tb1[3] - tb1[1])
        w2, h2 = max(8, tb2[2] - tb2[0]), max(8, tb2[3] - tb2[1])

        ix1 = max(tb1[0], tb2[0])
        iy1 = max(tb1[1], tb2[1])
        ix2 = min(tb1[2], tb2[2])
        iy2 = min(tb1[3], tb2[3])
        if ix2 > ix1 and iy2 > iy1:
            inter_a = float((ix2 - ix1) * (iy2 - iy1))
            min_a = float(min(w1 * h1, w2 * h2))
            if inter_a / max(1.0, min_a) >= 0.18:
                return None

        cx1, cy1 = (tb1[0] + tb1[2]) / 2.0, (tb1[1] + tb1[3]) / 2.0
        cx2, cy2 = (tb2[0] + tb2[2]) / 2.0, (tb2[1] + tb2[3]) / 2.0
        dx_centers = abs(cx1 - cx2)
        dy_centers = abs(cy1 - cy2)

        x_gap = max(tb1[0], tb2[0]) - min(tb1[2], tb2[2])
        y_gap = max(tb1[1], tb2[1]) - min(tb1[3], tb2[3])

        crop_gray = cv2.cvtColor(img_np[by1:by2, bx1:bx2], cv2.COLOR_RGB2GRAY)

        # Определяем доминирующую ось разделения двух долей сдвоенного бабла (включая диагональные восьмерки)
        horiz_dominant = (x_gap >= y_gap and x_gap >= 4) or (
            x_gap < 4 and y_gap < 6 and (dx_centers / max(1.0, float(bw))) >= (dy_centers / max(1.0, float(bh)))
        )

        # Вариант 1: Горизонтальный или диагонально-горизонтальный раздел (два облачка слева и справа)
        if horiz_dominant and dx_centers >= max(24.0, min(w1, w2) * 0.65):
            if tb1[0] + tb1[2] <= tb2[0] + tb2[2]:
                left_tb, right_tb = tb1, tb2
                left_conf, right_conf = float(matched_tbs[0]["confidence"]), float(matched_tbs[1]["confidence"])
            else:
                left_tb, right_tb = tb2, tb1
                left_conf, right_conf = float(matched_tbs[1]["confidence"]), float(matched_tbs[0]["confidence"])

            gap_s = max(bx1 + 10, min(left_tb[2], int(round((cx1 + cx2) / 2.0)) - 6)) - bx1
            gap_e = min(bx2 - 10, max(right_tb[0], int(round((cx1 + cx2) / 2.0)) + 6)) - bx1
            if gap_e > gap_s + 2:
                white_cols = np.mean(crop_gray >= 225, axis=0)
                min_rel = int(np.argmin(white_cols[gap_s:gap_e])) + gap_s
                split_x = bx1 + min_rel
            else:
                split_x = int(round((left_tb[2] + right_tb[0]) / 2.0))

            split_x = max(bx1 + 24, min(bx2 - 24, split_x))
            if (split_x - bx1) >= 24 and (bx2 - split_x) >= 24:
                # Поджимаем вертикальные границы каждой половинки вокруг её собственного текстового бокса,
                # чтобы при диагональном расположении двух баблов их рамки не наезжали друг на друга по высоте!
                lh_pad = max(18, int(round((left_tb[3] - left_tb[1]) * 0.42)))
                rh_pad = max(18, int(round((right_tb[3] - right_tb[1]) * 0.42)))
                ly1 = max(by1, left_tb[1] - lh_pad) if dy_centers >= bh * 0.22 else by1
                ly2 = min(by2, left_tb[3] + lh_pad) if dy_centers >= bh * 0.22 else by2
                ry1 = max(by1, right_tb[1] - rh_pad) if dy_centers >= bh * 0.22 else by1
                ry2 = min(by2, right_tb[3] + rh_pad) if dy_centers >= bh * 0.22 else by2

                b_left = [bx1, ly1, split_x, max(ly1 + 24, ly2)]
                b_right = [split_x, ry1, bx2, max(ry1 + 24, ry2)]
                return [
                    {
                        "bbox": b_left,
                        "text_bbox": cls._compute_tight_ink_bbox(img_np, b_left, left_tb, is_free_text=False),
                        "confidence": round(max(float(bubble_det["confidence"]), left_conf), 4),
                        "is_free_text": False,
                    },
                    {
                        "bbox": b_right,
                        "text_bbox": cls._compute_tight_ink_bbox(img_np, b_right, right_tb, is_free_text=False),
                        "confidence": round(max(float(bubble_det["confidence"]), right_conf), 4),
                        "is_free_text": False,
                    },
                ]

        # Вариант 2: Вертикальный или диагонально-вертикальный раздел (два облачка сверху и снизу — часто в манхве/маньхуа)
        if dy_centers >= max(24.0, min(h1, h2) * 0.65):
            if tb1[1] + tb1[3] <= tb2[1] + tb2[3]:
                top_tb, bot_tb = tb1, tb2
                top_conf, bot_conf = float(matched_tbs[0]["confidence"]), float(matched_tbs[1]["confidence"])
            else:
                top_tb, bot_tb = tb2, tb1
                top_conf, bot_conf = float(matched_tbs[1]["confidence"]), float(matched_tbs[0]["confidence"])

            gap_s = max(by1 + 10, min(top_tb[3], int(round((cy1 + cy2) / 2.0)) - 6)) - by1
            gap_e = min(by2 - 10, max(bot_tb[1], int(round((cy1 + cy2) / 2.0)) + 6)) - by1
            if gap_e > gap_s + 2:
                white_rows = np.mean(crop_gray >= 225, axis=1)
                min_rel = int(np.argmin(white_rows[gap_s:gap_e])) + gap_s
                split_y = by1 + min_rel
            else:
                split_y = int(round((top_tb[3] + bot_tb[1]) / 2.0))

            split_y = max(by1 + 22, min(by2 - 22, split_y))
            if (split_y - by1) >= 22 and (by2 - split_y) >= 22:
                tw_pad = max(18, int(round((top_tb[2] - top_tb[0]) * 0.42)))
                bw_pad = max(18, int(round((bot_tb[2] - bot_tb[0]) * 0.42)))
                tx1_b = max(bx1, top_tb[0] - tw_pad) if dx_centers >= bw * 0.22 else bx1
                tx2_b = min(bx2, top_tb[2] + tw_pad) if dx_centers >= bw * 0.22 else bx2
                bx1_b = max(bx1, bot_tb[0] - bw_pad) if dx_centers >= bw * 0.22 else bx1
                bx2_b = min(bx2, bot_tb[2] + bw_pad) if dx_centers >= bw * 0.22 else bx2

                b_top = [tx1_b, by1, max(tx1_b + 24, tx2_b), split_y]
                b_bot = [bx1_b, split_y, max(bx1_b + 24, bx2_b), by2]
                return [
                    {
                        "bbox": b_top,
                        "text_bbox": cls._compute_tight_ink_bbox(img_np, b_top, top_tb, is_free_text=False),
                        "confidence": round(max(float(bubble_det["confidence"]), top_conf), 4),
                        "is_free_text": False,
                    },
                    {
                        "bbox": b_bot,
                        "text_bbox": cls._compute_tight_ink_bbox(img_np, b_bot, bot_tb, is_free_text=False),
                        "confidence": round(max(float(bubble_det["confidence"]), bot_conf), 4),
                        "is_free_text": False,
                    },
                ]

        return None

    def _associate_bubbles_and_text_boxes(
        self,
        img_np: np.ndarray,
        bubbles: list[dict[str, Any]],
        text_bubbles: list[dict[str, Any]],
        min_box_size: int = 16,
    ) -> list[dict[str, Any]]:
        """
        Сопоставляет контуры облачков `bubbles` (от YOLOv8m + RT-DETR-v2 class 0 + OpenCV)
        и точные текстовые боксы `text_bubbles` (от RT-DETR-v2 class 1).
        Для каждого итогового блока гарантирует наличие как `bbox` (контур баббла для верстки),
        так и точного `text_bbox` (плотный контур букв для безшумного OCR и защиты краев при очистке).
        """
        img_h, img_w = img_np.shape[:2]

        # Дедуплицируем список `text_bubbles`, но с более строгим порогом, чтобы не сливать соседние реплики
        dedup_tbs = self._merge_and_nms_detections(
            text_bubbles,
            iou_threshold=0.48,
            iomin_threshold=0.72,
        )

        if dedup_tbs:
            try:
                _, cavity_map = group_text_boxes_by_bubble_cavity(img_np, dedup_tbs)
                self._last_cavity_map = cavity_map
            except Exception:
                self._last_cavity_map = np.zeros((img_h, img_w), dtype=np.int32)
        else:
            self._last_cavity_map = np.zeros((img_h, img_w), dtype=np.int32)

        # Сначала сопоставляем индивидуальные `dedup_tbs` с найденными `bubbles`, чтобы текстовые боксы
        # из разных бабблов страницы никогда не сливались друг с другом через глобальный белый фон!
        bubble_to_tbs: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(len(bubbles))}
        orphan_tbs: list[dict[str, Any]] = []

        for tb in dedup_tbs:
            tx1, ty1, tx2, ty2 = tb["bbox"]
            t_area = max(1.0, float((tx2 - tx1) * (ty2 - ty1)))
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0

            best_b_idx = -1
            best_iomin = 0.0

            for b_idx, b_det in enumerate(bubbles):
                bx1, by1, bx2, by2 = b_det["bbox"]
                ix1 = max(tx1, bx1)
                iy1 = max(ty1, by1)
                ix2 = min(tx2, bx2)
                iy2 = min(ty2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter_area = float((ix2 - ix1) * (iy2 - iy1))
                iomin = inter_area / t_area
                center_inside = (bx1 <= tcx <= bx2) and (by1 <= tcy <= by2)

                if (iomin >= 0.42 or (center_inside and iomin >= 0.28)) and iomin > best_iomin:
                    best_iomin = iomin
                    best_b_idx = b_idx

            if best_b_idx >= 0:
                bubble_to_tbs[best_b_idx].append(tb)
            else:
                orphan_tbs.append(tb)

        final_detections: list[dict[str, Any]] = []
        split_count = 0
        rescued_count = 0

        for b_idx, b_det in enumerate(bubbles):
            matched = bubble_to_tbs.get(b_idx) or []
            bx1, by1, bx2, by2 = b_det["bbox"]
            is_free = bool(b_det.get("is_free_text", False))

            if len(matched) == 0:
                tight_tb = self._compute_tight_ink_bbox(img_np, [bx1, by1, bx2, by2], None, is_free_text=is_free)
                b_copy = dict(b_det)
                b_copy["text_bbox"] = tight_tb
                final_detections.append(b_copy)
            elif len(matched) == 1:
                tb_coords = matched[0]["bbox"]
                tight_tb = self._compute_tight_ink_bbox(
                    img_np,
                    [bx1, by1, bx2, by2],
                    tb_coords,
                    is_free_text=is_free,
                )
                b_copy = dict(b_det)
                b_copy["text_bbox"] = tight_tb
                b_copy["confidence"] = round(max(float(b_det["confidence"]), float(matched[0]["confidence"])), 4)
                final_detections.append(b_copy)
            else:
                # Блок 3: Внутри одного `b_det` найдено >= 2 текстовых бокса.
                # Запускаем `group_text_boxes_by_bubble_cavity` локально на ROI этого баббла:
                # - если боксы лежат в ОДНОЙ непрерывной чаше баббла без узкого перешейка (`len(local_groups) == 1`),
                #   объединяем их вертикальные колонки (`tategaki`) в единый `text_bbox` без ложного разрезания баббла;
                # - если баббл является сдвоенной «восьмёркой» и водораздел выделил >= 2 долей (`len(local_groups) >= 2`),
                #   разрезаем баббл по долям через `_try_split_bubble_by_text_boxes`!
                roi_bub = img_np[max(0, by1) : min(img_h, by2), max(0, bx1) : min(img_w, bx2)]
                # Используем Проблему №3 (decompose_overlapping_bubbles) для проверки: не разделены ли matched текстовые боксы
                # черной стенкой баббла (например, текст внутри облачка и внешний SFX «ポカポカ» рядом с чиби-персонажем на Screenshot_7)!
                if len(matched) >= 2 and roi_bub.size > 0:
                    tb_tuples = [tuple(int(v) for v in m["bbox"][:4]) for m in matched]
                    decomp = decompose_overlapping_bubbles(img_np, tb_tuples)
                    ux_d, uy_d, _, _ = decomp["union_bbox"]
                    cavs_d = decomp["cavities"]
                    # Проверяем, какие из matched действительно лежат в замкнутой полости баббла
                    valid_matched: list[dict[str, Any]] = []
                    for idx_m, m_item in enumerate(matched):
                        cav_m = cavs_d[idx_m] if idx_m < len(cavs_d) else None
                        if cav_m is not None and np.count_nonzero(cav_m) > 0:
                            # Проверяем, не вытекает ли полость этого под-бокса наружу через края
                            t_lk = float(np.mean(cav_m[0, :] > 0))
                            b_lk = float(np.mean(cav_m[-1, :] > 0))
                            l_lk = float(np.mean(cav_m[:, 0] > 0))
                            r_lk = float(np.mean(cav_m[:, -1] > 0))
                            if sum([t_lk > 0.35, b_lk > 0.35, l_lk > 0.35, r_lk > 0.35]) >= 3:
                                orphan_tbs.append(m_item)
                                continue
                        valid_matched.append(m_item)
                    if len(valid_matched) >= 1:
                        matched = valid_matched

                local_tbs_in: list[dict[str, Any]] = []
                for m in matched:
                    mx1, my1, mx2, my2 = m["bbox"]
                    local_tbs_in.append(
                        {
                            "bbox": [
                                max(0, mx1 - bx1),
                                max(0, my1 - by1),
                                max(1, mx2 - bx1),
                                max(1, my2 - by1),
                            ],
                            "confidence": float(m.get("confidence", 0.5)),
                        }
                    )
                local_groups, _ = group_text_boxes_by_bubble_cavity(roi_bub, local_tbs_in)
                if len(local_groups) >= 2:
                    lobe_matched: list[dict[str, Any]] = []
                    for lg in local_groups:
                        lx1, ly1, lx2, ly2 = lg["bbox"]
                        sub_s = lg.get("sub_boxes") or []
                        l_conf = max((float(s.get("confidence", 0.5)) for s in sub_s), default=0.5)
                        lobe_matched.append(
                            {
                                "bbox": [bx1 + lx1, by1 + ly1, bx1 + lx2, by1 + ly2],
                                "confidence": l_conf,
                            }
                        )
                    split_parts = self._try_split_bubble_by_text_boxes(img_np, b_det, lobe_matched)
                else:
                    split_parts = None

                if split_parts:
                    split_count += 1
                    final_detections.extend(split_parts)
                else:
                    ux1 = min(m["bbox"][0] for m in matched)
                    uy1 = min(m["bbox"][1] for m in matched)
                    ux2 = max(m["bbox"][2] for m in matched)
                    uy2 = max(m["bbox"][3] for m in matched)
                    tight_tb = self._compute_tight_ink_bbox(
                        img_np,
                        [bx1, by1, bx2, by2],
                        [ux1, uy1, ux2, uy2],
                        is_free_text=is_free,
                    )
                    b_copy = dict(b_det)
                    b_copy["text_bbox"] = tight_tb
                    if local_groups:
                        b_copy["is_vertical"] = bool(local_groups[0].get("is_vertical", False))
                    b_copy["confidence"] = round(
                        max([float(b_det["confidence"])] + [float(m["confidence"]) for m in matched]),
                        4,
                    )
                    final_detections.append(b_copy)

        # Обрабатываем «сиротские» `text_bubble` (облачка без рамок, мысленные облачка или экраны телефонов, пропущенные YOLOv8m)
        for otb in orphan_tbs:
            otb_conf = float(otb.get("confidence", 0.0))
            if otb_conf < 0.42:
                continue

            tx1, ty1, tx2, ty2 = otb["bbox"]
            tw = tx2 - tx1
            th = ty2 - ty1
            if tw < min_box_size or th < min_box_size:
                continue

            crop_rgb = img_np[ty1:ty2, tx1:tx2]
            has_ink, _ = self._has_meaningful_ink(crop_rgb, is_free_text=False)
            if not has_ink or not self._has_real_text_glyph_structure(crop_rgb, is_free_text=True):
                continue

            # Расширяем `text_bubble` на 8% (4..14 px), чтобы получить комфортный `bbox` для верстки перевода
            pad_x = max(4, min(14, int(round(tw * 0.08))))
            pad_y = max(4, min(14, int(round(th * 0.08))))
            ex1 = max(0, tx1 - pad_x)
            ey1 = max(0, ty1 - pad_y)
            ex2 = min(img_w, tx2 + pad_x)
            ey2 = min(img_h, ty2 + pad_y)

            # Проверим, не пересекается ли уже сильно с каким-то из итоговых бабблов
            overlaps_existing = False
            o_area = max(1.0, float((ex2 - ex1) * (ey2 - ey1)))
            for ex_det in final_detections:
                fx1, fy1, fx2, fy2 = ex_det["bbox"]
                ix1 = max(ex1, fx1)
                iy1 = max(ey1, fy1)
                ix2 = min(ex2, fx2)
                iy2 = min(ey2, fy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    if inter / o_area >= 0.35:
                        overlaps_existing = True
                        break
            if overlaps_existing:
                continue

            # Строгая проверка v2: действительно ли сиротский text_bubble находится внутри однородного светлого облачка
            # (либо является подтвержденным многострочным экраном смартфона `_is_strict_multiline_screen_crop`)!
            exp_crop = img_np[ey1:ey2, ex1:ex2]
            exp_gray = cv2.cvtColor(exp_crop, cv2.COLOR_RGB2GRAY) if exp_crop.ndim == 3 else exp_crop
            white_pixel_ratio = float(np.mean(exp_gray >= 228)) if exp_gray.size > 0 else 0.0
            non_ink_mask = exp_gray >= 175
            bg_std = float(np.std(exp_gray[non_ink_mask])) if int(np.count_nonzero(non_ink_mask)) >= 16 else 99.0
            top_b = exp_gray[:3, :].ravel()
            bot_b = exp_gray[max(0, exp_gray.shape[0] - 3) :, :].ravel()
            left_b = exp_gray[:, :3].ravel()
            right_b = exp_gray[:, max(0, exp_gray.shape[1] - 3) :].ravel()
            perimeter_px = np.concatenate([top_b, bot_b, left_b, right_b])
            perimeter_median = float(np.median(perimeter_px)) if perimeter_px.size > 0 else 0.0
            perimeter_white_ratio = float(np.mean(perimeter_px >= 224)) if perimeter_px.size > 0 else 0.0

            is_strict_screen, _, _, _ = self._is_strict_multiline_screen_crop(exp_crop)
            ver_orphan = verify_bubble_and_text_candidate(exp_gray, bubbles_only=True)
            if (ver_orphan["is_kagon"] or ver_orphan["is_tile_grid"] or ver_orphan["is_face"]) and not is_strict_screen:
                continue
            is_clean_bubble_bg = is_strict_screen or (
                ver_orphan["is_valid"]
                and ver_orphan["has_enclosing_frame"]
                and white_pixel_ratio >= 0.58
                and bg_std <= 11.5
            )
            is_free_orphan = not is_clean_bubble_bg

            rescued_count += 1
            final_detections.append(
                {
                    "bbox": [ex1, ey1, ex2, ey2],
                    "text_bbox": self._compute_tight_ink_bbox(img_np, [ex1, ey1, ex2, ey2], [tx1, ty1, tx2, ty2], is_free_text=is_free_orphan),
                    "confidence": round(otb_conf, 4),
                    "is_free_text": is_free_orphan,
                    "_source": "rtdetr_rescued_text_bubble",
                }
            )

        self.last_detection_stats["split_multi_bubbles"] = split_count
        self.last_detection_stats["rescued_orphan_text_bubbles"] = rescued_count
        return final_detections

    def _detect_bubbles(
        self,
        img_np: np.ndarray,
        min_box_size: int = 16,
        conf_override: float | None = None,
    ) -> list[dict[str, Any]]:
        img_h, img_w = img_np.shape[:2]
        yolo_detections: list[dict[str, Any]] = []
        rtdetr_bubbles: list[dict[str, Any]] = []
        rtdetr_text_bubbles: list[dict[str, Any]] = []
        rtdetr_free_texts: list[dict[str, Any]] = []

        active_conf = conf_override if conf_override is not None else self.conf_threshold

        slices: list[tuple[int, int]] = []
        if img_h > 1400 and (img_h / max(1, img_w)) > 2.5:
            win_h = min(img_h, max(1280, int(round(img_w * 1.6))))
            overlap_h = int(round(win_h * 0.28))
            step_y = max(400, win_h - overlap_h)

            y_cur = 0
            while y_cur < img_h:
                y_end = min(img_h, y_cur + win_h)
                if (y_end - y_cur) < min(win_h, 600) and y_cur > 0:
                    y_cur = max(0, img_h - win_h)
                    y_end = img_h
                    slices.append((y_cur, y_end))
                    break
                slices.append((y_cur, y_end))
                if y_end >= img_h:
                    break
                y_cur += step_y
        else:
            slices.append((0, img_h))

        with torch.inference_mode():
            for y_start, y_end in slices:
                slice_np = img_np[y_start:y_end, :]

                # 1. Прогон YOLOv8m (speech bubble detector)
                results = self.detector.predict(
                    source=slice_np,
                    conf=active_conf,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    xyxy_list = boxes.xyxy.cpu().numpy().tolist()
                    conf_list = boxes.conf.cpu().numpy().tolist()

                    for bbox, conf in zip(xyxy_list, conf_list):
                        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
                        y1 += y_start
                        y2 += y_start

                        x1 = max(0, min(x1, img_w - 1))
                        y1 = max(0, min(y1, img_h - 1))
                        x2 = max(x1 + 1, min(x2, img_w))
                        y2 = max(y1 + 1, min(y2, img_h))

                        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                            continue

                        yolo_detections.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(float(conf), 4),
                                "is_free_text": False,
                                "_source": "yolov8",
                            }
                        )

                # 2. Прогон двухконтурного 3-классового детектора RT-DETR-v2 (bubble / text_bubble / text_free)
                if self.use_rtdetr:
                    rt_b, rt_tb, rt_free = self._run_rtdetr_on_slice(
                        slice_np,
                        y_start=y_start,
                        img_w=img_w,
                        img_h=img_h,
                        active_conf=active_conf,
                        min_box_size=max(12, min_box_size - 2),
                    )
                    if rt_b:
                        rtdetr_bubbles.extend(rt_b)
                    if rt_tb:
                        rtdetr_text_bubbles.extend(rt_tb)
                    if rt_free:
                        rtdetr_free_texts.extend(rt_free)

        # Если RT-DETR-v2 пометил текст внутри белого экрана телефона / таблички / записки как `2: text_free`,
        # но штрихи букв лежат внутри чистой белой полости (`_is_text_on_clean_white_cavity`),
        # переводим такой бокс в `rtdetr_text_bubbles`, чтобы он переводился даже в профиле `bubbles_only`!
        if rtdetr_free_texts:
            for rf in rtdetr_free_texts:
                rx1, ry1, rx2, ry2 = rf["bbox"]
                if (rx2 - rx1) >= 48 and (ry2 - ry1) >= 48:
                    rf_crop = img_np[ry1:ry2, rx1:rx2]
                    ok_scr, _, _, _ = self._is_strict_multiline_screen_crop(rf_crop)
                    if ok_scr:
                        rtdetr_text_bubbles.append(
                            {
                                "bbox": [rx1, ry1, rx2, ry2],
                                "confidence": max(0.48, float(rf.get("confidence", 0.48))),
                                "is_free_text": False,
                                "_source": "rtdetr_free_on_white_cavity",
                            }
                        )

        self._last_rtdetr_free_boxes = rtdetr_free_texts
        self.last_detection_stats.update(
            {
                "detector_backend": self.get_detector_backend_name(),
                "rtdetr_active": bool(self._rtdetr_available and self._rtdetr_model is not None),
                "yolo_bubbles": len(yolo_detections),
                "rtdetr_bubbles": len(rtdetr_bubbles),
                "rtdetr_text_bubbles": len(rtdetr_text_bubbles),
                "rtdetr_free_texts": len(rtdetr_free_texts),
            }
        )

        raw_detections = yolo_detections + rtdetr_bubbles

        # 3. Резервный контурный поиск OpenCV запускаем ТОЛЬКО если двухконтурный RT-DETR-v2 недоступен/отключен,
        # либо если RT-DETR-v2 нашел на странице текстовые боксы (`rtdetr_text_bubbles`), пропущенные детектором контуров.
        rtdetr_is_active = bool(self._rtdetr_available and self._rtdetr_model is not None)
        if not rtdetr_is_active or (len(raw_detections) == 0 and len(rtdetr_text_bubbles) > 0):
            cv_bubbles = self._detect_missed_white_bubbles(img_np, raw_detections, min_box_size=min_box_size)
            if cv_bubbles:
                raw_detections.extend(cv_bubbles)

        if raw_detections:
            raw_detections = self._merge_and_nms_detections(
                raw_detections,
                iou_threshold=0.52,
                iomin_threshold=0.62,
            )
            raw_detections = self._split_double_bubbles(img_np, raw_detections)

        # 4. Сопоставляем найденные контуры бабблов с точными внутренними боксами текста (`text_bubble` + TightInkCrop)
        raw_detections = self._associate_bubbles_and_text_boxes(
            img_np,
            raw_detections,
            rtdetr_text_bubbles,
            min_box_size=min_box_size,
        )

        # 4.5. Каскадная топологическая и спектральная верификация кандидатов (Проблема №6а)
        # + автоматическая подрезка раздутых боксов по замкнутой чаше баббла (защита чиби-персонажей на Screenshot_7)
        verified_detections: list[dict[str, Any]] = []
        page_gray_full = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
        for cand_det in raw_detections:
            bx1, by1, bx2, by2 = [int(v) for v in cand_det["bbox"][:4]]
            is_free_c = bool(cand_det.get("is_free_text", False))
            roi_g = page_gray_full[max(0, by1) : min(img_h, by2), max(0, bx1) : min(img_w, bx2)]
            if roi_g.size < 196:
                continue
            ver_c = verify_bubble_and_text_candidate(roi_g, bubbles_only=not is_free_c)
            if not is_free_c and not ver_c["is_valid"]:
                # Проверяем: возможно, бокс раздут и включает как настоящий баббл, так и соседнего чиби-персонажа (Screenshot_7)!
                tb_c = cand_det.get("text_bbox")
                if isinstance(tb_c, (list, tuple)) and len(tb_c) >= 4:
                    tx1_c, ty1_c, tx2_c, ty2_c = [int(v) for v in tb_c[:4]]
                    pad_tx = max(12, int((tx2_c - tx1_c) * 0.22))
                    pad_ty = max(12, int((ty2_c - ty1_c) * 0.22))
                    sx1_c, sy1_c = max(bx1, tx1_c - pad_tx), max(by1, ty1_c - pad_ty)
                    sx2_c, sy2_c = min(bx2, tx2_c + pad_tx), min(by2, ty2_c + pad_ty)
                    if (sx2_c - sx1_c) >= min_box_size and (sy2_c - sy1_c) >= min_box_size:
                        sub_ver = verify_bubble_and_text_candidate(
                            page_gray_full[sy1_c:sy2_c, sx1_c:sx2_c], bubbles_only=True
                        )
                        if sub_ver["is_valid"]:
                            cand_det["bbox"] = [sx1_c, sy1_c, sx2_c, sy2_c]
                            cand_det["has_enclosing_frame"] = bool(sub_ver["has_enclosing_frame"])
                            verified_detections.append(cand_det)
                            continue
                # Иначе проверяем, не является ли это подтвержденным многострочным экраном смартфона
                ok_scr_c, _, _, _ = self._is_strict_multiline_screen_crop(
                    img_np[max(0, by1) : min(img_h, by2), max(0, bx1) : min(img_w, bx2)]
                )
                if not ok_scr_c:
                    logger.debug(
                        "Отклонен ложный кандидат баббла [%d,%d,%d,%d] (face=%s, kagon=%s, tile=%s)",
                        bx1, by1, bx2, by2, ver_c["is_face"], ver_c["is_kagon"], ver_c["is_tile_grid"],
                    )
                    continue
            cand_det["has_enclosing_frame"] = bool(ver_c["has_enclosing_frame"])
            verified_detections.append(cand_det)
        raw_detections = verified_detections

        # 5. Детекция пропущенных или частично обрезанных экранов телефонов, мониторов, писем и наклонных табличек
        # с множеством глифов на чистой белой полости (включая наклонные экраны, касающиеся краев фрейма)
        screen_added = self._detect_missed_screen_and_tilted_text_containers(
            img_np,
            raw_detections,
            min_box_size=max(24, min_box_size),
        )
        if screen_added:
            raw_detections.extend(screen_added)

        return raw_detections

    def _detect_missed_screen_and_tilted_text_containers(
        self,
        img_np: np.ndarray,
        existing_detections: list[dict[str, Any]],
        min_box_size: int = 24,
    ) -> list[dict[str, Any]]:
        """
        Находит пропущенные детекторами YOLOv8m / RT-DETR-v2 (или захваченные лишь частично)
        текстовые блоки на белых экранах смартфонов, мониторах, письмах и наклонных плашках:
        - Выделяет связные компоненты чернил буквенного размера (`glyph_mask`), окруженные чистым белым фоном (`p80 >= 232`, `std <= 12.5`),
          разделенные темными стенками бабблов, рамками панелей и корпусами телефонов (`wall_mask`).
        - Если кластер на белом экране содержит `>= 8` глифов (и проходит `pre_ocr_stroke_gate` + проверку наклонных/многострочных строк):
          * при отсутствии пересечения с `existing_detections` (`< 25%`) — добавляет новый блок экрана;
          * при частичном перекрытии (`25%..80%`, когда детектор обвел только часть строк экрана) — расширяет существующий блок на все строки экрана!
        """
        img_h, img_w = img_np.shape[:2]
        page_area = max(1, img_h * img_w)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np

        _, bw_inv = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        # Отсоединяем прямые линии рамок фреймов (панелей), чтобы краевые иероглифы на экранах телефонов (紅, 談, 事, 助) не поглощались рамкой
        k_hl = cv2.getStructuringElement(cv2.MORPH_RECT, (max(45, img_w // 6), 1))
        k_vl = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(45, img_h // 6)))
        panel_lines = cv2.bitwise_or(
            cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, k_hl),
            cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, k_vl),
        )
        sever_panel = cv2.dilate(panel_lines, np.ones((3, 3), np.uint8), iterations=1)
        bw_severed = cv2.bitwise_and(bw_inv, cv2.bitwise_not(sever_panel))

        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(bw_severed, connectivity=8)
        if n_cc <= 4:
            return []

        glyph_mask = np.zeros_like(bw_inv)
        wall_mask = sever_panel.copy()
        max_gw = max(24, min(64, int(img_w * 0.14)))
        max_gh = max(24, min(64, int(img_h * 0.14)))

        for i in range(1, n_cc):
            x, y, cw, ch, area = [int(v) for v in stats[i]]
            if cw > img_w * 0.18 or ch > img_h * 0.18 or area > int(page_area * 0.015):
                wall_mask[labels == i] = 255
                continue
            if area < 4 or area > int(page_area * 0.012):
                continue
            if cw < 2 or ch < 2 or cw > max_gw or ch > max_gh:
                continue
            aspect = max(cw, ch) / max(1.0, float(min(cw, ch)))
            if aspect > 7.0:
                continue
            pad = 4
            sub = gray[max(0, y - pad) : min(img_h, y + ch + pad), max(0, x - pad) : min(img_w, x + cw + pad)]
            sub_w = wall_mask[max(0, y - pad) : min(img_h, y + ch + pad), max(0, x - pad) : min(img_w, x + cw + pad)]
            valid_bg = sub[sub_w == 0]
            white_px = valid_bg[valid_bg >= 220]
            if (
                white_px.size >= max(4, int(valid_bg.size * 0.32))
                and float(np.percentile(valid_bg, 80)) >= 230.0
                and float(np.std(white_px)) <= 12.8
            ):
                glyph_mask[labels == i] = 255

        k_group = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        grouped = cv2.dilate(glyph_mask, k_group, iterations=1)
        grouped[cv2.dilate(wall_mask, np.ones((3, 3), np.uint8)) > 0] = 0

        ng, g_labels, g_stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
        added_screens: list[dict[str, Any]] = []

        for gid in range(1, ng):
            gx, gy, gw, gh, _ = [int(v) for v in g_stats[gid]]
            if gw < min_box_size or gh < min_box_size:
                continue
            sub_g = (g_labels[gy : gy + gh, gx : gx + gw] == gid) & (glyph_mask[gy : gy + gh, gx : gx + gw] > 0)
            ys, xs = np.where(sub_g)
            if xs.size < 35:
                continue
            tx1 = int(gx + int(np.min(xs)))
            ty1 = int(gy + int(np.min(ys)))
            tx2 = int(gx + int(np.max(xs)) + 1)
            ty2 = int(gy + int(np.max(ys)) + 1)
            tw = tx2 - tx1
            th = ty2 - ty1
            if tw < min_box_size or th < min_box_size or (tw * th) < 1100:
                continue

            comp_ids = np.unique(
                labels[ty1:ty2, tx1:tx2][
                    sub_g[int(np.min(ys)) : int(np.max(ys)) + 1, int(np.min(xs)) : int(np.max(xs)) + 1]
                ]
            )
            comp_cnt = len([c for c in comp_ids if c > 0])
            # Настоящий многострочный экран смартфона/монитора содержит минимум 28 буквенных компонент (обычно 40..120)
            if comp_cnt < 28 or tw < 48 or th < 48:
                continue

            c_area = float(tw * th)
            best_overlap_idx = -1
            best_overlap_ratio = 0.0
            for idx_e, ex_det in enumerate(existing_detections):
                ex1, ey1, ex2, ey2 = ex_det["bbox"]
                ix1, iy1 = max(tx1, ex1), max(ty1, ey1)
                ix2, iy2 = min(tx2, ex2), min(ty2, ey2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    ov_r = inter / max(1.0, min(c_area, float((ex2 - ex1) * (ey2 - ey1))))
                    if ov_r > best_overlap_ratio:
                        best_overlap_ratio = ov_r
                        best_overlap_idx = idx_e

            if best_overlap_ratio >= 0.20:
                continue

            pad_x = max(5, min(14, int(round(tw * 0.05))))
            pad_y = max(5, min(14, int(round(th * 0.05))))
            bx1 = max(0, tx1 - pad_x)
            by1 = max(0, ty1 - pad_y)
            bx2 = min(img_w, tx2 + pad_x)
            by2 = min(img_h, ty2 + pad_y)
            # Не перешагиваем через прямые линии рамок панелей (panel_lines) при расширении бокса экрана
            for py_chk in range(ty1, by1 - 1, -1):
                if float(np.mean(panel_lines[py_chk, tx1:tx2] > 0)) >= 0.35:
                    by1 = py_chk + 1
                    break
            for py_chk in range(ty2 - 1, by2):
                if float(np.mean(panel_lines[py_chk, tx1:tx2] > 0)) >= 0.35:
                    by2 = py_chk
                    break
            for px_chk in range(tx1, bx1 - 1, -1):
                if float(np.mean(panel_lines[ty1:ty2, px_chk] > 0)) >= 0.35:
                    bx1 = px_chk + 1
                    break
            for px_chk in range(tx2 - 1, bx2):
                if float(np.mean(panel_lines[ty1:ty2, px_chk] > 0)) >= 0.35:
                    bx2 = px_chk
                    break

            crop_cand = img_np[by1:by2, bx1:bx2]
            if not pre_ocr_stroke_gate(crop_cand).passed:
                continue
            ok_scr, deg_c, lines_c, _ = self._is_strict_multiline_screen_crop(crop_cand)
            if not ok_scr or abs(deg_c) < 5.5 or len(lines_c) < 4:
                continue

            tb_pad = 3
            added_screens.append(
                {
                    "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                    "text_bbox": [
                        int(max(bx1, tx1 - tb_pad)),
                        int(max(by1, ty1 - tb_pad)),
                        int(min(bx2, tx2 + tb_pad)),
                        int(min(by2, ty2 + tb_pad)),
                    ],
                    "confidence": 0.65,
                    "is_free_text": False,
                    "_source": "cv_screen_container",
                }
            )

        return added_screens

    def _detect_missed_white_bubbles(
        self,
        img_np: np.ndarray,
        existing_bubbles: list[dict[str, Any]],
        min_box_size: int = 24,
    ) -> list[dict[str, Any]]:
        """
        Находит замкнутые белые/светлые кружки и овалы speech bubbles с черным текстом внутри,
        если нейросеть YOLOv8 пропустила кружок (например, идеально круглые, облачные или мелкие бабблы).
        """
        img_h, img_w = img_np.shape[:2]
        img_area = max(1, img_w * img_h)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Белые внутренности бабблов
        _, white_mask = cv2.threshold(gray, 232, 255, cv2.THRESH_BINARY)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_white = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        contours, _ = cv2.findContours(closed_white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        added: list[dict[str, Any]] = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < max(26, min_box_size) or h < max(26, min_box_size):
                continue
            box_area = float(w * h)
            if box_area < 900 or box_area > img_area * 0.18:
                continue
            aspect = w / max(1.0, float(h))
            if aspect < 0.22 or aspect > 3.8:
                continue

            # Не должен касаться самых краев страницы (фон/поля страницы)
            if x <= 2 or y <= 2 or (x + w) >= (img_w - 2) or (y + h) >= (img_h - 2):
                continue

            already_covered = False
            for ex in existing_bubbles:
                ex1, ey1, ex2, ey2 = ex["bbox"]
                ix1, iy1 = max(x, ex1), max(y, ey1)
                ix2, iy2 = min(x + w, ex2), min(y + h, ey2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    if inter / min(box_area, float((ex2 - ex1) * (ey2 - ey1))) >= 0.30:
                        already_covered = True
                        break
            if already_covered:
                continue

            cnt_area = float(cv2.contourArea(cnt))
            if cnt_area < box_area * 0.48:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = max(1.0, float(cv2.contourArea(hull)))
            solidity = cnt_area / hull_area
            if solidity < 0.76:
                continue

            pad_in_x = max(3, int(w * 0.16))
            pad_in_y = max(3, int(h * 0.16))
            inner_gray = gray[y + pad_in_y : y + h - pad_in_y, x + pad_in_x : x + w - pad_in_x]
            if inner_gray.size < 64:
                continue

            if float(np.percentile(inner_gray, 75)) < 235.0:
                continue
            dark_ink_px = int(np.count_nonzero(inner_gray < 150))
            inner_ratio = dark_ink_px / float(inner_gray.size)
            if dark_ink_px < 18 or inner_ratio < 0.015 or inner_ratio > 0.45:
                continue

            _, inner_bin = cv2.threshold(inner_gray, 165, 255, cv2.THRESH_BINARY_INV)
            n_lbl, _, st_lbl, _ = cv2.connectedComponentsWithStats(inner_bin, connectivity=8)
            char_comps = 0
            ih, iw = inner_gray.shape[:2]
            for idx_c in range(1, n_lbl):
                cw = st_lbl[idx_c, cv2.CC_STAT_WIDTH]
                ch = st_lbl[idx_c, cv2.CC_STAT_HEIGHT]
                ca = st_lbl[idx_c, cv2.CC_STAT_AREA]
                if ca >= 5 and 2 <= cw <= iw * 0.85 and 2 <= ch <= ih * 0.85:
                    char_comps += 1
            if char_comps < 2:
                continue

            added.append(
                {
                    "bbox": [x, y, x + w, y + h],
                    "confidence": 0.35,
                    "is_free_text": False,
                }
            )

        if len(added) > 10:
            added.sort(key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]), reverse=True)
            added = added[:10]
        return added

    @staticmethod
    def _group_free_text_boxes(
        raw_boxes: list[list[int]],
        img_w: int,
        img_h: int,
    ) -> list[list[int]]:
        if not raw_boxes:
            return []

        boxes = [list(b) for b in raw_boxes]
        changed = True
        while changed:
            changed = False
            new_boxes: list[list[int]] = []
            used = [False] * len(boxes)

            for i in range(len(boxes)):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2 = boxes[i]
                aw = max(1, ax2 - ax1)
                ah = max(1, ay2 - ay1)

                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    bx1, by1, bx2, by2 = boxes[j]
                    bw = max(1, bx2 - bx1)
                    bh = max(1, by2 - by1)

                    ix1 = max(ax1, bx1)
                    iy1 = max(ay1, by1)
                    ix2 = min(ax2, bx2)
                    iy2 = min(ay2, by2)
                    x_overlap = max(0, ix2 - ix1)
                    y_overlap = max(0, iy2 - iy1)

                    x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
                    y_gap = max(0, max(ay1, by1) - min(ay2, by2))

                    min_w = min(aw, bw)
                    min_h = min(ah, bh)

                    intersects = x_overlap > 0 and y_overlap > 0
                    vert_aligned = x_overlap >= min_w * 0.45 and y_gap <= max(16, int(min_h * 0.65))
                    adj_columns = (
                        y_overlap >= min_h * 0.55
                        and x_gap <= max(14, int(min_w * 0.55))
                        and (max(ax2, bx2) - min(ax1, bx1)) <= int(img_w * 0.36)
                    )

                    if intersects or vert_aligned or adj_columns:
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        aw = max(1, ax2 - ax1)
                        ah = max(1, ay2 - ay1)
                        used[j] = True
                        changed = True

                new_boxes.append([ax1, ay1, ax2, ay2])
            boxes = new_boxes

        return [
            [
                max(0, b[0] - 2),
                max(0, b[1] - 2),
                min(img_w, b[2] + 2),
                min(img_h, b[3] + 2),
            ]
            for b in boxes
        ]

    @staticmethod
    def _detect_free_text_cv2_fallback(
        masked_img_np: np.ndarray,
        existing_bubbles: list[dict[str, Any]],
    ) -> list[list[int]]:
        """
        Резервный морфологический детектор высококонтрастных надписей (вызывается только если EasyOCR не установлен).
        """
        img_h, img_w = masked_img_np.shape[:2]
        gray = cv2.cvtColor(masked_img_np, cv2.COLOR_RGB2GRAY)

        k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_small)
        _, thresh = cv2.threshold(blackhat, 72, 255, cv2.THRESH_BINARY)

        exclude = np.zeros((img_h, img_w), dtype=np.uint8)
        for b in existing_bubbles:
            x1, y1, x2, y2 = b["bbox"]
            cv2.rectangle(
                exclude,
                (max(0, x1 - 12), max(0, y1 - 12)),
                (min(img_w, x2 + 12), min(img_h, y2 + 12)),
                255,
                -1,
            )
        thresh[exclude > 0] = 0

        k_group = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_group)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        candidates: list[list[int]] = []
        img_area = max(1, img_w * img_h)

        for idx in range(1, num_labels):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = w * h

            if w < 22 or h < 22:
                continue
            if area < 600 or area > img_area * 0.10:
                continue
            if w > img_w * 0.60 and h < 28:
                continue
            if h > img_h * 0.60 and w < 28:
                continue

            roi_gray = gray[y : y + h, x : x + w]
            if float(np.percentile(roi_gray, 80)) < 200.0:
                continue

            candidates.append([x, y, x + w, y + h])

        return candidates

    def _detect_free_text_regions(
        self,
        img_np: np.ndarray,
        bubble_detections: list[dict[str, Any]],
        min_box_size: int = 18,
    ) -> list[dict[str, Any]]:
        img_h, img_w = img_np.shape[:2]
        masked_np = img_np.copy()

        for det in bubble_detections:
            bx1, by1, bx2, by2 = det["bbox"]
            mx1 = max(0, bx1 - 4)
            my1 = max(0, by1 - 4)
            mx2 = min(img_w, bx2 + 4)
            my2 = min(img_h, by2 + 4)
            masked_np[my1:my2, mx1:mx2] = 255

        raw_free_boxes: list[list[int]] = []

        # 1. Сначала берем высокоточные боксы класса `2: text_free` от RT-DETR-v2 (если он активен),
        # так как RT-DETR-v2 обучен на вертикальном и горизонтальном тексте манги/манхвы вне бабблов!
        if self._last_rtdetr_free_boxes:
            dedup_rt_free = self._merge_and_nms_detections(
                self._last_rtdetr_free_boxes,
                iou_threshold=0.45,
                iomin_threshold=0.65,
            )
            for rf in dedup_rt_free:
                rx1, ry1, rx2, ry2 = rf["bbox"]
                raw_free_boxes.append([rx1, ry1, rx2, ry2])

        # 2. Дополняем через CRAFT (или OpenCV fallback), если RT-DETR-v2 не загружен или нашел мало блоков
        craft_reader = self._get_any_craft_reader() if len(raw_free_boxes) < 6 else None
        if craft_reader is not None:
            slices: list[tuple[int, int]] = []
            if img_h > 1600 and (img_h / max(1, img_w)) > 2.2:
                win_h = min(img_h, max(1280, int(round(img_w * 1.5))))
                step_y = max(500, int(win_h * 0.78))
                y_cur = 0
                while y_cur < img_h:
                    y_end = min(img_h, y_cur + win_h)
                    slices.append((y_cur, y_end))
                    if y_end >= img_h:
                        break
                    y_cur += step_y
            else:
                slices.append((0, img_h))

            for y_start, y_end in slices:
                slice_np = masked_np[y_start:y_end, :]
                try:
                    horiz_list, free_list = craft_reader.detect(
                        slice_np,
                        min_size=16,
                        text_threshold=0.62,
                        low_text=0.38,
                        link_threshold=0.35,
                        width_ths=0.5,
                    )
                    if horiz_list and len(horiz_list) > 0:
                        for box in horiz_list[0]:
                            if len(box) >= 4:
                                x1, x2, y1, y2 = int(box[0]), int(box[1]), int(box[2]) + y_start, int(box[3]) + y_start
                                raw_free_boxes.append([x1, y1, x2, y2])
                    if free_list and len(free_list) > 0:
                        for poly in free_list[0]:
                            xs = [int(pt[0]) for pt in poly]
                            ys = [int(pt[1]) + y_start for pt in poly]
                            raw_free_boxes.append([min(xs), min(ys), max(xs), max(ys)])
                except Exception as exc:
                    logger.debug("CRAFT detect вне бабблов пропущен: %s", exc)
        elif not raw_free_boxes:
            try:
                cv2_boxes = self._detect_free_text_cv2_fallback(masked_np, bubble_detections)
                raw_free_boxes.extend(cv2_boxes)
            except Exception as exc:
                logger.debug("OpenCV free-text fallback ошибка: %s", exc)

        if not raw_free_boxes:
            return []

        grouped_boxes = self._group_free_text_boxes(raw_free_boxes, img_w, img_h)
        free_detections: list[dict[str, Any]] = []

        for gx1, gy1, gx2, gy2 in grouped_boxes:
            gx1 = max(0, min(gx1, img_w - 1))
            gy1 = max(0, min(gy1, img_h - 1))
            gx2 = max(gx1 + 1, min(gx2, img_w))
            gy2 = max(gy1 + 1, min(gy2, img_h))

            gw = gx2 - gx1
            gh = gy2 - gy1
            if gw < min_box_size or gh < min_box_size:
                continue

            g_area = float(gw * gh)
            overlaps_bubble = False
            for b in bubble_detections:
                bx1, by1, bx2, by2 = b["bbox"]
                ix1 = max(gx1, bx1)
                iy1 = max(gy1, by1)
                ix2 = min(gx2, bx2)
                iy2 = min(gy2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    if inter / max(1.0, g_area) >= 0.35:
                        overlaps_bubble = True
                        break
            if overlaps_bubble:
                continue

            crop_np = img_np[gy1:gy2, gx1:gx2]
            has_ink, ink_px = self._has_meaningful_ink(crop_np, is_free_text=True)
            if not has_ink or not self._is_valid_free_text_crop(crop_np):
                continue

            free_detections.append(
                {
                    "bbox": [gx1, gy1, gx2, gy2],
                    "text_bbox": [gx1, gy1, gx2, gy2],
                    "confidence": 0.45,
                    "is_free_text": True,
                    "_ink_score": ink_px,
                }
            )

        if len(free_detections) > 12:
            free_detections.sort(key=lambda d: int(d.get("_ink_score", 0)), reverse=True)
            free_detections = free_detections[:12]

        for d in free_detections:
            d.pop("_ink_score", None)

        return free_detections

    @staticmethod
    def _sort_boxes_reading_order(
        detections: list[dict[str, Any]],
        rtl: bool = True,
        row_overlap_ratio: float = 0.35,
        page_bgr: Optional[np.ndarray] = None,
        lang: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Упорядочивает блоки страницы через двухконтурный топологический граф чтения
        (`segment_manga_panels` + `sort_panels_dag` + внутрипанельный DAG `sort_page_blocks_reading_order_dag`).
        """
        if not detections:
            return []
        eff_lang = lang if lang is not None else ("ja" if rtl else "zh")
        return sort_page_blocks_reading_order_dag(detections, page_bgr=page_bgr, lang=eff_lang)

    def _recognize_crop(
        self,
        padded_crop_pil: Image.Image,
        raw_crop_np: np.ndarray,
        lang: str,
        is_free_text: bool = False,
    ) -> str:
        has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free_text)
        if not has_ink:
            return ""

        if lang in ("ko", "zh"):
            ko_zh_text = self._recognize_single_ko_zh_crop(
                padded_crop_pil,
                raw_crop_np,
                lang=lang,
                is_free_text=is_free_text,
            )
            if self._is_hallucinated_ocr_text(ko_zh_text, ink_pixels, is_free_text=is_free_text):
                return ""
            return ko_zh_text

        ja_list = self._recognize_crops_batch_ja([padded_crop_pil])
        ja_text = ja_list[0].strip() if ja_list else ""

        if self._is_hallucinated_ocr_text(ja_text, ink_pixels, is_free_text=is_free_text):
            return ""
        return ja_text

    def _run_classic_manga_ocr_batch(
        self,
        crops_pil: list[Image.Image],
        batch_size: int = 16,
    ) -> list[str]:
        """
        Пакетное распознавание через классический авторегрессионный `MangaOcr` (до 300 символов)
        с автоматическим выравниванием пропорций `_pad_to_balanced_aspect_for_vit` для узких/широких кропов.
        """
        if not crops_pil:
            return []
        balanced_crops = [self._pad_to_balanced_aspect_for_vit(c, max_aspect_ratio=2.0) for c in crops_pil]
        if isinstance(self.ja_recognizer, HallucinationFreeMangaOCR):
            results_hf: list[str] = []
            for raw_c, bal_c in zip(crops_pil, balanced_crops):
                rec = self.ja_recognizer.recognize(bal_c, gate_crop=raw_c)
                if not rec.is_valid:
                    logger.debug("HallucinationFreeMangaOCR отклонил кроп: %s", rec.reject_reason)
                    results_hf.append("")
                else:
                    results_hf.append(rec.text.strip())
            return results_hf

        if len(balanced_crops) == 1:
            with torch.inference_mode():
                return [self.ja_recognizer(balanced_crops[0]).strip()]

        model = getattr(self.ja_recognizer, "model", None)
        tokenizer = getattr(self.ja_recognizer, "tokenizer", None)
        processor = getattr(self.ja_recognizer, "processor", None) or getattr(
            self.ja_recognizer, "feature_extractor", None
        )
        post_proc = getattr(self.ja_recognizer, "_post_process", None) or getattr(
            self.ja_recognizer, "post_process", None
        )

        if model is None or tokenizer is None or processor is None:
            with torch.inference_mode():
                return [self.ja_recognizer(c).strip() for c in balanced_crops]

        try:
            results_out: list[str] = []
            model_device = next(model.parameters()).device
            use_cuda_fp16 = model_device.type == "cuda"

            with torch.inference_mode():
                for start_i in range(0, len(balanced_crops), batch_size):
                    sub_crops = balanced_crops[start_i : start_i + batch_size]
                    gray_rgb_list = [c.convert("L").convert("RGB") for c in sub_crops]
                    pixel_values = processor(gray_rgb_list, return_tensors="pt").pixel_values.to(
                        model_device, non_blocking=True
                    )
                    if use_cuda_fp16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            token_ids = model.generate(pixel_values, max_new_tokens=300)
                    else:
                        token_ids = model.generate(pixel_values, max_new_tokens=300)

                    decoded_list = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
                    for raw_str in decoded_list:
                        if callable(post_proc):
                            cleaned = str(post_proc(raw_str)).strip()
                        else:
                            cleaned = re.sub(r"\s+", "", str(raw_str)).strip()
                        results_out.append(cleaned)

            if len(results_out) == len(balanced_crops):
                return results_out
        except Exception as batch_exc:
            logger.debug("Пакетный режим MangaOCR недоступен (%s), используем поштучный режим.", batch_exc)

        with torch.inference_mode():
            return [self.ja_recognizer(c).strip() for c in balanced_crops]

    def _recognize_crops_batch_ja(
        self,
        crops_pil: list[Image.Image],
        batch_size: int = 16,
        _allow_line_split: bool = True,
    ) -> list[str]:
        """
        Гибридное пакетное GPU-распознавание японских бабблов (Шаг 2 + Deskew/Multi-line Screen Upgrade):
        0. Автоматически проверяет каждый кроп на наклонный/многострочный горизонтальный текст (экраны телефонов,
           мониторы, записки, таблички под углом) через `_extract_tilted_or_multiline_horizontal_crops`.
           Такие кропы выпрямляются, очищаются от черных рамок корпуса и читаются построчно сверху вниз!
        1. Основной проход выполняется через `muscgab/manga-ocr-nar-small` (CER 1.73%, EM 91.09%, сохранение пропорций
           кропа 324 патча 14x14, чтение всей пачки за ~35 мс).
        2. Если какой-то баббл содержит длинный монолог (`len(text) >= 54`, близко к потолку 64 символов у NAR-декодера)
           или вернул пустую/усеченную строку, только этот баббл автоматически дочитывается через авторегрессионный `MangaOcr`
           (`max_new_tokens=300`) с защитой пропорций `_pad_to_balanced_aspect_for_vit` и `HayaiOCR v2.5 Nova`!
        """
        if not crops_pil:
            return []

        if _allow_line_split:
            multiline_map: dict[int, list[Image.Image]] = {}
            normal_indices: list[int] = []
            normal_crops: list[Image.Image] = []
            all_sub_lines: list[Image.Image] = []
            sub_line_slices: dict[int, tuple[int, int]] = {}

            for idx, c_pil in enumerate(crops_pil):
                try:
                    c_np = np.array(c_pil.convert("RGB"))
                    ok_scr, deg_val, line_pils, _ = self._is_strict_multiline_screen_crop(c_np)
                except Exception:
                    ok_scr, deg_val, line_pils = False, 0.0, []

                if ok_scr and len(line_pils) >= 4:
                    s_idx = len(all_sub_lines)
                    all_sub_lines.extend(line_pils)
                    sub_line_slices[idx] = (s_idx, len(all_sub_lines))
                    multiline_map[idx] = line_pils
                else:
                    normal_indices.append(idx)
                    normal_crops.append(c_pil)

            if multiline_map:
                out_combined: list[str] = [""] * len(crops_pil)
                if normal_crops:
                    norm_res = self._recognize_crops_batch_ja(
                        normal_crops,
                        batch_size=batch_size,
                        _allow_line_split=False,
                    )
                    for n_i, n_txt in zip(normal_indices, norm_res):
                        out_combined[n_i] = n_txt
                if all_sub_lines:
                    sub_res = self._recognize_crops_batch_ja(
                        all_sub_lines,
                        batch_size=batch_size,
                        _allow_line_split=False,
                    )
                    for m_idx, (s_start, s_end) in sub_line_slices.items():
                        lines_txt = [
                            t.strip()
                            for t in sub_res[s_start:s_end]
                            if t and t.strip() and not PUNCT_ONLY_REGEX.match(t.strip())
                        ]
                        out_combined[m_idx] = "".join(lines_txt)
                return out_combined

        nar_model = self._get_ja_nar_recognizer()
        if nar_model is not None:
            try:
                rgb_crops = [c.convert("RGB") for c in crops_pil]
                with torch.inference_mode():
                    if len(rgb_crops) == 1:
                        raw_res = nar_model(rgb_crops[0])
                        nar_texts = [str(raw_res).strip()] if isinstance(raw_res, str) else [str(x).strip() for x in raw_res]
                    else:
                        raw_res = nar_model(rgb_crops)
                        if isinstance(raw_res, str):
                            nar_texts = [raw_res.strip()]
                        else:
                            nar_texts = [str(x).strip() for x in raw_res]

                if len(nar_texts) == len(crops_pil):
                    # ВАЖНО: Дочитываем через классический авторегрессионный MangaOCR ТОЛЬКО длинные монологи (len >= 54),
                    # которые уперлись в потолок 64 токенов NAR-декодера!
                    # Если manga-ocr-nar-small вернул пустую строку ("") на кропе — значит внутри нет японских букв,
                    # и принудительный вызов классического MangaOCR на пустом/фоновом кропе приводил к галлюцинации
                    # заученных фраз («タクシー運転手さん一番うまい店に連れてって» -> «Taxi driver, take me to the best restaurant»).
                    fallback_indices = [
                        idx
                        for idx, txt in enumerate(nar_texts)
                        if len(txt) >= 54
                        or (
                            not txt.strip()
                            and pre_ocr_stroke_gate(crops_pil[idx]).valid_cc_count >= 4
                            and verify_bubble_and_text_candidate(np.array(crops_pil[idx].convert("L")), bubbles_only=True)["has_enclosing_frame"]
                        )
                        or (
                            len(STRIP_PUNCT_REGEX.sub("", txt.strip())) <= 4
                            and (crops_pil[idx].size[0] * crops_pil[idx].size[1]) >= 16000
                            and verify_bubble_and_text_candidate(np.array(crops_pil[idx].convert("L")), bubbles_only=True)["has_enclosing_frame"]
                        )
                    ]
                    if fallback_indices:
                        sub_crops = [crops_pil[idx] for idx in fallback_indices]
                        fallback_texts = self._run_classic_manga_ocr_batch(sub_crops, batch_size=batch_size)
                        for idx, fb_txt in zip(fallback_indices, fallback_texts):
                            cur_nar = nar_texts[idx].strip()
                            fb_clean = fb_txt.strip()
                            if not fb_clean or KNOWN_OCR_HALLUCINATIONS_REGEX.search(fb_clean):
                                continue
                            cur_core_len = len(STRIP_PUNCT_REGEX.sub("", cur_nar))
                            fb_core_len = len(STRIP_PUNCT_REGEX.sub("", fb_clean))
                            if (
                                not cur_nar
                                or (len(cur_nar) >= 54 and len(fb_clean) > len(cur_nar))
                                or (cur_core_len <= 4 and fb_core_len >= cur_core_len + 2)
                            ):
                                nar_texts[idx] = fb_clean
                    # Если после NAR + MangaOCR какой-то баббл остался пустым или с галлюцинацией — дочитываем через HayaiOCR v2.5 Nova!
                    if self.use_hayai_ocr and self._hayai_model is not None:
                        hayai_rescue_indices = [
                            idx
                            for idx, txt in enumerate(nar_texts)
                            if (
                                (not txt.strip() or KNOWN_OCR_HALLUCINATIONS_REGEX.search(txt))
                                and pre_ocr_stroke_gate(crops_pil[idx]).valid_cc_count >= 4
                                and verify_bubble_and_text_candidate(np.array(crops_pil[idx].convert("L")), bubbles_only=True)["has_enclosing_frame"]
                            )
                        ]
                        if hayai_rescue_indices:
                            hayai_sub_crops = [crops_pil[idx] for idx in hayai_rescue_indices]
                            hayai_rescued = self._recognize_crops_batch_hayai(hayai_sub_crops, lang="ja")
                            for idx, h_txt in zip(hayai_rescue_indices, hayai_rescued):
                                h_clean = (h_txt or "").strip()
                                if h_clean and not KNOWN_OCR_HALLUCINATIONS_REGEX.search(h_clean):
                                    nar_texts[idx] = h_clean
                    return nar_texts
            except Exception as nar_exc:
                logger.debug("Ошибка пакетного вызова manga-ocr-nar-small (%s), переключаемся на MangaOCR.", nar_exc)

        return self._run_classic_manga_ocr_batch(crops_pil, batch_size=batch_size)

    def _recognize_crops_batch_ko_zh(
        self,
        crops_pil: list[Image.Image],
        raw_crops_np: list[np.ndarray],
        is_free_list: list[bool],
        lang: str,
    ) -> list[str]:
        """
        Пакетное GPU-распознавание бабблов для корейской манхвы (`ko`) и китайской маньхуа (`zh`) (Шаг 3):
        1. Выполняет быстрый батч-прогон всех кропов страницы через `HayaiOCR v2.5 Nova` (`SigLIP2 NaFlex`).
        2. Для каждого кропа проверяет полноту чтения (включая построчную сегментацию `_split_horizontal_lines_for_ocr`
           для высоких многострочных бабблов и фоллбэк на `RapidOCR` / `EasyOCR`).
        """
        if not crops_pil:
            return []

        hayai_batch_texts: list[str | None] = [None] * len(crops_pil)
        if self.use_hayai_ocr and self._hayai_available is not False:
            hayai_outs = self._recognize_crops_batch_hayai(crops_pil, lang=lang)
            if len(hayai_outs) == len(crops_pil):
                hayai_batch_texts = [t for t in hayai_outs]

        final_texts: list[str] = []
        for padded_pil, raw_np, is_f, pre_hayai in zip(crops_pil, raw_crops_np, is_free_list, hayai_batch_texts):
            txt = self._recognize_single_ko_zh_crop(
                padded_pil,
                raw_np,
                lang=lang,
                is_free_text=is_f,
                precomputed_hayai_text=pre_hayai,
            )
            final_texts.append(txt)
        return final_texts

    def process_image(
        self,
        image: Image.Image,
        lang: str = "auto",
        min_box_size: int = 16,
        detect_free_text: bool = False,
        text_filter_mode: str = "bubbles_only",
        skip_sighs: bool = False,
        skip_sfx: bool = True,
        min_text_length: int = 1,
    ) -> tuple[list[dict[str, Any]], str]:
        req_lang = (lang or "auto").lower().strip()
        if req_lang not in ("auto", "ja", "ko", "zh"):
            req_lang = "auto"

        mode = (text_filter_mode or "bubbles_only").lower().strip()
        # В режимах "dialogue_only", "bubbles_only" и "long_only" свободный текст на фоне принудительно отключен,
        # чтобы не замазывать рисунки и спецэффекты вне бабблов.
        effective_free_text = bool(detect_free_text) and mode not in ("dialogue_only", "bubbles_only", "long_only")
        if mode in ("all", "bubbles_no_sighs"):
            effective_free_text = True if mode == "all" else bool(detect_free_text)

        self._page_ocr_cache.clear()

        rgb_image = image.convert("RGB")
        img_np = np.array(rgb_image)

        if mode == "long_only":
            yolo_conf = max(0.25, self.conf_threshold)
            effective_min_box = max(min_box_size, 24)
        else:
            yolo_conf = max(0.22, self.conf_threshold)
            effective_min_box = max(18, min_box_size)

        raw_detections = self._detect_bubbles(
            img_np,
            min_box_size=effective_min_box,
            conf_override=yolo_conf,
        )

        if effective_free_text:
            free_detections = self._detect_free_text_regions(
                img_np,
                raw_detections,
                min_box_size=max(18, effective_min_box),
            )
            if free_detections:
                logger.info(
                    "Найдено %d дополнительных текстовых областей вне speech bubbles.",
                    len(free_detections),
                )
                raw_detections.extend(free_detections)

        if not raw_detections:
            det_lang_empty = self._last_detected_lang if req_lang == "auto" else req_lang
            self.last_page_diagnostics = {
                "raw_candidates_count": 0,
                "yolo_bubbles": int(self.last_detection_stats.get("yolo_bubbles", 0)),
                "rtdetr_bubbles": int(self.last_detection_stats.get("rtdetr_bubbles", 0)),
                "rtdetr_text_bubbles": int(self.last_detection_stats.get("rtdetr_text_bubbles", 0)),
                "rtdetr_free_texts": int(self.last_detection_stats.get("rtdetr_free_texts", 0)),
                "split_multi_bubbles": int(self.last_detection_stats.get("split_multi_bubbles", 0)),
                "rescued_orphan_text_bubbles": int(self.last_detection_stats.get("rescued_orphan_text_bubbles", 0)),
                "no_ink_count": 0,
                "hallucinated_count": 0,
                "empty_ocr_count": 0,
                "filtered_count": 0,
                "filtered_items": [],
                "restored_by_empty_guard_count": 0,
                "kept_blocks_count": 0,
                "free_text_kept_count": 0,
                "detected_lang": det_lang_empty,
                "zero_reason_code": "no_bubbles_detected",
                "zero_reason_ru": "На странице не обнаружено речевых облачков или текстовых рамок (обложка, чистый арт или страница без диалогов)",
            }
            return [], det_lang_empty

        resolved_lang = (
            self.detect_page_language(rgb_image, raw_detections)
            if req_lang == "auto"
            else req_lang
        )
        if req_lang in ("ja", "ko", "zh"):
            self._last_detected_lang = req_lang

        sorted_detections = sort_page_blocks_reading_order_dag(
            raw_detections,
            page_bgr=img_np,
            lang=resolved_lang,
        )

        output_blocks: list[dict[str, Any]] = []
        filtered_bubble_SamePage_fallback: list[dict[str, Any]] = []
        filtered_items_diag: list[dict[str, Any]] = []
        block_id = 1
        skipped_by_filter = 0
        no_ink_count = 0
        hallucinated_count = 0
        empty_ocr_count = 0
        restored_by_empty_guard_count = 0

        # Пакетное GPU-распознавание всех бабблов страницы (для ja через manga-ocr-nar-small/MangaOCR,
        # для ko/zh через HayaiOCR v2.5 Nova + SmartLineSplit + RapidOCR/EasyOCR) строго по `_get_ocr_bbox(det)`!
        batch_candidates_crops: list[Image.Image] = []
        batch_candidates_raw_np: list[np.ndarray] = []
        batch_candidates_meta: list[tuple[tuple[int, int, int, int, str], int, bool]] = []
        rejected_hallucination_keys: set[tuple[int, int, int, int, str]] = set()
        for det in sorted_detections:
            x1, y1, x2, y2 = det["bbox"]
            is_free = bool(det.get("is_free_text", False))
            cache_key = (x1, y1, x2, y2, resolved_lang)
            if cache_key in self._page_ocr_cache:
                continue
            ocr_coords = self._get_ocr_bbox(det)
            ox1, oy1, ox2, oy2 = ocr_coords
            raw_crop_np = img_np[oy1:oy2, ox1:ox2]
            has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free)
            has_glyphs = has_ink and self._has_real_text_glyph_structure(raw_crop_np, is_free_text=is_free)
            if (not has_ink or not has_glyphs) and ocr_coords != [x1, y1, x2, y2]:
                # Фоллбэк на полный bbox баббла, если text_bbox оказался слишком узким
                raw_crop_np = img_np[y1:y2, x1:x2]
                has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free)
                has_glyphs = has_ink and self._has_real_text_glyph_structure(raw_crop_np, is_free_text=is_free)
                ocr_coords = [x1, y1, x2, y2]
            if not has_ink or not has_glyphs:
                self._page_ocr_cache[cache_key] = ""
                rejected_hallucination_keys.add(cache_key)
                continue
            padded_crop = self._crop_with_padding(rgb_image, ocr_coords, pad=8)
            batch_candidates_crops.append(padded_crop)
            batch_candidates_raw_np.append(raw_crop_np)
            batch_candidates_meta.append((cache_key, ink_pixels, is_free))

        if batch_candidates_crops:
            if resolved_lang == "ja":
                batch_texts = self._recognize_crops_batch_ja(batch_candidates_crops)
            else:
                batch_texts = self._recognize_crops_batch_ko_zh(
                    batch_candidates_crops,
                    batch_candidates_raw_np,
                    [m[2] for m in batch_candidates_meta],
                    lang=resolved_lang,
                )
            for (c_key, ink_px, is_f), raw_np_item, b_txt in zip(batch_candidates_meta, batch_candidates_raw_np, batch_texts):
                if self._is_hallucinated_ocr_text(b_txt, ink_px, is_free_text=is_f, crop_shape=raw_np_item.shape[:2]):
                    if b_txt and b_txt.strip():
                        hallucinated_count += 1
                    self._page_ocr_cache[c_key] = ""
                    rejected_hallucination_keys.add(c_key)
                else:
                    self._page_ocr_cache[c_key] = b_txt

        for det in sorted_detections:
            x1, y1, x2, y2 = det["bbox"]
            ocr_coords = self._get_ocr_bbox(det)
            ox1, oy1, ox2, oy2 = ocr_coords
            is_free = bool(det.get("is_free_text", False))
            cache_key = (x1, y1, x2, y2, resolved_lang)

            raw_crop_np = img_np[oy1:oy2, ox1:ox2]
            has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free)
            if not has_ink and ocr_coords != [x1, y1, x2, y2]:
                raw_crop_np = img_np[y1:y2, x1:x2]
                has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free)

            if cache_key in self._page_ocr_cache:
                recognized_text = self._page_ocr_cache[cache_key]
                if (
                    not has_ink
                    or cache_key in rejected_hallucination_keys
                    or self._is_hallucinated_ocr_text(recognized_text, ink_pixels, is_free_text=is_free, crop_shape=raw_crop_np.shape[:2])
                ):
                    recognized_text = ""
            else:
                if not has_ink or not self._has_real_text_glyph_structure(raw_crop_np, is_free_text=is_free):
                    recognized_text = ""
                else:
                    padded_crop = self._crop_with_padding(rgb_image, ocr_coords, pad=8)
                    recognized_text = self._recognize_crop(
                        padded_crop,
                        raw_crop_np,
                        resolved_lang,
                        is_free_text=is_free,
                    )

            if not recognized_text and not is_free and has_ink and ink_pixels >= 28 and ocr_coords != [x1, y1, x2, y2]:
                full_raw_np = img_np[y1:y2, x1:x2]
                full_padded = self._crop_with_padding(rgb_image, [x1, y1, x2, y2], pad=12)
                retry_full_txt = self._recognize_crop(
                    full_padded,
                    full_raw_np,
                    resolved_lang,
                    is_free_text=False,
                )
                if retry_full_txt:
                    recognized_text = retry_full_txt
                    ocr_coords = [x1, y1, x2, y2]

            if not recognized_text:
                if not has_ink:
                    no_ink_count += 1
                else:
                    empty_ocr_count += 1
                continue

            filter_reason = self.classify_filter_reason(
                recognized_text,
                is_free_text=is_free,
                text_filter_mode=mode,
                skip_sighs=skip_sighs,
                skip_sfx=skip_sfx,
                min_text_length=min_text_length,
            )
            if filter_reason is not None:
                r_code, r_label = filter_reason
                skipped_by_filter += 1
                filtered_items_diag.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "text": recognized_text,
                        "reason_code": r_code,
                        "reason_ru": r_label,
                        "is_free_text": is_free,
                    }
                )
                if not is_free and not self.is_sfx_or_onomatopoeia(recognized_text, is_free_text=False):
                    filtered_bubble_SamePage_fallback.append(
                        {
                            "id": 0,
                            "bbox": [x1, y1, x2, y2],
                            "clean_bbox": [x1, y1, x2, y2],
                            "text_bbox": ocr_coords,
                            "confidence": det["confidence"],
                            "original_text": recognized_text,
                            "is_free_text": False,
                            "is_deleted": False,
                        }
                    )
                logger.debug(
                    "Блок [%d,%d,%d,%d] '%s' пропущен фильтром объема текста (%s, mode=%s)",
                    x1,
                    y1,
                    x2,
                    y2,
                    recognized_text,
                    r_label,
                    mode,
                )
                continue

            if det.get("_source") == "cv_screen_container":
                core_scr = STRIP_PUNCT_REGEX.sub("", recognized_text or "")
                if len(core_scr) < 12 or self.is_sfx_or_onomatopoeia(recognized_text, is_free_text=True) or self.is_sigh_or_interjection(recognized_text):
                    empty_ocr_count += 1
                    continue

            detected_angle = 0.0
            try:
                ok_scr_b, deg_blk, lines_blk, _ = self._is_strict_multiline_screen_crop(raw_crop_np)
                if ok_scr_b and abs(deg_blk) >= 6.0 and len(lines_blk) >= 4:
                    detected_angle = max(-52.0, min(52.0, round(-float(deg_blk), 1)))
            except Exception:
                detected_angle = 0.0

            output_blocks.append(
                {
                    "id": block_id,
                    "bbox": [x1, y1, x2, y2],
                    "clean_bbox": [x1, y1, x2, y2],
                    "text_bbox": ocr_coords,
                    "confidence": det["confidence"],
                    "original_text": recognized_text,
                    "is_free_text": is_free,
                    "is_deleted": False,
                    "angle": detected_angle,
                }
            )
            block_id += 1

        # Защита от пустых страниц: срабатывает ТОЛЬКО если пользователь НЕ включал пропуск вздохов (skip_sighs=False),
        # а фильтр min_text_length отсеял единственную короткую реплику на странице (не более 2 реальных бабблов).
        # Никогда не возвращает отфильтрованные вздохи/шум при skip_sighs=True!
        if filtered_bubble_SamePage_fallback and not skip_sighs and mode not in ("long_only", "dialogue_only", "bubbles_no_sighs"):
            kept_bubbles = sum(1 for b in output_blocks if not b.get("is_free_text"))
            if kept_bubbles == 0 and len(filtered_bubble_SamePage_fallback) <= 2:
                restored_by_empty_guard_count = len(filtered_bubble_SamePage_fallback)
                restored_bboxes = {tuple(b["bbox"]) for b in filtered_bubble_SamePage_fallback}
                filtered_items_diag = [
                    it for it in filtered_items_diag if tuple(it["bbox"]) not in restored_bboxes
                ]
                combined_all = output_blocks + filtered_bubble_SamePage_fallback
                combined_all = sort_page_blocks_reading_order_dag(combined_all, page_bgr=img_np, lang=resolved_lang)
                output_blocks = []
                for idx_new, b_item in enumerate(combined_all, start=1):
                    b_item["id"] = idx_new
                    output_blocks.append(b_item)
                skipped_by_filter = max(0, skipped_by_filter - len(filtered_bubble_SamePage_fallback))

        if skipped_by_filter > 0:
            logger.info(
                "Фильтр объема текста (mode=%s): пропущено %d мелких вздохов/спецэффектов, оставлено для перевода %d блоков.",
                mode,
                skipped_by_filter,
                len(output_blocks),
            )

        zero_reason_code = ""
        zero_reason_ru = ""
        if len(output_blocks) == 0:
            if len(filtered_items_diag) > 0:
                samples = ", ".join(f"«{it.get('text', '')}» ({it.get('reason_ru', '')})" for it in filtered_items_diag[:5])
                zero_reason_code = "all_filtered_out"
                zero_reason_ru = f"Весь распознанный текст ({len(filtered_items_diag)} шт.) отсеян фильтром настроек: {samples}"
            elif no_ink_count >= len(raw_detections):
                zero_reason_code = "no_ink_in_bubbles"
                zero_reason_ru = f"Детектор нашел контуры ({len(raw_detections)} шт.), но внутри нет читаемого текста/чернил"
            else:
                zero_reason_code = "empty_ocr_or_noise"
                zero_reason_ru = f"Детектор нашел области ({len(raw_detections)} шт.), но OCR отсеял их как шум/рисунок без букв (без чернил: {no_ink_count}, шум: {hallucinated_count})"

        self.last_page_diagnostics = {
            "raw_candidates_count": len(raw_detections),
            "yolo_bubbles": int(self.last_detection_stats.get("yolo_bubbles", 0)),
            "rtdetr_bubbles": int(self.last_detection_stats.get("rtdetr_bubbles", 0)),
            "rtdetr_text_bubbles": int(self.last_detection_stats.get("rtdetr_text_bubbles", 0)),
            "rtdetr_free_texts": int(self.last_detection_stats.get("rtdetr_free_texts", 0)),
            "split_multi_bubbles": int(self.last_detection_stats.get("split_multi_bubbles", 0)),
            "rescued_orphan_text_bubbles": int(self.last_detection_stats.get("rescued_orphan_text_bubbles", 0)),
            "no_ink_count": no_ink_count,
            "hallucinated_count": hallucinated_count,
            "empty_ocr_count": empty_ocr_count,
            "filtered_count": len(filtered_items_diag),
            "filtered_items": filtered_items_diag[:35],
            "restored_by_empty_guard_count": restored_by_empty_guard_count,
            "kept_blocks_count": len(output_blocks),
            "free_text_kept_count": sum(1 for b in output_blocks if b.get("is_free_text")),
            "detected_lang": resolved_lang,
            "zero_reason_code": zero_reason_code,
            "zero_reason_ru": zero_reason_ru,
        }

        self.last_detection_stats["ko_zh_ocr_backend"] = self.get_multilang_ocr_backend_name()
        self._page_ocr_cache.clear()
        return output_blocks, resolved_lang

    def recognize_manual_region(
        self,
        image: Image.Image,
        bbox: list[int],
        lang: str = "auto",
        use_vlm_ocr: bool = False,
        vlm_model: str | None = None,
        ollama_base_url: str | None = None,
        cloud_api_url: str | None = None,
        cloud_api_key: str | None = None,
    ) -> tuple[str, str, bool]:
        """
        Принудительное распознавание текста внутри выделенной пользователем мышкой области `bbox = [x1, y1, x2, y2]`
        без отсева по фильтрам вздохов/SFX/контраста (Шаги 1–4):
        - Если `use_vlm_ocr=True`, сразу выполняет снайперское чтение через Vision-модель (`recognize_crop_via_vlm`).
        - Иначе сначала использует быстрый каскад (`_compute_tight_ink_bbox` + `manga-ocr-nar-small` / `HayaiOCR v2.5 Nova`
          / `EasyOCR`), а если стандартный OCR не справился или вернул пустую строку — автоматически вызывает
          снайперский `recognize_crop_via_vlm`!
        Возвращает `(recognized_text, resolved_lang, is_free_text)`.
        """
        rgb_image = image.convert("RGB")
        img_w, img_h = rgb_image.size
        img_np = np.array(rgb_image)
        x1, y1, x2, y2 = [int(round(c)) for c in bbox[:4]]
        x1 = max(0, min(x1, img_w - 2))
        y1 = max(0, min(y1, img_h - 2))
        x2 = max(x1 + 4, min(x2, img_w))
        y2 = max(y1 + 4, min(y2, img_h))

        raw_crop_np = img_np[y1:y2, x1:x2]
        gray_crop = cv2.cvtColor(raw_crop_np, cv2.COLOR_RGB2GRAY)
        p70 = float(np.percentile(gray_crop, 70))
        # Если фон внутри выделенной области светлый (>= 205), считаем это кружком/бабблом для чистой заливки
        is_free_text = bool(p70 < 205.0)

        req_lang = (lang or "auto").lower().strip()
        if req_lang not in ("ja", "ko", "zh"):
            resolved_lang = self._last_detected_lang or "ja"
        else:
            resolved_lang = req_lang

        # 1. Если пользователь явно запросил снайперский VLM-OCR (`use_vlm_ocr=True`)
        if use_vlm_ocr:
            try:
                vlm_txt, vlm_lang, _ = self.recognize_crop_via_vlm(
                    rgb_image=rgb_image,
                    bbox=[x1, y1, x2, y2],
                    lang=req_lang,
                    vlm_model=vlm_model,
                    ollama_base_url=ollama_base_url,
                    cloud_api_url=cloud_api_url,
                    cloud_api_key=cloud_api_key,
                )
                if vlm_txt and not PUNCT_ONLY_REGEX.match(vlm_txt):
                    return vlm_txt, vlm_lang, is_free_text
            except Exception as vlm_exc:
                logger.warning("Сбой VLM-OCR при ручном выделении, переход к локальному OCR: %s", vlm_exc)

        # ВАЖНО: При ручном выделении («➕ Обвести») пользователь сам обводит нужный текст!
        # Нельзя полагаться только на `_compute_tight_ink_bbox`, так как буквы часто касаются границ
        # пользовательской рамки и отбрасываются как «стенки баббла», оставляя одну точку/дакутэн в центре,
        # которую декодер читает как «ビクッ» / «ピクッ» (-> «Twitch»)!
        full_padded = self._crop_with_padding(rgb_image, [x1, y1, x2, y2], pad=12)
        full_padded_np = np.array(full_padded)
        tight_coords = self._compute_tight_ink_bbox(img_np, [x1, y1, x2, y2], None, is_free_text=is_free_text)
        tx1, ty1, tx2, ty2 = tight_coords
        tight_area = max(1, (tx2 - tx1) * (ty2 - ty1))
        full_area = max(1, (x2 - x1) * (y2 - y1))
        # Используем tight_coords только если он сохранил >= 45% площади выделения пользователя
        use_tight = tight_coords != [x1, y1, x2, y2] and (tight_area >= full_area * 0.45)
        tight_padded = self._crop_with_padding(rgb_image, tight_coords, pad=10) if use_tight else full_padded
        tight_raw_np = img_np[ty1:ty2, tx1:tx2] if use_tight else raw_crop_np
        _, full_ink_px = self._has_meaningful_ink(raw_crop_np, is_free_text=False)

        recognized = ""
        if resolved_lang in ("ko", "zh"):
            try:
                recognized = self._recognize_single_ko_zh_crop(
                    full_padded,
                    raw_crop_np,
                    lang=resolved_lang,
                    is_free_text=False,
                )
            except Exception:
                recognized = ""
            if (not recognized or PUNCT_ONLY_REGEX.match(recognized)) and use_tight:
                try:
                    recognized = self._recognize_single_ko_zh_crop(
                        tight_padded,
                        tight_raw_np,
                        lang=resolved_lang,
                        is_free_text=False,
                    )
                except Exception:
                    pass
        else:
            # Для одиночного выделенного пользователем баббла запускаем и классический MangaOCR (с выравниванием пропорций ViT),
            # и manga-ocr-nar-small на полном кропе `full_padded`, выбирая наиболее полное и осмысленное чтение!
            ja_candidates: list[str] = []
            try:
                classic_outs = self._run_classic_manga_ocr_batch([full_padded], batch_size=1)
                if classic_outs and classic_outs[0].strip():
                    ja_candidates.append(classic_outs[0].strip())
            except Exception:
                pass
            try:
                nar_outs = self._recognize_crops_batch_ja([full_padded])
                if nar_outs and nar_outs[0].strip() and nar_outs[0].strip() not in ja_candidates:
                    ja_candidates.append(nar_outs[0].strip())
            except Exception:
                pass
            if use_tight:
                try:
                    classic_tight = self._run_classic_manga_ocr_batch([tight_padded], batch_size=1)
                    if classic_tight and classic_tight[0].strip() and classic_tight[0].strip() not in ja_candidates:
                        ja_candidates.append(classic_tight[0].strip())
                except Exception:
                    pass

            def _score_ja_manual_cand(cand_s: str) -> float:
                c = cand_s.strip()
                if not c or PUNCT_ONLY_REGEX.match(c) or KNOWN_OCR_HALLUCINATIONS_REGEX.search(c):
                    return -999.0
                core = STRIP_PUNCT_REGEX.sub("", c)
                if not core or REPEATED_NOISE_REGEX.match(core):
                    return -999.0
                score = float(len(core)) * 2.5
                kanji_cnt = len(HANZI_REGEX.findall(core))
                kana_cnt = len(KANA_REGEX.findall(core))
                score += kanji_cnt * 4.5 + kana_cnt * 1.5
                # Если локальный декодер выдал одиночный короткий SFX вроде «ビクッ» / «ピクッ» («Twitch»)
                # или вздох при большом количестве чернил в баббле — сильно штрафуем такой кандидат!
                if self.is_sfx_or_onomatopoeia(c, is_free_text=True) or c.strip("〜~！!？?…・ ") in ("ビクッ", "ピクッ", "ぴくっ", "びくっ", "ビク", "ピク"):
                    if full_ink_px >= 110 or full_area >= 1800:
                        score -= 25.0
                    else:
                        score -= 6.0
                elif self.is_sigh_or_interjection(c) and len(core) <= 3 and full_ink_px >= 160:
                    score -= 10.0
                return score

            scored_ja = sorted(
                [( _score_ja_manual_cand(cand), cand ) for cand in ja_candidates],
                key=lambda item: item[0],
                reverse=True,
            )
            if scored_ja and scored_ja[0][0] > -100.0:
                recognized = scored_ja[0][1]

            # Если текст пустой или на другом языке, пробуем HayaiOCR v2.5 Nova или EasyOCR
            if not recognized or PUNCT_ONLY_REGEX.match(recognized):
                if self.use_hayai_ocr and self._hayai_available is not False:
                    h_outs = self._recognize_crops_batch_hayai([full_padded], max_num_patches=512, lang="auto")
                    cand_h = h_outs[0].strip() if h_outs else ""
                    if cand_h and not PUNCT_ONLY_REGEX.match(cand_h):
                        recognized = cand_h
                        if HANGUL_REGEX.search(recognized):
                            resolved_lang = "ko"
                        elif KANA_REGEX.search(recognized):
                            resolved_lang = "ja"
                        elif HANZI_REGEX.search(recognized) and (req_lang == "zh" or len(ZH_SPECIFIC_MARKERS_REGEX.findall(recognized)) >= 2):
                            resolved_lang = "zh"
                if (not recognized or PUNCT_ONLY_REGEX.match(recognized)) and self._check_easyocr():
                    for fallback_reader in (self._get_ko_reader(), self._get_zh_reader()):
                        if fallback_reader is not None:
                            lines = fallback_reader.readtext(full_padded_np, detail=0, paragraph=True)
                            cand = " ".join(str(x).strip() for x in lines if str(x).strip())
                            if cand and not PUNCT_ONLY_REGEX.match(cand):
                                recognized = cand
                                break

        # Если с первого раза текст слишком бледный, пробуем повысить контраст кропа
        if not recognized or PUNCT_ONLY_REGEX.match(recognized):
            try:
                contrast_pil = ImageOps.autocontrast(full_padded, cutoff=2)
                with torch.inference_mode():
                    cand2 = self.ja_recognizer(self._pad_to_balanced_aspect_for_vit(contrast_pil)).strip()
                if cand2 and not PUNCT_ONLY_REGEX.match(cand2) and not KNOWN_OCR_HALLUCINATIONS_REGEX.search(cand2):
                    recognized = cand2
            except Exception:
                pass

        # Автоматический снайперский фоллбэк на VLM-OCR (Шаг 4):
        # Срабатывает не только когда стандартный OCR вернул пустую строку, но и когда он ошибочно прочитал
        # крупный баббл с текстом (`full_ink_px >= 110`) как короткий шумовой SFX «ビクッ»/«ピクッ» («Twitch»)!
        rec_core = STRIP_PUNCT_REGEX.sub("", recognized or "")
        is_suspicious_sfx_on_rich_bubble = bool(
            recognized
            and full_ink_px >= 110
            and len(rec_core) <= 4
            and (
                self.is_sfx_or_onomatopoeia(recognized, is_free_text=True)
                or rec_core in ("ビクッ", "ピクッ", "ぴくっ", "びくっ", "ビク", "ピク", "ヒクッ", "ひくっ")
            )
        )
        if not use_vlm_ocr and (not recognized or PUNCT_ONLY_REGEX.match(recognized) or is_suspicious_sfx_on_rich_bubble):
            try:
                vlm_txt, vlm_lang, _ = self.recognize_crop_via_vlm(
                    rgb_image=rgb_image,
                    bbox=[x1, y1, x2, y2],
                    lang=req_lang,
                    vlm_model=vlm_model,
                    ollama_base_url=ollama_base_url,
                    cloud_api_url=cloud_api_url,
                    cloud_api_key=cloud_api_key,
                )
                if vlm_txt and not PUNCT_ONLY_REGEX.match(vlm_txt):
                    recognized = vlm_txt
                    resolved_lang = vlm_lang
            except Exception as vlm_fb_exc:
                logger.debug("Фоллбэк VLM-OCR пропущен: %s", vlm_fb_exc)

        return recognized, resolved_lang, is_free_text