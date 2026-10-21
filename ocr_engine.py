"""
Мультиязычный модуль детекции бабблов, свободных надписей и распознавания текста (v1.0.0: JA / KO / ZH + AUTO).

Изменения v1.0.0:
1. Добавлена гибкая фильтрация объема переводимого текста на странице (`text_filter_mode`, `skip_sighs`,
   `skip_sfx`, `min_text_length`):
   - Автоматический пропуск коротких вздохов, стонов и одиночных междометий («а», «о», «ах», «ох», «мм», «ха»,
     «э», «あ», «あっ», «お», «んっ», «はぁ» и т.п.), чтобы программа не замазывала мелкие облачка и рисунок.
   - Фильтрация звуковых спецэффектов и ономатопеи (SFX катаканой/хангылем на фоне и в экшен-сценах).
   - Настраиваемый порог минимальной длины текста (`min_text_length`) и режим «Минимум замазки (только длинные фразы)».
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from manga_ocr import MangaOcr
from PIL import Image, ImageOps
from ultralytics import YOLO

logger = logging.getLogger(__name__)

HF_YOLO_REPO = "ogkalu/comic-speech-bubble-detector-yolov8m"
HF_YOLO_FILENAME = "comic-speech-bubble-detector.pt"

HANGUL_REGEX = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
KANA_REGEX = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
KATAKANA_ONLY_REGEX = re.compile(r"^[\u30a0-\u30ff]+$")
HANZI_REGEX = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
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

# Словарь ТОЛЬКО чистых вздохов, стонов и охов (без диалоговых слов типа いや / やだ / おい / うん!)
SIGH_EXACT_SET: set[str] = {
    # Японские вздохи и стоны (хирагана)
    "あ", "あっ", "ああ", "ああっ", "あぁ", "あぁっ", "あん", "あんっ", "ああん", "あう", "あうっ", "あふ", "あふっ",
    "お", "おっ", "おお", "おおっ", "おぉ", "おぉっ",
    "う", "うっ", "うぅ", "うぅっ", "うう", "ううっ", "うあ", "うあっ",
    "え", "えっ", "えぇ", "えぇっ",
    "い", "いっ",
    "ん", "んっ", "んん", "んんっ", "んぅ", "んぅっ", "んあ", "んあっ", "んお", "んふ", "んふっ", "んく", "んぐ",
    "は", "はっ", "はぁ", "はぁっ", "はぁはぁ", "はあ", "はあっ", "はあはあ", "はふ", "はふっ",
    "ふ", "ふっ", "ふぅ", "ふぅっ", "ふう", "ふうっ", "ふあ", "ふわ", "ふわぁ",
    "ひ", "ひっ", "ひゃ", "ひゃっ", "ひゃあ", "ひゃん", "ひぃ", "ひぃっ",
    "きゃ", "きゃっ", "きゃあ", "きゃあっ",
    "く", "くっ", "くぅ", "くぅっ", "ぐ", "ぐっ", "ぐぅ",
    "ち", "ちっ", "む", "むっ", "むぅ", "ぷ", "ぷっ",
    "わ", "わっ", "へ", "へっ", "ほ", "ほっ", "ほぉ",
    # Японские вздохи (катакана)
    "ア", "アッ", "アア", "アアッ", "アァ", "アァッ", "アン", "アンッ",
    "オ", "オッ", "オオ", "オオッ", "オォ",
    "ウ", "ウッ", "ウゥ", "ウゥッ",
    "エ", "エッ", "エェ",
    "ン", "ンッ", "ンン", "ンンッ", "ンゥ", "ンア",
    "ハ", "ハッ", "ハァ", "ハァッ", "ハァハァ", "ハア", "ハアッ",
    "フ", "フッ", "フゥ", "フゥッ", "フウ",
    "ヒ", "ヒッ", "ヒャ", "ヒャッ", "ヒャア", "ヒィ", "ヒィッ",
    "キャ", "キャッ", "キャア",
    "ク", "クッ", "クゥ", "グ", "グッ", "チ", "チッ",
    "ム", "ムッ", "プ", "プッ", "ワ", "ワッ", "ヘ", "ヘッ", "ホ", "ホッ", "ホォ",
    # Корейские вздохи
    "아", "아앗", "아아", "어", "어엇", "어어", "오", "오옷", "오오",
    "우", "우웃", "으", "으음", "음", "하", "하아", "하앗", "후", "후우", "후웃",
    "헉", "흡", "윽", "앗", "엇", "헛", "흑", "흐윽", "흐음", "캬", "꺄", "꺄아", "히익", "익",
    # Китайские вздохи
    "啊", "啊啊", "哦", "哦哦", "噢", "嗯", "嗯嗯", "呃", "唔", "哈", "呼", "呀", "哎", "诶", "咦", "嘶", "哼", "哇", "喔",
    # Русские и английские короткие вздохи
    "а", "о", "э", "у", "и", "ы", "м", "мм", "ммм", "ах", "аах", "ахн", "ох", "оох", "ух", "уух", "эх", "ээх",
    "ха", "хаа", "фу", "фух", "уф", "кья", "хья", "хм", "хмм", "пф", "пфф", "тц",
    "a", "o", "e", "u", "m", "mm", "mmm", "mmh", "ah", "aah", "ahn", "oh", "ooh", "uh", "eh", "ha", "haa",
    "phew", "kya", "kyaa", "hya", "eek", "ugh", "gah", "ngh", "hm", "hmm", "tch", "tsk", "huh", "wah",
}

# Слова из гласных, которые являются осмысленными репликами диалога, а НЕ вздохами
DIALOGUE_VOWEL_WORDS_JA: set[str] = {
    "おい", "オイ", "いい", "イイ", "いいえ", "イイエ", "うん", "ウン",
    "ええ", "エエ", "おう", "オウ", "あい", "アイ", "うえ", "ウエ",
    "あおい", "アオイ", "おおい", "オオイ", "いう", "イウ", "あう", "アウ",
}

# Регулярное выражение для комбинаций вздохов (например, «あ…あっ», «はぁ…はぁ…», «んっ…あ», «あああ»)
SIGH_COMBO_REGEX = re.compile(
    r"^(?:[あぁアァ]+[っッー]*|[おぉオォ]+[っッー]*|[うぅウゥ]+[っッー]*|[えぇエェ]+[っッー]*|[んン]+[っッー]*|"
    r"はぁ|はあ|ハァ|ハア|ふぅ|ふう|フゥ|フウ|ふっ|フッ|はっ|ハッ|"
    r"あん|アン|あっ|アッ|おっ|オッ|えっ|エッ|うっ|ウッ|んっ|ンッ|"
    r"きゃ|キャ|ひゃ|ヒャ|くっ|クッ|ぐっ|グッ|ちっ|チッ|わっ|ワッ|"
    r"ほっ|ホッ|へっ|ヘッ|あふ|アフ|んふ|ンフ|うふ|ウフ)+$"
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
    """Мультиязычный движок: детекция бабблов (YOLOv8) + свободного текста + OCR (MangaOCR / EasyOCR)."""

    def __init__(
        self,
        models_dir: str | Path = "models",
        device: str | None = None,
        conf_threshold: float = 0.22,
        iou_threshold: float = 0.45,
        imgsz: int = 1280,
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

        logger.info("Инициализация MangaOCREngine (v1.0.0) на устройстве: %s", self.device)
        self.detector = self._load_detector()
        self.ja_recognizer = self._load_manga_ocr()

        self._ko_reader: Any | None = None
        self._zh_reader: Any | None = None
        self._craft_reader: Any | None = None
        self._easyocr_available: bool | None = None

        self._page_ocr_cache: dict[tuple[int, int, int, int, str], str] = {}
        self._last_detected_lang: str = "ja"

        logger.info("Базовые модели (YOLOv8 + MangaOCR) успешно загружены.")

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

    def _load_manga_ocr(self) -> MangaOcr:
        force_cpu = not self.device.startswith("cuda")
        return MangaOcr(force_cpu=force_cpu)

    def _check_easyocr(self) -> bool:
        if self._easyocr_available is None:
            try:
                import easyocr  # noqa: F401

                self._easyocr_available = True
            except ImportError:
                logger.warning(
                    "Библиотека easyocr не установлена! Для корейского, китайского и CRAFT-детекции запустите install_multilang_ocr.bat"
                )
                self._easyocr_available = False
        return self._easyocr_available

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
    def _crop_with_padding(
        rgb_image: Image.Image,
        bbox: tuple[int, int, int, int] | list[int],
        pad: int = 8,
    ) -> Image.Image:
        img_w, img_h = rgb_image.size
        x1, y1, x2, y2 = bbox
        cx1 = max(0, x1 - 4)
        cy1 = max(0, y1 - 4)
        cx2 = min(img_w, x2 + 4)
        cy2 = min(img_h, y2 + 4)
        crop = rgb_image.crop((cx1, cy1, cx2, cy2))
        if pad > 0:
            crop = ImageOps.expand(crop, border=pad, fill=(255, 255, 255))
        return crop

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
    def _is_hallucinated_ocr_text(text: str, ink_pixels: int, is_free_text: bool = False) -> bool:
        clean = text.strip()
        if not clean:
            return True
        if PUNCT_ONLY_REGEX.match(clean):
            return True
        if not MEANINGFUL_CHAR_REGEX.search(clean):
            return True
        if ink_pixels < 38 and len(clean) > 4:
            return True

        if is_free_text:
            norm = re.sub(r"\s+", "", clean)
            if len(norm) <= 1 and ink_pixels < 140:
                return True
            if REPEATED_NOISE_REGEX.match(norm):
                return True
            if ink_pixels < 85 and len(norm) >= 8:
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

        # Если строка состоит только из вздохов/стонов (до 7 символов без кандзи)
        if len(core) <= 7 and not HANZI_REGEX.search(core):
            if SIGH_COMBO_REGEX.match(core) or SIGH_COMBO_REGEX.match(collapsed):
                return True

        # Одиночный символ хираганы/катаканы/хангыля/латиницы/кириллицы (не кандзи!) — почти всегда звук/вздох
        if len(core) == 1 and not HANZI_REGEX.search(core):
            return True

        return False

    @classmethod
    def is_sfx_or_onomatopoeia(cls, text: str, is_free_text: bool = False) -> bool:
        """
        Определяет, является ли блок звуковым спецэффектом (SFX / ономатопеей).
        ВАЖНО: Внутри кружков/бабблов (`is_free_text=False`) короткие слова катаканой (マジ, ダメ, ウソ, バカ, イヤ, ナニ, ココ)
        являются живыми репликами персонажей, поэтому внутри бабблов отсекаются ТОЛЬКО явные звуки из `SFX_EXACT_SET`!
        """
        if not text:
            return True
        core = STRIP_PUNCT_REGEX.sub("", text.strip())
        if not core:
            return True
        if core in SFX_EXACT_SET or core.lower() in SFX_EXACT_SET:
            return True

        collapsed = re.sub(r"(.)\1{2,}", r"\1", core)
        if collapsed in SFX_EXACT_SET:
            return True

        # Для текста ВНЕ бабблов (на фоне) отсекаем звуковую ономатопею катаканой и повторы слогов
        if is_free_text and not HANZI_REGEX.search(core):
            if len(core) <= 8 and REDUPLICATED_SFX_REGEX.match(core):
                return True
            if KATAKANA_ONLY_REGEX.match(core) and len(core) <= 6:
                return True

        return False

    @classmethod
    def should_filter_out_block(
        cls,
        text: str,
        is_free_text: bool = False,
        text_filter_mode: str = "dialogue_only",
        skip_sighs: bool = True,
        skip_sfx: bool = True,
        min_text_length: int = 2,
    ) -> bool:
        """
        Проверяет, нужно ли пропустить (не замазывать и не переводить) данный текстовый блок
        согласно настройкам объема перевода.
        """
        core = STRIP_PUNCT_REGEX.sub("", (text or "").strip())
        if not core:
            return True

        mode = (text_filter_mode or "dialogue_only").lower().strip()
        eff_skip_sighs = bool(skip_sighs) or mode in ("dialogue_only", "bubbles_no_sighs", "long_only")
        eff_skip_sfx = bool(skip_sfx) or mode in ("dialogue_only", "bubbles_no_sighs", "bubbles_only", "long_only")
        eff_min_len = max(1, int(min_text_length or 1))
        if mode == "long_only":
            eff_min_len = max(eff_min_len, 4)
        elif mode == "all" and not skip_sighs and not skip_sfx and min_text_length <= 1:
            eff_min_len = 1

        if mode in ("dialogue_only", "bubbles_only", "long_only") and is_free_text:
            return True

        # Одиночный кандзи внутри баббла (например 何 / 誰 / 嘘 / 嫌 / 悪) — это полноценное слово, никогда не отбрасываем при min_len <= 2!
        has_kanji = bool(HANZI_REGEX.search(core))
        effective_char_len = len(core) + (1 if (has_kanji and len(core) == 1 and eff_min_len <= 2) else 0)
        if effective_char_len < eff_min_len:
            return True

        if eff_skip_sighs and cls.is_sigh_or_interjection(text):
            return True

        if eff_skip_sfx and cls.is_sfx_or_onomatopoeia(text, is_free_text=is_free_text):
            return True

        return False

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

    @staticmethod
    def _read_easyocr_crop_with_stats(
        reader: Any,
        crop_np: np.ndarray,
        target_regex: re.Pattern[str],
        min_conf: float = 0.15,
    ) -> tuple[str, int, float]:
        res = reader.readtext(crop_np, detail=1, paragraph=False)
        texts: list[str] = []
        total_chars = 0
        weighted_conf_sum = 0.0

        for item in res:
            if len(item) < 3:
                continue
            _, raw_txt, conf = item[0], str(item[1]).strip(), float(item[2])
            if not raw_txt or conf < min_conf:
                continue
            texts.append(raw_txt)
            matched_chars = len(target_regex.findall(raw_txt))
            if matched_chars > 0:
                total_chars += matched_chars
                weighted_conf_sum += conf * matched_chars

        joined_text = " ".join(texts).strip()
        avg_conf = (weighted_conf_sum / total_chars) if total_chars > 0 else 0.0
        return joined_text, total_chars, avg_conf

    def detect_page_language(
        self,
        rgb_image: Image.Image,
        raw_detections: list[dict[str, Any]],
    ) -> str:
        if not raw_detections or not self._check_easyocr():
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
        )[:6]

        crops_data: list[tuple[tuple[int, int, int, int], Image.Image, np.ndarray, bool]] = []
        vertical_votes = 0
        horizontal_votes = 0

        for det in top_boxes:
            x1, y1, x2, y2 = det["bbox"]
            bbox_key = (x1, y1, x2, y2)
            raw_crop_np = np.array(rgb_image.crop(bbox_key))
            has_ink, _ = self._has_meaningful_ink(raw_crop_np, is_free_text=bool(det.get("is_free_text", False)))
            if not has_ink:
                continue
            padded_pil = self._crop_with_padding(rgb_image, bbox_key, pad=8)
            padded_np = np.array(padded_pil)
            is_vert = self._is_vertical_text_crop(raw_crop_np)
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

        if vertical_votes >= horizontal_votes:
            _evaluate_ja()
            ja_total = kana_chars + ja_hanzi_chars
            kana_ratio = kana_chars / max(1, ja_total)
            if kana_chars >= 2 and kana_ratio >= 0.22:
                self._last_detected_lang = "ja"
                return "ja"

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

        if hangul_chars >= 2 and ko_score >= 0.54:
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

        _evaluate_ja()
        ja_total = kana_chars + ja_hanzi_chars
        kana_ratio = kana_chars / max(1, ja_total)

        if hangul_chars >= 2 and ko_score >= 0.48 and ko_score >= (zh_score - 0.05):
            self._last_detected_lang = "ko"
            return "ko"

        if kana_chars >= 2 and kana_ratio >= 0.22:
            if vertical_votes >= horizontal_votes or zh_score < 0.74:
                self._last_detected_lang = "ja"
                return "ja"

        if (kana_chars + ja_hanzi_chars + hanzi_chars) <= 3 and vertical_votes >= horizontal_votes:
            return self._last_detected_lang

        if hanzi_chars >= 2 and zh_score >= 0.60:
            if kana_ratio < 0.20 or (horizontal_votes > vertical_votes and zh_score >= 0.65):
                self._last_detected_lang = "zh"
                return "zh"

        if vertical_votes > horizontal_votes and kana_chars >= 1:
            self._last_detected_lang = "ja"
            return "ja"

        if hanzi_chars >= 3 and zh_score >= 0.55 and kana_chars == 0:
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
                        existing["bbox"] = [
                            min(cx1, ex1),
                            min(cy1, ey1),
                            max(cx2, ex2),
                            max(cy2, ey2),
                        ]
                        existing["confidence"] = round(max(float(existing["confidence"]), c_conf), 4)
                        existing["is_free_text"] = bool(existing.get("is_free_text", False) and c_free)
                        absorbed = True
                        break

            if not absorbed:
                merged.append(
                    {
                        "bbox": [cx1, cy1, cx2, cy2],
                        "confidence": round(c_conf, 4),
                        "is_free_text": c_free,
                    }
                )

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

    def _detect_bubbles(
        self,
        img_np: np.ndarray,
        min_box_size: int = 16,
        conf_override: float | None = None,
    ) -> list[dict[str, Any]]:
        img_h, img_w = img_np.shape[:2]
        raw_detections: list[dict[str, Any]] = []
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

                        raw_detections.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(float(conf), 4),
                                "is_free_text": False,
                            }
                        )

        # Дополнительно ищем светлые круглые/овальные облачка с текстом, которые мог пропустить YOLOv8
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

        return raw_detections

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
        # Слегка замыкаем буквы внутри белого кружка, чтобы получить цельный контур кружка,
        # но сначала найдем сами замкнутые контуры белых областей
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

            # Сначала быстро проверяем по прямоугольнику, не пересекается ли уже с найденным YOLO бабблом
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

            # Проверяем наличие темного контура вокруг кружка и четких штрихов букв в центре кружка
            pad_in_x = max(3, int(w * 0.16))
            pad_in_y = max(3, int(h * 0.16))
            inner_gray = gray[y + pad_in_y : y + h - pad_in_y, x + pad_in_x : x + w - pad_in_x]
            if inner_gray.size < 64:
                continue

            # Внутри кружка фон должен быть белым, но обязательно должны быть темные пиксели букв
            if float(np.percentile(inner_gray, 75)) < 235.0:
                continue
            dark_ink_px = int(np.count_nonzero(inner_gray < 150))
            inner_ratio = dark_ink_px / float(inner_gray.size)
            if dark_ink_px < 18 or inner_ratio < 0.015 or inner_ratio > 0.45:
                continue

            # Проверим, что внутри есть минимум 2 отдельные связные компоненты (буквы/символы), а не просто линия рисунка
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

        craft_reader = self._get_any_craft_reader()
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
        else:
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
    ) -> list[dict[str, Any]]:
        if not detections:
            return []

        sorted_by_y = sorted(detections, key=lambda d: (d["bbox"][1] + d["bbox"][3]) / 2.0)
        rows: list[list[dict[str, Any]]] = []

        for det in sorted_by_y:
            x1, y1, x2, y2 = det["bbox"]
            box_h = max(1, y2 - y1)
            box_cy = (y1 + y2) / 2.0

            placed = False
            for row in rows:
                row_y1 = min(item["bbox"][1] for item in row)
                row_y2 = max(item["bbox"][3] for item in row)
                row_h = max(1, row_y2 - row_y1)
                row_cy = sum((item["bbox"][1] + item["bbox"][3]) / 2.0 for item in row) / len(row)

                overlap = min(y2, row_y2) - max(y1, row_y1)
                if overlap > min(box_h, row_h) * row_overlap_ratio and abs(box_cy - row_cy) < max(box_h, row_h) * 0.75:
                    row.append(det)
                    placed = True
                    break

            if not placed:
                rows.append([det])

        ordered: list[dict[str, Any]] = []
        for row in rows:
            row.sort(
                key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2.0,
                reverse=rtl,
            )
            ordered.extend(row)

        return ordered

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

        padded_np = np.array(padded_crop_pil)
        min_conf = 0.28 if is_free_text else 0.14

        if lang == "ko":
            reader = self._get_ko_reader()
            if reader is None:
                raise RuntimeError(
                    "Для перевода корейской манхвы запустите install_multilang_ocr.bat (установит EasyOCR)."
                )
            txt, _, _ = self._read_easyocr_crop_with_stats(
                reader, padded_np, HANGUL_REGEX, min_conf=min_conf
            )
            if txt:
                return txt
            if is_free_text:
                return ""
            lines = reader.readtext(padded_np, detail=0, paragraph=True)
            return " ".join(str(x).strip() for x in lines if str(x).strip())

        if lang == "zh":
            reader = self._get_zh_reader()
            if reader is None:
                raise RuntimeError(
                    "Для перевода китайской маньхуа запустите install_multilang_ocr.bat (установит EasyOCR)."
                )
            txt, _, _ = self._read_easyocr_crop_with_stats(
                reader, padded_np, HANZI_REGEX, min_conf=min_conf
            )
            if txt:
                return txt
            if is_free_text:
                return ""
            lines = reader.readtext(padded_np, detail=0, paragraph=True)
            return " ".join(str(x).strip() for x in lines if str(x).strip())

        with torch.inference_mode():
            ja_text = self.ja_recognizer(padded_crop_pil).strip()

        if self._is_hallucinated_ocr_text(ja_text, ink_pixels, is_free_text=is_free_text):
            return ""
        return ja_text

    def _recognize_crops_batch_ja(
        self,
        crops_pil: list[Image.Image],
        batch_size: int = 16,
    ) -> list[str]:
        """
        Пакетное (batched) GPU-распознавание японских бабблов через внутреннюю модель `MangaOcr`
        с поддержкой FP16 (`torch.autocast("cuda", dtype=torch.float16)`).
        Ускоряет OCR страницы с 12–20 бабблами в 3–5 раз по сравнению с поштучным вызовом.
        При любой несовместимости версий мгновенно откатывается к поштучному вызову `self.ja_recognizer(crop)`.
        """
        if not crops_pil:
            return []
        if len(crops_pil) == 1:
            with torch.inference_mode():
                return [self.ja_recognizer(crops_pil[0]).strip()]

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
                return [self.ja_recognizer(c).strip() for c in crops_pil]

        try:
            results_out: list[str] = []
            model_device = next(model.parameters()).device
            use_cuda_fp16 = model_device.type == "cuda"

            with torch.inference_mode():
                for start_i in range(0, len(crops_pil), batch_size):
                    sub_crops = crops_pil[start_i : start_i + batch_size]
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

            if len(results_out) == len(crops_pil):
                return results_out
        except Exception as batch_exc:
            logger.debug("Пакетный режим MangaOCR недоступен (%s), используем поштучный режим.", batch_exc)

        with torch.inference_mode():
            return [self.ja_recognizer(c).strip() for c in crops_pil]

    def process_image(
        self,
        image: Image.Image,
        lang: str = "auto",
        min_box_size: int = 16,
        detect_free_text: bool = False,
        text_filter_mode: str = "dialogue_only",
        skip_sighs: bool = True,
        skip_sfx: bool = True,
        min_text_length: int = 2,
    ) -> tuple[list[dict[str, Any]], str]:
        req_lang = (lang or "auto").lower().strip()
        if req_lang not in ("auto", "ja", "ko", "zh"):
            req_lang = "auto"

        mode = (text_filter_mode or "dialogue_only").lower().strip()
        # В режимах "dialogue_only", "bubbles_only" и "long_only" свободный текст на фоне принудительно отключен,
        # чтобы не замазывать рисунки и спецэффекты вне бабблов.
        effective_free_text = bool(detect_free_text) and mode not in ("dialogue_only", "bubbles_only", "long_only")
        if mode in ("all", "bubbles_no_sighs"):
            effective_free_text = True if mode == "all" else bool(detect_free_text)

        self._page_ocr_cache.clear()

        rgb_image = image.convert("RGB")
        img_np = np.array(rgb_image)

        if mode == "long_only":
            yolo_conf = 0.22
            effective_min_box = max(min_box_size, 24)
        else:
            # Сниженный порог 0.15 позволяет YOLOv8 находить все круглые, овальные и нестандартные облачка с текстом
            yolo_conf = min(self.conf_threshold, 0.15)
            effective_min_box = max(16, min_box_size)

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
            return [], (self._last_detected_lang if req_lang == "auto" else req_lang)

        resolved_lang = (
            self.detect_page_language(rgb_image, raw_detections)
            if req_lang == "auto"
            else req_lang
        )
        if req_lang in ("ja", "ko", "zh"):
            self._last_detected_lang = req_lang

        sorted_detections = self._sort_boxes_reading_order(
            raw_detections,
            rtl=(resolved_lang == "ja"),
        )

        output_blocks: list[dict[str, Any]] = []
        block_id = 1
        skipped_by_filter = 0

        # Если язык страницы японский (ja), предварительно распознаем все бабблы одним пакетным GPU-вызовом!
        if resolved_lang == "ja":
            batch_candidates_crops: list[Image.Image] = []
            batch_candidates_meta: list[tuple[tuple[int, int, int, int, str], int, bool]] = []
            for det in sorted_detections:
                x1, y1, x2, y2 = det["bbox"]
                is_free = bool(det.get("is_free_text", False))
                cache_key = (x1, y1, x2, y2, resolved_lang)
                if cache_key in self._page_ocr_cache:
                    continue
                raw_crop_np = img_np[y1:y2, x1:x2]
                has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free)
                if not has_ink:
                    self._page_ocr_cache[cache_key] = ""
                    continue
                padded_crop = self._crop_with_padding(rgb_image, (x1, y1, x2, y2), pad=8)
                batch_candidates_crops.append(padded_crop)
                batch_candidates_meta.append((cache_key, ink_pixels, is_free))

            if batch_candidates_crops:
                batch_texts = self._recognize_crops_batch_ja(batch_candidates_crops)
                for (c_key, ink_px, is_f), b_txt in zip(batch_candidates_meta, batch_texts):
                    if self._is_hallucinated_ocr_text(b_txt, ink_px, is_free_text=is_f):
                        self._page_ocr_cache[c_key] = ""
                    else:
                        self._page_ocr_cache[c_key] = b_txt

        for det in sorted_detections:
            x1, y1, x2, y2 = det["bbox"]
            is_free = bool(det.get("is_free_text", False))
            cache_key = (x1, y1, x2, y2, resolved_lang)

            raw_crop_np = img_np[y1:y2, x1:x2]
            if cache_key in self._page_ocr_cache:
                recognized_text = self._page_ocr_cache[cache_key]
                has_ink, ink_pixels = self._has_meaningful_ink(raw_crop_np, is_free_text=is_free)
                if not has_ink or self._is_hallucinated_ocr_text(recognized_text, ink_pixels, is_free_text=is_free):
                    recognized_text = ""
            else:
                padded_crop = self._crop_with_padding(rgb_image, (x1, y1, x2, y2), pad=8)
                recognized_text = self._recognize_crop(
                    padded_crop,
                    raw_crop_np,
                    resolved_lang,
                    is_free_text=is_free,
                )

            if not recognized_text:
                continue

            if self.should_filter_out_block(
                recognized_text,
                is_free_text=is_free,
                text_filter_mode=mode,
                skip_sighs=skip_sighs,
                skip_sfx=skip_sfx,
                min_text_length=min_text_length,
            ):
                skipped_by_filter += 1
                logger.debug(
                    "Блок [%d,%d,%d,%d] '%s' пропущен фильтром объема текста (mode=%s, skip_sighs=%s, skip_sfx=%s)",
                    x1,
                    y1,
                    x2,
                    y2,
                    recognized_text,
                    mode,
                    skip_sighs,
                    skip_sfx,
                )
                continue

            output_blocks.append(
                {
                    "id": block_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": det["confidence"],
                    "original_text": recognized_text,
                    "is_free_text": is_free,
                    "is_deleted": False,
                }
            )
            block_id += 1

        if skipped_by_filter > 0:
            logger.info(
                "Фильтр объема текста (mode=%s): пропущено %d мелких вздохов/спецэффектов, оставлено для перевода %d блоков.",
                mode,
                skipped_by_filter,
                len(output_blocks),
            )

        self._page_ocr_cache.clear()
        return output_blocks, resolved_lang

    def recognize_manual_region(
        self,
        image: Image.Image,
        bbox: list[int],
        lang: str = "auto",
    ) -> tuple[str, str, bool]:
        """
        Принудительное распознавание текста внутри выделенной пользователем мышкой области `bbox = [x1, y1, x2, y2]`
        без отсева по фильтрам вздохов/SFX/контраста.
        Возвращает `(recognized_text, resolved_lang, is_free_text)`.
        """
        rgb_image = image.convert("RGB")
        img_w, img_h = rgb_image.size
        x1, y1, x2, y2 = [int(round(c)) for c in bbox[:4]]
        x1 = max(0, min(x1, img_w - 2))
        y1 = max(0, min(y1, img_h - 2))
        x2 = max(x1 + 4, min(x2, img_w))
        y2 = max(y1 + 4, min(y2, img_h))

        raw_crop_np = np.array(rgb_image.crop((x1, y1, x2, y2)))
        gray_crop = cv2.cvtColor(raw_crop_np, cv2.COLOR_RGB2GRAY)
        p70 = float(np.percentile(gray_crop, 70))
        # Если фон внутри выделенной области светлый (>= 205), считаем это кружком/бабблом для чистой заливки
        is_free_text = bool(p70 < 205.0)

        req_lang = (lang or "auto").lower().strip()
        if req_lang not in ("ja", "ko", "zh"):
            resolved_lang = self._last_detected_lang or "ja"
        else:
            resolved_lang = req_lang

        padded_crop = self._crop_with_padding(rgb_image, (x1, y1, x2, y2), pad=10)
        padded_np = np.array(padded_crop)

        recognized = ""
        if resolved_lang == "ko" and self._check_easyocr():
            reader = self._get_ko_reader()
            if reader is not None:
                lines = reader.readtext(padded_np, detail=0, paragraph=True)
                recognized = " ".join(str(x).strip() for x in lines if str(x).strip())
        elif resolved_lang == "zh" and self._check_easyocr():
            reader = self._get_zh_reader()
            if reader is not None:
                lines = reader.readtext(padded_np, detail=0, paragraph=True)
                recognized = " ".join(str(x).strip() for x in lines if str(x).strip())
        else:
            with torch.inference_mode():
                recognized = self.ja_recognizer(padded_crop).strip()
            # Если текст пустой или на другом языке, пробуем EasyOCR при наличии
            if (not recognized or PUNCT_ONLY_REGEX.match(recognized)) and self._check_easyocr():
                for fallback_reader in (self._get_ko_reader(), self._get_zh_reader()):
                    if fallback_reader is not None:
                        lines = fallback_reader.readtext(padded_np, detail=0, paragraph=True)
                        cand = " ".join(str(x).strip() for x in lines if str(x).strip())
                        if cand and not PUNCT_ONLY_REGEX.match(cand):
                            recognized = cand
                            break

        # Если с первого раза текст слишком бледный, пробуем повысить контраст кропа
        if not recognized or PUNCT_ONLY_REGEX.match(recognized):
            try:
                contrast_pil = ImageOps.autocontrast(padded_crop, cutoff=2)
                with torch.inference_mode():
                    cand2 = self.ja_recognizer(contrast_pil).strip()
                if cand2 and not PUNCT_ONLY_REGEX.match(cand2):
                    recognized = cand2
            except Exception:
                pass

        return recognized.strip(), resolved_lang, is_free_text
