import unittest
import numpy as np
from PIL import Image

from cleaner import BubbleCleaner


class TestTruncatedPolyTail(unittest.TestCase):
    def test_python_cleaner_truncated_tall_bubble_polygon_cleans_tail(self) -> None:
        """Регрессия 002.webp: обрезанный на y=150 полигон высокого баббла обязан дочищать нижний хвост 'ああ!!' без горизонтального шва."""
        h, w = 260, 120
        arr = np.full((h, w, 3), 115, dtype=np.uint8)
        # Белая полость баббла [20..100, 15..240] с чёрной замкнутой рамкой 3px
        arr[15:241, 20:101] = 255
        arr[15:18, 20:101] = 12
        arr[238:241, 20:101] = 12
        arr[15:241, 20:23] = 12
        arr[15:241, 98:101] = 12

        # Верхние символы (y=45..95), символ прямо на границе обреза полигона (y=142..158) и нижние 'ああ!!' (y=185..222)
        for y0, y1 in ((45, 95), (142, 158), (185, 222)):
            arr[y0:y1:2, 48:72:2] = 15

        cleaner = BubbleCleaner(use_lama=False, models_dir="__nonexistent_models__")
        cleaner.stroke_segmenter = None
        block = {
            "id": 1,
            "bbox": [20, 15, 100, 240],
            "clean_bbox": [20, 15, 100, 240],
            "text_bbox": [44, 40, 76, 226],
            "bubble_polygon": [[22, 17], [98, 17], [98, 150], [22, 150]],
            "original_text": "孫ができたーああ!!",
            "translated_text": "Mago's having a baby!",
            "is_free_text": False,
        }

        cleaned_pil, _ = cleaner.clean_image(Image.fromarray(arr), [block])
        out = np.asarray(cleaned_pil)

        # 1. И на шве y=142..158, и внизу y=185..222 все чёрные штрихи текста полностью стёрты в белый!
        self.assertGreaterEqual(float(out[142:158, 46:74].min()), 245.0)
        self.assertGreaterEqual(float(out[185:222, 46:74].min()), 245.0)
        # 2. Чёрная нижняя рамка баббла на y=239 цела!
        self.assertLess(float(out[239, 60].mean()), 35.0)


if __name__ == "__main__":
    unittest.main()
