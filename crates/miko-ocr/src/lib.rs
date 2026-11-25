pub mod detectors;
pub mod filter;
pub mod nar;
pub mod pipeline;
pub mod preprocess;
pub mod reading_order;
pub mod tight_ink;

pub use detectors::{box_iomin, box_iou, fuse_detections, nms_boxes, BubbleDetection};
pub use filter::{
    classify_filter_reason, is_hallucinated_ocr_text, is_punct_only, FilterReason,
    OcrBackendInfo,
};
pub use nar::{NarError, NarRecognizer, NarVocabulary};
pub use pipeline::{OcrOptions, OcrPageResult, OcrTimings, PipelineError, RustOcrEngine};
pub use preprocess::{
    crop_with_padding, letterbox_yolo_chw, pad_grid, preprocess_nar_crop, resize_rtdetr_chw,
    LetterboxMeta, NarPreprocessOutput,
};
pub use reading_order::sort_detections_reading_order;
pub use tight_ink::{compute_tight_ink_bbox, has_meaningful_ink};

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_numeric_dialogue_and_hallucination_rules() {
        assert!(!is_hallucinated_ocr_text("100", 80, false));
        assert!(!is_hallucinated_ocr_text("2026", 110, false));
        assert!(is_hallucinated_ocr_text("1111", 80, false));
        assert!(is_hallucinated_ocr_text("777", 80, false));

        assert!(classify_filter_reason("100", false, "dialogue_only", 2, 0.92, true).is_none());
        assert!(classify_filter_reason("111", false, "dialogue_only", 2, 0.92, true).is_some());
        assert!(classify_filter_reason("100", true, "dialogue_only", 2, 0.92, false).is_some());
    }

    #[test]
    fn test_nar_preprocess_and_vocab_eos_decode() {
        let (_s, gh, gw, rw, rh) = pad_grid(120, 60, 324);
        assert!(gh * gw <= 324 && rw > 0 && rh > 0);
        let crop = RgbImage::from_pixel(60, 120, Rgb([250, 250, 250]));
        let prep = preprocess_nar_crop(&crop);
        assert_eq!(prep.p.len(), 324 * 588);
        assert_eq!(prep.h as usize, gh);
        assert_eq!(prep.w as usize, gw);
        assert_eq!(prep.v[gh * gw - 1], 1.0);
        if gh * gw < 324 {
            assert_eq!(prep.v[gh * gw], 0.0);
        }

        let vocab = NarVocabulary::from_chars(vec!["あ".into(), "い".into(), "う".into()]);
        assert_eq!(vocab.decode_ids(&[0, 2, 3, 1]), "あう");
    }

    #[test]
    fn test_colored_webtoon_tight_ink_bbox_and_rtl_order() {
        let mut page = RgbImage::from_pixel(180, 180, Rgb([255, 255, 255]));
        for y in 55..115 {
            for x in 68..108 {
                page.put_pixel(x, y, Rgb([255, 232, 35]));
            }
        }
        let tb = compute_tight_ink_bbox(&page, [20, 20, 160, 160], None, false);
        assert!((55..=72).contains(&tb[0]) && (104..=120).contains(&tb[2]), "tb={tb:?}");

        let mut dets = vec![
            BubbleDetection {
                bbox: [20, 20, 100, 120],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.9,
                is_free_text: false,
                has_enclosing_frame: true,
            },
            BubbleDetection {
                bbox: [150, 20, 230, 120],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.9,
                is_free_text: false,
                has_enclosing_frame: true,
            },
        ];
        sort_detections_reading_order(&mut dets, true);
        assert_eq!(dets[0].bbox[0], 150, "RTL должен ставить правый баббл первым");
    }
}
