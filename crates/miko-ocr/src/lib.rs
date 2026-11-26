pub mod detectors;
pub mod filter;
pub mod nar;
pub mod pipeline;
pub mod preprocess;
pub mod reading_order;
pub mod tight_ink;

pub use detectors::{
    box_iomin, box_iou, fuse_detections, nms_boxes, suppress_parent_containers,
    trace_concave_scanline_polygon, BalloonPolygon, BubbleDetection,
};
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
pub use tight_ink::{compute_tight_ink_bbox, has_meaningful_ink, verify_closed_bubble_cavity};

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

    #[test]
    fn test_concave_scanline_and_conjoined_triple_bubble_split() {
        // Регрессия 005.webp: 3 вертикально соприкасающихся баббла на одном общем полигоне YOLO26s-seg
        let mut rows = Vec::new();
        for y in 490..=1540 {
            let (lx, rx) = if (750..=880).contains(&y) {
                (978, 1054) // узкая талия среднего баббла #3 (справа на x=1060..1155 волосы и рамка!)
            } else {
                (962, 1158)
            };
            rows.push((y, lx, rx));
        }
        let concave_poly = trace_concave_scanline_polygon(&rows, 96);
        assert!(concave_poly.len() >= 8);
        // Проверяем, что в районе талии y=810 правая стенка не натянулась до 1158 (как делал convex_hull_2d)
        let waist_max_x = concave_poly
            .iter()
            .filter(|&&[_, y]| (780..=850).contains(&y))
            .map(|&[x, _]| x)
            .max()
            .unwrap_or(0);
        assert!(
            waist_max_x <= 1065,
            "expected concave waist <= 1065, got {waist_max_x}"
        );

        let rt0 = vec![
            BubbleDetection {
                bbox: [997, 483, 1164, 789],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.92,
                is_free_text: false,
                has_enclosing_frame: true,
            },
            BubbleDetection {
                bbox: [974, 749, 1058, 883],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.89,
                is_free_text: false,
                has_enclosing_frame: true,
            },
            BubbleDetection {
                bbox: [950, 874, 1153, 1566],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.95,
                is_free_text: false,
                has_enclosing_frame: true,
            },
        ];
        let rt1 = vec![
            BubbleDetection {
                bbox: [997, 521, 1133, 787],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.91,
                is_free_text: false,
                has_enclosing_frame: true,
            },
            BubbleDetection {
                bbox: [1000, 749, 1050, 871],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.88,
                is_free_text: false,
                has_enclosing_frame: true,
            },
            BubbleDetection {
                bbox: [950, 884, 1153, 1499],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.94,
                is_free_text: false,
                has_enclosing_frame: true,
            },
        ];
        let seg = vec![BalloonPolygon {
            bbox: [962, 490, 1158, 1542],
            confidence: 0.90,
            polygon: concave_poly,
        }];

        let fused = fuse_detections(rt0, rt1, vec![], vec![], &seg, 0.35, false);
        assert_eq!(fused.len(), 3);
        let p0 = fused[0].bubble_polygon.as_ref().expect("poly #0");
        let p1 = fused[1].bubble_polygon.as_ref().expect("poly #1");
        let p2 = fused[2].bubble_polygon.as_ref().expect("poly #2");

        let max_y0 = p0.iter().map(|p| p[1]).max().unwrap();
        let min_y1 = p1.iter().map(|p| p[1]).min().unwrap();
        let max_y1 = p1.iter().map(|p| p[1]).max().unwrap();
        let min_y2 = p2.iter().map(|p| p[1]).min().unwrap();
        assert!(max_y0 <= 795, "top bubble leaked down to {max_y0}");
        assert!(min_y1 >= 743 && max_y1 <= 882, "mid bubble y={min_y1}..{max_y1}");
        assert!(min_y2 >= 870, "bottom bubble leaked up to {min_y2}");
    }

    #[test]
    fn test_suppress_parent_matryoshka_and_borderless_sfx_cavity_check() {
        // 1. Регрессия 018.webp (#8, #9, #10): родительский бокс #9 накрывает оба под-баббла #8 и #10
        let mut cands = vec![
            BubbleDetection {
                bbox: [745, 425, 842, 726],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.91,
                is_free_text: false,
                has_enclosing_frame: true,
            },
            BubbleDetection {
                bbox: [636, 424, 842, 722], // дубль-матрёшка поверх обоих!
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.94,
                is_free_text: false,
                has_enclosing_frame: true,
            },
            BubbleDetection {
                bbox: [636, 426, 747, 705],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.90,
                is_free_text: false,
                has_enclosing_frame: true,
            },
        ];
        suppress_parent_containers(&mut cands);
        assert_eq!(cands.len(), 2, "parent matryoshka container must be dropped");

        // 1b. Регрессия 021.webp (#5, #6, #7): широкий единый баббл #6 и два боковых осколка колонок (#5 и #7, сумма < 55%) ->
        // сохраняем именно целый баббл #6, а боковые осколки подавляем!
        let mut single_with_frags = vec![
            BubbleDetection {
                bbox: [311, 62, 408, 328],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.85,
                is_free_text: false,
                has_enclosing_frame: true,
            },
            BubbleDetection {
                bbox: [148, 61, 408, 348],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.93,
                is_free_text: false,
                has_enclosing_frame: true,
            },
            BubbleDetection {
                bbox: [147, 64, 214, 235],
                text_bbox: None,
                bubble_polygon: None,
                confidence: 0.78,
                is_free_text: false,
                has_enclosing_frame: true,
            },
        ];
        suppress_parent_containers(&mut single_with_frags);
        assert_eq!(single_with_frags.len(), 1, "sub-column fragments inside a single bubble must be dropped");
        assert_eq!(single_with_frags[0].bbox, [148, 61, 408, 348]);

        // 2. Регрессия 013.webp #14 ("Eeeek!"): SFX на скринтоне без рамки vs реальная закрытая плашка
        let mut art_img = RgbImage::from_pixel(160, 160, Rgb([210, 210, 210]));
        for y in 30..130 {
            for x in 30..130 {
                if (x + y) % 2 == 0 {
                    art_img.put_pixel(x, y, Rgb([145, 145, 150]));
                } else {
                    art_img.put_pixel(x, y, Rgb([250, 250, 250]));
                }
            }
        }
        assert!(
            !verify_closed_bubble_cavity(&art_img, [35, 35, 125, 125]),
            "screentone SFX must NOT pass closed cavity check"
        );

        let mut framed_img = RgbImage::from_pixel(160, 160, Rgb([180, 180, 185]));
        for y in 30..130 {
            for x in 30..130 {
                let border = x <= 32 || x >= 127 || y <= 32 || y >= 127;
                framed_img.put_pixel(
                    x,
                    y,
                    if border {
                        Rgb([15, 15, 15])
                    } else {
                        Rgb([255, 255, 255])
                    },
                );
            }
        }
        assert!(
            verify_closed_bubble_cavity(&framed_img, [30, 30, 130, 130]),
            "closed rectangular narration box must pass cavity check"
        );
    }
}
