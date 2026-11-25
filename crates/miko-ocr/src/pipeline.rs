use crate::detectors::{
    fuse_detections, RtDetrDetector, YoloBubbleDetector, YoloSegDetector,
};
use crate::filter::{classify_filter_reason, is_hallucinated_ocr_text, is_punct_only};
use crate::nar::{NarError, NarRecognizer};
use crate::preprocess::crop_with_padding;
use crate::reading_order::sort_detections_reading_order;
use crate::tight_ink::{compute_tight_ink_bbox, has_meaningful_ink};
use image::RgbImage;
use miko_core::{normalize_clean_bbox, TextBlock};
use miko_ort::{ExecutionProviderKind, OrtError};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("Ошибка изображения: {0}")]
    Image(#[from] image::ImageError),
    #[error("Ошибка ONNX Runtime: {0}")]
    Ort(#[from] OrtError),
    #[error("Ошибка NAR-OCR: {0}")]
    Nar(#[from] NarError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrOptions {
    pub detect_free_text: bool,
    pub text_filter_mode: String,
    pub min_text_length: usize,
    pub conf_threshold: f32,
    pub rtl: bool,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            detect_free_text: false,
            text_filter_mode: "dialogue_only".to_string(),
            min_text_length: 1,
            conf_threshold: 0.35,
            rtl: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OcrTimings {
    pub detect_ms: u128,
    pub seg_ms: u128,
    pub ocr_ms: u128,
    pub total_ms: u128,
    pub ep: ExecutionProviderKind,
}

#[derive(Debug, Clone, Serialize)]
pub struct OcrPageResult {
    pub width: u32,
    pub height: u32,
    pub blocks: Vec<TextBlock>,
    pub timings: OcrTimings,
}

/// Полный нативный GPU OCR-движок на Rust (`RT-DETR-v2 + YOLOv8m + YOLO26s-seg + manga-ocr-nar-small`).
pub struct RustOcrEngine {
    rtdetr: RtDetrDetector,
    yolo_bubble: Option<YoloBubbleDetector>,
    yolo_seg: Option<YoloSegDetector>,
    nar: NarRecognizer,
}

impl RustOcrEngine {
    pub fn load(base_dir: &Path, prefer_gpu: bool) -> Result<Self, PipelineError> {
        let onnx_dir = base_dir.join("models").join("onnx");
        let rtdetr = RtDetrDetector::load(&onnx_dir.join("rtdetr_v2_comic.onnx"), prefer_gpu)?;

        let yolo_path = onnx_dir.join("yolo_bubble_det.onnx");
        let yolo_bubble = if yolo_path.is_file() {
            Some(YoloBubbleDetector::load(&yolo_path, prefer_gpu)?)
        } else {
            None
        };

        let seg_path = onnx_dir.join("yolo26s_seg.onnx");
        let yolo_seg = if seg_path.is_file() {
            Some(YoloSegDetector::load(&seg_path, prefer_gpu)?)
        } else {
            None
        };

        let nar = NarRecognizer::load(
            &onnx_dir.join("manga_ocr_nar_small.onnx"),
            &base_dir
                .join("models")
                .join("manga-ocr-nar-small")
                .join("nar_vocabulary_6817.json"),
            prefer_gpu,
        )?;

        Ok(Self {
            rtdetr,
            yolo_bubble,
            yolo_seg,
            nar,
        })
    }

    pub fn process_image_path(
        &mut self,
        image_path: &Path,
        opts: &OcrOptions,
    ) -> Result<OcrPageResult, PipelineError> {
        let dyn_im = image::open(image_path)?;
        let rgb = dyn_im.to_rgb8();
        self.process_rgb_image(&rgb, opts)
    }

    pub fn process_rgb_image(
        &mut self,
        img: &RgbImage,
        opts: &OcrOptions,
    ) -> Result<OcrPageResult, PipelineError> {
        let t_total = Instant::now();
        let (w, h) = img.dimensions();

        let t_det = Instant::now();
        let (rt0, rt1, rt2) = self.rtdetr.detect(img, opts.conf_threshold)?;
        let yolo_dets = if let Some(ref mut yb) = self.yolo_bubble {
            yb.detect(img, opts.conf_threshold)?
        } else {
            Vec::new()
        };
        let detect_ms = t_det.elapsed().as_millis();

        let t_seg = Instant::now();
        let balloons = if let Some(ref mut seg) = self.yolo_seg {
            seg.detect_balloons(img, opts.conf_threshold)?
        } else {
            Vec::new()
        };
        let seg_ms = t_seg.elapsed().as_millis();

        let mut fused = fuse_detections(
            rt0,
            rt1,
            rt2,
            yolo_dets,
            &balloons,
            opts.conf_threshold,
            opts.detect_free_text,
        );
        sort_detections_reading_order(&mut fused, opts.rtl);

        let t_ocr = Instant::now();
        let mut blocks = Vec::new();
        for det in fused {
            let tight_tb = compute_tight_ink_bbox(img, det.bbox, det.text_bbox, det.is_free_text);
            let ocr_box = if !det.is_free_text { tight_tb } else { det.bbox };
            let raw_crop = crop_with_padding(img, ocr_box, 0);
            let (has_ink, ink_px) = has_meaningful_ink(&raw_crop, det.is_free_text);
            if !has_ink {
                continue;
            }

            let padded_crop = crop_with_padding(img, ocr_box, 8);
            let mut recognized = self.nar.recognize_crop(&padded_crop)?.trim().to_string();
            if is_hallucinated_ocr_text(&recognized, ink_px, det.is_free_text) {
                if !det.is_free_text && ocr_box != det.bbox {
                    let fallback_crop = crop_with_padding(img, det.bbox, 10);
                    let fb_txt = self.nar.recognize_crop(&fallback_crop)?.trim().to_string();
                    if !is_hallucinated_ocr_text(&fb_txt, ink_px, false) {
                        recognized = fb_txt;
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            if is_punct_only(&recognized) {
                continue;
            }

            if classify_filter_reason(
                &recognized,
                det.is_free_text,
                &opts.text_filter_mode,
                opts.min_text_length,
                det.confidence,
                det.has_enclosing_frame,
            )
            .is_some()
            {
                continue;
            }

            let mut blk = TextBlock {
                id: (blocks.len() + 1) as i64,
                bbox: det.bbox,
                clean_bbox: Some(det.bbox),
                text_bbox: Some(tight_tb),
                bubble_polygon: det.bubble_polygon,
                confidence: det.confidence,
                original_text: recognized,
                is_free_text: det.is_free_text,
                ..Default::default()
            };
            normalize_clean_bbox(&mut blk);
            blocks.push(blk);
        }
        let ocr_ms = t_ocr.elapsed().as_millis();

        Ok(OcrPageResult {
            width: w,
            height: h,
            blocks,
            timings: OcrTimings {
                detect_ms,
                seg_ms,
                ocr_ms,
                total_ms: t_total.elapsed().as_millis(),
                ep: self.nar.ep,
            },
        })
    }
}
