use crate::detectors::{
    fuse_detections, RtDetrDetector, YoloBubbleDetector, YoloSegDetector,
};
use crate::filter::{
    classify_filter_reason, is_hallucinated_ocr_text, is_likely_sfx_text, is_punct_only,
    sanitize_nar_stutter,
};
use crate::nar::{NarError, NarRecognizer};
use crate::preprocess::crop_with_padding;
use crate::reading_order::sort_detections_reading_order;
use crate::tight_ink::{compute_tight_ink_bbox, has_meaningful_ink, verify_closed_bubble_cavity};
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

        let onnx_vocab = onnx_dir.join("nar_vocabulary_6817.json");
        let legacy_vocab = base_dir
            .join("models")
            .join("manga-ocr-nar-small")
            .join("nar_vocabulary_6817.json");
        let vocab_path = if onnx_vocab.is_file() {
            onnx_vocab
        } else {
            legacy_vocab
        };
        let nar = NarRecognizer::load(
            &onnx_dir.join("manga_ocr_nar_small.onnx"),
            &vocab_path,
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

        let mode_lc = opts.text_filter_mode.trim().to_ascii_lowercase();
        let allow_free_pass = opts.detect_free_text
            || matches!(
                mode_lc.as_str(),
                "dialogue_only" | "bubbles_no_sighs" | "all"
            );

        let mut fused = fuse_detections(
            rt0,
            rt1,
            rt2,
            yolo_dets,
            &balloons,
            opts.conf_threshold,
            allow_free_pass,
        );
        merge_split_column_strips(img, &balloons, &mut fused);
        sort_detections_reading_order(&mut fused, opts.rtl);

        let t_ocr = Instant::now();
        let mut blocks = Vec::new();
        for mut det in fused {
            // Если YOLO26s-seg не нашёл полигон баббла, отбрасываем слабые одиночные срабатывания
            // (убирает галлюцинации по лицам `112.webp #7 "ドる"` с conf=0.41 и т.п.),
            // иначе проверяем наличие реальной замкнутой светлой полости.
            if !det.is_free_text && det.bubble_polygon.is_none() {
                if det.confidence < 0.56 && !det.confirmed_by_two_detectors {
                    continue;
                }
                if verify_closed_bubble_cavity(img, det.bbox) {
                    det.has_enclosing_frame = true;
                } else {
                    det.is_free_text = true;
                    det.has_enclosing_frame = false;
                }
            }
            if det.is_free_text {
                if !allow_free_pass {
                    continue;
                }
                if !opts.detect_free_text && det.confidence < 0.45 {
                    continue;
                }
            }

            let tight_tb = compute_tight_ink_bbox(img, det.bbox, det.text_bbox, det.is_free_text);
            let ocr_box = if !det.is_free_text { tight_tb } else { det.bbox };
            let raw_crop = crop_with_padding(img, ocr_box, 0);
            let (has_ink, ink_px) = has_meaningful_ink(&raw_crop, det.is_free_text);
            if !has_ink {
                continue;
            }

            let padded_crop = crop_with_padding(img, ocr_box, 8);
            let mut recognized = recognize_nar_adaptive(&mut self.nar, &padded_crop)?;
            if is_hallucinated_ocr_text(&recognized, ink_px, det.is_free_text) {
                if !det.is_free_text && ocr_box != det.bbox {
                    let fallback_crop = crop_with_padding(img, det.bbox, 10);
                    let fb_txt = recognize_nar_adaptive(&mut self.nar, &fallback_crop)?;
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
            if !det.is_free_text
                && det.bubble_polygon.is_none()
                && !det.confirmed_by_two_detectors
                && det.confidence < 0.65
                && recognized.chars().count() <= 2
            {
                continue;
            }
            if mode_lc == "dialogue_only"
                && det.bubble_polygon.is_none()
                && is_likely_sfx_text(&recognized)
            {
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
        dedup_adjacent_text_overlaps(&mut blocks);
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

/// Склеивает ошибочно разрезанный на узкую одноколоночную полоску (`<= 78px`) единый баббл (`021.webp #3 + #4`),
/// если между полоской и соседней половиной нет тёмной перегородки и их вертикальные границы совпадают.
fn merge_split_column_strips(
    img: &RgbImage,
    seg_balloons: &[crate::detectors::BalloonPolygon],
    dets: &mut Vec<crate::detectors::BubbleDetection>,
) {
    let (iw, ih) = (img.width() as i32, img.height() as i32);
    let lum_at = |x: i32, y: i32| -> u8 {
        let px = img.get_pixel(x.clamp(0, iw - 1) as u32, y.clamp(0, ih - 1) as u32);
        (((77u32 * px[0] as u32) + (150u32 * px[1] as u32) + (29u32 * px[2] as u32)) >> 8) as u8
    };
    let is_open_vertical_seam = |seam_x: i32, y_top: i32, y_bot: i32| -> bool {
        let span = (y_bot - y_top).max(1);
        let y0 = y_top + (span * 20) / 100;
        let y1 = y_bot - (span * 20) / 100;
        if y1 <= y0 + 8 {
            return false;
        }
        // Проверяем полосу шириной ±3px вокруг шва между колонками:
        // если там настоящая чёрная стенка баббла, почти на каждом Y найдётся тёмный пиксель стенки;
        // если это просто белый зазор между японскими столбиками одного баббла — >80% строк чисто белые!
        let mut clear_rows = 0usize;
        let mut total_rows = 0usize;
        let mut y = y0;
        while y <= y1 {
            total_rows += 1;
            let row_min = (-2..=2)
                .map(|dx| lum_at(seam_x + dx, y))
                .min()
                .unwrap_or(0);
            if row_min >= 220 {
                clear_rows += 1;
            }
            y += 2;
        }
        total_rows > 0 && clear_rows * 100 >= total_rows * 78
    };

    let mut merged_any = true;
    while merged_any && dets.len() >= 2 {
        merged_any = false;
        'outer: for i in 0..dets.len() {
            if dets[i].is_free_text {
                continue;
            }
            let bi = dets[i].bbox;
            let wi = bi[2] - bi[0];
            let hi = bi[3] - bi[1];
            for j in (i + 1)..dets.len() {
                if dets[j].is_free_text {
                    continue;
                }
                let bj = dets[j].bbox;
                let wj = bj[2] - bj[0];
                let hj = bj[3] - bj[1];
                let min_w = wi.min(wj);
                let narrow_h = if wi <= wj { hi } else { hj };
                if min_w > 78 || narrow_h < min_w * 2 {
                    continue;
                }
                let gap_x = (bi[0] - bj[2]).max(bj[0] - bi[2]);
                if gap_x > 6 || (bi[1] - bj[1]).abs() > 22 || (bi[3] - bj[3]).abs() > 22 {
                    continue;
                }
                let seam_x = if bi[0] < bj[0] {
                    (bi[2] + bj[0]) / 2
                } else {
                    (bj[2] + bi[0]) / 2
                };
                let y_top = bi[1].max(bj[1]);
                let y_bot = bi[3].min(bj[3]);
                if !is_open_vertical_seam(seam_x, y_top, y_bot) {
                    continue;
                }

                let union_box = [
                    bi[0].min(bj[0]),
                    bi[1].min(bj[1]),
                    bi[2].max(bj[2]),
                    bi[3].max(bj[3]),
                ];
                let union_tb = match (dets[i].text_bbox, dets[j].text_bbox) {
                    (Some(ta), Some(tb)) => Some([
                        ta[0].min(tb[0]),
                        ta[1].min(tb[1]),
                        ta[2].max(tb[2]),
                        ta[3].max(tb[3]),
                    ]),
                    _ => None,
                };
                dets[i].bbox = union_box;
                dets[i].text_bbox = union_tb;
                dets[i].confidence = dets[i].confidence.max(dets[j].confidence);
                // Переподбираем полигон под объединённый bbox
                let clip_r = [
                    union_box[0] - 4,
                    union_box[1] - 4,
                    union_box[2] + 4,
                    union_box[3] + 4,
                ];
                for seg in seg_balloons {
                    if crate::detectors::box_iomin(seg.bbox, union_box) >= 0.55 {
                        let clipped = miko_core::clip_polygon_to_rect(&seg.polygon, clip_r);
                        if clipped.len() >= 3 {
                            dets[i].bubble_polygon = Some(clipped);
                            break;
                        }
                    }
                }
                dets.remove(j);
                merged_any = true;
                break 'outer;
            }
        }
    }
}

fn find_clean_vertical_split_x(crop: &RgbImage) -> Option<u32> {
    let (cw, ch) = crop.dimensions();
    if cw < 240 || ch == 0 || cw < ch * 4 || ch > 76 {
        return None;
    }
    let x_start = (cw * 32) / 100;
    let x_end = (cw * 68) / 100;
    if x_end <= x_start + 6 {
        return None;
    }
    let mid = cw / 2;
    let y0 = (ch * 10) / 100;
    let y1 = ch.saturating_sub(y0).max(y0 + 1);
    let col_h = (y1 - y0).max(1);

    let is_blank_col = |x: u32| -> bool {
        let mut light = 0u32;
        for y in y0..y1 {
            let px = crop.get_pixel(x, y);
            let lum = ((77u32 * px[0] as u32) + (150u32 * px[1] as u32) + (29u32 * px[2] as u32)) >> 8;
            if lum >= 215 {
                light += 1;
            }
        }
        light * 100 >= col_h * 95
    };

    let mut best_x: Option<u32> = None;
    let mut best_dist = u32::MAX;
    for x in (x_start + 1)..(x_end - 1) {
        if is_blank_col(x - 1) && is_blank_col(x) && is_blank_col(x + 1) {
            let dist = x.abs_diff(mid);
            if dist < best_dist {
                best_dist = dist;
                best_x = Some(x);
            }
        }
    }
    best_x
}

fn recognize_nar_adaptive(nar: &mut NarRecognizer, crop: &RgbImage) -> Result<String, NarError> {
    if let Some(split_x) = find_clean_vertical_split_x(crop) {
        let (cw, ch) = crop.dimensions();
        if split_x >= 48 && cw.saturating_sub(split_x) >= 48 {
            let left = image::imageops::crop_imm(crop, 0, 0, split_x, ch).to_image();
            let right =
                image::imageops::crop_imm(crop, split_x, 0, cw - split_x, ch).to_image();
            let l_txt = sanitize_nar_stutter(nar.recognize_crop(&left)?.trim());
            let r_txt = sanitize_nar_stutter(nar.recognize_crop(&right)?.trim());
            let joined = format!("{l_txt}{r_txt}");
            if !joined.trim().is_empty() {
                return Ok(sanitize_nar_stutter(&joined));
            }
        }
    }
    let raw = nar.recognize_crop(crop)?;
    Ok(sanitize_nar_stutter(raw.trim()))
}

fn is_sentence_stop_punct(c: char) -> bool {
    matches!(c, '?' | '!' | '？' | '！' | '。' | '」' | '』')
}

/// Убирает дублирующиеся огрызки текста на стыке соседних/перекрывающихся рамок
/// (например, `'誰がお兄さんだ?'` + `'さんだ?私からすればお前らまだ子供じゃん'` -> `'私からすればお前らまだ子供じゃん'`,
/// `'とても似合っていますよ!う上かう'` + `'すよ!お店から制服を...'` -> `'とても似合っていますよ!'` + `'お店から制服を...'`).
pub fn dedup_adjacent_text_overlaps(blocks: &mut [TextBlock]) {
    let n = blocks.len();
    if n < 2 {
        return;
    }
    for j in 0..n {
        if blocks[j].is_deleted {
            continue;
        }
        let bj = blocks[j].bbox;
        let tj_chars: Vec<char> = blocks[j].original_text.trim().chars().collect();
        if tj_chars.len() < 7 {
            continue;
        }

        // Ищем разделительный знак конца чужой фразы (`?`, `!`, `。`) в первых 2..=10 символах блока `j`
        let max_pref = tj_chars.len().saturating_sub(4).min(10);
        let mut split_pos: Option<usize> = None;
        for idx in 1..max_pref {
            if is_sentence_stop_punct(tj_chars[idx]) {
                let mut end_idx = idx + 1;
                while end_idx < max_pref && is_sentence_stop_punct(tj_chars[end_idx]) {
                    end_idx += 1;
                }
                split_pos = Some(end_idx);
                break;
            }
        }
        let Some(pref_end) = split_pos else {
            continue;
        };
        if tj_chars.len().saturating_sub(pref_end) < 4 {
            continue;
        }
        let prefix: String = tj_chars[..pref_end].iter().collect();
        let pref_char_len = pref_end;

        let mut matched_neighbor: Option<(usize, usize)> = None;
        for i in 0..n {
            if i == j || blocks[i].is_deleted {
                continue;
            }
            let bi = blocks[i].bbox;
            let gap_x = (bi[0] - bj[2]).max(bj[0] - bi[2]);
            let gap_y = (bi[1] - bj[3]).max(bj[1] - bi[3]);
            if gap_x > 36 || gap_y > 36 {
                continue;
            }
            let ti = blocks[i].original_text.trim();
            let ti_chars: Vec<char> = ti.chars().collect();
            if ti_chars.len() <= pref_char_len {
                continue;
            }
            // Проверяем, стоит ли `prefix` в самом конце `ti` (допуская до 4 символов шумового хвоста после `!`/`?`)
            let tail_start_min = ti_chars.len().saturating_sub(pref_char_len + 4);
            let tail_slice: String = ti_chars[tail_start_min..].iter().collect();
            if let Some(byte_off) = tail_slice.rfind(&prefix) {
                let after_pref = &tail_slice[byte_off + prefix.len()..];
                let after_len = after_pref.chars().count();
                if after_len <= 4 && !after_pref.chars().any(is_sentence_stop_punct) {
                    let cut_chars = ti_chars.len() - after_len;
                    matched_neighbor = Some((i, cut_chars));
                    break;
                }
            }
        }

        if let Some((nbr_idx, nbr_keep_chars)) = matched_neighbor {
            let new_j: String = tj_chars[pref_end..].iter().collect();
            blocks[j].original_text = new_j.trim().to_string();

            let nbr_chars: Vec<char> = blocks[nbr_idx].original_text.trim().chars().collect();
            if nbr_keep_chars < nbr_chars.len() && nbr_keep_chars >= 3 {
                blocks[nbr_idx].original_text =
                    nbr_chars[..nbr_keep_chars].iter().collect::<String>();
            }
        }
    }
}
