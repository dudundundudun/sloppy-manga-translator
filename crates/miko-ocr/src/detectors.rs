use crate::preprocess::{letterbox_yolo_chw, resize_rtdetr_chw};
use image::imageops::crop_imm;
use image::RgbImage;
use miko_core::{clip_polygon_to_rect, polygon_area_abs};
use miko_ort::ort::session::Session;
use miko_ort::ort::value::Tensor;
use miko_ort::{create_session, ExecutionProviderKind, OrtError};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BubbleDetection {
    pub bbox: [i32; 4],
    pub text_bbox: Option<[i32; 4]>,
    pub bubble_polygon: Option<Vec<[i32; 2]>>,
    pub confidence: f32,
    pub is_free_text: bool,
    pub has_enclosing_frame: bool,
    #[serde(default)]
    pub confirmed_by_two_detectors: bool,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn box_iou(a: [i32; 4], b: [i32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let inter = ((ix2 - ix1).max(0) as f32) * ((iy2 - iy1).max(0) as f32);
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = ((a[2] - a[0]).max(1) as f32) * ((a[3] - a[1]).max(1) as f32);
    let area_b = ((b[2] - b[0]).max(1) as f32) * ((b[3] - b[1]).max(1) as f32);
    inter / (area_a + area_b - inter).max(1.0)
}

pub fn box_iomin(a: [i32; 4], b: [i32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let inter = ((ix2 - ix1).max(0) as f32) * ((iy2 - iy1).max(0) as f32);
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = ((a[2] - a[0]).max(1) as f32) * ((a[3] - a[1]).max(1) as f32);
    let area_b = ((b[2] - b[0]).max(1) as f32) * ((b[3] - b[1]).max(1) as f32);
    inter / area_a.min(area_b).max(1.0)
}

pub fn nms_boxes(cands: &mut Vec<BubbleDetection>, iou_thresh: f32) {
    cands.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<BubbleDetection> = Vec::with_capacity(cands.len());
    for c in cands.drain(..) {
        if !kept.iter().any(|k| box_iou(k.bbox, c.bbox) >= iou_thresh) {
            kept.push(c);
        }
    }
    *cands = kept;
}

pub struct RtDetrDetector {
    session: Session,
    pub ep: ExecutionProviderKind,
}

impl RtDetrDetector {
    pub fn load(onnx_path: &Path, prefer_gpu: bool) -> Result<Self, OrtError> {
        let (session, ep) = create_session(onnx_path, prefer_gpu)?;
        Ok(Self { session, ep })
    }

    fn run_slice(
        &mut self,
        slice_rgb: &RgbImage,
        y_offset: i32,
        img_w: i32,
        img_h: i32,
        conf_thresh: f32,
    ) -> Result<
        (
            Vec<BubbleDetection>,
            Vec<BubbleDetection>,
            Vec<BubbleDetection>,
        ),
        OrtError,
    > {
        let chw = resize_rtdetr_chw(slice_rgb, 640);
        let input_tensor = Tensor::from_array(([1usize, 3, 640, 640], chw))?;
        let in_name = self
            .session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "pixel_values".to_string());
        let outputs = self
            .session
            .run(miko_ort::ort::inputs![in_name => input_tensor])?;

        let (logits_shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        let (_, boxes) = outputs["pred_boxes"].try_extract_tensor::<f32>()?;
        let num_queries = logits_shape.get(1).copied().unwrap_or(300).max(0) as usize;
        let num_classes = logits_shape.get(2).copied().unwrap_or(3).max(0) as usize;

        let sw = slice_rgb.width() as f32;
        let sh = slice_rgb.height() as f32;
        let low_thresh = conf_thresh.min(0.25).max(0.18);

        let mut cls0 = Vec::new();
        let mut cls1 = Vec::new();
        let mut cls2 = Vec::new();

        for q in 0..num_queries {
            let l_base = q * num_classes;
            let mut best_cls = 0usize;
            let mut best_prob = 0.0f32;
            for c in 0..num_classes {
                let p = sigmoid(logits[l_base + c]);
                if p > best_prob {
                    best_prob = p;
                    best_cls = c;
                }
            }
            if best_prob < low_thresh {
                continue;
            }
            let b_base = q * 4;
            let cx = boxes[b_base] * sw;
            let cy = boxes[b_base + 1] * sh;
            let bw = boxes[b_base + 2] * sw;
            let bh = boxes[b_base + 3] * sh;
            let x1 = ((cx - bw * 0.5).round() as i32).clamp(0, img_w - 1);
            let y1 = (((cy - bh * 0.5).round() as i32) + y_offset).clamp(0, img_h - 1);
            let x2 = ((cx + bw * 0.5).round() as i32).clamp(x1 + 1, img_w);
            let y2 = (((cy + bh * 0.5).round() as i32) + y_offset).clamp(y1 + 1, img_h);
            if (x2 - x1) < 12 || (y2 - y1) < 12 {
                continue;
            }
            let det = BubbleDetection {
                bbox: [x1, y1, x2, y2],
                text_bbox: None,
                bubble_polygon: None,
                confidence: best_prob,
                is_free_text: best_cls == 2,
                has_enclosing_frame: best_cls != 2,
                confirmed_by_two_detectors: false,
            };
            match best_cls {
                0 => cls0.push(det),
                1 => cls1.push(det),
                _ => cls2.push(det),
            }
        }

        Ok((cls0, cls1, cls2))
    }

    pub fn detect(
        &mut self,
        img: &RgbImage,
        conf_thresh: f32,
    ) -> Result<
        (
            Vec<BubbleDetection>,
            Vec<BubbleDetection>,
            Vec<BubbleDetection>,
        ),
        OrtError,
    > {
        let (w, h) = img.dimensions();
        let mut all0 = Vec::new();
        let mut all1 = Vec::new();
        let mut all2 = Vec::new();

        if h as f32 > (w as f32) * 1.8 {
            let slice_h = ((w as f32 * 1.45).round() as u32).max(640).min(h);
            let step = (slice_h * 3 / 4).max(200);
            let mut y = 0u32;
            while y < h {
                let end_y = (y + slice_h).min(h);
                let real_y = end_y.saturating_sub(slice_h);
                let sub = crop_imm(img, 0, real_y, w, end_y - real_y).to_image();
                let (c0, c1, c2) =
                    self.run_slice(&sub, real_y as i32, w as i32, h as i32, conf_thresh)?;
                all0.extend(c0);
                all1.extend(c1);
                all2.extend(c2);
                if end_y >= h {
                    break;
                }
                y += step;
            }
        } else {
            let (c0, c1, c2) = self.run_slice(img, 0, w as i32, h as i32, conf_thresh)?;
            all0 = c0;
            all1 = c1;
            all2 = c2;
        }

        nms_boxes(&mut all0, 0.55);
        nms_boxes(&mut all1, 0.55);
        nms_boxes(&mut all2, 0.50);
        Ok((all0, all1, all2))
    }
}

pub struct YoloBubbleDetector {
    session: Session,
    pub ep: ExecutionProviderKind,
}

impl YoloBubbleDetector {
    pub fn load(onnx_path: &Path, prefer_gpu: bool) -> Result<Self, OrtError> {
        let (session, ep) = create_session(onnx_path, prefer_gpu)?;
        Ok(Self { session, ep })
    }

    pub fn detect(
        &mut self,
        img: &RgbImage,
        conf_thresh: f32,
    ) -> Result<Vec<BubbleDetection>, OrtError> {
        let (iw, ih) = (img.width() as i32, img.height() as i32);
        let (chw, meta) = letterbox_yolo_chw(img, 640);
        let in_tensor = Tensor::from_array(([1usize, 3, 640, 640], chw))?;
        let in_name = self
            .session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());
        let outputs = self
            .session
            .run(miko_ort::ort::inputs![in_name => in_tensor])?;
        let out_val = outputs.values().next().expect("yolo output0");
        let (shape, data) = out_val.try_extract_tensor::<f32>()?;
        let ch = shape.get(1).copied().unwrap_or(6) as usize;
        let anchors = shape.get(2).copied().unwrap_or(8400) as usize;
        let num_classes = ch.saturating_sub(4);

        let mut cands = Vec::new();
        for i in 0..anchors {
            let mut best_cls = 0usize;
            let mut best_score = 0.0f32;
            for c in 0..num_classes {
                let s = data[(4 + c) * anchors + i];
                if s > best_score {
                    best_score = s;
                    best_cls = c;
                }
            }
            if best_score < conf_thresh {
                continue;
            }
            let cx = (data[i] - meta.pad_x) / meta.scale;
            let cy = (data[anchors + i] - meta.pad_y) / meta.scale;
            let bw = data[2 * anchors + i] / meta.scale;
            let bh = data[3 * anchors + i] / meta.scale;

            let x1 = ((cx - bw * 0.5).round() as i32).clamp(0, iw - 1);
            let y1 = ((cy - bh * 0.5).round() as i32).clamp(0, ih - 1);
            let x2 = ((cx + bw * 0.5).round() as i32).clamp(x1 + 1, iw);
            let y2 = ((cy + bh * 0.5).round() as i32).clamp(y1 + 1, ih);
            if (x2 - x1) < 12 || (y2 - y1) < 12 {
                continue;
            }
            cands.push(BubbleDetection {
                bbox: [x1, y1, x2, y2],
                text_bbox: None,
                bubble_polygon: None,
                confidence: best_score,
                is_free_text: best_cls == 1,
                has_enclosing_frame: best_cls != 1,
                confirmed_by_two_detectors: false,
            });
        }
        nms_boxes(&mut cands, 0.45);
        Ok(cands)
    }
}

/// Строит замкнутый невыпуклый полигон по сканлайн-хордам `(y, left_x, right_x)`,
/// сохраняя узкие перешейки («талию») сдвоенных и вертикально слепленных бабблов без натяжения выпуклой оболочки.
pub fn trace_concave_scanline_polygon(
    rows: &[(i32, i32, i32)],
    max_vertices: usize,
) -> Vec<[i32; 2]> {
    if rows.len() < 2 {
        return Vec::new();
    }
    let n = rows.len();
    // Лёгкое 3-точечное медианное сглаживание краёв против ступенек прото-маски 160x160
    let mut smooth_l = Vec::with_capacity(n);
    let mut smooth_r = Vec::with_capacity(n);
    for i in 0..n {
        let i0 = i.saturating_sub(1);
        let i2 = (i + 1).min(n - 1);
        let mut ls = [rows[i0].1, rows[i].1, rows[i2].1];
        let mut rs = [rows[i0].2, rows[i].2, rows[i2].2];
        ls.sort_unstable();
        rs.sort_unstable();
        let lx = ls[1];
        let rx = rs[1].max(lx + 1);
        smooth_l.push([lx, rows[i].0]);
        smooth_r.push([rx, rows[i].0]);
    }

    let half_cap = (max_vertices.max(12) / 2).max(6);
    let subsample_chain = |chain: &[[i32; 2]]| -> Vec<[i32; 2]> {
        if chain.len() <= half_cap {
            return chain.to_vec();
        }
        let mut sub = Vec::with_capacity(half_cap);
        let last = chain.len() - 1;
        for k in 0..half_cap {
            let idx = (k * last + (half_cap - 1) / 2) / (half_cap - 1);
            let pt = chain[idx.min(last)];
            if sub.last().copied() != Some(pt) {
                sub.push(pt);
            }
        }
        sub
    };

    let left_pts = subsample_chain(&smooth_l);
    let right_pts = subsample_chain(&smooth_r);

    let mut poly = Vec::with_capacity(left_pts.len() + right_pts.len());
    for pt in left_pts {
        if poly.last().copied() != Some(pt) {
            poly.push(pt);
        }
    }
    for &pt in right_pts.iter().rev() {
        if poly.last().copied() != Some(pt) {
            poly.push(pt);
        }
    }
    if poly.len() >= 2 && poly.first() == poly.last() {
        poly.pop();
    }
    if poly.len() < 3 || polygon_area_abs(&poly) < 24.0 {
        return Vec::new();
    }
    poly
}

#[derive(Debug, Clone)]
pub struct BalloonPolygon {
    pub bbox: [i32; 4],
    pub confidence: f32,
    pub polygon: Vec<[i32; 2]>,
}

pub struct YoloSegDetector {
    session: Session,
    pub ep: ExecutionProviderKind,
}

impl YoloSegDetector {
    pub fn load(onnx_path: &Path, prefer_gpu: bool) -> Result<Self, OrtError> {
        let (session, ep) = create_session(onnx_path, prefer_gpu)?;
        Ok(Self { session, ep })
    }

    fn run_seg_slice(
        &mut self,
        slice_rgb: &RgbImage,
        y_offset: i32,
        img_w: i32,
        img_h: i32,
        conf_thresh: f32,
    ) -> Result<Vec<BalloonPolygon>, OrtError> {
        let (sw, sh) = (slice_rgb.width() as i32, slice_rgb.height() as i32);
        let (chw, meta) = letterbox_yolo_chw(slice_rgb, 640);
        let in_tensor = Tensor::from_array(([1usize, 3, 640, 640], chw))?;
        let in_name = self
            .session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());
        let outputs = self
            .session
            .run(miko_ort::ort::inputs![in_name => in_tensor])?;

        let (s0, d0) = outputs["output0"].try_extract_tensor::<f32>()?;
        let (s1, d1) = outputs["output1"].try_extract_tensor::<f32>()?;
        let ch0 = s0.get(1).copied().unwrap_or(39) as usize;
        let anchors = s0.get(2).copied().unwrap_or(8400) as usize;
        let nm = s1.get(1).copied().unwrap_or(32) as usize;
        let mh = s1.get(2).copied().unwrap_or(160) as usize;
        let mw = s1.get(3).copied().unwrap_or(160) as usize;
        let nc = ch0.saturating_sub(4 + nm);

        // Класс 2 = `balloon` в manga109-yolov26s-seg
        let target_cls = 2usize.min(nc.saturating_sub(1));
        struct RawMaskDet {
            bbox: [i32; 4],
            lb_box: [f32; 4],
            conf: f32,
            coeff: Vec<f32>,
        }
        let mut raw_dets = Vec::new();
        for i in 0..anchors {
            let mut best_cls = 0usize;
            let mut best_score = 0.0f32;
            for c in 0..nc {
                let s = d0[(4 + c) * anchors + i];
                if s > best_score {
                    best_score = s;
                    best_cls = c;
                }
            }
            if best_cls != target_cls || best_score < conf_thresh.min(0.28) {
                continue;
            }
            let lcx = d0[i];
            let lcy = d0[anchors + i];
            let lbw = d0[2 * anchors + i];
            let lbh = d0[3 * anchors + i];

            let x1 = (((lcx - lbw * 0.5) - meta.pad_x) / meta.scale).round() as i32;
            let y1 = (((lcy - lbh * 0.5) - meta.pad_y) / meta.scale).round() as i32;
            let x2 = (((lcx + lbw * 0.5) - meta.pad_x) / meta.scale).round() as i32;
            let y2 = (((lcy + lbh * 0.5) - meta.pad_y) / meta.scale).round() as i32;
            let bx1 = x1.clamp(0, (sw - 1).max(0)).clamp(0, (img_w - 1).max(0));
            let by1 = (y1.clamp(0, (sh - 1).max(0)) + y_offset).clamp(0, (img_h - 1).max(0));
            let bx2 = x2.clamp(bx1 + 1, sw.max(1)).clamp(bx1 + 1, img_w.max(1));
            let by2 =
                (y2.clamp((by1 - y_offset) + 1, sh.max(1)) + y_offset).clamp(by1 + 1, img_h.max(1));
            if (bx2 - bx1) < 16 || (by2 - by1) < 16 {
                continue;
            }
            let mut coeff = Vec::with_capacity(nm);
            for k in 0..nm {
                coeff.push(d0[(4 + nc + k) * anchors + i]);
            }
            raw_dets.push(RawMaskDet {
                bbox: [bx1, by1, bx2, by2],
                lb_box: [
                    lcx - lbw * 0.5,
                    lcy - lbh * 0.5,
                    lcx + lbw * 0.5,
                    lcy + lbh * 0.5,
                ],
                conf: best_score,
                coeff,
            });
        }

        raw_dets.sort_by(|a, b| {
            b.conf
                .partial_cmp(&a.conf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut out = Vec::new();
        let stride = mh * mw;
        let scale_to_proto = (mw as f32) / 640.0;

        for r in raw_dets {
            if out
                .iter()
                .any(|k: &BalloonPolygon| box_iou(k.bbox, r.bbox) >= 0.5)
            {
                continue;
            }
            let px1 = (r.lb_box[0] * scale_to_proto)
                .floor()
                .clamp(0.0, (mw - 1) as f32) as usize;
            let py1 = (r.lb_box[1] * scale_to_proto)
                .floor()
                .clamp(0.0, (mh - 1) as f32) as usize;
            let px2 = (r.lb_box[2] * scale_to_proto)
                .ceil()
                .clamp((px1 + 1) as f32, mw as f32) as usize;
            let py2 = (r.lb_box[3] * scale_to_proto)
                .ceil()
                .clamp((py1 + 1) as f32, mh as f32) as usize;

            let mut scan_rows: Vec<(i32, i32, i32)> = Vec::with_capacity(py2.saturating_sub(py1));
            for py in py1..py2 {
                let mut row_left: Option<usize> = None;
                let mut row_right: Option<usize> = None;
                for px in px1..px2 {
                    let mut sum = 0.0f32;
                    let off = py * mw + px;
                    for k in 0..nm {
                        sum += r.coeff[k] * d1[k * stride + off];
                    }
                    if sigmoid(sum) > 0.45 {
                        if row_left.is_none() {
                            row_left = Some(px);
                        }
                        row_right = Some(px);
                    }
                }
                if let (Some(lx), Some(rx)) = (row_left, row_right) {
                    let ox_l =
                        (((lx as f32) / scale_to_proto - meta.pad_x) / meta.scale).round() as i32;
                    let ox_r = ((((rx + 1) as f32) / scale_to_proto - meta.pad_x) / meta.scale)
                        .round() as i32;
                    let oy = (((py as f32 + 0.5) / scale_to_proto - meta.pad_y) / meta.scale)
                        .round() as i32
                        + y_offset;
                    let cl = ox_l.clamp(r.bbox[0], r.bbox[2]).clamp(0, img_w);
                    let cr = ox_r.clamp(cl, r.bbox[2]).clamp(0, img_w);
                    let cy = oy.clamp(r.bbox[1], r.bbox[3]).clamp(0, img_h);
                    if cr > cl {
                        if let Some(last) = scan_rows.last_mut() {
                            if last.0 == cy {
                                last.1 = last.1.min(cl);
                                last.2 = last.2.max(cr);
                                continue;
                            }
                        }
                        scan_rows.push((cy, cl, cr));
                    }
                }
            }
            let poly = trace_concave_scanline_polygon(&scan_rows, 96);
            if poly.len() >= 3 {
                out.push(BalloonPolygon {
                    bbox: r.bbox,
                    confidence: r.conf,
                    polygon: poly,
                });
            }
        }
        Ok(out)
    }

    pub fn detect_balloons(
        &mut self,
        img: &RgbImage,
        conf_thresh: f32,
    ) -> Result<Vec<BalloonPolygon>, OrtError> {
        let (w, h) = img.dimensions();
        let (iw, ih) = (w as i32, h as i32);
        let mut all = self.run_seg_slice(img, 0, iw, ih, conf_thresh)?;

        // На высоких страницах манги и вебтунах делаем перекрывающиеся вертикальные срезы,
        // чтобы прото-маска 160x160 не теряла разрешение и не обрубала длинные бабблы (002.webp).
        if h as f32 > (w as f32) * 1.38 {
            let slice_h = ((w as f32 * 1.25).round() as u32).max(640).min(h);
            let step = (slice_h * 3 / 5).max(200);
            let mut y = 0u32;
            while y < h {
                let end_y = (y + slice_h).min(h);
                let real_y = end_y.saturating_sub(slice_h);
                let sub = crop_imm(img, 0, real_y, w, end_y - real_y).to_image();
                let slice_dets = self.run_seg_slice(&sub, real_y as i32, iw, ih, conf_thresh)?;
                all.extend(slice_dets);
                if end_y >= h {
                    break;
                }
                y += step;
            }
        }

        // Предпочитаем более полные по площади полигоны при высоком перекрытии IoU,
        // чтобы полноразмерный баббл с полного кадра побеждал срезанный на границе тайла (и наоборот — чёткий локальный побеждал пропущенный).
        all.sort_by(|a, b| {
            let score_a = a.confidence * (polygon_area_abs(&a.polygon).sqrt() as f32).max(1.0);
            let score_b = b.confidence * (polygon_area_abs(&b.polygon).sqrt() as f32).max(1.0);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut merged: Vec<BalloonPolygon> = Vec::with_capacity(all.len());
        for cand in all {
            if !merged.iter().any(|k| box_iou(k.bbox, cand.bbox) >= 0.55) {
                merged.push(cand);
            }
        }
        Ok(merged)
    }
}

fn compute_neighbor_aware_clip_rect(idx: usize, bubbles: &[BubbleDetection]) -> [i32; 4] {
    let b = &bubbles[idx];
    let mut rect = [b.bbox[0] - 6, b.bbox[1] - 6, b.bbox[2] + 6, b.bbox[3] + 6];
    if let Some(tb) = b.text_bbox {
        rect[0] = rect[0].min(tb[0] - 4);
        rect[1] = rect[1].min(tb[1] - 4);
        rect[2] = rect[2].max(tb[2] + 4);
        rect[3] = rect[3].max(tb[3] + 4);
    }

    let anchor_i = b.text_bbox.unwrap_or(b.bbox);
    let cx_i = (anchor_i[0] + anchor_i[2]) as f32 * 0.5;
    let cy_i = (anchor_i[1] + anchor_i[3]) as f32 * 0.5;

    for (j, other) in bubbles.iter().enumerate() {
        if j == idx || other.is_free_text {
            continue;
        }
        // Проверяем, соприкасаются или пересекаются ли соседние бабблы
        let near_x = other.bbox[0] <= rect[2] + 8 && other.bbox[2] >= rect[0] - 8;
        let near_y = other.bbox[1] <= rect[3] + 8 && other.bbox[3] >= rect[1] - 8;
        if !(near_x && near_y) {
            continue;
        }

        let anchor_j = other.text_bbox.unwrap_or(other.bbox);
        let cx_j = (anchor_j[0] + anchor_j[2]) as f32 * 0.5;
        let cy_j = (anchor_j[1] + anchor_j[3]) as f32 * 0.5;
        let dx = (cx_j - cx_i).abs();
        let dy = (cy_j - cy_i).abs();

        // Определяем ось разделения по зазору между текстовыми ядрами (или центрами)
        let gap_x = (anchor_j[0] - anchor_i[2]).max(anchor_i[0] - anchor_j[2]);
        let gap_y = (anchor_j[1] - anchor_i[3]).max(anchor_i[1] - anchor_j[3]);
        let split_vertically = if gap_y > gap_x {
            true
        } else if gap_x > gap_y {
            false
        } else {
            dy >= dx * 0.85
        };

        if split_vertically {
            if cy_j > cy_i {
                let seam = (anchor_i[3] + anchor_j[1]) / 2;
                let safe_bottom = seam.max(anchor_i[3] + 2).min(b.bbox[3] + 4);
                rect[3] = rect[3].min(safe_bottom);
            } else {
                let seam = (anchor_j[3] + anchor_i[1]) / 2;
                let safe_top = seam.min(anchor_i[1] - 2).max(b.bbox[1] - 4);
                rect[1] = rect[1].max(safe_top);
            }
        } else if cx_j > cx_i {
            let seam = (anchor_i[2] + anchor_j[0]) / 2;
            let safe_right = seam.max(anchor_i[2] + 2).min(b.bbox[2] + 4);
            rect[2] = rect[2].min(safe_right);
        } else {
            let seam = (anchor_j[2] + anchor_i[0]) / 2;
            let safe_left = seam.min(anchor_i[0] - 2).max(b.bbox[0] - 4);
            rect[0] = rect[0].max(safe_left);
        }
    }

    if rect[2] <= rect[0] + 4 {
        rect[0] = b.bbox[0];
        rect[2] = b.bbox[2];
    }
    if rect[3] <= rect[1] + 4 {
        rect[1] = b.bbox[1];
        rect[3] = b.bbox[3];
    }
    rect
}

/// Разрешает вложенные контейнеры («матрёшки»):
/// 1) Если родитель целиком накрывает 2+ полноценных под-баббла с суммарной площадью >= 68% (`018.webp #8/#9/#10`) — удаляет лишнего родителя.
/// 2) Если внутри одного широкого баббла детектор дополнительно выделил узкие осколки отдельных японских колонок с суммарной площадью < 62% (`021.webp #5/#7` внутри `#6`) — удаляет осколки и сохраняет целый баббл!
pub fn suppress_parent_containers(bubbles: &mut Vec<BubbleDetection>) {
    let n = bubbles.len();
    if n < 2 {
        return;
    }
    let areas: Vec<f32> = bubbles
        .iter()
        .map(|b| ((b.bbox[2] - b.bbox[0]).max(1) as f32) * ((b.bbox[3] - b.bbox[1]).max(1) as f32))
        .collect();
    let mut drop_flags = vec![false; n];

    for i in 0..n {
        if drop_flags[i] || bubbles[i].is_free_text {
            continue;
        }
        let area_p = areas[i];
        let mut children: Vec<usize> = Vec::new();
        for j in 0..n {
            if i == j || drop_flags[j] || bubbles[j].is_free_text {
                continue;
            }
            if areas[j] >= area_p * 0.86 {
                continue;
            }
            if box_iomin(bubbles[i].bbox, bubbles[j].bbox) >= 0.72 {
                if !children
                    .iter()
                    .any(|&ck| box_iou(bubbles[ck].bbox, bubbles[j].bbox) >= 0.35)
                {
                    children.push(j);
                }
            }
        }
        if children.len() >= 2 {
            let sum_child_area: f32 = children.iter().map(|&ck| areas[ck]).sum();
            if sum_child_area >= area_p * 0.68 {
                drop_flags[i] = true;
            } else if sum_child_area < area_p * 0.62 {
                for &ck in &children {
                    if areas[ck] < area_p * 0.58
                        && box_iomin(bubbles[i].bbox, bubbles[ck].bbox) >= 0.80
                    {
                        drop_flags[ck] = true;
                    }
                }
            }
        } else if children.len() == 1 {
            let ck = children[0];
            let cw = bubbles[ck].bbox[2] - bubbles[ck].bbox[0];
            let pw = bubbles[i].bbox[2] - bubbles[i].bbox[0];
            if areas[ck] < area_p * 0.52
                && cw < (pw * 60 / 100)
                && box_iomin(bubbles[i].bbox, bubbles[ck].bbox) >= 0.85
            {
                drop_flags[ck] = true;
            }
        }
    }

    if drop_flags.iter().any(|&d| d) {
        let mut idx = 0usize;
        bubbles.retain(|_| {
            let keep = !drop_flags[idx];
            idx += 1;
            keep
        });
    }
}

/// Объединяет детекции RT-DETR-v2 + YOLOv8m + полигоны YOLO26s-seg.
pub fn fuse_detections(
    rt_cls0: Vec<BubbleDetection>,
    rt_cls1: Vec<BubbleDetection>,
    rt_cls2: Vec<BubbleDetection>,
    yolo_dets: Vec<BubbleDetection>,
    seg_balloons: &[BalloonPolygon],
    conf_thresh: f32,
    detect_free_text: bool,
) -> Vec<BubbleDetection> {
    let mut bubbles = Vec::new();

    // 1. Парим внешние бабблы RT-DETR cls0 с внутренним текстом cls1
    for mut b in rt_cls0 {
        if b.confidence < conf_thresh {
            continue;
        }
        let mut best_tb: Option<[i32; 4]> = None;
        let mut best_ov = 0.0f32;
        for t in &rt_cls1 {
            let ov = box_iomin(b.bbox, t.bbox);
            if ov > 0.55 && ov > best_ov {
                best_ov = ov;
                best_tb = Some(t.bbox);
            }
        }
        b.text_bbox = best_tb;
        b.is_free_text = false;
        b.has_enclosing_frame = true;
        bubbles.push(b);
    }

    // 2. Добавляем unmatched text_bubble из RT-DETR cls1, затем сверяем с YOLOv8m
    // и фиксируем подтверждение двумя независимыми детекторами (`confirmed_by_two_detectors`).
    for t in rt_cls1 {
        if t.confidence < conf_thresh {
            continue;
        }
        if t.is_free_text {
            if detect_free_text && !bubbles.iter().any(|b| box_iou(b.bbox, t.bbox) >= 0.45) {
                bubbles.push(t);
            }
            continue;
        }
        if !bubbles
            .iter()
            .any(|b| box_iou(b.bbox, t.bbox) >= 0.42 || box_iomin(b.bbox, t.bbox) >= 0.60)
        {
            bubbles.push(t);
        }
    }

    for y in yolo_dets {
        if y.confidence < conf_thresh {
            continue;
        }
        if y.is_free_text {
            if !detect_free_text {
                continue;
            }
            if let Some(existing) = bubbles
                .iter_mut()
                .find(|b| b.is_free_text && box_iou(b.bbox, y.bbox) >= 0.35)
            {
                existing.confirmed_by_two_detectors = true;
                existing.confidence = existing.confidence.max(y.confidence);
            } else if !bubbles.iter().any(|b| box_iou(b.bbox, y.bbox) >= 0.45) {
                bubbles.push(y);
            }
            continue;
        }
        if let Some(existing) = bubbles.iter_mut().find(|b| {
            !b.is_free_text
                && (box_iou(b.bbox, y.bbox) >= 0.35 || box_iomin(b.bbox, y.bbox) >= 0.55)
        }) {
            existing.confirmed_by_two_detectors = true;
            existing.confidence = existing.confidence.max(y.confidence);
        } else if !bubbles
            .iter()
            .any(|b| box_iou(b.bbox, y.bbox) >= 0.42 || box_iomin(b.bbox, y.bbox) >= 0.60)
        {
            bubbles.push(y);
        }
    }

    if detect_free_text {
        for f in rt_cls2 {
            if f.confidence < conf_thresh {
                continue;
            }
            if let Some(existing) = bubbles
                .iter_mut()
                .find(|b| b.is_free_text && box_iou(b.bbox, f.bbox) >= 0.35)
            {
                existing.confirmed_by_two_detectors = true;
                existing.confidence = existing.confidence.max(f.confidence);
            } else if !bubbles.iter().any(|b| box_iou(b.bbox, f.bbox) >= 0.45) {
                bubbles.push(f);
            }
        }
    }

    // 2.5. Подавляем родительские дубли-матрёшки поверх сдвоенных/тройных бабблов (`018.webp`)
    suppress_parent_containers(&mut bubbles);

    // 3. Прикрепляем полигоны бабблов от YOLO26s-seg с обязательной отсечкой Сазерленда–Ходжмана
    // по индивидуальному окну каждого баббла (чтобы сдвоенные/тройные бабблы никогда не делили один общий полигон!).
    let clip_rects: Vec<[i32; 4]> = (0..bubbles.len())
        .map(|idx| compute_neighbor_aware_clip_rect(idx, &bubbles))
        .collect();

    for (idx, b) in bubbles.iter_mut().enumerate() {
        if b.is_free_text {
            continue;
        }
        let clip_rect = clip_rects[idx];
        let mut best_clipped: Option<Vec<[i32; 2]>> = None;
        let mut best_score = 0.0f32;

        for seg in seg_balloons {
            let s = box_iou(b.bbox, seg.bbox).max(box_iomin(b.bbox, seg.bbox) * 0.85);
            if s <= 0.35 {
                continue;
            }
            let clipped = clip_polygon_to_rect(&seg.polygon, clip_rect);
            if clipped.len() < 3 {
                continue;
            }
            let b_area =
                ((b.bbox[2] - b.bbox[0]).max(1) as f64) * ((b.bbox[3] - b.bbox[1]).max(1) as f64);
            let c_area = polygon_area_abs(&clipped);
            if c_area < b_area * 0.15 {
                continue;
            }
            let eff_score = s * ((c_area / b_area).min(1.0) as f32).sqrt();
            if eff_score > best_score {
                best_score = eff_score;
                best_clipped = Some(clipped);
            }
        }
        if let Some(p) = best_clipped {
            b.bubble_polygon = Some(p);
            b.has_enclosing_frame = true;
        } else {
            b.has_enclosing_frame = false;
        }
    }

    bubbles
}
