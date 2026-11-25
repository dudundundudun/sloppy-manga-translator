use crate::preprocess::{letterbox_yolo_chw, resize_rtdetr_chw};
use image::imageops::crop_imm;
use image::RgbImage;
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
    ) -> Result<(Vec<BubbleDetection>, Vec<BubbleDetection>, Vec<BubbleDetection>), OrtError> {
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
    ) -> Result<(Vec<BubbleDetection>, Vec<BubbleDetection>, Vec<BubbleDetection>), OrtError> {
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

    pub fn detect(&mut self, img: &RgbImage, conf_thresh: f32) -> Result<Vec<BubbleDetection>, OrtError> {
        let (iw, ih) = (img.width() as i32, img.height() as i32);
        let (chw, meta) = letterbox_yolo_chw(img, 640);
        let in_tensor = Tensor::from_array(([1usize, 3, 640, 640], chw))?;
        let in_name = self
            .session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());
        let outputs = self.session.run(miko_ort::ort::inputs![in_name => in_tensor])?;
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
            });
        }
        nms_boxes(&mut cands, 0.45);
        Ok(cands)
    }
}

/// Выпуклая оболочка Эндрю для построения чистого упорядоченного полигона по точкам маски (<= 48 вершин).
fn convex_hull_2d(pts: &mut Vec<[i32; 2]>) -> Vec<[i32; 2]> {
    pts.sort_unstable();
    pts.dedup();
    if pts.len() < 3 {
        return pts.clone();
    }
    let cross = |o: [i32; 2], a: [i32; 2], b: [i32; 2]| -> i64 {
        (a[0] - o[0]) as i64 * (b[1] - o[1]) as i64 - (a[1] - o[1]) as i64 * (b[0] - o[0]) as i64
    };
    let mut lower = Vec::new();
    for &p in pts.iter() {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    if lower.len() > 48 {
        let step = (lower.len() as f32) / 48.0;
        let mut sub = Vec::with_capacity(48);
        for k in 0..48 {
            let idx = ((k as f32 * step).floor() as usize).min(lower.len() - 1);
            sub.push(lower[idx]);
        }
        return sub;
    }
    lower
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

    pub fn detect_balloons(
        &mut self,
        img: &RgbImage,
        conf_thresh: f32,
    ) -> Result<Vec<BalloonPolygon>, OrtError> {
        let (iw, ih) = (img.width() as i32, img.height() as i32);
        let (chw, meta) = letterbox_yolo_chw(img, 640);
        let in_tensor = Tensor::from_array(([1usize, 3, 640, 640], chw))?;
        let in_name = self
            .session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());
        let outputs = self.session.run(miko_ort::ort::inputs![in_name => in_tensor])?;

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
            let bx1 = x1.clamp(0, iw - 1);
            let by1 = y1.clamp(0, ih - 1);
            let bx2 = x2.clamp(bx1 + 1, iw);
            let by2 = y2.clamp(by1 + 1, ih);
            if (bx2 - bx1) < 16 || (by2 - by1) < 16 {
                continue;
            }
            let mut coeff = Vec::with_capacity(nm);
            for k in 0..nm {
                coeff.push(d0[(4 + nc + k) * anchors + i]);
            }
            raw_dets.push(RawMaskDet {
                bbox: [bx1, by1, bx2, by2],
                lb_box: [lcx - lbw * 0.5, lcy - lbh * 0.5, lcx + lbw * 0.5, lcy + lbh * 0.5],
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
            if out.iter().any(|k: &BalloonPolygon| box_iou(k.bbox, r.bbox) >= 0.5) {
                continue;
            }
            let px1 = (r.lb_box[0] * scale_to_proto).floor().clamp(0.0, (mw - 1) as f32) as usize;
            let py1 = (r.lb_box[1] * scale_to_proto).floor().clamp(0.0, (mh - 1) as f32) as usize;
            let px2 = (r.lb_box[2] * scale_to_proto).ceil().clamp((px1 + 1) as f32, mw as f32) as usize;
            let py2 = (r.lb_box[3] * scale_to_proto).ceil().clamp((py1 + 1) as f32, mh as f32) as usize;

            let mut boundary_pts = Vec::new();
            for py in py1..py2 {
                let mut row_left: Option<usize> = None;
                let mut row_right: Option<usize> = None;
                for px in px1..px2 {
                    let mut sum = 0.0f32;
                    let off = py * mw + px;
                    for k in 0..nm {
                        sum += r.coeff[k] * d1[k * stride + off];
                    }
                    if sigmoid(sum) > 0.5 {
                        if row_left.is_none() {
                            row_left = Some(px);
                        }
                        row_right = Some(px);
                    }
                }
                if let (Some(lx), Some(rx)) = (row_left, row_right) {
                    let map_pt = |gx: usize, gy: usize| -> [i32; 2] {
                        let ox = (((gx as f32 + 0.5) / scale_to_proto - meta.pad_x) / meta.scale)
                            .round() as i32;
                        let oy = (((gy as f32 + 0.5) / scale_to_proto - meta.pad_y) / meta.scale)
                            .round() as i32;
                        [ox.clamp(0, iw), oy.clamp(0, ih)]
                    };
                    boundary_pts.push(map_pt(lx, py));
                    if rx > lx {
                        boundary_pts.push(map_pt(rx, py));
                    }
                }
            }
            let poly = convex_hull_2d(&mut boundary_pts);
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

    // 2. Добавляем unmatched text_bubble из cls1 и из YOLOv8m
    for t in rt_cls1.into_iter().chain(yolo_dets.into_iter()) {
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

    if detect_free_text {
        for f in rt_cls2 {
            if f.confidence >= conf_thresh && !bubbles.iter().any(|b| box_iou(b.bbox, f.bbox) >= 0.45)
            {
                bubbles.push(f);
            }
        }
    }

    // 3. Прикрепляем полигоны бабблов от YOLO26s-seg
    for b in bubbles.iter_mut() {
        if b.is_free_text {
            continue;
        }
        let mut best_poly: Option<&Vec<[i32; 2]>> = None;
        let mut best_score = 0.0f32;
        for seg in seg_balloons {
            let s = box_iou(b.bbox, seg.bbox).max(box_iomin(b.bbox, seg.bbox) * 0.85);
            if s > 0.35 && s > best_score {
                best_score = s;
                best_poly = Some(&seg.polygon);
            }
        }
        if let Some(p) = best_poly {
            b.bubble_polygon = Some(p.clone());
        }
    }

    bubbles
}
