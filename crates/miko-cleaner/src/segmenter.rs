use image::imageops::{crop_imm, resize, FilterType};
use image::RgbImage;
use miko_ort::ort::session::Session;
use miko_ort::ort::value::Tensor;
use miko_ort::{create_session, ExecutionProviderKind, OrtError};
use std::path::Path;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct LetterboxMeta {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

fn letterbox_640_chw(img: &RgbImage) -> (Vec<f32>, LetterboxMeta) {
    let target = 640u32;
    let (w, h) = img.dimensions();
    let scale = (target as f32 / w.max(1) as f32).min(target as f32 / h.max(1) as f32);
    let nw = ((w as f32 * scale).round() as u32).clamp(1, target);
    let nh = ((h as f32 * scale).round() as u32).clamp(1, target);
    let pad_x = ((target - nw) / 2) as usize;
    let pad_y = ((target - nh) / 2) as usize;

    let resized = resize(img, nw, nh, FilterType::Triangle);
    let t = target as usize;
    let plane = t * t;
    let fill = 114.0f32 / 255.0f32;
    let mut chw = vec![fill; 3 * plane];
    let raw = resized.as_raw();
    let rw = nw as usize;
    let rh = nh as usize;

    for y in 0..rh {
        let dst_row = (y + pad_y) * t + pad_x;
        let src_row = y * rw * 3;
        for x in 0..rw {
            let si = src_row + x * 3;
            let di = dst_row + x;
            chw[di] = raw[si] as f32 / 255.0;
            chw[plane + di] = raw[si + 1] as f32 / 255.0;
            chw[2 * plane + di] = raw[si + 2] as f32 / 255.0;
        }
    }

    (
        chw,
        LetterboxMeta {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
        },
    )
}

fn iou_f32(a: [f32; 4], b: [f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    if inter <= 0.0 {
        return 0.0;
    }
    let aa = (a[2] - a[0]).max(1.0) * (a[3] - a[1]).max(1.0);
    let ab = (b[2] - b[0]).max(1.0) * (b[3] - b[1]).max(1.0);
    inter / (aa + ab - inter).max(1.0)
}

/// GPU-сегментатор текстовых штрихов на базе `comic_text_segmenter.onnx` (YOLOv8m-seg).
pub struct NeuralStrokeSegmenter {
    session: Session,
    pub ep: ExecutionProviderKind,
}

impl NeuralStrokeSegmenter {
    pub fn load(onnx_path: &Path, prefer_gpu: bool) -> Result<Self, OrtError> {
        let (session, ep) = create_session(onnx_path, prefer_gpu)?;
        Ok(Self { session, ep })
    }

    fn run_slice(&mut self, slice_rgb: &RgbImage, conf_thresh: f32) -> Result<Vec<u8>, OrtError> {
        let (sw, sh) = (slice_rgb.width() as usize, slice_rgb.height() as usize);
        let mut out_mask = vec![0u8; sw * sh];
        if sw == 0 || sh == 0 {
            return Ok(out_mask);
        }

        let (chw, meta) = letterbox_640_chw(slice_rgb);
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
        let ch0 = s0.get(1).copied().unwrap_or(37) as usize;
        let anchors = s0.get(2).copied().unwrap_or(8400) as usize;
        let nm = s1.get(1).copied().unwrap_or(32) as usize;
        let mh = s1.get(2).copied().unwrap_or(160) as usize;
        let mw = s1.get(3).copied().unwrap_or(160) as usize;
        let nc = ch0.saturating_sub(4 + nm).max(1);

        struct Cand {
            lb_box: [f32; 4],
            conf: f32,
            coeff: Vec<f32>,
        }
        let mut cands = Vec::new();
        for i in 0..anchors {
            let mut best_score = 0.0f32;
            for c in 0..nc {
                best_score = best_score.max(d0[(4 + c) * anchors + i]);
            }
            if best_score < conf_thresh {
                continue;
            }
            let cx = d0[i];
            let cy = d0[anchors + i];
            let bw = d0[2 * anchors + i];
            let bh = d0[3 * anchors + i];
            if bw < 2.0 || bh < 2.0 {
                continue;
            }
            let mut coeff = Vec::with_capacity(nm);
            for k in 0..nm {
                coeff.push(d0[(4 + nc + k) * anchors + i]);
            }
            cands.push(Cand {
                lb_box: [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5],
                conf: best_score,
                coeff,
            });
        }

        cands.sort_by(|a, b| {
            b.conf
                .partial_cmp(&a.conf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept: Vec<Cand> = Vec::with_capacity(cands.len().min(200));
        for c in cands {
            if !kept.iter().any(|k| iou_f32(k.lb_box, c.lb_box) >= 0.55) {
                kept.push(c);
                if kept.len() >= 200 {
                    break;
                }
            }
        }

        let proto_stride = mh * mw;
        let scale_to_proto = (mw as f32) / 640.0;

        for det in &kept {
            let px1 = (det.lb_box[0] * scale_to_proto)
                .floor()
                .clamp(0.0, (mw - 1) as f32) as usize;
            let py1 = (det.lb_box[1] * scale_to_proto)
                .floor()
                .clamp(0.0, (mh - 1) as f32) as usize;
            let px2 = (det.lb_box[2] * scale_to_proto)
                .ceil()
                .clamp((px1 + 1) as f32, mw as f32) as usize;
            let py2 = (det.lb_box[3] * scale_to_proto)
                .ceil()
                .clamp((py1 + 1) as f32, mh as f32) as usize;

            // Преобразуем диапазон proto-координат в исходные координаты slice_rgb
            let ox1 = (((px1 as f32) / scale_to_proto - meta.pad_x) / meta.scale)
                .floor()
                .clamp(0.0, (sw - 1) as f32) as usize;
            let oy1 = (((py1 as f32) / scale_to_proto - meta.pad_y) / meta.scale)
                .floor()
                .clamp(0.0, (sh - 1) as f32) as usize;
            let ox2 = (((px2 as f32) / scale_to_proto - meta.pad_x) / meta.scale)
                .ceil()
                .clamp((ox1 + 1) as f32, sw as f32) as usize;
            let oy2 = (((py2 as f32) / scale_to_proto - meta.pad_y) / meta.scale)
                .ceil()
                .clamp((oy1 + 1) as f32, sh as f32) as usize;

            for sy in oy1..oy2 {
                let lby = sy as f32 * meta.scale + meta.pad_y;
                let pyf = (lby * scale_to_proto - 0.5).clamp(0.0, (mh - 1) as f32);
                let py0 = (pyf.floor() as usize).min(mh - 1);
                let py_hi = (py0 + 1).min(mh - 1);
                let wy = pyf - py0 as f32;
                let row_off = sy * sw;

                for sx in ox1..ox2 {
                    let lbx = sx as f32 * meta.scale + meta.pad_x;
                    let pxf = (lbx * scale_to_proto - 0.5).clamp(0.0, (mw - 1) as f32);
                    let px0 = (pxf.floor() as usize).min(mw - 1);
                    let px_hi = (px0 + 1).min(mw - 1);
                    let wx = pxf - px0 as f32;

                    let o00 = py0 * mw + px0;
                    let o01 = py0 * mw + px_hi;
                    let o10 = py_hi * mw + px0;
                    let o11 = py_hi * mw + px_hi;

                    let w00 = (1.0 - wx) * (1.0 - wy);
                    let w01 = wx * (1.0 - wy);
                    let w10 = (1.0 - wx) * wy;
                    let w11 = wx * wy;

                    let mut acc = 0.0f32;
                    for k in 0..nm {
                        let base = k * proto_stride;
                        let pval = d1[base + o00] * w00
                            + d1[base + o01] * w01
                            + d1[base + o10] * w10
                            + d1[base + o11] * w11;
                        acc += det.coeff[k] * pval;
                    }
                    let prob = sigmoid(acc);
                    if prob >= 0.25 {
                        let v = (prob * 255.0).round().clamp(0.0, 255.0) as u8;
                        if v > out_mask[row_off + sx] {
                            out_mask[row_off + sx] = v;
                        }
                    }
                }
            }
        }

        Ok(out_mask)
    }

    /// Строит вероятностную маску текстовых штрихов `0..255` для всей страницы (с нарезкой для длинных вебтунов).
    pub fn segment_page(&mut self, img: &RgbImage, conf_thresh: f32) -> Result<Vec<u8>, OrtError> {
        let (w, h) = img.dimensions();
        let wu = w as usize;
        let hu = h as usize;
        if wu == 0 || hu == 0 {
            return Ok(Vec::new());
        }

        if (h as f32) <= (w as f32) * 1.8 {
            return self.run_slice(img, conf_thresh);
        }

        let mut full_mask = vec![0u8; wu * hu];
        let slice_h = ((w as f32 * 1.35).round() as u32).max(640).min(h);
        let overlap = ((w as f32 * 0.25).round() as u32).min(slice_h / 2);
        let step = (slice_h - overlap).max(200);
        let mut y = 0u32;

        while y < h {
            let end_y = (y + slice_h).min(h);
            let real_y = end_y.saturating_sub(slice_h);
            let sh = end_y - real_y;
            let sub = crop_imm(img, 0, real_y, w, sh).to_image();
            let sub_mask = self.run_slice(&sub, conf_thresh)?;
            let oy = real_y as usize;
            let shu = sh as usize;
            for sy in 0..shu {
                let dst_row = (oy + sy) * wu;
                let src_row = sy * wu;
                for x in 0..wu {
                    let v = sub_mask[src_row + x];
                    if v > full_mask[dst_row + x] {
                        full_mask[dst_row + x] = v;
                    }
                }
            }
            if end_y >= h {
                break;
            }
            y += step;
        }

        Ok(full_mask)
    }
}
