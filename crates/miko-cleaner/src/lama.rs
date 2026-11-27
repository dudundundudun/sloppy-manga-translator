use image::RgbImage;
use miko_ort::ort::session::Session;
use miko_ort::ort::value::Tensor;
use miko_ort::{create_session, ExecutionProviderKind, OrtError};
use std::path::Path;

const LAMA_SIZE: usize = 512;

/// Нативный GPU-инпейнтер на базе `anime_lama.onnx` (`dreMaz/AnimeMangaInpainting`, 512×512).
pub struct AnimeLamaInpainter {
    session: Session,
    pub ep: ExecutionProviderKind,
}

#[inline]
fn reflect_idx(i: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (len as isize - 1) * 2;
    let mut m = i % period;
    if m < 0 {
        m += period;
    }
    if m >= len as isize {
        (period - m) as usize
    } else {
        m as usize
    }
}

#[inline]
fn sample_chw_bilinear(chw: &[f32], w: usize, h: usize, c: usize, fx: f32, fy: f32) -> f32 {
    let x = fx.clamp(0.0, (w - 1) as f32);
    let y = fy.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let plane = w * h;
    let off = c * plane;
    let v00 = chw[off + y0 * w + x0];
    let v01 = chw[off + y0 * w + x1];
    let v10 = chw[off + y1 * w + x0];
    let v11 = chw[off + y1 * w + x1];
    let top = v00 * (1.0 - dx) + v01 * dx;
    let bot = v10 * (1.0 - dx) + v11 * dx;
    top * (1.0 - dy) + bot * dy
}

#[inline]
fn sample_rgb_bilinear_norm(rgb: &[u8], w: usize, h: usize, c: usize, fx: f32, fy: f32) -> f32 {
    let x = fx.clamp(0.0, (w - 1) as f32);
    let y = fy.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let v00 = rgb[(y0 * w + x0) * 3 + c] as f32;
    let v01 = rgb[(y0 * w + x1) * 3 + c] as f32;
    let v10 = rgb[(y1 * w + x0) * 3 + c] as f32;
    let v11 = rgb[(y1 * w + x1) * 3 + c] as f32;
    let top = v00 * (1.0 - dx) + v01 * dx;
    let bot = v10 * (1.0 - dx) + v11 * dx;
    (top * (1.0 - dy) + bot * dy) / 255.0
}

#[inline]
fn downsample_mask_box_any(mask: &[u8], src_w: usize, src_h: usize, tx: usize, ty: usize) -> f32 {
    let x0 = (tx * src_w) / LAMA_SIZE;
    let x1 = (((tx + 1) * src_w).div_ceil(LAMA_SIZE)).clamp(x0 + 1, src_w);
    let y0 = (ty * src_h) / LAMA_SIZE;
    let y1 = (((ty + 1) * src_h).div_ceil(LAMA_SIZE)).clamp(y0 + 1, src_h);
    for y in y0..y1 {
        let row = y * src_w;
        for x in x0..x1 {
            if mask[row + x] > 0 {
                return 1.0;
            }
        }
    }
    0.0
}

impl AnimeLamaInpainter {
    pub fn load(onnx_path: &Path, prefer_gpu: bool) -> Result<Self, OrtError> {
        let (session, ep) = create_session(onnx_path, prefer_gpu)?;
        Ok(Self { session, ep })
    }

    fn run_512(&mut self, img_chw: Vec<f32>, mask_chw: Vec<f32>) -> Result<Vec<f32>, OrtError> {
        let img_t = Tensor::from_array(([1usize, 3, LAMA_SIZE, LAMA_SIZE], img_chw))?;
        let mask_t = Tensor::from_array(([1usize, 1, LAMA_SIZE, LAMA_SIZE], mask_chw))?;
        let outputs = self
            .session
            .run(miko_ort::ort::inputs!["image" => img_t, "mask" => mask_t])?;
        if let Some(v) = outputs.get("inpainted") {
            let (_shape, slice) = v.try_extract_tensor::<f32>()?;
            return Ok(slice.to_vec());
        }
        if let Some((_, v)) = outputs.iter().next() {
            let (_shape, slice) = v.try_extract_tensor::<f32>()?;
            return Ok(slice.to_vec());
        }
        Err(OrtError::UnsupportedInput {
            name: "inpainted".to_string(),
            reason: "Модель anime_lama.onnx не вернула выходной тензор".to_string(),
        })
    }

    /// Инпейнтит кроп `crop_rect` с использованием широкого `512×512` контекста исходной страницы.
    /// Возвращает RGB-буфер размера `(cw * ch * 3)`, изменяя ТОЛЬКО пиксели с `crop_mask[i] > 0`.
    pub fn inpaint_on_page(
        &mut self,
        page: &RgbImage,
        crop_rect: [i32; 4],
        crop_mask: &[u8],
    ) -> Result<Vec<u8>, OrtError> {
        let (pw, ph) = page.dimensions();
        let rx0 = crop_rect[0].clamp(0, pw as i32);
        let ry0 = crop_rect[1].clamp(0, ph as i32);
        let rx1 = crop_rect[2].clamp(rx0, pw as i32);
        let ry1 = crop_rect[3].clamp(ry0, ph as i32);
        let cw = (rx1 - rx0) as usize;
        let ch = (ry1 - ry0) as usize;
        let mut out = vec![0u8; cw * ch * 3];
        if cw == 0 || ch == 0 || crop_mask.len() < cw * ch {
            return Ok(out);
        }

        let pwi = pw as usize;
        let phi = ph as usize;
        let raw_page = page.as_raw();
        for ry in 0..ch {
            let py = (ry0 as usize) + ry;
            let src_off = (py * pwi + (rx0 as usize)) * 3;
            let dst_off = ry * cw * 3;
            out[dst_off..dst_off + cw * 3].copy_from_slice(&raw_page[src_off..src_off + cw * 3]);
        }

        if !crop_mask[..cw * ch].iter().any(|&v| v > 0) {
            return Ok(out);
        }

        let plane = LAMA_SIZE * LAMA_SIZE;
        if cw <= 464 && ch <= 464 {
            let cx = (rx0 + rx1) / 2;
            let cy = (ry0 + ry1) / 2;
            let wx0 = if pw as i32 >= LAMA_SIZE as i32 {
                (cx - LAMA_SIZE as i32 / 2).clamp(0, pw as i32 - LAMA_SIZE as i32)
            } else {
                (pw as i32 - LAMA_SIZE as i32) / 2
            };
            let wy0 = if ph as i32 >= LAMA_SIZE as i32 {
                (cy - LAMA_SIZE as i32 / 2).clamp(0, ph as i32 - LAMA_SIZE as i32)
            } else {
                (ph as i32 - LAMA_SIZE as i32) / 2
            };

            let mut img_chw = vec![0.0f32; 3 * plane];
            let mut mask_chw = vec![0.0f32; plane];
            for ty in 0..LAMA_SIZE {
                let py_i = wy0 + ty as i32;
                let py_ref = reflect_idx(py_i as isize, phi);
                let in_y = py_i >= ry0 && py_i < ry1;
                for tx in 0..LAMA_SIZE {
                    let px_i = wx0 + tx as i32;
                    let px_ref = reflect_idx(px_i as isize, pwi);
                    let src_idx = (py_ref * pwi + px_ref) * 3;
                    let dst_idx = ty * LAMA_SIZE + tx;
                    img_chw[dst_idx] = raw_page[src_idx] as f32 / 255.0;
                    img_chw[plane + dst_idx] = raw_page[src_idx + 1] as f32 / 255.0;
                    img_chw[2 * plane + dst_idx] = raw_page[src_idx + 2] as f32 / 255.0;

                    if in_y && px_i >= rx0 && px_i < rx1 {
                        let mx = (px_i - rx0) as usize;
                        let my = (py_i - ry0) as usize;
                        if crop_mask[my * cw + mx] > 0 {
                            mask_chw[dst_idx] = 1.0;
                        }
                    }
                }
            }

            let pred = self.run_512(img_chw, mask_chw)?;
            if pred.len() >= 3 * plane {
                for ry in 0..ch {
                    let ty = (ry0 + ry as i32 - wy0) as usize;
                    for rx in 0..cw {
                        let ci = ry * cw + rx;
                        if crop_mask[ci] > 0 {
                            let tx = (rx0 + rx as i32 - wx0) as usize;
                            let si = ty * LAMA_SIZE + tx;
                            let di = ci * 3;
                            out[di] = (pred[si].clamp(0.0, 1.0) * 255.0).round() as u8;
                            out[di + 1] = (pred[plane + si].clamp(0.0, 1.0) * 255.0).round() as u8;
                            out[di + 2] =
                                (pred[2 * plane + si].clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    }
                }
            }
            return Ok(out);
        }

        // Большой блок (> 464px): вырезаем максимально широкое квадратное окно вокруг кропа и масштабируем в 512×512
        let side = (cw.max(ch) + 64).min(pwi.max(phi));
        let win_w = side.min(pwi);
        let win_h = side.min(phi);
        let cx = (rx0 + rx1) / 2;
        let cy = (ry0 + ry1) / 2;
        let wx0 = (cx - win_w as i32 / 2).clamp(0, (pwi - win_w) as i32) as usize;
        let wy0 = (cy - win_h as i32 / 2).clamp(0, (phi - win_h) as i32) as usize;

        let mut win_rgb = vec![0u8; win_w * win_h * 3];
        let mut win_mask = vec![0u8; win_w * win_h];
        for wy in 0..win_h {
            let py = wy0 + wy;
            let src_off = (py * pwi + wx0) * 3;
            let dst_off = wy * win_w * 3;
            win_rgb[dst_off..dst_off + win_w * 3]
                .copy_from_slice(&raw_page[src_off..src_off + win_w * 3]);
        }
        for ry in 0..ch {
            let py = ry0 as usize + ry;
            if py < wy0 || py >= wy0 + win_h {
                continue;
            }
            let wy = py - wy0;
            for rx in 0..cw {
                let px = rx0 as usize + rx;
                if px >= wx0 && px < wx0 + win_w && crop_mask[ry * cw + rx] > 0 {
                    win_mask[wy * win_w + (px - wx0)] = 255;
                }
            }
        }

        self.inpaint_crop_rgb(&mut win_rgb, &win_mask, win_w, win_h)?;
        for ry in 0..ch {
            let py = ry0 as usize + ry;
            if py < wy0 || py >= wy0 + win_h {
                continue;
            }
            let wy = py - wy0;
            for rx in 0..cw {
                let ci = ry * cw + rx;
                let px = rx0 as usize + rx;
                if px >= wx0 && px < wx0 + win_w && crop_mask[ci] > 0 {
                    let si = (wy * win_w + (px - wx0)) * 3;
                    let di = ci * 3;
                    out[di] = win_rgb[si];
                    out[di + 1] = win_rgb[si + 1];
                    out[di + 2] = win_rgb[si + 2];
                }
            }
        }
        Ok(out)
    }

    /// Инпейнтит предварительно подготовленный (экранированный) RGB-кроп `cw × ch` на месте,
    /// модифицируя ТОЛЬКО пиксели с `mask[i] > 0`.
    pub fn inpaint_crop_rgb(
        &mut self,
        crop_rgb: &mut [u8],
        mask: &[u8],
        cw: usize,
        ch: usize,
    ) -> Result<(), OrtError> {
        if cw == 0 || ch == 0 || crop_rgb.len() < cw * ch * 3 || mask.len() < cw * ch {
            return Ok(());
        }
        if !mask[..cw * ch].iter().any(|&v| v > 0) {
            return Ok(());
        }

        let plane = LAMA_SIZE * LAMA_SIZE;
        let mut img_chw = vec![0.0f32; 3 * plane];
        let mut mask_chw = vec![0.0f32; plane];

        if cw <= LAMA_SIZE && ch <= LAMA_SIZE {
            let ox = (LAMA_SIZE - cw) / 2;
            let oy = (LAMA_SIZE - ch) / 2;
            for ty in 0..LAMA_SIZE {
                let sy = reflect_idx(ty as isize - oy as isize, ch);
                let in_y = ty >= oy && ty < oy + ch;
                for tx in 0..LAMA_SIZE {
                    let sx = reflect_idx(tx as isize - ox as isize, cw);
                    let s = (sy * cw + sx) * 3;
                    let d = ty * LAMA_SIZE + tx;
                    img_chw[d] = crop_rgb[s] as f32 / 255.0;
                    img_chw[plane + d] = crop_rgb[s + 1] as f32 / 255.0;
                    img_chw[2 * plane + d] = crop_rgb[s + 2] as f32 / 255.0;

                    if in_y && tx >= ox && tx < ox + cw && mask[(ty - oy) * cw + (tx - ox)] > 0 {
                        mask_chw[d] = 1.0;
                    }
                }
            }

            let pred = self.run_512(img_chw, mask_chw)?;
            if pred.len() >= 3 * plane {
                for ry in 0..ch {
                    let ty = oy + ry;
                    for rx in 0..cw {
                        let ci = ry * cw + rx;
                        if mask[ci] > 0 {
                            let si = ty * LAMA_SIZE + (ox + rx);
                            let di = ci * 3;
                            crop_rgb[di] = (pred[si].clamp(0.0, 1.0) * 255.0).round() as u8;
                            crop_rgb[di + 1] =
                                (pred[plane + si].clamp(0.0, 1.0) * 255.0).round() as u8;
                            crop_rgb[di + 2] =
                                (pred[2 * plane + si].clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    }
                }
            }
            return Ok(());
        }

        // Кроп крупнее 512px по одной/обеим осям: билинейный ресайз в 512×512 и обратно
        let sx_scale = (cw - 1) as f32 / (LAMA_SIZE - 1) as f32;
        let sy_scale = (ch - 1) as f32 / (LAMA_SIZE - 1) as f32;
        for ty in 0..LAMA_SIZE {
            let fy = ty as f32 * sy_scale;
            for tx in 0..LAMA_SIZE {
                let fx = tx as f32 * sx_scale;
                let d = ty * LAMA_SIZE + tx;
                img_chw[d] = sample_rgb_bilinear_norm(crop_rgb, cw, ch, 0, fx, fy);
                img_chw[plane + d] = sample_rgb_bilinear_norm(crop_rgb, cw, ch, 1, fx, fy);
                img_chw[2 * plane + d] = sample_rgb_bilinear_norm(crop_rgb, cw, ch, 2, fx, fy);
                mask_chw[d] = downsample_mask_box_any(mask, cw, ch, tx, ty);
            }
        }

        let pred = self.run_512(img_chw, mask_chw)?;
        if pred.len() >= 3 * plane {
            let inv_x = (LAMA_SIZE - 1) as f32 / (cw - 1).max(1) as f32;
            let inv_y = (LAMA_SIZE - 1) as f32 / (ch - 1).max(1) as f32;
            for ry in 0..ch {
                let fy = ry as f32 * inv_y;
                for rx in 0..cw {
                    let ci = ry * cw + rx;
                    if mask[ci] > 0 {
                        let fx = rx as f32 * inv_x;
                        let di = ci * 3;
                        crop_rgb[di] = (sample_chw_bilinear(&pred, LAMA_SIZE, LAMA_SIZE, 0, fx, fy)
                            .clamp(0.0, 1.0)
                            * 255.0)
                            .round() as u8;
                        crop_rgb[di + 1] =
                            (sample_chw_bilinear(&pred, LAMA_SIZE, LAMA_SIZE, 1, fx, fy)
                                .clamp(0.0, 1.0)
                                * 255.0)
                                .round() as u8;
                        crop_rgb[di + 2] =
                            (sample_chw_bilinear(&pred, LAMA_SIZE, LAMA_SIZE, 2, fx, fy)
                                .clamp(0.0, 1.0)
                                * 255.0)
                                .round() as u8;
                    }
                }
            }
        }
        Ok(())
    }
}
