use image::imageops::{resize, FilterType};
use image::{Rgb, RgbImage};

pub const NAR_BUDGET: usize = 324;
pub const NAR_PATCH: usize = 14;
pub const NAR_FEAT: usize = 588;
pub const NAR_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const NAR_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

#[derive(Debug, Clone, Copy)]
pub struct LetterboxMeta {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub target: u32,
}

#[derive(Debug, Clone)]
pub struct NarPreprocessOutput {
    pub p: Vec<f32>,
    pub v: Vec<f32>,
    pub r: Vec<i32>,
    pub c: Vec<i32>,
    pub h: i32,
    pub w: i32,
}

/// Подготовка изображения под RT-DETR-v2 (`640x640` CHW, `/ 255.0`, без вычитания mean/std).
pub fn resize_rtdetr_chw(img: &RgbImage, target: u32) -> Vec<f32> {
    let resized = resize(img, target, target, FilterType::Triangle);
    let tw = target as usize;
    let th = target as usize;
    let plane = tw * th;
    let mut chw = vec![0.0f32; 3 * plane];
    let raw = resized.as_raw();
    for i in 0..plane {
        let off = i * 3;
        chw[i] = (raw[off] as f32) / 255.0;
        chw[plane + i] = (raw[off + 1] as f32) / 255.0;
        chw[2 * plane + i] = (raw[off + 2] as f32) / 255.0;
    }
    chw
}

/// Letterbox-препроцессинг под YOLOv8 / YOLO26-seg (`640x640` CHW, серый паддинг `114/255.0`).
pub fn letterbox_yolo_chw(img: &RgbImage, target: u32) -> (Vec<f32>, LetterboxMeta) {
    let (w, h) = img.dimensions();
    let r = (target as f32 / w.max(1) as f32).min(target as f32 / h.max(1) as f32);
    let nw = ((w as f32 * r).round() as u32).max(1).min(target);
    let nh = ((h as f32 * r).round() as u32).max(1).min(target);
    let pad_x_u = (target - nw) / 2;
    let pad_y_u = (target - nh) / 2;

    let scaled = resize(img, nw, nh, FilterType::Triangle);
    let tw = target as usize;
    let th = target as usize;
    let plane = tw * th;
    let fill = 114.0f32 / 255.0;
    let mut chw = vec![fill; 3 * plane];
    let raw = scaled.as_raw();
    let nwu = nw as usize;
    let nhu = nh as usize;
    let px = pad_x_u as usize;
    let py = pad_y_u as usize;

    for y in 0..nhu {
        let src_row = y * nwu * 3;
        let dst_row = (py + y) * tw + px;
        for x in 0..nwu {
            let s = src_row + x * 3;
            let d = dst_row + x;
            chw[d] = (raw[s] as f32) / 255.0;
            chw[plane + d] = (raw[s + 1] as f32) / 255.0;
            chw[2 * plane + d] = (raw[s + 2] as f32) / 255.0;
        }
    }

    let meta = LetterboxMeta {
        scale: r,
        pad_x: pad_x_u as f32,
        pad_y: pad_y_u as f32,
        target,
    };
    (chw, meta)
}

/// Точный бинарный поиск масштаба сетки патчей `14x14` (`<= 324` патчей) для `manga-ocr-nar-small`.
pub fn pad_grid(height: u32, width: u32, budget: usize) -> (f64, usize, usize, u32, u32) {
    let h_f = height.max(1) as f64;
    let w_f = width.max(1) as f64;
    let patch_f = NAR_PATCH as f64;
    let mut lo = 1e-4f64;
    let mut hi = 1000.0f64;
    for _ in 0..60 {
        let mid = (lo + hi) * 0.5;
        let gh = ((h_f * mid) / patch_f).ceil() as usize;
        let gw = ((w_f * mid) / patch_f).ceil() as usize;
        if gh.saturating_mul(gw) <= budget {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let gh = ((h_f * lo) / patch_f).ceil().max(1.0) as usize;
    let gw = ((w_f * lo) / patch_f).ceil().max(1.0) as usize;
    let rw = ((w_f * lo).round() as u32)
        .max(1)
        .min((gw * NAR_PATCH) as u32);
    let rh = ((h_f * lo).round() as u32)
        .max(1)
        .min((gh * NAR_PATCH) as u32);
    (lo, gh, gw, rw, rh)
}

fn median_u8(vals: &mut [u8]) -> u8 {
    if vals.is_empty() {
        return 255;
    }
    let mid = vals.len() / 2;
    vals.select_nth_unstable(mid);
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        let upper = vals[mid] as u16;
        let lower = *vals[..mid].iter().max().unwrap_or(&vals[mid]) as u16;
        ((upper + lower) / 2) as u8
    }
}

/// Вырезает кроп с белой/граничной каймой `pad` px (аналог `_crop_with_padding`).
pub fn crop_with_padding(img: &RgbImage, bbox: [i32; 4], pad: i32) -> RgbImage {
    let (iw, ih) = img.dimensions();
    let x1 = bbox[0].clamp(0, iw.saturating_sub(1) as i32) as u32;
    let y1 = bbox[1].clamp(0, ih.saturating_sub(1) as i32) as u32;
    let x2 = bbox[2].clamp(x1 as i32 + 1, iw as i32) as u32;
    let y2 = bbox[3].clamp(y1 as i32 + 1, ih as i32) as u32;
    let cw = x2 - x1;
    let ch = y2 - y1;
    let p = pad.max(0) as u32;

    let mut out = RgbImage::from_pixel(cw + 2 * p, ch + 2 * p, Rgb([255, 255, 255]));
    for y in 0..ch {
        for x in 0..cw {
            let px = img.get_pixel(x1 + x, y1 + y);
            out.put_pixel(p + x, p + y, *px);
        }
    }
    out
}

/// Полный порт `manga_ocr_nar_small/preprocess.py` для подачи прямо в `manga_ocr_nar_small.onnx`.
pub fn preprocess_nar_crop(crop: &RgbImage) -> NarPreprocessOutput {
    let (_, gh, gw, rw, rh) = pad_grid(crop.height(), crop.width(), NAR_BUDGET);
    let small = resize(crop, rw, rh, FilterType::CatmullRom);
    let sraw = small.as_raw();
    let rwu = rw as usize;
    let rhu = rh as usize;

    let border_cap = 2 * rwu + 2 * rhu;
    let mut rs = Vec::with_capacity(border_cap);
    let mut gs = Vec::with_capacity(border_cap);
    let mut bs = Vec::with_capacity(border_cap);
    for x in 0..rwu {
        let top = x * 3;
        let bot = ((rhu - 1) * rwu + x) * 3;
        rs.push(sraw[top]);
        gs.push(sraw[top + 1]);
        bs.push(sraw[top + 2]);
        rs.push(sraw[bot]);
        gs.push(sraw[bot + 1]);
        bs.push(sraw[bot + 2]);
    }
    for y in 0..rhu {
        let left = (y * rwu) * 3;
        let right = (y * rwu + rwu - 1) * 3;
        rs.push(sraw[left]);
        gs.push(sraw[left + 1]);
        bs.push(sraw[left + 2]);
        rs.push(sraw[right]);
        gs.push(sraw[right + 1]);
        bs.push(sraw[right + 2]);
    }
    let fill_rgb = [median_u8(&mut rs), median_u8(&mut gs), median_u8(&mut bs)];

    let cw = gw * NAR_PATCH;
    let ch = gh * NAR_PATCH;
    let mut canvas = vec![0u8; cw * ch * 3];
    for px in canvas.chunks_exact_mut(3) {
        px.copy_from_slice(&fill_rgb);
    }
    for y in 0..rhu {
        let src_row = &sraw[y * rwu * 3..(y + 1) * rwu * 3];
        let dst_off = y * cw * 3;
        canvas[dst_off..dst_off + rwu * 3].copy_from_slice(src_row);
    }

    let n_valid = (gh * gw).min(NAR_BUDGET);
    let mut p = vec![0.0f32; NAR_BUDGET * NAR_FEAT];
    let mut v = vec![0.0f32; NAR_BUDGET];
    let mut r = vec![0i32; NAR_BUDGET];
    let mut c = vec![0i32; NAR_BUDGET];

    for patch_idx in 0..n_valid {
        let py = patch_idx / gw;
        let px = patch_idx % gw;
        v[patch_idx] = 1.0;
        r[patch_idx] = py as i32;
        c[patch_idx] = px as i32;

        let p_base = patch_idx * NAR_FEAT;
        for ch_idx in 0..3usize {
            let mean = NAR_MEAN[ch_idx];
            let std = NAR_STD[ch_idx];
            let ch_off = ch_idx * (NAR_PATCH * NAR_PATCH);
            for dy in 0..NAR_PATCH {
                let cy = py * NAR_PATCH + dy;
                let row_base = cy * cw * 3;
                for dx in 0..NAR_PATCH {
                    let cx = px * NAR_PATCH + dx;
                    let raw_val = canvas[row_base + cx * 3 + ch_idx] as f32;
                    let norm_val = (raw_val / 255.0 - mean) / std;
                    p[p_base + ch_off + dy * NAR_PATCH + dx] = norm_val;
                }
            }
        }
    }

    NarPreprocessOutput {
        p,
        v,
        r,
        c,
        h: gh as i32,
        w: gw as i32,
    }
}
