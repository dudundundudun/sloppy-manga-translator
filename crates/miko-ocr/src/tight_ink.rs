use image::RgbImage;
use miko_cleaner::connected_components;

fn luma_u8(r: u8, g: u8, b: u8) -> u8 {
    ((0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32).round() as i32).clamp(0, 255) as u8
}

fn percentile_u8(sorted_or_raw: &[u8], pct: f32) -> f32 {
    if sorted_or_raw.is_empty() {
        return 255.0;
    }
    let mut buf = sorted_or_raw.to_vec();
    let max_idx = buf.len() - 1;
    let idx = (((max_idx) as f32) * (pct / 100.0)).round() as usize;
    let (_, val, _) = buf.select_nth_unstable(idx.min(max_idx));
    *val as f32
}

/// Быстрая проверка наличия реальных текстовых чернил (яркость + цветовая насыщенность для вебтунов).
pub fn has_meaningful_ink(crop: &RgbImage, is_free_text: bool) -> (bool, usize) {
    let raw = crop.as_raw();
    let n = (crop.width() * crop.height()) as usize;
    if n < 36 {
        return (false, 0);
    }
    let mut lumas = Vec::with_capacity(n);
    let mut chromas = Vec::with_capacity(n);
    for px in raw.chunks_exact(3) {
        lumas.push(luma_u8(px[0], px[1], px[2]));
        let max_c = px[0].max(px[1]).max(px[2]);
        let min_c = px[0].min(px[1]).min(px[2]);
        chromas.push(max_c - min_c);
    }

    let p50 = percentile_u8(&lumas, 50.0);
    let p75 = percentile_u8(&lumas, 75.0);
    let p90 = percentile_u8(&lumas, 90.0);
    let p10 = percentile_u8(&lumas, 10.0);
    let chroma_med = percentile_u8(&chromas, 50.0);

    let dark_cnt = if p75 >= 200.0 {
        let thr = (p75 - 24.0).max(145.0) as u8;
        lumas
            .iter()
            .zip(chromas.iter())
            .filter(|(&y, &c)| y < thr || (chroma_med <= 14.0 && c >= 42))
            .count()
    } else if p50 < 110.0 && p90 >= 170.0 {
        let thr = (p50 + 42.0).max(165.0) as u8;
        lumas.iter().filter(|&&y| y > thr).count()
    } else {
        if (p90 - p10) < if is_free_text { 32.0 } else { 22.0 } {
            return (false, 0);
        }
        let thr = (p50 - 30.0).max(95.0) as u8;
        lumas
            .iter()
            .zip(chromas.iter())
            .filter(|(&y, &c)| y < thr || c >= 48)
            .count()
    };

    let min_ink = if is_free_text { 20 } else { 12 };
    (dark_cnt >= min_ink, dark_cnt)
}

/// Вычисляет плотный `text_bbox` внутри баббла (включая цветные буквы вебтунов и небольшой выступ за рамку <= 14 px).
pub fn compute_tight_ink_bbox(
    img: &RgbImage,
    bubble_bbox: [i32; 4],
    hint_text_bbox: Option<[i32; 4]>,
    is_free_text: bool,
) -> [i32; 4] {
    let img_w = img.width() as i32;
    let img_h = img.height() as i32;
    let bx1 = bubble_bbox[0].clamp(0, img_w - 2);
    let by1 = bubble_bbox[1].clamp(0, img_h - 2);
    let bx2 = bubble_bbox[2].clamp(bx1 + 2, img_w);
    let by2 = bubble_bbox[3].clamp(by1 + 2, img_h);
    let bw = bx2 - bx1;
    let bh = by2 - by1;
    let max_overhang = 24i32.min(((bw.min(bh) as f32) * 0.25) as i32).max(4);

    let valid_hint = hint_text_bbox.and_then(|h| {
        let hx1 = h[0].clamp((bx1 - max_overhang).max(0), img_w - 2);
        let hy1 = h[1].clamp((by1 - max_overhang).max(0), img_h - 2);
        let hx2 = h[2].clamp(hx1 + 2, (bx2 + max_overhang).min(img_w));
        let hy2 = h[3].clamp(hy1 + 2, (by2 + max_overhang).min(img_h));
        if (hx2 - hx1) >= 10 && (hy2 - hy1) >= 10 {
            Some([hx1, hy1, hx2, hy2])
        } else {
            None
        }
    });

    if bw < 22 || bh < 22 || is_free_text {
        return valid_hint.unwrap_or([bx1, by1, bx2, by2]);
    }

    let pad_in_x = ((bw as f32) * 0.02).round().max(1.0) as i32;
    let pad_in_y = ((bh as f32) * 0.02).round().max(1.0) as i32;
    let ix1 = (bx1 + pad_in_x).min(bx2 - 2);
    let iy1 = (by1 + pad_in_y).min(by2 - 2);
    let ix2 = (bx2 - pad_in_x).max(ix1 + 2);
    let iy2 = (by2 - pad_in_y).max(iy1 + 2);
    let rw = (ix2 - ix1) as usize;
    let rh = (iy2 - iy1) as usize;
    if rw < 12 || rh < 12 {
        return valid_hint.unwrap_or([bx1, by1, bx2, by2]);
    }

    let mut lumas = Vec::with_capacity(rw * rh);
    let mut chromas = Vec::with_capacity(rw * rh);
    for y in iy1..iy2 {
        for x in ix1..ix2 {
            let p = img.get_pixel(x as u32, y as u32);
            lumas.push(luma_u8(p[0], p[1], p[2]));
            chromas.push(p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2]));
        }
    }

    let p75 = percentile_u8(&lumas, 75.0);
    let p50 = percentile_u8(&lumas, 50.0);
    let chroma_med = percentile_u8(&chromas, 50.0);
    let mut binary = vec![0u8; rw * rh];

    if p75 >= 210.0 {
        let ink_thr = (p75 - 18.0).max(165.0) as u8;
        let color_thr = if chroma_med <= 14.0 {
            ((chroma_med + 32.0).round() as u8).max(42)
        } else {
            255
        };
        for i in 0..rw * rh {
            if lumas[i] < ink_thr || chromas[i] >= color_thr {
                binary[i] = 255;
            }
        }
    } else if p50 >= 127.0 {
        let thr = (p50 - 25.0).max(95.0) as u8;
        for i in 0..rw * rh {
            if lumas[i] < thr {
                binary[i] = 255;
            }
        }
    } else {
        let thr = (p50 + 35.0).min(175.0) as u8;
        for i in 0..rw * rh {
            if lumas[i] > thr {
                binary[i] = 255;
            }
        }
    }

    let cc = connected_components(&binary, rw, rh);
    let mut min_cx = rw;
    let mut min_cy = rh;
    let mut max_cx = 0usize;
    let mut max_cy = 0usize;
    let mut kept_px = 0usize;

    for comp in &cc.stats {
        if comp.area < 3 {
            continue;
        }
        let touches_l = comp.left <= 1;
        let touches_t = comp.top <= 1;
        let touches_r = comp.left + comp.width >= rw.saturating_sub(1);
        let touches_b = comp.top + comp.height >= rh.saturating_sub(1);
        let touches = [touches_l, touches_t, touches_r, touches_b]
            .into_iter()
            .filter(|&b| b)
            .count();

        if touches >= 2 && (comp.width as f32 > rw as f32 * 0.65 || comp.height as f32 > rh as f32 * 0.65) {
            continue;
        }
        if touches >= 1 && (comp.width as f32 > rw as f32 * 0.78 || comp.height as f32 > rh as f32 * 0.78) {
            continue;
        }
        if touches >= 1
            && (comp.width <= 4 || comp.height <= 4)
            && (comp.width as f32 > rw as f32 * 0.35 || comp.height as f32 > rh as f32 * 0.35)
        {
            continue;
        }

        min_cx = min_cx.min(comp.left);
        min_cy = min_cy.min(comp.top);
        max_cx = max_cx.max(comp.left + comp.width);
        max_cy = max_cy.max(comp.top + comp.height);
        kept_px += comp.area;
    }

    if kept_px < 12 || max_cx <= min_cx || max_cy <= min_cy {
        return valid_hint.unwrap_or([bx1, by1, bx2, by2]);
    }

    let clamp_x1 = valid_hint.map_or(bx1, |h| bx1.min(h[0]));
    let clamp_y1 = valid_hint.map_or(by1, |h| by1.min(h[1]));
    let clamp_x2 = valid_hint.map_or(bx2, |h| bx2.max(h[2]));
    let clamp_y2 = valid_hint.map_or(by2, |h| by2.max(h[3]));

    let ink_x1 = (ix1 + min_cx as i32 - 4).max(clamp_x1);
    let ink_y1 = (iy1 + min_cy as i32 - 4).max(clamp_y1);
    let ink_x2 = (ix1 + max_cx as i32 + 4).min(clamp_x2);
    let ink_y2 = (iy1 + max_cy as i32 + 4).min(clamp_y2);

    if (ink_x2 - ink_x1) < 10 || (ink_y2 - ink_y1) < 10 {
        return valid_hint.unwrap_or([bx1, by1, bx2, by2]);
    }
    [ink_x1, ink_y1, ink_x2, ink_y2]
}
