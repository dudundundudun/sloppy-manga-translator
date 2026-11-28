use crate::cache::compute_clean_state_hash_opts;
use crate::cv::{
    compute_rgb_median_and_std, connected_components, dilate_ellipse, distance_transform_l2,
    draw_circle, draw_polyline, erode_ellipse, fill_polygon, hermite_wall_bridge,
    inpaint_laplace_sor,
};
use crate::disk_cache::CleanDiskCache;
use crate::lama::AnimeLamaInpainter;
use crate::segmenter::NeuralStrokeSegmenter;
use image::RgbImage;
use miko_core::{clip_polygon_to_rect, TextBlock};
use miko_ort::OrtError;
use serde::Serialize;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;
use thiserror::Error;

pub const CLEAN_ALGO_VERSION: &str = "rust_lama_v4";

#[derive(Debug, Error)]
pub enum CleanerError {
    #[error("Ошибка ORT при сегментации текста: {0}")]
    Ort(#[from] OrtError),
    #[error("Ошибка дискового кэша WebP: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
pub struct CleanOptions {
    pub dilate_size: u8,
    pub pipeline_mode: String,
    pub use_neural_segmenter: bool,
    pub use_cache: bool,
    pub neural_conf: f32,
    pub skip_untranslated: bool,
}

impl Default for CleanOptions {
    fn default() -> Self {
        Self {
            dilate_size: 9,
            pipeline_mode: "balanced".to_string(),
            use_neural_segmenter: true,
            use_cache: true,
            neural_conf: 0.25,
            skip_untranslated: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CleanTimings {
    pub ep: String,
    pub seg_ms: u128,
    pub mask_ms: u128,
    pub inpaint_ms: u128,
    pub total_ms: u128,
    pub cache_hit: bool,
}

pub struct CleanPageResult {
    pub cleaned: RgbImage,
    pub mask_nonzero: usize,
    pub blocks_cleaned: usize,
    pub timings: CleanTimings,
}

pub struct RustCleanerEngine {
    segmenter: Option<NeuralStrokeSegmenter>,
    lama: Option<AnimeLamaInpainter>,
    cache: CleanDiskCache,
    pub ep: String,
}

#[inline]
fn rgb_lum(r: u8, g: u8, b: u8) -> u8 {
    (((77u32 * r as u32) + (150u32 * g as u32) + (29u32 * b as u32)) >> 8) as u8
}

#[inline]
fn rgb_chroma(r: u8, g: u8, b: u8) -> u8 {
    r.max(g).max(b) - r.min(g).min(b)
}

#[inline]
fn rgb_l1_dist(r: u8, g: u8, b: u8, bg: [u8; 3]) -> i32 {
    (r as i32 - bg[0] as i32).abs()
        + (g as i32 - bg[1] as i32).abs()
        + (b as i32 - bg[2] as i32).abs()
}

fn is_block_skipped(blk: &TextBlock, skip_untranslated: bool) -> bool {
    if blk.is_deleted
        || (blk.translated_text.trim().is_empty() && blk.original_text.trim().is_empty())
    {
        return true;
    }
    if skip_untranslated && blk.translated_text.trim().is_empty() {
        let has_strokes = blk
            .custom_mask_strokes
            .as_ref()
            .is_some_and(|s| !s.is_empty());
        if !has_strokes {
            return true;
        }
    }
    false
}

fn effective_clean_bbox(blk: &TextBlock) -> [i32; 4] {
    let mut cb = blk.clean_bbox.unwrap_or(blk.bbox);
    if !blk.clean_bbox_modified {
        if let Some(tb) = blk.text_bbox {
            if (tb[2] - tb[0]) >= 8 && (tb[3] - tb[1]) >= 8 {
                cb[0] = cb[0].min(tb[0]);
                cb[1] = cb[1].min(tb[1]);
                cb[2] = cb[2].max(tb[2]);
                cb[3] = cb[3].max(tb[3]);
            }
        }
    }
    cb
}

fn clipped_block_polygon(blk: &TextBlock, img_w: i32, img_h: i32) -> Option<Vec<[i32; 2]>> {
    if blk.is_free_text || blk.clean_bbox_modified {
        return None;
    }
    let poly = blk.bubble_polygon.as_ref()?;
    if poly.len() < 3 {
        return None;
    }
    let eb = effective_clean_bbox(blk);
    let clip_rect = [
        (eb[0] - 8).clamp(0, (img_w - 1).max(0)),
        (eb[1] - 8).clamp(0, (img_h - 1).max(0)),
        (eb[2] + 8).clamp(1, img_w.max(1)),
        (eb[3] + 8).clamp(1, img_h.max(1)),
    ];
    let clipped = clip_polygon_to_rect(poly, clip_rect);
    if clipped.len() >= 3 {
        Some(clipped)
    } else {
        None
    }
}

fn block_crop_rect(
    blk: &TextBlock,
    clipped_poly: Option<&[[i32; 2]]>,
    img_w: i32,
    img_h: i32,
    pad: i32,
) -> [i32; 4] {
    let mut b = effective_clean_bbox(blk);
    if let Some(tb) = blk.text_bbox {
        b[0] = b[0].min(tb[0]);
        b[1] = b[1].min(tb[1]);
        b[2] = b[2].max(tb[2]);
        b[3] = b[3].max(tb[3]);
    }
    if let Some(poly) = clipped_poly {
        for &[px, py] in poly {
            b[0] = b[0].min(px);
            b[1] = b[1].min(py);
            b[2] = b[2].max(px);
            b[3] = b[3].max(py);
        }
    }
    let x1 = (b[0] - pad).clamp(0, (img_w - 1).max(0));
    let y1 = (b[1] - pad).clamp(0, (img_h - 1).max(0));
    let x2 = (b[2] + pad).clamp(x1 + 1, img_w.max(1));
    let y2 = (b[3] + pad).clamp(y1 + 1, img_h.max(1));
    [x1, y1, x2, y2]
}

fn copy_crop_rgb(img: &RgbImage, rect: [i32; 4]) -> Vec<u8> {
    let iw = img.width() as usize;
    let raw = img.as_raw();
    let x0 = rect[0] as usize;
    let y0 = rect[1] as usize;
    let cw = (rect[2] - rect[0]) as usize;
    let ch = (rect[3] - rect[1]) as usize;
    let mut out = Vec::with_capacity(cw * ch * 3);
    for ry in 0..ch {
        let src_start = ((y0 + ry) * iw + x0) * 3;
        out.extend_from_slice(&raw[src_start..src_start + cw * 3]);
    }
    out
}

fn copy_crop_u8(plane: &[u8], full_w: usize, rect: [i32; 4]) -> Vec<u8> {
    let x0 = rect[0] as usize;
    let y0 = rect[1] as usize;
    let cw = (rect[2] - rect[0]) as usize;
    let ch = (rect[3] - rect[1]) as usize;
    let mut out = Vec::with_capacity(cw * ch);
    for ry in 0..ch {
        let src_start = (y0 + ry) * full_w + x0;
        out.extend_from_slice(&plane[src_start..src_start + cw]);
    }
    out
}

/// Вычисляет перцентиль `pct` (`0..=100`) яркости по пикселям, где `selector > 0`.
fn selector_lum_percentile(crop_rgb: &[u8], selector: &[u8], pct: usize) -> Option<u8> {
    let mut hist = [0u32; 256];
    let mut total = 0u32;
    for (i, &sel) in selector.iter().enumerate() {
        if sel == 0 {
            continue;
        }
        let lum = rgb_lum(crop_rgb[i * 3], crop_rgb[i * 3 + 1], crop_rgb[i * 3 + 2]);
        hist[lum as usize] += 1;
        total += 1;
    }
    if total == 0 {
        return None;
    }
    let target = ((total as u64 * pct.min(100) as u64) / 100).max(1) as u32;
    let mut acc = 0u32;
    for (val, &cnt) in hist.iter().enumerate() {
        acc += cnt;
        if acc >= target {
            return Some(val as u8);
        }
    }
    Some(255)
}

/// Строит попиксельно точную маску внутренней полости баббла (`true_cavity`),
/// отсекая внешние ложбинки сдвоенных бабблов (`011.webp`), чёрные контуры и волосы (`004.webp`, `005.webp`),
/// но доращивая усечённые хвосты (`002.webp`) и белые перешейки строго до реальной тёмной стенки.
fn build_effective_poly_mask(
    crop_rgb: &[u8],
    cw: usize,
    ch: usize,
    x0: i32,
    y0: i32,
    blk: &TextBlock,
    local_poly: &[[i32; 2]],
) -> Vec<u8> {
    let raw_poly_mask = fill_polygon(local_poly, cw, ch);
    let n = cw * ch;
    if n == 0 {
        return raw_poly_mask;
    }

    let mut eb = effective_clean_bbox(blk);
    if let Some(tb) = blk.text_bbox {
        eb[0] = eb[0].min(tb[0]);
        eb[1] = eb[1].min(tb[1]);
        eb[2] = eb[2].max(tb[2]);
        eb[3] = eb[3].max(tb[3]);
    }
    let bx1 = (eb[0] - x0 - 2).clamp(0, cw as i32) as usize;
    let by1 = (eb[1] - y0 - 2).clamp(0, ch as i32) as usize;
    let bx2 = (eb[2] - x0 + 2).clamp(bx1 as i32, cw as i32) as usize;
    let by2 = (eb[3] - y0 + 2).clamp(by1 as i32, ch as i32) as usize;
    if bx2 <= bx1 + 6 || by2 <= by1 + 6 {
        return raw_poly_mask;
    }

    let seed_eroded = erode_ellipse(&raw_poly_mask, cw, ch, 4);
    let seed_ref = if seed_eroded.iter().any(|&v| v > 0) {
        &seed_eroded
    } else {
        &raw_poly_mask
    };
    let Some(p80_lum) = selector_lum_percentile(crop_rgb, seed_ref, 80) else {
        return raw_poly_mask;
    };
    // Для тёмных/инверсных бабблов оставляем геометрический полигон без светлого flood-fill
    if p80_lum < 195 {
        return raw_poly_mask;
    }

    let mut bright_sel = vec![0u8; n];
    for (i, &s) in seed_ref.iter().enumerate() {
        if s > 0 {
            let lum = rgb_lum(crop_rgb[i * 3], crop_rgb[i * 3 + 1], crop_rgb[i * 3 + 2]);
            if lum >= p80_lum.saturating_sub(24) {
                bright_sel[i] = 255;
            }
        }
    }
    let (mut bg_rgb, _) =
        compute_rgb_median_and_std(crop_rgb, &bright_sel).unwrap_or(([255, 255, 255], 0.0));
    if bg_rgb[0] >= 240 && bg_rgb[1] >= 240 && bg_rgb[2] >= 240 {
        bg_rgb = [255, 255, 255];
    }
    let lum_floor = p80_lum.saturating_sub(42).max(192);

    // Разрешённая зона поиска полости: полигон (+2px) ∪ effective_clean_bbox
    let poly_dil2 = dilate_ellipse(&raw_poly_mask, cw, ch, 2);
    let mut allow_zone = vec![false; n];
    for ry in 0..ch {
        let row = ry * cw;
        let in_eb_y = ry >= by1 && ry < by2;
        for rx in 0..cw {
            let i = row + rx;
            if poly_dil2[i] > 0 || (in_eb_y && rx >= bx1 && rx < bx2) {
                allow_zone[i] = true;
            }
        }
    }

    let mut walkable = vec![false; n];
    for i in 0..n {
        if !allow_zone[i] {
            continue;
        }
        let r = crop_rgb[i * 3];
        let g = crop_rgb[i * 3 + 1];
        let b = crop_rgb[i * 3 + 2];
        let lum = rgb_lum(r, g, b);
        if lum >= lum_floor && rgb_l1_dist(r, g, b, bg_rgb) <= 56 {
            walkable[i] = true;
        }
    }

    // Стартуем BFS строго из центрального ядра текстового бокса / глубокой эрозии полигона,
    // чтобы никогда не засеять внешнюю ложбинку сдвоенного баббла (011.webp) снаружи чёрной стенки!
    let anchor = blk.text_bbox.unwrap_or(eb);
    let ax1 = (anchor[0] - x0).clamp(0, cw as i32) as usize;
    let ay1 = (anchor[1] - y0).clamp(0, ch as i32) as usize;
    let ax2 = (anchor[2] - x0).clamp(ax1 as i32, cw as i32) as usize;
    let ay2 = (anchor[3] - y0).clamp(ay1 as i32, ch as i32) as usize;
    let pad_ax = ((ax2 - ax1) / 6).min(14);
    let pad_ay = ((ay2 - ay1) / 6).min(14);
    let sx1 = (ax1 + pad_ax).min(ax2.saturating_sub(1));
    let sy1 = (ay1 + pad_ay).min(ay2.saturating_sub(1));
    let sx2 = ax2.saturating_sub(pad_ax).max(sx1 + 1);
    let sy2 = ay2.saturating_sub(pad_ay).max(sy1 + 1);

    let mut cav_flood = vec![0u8; n];
    let mut q: VecDeque<(usize, usize)> = VecDeque::new();
    for ry in sy1..sy2 {
        let row = ry * cw;
        for rx in sx1..sx2 {
            let i = row + rx;
            if walkable[i] && (raw_poly_mask[i] > 0 || blk.text_bbox.is_some()) {
                cav_flood[i] = 255;
                q.push_back((rx, ry));
            }
        }
    }
    if q.is_empty() {
        let deep = erode_ellipse(&raw_poly_mask, cw, ch, 5);
        let fallback_seed = if deep.iter().any(|&v| v > 0) {
            &deep
        } else {
            seed_ref
        };
        for ry in 0..ch {
            let row = ry * cw;
            for rx in 0..cw {
                let i = row + rx;
                if fallback_seed[i] > 0 && walkable[i] {
                    cav_flood[i] = 255;
                    q.push_back((rx, ry));
                }
            }
        }
    }
    if q.is_empty() {
        return raw_poly_mask;
    }

    while let Some((x, y)) = q.pop_front() {
        let nbrs = [
            (x.wrapping_sub(1), y, x > 0),
            (x + 1, y, x + 1 < cw),
            (x, y.wrapping_sub(1), y > 0),
            (x, y + 1, y + 1 < ch),
        ];
        for (nx, ny, ok) in nbrs {
            if !ok {
                continue;
            }
            let ni = ny * cw + nx;
            if cav_flood[ni] == 0 && walkable[ni] {
                cav_flood[ni] = 255;
                q.push_back((nx, ny));
            }
        }
    }

    // Если баббл открытый/облачный без рамки и `cav_flood` вытек за `raw_poly_mask` на широком периметре `eb`,
    // ограничиваем выход за пределы `poly_dil2`, но внутри полигона всё равно отсекаем любые тёмные внешние углы!
    let mut perim_leak = 0usize;
    for rx in bx1..bx2 {
        let i_top = by1 * cw + rx;
        let i_bot = (by2 - 1) * cw + rx;
        if cav_flood[i_top] > 0 && poly_dil2[i_top] == 0 {
            perim_leak += 1;
        }
        if cav_flood[i_bot] > 0 && poly_dil2[i_bot] == 0 {
            perim_leak += 1;
        }
    }
    for ry in by1..by2 {
        let i_l = ry * cw + bx1;
        let i_r = ry * cw + (bx2 - 1);
        if cav_flood[i_l] > 0 && poly_dil2[i_l] == 0 {
            perim_leak += 1;
        }
        if cav_flood[i_r] > 0 && poly_dil2[i_r] == 0 {
            perim_leak += 1;
        }
    }
    let perim_total = 2 * ((bx2 - bx1) + (by2 - by1)).max(1);
    if perim_leak * 100 > perim_total * 18 {
        for i in 0..n {
            if poly_dil2[i] == 0 {
                cav_flood[i] = 0;
            }
        }
    }

    // Мягкое замыкание штрихов текста (толщиной до 8px), коснувшихся или пересёкших стенку баббла:
    // `safe_from_notch` гарантирует, что тёмная галочка V-седловины сдвоенного баббла (`011.webp`)
    // на границе с внешним карманом `closed_cand == 0` никогда не будет поглощена полостью!
    let inner_guard = erode_ellipse(&raw_poly_mask, cw, ch, 3);
    let closed_cand = erode_ellipse(&dilate_ellipse(&cav_flood, cw, ch, 4), cw, ch, 4);
    let safe_from_notch = erode_ellipse(&closed_cand, cw, ch, 3);
    for i in 0..n {
        if inner_guard[i] > 0 && closed_cand[i] > 0 {
            let lum = rgb_lum(crop_rgb[i * 3], crop_rgb[i * 3 + 1], crop_rgb[i * 3 + 2]);
            if safe_from_notch[i] > 0 || lum >= 95 {
                cav_flood[i] = 255;
            }
        }
    }

    // Заполняем внутренние дырки внутри `cav_flood` (сами чёрные/цветные буквы внутри белой полости)
    let mut ext_vis = vec![false; n];
    let mut eq: VecDeque<(usize, usize)> = VecDeque::new();
    for x in 0..cw {
        for &y in &[0usize, ch.saturating_sub(1)] {
            let i = y * cw + x;
            if cav_flood[i] == 0 && !ext_vis[i] {
                ext_vis[i] = true;
                eq.push_back((x, y));
            }
        }
    }
    for y in 0..ch {
        for &x in &[0usize, cw.saturating_sub(1)] {
            let i = y * cw + x;
            if cav_flood[i] == 0 && !ext_vis[i] {
                ext_vis[i] = true;
                eq.push_back((x, y));
            }
        }
    }
    while let Some((x, y)) = eq.pop_front() {
        let nbrs = [
            (x.wrapping_sub(1), y, x > 0),
            (x + 1, y, x + 1 < cw),
            (x, y.wrapping_sub(1), y > 0),
            (x, y + 1, y + 1 < ch),
        ];
        for (nx, ny, ok) in nbrs {
            if !ok {
                continue;
            }
            let ni = ny * cw + nx;
            if !ext_vis[ni] && cav_flood[ni] == 0 {
                ext_vis[ni] = true;
                eq.push_back((nx, ny));
            }
        }
    }

    let mut cav_filled = vec![0u8; n];
    let mut filled_area = 0usize;
    for i in 0..n {
        if !ext_vis[i] {
            cav_filled[i] = 255;
            filled_area += 1;
        }
    }

    let raw_area = raw_poly_mask.iter().filter(|&&v| v > 0).count().max(1);
    if filled_area * 100 < raw_area * 22 {
        return raw_poly_mask;
    }

    // Расширяем чистую полость на 1px (строго не заходя на тёмный контур `lum < 135`),
    // чтобы последующая эрозия отступала ровно от внутренней грани чёрной стенки!
    let cav_pad1 = dilate_ellipse(&cav_filled, cw, ch, 1);
    let mut out = vec![0u8; n];
    for i in 0..n {
        if cav_filled[i] > 0 {
            out[i] = 255;
        } else if cav_pad1[i] > 0 {
            let lum = rgb_lum(crop_rgb[i * 3], crop_rgb[i * 3 + 1], crop_rgb[i * 3 + 2]);
            if lum >= 145 {
                out[i] = 255;
            }
        }
    }
    out
}

/// Вычисляет робастный цвет и признак однородности плоского баббла по стандарту PanelCleaner
/// (нормализация околобелого >= 240 -> 255 и отбрасывание краевых тёмных выбросов),
/// чтобы чистые белые/однотонные бабблы никогда не проваливались в размытие Лапласа (`inpaint_laplace_sor`).
fn evaluate_flat_bubble_bg(crop_rgb: &[u8], selector: &[u8]) -> ([u8; 3], bool) {
    let Some(p75_lum) = selector_lum_percentile(crop_rgb, selector, 75) else {
        return ([255, 255, 255], true);
    };

    // Если 75-й перцентиль полости околобелый (>= 236) — проверяем однородность светлого фона,
    // полностью игнорируя случайные тёмные пиксели у рамки или остатков штрихов (`lum < 165`)!
    if p75_lum >= 236 {
        let mut bright_total = 0usize;
        let mut white_inliers = 0usize;
        for (i, &sel) in selector.iter().enumerate() {
            if sel == 0 {
                continue;
            }
            let r = crop_rgb[i * 3];
            let g = crop_rgb[i * 3 + 1];
            let b = crop_rgb[i * 3 + 2];
            let lum = rgb_lum(r, g, b);
            if lum >= 165 {
                bright_total += 1;
                if lum >= 232 && rgb_chroma(r, g, b) <= 24 {
                    white_inliers += 1;
                }
            }
        }
        if bright_total == 0 || white_inliers * 100 >= bright_total * 62 {
            return ([255, 255, 255], true);
        }
    }

    let Some((mut bg_rgb, bg_std)) = compute_rgb_median_and_std(crop_rgb, selector) else {
        return ([255, 255, 255], true);
    };
    let mut total = 0usize;
    let mut inliers = 0usize;
    let mut sum_lum = 0.0f64;
    let mut sum_lum_sq = 0.0f64;
    for (i, &sel) in selector.iter().enumerate() {
        if sel == 0 {
            continue;
        }
        let r = crop_rgb[i * 3];
        let g = crop_rgb[i * 3 + 1];
        let b = crop_rgb[i * 3 + 2];
        let lum = rgb_lum(r, g, b);
        if p75_lum >= 190 && lum < 110 {
            continue;
        }
        total += 1;
        if rgb_l1_dist(r, g, b, bg_rgb) <= 40 {
            inliers += 1;
            let lf = lum as f64;
            sum_lum += lf;
            sum_lum_sq += lf * lf;
        }
    }
    let inlier_ratio = if total > 0 {
        (inliers as f32) / (total as f32)
    } else {
        1.0
    };
    let inlier_std = if inliers > 1 {
        let m = sum_lum / (inliers as f64);
        ((sum_lum_sq / (inliers as f64) - m * m).max(0.0)).sqrt() as f32
    } else {
        0.0
    };
    if bg_rgb[0] >= 240 && bg_rgb[1] >= 240 && bg_rgb[2] >= 240 {
        bg_rgb = [255, 255, 255];
    }
    let is_flat = bg_std <= 12.0 || (inlier_ratio >= 0.70 && inlier_std <= 8.0);
    (bg_rgb, is_flat)
}

fn build_tract_a_crop_mask(
    crop_rgb: &[u8],
    cw: usize,
    ch: usize,
    x0: i32,
    y0: i32,
    blk: &TextBlock,
    poly: &[[i32; 2]],
    neural_crop: Option<&[u8]>,
    default_dilate: u8,
) -> (Vec<u8>, Vec<u8>) {
    let local_poly: Vec<[i32; 2]> = poly.iter().map(|&[px, py]| [px - x0, py - y0]).collect();
    let poly_mask = build_effective_poly_mask(crop_rgb, cw, ch, x0, y0, blk, &local_poly);

    let mut min_px = cw as i32;
    let mut max_px = 0i32;
    let mut min_py = ch as i32;
    let mut max_py = 0i32;
    for ry in 0..ch {
        let row = ry * cw;
        for rx in 0..cw {
            if poly_mask[row + rx] > 0 {
                min_px = min_px.min(rx as i32);
                max_px = max_px.max(rx as i32 + 1);
                min_py = min_py.min(ry as i32);
                max_py = max_py.max(ry as i32 + 1);
            }
        }
    }
    if max_px <= min_px {
        min_px = 0;
        max_px = cw as i32;
        min_py = 0;
        max_py = ch as i32;
    }
    let min_dim = (max_px - min_px).min(max_py - min_py).max(1);
    let poly_w = (max_px - min_px).max(1) as usize;
    let poly_h = (max_py - min_py).max(1) as usize;
    let erode_r = if min_dim < 120 {
        3usize
    } else if min_dim < 260 {
        4usize
    } else {
        5usize
    };
    let mut shrunk_poly = erode_ellipse(&poly_mask, cw, ch, erode_r);
    if !shrunk_poly.iter().any(|&v| v > 0) {
        shrunk_poly = poly_mask.clone();
    }

    let poly_area = shrunk_poly.iter().filter(|&&v| v > 0).count().max(1);
    let (bg_rgb, _) =
        compute_rgb_median_and_std(crop_rgb, &shrunk_poly).unwrap_or(([255, 255, 255], 0.0));
    let bg_lum = rgb_lum(bg_rgb[0], bg_rgb[1], bg_rgb[2]);
    let is_dark_bubble = bg_lum < 110;

    // Если есть явный `text_bbox`, чистый CV-контраст без нейроподтверждения работает строго в окрестности текста (+18px),
    // чтобы не стирать линии рамок, хвостики бабблов и волосы на периферии полигона!
    let cv_roi_local: Option<[usize; 4]> = blk.text_bbox.and_then(|tb| {
        if (tb[2] - tb[0]) >= 6 && (tb[3] - tb[1]) >= 6 {
            let rx1 = (tb[0] - x0 - 18).clamp(0, cw as i32) as usize;
            let ry1 = (tb[1] - y0 - 18).clamp(0, ch as i32) as usize;
            let rx2 = (tb[2] - x0 + 18).clamp(rx1 as i32, cw as i32) as usize;
            let ry2 = (tb[3] - y0 + 18).clamp(ry1 as i32, ch as i32) as usize;
            Some([rx1, ry1, rx2, ry2])
        } else {
            None
        }
    });
    let tb_core_local: Option<[usize; 4]> = blk.text_bbox.and_then(|tb| {
        if (tb[2] - tb[0]) >= 6 && (tb[3] - tb[1]) >= 6 {
            let rx1 = (tb[0] - x0 + 2).clamp(0, cw as i32) as usize;
            let ry1 = (tb[1] - y0 + 2).clamp(0, ch as i32) as usize;
            let rx2 = (tb[2] - x0 - 2).clamp(rx1 as i32, cw as i32) as usize;
            let ry2 = (tb[3] - y0 - 2).clamp(ry1 as i32, ch as i32) as usize;
            Some([rx1, ry1, rx2, ry2])
        } else {
            None
        }
    });

    let n = cw * ch;
    let mut cand_ink = vec![0u8; n];
    for ry in 0..ch {
        let row = ry * cw;
        for rx in 0..cw {
            let i = row + rx;
            if shrunk_poly[i] == 0 {
                continue;
            }
            let in_cv_roi =
                cv_roi_local.is_none_or(|b| rx >= b[0] && rx < b[2] && ry >= b[1] && ry < b[3]);
            let nn_hit = neural_crop.is_some_and(|nm| nm[i] >= 100);
            if !in_cv_roi && !nn_hit {
                continue;
            }

            let r = crop_rgb[i * 3];
            let g = crop_rgb[i * 3 + 1];
            let b = crop_rgb[i * 3 + 2];
            let lum = rgb_lum(r, g, b);
            let chroma = rgb_chroma(r, g, b);
            let l1 = rgb_l1_dist(r, g, b, bg_rgb);

            let contrast_hit = if is_dark_bubble {
                lum > bg_lum.saturating_add(36)
            } else {
                lum.saturating_add(36) < bg_lum
            };
            let color_hit = chroma >= 42 && l1 >= 60;
            if nn_hit || (in_cv_roi && (contrast_hit || color_hit)) {
                cand_ink[i] = 255;
            }
        }
    }

    // Защита чиби-арта, волос и толстого контура баббла внутри shrunk_poly
    let cc = connected_components(&cand_ink, cw, ch);
    if !cc.stats.is_empty() {
        let dt = distance_transform_l2(&cand_ink, cw, ch);
        let rim_inner = erode_ellipse(&shrunk_poly, cw, ch, 2);
        let mut max_dt = vec![0.0f32; cc.stats.len()];
        let mut nn_hits = vec![0usize; cc.stats.len()];
        let mut rim_hits = vec![0usize; cc.stats.len()];
        for (i, &lbl) in cc.labels.iter().enumerate() {
            if lbl == 0 {
                continue;
            }
            let idx = (lbl - 1) as usize;
            if dt[i] > max_dt[idx] {
                max_dt[idx] = dt[i];
            }
            if neural_crop.is_some_and(|nm| nm[i] >= 90) {
                nn_hits[idx] += 1;
            }
            if rim_inner[i] == 0 {
                rim_hits[idx] += 1;
            }
        }
        let mut reject_lbl = vec![false; cc.stats.len() + 1];
        let mut any_rejected = false;
        for st in &cc.stats {
            let idx = (st.label - 1) as usize;
            let nn_ratio = (nn_hits[idx] as f32) / (st.area.max(1) as f32);
            let rim_ratio = (rim_hits[idx] as f32) / (st.area.max(1) as f32);
            let strictly_in_tb = tb_core_local.is_some_and(|tb| {
                st.left >= tb[0]
                    && (st.left + st.width) <= tb[2]
                    && st.top >= tb[1]
                    && (st.top + st.height) <= tb[3]
            });

            let is_wall_rim = rim_ratio >= 0.65
                && (st.width * 100 >= poly_w * 60 || st.height * 100 >= poly_h * 60)
                && (neural_crop.is_none() || nn_ratio < 0.30);
            let is_hair_or_panel_intrusion = !strictly_in_tb
                && rim_hits[idx] >= 3
                && st.area >= 95
                && max_dt[idx] >= 4.2
                && (neural_crop.is_none() || nn_ratio < 0.35);

            if is_wall_rim || is_hair_or_panel_intrusion {
                reject_lbl[st.label as usize] = true;
                any_rejected = true;
            } else if st.area >= 450 && max_dt[idx] >= 11.5 {
                let reject = if neural_crop.is_some() {
                    nn_ratio < 0.35
                } else {
                    max_dt[idx] >= 16.0 && (st.area * 100) > (poly_area * 40)
                };
                if reject {
                    reject_lbl[st.label as usize] = true;
                    any_rejected = true;
                }
            }
        }
        if any_rejected {
            for (i, &lbl) in cc.labels.iter().enumerate() {
                if lbl > 0 && reject_lbl[lbl as usize] {
                    cand_ink[i] = 0;
                }
            }
        }
    }

    // Нависающие штрихи текста за пределами контура баббла (только при явном text_bbox или нейро-маске)
    let mut outside_raw = vec![0u8; n];
    let mut has_outside = false;
    let overhang_box = if blk.text_bbox.is_some() || neural_crop.is_some() {
        Some(blk.text_bbox.unwrap_or_else(|| effective_clean_bbox(blk)))
    } else {
        None
    };

    if let Some(tb) = overhang_box {
        let tx1 = (tb[0] - x0 - 4).clamp(0, cw as i32) as usize;
        let ty1 = (tb[1] - y0 - 4).clamp(0, ch as i32) as usize;
        let tx2 = (tb[2] - x0 + 4).clamp(tx1 as i32, cw as i32) as usize;
        let ty2 = (tb[3] - y0 + 4).clamp(ty1 as i32, ch as i32) as usize;

        // Внешний фон вокруг баббла (всегда вычисляем, чтобы в нейро-режиме тоже отсекать ложные нависания)
        let ext_ring = dilate_ellipse(&poly_mask, cw, ch, 12);
        let mut ext_sel = vec![0u8; n];
        for i in 0..n {
            if ext_ring[i] > 0 && poly_mask[i] == 0 {
                ext_sel[i] = 255;
            }
        }
        let ext_bg_rgb = compute_rgb_median_and_std(crop_rgb, &ext_sel)
            .map(|(c, _)| c)
            .unwrap_or(bg_rgb);
        let ext_lum = rgb_lum(ext_bg_rgb[0], ext_bg_rgb[1], ext_bg_rgb[2]);

        for ry in ty1..ty2 {
            let row = ry * cw;
            for rx in tx1..tx2 {
                let i = row + rx;
                if poly_mask[i] > 0 {
                    continue;
                }
                let r = crop_rgb[i * 3];
                let g = crop_rgb[i * 3 + 1];
                let b = crop_rgb[i * 3 + 2];
                let lum = rgb_lum(r, g, b);
                let chroma = rgb_chroma(r, g, b);
                let l1_ext = rgb_l1_dist(r, g, b, ext_bg_rgb);
                let high_contrast =
                    (lum as i32 - ext_lum as i32).abs() > 42 || (chroma >= 45 && l1_ext >= 60);

                // Грубая 160×160 маска segmenter не должна слепо помечать ровный фон или рамку за стеной баббла
                let nn_over = neural_crop.is_some_and(|nm| nm[i] >= 150) && high_contrast;
                let cv_over = neural_crop.is_none() && blk.text_bbox.is_some() && high_contrast;
                if nn_over || cv_over {
                    outside_raw[i] = 255;
                    has_outside = true;
                }
            }
        }

        if has_outside {
            // Отсекаем собственную чёрную рамку баббла и край панели (`006.webp #2`, V-седловину `011.webp`):
            // настоящий нависающий штрих текста обязан выходить за оболочку стенки (`poly_wall_shell == 0`)
            // И соединяться с внутренними чернилами `cand_ink`!
            let poly_wall_shell = dilate_ellipse(&poly_mask, cw, ch, 5);
            let in_bridge_check = dilate_ellipse(&cand_ink, cw, ch, 6);
            let occ = connected_components(&outside_raw, cw, ch);
            has_outside = false;
            if !occ.stats.is_empty() {
                let mut beyond_cnt = vec![0usize; occ.stats.len()];
                let mut touches_in = vec![false; occ.stats.len()];
                for (i, &lbl) in occ.labels.iter().enumerate() {
                    if lbl == 0 {
                        continue;
                    }
                    let idx = (lbl - 1) as usize;
                    if poly_wall_shell[i] == 0 {
                        beyond_cnt[idx] += 1;
                    }
                    if in_bridge_check[i] > 0 {
                        touches_in[idx] = true;
                    }
                }
                for (i, &lbl) in occ.labels.iter().enumerate() {
                    if lbl == 0 {
                        continue;
                    }
                    let idx = (lbl - 1) as usize;
                    if beyond_cnt[idx] >= 3 && touches_in[idx] {
                        has_outside = true;
                    } else {
                        outside_raw[i] = 0;
                    }
                }
            }
        }

        if has_outside {
            let d = dilate_ellipse(&outside_raw, cw, ch, 2);
            let d_out_bridge = dilate_ellipse(&outside_raw, cw, ch, 4);
            let d_in_bridge = dilate_ellipse(&cand_ink, cw, ch, 5);
            let mut clamped = vec![0u8; n];
            for ry in ty1..ty2 {
                let row = ry * cw;
                for rx in tx1..tx2 {
                    let i = row + rx;
                    if d[i] > 0 && poly_mask[i] == 0 {
                        clamped[i] = 255;
                    } else if poly_mask[i] > 0 && d_out_bridge[i] > 0 && d_in_bridge[i] > 0 {
                        clamped[i] = 255;
                    }
                }
            }
            outside_raw = clamped;
        }
    }
    let outside_dil = outside_raw;

    // Честный учет ползунка `mask_dilate`: при `> 10` ослабляем зажим до `poly_mask`
    let eff_dil = blk.mask_dilate.unwrap_or(default_dilate).clamp(1, 28) as usize;
    let dil_r = (eff_dil / 2).max(1);
    let dil_in = dilate_ellipse(&cand_ink, cw, ch, dil_r);
    let clamp_region = if eff_dil > 10 {
        &poly_mask
    } else {
        &shrunk_poly
    };

    let mut out_mask = vec![0u8; n];
    for i in 0..n {
        if (dil_in[i] > 0 && clamp_region[i] > 0) || outside_dil[i] > 0 {
            out_mask[i] = 255;
        }
    }

    (out_mask, poly_mask)
}

/// Восстанавливает попиксельную замкнутую светлую полость для блоков без полигона (Tract B),
/// чтобы не закрашивать белым квадратом шипы баббла и фон вокруг (`013.webp #14`).
fn extract_tract_b_white_cavity(
    crop_rgb: &[u8],
    cw: usize,
    ch: usize,
    bx1: usize,
    by1: usize,
    bx2: usize,
    by2: usize,
) -> Option<Vec<u8>> {
    if cw < 12 || ch < 12 || bx2 <= bx1 + 8 || by2 <= by1 + 8 {
        return None;
    }
    let n = cw * ch;
    let bw = bx2 - bx1;
    let bh = by2 - by1;
    let cx = (bx1 + bx2) / 2;
    let cy = (by1 + by2) / 2;
    let rx = (bw * 18 / 100).max(4);
    let ry = (bh * 18 / 100).max(4);

    let mut vis = vec![0u8; n];
    let mut q = VecDeque::new();
    for y in cy.saturating_sub(ry)..=(cy + ry).min(ch - 1) {
        for x in cx.saturating_sub(rx)..=(cx + rx).min(cw - 1) {
            let i = y * cw + x;
            let lum = rgb_lum(crop_rgb[i * 3], crop_rgb[i * 3 + 1], crop_rgb[i * 3 + 2]);
            let chr = rgb_chroma(crop_rgb[i * 3], crop_rgb[i * 3 + 1], crop_rgb[i * 3 + 2]);
            if lum >= 200 && chr <= 26 && vis[i] == 0 {
                vis[i] = 255;
                q.push_back((x, y));
            }
        }
    }
    if q.is_empty() {
        return None;
    }

    while let Some((x, y)) = q.pop_front() {
        let nbrs = [
            (x.wrapping_sub(1), y, x > 0),
            (x + 1, y, x + 1 < cw),
            (x, y.wrapping_sub(1), y > 0),
            (x, y + 1, y + 1 < ch),
        ];
        for (nx, ny, ok) in nbrs {
            if !ok {
                continue;
            }
            let ni = ny * cw + nx;
            if vis[ni] == 0 {
                let lum = rgb_lum(crop_rgb[ni * 3], crop_rgb[ni * 3 + 1], crop_rgb[ni * 3 + 2]);
                if lum >= 195 {
                    vis[ni] = 255;
                    q.push_back((nx, ny));
                }
            }
        }
    }

    let mut leaked = 0usize;
    let perim_total = 2 * cw + 2 * ch;
    for x in 0..cw {
        if vis[x] > 0 {
            leaked += 1;
        }
        if vis[(ch - 1) * cw + x] > 0 {
            leaked += 1;
        }
    }
    for y in 0..ch {
        if vis[y * cw] > 0 {
            leaked += 1;
        }
        if vis[y * cw + (cw - 1)] > 0 {
            leaked += 1;
        }
    }
    if perim_total > 0 && leaked * 100 > perim_total * 16 {
        return None;
    }

    let mut inside_bbox_cnt = 0usize;
    for y in by1..by2 {
        let row = y * cw;
        for x in bx1..bx2 {
            if vis[row + x] > 0 {
                inside_bbox_cnt += 1;
            }
        }
    }
    let bbox_area = (bw * bh).max(1);
    if inside_bbox_cnt * 100 < bbox_area * 18 {
        return None;
    }

    // Заполняем внутренние «дыры» (тёмные буквы внутри белой полости) внешним обходом по краям кропа,
    // чтобы тёмные штрихи текста тоже попадали в `safe_cav`, сохраняя стенки и фон снаружи.
    let mut ext_vis = vec![false; n];
    let mut eq = VecDeque::new();
    for x in 0..cw {
        for y in [0usize, ch - 1] {
            let i = y * cw + x;
            if vis[i] == 0 && !ext_vis[i] {
                ext_vis[i] = true;
                eq.push_back((x, y));
            }
        }
    }
    for y in 0..ch {
        for x in [0usize, cw - 1] {
            let i = y * cw + x;
            if vis[i] == 0 && !ext_vis[i] {
                ext_vis[i] = true;
                eq.push_back((x, y));
            }
        }
    }
    while let Some((x, y)) = eq.pop_front() {
        let nbrs = [
            (x.wrapping_sub(1), y, x > 0),
            (x + 1, y, x + 1 < cw),
            (x, y.wrapping_sub(1), y > 0),
            (x, y + 1, y + 1 < ch),
        ];
        for (nx, ny, ok) in nbrs {
            if !ok {
                continue;
            }
            let ni = ny * cw + nx;
            if !ext_vis[ni] && vis[ni] == 0 {
                ext_vis[ni] = true;
                eq.push_back((nx, ny));
            }
        }
    }

    let mut cav_filled = vec![0u8; n];
    for i in 0..n {
        if !ext_vis[i] {
            cav_filled[i] = 255;
        }
    }

    let safe_cav = erode_ellipse(&cav_filled, cw, ch, 2);
    if !safe_cav.iter().any(|&v| v > 0) {
        return None;
    }
    Some(safe_cav)
}

fn build_tract_b_crop_mask(
    crop_rgb: &[u8],
    cw: usize,
    ch: usize,
    x0: i32,
    y0: i32,
    blk: &TextBlock,
    neural_crop: Option<&[u8]>,
    default_dilate: u8,
) -> (Vec<u8>, Option<Vec<u8>>) {
    let eb = effective_clean_bbox(blk);
    let bx1 = (eb[0] - x0).clamp(0, cw as i32) as usize;
    let by1 = (eb[1] - y0).clamp(0, ch as i32) as usize;
    let bx2 = (eb[2] - x0).clamp(bx1 as i32, cw as i32) as usize;
    let by2 = (eb[3] - y0).clamp(by1 as i32, ch as i32) as usize;
    let n = cw * ch;
    let mut cand = vec![0u8; n];
    if bx1 >= bx2 || by1 >= by2 {
        return (cand, None);
    }

    // Путь B1: закрытый светлый баббл без YOLO-полигона (`!blk.is_free_text`) —
    // восстанавливаем реальную полость и чистим СТРОГО внутри неё (без белых квадратов!)
    if !blk.is_free_text {
        if let Some(safe_cav) = extract_tract_b_white_cavity(crop_rgb, cw, ch, bx1, by1, bx2, by2) {
            for ry in by1..by2 {
                let row = ry * cw;
                for rx in bx1..bx2 {
                    let i = row + rx;
                    if safe_cav[i] == 0 {
                        continue;
                    }
                    let r = crop_rgb[i * 3];
                    let g = crop_rgb[i * 3 + 1];
                    let b = crop_rgb[i * 3 + 2];
                    let lum = rgb_lum(r, g, b);
                    let chroma = rgb_chroma(r, g, b);
                    let nn_hit = neural_crop.is_some_and(|nm| nm[i] >= 110);
                    if nn_hit || lum < 195 || chroma >= 45 {
                        cand[i] = 255;
                    }
                }
            }
            let dil_r =
                (blk.mask_dilate.unwrap_or(default_dilate).clamp(1, 28) as usize / 2).max(1);
            let dil = dilate_ellipse(&cand, cw, ch, dil_r);
            let mut out = vec![0u8; n];
            for i in 0..n {
                if dil[i] > 0 && safe_cav[i] > 0 {
                    out[i] = 255;
                }
            }
            return (out, Some(safe_cav));
        }
    }

    // Путь B2 (`blk.is_free_text` или безрамочный баббл без замкнутой полости):
    // раздельная фильтрация тёмных и светлых штрихов (чтобы белая обводка не склеивала колонки иероглифов в один сплошной блок!)
    // + захват белой обводки (`白フチ`) на тёмных/серых скринтонах для бесшовного LaMa-инпейнтинга!
    let iw = cw + 1;
    let mut integ = vec![0u64; iw * (ch + 1)];
    for y in 0..ch {
        let mut r_acc = 0u64;
        for x in 0..cw {
            let i = y * cw + x;
            r_acc += rgb_lum(crop_rgb[i * 3], crop_rgb[i * 3 + 1], crop_rgb[i * 3 + 2]) as u64;
            integ[(y + 1) * iw + (x + 1)] = integ[y * iw + (x + 1)] + r_acc;
        }
    }

    let mut dark_cand = vec![0u8; n];
    let mut bright_cand = vec![0u8; n];
    let win_r = 8usize;
    for ry in by1..by2 {
        let row = ry * cw;
        let wy0 = ry.saturating_sub(win_r);
        let wy1 = (ry + win_r + 1).min(ch);
        for rx in bx1..bx2 {
            let wx0 = rx.saturating_sub(win_r);
            let wx1 = (rx + win_r + 1).min(cw);
            let area_w = ((wx1 - wx0) * (wy1 - wy0)).max(1) as u64;
            let sum_w = integ[wy1 * iw + wx1] + integ[wy0 * iw + wx0]
                - integ[wy0 * iw + wx1]
                - integ[wy1 * iw + wx0];
            let loc_mean = (sum_w / area_w) as i32;

            let i = row + rx;
            let r = crop_rgb[i * 3];
            let g = crop_rgb[i * 3 + 1];
            let b = crop_rgb[i * 3 + 2];
            let lum = rgb_lum(r, g, b) as i32;
            let chroma = rgb_chroma(r, g, b);

            let dark_stroke = lum < 105 && loc_mean - lum >= 26;
            let bright_stroke = lum > 220 && lum - loc_mean >= 42;
            let nn_hit = neural_crop.is_some_and(|nm| nm[i] >= 125)
                && ((loc_mean - lum).abs() >= 24 || lum < 95 || lum > 225 || chroma >= 45);
            if dark_stroke || (lum < 115 && nn_hit) {
                dark_cand[i] = 255;
            } else if bright_stroke || chroma >= 48 || (lum >= 115 && nn_hit) {
                bright_cand[i] = 255;
            }
        }
    }

    let bw = (bx2 - bx1).max(1);
    let bh = (by2 - by1).max(1);
    let box_area = bw * bh;
    let filter_polarity = |plane: &mut [u8], max_dt_limit: f32, max_area_pct: usize| {
        let cc = connected_components(plane, cw, ch);
        if cc.stats.is_empty() {
            return;
        }
        let dt = distance_transform_l2(plane, cw, ch);
        let mut max_dt = vec![0.0f32; cc.stats.len()];
        for (i, &lbl) in cc.labels.iter().enumerate() {
            if lbl > 0 {
                let idx = (lbl - 1) as usize;
                if dt[i] > max_dt[idx] {
                    max_dt[idx] = dt[i];
                }
            }
        }
        let mut reject = vec![false; cc.stats.len() + 1];
        for st in &cc.stats {
            let idx = (st.label - 1) as usize;
            let touches_edge = st.left <= bx1 + 1
                || (st.left + st.width) + 1 >= bx2
                || st.top <= by1 + 1
                || (st.top + st.height) + 1 >= by2;
            let too_tiny_dot = st.area < 6;
            let too_thick_blob =
                max_dt[idx] > max_dt_limit || st.area * 100 > box_area * max_area_pct;
            let spans_frame = touches_edge
                && max_dt[idx] <= 3.5
                && ((st.width <= 4 && st.height * 100 > bh * 90)
                    || (st.height <= 4 && st.width * 100 > bw * 90));
            if too_tiny_dot || too_thick_blob || spans_frame {
                reject[st.label as usize] = true;
            }
        }
        for (i, &lbl) in cc.labels.iter().enumerate() {
            if lbl > 0 && reject[lbl as usize] {
                plane[i] = 0;
            }
        }
    };

    filter_polarity(&mut dark_cand, 9.0, 42);
    filter_polarity(&mut bright_cand, 9.5, 42);

    let has_dark_strokes = dark_cand.iter().filter(|&&v| v > 0).count() >= 16;
    for i in 0..n {
        if dark_cand[i] > 0 || (!has_dark_strokes && bright_cand[i] > 0) {
            cand[i] = 255;
        }
    }

    // Захват белой обводки (`白フチ`) вокруг тёмного текста на серых скринтонах (`bg_lum < 220`)
    let mut ring_sel = vec![0u8; n];
    let rx0 = bx1.saturating_sub(10);
    let ry0 = by1.saturating_sub(10);
    let rx1 = (bx2 + 10).min(cw);
    let ry1 = (by2 + 10).min(ch);
    for ry in ry0..ry1 {
        let row = ry * cw;
        for rx in rx0..rx1 {
            if rx < bx1 || rx >= bx2 || ry < by1 || ry >= by2 {
                ring_sel[row + rx] = 255;
            }
        }
    }
    let bg_lum = compute_rgb_median_and_std(crop_rgb, &ring_sel)
        .map(|(c, _)| rgb_lum(c[0], c[1], c[2]))
        .unwrap_or(245);
    if bg_lum < 220 && has_dark_strokes {
        let halo_zone = dilate_ellipse(&dark_cand, cw, ch, 5);
        let hx0 = bx1.saturating_sub(4);
        let hy0 = by1.saturating_sub(4);
        let hx1 = (bx2 + 4).min(cw);
        let hy1 = (by2 + 4).min(ch);
        for ry in hy0..hy1 {
            let row = ry * cw;
            for rx in hx0..hx1 {
                let i = row + rx;
                if halo_zone[i] > 0 && cand[i] == 0 {
                    let lum = rgb_lum(crop_rgb[i * 3], crop_rgb[i * 3 + 1], crop_rgb[i * 3 + 2]);
                    if lum >= 195 && lum >= bg_lum.saturating_add(35) {
                        cand[i] = 255;
                    }
                }
            }
        }
    }

    let eff_r = if (blk.is_free_text || neural_crop.is_none()) && blk.mask_dilate.is_none() {
        2usize
    } else {
        (blk.mask_dilate.unwrap_or(default_dilate).clamp(1, 28) as usize / 2).max(1)
    };
    let dil = dilate_ellipse(&cand, cw, ch, eff_r);

    let cx1 = bx1.saturating_sub(5);
    let cy1 = by1.saturating_sub(5);
    let cx2 = (bx2 + 5).min(cw);
    let cy2 = (by2 + 5).min(ch);
    let mut out = vec![0u8; n];
    for ry in cy1..cy2 {
        let row = ry * cw;
        for rx in cx1..cx2 {
            let i = row + rx;
            if dil[i] > 0 {
                out[i] = 255;
            }
        }
    }
    (out, None)
}

fn fill_masked_with_rgb(rgb: &mut [u8], mask: &[u8], color: [u8; 3]) {
    for (i, &m) in mask.iter().enumerate() {
        if m > 0 {
            rgb[i * 3] = color[0];
            rgb[i * 3 + 1] = color[1];
            rgb[i * 3 + 2] = color[2];
        }
    }
}

/// Замыкает разрывы контура баббла (если нависший текст пересёк стенку) кубическим сплайном Эрмита.
fn repair_broken_bubble_walls(
    crop_rgb: &mut [u8],
    cw: usize,
    ch: usize,
    crop_mask: &[u8],
    poly_mask: &[u8],
    local_poly: &[[i32; 2]],
) {
    if local_poly.len() < 3 || cw == 0 || ch == 0 {
        return;
    }
    let eroded = erode_ellipse(poly_mask, cw, ch, 3);
    let mut wall_ring = vec![0u8; cw * ch];
    let mut wall_broken = false;
    for i in 0..cw * ch {
        if poly_mask[i] > 0 && eroded[i] == 0 {
            wall_ring[i] = 255;
            if crop_mask[i] > 0 {
                wall_broken = true;
            }
        }
    }
    if !wall_broken {
        return;
    }

    // Берём цвет стенки из неповреждённых тёмных пикселей контура
    let mut dark_wall_sel = vec![0u8; cw * ch];
    for i in 0..cw * ch {
        if wall_ring[i] > 0 && crop_mask[i] == 0 {
            let lum = rgb_lum(crop_rgb[i * 3], crop_rgb[i * 3 + 1], crop_rgb[i * 3 + 2]);
            if lum < 115 {
                dark_wall_sel[i] = 255;
            }
        }
    }
    let wall_rgb = compute_rgb_median_and_std(crop_rgb, &dark_wall_sel)
        .map(|(c, _)| c)
        .unwrap_or([24, 24, 24]);

    // Уплотняем полигон до ~1.5px шага вдоль сторон
    let mut contour: Vec<[f32; 2]> = Vec::new();
    let m = local_poly.len();
    for i in 0..m {
        let a = local_poly[i];
        let b = local_poly[(i + 1) % m];
        let dx = (b[0] - a[0]) as f32;
        let dy = (b[1] - a[1]) as f32;
        let dist = (dx * dx + dy * dy).sqrt();
        let steps = (dist / 1.5).ceil().max(1.0) as usize;
        for s in 0..steps {
            let t = (s as f32) / (steps as f32);
            contour.push([a[0] as f32 + t * dx, a[1] as f32 + t * dy]);
        }
    }
    if contour.len() < 8 {
        return;
    }

    let is_pt_masked = |p: [f32; 2]| -> bool {
        let cx = p[0].round() as i32;
        let cy = p[1].round() as i32;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let x = cx + dx;
                let y = cy + dy;
                if x >= 0 && y >= 0 && (x as usize) < cw && (y as usize) < ch {
                    if crop_mask[(y as usize) * cw + (x as usize)] > 0 {
                        return true;
                    }
                }
            }
        }
        false
    };

    let masked_flags: Vec<bool> = contour.iter().map(|&p| is_pt_masked(p)).collect();
    let Some(first_unmasked) = masked_flags.iter().position(|&f| !f) else {
        return;
    };

    let clen = contour.len();
    let norm_tan = |p_from: [f32; 2], p_to: [f32; 2]| -> [f32; 2] {
        let dx = p_to[0] - p_from[0];
        let dy = p_to[1] - p_from[1];
        let d = (dx * dx + dy * dy).sqrt();
        if d > 1e-4 {
            [dx / d, dy / d]
        } else {
            [1.0, 0.0]
        }
    };

    let mut bridge_mask = vec![0u8; cw * ch];
    let mut idx = 0usize;
    while idx < clen {
        let cur = (first_unmasked + idx) % clen;
        if !masked_flags[cur] {
            idx += 1;
            continue;
        }
        let gap_start_offset = idx;
        while idx < clen && masked_flags[(first_unmasked + idx) % clen] {
            idx += 1;
        }
        let gap_len = idx - gap_start_offset;
        if gap_len < 2 {
            continue;
        }
        let a_idx = (first_unmasked + gap_start_offset + clen - 1) % clen;
        let b_idx = (first_unmasked + idx) % clen;
        let a_prev = (a_idx + clen - 6.min(clen - 1)) % clen;
        let b_next = (b_idx + 6.min(clen - 1)) % clen;

        let p_a = contour[a_idx];
        let p_b = contour[b_idx];
        // Строим мост Эрмита ТОЛЬКО если оба конца разрыва реально лежат на тёмной стенке баббла,
        // а не на искусственном срезе полигона в белом перешейке сдвоенного баббла (005.webp)!
        let has_dark_anchor = |pt: [f32; 2]| -> bool {
            let cx = pt[0].round() as i32;
            let cy = pt[1].round() as i32;
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let x = cx + dx;
                    let y = cy + dy;
                    if x >= 0 && y >= 0 && (x as usize) < cw && (y as usize) < ch {
                        let k = (y as usize) * cw + (x as usize);
                        if crop_mask[k] == 0
                            && rgb_lum(crop_rgb[k * 3], crop_rgb[k * 3 + 1], crop_rgb[k * 3 + 2])
                                < 135
                        {
                            return true;
                        }
                    }
                }
            }
            false
        };
        if !has_dark_anchor(p_a) || !has_dark_anchor(p_b) {
            continue;
        }
        let t_a = norm_tan(contour[a_prev], p_a);
        let t_b = norm_tan(p_b, contour[b_next]);
        let bridge_pts = hermite_wall_bridge(p_a, t_a, p_b, t_b, (gap_len * 2).max(10));
        draw_polyline(&mut bridge_mask, cw, ch, &bridge_pts, 1, 255);
    }

    for (i, &bm) in bridge_mask.iter().enumerate() {
        if bm > 0 && crop_mask[i] > 0 {
            crop_rgb[i * 3] = wall_rgb[0];
            crop_rgb[i * 3 + 1] = wall_rgb[1];
            crop_rgb[i * 3 + 2] = wall_rgb[2];
        }
    }
}

impl RustCleanerEngine {
    pub fn load(base_dir: &Path, prefer_gpu: bool) -> Self {
        let cache_dir = base_dir.join(".miko").join("clean_cache");
        let cache = CleanDiskCache::new(cache_dir, 48, 240);
        let onnx_dir = base_dir.join("models").join("onnx");
        let seg_path = onnx_dir.join("comic_text_segmenter.onnx");
        let lama_path = onnx_dir.join("anime_lama.onnx");

        let segmenter = if seg_path.is_file() {
            match NeuralStrokeSegmenter::load(&seg_path, prefer_gpu) {
                Ok(seg) => {
                    tracing::info!(
                        "RustCleanerEngine загрузил comic_text_segmenter.onnx на {}",
                        seg.ep
                    );
                    Some(seg)
                }
                Err(err) => {
                    tracing::warn!(
                        "Не удалось загрузить comic_text_segmenter.onnx ({}); работаем в CV-режиме",
                        err
                    );
                    None
                }
            }
        } else {
            None
        };

        let lama = if lama_path.is_file() {
            match AnimeLamaInpainter::load(&lama_path, prefer_gpu) {
                Ok(inp) => {
                    tracing::info!("RustCleanerEngine загрузил anime_lama.onnx на {}", inp.ep);
                    Some(inp)
                }
                Err(err) => {
                    tracing::warn!(
                        "Не удалось загрузить anime_lama.onnx ({}); фоллбэк на Laplace SOR",
                        err
                    );
                    None
                }
            }
        } else {
            None
        };

        let base_ep = segmenter
            .as_ref()
            .map(|s| s.ep.to_string())
            .or_else(|| lama.as_ref().map(|l| l.ep.to_string()))
            .unwrap_or_else(|| "CPU-CV".to_string());
        let ep = if lama.is_some() {
            format!("{base_ep}+LaMa")
        } else {
            base_ep
        };

        Self {
            segmenter,
            lama,
            cache,
            ep,
        }
    }

    pub fn new_cv_only(cache_dir: PathBuf) -> Self {
        Self {
            segmenter: None,
            lama: None,
            cache: CleanDiskCache::new(cache_dir, 48, 240),
            ep: "CPU-CV".to_string(),
        }
    }

    pub fn cache_mut(&mut self) -> &mut CleanDiskCache {
        &mut self.cache
    }

    /// Строит бинарную маску текста (`0`/`255`) по всем активным блокам страницы.
    pub fn build_text_mask(
        &self,
        img: &RgbImage,
        blocks: &[TextBlock],
        neural_mask: Option<&[u8]>,
        opts: &CleanOptions,
    ) -> Vec<u8> {
        let (w, h) = img.dimensions();
        let wu = w as usize;
        let hu = h as usize;
        let mut full_mask = vec![0u8; wu * hu];
        if wu == 0 || hu == 0 {
            return full_mask;
        }
        let nn_slice = neural_mask.filter(|nm| nm.len() >= wu * hu);

        for blk in blocks {
            if is_block_skipped(blk, opts.skip_untranslated) {
                continue;
            }
            let cpoly = clipped_block_polygon(blk, w as i32, h as i32);
            let rect = block_crop_rect(blk, cpoly.as_deref(), w as i32, h as i32, 24);
            let x0 = rect[0] as usize;
            let y0 = rect[1] as usize;
            let cw = (rect[2] - rect[0]) as usize;
            let ch = (rect[3] - rect[1]) as usize;
            let crop_rgb = copy_crop_rgb(img, rect);
            let nn_crop = nn_slice.map(|nm| copy_crop_u8(nm, wu, rect));

            let crop_mask = if let Some(ref poly) = cpoly {
                let (m, _) = build_tract_a_crop_mask(
                    &crop_rgb,
                    cw,
                    ch,
                    rect[0],
                    rect[1],
                    blk,
                    poly,
                    nn_crop.as_deref(),
                    opts.dilate_size,
                );
                m
            } else {
                let (m, _) = build_tract_b_crop_mask(
                    &crop_rgb,
                    cw,
                    ch,
                    rect[0],
                    rect[1],
                    blk,
                    nn_crop.as_deref(),
                    opts.dilate_size,
                );
                m
            };

            for ry in 0..ch {
                let dst_row = (y0 + ry) * wu + x0;
                let src_row = ry * cw;
                for rx in 0..cw {
                    if crop_mask[src_row + rx] > 0 {
                        full_mask[dst_row + rx] = 255;
                    }
                }
            }
        }

        // Ручная кисть пользователя (add / erase) имеет абсолютный приоритет поверх авто-маски
        for blk in blocks {
            if is_block_skipped(blk, opts.skip_untranslated) {
                continue;
            }
            if let Some(ref strokes) = blk.custom_mask_strokes {
                for s in strokes {
                    let val = if s.mode.eq_ignore_ascii_case("erase") {
                        0
                    } else {
                        255
                    };
                    draw_circle(&mut full_mask, wu, hu, s.x, s.y, s.r.clamp(1, 80), val);
                }
            }
        }

        full_mask
    }

    pub fn clean_image(
        &mut self,
        img: &RgbImage,
        blocks: &[TextBlock],
        opts: &CleanOptions,
        src_file: Option<&str>,
        src_mtime: i64,
    ) -> Result<CleanPageResult, CleanerError> {
        let t_total = Instant::now();
        let active_blocks = blocks
            .iter()
            .filter(|b| !is_block_skipped(b, opts.skip_untranslated))
            .count();

        let cache_key = if opts.use_cache {
            src_file.map(|sf| {
                compute_clean_state_hash_opts(
                    blocks,
                    sf,
                    src_mtime,
                    opts.dilate_size,
                    CLEAN_ALGO_VERSION,
                    &opts.pipeline_mode,
                    opts.skip_untranslated,
                )
            })
        } else {
            None
        };

        if let Some(ref key) = cache_key {
            if let Some(cached_img) = self.cache.get(key) {
                return Ok(CleanPageResult {
                    cleaned: cached_img,
                    mask_nonzero: 0,
                    blocks_cleaned: active_blocks,
                    timings: CleanTimings {
                        ep: self.ep.clone(),
                        seg_ms: 0,
                        mask_ms: 0,
                        inpaint_ms: 0,
                        total_ms: t_total.elapsed().as_millis(),
                        cache_hit: true,
                    },
                });
            }
        }

        let t_seg = Instant::now();
        let neural_mask = if opts.use_neural_segmenter {
            if let Some(ref mut seg) = self.segmenter {
                Some(seg.segment_page(img, opts.neural_conf)?)
            } else {
                None
            }
        } else {
            None
        };
        let seg_ms = t_seg.elapsed().as_millis();

        let mut res = self.clean_image_with_neural_mask(img, blocks, neural_mask.as_deref(), opts);
        res.timings.seg_ms = seg_ms;
        res.timings.total_ms = t_total.elapsed().as_millis();

        if let Some(ref key) = cache_key {
            self.cache.put(key, &res.cleaned)?;
        }

        Ok(res)
    }

    /// Детерминированная очистка страницы с переданной (опционально синтетической в тестах) нейро-маской.
    pub fn clean_image_with_neural_mask(
        &mut self,
        img: &RgbImage,
        blocks: &[TextBlock],
        neural_mask: Option<&[u8]>,
        opts: &CleanOptions,
    ) -> CleanPageResult {
        let t_total = Instant::now();
        let (w, h) = img.dimensions();
        let wu = w as usize;

        let t_mask = Instant::now();
        let full_mask = self.build_text_mask(img, blocks, neural_mask, opts);
        let mask_ms = t_mask.elapsed().as_millis();
        let mask_nonzero = full_mask.iter().filter(|&&v| v > 0).count();

        let t_inpaint = Instant::now();
        let mut cleaned = img.clone();
        let orig_raw = img.as_raw();
        let nn_slice = neural_mask.filter(|nm| nm.len() >= wu * (h as usize));
        let mut blocks_cleaned = 0usize;

        for blk in blocks {
            if is_block_skipped(blk, opts.skip_untranslated) {
                continue;
            }
            blocks_cleaned += 1;

            let cpoly = clipped_block_polygon(blk, w as i32, h as i32);
            let rect = block_crop_rect(blk, cpoly.as_deref(), w as i32, h as i32, 24);
            let x0 = rect[0] as usize;
            let y0 = rect[1] as usize;
            let cw = (rect[2] - rect[0]) as usize;
            let ch = (rect[3] - rect[1]) as usize;
            let n = cw * ch;

            // Строим СОБСТВЕННУЮ маску текущего блока (чтобы `pad=24` никогда не захватывал чужую `full_mask`
            // соседнего сдвоенного баббла как мнимую `outside_mask` — именно это давало серый след `224` на `005.webp`!),
            // с наложением ручных кистей `add` / `erase` текущего блока и глобального `full_mask` (для учёта ластика).
            let orig_crop_rgb = copy_crop_rgb(img, rect);
            let nn_crop = nn_slice.map(|nm| copy_crop_u8(nm, wu, rect));
            let mut tract_b_safe_cav: Option<Vec<u8>> = None;
            let mut crop_mask = if let Some(ref poly) = cpoly {
                let (m, _) = build_tract_a_crop_mask(
                    &orig_crop_rgb,
                    cw,
                    ch,
                    rect[0],
                    rect[1],
                    blk,
                    poly,
                    nn_crop.as_deref(),
                    opts.dilate_size,
                );
                m
            } else {
                let (m, cav) = build_tract_b_crop_mask(
                    &orig_crop_rgb,
                    cw,
                    ch,
                    rect[0],
                    rect[1],
                    blk,
                    nn_crop.as_deref(),
                    opts.dilate_size,
                );
                tract_b_safe_cav = cav;
                m
            };
            if let Some(ref strokes) = blk.custom_mask_strokes {
                for s in strokes {
                    let val = if s.mode.eq_ignore_ascii_case("erase") {
                        0
                    } else {
                        255
                    };
                    draw_circle(
                        &mut crop_mask,
                        cw,
                        ch,
                        s.x - rect[0],
                        s.y - rect[1],
                        s.r.clamp(1, 80),
                        val,
                    );
                }
            }
            let global_crop_mask = copy_crop_u8(&full_mask, wu, rect);
            for i in 0..n {
                if global_crop_mask[i] == 0 {
                    crop_mask[i] = 0;
                }
            }
            if !crop_mask.iter().any(|&v| v > 0) {
                continue;
            }

            let mut crop_rgb = copy_crop_rgb(&cleaned, rect);
            if let Some(ref poly) = cpoly {
                let local_poly: Vec<[i32; 2]> = poly
                    .iter()
                    .map(|&[px, py]| [px - rect[0], py - rect[1]])
                    .collect();
                let raw_poly_mask = fill_polygon(&local_poly, cw, ch);
                let mut poly_mask = build_effective_poly_mask(
                    &crop_rgb,
                    cw,
                    ch,
                    rect[0],
                    rect[1],
                    blk,
                    &local_poly,
                );

                let mut inside_mask = vec![0u8; n];
                let mut outside_mask = vec![0u8; n];
                let mut has_inside = false;
                let mut has_outside = false;
                for i in 0..n {
                    if crop_mask[i] == 0 {
                        continue;
                    }
                    if poly_mask[i] > 0 {
                        inside_mask[i] = 255;
                        has_inside = true;
                    } else {
                        outside_mask[i] = 255;
                        has_outside = true;
                    }
                }

                // Если компонента `outside_mask` примыкает к светлой полости или сама окружена светлым перешейком
                // сдвоенного баббла (как `005.webp #3 "Huh?"` рядом с тёмными волосами), переводим её в `inside_mask`,
                // чтобы серый внешний фон с волос/рубашки НИКОГДА не затекал Лапласом внутрь белого баббла!
                if has_outside {
                    let probe_inner = erode_ellipse(&poly_mask, cw, ch, 2);
                    let probe_ref = if probe_inner.iter().any(|&v| v > 0) {
                        &probe_inner
                    } else {
                        &poly_mask
                    };
                    let probe_p75 =
                        selector_lum_percentile(&crop_rgb, probe_ref, 75).unwrap_or(255);
                    if probe_p75 >= 225 {
                        let out_cc = connected_components(&outside_mask, cw, ch);
                        if !out_cc.stats.is_empty() {
                            let mut promote_lbl = vec![false; out_cc.stats.len() + 1];
                            for st in &out_cc.stats {
                                let x_a = st.left.saturating_sub(4);
                                let y_a = st.top.saturating_sub(4);
                                let x_b = (st.left + st.width + 4).min(cw);
                                let y_b = (st.top + st.height + 4).min(ch);
                                let mut ext_non_wall_cnt = 0usize;
                                let mut ext_white_cnt = 0usize;
                                for ry in y_a..y_b {
                                    let row = ry * cw;
                                    for rx in x_a..x_b {
                                        let k = row + rx;
                                        // Оцениваем СТРОГО внешний контекст за пределами `poly_mask` (`poly_mask[k] == 0`),
                                        // чтобы не перепутать нависающий наружу на тёмный фон штрих с белым перешейком баббла!
                                        if poly_mask[k] > 0 || crop_mask[k] != 0 {
                                            continue;
                                        }
                                        let lum = rgb_lum(
                                            crop_rgb[k * 3],
                                            crop_rgb[k * 3 + 1],
                                            crop_rgb[k * 3 + 2],
                                        );
                                        if lum >= 75 {
                                            ext_non_wall_cnt += 1;
                                            if lum >= 230 {
                                                ext_white_cnt += 1;
                                            }
                                        }
                                    }
                                }
                                if ext_non_wall_cnt >= 6
                                    && ext_white_cnt * 100 >= ext_non_wall_cnt * 60
                                {
                                    promote_lbl[st.label as usize] = true;
                                }
                            }
                            has_outside = false;
                            for (i, &lbl) in out_cc.labels.iter().enumerate() {
                                if lbl > 0 {
                                    if promote_lbl[lbl as usize] {
                                        outside_mask[i] = 0;
                                        inside_mask[i] = 255;
                                        poly_mask[i] = 255;
                                        has_inside = true;
                                    } else {
                                        has_outside = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // 1. Очистка текста ВНУТРИ баббла: чистая плоская заливка для белых/однотонных бабблов (PanelCleaner),
                //    а для реальных градиентов — экранированный Лаплас SOR с Anti-Stain Guard!
                if has_inside {
                    let interior_only = erode_ellipse(&poly_mask, cw, ch, 2);
                    let mut clean_interior_sel = vec![0u8; n];
                    let mut interior_cnt = 0usize;
                    for i in 0..n {
                        if interior_only[i] > 0 && crop_mask[i] == 0 {
                            clean_interior_sel[i] = 255;
                            interior_cnt += 1;
                        }
                    }
                    if interior_cnt == 0 {
                        for i in 0..n {
                            if poly_mask[i] > 0 && crop_mask[i] == 0 {
                                clean_interior_sel[i] = 255;
                            }
                        }
                    }
                    let (bg_rgb, is_flat_bubble) =
                        evaluate_flat_bubble_bg(&crop_rgb, &clean_interior_sel);
                    let bg_lum = rgb_lum(bg_rgb[0], bg_rgb[1], bg_rgb[2]);

                    if is_flat_bubble {
                        fill_masked_with_rgb(&mut crop_rgb, &inside_mask, bg_rgb);
                    } else {
                        let mut work_in = crop_rgb.clone();
                        // Экранируем И внешность, И любые тёмные пиксели контура/волос/чиби внутри `poly_mask`,
                        // чтобы чёрный цвет (0..30) никогда не диффундировал серым туманом в светлый баббл!
                        for i in 0..n {
                            let r = work_in[i * 3];
                            let g = work_in[i * 3 + 1];
                            let b = work_in[i * 3 + 2];
                            let lum = rgb_lum(r, g, b);
                            let dark_stain_on_light_bg =
                                bg_lum >= 175 && lum < bg_lum.saturating_sub(34);
                            if poly_mask[i] == 0 || (inside_mask[i] == 0 && dark_stain_on_light_bg)
                            {
                                work_in[i * 3] = bg_rgb[0];
                                work_in[i * 3 + 1] = bg_rgb[1];
                                work_in[i * 3 + 2] = bg_rgb[2];
                            }
                        }
                        let used_lama = if let Some(ref mut lama) = self.lama {
                            lama.inpaint_crop_rgb(&mut work_in, &inside_mask, cw, ch)
                                .is_ok()
                        } else {
                            false
                        };
                        if !used_lama {
                            inpaint_laplace_sor(&mut work_in, &inside_mask, cw, ch, 180, 1.75);
                        }
                        let min_floor = if bg_lum >= 195 {
                            bg_lum.saturating_sub(24)
                        } else {
                            0
                        };
                        for i in 0..n {
                            if inside_mask[i] > 0 {
                                let mut r = work_in[i * 3];
                                let mut g = work_in[i * 3 + 1];
                                let mut b = work_in[i * 3 + 2];
                                if min_floor > 0 && rgb_lum(r, g, b) < min_floor {
                                    r = r.max(min_floor);
                                    g = g.max(min_floor);
                                    b = b.max(min_floor);
                                }
                                crop_rgb[i * 3] = r;
                                crop_rgb[i * 3 + 1] = g;
                                crop_rgb[i * 3 + 2] = b;
                            }
                        }
                    }
                }

                // 2. Zero-Bleed Context Shielding + Hermite Wall Bridge для нависающих штрихов СНАРУЖИ
                if has_outside {
                    let mut shield_bubble = poly_mask.clone();
                    for i in 0..n {
                        if raw_poly_mask[i] > 0 {
                            shield_bubble[i] = 255;
                        }
                    }
                    let ext_ring = dilate_ellipse(&shield_bubble, cw, ch, 16);
                    let mut ext_sel = vec![0u8; n];
                    for i in 0..n {
                        if ext_ring[i] > 0 && shield_bubble[i] == 0 && crop_mask[i] == 0 {
                            ext_sel[i] = 255;
                        }
                    }
                    let ext_bg = compute_rgb_median_and_std(&crop_rgb, &ext_sel)
                        .map(|(c, _)| c)
                        .unwrap_or([245, 245, 245]);

                    let mut work_out = crop_rgb.clone();
                    // Забиваем внутренность баббла внешним цветом, чтобы белый цвет не вытекал наружу!
                    fill_masked_with_rgb(&mut work_out, &shield_bubble, ext_bg);
                    let used_lama = if let Some(ref mut lama) = self.lama {
                        lama.inpaint_crop_rgb(&mut work_out, &outside_mask, cw, ch)
                            .is_ok()
                    } else {
                        false
                    };
                    if !used_lama {
                        inpaint_laplace_sor(&mut work_out, &outside_mask, cw, ch, 160, 1.75);
                    }
                    for i in 0..n {
                        if outside_mask[i] > 0 {
                            crop_rgb[i * 3] = work_out[i * 3];
                            crop_rgb[i * 3 + 1] = work_out[i * 3 + 1];
                            crop_rgb[i * 3 + 2] = work_out[i * 3 + 2];
                        }
                    }

                    // Замыкаем разорванную стенку баббла сплайном Эрмита
                    repair_broken_bubble_walls(
                        &mut crop_rgb,
                        cw,
                        ch,
                        &crop_mask,
                        &raw_poly_mask,
                        &local_poly,
                    );
                }
            } else {
                // Tract B:
                // Путь B1 (`!is_free_text` с доказанной замкнутой белой полостью `safe_cav`) -> чистая плоская заливка строго внутри полости;
                // Путь B2 (`is_free_text` или безрамочный баббл на фоне/скринтоне) -> ТОЛЬКО нейро-инпейнтинг Anime-LaMa (или Laplace SOR), без белых квадратов!
                let mut did_flat_fill = false;
                if !blk.is_free_text {
                    if let Some(ref safe_cav) = tract_b_safe_cav {
                        let mut cav_sel = vec![0u8; n];
                        let mut cav_cnt = 0usize;
                        for i in 0..n {
                            if safe_cav[i] > 0 && crop_mask[i] == 0 {
                                cav_sel[i] = 255;
                                cav_cnt += 1;
                            }
                        }
                        if cav_cnt >= 12 {
                            let (bg_rgb, is_flat) = evaluate_flat_bubble_bg(&crop_rgb, &cav_sel);
                            if is_flat && rgb_lum(bg_rgb[0], bg_rgb[1], bg_rgb[2]) >= 228 {
                                fill_masked_with_rgb(&mut crop_rgb, &crop_mask, bg_rgb);
                                did_flat_fill = true;
                            }
                        }
                    }
                }
                if !did_flat_fill {
                    let lama_res = if let Some(ref mut lama) = self.lama {
                        lama.inpaint_on_page(&cleaned, rect, &crop_mask).ok()
                    } else {
                        None
                    };
                    if let Some(inpainted) = lama_res {
                        crop_rgb = inpainted;
                    } else {
                        inpaint_laplace_sor(&mut crop_rgb, &crop_mask, cw, ch, 160, 1.75);
                    }
                }
            }

            let dst_raw = cleaned.as_mut();
            for ry in 0..ch {
                let dst_row = ((y0 + ry) * wu + x0) * 3;
                let src_row = ry * cw * 3;
                let m_row = ry * cw;
                for rx in 0..cw {
                    if crop_mask[m_row + rx] > 0 {
                        let d = dst_row + rx * 3;
                        let s = src_row + rx * 3;
                        dst_raw[d] = crop_rgb[s];
                        dst_raw[d + 1] = crop_rgb[s + 1];
                        dst_raw[d + 2] = crop_rgb[s + 2];
                    }
                }
            }
        }

        // Гарантируем, что пиксели вне `full_mask` (в т.ч. после erase-кисти) остаются побайтово оригинальными
        let dst_raw = cleaned.as_mut();
        for (i, &m) in full_mask.iter().enumerate() {
            if m == 0 {
                dst_raw[i * 3] = orig_raw[i * 3];
                dst_raw[i * 3 + 1] = orig_raw[i * 3 + 1];
                dst_raw[i * 3 + 2] = orig_raw[i * 3 + 2];
            }
        }

        let inpaint_ms = t_inpaint.elapsed().as_millis();
        CleanPageResult {
            cleaned,
            mask_nonzero,
            blocks_cleaned,
            timings: CleanTimings {
                ep: self.ep.clone(),
                seg_ms: 0,
                mask_ms,
                inpaint_ms,
                total_ms: t_total.elapsed().as_millis(),
                cache_hit: false,
            },
        }
    }
}
