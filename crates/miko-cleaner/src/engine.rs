use crate::cache::compute_clean_state_hash;
use crate::cv::{
    compute_rgb_median_and_std, connected_components, dilate_ellipse, distance_transform_l2,
    draw_circle, draw_polyline, erode_ellipse, fill_polygon, hermite_wall_bridge,
    inpaint_laplace_sor, otsu_threshold,
};
use crate::disk_cache::CleanDiskCache;
use crate::segmenter::NeuralStrokeSegmenter;
use image::RgbImage;
use miko_core::{clip_polygon_to_rect, TextBlock};
use miko_ort::OrtError;
use serde::Serialize;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;
use thiserror::Error;

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
}

impl Default for CleanOptions {
    fn default() -> Self {
        Self {
            dilate_size: 9,
            pipeline_mode: "balanced".to_string(),
            use_neural_segmenter: true,
            use_cache: true,
            neural_conf: 0.25,
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

fn is_block_skipped(blk: &TextBlock) -> bool {
    blk.is_deleted
        || (blk.translated_text.trim().is_empty() && blk.original_text.trim().is_empty())
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

/// Растеризует полигон баббла и доращивает полость внутри `effective_clean_bbox`,
/// если YOLO26s-seg обрезал хвост высокого баббла (как `ああ!!` на `002.webp`), останавливаясь строго у тёмных стенок.
fn build_effective_poly_mask(
    crop_rgb: &[u8],
    cw: usize,
    ch: usize,
    x0: i32,
    y0: i32,
    blk: &TextBlock,
    local_poly: &[[i32; 2]],
) -> Vec<u8> {
    let mut poly_mask = fill_polygon(local_poly, cw, ch);
    let n = cw * ch;
    if n == 0 {
        return poly_mask;
    }

    let eb = effective_clean_bbox(blk);
    let bx1 = (eb[0] - x0).clamp(0, cw as i32) as usize;
    let by1 = (eb[1] - y0).clamp(0, ch as i32) as usize;
    let bx2 = (eb[2] - x0).clamp(bx1 as i32, cw as i32) as usize;
    let by2 = (eb[3] - y0).clamp(by1 as i32, ch as i32) as usize;
    if bx2 <= bx1 + 8 || by2 <= by1 + 8 {
        return poly_mask;
    }

    let seed_eroded = erode_ellipse(&poly_mask, cw, ch, 3);
    let seed_ref = if seed_eroded.iter().any(|&v| v > 0) {
        &seed_eroded
    } else {
        &poly_mask
    };
    let Some((bg_rgb, _)) = compute_rgb_median_and_std(crop_rgb, seed_ref) else {
        return poly_mask;
    };
    let bg_lum = rgb_lum(bg_rgb[0], bg_rgb[1], bg_rgb[2]);
    if bg_lum < 200 {
        return poly_mask;
    }

    // Проверяем, осталась ли часть `text_bbox` или `clean_bbox` со светлой полостью снаружи `poly_mask`
    let check_box = blk.text_bbox.unwrap_or(eb);
    let tx1 = (check_box[0] - x0).clamp(bx1 as i32, bx2 as i32) as usize;
    let ty1 = (check_box[1] - y0).clamp(by1 as i32, by2 as i32) as usize;
    let tx2 = (check_box[2] - x0).clamp(tx1 as i32, bx2 as i32) as usize;
    let ty2 = (check_box[3] - y0).clamp(ty1 as i32, by2 as i32) as usize;
    let mut uncovered_bright = 0usize;
    for ry in ty1..ty2 {
        let row = ry * cw;
        for rx in tx1..tx2 {
            let i = row + rx;
            if poly_mask[i] == 0 {
                let r = crop_rgb[i * 3];
                let g = crop_rgb[i * 3 + 1];
                let b = crop_rgb[i * 3 + 2];
                if rgb_lum(r, g, b) >= 205 && rgb_l1_dist(r, g, b, bg_rgb) <= 45 {
                    uncovered_bright += 1;
                }
            }
        }
    }
    if uncovered_bright < 24 {
        return poly_mask;
    }

    // 4-связный BFS по светлой полости баббла внутри `eb`, строго запертый тёмными стенками баббла
    let mut walkable = vec![false; n];
    for ry in by1..by2 {
        let row = ry * cw;
        for rx in bx1..bx2 {
            let i = row + rx;
            let r = crop_rgb[i * 3];
            let g = crop_rgb[i * 3 + 1];
            let b = crop_rgb[i * 3 + 2];
            let lum = rgb_lum(r, g, b);
            if lum >= 205 && rgb_l1_dist(r, g, b, bg_rgb) <= 48 {
                walkable[i] = true;
            }
        }
    }

    let mut cav_flood = vec![0u8; n];
    let mut q: VecDeque<(usize, usize)> = VecDeque::new();
    for ry in by1..by2 {
        let row = ry * cw;
        for rx in bx1..bx2 {
            let i = row + rx;
            if seed_ref[i] > 0 && walkable[i] {
                cav_flood[i] = 255;
                q.push_back((rx, ry));
            }
        }
    }
    while let Some((x, y)) = q.pop_front() {
        let nbrs = [
            (x.wrapping_sub(1), y, x > bx1),
            (x + 1, y, x + 1 < bx2),
            (x, y.wrapping_sub(1), y > by1),
            (x, y + 1, y + 1 < by2),
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

    // Защита от открытых/облачных бабблов: если разлитая полость касается рамки `eb` слишком широко, отклоняем!
    let mut perim_touch = 0usize;
    for rx in bx1..bx2 {
        if cav_flood[by1 * cw + rx] > 0 {
            perim_touch += 1;
        }
        if cav_flood[(by2 - 1) * cw + rx] > 0 {
            perim_touch += 1;
        }
    }
    for ry in by1..by2 {
        if cav_flood[ry * cw + bx1] > 0 {
            perim_touch += 1;
        }
        if cav_flood[ry * cw + (bx2 - 1)] > 0 {
            perim_touch += 1;
        }
    }
    let perim_total = 2 * ((bx2 - bx1) + (by2 - by1)).max(1);
    // У сдвоенных бабблов (005.webp) узкий белый перешеек в другой баббл даёт ~5–12% касания периметра `eb`;
    // открытый же фон без рамки даёт > 22% касания. Порог 14% безопасно доращивает полость у перешейков!
    if perim_touch * 100 > perim_total * 14 {
        return poly_mask;
    }

    // Заполняем внутренние дырки внутри `cav_flood` (сами чёрные буквы `ああ!!`, полностью окружённые белой полостью)
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
    for i in 0..n {
        if !ext_vis[i] {
            cav_filled[i] = 255;
        }
    }
    // Лёгкое расширение на 1px, чтобы `erode_ellipse` в `build_tract_a_crop_mask` отступил ровно от чёрной стенки
    let cav_padded = dilate_ellipse(&cav_filled, cw, ch, 1);
    for i in 0..n {
        if cav_padded[i] > 0 {
            poly_mask[i] = 255;
        }
    }
    poly_mask
}

/// Вычисляет робастный цвет и признак однородности плоского баббла, игнорируя <= 20% краевых антиалиас-пикселей,
/// чтобы чистые белые/однотонные бабблы никогда не проваливались в размытие Лапласа (`inpaint_laplace_sor`).
fn evaluate_flat_bubble_bg(crop_rgb: &[u8], selector: &[u8]) -> ([u8; 3], bool) {
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
        total += 1;
        let r = crop_rgb[i * 3];
        let g = crop_rgb[i * 3 + 1];
        let b = crop_rgb[i * 3 + 2];
        if rgb_l1_dist(r, g, b, bg_rgb) <= 36 {
            inliers += 1;
            let lum = rgb_lum(r, g, b) as f64;
            sum_lum += lum;
            sum_lum_sq += lum * lum;
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
    if bg_rgb[0] >= 244 && bg_rgb[1] >= 244 && bg_rgb[2] >= 244 {
        bg_rgb = [255, 255, 255];
    }
    let is_flat = bg_std <= 10.0 || (inlier_ratio >= 0.80 && inlier_std <= 6.5);
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
            let in_cv_roi = cv_roi_local
                .is_none_or(|b| rx >= b[0] && rx < b[2] && ry >= b[1] && ry < b[3]);
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

        // Внешний фон вокруг баббла (чтобы нависающие штрихи сравнивались с внешним фоном, а не внутренностью баббла)
        let ext_bg_rgb = if neural_crop.is_none() {
            let ext_ring = dilate_ellipse(&poly_mask, cw, ch, 12);
            let mut ext_sel = vec![0u8; n];
            for i in 0..n {
                if ext_ring[i] > 0 && poly_mask[i] == 0 {
                    ext_sel[i] = 255;
                }
            }
            compute_rgb_median_and_std(crop_rgb, &ext_sel)
                .map(|(c, _)| c)
                .unwrap_or(bg_rgb)
        } else {
            bg_rgb
        };
        let ext_lum = rgb_lum(ext_bg_rgb[0], ext_bg_rgb[1], ext_bg_rgb[2]);

        for ry in ty1..ty2 {
            let row = ry * cw;
            for rx in tx1..tx2 {
                let i = row + rx;
                if poly_mask[i] > 0 {
                    continue;
                }
                let nn_over = neural_crop.is_some_and(|nm| nm[i] >= 120);
                let cv_over = if neural_crop.is_none() && blk.text_bbox.is_some() {
                    let r = crop_rgb[i * 3];
                    let g = crop_rgb[i * 3 + 1];
                    let b = crop_rgb[i * 3 + 2];
                    let lum = rgb_lum(r, g, b);
                    let chroma = rgb_chroma(r, g, b);
                    let l1_ext = rgb_l1_dist(r, g, b, ext_bg_rgb);
                    (lum as i32 - ext_lum as i32).abs() > 42 || (chroma >= 45 && l1_ext >= 60)
                } else {
                    false
                };
                if nn_over || cv_over {
                    outside_raw[i] = 255;
                    has_outside = true;
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
    let clamp_region = if eff_dil > 10 { &poly_mask } else { &shrunk_poly };

    let mut out_mask = vec![0u8; n];
    for i in 0..n {
        if (dil_in[i] > 0 && clamp_region[i] > 0) || outside_dil[i] > 0 {
            out_mask[i] = 255;
        }
    }

    (out_mask, poly_mask)
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
) -> Vec<u8> {
    let eb = effective_clean_bbox(blk);
    let bx1 = (eb[0] - x0).clamp(0, cw as i32) as usize;
    let by1 = (eb[1] - y0).clamp(0, ch as i32) as usize;
    let bx2 = (eb[2] - x0).clamp(bx1 as i32, cw as i32) as usize;
    let by2 = (eb[3] - y0).clamp(by1 as i32, ch as i32) as usize;
    let n = cw * ch;
    let mut cand = vec![0u8; n];
    if bx1 >= bx2 || by1 >= by2 {
        return cand;
    }

    let mut nn_hits = 0usize;
    if let Some(nm) = neural_crop {
        for ry in by1..by2 {
            let row = ry * cw;
            for rx in bx1..bx2 {
                let i = row + rx;
                if nm[i] >= 90 {
                    cand[i] = 255;
                    nn_hits += 1;
                }
            }
        }
    }

    if nn_hits < 15 {
        // Определяем фоновый цвет по узкой рамке вокруг bbox внутри кропа
        let mut border_sel = vec![0u8; n];
        let ox1 = bx1.saturating_sub(4);
        let oy1 = by1.saturating_sub(4);
        let ox2 = (bx2 + 4).min(cw);
        let oy2 = (by2 + 4).min(ch);
        for ry in oy1..oy2 {
            let row = ry * cw;
            for rx in ox1..ox2 {
                if rx < bx1 || rx >= bx2 || ry < by1 || ry >= by2 {
                    border_sel[row + rx] = 255;
                }
            }
        }
        let border_lum = compute_rgb_median_and_std(crop_rgb, &border_sel)
            .map(|(rgb, _)| rgb_lum(rgb[0], rgb[1], rgb[2]))
            .unwrap_or(245);

        let mut box_gray = Vec::with_capacity((bx2 - bx1) * (by2 - by1));
        for ry in by1..by2 {
            let row = ry * cw;
            for rx in bx1..bx2 {
                let i = row + rx;
                box_gray.push(rgb_lum(crop_rgb[i * 3], crop_rgb[i * 3 + 1], crop_rgb[i * 3 + 2]));
            }
        }
        let otsu = otsu_threshold(&box_gray);
        let dark_bg = border_lum < 110;

        for ry in by1..by2 {
            let row = ry * cw;
            for rx in bx1..bx2 {
                let i = row + rx;
                let r = crop_rgb[i * 3];
                let g = crop_rgb[i * 3 + 1];
                let b = crop_rgb[i * 3 + 2];
                let lum = rgb_lum(r, g, b);
                let chroma = rgb_chroma(r, g, b);
                let contrast = (lum as i32 - border_lum as i32).abs() > 38;
                let otsu_hit = if dark_bg { lum > otsu } else { lum < otsu };
                if (contrast && otsu_hit) || chroma >= 45 {
                    cand[i] = 255;
                }
            }
        }
    }

    let eff_r = if blk.is_free_text && blk.mask_dilate.is_none() {
        2usize
    } else {
        (blk.mask_dilate.unwrap_or(default_dilate).clamp(1, 28) as usize / 2).max(1)
    };
    let dil = dilate_ellipse(&cand, cw, ch, eff_r);

    let cx1 = bx1.saturating_sub(4);
    let cy1 = by1.saturating_sub(4);
    let cx2 = (bx2 + 4).min(cw);
    let cy2 = (by2 + 4).min(ch);
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
    out
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
        let seg_path = base_dir
            .join("models")
            .join("onnx")
            .join("comic_text_segmenter.onnx");

        if seg_path.is_file() {
            match NeuralStrokeSegmenter::load(&seg_path, prefer_gpu) {
                Ok(seg) => {
                    let ep = seg.ep.to_string();
                    tracing::info!("RustCleanerEngine загрузил comic_text_segmenter.onnx на {}", ep);
                    return Self {
                        segmenter: Some(seg),
                        cache,
                        ep,
                    };
                }
                Err(err) => {
                    tracing::warn!(
                        "Не удалось загрузить comic_text_segmenter.onnx ({}); работаем в CV-режиме",
                        err
                    );
                }
            }
        }
        Self {
            segmenter: None,
            cache,
            ep: "CPU-CV".to_string(),
        }
    }

    pub fn new_cv_only(cache_dir: PathBuf) -> Self {
        Self {
            segmenter: None,
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
            if is_block_skipped(blk) {
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
                build_tract_b_crop_mask(
                    &crop_rgb,
                    cw,
                    ch,
                    rect[0],
                    rect[1],
                    blk,
                    nn_crop.as_deref(),
                    opts.dilate_size,
                )
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
            if is_block_skipped(blk) {
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
        let active_blocks = blocks.iter().filter(|b| !is_block_skipped(b)).count();

        let cache_key = if opts.use_cache {
            src_file.map(|sf| {
                compute_clean_state_hash(
                    blocks,
                    sf,
                    src_mtime,
                    opts.dilate_size,
                    "rust_harmonic_v2",
                    &opts.pipeline_mode,
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
        &self,
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
            if is_block_skipped(blk) {
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
                build_tract_b_crop_mask(
                    &orig_crop_rgb,
                    cw,
                    ch,
                    rect[0],
                    rect[1],
                    blk,
                    nn_crop.as_deref(),
                    opts.dilate_size,
                )
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
                let mut poly_mask =
                    build_effective_poly_mask(&crop_rgb, cw, ch, rect[0], rect[1], blk, &local_poly);

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

                // Если компонента `outside_mask` окружена тем же чистым светлым фоном баббла (например, в белом перешейке
                // сдвоенного баббла `005.webp`, где грубая маска YOLO26s-seg чуть уже реальной белой полости),
                // переводим её во внутреннюю заливку `inside_mask`, чтобы в белый баббл НЕ затекал серый скринтон с рубашки!
                if has_outside {
                    let probe_inner = erode_ellipse(&poly_mask, cw, ch, 2);
                    let probe_ref = if probe_inner.iter().any(|&v| v > 0) {
                        &probe_inner
                    } else {
                        &poly_mask
                    };
                    let probe_bg = compute_rgb_median_and_std(&crop_rgb, probe_ref)
                        .map(|(c, _)| c)
                        .unwrap_or([255, 255, 255]);
                    let probe_lum = rgb_lum(probe_bg[0], probe_bg[1], probe_bg[2]);
                    if probe_lum >= 225 {
                        let out_cc = connected_components(&outside_mask, cw, ch);
                        if !out_cc.stats.is_empty() {
                            let mut promote_lbl = vec![false; out_cc.stats.len() + 1];
                            for st in &out_cc.stats {
                                let x_a = st.left.saturating_sub(4);
                                let y_a = st.top.saturating_sub(4);
                                let x_b = (st.left + st.width + 4).min(cw);
                                let y_b = (st.top + st.height + 4).min(ch);
                                let mut non_wall_cnt = 0usize;
                                let mut match_cnt = 0usize;
                                for ry in y_a..y_b {
                                    let row = ry * cw;
                                    for rx in x_a..x_b {
                                        let k = row + rx;
                                        if crop_mask[k] != 0 {
                                            continue;
                                        }
                                        let near = (rx >= st.left.saturating_sub(4)
                                            && rx < (st.left + st.width + 4).min(cw))
                                            && (ry >= st.top.saturating_sub(4)
                                                && ry < (st.top + st.height + 4).min(ch));
                                        if near {
                                            let r = crop_rgb[k * 3];
                                            let g = crop_rgb[k * 3 + 1];
                                            let b = crop_rgb[k * 3 + 2];
                                            let lum = rgb_lum(r, g, b);
                                            // Игнорируем сам чёрный контур баббла (`lum < 75`), сравнивая только фон полости vs внешний фон
                                            if lum >= 75 {
                                                non_wall_cnt += 1;
                                                if lum >= 235
                                                    && rgb_l1_dist(r, g, b, probe_bg) <= 36
                                                {
                                                    match_cnt += 1;
                                                }
                                            }
                                        }
                                    }
                                }
                                if non_wall_cnt >= 6 && match_cnt * 100 >= non_wall_cnt * 55 {
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

                // 1. Очистка текста ВНУТРИ баббла (робастная чистая заливка для плоских бабблов, SOR Лаплас для градиентов)
                if has_inside {
                    let interior_only = erode_ellipse(&poly_mask, cw, ch, 3);
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

                    if is_flat_bubble {
                        fill_masked_with_rgb(&mut crop_rgb, &inside_mask, bg_rgb);
                    } else {
                        let mut work_in = crop_rgb.clone();
                        // Щит для внешности и неснятых тёмных линий у рамки, чтобы тёмный цвет не затекал в полость
                        for i in 0..n {
                            if poly_mask[i] == 0 {
                                work_in[i * 3] = bg_rgb[0];
                                work_in[i * 3 + 1] = bg_rgb[1];
                                work_in[i * 3 + 2] = bg_rgb[2];
                            }
                        }
                        inpaint_laplace_sor(&mut work_in, &inside_mask, cw, ch, 180, 1.75);
                        for i in 0..n {
                            if inside_mask[i] > 0 {
                                crop_rgb[i * 3] = work_in[i * 3];
                                crop_rgb[i * 3 + 1] = work_in[i * 3 + 1];
                                crop_rgb[i * 3 + 2] = work_in[i * 3 + 2];
                            }
                        }
                    }
                }

                // 2. Zero-Bleed Context Shielding + Hermite Wall Bridge для нависающих штрихов СНАРУЖИ
                if has_outside {
                    let ext_ring = dilate_ellipse(&poly_mask, cw, ch, 16);
                    let mut ext_sel = vec![0u8; n];
                    for i in 0..n {
                        if ext_ring[i] > 0 && poly_mask[i] == 0 && crop_mask[i] == 0 {
                            ext_sel[i] = 255;
                        }
                    }
                    let ext_bg = compute_rgb_median_and_std(&crop_rgb, &ext_sel)
                        .map(|(c, _)| c)
                        .unwrap_or([245, 245, 245]);

                    let mut work_out = crop_rgb.clone();
                    // Забиваем внутренность баббла внешним цветом, чтобы белый цвет не вытекал наружу!
                    fill_masked_with_rgb(&mut work_out, &poly_mask, ext_bg);
                    inpaint_laplace_sor(&mut work_out, &outside_mask, cw, ch, 160, 1.75);
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
                        &poly_mask,
                        &local_poly,
                    );
                }
            } else {
                // Tract B: если это обычный светлый закрытый баббл (`!is_free_text`) с чистым однотонным фоном вокруг букв,
                // заливаем его резким фоновым цветом без размытия Лапласа; для текстур и free-text — Лаплас SOR!
                let mut did_flat_fill = false;
                if !blk.is_free_text {
                    let ring = dilate_ellipse(&crop_mask, cw, ch, 4);
                    let mut ring_sel = vec![0u8; n];
                    let mut ring_cnt = 0usize;
                    for i in 0..n {
                        if ring[i] > 0 && crop_mask[i] == 0 {
                            ring_sel[i] = 255;
                            ring_cnt += 1;
                        }
                    }
                    if ring_cnt >= 16 {
                        let (bg_rgb, is_flat) = evaluate_flat_bubble_bg(&crop_rgb, &ring_sel);
                        if is_flat && rgb_lum(bg_rgb[0], bg_rgb[1], bg_rgb[2]) >= 232 {
                            fill_masked_with_rgb(&mut crop_rgb, &crop_mask, bg_rgb);
                            did_flat_fill = true;
                        }
                    }
                }
                if !did_flat_fill {
                    inpaint_laplace_sor(&mut crop_rgb, &crop_mask, cw, ch, 160, 1.75);
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
