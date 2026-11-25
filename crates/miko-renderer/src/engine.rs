use crate::layout::{harmonize_page_font_sizes, solve_bubble_layout, BubbleLayout};
use crate::painter::{
    draw_text_line, measure_text_width, parse_hex_color, resolve_font_for_text, FontCache,
};
use crate::typography::format_comic_typography;
use ab_glyph::{Font, FontVec, PxScale, ScaleFont};
use image::RgbImage;
use miko_cleaner::{compute_rgb_median_and_std, dilate_ellipse, erode_ellipse, fill_polygon};
use miko_core::TextBlock;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RendererError {
    #[error("Не найден подходящий шрифт для отрисовки текста")]
    NoFontAvailable,
    #[error("Ошибка шрифта: {0}")]
    FontError(String),
}

#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub min_font_size: i32,
    pub max_font_size: i32,
    pub uppercase: bool,
    pub overflow_mode: String,
    pub default_font_path: Option<PathBuf>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            min_font_size: 12,
            max_font_size: 36,
            uppercase: false,
            overflow_mode: "shrink".to_string(),
            default_font_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RenderTimings {
    pub layout_ms: u128,
    pub paint_ms: u128,
    pub total_ms: u128,
}

pub struct RenderPageResult {
    pub rendered: RgbImage,
    pub blocks_rendered: usize,
    pub timings: RenderTimings,
}

/// Движок вёрстки и отрисовки переведённого текста на чистом Rust (`ab_glyph`).
pub struct RustRendererEngine {
    fonts_dir: PathBuf,
    font_cache: FontCache,
}

#[inline]
fn rgb_lum(c: [u8; 3]) -> u8 {
    (((77u32 * c[0] as u32) + (150u32 * c[1] as u32) + (29u32 * c[2] as u32)) >> 8) as u8
}

fn build_cavity_mask(
    blk: &TextBlock,
    img_w: i32,
    img_h: i32,
) -> ([i32; 4], Vec<u8>, usize, usize) {
    let b = blk.bbox;
    let x1 = b[0].clamp(0, (img_w - 1).max(0));
    let y1 = b[1].clamp(0, (img_h - 1).max(0));
    let x2 = b[2].clamp(x1 + 1, img_w.max(1));
    let y2 = b[3].clamp(y1 + 1, img_h.max(1));
    let cw = (x2 - x1) as usize;
    let ch = (y2 - y1) as usize;

    if !blk.is_free_text {
        if let Some(ref poly) = blk.bubble_polygon {
            if poly.len() >= 3 {
                let mut px1 = x1;
                let mut py1 = y1;
                let mut px2 = x2;
                let mut py2 = y2;
                for &[px, py] in poly {
                    px1 = px1.min(px);
                    py1 = py1.min(py);
                    px2 = px2.max(px);
                    py2 = py2.max(py);
                }
                let rx1 = px1.clamp(0, (img_w - 1).max(0));
                let ry1 = py1.clamp(0, (img_h - 1).max(0));
                let rx2 = px2.clamp(rx1 + 1, img_w.max(1));
                let ry2 = py2.clamp(ry1 + 1, img_h.max(1));
                let rw = (rx2 - rx1) as usize;
                let rh = (ry2 - ry1) as usize;
                let local_poly: Vec<[i32; 2]> = poly.iter().map(|&[px, py]| [px - rx1, py - ry1]).collect();
                let raw_mask = fill_polygon(&local_poly, rw, rh);
                let mut shrunk = erode_ellipse(&raw_mask, rw, rh, 6);
                if !shrunk.iter().any(|&v| v > 0) {
                    shrunk = raw_mask;
                }
                return ([rx1, ry1, rx2, ry2], shrunk, rw, rh);
            }
        }
    }

    // Фоллбэк: овал/прямоугольник внутри bbox (с отступом 5% от краёв)
    let mut mask = vec![0u8; cw * ch];
    let cx = cw as f32 * 0.5;
    let cy = ch as f32 * 0.5;
    let rx = (cw as f32 * 0.45).max(4.0);
    let ry = (ch as f32 * 0.45).max(4.0);
    for y in 0..ch {
        let dy = (y as f32 + 0.5 - cy) / ry;
        let row = y * cw;
        for x in 0..cw {
            let dx = (x as f32 + 0.5 - cx) / rx;
            if blk.is_free_text || (dx * dx + dy * dy <= 1.0) {
                mask[row + x] = 255;
            }
        }
    }
    ([x1, y1, x2, y2], mask, cw, ch)
}

impl RustRendererEngine {
    pub fn new(fonts_dir: PathBuf) -> Self {
        Self {
            fonts_dir,
            font_cache: FontCache::new(),
        }
    }

    fn resolve_block_font(&self, blk: &TextBlock, opts: &RenderOptions, text: &str) -> Result<Arc<FontVec>, RendererError> {
        let pref = blk
            .font_family
            .as_deref()
            .or_else(|| opts.default_font_path.as_ref().and_then(|p| p.to_str()));
        let font_path = resolve_font_for_text(&self.fonts_dir, &self.font_cache, pref, text)
            .ok_or(RendererError::NoFontAvailable)?;
        self.font_cache
            .load_font(&font_path)
            .map_err(RendererError::FontError)
    }

    /// Двухпроходная вёрстка и рендер переведённых блоков на очищенную страницу.
    pub fn render_page(
        &self,
        cleaned_img: &RgbImage,
        blocks: &mut [TextBlock],
        opts: &RenderOptions,
    ) -> Result<RenderPageResult, RendererError> {
        let t_total = Instant::now();
        let mut rendered = cleaned_img.clone();
        let (iw, ih) = (cleaned_img.width() as i32, cleaned_img.height() as i32);

        struct PreparedBlock {
            blk_idx: usize,
            rect: [i32; 4],
            cavity: Vec<u8>,
            cw: usize,
            ch: usize,
            formatted: String,
            font: Arc<FontVec>,
            layout: BubbleLayout,
            is_fixed: bool,
        }

        let t_layout = Instant::now();
        let mut prepared: Vec<PreparedBlock> = Vec::new();
        let mut raw_sizes: Vec<(usize, i32, usize, bool)> = Vec::new();

        for (idx, blk) in blocks.iter().enumerate() {
            if blk.is_deleted {
                continue;
            }
            let raw_text = if !blk.translated_text.trim().is_empty() {
                blk.translated_text.trim()
            } else {
                continue;
            };
            let formatted = format_comic_typography(raw_text, opts.uppercase);
            if formatted.is_empty() {
                continue;
            }
            let font = self.resolve_block_font(blk, opts, &formatted)?;
            let (rect, cavity, cw, ch) = build_cavity_mask(blk, iw, ih);
            let layout = solve_bubble_layout(
                &formatted,
                &cavity,
                cw,
                ch,
                opts.min_font_size,
                opts.max_font_size,
                blk.font_size,
                |s, fs| measure_text_width(&font, fs, s),
            );
            let is_fixed = blk.font_size.is_some() || blk.is_free_text;
            let char_cnt = formatted.chars().count();
            let pos = prepared.len();
            raw_sizes.push((pos, layout.font_size, char_cnt, is_fixed));
            prepared.push(PreparedBlock {
                blk_idx: idx,
                rect,
                cavity,
                cw,
                ch,
                formatted,
                font,
                layout,
                is_fixed,
            });
        }

        // Проход 2: гармонизация кеглей на уровне страницы
        let harmonized = harmonize_page_font_sizes(&raw_sizes);
        for (pos, item) in prepared.iter_mut().enumerate() {
            let target_fs = harmonized[pos];
            if !item.is_fixed && target_fs != item.layout.font_size {
                item.layout = solve_bubble_layout(
                    &item.formatted,
                    &item.cavity,
                    item.cw,
                    item.ch,
                    opts.min_font_size,
                    target_fs,
                    Some(target_fs),
                    |s, fs| measure_text_width(&item.font, fs, s),
                );
            }
            blocks[item.blk_idx].rendered_font_size = Some(item.layout.font_size);
        }
        let layout_ms = t_layout.elapsed().as_millis();

        // Отрисовка каждого блока
        let t_paint = Instant::now();
        let blocks_rendered = prepared.len();
        for item in prepared {
            let blk = &blocks[item.blk_idx];
            let [rx0, ry0, _, _] = item.rect;

            // Определяем полярность фона внутри полости
            let mut crop_rgb = Vec::with_capacity(item.cw * item.ch * 3);
            for ry in 0..item.ch {
                let y = ry0 as usize + ry;
                for rx in 0..item.cw {
                    let x = rx0 as usize + rx;
                    let p = cleaned_img.get_pixel(x as u32, y as u32);
                    crop_rgb.push(p[0]);
                    crop_rgb.push(p[1]);
                    crop_rgb.push(p[2]);
                }
            }
            let bg_rgb = compute_rgb_median_and_std(&crop_rgb, &item.cavity)
                .map(|(c, _)| c)
                .unwrap_or([250, 250, 250]);
            let is_dark_bg = rgb_lum(bg_rgb) < 120;

            let fill_rgb = blk
                .text_color
                .as_deref()
                .and_then(parse_hex_color)
                .unwrap_or_else(|| {
                    if is_dark_bg || blk.is_free_text {
                        [252, 252, 252]
                    } else {
                        [18, 18, 22]
                    }
                });

            let stroke_opt = blk
                .stroke_color
                .as_deref()
                .and_then(parse_hex_color)
                .map(|sc| (sc, blk.stroke_width.unwrap_or(2)))
                .or_else(|| {
                    if blk.is_free_text || is_dark_bg {
                        Some(([16, 16, 20], 2))
                    } else {
                        None
                    }
                });

            // Если включена подложка bg_style ("box" / "pill")
            if blk.bg_style == "box" || blk.bg_style == "pill" {
                let pad = blk.bg_padding.unwrap_or(6).max(2);
                let bg_fill = blk
                    .bg_color
                    .as_deref()
                    .and_then(parse_hex_color)
                    .unwrap_or([255, 255, 255]);
                let box_w = item.layout.max_line_w.ceil() as i32 + 2 * pad;
                let box_h =
                    (item.layout.lines.len() as f32 * item.layout.line_height).ceil() as i32 + 2 * pad;
                let cx = rx0 + item.layout.center_xy[0];
                let cy = ry0 + item.layout.center_xy[1];
                let x_a = (cx - box_w / 2).clamp(0, iw - 1);
                let y_a = (cy - box_h / 2).clamp(0, ih - 1);
                let x_b = (cx + box_w / 2).clamp(x_a + 1, iw);
                let y_b = (cy + box_h / 2).clamp(y_a + 1, ih);
                let mut pill_mask = vec![255u8; ((x_b - x_a) * (y_b - y_a)) as usize];
                if blk.bg_style == "pill" {
                    let mw = (x_b - x_a) as usize;
                    let mh = (y_b - y_a) as usize;
                    pill_mask = dilate_ellipse(&pill_mask, mw, mh, 0);
                }
                for py in y_a..y_b {
                    for px in x_a..x_b {
                        let m = pill_mask[((py - y_a) as usize) * ((x_b - x_a) as usize) + (px - x_a) as usize];
                        if m > 0 {
                            rendered.put_pixel(px as u32, py as u32, image::Rgb(bg_fill));
                        }
                    }
                }
            }

            let px_size = item.layout.font_size as f32;
            let scaled = item.font.as_scaled(PxScale::from(px_size));
            let ascent = scaled.ascent();
            let line_h = item.layout.line_height;
            let n_lines = item.layout.lines.len();
            let total_h = (n_lines as f32) * line_h;

            let center_page_x = (rx0 + item.layout.center_xy[0]) as f32;
            let top_page_y = (ry0 + item.layout.center_xy[1]) as f32 - total_h * 0.5;

            for (k, line) in item.layout.lines.iter().enumerate() {
                let lw = measure_text_width(&item.font, px_size, line);
                let x_left = match blk.align.as_str() {
                    "left" => center_page_x - item.layout.max_line_w * 0.5,
                    "right" => center_page_x + item.layout.max_line_w * 0.5 - lw,
                    _ => center_page_x - lw * 0.5,
                };
                let y_baseline = top_page_y + (k as f32) * line_h + ascent;
                draw_text_line(
                    &mut rendered,
                    &item.font,
                    px_size,
                    x_left,
                    y_baseline,
                    line,
                    fill_rgb,
                    stroke_opt,
                );
            }
        }
        let paint_ms = t_paint.elapsed().as_millis();

        Ok(RenderPageResult {
            rendered,
            blocks_rendered,
            timings: RenderTimings {
                layout_ms,
                paint_ms,
                total_ms: t_total.elapsed().as_millis(),
            },
        })
    }
}
