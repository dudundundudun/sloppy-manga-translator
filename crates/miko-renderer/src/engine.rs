use crate::layout::{harmonize_page_font_sizes, solve_bubble_layout, BubbleLayout};
use crate::painter::{
    draw_text_line, measure_text_width, parse_hex_color, resolve_font_for_text, FontCache,
};
use crate::typography::format_comic_typography;
use ab_glyph::{Font, FontVec, PxScale, ScaleFont};
use image::RgbImage;
use miko_cleaner::{compute_rgb_median_and_std, dilate_ellipse, erode_ellipse, fill_polygon};
use miko_core::{clip_polygon_to_rect, TextBlock};
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

fn build_oval_fallback_mask(cw: usize, ch: usize, is_free_text: bool) -> Vec<u8> {
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
            if is_free_text || (dx * dx + dy * dy <= 1.0) {
                mask[row + x] = 255;
            }
        }
    }
    mask
}

/// Строит маску доступной полости СТРОГО в границах `blk.bbox` (отсекая общий полигон
/// сдвоенного/тройного баббла, чтобы соседние блоки никогда не получали чужую площадь!).
fn build_cavity_mask(
    blk: &TextBlock,
    all_blocks: &[TextBlock],
    self_idx: usize,
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
    let oval_mask = build_oval_fallback_mask(cw, ch, blk.is_free_text);

    let mut mask = if !blk.is_free_text {
        if let Some(ref poly) = blk.bubble_polygon {
            let clipped = clip_polygon_to_rect(poly, [x1, y1, x2, y2]);
            if clipped.len() >= 3 {
                let local_poly: Vec<[i32; 2]> =
                    clipped.iter().map(|&[px, py]| [px - x1, py - y1]).collect();
                let raw_mask = fill_polygon(&local_poly, cw, ch);
                let erode_r = if cw.min(ch) < 90 { 3 } else { 5 };
                let mut shrunk = erode_ellipse(&raw_mask, cw, ch, erode_r);
                if !shrunk.iter().any(|&v| v > 0) {
                    shrunk = raw_mask;
                }
                // Если полигон покрывает < 35% площади `bbox` (например, усечённый хвост `002.webp`),
                // объединяем с внутренним овалом `bbox`, чтобы вёрстка использовала всю высоту баббла!
                let area = shrunk.iter().filter(|&&v| v > 0).count();
                if area * 100 < (cw * ch) * 35 {
                    for i in 0..(cw * ch) {
                        if oval_mask[i] > 0 {
                            shrunk[i] = 255;
                        }
                    }
                }
                shrunk
            } else {
                oval_mask
            }
        } else {
            oval_mask
        }
    } else {
        oval_mask
    };

    // Вычитаем текстовые ядра соседних бабблов при частичном нахлёсте `bbox` (как `#2` и `#3` на `005.webp`),
    // чтобы солвер вёрстки никогда не размещал центр текущего текста поверх чужой реплики.
    if !blk.is_free_text {
        let self_anchor = blk.text_bbox.unwrap_or(blk.bbox);
        let self_cy = (self_anchor[1] + self_anchor[3]) / 2;
        for (j, other) in all_blocks.iter().enumerate() {
            if j == self_idx || other.is_deleted || other.is_free_text {
                continue;
            }
            if other.translated_text.trim().is_empty() && other.original_text.trim().is_empty() {
                continue;
            }
            let ob = other.text_bbox.unwrap_or(other.bbox);
            let ix1 = (ob[0] - 4).max(x1);
            let iy1 = (ob[1] - 4).max(y1);
            let ix2 = (ob[2] + 4).min(x2);
            let iy2 = (ob[3] + 4).min(y2);
            if ix2 > ix1 && iy2 > iy1 {
                let other_cy = (ob[1] + ob[3]) / 2;
                if (other_cy - self_cy).abs() >= 16 {
                    let mut saved_mask = mask.clone();
                    for py in iy1..iy2 {
                        let row = ((py - y1) as usize) * cw;
                        for px in ix1..ix2 {
                            saved_mask[row + (px - x1) as usize] = 0;
                        }
                    }
                    if saved_mask.iter().filter(|&&v| v > 0).count() >= 64 {
                        mask = saved_mask;
                    }
                }
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
            let (rect, cavity, cw, ch) = build_cavity_mask(blk, blocks, idx, iw, ih);
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

        // Проход 3: защита от вертикального наложения текстовых боксов в соприкасающихся бабблах
        for pass in 0..2 {
            let _ = pass;
            let len = prepared.len();
            for i in 0..len {
                for j in (i + 1)..len {
                    let wi = prepared[i].layout.max_line_w * 0.5;
                    let hi = (prepared[i].layout.lines.len() as f32
                        * prepared[i].layout.line_height)
                        * 0.5;
                    let wj = prepared[j].layout.max_line_w * 0.5;
                    let hj = (prepared[j].layout.lines.len() as f32
                        * prepared[j].layout.line_height)
                        * 0.5;

                    let cxi = (prepared[i].rect[0] + prepared[i].layout.center_xy[0]) as f32;
                    let cyi = (prepared[i].rect[1] + prepared[i].layout.center_xy[1]) as f32;
                    let cxj = (prepared[j].rect[0] + prepared[j].layout.center_xy[0]) as f32;
                    let cyj = (prepared[j].rect[1] + prepared[j].layout.center_xy[1]) as f32;

                    let ov_x = (wi + wj + 4.0) - (cxi - cxj).abs();
                    let ov_y = (hi + hj + 4.0) - (cyi - cyj).abs();
                    if ov_x > 0.0 && ov_y > 0.0 {
                        let push = ((ov_y * 0.5).ceil() as i32).max(2);
                        if cyi <= cyj {
                            let min_i = (hi.ceil() as i32 + 2).min(prepared[i].ch as i32 / 2);
                            let max_j = (prepared[j].ch as i32 - hj.ceil() as i32 - 2)
                                .max(prepared[j].ch as i32 / 2);
                            prepared[i].layout.center_xy[1] =
                                (prepared[i].layout.center_xy[1] - push).max(min_i);
                            prepared[j].layout.center_xy[1] =
                                (prepared[j].layout.center_xy[1] + push).min(max_j);
                        } else {
                            let max_i = (prepared[i].ch as i32 - hi.ceil() as i32 - 2)
                                .max(prepared[i].ch as i32 / 2);
                            let min_j = (hj.ceil() as i32 + 2).min(prepared[j].ch as i32 / 2);
                            prepared[i].layout.center_xy[1] =
                                (prepared[i].layout.center_xy[1] + push).min(max_i);
                            prepared[j].layout.center_xy[1] =
                                (prepared[j].layout.center_xy[1] - push).max(min_j);
                        }
                    }
                }
            }
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
