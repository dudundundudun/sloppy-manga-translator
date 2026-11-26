use crate::layout::{
    classify_bubble_role, compute_page_target_base_fs, harmonize_page_roles,
    solve_bubble_layout_anchored, BubbleLayout, BubbleTextRole, PageHarmonizeInput,
};
use crate::painter::{
    check_text_lines_wall_contact, draw_text_line_scaled, measure_text_width_scaled,
    parse_hex_color, resolve_font_for_text, sanitize_text_for_font, FontCache,
};
use crate::typography::{compact_scream_for_narrow_bubble, format_comic_typography};
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

/// Проверяет, пересекает ли отрезок между центрами двух бабблов тёмную стенку/рамку на `cleaned_img`.
fn has_dark_wall_between(img: &RgbImage, a: [i32; 2], b: [i32; 2]) -> bool {
    let iw = img.width() as i32;
    let ih = img.height() as i32;
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let steps = dx.abs().max(dy.abs()).max(1);
    let mut dark_px = 0usize;
    for s in 0..=steps {
        let t = (s as f32) / (steps as f32);
        let px = (a[0] as f32 + t * (dx as f32)).round() as i32;
        let py = (a[1] as f32 + t * (dy as f32)).round() as i32;
        if px >= 0 && px < iw && py >= 0 && py < ih {
            let p = img.get_pixel(px as u32, py as u32);
            if rgb_lum([p[0], p[1], p[2]]) < 120 {
                dark_px += 1;
                if dark_px >= 2 {
                    return true;
                }
            }
        }
    }
    false
}

/// Строит маску доступной полости СТРОГО в границах `blk.bbox`:
/// доращивает обрезанные сбоку/снизу полигоны по реальной светлой полости на `cleaned_img` (фикс `014.webp #4`!)
/// и разделяет общие полости сдвоенных бабблов строго по шву между центрами без отпиливания ширины отдельным бабблом сверху!
fn build_cavity_mask(
    cleaned_img: &RgbImage,
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
    let min_dim = cw.min(ch);
    let erode_r = if min_dim < 75 {
        2
    } else if min_dim < 115 {
        3
    } else if min_dim < 155 {
        4
    } else {
        5
    };

    let mut mask = if !blk.is_free_text {
        if let Some(ref poly) = blk.bubble_polygon {
            let clipped = clip_polygon_to_rect(poly, [x1, y1, x2, y2]);
            if clipped.len() >= 3 {
                let local_poly: Vec<[i32; 2]> =
                    clipped.iter().map(|&[px, py]| [px - x1, py - y1]).collect();
                let mut raw_mask = fill_polygon(&local_poly, cw, ch);
                let mut flood_succeeded = false;

                // Попиксельное уточнение полости на `cleaned_img` (где буквы уже стёрты в белый, а стенки бабблов целы):
                // возвращает потерянную ширину, если прото-сетка YOLO26s-seg срезала край баббла (`014.webp #4`)!
                let anchor = blk.text_bbox.unwrap_or(blk.bbox);
                let seed_x = (((anchor[0] + anchor[2]) / 2) - x1).clamp(0, cw as i32 - 1) as usize;
                let seed_y = (((anchor[1] + anchor[3]) / 2) - y1).clamp(0, ch as i32 - 1) as usize;
                let center_p = cleaned_img.get_pixel((x1 + seed_x as i32) as u32, (y1 + seed_y as i32) as u32);
                if rgb_lum([center_p[0], center_p[1], center_p[2]]) >= 225 {
                    let mut flood = vec![0u8; cw * ch];
                    let mut q = std::collections::VecDeque::new();
                    flood[seed_y * cw + seed_x] = 255;
                    q.push_back((seed_x, seed_y));
                    while let Some((fx, fy)) = q.pop_front() {
                        let nbrs = [
                            (fx.wrapping_sub(1), fy, fx > 0),
                            (fx + 1, fy, fx + 1 < cw),
                            (fx, fy.wrapping_sub(1), fy > 0),
                            (fx, fy + 1, fy + 1 < ch),
                        ];
                        for (nx, ny, ok) in nbrs {
                            if !ok {
                                continue;
                            }
                            let ni = ny * cw + nx;
                            if flood[ni] == 0 {
                                let p = cleaned_img
                                    .get_pixel((x1 + nx as i32) as u32, (y1 + ny as i32) as u32);
                                if rgb_lum([p[0], p[1], p[2]]) >= 205 {
                                    flood[ni] = 255;
                                    q.push_back((nx, ny));
                                }
                            }
                        }
                    }
                    let mut perim_touch = 0usize;
                    for rx in 0..cw {
                        if flood[rx] > 0 {
                            perim_touch += 1;
                        }
                        if flood[(ch - 1) * cw + rx] > 0 {
                            perim_touch += 1;
                        }
                    }
                    for ry in 0..ch {
                        if flood[ry * cw] > 0 {
                            perim_touch += 1;
                        }
                        if flood[ry * cw + (cw - 1)] > 0 {
                            perim_touch += 1;
                        }
                    }
                    let perim_total = 2 * (cw + ch).max(1);
                    if perim_touch * 100 <= perim_total * 24 {
                        // Заменяем грубый полигон на попиксельно точную светлую полость (исключая внешние рамки!)
                        let flood_area = flood.iter().filter(|&&v| v > 0).count();
                        if flood_area * 100 >= (cw * ch) * 28 {
                            raw_mask = flood;
                            flood_succeeded = true;
                        }
                    }
                }

                let mut shrunk = erode_ellipse(&raw_mask, cw, ch, erode_r);
                if !shrunk.iter().any(|&v| v > 0) {
                    shrunk = raw_mask;
                }
                let area = shrunk.iter().filter(|&&v| v > 0).count();
                if (!flood_succeeded && area * 100 < (cw * ch) * 30) || area < 64 {
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

    // Разделяем только РЕАЛЬНО общую полость сдвоенных бабблов без тёмной перегородки между центрами,
    // и режем строго по плоскости шва (по Y при вертикальном соседстве, по X при горизонтальном),
    // чтобы свесившийся сверху чужой баббл (`014.webp #1`) никогда не отпиливал половину ширины у нижнего (`014.webp #4`)!
    if !blk.is_free_text {
        let self_anchor = blk.text_bbox.unwrap_or(blk.bbox);
        let self_cx = (self_anchor[0] + self_anchor[2]) / 2;
        let self_cy = (self_anchor[1] + self_anchor[3]) / 2;
        for (j, other) in all_blocks.iter().enumerate() {
            if j == self_idx || other.is_deleted || other.is_free_text {
                continue;
            }
            if other.translated_text.trim().is_empty() && other.original_text.trim().is_empty() {
                continue;
            }
            let ob = other.text_bbox.unwrap_or(other.bbox);
            let ix1 = ob[0].max(x1);
            let iy1 = ob[1].max(y1);
            let ix2 = ob[2].min(x2);
            let iy2 = ob[3].min(y2);
            if ix2 <= ix1 || iy2 <= iy1 {
                continue;
            }
            let other_cx = (ob[0] + ob[2]) / 2;
            let other_cy = (ob[1] + ob[3]) / 2;
            if has_dark_wall_between(cleaned_img, [self_cx, self_cy], [other_cx, other_cy]) {
                continue;
            }

            let dx = (other_cx - self_cx).abs();
            let dy = (other_cy - self_cy).abs();
            if dy < 14 && dx < 14 {
                continue;
            }

            // Выбираем ось шва по нормированному смещению относительно средних габаритов двух блоков,
            // чтобы высокие соседние колонки со сдвигом по высоте (`015.webp #5/#6`: dx=99, dy=109, но w~130, h~300)
            // разрезались ВЕРТИКАЛЬНЫМ швом `seam_x`, а не рубились пополам по горизонтали!
            let self_w = (self_anchor[2] - self_anchor[0]).max(16) as f32;
            let self_h = (self_anchor[3] - self_anchor[1]).max(16) as f32;
            let other_w = (ob[2] - ob[0]).max(16) as f32;
            let other_h = (ob[3] - ob[1]).max(16) as f32;
            let avg_w = (self_w + other_w) * 0.5;
            let avg_h = (self_h + other_h) * 0.5;
            let norm_dx = (dx as f32) / avg_w;
            let norm_dy = (dy as f32) / avg_h;

            let mut saved_mask = mask.clone();
            if norm_dy >= norm_dx {
                if other_cy > self_cy {
                    let seam_y = ((self_anchor[3] + ob[1]) / 2).clamp(y1, y2);
                    for py in seam_y..y2 {
                        let row = ((py - y1) as usize) * cw;
                        saved_mask[row..row + cw].fill(0);
                    }
                } else {
                    let seam_y = ((ob[3] + self_anchor[1]) / 2).clamp(y1, y2);
                    for py in y1..seam_y {
                        let row = ((py - y1) as usize) * cw;
                        saved_mask[row..row + cw].fill(0);
                    }
                }
            } else if other_cx > self_cx {
                let seam_x = ((self_anchor[2] + ob[0]) / 2).clamp(x1, x2);
                for py in y1..y2 {
                    let row = ((py - y1) as usize) * cw;
                    for px in seam_x..x2 {
                        saved_mask[row + (px - x1) as usize] = 0;
                    }
                }
            } else {
                let seam_x = ((ob[2] + self_anchor[0]) / 2).clamp(x1, x2);
                for py in y1..y2 {
                    let row = ((py - y1) as usize) * cw;
                    for px in x1..seam_x {
                        saved_mask[row + (px - x1) as usize] = 0;
                    }
                }
            }

            if saved_mask.iter().filter(|&&v| v > 0).count() >= 64 {
                mask = saved_mask;
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
        let anchor_fs = compute_page_target_base_fs(
            cleaned_img.width(),
            cleaned_img.height(),
            opts.min_font_size,
            opts.max_font_size,
        );
        let page_area = (iw.max(1) as usize) * (ih.max(1) as usize);
        let mut prepared: Vec<PreparedBlock> = Vec::new();
        let mut harm_inputs: Vec<PageHarmonizeInput> = Vec::new();

        for (idx, blk) in blocks.iter().enumerate() {
            if blk.is_deleted {
                continue;
            }
            let raw_text = if !blk.translated_text.trim().is_empty() {
                blk.translated_text.trim()
            } else {
                continue;
            };
            // Защита при перерисовке кэша: различаем две противоположные ситуации!
            // 1) Родительская обёртка-матрёшка поверх 2+ полноценных бабблов (сумма площадей детей >= 68% от родителя, `018.webp #9`) -> пропускаем родителя!
            // 2) Осколочные узкие столбики внутри ОДНОГО широкого баббла (сумма площадей детей < 68% от родителя, `021.webp #5/#7` внутри `#6`) -> пропускаем осколки, рендерим полный баббл `#6`!
            if !blk.is_free_text {
                let area_self = ((blk.bbox[2] - blk.bbox[0]).max(1) as i64)
                    * ((blk.bbox[3] - blk.bbox[1]).max(1) as i64);
                let mut child_cnt = 0usize;
                let mut child_area_sum = 0i64;
                let mut enclosing_parent_idx: Option<usize> = None;

                for (j, oth) in blocks.iter().enumerate() {
                    if j == idx
                        || oth.is_deleted
                        || oth.is_free_text
                        || oth.translated_text.trim().is_empty()
                    {
                        continue;
                    }
                    let area_o = ((oth.bbox[2] - oth.bbox[0]).max(1) as i64)
                        * ((oth.bbox[3] - oth.bbox[1]).max(1) as i64);
                    let ix1 = blk.bbox[0].max(oth.bbox[0]);
                    let iy1 = blk.bbox[1].max(oth.bbox[1]);
                    let ix2 = blk.bbox[2].min(oth.bbox[2]);
                    let iy2 = blk.bbox[3].min(oth.bbox[3]);
                    let inter = ((ix2 - ix1).max(0) as i64) * ((iy2 - iy1).max(0) as i64);

                    if area_o * 100 < area_self * 86 && inter * 100 >= area_o * 72 {
                        child_cnt += 1;
                        child_area_sum += area_o;
                    }
                    if area_self * 100 < area_o * 60 && inter * 100 >= area_self * 78 {
                        enclosing_parent_idx = Some(j);
                    }
                }
                if child_cnt >= 2 && child_area_sum * 100 >= area_self * 68 {
                    continue;
                }
                if let Some(p_idx) = enclosing_parent_idx {
                    let p_box = blocks[p_idx].bbox;
                    let area_p = ((p_box[2] - p_box[0]).max(1) as i64)
                        * ((p_box[3] - p_box[1]).max(1) as i64);
                    let mut p_child_sum = 0i64;
                    let mut p_child_cnt = 0usize;
                    for (k, cand) in blocks.iter().enumerate() {
                        if k == p_idx
                            || cand.is_deleted
                            || cand.is_free_text
                            || cand.translated_text.trim().is_empty()
                        {
                            continue;
                        }
                        let area_c = ((cand.bbox[2] - cand.bbox[0]).max(1) as i64)
                            * ((cand.bbox[3] - cand.bbox[1]).max(1) as i64);
                        if area_c * 100 >= area_p * 86 {
                            continue;
                        }
                        let ix1 = p_box[0].max(cand.bbox[0]);
                        let iy1 = p_box[1].max(cand.bbox[1]);
                        let ix2 = p_box[2].min(cand.bbox[2]);
                        let iy2 = p_box[3].min(cand.bbox[3]);
                        let inter_c = ((ix2 - ix1).max(0) as i64) * ((iy2 - iy1).max(0) as i64);
                        if inter_c * 100 >= area_c * 72 {
                            p_child_cnt += 1;
                            p_child_sum += area_c;
                        }
                    }
                    // Если родитель не подавлен (его дети покрывают < 68% площади или ребёнок вообще один) — подавляем узкий осколок!
                    if p_child_cnt < 2 || p_child_sum * 100 < area_p * 68 {
                        continue;
                    }
                }
            }

            let (rect, cavity, cw, ch) = build_cavity_mask(cleaned_img, blk, blocks, idx, iw, ih);
            let mut formatted = format_comic_typography(raw_text, opts.uppercase);
            if !blk.is_free_text {
                formatted = compact_scream_for_narrow_bubble(&formatted, cw, ch);
            }
            if formatted.is_empty() {
                continue;
            }
            let role = classify_bubble_role(&formatted, &blk.style_preset, cw, ch, blk.is_free_text);
            let font = self.resolve_block_font(blk, opts, &formatted)?;
            formatted = sanitize_text_for_font(&font, &formatted);
            let allow_emergency =
                blk.overflow_mode.as_deref().unwrap_or(&opts.overflow_mode) != "overflow";

            let cavity_area = cavity.iter().filter(|&&v| v > 0).count();
            // В Проходе 1 сразу ограничиваем верхний поиск разумным потолком роли, а воздух требуем выше anchor_fs
            let pass1_max_fs = match role {
                BubbleTextRole::Dialogue => (anchor_fs + 3).min(opts.max_font_size),
                BubbleTextRole::Shout => opts.max_font_size,
                BubbleTextRole::Aside => anchor_fs.min(opts.max_font_size),
            }
            .max(opts.min_font_size);

            let comfort_ref = match role {
                BubbleTextRole::Shout => Some(anchor_fs + 2),
                _ => Some(anchor_fs),
            };

            let layout = solve_bubble_layout_anchored(
                &formatted,
                &cavity,
                cw,
                ch,
                opts.min_font_size,
                pass1_max_fs,
                blk.font_size,
                comfort_ref,
                allow_emergency,
                |s, fs, sx| measure_text_width_scaled(&font, fs, sx, s),
            );
            let is_fixed = blk.font_size.is_some();
            let char_cnt = formatted.chars().count();
            harm_inputs.push(PageHarmonizeInput {
                raw_fs: layout.font_size,
                chars: char_cnt,
                is_fixed,
                role,
                cavity_area,
                cav_h: ch,
            });
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

        // Проход 2: гармонизация кеглей вокруг базового размера страницы с учётом роли (диалог vs крик vs шёпот)
        let harmonized = harmonize_page_roles(
            &harm_inputs,
            anchor_fs,
            page_area,
            ih.max(1) as usize,
            opts.min_font_size,
            opts.max_font_size,
        );
        for (pos, item) in prepared.iter_mut().enumerate() {
            let target_fs = harmonized[pos];
            if !item.is_fixed && target_fs < item.layout.font_size {
                let blk_ref = &blocks[item.blk_idx];
                let allow_emergency =
                    blk_ref.overflow_mode.as_deref().unwrap_or(&opts.overflow_mode) != "overflow";
                let comfort_ref = match harm_inputs[pos].role {
                    BubbleTextRole::Shout => Some(anchor_fs + 2),
                    _ => Some(anchor_fs),
                };
                // Пересчитываем вёрстку с верхним пределом `target_fs` (а НЕ `fixed_fs`), чтобы сохранялись
                // оптимальное сжатие и центровка по хордам без принудительного выхода за стенки!
                item.layout = solve_bubble_layout_anchored(
                    &item.formatted,
                    &item.cavity,
                    item.cw,
                    item.ch,
                    opts.min_font_size,
                    target_fs,
                    None,
                    comfort_ref,
                    allow_emergency,
                    |s, fs, sx| measure_text_width_scaled(&item.font, fs, sx, s),
                );
            }
            blocks[item.blk_idx].rendered_font_size = Some(item.layout.font_size);
        }

        // Проход 3: защита от вертикального наложения текстовых боксов в соприкасающихся бабблах
        let cavity_y_bounds = |cavity: &[u8], cw: usize, ch: usize| -> (i32, i32) {
            let mut y_min = ch as i32;
            let mut y_max = 0i32;
            for y in 0..ch {
                let row = y * cw;
                if cavity[row..row + cw].iter().any(|&v| v > 0) {
                    y_min = y_min.min(y as i32);
                    y_max = y_max.max((y + 1) as i32);
                }
            }
            if y_min >= y_max {
                (0, ch as i32)
            } else {
                (y_min, y_max)
            }
        };
        for _pass in 0..2 {
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

                    let abs_dx = (cxi - cxj).abs();
                    let ov_x = (wi + wj) - abs_dx;
                    let ov_y = (hi + hj + 4.0) - (cyi - cyj).abs();
                    // Раздвигаем по вертикали только реально накладывающиеся друг над другом текстовые блоки,
                    // не трогая стоящие рядом бок о бок колонки при касании краёв!
                    if ov_x > 6.0 && abs_dx < (wi + wj) * 0.72 && ov_y > 0.0 {
                        let push = ((ov_y * 0.5).ceil() as i32).max(2);
                        let (yi_top, yi_bot) =
                            cavity_y_bounds(&prepared[i].cavity, prepared[i].cw, prepared[i].ch);
                        let (yj_top, yj_bot) =
                            cavity_y_bounds(&prepared[j].cavity, prepared[j].cw, prepared[j].ch);
                        let yi_mid = (yi_top + yi_bot) / 2;
                        let yj_mid = (yj_top + yj_bot) / 2;
                        if cyi <= cyj {
                            let min_i = (yi_top + hi.ceil() as i32 + 2).min(yi_mid);
                            let max_j = (yj_bot - hj.ceil() as i32 - 2).max(yj_mid);
                            prepared[i].layout.center_xy[1] =
                                (prepared[i].layout.center_xy[1] - push).max(min_i);
                            prepared[j].layout.center_xy[1] =
                                (prepared[j].layout.center_xy[1] + push).min(max_j);
                        } else {
                            let max_i = (yi_bot - hi.ceil() as i32 - 2).max(yi_mid);
                            let min_j = (yj_top + hj.ceil() as i32 + 2).min(yj_mid);
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

            let mut stroke_opt = blk
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
            let scale_x = item.layout.scale_x;
            let scaled = item.font.as_scaled(PxScale {
                x: px_size * scale_x,
                y: px_size,
            });
            let ascent = scaled.ascent();
            let line_h = item.layout.line_height;
            let n_lines = item.layout.lines.len();
            let total_h = (n_lines as f32) * line_h;

            let center_page_x = (rx0 + item.layout.center_xy[0]) as f32;
            let top_page_y = (ry0 + item.layout.center_xy[1]) as f32 - total_h * 0.5;

            let mut line_draws: Vec<(f32, f32, &str)> = Vec::with_capacity(n_lines);
            for (k, line) in item.layout.lines.iter().enumerate() {
                let lw = measure_text_width_scaled(&item.font, px_size, scale_x, line);
                let line_cx = rx0 as f32
                    + item
                        .layout
                        .line_centers_x
                        .get(k)
                        .copied()
                        .unwrap_or(item.layout.center_xy[0] as f32);
                let x_left = match blk.align.as_str() {
                    "left" => center_page_x - item.layout.max_line_w * 0.5,
                    "right" => center_page_x + item.layout.max_line_w * 0.5 - lw,
                    _ => line_cx - lw * 0.5,
                };
                let y_baseline = top_page_y + (k as f32) * line_h + ascent;
                line_draws.push((x_left, y_baseline, line.as_str()));
            }

            // Адаптивная защитная обводка: включается автоматически, если хотя бы одна буква
            // касается тёмной стенки баббла, лайнарта или выходит за чистую белую полость.
            if stroke_opt.is_none() && blk.bg_style != "box" && blk.bg_style != "pill" {
                let is_dark_text = rgb_lum(fill_rgb) < 128;
                if check_text_lines_wall_contact(
                    cleaned_img,
                    &item.cavity,
                    item.rect,
                    item.cw,
                    item.ch,
                    &item.font,
                    px_size,
                    scale_x,
                    &line_draws,
                    is_dark_text,
                ) {
                    let halo_rgb = if is_dark_text {
                        if rgb_lum(bg_rgb) >= 220 {
                            bg_rgb
                        } else {
                            [255, 255, 255]
                        }
                    } else {
                        [16, 16, 20]
                    };
                    let halo_w = ((px_size * 0.18).round() as i32).clamp(3, 6);
                    stroke_opt = Some((halo_rgb, halo_w));
                }
            }

            for (x_left, y_baseline, line) in line_draws {
                draw_text_line_scaled(
                    &mut rendered,
                    &item.font,
                    px_size,
                    scale_x,
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
