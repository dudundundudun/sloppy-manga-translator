use ab_glyph::{point, Font, FontVec, PxScale, ScaleFont};
use image::RgbImage;
use miko_cleaner::distance_transform_l2;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Кэш загруженных шрифтов (TTF / OTF / TTC).
pub struct FontCache {
    fonts: Mutex<HashMap<PathBuf, Arc<FontVec>>>,
}

impl Default for FontCache {
    fn default() -> Self {
        Self::new()
    }
}

impl FontCache {
    pub fn new() -> Self {
        Self {
            fonts: Mutex::new(HashMap::new()),
        }
    }

    pub fn load_font(&self, path: &Path) -> Result<Arc<FontVec>, String> {
        {
            let guard = self.fonts.lock().map_err(|_| "font mutex poisoned")?;
            if let Some(f) = guard.get(path) {
                return Ok(f.clone());
            }
        }
        let bytes = std::fs::read(path).map_err(|e| format!("Ошибка чтения шрифта {path:?}: {e}"))?;
        let is_ttc = path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("ttc"));
        let font = if is_ttc {
            FontVec::try_from_vec_and_index(bytes, 0)
        } else {
            FontVec::try_from_vec(bytes)
        }
        .map_err(|e| format!("Ошибка парсинга шрифта {path:?}: {e}"))?;

        let arc = Arc::new(font);
        if let Ok(mut guard) = self.fonts.lock() {
            guard.insert(path.to_path_buf(), arc.clone());
        }
        Ok(arc)
    }

    pub fn font_supports_text(&self, path: &Path, sample: &str) -> bool {
        let Ok(f) = self.load_font(path) else {
            return false;
        };
        let mut tested = 0usize;
        for ch in sample.chars() {
            if ch.is_alphabetic() {
                if f.glyph_id(ch).0 == 0 {
                    return false;
                }
                tested += 1;
                if tested >= 6 {
                    break;
                }
            }
        }
        true
    }
}

/// Подбирает лучший шрифт (из `fonts/` или системных), гарантируя поддержку символов текста (RU/EN).
pub fn resolve_font_for_text(
    fonts_dir: &Path,
    cache: &FontCache,
    preferred: Option<&str>,
    sample_text: &str,
) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(pref) = preferred {
        let trimmed = pref.trim();
        let p = Path::new(trimmed);
        if p.is_file() {
            candidates.push(p.to_path_buf());
        } else if !trimmed.is_empty() {
            let in_dir = fonts_dir.join(trimmed);
            if in_dir.is_file() {
                candidates.push(in_dir);
            } else if let Ok(rd) = std::fs::read_dir(fonts_dir) {
                for entry in rd.flatten() {
                    let fp = entry.path();
                    if !fp.is_file() {
                        continue;
                    }
                    let fname = fp.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    let fstem = fp.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    if fname.eq_ignore_ascii_case(trimmed) || fstem.eq_ignore_ascii_case(trimmed) {
                        candidates.push(fp);
                    }
                }
            }
        }
    }

    let priority_names = [
        "v_CCWildWordsLower.ttf",
        "v_CCWildWords.ttf",
        "CCWildWords.ttf",
        "AnimeAcev05.ttf",
        "animeace2_reg.ttf",
        "MangaTemple.ttf",
        "Chinacat.ttf",
        "cc-wild-words-roman.ttf",
        "BalsamiqSans-Bold.ttf",
        "Pangolin-Regular.ttf",
        "ComicNeue-Bold.ttf",
        "Neucha.ttf",
    ];
    for name in priority_names {
        let p = fonts_dir.join(name);
        if p.is_file() {
            candidates.push(p);
        }
    }

    if let Ok(rd) = std::fs::read_dir(fonts_dir) {
        let mut extra: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|s| s.to_str())
                        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "ttf" | "otf" | "ttc"))
            })
            .collect();
        extra.sort();
        candidates.extend(extra);
    }

    for sys in [
        "C:/Windows/Fonts/trebucbd.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/comicbd.ttf",
        "C:/Windows/Fonts/comic.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ] {
        let p = PathBuf::from(sys);
        if p.is_file() {
            candidates.push(p);
        }
    }

    for c in &candidates {
        if cache.font_supports_text(c, sample_text) {
            return Some(c.clone());
        }
    }
    candidates.into_iter().next()
}

/// Заменяет типографские символы без глифа в выбранном шрифте (`—`, `…`, `’`, `“`) на поддерживаемые аналоги.
pub fn sanitize_text_for_font(font: &FontVec, text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_whitespace() || font.glyph_id(ch).0 != 0 {
            out.push(ch);
            continue;
        }
        match ch {
            '—' => {
                if font.glyph_id('-').0 != 0 {
                    out.push_str("--");
                } else if font.glyph_id('–').0 != 0 {
                    out.push('–');
                } else {
                    out.push(ch);
                }
            }
            '–' => {
                if font.glyph_id('-').0 != 0 {
                    out.push('-');
                } else {
                    out.push(ch);
                }
            }
            '…' => {
                if font.glyph_id('.').0 != 0 {
                    out.push_str("...");
                } else {
                    out.push(ch);
                }
            }
            '’' | '‘' => {
                if font.glyph_id('\'').0 != 0 {
                    out.push('\'');
                } else {
                    out.push(ch);
                }
            }
            '“' | '”' | '«' | '»' => {
                if font.glyph_id('"').0 != 0 {
                    out.push('"');
                } else {
                    out.push(ch);
                }
            }
            _ => out.push(ch),
        }
    }
    out
}

#[inline]
fn rgb_lum_u8(c: [u8; 3]) -> u8 {
    (((77u32 * c[0] as u32) + (150u32 * c[1] as u32) + (29u32 * c[2] as u32)) >> 8) as u8
}

/// Проверяет, касается ли хотя бы один глиф блока (с буфером ~3 px) тёмной стенки баббла,
/// внешнего фона за пределами полости или контрастного рисунка.
#[allow(clippy::too_many_arguments)]
pub fn check_text_lines_wall_contact(
    cleaned_img: &RgbImage,
    cavity: &[u8],
    rect: [i32; 4],
    cw: usize,
    _ch: usize,
    font: &FontVec,
    px_size: f32,
    scale_x: f32,
    lines: &[(f32, f32, &str)],
    is_dark_text: bool,
) -> bool {
    let iw = cleaned_img.width() as i32;
    let ih = cleaned_img.height() as i32;
    let py = px_size.max(1.0);
    let px = (py * scale_x.clamp(0.85, 1.05)).max(1.0);
    let scale = PxScale { x: px, y: py };
    let scaled = font.as_scaled(scale);
    let offsets: [(i32, i32); 9] = [
        (0, 0),
        (-3, 0),
        (3, 0),
        (0, -3),
        (0, 3),
        (-2, -2),
        (2, -2),
        (-2, 2),
        (2, 2),
    ];

    let mut contact_hits = 0usize;

    for &(x_left, y_baseline, text) in lines {
        let mut cursor_x = x_left;
        let mut prev_gid = None;
        for ch_sym in text.chars() {
            let gid = font.glyph_id(ch_sym);
            if let Some(prev) = prev_gid {
                cursor_x += scaled.kern(prev, gid);
            }
            let g = gid.with_scale_and_position(scale, point(cursor_x, y_baseline));
            cursor_x += scaled.h_advance(gid);
            prev_gid = Some(gid);

            if let Some(outlined) = font.outline_glyph(g) {
                let b = outlined.px_bounds();
                let gx0 = b.min.x.floor() as i32;
                let gy0 = b.min.y.floor() as i32;
                let mut hit_found = false;
                outlined.draw(|lx, ly, cov| {
                    if hit_found || cov <= 0.25 {
                        return;
                    }
                    let px_i = gx0 + lx as i32;
                    let py_i = gy0 + ly as i32;
                    for &(dx, dy) in &offsets {
                        let nx = px_i + dx;
                        let ny = py_i + dy;
                        if nx < 0 || nx >= iw || ny < 0 || ny >= ih {
                            continue;
                        }
                        let in_cavity = nx >= rect[0]
                            && nx < rect[2]
                            && ny >= rect[1]
                            && ny < rect[3]
                            && cavity[((ny - rect[1]) as usize) * cw + ((nx - rect[0]) as usize)]
                                > 0;
                        let p = cleaned_img.get_pixel(nx as u32, ny as u32);
                        let lum = rgb_lum_u8([p[0], p[1], p[2]]);
                        let conflict = if is_dark_text {
                            lum < 215 || (!in_cavity && lum < 232)
                        } else {
                            lum > 95
                        };
                        if conflict {
                            contact_hits += 1;
                            if contact_hits >= 3 {
                                hit_found = true;
                                return;
                            }
                        }
                    }
                });
                if contact_hits >= 3 {
                    return true;
                }
            }
        }
    }
    false
}

/// Точное измерение ширины строки шрифтом (advance + kerning).
pub fn measure_text_width(font: &FontVec, px_size: f32, text: &str) -> f32 {
    measure_text_width_scaled(font, px_size, 1.0, text)
}

/// Измерение ширины строки с учётом анаморфного горизонтального коэффициента `scale_x` (`0.88..=1.0`).
pub fn measure_text_width_scaled(font: &FontVec, px_size: f32, scale_x: f32, text: &str) -> f32 {
    let py = px_size.max(1.0);
    let px = (py * scale_x.clamp(0.85, 1.05)).max(1.0);
    let scale = PxScale { x: px, y: py };
    let scaled = font.as_scaled(scale);
    let mut total = 0.0f32;
    let mut prev_gid = None;
    for ch in text.chars() {
        let gid = font.glyph_id(ch);
        if let Some(prev) = prev_gid {
            total += scaled.kern(prev, gid);
        }
        total += scaled.h_advance(gid);
        prev_gid = Some(gid);
    }
    total
}

pub fn parse_hex_color(s: &str) -> Option<[u8; 3]> {
    let hex = s.trim().trim_start_matches('#');
    if hex.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some([r, g, b])
}

#[inline]
fn blend_pixel(img: &mut RgbImage, x: usize, y: usize, color: [u8; 3], alpha: f32) {
    if alpha <= 0.0 || x >= img.width() as usize || y >= img.height() as usize {
        return;
    }
    let a = alpha.clamp(0.0, 1.0);
    let inv = 1.0 - a;
    let px = img.get_pixel_mut(x as u32, y as u32);
    px[0] = (color[0] as f32 * a + px[0] as f32 * inv).round() as u8;
    px[1] = (color[1] as f32 * a + px[1] as f32 * inv).round() as u8;
    px[2] = (color[2] as f32 * a + px[2] as f32 * inv).round() as u8;
}

/// Рисует одну строку текста с антиалиасингом и опциональной обводкой/ореолом (`stroke`).
pub fn draw_text_line(
    img: &mut RgbImage,
    font: &FontVec,
    px_size: f32,
    x_left: f32,
    y_baseline: f32,
    text: &str,
    fill: [u8; 3],
    stroke: Option<([u8; 3], i32)>,
) {
    draw_text_line_scaled(img, font, px_size, 1.0, x_left, y_baseline, text, fill, stroke);
}

/// Рисует строку текста с векторным горизонтальным сжатием `scale_x` (`0.88..=1.0` для узких вертикальных бабблов).
#[allow(clippy::too_many_arguments)]
pub fn draw_text_line_scaled(
    img: &mut RgbImage,
    font: &FontVec,
    px_size: f32,
    scale_x: f32,
    x_left: f32,
    y_baseline: f32,
    text: &str,
    fill: [u8; 3],
    stroke: Option<([u8; 3], i32)>,
) {
    let py = px_size.max(1.0);
    let px = (py * scale_x.clamp(0.85, 1.05)).max(1.0);
    let scale = PxScale { x: px, y: py };
    let scaled = font.as_scaled(scale);
    let stroke_r = stroke.map(|(_, w)| w.clamp(0, 8)).unwrap_or(0) as usize;
    let pad = stroke_r as i32 + 4;

    let mut glyphs = Vec::new();
    let mut cursor_x = x_left;
    let mut prev_gid = None;
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    for ch in text.chars() {
        let gid = font.glyph_id(ch);
        if let Some(prev) = prev_gid {
            cursor_x += scaled.kern(prev, gid);
        }
        let g = gid.with_scale_and_position(scale, point(cursor_x, y_baseline));
        cursor_x += scaled.h_advance(gid);
        prev_gid = Some(gid);

        if let Some(outlined) = font.outline_glyph(g) {
            let b = outlined.px_bounds();
            min_x = min_x.min(b.min.x.floor() as i32);
            min_y = min_y.min(b.min.y.floor() as i32);
            max_x = max_x.max(b.max.x.ceil() as i32);
            max_y = max_y.max(b.max.y.ceil() as i32);
            glyphs.push(outlined);
        }
    }

    if glyphs.is_empty() || min_x >= max_x || min_y >= max_y {
        return;
    }

    let ox = min_x - pad;
    let oy = min_y - pad;
    let bw = ((max_x - min_x) + 2 * pad).max(1) as usize;
    let bh = ((max_y - min_y) + 2 * pad).max(1) as usize;
    let mut alpha_buf = vec![0u8; bw * bh];

    for outlined in glyphs {
        let b = outlined.px_bounds();
        let gx0 = b.min.x.floor() as i32;
        let gy0 = b.min.y.floor() as i32;
        outlined.draw(|lx, ly, cov| {
            let ax = (gx0 + lx as i32 - ox) as usize;
            let ay = (gy0 + ly as i32 - oy) as usize;
            if ax < bw && ay < bh {
                let idx = ay * bw + ax;
                let a = (cov.clamp(0.0, 1.0) * 255.0).round() as u8;
                if a > alpha_buf[idx] {
                    alpha_buf[idx] = a;
                }
            }
        });
    }

    if let Some((stroke_rgb, _)) = stroke {
        if stroke_r > 0 {
            let bin: Vec<u8> = alpha_buf
                .iter()
                .map(|&a| if a >= 72 { 0 } else { 255 })
                .collect();
            let dist = distance_transform_l2(&bin, bw, bh);
            let r = stroke_r as f32;
            for y in 0..bh {
                let py = oy + y as i32;
                if py < 0 || py >= img.height() as i32 {
                    continue;
                }
                for x in 0..bw {
                    let px = ox + x as i32;
                    if px < 0 || px >= img.width() as i32 {
                        continue;
                    }
                    let d = dist[y * bw + x];
                    let core_a = (r + 0.5 - d).clamp(0.0, 1.0);
                    let feather_a = if stroke_r >= 2 && d > r {
                        let over = d - r;
                        0.35 * (-over * over / (2.0 * 1.4 * 1.4)).exp()
                    } else {
                        0.0
                    };
                    let sa = core_a.max(feather_a);
                    if sa > 0.01 {
                        blend_pixel(img, px as usize, py as usize, stroke_rgb, sa);
                    }
                }
            }
        }
    }

    for y in 0..bh {
        let py = oy + y as i32;
        if py < 0 || py >= img.height() as i32 {
            continue;
        }
        for x in 0..bw {
            let px = ox + x as i32;
            if px < 0 || px >= img.width() as i32 {
                continue;
            }
            let a = alpha_buf[y * bw + x];
            if a > 0 {
                blend_pixel(img, px as usize, py as usize, fill, (a as f32) / 255.0);
            }
        }
    }
}
