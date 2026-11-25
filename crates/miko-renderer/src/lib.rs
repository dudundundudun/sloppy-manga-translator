pub mod engine;
pub mod layout;
pub mod painter;
pub mod typography;

pub use engine::{RenderOptions, RenderPageResult, RenderTimings, RendererError, RustRendererEngine};
pub use layout::{
    compute_scanline_chords, find_optimal_box_center, harmonize_page_font_sizes,
    solve_bubble_layout, BubbleLayout,
};
pub use painter::{
    draw_text_line, measure_text_width, parse_hex_color, resolve_font_for_text, FontCache,
};
pub use typography::{auto_diamond_wrap_lines, format_comic_typography, hyphenate_word};

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontInfo {
    pub name: String,
    pub filename: String,
    pub path: String,
}

/// Сканирует директорию шрифтов (`fonts/`) без Python/Pillow.
pub fn list_available_fonts(fonts_dir: &Path) -> Vec<FontInfo> {
    let Ok(entries) = std::fs::read_dir(fonts_dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !matches!(ext.as_str(), "ttf" | "otf" | "ttc") {
            continue;
        }
        let filename = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let name = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&filename)
            .to_string();
        out.push(FontInfo {
            name,
            filename,
            path: p.to_string_lossy().to_string(),
        });
    }
    out.sort_by(|a, b| a.filename.cmp(&b.filename));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use miko_core::TextBlock;
    use std::path::PathBuf;

    #[test]
    fn test_typography_and_hyphenation() {
        assert_eq!(format_comic_typography("hey...  i know -- right", false), "hey… I know — right");
        assert_eq!(format_comic_typography("привет", true), "ПРИВЕТ");

        let ru_parts = hyphenate_word("приключение");
        assert!(ru_parts.len() >= 2 && ru_parts.join("") == "приключение");
        let en_parts = hyphenate_word("unbelievable");
        assert!(en_parts.len() >= 2 && en_parts.join("") == "unbelievable");

        let lines = auto_diamond_wrap_lines(
            "Привет друг как твои дела сегодня на улице солнце",
            140.0,
            |s| (s.chars().count() as f32) * 10.0,
        );
        assert!(lines.len() >= 3);
        for l in &lines {
            assert!((l.chars().count() as f32) * 10.0 <= 145.0, "line too wide: {l}");
        }
    }

    #[test]
    fn test_optimal_box_center_and_harmonize() {
        let (w, h) = (60usize, 60usize);
        let mut mask = vec![255u8; w * h];
        // Стенка слева препятствует (0..25), валидная полость справа
        for y in 0..h {
            for x in 0..25 {
                mask[y * w + x] = 0;
            }
        }
        let c = find_optimal_box_center(&mask, w, h, 20, 20);
        assert!(c[0] >= 35, "center should stay inside valid right region: {:?}", c);

        let harmonized = harmonize_page_font_sizes(&[
            (0, 18, 25, false),
            (1, 19, 30, false),
            (2, 42, 3, false), // короткая реплика-выброс 42px должна приглушиться!
        ]);
        assert!(harmonized[2] < 42 && harmonized[2] <= 24, "harmonized={harmonized:?}");
    }

    #[test]
    fn test_render_page_cyrillic_light_and_dark_bubbles() {
        let fonts_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fonts");
        let engine = RustRendererEngine::new(fonts_dir);

        let mut img = image::RgbImage::from_pixel(320, 320, image::Rgb([180, 180, 180]));
        // Белый баббл [20..150, 20..140]
        for y in 20..140 {
            for x in 20..150 {
                img.put_pixel(x, y, image::Rgb([252, 252, 252]));
            }
        }
        // Тёмный баббл [160..300, 160..290]
        for y in 160..290 {
            for x in 160..300 {
                img.put_pixel(x, y, image::Rgb([18, 18, 22]));
            }
        }

        let mut blocks = vec![
            TextBlock {
                id: 1,
                bbox: [20, 20, 150, 140],
                bubble_polygon: Some(vec![[20, 20], [150, 20], [150, 140], [20, 140]]),
                original_text: "こんにちは".into(),
                translated_text: "Привет, как дела!".into(),
                ..Default::default()
            },
            TextBlock {
                id: 2,
                bbox: [160, 160, 300, 290],
                bubble_polygon: Some(vec![[160, 160], [300, 160], [300, 290], [160, 290]]),
                original_text: "何だと！？".into(),
                translated_text: "Что ты сказал?!".into(),
                ..Default::default()
            },
        ];

        let res = engine
            .render_page(&img, &mut blocks, &RenderOptions::default())
            .expect("render_page");
        assert_eq!(res.blocks_rendered, 2);
        assert!(blocks[0].rendered_font_size.unwrap_or(0) >= 12);
        assert!(blocks[1].rendered_font_size.unwrap_or(0) >= 12);

        // В белом баббле появились тёмные чернила кириллицы
        let mut dark_in_white = 0usize;
        for y in 25..135 {
            for x in 25..145 {
                if res.rendered.get_pixel(x, y)[0] < 100 {
                    dark_in_white += 1;
                }
            }
        }
        assert!(dark_in_white > 40, "expected dark text in white bubble, got {dark_in_white}");

        // В тёмном баббле появились светлые чернила кириллицы
        let mut light_in_dark = 0usize;
        for y in 165..285 {
            for x in 165..295 {
                if res.rendered.get_pixel(x, y)[0] > 200 {
                    light_in_dark += 1;
                }
            }
        }
        assert!(light_in_dark > 40, "expected bright text in dark bubble, got {light_in_dark}");
    }
}
