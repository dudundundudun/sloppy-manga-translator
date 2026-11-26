pub mod engine;
pub mod layout;
pub mod painter;
pub mod typography;

pub use engine::{RenderOptions, RenderPageResult, RenderTimings, RendererError, RustRendererEngine};
pub use layout::{
    compute_scanline_chords, find_optimal_box_center, harmonize_page_font_sizes,
    solve_bubble_layout, solve_bubble_layout_scaled, solve_bubble_layout_with_options,
    BubbleLayout,
};
pub use painter::{
    check_text_lines_wall_contact, draw_text_line, draw_text_line_scaled, measure_text_width,
    measure_text_width_scaled, parse_hex_color, resolve_font_for_text, sanitize_text_for_font,
    FontCache,
};
pub use typography::{
    auto_diamond_wrap_lines, compact_scream_for_narrow_bubble, format_comic_typography,
    hyphenate_word,
};

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontInfo {
    pub id: String,
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
            id: filename.clone(),
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
        assert_eq!(format_comic_typography("What if we have a daughter.?", false), "What if we have a daughter…?");
        assert_eq!(format_comic_typography(".Probably.", false), "…Probably.");
        assert_eq!(format_comic_typography("привет", true), "ПРИВЕТ");
        assert_eq!(
            compact_scream_for_narrow_bubble("Whaaaaaaaaaat?!", 140, 620),
            "Whaaaat?!"
        );

        let ru_parts = hyphenate_word("приключение");
        assert!(ru_parts.len() >= 2 && ru_parts.join("") == "приключение");
        let en_parts = hyphenate_word("unbelievable!");
        assert!(en_parts.len() >= 2 && en_parts.join("") == "unbelievable!");
        let compound = hyphenate_word("matter-of-factly.");
        assert_eq!(compound, vec!["matter-", "of-", "factly."]);
        assert_eq!(hyphenate_word("strength"), vec!["strength"]);

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

    #[test]
    fn test_conjoined_bubbles_with_shared_polygon_never_stack_text() {
        let fonts_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fonts");
        let engine = RustRendererEngine::new(fonts_dir);

        let img = image::RgbImage::from_pixel(240, 600, image::Rgb([255, 255, 255]));
        // Один гигантский общий полигон на все 3 баббла (имитация старого состояния 005.webp)
        let shared_poly = vec![[30, 30], [210, 30], [210, 560], [30, 560]];
        let mut blocks = vec![
            TextBlock {
                id: 1,
                bbox: [40, 30, 200, 190],
                text_bbox: Some([55, 50, 185, 175]),
                bubble_polygon: Some(shared_poly.clone()),
                original_text: "俺ぁ家族の為に".into(),
                translated_text: "I work hard for my family".into(),
                ..Default::default()
            },
            TextBlock {
                id: 2,
                bbox: [40, 185, 145, 285],
                text_bbox: Some([55, 195, 130, 275]),
                bubble_polygon: Some(shared_poly.clone()),
                original_text: "って?".into(),
                translated_text: "Huh?".into(),
                ..Default::default()
            },
            TextBlock {
                id: 3,
                bbox: [35, 280, 205, 560],
                text_bbox: Some([50, 300, 190, 530]),
                bubble_polygon: Some(shared_poly),
                original_text: "うおおおおお?!".into(),
                translated_text: "WHOOOOOOA?!".into(),
                ..Default::default()
            },
        ];

        let res = engine
            .render_page(&img, &mut blocks, &RenderOptions::default())
            .expect("render conjoined");
        assert_eq!(res.blocks_rendered, 3);

        // В каждом из трёх вертикальных диапазонов должен быть свой отрендеренный текст!
        let count_ink_band = |y0: u32, y1: u32| -> usize {
            let mut cnt = 0usize;
            for y in y0..y1 {
                for x in 35..205 {
                    if res.rendered.get_pixel(x, y)[0] < 120 {
                        cnt += 1;
                    }
                }
            }
            cnt
        };
        let top_ink = count_ink_band(40, 180);
        let mid_ink = count_ink_band(190, 280);
        let bot_ink = count_ink_band(300, 550);
        assert!(top_ink > 30, "top bubble missing text: {top_ink}");
        assert!(mid_ink > 15, "mid bubble missing text: {mid_ink}");
        assert!(bot_ink > 30, "bottom bubble missing text: {bot_ink}");
    }

    #[test]
    fn test_overhanging_neighbor_with_wall_does_not_crush_font_size() {
        // Регрессия 014.webp (#1 vs #4): верхний вертикальный крик #1 с чёрной рамкой
        // частично перекрывает правый верхний угол прямоугольника #4, но разделён с ним чёрной стенкой!
        let fonts_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fonts");
        let engine = RustRendererEngine::new(fonts_dir);

        let mut img = image::RgbImage::from_pixel(300, 420, image::Rgb([30, 30, 35]));
        // Нижний баббл #4: [40..150, 180..400] с чисто белой полостью и чёрной рамкой
        for y in 180..400 {
            for x in 40..150 {
                let wall = x <= 42 || x >= 147 || y <= 182 || y >= 397;
                img.put_pixel(
                    x,
                    y,
                    if wall {
                        image::Rgb([10, 10, 10])
                    } else {
                        image::Rgb([255, 255, 255])
                    },
                );
            }
        }
        // Верхний баббл #1: [90..240, 10..260] (срезан чёрной стенкой выше #4 в левой зоне, но его bbox нависает)
        for y in 10..176 {
            for x in 90..240 {
                let wall = x <= 92 || x >= 237 || y <= 12 || y >= 173;
                img.put_pixel(
                    x,
                    y,
                    if wall {
                        image::Rgb([10, 10, 10])
                    } else {
                        image::Rgb([255, 255, 255])
                    },
                );
            }
        }

        let mut blocks = vec![
            TextBlock {
                id: 1,
                bbox: [90, 10, 240, 310],
                text_bbox: Some([90, 20, 240, 300]),
                bubble_polygon: Some(vec![[93, 13], [236, 13], [236, 172], [93, 172]]),
                original_text: "のぉおおおおお!!!".into(),
                translated_text: "Nooooooo!!!".into(),
                ..Default::default()
            },
            TextBlock {
                id: 4,
                bbox: [40, 180, 150, 400],
                text_bbox: Some([44, 195, 120, 390]),
                // Усечённый справа полигон (до x=122 вместо 147), как на 014.webp #4!
                bubble_polygon: Some(vec![[43, 183], [122, 183], [122, 396], [43, 396]]),
                original_text: "この世界の風景は宇宙なんですよ".into(),
                translated_text: "This world's scenery is outer space.".into(),
                ..Default::default()
            },
        ];

        let _ = engine
            .render_page(&img, &mut blocks, &RenderOptions::default())
            .expect("render 014 regression");
        let fs4 = blocks[1].rendered_font_size.unwrap_or(0);
        assert!(
            fs4 >= 17,
            "expected healthy font size >= 17px on 014.webp #4 regression, got {fs4}px"
        );
    }

    #[test]
    fn test_asymmetric_conjoined_oval_no_wall_overflow() {
        let fonts_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fonts");
        let engine = RustRendererEngine::new(fonts_dir);

        let mut img = image::RgbImage::from_pixel(320, 280, image::Rgb([245, 245, 245]));
        // Сдвоенный асимметричный баббл: левый овал [20..150, 50..255], правый овал [135..300, 15..220]
        // с открытым белым перешейком посередине и чёрной стенкой [10,10,10] по периметру.
        let mut wall_pixels: Vec<(u32, u32)> = Vec::new();
        for y in 0..280u32 {
            for x in 0..320u32 {
                let d1 = ((x as f32 - 85.0) / 62.0).powi(2) + ((y as f32 - 152.0) / 100.0).powi(2);
                let d2 = ((x as f32 - 218.0) / 78.0).powi(2) + ((y as f32 - 118.0) / 98.0).powi(2);
                let in_union = d1 <= 1.0 || d2 <= 1.0;
                if in_union {
                    let is_inner = d1 <= 0.90 || d2 <= 0.90;
                    if is_inner {
                        img.put_pixel(x, y, image::Rgb([255, 255, 255]));
                    } else {
                        img.put_pixel(x, y, image::Rgb([10, 10, 10]));
                        wall_pixels.push((x, y));
                    }
                }
            }
        }

        let mut blocks = vec![
            TextBlock {
                id: 1,
                bbox: [20, 50, 155, 255],
                text_bbox: Some([42, 80, 132, 225]),
                bubble_polygon: Some(vec![[20, 50], [155, 50], [155, 255], [20, 255]]),
                original_text: "未来から来たのか知らないが".into(),
                translated_text: "I don't know if you're from the future, but.".into(),
                ..Default::default()
            },
            TextBlock {
                id: 2,
                bbox: [140, 15, 300, 220],
                text_bbox: Some([160, 45, 280, 185]),
                bubble_polygon: Some(vec![[140, 15], [300, 15], [300, 220], [140, 220]]),
                original_text: "目の前に現れたんだ".into(),
                translated_text: "Well, you did appear right in front of me.".into(),
                ..Default::default()
            },
        ];

        let res = engine
            .render_page(&img, &mut blocks, &RenderOptions::default())
            .expect("render asymmetric ovals");
        assert_eq!(res.blocks_rendered, 2);

        // Проверяем, что левая чёрная стенка и верхняя дуга не перетёрты текстом и не получили посторонней обводки
        let mut damaged = 0usize;
        for &(wx, wy) in &wall_pixels {
            let p = res.rendered.get_pixel(wx, wy);
            if p[0] > 40 || p[1] > 40 || p[2] > 40 {
                damaged += 1;
            }
        }
        assert_eq!(
            damaged, 0,
            "expected zero wall pixels damaged by text/halo in asymmetric oval, got {damaged}"
        );
    }

    #[test]
    fn test_adaptive_white_halo_when_text_touches_wall() {
        let fonts_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fonts");
        let engine = RustRendererEngine::new(fonts_dir);

        let mut img = image::RgbImage::from_pixel(140, 140, image::Rgb([15, 15, 15]));
        // Узкий белый баббл [20..120, 25..115] с чёрной стенкой вокруг
        for y in 28..112u32 {
            for x in 24..116u32 {
                img.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }

        // Принудительно ставим слишком крупный кегль 28px, чтобы текст точно задел чёрную стенку по краям!
        let mut blocks = vec![TextBlock {
            id: 1,
            bbox: [20, 25, 120, 115],
            bubble_polygon: Some(vec![[24, 28], [116, 28], [116, 112], [24, 112]]),
            font_size: Some(28),
            original_text: "テスト".into(),
            translated_text: "UNBELIEVABLE!".into(),
            ..Default::default()
        }];

        let res = engine
            .render_page(&img, &mut blocks, &RenderOptions::default())
            .expect("render forced overflow with halo");

        // На чёрной рамке слева/справа от баббла (x < 24 или x >= 116) должны появиться белые пиксели защитного ореола!
        let mut halo_on_wall = 0usize;
        for y in 25..115u32 {
            for x in 0..24u32 {
                if res.rendered.get_pixel(x, y)[0] > 180 {
                    halo_on_wall += 1;
                }
            }
            for x in 116..140u32 {
                if res.rendered.get_pixel(x, y)[0] > 180 {
                    halo_on_wall += 1;
                }
            }
        }
        assert!(
            halo_on_wall >= 15,
            "expected adaptive white halo on dark wall when text overflows, got {halo_on_wall} px"
        );
    }

    #[test]
    fn test_cc_wild_words_em_dash_fallback() {
        let fonts_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fonts");
        let font_path = fonts_dir.join("cc-wild-words-roman.ttf");
        if !font_path.is_file() {
            return;
        }
        let cache = FontCache::new();
        let font = cache.load_font(&font_path).expect("load cc-wild-words");
        let sanitized = sanitize_text_for_font(&font, "Wait—what… ’tis “fine”");
        assert!(!sanitized.contains('—'), "em-dash should fallback: {sanitized}");
        assert!(sanitized.contains("--"), "expected '--' fallback: {sanitized}");
    }
}
