pub mod engine;
pub mod layout;
pub mod painter;
pub mod typography;

pub use engine::{
    RenderOptions, RenderPageResult, RenderTimings, RendererError, RustRendererEngine,
};
pub use layout::{
    classify_bubble_role, compute_page_target_base_fs, compute_scanline_chords,
    find_optimal_box_center, harmonize_page_font_sizes, harmonize_page_roles, solve_bubble_layout,
    solve_bubble_layout_anchored, solve_bubble_layout_scaled, solve_bubble_layout_with_options,
    BubbleLayout, BubbleTextRole, PageHarmonizeInput,
};
pub use painter::{
    check_text_lines_wall_contact, draw_text_line, draw_text_line_scaled, measure_text_width,
    measure_text_width_scaled, parse_hex_color, resolve_font_for_text, sanitize_text_for_font,
    FontCache,
};
pub use typography::{
    auto_diamond_wrap_lines, auto_diamond_wrap_lines_opts, compact_scream_for_narrow_bubble,
    format_comic_typography, hyphenate_word,
};

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontInfo {
    pub id: String,
    pub name: String,
    pub filename: String,
    pub path: String,
    #[serde(default)]
    pub supports_ru: bool,
    #[serde(default)]
    pub supports_en: bool,
}

fn font_sort_rank(filename: &str) -> u8 {
    let low = filename.to_ascii_lowercase();
    if low.contains("wild")
        || low.contains("anime")
        || low.contains("manga")
        || low.contains("chinacat")
    {
        0 // Классические манга-шрифты всегда на первом месте
    } else if matches!(
        low.as_str(),
        "balsamiqsans-bold.ttf"
            | "pangolin-regular.ttf"
            | "neucha.ttf"
            | "russoone-regular.ttf"
            | "comicneue-bold.ttf"
            | "bangers-regular.ttf"
    ) {
        2 // Базовый авто-пак Google Fonts ниже пользовательских шрифтов
    } else {
        1 // Любые пользовательские шрифты, закинутые вручную в fonts/
    }
}

/// Сканирует директорию шрифтов (`fonts/`) без Python/Pillow и проверяет поддержку RU/EN глифов.
pub fn list_available_fonts(fonts_dir: &Path) -> Vec<FontInfo> {
    let Ok(entries) = std::fs::read_dir(fonts_dir) else {
        return Vec::new();
    };
    let cache = FontCache::new();
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
        let supports_ru = cache.font_supports_text(&p, "Привет мир");
        let supports_en = cache.font_supports_text(&p, "Hello World");
        out.push(FontInfo {
            id: filename.clone(),
            name,
            filename,
            path: p.to_string_lossy().to_string(),
            supports_ru,
            supports_en,
        });
    }
    out.sort_by(|a, b| {
        font_sort_rank(&a.filename)
            .cmp(&font_sort_rank(&b.filename))
            .then_with(|| {
                a.filename
                    .to_ascii_lowercase()
                    .cmp(&b.filename.to_ascii_lowercase())
            })
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use miko_core::TextBlock;
    use std::path::PathBuf;

    #[test]
    fn test_typography_and_hyphenation() {
        assert_eq!(
            format_comic_typography("hey...  i know -- right", false),
            "hey… I know — right"
        );
        assert_eq!(
            format_comic_typography("What if we have a daughter.?", false),
            "What if we have a daughter…?"
        );
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
            assert!(
                (l.chars().count() as f32) * 10.0 <= 145.0,
                "line too wide: {l}"
            );
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
        assert!(
            c[0] >= 35,
            "center should stay inside valid right region: {:?}",
            c
        );

        let harmonized = harmonize_page_font_sizes(&[
            (0, 18, 25, false),
            (1, 19, 30, false),
            (2, 42, 3, false), // короткая реплика-выброс 42px должна приглушиться!
        ]);
        assert!(
            harmonized[2] < 42 && harmonized[2] <= 24,
            "harmonized={harmonized:?}"
        );
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
        assert!(
            dark_in_white > 40,
            "expected dark text in white bubble, got {dark_in_white}"
        );

        // В тёмном баббле появились светлые чернила кириллицы
        let mut light_in_dark = 0usize;
        for y in 165..285 {
            for x in 165..295 {
                if res.rendered.get_pixel(x, y)[0] > 200 {
                    light_in_dark += 1;
                }
            }
        }
        assert!(
            light_in_dark > 40,
            "expected bright text in dark bubble, got {light_in_dark}"
        );
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
        assert!(
            !sanitized.contains('—'),
            "em-dash should fallback: {sanitized}"
        );
        assert!(
            sanitized.contains("--"),
            "expected '--' fallback: {sanitized}"
        );
    }

    #[test]
    fn test_tall_flared_scream_bubble_stays_vertically_centered() {
        // Регрессия 012.webp #1 ("BIG BANG!") и 013.webp #1 ("WAAAH!!"):
        // высокий крик слегка шире у самого потолка (115px) и чуть уже посередине (105px) —
        // текст НЕ должен улетать под потолок, оставляя 80% баббла снизу пустыми!
        let (w, h) = (136usize, 380usize);
        let mut mask = vec![0u8; w * h];
        for y in 12..368 {
            let half = if y < 65 { 56usize } else { 51usize };
            let cx = w / 2;
            for x in (cx - half)..(cx + half) {
                mask[y * w + x] = 255;
            }
        }
        let lay = solve_bubble_layout_with_options(
            "WAAAH!!",
            &mask,
            w,
            h,
            12,
            32,
            None,
            true,
            |s, fs, sx| (s.chars().count() as f32) * fs * 0.62 * sx,
        );
        let cy = lay.center_xy[1];
        assert!(
            (130..=250).contains(&cy),
            "scream text jumped out of vertical center: cy={cy} (expected 130..=250 in h={h})"
        );
    }

    #[test]
    fn test_no_midline_hyphens_and_english_digraphs() {
        // Регрессия артефактов BRE-AKFAST, TOGE-THER, THO-USAND, APPE-ARED, BLA-CKOUT, ATTA-CHED, STAN-CES
        for w in [
            "breakfast",
            "thousand",
            "cooperate",
            "appeared",
            "attached?",
        ] {
            let parts = hyphenate_word(w);
            let joined_hy = parts.join("-");
            assert!(
                !joined_hy.contains("bre-ak")
                    && !joined_hy.contains("tho-us")
                    && !joined_hy.contains("coo-pe")
                    && !joined_hy.contains("appe-ar")
                    && !joined_hy.contains("atta-ched"),
                "bad English digraph/silent-ed split for {w}: {parts:?}"
            );
        }
        assert_eq!(hyphenate_word("blackout"), vec!["black", "out"]);

        let (w, h) = (190usize, 250usize);
        let mask = vec![255u8; w * h];
        let samples = [
            "THOUGHT YOU'D COME HERE FOR BREAKFAST FOR ONCE, BUT YOU'RE JUST BRAGGING ABOUT BEING PREGNANT?",
            "I DON'T EVEN REMEMBER THAT TIME WE GOT BLACKOUT DRUNK TOGETHER.",
            "GOT IT?! UNDER NO CIRCUMSTANCES TELL ANYONE ABOUT THE FUTURE. DON'T DO ANYTHING THAT COULD CHANGE HISTORY!",
        ];
        for sample in samples {
            let lay = solve_bubble_layout_with_options(
                sample,
                &mask,
                w,
                h,
                12,
                28,
                None,
                true,
                |s, fs, sx| (s.chars().count() as f32) * fs * 0.58 * sx,
            );
            for line in &lay.lines {
                // Внутри строки (кроме самого последнего символа при реальном переносе на следующую строку)
                // НИКОГДА не должно появляться искусственных дефисов между буквами!
                let body = line.strip_suffix('-').unwrap_or(line);
                assert!(
                    !body.contains('-'),
                    "unexpected mid-line hyphen in line {line:?} (all lines: {:?})",
                    lay.lines
                );
            }
        }
    }

    #[test]
    fn test_staggered_side_by_side_conjoined_scream_bubbles_stay_centered() {
        // Точная регрессия 015.webp (#5 [473, 424, 652, 763] и #6 [374, 495, 504, 868]):
        // два высоких баббла крика стоят БОК О БОК с открытым белым перешейком, но смещены по Y (dy=109 > dx=99).
        // Шов обязан быть вертикальным (seam_x), чтобы оба текста центрировались по всей высоте своих бабблов!
        let fonts_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fonts");
        let engine = RustRendererEngine::new(fonts_dir);

        let mut img = image::RgbImage::from_pixel(320, 500, image::Rgb([30, 30, 35]));
        // Правый баббл [123..302, 24..363], левый баббл [24..154, 95..468], общий белый перешеек без чёрной стены между ними
        for y in 24..363u32 {
            for x in 123..302u32 {
                let wall =
                    x >= 299 || y <= 26 || y >= 360 || (x <= 125 && !(100..=355).contains(&y));
                img.put_pixel(
                    x,
                    y,
                    if wall {
                        image::Rgb([12, 12, 12])
                    } else {
                        image::Rgb([255, 255, 255])
                    },
                );
            }
        }
        for y in 95..468u32 {
            for x in 24..154u32 {
                let wall =
                    x <= 26 || y <= 97 || y >= 465 || (x >= 151 && !(100..=355).contains(&y));
                if img.get_pixel(x, y)[0] != 255 {
                    img.put_pixel(
                        x,
                        y,
                        if wall {
                            image::Rgb([12, 12, 12])
                        } else {
                            image::Rgb([255, 255, 255])
                        },
                    );
                }
            }
        }

        let mut blocks = vec![
            TextBlock {
                id: 5,
                bbox: [123, 24, 302, 363],
                text_bbox: Some([123, 27, 272, 326]),
                bubble_polygon: Some(vec![[125, 27], [298, 27], [298, 359], [125, 359]]),
                original_text: "大事なことは最初に言え!!".into(),
                translated_text: "Could you tell me important stuff like that first?!".into(),
                ..Default::default()
            },
            TextBlock {
                id: 6,
                bbox: [24, 95, 154, 468],
                text_bbox: Some([42, 134, 154, 437]),
                bubble_polygon: Some(vec![[27, 98], [150, 98], [150, 464], [27, 464]]),
                original_text: "あぶなかった死ぬとこだった!".into(),
                translated_text: "That was insanely close! I almost died!".into(),
                ..Default::default()
            },
        ];

        let res = engine
            .render_page(
                &img,
                &mut blocks,
                &RenderOptions {
                    uppercase: true,
                    ..Default::default()
                },
            )
            .expect("render 015 regression");

        // Правый баббл (y=24..363, середина ~193) должен иметь чернила НИЖЕ бывшего ложного среза y=230!
        let mut right_lower_ink = 0usize;
        for y in 215..320u32 {
            for x in 145..295u32 {
                if res.rendered.get_pixel(x, y)[0] < 100 {
                    right_lower_ink += 1;
                }
            }
        }
        // Левый баббл (y=95..468, середина ~281) должен иметь чернила ВЫШЕ бывшего ложного среза y=230!
        let mut left_upper_ink = 0usize;
        for y in 150..240u32 {
            for x in 30..135u32 {
                if res.rendered.get_pixel(x, y)[0] < 100 {
                    left_upper_ink += 1;
                }
            }
        }
        assert!(
            right_lower_ink > 30,
            "right bubble text was pushed to top ceiling (right_lower_ink={right_lower_ink})"
        );
        assert!(
            left_upper_ink > 30,
            "left bubble text was pushed to bottom floor (left_upper_ink={left_upper_ink})"
        );
    }

    #[test]
    fn test_target_anchored_page_consistency_and_shout_hierarchy() {
        // На полноценной странице манги (1168x1650) короткие и длинные диалоги должны держаться
        // в ровном коридоре ~19..22px без раздувания "This one." до 30px, а крик "Whaaaaat?!" — быть заметно крупнее!
        let fonts_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fonts");
        let engine = RustRendererEngine::new(fonts_dir);

        let img = image::RgbImage::from_pixel(1168, 1650, image::Rgb([255, 255, 255]));
        let mut blocks = vec![
            TextBlock {
                id: 1,
                bbox: [100, 100, 260, 280],
                bubble_polygon: Some(vec![[105, 105], [255, 105], [255, 275], [105, 275]]),
                original_text: "これだ".into(),
                translated_text: "This one.".into(),
                ..Default::default()
            },
            TextBlock {
                id: 2,
                bbox: [300, 100, 480, 390],
                bubble_polygon: Some(vec![[305, 105], [475, 105], [475, 385], [305, 385]]),
                original_text: "長いセリフ".into(),
                translated_text: "More precisely, this, the device on my wrist, and my genes."
                    .into(),
                ..Default::default()
            },
            TextBlock {
                id: 3,
                bbox: [520, 100, 670, 325],
                bubble_polygon: Some(vec![[525, 105], [665, 105], [665, 320], [525, 320]]),
                original_text: "ひ孫".into(),
                translated_text: "So what does my great-granddaughter want with me?".into(),
                ..Default::default()
            },
            TextBlock {
                id: 4,
                bbox: [720, 100, 960, 520],
                bubble_polygon: Some(vec![[725, 105], [955, 105], [955, 515], [725, 515]]),
                original_text: "なああああに?!".into(),
                translated_text: "Whaaaaat?!".into(),
                ..Default::default()
            },
        ];

        let opts = RenderOptions {
            min_font_size: 12,
            max_font_size: 32,
            ..Default::default()
        };
        let _ = engine
            .render_page(&img, &mut blocks, &opts)
            .expect("render page consistency");

        let fs_short = blocks[0].rendered_font_size.unwrap();
        let fs_long = blocks[1].rendered_font_size.unwrap();
        let fs_compound = blocks[2].rendered_font_size.unwrap();
        let fs_shout = blocks[3].rendered_font_size.unwrap();

        assert!(
            (18..=22).contains(&fs_short),
            "short dialogue inflated too much: fs_short={fs_short}"
        );
        assert!(
            (18..=22).contains(&fs_long),
            "long dialogue out of base band: fs_long={fs_long}"
        );
        assert!(
            fs_compound >= 17,
            "great-granddaughter compound crushed font size: fs_compound={fs_compound}"
        );
        assert!(
            (fs_short - fs_long).abs() <= 3,
            "dialogue sizes jumped too much: short={fs_short} vs long={fs_long}"
        );
        assert!(
            fs_shout >= fs_short + 3,
            "shout should stand out above normal dialogue: shout={fs_shout} vs short={fs_short}"
        );
    }

    #[test]
    fn test_free_text_on_grey_screentone_renders_black_with_white_halo() {
        let fonts_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fonts");
        let engine = RustRendererEngine::new(fonts_dir);

        // Тёмно-серый скринтон без баббла (скриншот 4)
        let img = image::RgbImage::from_pixel(300, 300, image::Rgb([85, 85, 90]));
        let mut blocks = vec![TextBlock {
            id: 1,
            bbox: [60, 60, 240, 240],
            bubble_polygon: None,
            is_free_text: true,
            original_text: "どうせ未来に帰るんだし…".into(),
            translated_text: "I'm going back to the future anyway…".into(),
            ..Default::default()
        }];

        let res = engine
            .render_page(&img, &mut blocks, &RenderOptions::default())
            .expect("render free_text screentone");
        assert_eq!(res.blocks_rendered, 1);

        let mut dark_glyph_px = 0usize;
        let mut white_halo_px = 0usize;
        for y in 65..235 {
            for x in 65..235 {
                let p = res.rendered.get_pixel(x, y);
                let lum = ((77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8) as u8;
                if lum < 45 {
                    dark_glyph_px += 1;
                } else if lum > 225 {
                    white_halo_px += 1;
                }
            }
        }
        assert!(
            dark_glyph_px >= 40,
            "expected black text pixels (<45 lum), got {dark_glyph_px}"
        );
        assert!(
            white_halo_px >= 80,
            "expected white halo pixels (>225 lum), got {white_halo_px}"
        );
    }

    #[test]
    fn test_glued_dots_apostrophe_hyphenation_and_narrow_sluuurp_rescue() {
        // Проверка всех трёх реальных багов со скриншотов пользователя:
        // 1) "Mmm.such a.strange.feeling." (133x287) — было 10px из-за точек без пробелов
        // 2) "Something's.c-coming." (136x232) — было 10px из-за точек и запрета переноса апострофа
        // 3) "Sluuurp" (66x202) — было 10px в узком вертикальном баббле
        // 4) "THOOM" (225x485) со спидлайнами внутри полигона — было 10px из-за урезания flood-маски
        assert_eq!(
            format_comic_typography("Mmm.such a.strange.feeling.", false),
            "Mmm… such a… strange… feeling."
        );
        assert_eq!(
            format_comic_typography("Something's.c-coming.", false),
            "Something's… c-coming."
        );
        assert!(
            hyphenate_word("Something's").len() >= 2,
            "expected Something's to hyphenate cleanly"
        );
        assert_eq!(
            compact_scream_for_narrow_bubble("Sluuurp", 66, 202),
            "Sluurp"
        );

        let fonts_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("fonts");
        let engine = RustRendererEngine::new(fonts_dir);

        let mut img = image::RgbImage::from_pixel(1280, 1750, image::Rgb([255, 255, 255]));
        // Внутри баббла #4 (THOOM, многоугольник 8 вершин) рисуем тёмные диагональные спидлайны, чтобы flood fill покрывал <50% полигона
        for y in 220..640u32 {
            for x in 730..895u32 {
                if (x + y) % 9 < 3 {
                    img.put_pixel(x, y, image::Rgb([35, 35, 35]));
                }
            }
        }

        let mut blocks = vec![
            TextBlock {
                id: 1,
                bbox: [1030, 100, 1163, 387],
                bubble_polygon: Some(vec![[1032, 105], [1157, 105], [1157, 375], [1032, 375]]),
                original_text: "ん…変な感じ…".into(),
                translated_text: "Mmm.such a.strange.feeling.".into(),
                ..Default::default()
            },
            TextBlock {
                id: 2,
                bbox: [450, 256, 586, 488],
                bubble_polygon: Some(vec![[454, 262], [584, 262], [584, 482], [454, 482]]),
                original_text: "なんか…くる…".into(),
                translated_text: "Something's.c-coming.".into(),
                ..Default::default()
            },
            TextBlock {
                id: 3,
                bbox: [154, 1082, 220, 1284],
                bubble_polygon: Some(vec![[156, 1086], [218, 1086], [218, 1280], [156, 1280]]),
                original_text: "じゅるっ".into(),
                translated_text: "Sluuurp".into(),
                ..Default::default()
            },
            TextBlock {
                id: 4,
                bbox: [700, 181, 925, 666],
                bubble_polygon: Some(vec![
                    [725, 217],
                    [812, 210],
                    [898, 217],
                    [904, 420],
                    [898, 642],
                    [812, 648],
                    [725, 642],
                    [718, 420],
                ]),
                style_preset: "shout".into(),
                original_text: "ドクン".into(),
                translated_text: "THOOM!!".into(),
                ..Default::default()
            },
        ];

        let opts = RenderOptions {
            min_font_size: 12,
            max_font_size: 32,
            font_scale_pct: 100,
            uppercase: true,
            ..Default::default()
        };
        let _ = engine
            .render_page(&img, &mut blocks, &opts)
            .expect("render screenshot regressions");

        let fs1 = blocks[0].rendered_font_size.unwrap_or(0);
        let fs2 = blocks[1].rendered_font_size.unwrap_or(0);
        let fs3 = blocks[2].rendered_font_size.unwrap_or(0);
        let fs4 = blocks[3].rendered_font_size.unwrap_or(0);

        assert!(
            fs1 >= 18,
            "Mmm.such a.strange.feeling. stayed too small: {fs1}px"
        );
        assert!(fs2 >= 18, "Something's.c-coming. stayed too small: {fs2}px");
        assert!(
            fs3 >= 14,
            "Sluuurp in narrow 66x202 bubble stayed too small: {fs3}px"
        );
        assert!(
            fs4 >= 25,
            "THOOM!! in 225x485 burst bubble stayed too small: {fs4}px"
        );
    }

    #[test]
    fn test_cc_wild_words_tilde_and_heart_never_become_bang_or_vanish() {
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
        let sanitized = sanitize_text_for_font(&font, "it~ ♥");
        assert_eq!(sanitized, "it~ ♥");
        assert!(!sanitized.contains('!'));

        let engine = RustRendererEngine::new(fonts_dir);
        let img = image::RgbImage::from_pixel(280, 140, image::Rgb([255, 255, 255]));
        let mut blocks_heart = vec![TextBlock {
            id: 1,
            bbox: [40, 20, 240, 120],
            bubble_polygon: Some(vec![[45, 25], [235, 25], [235, 115], [45, 115]]),
            original_text: "♥".into(),
            translated_text: "♥".into(),
            ..Default::default()
        }];
        let res_h = engine
            .render_page(
                &img,
                &mut blocks_heart,
                &RenderOptions {
                    default_font_path: Some(font_path.clone()),
                    ..Default::default()
                },
            )
            .expect("render procedural heart");
        let mut dark_heart_px = 0usize;
        for y in 20..120u32 {
            for x in 40..240u32 {
                if res_h.rendered.get_pixel(x, y)[0] < 120 {
                    dark_heart_px += 1;
                }
            }
        }
        assert!(
            dark_heart_px >= 15,
            "procedural heart drew too few pixels: {dark_heart_px}"
        );

        let mut blocks_tilde = vec![TextBlock {
            id: 1,
            bbox: [20, 20, 260, 120],
            bubble_polygon: Some(vec![[25, 25], [255, 25], [255, 115], [25, 115]]),
            original_text: "残念ですね〜".into(),
            translated_text: "it~ ♥".into(),
            ..Default::default()
        }];
        let res_t = engine
            .render_page(
                &img,
                &mut blocks_tilde,
                &RenderOptions {
                    default_font_path: Some(font_path),
                    ..Default::default()
                },
            )
            .expect("render tilde and heart");
        let mut right_tail_ink = 0usize;
        for y in 25..115u32 {
            for x in 130..255u32 {
                if res_t.rendered.get_pixel(x, y)[0] < 120 {
                    right_tail_ink += 1;
                }
            }
        }
        assert!(
            right_tail_ink >= 15,
            "tilde/heart right tail ink missing: {right_tail_ink}"
        );
    }

    #[test]
    fn test_dehyphenation_prefers_minor_fontsize_stepdown_over_broken_words() {
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

        for (text, w, h, min_expected_fs) in [
            ("You lecherous monk.", 121usize, 199usize, 16i32),
            ("YOU LECHEROUS MONK.", 121usize, 199usize, 16i32),
            (
                "BUT I AM INTERESTED IN THE ACT OF SEX ITSELF.",
                143usize,
                270usize,
                17i32,
            ),
        ] {
            let mut mask = vec![0u8; w * h];
            let cx = w as f32 * 0.5;
            let cy = h as f32 * 0.5;
            let rx = (w as f32 * 0.45).max(4.0);
            let ry = (h as f32 * 0.45).max(4.0);
            for y in 0..h {
                let dy = (y as f32 + 0.5 - cy) / ry;
                for x in 0..w {
                    let dx = (x as f32 + 0.5 - cx) / rx;
                    if dx * dx + dy * dy <= 1.0 {
                        mask[y * w + x] = 255;
                    }
                }
            }
            let lay = solve_bubble_layout_anchored(
                text,
                &mask,
                w,
                h,
                12,
                23,
                None,
                Some(20),
                true,
                |s, fs, sx| measure_text_width_scaled(&font, fs, sx, s),
            );
            assert!(
                lay.font_size >= min_expected_fs,
                "expected readable font_size >= {min_expected_fs}px for {text:?}, got {}px (lines: {:?})",
                lay.font_size,
                lay.lines
            );
            for line in &lay.lines {
                assert!(
                    !line.ends_with('-'),
                    "expected whole words without soft hyphen break in {text:?}, got line {line:?} (all: {:?})",
                    lay.lines
                );
            }
        }
    }

    #[test]
    fn test_ellipsis_typography_repairs() {
        assert_eq!(
            format_comic_typography("Mmm. It feels. strange.", false),
            "Mmm. It feels… strange."
        );
        assert_eq!(
            format_comic_typography("I don't hate it, but.", false),
            "I don't hate it, but…"
        );
        assert_eq!(
            format_comic_typography("Wait. hold on", false),
            "Wait… hold on"
        );
        assert_eq!(
            format_comic_typography("Mr. Smith went home.", false),
            "Mr. Smith went home."
        );
    }
}
