pub mod cache;
pub mod cv;
pub mod disk_cache;
pub mod engine;
pub mod segmenter;

pub use cache::compute_clean_state_hash;
pub use cv::{
    compute_rgb_median_and_std, connected_components, dilate_ellipse, distance_transform_l2,
    draw_circle, draw_polyline, erode_ellipse, fill_polygon, hermite_wall_bridge,
    inpaint_laplace_sor, otsu_threshold, CcComponent, CcResult,
};
pub use disk_cache::CleanDiskCache;
pub use engine::{CleanOptions, CleanPageResult, CleanTimings, CleanerError, RustCleanerEngine};
pub use segmenter::NeuralStrokeSegmenter;

#[cfg(test)]
mod tests {
    use super::*;
    use miko_core::TextBlock;

    #[test]
    fn test_otsu_cc_dt_and_hermite() {
        let mut gray = vec![245u8; 64];
        for y in 2..6 {
            for x in 2..6 {
                gray[y * 8 + x] = 20;
            }
        }
        let thr = otsu_threshold(&gray);
        assert!(thr > 40 && thr < 230, "unexpected otsu threshold: {thr}");

        let bin: Vec<u8> = gray.iter().map(|&p| if p < thr { 255 } else { 0 }).collect();
        let cc = connected_components(&bin, 8, 8);
        assert_eq!(cc.stats.len(), 1);
        assert_eq!(cc.stats[0].area, 16);

        let dt = distance_transform_l2(&bin, 8, 8);
        let center_dt = dt[4 * 8 + 4];
        assert!(center_dt >= 1.9 && center_dt <= 2.5, "center_dt={center_dt}");

        let pts = hermite_wall_bridge([0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [1.0, 0.0], 11);
        assert_eq!(pts.len(), 11);
        assert_eq!(pts[0], [0, 0]);
        assert_eq!(pts[10], [10, 0]);
    }

    #[test]
    fn test_clean_hash_invariant_to_bbox_drag() {
        let b1 = TextBlock {
            id: 1,
            bbox: [20, 20, 100, 100],
            clean_bbox: Some([20, 20, 100, 100]),
            text_bbox: Some([12, 30, 90, 90]),
            bubble_polygon: Some(vec![[20, 20], [100, 20], [100, 100], [20, 100]]),
            original_text: "テスト".into(),
            translated_text: "Тест".into(),
            ..Default::default()
        };
        let b2 = TextBlock {
            bbox: [45, 55, 125, 135],
            bbox_modified: true,
            ..b1.clone()
        };
        let h1 = compute_clean_state_hash(&[b1], "p1.webp", 100, 9, "anime_lama_sota", "ultra");
        let h2 = compute_clean_state_hash(&[b2], "p1.webp", 100, 9, "anime_lama_sota", "ultra");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_polygon_raster_and_euclidean_morphology() {
        let poly = vec![[10, 10], [40, 10], [40, 30], [10, 30]];
        let mask = fill_polygon(&poly, 60, 50);
        assert_eq!(mask[20 * 60 + 25], 255);
        assert_eq!(mask[5 * 60 + 25], 0);

        let eroded = erode_ellipse(&mask, 60, 50, 3);
        assert_eq!(eroded[11 * 60 + 25], 0);
        assert_eq!(eroded[20 * 60 + 25], 255);

        let dilated = dilate_ellipse(&eroded, 60, 50, 3);
        assert!(dilated[12 * 60 + 25] == 255);
        assert_eq!(dilated[2 * 60 + 25], 0);
    }

    #[test]
    fn test_laplace_sor_smooth_gradient() {
        let (w, h) = (40usize, 40usize);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            let v = (140 + y * 2).min(255) as u8;
            for x in 0..w {
                let idx = (y * w + x) * 3;
                rgb[idx] = v;
                rgb[idx + 1] = v;
                rgb[idx + 2] = v;
            }
        }
        let mut mask = vec![0u8; w * h];
        for y in 15..25 {
            for x in 15..25 {
                let i = y * w + x;
                mask[i] = 255;
                rgb[i * 3] = 10;
                rgb[i * 3 + 1] = 10;
                rgb[i * 3 + 2] = 10;
            }
        }
        inpaint_laplace_sor(&mut rgb, &mask, w, h, 200, 1.75);
        for y in 15..25 {
            let expected = (140 + y * 2) as i32;
            for x in 15..25 {
                let got = rgb[(y * w + x) * 3] as i32;
                assert!(
                    (got - expected).abs() <= 4,
                    "gradient mismatch at ({x},{y}): got={got}, expected={expected}"
                );
            }
        }
        // Пиксель вне маски остался нетронутым
        assert_eq!(rgb[(5 * w + 5) * 3], 150);
    }

    #[test]
    fn test_tract_a_mask_dilate_and_colored_ink() {
        let mut img = image::RgbImage::from_pixel(120, 120, image::Rgb([30, 30, 30]));
        // Рисуем белый баббл [20..100, 20..100] с тёмной рамкой 3px и цветным (красным) текстом внутри
        for y in 20..100 {
            for x in 20..100 {
                let on_border = x < 23 || x >= 97 || y < 23 || y >= 97;
                let px = if on_border {
                    image::Rgb([15, 15, 15])
                } else {
                    image::Rgb([250, 250, 250])
                };
                img.put_pixel(x, y, px);
            }
        }
        for y in 52..68 {
            for x in 46..74 {
                if (x + y) % 3 == 0 {
                    img.put_pixel(x, y, image::Rgb([210, 45, 55]));
                }
            }
        }

        let base_block = TextBlock {
            id: 1,
            bbox: [20, 20, 100, 100],
            bubble_polygon: Some(vec![[20, 20], [100, 20], [100, 100], [20, 100]]),
            original_text: "テスト".into(),
            ..Default::default()
        };
        let mut b_small = base_block.clone();
        b_small.mask_dilate = Some(2);
        let mut b_large = base_block.clone();
        b_large.mask_dilate = Some(14);

        let engine = RustCleanerEngine::new_cv_only(std::env::temp_dir().join("miko_t_dil"));
        let opts = CleanOptions {
            use_neural_segmenter: false,
            use_cache: false,
            ..Default::default()
        };
        let m_small = engine.build_text_mask(&img, &[b_small], None, &opts);
        let m_large = engine.build_text_mask(&img, &[b_large], None, &opts);
        let c_small: usize = m_small.iter().filter(|&&v| v > 0).count();
        let c_large: usize = m_large.iter().filter(|&&v| v > 0).count();
        assert!(c_small > 40 && c_large > c_small, "small={c_small} large={c_large}");

        let cleaned = engine.clean_image_with_neural_mask(&img, &[base_block], None, &opts);
        // Красный текст внутри стёрт в белый, рамка баббла [21, 50] сохранена тёмной!
        assert!(cleaned.cleaned.get_pixel(57, 60)[0] > 235);
        assert!(cleaned.cleaned.get_pixel(21, 50)[0] < 40);
    }

    #[test]
    fn test_chibi_art_protection_with_neural_mask() {
        let mut img = image::RgbImage::from_pixel(140, 140, image::Rgb([252, 252, 252]));
        // Большой плотный тёмный чиби-рисунок 26x26 (area=676, max_dt >= 13) и мелкий штрих текста
        for y in 40..66 {
            for x in 35..61 {
                img.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        for y in 85..100 {
            for x in 55..85 {
                if x % 4 == 0 {
                    img.put_pixel(x, y, image::Rgb([20, 20, 20]));
                }
            }
        }

        let mut neural = vec![0u8; 140 * 140];
        // Нейросеть подтверждает только нижнюю область текста, но НЕ чиби-рисунок!
        for y in 85..100 {
            for x in 55..85 {
                neural[y * 140 + x] = 220;
            }
        }

        let blk = TextBlock {
            id: 1,
            bbox: [15, 15, 125, 125],
            bubble_polygon: Some(vec![[15, 15], [125, 15], [125, 125], [15, 125]]),
            original_text: "あ".into(),
            ..Default::default()
        };
        let engine = RustCleanerEngine::new_cv_only(std::env::temp_dir().join("miko_t_chibi"));
        let opts = CleanOptions::default();
        let mask = engine.build_text_mask(&img, &[blk], Some(&neural), &opts);

        assert_eq!(mask[52 * 140 + 48], 0, "chibi blob center must be protected!");
        assert_eq!(mask[92 * 140 + 60], 255, "text stroke must be masked!");
    }

    #[test]
    fn test_zero_bleed_and_hermite_wall_bridge() {
        // Тёмный внешний фон [25, 25, 35], белый баббл [30..90, 30..90] с тёмной стенкой [20, 20, 20],
        // и нависающий чёрный штрих, перерезающий правую стенку x=90 наружу до x=106.
        let mut img = image::RgbImage::from_pixel(120, 120, image::Rgb([25, 25, 35]));
        for y in 30..91 {
            for x in 30..91 {
                let wall = x <= 32 || x >= 88 || y <= 32 || y >= 88;
                img.put_pixel(
                    x,
                    y,
                    if wall {
                        image::Rgb([18, 18, 18])
                    } else {
                        image::Rgb([250, 250, 250])
                    },
                );
            }
        }
        // Нависающий штрих внутри+через стенку+снаружи [75..105, 58..63]
        for y in 58..63 {
            for x in 75..106 {
                img.put_pixel(x, y, image::Rgb([240, 40, 40]));
            }
        }

        let blk = TextBlock {
            id: 1,
            bbox: [30, 30, 110, 90],
            clean_bbox: Some([30, 30, 90, 90]),
            text_bbox: Some([45, 55, 108, 66]),
            bubble_polygon: Some(vec![[30, 30], [90, 30], [90, 90], [30, 90]]),
            original_text: "ドン!".into(),
            ..Default::default()
        };
        let engine = RustCleanerEngine::new_cv_only(std::env::temp_dir().join("miko_t_zb"));
        let opts = CleanOptions {
            use_neural_segmenter: false,
            use_cache: false,
            ..Default::default()
        };
        let res = engine.clean_image_with_neural_mask(&img, &[blk], None, &opts);

        // 1. Внутри зачищено в белый
        assert!(res.cleaned.get_pixel(78, 60)[0] > 230);
        // 2. Снаружи у x=102 фон тёмный (белый не вытек из баббла!)
        let ext = res.cleaned.get_pixel(102, 60);
        assert!(ext[0] < 70 && ext[2] < 80, "white bleed detected outside: {:?}", ext);
        // 3. Разорванная правая стенка x=89..90 замкнута тёмным контуром!
        let wall_mid = res.cleaned.get_pixel(90, 60);
        assert!(wall_mid[0] < 90, "hermite wall bridge did not close wall: {:?}", wall_mid);
    }

    #[test]
    fn test_clean_disk_cache_lossless_webp_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "miko_webp_cache_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let mut cache = CleanDiskCache::new(dir.clone(), 4, 2);
        let mut img = image::RgbImage::new(32, 24);
        for y in 0..24 {
            for x in 0..32 {
                img.put_pixel(x, y, image::Rgb([(x * 7) as u8, (y * 9) as u8, ((x + y) * 3) as u8]));
            }
        }
        cache.put("key_one", &img).expect("put webp 1");
        cache.clear_ram();
        let loaded = cache.get("key_one").expect("get webp 1 from disk");
        assert_eq!(loaded.as_raw(), img.as_raw(), "WebP must be bit-exact lossless!");

        std::thread::sleep(std::time::Duration::from_millis(20));
        cache.put("key_two", &img).expect("put webp 2");
        std::thread::sleep(std::time::Duration::from_millis(20));
        cache.put("key_three", &img).expect("put webp 3 triggers prune max_disk=2");

        let count_webp = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("webp"))
            .count();
        assert_eq!(count_webp, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
