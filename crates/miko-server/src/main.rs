use miko_server::{build_router, AppState};
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8000;
    let mut base_dir = PathBuf::from(".");
    let mut self_check = false;
    let mut verify_onnx = false;
    let mut ocr_image: Option<PathBuf> = None;
    let mut clean_image: Option<PathBuf> = None;
    let mut translate_image: Option<PathBuf> = None;
    let mut translate_folder: Option<PathBuf> = None;
    let mut out_image: Option<PathBuf> = None;
    let mut target_lang: String = "ru".to_string();
    let mut mock_text: Option<String> = None;
    let mut repeat_n: usize = 1;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => {
                if let Some(v) = args.next() {
                    host = v;
                }
            }
            "--port" => {
                if let Some(v) = args.next() {
                    if let Ok(p) = v.parse::<u16>() {
                        port = p;
                    }
                }
            }
            "--base-dir" => {
                if let Some(v) = args.next() {
                    base_dir = PathBuf::from(v);
                }
            }
            "--self-check" => {
                self_check = true;
            }
            "--verify-onnx" => {
                verify_onnx = true;
            }
            "--ocr" => {
                if let Some(v) = args.next() {
                    ocr_image = Some(PathBuf::from(v));
                }
            }
            "--clean" => {
                if let Some(v) = args.next() {
                    clean_image = Some(PathBuf::from(v));
                }
            }
            "--out" => {
                if let Some(v) = args.next() {
                    out_image = Some(PathBuf::from(v));
                }
            }
            "--translate" => {
                if let Some(v) = args.next() {
                    translate_image = Some(PathBuf::from(v));
                }
            }
            "--translate-folder" => {
                if let Some(v) = args.next() {
                    translate_folder = Some(PathBuf::from(v));
                }
            }
            "--target-lang" => {
                if let Some(v) = args.next() {
                    target_lang = v;
                }
            }
            "--mock-text" => {
                if let Some(v) = args.next() {
                    mock_text = Some(v);
                }
            }
            "--repeat" => {
                if let Some(v) = args.next() {
                    if let Ok(n) = v.parse::<usize>() {
                        repeat_n = n.max(1);
                    }
                }
            }
            _ => {}
        }
    }

    let canon_base = std::fs::canonicalize(&base_dir).unwrap_or(base_dir);
    let state = AppState::init(canon_base)?;

    if self_check {
        let ui_exists = state.base_dir.join("web_studio.html").exists();
        let metrics = state.db.metrics()?;
        println!(
            "SELF_CHECK OK | version={} | startup_ms={} | ui_exists={} | fonts={} | onnx_models={} | db_glossary={}",
            miko_server::APP_VERSION_RS,
            state.startup_ms,
            ui_exists,
            state.get_available_fonts().len(),
            state.ort_inventory.models.len(),
            metrics.glossary_rows
        );
        return Ok(());
    }

    if verify_onnx {
        println!(
            "ONNX INVENTORY | dir={} | models={} | cuda_compiled={} | dml_compiled={}",
            state.ort_inventory.onnx_dir,
            state.ort_inventory.models.len(),
            state.ort_inventory.cuda_ep_compiled,
            state.ort_inventory.directml_ep_compiled
        );
        for entry in &state.ort_inventory.models {
            match miko_ort::verify_onnx_model(std::path::Path::new(&entry.path), true) {
                Ok(rep) => {
                    println!(
                        "  [OK] {:24} | ep={:22} | load={} ms | infer={} ms | {} -> {}",
                        rep.model_name,
                        rep.ep,
                        rep.load_ms,
                        rep.infer_ms,
                        rep.inputs_summary,
                        rep.outputs_summary
                    );
                }
                Err(err) => {
                    println!("  [ERR] {:24} | {}", entry.name, err);
                }
            }
        }
        return Ok(());
    }

    if let Some(img_path) = ocr_image {
        let t_load = std::time::Instant::now();
        let mut engine = miko_ocr::RustOcrEngine::load(&state.base_dir, true)?;
        let load_ms = t_load.elapsed().as_millis();
        let opts = miko_ocr::OcrOptions::default();

        let mut last_res = None;
        for run_idx in 1..=repeat_n {
            let res = engine.process_image_path(&img_path, &opts)?;
            println!(
                "RUN {}/{} | ep={} | load={} ms | detect={} ms | seg={} ms | ocr={} ms | total={} ms | blocks={}",
                run_idx,
                repeat_n,
                res.timings.ep,
                load_ms,
                res.timings.detect_ms,
                res.timings.seg_ms,
                res.timings.ocr_ms,
                res.timings.total_ms,
                res.blocks.len()
            );
            last_res = Some(res);
        }
        if let Some(res) = last_res {
            for b in &res.blocks {
                let poly_pts = b.bubble_polygon.as_ref().map(|p| p.len()).unwrap_or(0);
                println!(
                    "  #{} | bbox={:?} | text_bbox={:?} | poly={} pts | conf={:.3} | text={}",
                    b.id, b.bbox, b.text_bbox, poly_pts, b.confidence, b.original_text
                );
            }
        }
        return Ok(());
    }

    if let Some(img_path) = clean_image {
        let t_ocr_load = std::time::Instant::now();
        let mut ocr = miko_ocr::RustOcrEngine::load(&state.base_dir, true)?;
        let ocr_load_ms = t_ocr_load.elapsed().as_millis();

        let t_clean_load = std::time::Instant::now();
        let mut cleaner = miko_cleaner::RustCleanerEngine::load(&state.base_dir, true);
        let clean_load_ms = t_clean_load.elapsed().as_millis();

        let rgb = image::open(&img_path)?.to_rgb8();
        let ocr_res = ocr.process_rgb_image(&rgb, &miko_ocr::OcrOptions::default())?;
        println!(
            "OCR READY | ep={} | load={} ms | ocr_total={} ms | blocks={}",
            ocr_res.timings.ep, ocr_load_ms, ocr_res.timings.total_ms, ocr_res.blocks.len()
        );

        let src_str = img_path.to_string_lossy().to_string();
        let mtime = std::fs::metadata(&img_path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let uncached_opts = miko_cleaner::CleanOptions {
            use_cache: false,
            ..Default::default()
        };
        let mut last_cleaned = None;
        for run_idx in 1..=repeat_n {
            let res = cleaner.clean_image(
                &rgb,
                &ocr_res.blocks,
                &uncached_opts,
                Some(&src_str),
                mtime,
            )?;
            println!(
                "CLEAN RUN {}/{} | ep={} | load={} ms | seg={} ms | mask={} ms | inpaint={} ms | total={} ms | blocks_cleaned={} | mask_px={}",
                run_idx,
                repeat_n,
                res.timings.ep,
                clean_load_ms,
                res.timings.seg_ms,
                res.timings.mask_ms,
                res.timings.inpaint_ms,
                res.timings.total_ms,
                res.blocks_cleaned,
                res.mask_nonzero
            );
            last_cleaned = Some(res.cleaned);
        }

        let cached_opts = miko_cleaner::CleanOptions {
            use_cache: true,
            ..Default::default()
        };
        let write_res = cleaner.clean_image(
            &rgb,
            &ocr_res.blocks,
            &cached_opts,
            Some(&src_str),
            mtime,
        )?;
        cleaner.cache_mut().clear_ram();
        let read_res = cleaner.clean_image(
            &rgb,
            &ocr_res.blocks,
            &cached_opts,
            Some(&src_str),
            mtime,
        )?;
        println!(
            "CACHE ROUNDTRIP | write_ms={} | disk_hit={} | read_ms={}",
            write_res.timings.total_ms, read_res.timings.cache_hit, read_res.timings.total_ms
        );

        if let (Some(op), Some(cleaned_img)) = (out_image, last_cleaned) {
            cleaned_img.save(&op)?;
            println!("SAVED CLEANED -> {}", op.display());
        }
        return Ok(());
    }

    if let Some(img_path) = translate_image {
        let mut ocr = miko_ocr::RustOcrEngine::load(&state.base_dir, true)?;
        let mut cleaner = miko_cleaner::RustCleanerEngine::load(&state.base_dir, true);
        let cfg = state.config.read().expect("cfg").clone();
        let rgb = image::open(&img_path)?.to_rgb8();

        let mut last_rendered = None;
        for run_idx in 1..=repeat_n {
            let t_e2e = std::time::Instant::now();
            let ocr_res = ocr.process_rgb_image(&rgb, &miko_ocr::OcrOptions::default())?;
            let ocr_ms = ocr_res.timings.total_ms;

            let t_tr = std::time::Instant::now();
            let mut blocks = ocr_res.blocks;
            if let Some(ref mock) = mock_text {
                for b in &mut blocks {
                    b.translated_text = mock.clone();
                }
            } else {
                let tr_res = state
                    .translator
                    .translate_blocks(
                        miko_translator::TranslateBatchRequest {
                            blocks,
                            target_lang: target_lang.clone(),
                            model: cfg.selected_model.clone(),
                            cloud_url: cfg.unsloth_url.clone(),
                            cloud_api_key: cfg.unsloth_api_key.clone(),
                            glossary: state.db.load_glossary("global").unwrap_or_default(),
                            ..Default::default()
                        },
                        Some(&state.db),
                    )
                    .await?;
                blocks = tr_res.blocks;
            }
            let translate_ms = t_tr.elapsed().as_millis();

            let clean_res = cleaner.clean_image(
                &rgb,
                &blocks,
                &miko_cleaner::CleanOptions {
                    use_cache: false,
                    ..Default::default()
                },
                None,
                0,
            )?;
            let clean_ms = clean_res.timings.total_ms;

            let render_res = state.renderer.render_page(
                &clean_res.cleaned,
                &mut blocks,
                &miko_renderer::RenderOptions {
                    min_font_size: cfg.min_font_size as i32,
                    max_font_size: cfg.max_font_size as i32,
                    uppercase: cfg.uppercase,
                    ..Default::default()
                },
            )?;
            let render_ms = render_res.timings.total_ms;
            let total_ms = t_e2e.elapsed().as_millis();
            println!(
                "E2E RUN {}/{} | ep_ocr={} | ep_clean={} | ocr={} ms | translate={} ms | clean={} ms | render={} ms | total={} ms | blocks={}",
                run_idx,
                repeat_n,
                ocr_res.timings.ep,
                clean_res.timings.ep,
                ocr_ms,
                translate_ms,
                clean_ms,
                render_ms,
                total_ms,
                blocks.len()
            );
            last_rendered = Some(render_res.rendered);
        }

        if let (Some(op), Some(final_img)) = (out_image, last_rendered) {
            final_img.save(&op)?;
            println!("SAVED E2E RENDERED -> {}", op.display());
        }
        return Ok(());
    }

    if let Some(folder_dir) = translate_folder {
        let source_dir = miko_server::folder_job::unwrap_source_folder(&folder_dir);
        let output_dir = miko_server::folder_job::output_dir_for_folder(&source_dir, &target_lang);
        let images = miko_server::folder_job::collect_folder_images(&source_dir);
        let cfg = state.config.read().expect("cfg").clone();
        let report = miko_server::folder_job::process_chapter_folder_sync(
            state.clone(),
            miko_server::folder_job::BatchFolderOptions {
                source_dir,
                output_dir,
                images,
                target_lang,
                model_override: None,
                detect_free_text: cfg.detect_free_text,
                text_filter_mode: cfg.text_filter_mode,
                skip_existing: false,
                force_retranslate: true,
                uppercase: cfg.uppercase,
                min_font_size: cfg.min_font_size as i32,
                max_font_size: cfg.max_font_size as i32,
                mock_text,
            },
        )
        .await?;
        let avg_ms = if report.processed > 0 {
            report.total_ms / (report.processed as u128)
        } else {
            0
        };
        println!(
            "CHAPTER DONE | status={} | total={} | processed={} | skipped={} | total_ms={} | avg_ms_per_page={} | out={}",
            report.status,
            report.total,
            report.processed,
            report.skipped,
            report.total_ms,
            avg_ms,
            report.output_dir
        );
        return Ok(());
    }

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let app = build_router(state.clone());
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        "Miko-chan Manga Studio (Rust) listening on http://{} (startup {} ms)",
        addr,
        state.startup_ms
    );

    let bg_state = state.clone();
    tokio::spawn(async move {
        let fonts_dir = bg_state.base_dir.join("fonts");
        if !miko_server::fonts::has_cyrillic_custom_font(&fonts_dir) {
            let base_ids: Vec<String> = miko_server::fonts::BASE_CYRILLIC_IDS
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            let rep = miko_server::fonts::download_and_install_fonts(
                &fonts_dir,
                Some(&base_ids),
                false,
            )
            .await;
            bg_state.refresh_fonts();
            tracing::info!(
                "Автозагрузка базовых манга-шрифтов: installed={:?}, skipped={:?}, failed={:?}",
                rep.installed,
                rep.skipped,
                rep.failed
            );
        }
    });

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = state.shutdown_notify.notified() => {},
                _ = tokio::signal::ctrl_c() => {},
            }
        })
        .await?;

    Ok(())
}
