use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use miko_server::{build_router, AppState};
use serde_json::{json, Value};
use std::path::PathBuf;
use tower::ServiceExt;

#[tokio::test]
async fn test_health_glossary_and_security_middleware() {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let state = AppState::init_in_memory(base_dir).expect("init_in_memory");
    let app = build_router(state);

    // 1. /health returns runtime=rust and masks API key
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
    let val: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(val["runtime"], "rust");
    assert_eq!(val["status"], "ready");
    assert_eq!(val["unsloth_api_key"], "");

    // 2. Security middleware blocks untrusted Origin and evil Host
    let evil_origin = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("host", "127.0.0.1:8000")
                .header("origin", "https://evil.example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(evil_origin.status(), StatusCode::OK); // /health is public telemetry

    let evil_priv = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/glossary")
                .header("host", "127.0.0.1:8000")
                .header("origin", "https://evil.example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(evil_priv.status(), StatusCode::FORBIDDEN);

    let evil_host = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("host", "rebind.evil.com:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(evil_host.status(), StatusCode::FORBIDDEN);

    // 3. POST/GET /glossary works via SQLite WAL + rejects UNC path
    let post_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/glossary")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "global_glossary": [{"src": "太郎", "dst": "Таро", "note": "mc"}]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(post_res.status(), StatusCode::OK);
    let post_body = to_bytes(post_res.into_body(), 1024 * 1024).await.unwrap();
    let post_val: Value = serde_json::from_slice(&post_body).unwrap();
    assert_eq!(post_val["global_glossary"].as_array().unwrap().len(), 1);

    let unc_res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/glossary?folder_path=%2F%2Fevil%2Fshare")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unc_res.status(), StatusCode::BAD_REQUEST);

    // 4. POST /clean_rust security & validation checks
    let clean_evil_origin = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/clean_rust")
                .header("host", "127.0.0.1:8000")
                .header("origin", "https://evil.example.com")
                .header("content-type", "application/json")
                .body(Body::from(json!({"image_path": "missing.png"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(clean_evil_origin.status(), StatusCode::FORBIDDEN);

    let clean_unc = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/clean_rust")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"image_path": "\\\\evil\\share\\p.png"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(clean_unc.status(), StatusCode::BAD_REQUEST);

    let clean_missing = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/clean_rust")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"image_path": "definitely_missing_page_xyz.png"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(clean_missing.status(), StatusCode::NOT_FOUND);

    // 5. Phase 4 endpoints: /local_image security, /auto_diamond_wrap, and /save_edited_page
    let li_unc = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/local_image?path=%2F%2Fevil%2Fshare%2Fa.png")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(li_unc.status(), StatusCode::BAD_REQUEST);

    let li_outside = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/local_image?path=C%3A%2FWindows%2Fwin.ini.png")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(li_outside.status(), StatusCode::FORBIDDEN);

    let wrap_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/auto_diamond_wrap")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"text": "Привет мир как дела сегодня солнечно!", "max_width_px": 140.0, "font_size": 18.0})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(wrap_res.status(), StatusCode::OK);
    let wrap_body = to_bytes(wrap_res.into_body(), 1024 * 1024).await.unwrap();
    let wrap_val: Value = serde_json::from_slice(&wrap_body).unwrap();
    assert!(wrap_val["lines"].as_array().unwrap().len() >= 2);

    // End-to-end /save_edited_page inside allowed root
    let tmp_dir = std::env::temp_dir().join(format!("miko_save_edited_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let src_p = tmp_dir.join("src.png");
    let dst_p = tmp_dir.join("out.png");
    let mut img = image::RgbImage::from_pixel(240, 200, image::Rgb([250, 250, 250]));
    for y in 70..130 {
        for x in 60..180 {
            if (x + y) % 5 == 0 {
                img.put_pixel(x, y, image::Rgb([15, 15, 15]));
            }
        }
    }
    img.save(&src_p).unwrap();

    let state_with_tmp = AppState::init_in_memory(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf(),
    )
    .unwrap();
    state_with_tmp.allowed_roots.register_root(&tmp_dir).unwrap();
    let app_tmp = build_router(state_with_tmp.clone());

    let save_res = app_tmp
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/save_edited_page")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "source_path": src_p.to_string_lossy(),
                        "dest_path": dst_p.to_string_lossy(),
                        "blocks": [{
                            "id": 1,
                            "bbox": [30, 30, 210, 170],
                            "bubble_polygon": [[30, 30], [210, 30], [210, 170], [30, 170]],
                            "original_text": "テスト",
                            "translated_text": "Готово!"
                        }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(save_res.status(), StatusCode::OK);
    let save_body = to_bytes(save_res.into_body(), 1024 * 1024).await.unwrap();
    let save_val: Value = serde_json::from_slice(&save_body).unwrap();
    assert_eq!(save_val["blocks_rendered"], 1);
    assert!(dst_p.is_file());
    assert_eq!(state_with_tmp.db.metrics().unwrap().page_state_rows, 1);

    let _ = std::fs::remove_dir_all(&tmp_dir);

    // 6. Phase 5 batch/chapter endpoints: /page_blocks, /folder_find_replace, /folder_report, /chapter_book, /folder_control
    let ch_dir = std::env::temp_dir().join(format!("miko_ch5_test_{}", std::process::id()));
    let ch_out = ch_dir.parent().unwrap().join(format!("{}_RU", ch_dir.file_name().unwrap().to_str().unwrap()));
    let _ = std::fs::remove_dir_all(&ch_dir);
    let _ = std::fs::remove_dir_all(&ch_out);
    std::fs::create_dir_all(&ch_dir).unwrap();
    std::fs::create_dir_all(&ch_out).unwrap();

    let p1 = ch_dir.join("001.png");
    let mut p_img = image::RgbImage::from_pixel(220, 180, image::Rgb([250, 250, 250]));
    for y in 60..120 {
        for x in 50..170 {
            if (x + y) % 5 == 0 {
                p_img.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
    }
    p_img.save(&p1).unwrap();

    let manifest = json!({
        "source_dir": ch_dir.to_string_lossy(),
        "target_lang": "ru",
        "phrase_cache": {},
        "pages": {
            "001.png": {
                "blocks_count": 1,
                "free_text_count": 0,
                "detected_lang": "ja",
                "model": "mock",
                "blocks": [{
                    "id": 1,
                    "bbox": [25, 25, 195, 155],
                    "bubble_polygon": [[25, 25], [195, 25], [195, 155], [25, 155]],
                    "original_text": "こんにちは",
                    "translated_text": "Старое слово!"
                }]
            }
        }
    });
    std::fs::write(ch_out.join(".translation_state.json"), manifest.to_string()).unwrap();

    let state5 = AppState::init_in_memory(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf(),
    )
    .unwrap();
    state5.allowed_roots.register_root(&ch_dir).unwrap();
    state5.allowed_roots.register_root(&ch_out).unwrap();
    let app5 = build_router(state5.clone());

    let pb_res = app5
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/page_blocks")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "source_path": p1.to_string_lossy(),
                        "source_dir": ch_dir.to_string_lossy(),
                        "target_lang": "ru"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(pb_res.status(), StatusCode::OK);
    let pb_val: Value = serde_json::from_slice(&to_bytes(pb_res.into_body(), 1024 * 1024).await.unwrap()).unwrap();
    assert_eq!(pb_val["blocks"][0]["romaji_text"], "konnichiha");

    // Dry-run Find & Replace
    let fr_dry = app5
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/folder_find_replace")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "folder_path": ch_dir.to_string_lossy(),
                        "target_lang": "ru",
                        "find_text": "Старое",
                        "replace_text": "Новое",
                        "dry_run": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(fr_dry.status(), StatusCode::OK);
    let fr_dry_val: Value = serde_json::from_slice(&to_bytes(fr_dry.into_body(), 1024 * 1024).await.unwrap()).unwrap();
    assert_eq!(fr_dry_val["matches_count"], 1);
    assert_eq!(fr_dry_val["rerendered_pages_count"], 0);

    // Apply Find & Replace (re-renders 001.png -> ch_out/001.png)
    let fr_apply = app5
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/folder_find_replace")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "folder_path": ch_dir.to_string_lossy(),
                        "target_lang": "ru",
                        "find_text": "Старое",
                        "replace_text": "Новое",
                        "dry_run": false
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(fr_apply.status(), StatusCode::OK);
    let fr_apply_val: Value = serde_json::from_slice(&to_bytes(fr_apply.into_body(), 1024 * 1024).await.unwrap()).unwrap();
    assert_eq!(fr_apply_val["rerendered_pages_count"], 1);
    assert!(ch_out.join("001.png").is_file());

    let encoded_ch = url::form_urlencoded::byte_serialize(ch_dir.to_string_lossy().as_bytes()).collect::<String>();
    let rep_res = app5
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/folder_report?folder_path={encoded_ch}&target_lang=ru"))
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rep_res.status(), StatusCode::OK);
    let rep_val: Value = serde_json::from_slice(&to_bytes(rep_res.into_body(), 1024 * 1024).await.unwrap()).unwrap();
    assert_eq!(rep_val["translated_pages"], 1);
    assert_eq!(rep_val["total_blocks"], 1);

    let book_res = app5
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/chapter_book?folder_path={encoded_ch}&target_lang=ru"))
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(book_res.status(), StatusCode::OK);
    let book_val: Value = serde_json::from_slice(&to_bytes(book_res.into_body(), 1024 * 1024).await.unwrap()).unwrap();
    assert!(book_val["script_text"].as_str().unwrap().contains("Новое слово!"));

    let ctrl_res = app5
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/folder_control")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(json!({"action": "toggle_cloud_only", "cloud_only": true}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ctrl_res.status(), StatusCode::OK);

    // 7. Full UI contract for /folder_inspect, /open_folder security, and /pick_folder CSRF guard
    let inspect_res = app5
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/folder_inspect")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "folder_path": ch_dir.to_string_lossy(),
                        "target_lang": "ru"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(inspect_res.status(), StatusCode::OK);
    let insp_val: Value =
        serde_json::from_slice(&to_bytes(inspect_res.into_body(), 1024 * 1024).await.unwrap()).unwrap();
    assert_eq!(insp_val["total_pages"], 1);
    assert_eq!(insp_val["translated_pages"], 1);
    assert_eq!(insp_val["pages"][0]["filename"], "001.png");
    assert_eq!(insp_val["pages"][0]["is_translated"], true);
    assert_eq!(insp_val["pages"][0]["has_saved_blocks"], true);
    assert!(insp_val["source_dir"].as_str().is_some());
    assert!(insp_val["output_dir"].as_str().is_some());

    let open_unc = app5
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/open_folder?path=%2F%2Fevil%2Fshare")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(open_unc.status(), StatusCode::BAD_REQUEST);

    let pick_csrf = app5
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/pick_folder?target_lang=en")
                .header("host", "127.0.0.1:8000")
                .header("origin", "https://evil.example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(pick_csrf.status(), StatusCode::FORBIDDEN);

    let fs_res = app5
        .oneshot(
            Request::builder()
                .uri("/folder_status")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(fs_res.status(), StatusCode::OK);
    let fs_val: Value =
        serde_json::from_slice(&to_bytes(fs_res.into_body(), 1024 * 1024).await.unwrap()).unwrap();
    assert!(fs_val.get("current").is_some());
    assert!(fs_val.get("elapsed_sec").is_some());
    assert!(fs_val.get("source_dir").is_some());
    assert!(fs_val.get("live").is_some());

    let _ = std::fs::remove_dir_all(&ch_dir);
    let _ = std::fs::remove_dir_all(&ch_out);
}

#[tokio::test]
async fn test_fonts_catalog_security_and_upload_validation() {
    use base64::Engine;

    let tmp_base = std::env::temp_dir().join(format!(
        "miko_fonts_srv_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    ));
    let _ = std::fs::remove_dir_all(&tmp_base);
    std::fs::create_dir_all(tmp_base.join("fonts")).unwrap();

    let state = AppState::init_in_memory(tmp_base.clone()).expect("init_in_memory");
    let app = build_router(state);

    // 1. GET /fonts/catalog returns recommended catalog
    let cat_res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/fonts/catalog")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cat_res.status(), StatusCode::OK);
    let cat_val: Value =
        serde_json::from_slice(&to_bytes(cat_res.into_body(), 1024 * 1024).await.unwrap()).unwrap();
    assert_eq!(cat_val["status"], "ok");
    assert!(cat_val["catalog"].as_array().unwrap().len() >= 4);

    // 2. Evil origin blocked on /fonts/install and /fonts/upload
    let evil_install = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/fonts/install")
                .header("host", "127.0.0.1:8000")
                .header("origin", "https://evil.example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(evil_install.status(), StatusCode::FORBIDDEN);

    let evil_upload = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/fonts/upload")
                .header("host", "127.0.0.1:8000")
                .header("origin", "https://evil.example.com")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "filename": "x.ttf", "data_base64": "AAAA" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(evil_upload.status(), StatusCode::FORBIDDEN);

    // 3. Path traversal and invalid TTF bytes are rejected with 400
    let bad_path = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/fonts/upload")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "filename": "../evil.ttf", "data_base64": "AAAA" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bad_path.status(), StatusCode::BAD_REQUEST);

    let bad_bytes = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/fonts/upload")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "filename": "not_a_font.ttf",
                        "data_base64": base64::engine::general_purpose::STANDARD.encode(vec![0u8; 2048])
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bad_bytes.status(), StatusCode::BAD_REQUEST);

    // 4. Valid TTF bytes (copy of cc-wild-words-roman.ttf) succeed and update available_fonts
    let real_ttf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fonts")
        .join("cc-wild-words-roman.ttf");
    if real_ttf_path.is_file() {
        let ttf_bytes = std::fs::read(&real_ttf_path).unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&ttf_bytes);
        let ok_upload = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/fonts/upload")
                    .header("host", "127.0.0.1:8000")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "filename": "custom-test.ttf",
                            "data_base64": b64
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ok_upload.status(), StatusCode::OK);
        let up_val: Value =
            serde_json::from_slice(&to_bytes(ok_upload.into_body(), 1024 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(up_val["status"], "ok");
        assert_eq!(up_val["font"]["filename"], "custom-test.ttf");
        assert_eq!(up_val["font_status"]["has_custom_fonts"], true);
    }

    let _ = std::fs::remove_dir_all(&tmp_base);
}

#[tokio::test]
async fn test_settings_persistence_across_restarts_and_deepseek_flash_default() {
    let tmp_base = std::env::temp_dir().join(format!(
        "miko_cfg_persist_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    ));
    let _ = std::fs::remove_dir_all(&tmp_base);
    std::fs::create_dir_all(tmp_base.join("fonts")).unwrap();

    // 1. Fresh AppState::init defaults to unsloth::deepseek-v4-flash
    let st1 = AppState::init(tmp_base.clone()).expect("init st1");
    {
        let cfg1 = st1.config.read().unwrap();
        assert_eq!(cfg1.selected_model, "unsloth::deepseek-v4-flash");
        assert!(cfg1.anti_censorship_prompt);
    }
    let app1 = build_router(st1);

    // 2. POST /settings writes changes to disk config.json
    let res = app1
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/settings")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "selected_model": "unsloth::deepseek-v4-flash",
                        "local_fallback_model_multilang": "ollama::huihui_ai/qwen3.5-abliterated:9b",
                        "uppercase": true,
                        "min_font_size": 14,
                        "max_font_size": 34,
                        "anti_censorship_prompt": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert!(tmp_base.join("config.json").is_file(), "config.json must be written on POST /settings");

    // 3. Simulated app restart reloads saved settings instead of resetting to defaults
    let st2 = AppState::init(tmp_base.clone()).expect("init st2");
    {
        let cfg2 = st2.config.read().unwrap();
        assert!(cfg2.uppercase);
        assert_eq!(cfg2.min_font_size, 14);
        assert_eq!(cfg2.max_font_size, 34);
        assert_eq!(
            cfg2.local_fallback_model_multilang,
            "ollama::huihui_ai/qwen3.5-abliterated:9b"
        );
    }

    let _ = std::fs::remove_dir_all(&tmp_base);
}

#[tokio::test]
async fn test_embedded_ui_and_setup_status_on_clean_portable_dir() {
    let tmp_base = std::env::temp_dir().join(format!(
        "miko_portable_clean_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&tmp_base);
    std::fs::create_dir_all(&tmp_base).unwrap();

    let state = AppState::init_in_memory(tmp_base.clone()).expect("init clean state");
    let app = build_router(state);

    // 1. UI serves embedded web_studio.html even when web_studio.html is absent from tmp_base
    let ui_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ui_res.status(), StatusCode::OK);
    let ui_bytes = axum::body::to_bytes(ui_res.into_body(), 2 * 1024 * 1024)
        .await
        .unwrap();
    let ui_str = String::from_utf8_lossy(&ui_bytes);
    assert!(
        ui_str.contains("rust-setup-modal"),
        "embedded UI must include First-Run Setup Wizard"
    );

    // 2. Embedded fox mascot PNG also serves 200 OK without loose PNG on disk
    let fox_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/mascot_fox.png")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(fox_res.status(), StatusCode::OK);

    // 3. /setup/status reports needs_setup=true on empty models directory and enforces Origin check
    let blocked_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/setup/status")
                .header("host", "127.0.0.1:8000")
                .header("origin", "https://evil.example")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(blocked_res.status(), StatusCode::FORBIDDEN);

    let setup_res = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/setup/status")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(setup_res.status(), StatusCode::OK);
    let setup_bytes = axum::body::to_bytes(setup_res.into_body(), 256 * 1024)
        .await
        .unwrap();
    let setup_json: serde_json::Value = serde_json::from_slice(&setup_bytes).unwrap();
    assert_eq!(setup_json["status"], "ok");
    assert_eq!(setup_json["needs_setup"], true);
    assert_eq!(setup_json["models_ready_count"], 0);
    assert_eq!(setup_json["models_total_count"], 6);

    let _ = std::fs::remove_dir_all(&tmp_base);
}

#[tokio::test]
async fn test_live_monitor_sync_start_and_controls() {
    let tmp_base = std::env::temp_dir().join(format!(
        "miko_livemon_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    ));
    let ch_dir = tmp_base.join("ch_live");
    std::fs::create_dir_all(&ch_dir).unwrap();
    let img = image::RgbImage::from_pixel(32, 32, image::Rgb([255, 255, 255]));
    img.save(ch_dir.join("001.png")).unwrap();

    let state = AppState::init_in_memory(tmp_base.clone()).expect("init_in_memory");
    let _ = state.allowed_roots.register_root(&ch_dir);
    let app = build_router(state.clone());

    // 1. /models returns structured {id, name, installed} objects for web_studio.html
    let mod_res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/models")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mod_res.status(), StatusCode::OK);
    let mod_val: Value =
        serde_json::from_slice(&to_bytes(mod_res.into_body(), 256 * 1024).await.unwrap()).unwrap();
    let models_arr = mod_val["models"].as_array().expect("models array");
    assert!(!models_arr.is_empty());
    assert!(models_arr[0].get("id").and_then(|v| v.as_str()).is_some());
    assert!(models_arr[0].get("name").and_then(|v| v.as_str()).is_some());
    assert_eq!(models_arr[0]["installed"], true);

    // 2. Live Monitor controls (pause_after_batch, abort_request, skip_stage, switch_model) return 200 OK
    for action_body in [
        serde_json::json!({"action": "pause_after_batch"}),
        serde_json::json!({"action": "abort_request"}),
        serde_json::json!({"action": "skip_stage"}),
        serde_json::json!({"action": "switch_model", "model": "unsloth::deepseek-v4-flash"}),
    ] {
        let ctrl_res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/folder_control")
                    .header("host", "127.0.0.1:8000")
                    .header("content-type", "application/json")
                    .body(Body::from(action_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ctrl_res.status(), StatusCode::OK);
    }

    // 3. Immediately upon POST /translate_folder, state.folder_job.status is ALREADY non-idle (no SSE race!)
    let tr_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/translate_folder")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "folder_path": ch_dir.to_string_lossy(),
                        "target_lang": "en",
                        "skip_existing": false
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(tr_res.status(), StatusCode::OK);
    let snap = state.folder_job.read().unwrap().clone();
    assert_ne!(
        snap.status, "idle",
        "folder_job must never still be idle when /translate_folder returns HTTP 200"
    );
    assert!(
        snap.live.get("run_id").and_then(|v| v.as_str()).is_some(),
        "live.run_id must be synchronously populated before HTTP 200"
    );

    let _ = state.folder_stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;

    // 4. Verify /chapter_qa_check and /chapter_qa_autofix + font_scale_pct settings roundtrip
    let set_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/settings")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "font_scale_pct": 115,
                        "default_font": "comic-test.ttf"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(set_res.status(), StatusCode::OK);

    let out_dir = tmp_base.join("ch_live_EN");
    std::fs::create_dir_all(&out_dir).unwrap();
    let big_img = image::RgbImage::from_pixel(400, 500, image::Rgb([255, 255, 255]));
    big_img.save(ch_dir.join("001.png")).unwrap();
    std::fs::write(
        out_dir.join(".translation_state.json"),
        serde_json::json!({
            "pages": {
                "001.png": {
                    "blocks_count": 1,
                    "blocks": [{
                        "id": 1,
                        "bbox": [50, 50, 200, 320],
                        "bubble_polygon": [[55, 55], [195, 55], [195, 315], [55, 315]],
                        "original_text": "テスト",
                        "translated_text": "Mmm.such a.strange.feeling.",
                        "rendered_font_size": 10
                    }]
                }
            },
            "phrase_cache": {}
        })
        .to_string(),
    )
    .unwrap();

    let qa_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/chapter_qa_check")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "folder_path": ch_dir.to_string_lossy(),
                        "target_lang": "en",
                        "tiny_threshold_px": 14
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(qa_res.status(), StatusCode::OK);
    let qa_val: Value =
        serde_json::from_slice(&to_bytes(qa_res.into_body(), 256 * 1024).await.unwrap()).unwrap();
    assert_eq!(qa_val["issues_count"], 1);

    let fix_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/chapter_qa_autofix")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "folder_path": ch_dir.to_string_lossy(),
                        "target_lang": "en",
                        "tiny_threshold_px": 14
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(fix_res.status(), StatusCode::OK);
    let fix_val: Value =
        serde_json::from_slice(&to_bytes(fix_res.into_body(), 256 * 1024).await.unwrap()).unwrap();
    assert_eq!(fix_val["fixed_pages"], 1);

    let qa_after_res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/chapter_qa_check")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "folder_path": ch_dir.to_string_lossy(),
                        "target_lang": "en",
                        "tiny_threshold_px": 14
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let qa_after: Value =
        serde_json::from_slice(&to_bytes(qa_after_res.into_body(), 256 * 1024).await.unwrap())
            .unwrap();
    assert_eq!(
        qa_after["issues_count"], 0,
        "autofix should raise glued-dot 10px bubble above threshold!"
    );

    let _ = std::fs::remove_dir_all(&tmp_base);
    let _ = std::fs::remove_dir_all(out_dir);
}

#[tokio::test]
async fn test_open_folder_path_quotes_file_unwrap_and_security() {
    let tmp_base = std::env::temp_dir().join(format!(
        "miko_open_path_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    ));
    let ext_manga = std::env::temp_dir().join(format!(
        "miko_ext_chapter_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp_base).unwrap();
    std::fs::create_dir_all(&ext_manga).unwrap();
    let img = image::RgbImage::from_pixel(320, 480, image::Rgb([255, 255, 255]));
    let page_path = ext_manga.join("001.png");
    img.save(&page_path).unwrap();

    let state = AppState::init_in_memory(tmp_base.clone()).expect("init");
    let app = build_router(state.clone());

    // 1. Evil origin must be rejected on /open_folder_path
    let evil = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/open_folder_path")
                .header("host", "127.0.0.1:8000")
                .header("origin", "https://evil.example.com")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "folder_path": ext_manga.to_string_lossy() }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(evil.status(), StatusCode::FORBIDDEN);

    // 2. UNC path must be rejected with 400
    let unc = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/open_folder_path")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "folder_path": r"\\192.168.1.50\share\manga" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unc.status(), StatusCode::BAD_REQUEST);

    // 3. Pasting a quoted file path ("C:\...\001.png" from Explorer 'Copy as path') unwraps to parent folder & registers root
    let quoted_file = format!("\"{}\"", page_path.to_string_lossy());
    let ok_res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/open_folder_path")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "folder_path": quoted_file, "target_lang": "en" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ok_res.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(ok_res.into_body(), 256 * 1024).await.unwrap()).unwrap();
    assert_eq!(body["total_pages"], 1);
    assert!(state.allowed_roots.is_within_allowed_roots(&ext_manga));

    let _ = std::fs::remove_dir_all(&tmp_base);
    let _ = std::fs::remove_dir_all(&ext_manga);
}

#[tokio::test]
async fn test_prompt_editor_preview_live_test_and_reset_persistence() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let tmp_base = std::env::temp_dir().join(format!(
        "miko_prompts_srv_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&tmp_base);
    std::fs::create_dir_all(tmp_base.join("fonts")).unwrap();

    // Mock OpenAI-compatible cloud endpoint to verify /test_prompt sends custom prompts
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
    let addr = listener.local_addr().unwrap();
    let captured_body = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let cap_clone = captured_body.clone();
    tokio::spawn(async move {
        if let Ok((mut sock, _)) = listener.accept().await {
            let mut buf = vec![0u8; 16384];
            let n = sock.read(&mut buf).await.unwrap_or(0);
            *cap_clone.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = r#"{"choices":[{"message":{"content":"{\"0\":\"Йоу, Сэмпай, ты серьёзно?!\"}"}}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes()).await;
        }
    });

    let st1 = AppState::init(tmp_base.clone()).expect("init st1");
    let app1 = build_router(st1);

    // 1. Save custom prompts + cloud URL via POST /settings
    let save_res = app1
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/settings")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "unsloth_url": format!("http://{addr}/v1"),
                        "unsloth_api_key": "sk-test-prompt",
                        "selected_model": "unsloth::deepseek-v4-flash",
                        "cloud_only_no_local_fallback": true,
                        "anti_censorship_prompt": true,
                        "anti_censorship_custom_prompt": "[18+ CUSTOM PREAMBLE]",
                        "custom_system_prompt": "Переводи дерзким сленгом на {{LANG}}!",
                        "custom_context_prompt": "Герои — байкеры из Иокогамы.",
                        "prompt_preset_id": "custom"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(save_res.status(), StatusCode::OK);

    // 2. GET /health must expose all saved prompt fields & factory defaults
    let h_res = app1
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("host", "127.0.0.1:8000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let h_val: Value =
        serde_json::from_slice(&to_bytes(h_res.into_body(), 512 * 1024).await.unwrap()).unwrap();
    assert_eq!(h_val["anti_censorship_custom_prompt"], "[18+ CUSTOM PREAMBLE]");
    assert_eq!(
        h_val["custom_system_prompt"],
        "Переводи дерзким сленгом на {{LANG}}!"
    );
    assert_eq!(h_val["custom_context_prompt"], "Герои — байкеры из Иокогамы.");
    assert_eq!(h_val["prompt_preset_id"], "custom");
    assert!(!h_val["default_system_prompt"].as_str().unwrap_or("").is_empty());

    // 3. POST /prompt_preview returns assembled system + user prompts and blocks evil origin
    let evil_prev = app1
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/prompt_preview")
                .header("host", "127.0.0.1:8000")
                .header("origin", "https://evil.example.com")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(evil_prev.status(), StatusCode::FORBIDDEN);

    let prev_res = app1
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/prompt_preview")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(json!({"target_lang": "ru"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(prev_res.status(), StatusCode::OK);
    let prev_val: Value =
        serde_json::from_slice(&to_bytes(prev_res.into_body(), 256 * 1024).await.unwrap()).unwrap();
    let sys_str = prev_val["system_prompt"].as_str().unwrap();
    assert!(sys_str.contains("[18+ CUSTOM PREAMBLE]"));
    assert!(sys_str.contains("Герои — байкеры из Иокогамы."));
    assert!(sys_str.contains("MANDATORY OUTPUT CONTRACT:"));

    // 4. POST /test_prompt executes live call against mock cloud with custom prompts
    let test_res = app1
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/test_prompt")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "text": "先輩、マジで言ってるの！？",
                        "target_lang": "ru"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(test_res.status(), StatusCode::OK);
    let test_val: Value =
        serde_json::from_slice(&to_bytes(test_res.into_body(), 256 * 1024).await.unwrap()).unwrap();
    assert_eq!(test_val["translated_text"], "Йоу, Сэмпай, ты серьёзно?!");
    assert!(captured_body.lock().unwrap().contains("[18+ CUSTOM PREAMBLE]"));

    // 5. Resetting custom prompts back to "" persists cleanly across restart
    let reset_res = app1
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/settings")
                .header("host", "127.0.0.1:8000")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "anti_censorship_custom_prompt": "",
                        "custom_system_prompt": "",
                        "custom_context_prompt": "",
                        "prompt_preset_id": "standard"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reset_res.status(), StatusCode::OK);

    let st2 = AppState::init(tmp_base.clone()).expect("init st2");
    {
        let cfg2 = st2.config.read().unwrap();
        assert!(cfg2.anti_censorship_custom_prompt.is_empty());
        assert!(cfg2.custom_system_prompt.is_empty());
        assert!(cfg2.custom_context_prompt.is_empty());
        assert_eq!(cfg2.prompt_preset_id, "standard");
    }

    let _ = std::fs::remove_dir_all(&tmp_base);
}
