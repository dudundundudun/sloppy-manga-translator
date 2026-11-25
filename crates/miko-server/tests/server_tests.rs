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
