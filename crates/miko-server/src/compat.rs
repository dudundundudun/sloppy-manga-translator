use crate::{bootstrap, folder_job, AppState};
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use miko_cleaner::{CleanOptions, RustCleanerEngine};
use miko_core::security::reject_unc_path;
use miko_core::TextBlock;
use miko_renderer::RenderOptions;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

pub fn append_log_to_disk(base_dir: &Path, message: &str) {
    let logs_dir = base_dir.join("logs");
    let _ = fs::create_dir_all(&logs_dir);
    let log_path = logs_dir.join("miko_studio.log");
    let ts = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&log_path) {
        let _ = writeln!(f, "[ts={ts}] {message}");
    }
}

pub async fn ux_telemetry_record() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

pub async fn ux_telemetry_report() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "total_events": 0,
        "summary": {}
    }))
}

pub async fn ux_telemetry_clear() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

pub async fn ux_telemetry_open_report(State(state): State<Arc<AppState>>) -> Json<Value> {
    let logs_dir = state.base_dir.join("logs");
    let _ = fs::create_dir_all(&logs_dir);
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("explorer.exe")
            .arg(&logs_dir)
            .spawn();
    }
    Json(json!({"status": "ok"}))
}

pub async fn get_diagnostics_report(State(state): State<Arc<AppState>>) -> Json<Value> {
    let log_path = state.base_dir.join("logs").join("miko_studio.log");
    let tail_lines: Vec<String> = fs::read_to_string(&log_path)
        .unwrap_or_default()
        .lines()
        .rev()
        .take(120)
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    Json(json!({
        "status": "ok",
        "runtime": "rust",
        "log_path": log_path.to_string_lossy(),
        "recent_logs": tail_lines,
        "report_text": tail_lines.join("\n"),
    }))
}

pub async fn download_diagnostics_log(State(state): State<Arc<AppState>>) -> Response {
    let log_path = state.base_dir.join("logs").join("miko_studio.log");
    let content = fs::read_to_string(&log_path).unwrap_or_else(|_| "Лог пока пуст.\n".to_string());
    (
        [
            (header::CONTENT_TYPE, "text/plain; charset=utf-8"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"miko_studio.log\"",
            ),
        ],
        content,
    )
        .into_response()
}

pub async fn open_diagnostics_logs(State(state): State<Arc<AppState>>) -> Json<Value> {
    let logs_dir = state.base_dir.join("logs");
    let _ = fs::create_dir_all(&logs_dir);
    let log_file = logs_dir.join("miko_studio.log");
    if !log_file.exists() {
        let _ = fs::write(&log_file, "Miko-chan Manga Studio (Rust) Log\n");
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("explorer.exe")
            .arg(&logs_dir)
            .spawn();
    }
    Json(json!({
        "status": "ok",
        "logs_dir": logs_dir.to_string_lossy()
    }))
}

pub async fn clear_diagnostics_logs(State(state): State<Arc<AppState>>) -> Json<Value> {
    let log_file = state.base_dir.join("logs").join("miko_studio.log");
    let _ = fs::write(&log_file, "");
    Json(json!({"status": "ok"}))
}

#[derive(Debug, Deserialize)]
pub struct OpenReportQuery {
    pub folder_path: String,
    pub target_lang: Option<String>,
}

pub async fn open_folder_report(
    State(state): State<Arc<AppState>>,
    Query(q): Query<OpenReportQuery>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&q.folder_path), "folder_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let src_dir = folder_job::unwrap_source_folder(Path::new(q.folder_path.trim()));
    if !state.allowed_roots.is_within_allowed_roots(&src_dir) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Папка вне разрешённых директорий"})),
        )
            .into_response();
    }
    let t_lang = q.target_lang.unwrap_or_else(|| "en".to_string());
    let out_dir = folder_job::output_dir_for_folder(&src_dir, &t_lang);
    let _ = fs::create_dir_all(&out_dir);
    let report_path = out_dir.join("TRANSLATION_REPORT.txt");
    if !report_path.is_file() {
        let manifest = folder_job::load_manifest(&out_dir);
        let pages_cnt = manifest["pages"].as_object().map(|m| m.len()).unwrap_or(0);
        let fallback_txt = format!(
            "Miko-chan Manga Studio (Rust) — Сводка главы\nИсходная папка: {}\nПапка перевода: {}\nСохранено переведённых страниц в кэше: {}\n",
            src_dir.display(),
            out_dir.display(),
            pages_cnt
        );
        let _ = fs::write(&report_path, fallback_txt);
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("notepad.exe")
            .arg(&report_path)
            .spawn();
    }
    Json(json!({
        "status": "ok",
        "report_path": report_path.to_string_lossy()
    }))
    .into_response()
}

pub async fn get_model_storage(State(state): State<Arc<AppState>>) -> Json<Value> {
    let inv = state.get_ort_inventory();
    let total_local_mb: f64 = inv.models.iter().map(|m| m.size_mb).sum();
    let local_weights: Vec<Value> = bootstrap::REQUIRED_ONNX_MODELS
        .iter()
        .map(|spec| {
            let p = state.base_dir.join("models").join("onnx").join(spec.filename);
            let installed = p.is_file() && p.metadata().map(|m| m.len()).unwrap_or(0) > 1_000_000;
            json!({
                "id": spec.id,
                "title": spec.title,
                "category": "local_weights",
                "role": "onnx",
                "installed": installed,
                "size_mb": spec.size_mb,
                "size_gb": ((spec.size_mb / 1024.0) * 100.0).round() / 100.0,
                "loaded_in_vram": false,
                "path_str": p.to_string_lossy(),
                "can_redownload": true
            })
        })
        .collect();

    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    let mut ollama_models = Vec::new();
    let mut total_ollama_bytes = 0u64;
    if let Ok(resp) = client.get("http://127.0.0.1:11434/api/tags").send().await {
        if let Ok(val) = resp.json::<Value>().await {
            if let Some(arr) = val.get("models").and_then(|v| v.as_array()) {
                for m in arr {
                    let name = m.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let sz = m.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
                    total_ollama_bytes += sz;
                    let gb = ((sz as f64) / (1024.0 * 1024.0 * 1024.0) * 100.0).round() / 100.0;
                    ollama_models.push(json!({
                        "id": name,
                        "name": name,
                        "size_bytes": sz,
                        "size_gb": gb
                    }));
                }
            }
        }
    }
    let total_ollama_gb =
        ((total_ollama_bytes as f64) / (1024.0 * 1024.0 * 1024.0) * 100.0).round() / 100.0;

    Json(json!({
        "status": "ok",
        "runtime": "rust",
        "ollama_models": ollama_models,
        "local_weights": local_weights,
        "total_ollama_gb": total_ollama_gb,
        "total_local_mb": (total_local_mb * 10.0).round() / 10.0,
    }))
}

#[derive(Debug, Deserialize)]
pub struct DeleteModelPayload {
    pub model_id: String,
    pub category: Option<String>,
}

pub async fn post_delete_model(Json(payload): Json<DeleteModelPayload>) -> Response {
    let clean = payload
        .model_id
        .trim()
        .strip_prefix("ollama::")
        .unwrap_or(payload.model_id.trim());
    let _ = payload.category;
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(15))
        .build()
        .unwrap_or_default();
    let res = client
        .delete("http://127.0.0.1:11434/api/delete")
        .json(&json!({ "name": clean }))
        .send()
        .await;
    match res {
        Ok(r) if r.status().is_success() => {
            Json(json!({"status": "ok", "deleted": clean})).into_response()
        }
        Ok(r) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"detail": format!("Ollama вернула HTTP {}", r.status())})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"detail": format!("Ошибка связи с Ollama: {e}")})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct CleanNoisePayload {
    pub source_dir: String,
    pub target_lang: Option<String>,
    pub remove_free_text: Option<bool>,
    pub remove_empty: Option<bool>,
}

pub async fn folder_clean_noise(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CleanNoisePayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.source_dir), "source_dir") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let src_dir = folder_job::unwrap_source_folder(Path::new(payload.source_dir.trim()));
    if !state.allowed_roots.is_within_allowed_roots(&src_dir) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Папка вне разрешённых директорий"})),
        )
            .into_response();
    }
    let t_lang = payload.target_lang.unwrap_or_else(|| "en".to_string());
    let out_dir = folder_job::output_dir_for_folder(&src_dir, &t_lang);
    let mut manifest = folder_job::load_manifest(&out_dir);
    let rem_free = payload.remove_free_text.unwrap_or(true);
    let rem_empty = payload.remove_empty.unwrap_or(true);
    let mut removed_blocks = 0usize;
    let mut touched_pages = 0usize;

    if let Some(pages) = manifest.get_mut("pages").and_then(|v| v.as_object_mut()) {
        for (_fname, pmeta) in pages.iter_mut() {
            let mut page_changed = false;
            if let Some(arr) = pmeta.get_mut("blocks").and_then(|v| v.as_array_mut()) {
                for b in arr.iter_mut() {
                    let already_del = b.get("is_deleted").and_then(|v| v.as_bool()).unwrap_or(false);
                    if already_del {
                        continue;
                    }
                    let is_free = b.get("is_free_text").and_then(|v| v.as_bool()).unwrap_or(false);
                    let tr = b
                        .get("translated_text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim();
                    if (rem_free && is_free) || (rem_empty && tr.is_empty()) {
                        b["is_deleted"] = json!(true);
                        removed_blocks += 1;
                        page_changed = true;
                    }
                }
            }
            if page_changed {
                touched_pages += 1;
            }
        }
    }
    let _ = folder_job::save_manifest(&out_dir, &manifest);

    Json(json!({
        "status": "ok",
        "removed_blocks": removed_blocks,
        "updated_pages": touched_pages,
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct ShortenBlockPayload {
    pub block_id: i64,
    pub source_path: Option<String>,
    pub dest_path: Option<String>,
    pub target_lang: Option<String>,
    pub uppercase: Option<bool>,
    pub min_font_size: Option<i32>,
    pub max_font_size: Option<i32>,
    pub overflow_mode: Option<String>,
    pub blocks: Vec<TextBlock>,
}

pub async fn shorten_block_text(
    State(state): State<Arc<AppState>>,
    Json(mut payload): Json<ShortenBlockPayload>,
) -> Response {
    let Some(target_idx) = payload.blocks.iter().position(|b| b.id == payload.block_id) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Блок не найден"})),
        )
            .into_response();
    };
    let current_tr = payload.blocks[target_idx].translated_text.trim().to_string();
    if current_tr.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "В выбранном баббле нет текста для сокращения"})),
        )
            .into_response();
    }

    let cfg = state.config.read().unwrap_or_else(|e| e.into_inner()).clone();
    let t_lang = payload.target_lang.unwrap_or_else(|| "ru".to_string());
    let prompt_text = format!(
        "Сократи эту реплику персонажа на 30% короче по длине символов, сохранив смысл и живую речь (верни ТОЛЬКО сокращённую фразу на языке {t_lang}): {current_tr}"
    );
    let shortened = match state
        .translator
        .translate_single_text(
            &prompt_text,
            &t_lang,
            &cfg.selected_model,
            &cfg.unsloth_url,
            &cfg.unsloth_api_key,
            &[],
            cfg.anti_censorship_prompt,
            None,
        )
        .await
    {
        Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => current_tr.clone(),
    };
    payload.blocks[target_idx].translated_text = shortened.clone();

    if let (Some(ref sp), Some(ref dp)) = (&payload.source_path, &payload.dest_path) {
        let src_p = PathBuf::from(sp.trim());
        let dst_p = PathBuf::from(dp.trim());
        if state.allowed_roots.is_within_allowed_roots(&src_p)
            && state.allowed_roots.is_within_allowed_roots(&dst_p)
            && src_p.is_file()
        {
            let st_clone = state.clone();
            let mut blks_clone = payload.blocks.clone();
            let up = payload.uppercase.unwrap_or(cfg.uppercase);
            let min_fs = payload.min_font_size.unwrap_or(cfg.min_font_size as i32);
            let max_fs = payload.max_font_size.unwrap_or(cfg.max_font_size as i32);
            let ovf = payload.overflow_mode.unwrap_or(cfg.overflow_mode);
            let _ = tokio::task::spawn_blocking(move || -> Result<(), String> {
                let rgb = image::open(&src_p).map_err(|e| e.to_string())?.to_rgb8();
                let mut cguard = st_clone.cleaner_engine.lock().unwrap_or_else(|e| e.into_inner());
                if cguard.is_none() {
                    *cguard = Some(RustCleanerEngine::load(&st_clone.base_dir, true));
                }
                let src_s = src_p.to_string_lossy().to_string();
                let cl = cguard
                    .as_mut()
                    .expect("cleaner")
                    .clean_image(&rgb, &blks_clone, &CleanOptions::default(), Some(&src_s), 0)
                    .map_err(|e| e.to_string())?;
                let rr = st_clone
                    .renderer
                    .render_page(
                        &cl.cleaned,
                        &mut blks_clone,
                        &RenderOptions {
                            min_font_size: min_fs,
                            max_font_size: max_fs,
                            uppercase: up,
                            overflow_mode: ovf,
                            default_font_path: None,
                        },
                    )
                    .map_err(|e| e.to_string())?;
                rr.rendered.save(&dst_p).map_err(|e| e.to_string())?;
                Ok(())
            })
            .await;
        }
    }

    Json(json!({
        "status": "ok",
        "block_id": payload.block_id,
        "shortened_text": shortened,
        "blocks": payload.blocks,
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct ChapterBookImportPayload {
    pub folder_path: String,
    pub target_lang: Option<String>,
    pub book_text: Option<String>,
}

pub async fn chapter_book_import(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChapterBookImportPayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.folder_path), "folder_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let src_dir = folder_job::unwrap_source_folder(Path::new(payload.folder_path.trim()));
    if !state.allowed_roots.is_within_allowed_roots(&src_dir) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Папка вне разрешённых директорий"})),
        )
            .into_response();
    }
    let t_lang = payload.target_lang.unwrap_or_else(|| "en".to_string());
    let out_dir = folder_job::output_dir_for_folder(&src_dir, &t_lang);
    let mut manifest = folder_job::load_manifest(&out_dir);
    let text = payload.book_text.unwrap_or_default();

    let mut cur_page = String::new();
    let mut updated_blocks = 0usize;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.starts_with("===") && line.ends_with("===") {
            cur_page = line.trim_matches('=').trim().to_string();
            continue;
        }
        if cur_page.is_empty() || !line.starts_with('[') {
            continue;
        }
        let Some(close_br) = line.find(']') else {
            continue;
        };
        let Ok(bid) = line[1..close_br].trim().parse::<i64>() else {
            continue;
        };
        let rest = line[close_br + 1..].trim();
        let new_tr = rest
            .split_once("||")
            .map(|(_, r)| r.trim())
            .unwrap_or(rest);
        if new_tr.is_empty() {
            continue;
        }
        if let Some(arr) = manifest
            .get_mut("pages")
            .and_then(|p| p.get_mut(&cur_page))
            .and_then(|pm| pm.get_mut("blocks"))
            .and_then(|v| v.as_array_mut())
        {
            for b in arr.iter_mut() {
                if b.get("id").and_then(|v| v.as_i64()) == Some(bid) {
                    b["translated_text"] = json!(new_tr);
                    updated_blocks += 1;
                }
            }
        }
    }
    let _ = folder_job::save_manifest(&out_dir, &manifest);
    Json(json!({
        "status": "ok",
        "updated_blocks": updated_blocks,
    }))
    .into_response()
}

pub async fn chapter_book_proofread(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChapterBookImportPayload>,
) -> Response {
    chapter_book_import(State(state), Json(payload)).await
}
