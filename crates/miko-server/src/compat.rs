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
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": err.to_string()})),
        )
            .into_response();
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
            let p = state
                .base_dir
                .join("models")
                .join("onnx")
                .join(spec.filename);
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

    let cfg = state
        .config
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let (_, fb_bare) = miko_translator::strip_model_prefix(&cfg.local_fallback_model_multilang);
    let fb_configured = if fb_bare.is_empty() || fb_bare.eq_ignore_ascii_case("auto") {
        None
    } else {
        Some(fb_bare.to_string())
    };
    let fb_installed = match fb_configured {
        Some(ref pref) => {
            let short = pref
                .split('/')
                .next_back()
                .unwrap_or(pref)
                .to_ascii_lowercase();
            ollama_models.iter().any(|m| {
                let n = m.get("name").and_then(|v| v.as_str()).unwrap_or("");
                n.eq_ignore_ascii_case(pref)
                    || (!short.is_empty() && n.to_ascii_lowercase().contains(&short))
            })
        }
        None => true,
    };

    Json(json!({
        "status": "ok",
        "runtime": "rust",
        "ollama_models": ollama_models,
        "local_weights": local_weights,
        "total_ollama_gb": total_ollama_gb,
        "total_local_mb": (total_local_mb * 10.0).round() / 10.0,
        "configured_fallback_model": fb_configured,
        "fallback_installed": fb_installed,
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
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": err.to_string()})),
        )
            .into_response();
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
                    let already_del = b
                        .get("is_deleted")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if already_del {
                        continue;
                    }
                    let is_free = b
                        .get("is_free_text")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
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
    pub block_id: Option<i64>,
    pub text: Option<String>,
    #[allow(dead_code)]
    pub original_text: Option<String>,
    pub source_path: Option<String>,
    pub dest_path: Option<String>,
    pub target_lang: Option<String>,
    pub uppercase: Option<bool>,
    pub min_font_size: Option<i32>,
    pub max_font_size: Option<i32>,
    pub overflow_mode: Option<String>,
    #[serde(default)]
    pub blocks: Vec<TextBlock>,
}

pub async fn shorten_block_text(
    State(state): State<Arc<AppState>>,
    Json(mut payload): Json<ShortenBlockPayload>,
) -> Response {
    let target_idx = payload
        .block_id
        .and_then(|bid| payload.blocks.iter().position(|b| b.id == bid));
    let current_tr = if let Some(idx) = target_idx {
        payload.blocks[idx].translated_text.trim().to_string()
    } else {
        payload.text.as_deref().unwrap_or("").trim().to_string()
    };
    if current_tr.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "В выбранном баббле нет текста для сокращения"})),
        )
            .into_response();
    }

    let cfg = state
        .config
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let t_lang = payload.target_lang.unwrap_or_else(|| "ru".to_string());
    let prompt_text = format!(
        "Сократи эту реплику персонажа на 30% короче по длине символов, сохранив смысл и живую речь (верни ТОЛЬКО сокращённую фразу на языке {t_lang}): {current_tr}"
    );
    let shorten_prompt_cfg = miko_translator::PromptConfig {
        anti_censorship: cfg.anti_censorship_prompt,
        anti_censorship_custom_prompt: cfg.anti_censorship_custom_prompt.clone(),
        custom_system_prompt: String::new(),
        custom_context_prompt: cfg.custom_context_prompt.clone(),
        adult_canon_en: cfg.adult_canon_en,
        ..Default::default()
    };
    let shortened = match state
        .translator
        .translate_single_text_with_prompts(
            &prompt_text,
            &t_lang,
            &cfg.selected_model,
            &cfg.local_fallback_model_multilang,
            cfg.cloud_only_no_local_fallback,
            &cfg.unsloth_url,
            &cfg.unsloth_api_key,
            &[],
            &shorten_prompt_cfg,
            false,
            None,
        )
        .await
    {
        Ok((s, _)) if !s.trim().is_empty() => s.trim().to_string(),
        _ => current_tr.clone(),
    };
    if let Some(idx) = target_idx {
        payload.blocks[idx].translated_text = shortened.clone();
    }

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
            let scale_pct = cfg.font_scale_pct;
            let def_font = cfg.resolve_default_font_path(&state.base_dir);
            let ovf = payload.overflow_mode.unwrap_or(cfg.overflow_mode);
            if let Some(updated) =
                tokio::task::spawn_blocking(move || -> Result<Vec<TextBlock>, String> {
                    let rgb = image::open(&src_p).map_err(|e| e.to_string())?.to_rgb8();
                    let mut cguard = st_clone
                        .cleaner_engine
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    if cguard.is_none() {
                        *cguard = Some(RustCleanerEngine::load(&st_clone.base_dir, true));
                    }
                    let src_s = src_p.to_string_lossy().to_string();
                    let cl = cguard
                        .as_mut()
                        .expect("cleaner")
                        .clean_image(
                            &rgb,
                            &blks_clone,
                            &CleanOptions {
                                skip_untranslated: true,
                                ..Default::default()
                            },
                            Some(&src_s),
                            0,
                        )
                        .map_err(|e| e.to_string())?;
                    let rr = st_clone
                        .renderer
                        .render_page(
                            &cl.cleaned,
                            &mut blks_clone,
                            &RenderOptions {
                                min_font_size: min_fs,
                                max_font_size: max_fs,
                                font_scale_pct: scale_pct,
                                uppercase: up,
                                overflow_mode: ovf,
                                default_font_path: def_font,
                            },
                        )
                        .map_err(|e| e.to_string())?;
                    rr.rendered.save(&dst_p).map_err(|e| e.to_string())?;
                    if let (Some(out_dir), Some(fname)) =
                        (dst_p.parent(), src_p.file_name().and_then(|s| s.to_str()))
                    {
                        if folder_job::manifest_path(out_dir).is_file() {
                            let mut man = folder_job::load_manifest(out_dir);
                            if man["pages"][fname].is_object() {
                                man["pages"][fname]["blocks"] = json!(blks_clone);
                                let _ = folder_job::save_manifest(out_dir, &man);
                            }
                        }
                    }
                    Ok::<Vec<TextBlock>, String>(blks_clone)
                })
                .await
                .ok()
                .and_then(|r| r.ok())
            {
                payload.blocks = updated;
            }
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
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": err.to_string()})),
        )
            .into_response();
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
        let new_tr = rest.split_once("||").map(|(_, r)| r.trim()).unwrap_or(rest);
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

#[derive(Debug, Deserialize)]
pub struct ChapterQaPayload {
    pub folder_path: String,
    pub target_lang: Option<String>,
    pub tiny_threshold_px: Option<i32>,
}

fn has_glued_dots(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    for i in 1..chars.len().saturating_sub(1) {
        if chars[i] == '.' && chars[i - 1].is_alphabetic() && chars[i + 1].is_alphabetic() {
            let left_two = i >= 2 && chars[i - 2].is_alphabetic();
            let right_two = i + 2 < chars.len() && chars[i + 2].is_alphabetic();
            if left_two || right_two {
                return true;
            }
        }
    }
    false
}

pub async fn chapter_qa_check(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChapterQaPayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.folder_path), "folder_path") {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": err.to_string()})),
        )
            .into_response();
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
    let images = folder_job::collect_folder_images(&src_dir);
    let manifest = folder_job::load_manifest(&out_dir);
    let threshold = payload.tiny_threshold_px.unwrap_or(14).clamp(9, 26);

    let cfg = state
        .config
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let is_en_canon = t_lang.trim().eq_ignore_ascii_case("en") && cfg.adult_canon_en;
    let qa_glossary = if is_en_canon {
        let global_g = state
            .db
            .load_glossary("global")
            .unwrap_or_else(|_| cfg.glossary.clone());
        let folder_g = state
            .db
            .load_glossary(&format!(
                "folder:{}",
                src_dir.to_string_lossy().trim().to_ascii_lowercase()
            ))
            .unwrap_or_default();
        let user_g = miko_translator::merge_glossaries(&global_g, &folder_g);
        let auto_lock: Vec<miko_core::GlossaryEntry> = manifest
            .get("name_lock")
            .cloned()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        miko_translator::merge_glossaries(&auto_lock, &user_g)
    } else {
        Vec::new()
    };
    let canon_db_opt = if is_en_canon {
        let ov = state.db.load_canon_overrides("en").unwrap_or_default();
        Some(miko_translator::merge_overrides(
            miko_translator::builtin_canon_en(),
            &ov,
        ))
    } else {
        None
    };

    let mut all_fs: Vec<i32> = Vec::new();
    let mut issues: Vec<Value> = Vec::new();
    let mut translated_pages = 0usize;

    for (idx, img_path) in images.iter().enumerate() {
        let fname = img_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let Some(raw_arr) = manifest["pages"][&fname]["blocks"].as_array() else {
            continue;
        };
        let Ok(blocks) = serde_json::from_value::<Vec<TextBlock>>(Value::Array(raw_arr.clone()))
        else {
            continue;
        };
        translated_pages += 1;

        let mut page_fs_list: Vec<i32> = blocks
            .iter()
            .filter(|b| !b.is_deleted && !b.translated_text.trim().is_empty())
            .filter_map(|b| b.font_size.or(b.rendered_font_size))
            .collect();
        page_fs_list.sort_unstable();
        let page_med = if page_fs_list.is_empty() {
            20
        } else {
            page_fs_list[page_fs_list.len() / 2]
        };

        for b in &blocks {
            if b.is_deleted {
                continue;
            }
            let tr = b.translated_text.trim();
            if tr.is_empty() {
                continue;
            }
            let fs = b.font_size.or(b.rendered_font_size).unwrap_or(page_med);
            all_fs.push(fs);

            let bw = (b.bbox[2] - b.bbox[0]).max(1);
            let bh = (b.bbox[3] - b.bbox[1]).max(1);
            let area = (bw as usize) * (bh as usize);
            let chars = tr.chars().count();
            let glued = has_glued_dots(tr);
            let is_decent_bubble = bw >= 52 && bh >= 80 && area >= 5_200;
            let is_tiny = fs <= threshold && is_decent_bubble;
            let is_severe_drop = fs <= (page_med - 6).min(15) && bw >= 80 && bh >= 110;
            let is_crowded = fs <= threshold + 1 && chars > ((area / 210).max(24));

            if is_tiny || is_severe_drop || glued {
                let rec_fs = (page_med - 2).clamp((threshold + 2).max(15), 22);
                let (kind, reason) = if glued {
                    (
                        "glued_dots",
                        format!("Склеенные точки без пробела ({fs} px в баббле {bw}×{bh} px)"),
                    )
                } else if is_crowded {
                    (
                        "crowded_bubble",
                        format!("Длинный текст ({chars} симв.) сжал шрифт до {fs} px в баббле {bw}×{bh} px — можно сжать фразу ИИ или поднять кегль"),
                    )
                } else {
                    (
                        "tiny_font",
                        format!("Мелкий шрифт {fs} px в баббле {bw}×{bh} px (средний по стр. {page_med} px)"),
                    )
                };
                issues.push(json!({
                    "page": fname,
                    "page_index": idx + 1,
                    "source_path": img_path.to_string_lossy(),
                    "dest_path": out_dir.join(&fname).to_string_lossy(),
                    "block_id": b.id,
                    "kind": kind,
                    "rendered_font_size": fs,
                    "recommended_font_size": rec_fs,
                    "bbox_size": format!("{bw}×{bh}"),
                    "is_crowded": is_crowded,
                    "text": tr,
                    "original_text": b.original_text,
                    "reason": reason,
                }));
            }

            if let Some(ref canon_db) = canon_db_opt {
                let qa = miko_translator::lint_pair(&b.original_text, tr, &qa_glossary, canon_db);
                if !qa.issues.is_empty() {
                    let primary_kind = qa.issues[0].kind.inspector_kind();
                    let suggested = if qa.text != tr {
                        Some(qa.text.clone())
                    } else {
                        None
                    };
                    let reason = qa
                        .issues
                        .iter()
                        .map(|i| i.message.as_str())
                        .collect::<Vec<_>>()
                        .join("; ");
                    issues.push(json!({
                        "page": fname,
                        "page_index": idx + 1,
                        "source_path": img_path.to_string_lossy(),
                        "dest_path": out_dir.join(&fname).to_string_lossy(),
                        "block_id": b.id,
                        "kind": primary_kind,
                        "suggested_text": suggested,
                        "needs_retry": qa.needs_retry,
                        "rendered_font_size": fs,
                        "recommended_font_size": fs,
                        "bbox_size": format!("{bw}×{bh}"),
                        "is_crowded": false,
                        "text": tr,
                        "original_text": b.original_text,
                        "reason": reason,
                    }));
                }
            }
        }
    }

    all_fs.sort_unstable();
    let median_fs = if all_fs.is_empty() {
        0
    } else {
        all_fs[all_fs.len() / 2]
    };
    let min_fs = all_fs.first().copied().unwrap_or(0);

    Json(json!({
        "status": "ok",
        "total_pages": images.len(),
        "translated_pages": translated_pages,
        "total_blocks": all_fs.len(),
        "median_font_size": median_fs,
        "min_rendered_font_size": min_fs,
        "threshold_px": threshold,
        "issues_count": issues.len(),
        "issues": issues,
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct ChapterQaAutofixPayload {
    pub folder_path: String,
    pub target_lang: Option<String>,
    pub tiny_threshold_px: Option<i32>,
    pub min_target_fs: Option<i32>,
    pub shorten_crowded_ai: Option<bool>,
}

pub async fn chapter_qa_autofix(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChapterQaAutofixPayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.folder_path), "folder_path") {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": err.to_string()})),
        )
            .into_response();
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
    let images = folder_job::collect_folder_images(&src_dir);
    let mut manifest = folder_job::load_manifest(&out_dir);
    let cfg = state
        .config
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let threshold = payload.tiny_threshold_px.unwrap_or(14).clamp(9, 26);
    let min_target = payload
        .min_target_fs
        .unwrap_or((threshold + 2).max(15))
        .clamp(12, 26);
    let use_ai_shorten = payload.shorten_crowded_ai.unwrap_or(false);

    let mut fixed_pages = 0usize;
    let mut fixed_blocks = 0usize;

    let is_en_canon = t_lang.trim().eq_ignore_ascii_case("en") && cfg.adult_canon_en;
    let qa_glossary = if is_en_canon {
        let global_g = state
            .db
            .load_glossary("global")
            .unwrap_or_else(|_| cfg.glossary.clone());
        let folder_g = state
            .db
            .load_glossary(&format!(
                "folder:{}",
                src_dir.to_string_lossy().trim().to_ascii_lowercase()
            ))
            .unwrap_or_default();
        let user_g = miko_translator::merge_glossaries(&global_g, &folder_g);
        let auto_lock: Vec<miko_core::GlossaryEntry> = manifest
            .get("name_lock")
            .cloned()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        miko_translator::merge_glossaries(&auto_lock, &user_g)
    } else {
        Vec::new()
    };
    let canon_overrides = state.db.load_canon_overrides("en").unwrap_or_default();
    let canon_db_opt = if is_en_canon {
        Some(miko_translator::merge_overrides(
            miko_translator::builtin_canon_en(),
            &canon_overrides,
        ))
    } else {
        None
    };

    for img_path in &images {
        let fname = img_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let Some(raw_arr) = manifest["pages"][&fname]["blocks"].as_array() else {
            continue;
        };
        let Ok(mut blocks) =
            serde_json::from_value::<Vec<TextBlock>>(Value::Array(raw_arr.clone()))
        else {
            continue;
        };

        let mut page_needs_fix = false;
        for b in &mut blocks {
            if b.is_deleted || b.translated_text.trim().is_empty() {
                continue;
            }
            if let Some(ref canon_db) = canon_db_opt {
                let qa = miko_translator::lint_pair(
                    &b.original_text,
                    &b.translated_text,
                    &qa_glossary,
                    canon_db,
                );
                if qa.text != b.translated_text.trim() {
                    b.translated_text = qa.text.clone();
                    page_needs_fix = true;
                    fixed_blocks += 1;
                }
                if qa.needs_retry && use_ai_shorten && !b.original_text.trim().is_empty() {
                    let hints: Vec<String> = qa
                        .issues
                        .iter()
                        .filter_map(|i| i.hint_for_retry.clone())
                        .collect();
                    let retry_cfg = miko_translator::PromptConfig {
                        anti_censorship: cfg.anti_censorship_prompt,
                        anti_censorship_custom_prompt: cfg.anti_censorship_custom_prompt.clone(),
                        custom_system_prompt: cfg.custom_system_prompt.clone(),
                        custom_context_prompt: miko_translator::compose_context_with_series_hint(
                            &cfg.custom_context_prompt,
                            Some(&src_dir),
                        ),
                        adult_canon_en: cfg.adult_canon_en,
                        qa_hints: hints,
                        canon_overrides: canon_overrides.clone(),
                        ..Default::default()
                    };
                    if let Ok((retr, _)) = state
                        .translator
                        .translate_single_text_with_prompts(
                            &b.original_text,
                            &t_lang,
                            &cfg.selected_model,
                            &cfg.local_fallback_model_multilang,
                            cfg.cloud_only_no_local_fallback,
                            &cfg.unsloth_url,
                            &cfg.unsloth_api_key,
                            &qa_glossary,
                            &retry_cfg,
                            false,
                            Some(&state.db),
                        )
                        .await
                    {
                        if !retr.trim().is_empty() {
                            b.translated_text = retr.trim().to_string();
                            page_needs_fix = true;
                            fixed_blocks += 1;
                        }
                    }
                }
            }

            let bw = (b.bbox[2] - b.bbox[0]).max(1);
            let bh = (b.bbox[3] - b.bbox[1]).max(1);
            let fs = b.font_size.or(b.rendered_font_size).unwrap_or(20);
            let glued = has_glued_dots(&b.translated_text);
            if (fs <= threshold && bw >= 52 && bh >= 80) || glued {
                page_needs_fix = true;
                fixed_blocks += 1;
                if glued {
                    b.translated_text =
                        miko_renderer::format_comic_typography(&b.translated_text, false);
                }
                // Если пользователь вручную зажал мелкий font_size — сбрасываем к Авто, чтобы сработал новый крупный укладчик
                if b.font_size.is_some_and(|f| f <= threshold) {
                    b.font_size = None;
                }
                if use_ai_shorten && b.translated_text.chars().count() > 28 {
                    let prompt_text = format!(
                        "Сократи эту реплику на 35% короче по символам для узкого облачка комикса, сохранив живой смысл (верни ТОЛЬКО короткую фразу на языке {t_lang}): {}",
                        b.translated_text.trim()
                    );
                    let shorten_prompt_cfg = miko_translator::PromptConfig {
                        anti_censorship: cfg.anti_censorship_prompt,
                        anti_censorship_custom_prompt: cfg.anti_censorship_custom_prompt.clone(),
                        custom_system_prompt: String::new(),
                        custom_context_prompt: cfg.custom_context_prompt.clone(),
                        adult_canon_en: cfg.adult_canon_en,
                        ..Default::default()
                    };
                    if let Ok((short_s, _)) = state
                        .translator
                        .translate_single_text_with_prompts(
                            &prompt_text,
                            &t_lang,
                            &cfg.selected_model,
                            &cfg.local_fallback_model_multilang,
                            cfg.cloud_only_no_local_fallback,
                            &cfg.unsloth_url,
                            &cfg.unsloth_api_key,
                            &[],
                            &shorten_prompt_cfg,
                            false,
                            None,
                        )
                        .await
                    {
                        if !short_s.trim().is_empty() {
                            b.translated_text = short_s.trim().to_string();
                        }
                    }
                }
            }
        }

        if !page_needs_fix {
            continue;
        }

        let st_cr = state.clone();
        let src_p = img_path.clone();
        let dst_p = out_dir.join(&fname);
        let cfg_c = cfg.clone();
        let eff_min_fs = (cfg_c.min_font_size as i32).max(min_target - 1);
        let eff_max_fs = (cfg_c.max_font_size as i32).max(eff_min_fs + 4);

        if let Ok(Ok(updated_blocks)) =
            tokio::task::spawn_blocking(move || -> Result<Vec<TextBlock>, String> {
                let rgb = image::open(&src_p).map_err(|e| e.to_string())?.to_rgb8();
                let mut cg = st_cr
                    .cleaner_engine
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if cg.is_none() {
                    *cg = Some(RustCleanerEngine::load(&st_cr.base_dir, true));
                }
                let src_s = src_p.to_string_lossy().to_string();
                let cl = cg
                    .as_mut()
                    .expect("cleaner")
                    .clean_image(
                        &rgb,
                        &blocks,
                        &CleanOptions {
                            skip_untranslated: true,
                            ..Default::default()
                        },
                        Some(&src_s),
                        0,
                    )
                    .map_err(|e| e.to_string())?;
                let def_font = cfg_c.resolve_default_font_path(&st_cr.base_dir);
                let mut rr = st_cr
                    .renderer
                    .render_page(
                        &cl.cleaned,
                        &mut blocks,
                        &RenderOptions {
                            min_font_size: eff_min_fs,
                            max_font_size: eff_max_fs,
                            font_scale_pct: cfg_c.font_scale_pct,
                            uppercase: cfg_c.uppercase,
                            overflow_mode: cfg_c.overflow_mode.clone(),
                            default_font_path: def_font.clone(),
                        },
                    )
                    .map_err(|e| e.to_string())?;

                // Если какой-то крупный баббл всё ещё остался <= threshold (напр. сверхплотная фраза без AI),
                // мягко фиксируем ему читаемый кегль `min_target` с защитной белой обводкой
                let mut forced_any = false;
                for b in &mut blocks {
                    let bw = (b.bbox[2] - b.bbox[0]).max(1);
                    let bh = (b.bbox[3] - b.bbox[1]).max(1);
                    if !b.is_deleted
                        && !b.translated_text.trim().is_empty()
                        && bw >= 62
                        && bh >= 95
                        && b.rendered_font_size.unwrap_or(20) <= threshold
                    {
                        b.font_size = Some(min_target);
                        forced_any = true;
                    }
                }
                if forced_any {
                    rr = st_cr
                        .renderer
                        .render_page(
                            &cl.cleaned,
                            &mut blocks,
                            &RenderOptions {
                                min_font_size: eff_min_fs,
                                max_font_size: eff_max_fs,
                                font_scale_pct: cfg_c.font_scale_pct,
                                uppercase: cfg_c.uppercase,
                                overflow_mode: "overflow".to_string(),
                                default_font_path: def_font,
                            },
                        )
                        .map_err(|e| e.to_string())?;
                }

                if let Some(par) = dst_p.parent() {
                    let _ = std::fs::create_dir_all(par);
                }
                rr.rendered.save(&dst_p).map_err(|e| e.to_string())?;
                Ok(blocks)
            })
            .await
        {
            manifest["pages"][&fname]["blocks"] = json!(updated_blocks);
            fixed_pages += 1;
        }
    }

    if fixed_pages > 0 {
        let _ = folder_job::save_manifest(&out_dir, &manifest);
    }

    Json(json!({
        "status": "ok",
        "fixed_pages": fixed_pages,
        "fixed_blocks": fixed_blocks,
    }))
    .into_response()
}
