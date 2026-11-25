use crate::AppState;
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use image::{ImageFormat, RgbImage};
use miko_cleaner::{compute_clean_state_hash, CleanOptions, RustCleanerEngine};
use miko_core::models::normalize_clean_bbox;
use miko_core::security::reject_unc_path;
use miko_core::TextBlock;
use miko_ocr::{OcrOptions, RustOcrEngine};
use miko_renderer::{auto_diamond_wrap_lines, measure_text_width, resolve_font_for_text, FontCache, RenderOptions};
use miko_translator::{merge_glossaries, TranslateBatchRequest};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, UNIX_EPOCH};

fn is_supported_ext(p: &Path) -> bool {
    p.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp" | "bmp"))
}

fn content_type_for_ext(p: &Path) -> &'static str {
    match p
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        _ => "image/png",
    }
}

pub fn encode_rgb_to_data_url(img: &RgbImage, fmt: &str) -> Result<String, String> {
    let mut buf = Cursor::new(Vec::new());
    let f = fmt.trim().to_ascii_lowercase();
    let (img_fmt, mime) = match f.as_str() {
        "jpeg" | "jpg" => (ImageFormat::Jpeg, "image/jpeg"),
        "webp" => (ImageFormat::WebP, "image/webp"),
        _ => (ImageFormat::Png, "image/png"),
    };
    img.write_to(&mut buf, img_fmt)
        .map_err(|e| format!("Ошибка кодирования изображения: {e}"))?;
    Ok(format!("data:{mime};base64,{}", B64.encode(buf.into_inner())))
}

pub fn decode_base64_rgb(raw: &str) -> Result<RgbImage, String> {
    let payload = raw
        .trim()
        .split_once(',')
        .map(|(_, rhs)| rhs)
        .unwrap_or_else(|| raw.trim());
    let bytes = B64
        .decode(payload)
        .map_err(|e| format!("Некорректный base64 изображения: {e}"))?;
    let im = image::load_from_memory(&bytes)
        .map_err(|e| format!("Не удалось декодировать изображение: {e}"))?;
    if (im.width() as u64) * (im.height() as u64) > 40_000_000 {
        return Err("Изображение превышает лимит 40 Мпикс".to_string());
    }
    Ok(im.to_rgb8())
}

fn file_mtime_secs(p: &Path) -> i64 {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[derive(Debug, Deserialize)]
pub struct LocalImageQuery {
    pub path: String,
}

pub async fn local_image(
    State(state): State<Arc<AppState>>,
    Query(q): Query<LocalImageQuery>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&q.path), "path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let img_p = Path::new(q.path.trim());
    if !is_supported_ext(img_p) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Неподдерживаемый формат изображения"})),
        )
            .into_response();
    }
    if !state.allowed_roots.is_within_allowed_roots(img_p) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Путь вне разрешённых директорий"})),
        )
            .into_response();
    }
    if !img_p.is_file() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Файл изображения не найден"})),
        )
            .into_response();
    }
    match tokio::fs::read(img_p).await {
        Ok(bytes) => ([(header::CONTENT_TYPE, content_type_for_ext(img_p))], bytes).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e.to_string()})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct FolderInspectPayload {
    pub folder_path: String,
    pub target_lang: Option<String>,
}

pub async fn folder_inspect(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<FolderInspectPayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.folder_path), "folder_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let dir_p = Path::new(payload.folder_path.trim());
    if !state.allowed_roots.is_within_allowed_roots(dir_p) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Доступ к папке вне разрешённых директорий запрещён"})),
        )
            .into_response();
    }
    if !dir_p.is_dir() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Папка не найдена"})),
        )
            .into_response();
    }

    let t_lang = payload
        .target_lang
        .as_deref()
        .unwrap_or("ru")
        .trim()
        .to_ascii_uppercase();
    let dir_name = dir_p
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("manga");
    let out_dir = dir_p
        .parent()
        .unwrap_or(dir_p)
        .join(format!("{dir_name}_{t_lang}"));

    let mut pages = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir_p) {
        let mut files: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && is_supported_ext(p))
            .collect();
        files.sort();
        for f in files {
            let name = f
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            let out_file = out_dir.join(&name);
            pages.push(json!({
                "name": name,
                "source_path": f.to_string_lossy(),
                "translated_exists": out_file.is_file(),
                "dest_path": out_file.to_string_lossy(),
            }));
        }
    }

    Json(json!({
        "status": "ok",
        "folder_path": dir_p.to_string_lossy(),
        "output_dir": out_dir.to_string_lossy(),
        "total_pages": pages.len(),
        "pages": pages,
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct AutoDiamondWrapPayload {
    pub text: String,
    pub max_width_px: Option<f32>,
    pub font_size: Option<f32>,
    pub font_family: Option<String>,
}

pub async fn auto_diamond_wrap(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<AutoDiamondWrapPayload>,
) -> Json<Value> {
    let fs = payload.font_size.unwrap_or(20.0).clamp(8.0, 120.0);
    let max_w = payload.max_width_px.unwrap_or(220.0).clamp(40.0, 2000.0);
    let cache = FontCache::new();
    let font_opt = resolve_font_for_text(
        &state.base_dir.join("fonts"),
        &cache,
        payload.font_family.as_deref(),
        &payload.text,
    )
    .and_then(|p| cache.load_font(&p).ok());

    let lines = auto_diamond_wrap_lines(&payload.text, max_w, |s| match font_opt {
        Some(ref f) => measure_text_width(f, fs, s),
        None => (s.chars().count() as f32) * fs * 0.56,
    });
    let wrapped_text = lines.join("\n");
    Json(json!({
        "status": "ok",
        "wrapped_text": wrapped_text,
        "lines": lines,
    }))
}

#[derive(Debug, Deserialize)]
pub struct RetranslateSinglePayload {
    pub text: String,
    pub target_lang: Option<String>,
    pub model: Option<String>,
    pub folder_path: Option<String>,
}

pub async fn retranslate_single_block(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RetranslateSinglePayload>,
) -> Response {
    if let Some(ref fp) = payload.folder_path {
        if let Err(err) = reject_unc_path(Some(fp), "folder_path") {
            return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()})))
                .into_response();
        }
    }
    let cfg = state.config.read().expect("config rwlock").clone();
    let global_g = state.db.load_glossary("global").unwrap_or(cfg.glossary);
    let folder_g = payload
        .folder_path
        .as_deref()
        .map(|fp| {
            state
                .db
                .load_glossary(&format!("folder:{}", fp.trim().to_ascii_lowercase()))
                .unwrap_or_default()
        })
        .unwrap_or_default();
    let merged = merge_glossaries(&global_g, &folder_g);

    let t_lang = payload.target_lang.unwrap_or_else(|| "ru".to_string());
    let model = payload.model.unwrap_or(cfg.selected_model);

    match state
        .translator
        .translate_single_text(
            &payload.text,
            &t_lang,
            &model,
            &cfg.unsloth_url,
            &cfg.unsloth_api_key,
            &merged,
            cfg.anti_censorship_prompt,
            Some(&state.db),
        )
        .await
    {
        Ok(translated) => Json(json!({
            "status": "ok",
            "original_text": payload.text,
            "translated_text": translated,
            "target_lang": t_lang,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"detail": e.to_string()})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct TranslatePayload {
    pub image: Option<String>,
    pub image_base64: Option<String>,
    pub lang: Option<String>,
    pub target_lang: Option<String>,
    pub model: Option<String>,
    pub ollama_model: Option<String>,
    pub uppercase: Option<bool>,
    pub min_font_size: Option<u32>,
    pub max_font_size: Option<u32>,
    pub detect_free_text: Option<bool>,
    pub text_filter_mode: Option<String>,
    pub force_retranslate: Option<bool>,
    pub return_image: Option<bool>,
    pub image_format: Option<String>,
}

async fn run_full_rgb_pipeline(
    state: Arc<AppState>,
    rgb: RgbImage,
    src_file_hint: Option<PathBuf>,
    target_lang: String,
    model_override: Option<String>,
    uppercase: bool,
    min_fs: i32,
    max_fs: i32,
    detect_free_text: bool,
    text_filter_mode: String,
    use_cache: bool,
    image_format: String,
    return_image: bool,
) -> Result<Value, String> {
    let t_total = Instant::now();
    let cfg = state.config.read().expect("config rwlock").clone();
    let (w, h) = rgb.dimensions();

    // 1. OCR на GPU/CPU
    let st_ocr = state.clone();
    let rgb_ocr = rgb.clone();
    let ocr_res = tokio::task::spawn_blocking(move || {
        let mut guard = st_ocr.ocr_engine.lock().expect("ocr mutex");
        if guard.is_none() {
            let eng = RustOcrEngine::load(&st_ocr.base_dir, true).map_err(|e| e.to_string())?;
            *guard = Some(eng);
        }
        guard
            .as_mut()
            .expect("ocr initialized")
            .process_rgb_image(
                &rgb_ocr,
                &OcrOptions {
                    detect_free_text,
                    text_filter_mode,
                    ..Default::default()
                },
            )
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())??;

    // 2. Перевод батча (async HTTP + SQLite кэш)
    let glossary = state.db.load_glossary("global").unwrap_or(cfg.glossary);
    let tr_res = state
        .translator
        .translate_blocks(
            TranslateBatchRequest {
                blocks: ocr_res.blocks,
                target_lang: target_lang.clone(),
                model: model_override.unwrap_or(cfg.selected_model),
                cloud_url: cfg.unsloth_url,
                cloud_api_key: cfg.unsloth_api_key,
                glossary,
                context_lines: Vec::new(),
                anti_censorship: cfg.anti_censorship_prompt,
                use_cache,
            },
            Some(&state.db),
        )
        .await
        .map_err(|e| e.to_string())?;

    // 3. Очистка (Neural Seg CUDA + Harmonic SOR) + Рендер (ab_glyph)
    let st_cr = state.clone();
    let mut blocks_final = tr_res.blocks;
    let src_str = src_file_hint
        .as_ref()
        .map(|p| p.to_string_lossy().to_string());
    let src_mtime = src_file_hint.as_deref().map(file_mtime_secs).unwrap_or(0);

    let (rendered_img, clean_ep, blocks_out) = tokio::task::spawn_blocking(move || -> Result<(RgbImage, String, Vec<TextBlock>), String> {
        let mut cguard = st_cr.cleaner_engine.lock().expect("cleaner mutex");
        if cguard.is_none() {
            *cguard = Some(RustCleanerEngine::load(&st_cr.base_dir, true));
        }
        let cleaner = cguard.as_mut().expect("cleaner init");
        let clean_res = cleaner
            .clean_image(
                &rgb,
                &blocks_final,
                &CleanOptions {
                    use_cache: use_cache && src_str.is_some(),
                    ..Default::default()
                },
                src_str.as_deref(),
                src_mtime,
            )
            .map_err(|e| e.to_string())?;
        let ep = clean_res.timings.ep.clone();

        let render_res = st_cr
            .renderer
            .render_page(
                &clean_res.cleaned,
                &mut blocks_final,
                &RenderOptions {
                    min_font_size: min_fs,
                    max_font_size: max_fs,
                    uppercase,
                    overflow_mode: cfg.overflow_mode,
                    default_font_path: None,
                },
            )
            .map_err(|e| e.to_string())?;

        Ok((render_res.rendered, ep, blocks_final))
    })
    .await
    .map_err(|e| e.to_string())??;

    let free_cnt = blocks_out.iter().filter(|b| b.is_free_text).count();
    let b64_img = if return_image {
        Some(encode_rgb_to_data_url(&rendered_img, &image_format)?)
    } else {
        None
    };

    if let Some(ref sp) = src_file_hint {
        let state_hash = compute_clean_state_hash(
            &blocks_out,
            &sp.to_string_lossy(),
            src_mtime,
            9,
            "rust_harmonic_v1",
            "balanced",
        );
        let _ = state
            .db
            .save_page_state(&sp.to_string_lossy(), &state_hash, &blocks_out);
    }

    Ok(json!({
        "status": "ok",
        "runtime": "rust",
        "lang": "auto",
        "detected_lang": tr_res.detected_lang,
        "target_lang": target_lang,
        "ollama_model": tr_res.active_model,
        "inpaint_backend": format!("Rust Harmonic ({clean_ep})"),
        "image_size": [w, h],
        "processing_time_sec": (t_total.elapsed().as_secs_f64() * 1000.0).round() / 1000.0,
        "blocks_count": blocks_out.len(),
        "free_text_blocks_count": free_cnt,
        "translated_image": b64_img,
        "blocks": blocks_out,
    }))
}

pub async fn translate_page(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<TranslatePayload>,
) -> Response {
    let raw_b64 = payload.image.or(payload.image_base64).unwrap_or_default();
    if raw_b64.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Поле 'image' (base64) обязательно"})),
        )
            .into_response();
    }
    let rgb = match decode_base64_rgb(&raw_b64) {
        Ok(im) => im,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"detail": e}))).into_response(),
    };
    let cfg = state.config.read().expect("config rwlock").clone();
    let t_lang = payload
        .target_lang
        .unwrap_or_else(|| "ru".to_string())
        .to_ascii_lowercase();
    let model = payload.ollama_model.or(payload.model);

    match run_full_rgb_pipeline(
        state,
        rgb,
        None,
        t_lang,
        model,
        payload.uppercase.unwrap_or(cfg.uppercase),
        payload.min_font_size.unwrap_or(cfg.min_font_size) as i32,
        payload.max_font_size.unwrap_or(cfg.max_font_size) as i32,
        payload.detect_free_text.unwrap_or(cfg.detect_free_text),
        payload.text_filter_mode.unwrap_or(cfg.text_filter_mode),
        !payload.force_retranslate.unwrap_or(false),
        payload.image_format.unwrap_or_else(|| "png".to_string()),
        payload.return_image.unwrap_or(true),
    )
    .await
    {
        Ok(val) => Json(val).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct TranslatePageFilePayload {
    pub source_path: String,
    pub dest_path: Option<String>,
    pub target_lang: Option<String>,
    pub model: Option<String>,
    pub ollama_model: Option<String>,
    pub uppercase: Option<bool>,
    pub min_font_size: Option<u32>,
    pub max_font_size: Option<u32>,
    pub detect_free_text: Option<bool>,
    pub force_retranslate: Option<bool>,
    pub return_image: Option<bool>,
    pub image_format: Option<String>,
}

pub async fn translate_page_file(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<TranslatePageFilePayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.source_path), "source_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    if let Some(ref dp) = payload.dest_path {
        if let Err(err) = reject_unc_path(Some(dp), "dest_path") {
            return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()})))
                .into_response();
        }
    }
    let src_p = PathBuf::from(payload.source_path.trim());
    if !src_p.is_file() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Исходный файл не найден"})),
        )
            .into_response();
    }
    if !state.allowed_roots.is_within_allowed_roots(&src_p) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Путь source_path вне разрешённых директорий"})),
        )
            .into_response();
    }
    if let Some(ref dp) = payload.dest_path {
        let dest_p = Path::new(dp.trim());
        let parent_chk = dest_p.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(dest_p);
        if !state.allowed_roots.is_within_allowed_roots(parent_chk) {
            return (
                StatusCode::FORBIDDEN,
                Json(json!({"detail": "Путь dest_path вне разрешённых директорий"})),
            )
                .into_response();
        }
    }

    let rgb = match image::open(&src_p) {
        Ok(im) => im.to_rgb8(),
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"detail": format!("Ошибка чтения изображения: {e}")})),
            )
                .into_response()
        }
    };
    let cfg = state.config.read().expect("config rwlock").clone();
    let t_lang = payload
        .target_lang
        .unwrap_or_else(|| "ru".to_string())
        .to_ascii_lowercase();
    let dest_opt = payload.dest_path.map(|s| PathBuf::from(s.trim()));

    match run_full_rgb_pipeline(
        state,
        rgb,
        Some(src_p),
        t_lang,
        payload.ollama_model.or(payload.model),
        payload.uppercase.unwrap_or(cfg.uppercase),
        payload.min_font_size.unwrap_or(cfg.min_font_size) as i32,
        payload.max_font_size.unwrap_or(cfg.max_font_size) as i32,
        payload.detect_free_text.unwrap_or(cfg.detect_free_text),
        cfg.text_filter_mode,
        !payload.force_retranslate.unwrap_or(false),
        payload.image_format.unwrap_or_else(|| "png".to_string()),
        payload.return_image.unwrap_or(false),
    )
    .await
    {
        Ok(mut val) => {
            if let (Some(dp), Some(b64)) = (dest_opt, val["translated_image"].as_str()) {
                if let Ok(im) = decode_base64_rgb(b64) {
                    if let Some(parent) = dp.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let _ = im.save(&dp);
                    val["saved_output_path"] = json!(dp.to_string_lossy());
                }
            }
            Json(val).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct SaveEditedPagePayload {
    pub source_path: String,
    pub dest_path: Option<String>,
    pub blocks: Vec<TextBlock>,
    pub uppercase: Option<bool>,
    pub min_font_size: Option<u32>,
    pub max_font_size: Option<u32>,
    pub return_image: Option<bool>,
    pub image_format: Option<String>,
}

pub async fn save_edited_page(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SaveEditedPagePayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.source_path), "source_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    if let Some(ref dp) = payload.dest_path {
        if let Err(err) = reject_unc_path(Some(dp), "dest_path") {
            return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()})))
                .into_response();
        }
    }
    let src_p = PathBuf::from(payload.source_path.trim());
    if !src_p.is_file() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Исходный файл не найден"})),
        )
            .into_response();
    }
    if !state.allowed_roots.is_within_allowed_roots(&src_p) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Путь source_path вне разрешённых директорий"})),
        )
            .into_response();
    }
    let dest_p = payload
        .dest_path
        .as_ref()
        .map(|s| PathBuf::from(s.trim()))
        .unwrap_or_else(|| src_p.with_extension("edited.png"));
    let dest_parent = dest_p.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(&dest_p);
    if !state.allowed_roots.is_within_allowed_roots(dest_parent) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Путь dest_path вне разрешённых директорий"})),
        )
            .into_response();
    }

    let mut blocks = payload.blocks;
    for b in &mut blocks {
        normalize_clean_bbox(b);
    }

    let cfg = state.config.read().expect("config rwlock").clone();
    let st_clone = state.clone();
    let src_clone = src_p.clone();
    let dest_clone = dest_p.clone();
    let uppercase = payload.uppercase.unwrap_or(cfg.uppercase);
    let min_fs = payload.min_font_size.unwrap_or(cfg.min_font_size) as i32;
    let max_fs = payload.max_font_size.unwrap_or(cfg.max_font_size) as i32;
    let ret_img = payload.return_image.unwrap_or(false);
    let img_fmt = payload.image_format.unwrap_or_else(|| "png".to_string());

    let res = tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let rgb = image::open(&src_clone)
            .map_err(|e| format!("Ошибка декодирования изображения: {e}"))?
            .to_rgb8();
        let src_str = src_clone.to_string_lossy().to_string();
        let mtime = file_mtime_secs(&src_clone);

        let mut cguard = st_clone.cleaner_engine.lock().expect("cleaner mutex");
        if cguard.is_none() {
            *cguard = Some(RustCleanerEngine::load(&st_clone.base_dir, true));
        }
        let clean_res = cguard
            .as_mut()
            .expect("cleaner init")
            .clean_image(
                &rgb,
                &blocks,
                &CleanOptions::default(),
                Some(&src_str),
                mtime,
            )
            .map_err(|e| e.to_string())?;

        let render_res = st_clone
            .renderer
            .render_page(
                &clean_res.cleaned,
                &mut blocks,
                &RenderOptions {
                    min_font_size: min_fs,
                    max_font_size: max_fs,
                    uppercase,
                    overflow_mode: cfg.overflow_mode,
                    default_font_path: None,
                },
            )
            .map_err(|e| e.to_string())?;

        if let Some(p) = dest_clone.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        render_res
            .rendered
            .save(&dest_clone)
            .map_err(|e| format!("Ошибка сохранения результата: {e}"))?;

        let hash = compute_clean_state_hash(
            &blocks,
            &src_str,
            mtime,
            9,
            "rust_harmonic_v1",
            "balanced",
        );
        let _ = st_clone.db.save_page_state(&src_str, &hash, &blocks);

        let b64_opt = if ret_img {
            Some(encode_rgb_to_data_url(&render_res.rendered, &img_fmt)?)
        } else {
            None
        };

        Ok(json!({
            "status": "ok",
            "runtime": "rust",
            "saved_output_path": dest_clone.to_string_lossy(),
            "blocks_rendered": render_res.blocks_rendered,
            "clean_timings": clean_res.timings,
            "render_timings": render_res.timings,
            "translated_image": b64_opt,
            "blocks": blocks,
        }))
    })
    .await;

    match res {
        Ok(Ok(val)) => Json(val).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e.to_string()})),
        )
            .into_response(),
    }
}
