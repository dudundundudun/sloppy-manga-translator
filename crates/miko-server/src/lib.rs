pub mod folder_job;
pub mod studio;

use axum::body::Body;
use axum::extract::{Query, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use miko_cleaner::{CleanOptions, RustCleanerEngine};
use miko_core::security::{
    is_trusted_host_header, is_trusted_local_origin, is_trusted_referer, mask_api_key,
    reject_unc_path, AllowedRoots, PRIVILEGED_ENDPOINTS,
};
use miko_core::{AppConfig, FolderJobState, GlossaryEntry, StudioDb, TextBlock};
use miko_ocr::{OcrOptions, RustOcrEngine};
use miko_ort::{scan_onnx_inventory, OrtInventory};
use miko_renderer::{list_available_fonts, FontInfo, RustRendererEngine};
use miko_translator::{
    export_glossary_to_sakura_text, export_glossary_to_tsv, merge_glossaries,
    normalize_glossary_entries, RustTranslator,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{watch, Notify};

pub const APP_VERSION_RS: &str = "7.0.0-rust-alpha";

pub struct AppState {
    pub base_dir: PathBuf,
    pub started_at: Instant,
    pub startup_ms: u128,
    pub db: StudioDb,
    pub config: RwLock<AppConfig>,
    pub folder_job: RwLock<FolderJobState>,
    pub folder_tx: watch::Sender<u64>,
    pub allowed_roots: AllowedRoots,
    pub available_fonts: Vec<FontInfo>,
    pub ort_inventory: OrtInventory,
    pub ocr_engine: Mutex<Option<RustOcrEngine>>,
    pub cleaner_engine: Mutex<Option<RustCleanerEngine>>,
    pub translator: Arc<RustTranslator>,
    pub renderer: Arc<RustRendererEngine>,
    pub folder_stop_flag: Arc<AtomicBool>,
    pub folder_pause_flag: Arc<AtomicBool>,
    pub shutdown_notify: Notify,
}

impl AppState {
    pub fn init(base_dir: PathBuf) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>> {
        let t0 = Instant::now();
        let db_path = base_dir.join(".miko").join("miko_studio.db");
        let db = StudioDb::open(&db_path)?;
        let cfg = AppConfig::load_defaults_from_config_json(&base_dir);
        if !cfg.glossary.is_empty() && db.load_glossary("global")?.is_empty() {
            let _ = db.replace_glossary("global", &cfg.glossary);
        }
        let fonts = list_available_fonts(&base_dir.join("fonts"));
        let ort_inventory = scan_onnx_inventory(&base_dir.join("models").join("onnx"));
        let allowed_roots = AllowedRoots::new();
        let _ = allowed_roots.register_root(&base_dir);
        let save_dir = PathBuf::from(&cfg.web_save_dir);
        if save_dir.exists() {
            let _ = allowed_roots.register_root(&save_dir);
        }
        let (folder_tx, _) = watch::channel(0u64);
        let startup_ms = t0.elapsed().as_millis();
        let renderer = Arc::new(RustRendererEngine::new(base_dir.join("fonts")));
        let translator = Arc::new(RustTranslator::new(None));

        Ok(Arc::new(Self {
            base_dir,
            started_at: t0,
            startup_ms,
            db,
            config: RwLock::new(cfg),
            folder_job: RwLock::new(FolderJobState::default()),
            folder_tx,
            allowed_roots,
            available_fonts: fonts,
            ort_inventory,
            ocr_engine: Mutex::new(None),
            cleaner_engine: Mutex::new(None),
            translator,
            renderer,
            folder_stop_flag: Arc::new(AtomicBool::new(false)),
            folder_pause_flag: Arc::new(AtomicBool::new(false)),
            shutdown_notify: Notify::new(),
        }))
    }

    pub fn init_in_memory(
        base_dir: PathBuf,
    ) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>> {
        let t0 = Instant::now();
        let db = StudioDb::open_in_memory()?;
        let cfg = AppConfig::default_with_base(&base_dir);
        let fonts = list_available_fonts(&base_dir.join("fonts"));
        let ort_inventory = scan_onnx_inventory(&base_dir.join("models").join("onnx"));
        let allowed_roots = AllowedRoots::new();
        let _ = allowed_roots.register_root(&base_dir);
        let (folder_tx, _) = watch::channel(0u64);
        let startup_ms = t0.elapsed().as_millis();
        let renderer = Arc::new(RustRendererEngine::new(base_dir.join("fonts")));
        let translator = Arc::new(RustTranslator::new(None));
        Ok(Arc::new(Self {
            base_dir,
            started_at: t0,
            startup_ms,
            db,
            config: RwLock::new(cfg),
            folder_job: RwLock::new(FolderJobState::default()),
            folder_tx,
            allowed_roots,
            available_fonts: fonts,
            ort_inventory,
            ocr_engine: Mutex::new(None),
            cleaner_engine: Mutex::new(None),
            translator,
            renderer,
            folder_stop_flag: Arc::new(AtomicBool::new(false)),
            folder_pause_flag: Arc::new(AtomicBool::new(false)),
            shutdown_notify: Notify::new(),
        }))
    }
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}

async fn local_security_middleware(req: Request<Body>, next: Next) -> Response {
    let path = req.uri().path();
    let is_priv = PRIVILEGED_ENDPOINTS
        .iter()
        .any(|&p| path == p || path.starts_with(&format!("{p}/")));

    let headers = req.headers();
    if !is_trusted_host_header(header_str(headers, "host")) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Недопустимый заголовок Host"})),
        )
            .into_response();
    }
    if is_priv {
        if !is_trusted_local_origin(header_str(headers, "origin"))
            || !is_trusted_referer(header_str(headers, "referer"))
        {
            return (
                StatusCode::FORBIDDEN,
                Json(json!({"detail": "Доступ с внешнего Origin/Referer запрещен"})),
            )
                .into_response();
        }
    }
    next.run(req).await
}

async fn serve_ui(State(state): State<Arc<AppState>>) -> Response {
    let html_path = state.base_dir.join("web_studio.html");
    match tokio::fs::read_to_string(&html_path).await {
        Ok(body) => Html(body).into_response(),
        Err(_) => Html("<h1>Miko-chan Manga Studio (Rust Core)</h1><p>web_studio.html not found in base_dir.</p>").into_response(),
    }
}

async fn serve_mascot(State(state): State<Arc<AppState>>) -> Response {
    let mascot_path = state.base_dir.join("mascot.png");
    match tokio::fs::read(&mascot_path).await {
        Ok(bytes) => ([(header::CONTENT_TYPE, "image/png")], bytes).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

#[derive(Serialize)]
struct HealthGpuInfo {
    cuda_available: bool,
    cuda_ep_compiled: bool,
    directml_ep_compiled: bool,
    onnx_ready_count: usize,
    runtime_note: &'static str,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    engines_ready: bool,
    runtime: &'static str,
    version: &'static str,
    gpu: HealthGpuInfo,
    ollama_model: String,
    active_model: String,
    selected_model: String,
    models: Vec<Value>,
    recommended_models: Vec<Value>,
    cloud_presets: Vec<Value>,
    unsloth_url: String,
    unsloth_api_key: &'static str,
    masked_api_key: String,
    has_unsloth_key: bool,
    auto_save_web_pages: bool,
    auto_export_chapter_book: bool,
    web_save_dir: String,
    text_filter_mode: String,
    skip_sighs: bool,
    skip_sfx: bool,
    detect_free_text: bool,
    min_text_length: u32,
    min_font_size: u32,
    max_font_size: u32,
    uppercase: bool,
    overflow_mode: String,
    batch_pages: u32,
    translation_profile: String,
    translation_profiles: Vec<Value>,
    pipeline_mode: String,
    pipeline_modes: Vec<Value>,
    use_image_context: bool,
    local_fallback_model_multilang: String,
    anti_censorship_prompt: bool,
    preflight_nsfw_split: bool,
    cloud_only_no_local_fallback: bool,
    cloud_fallback_model: String,
    ultra_proofread_mode: String,
    ui_settings: Value,
    glossary: Vec<GlossaryEntry>,
    glossary_count: usize,
    supported_source_langs: [&'static str; 4],
    default_target_lang: &'static str,
    supported_target_langs: [&'static str; 3],
    easyocr_installed: bool,
    ocr_engine_updated: bool,
    inpaint_backend: &'static str,
    font: Option<String>,
    available_fonts: Vec<FontInfo>,
    folder_job: FolderJobState,
}

async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let cfg = state.config.read().expect("config rwlock").clone();
    let job = state.folder_job.read().expect("job rwlock").clone();
    let global_glossary = state
        .db
        .load_glossary("global")
        .unwrap_or_else(|_| cfg.glossary.clone());
    let first_font = state.available_fonts.first().map(|f| f.path.clone());
    let glossary_count = global_glossary.len();

    Json(HealthResponse {
        status: "ready",
        engines_ready: true,
        runtime: "rust",
        version: APP_VERSION_RS,
        gpu: HealthGpuInfo {
            cuda_available: state.ort_inventory.cuda_ep_compiled,
            cuda_ep_compiled: state.ort_inventory.cuda_ep_compiled,
            directml_ep_compiled: state.ort_inventory.directml_ep_compiled,
            onnx_ready_count: state.ort_inventory.models.len(),
            runtime_note: "Phase 5 Full Rust Studio (Batch Folder + SSE + OCR + Cleaner + Translator + Renderer)",
        },
        ollama_model: cfg.selected_model.clone(),
        active_model: cfg.selected_model.clone(),
        selected_model: cfg.selected_model,
        models: Vec::new(),
        recommended_models: Vec::new(),
        cloud_presets: Vec::new(),
        unsloth_url: cfg.unsloth_url,
        unsloth_api_key: "",
        masked_api_key: mask_api_key(Some(&cfg.unsloth_api_key)),
        has_unsloth_key: !cfg.unsloth_api_key.trim().is_empty(),
        auto_save_web_pages: cfg.auto_save_web_pages,
        auto_export_chapter_book: cfg.auto_export_chapter_book,
        web_save_dir: cfg.web_save_dir,
        text_filter_mode: cfg.text_filter_mode,
        skip_sighs: cfg.skip_sighs,
        skip_sfx: cfg.skip_sfx,
        detect_free_text: cfg.detect_free_text,
        min_text_length: cfg.min_text_length,
        min_font_size: cfg.min_font_size,
        max_font_size: cfg.max_font_size,
        uppercase: cfg.uppercase,
        overflow_mode: cfg.overflow_mode,
        batch_pages: cfg.batch_pages,
        translation_profile: cfg.translation_profile,
        translation_profiles: Vec::new(),
        pipeline_mode: cfg.pipeline_mode,
        pipeline_modes: Vec::new(),
        use_image_context: cfg.use_image_context,
        local_fallback_model_multilang: cfg.local_fallback_model_multilang,
        anti_censorship_prompt: cfg.anti_censorship_prompt,
        preflight_nsfw_split: cfg.preflight_nsfw_split,
        cloud_only_no_local_fallback: cfg.cloud_only_no_local_fallback,
        cloud_fallback_model: cfg.cloud_fallback_model,
        ultra_proofread_mode: cfg.ultra_proofread_mode,
        ui_settings: cfg.ui_settings,
        glossary: global_glossary,
        glossary_count,
        supported_source_langs: ["auto", "ja", "ko", "zh"],
        default_target_lang: "en",
        supported_target_langs: ["en", "ru", "romaji"],
        easyocr_installed: false,
        ocr_engine_updated: true,
        inpaint_backend: "Rust Harmonic + Neural Seg (Phase 3)",
        font: first_font,
        available_fonts: state.available_fonts.clone(),
        folder_job: job,
    })
}

async fn get_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let cfg = state.config.read().expect("config rwlock").clone();
    let mut models = state.translator.list_ollama_models().await;
    if !cfg.selected_model.trim().is_empty()
        && cfg.selected_model != "auto"
        && !models.iter().any(|m| m == &cfg.selected_model)
    {
        models.insert(0, cfg.selected_model.clone());
    }
    Json(json!({
        "status": "ok",
        "runtime": "rust",
        "active_model": cfg.selected_model,
        "models": models,
    }))
}

async fn get_folder_status(State(state): State<Arc<AppState>>) -> Json<FolderJobState> {
    let job = state.folder_job.read().expect("job rwlock").clone();
    Json(job)
}

async fn stream_folder_status(
    State(state): State<Arc<AppState>>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.folder_tx.subscribe();
    let stream = futures_util::stream::unfold(
        (state, rx, false),
        |(st, mut rx, sent_initial)| async move {
            if sent_initial {
                let _ = tokio::time::timeout(Duration::from_secs(5), rx.changed()).await;
            }
            let snap = st.folder_job.read().expect("job rwlock").clone();
            let data = serde_json::to_string(&snap).unwrap_or_else(|_| "{}".to_string());
            let ev = Event::default().event("status").data(data);
            Some((Ok(ev), (st, rx, true)))
        },
    );
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(5)))
}

#[derive(Debug, Deserialize)]
struct GlossaryQuery {
    folder_path: Option<String>,
}

async fn get_glossary(
    State(state): State<Arc<AppState>>,
    Query(q): Query<GlossaryQuery>,
) -> Response {
    if let Some(ref fp) = q.folder_path {
        if let Err(err) = reject_unc_path(Some(fp), "folder_path") {
            return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
        }
    }
    let global_list = state.db.load_glossary("global").unwrap_or_default();
    let folder_scope = q
        .folder_path
        .as_deref()
        .map(|s| format!("folder:{}", s.trim().to_ascii_lowercase()))
        .unwrap_or_default();
    let folder_list = if folder_scope.is_empty() {
        Vec::new()
    } else {
        state.db.load_glossary(&folder_scope).unwrap_or_default()
    };
    let merged = merge_glossaries(&global_list, &folder_list);
    Json(json!({
        "status": "ok",
        "global_glossary": global_list,
        "folder_glossary": folder_list,
        "merged_glossary": merged,
        "sakura_text": export_glossary_to_sakura_text(&merged),
        "tsv_text": export_glossary_to_tsv(&merged),
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
struct GlossarySavePayload {
    global_glossary: Option<Vec<GlossaryEntry>>,
    folder_glossary: Option<Vec<GlossaryEntry>>,
    folder_path: Option<String>,
}

async fn save_glossary(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<GlossarySavePayload>,
) -> Response {
    if let Some(ref fp) = payload.folder_path {
        if let Err(err) = reject_unc_path(Some(fp), "folder_path") {
            return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
        }
    }
    if let Some(ref g) = payload.global_glossary {
        let norm = normalize_glossary_entries(g);
        if let Err(e) = state.db.replace_glossary("global", &norm) {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"detail": e.to_string()}))).into_response();
        }
        if let Ok(mut cfg) = state.config.write() {
            cfg.glossary = norm;
        }
    }
    if let (Some(ref fg), Some(ref fp)) = (&payload.folder_glossary, &payload.folder_path) {
        let scope = format!("folder:{}", fp.trim().to_ascii_lowercase());
        let norm_f = normalize_glossary_entries(fg);
        if let Err(e) = state.db.replace_glossary(&scope, &norm_f) {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"detail": e.to_string()}))).into_response();
        }
    }
    let global_list = state.db.load_glossary("global").unwrap_or_default();
    let folder_list = payload
        .folder_path
        .as_deref()
        .map(|fp| {
            let scope = format!("folder:{}", fp.trim().to_ascii_lowercase());
            state.db.load_glossary(&scope).unwrap_or_default()
        })
        .unwrap_or_default();
    let merged = merge_glossaries(&global_list, &folder_list);
    Json(json!({
        "status": "ok",
        "global_glossary": global_list,
        "folder_glossary": folder_list,
        "merged_glossary": merged,
        "sakura_text": export_glossary_to_sakura_text(&merged),
        "tsv_text": export_glossary_to_tsv(&merged),
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
struct SettingsPayload {
    model: Option<String>,
    selected_model: Option<String>,
    unsloth_url: Option<String>,
    unsloth_api_key: Option<String>,
    pipeline_mode: Option<String>,
    translation_profile: Option<String>,
    text_filter_mode: Option<String>,
    skip_sighs: Option<bool>,
    skip_sfx: Option<bool>,
    detect_free_text: Option<bool>,
    min_text_length: Option<u32>,
    min_font_size: Option<u32>,
    max_font_size: Option<u32>,
    uppercase: Option<bool>,
    overflow_mode: Option<String>,
    batch_pages: Option<u32>,
    auto_save_web_pages: Option<bool>,
    web_save_dir: Option<String>,
}

async fn update_settings(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SettingsPayload>,
) -> Response {
    if let Some(ref sd) = payload.web_save_dir {
        if let Err(err) = reject_unc_path(Some(sd), "web_save_dir") {
            return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
        }
    }
    let mut cfg = state.config.write().expect("config rwlock");
    if let Some(m) = payload.selected_model.or(payload.model) {
        if !m.trim().is_empty() {
            cfg.selected_model = m.trim().to_string();
        }
    }
    if let Some(u) = payload.unsloth_url {
        cfg.unsloth_url = u.trim().to_string();
    }
    if let Some(k) = payload.unsloth_api_key {
        let trimmed = k.trim();
        if !trimmed.is_empty() {
            cfg.unsloth_api_key = trimmed.to_string();
        }
    }
    if let Some(pm) = payload.pipeline_mode {
        cfg.pipeline_mode = pm;
    }
    if let Some(tp) = payload.translation_profile {
        cfg.translation_profile = tp;
    }
    if let Some(tfm) = payload.text_filter_mode {
        cfg.text_filter_mode = tfm;
    }
    if let Some(v) = payload.skip_sighs {
        cfg.skip_sighs = v;
    }
    if let Some(v) = payload.skip_sfx {
        cfg.skip_sfx = v;
    }
    if let Some(v) = payload.detect_free_text {
        cfg.detect_free_text = v;
    }
    if let Some(v) = payload.min_text_length {
        cfg.min_text_length = v.clamp(1, 50);
    }
    if let Some(v) = payload.min_font_size {
        cfg.min_font_size = v.clamp(6, 200);
    }
    if let Some(v) = payload.max_font_size {
        cfg.max_font_size = v.clamp(8, 300);
    }
    if let Some(v) = payload.uppercase {
        cfg.uppercase = v;
    }
    if let Some(v) = payload.overflow_mode {
        cfg.overflow_mode = v;
    }
    if let Some(v) = payload.batch_pages {
        cfg.batch_pages = v.clamp(1, 200);
    }
    if let Some(v) = payload.auto_save_web_pages {
        cfg.auto_save_web_pages = v;
    }
    if let Some(v) = payload.web_save_dir {
        if !v.trim().is_empty() {
            cfg.web_save_dir = v.trim().to_string();
        }
    }
    Json(json!({
        "status": "ok",
        "selected_model": cfg.selected_model,
        "pipeline_mode": cfg.pipeline_mode,
        "masked_api_key": mask_api_key(Some(&cfg.unsloth_api_key)),
    }))
    .into_response()
}

async fn get_diagnostics(State(state): State<Arc<AppState>>) -> Json<Value> {
    let metrics = state.db.metrics().unwrap_or_default();
    Json(json!({
        "status": "ok",
        "runtime": "rust",
        "version": APP_VERSION_RS,
        "startup_ms": state.startup_ms,
        "uptime_ms": state.started_at.elapsed().as_millis(),
        "fonts_count": state.available_fonts.len(),
        "onnx_ready_count": state.ort_inventory.models.len(),
        "onnx_models": state.ort_inventory.models,
        "cuda_ep_compiled": state.ort_inventory.cuda_ep_compiled,
        "directml_ep_compiled": state.ort_inventory.directml_ep_compiled,
        "sqlite": {
            "translation_cache_rows": metrics.translation_cache_rows,
            "glossary_rows": metrics.glossary_rows,
            "page_state_rows": metrics.page_state_rows,
        }
    }))
}

#[derive(Debug, Deserialize)]
struct OcrRustPayload {
    image_path: String,
    detect_free_text: Option<bool>,
    text_filter_mode: Option<String>,
    rtl: Option<bool>,
}

async fn run_ocr_rust(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<OcrRustPayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.image_path), "image_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let img_p = Path::new(payload.image_path.trim());
    if !img_p.is_file() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Файл изображения не найден"})),
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

    let opts = OcrOptions {
        detect_free_text: payload.detect_free_text.unwrap_or(false),
        text_filter_mode: payload
            .text_filter_mode
            .unwrap_or_else(|| "dialogue_only".to_string()),
        rtl: payload.rtl.unwrap_or(true),
        ..Default::default()
    };
    let img_owned = img_p.to_path_buf();
    let st_clone = state.clone();

    let res = tokio::task::spawn_blocking(move || {
        let mut guard = st_clone.ocr_engine.lock().expect("ocr mutex");
        if guard.is_none() {
            let eng = RustOcrEngine::load(&st_clone.base_dir, true)?;
            *guard = Some(eng);
        }
        guard
            .as_mut()
            .expect("initialized")
            .process_image_path(&img_owned, &opts)
    })
    .await;

    match res {
        Ok(Ok(page)) => Json(json!({
            "status": "ok",
            "runtime": "rust",
            "width": page.width,
            "height": page.height,
            "timings": page.timings,
            "blocks": page.blocks,
        }))
        .into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e.to_string()})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e.to_string()})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct CleanRustPayload {
    image_path: String,
    blocks: Option<Vec<TextBlock>>,
    out_path: Option<String>,
    dilate_size: Option<u8>,
    use_neural_segmenter: Option<bool>,
    use_cache: Option<bool>,
}

async fn run_clean_rust(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CleanRustPayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.image_path), "image_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    if let Some(ref op) = payload.out_path {
        if let Err(err) = reject_unc_path(Some(op), "out_path") {
            return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()})))
                .into_response();
        }
    }
    let img_p = Path::new(payload.image_path.trim());
    if !img_p.is_file() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Файл изображения не найден"})),
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
    if let Some(ref op) = payload.out_path {
        let out_p = Path::new(op.trim());
        let check_target = out_p.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(out_p);
        if !state.allowed_roots.is_within_allowed_roots(check_target) {
            return (
                StatusCode::FORBIDDEN,
                Json(json!({"detail": "Выходной путь вне разрешённых директорий"})),
            )
                .into_response();
        }
    }

    let img_owned = img_p.to_path_buf();
    let out_owned = payload.out_path.as_ref().map(|s| PathBuf::from(s.trim()));
    let supplied_blocks = payload.blocks;
    let opts = CleanOptions {
        dilate_size: payload.dilate_size.unwrap_or(9).clamp(1, 28),
        use_neural_segmenter: payload.use_neural_segmenter.unwrap_or(true),
        use_cache: payload.use_cache.unwrap_or(true),
        ..Default::default()
    };
    let st_clone = state.clone();

    let res = tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let rgb = image::open(&img_owned)
            .map_err(|e| format!("Ошибка декодирования изображения: {e}"))?
            .to_rgb8();
        let mtime = std::fs::metadata(&img_owned)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let blocks = match supplied_blocks {
            Some(b) if !b.is_empty() => b,
            _ => {
                let mut ocr_guard = st_clone.ocr_engine.lock().expect("ocr mutex");
                if ocr_guard.is_none() {
                    let eng = RustOcrEngine::load(&st_clone.base_dir, true)
                        .map_err(|e| e.to_string())?;
                    *ocr_guard = Some(eng);
                }
                let ocr_res = ocr_guard
                    .as_mut()
                    .expect("ocr initialized")
                    .process_rgb_image(&rgb, &OcrOptions::default())
                    .map_err(|e| e.to_string())?;
                ocr_res.blocks
            }
        };

        let mut clean_guard = st_clone.cleaner_engine.lock().expect("cleaner mutex");
        if clean_guard.is_none() {
            *clean_guard = Some(RustCleanerEngine::load(&st_clone.base_dir, true));
        }
        let src_str = img_owned.to_string_lossy().to_string();
        let clean_res = clean_guard
            .as_mut()
            .expect("cleaner initialized")
            .clean_image(&rgb, &blocks, &opts, Some(&src_str), mtime)
            .map_err(|e| e.to_string())?;

        if let Some(ref op) = out_owned {
            clean_res
                .cleaned
                .save(op)
                .map_err(|e| format!("Ошибка сохранения результата: {e}"))?;
        }

        Ok(json!({
            "status": "ok",
            "runtime": "rust",
            "width": clean_res.cleaned.width(),
            "height": clean_res.cleaned.height(),
            "blocks_cleaned": clean_res.blocks_cleaned,
            "mask_nonzero": clean_res.mask_nonzero,
            "timings": clean_res.timings,
            "out_path": out_owned.map(|p| p.to_string_lossy().to_string()),
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

async fn shutdown_server(State(state): State<Arc<AppState>>) -> Json<Value> {
    state.shutdown_notify.notify_waiters();
    Json(json!({"status": "shutting_down"}))
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(serve_ui))
        .route("/ui", get(serve_ui))
        .route("/mascot.png", get(serve_mascot))
        .route("/health", get(health_check))
        .route("/models", get(get_models))
        .route("/folder_status", get(get_folder_status))
        .route("/folder_stream", get(stream_folder_status))
        .route("/glossary", get(get_glossary).post(save_glossary))
        .route("/settings", post(update_settings))
        .route("/ocr_rust", post(run_ocr_rust))
        .route("/clean_rust", post(run_clean_rust))
        .route("/local_image", get(studio::local_image))
        .route("/folder_inspect", post(studio::folder_inspect))
        .route("/auto_diamond_wrap", post(studio::auto_diamond_wrap))
        .route("/retranslate_single_block", post(studio::retranslate_single_block))
        .route("/translate", post(studio::translate_page))
        .route("/translate_page_file", post(studio::translate_page_file))
        .route("/save_edited_page", post(studio::save_edited_page))
        .route("/page_blocks", post(folder_job::page_blocks))
        .route("/page_romaji", post(folder_job::page_romaji))
        .route("/translate_manual_region", post(folder_job::translate_manual_region))
        .route("/translate_folder", post(folder_job::translate_folder))
        .route("/folder_stop", post(folder_job::folder_stop))
        .route("/folder_control", post(folder_job::folder_control))
        .route("/folder_find_replace", post(folder_job::folder_find_replace))
        .route("/folder_report", get(folder_job::folder_report))
        .route("/chapter_book", get(folder_job::chapter_book))
        .route("/diagnostics", get(get_diagnostics))
        .route("/shutdown", post(shutdown_server))
        .layer(middleware::from_fn(local_security_middleware))
        .with_state(state)
}
