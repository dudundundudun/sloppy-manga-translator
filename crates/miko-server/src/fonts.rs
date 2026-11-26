use crate::AppState;
use ab_glyph::FontVec;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use base64::engine::general_purpose::STANDARD as BASE64_STD;
use base64::Engine;
use miko_renderer::{list_available_fonts, FontCache, FontInfo};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
pub struct RecommendedFontSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub filename: &'static str,
    pub role: &'static str,
    pub supports_ru: bool,
    pub supports_en: bool,
    pub url: &'static str,
    pub desc: &'static str,
}

pub const RECOMMENDED_FONT_PACK: &[RecommendedFontSpec] = &[
    RecommendedFontSpec {
        id: "balsamiq-bold",
        name: "Balsamiq Sans Bold (RU/EN)",
        filename: "BalsamiqSans-Bold.ttf",
        role: "dialogue",
        supports_ru: true,
        supports_en: true,
        url: "https://raw.githubusercontent.com/google/fonts/main/ofl/balsamiqsans/BalsamiqSans-Bold.ttf",
        desc: "Мягкий комиксный гротеск для диалогов с полной поддержкой кириллицы и латиницы",
    },
    RecommendedFontSpec {
        id: "pangolin",
        name: "Pangolin (RU/EN)",
        filename: "Pangolin-Regular.ttf",
        role: "dialogue",
        supports_ru: true,
        supports_en: true,
        url: "https://raw.githubusercontent.com/google/fonts/main/ofl/pangolin/Pangolin-Regular.ttf",
        desc: "Рукописный комикс-шрифт для живых диалогов (RU + EN)",
    },
    RecommendedFontSpec {
        id: "neucha",
        name: "Neucha (RU/EN)",
        filename: "Neucha.ttf",
        role: "thought",
        supports_ru: true,
        supports_en: true,
        url: "https://raw.githubusercontent.com/google/fonts/main/ofl/neucha/Neucha.ttf",
        desc: "Аккуратный рукописный шрифт для мыслей, дневников и закадровых вставок (RU + EN)",
    },
    RecommendedFontSpec {
        id: "russo-one",
        name: "Russo One (RU/EN)",
        filename: "RussoOne-Regular.ttf",
        role: "shout",
        supports_ru: true,
        supports_en: true,
        url: "https://raw.githubusercontent.com/google/fonts/main/ofl/russoone/RussoOne-Regular.ttf",
        desc: "Жирный ударный шрифт для криков, табличек и SFX (RU + EN)",
    },
    RecommendedFontSpec {
        id: "comic-neue-bold",
        name: "Comic Neue Bold (EN)",
        filename: "ComicNeue-Bold.ttf",
        role: "dialogue",
        supports_ru: false,
        supports_en: true,
        url: "https://raw.githubusercontent.com/google/fonts/main/ofl/comicneue/ComicNeue-Bold.ttf",
        desc: "Современный чистый комиксный шрифт для английских бабблов",
    },
    RecommendedFontSpec {
        id: "bangers",
        name: "Bangers (EN)",
        filename: "Bangers-Regular.ttf",
        role: "sfx",
        supports_ru: false,
        supports_en: true,
        url: "https://raw.githubusercontent.com/google/fonts/main/ofl/bangers/Bangers-Regular.ttf",
        desc: "Классический экспрессивный комикс-шрифт для криков и звуковых эффектов (EN)",
    },
];

pub const BASE_CYRILLIC_IDS: &[&str] = &[
    "balsamiq-bold",
    "pangolin",
    "neucha",
    "russo-one",
];

#[derive(Debug, Clone, Serialize, Default)]
pub struct FontInstallReport {
    pub installed: Vec<String>,
    pub skipped: Vec<String>,
    pub failed: Vec<String>,
    pub windows_installed: Vec<String>,
}

pub fn has_cyrillic_custom_font(fonts_dir: &Path) -> bool {
    let list = list_available_fonts(fonts_dir);
    if list.is_empty() {
        return false;
    }
    let cache = FontCache::new();
    list.iter()
        .any(|f| cache.font_supports_text(Path::new(&f.path), "Привет мир"))
}

pub fn build_font_status(fonts_dir: &Path, available_fonts: &[FontInfo]) -> Value {
    let cache = FontCache::new();
    let has_custom = !available_fonts.is_empty();
    let has_cyrillic = available_fonts
        .iter()
        .any(|f| cache.font_supports_text(Path::new(&f.path), "Привет мир"));
    let first_font = available_fonts.first().map(|f| f.path.clone());
    json!({
        "has_custom_fonts": has_custom,
        "has_cyrillic_font": has_cyrillic,
        "installed_count": available_fonts.len(),
        "fonts_dir": fonts_dir.to_string_lossy().to_string(),
        "active_default_font": first_font,
    })
}

fn validate_font_bytes(bytes: &[u8], ext: &str) -> Result<(), String> {
    if bytes.len() < 1024 || bytes.len() > 20 * 1024 * 1024 {
        return Err(format!("Некорректный размер файла шрифта: {} байт", bytes.len()));
    }
    let res = if ext.eq_ignore_ascii_case("ttc") {
        FontVec::try_from_vec_and_index(bytes.to_vec(), 0)
    } else {
        FontVec::try_from_vec(bytes.to_vec())
    };
    res.map(|_| ())
        .map_err(|e| format!("Файл не является корректным TTF/OTF шрифтом: {e}"))
}

fn atomic_write_font(dst_path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = dst_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Ошибка создания папки шрифтов: {e}"))?;
    }
    let tmp_path = dst_path.with_extension("part");
    std::fs::write(&tmp_path, bytes).map_err(|e| format!("Ошибка записи шрифта: {e}"))?;
    std::fs::rename(&tmp_path, dst_path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        format!("Ошибка переименования шрифта: {e}")
    })?;
    Ok(())
}

#[cfg(windows)]
pub fn install_font_for_windows_user(src_path: &Path, display_name: &str) -> Result<(), String> {
    let local_app = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .ok_or_else(|| "Не найдена переменная LOCALAPPDATA".to_string())?;
    let win_fonts = local_app.join("Microsoft").join("Windows").join("Fonts");
    std::fs::create_dir_all(&win_fonts)
        .map_err(|e| format!("Не удалось создать папку шрифтов пользователя Windows: {e}"))?;
    let fname = src_path
        .file_name()
        .ok_or_else(|| "Пустое имя файла шрифта".to_string())?;
    let dst = win_fonts.join(fname);
    std::fs::copy(src_path, &dst).map_err(|e| format!("Ошибка копирования в Windows Fonts: {e}"))?;

    let safe_name: String = display_name
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.'))
        .collect();
    let reg_val = format!("{} (TrueType)", safe_name.trim());
    let status = std::process::Command::new("reg")
        .args([
            "add",
            r"HKCU\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
            "/v",
            &reg_val,
            "/t",
            "REG_SZ",
            "/d",
            &dst.to_string_lossy(),
            "/f",
        ])
        .output();
    match status {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => Err(format!(
            "reg add завершился с кодом {:?}",
            out.status.code()
        )),
        Err(e) => Err(format!("Не удалось запустить reg: {e}")),
    }
}

#[cfg(not(windows))]
pub fn install_font_for_windows_user(_src_path: &Path, _display_name: &str) -> Result<(), String> {
    Ok(())
}

pub async fn download_and_install_fonts(
    fonts_dir: &Path,
    font_ids: Option<&[String]>,
    install_windows_user: bool,
) -> FontInstallReport {
    let mut report = FontInstallReport::default();
    if let Err(e) = std::fs::create_dir_all(fonts_dir) {
        report.failed.push(format!("mkdir fonts/: {e}"));
        return report;
    }

    let client = match reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .user_agent("MikoStudio/7.0")
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            report.failed.push(format!("http client: {e}"));
            return report;
        }
    };

    let targets: Vec<&RecommendedFontSpec> = match font_ids {
        Some(ids) if !ids.is_empty() => RECOMMENDED_FONT_PACK
            .iter()
            .filter(|spec| ids.iter().any(|id| id.eq_ignore_ascii_case(spec.id)))
            .collect(),
        _ => RECOMMENDED_FONT_PACK.iter().collect(),
    };

    for spec in targets {
        let dst_path = fonts_dir.join(spec.filename);
        let ext = dst_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("ttf");

        let already_valid = std::fs::read(&dst_path)
            .ok()
            .is_some_and(|b| validate_font_bytes(&b, ext).is_ok());

        if already_valid {
            report.skipped.push(spec.filename.to_string());
            if install_windows_user && install_font_for_windows_user(&dst_path, spec.name).is_ok() {
                report.windows_installed.push(spec.filename.to_string());
            }
            continue;
        }

        let resp = match client.get(spec.url).send().await {
            Ok(r) => r,
            Err(e) => {
                report.failed.push(format!("{}: {e}", spec.filename));
                continue;
            }
        };
        if !resp.status().is_success() {
            report
                .failed
                .push(format!("{}: HTTP {}", spec.filename, resp.status()));
            continue;
        }
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                report.failed.push(format!("{}: read body {e}", spec.filename));
                continue;
            }
        };
        if let Err(e) = validate_font_bytes(&bytes, ext) {
            report.failed.push(format!("{}: {e}", spec.filename));
            continue;
        }
        if let Err(e) = atomic_write_font(&dst_path, &bytes) {
            report.failed.push(format!("{}: {e}", spec.filename));
            continue;
        }

        report.installed.push(spec.filename.to_string());
        if install_windows_user && install_font_for_windows_user(&dst_path, spec.name).is_ok() {
            report.windows_installed.push(spec.filename.to_string());
        }
    }

    report
}

pub fn sanitize_font_filename(raw_filename: &str) -> Result<String, String> {
    let trimmed = raw_filename.trim();
    if trimmed.is_empty()
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains("..")
        || trimmed.contains('\0')
    {
        return Err("Недопустимое имя файла шрифта".to_string());
    }
    let clean: String = trimmed
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let lower = clean.to_ascii_lowercase();
    if !(lower.ends_with(".ttf") || lower.ends_with(".otf")) {
        return Err("Разрешены только шрифты формата .ttf и .otf".to_string());
    }
    if clean.len() < 5 {
        return Err("Слишком короткое имя файла шрифта".to_string());
    }
    Ok(clean)
}

pub fn upload_custom_font(
    fonts_dir: &Path,
    raw_filename: &str,
    data_base64: &str,
    install_windows_user: bool,
) -> Result<FontInfo, String> {
    let safe_name = sanitize_font_filename(raw_filename)?;
    let b64_clean = data_base64
        .trim()
        .split_once(',')
        .map(|(_, rest)| rest)
        .unwrap_or(data_base64.trim());
    if b64_clean.len() > 21 * 1024 * 1024 {
        return Err("Размер файла шрифта превышает лимит 15 МБ".to_string());
    }
    let bytes = BASE64_STD
        .decode(b64_clean)
        .map_err(|e| format!("Ошибка декодирования base64: {e}"))?;
    if bytes.len() > 15 * 1024 * 1024 {
        return Err("Размер файла шрифта превышает лимит 15 МБ".to_string());
    }
    let ext = Path::new(&safe_name)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("ttf");
    validate_font_bytes(&bytes, ext)?;

    let dst_path = fonts_dir.join(&safe_name);
    atomic_write_font(&dst_path, &bytes)?;

    let stem = Path::new(&safe_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&safe_name)
        .to_string();
    if install_windows_user {
        let _ = install_font_for_windows_user(&dst_path, &stem);
    }

    Ok(FontInfo {
        id: safe_name.clone(),
        name: stem,
        filename: safe_name,
        path: dst_path.to_string_lossy().to_string(),
    })
}

pub async fn get_fonts_catalog(State(state): State<Arc<AppState>>) -> Json<Value> {
    let fonts_dir = state.base_dir.join("fonts");
    let available = state.refresh_fonts();
    let status = build_font_status(&fonts_dir, &available);
    let catalog: Vec<Value> = RECOMMENDED_FONT_PACK
        .iter()
        .map(|spec| {
            let is_installed = available
                .iter()
                .any(|f| f.filename.eq_ignore_ascii_case(spec.filename));
            json!({
                "id": spec.id,
                "name": spec.name,
                "filename": spec.filename,
                "role": spec.role,
                "supports_ru": spec.supports_ru,
                "supports_en": spec.supports_en,
                "desc": spec.desc,
                "installed": is_installed,
            })
        })
        .collect();
    Json(json!({
        "status": "ok",
        "catalog": catalog,
        "available_fonts": available,
        "font_status": status,
    }))
}

#[derive(Debug, Deserialize)]
pub struct InstallFontsPayload {
    pub font_ids: Option<Vec<String>>,
    pub install_windows_user: Option<bool>,
}

pub async fn install_fonts(
    State(state): State<Arc<AppState>>,
    body: Option<Json<InstallFontsPayload>>,
) -> Json<Value> {
    let payload = body.map(|Json(p)| p).unwrap_or(InstallFontsPayload {
        font_ids: None,
        install_windows_user: Some(false),
    });
    let fonts_dir = state.base_dir.join("fonts");
    let win_install = payload.install_windows_user.unwrap_or(false);
    let report =
        download_and_install_fonts(&fonts_dir, payload.font_ids.as_deref(), win_install).await;
    let available = state.refresh_fonts();
    let status = build_font_status(&fonts_dir, &available);

    Json(json!({
        "status": if report.failed.is_empty() || !report.installed.is_empty() || !report.skipped.is_empty() { "ok" } else { "error" },
        "installed": report.installed,
        "skipped": report.skipped,
        "failed": report.failed,
        "windows_installed": report.windows_installed,
        "available_fonts": available,
        "font_status": status,
    }))
}

#[derive(Debug, Deserialize)]
pub struct UploadFontPayload {
    pub filename: String,
    pub data_base64: String,
    pub install_windows_user: Option<bool>,
}

pub async fn upload_font(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<UploadFontPayload>,
) -> Response {
    let fonts_dir = state.base_dir.join("fonts");
    let win_install = payload.install_windows_user.unwrap_or(false);
    match upload_custom_font(&fonts_dir, &payload.filename, &payload.data_base64, win_install) {
        Ok(info) => {
            let available = state.refresh_fonts();
            let status = build_font_status(&fonts_dir, &available);
            Json(json!({
                "status": "ok",
                "font": info,
                "available_fonts": available,
                "font_status": status,
            }))
            .into_response()
        }
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": "error", "detail": err })),
        )
            .into_response(),
    }
}
