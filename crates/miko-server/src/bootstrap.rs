use crate::{fonts, AppState};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use flate2::read::DeflateDecoder;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

pub const DEFAULT_CUSTOM_ONNX_BASE_URL: &str =
    "https://github.com/dudundundudun/sloppy-manga-translator/releases/download/onnx-models-v1";

pub const HF_NAR_ONNX_URL: &str =
    "https://huggingface.co/muscgab/manga-ocr-nar-small/resolve/81ff4f3313263d1da349946a9dd20d14e0fb0e66/onnx/mangaocr_nar_small_fp32.onnx";

#[derive(Debug, Clone, Serialize)]
pub struct OnnxModelSpec {
    pub id: &'static str,
    pub filename: &'static str,
    pub title: &'static str,
    pub size_mb: f64,
    pub sha256: &'static str,
    pub source_kind: &'static str,
}

pub const REQUIRED_ONNX_MODELS: &[OnnxModelSpec] = &[
    OnnxModelSpec {
        id: "manga_ocr_nar_small",
        filename: "manga_ocr_nar_small.onnx",
        title: "Японский OCR manga-ocr-nar-small (HuggingFace оригинал)",
        size_mb: 203.22,
        sha256: "5ee628dfd40c59c522022d87c31f8f903104198877b9ba4cf795603dc540dcf5",
        source_kind: "huggingface_official",
    },
    OnnxModelSpec {
        id: "rtdetr_v2_comic",
        filename: "rtdetr_v2_comic.onnx",
        title: "Детектор бабблов и текста RT-DETR-v2",
        size_mb: 163.70,
        sha256: "93c66f583755f9e0f55971f419116f49467ff51b88f9567d121a9d8f9b45a7f6",
        source_kind: "github_onnx_pack",
    },
    OnnxModelSpec {
        id: "yolo_bubble_det",
        filename: "yolo_bubble_det.onnx",
        title: "Детектор речевых облачков YOLOv8m",
        size_mb: 98.79,
        sha256: "41ab92821b1b1e29a75efb84d760b0f3c580e87352185a4eff643a9653dd05b4",
        source_kind: "github_onnx_pack",
    },
    OnnxModelSpec {
        id: "yolo26s_seg",
        filename: "yolo26s_seg.onnx",
        title: "Сегментатор геометрии бабблов YOLO26s-seg",
        size_mb: 39.79,
        sha256: "3edc2c0a570eaa9a8996318aac970273d637b0212a6b2a94186125bb77a723fe",
        source_kind: "github_onnx_pack",
    },
    OnnxModelSpec {
        id: "comic_text_segmenter",
        filename: "comic_text_segmenter.onnx",
        title: "Нейросетевой сегментатор штрихов букв",
        size_mb: 104.07,
        sha256: "d4eb688af0f3b952951f15b66a2481a32cb1147ccb291df416785db7da6650cf",
        source_kind: "github_onnx_pack",
    },
    OnnxModelSpec {
        id: "anime_lama",
        filename: "anime_lama.onnx",
        title: "Инпейнтинг скринтонов Anime-LaMa (Real-DFT)",
        size_mb: 196.60,
        sha256: "f89c5ee0577c6b82919d853a6715252b864d08d37af8c13931d50115b876c51d",
        source_kind: "github_onnx_pack",
    },
];

#[derive(Debug, Clone)]
struct PypiWheelSpec {
    package: &'static str,
    title: &'static str,
    min_major: u64,
    max_major: u64,
    min_minor: u64,
    indicator_dll: &'static str,
    approx_mb: f64,
}

const PYPI_CUDA_WHEELS: &[PypiWheelSpec] = &[
    PypiWheelSpec {
        package: "onnxruntime-gpu",
        title: "Microsoft ONNX Runtime GPU (CUDA 13)",
        min_major: 1,
        max_major: 1,
        min_minor: 31,
        indicator_dll: "onnxruntime_providers_cuda.dll",
        approx_mb: 185.0,
    },
    PypiWheelSpec {
        package: "nvidia-cuda-runtime",
        title: "NVIDIA CUDA 13 Runtime (cudart)",
        min_major: 13,
        max_major: 13,
        min_minor: 0,
        indicator_dll: "cudart64_13.dll",
        approx_mb: 1.0,
    },
    PypiWheelSpec {
        package: "nvidia-cublas",
        title: "NVIDIA cuBLAS 13 (тензорные ядра)",
        min_major: 13,
        max_major: 13,
        min_minor: 0,
        indicator_dll: "cublas64_13.dll",
        approx_mb: 290.0,
    },
    PypiWheelSpec {
        package: "nvidia-cufft",
        title: "NVIDIA cuFFT (Фурье для Anime-LaMa)",
        min_major: 12,
        max_major: 13,
        min_minor: 0,
        indicator_dll: "cufft64_12.dll",
        approx_mb: 145.0,
    },
    PypiWheelSpec {
        package: "nvidia-cudnn-cu13",
        title: "NVIDIA cuDNN 9 (нейросетевые ядра)",
        min_major: 9,
        max_major: 9,
        min_minor: 0,
        indicator_dll: "cudnn64_9.dll",
        approx_mb: 310.0,
    },
    PypiWheelSpec {
        package: "nvidia-cuda-nvrtc",
        title: "NVIDIA NVRTC 13 (JIT компилятор ядер)",
        min_major: 13,
        max_major: 13,
        min_minor: 0,
        indicator_dll: "nvrtc64_130_0.dll",
        approx_mb: 48.0,
    },
];

const PYPI_CPU_WHEELS: &[PypiWheelSpec] = &[PypiWheelSpec {
    package: "onnxruntime",
    title: "Microsoft ONNX Runtime (CPU x64)",
    min_major: 1,
    max_major: 1,
    min_minor: 31,
    indicator_dll: "onnxruntime.dll",
    approx_mb: 14.0,
}];

#[derive(Debug, Clone, Serialize)]
pub struct DetectedGpu {
    pub has_nvidia: bool,
    pub gpu_name: String,
    pub recommended_profile: &'static str,
}

static CACHED_GPU: OnceLock<DetectedGpu> = OnceLock::new();

#[cfg(windows)]
fn run_hidden_command(program: &str, args: &[&str]) -> Option<String> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let out = Command::new(program)
        .args(args)
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

#[cfg(not(windows))]
fn run_hidden_command(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

pub fn detect_system_gpu() -> DetectedGpu {
    CACHED_GPU
        .get_or_init(|| {
            if let Some(raw) = run_hidden_command(
                "nvidia-smi",
                &["--query-gpu=name,memory.total", "--format=csv,noheader"],
            ) {
                let first_line = raw.lines().next().unwrap_or("").trim();
                if !first_line.is_empty() {
                    return DetectedGpu {
                        has_nvidia: true,
                        gpu_name: first_line.to_string(),
                        recommended_profile: "cuda",
                    };
                }
            }
            #[cfg(windows)]
            if let Some(raw) = run_hidden_command(
                "powershell",
                &[
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; '",
                ],
            ) {
                let has_nv = raw.to_ascii_lowercase().contains("nvidia");
                return DetectedGpu {
                    has_nvidia: has_nv,
                    gpu_name: raw,
                    recommended_profile: if has_nv { "cuda" } else { "cpu" },
                };
            }
            DetectedGpu {
                has_nvidia: false,
                gpu_name: "Стандартный видеоадаптер (CPU)".to_string(),
                recommended_profile: "cpu",
            }
        })
        .clone()
}

#[derive(Debug, Clone, Serialize)]
pub struct BootstrapProgress {
    pub status: String, // "idle" | "running" | "completed" | "cancelled" | "error"
    pub profile: String,
    pub phase: String,
    pub current_item: String,
    pub step_index: usize,
    pub step_total: usize,
    pub item_downloaded_mb: f64,
    pub item_total_mb: f64,
    pub progress_pct: u8,
    pub speed_mbps: f64,
    pub error: Option<String>,
    pub active_ep: Option<String>,
    pub logs: Vec<String>,
}

impl Default for BootstrapProgress {
    fn default() -> Self {
        Self {
            status: "idle".to_string(),
            profile: "cuda".to_string(),
            phase: String::new(),
            current_item: String::new(),
            step_index: 0,
            step_total: 0,
            item_downloaded_mb: 0.0,
            item_total_mb: 0.0,
            progress_pct: 0,
            speed_mbps: 0.0,
            error: None,
            active_ep: None,
            logs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaPullProgress {
    pub status: String, // "idle" | "downloading" | "completed" | "error"
    pub model: String,
    pub progress_pct: u8,
    pub message: String,
    pub error: Option<String>,
}

impl Default for OllamaPullProgress {
    fn default() -> Self {
        Self {
            status: "idle".to_string(),
            model: String::new(),
            progress_pct: 0,
            message: "Ожидание".to_string(),
            error: None,
        }
    }
}

#[derive(Deserialize)]
struct SetupSourcesOverride {
    custom_onnx_base_url: Option<String>,
}

pub fn resolve_custom_onnx_base_url(base_dir: &Path) -> String {
    if let Ok(env_url) = std::env::var("MIKO_ONNX_RELEASE_BASE_URL") {
        let t = env_url.trim().trim_end_matches('/');
        if !t.is_empty() {
            return t.to_string();
        }
    }
    let cfg_path = base_dir.join("setup_sources.json");
    if let Ok(raw) = fs::read_to_string(&cfg_path) {
        if let Ok(parsed) = serde_json::from_str::<SetupSourcesOverride>(&raw) {
            if let Some(u) = parsed.custom_onnx_base_url {
                let t = u.trim().trim_end_matches('/');
                if !t.is_empty() {
                    return t.to_string();
                }
            }
        }
    }
    DEFAULT_CUSTOM_ONNX_BASE_URL.to_string()
}

pub fn check_cuda_dlls_present(base_dir: &Path) -> bool {
    let ort_dir = base_dir.join("models").join("ort_runtime");
    if !ort_dir.join("onnxruntime.dll").is_file()
        || !ort_dir.join("onnxruntime_providers_cuda.dll").is_file()
    {
        return false;
    }
    if ort_dir.join("cublas64_13.dll").is_file() && ort_dir.join("cudnn64_9.dll").is_file() {
        return true;
    }
    if let Ok(extra) = std::env::var("MIKO_CUDA_DLL_DIR") {
        let p = Path::new(extra.trim());
        if p.join("cublas64_13.dll").is_file() {
            return true;
        }
    }
    false
}

pub fn parse_numeric_version(ver: &str) -> Option<Vec<u64>> {
    let lower = ver.trim().to_ascii_lowercase();
    if lower.is_empty()
        || lower.contains('a')
        || lower.contains('b')
        || lower.contains("rc")
        || lower.contains("dev")
        || lower.contains("post")
    {
        return None;
    }
    let mut nums = Vec::new();
    for part in lower.split('.') {
        let parsed = part.parse::<u64>().ok()?;
        nums.push(parsed);
    }
    if nums.is_empty() {
        None
    } else {
        Some(nums)
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedWheel {
    pub version: String,
    pub filename: String,
    pub url: String,
    pub sha256: Option<String>,
    pub size_bytes: u64,
}

pub fn select_best_win_wheel_from_pypi_json(
    pypi_json: &Value,
    min_major: u64,
    max_major: u64,
    min_minor: u64,
) -> Option<ResolvedWheel> {
    let releases = pypi_json.get("releases")?.as_object()?;
    let mut candidates: Vec<(Vec<u64>, String, &Vec<Value>)> = Vec::new();

    for (ver_str, files_val) in releases {
        let Some(ver_nums) = parse_numeric_version(ver_str) else {
            continue;
        };
        let major = *ver_nums.first().unwrap_or(&0);
        let minor = *ver_nums.get(1).unwrap_or(&0);
        if major < min_major || major > max_major {
            continue;
        }
        if major == min_major && minor < min_minor {
            continue;
        }
        if let Some(arr) = files_val.as_array() {
            candidates.push((ver_nums, ver_str.clone(), arr));
        }
    }

    candidates.sort_by(|a, b| b.0.cmp(&a.0));

    for (_nums, ver_str, files) in candidates {
        let mut win_wheels: Vec<&Value> = files
            .iter()
            .filter(|f| {
                let yanked = f.get("yanked").and_then(|v| v.as_bool()).unwrap_or(false);
                let fname = f.get("filename").and_then(|v| v.as_str()).unwrap_or("");
                !yanked && fname.ends_with(".whl") && fname.contains("win_amd64")
            })
            .collect();

        if win_wheels.is_empty() {
            continue;
        }
        win_wheels.sort_by_key(|f| {
            let fname = f.get("filename").and_then(|v| v.as_str()).unwrap_or("");
            if fname.contains("py3-none") {
                0
            } else if fname.contains("cp312") {
                1
            } else {
                2
            }
        });
        let chosen = win_wheels[0];
        let url = chosen.get("url").and_then(|v| v.as_str())?.to_string();
        let filename = chosen.get("filename").and_then(|v| v.as_str())?.to_string();
        let size_bytes = chosen.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
        let sha256 = chosen
            .get("digests")
            .and_then(|d| d.get("sha256"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        return Some(ResolvedWheel {
            version: ver_str,
            filename,
            url,
            sha256,
            size_bytes,
        });
    }
    None
}

fn read_u16_le(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn read_u64_le(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

/// Извлекает все `.dll` файлы из ZIP / `.whl` архива напрямую в `dest_dir` (чистый Rust + `flate2`, с поддержкой ZIP64).
pub fn extract_dlls_from_zip(zip_path: &Path, dest_dir: &Path) -> Result<Vec<String>, String> {
    fs::create_dir_all(dest_dir)
        .map_err(|e| format!("Ошибка создания папки {:?}: {e}", dest_dir))?;
    let mut file = File::open(zip_path)
        .map_err(|e| format!("Не удалось открыть архив {:?}: {e}", zip_path))?;
    let file_len = file
        .metadata()
        .map_err(|e| format!("Ошибка чтения размера архива: {e}"))?
        .len();
    if file_len < 22 {
        return Err("Архив поврежден (слишком мал)".to_string());
    }

    let tail_len = file_len.min(65_557) as usize;
    let tail_start = file_len - tail_len as u64;
    file.seek(SeekFrom::Start(tail_start))
        .map_err(|e| e.to_string())?;
    let mut tail = vec![0u8; tail_len];
    file.read_exact(&mut tail).map_err(|e| e.to_string())?;

    let eocd_rel = (0..=tail_len - 22)
        .rev()
        .find(|&i| &tail[i..i + 4] == b"PK\x05\x06")
        .ok_or_else(|| "Не найден заголовок EOCD в ZIP-архиве".to_string())?;

    let mut cd_size = read_u32_le(&tail, eocd_rel + 12) as u64;
    let mut cd_offset = read_u32_le(&tail, eocd_rel + 16) as u64;

    if (cd_offset == 0xFFFF_FFFF || cd_size == 0xFFFF_FFFF) && eocd_rel >= 20 {
        let loc = eocd_rel - 20;
        if &tail[loc..loc + 4] == b"PK\x06\x07" {
            let zip64_eocd_off = read_u64_le(&tail, loc + 8);
            file.seek(SeekFrom::Start(zip64_eocd_off))
                .map_err(|e| e.to_string())?;
            let mut z64 = [0u8; 56];
            file.read_exact(&mut z64).map_err(|e| e.to_string())?;
            if &z64[0..4] == b"PK\x06\x06" {
                cd_size = read_u64_le(&z64, 40);
                cd_offset = read_u64_le(&z64, 48);
            }
        }
    }

    file.seek(SeekFrom::Start(cd_offset))
        .map_err(|e| e.to_string())?;
    let mut cd_buf = vec![0u8; cd_size as usize];
    file.read_exact(&mut cd_buf).map_err(|e| e.to_string())?;

    struct DllEntry {
        basename: String,
        method: u16,
        comp_size: u64,
        local_header_offset: u64,
    }

    let mut entries = Vec::new();
    let mut pos = 0usize;
    while pos + 46 <= cd_buf.len() && &cd_buf[pos..pos + 4] == b"PK\x01\x02" {
        let method = read_u16_le(&cd_buf, pos + 10);
        let mut comp_size = read_u32_le(&cd_buf, pos + 20) as u64;
        let uncomp_size32 = read_u32_le(&cd_buf, pos + 24);
        let name_len = read_u16_le(&cd_buf, pos + 28) as usize;
        let extra_len = read_u16_le(&cd_buf, pos + 30) as usize;
        let comment_len = read_u16_le(&cd_buf, pos + 32) as usize;
        let mut local_off = read_u32_le(&cd_buf, pos + 42) as u64;

        let name_end = pos + 46 + name_len;
        if name_end > cd_buf.len() {
            break;
        }
        let raw_name = String::from_utf8_lossy(&cd_buf[pos + 46..name_end]).to_string();
        let extra_slice_end = (name_end + extra_len).min(cd_buf.len());
        if comp_size == 0xFFFF_FFFF || uncomp_size32 == 0xFFFF_FFFF || local_off == 0xFFFF_FFFF {
            let extra = &cd_buf[name_end..extra_slice_end];
            let mut ep = 0usize;
            while ep + 4 <= extra.len() {
                let tag = read_u16_le(extra, ep);
                let sz = read_u16_le(extra, ep + 2) as usize;
                if ep + 4 + sz > extra.len() {
                    break;
                }
                if tag == 0x0001 {
                    let mut zoff = ep + 4;
                    if uncomp_size32 == 0xFFFF_FFFF && zoff + 8 <= ep + 4 + sz {
                        zoff += 8;
                    }
                    if comp_size == 0xFFFF_FFFF && zoff + 8 <= ep + 4 + sz {
                        comp_size = read_u64_le(extra, zoff);
                        zoff += 8;
                    }
                    if local_off == 0xFFFF_FFFF && zoff + 8 <= ep + 4 + sz {
                        local_off = read_u64_le(extra, zoff);
                    }
                }
                ep += 4 + sz;
            }
        }

        if raw_name.to_ascii_lowercase().ends_with(".dll") && !raw_name.ends_with('/') {
            if let Some(base) = Path::new(&raw_name).file_name().and_then(|s| s.to_str()) {
                entries.push(DllEntry {
                    basename: base.to_string(),
                    method,
                    comp_size,
                    local_header_offset: local_off,
                });
            }
        }
        pos = name_end + extra_len + comment_len;
    }

    let mut extracted = Vec::new();
    for ent in entries {
        file.seek(SeekFrom::Start(ent.local_header_offset))
            .map_err(|e| e.to_string())?;
        let mut lh = [0u8; 30];
        file.read_exact(&mut lh).map_err(|e| e.to_string())?;
        if &lh[0..4] != b"PK\x03\x04" {
            return Err(format!("Некорректный Local Header для {}", ent.basename));
        }
        let l_name_len = read_u16_le(&lh, 26) as u64;
        let l_extra_len = read_u16_le(&lh, 28) as u64;
        let data_start = ent.local_header_offset + 30 + l_name_len + l_extra_len;
        file.seek(SeekFrom::Start(data_start))
            .map_err(|e| e.to_string())?;

        let dst_final = dest_dir.join(&ent.basename);
        let dst_part = dest_dir.join(format!("{}.part", ent.basename));
        {
            let mut out_f = File::create(&dst_part)
                .map_err(|e| format!("Ошибка записи {:?}: {e}", dst_part))?;
            let limited = (&mut file).take(ent.comp_size);
            match ent.method {
                0 => {
                    let mut reader = limited;
                    io::copy(&mut reader, &mut out_f)
                        .map_err(|e| format!("Ошибка копирования {}: {e}", ent.basename))?;
                }
                8 => {
                    let mut decoder = DeflateDecoder::new(limited);
                    io::copy(&mut decoder, &mut out_f)
                        .map_err(|e| format!("Ошибка распаковки {}: {e}", ent.basename))?;
                }
                other => {
                    return Err(format!(
                        "Неподдерживаемый метод сжатия ZIP ({other}) для {}",
                        ent.basename
                    ));
                }
            }
            out_f.flush().map_err(|e| e.to_string())?;
        }
        if dst_final.exists() {
            let _ = fs::remove_file(&dst_final);
        }
        fs::rename(&dst_part, &dst_final)
            .map_err(|e| format!("Не удалось переименовать {:?}: {e}", dst_final))?;
        extracted.push(ent.basename);
    }

    Ok(extracted)
}

pub fn compute_file_sha256(path: &Path) -> Result<String, String> {
    let mut f = File::open(path).map_err(|e| format!("Ошибка открытия {:?}: {e}", path))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn push_setup_log(state: &AppState, line: impl Into<String>) {
    let msg = line.into();
    tracing::info!("[SETUP] {}", msg);
    if let Ok(mut st) = state.bootstrap_state.write() {
        st.logs.push(msg);
        if st.logs.len() > 80 {
            let drop_n = st.logs.len() - 80;
            st.logs.drain(0..drop_n);
        }
    }
}

async fn download_with_resume_and_sha256(
    client: &reqwest::Client,
    state: &AppState,
    cancel_flag: &AtomicBool,
    url: &str,
    dest_path: &Path,
    expected_sha256: Option<&str>,
    label: &str,
    step_idx: usize,
    step_total: usize,
) -> Result<(), String> {
    if cancel_flag.load(Ordering::Relaxed) {
        return Err("CANCELLED".to_string());
    }
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Ошибка создания папки: {e}"))?;
    }
    if dest_path.is_file() {
        if let Some(exp) = expected_sha256 {
            if let Ok(actual) = compute_file_sha256(dest_path) {
                if actual.eq_ignore_ascii_case(exp) {
                    push_setup_log(state, format!("✓ Уже установлен и проверен: {label}"));
                    return Ok(());
                }
            }
        } else if dest_path.metadata().map(|m| m.len()).unwrap_or(0) > 1024 {
            return Ok(());
        }
    }

    let part_path = dest_path.with_extension("onnx_dl_part");
    let mut existing_len = part_path.metadata().map(|m| m.len()).unwrap_or(0);

    let mut req = client
        .get(url)
        .header("User-Agent", "MikoMangaStudio-Rust/7.0");
    if existing_len > 0 {
        req = req.header("Range", format!("bytes={existing_len}-"));
    }

    let mut resp = req
        .send()
        .await
        .map_err(|e| format!("Ошибка сети при скачивании {label}: {e}"))?;

    if resp.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        let _ = fs::remove_file(&part_path);
        existing_len = 0;
        resp = client
            .get(url)
            .header("User-Agent", "MikoMangaStudio-Rust/7.0")
            .send()
            .await
            .map_err(|e| format!("Ошибка повторного запроса {label}: {e}"))?;
    }

    if !resp.status().is_success() {
        return Err(format!(
            "Сервер вернул HTTP {} при скачивании {} ({})",
            resp.status(),
            label,
            url
        ));
    }

    let is_partial = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    if !is_partial {
        existing_len = 0;
    }

    let total_bytes = resp
        .content_length()
        .map(|cl| cl + existing_len)
        .unwrap_or(0);

    let mut out_file = if is_partial && existing_len > 0 {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part_path)
            .map_err(|e| format!("Ошибка открытия {:?}: {e}", part_path))?
    } else {
        File::create(&part_path).map_err(|e| format!("Ошибка создания {:?}: {e}", part_path))?
    };

    let mut downloaded = existing_len;
    let t_start = Instant::now();
    let mut last_ui_tick = Instant::now();

    while let Some(chunk) = resp.chunk().await.map_err(|e| {
        format!("Обрыв загрузки {label}: {e} (нажмите «Скачать» ещё раз для докачки)")
    })? {
        if cancel_flag.load(Ordering::Relaxed) {
            let _ = out_file.flush();
            return Err("CANCELLED".to_string());
        }
        out_file
            .write_all(&chunk)
            .map_err(|e| format!("Ошибка записи на диск: {e}"))?;
        downloaded += chunk.len() as u64;

        if last_ui_tick.elapsed() >= Duration::from_millis(140) {
            last_ui_tick = Instant::now();
            let elapsed_s = t_start.elapsed().as_secs_f64().max(0.05);
            let session_bytes = downloaded.saturating_sub(existing_len);
            let speed_mbps = (session_bytes as f64 / (1024.0 * 1024.0)) / elapsed_s;
            let dl_mb = (downloaded as f64) / (1024.0 * 1024.0);
            let tot_mb = (total_bytes as f64) / (1024.0 * 1024.0);
            let item_frac = if total_bytes > 0 {
                (downloaded as f64 / total_bytes as f64).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let overall_pct = (((step_idx as f64 - 1.0 + item_frac) / (step_total.max(1) as f64))
                * 100.0)
                .round()
                .clamp(1.0, 99.0) as u8;

            if let Ok(mut st) = state.bootstrap_state.write() {
                st.current_item = label.to_string();
                st.step_index = step_idx;
                st.step_total = step_total;
                st.item_downloaded_mb = (dl_mb * 10.0).round() / 10.0;
                st.item_total_mb = (tot_mb * 10.0).round() / 10.0;
                st.speed_mbps = (speed_mbps * 10.0).round() / 10.0;
                st.progress_pct = overall_pct;
            }
        }
    }
    out_file.flush().map_err(|e| e.to_string())?;
    drop(out_file);

    if let Some(exp) = expected_sha256 {
        if let Ok(mut st) = state.bootstrap_state.write() {
            st.current_item = format!("Проверка SHA-256: {label}");
        }
        let actual = compute_file_sha256(&part_path)?;
        if !actual.eq_ignore_ascii_case(exp) {
            let _ = fs::remove_file(&part_path);
            return Err(format!(
                "Контрольная сумма SHA-256 не совпала для {label}: ожидалось {exp}, получено {actual}"
            ));
        }
    }

    if dest_path.exists() {
        let _ = fs::remove_file(dest_path);
    }
    fs::rename(&part_path, dest_path)
        .map_err(|e| format!("Ошибка сохранения {:?}: {e}", dest_path))?;
    Ok(())
}

pub fn build_setup_status_json(state: &AppState) -> Value {
    let gpu = detect_system_gpu();
    let inv = state.get_ort_inventory();
    let onnx_dir = state.base_dir.join("models").join("onnx");
    let ort_dir = state.base_dir.join("models").join("ort_runtime");
    let ort_dll_ready = inv.ort_dylib_path.is_some() || ort_dir.join("onnxruntime.dll").is_file();
    let cuda_dlls_ready = check_cuda_dlls_present(&state.base_dir);
    let fonts_ready = fonts::has_cyrillic_custom_font(&state.base_dir.join("fonts"));

    let mut model_items = Vec::new();
    let mut ready_models_count = 0usize;
    let mut missing_models_mb = 0.0f64;
    for spec in REQUIRED_ONNX_MODELS {
        let p = onnx_dir.join(spec.filename);
        let installed = p.is_file() && p.metadata().map(|m| m.len()).unwrap_or(0) > 1_000_000;
        if installed {
            ready_models_count += 1;
        } else {
            missing_models_mb += spec.size_mb;
        }
        model_items.push(json!({
            "id": spec.id,
            "filename": spec.filename,
            "title": spec.title,
            "size_mb": spec.size_mb,
            "source_kind": spec.source_kind,
            "installed": installed,
        }));
    }

    let missing_cuda_mb: f64 = PYPI_CUDA_WHEELS
        .iter()
        .filter(|w| !ort_dir.join(w.indicator_dll).is_file())
        .map(|w| w.approx_mb)
        .sum();
    let missing_cpu_mb: f64 = if ort_dll_ready { 0.0 } else { 14.0 };

    let needs_setup = !ort_dll_ready || ready_models_count < REQUIRED_ONNX_MODELS.len();
    let ollama_cli_installed = run_hidden_command("where", &["ollama"]).is_some();
    let progress = state
        .bootstrap_state
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let cfg = state
        .config
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    json!({
        "status": "ok",
        "needs_setup": needs_setup,
        "ort_dll_ready": ort_dll_ready,
        "cuda_dlls_ready": cuda_dlls_ready,
        "fonts_ready": fonts_ready,
        "models_ready_count": ready_models_count,
        "models_total_count": REQUIRED_ONNX_MODELS.len(),
        "models": model_items,
        "gpu": gpu,
        "estimates_mb": {
            "cuda_total_mb": ((missing_models_mb + if cuda_dlls_ready { 0.0 } else { missing_cuda_mb }) * 10.0).round() / 10.0,
            "cpu_total_mb": ((missing_models_mb + missing_cpu_mb) * 10.0).round() / 10.0,
        },
        "custom_onnx_base_url": resolve_custom_onnx_base_url(&state.base_dir),
        "has_cloud_api_key": !cfg.unsloth_api_key.trim().is_empty(),
        "ollama_cli_installed": ollama_cli_installed,
        "progress": progress,
    })
}

pub async fn get_setup_status(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(build_setup_status_json(&state))
}

#[derive(Debug, Deserialize)]
pub struct SetupInstallPayload {
    pub profile: Option<String>, // "cuda" | "cpu"
    pub custom_onnx_base_url: Option<String>,
}

pub async fn post_setup_install(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SetupInstallPayload>,
) -> Response {
    {
        let st = state
            .bootstrap_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if st.status == "running" {
            return (
                StatusCode::CONFLICT,
                Json(json!({"detail": "Установка уже выполняется"})),
            )
                .into_response();
        }
    }

    if let Some(ref custom_url) = payload.custom_onnx_base_url {
        let trimmed = custom_url.trim().trim_end_matches('/');
        if !trimmed.is_empty() {
            let override_json = json!({ "custom_onnx_base_url": trimmed });
            let _ = fs::write(
                state.base_dir.join("setup_sources.json"),
                serde_json::to_string_pretty(&override_json).unwrap_or_default(),
            );
        }
    }

    let gpu = detect_system_gpu();
    let profile = payload
        .profile
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| s == "cuda" || s == "cpu")
        .unwrap_or_else(|| gpu.recommended_profile.to_string());

    state.bootstrap_cancel_flag.store(false, Ordering::SeqCst);
    {
        let mut st = state
            .bootstrap_state
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *st = BootstrapProgress {
            status: "running".to_string(),
            profile: profile.clone(),
            phase: "Подготовка к установке...".to_string(),
            current_item: "Проверка компонентов".to_string(),
            step_index: 0,
            step_total: 1,
            progress_pct: 1,
            ..Default::default()
        };
    }

    let bg_state = state.clone();
    let profile_owned = profile.clone();
    tokio::spawn(async move {
        let res = run_bootstrap_pipeline(&bg_state, &profile_owned).await;
        match res {
            Ok(active_ep) => {
                bg_state.refresh_fonts();
                bg_state.refresh_ort_inventory();
                if let Ok(mut g) = bg_state.ocr_engine.lock() {
                    *g = None;
                }
                if let Ok(mut g) = bg_state.cleaner_engine.lock() {
                    *g = None;
                }
                push_setup_log(
                    &bg_state,
                    format!("🎉 Установка успешно завершена! Активный ускоритель: {active_ep}"),
                );
                if let Ok(mut st) = bg_state.bootstrap_state.write() {
                    st.status = "completed".to_string();
                    st.phase = "Готово к работе!".to_string();
                    st.current_item = format!("Все модели установлены ({active_ep})");
                    st.progress_pct = 100;
                    st.active_ep = Some(active_ep);
                }
            }
            Err(err) if err == "CANCELLED" => {
                push_setup_log(
                    &bg_state,
                    "⏸ Загрузка приостановлена пользователем (можно продолжить с того же места).",
                );
                if let Ok(mut st) = bg_state.bootstrap_state.write() {
                    st.status = "cancelled".to_string();
                    st.phase = "Пауза".to_string();
                }
            }
            Err(err) => {
                push_setup_log(&bg_state, format!("❌ Ошибка установки: {err}"));
                if let Ok(mut st) = bg_state.bootstrap_state.write() {
                    st.status = "error".to_string();
                    st.phase = "Ошибка установки".to_string();
                    st.error = Some(err);
                }
            }
        }
    });

    Json(json!({
        "status": "started",
        "profile": profile,
    }))
    .into_response()
}

pub async fn post_setup_cancel(State(state): State<Arc<AppState>>) -> Json<Value> {
    state.bootstrap_cancel_flag.store(true, Ordering::SeqCst);
    Json(json!({"status": "cancelling"}))
}

async fn run_bootstrap_pipeline(state: &Arc<AppState>, profile: &str) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(3600))
        .build()
        .map_err(|e| format!("Ошибка HTTP-клиента: {e}"))?;

    let ort_dir = state.base_dir.join("models").join("ort_runtime");
    let onnx_dir = state.base_dir.join("models").join("onnx");
    let tmp_wheels_dir = state.base_dir.join("models").join(".tmp_wheels");
    fs::create_dir_all(&ort_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(&onnx_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(&tmp_wheels_dir).map_err(|e| e.to_string())?;

    let wheel_specs: &[PypiWheelSpec] = if profile == "cuda" {
        if check_cuda_dlls_present(&state.base_dir) {
            &[]
        } else {
            PYPI_CUDA_WHEELS
        }
    } else {
        PYPI_CPU_WHEELS
    };

    let wheels_needed: Vec<&PypiWheelSpec> = wheel_specs
        .iter()
        .filter(|w| !ort_dir.join(w.indicator_dll).is_file())
        .collect();

    let models_needed: Vec<&OnnxModelSpec> = REQUIRED_ONNX_MODELS
        .iter()
        .filter(|m| {
            let p = onnx_dir.join(m.filename);
            !p.is_file() || p.metadata().map(|md| md.len()).unwrap_or(0) < 1_000_000
        })
        .collect();

    let step_total = 1 + wheels_needed.len() + models_needed.len();
    let mut step_idx = 1usize;

    // 1. Fonts from Google Fonts
    {
        if let Ok(mut st) = state.bootstrap_state.write() {
            st.phase = "Шаг 1: Установка манга-шрифтов".to_string();
            st.current_item = "Шрифты Google Fonts (RU/EN)".to_string();
            st.step_index = step_idx;
            st.step_total = step_total;
        }
        let fonts_dir = state.base_dir.join("fonts");
        let rep = fonts::download_and_install_fonts(&fonts_dir, None, false).await;
        push_setup_log(
            state,
            format!(
                "Шрифты: установлено {}, уже было {}",
                rep.installed.len(),
                rep.skipped.len()
            ),
        );
    }

    // 2. Official ONNX Runtime + NVIDIA CUDA/cuDNN wheels from PyPI CDN
    for wspec in wheels_needed {
        step_idx += 1;
        if state.bootstrap_cancel_flag.load(Ordering::Relaxed) {
            return Err("CANCELLED".to_string());
        }
        if let Ok(mut st) = state.bootstrap_state.write() {
            st.phase = "Шаг 2: Официальный рантайм Microsoft / NVIDIA".to_string();
            st.current_item = format!("Поиск пакета {} на PyPI...", wspec.package);
            st.step_index = step_idx;
            st.step_total = step_total;
        }
        let meta_url = format!("https://pypi.org/pypi/{}/json", wspec.package);
        let meta_json: Value = client
            .get(&meta_url)
            .header("User-Agent", "MikoMangaStudio-Rust/7.0")
            .send()
            .await
            .map_err(|e| format!("Не удалось запросить PyPI для {}: {e}", wspec.package))?
            .json()
            .await
            .map_err(|e| format!("Некорректный JSON от PyPI для {}: {e}", wspec.package))?;

        let resolved = select_best_win_wheel_from_pypi_json(
            &meta_json,
            wspec.min_major,
            wspec.max_major,
            wspec.min_minor,
        )
        .ok_or_else(|| {
            format!(
                "На PyPI не найден совместимый Windows x64 пакет для {}",
                wspec.package
            )
        })?;

        push_setup_log(
            state,
            format!("Скачивание {} v{}...", wspec.title, resolved.version),
        );
        let wheel_path = tmp_wheels_dir.join(&resolved.filename);
        download_with_resume_and_sha256(
            &client,
            state,
            &state.bootstrap_cancel_flag,
            &resolved.url,
            &wheel_path,
            resolved.sha256.as_deref(),
            wspec.title,
            step_idx,
            step_total,
        )
        .await?;

        if let Ok(mut st) = state.bootstrap_state.write() {
            st.current_item = format!("Распаковка DLL: {}...", wspec.title);
        }
        let wheel_clone = wheel_path.clone();
        let ort_clone = ort_dir.clone();
        let extracted =
            tokio::task::spawn_blocking(move || extract_dlls_from_zip(&wheel_clone, &ort_clone))
                .await
                .map_err(|e| format!("Ошибка потока распаковки: {e}"))??;
        let _ = fs::remove_file(&wheel_path);
        push_setup_log(
            state,
            format!("✓ Распаковано {} DLL из {}", extracted.len(), wspec.package),
        );
    }

    let _ = fs::remove_dir_all(&tmp_wheels_dir);

    // 3. ONNX Models (HuggingFace official + GitHub Release onnx-models-v1)
    let custom_base = resolve_custom_onnx_base_url(&state.base_dir);
    for mspec in models_needed {
        step_idx += 1;
        if state.bootstrap_cancel_flag.load(Ordering::Relaxed) {
            return Err("CANCELLED".to_string());
        }
        if let Ok(mut st) = state.bootstrap_state.write() {
            st.phase = "Шаг 3: Загрузка нейросетей распознавания и очистки".to_string();
            st.current_item = mspec.title.to_string();
            st.step_index = step_idx;
            st.step_total = step_total;
        }
        let dest_onnx = onnx_dir.join(mspec.filename);
        if mspec.source_kind == "huggingface_official" {
            push_setup_log(
                state,
                format!(
                    "Скачивание {} с официального HuggingFace автора...",
                    mspec.filename
                ),
            );
            let primary_res = download_with_resume_and_sha256(
                &client,
                state,
                &state.bootstrap_cancel_flag,
                HF_NAR_ONNX_URL,
                &dest_onnx,
                Some(mspec.sha256),
                mspec.title,
                step_idx,
                step_total,
            )
            .await;
            if let Err(primary_err) = primary_res {
                if primary_err == "CANCELLED" {
                    return Err(primary_err);
                }
                let fallback_url = format!("{}/{}", custom_base, mspec.filename);
                push_setup_log(
                    state,
                    format!(
                        "HuggingFace недоступен ({primary_err}), пробуем зеркало GitHub Releases..."
                    ),
                );
                download_with_resume_and_sha256(
                    &client,
                    state,
                    &state.bootstrap_cancel_flag,
                    &fallback_url,
                    &dest_onnx,
                    Some(mspec.sha256),
                    mspec.title,
                    step_idx,
                    step_total,
                )
                .await?;
            }
        } else {
            let url = format!("{}/{}", custom_base, mspec.filename);
            push_setup_log(state, format!("Скачивание {}...", mspec.filename));
            download_with_resume_and_sha256(
                &client,
                state,
                &state.bootstrap_cancel_flag,
                &url,
                &dest_onnx,
                Some(mspec.sha256),
                mspec.title,
                step_idx,
                step_total,
            )
            .await
            .map_err(|e| {
                if e.contains("HTTP 404") {
                    format!(
                        "{e}. Проверьте, что в репозитории опубликован релиз с тегом 'onnx-models-v1' и прикреплён файл '{}'",
                        mspec.filename
                    )
                } else {
                    e
                }
            })?;
        }
    }

    // 4. Verify ONNX session & detect active ExecutionProvider
    let test_model = onnx_dir.join("yolo26s_seg.onnx");
    let prefer_gpu = profile == "cuda";
    let ep_label = tokio::task::spawn_blocking(move || {
        miko_ort::verify_onnx_model(&test_model, prefer_gpu)
            .map(|rep| rep.ep.to_string())
            .unwrap_or_else(|_| "CPUExecutionProvider".to_string())
    })
    .await
    .unwrap_or_else(|_| "CPUExecutionProvider".to_string());

    Ok(ep_label)
}

pub async fn post_install_ollama(State(state): State<Arc<AppState>>) -> Response {
    let st_clone = state.clone();
    tokio::spawn(async move {
        push_setup_log(&st_clone, "Запуск официального установщика Ollama...");
        #[cfg(windows)]
        {
            let winget_ok = run_hidden_command(
                "winget",
                &[
                    "install",
                    "-e",
                    "--id",
                    "Ollama.Ollama",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ],
            )
            .is_some();
            if !winget_ok {
                let _ = Command::new("cmd")
                    .args([
                        "/C",
                        "start",
                        "",
                        "https://ollama.com/download/OllamaSetup.exe",
                    ])
                    .spawn();
            } else {
                let _ = Command::new("cmd")
                    .args(["/C", "start", "/B", "ollama", "serve"])
                    .spawn();
            }
        }
    });
    Json(json!({
        "status": "started",
        "message": "Запущена установка Ollama (через winget / официальный сайт ollama.com)"
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct PullModelPayload {
    pub model: String,
    pub set_as_default: Option<bool>,
}

pub async fn post_pull_model(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<PullModelPayload>,
) -> Response {
    let raw_model = payload.model.trim();
    let clean_tag = raw_model
        .strip_prefix("ollama::")
        .unwrap_or(raw_model)
        .trim();
    if clean_tag.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Не указано имя модели Ollama"})),
        )
            .into_response();
    }

    {
        let st = state
            .ollama_pull_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if st.status == "downloading" {
            return (
                StatusCode::CONFLICT,
                Json(json!({"detail": "Другая модель уже скачивается"})),
            )
                .into_response();
        }
    }

    let model_owned = clean_tag.to_string();
    let set_def = payload.set_as_default.unwrap_or(true);
    {
        let mut st = state
            .ollama_pull_state
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *st = OllamaPullProgress {
            status: "downloading".to_string(),
            model: model_owned.clone(),
            progress_pct: 1,
            message: format!("Подключение к Ollama: {model_owned}..."),
            error: None,
        };
    }

    let bg_state = state.clone();
    let ret_model = model_owned.clone();
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(7200))
            .build()
            .unwrap_or_default();
        let res = client
            .post("http://127.0.0.1:11434/api/pull")
            .json(&json!({ "name": model_owned, "stream": true }))
            .send()
            .await;

        let mut resp = match res {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                if let Ok(mut st) = bg_state.ollama_pull_state.write() {
                    st.status = "error".to_string();
                    st.error = Some(format!("Ollama вернула HTTP {}", r.status()));
                    st.message = "Ошибка скачивания из Ollama".to_string();
                }
                return;
            }
            Err(e) => {
                if let Ok(mut st) = bg_state.ollama_pull_state.write() {
                    st.status = "error".to_string();
                    st.error = Some(format!("Ollama не запущена или недоступна: {e}"));
                    st.message = "Сначала установите и запустите Ollama".to_string();
                }
                return;
            }
        };

        let mut buf = Vec::new();
        while let Ok(Some(chunk)) = resp.chunk().await {
            buf.extend_from_slice(&chunk);
            while let Some(nl_pos) = buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buf.drain(..=nl_pos).collect();
                if let Ok(val) = serde_json::from_slice::<Value>(&line_bytes) {
                    if let Some(err_s) = val.get("error").and_then(|v| v.as_str()) {
                        if let Ok(mut st) = bg_state.ollama_pull_state.write() {
                            st.status = "error".to_string();
                            st.error = Some(err_s.to_string());
                            st.message = format!("Ошибка: {err_s}");
                        }
                        return;
                    }
                    let status_txt = val
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Скачивание...");
                    let completed = val.get("completed").and_then(|v| v.as_u64()).unwrap_or(0);
                    let total = val.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
                    let pct = if total > 0 {
                        ((completed as f64 / total as f64) * 100.0)
                            .round()
                            .clamp(1.0, 99.0) as u8
                    } else {
                        5
                    };
                    if let Ok(mut st) = bg_state.ollama_pull_state.write() {
                        st.progress_pct = pct;
                        st.message = format!("{status_txt} ({pct}%)");
                    }
                }
            }
        }

        if set_def {
            if let Ok(mut cfg) = bg_state.config.write() {
                cfg.selected_model = format!("ollama::{model_owned}");
                bg_state.persist_config_if_enabled(&cfg);
            }
        }
        if let Ok(mut st) = bg_state.ollama_pull_state.write() {
            st.status = "completed".to_string();
            st.progress_pct = 100;
            st.message = format!("Модель {model_owned} готова!");
        }
    });

    Json(json!({
        "status": "started",
        "model": ret_model,
    }))
    .into_response()
}

pub async fn get_pull_status(State(state): State<Arc<AppState>>) -> Json<OllamaPullProgress> {
    let st = state
        .ollama_pull_state
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    Json(st)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pypi_wheel_version_selector_skips_linux_only_release() {
        let mock_pypi = json!({
            "releases": {
                "13.10.1.6": [
                    {
                        "filename": "nvidia_cublas-13.10.1.6-py3-none-manylinux_x86_64.whl",
                        "url": "https://files.pythonhosted.org/linux.whl",
                        "size": 100,
                        "yanked": false
                    }
                ],
                "13.8.0.4": [
                    {
                        "filename": "nvidia_cublas-13.8.0.4-py3-none-win_amd64.whl",
                        "url": "https://files.pythonhosted.org/win.whl",
                        "size": 200,
                        "digests": { "sha256": "abc123" },
                        "yanked": false
                    }
                ],
                "13.9.0rc1": [
                    {
                        "filename": "nvidia_cublas-13.9.0rc1-py3-none-win_amd64.whl",
                        "url": "https://files.pythonhosted.org/rc.whl",
                        "size": 300,
                        "yanked": false
                    }
                ]
            }
        });

        let best = select_best_win_wheel_from_pypi_json(&mock_pypi, 13, 13, 0)
            .expect("must select 13.8.0.4");
        assert_eq!(best.version, "13.8.0.4");
        assert_eq!(best.sha256.as_deref(), Some("abc123"));
    }

    #[test]
    fn test_extract_dlls_from_deflate_zip() {
        use flate2::write::DeflateEncoder;
        use flate2::Compression;

        let tmp_dir = std::env::temp_dir().join(format!("miko_zip_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&tmp_dir).unwrap();
        let zip_path = tmp_dir.join("test_pkg.whl");

        let payload = b"MZ_FAKE_DLL_CONTENT_1234567890_HELLO_CUDA";
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(payload).unwrap();
        let compressed = enc.finish().unwrap();

        let name = b"nvidia/cublas/bin/test_dummy.dll";
        let mut zip_bytes = Vec::new();
        // Local file header
        zip_bytes.extend_from_slice(b"PK\x03\x04");
        zip_bytes.extend_from_slice(&20u16.to_le_bytes()); // version needed
        zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // flags
        zip_bytes.extend_from_slice(&8u16.to_le_bytes()); // deflate
        zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // mod time
        zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // mod date
        zip_bytes.extend_from_slice(&0u32.to_le_bytes()); // crc32 (unused by reader)
        zip_bytes.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        zip_bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        zip_bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // extra len
        zip_bytes.extend_from_slice(name);
        zip_bytes.extend_from_slice(&compressed);

        let cd_offset = zip_bytes.len() as u32;
        // Central directory header
        zip_bytes.extend_from_slice(b"PK\x01\x02");
        zip_bytes.extend_from_slice(&20u16.to_le_bytes()); // ver made
        zip_bytes.extend_from_slice(&20u16.to_le_bytes()); // ver needed
        zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // flags
        zip_bytes.extend_from_slice(&8u16.to_le_bytes()); // deflate
        zip_bytes.extend_from_slice(&0u16.to_le_bytes());
        zip_bytes.extend_from_slice(&0u16.to_le_bytes());
        zip_bytes.extend_from_slice(&0u32.to_le_bytes());
        zip_bytes.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        zip_bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        zip_bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // extra
        zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // comment
        zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // disk
        zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // iattr
        zip_bytes.extend_from_slice(&0u32.to_le_bytes()); // eattr
        zip_bytes.extend_from_slice(&0u32.to_le_bytes()); // local header offset 0
        zip_bytes.extend_from_slice(name);
        let cd_size = (zip_bytes.len() as u32) - cd_offset;

        // EOCD
        zip_bytes.extend_from_slice(b"PK\x05\x06");
        zip_bytes.extend_from_slice(&0u16.to_le_bytes());
        zip_bytes.extend_from_slice(&0u16.to_le_bytes());
        zip_bytes.extend_from_slice(&1u16.to_le_bytes());
        zip_bytes.extend_from_slice(&1u16.to_le_bytes());
        zip_bytes.extend_from_slice(&cd_size.to_le_bytes());
        zip_bytes.extend_from_slice(&cd_offset.to_le_bytes());
        zip_bytes.extend_from_slice(&0u16.to_le_bytes());

        fs::write(&zip_path, &zip_bytes).unwrap();
        let out_dir = tmp_dir.join("out");
        let extracted = extract_dlls_from_zip(&zip_path, &out_dir).unwrap();
        assert_eq!(extracted, vec!["test_dummy.dll"]);
        assert_eq!(fs::read(out_dir.join("test_dummy.dll")).unwrap(), payload);
        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
