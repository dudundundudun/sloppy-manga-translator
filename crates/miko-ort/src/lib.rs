pub use ort;
use ort::ep;
use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, ValueType};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OrtError {
    #[error("Файл модели не найден: {0}")]
    ModelNotFound(PathBuf),
    #[error("Библиотека onnxruntime.dll не найдена (положите в models/ort_runtime/onnxruntime.dll или задайте ORT_DYLIB_PATH)")]
    DylibNotFound,
    #[error("Ошибка динамической загрузки ONNX Runtime: {0}")]
    DynamicLoad(String),
    #[error("Ошибка ONNX Runtime: {0}")]
    Runtime(#[from] ort::Error),
    #[error("Неподдерживаемый вход модели '{name}': {reason}")]
    UnsupportedInput { name: String, reason: String },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProviderKind {
    Cuda,
    DirectMl,
    Cpu,
}

impl std::fmt::Display for ExecutionProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda => f.write_str("CUDAExecutionProvider"),
            Self::DirectMl => f.write_str("DmlExecutionProvider"),
            Self::Cpu => f.write_str("CPUExecutionProvider"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrtModelEntry {
    pub name: String,
    pub path: String,
    pub size_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OrtInventory {
    pub onnx_dir: String,
    pub ort_dylib_path: Option<String>,
    pub models: Vec<OrtModelEntry>,
    pub cuda_ep_compiled: bool,
    pub directml_ep_compiled: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelVerifyReport {
    pub model_name: String,
    pub ep: ExecutionProviderKind,
    pub load_ms: u128,
    pub infer_ms: u128,
    pub inputs_summary: String,
    pub outputs_summary: String,
}

static ORT_ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

#[cfg(windows)]
fn ort_dylib_filename() -> &'static str {
    "onnxruntime.dll"
}

#[cfg(not(windows))]
fn ort_dylib_filename() -> &'static str {
    "libonnxruntime.so"
}

/// Ищет `onnxruntime.dll`: сначала `ORT_DYLIB_PATH`, затем `<base_dir>/models/ort_runtime/`, затем CWD/exe.
pub fn find_ort_dylib(base_dir: Option<&Path>) -> Option<PathBuf> {
    if let Ok(env_p) = std::env::var("ORT_DYLIB_PATH") {
        let p = PathBuf::from(env_p.trim());
        if p.is_file() {
            return Some(p);
        }
    }
    let dll_name = ort_dylib_filename();
    let mut candidates = Vec::new();
    if let Some(bd) = base_dir {
        candidates.push(bd.join("models").join("ort_runtime").join(dll_name));
    }
    candidates.push(PathBuf::from("models").join("ort_runtime").join(dll_name));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join(dll_name));
            candidates.push(exe_dir.join("models").join("ort_runtime").join(dll_name));
        }
    }
    candidates.into_iter().find(|p| p.is_file())
}

#[cfg(windows)]
fn auto_discover_cuda_dll_dirs() -> Vec<PathBuf> {
    let mut found = Vec::new();
    for env_key in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(val) = std::env::var(env_key) {
            let bin = PathBuf::from(val.trim()).join("bin");
            if bin.is_dir() {
                found.push(bin);
            }
        }
    }
    let mut site_roots = Vec::new();
    if let Ok(appdata) = std::env::var("APPDATA") {
        site_roots.push(PathBuf::from(appdata).join("Python"));
    }
    if let Ok(localapp) = std::env::var("LOCALAPPDATA") {
        site_roots.push(PathBuf::from(localapp).join("Programs").join("Python"));
    }
    site_roots.push(PathBuf::from(r"C:\"));
    for root in site_roots {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if !s.to_ascii_lowercase().starts_with("python3") {
                continue;
            }
            for sub in [
                entry.path().join("site-packages").join("torch").join("lib"),
                entry
                    .path()
                    .join("Lib")
                    .join("site-packages")
                    .join("torch")
                    .join("lib"),
            ] {
                if sub.join("cublas64_13.dll").is_file() || sub.join("cudart64_13.dll").is_file() {
                    found.push(sub);
                }
            }
        }
    }
    found
}

#[cfg(windows)]
fn prepend_windows_dll_search_dirs(dll_path: &Path) {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut ort_has_cuda_libs = false;
    if let Some(parent) = dll_path.parent() {
        ort_has_cuda_libs = parent.join("cublas64_13.dll").is_file();
        dirs.push(parent.to_path_buf());
    }
    if let Ok(extra_dir) = std::env::var("MIKO_CUDA_DLL_DIR") {
        let trimmed = extra_dir.trim();
        if !trimmed.is_empty() && Path::new(trimmed).is_dir() {
            dirs.push(PathBuf::from(trimmed));
        }
    } else if !ort_has_cuda_libs {
        dirs.extend(auto_discover_cuda_dll_dirs());
    }
    if dirs.is_empty() {
        return;
    }
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut combined = std::ffi::OsString::new();
    for d in dirs {
        combined.push(d.as_os_str());
        combined.push(";");
    }
    combined.push(current);
    std::env::set_var("PATH", combined);
}

/// Инициализирует единое окружение ONNX Runtime (`load-dynamic`) для текущего процесса.
pub fn init_ort_runtime(
    base_dir: Option<&Path>,
) -> Result<&'static ort::environment::Environment, OrtError> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env);
    }
    let dll_path = find_ort_dylib(base_dir).ok_or(OrtError::DylibNotFound)?;
    #[cfg(windows)]
    prepend_windows_dll_search_dirs(&dll_path);

    let builder = ort::init_from(&dll_path).map_err(|e| OrtError::DynamicLoad(e.to_string()))?;
    let env = builder
        .with_name("miko-studio-rs")
        .with_telemetry(false)
        .build()?;
    let _ = ORT_ENV.set(env);
    Ok(ORT_ENV.get().expect("ORT_ENV initialized"))
}

pub fn scan_onnx_inventory(onnx_dir: &Path) -> OrtInventory {
    let base_dir = onnx_dir.parent().and_then(|p| p.parent());
    let dylib_path = find_ort_dylib(base_dir);
    let cuda_dll_exists = dylib_path
        .as_ref()
        .and_then(|p| p.parent())
        .is_some_and(|dir| dir.join("onnxruntime_providers_cuda.dll").is_file());

    let (cuda_ep_compiled, directml_ep_compiled) = if ORT_ENV.get().is_some() {
        use ort::ep::ExecutionProvider;
        (
            ep::CUDA::is_available().unwrap_or(cuda_dll_exists),
            ep::DirectML::is_available().unwrap_or(false),
        )
    } else {
        (cuda_dll_exists, false)
    };

    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(onnx_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let ext = p
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if ext != "onnx" {
                continue;
            }
            let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let name = p
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            models.push(OrtModelEntry {
                name,
                path: p.to_string_lossy().to_string(),
                size_mb: ((size_bytes as f64) / (1024.0 * 1024.0) * 100.0).round() / 100.0,
            });
        }
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));

    OrtInventory {
        onnx_dir: onnx_dir.to_string_lossy().to_string(),
        ort_dylib_path: dylib_path.map(|p| p.to_string_lossy().to_string()),
        models,
        cuda_ep_compiled,
        directml_ep_compiled,
    }
}

/// Создает ONNX Runtime Session с каскадным фоллбэком CUDA -> DirectML -> CPU.
pub fn create_session(
    model_path: &Path,
    prefer_gpu: bool,
) -> Result<(Session, ExecutionProviderKind), OrtError> {
    if !model_path.exists() {
        return Err(OrtError::ModelNotFound(model_path.to_path_buf()));
    }
    let base_dir = model_path
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent());
    let env = init_ort_runtime(base_dir)?;

    if prefer_gpu {
        if let Ok(builder) = Session::builder(env) {
            let cuda_ep = ep::CUDA::default()
                .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
                .build()
                .error_on_failure();
            if let Ok(mut builder) = builder.with_execution_providers([cuda_ep]) {
                if let Ok(session) = builder.commit_from_file(model_path) {
                    return Ok((session, ExecutionProviderKind::Cuda));
                }
            }
        }
        if let Ok(builder) = Session::builder(env) {
            if let Ok(mut builder) = builder
                .with_execution_providers([ep::DirectML::default().build().error_on_failure()])
            {
                if let Ok(session) = builder.commit_from_file(model_path) {
                    return Ok((session, ExecutionProviderKind::DirectMl));
                }
            }
        }
    }

    let session = Session::builder(env)?.commit_from_file(model_path)?;
    Ok((session, ExecutionProviderKind::Cpu))
}

fn resolve_dummy_shape(raw_dims: &[i64], name: &str) -> Vec<usize> {
    let n = raw_dims.len();
    raw_dims
        .iter()
        .enumerate()
        .map(|(idx, &d)| {
            if d > 0 {
                d as usize
            } else if n == 4 && idx >= 2 {
                if name.contains("image") || name.contains("mask") {
                    512
                } else {
                    640
                }
            } else {
                1
            }
        })
        .collect()
}

fn build_dummy_input<'a>(
    name: &str,
    ty: TensorElementType,
    shape: &[usize],
) -> Result<SessionInputValue<'a>, OrtError> {
    let numel: usize = shape.iter().product::<usize>().max(1);
    match ty {
        TensorElementType::Float32 => {
            let t = Tensor::from_array((shape.to_vec(), vec![0.0f32; numel]))?;
            Ok(t.into())
        }
        TensorElementType::Int32 => {
            let val = if name == "H" || name == "W" {
                1i32
            } else {
                0i32
            };
            let t = Tensor::from_array((shape.to_vec(), vec![val; numel]))?;
            Ok(t.into())
        }
        TensorElementType::Int64 => {
            let val = if name == "H" || name == "W" {
                1i64
            } else {
                0i64
            };
            let t = Tensor::from_array((shape.to_vec(), vec![val; numel]))?;
            Ok(t.into())
        }
        TensorElementType::Bool => {
            let t = Tensor::from_array((shape.to_vec(), vec![true; numel]))?;
            Ok(t.into())
        }
        TensorElementType::Uint8 => {
            let t = Tensor::from_array((shape.to_vec(), vec![0u8; numel]))?;
            Ok(t.into())
        }
        other => Err(OrtError::UnsupportedInput {
            name: name.to_string(),
            reason: format!("Неподдерживаемый тип тензора {other:?}"),
        }),
    }
}

/// Загружает ONNX-модель в Rust, прогоняет реальный инференс на синтетическом тензоре и замеряет латентность.
pub fn verify_onnx_model(
    model_path: &Path,
    prefer_gpu: bool,
) -> Result<ModelVerifyReport, OrtError> {
    let t_load = Instant::now();
    let (mut session, ep) = create_session(model_path, prefer_gpu)?;
    let load_ms = t_load.elapsed().as_millis();

    let mut inputs_desc = Vec::new();
    let mut named_inputs: Vec<(String, SessionInputValue<'static>)> = Vec::new();

    for inlet in session.inputs() {
        let name = inlet.name().to_string();
        match inlet.dtype() {
            ValueType::Tensor { ty, shape, .. } => {
                let concr_shape = resolve_dummy_shape(shape, &name);
                inputs_desc.push(format!("{name}:{concr_shape:?}"));
                let val = build_dummy_input(&name, *ty, &concr_shape)?;
                named_inputs.push((name, val));
            }
            other => {
                return Err(OrtError::UnsupportedInput {
                    name,
                    reason: format!("Неподдерживаемый ValueType {other}"),
                });
            }
        }
    }

    let t_inf = Instant::now();
    let outputs = session.run(named_inputs)?;
    let infer_ms = t_inf.elapsed().as_millis();

    let mut outputs_desc = Vec::new();
    for (out_name, out_val) in outputs.iter() {
        if let Some(shape) = out_val.dtype().tensor_shape() {
            outputs_desc.push(format!("{out_name}:{:?}", &**shape));
        } else {
            outputs_desc.push(out_name.to_string());
        }
    }

    Ok(ModelVerifyReport {
        model_name: model_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown.onnx")
            .to_string(),
        ep,
        load_ms,
        infer_ms,
        inputs_summary: inputs_desc.join(", "),
        outputs_summary: outputs_desc.join(", "),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_missing_dir_and_missing_model() {
        let missing = Path::new("nonexistent_dir_xyz_999");
        let inv = scan_onnx_inventory(missing);
        assert!(inv.models.is_empty());
        assert!(create_session(&missing.join("fake.onnx"), false).is_err());
    }
}
