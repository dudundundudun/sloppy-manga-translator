use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use thiserror::Error;
use url::Url;

pub const PRIVILEGED_ENDPOINTS: &[&str] = &[
    "/models",
    "/pull_status",
    "/model_storage",
    "/translate",
    "/translate_page_file",
    "/page_blocks",
    "/page_romaji",
    "/translate_manual_region",
    "/retranslate_single_block",
    "/auto_diamond_wrap",
    "/shorten_block_text",
    "/folder_status",
    "/folder_report",
    "/ux_telemetry/record",
    "/ux_telemetry/report",
    "/local_image",
    "/settings",
    "/delete_model",
    "/shutdown",
    "/open_folder",
    "/open_web_save_folder",
    "/pick_folder",
    "/pick_file",
    "/pick_web_save_folder",
    "/open_folder_path",
    "/folder_inspect",
    "/translate_folder",
    "/folder_stop",
    "/folder_control",
    "/folder_stream",
    "/save_edited_page",
    "/glossary",
    "/pull_model",
    "/model_stack",
    "/folder_find_replace",
    "/folder_clean_noise",
    "/open_folder_report",
    "/chapter_book",
    "/chapter_book/import",
    "/chapter_book/proofread",
    "/chapter_qa_check",
    "/chapter_qa_autofix",
    "/diagnostics",
    "/diagnostics/report",
    "/diagnostics/download_log",
    "/diagnostics/open_logs",
    "/diagnostics/clear",
    "/ux_telemetry/open_report",
    "/ux_telemetry/clear",
    "/ocr_rust",
    "/clean_rust",
    "/fonts",
    "/setup",
    "/prompt_preview",
    "/test_prompt",
    "/canon",
    "/chapter_analysis",
];

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SecurityError {
    #[error("Поле {field} содержит небезопасный UNC/сетевой путь или NUL-байт")]
    UnsafeUncPath { field: String },
    #[error("Корневая директория диска запрещена")]
    FilesystemRootForbidden,
    #[error("Доступ к пути вне разрешенных директорий запрещен")]
    OutsideAllowedRoots,
}

pub fn mask_api_key(raw_key: Option<&str>) -> String {
    let k = raw_key.unwrap_or("").trim();
    if k.is_empty() {
        return String::new();
    }
    let chars: Vec<char> = k.chars().collect();
    if chars.len() > 10 {
        let head: String = chars[..5].iter().collect();
        let tail: String = chars[chars.len() - 4..].iter().collect();
        format!("{head}...{tail}")
    } else {
        "***".to_string()
    }
}

fn is_trusted_local_hostname(host: &str) -> bool {
    let clean = host
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches('.')
        .to_ascii_lowercase();
    matches!(clean.as_str(), "127.0.0.1" | "localhost" | "::1")
}

pub fn is_trusted_local_origin(origin: Option<&str>) -> bool {
    let Some(raw) = origin else {
        return true;
    };
    let s = raw.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("null") {
        return false;
    }
    let Ok(parsed) = Url::parse(s) else {
        return false;
    };
    let scheme = parsed.scheme().to_ascii_lowercase();
    if matches!(scheme.as_str(), "chrome-extension" | "moz-extension") {
        return parsed.host_str().is_some_and(|h| !h.is_empty());
    }
    if matches!(scheme.as_str(), "http" | "https") {
        return parsed.host_str().is_some_and(is_trusted_local_hostname);
    }
    false
}

pub fn is_trusted_host_header(host: Option<&str>) -> bool {
    let Some(raw) = host else {
        return true;
    };
    let h = raw.trim();
    if h.is_empty() {
        return true;
    }
    if h.starts_with('[') {
        let Some(end_idx) = h.find(']') else {
            return false;
        };
        let after = &h[end_idx + 1..];
        if !after.is_empty() {
            if !after.starts_with(':')
                || after.len() == 1
                || !after[1..].chars().all(|c| c.is_ascii_digit())
            {
                return false;
            }
        }
        return is_trusted_local_hostname(&h[1..end_idx]);
    }
    let mut parts = h.split(':');
    let hostname = parts.next().unwrap_or("");
    if let Some(port) = parts.next() {
        if port.is_empty() || !port.chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
    }
    if parts.next().is_some() {
        return false;
    }
    is_trusted_local_hostname(hostname)
}

pub fn is_trusted_referer(referer: Option<&str>) -> bool {
    let Some(raw) = referer else {
        return true;
    };
    let s = raw.trim();
    if s.is_empty() {
        return true;
    }
    let Ok(parsed) = Url::parse(s) else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    let origin = match parsed.port() {
        Some(p) => format!("{}://{}:{}", parsed.scheme(), host, p),
        None => format!("{}://{}", parsed.scheme(), host),
    };
    is_trusted_local_origin(Some(&origin))
}

pub fn reject_unc_path(raw_path: Option<&str>, field_name: &str) -> Result<(), SecurityError> {
    let Some(s) = raw_path else {
        return Ok(());
    };
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    let normalized = trimmed.replace('/', "\\");
    if s.contains('\0') || normalized.starts_with("\\\\") {
        return Err(SecurityError::UnsafeUncPath {
            field: field_name.to_string(),
        });
    }
    Ok(())
}

pub fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_ipv4(v4),
        IpAddr::V6(v6) => is_blocked_ipv6(v6),
    }
}

fn is_blocked_ipv4(v4: Ipv4Addr) -> bool {
    let octets = v4.octets();
    let is_cgnat = octets[0] == 100 && (64..=127).contains(&octets[1]);
    v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_multicast()
        || v4.is_broadcast()
        || v4.is_unspecified()
        || v4.is_documentation()
        || is_cgnat
}

fn is_blocked_ipv6(v6: Ipv6Addr) -> bool {
    if let Some(mapped) = v6.to_ipv4_mapped() {
        return is_blocked_ipv4(mapped);
    }
    let segs = v6.segments();
    let is_ula = (segs[0] & 0xfe00) == 0xfc00;
    let is_link_local = (segs[0] & 0xffc0) == 0xfe80;
    v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() || is_ula || is_link_local
}

pub fn is_filesystem_root(path: &Path) -> bool {
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    resolved.parent().is_none()
}

#[derive(Debug, Default)]
pub struct AllowedRoots {
    roots: RwLock<HashSet<PathBuf>>,
}

impl AllowedRoots {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_root(&self, path: &Path) -> Result<PathBuf, SecurityError> {
        let raw_str = path.to_string_lossy();
        reject_unc_path(Some(&raw_str), "root")?;
        if is_filesystem_root(path) {
            return Err(SecurityError::FilesystemRootForbidden);
        }
        let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        if canon.parent().is_none() {
            return Err(SecurityError::FilesystemRootForbidden);
        }
        if let Ok(mut guard) = self.roots.write() {
            guard.insert(canon.clone());
        }
        Ok(canon)
    }

    pub fn is_within_allowed_roots(&self, candidate: &Path) -> bool {
        let raw_str = candidate.to_string_lossy();
        if reject_unc_path(Some(&raw_str), "path").is_err() {
            return false;
        }
        let canon = std::fs::canonicalize(candidate).unwrap_or_else(|_| candidate.to_path_buf());
        if canon.parent().is_none() {
            return false;
        }
        let Ok(guard) = self.roots.read() else {
            return false;
        };
        guard
            .iter()
            .any(|root| canon == *root || canon.starts_with(root))
    }
}
