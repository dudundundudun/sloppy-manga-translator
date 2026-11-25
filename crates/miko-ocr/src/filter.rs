use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FilterReason {
    pub code: String,
    pub label_ru: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OcrBackendInfo {
    pub engine: &'static str,
    pub phase: &'static str,
    pub onnx_pipeline_ready: bool,
}

fn punct_only_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^[\s\p{P}\p{S}・…ー〜\-_=+*/\\|<>「」『』（）()\[\]{}]+$").expect("valid regex")
    })
}

fn strip_punct_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[\s\p{P}・…〜\-_=+*/\\|<>「」『』（）()\[\]{}]+").expect("valid regex")
    })
}

fn digits_only_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[0-9０-９]+$").expect("valid regex"))
}

fn meaningful_char_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[\p{L}\p{N}]").expect("valid regex"))
}

pub fn is_punct_only(text: &str) -> bool {
    let t = text.trim();
    t.is_empty() || punct_only_re().is_match(t)
}

fn is_hatching_digit_noise(norm: &str) -> bool {
    let chars: Vec<char> = norm.chars().collect();
    if chars.is_empty() {
        return true;
    }
    let uniq: HashSet<char> = chars.iter().copied().collect();
    if chars.len() >= 2 && uniq.len() == 1 && matches!(chars[0], '0' | '1' | '7' | '０' | '１' | '７')
    {
        return true;
    }
    if chars.len() >= 3 && uniq.len() == 1 {
        return true;
    }
    if chars.len() >= 3 && uniq.iter().all(|c| matches!(c, '1' | '7' | '１' | '７')) {
        return true;
    }
    chars.len() >= 7
}

pub fn is_hallucinated_ocr_text(text: &str, ink_pixels: usize, is_free_text: bool) -> bool {
    let clean = text.trim();
    if clean.is_empty() || punct_only_re().is_match(clean) || !meaningful_char_re().is_match(clean)
    {
        return true;
    }
    let norm = strip_punct_re().replace_all(clean, "");
    if norm.is_empty() {
        return true;
    }
    if digits_only_re().is_match(&norm) {
        return is_free_text || is_hatching_digit_noise(&norm) || ink_pixels < 28;
    }
    let char_len = norm.chars().count();
    if char_len >= 8 && ink_pixels < char_len * 11 {
        return true;
    }
    if is_free_text && ink_pixels < 42 && char_len > 3 {
        return true;
    }
    if !is_free_text && ink_pixels < 22 && char_len > 4 {
        return true;
    }
    false
}

/// Фильтр объёма текста с сохранением осмысленных числовых реплик в уверенных диалоговых бабблах (`>= 0.78`).
pub fn classify_filter_reason(
    text: &str,
    is_free_text: bool,
    text_filter_mode: &str,
    min_text_length: usize,
    confidence: f32,
    has_enclosing_frame: bool,
) -> Option<FilterReason> {
    let core = strip_punct_re().replace_all(text.trim(), "");
    if core.is_empty() {
        return Some(FilterReason {
            code: "empty_punct".to_string(),
            label_ru: "Только знаки препинания / пустая строка".to_string(),
        });
    }
    let mode = text_filter_mode.trim().to_ascii_lowercase();
    let eff_min_len = if mode == "long_only" {
        min_text_length.max(4)
    } else if !is_free_text {
        1
    } else {
        min_text_length.max(1)
    };

    if matches!(mode.as_str(), "dialogue_only" | "bubbles_only" | "long_only") && is_free_text {
        return Some(FilterReason {
            code: "free_text_disabled".to_string(),
            label_ru: format!("Надпись вне баббла (отключено в профиле {mode})"),
        });
    }

    if mode != "all" && digits_only_re().is_match(&core) {
        let len = core.chars().count();
        let is_confident_dialogue_num = !is_free_text
            && has_enclosing_frame
            && confidence >= 0.78
            && (1..=6).contains(&len)
            && !is_hatching_digit_noise(&core);
        if !is_confident_dialogue_num {
            return Some(FilterReason {
                code: "numeric_noise".to_string(),
                label_ru: "Числовой артефакт OCR / цифры без букв".to_string(),
            });
        }
    }

    if core.chars().count() < eff_min_len {
        return Some(FilterReason {
            code: "min_length".to_string(),
            label_ru: format!("Короткий текст (< порога {eff_min_len})"),
        });
    }

    None
}
