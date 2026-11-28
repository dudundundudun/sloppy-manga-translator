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

#[inline]
fn is_katakana_char(c: char) -> bool {
    ('\u{30A0}'..='\u{30FF}').contains(&c)
}

#[inline]
fn is_hiragana_char(c: char) -> bool {
    ('\u{3040}'..='\u{309F}').contains(&c)
}

#[inline]
fn is_kanji_char(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c) || ('\u{3400}'..='\u{4DBF}').contains(&c)
}

/// Определяет типичные звукоподражания / SFX (`ドクン`, `ざわっ`, `ぎゅっ`, `ごろん`, `どきどき`),
/// которые не нужно переводить поверх фона как реплики диалога.
pub fn is_likely_sfx_text(text: &str) -> bool {
    let core = strip_punct_re().replace_all(text.trim(), "");
    if core.is_empty() {
        return false;
    }
    let body: Vec<char> = core
        .chars()
        .filter(|&c| !matches!(c, 'っ' | 'ッ' | 'ー' | '〜'))
        .collect();
    if body.is_empty() {
        return true;
    }
    let len = body.len();
    let all_kata = body.iter().all(|&c| is_katakana_char(c));
    let all_hira = body.iter().all(|&c| is_hiragana_char(c));

    if all_kata && len <= 6 {
        return true;
    }
    if all_hira && len <= 4 {
        return true;
    }
    if (all_kata || all_hira) && (len == 4 || len == 6 || len == 8) {
        let half = len / 2;
        if body[..half] == body[half..] {
            return true;
        }
        if len >= 4 && body.chunks(2).all(|ch| ch == &body[..2]) {
            return true;
        }
    }
    false
}

/// Отличает связные диалоги/мысли поверх скринтона (`どうせ未来に帰るんだし…`, `適当な事してもバレないんじゃね？`)
/// от шумовых звуков (`ドクン`, `ざわっ`) и коротких галлюцинаций OCR (`ドる`).
pub fn is_screentone_dialogue_text(text: &str) -> bool {
    if is_punct_only(text) || is_likely_sfx_text(text) {
        return false;
    }
    let core = strip_punct_re().replace_all(text.trim(), "");
    if core.is_empty() || digits_only_re().is_match(&core) {
        return false;
    }
    let char_len = core.chars().count();
    let has_kanji = core.chars().any(is_kanji_char);
    (has_kanji && char_len >= 4) || char_len >= 7
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

    if is_free_text {
        match mode.as_str() {
            "bubbles_only" | "long_only" => {
                return Some(FilterReason {
                    code: "free_text_disabled".to_string(),
                    label_ru: format!("Надпись вне баббла (отключено в профиле {mode})"),
                });
            }
            "dialogue_only" => {
                if !(confidence >= 0.45 && is_screentone_dialogue_text(text)) {
                    return Some(FilterReason {
                        code: "free_text_disabled".to_string(),
                        label_ru: "Звук / короткая надпись вне баббла (профиль dialogue_only)"
                            .to_string(),
                    });
                }
            }
            "bubbles_no_sighs" => {
                if is_likely_sfx_text(text) {
                    return Some(FilterReason {
                        code: "sfx_filtered".to_string(),
                        label_ru: "Звуковой эффект / ономатопея (SFX)".to_string(),
                    });
                }
            }
            _ => {}
        }
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

fn is_natural_jp_double_char(c: char) -> bool {
    matches!(
        c,
        'あ' | 'い'
            | 'う'
            | 'え'
            | 'お'
            | 'ぁ'
            | 'ぃ'
            | 'ぅ'
            | 'ぇ'
            | 'ぉ'
            | 'ん'
            | 'っ'
            | 'ア'
            | 'イ'
            | 'ウ'
            | 'エ'
            | 'オ'
            | 'ン'
            | 'ッ'
            | 'ー'
            | '〜'
            | '~'
            | '…'
            | '・'
            | '.'
            | '!'
            | '?'
            | '！'
            | '？'
            | 'こ'
            | 'ま'
            | '人'
            | '時'
            | '日'
            | '我'
    )
}

fn is_expressive_scream_char(c: char) -> bool {
    matches!(
        c,
        'あ' | 'い'
            | 'う'
            | 'え'
            | 'お'
            | 'ぁ'
            | 'ぃ'
            | 'ぅ'
            | 'ぇ'
            | 'ぉ'
            | 'ん'
            | 'っ'
            | 'ア'
            | 'イ'
            | 'ウ'
            | 'エ'
            | 'オ'
            | 'ン'
            | 'ッ'
            | 'ー'
            | '〜'
            | '~'
            | '…'
            | '.'
            | '!'
            | '?'
            | '！'
            | '？'
    )
}

/// Схлопывает аномальное парное заикание и вырожденные хвосты `manga-ocr-nar-small`
/// на переполненных длинных горизонтальных строках (например, `'今今ははチチュームを5着描きまたたが'` -> `'今はチチュームを5着描きたが'`,
/// `'これれら着着て'` -> `'これら着て'`, `'24444422ahhhhh0000000...ののののの'` -> без мусорных серий),
/// сохраняя нетронутыми обычные реплики и эмоциональные возгласы (`'おおおおぉ!!'`, `'ああ…'`).
pub fn sanitize_nar_stutter(text: &str) -> String {
    let trimmed = text.trim();
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.len() < 10 {
        return trimmed.to_string();
    }

    // 1. Срезаем вырожденные серии по >= 4 одинаковых не-экспрессивных символов подряд (`44444`, `hhhhh`, `0000000`, `ののののの`, `そそそそそ`)
    let mut stage1: Vec<char> = Vec::with_capacity(chars.len());
    let mut idx = 0usize;
    let mut had_run_garbage = false;
    while idx < chars.len() {
        let ch = chars[idx];
        let mut run_end = idx + 1;
        while run_end < chars.len() && chars[run_end] == ch {
            run_end += 1;
        }
        let run_len = run_end - idx;
        if run_len >= 4 && !is_expressive_scream_char(ch) {
            had_run_garbage = true;
            if ch.is_ascii_alphanumeric() {
                // Вырожденный ASCII/числовой шум (`24444422ahhhhh0000000`) пропускаем до одного символа или убираем буквенный спам
                if ch.is_ascii_digit() {
                    stage1.push(ch);
                }
            } else {
                stage1.push(ch);
            }
        } else {
            stage1.extend_from_slice(&chars[idx..run_end]);
        }
        idx = run_end;
    }

    // 2. Проверяем плотность неестественных парных заиканий (`今今`, `はは`, `のの`, `でで`, `るる`, `着着`, `これれら`)
    let mut suspicious_pairs = 0usize;
    let mut i = 0usize;
    while i + 1 < stage1.len() {
        let ch = stage1[i];
        if ch == stage1[i + 1] && !is_natural_jp_double_char(ch) && !ch.is_ascii_digit() {
            suspicious_pairs += 1;
            i += 2;
        } else {
            i += 1;
        }
    }

    let should_destutter = suspicious_pairs >= 3
        || (suspicious_pairs >= 2 && (stage1.len() >= 18 || had_run_garbage));
    if !should_destutter {
        return stage1.into_iter().collect::<String>().trim().to_string();
    }

    let mut out = String::with_capacity(stage1.len());
    let mut k = 0usize;
    while k < stage1.len() {
        let ch = stage1[k];
        if k + 1 < stage1.len()
            && ch == stage1[k + 1]
            && !is_natural_jp_double_char(ch)
            && !ch.is_ascii_digit()
        {
            out.push(ch);
            k += 2;
            while k < stage1.len() && stage1[k] == ch {
                k += 1;
            }
        } else {
            out.push(ch);
            k += 1;
        }
    }
    out.trim().to_string()
}
