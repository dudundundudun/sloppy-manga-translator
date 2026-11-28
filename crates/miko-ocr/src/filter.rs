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
        Regex::new(r"^[\s\p{P}\p{S}・…ー〜\-_=+*/\\|<>「」『』（）()\[\]{}]+$")
            .expect("valid regex")
    })
}

fn strip_punct_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[\s\p{P}\p{S}・…〜~\-_=+*/\\|<>「」『』（）()\[\]{}♡♥♪]+")
            .expect("valid regex")
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
    if chars.len() >= 2
        && uniq.len() == 1
        && matches!(chars[0], '0' | '1' | '7' | '０' | '１' | '７')
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

#[inline]
fn kata_to_hira_char(c: char) -> char {
    let cp = c as u32;
    if (0x30A1..=0x30F6).contains(&cp) {
        char::from_u32(cp - 0x60).unwrap_or(c)
    } else {
        c
    }
}

/// Белый список коротких диалоговых реплик, которые НИКОГДА не должны отбрасываться как рисованный SFX.
pub fn is_dialogue_whitelist(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.chars().any(is_kanji_char) {
        return true;
    }
    let core_orig = strip_punct_re().replace_all(trimmed, "");
    if core_orig.is_empty() {
        return false;
    }
    // Короткие вопросительные реплики (`は?`, `へ?`, `ええ?`)
    if (trimmed.contains('?') || trimmed.contains('？'))
        && matches!(
            core_orig.as_ref(),
            "は" | "へ" | "え" | "ええ" | "あ" | "ん" | "なに" | "マジ"
        )
    {
        return true;
    }
    let stripped_sokuon: String = core_orig
        .chars()
        .filter(|&c| !matches!(c, 'っ' | 'ッ' | 'ー'))
        .collect();
    // Катакана `イク` / `イクッ` / `イっちゃう` — это возглас оргазма в баббле, не фоновый SFX
    for kw in ["イク", "イっちゃ", "イキ", "イく"] {
        if core_orig.starts_with(kw) || stripped_sokuon == "イク" {
            return true;
        }
    }
    let hira: String = stripped_sokuon.chars().map(kata_to_hira_char).collect();
    matches!(
        hira.as_str(),
        "すき"
            | "だいすき"
            | "だめ"
            | "だめえ"
            | "やめ"
            | "やめて"
            | "いや"
            | "やだ"
            | "もっと"
            | "もっ"
            | "はい"
            | "うん"
            | "いい"
            | "なに"
            | "そこ"
            | "ここ"
            | "おく"
            | "まって"
    )
}

/// Чистый вздох или короткий стон без слов (`はぁ…`, `ふう…`, `んんっ`, `あっ`, `あん…`, `ひゃっ`).
pub fn is_pure_sigh_text(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.contains('?') || trimmed.contains('？') {
        return false;
    }
    if trimmed.chars().any(is_kanji_char) || trimmed.chars().any(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    let core = strip_punct_re().replace_all(trimmed, "");
    if core.is_empty() {
        return false;
    }
    let hira: String = core
        .chars()
        .filter(|&c| !matches!(c, 'っ' | 'ッ' | 'ー'))
        .map(kata_to_hira_char)
        .collect();
    if hira.is_empty() || hira.chars().count() > 6 {
        return false;
    }
    matches!(
        hira.as_str(),
        "はぁ"
            | "はあ"
            | "はぁはぁ"
            | "はあはあ"
            | "ふう"
            | "ふぅ"
            | "ふー"
            | "ん"
            | "んん"
            | "んんん"
            | "あ"
            | "ああ"
            | "あぁ"
            | "あん"
            | "ああん"
            | "ひゃ"
            | "ひゃん"
            | "ひゃあ"
            | "う"
            | "うう"
            | "うぅ"
            | "くぅ"
            | "ふぁ"
            | "ひぃ"
    )
}

const DRAWN_SFX_STEMS: &[&str] = &[
    "ちゅちる",
    "ちゅる",
    "じゅる",
    "ぢゅる",
    "ちゅぱ",
    "ちゅう",
    "じゅぽ",
    "じゅぷ",
    "ぐちゅ",
    "くちゅ",
    "ぬちゅ",
    "ずちゅ",
    "ぬぷ",
    "ずぷ",
    "ずぶ",
    "ずぼ",
    "にゅる",
    "ぬる",
    "ぱんぱん",
    "びくびく",
    "びくん",
    "びく",
    "どぴゅ",
    "びゅる",
    "どぷ",
    "ぷしゃ",
    "ごくん",
    "ごく",
    "ぺろ",
    "れろ",
    "ぴちゃ",
    "くちゃ",
    "どくん",
    "どきどき",
    "ざわ",
    "ごろん",
    "ちゅ",
    "ぎゅ",
    "びゅ",
];

fn collapse_repeated_tail_and_runs(hira: &str) -> String {
    let chars: Vec<char> = hira.chars().collect();
    if chars.is_empty() {
        return String::new();
    }
    let mut out = Vec::with_capacity(chars.len());
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        out.push(ch);
        i += 1;
        // Схлопываем тянущиеся хвосты вроде `ちゅるるるる` -> `ちゅる`
        while i < chars.len() && chars[i] == ch {
            i += 1;
        }
    }
    out.into_iter().collect()
}

fn covers_all_by_sfx_stems(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut rest = s;
    let mut steps = 0usize;
    while !rest.is_empty() {
        let mut matched_len = 0usize;
        for &stem in DRAWN_SFX_STEMS {
            if rest.starts_with(stem) && stem.len() > matched_len {
                matched_len = stem.len();
            }
        }
        if matched_len == 0 {
            return false;
        }
        rest = &rest[matched_len..];
        steps += 1;
    }
    steps >= 1
}

/// Рисованная ономатопея 18+/общая: кана из SFX-лексикона + `っ`, `ー`, `〜`, `♡`, `♥`, `!`, `…`,
/// например `ちゅるるるる～っ♡♡`, `じゅぽじゅぽ`, `ちゅちる` (OCR-искажение), `ぐちゅっ`, `びくびくっ`.
pub fn is_drawn_sfx_text(text: &str) -> bool {
    is_drawn_sfx_with_shape(text, None)
}

/// Расширенная проверка рисованного SFX с опциональным усиливающим признаком узкого вертикального бокса.
pub fn is_drawn_sfx_with_shape(text: &str, bbox: Option<[i32; 4]>) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || is_dialogue_whitelist(trimmed) || is_pure_sigh_text(trimmed) {
        return false;
    }
    if trimmed.chars().any(is_kanji_char) || trimmed.chars().any(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    let core = strip_punct_re().replace_all(trimmed, "");
    if core.is_empty() {
        return false;
    }
    let all_kana = core
        .chars()
        .all(|c| is_hiragana_char(c) || is_katakana_char(c) || matches!(c, 'ー'));
    if !all_kana {
        return false;
    }
    let hira_raw: String = core
        .chars()
        .filter(|&c| !matches!(c, 'っ' | 'ッ' | 'ー'))
        .map(kata_to_hira_char)
        .collect();
    if hira_raw.is_empty() {
        return true;
    }
    let collapsed = collapse_repeated_tail_and_runs(&hira_raw);
    let collapsed_len = collapsed.chars().count();
    if collapsed_len > 10 {
        return false;
    }

    // Прямое покрытие лексиконом SFX (`ちゅるるるる`, `じゅぽじゅぽ`, `ちゅちる`, `ぐちゅっ`, `びくびくっ`, `ドクン`)
    if covers_all_by_sfx_stems(&hira_raw) || covers_all_by_sfx_stems(&collapsed) {
        // Одиночные короткие двусимвольные корни (`ちゅ`, `ぎゅ`, `びゅ`) считаем SFX, если есть маркер звука (`っ`/`ー`/`〜`/`♡`/`♥`) или повтор
        if collapsed_len <= 2 && hira_raw.chars().count() <= 2 {
            let has_sfx_marker = trimmed
                .chars()
                .any(|c| matches!(c, 'っ' | 'ッ' | 'ー' | '〜' | '~' | '♡' | '♥'));
            if has_sfx_marker {
                return true;
            }
        } else {
            return true;
        }
    }

    // Искажённые OCR варианты длинного рисованного звука (`ちゅるる` -> `ちゅちる` и т.п.):
    // начинается с известного многоморового SFX-префикса и целиком состоит из типичных слогов SFX
    const SFX_CHAR_POOL: &[char] = &[
        'ち', 'ゅ', 'る', 'じ', 'ぢ', 'ぽ', 'ぷ', 'ぱ', 'ぐ', 'く', 'ぬ', 'ず', 'ぶ', 'ぼ', 'び',
        'ぴ', 'ど', 'ん', 'ご', 'ぺ', 'れ', 'ろ', 'し', 'ゃ',
    ];
    if (3..=6).contains(&collapsed_len)
        && ["ちゅ", "じゅ", "ぢゅ", "ぐちゅ", "くちゅ", "どぴゅ", "びゅ"]
            .iter()
            .any(|p| collapsed.starts_with(p))
        && collapsed.chars().all(|c| SFX_CHAR_POOL.contains(&c))
    {
        return true;
    }

    // Усиливающий геометрический признак: узкий высокий вертикальный столбик (`w < 160 && h > 2.5*w`)
    if let Some([x1, y1, x2, y2]) = bbox {
        let w = (x2 - x1).max(1);
        let h = (y2 - y1).max(1);
        if w < 160 && h * 2 > w * 5 && collapsed_len <= 8 {
            let pool_hits = collapsed
                .chars()
                .filter(|c| SFX_CHAR_POOL.contains(c))
                .count();
            if pool_hits * 100 >= collapsed_len * 60
                && DRAWN_SFX_STEMS.iter().any(|st| collapsed.contains(st))
            {
                return true;
            }
        }
    }

    false
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

    let should_destutter =
        suspicious_pairs >= 3 || (suspicious_pairs >= 2 && (stage1.len() >= 18 || had_run_garbage));
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
