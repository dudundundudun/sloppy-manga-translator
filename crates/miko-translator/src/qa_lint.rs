use crate::canon::{CanonCategory, CanonDb};
use crate::names::builtin_fandom_db;
use crate::prompts::has_foreign_script_leak;
use miko_core::GlossaryEntry;
use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QaMode {
    Off,
    Fix,
    FixAndRetry,
}

impl Default for QaMode {
    fn default() -> Self {
        Self::FixAndRetry
    }
}

impl QaMode {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "disabled" | "none" => Self::Off,
            "fix" | "fix_only" => Self::Fix,
            _ => Self::FixAndRetry,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fix => "fix",
            Self::FixAndRetry => "fix_and_retry",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QaKind {
    NameDrift,
    CanonForbidden,
    GrammarAnchor,
    StyleBan,
    UntranslatedJp,
}

impl QaKind {
    pub fn inspector_kind(&self) -> &'static str {
        match self {
            Self::NameDrift => "name_drift",
            Self::CanonForbidden => "canon_slang",
            Self::GrammarAnchor => "grammar_anchor",
            Self::StyleBan => "style_ban",
            Self::UntranslatedJp => "canon_slang",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QaIssue {
    pub kind: QaKind,
    pub rule_id: String,
    pub message: String,
    pub fixed: bool,
    pub hint_for_retry: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QaResult {
    pub text: String,
    pub issues: Vec<QaIssue>,
    pub needs_retry: bool,
}

fn cap_word_token_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[A-Z\u{00C0}-\u{00DE}][a-zA-Z\u{00C0}-\u{017F}'’\-]{2,}")
            .expect("valid token regex")
    })
}

fn levenshtein_ascii_ci(a: &str, b: &str) -> usize {
    let ac: Vec<char> = a.chars().flat_map(|c| c.to_lowercase()).collect();
    let bc: Vec<char> = b.chars().flat_map(|c| c.to_lowercase()).collect();
    let m = ac.len();
    let n = bc.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = usize::from(ac[i - 1] != bc[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

fn common_prefix_len_ci(a: &str, b: &str) -> usize {
    a.chars()
        .flat_map(|c| c.to_lowercase())
        .zip(b.chars().flat_map(|c| c.to_lowercase()))
        .take_while(|(x, y)| x == y)
        .count()
}

fn contains_term_word_ci(haystack: &str, term: &str) -> bool {
    let t = term.trim();
    if t.is_empty() {
        return true;
    }
    let pat = format!(r"(?i)\b{}(?:s|es|'s|’s)?\b", regex::escape(t));
    if let Ok(re) = Regex::new(&pat) {
        if re.is_match(haystack) {
            return true;
        }
    }
    haystack.to_lowercase().contains(&t.to_lowercase())
}

fn apply_case_preserving_regex(re: &Regex, text: &str, replacement_tpl: &str) -> String {
    re.replace_all(text, |caps: &regex::Captures<'_>| {
        let matched = caps.get(0).map(|m| m.as_str()).unwrap_or("");
        let mut expanded = String::new();
        caps.expand(replacement_tpl, &mut expanded);

        let matched_starts_upper = matched
            .chars()
            .find(|c| c.is_alphabetic())
            .is_some_and(|c| c.is_uppercase());
        let repl_starts_lower = expanded
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase());

        if matched_starts_upper && repl_starts_lower {
            let mut chs = expanded.chars();
            if let Some(first) = chs.next() {
                return format!("{}{}", first.to_ascii_uppercase(), chs.as_str());
            }
        }
        expanded
    })
    .into_owned()
}

fn lookup_known_mistranslations(jp_src: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for s in &builtin_fandom_db().series {
        if let Some(list) = s.known_mistranslations.get(jp_src) {
            for item in list {
                if !out.contains(item) {
                    out.push(item.clone());
                }
            }
        }
    }
    // Сначала более длинные фразы (`"the bleach"` раньше `"bleach"`), чтобы артикль не оставался висеть
    out.sort_by(|a, b| b.len().cmp(&a.len()));
    out
}

fn try_fix_known_mistranslation(text: &str, jp_src: &str, dst: &str) -> Option<String> {
    let bad_list = lookup_known_mistransations_cached(jp_src);
    if bad_list.is_empty() {
        return None;
    }
    let mut cur = text.to_string();
    let mut changed = false;
    for bad in bad_list {
        let pat = format!(r"\b{}\b", regex::escape(&bad));
        if let Ok(re) = RegexBuilder::new(&pat).case_insensitive(true).build() {
            if re.is_match(&cur) {
                cur = apply_case_preserving_regex(&re, &cur, dst);
                changed = true;
            }
        }
    }
    if changed {
        Some(cur)
    } else {
        None
    }
}

fn lookup_known_mistransations_cached(jp_src: &str) -> Vec<String> {
    lookup_known_mistranslations(jp_src)
}

fn try_fix_fuzzy_capitalized_name(text: &str, target_name: &str) -> Option<String> {
    if target_name.chars().count() < 4 {
        return None;
    }
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();
    for m in cap_word_token_regex().find_iter(text) {
        let tok = m.as_str();
        let (stem, suffix) = if let Some(s) = tok.strip_suffix("'s") {
            (s, "'s")
        } else if let Some(s) = tok.strip_suffix("’s") {
            (s, "’s")
        } else {
            (tok, "")
        };
        if stem.eq_ignore_ascii_case(target_name) {
            continue;
        }
        let pref = common_prefix_len_ci(stem, target_name);
        let dist = levenshtein_ascii_ci(stem, target_name);
        if pref >= 3 && dist <= 2 {
            replacements.push((m.start(), m.end(), format!("{target_name}{suffix}")));
        }
    }
    if replacements.is_empty() {
        return None;
    }
    let mut out = text.to_string();
    for (start, end, repl) in replacements.into_iter().rev() {
        out.replace_range(start..end, &repl);
    }
    Some(out)
}

fn starts_with_latin_uppercase(s: &str) -> bool {
    s.chars().next().is_some_and(|c| {
        c.is_uppercase() && (c.is_ascii_alphabetic() || ('\u{00C0}'..='\u{017F}').contains(&c))
    })
}

/// Проверяет переведённую английскую реплику против глоссария имён, базы 18+ канона и грамматических якорей.
pub fn lint_pair(
    src_jp: &str,
    tr_en: &str,
    glossary: &[GlossaryEntry],
    canon: &CanonDb,
) -> QaResult {
    let mut cur_text = tr_en.trim().to_string();
    let mut issues: Vec<QaIssue> = Vec::new();
    let mut needs_retry = false;

    if src_jp.trim().is_empty() || cur_text.is_empty() {
        return QaResult {
            text: cur_text,
            issues,
            needs_retry: false,
        };
    }

    // 1. NameDrift + Fandom Terms
    for g in glossary {
        let g_src = g.src.trim();
        let g_dst = g.dst.trim();
        if g_src.is_empty() || g_dst.is_empty() || !src_jp.contains(g_src) {
            continue;
        }
        let is_proper_name = starts_with_latin_uppercase(g_dst);
        let is_fandom_term = g.note.starts_with("auto-lock:fandom-term:")
            || !lookup_known_mistranslations(g_src).is_empty();
        if !is_proper_name && !is_fandom_term {
            continue;
        }
        if contains_term_word_ci(&cur_text, g_dst) {
            continue;
        }

        // Сначала проверяем известные ложные друзья переводчика (`"the bleach"` -> `"Heiter"`, `"Rough, monk"` -> `"Corrupt priest"`)
        if let Some(fixed_mistr) = try_fix_known_mistranslation(&cur_text, g_src, g_dst) {
            cur_text = fixed_mistr;
            issues.push(QaIssue {
                kind: QaKind::NameDrift,
                rule_id: format!("name_mistranslation:{g_src}"),
                message: format!("Исправлена буквальная ошибка имени/термина {g_src} → {g_dst}"),
                fixed: true,
                hint_for_retry: None,
            });
            continue;
        }

        // Затем ищем искажённые имена по Левенштейну <= 2 и общему префиксу >= 3 (`Himm` / `Himmle` -> `Himmel`)
        if is_proper_name {
            if let Some(fixed_fuzzy) = try_fix_fuzzy_capitalized_name(&cur_text, g_dst) {
                cur_text = fixed_fuzzy;
                issues.push(QaIssue {
                    kind: QaKind::NameDrift,
                    rule_id: format!("name_fuzzy:{g_src}"),
                    message: format!("Исправлена опечатка в имени {g_src} → {g_dst}"),
                    fixed: true,
                    hint_for_retry: None,
                });
                continue;
            }
        }

        // Если это общий короткий титул вроде `勇者 => Hero`, а в строке есть более длинный `勇者一行`, не спамим ложным retry
        if g_src == "勇者"
            && src_jp.contains("勇者一行")
            && contains_term_word_ci(&cur_text, "Hero")
        {
            continue;
        }

        needs_retry = true;
        let hint = if is_proper_name {
            format!("Must use the exact name \"{g_dst}\" for {g_src}.")
        } else {
            format!("Must translate {g_src} as \"{g_dst}\".")
        };
        issues.push(QaIssue {
            kind: QaKind::NameDrift,
            rule_id: format!("name_missing:{g_src}"),
            message: format!("Потеряно или искажено каноничное имя/термин {g_src} → {g_dst}"),
            fixed: false,
            hint_for_retry: Some(hint),
        });
    }

    // 2. CanonForbidden + GrammarAnchor + StyleBan
    for entry in &canon.entries {
        match entry.category {
            CanonCategory::StyleBan => {
                if entry.id == "style_lol"
                    && (src_jp.contains('笑') || src_jp.contains('ｗ') || src_jp.contains('w'))
                {
                    continue;
                }
                let matched_ban = entry.forbid_en.iter().any(|re| re.is_match(&cur_text));
                if matched_ban {
                    issues.push(QaIssue {
                        kind: QaKind::StyleBan,
                        rule_id: entry.id.clone(),
                        message: format!(
                            "Нежелательный современный сленг в классическом 18+ регистре ({})",
                            entry.id
                        ),
                        fixed: false,
                        hint_for_retry: None,
                    });
                }
            }
            CanonCategory::Grammar => {
                if !entry.matches_jp(src_jp) || entry.required_en_any.is_empty() {
                    continue;
                }
                let has_anchor = entry
                    .required_en_any
                    .iter()
                    .any(|re| re.is_match(&cur_text));
                if !has_anchor {
                    needs_retry = true;
                    let hint = if entry.id == "grammar_igai" {
                        "The source says \"everyone EXCEPT / other than\" (以外) — preserve the exception using 'except' or 'other than'.".to_string()
                    } else if !entry.note.is_empty() {
                        entry.note.clone()
                    } else {
                        format!("Preserve grammar anchor: {}", entry.en)
                    };
                    issues.push(QaIssue {
                        kind: QaKind::GrammarAnchor,
                        rule_id: entry.id.clone(),
                        message: format!(
                            "Потеряна грамматическая конструкция исключения ({})",
                            entry.id
                        ),
                        fixed: false,
                        hint_for_retry: Some(hint),
                    });
                }
            }
            _ => {
                if entry.forbid_en.is_empty() || !entry.matches_jp(src_jp) {
                    continue;
                }
                let hit_forbidden = entry.forbid_en.iter().any(|re| re.is_match(&cur_text));
                if !hit_forbidden {
                    continue;
                }

                let before = cur_text.clone();
                for (af_re, af_to) in &entry.auto_fix {
                    if af_re.is_match(&cur_text) {
                        cur_text = apply_case_preserving_regex(af_re, &cur_text, af_to);
                    }
                }
                let still_forbidden = entry.forbid_en.iter().any(|re| re.is_match(&cur_text));
                if !still_forbidden && cur_text != before {
                    issues.push(QaIssue {
                        kind: QaKind::CanonForbidden,
                        rule_id: entry.id.clone(),
                        message: format!(
                            "Автозамена неканоничного 18+ сленга по правилу {} → «{}»",
                            entry.id, entry.en
                        ),
                        fixed: true,
                        hint_for_retry: None,
                    });
                } else {
                    needs_retry = true;
                    let jp_lbl = entry
                        .raw_jp
                        .first()
                        .cloned()
                        .or_else(|| entry.jp_regex_raw.clone())
                        .unwrap_or_else(|| entry.id.clone());
                    let hint = if !entry.note.is_empty() {
                        format!("Translate {} as \"{}\". {}", jp_lbl, entry.en, entry.note)
                    } else {
                        format!(
                            "Translate {} using classic adult-manga English: \"{}\".",
                            jp_lbl, entry.en
                        )
                    };
                    issues.push(QaIssue {
                        kind: QaKind::CanonForbidden,
                        rule_id: entry.id.clone(),
                        message: format!(
                            "Нарушен 18+ канон ({}): ожидается «{}»",
                            entry.id, entry.en
                        ),
                        fixed: false,
                        hint_for_retry: Some(hint),
                    });
                }
            }
        }
    }

    // 3. UntranslatedJp / Foreign script leak
    if has_foreign_script_leak(&cur_text, "en") {
        needs_retry = true;
        issues.push(QaIssue {
            kind: QaKind::UntranslatedJp,
            rule_id: "untranslated_jp".to_string(),
            message: "В переводе остались непереведённые японские или посторонние символы"
                .to_string(),
            fixed: false,
            hint_for_retry: Some(
                "Output pure English only — do not leave any Japanese kana/kanji or foreign characters."
                    .to_string(),
            ),
        });
    }

    QaResult {
        text: cur_text,
        issues,
        needs_retry,
    }
}
