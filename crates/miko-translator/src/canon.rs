use miko_core::{AutoFixRaw, CanonOverride};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::sync::OnceLock;

const BUILTIN_CANON_EN_JSON: &str = include_str!("../assets/canon_en_adult.json");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonCategory {
    Anatomy,
    Act,
    Descriptor,
    Moan,
    Sfx,
    Grammar,
    StyleBan,
}

impl CanonCategory {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "anatomy" => Self::Anatomy,
            "act" => Self::Act,
            "descriptor" => Self::Descriptor,
            "moan" => Self::Moan,
            "sfx" => Self::Sfx,
            "grammar" => Self::Grammar,
            "style_ban" | "styleban" => Self::StyleBan,
            _ => Self::Act,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Anatomy => "anatomy",
            Self::Act => "act",
            Self::Descriptor => "descriptor",
            Self::Moan => "moan",
            Self::Sfx => "sfx",
            Self::Grammar => "grammar",
            Self::StyleBan => "style_ban",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CanonEntry {
    pub id: String,
    pub category: CanonCategory,
    /// Предвычисленные подстроки для быстрого поиска (с учётом `expand_kana`).
    pub jp: Vec<String>,
    /// Исходные подстроки без дублирования каны (для отображения в промпте и UI).
    pub raw_jp: Vec<String>,
    pub expand_kana: bool,
    pub jp_regex: Option<Regex>,
    pub jp_regex_raw: Option<String>,
    pub en: String,
    pub en_alts: Vec<String>,
    pub forbid_en: Vec<Regex>,
    pub forbid_en_raw: Vec<String>,
    pub required_en_any: Vec<Regex>,
    pub required_en_any_raw: Vec<String>,
    pub auto_fix: Vec<(Regex, String)>,
    pub auto_fix_raw: Vec<AutoFixRaw>,
    pub note: String,
    pub priority: i32,
}

#[derive(Debug, Clone)]
pub struct CanonDb {
    pub version: String,
    pub entries: Vec<CanonEntry>,
}

#[derive(Debug, Deserialize)]
struct RawCanonFile {
    version: String,
    entries: Vec<RawCanonEntry>,
}

#[derive(Debug, Deserialize)]
struct RawCanonEntry {
    id: String,
    category: String,
    #[serde(default)]
    jp: Vec<String>,
    #[serde(default)]
    expand_kana: bool,
    #[serde(default)]
    jp_regex: Option<String>,
    #[serde(default)]
    en: String,
    #[serde(default)]
    en_alts: Vec<String>,
    #[serde(default)]
    forbid_en: Vec<String>,
    #[serde(default)]
    required_en_any: Vec<String>,
    #[serde(default)]
    auto_fix: Vec<AutoFixRaw>,
    #[serde(default)]
    note: String,
    #[serde(default = "default_priority")]
    priority: i32,
}

fn default_priority() -> i32 {
    80
}

fn kata_to_hira_str(s: &str) -> String {
    s.chars()
        .map(|ch| {
            let cp = ch as u32;
            if (0x30A1..=0x30F6).contains(&cp) {
                char::from_u32(cp - 0x60).unwrap_or(ch)
            } else {
                ch
            }
        })
        .collect()
}

fn hira_to_kata_str(s: &str) -> String {
    s.chars()
        .map(|ch| {
            let cp = ch as u32;
            if (0x3041..=0x3096).contains(&cp) {
                char::from_u32(cp + 0x60).unwrap_or(ch)
            } else {
                ch
            }
        })
        .collect()
}

fn expand_jp_variants(raw_jp: &[String], expand_kana: bool) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for item in raw_jp {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.to_string()) {
            out.push(trimmed.to_string());
        }
        if expand_kana {
            let hira = kata_to_hira_str(trimmed);
            if seen.insert(hira.clone()) {
                out.push(hira);
            }
            let kata = hira_to_kata_str(trimmed);
            if seen.insert(kata.clone()) {
                out.push(kata);
            }
        }
    }
    out
}

fn compile_raw_entry(raw: RawCanonEntry) -> Option<CanonEntry> {
    let id = raw.id.trim().to_string();
    if id.is_empty() {
        return None;
    }
    let category = CanonCategory::from_str_lossy(&raw.category);
    let raw_jp: Vec<String> = raw
        .jp
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let jp = expand_jp_variants(&raw_jp, raw.expand_kana);

    let jp_regex_raw = raw
        .jp_regex
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let jp_regex = match jp_regex_raw {
        Some(ref pat) => match Regex::new(pat) {
            Ok(re) => Some(re),
            Err(err) => {
                tracing::error!("Canon entry '{}': invalid jp_regex '{}': {}", id, pat, err);
                return None;
            }
        },
        None => None,
    };

    let mut forbid_en = Vec::new();
    let mut forbid_en_raw = Vec::new();
    for pat in &raw.forbid_en {
        let p = pat.trim();
        if p.is_empty() {
            continue;
        }
        match Regex::new(p) {
            Ok(re) => {
                forbid_en.push(re);
                forbid_en_raw.push(p.to_string());
            }
            Err(err) => {
                tracing::error!("Canon entry '{}': invalid forbid_en '{}': {}", id, p, err);
            }
        }
    }

    let mut required_en_any = Vec::new();
    let mut required_en_any_raw = Vec::new();
    for pat in &raw.required_en_any {
        let p = pat.trim();
        if p.is_empty() {
            continue;
        }
        match Regex::new(p) {
            Ok(re) => {
                required_en_any.push(re);
                required_en_any_raw.push(p.to_string());
            }
            Err(err) => {
                tracing::error!(
                    "Canon entry '{}': invalid required_en_any '{}': {}",
                    id,
                    p,
                    err
                );
            }
        }
    }

    let mut auto_fix = Vec::new();
    let mut auto_fix_raw = Vec::new();
    for af in &raw.auto_fix {
        let p = af.from.trim();
        if p.is_empty() {
            continue;
        }
        match Regex::new(p) {
            Ok(re) => {
                auto_fix.push((re, af.to.clone()));
                auto_fix_raw.push(AutoFixRaw {
                    from: p.to_string(),
                    to: af.to.clone(),
                });
            }
            Err(err) => {
                tracing::error!("Canon entry '{}': invalid auto_fix '{}': {}", id, p, err);
            }
        }
    }

    Some(CanonEntry {
        id,
        category,
        jp,
        raw_jp,
        expand_kana: raw.expand_kana,
        jp_regex,
        jp_regex_raw,
        en: raw.en.trim().to_string(),
        en_alts: raw
            .en_alts
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        forbid_en,
        forbid_en_raw,
        required_en_any,
        required_en_any_raw,
        auto_fix,
        auto_fix_raw,
        note: raw.note.trim().to_string(),
        priority: raw.priority,
    })
}

/// Валидирует пользовательский override перед записью в SQLite (возвращает текст ошибки регэкспа для HTTP 400).
pub fn validate_canon_override(ov: &CanonOverride) -> Result<(), String> {
    if ov.id.trim().is_empty() {
        return Err("Поле 'id' в canon override не может быть пустым".to_string());
    }
    if let Some(ref pat) = ov.jp_regex {
        let trimmed = pat.trim();
        if !trimmed.is_empty() {
            Regex::new(trimmed).map_err(|e| format!("Некорректный jp_regex в '{}': {e}", ov.id))?;
        }
    }
    for pat in &ov.forbid_en {
        let trimmed = pat.trim();
        if !trimmed.is_empty() {
            Regex::new(trimmed)
                .map_err(|e| format!("Некорректный forbid_en '{}' в '{}': {e}", trimmed, ov.id))?;
        }
    }
    for pat in &ov.required_en_any {
        let trimmed = pat.trim();
        if !trimmed.is_empty() {
            Regex::new(trimmed).map_err(|e| {
                format!(
                    "Некорректный required_en_any '{}' в '{}': {e}",
                    trimmed, ov.id
                )
            })?;
        }
    }
    for af in &ov.auto_fix {
        let trimmed = af.from.trim();
        if !trimmed.is_empty() {
            Regex::new(trimmed).map_err(|e| {
                format!(
                    "Некорректный auto_fix.from '{}' в '{}': {e}",
                    trimmed, ov.id
                )
            })?;
        }
    }
    Ok(())
}

pub fn canon_entry_to_override(entry: &CanonEntry) -> CanonOverride {
    CanonOverride {
        id: entry.id.clone(),
        category: entry.category.as_str().to_string(),
        jp: entry.raw_jp.clone(),
        expand_kana: entry.expand_kana,
        jp_regex: entry.jp_regex_raw.clone(),
        en: entry.en.clone(),
        en_alts: entry.en_alts.clone(),
        forbid_en: entry.forbid_en_raw.clone(),
        required_en_any: entry.required_en_any_raw.clone(),
        auto_fix: entry.auto_fix_raw.clone(),
        note: entry.note.clone(),
        priority: entry.priority,
        enabled: true,
    }
}

fn compile_override(ov: &CanonOverride) -> Option<CanonEntry> {
    compile_raw_entry(RawCanonEntry {
        id: ov.id.clone(),
        category: ov.category.clone(),
        jp: ov.jp.clone(),
        expand_kana: ov.expand_kana,
        jp_regex: ov.jp_regex.clone(),
        en: ov.en.clone(),
        en_alts: ov.en_alts.clone(),
        forbid_en: ov.forbid_en.clone(),
        required_en_any: ov.required_en_any.clone(),
        auto_fix: ov.auto_fix.clone(),
        note: ov.note.clone(),
        priority: ov.priority,
    })
}

/// Загружает встроенную 18+ базу классического английского сканлейта (panic-free).
pub fn builtin_canon_en() -> &'static CanonDb {
    static DB: OnceLock<CanonDb> = OnceLock::new();
    DB.get_or_init(
        || match serde_json::from_str::<RawCanonFile>(BUILTIN_CANON_EN_JSON) {
            Ok(parsed) => {
                let entries = parsed
                    .entries
                    .into_iter()
                    .filter_map(compile_raw_entry)
                    .collect();
                CanonDb {
                    version: parsed.version,
                    entries,
                }
            }
            Err(err) => {
                tracing::error!(
                    "Не удалось распарсить встроенный canon_en_adult.json: {}",
                    err
                );
                CanonDb {
                    version: "0.0.0-error".to_string(),
                    entries: Vec::new(),
                }
            }
        },
    )
}

/// Сливает встроенную базу с пользовательскими правками из SQLite (`enabled=false` отключает запись по `id`).
pub fn merge_overrides(base: &CanonDb, overrides: &[CanonOverride]) -> CanonDb {
    if overrides.is_empty() {
        return base.clone();
    }
    let mut entries = base.entries.clone();
    let mut ov_hash = Sha256::new();
    for ov in overrides {
        let id = ov.id.trim();
        if id.is_empty() {
            continue;
        }
        ov_hash.update(id.as_bytes());
        ov_hash.update([u8::from(ov.enabled)]);
        if let Ok(bytes) = serde_json::to_vec(ov) {
            ov_hash.update(&bytes);
        }

        if !ov.enabled {
            entries.retain(|e| e.id != id);
            continue;
        }
        if let Some(compiled) = compile_override(ov) {
            if let Some(pos) = entries.iter().position(|e| e.id == id) {
                entries[pos] = compiled;
            } else {
                entries.push(compiled);
            }
        }
    }
    let suffix = &hex::encode(ov_hash.finalize())[..8];
    CanonDb {
        version: format!("{}+ov_{suffix}", base.version),
        entries,
    }
}

impl CanonEntry {
    /// Проверяет, срабатывает ли триггер записи на исходной японской реплике.
    pub fn matches_jp(&self, text: &str) -> bool {
        if text.is_empty() {
            return false;
        }
        if self.jp.iter().any(|needle| text.contains(needle.as_str())) {
            return true;
        }
        if let Some(ref re) = self.jp_regex {
            if re.is_match(text) {
                return true;
            }
        }
        false
    }

    fn prompt_trigger_label(&self) -> String {
        if !self.raw_jp.is_empty() {
            let take_n = self.raw_jp.len().min(3);
            return self.raw_jp[..take_n].join(" / ");
        }
        match self.id.as_str() {
            "climax_iku" => "イク / イっちゃう / イキそう".to_string(),
            "desc_ero_person" => "エロ坊主 / エロ親父 (エロ + person)".to_string(),
            "grammar_igai" => "〜以外 / 〜以外は".to_string(),
            "grammar_dake_shika" => "〜だけ / 〜しか".to_string(),
            _ => self.jp_regex_raw.clone().unwrap_or_else(|| self.id.clone()),
        }
    }

    fn clean_forbidden_preview(&self) -> Option<String> {
        if self.forbid_en_raw.is_empty() {
            return None;
        }
        let mut words = Vec::new();
        for raw in &self.forbid_en_raw {
            let cleaned = raw
                .replace("(?i)", "")
                .replace("\\b", "")
                .replace("\\W*", "")
                .replace("\\s+", " ")
                .replace("\\s*", "")
                .replace("(?:e|ed|ing)", "ed")
                .replace("(?:her|you|him)", "her")
                .replace("(?:ie)?", "")
                .replace("?", "")
                .replace("+", "")
                .trim_matches(['^', '$', ' ', '\\'])
                .to_string();
            let short = if cleaned.contains("(?:") {
                continue;
            } else {
                cleaned
            };
            if !short.is_empty() && short.len() <= 22 && !words.contains(&short) {
                words.push(short);
            }
            if words.len() >= 4 {
                break;
            }
        }
        if words.is_empty() {
            None
        } else {
            Some(words.join(", "))
        }
    }
}

/// Возвращает совпавшие записи базы канона для списка исходных строк, отсортированные по убыванию `priority`.
pub fn match_canon<'a>(db: &'a CanonDb, src_texts: &[String]) -> Vec<&'a CanonEntry> {
    if src_texts.is_empty() || db.entries.is_empty() {
        return Vec::new();
    }
    let mut hits: Vec<&'a CanonEntry> = db
        .entries
        .iter()
        .filter(|entry| {
            entry.category != CanonCategory::StyleBan
                && src_texts.iter().any(|line| entry.matches_jp(line))
        })
        .collect();
    hits.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.id.cmp(&b.id)));
    hits
}

/// Строит компактный блок `ADULT MANGA CANON` только из реально встретившихся в пачке терминов (`<= max_entries`).
pub fn build_canon_prompt_block(hits: &[&CanonEntry], max_entries: usize) -> String {
    if hits.is_empty() || max_entries == 0 {
        return String::new();
    }
    let mut sorted: Vec<&CanonEntry> = hits
        .iter()
        .copied()
        .filter(|e| e.category != CanonCategory::StyleBan && !e.en.is_empty())
        .collect();
    sorted.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.id.cmp(&b.id)));
    sorted.dedup_by(|a, b| a.id == b.id);
    if sorted.is_empty() {
        return String::new();
    }

    let mut out = String::from(
        "ADULT MANGA CANON (classic English scanlation register — mandatory for matched terms):\n",
    );
    for entry in sorted.into_iter().take(max_entries) {
        let trig = entry.prompt_trigger_label();
        let mut suffix = String::new();
        if let Some(forb) = entry.clean_forbidden_preview() {
            suffix = format!(" (never: {forb})");
        } else if !entry.note.is_empty() {
            suffix = format!(" ({})", entry.note);
        }
        out.push_str(&format!("- {} => \"{}\"{}\n", trig, entry.en, suffix));
    }
    out.push_str(
        "STYLE: classic adult-manga English; no modern meme/Twitter slang; moans as Ahn…/Mmh…/Haah…\n",
    );
    out
}

/// Короткий детерминированный отпечаток совпавших правил канона для ключа SQLite-кэша.
pub fn canon_hits_fingerprint(db_version: &str, hits: &[&CanonEntry]) -> String {
    if hits.is_empty() {
        return String::new();
    }
    let mut ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    let mut hasher = Sha256::new();
    hasher.update(db_version.as_bytes());
    hasher.update(b":");
    for id in ids {
        hasher.update(id.as_bytes());
        hasher.update(b",");
    }
    hex::encode(hasher.finalize())[..16].to_string()
}

fn is_trailing_moan_punct(c: char) -> bool {
    matches!(
        c,
        '…' | '‥' | '.' | '!' | '！' | '?' | '？' | '~' | '〜' | '♡' | '♥' | '♪' | ' ' | '　'
    )
}

fn normalize_trailing_punct_for_en(raw_tail: &str, default_en: &str) -> String {
    let mut out = String::new();
    let mut chars = raw_tail.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '！' => out.push('!'),
            '？' => out.push('?'),
            '〜' => out.push('~'),
            '‥' => out.push('…'),
            '.' => {
                let mut dots = 1usize;
                while chars.peek() == Some(&'.') {
                    chars.next();
                    dots += 1;
                }
                if dots >= 2 {
                    out.push('…');
                } else {
                    out.push('.');
                }
            }
            ' ' | '　' => {}
            other => out.push(other),
        }
    }
    if out.is_empty() {
        // Сохраняем дефолтную пунктуацию из самого шаблона (`Ah!`, `Phew…`, `Oh my~`)
        default_en
            .chars()
            .skip_while(|c| c.is_ascii_alphabetic() || *c == ',' || *c == ' ' || *c == '\'')
            .collect()
    } else {
        out
    }
}

/// Детерминированный перевод одиночных стонов и вздохов без обращения к LLM (`ふう…` → `Phew…`, `あん…` → `Ahn…`).
pub fn match_pure_moan_en(db: &CanonDb, src_jp: &str) -> Option<String> {
    let trimmed = src_jp.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 14 {
        return None;
    }
    // Отделяем ведущие кавычки/многоточия и хвостовую пунктуацию
    let without_lead = trimmed.trim_start_matches(['…', '‥', '.', ' ', '　']);
    let lead_ellipsis = without_lead.len() < trimmed.len()
        && trimmed
            .chars()
            .take_while(|c| matches!(c, '…' | '‥' | '.'))
            .count()
            >= 1;

    let mut split_byte = without_lead.len();
    for (idx, ch) in without_lead.char_indices().rev() {
        if is_trailing_moan_punct(ch) {
            split_byte = idx;
        } else {
            break;
        }
    }
    let core_raw = &without_lead[..split_byte];
    let tail_raw = &without_lead[split_byte..];
    if core_raw.is_empty() {
        return None;
    }

    // Нормализуем удлинения в конце стона (`あんっ` -> `あん`, `ふぅー` -> `ふぅ`)
    let core_clean: String = core_raw
        .trim_end_matches(['っ', 'ッ', 'ー'])
        .chars()
        .collect();
    let candidates_to_try = [core_raw, core_clean.as_str()];

    for entry in &db.entries {
        if entry.category != CanonCategory::Moan || entry.en.is_empty() {
            continue;
        }
        let matched = entry.jp.iter().any(|jp_form| {
            candidates_to_try
                .iter()
                .any(|cand| !cand.is_empty() && *cand == jp_form.as_str())
        });
        if matched {
            let primary_en = entry.en.split('/').next().unwrap_or(&entry.en).trim();
            // Сдвоенный вздох `はぁはぁ` -> `Haah… haah`
            let chosen_en = if matches!(core_clean.as_str(), "はぁはぁ" | "はあはあ" | "ハァハァ")
                && !entry.en_alts.is_empty()
            {
                entry.en_alts[0].as_str()
            } else {
                primary_en
            };
            let base_en = chosen_en.trim_end_matches(|c: char| is_trailing_moan_punct(c));
            let tail_en = normalize_trailing_punct_for_en(tail_raw, chosen_en);
            let prefix = if lead_ellipsis { "…" } else { "" };
            return Some(format!("{prefix}{base_en}{tail_en}"));
        }
    }
    None
}
