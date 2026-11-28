use crate::canon::{builtin_canon_en, CanonCategory};
use crate::prompts::kana_to_romaji;
use miko_core::GlossaryEntry;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const BUILTIN_FANDOM_JSON: &str = include_str!("../assets/fandom_names.json");

/// Стоп-лист частых нарицательных слов катаканой, которые НЕ являются именами персонажей.
const KATAKANA_STOPLIST: &[&str] = &[
    "セックス",
    "エッチ",
    "エロ",
    "エルフ",
    "ドワーフ",
    "シスター",
    "ナース",
    "ビキニ",
    "マイクロビキニ",
    "ネコミミ",
    "バニー",
    "メイド",
    "コスプレ",
    "キス",
    "キッス",
    "フェラ",
    "フェラチオ",
    "クンニ",
    "パイズリ",
    "手コキ",
    "足コキ",
    "アナル",
    "チンポ",
    "チンコ",
    "チンチン",
    "マンコ",
    "オマンコ",
    "オナニー",
    "ザーメン",
    "アソコ",
    "デカブツ",
    "デカチン",
    "デカマラ",
    "キツキツ",
    "バック",
    "スタイル",
    "タイトル",
    "オーケー",
    "ベッド",
    "ホテル",
    "トイレ",
    "シャワー",
    "コンドーム",
    "ゴム",
    "バイブ",
    "ローター",
    "ミルク",
    "パンツ",
    "ブラジャー",
    "スカート",
    "モンスター",
    "スライム",
    "ゴブリン",
    "オーク",
    "ドラゴン",
    "マジック",
    "パーティー",
    "パーティ",
    "ダンジョン",
    "ギルド",
    "クエスト",
    "スキル",
    "レベル",
    "ステータス",
    "アイテム",
    "ポーション",
    "ヒーロー",
    "プリースト",
    "ウォーリアー",
    "メイジ",
    "ページ",
    "コメント",
    "シリーズ",
    "オリジナル",
    "パロディ",
    "デジタル",
    "カラー",
    "モノクロ",
    "イラスト",
    "マンガ",
    "アニメ",
    "ゲーム",
    "タイプ",
    "サイズ",
    "ポイント",
    "チャンス",
    "サービス",
    "プレゼント",
    "メッセージ",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NameCandidate {
    pub jp: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FandomSeries {
    pub id: String,
    pub aliases: Vec<String>,
    pub names: HashMap<String, String>,
    #[serde(default)]
    pub known_mistranslations: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub terms: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FandomDb {
    pub series: Vec<FandomSeries>,
}

/// Загружает встроенную фандом-таблицу канонических имён и терминов (panic-free).
pub fn builtin_fandom_db() -> &'static FandomDb {
    static DB: OnceLock<FandomDb> = OnceLock::new();
    DB.get_or_init(
        || match serde_json::from_str::<FandomDb>(BUILTIN_FANDOM_JSON) {
            Ok(parsed) => parsed,
            Err(err) => {
                tracing::error!(
                    "Не удалось распарсить встроенный fandom_names.json: {}",
                    err
                );
                FandomDb::default()
            }
        },
    )
}

/// Находит подходящий тайтл в фандом-таблице по подсказке из названия папки (`series_hint`).
pub fn match_fandom_series<'a>(
    fandom: &'a FandomDb,
    series_hint: Option<&str>,
) -> Option<&'a FandomSeries> {
    let hint = series_hint?.trim();
    if hint.is_empty() {
        return None;
    }
    let hint_lc = hint.to_lowercase();
    fandom.series.iter().find(|s| {
        s.aliases
            .iter()
            .any(|alias| !alias.is_empty() && hint_lc.contains(&alias.to_lowercase()))
    })
}

/// Резервный поиск фандома по уникальным именам персонажей, если папка названа нейтрально (`ch01`).
pub fn infer_fandom_series_from_candidates<'a>(
    fandom: &'a FandomDb,
    cands: &[NameCandidate],
) -> Option<&'a FandomSeries> {
    let mut best: Option<(&'a FandomSeries, usize)> = None;
    for s in &fandom.series {
        let matched_distinct = cands
            .iter()
            .filter(|c| c.jp.chars().count() >= 3 && s.names.contains_key(&c.jp))
            .count();
        if matched_distinct >= 2 && best.map_or(true, |(_, prev)| matched_distinct > prev) {
            best = Some((s, matched_distinct));
        }
    }
    best.map(|(s, _)| s)
}

#[inline]
fn is_katakana_name_char(c: char) -> bool {
    ('\u{30A1}'..='\u{30FA}').contains(&c) || matches!(c, 'ー' | '・')
}

fn strip_katakana_honorific(s: &str) -> &str {
    let mut cur = s.trim_matches('・').trim_start_matches('ー');
    for sfx in ["チャン", "サマ", "サン", "クン", "タチ", "ドノ"] {
        if let Some(stem) = cur.strip_suffix(sfx) {
            if stem.chars().count() >= 3 {
                cur = stem.trim_matches('・').trim_start_matches('ー');
                break;
            }
        }
    }
    cur
}

fn is_likely_katakana_sfx(word: &str) -> bool {
    let body: Vec<char> = word
        .chars()
        .filter(|&c| !matches!(c, 'ッ' | 'ー' | '・'))
        .collect();
    if body.len() < 2 {
        return true;
    }
    let len = body.len();
    // Повторы слогов (`ドキドキ`, `ビクビク`, `パンパン`, `ジュポジュポ`)
    if (len == 4 || len == 6 || len == 8) && body[..len / 2] == body[len / 2..] {
        return true;
    }
    if len >= 4 && len % 2 == 0 && body.chunks(2).all(|ch| ch == &body[..2]) {
        return true;
    }
    // Короткие обрывистые звуки на `ッ` (`ギュッ`, `ビクッ`, `チュッ`, `ドキッ`)
    if word.ends_with('ッ') && word.chars().count() <= 4 {
        return true;
    }
    for sfx_pref in [
        "チュ",
        "ジュ",
        "グチュ",
        "クチュ",
        "ヌプ",
        "ズプ",
        "ズブ",
        "ビク",
        "ドピュ",
        "ビュル",
        "ゴク",
        "ドクン",
        "ビクン",
    ] {
        if word.starts_with(sfx_pref) {
            return true;
        }
    }
    false
}

fn build_katakana_stoplist() -> &'static HashSet<String> {
    static STOP: OnceLock<HashSet<String>> = OnceLock::new();
    STOP.get_or_init(|| {
        let mut set: HashSet<String> = KATAKANA_STOPLIST.iter().map(|s| (*s).to_string()).collect();
        for entry in &builtin_canon_en().entries {
            if matches!(
                entry.category,
                CanonCategory::Anatomy
                    | CanonCategory::Act
                    | CanonCategory::Descriptor
                    | CanonCategory::Moan
                    | CanonCategory::Sfx
            ) {
                for j in &entry.jp {
                    if j.chars().all(is_katakana_name_char) {
                        set.insert(j.clone());
                    }
                }
            }
        }
        set
    })
}

fn all_known_fandom_katakana_names() -> &'static HashSet<String> {
    static KNOWN: OnceLock<HashSet<String>> = OnceLock::new();
    KNOWN.get_or_init(|| {
        let mut set = HashSet::new();
        for s in &builtin_fandom_db().series {
            for k in s.names.keys() {
                if k.chars().all(is_katakana_name_char) {
                    set.insert(k.clone());
                }
            }
        }
        set
    })
}

/// Извлекает кандидатов в имена персонажей катаканой (длина >= 3, без гонорификов, стоп-слов и SFX).
pub fn extract_katakana_name_candidates(lines: &[String]) -> Vec<NameCandidate> {
    let stoplist = build_katakana_stoplist();
    let known_fandom = all_known_fandom_katakana_names();
    let mut counts: HashMap<String, usize> = HashMap::new();

    for line in lines {
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            if is_katakana_name_char(chars[i]) {
                let start = i;
                while i < chars.len() && is_katakana_name_char(chars[i]) {
                    i += 1;
                }
                let raw_run: String = chars[start..i].iter().collect();
                // Если дальше в строке сразу идёт хирагана гонорифика/частицы — они уже вне `raw_run`,
                // срезаем только катакана-суффиксы и краевые разделители
                let cleaned = strip_katakana_honorific(&raw_run);
                let char_len = cleaned.chars().count();
                if char_len < 3 || char_len > 14 {
                    continue;
                }
                if stoplist.contains(cleaned) {
                    continue;
                }
                if !known_fandom.contains(cleaned) && is_likely_katakana_sfx(cleaned) {
                    continue;
                }
                *counts.entry(cleaned.to_string()).or_insert(0) += 1;
            } else {
                i += 1;
            }
        }
    }

    let mut out: Vec<NameCandidate> = counts
        .into_iter()
        .map(|(jp, count)| NameCandidate { jp, count })
        .collect();
    out.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.jp.cmp(&b.jp)));
    out
}

fn title_case_romaji_name(jp_kata: &str) -> String {
    jp_kata
        .split('・')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            let rom = kana_to_romaji(part.trim());
            let mut chars = rom.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Строит автоматический `NameLock`-глоссарий для главы с учётом фандом-таблицы и пользовательского глоссария.
pub fn resolve_name_lock(
    cands: &[NameCandidate],
    series_hint: Option<&str>,
    fandom: &FandomDb,
    user_glossary: &[GlossaryEntry],
) -> Vec<GlossaryEntry> {
    let user_srcs: HashSet<&str> = user_glossary.iter().map(|g| g.src.trim()).collect();
    let matched_series = match_fandom_series(fandom, series_hint)
        .or_else(|| infer_fandom_series_from_candidates(fandom, cands));

    let mut out: Vec<GlossaryEntry> = Vec::new();
    let mut seen_src: HashSet<String> = HashSet::new();

    for cand in cands {
        let src = cand.jp.trim();
        if src.is_empty() || user_srcs.contains(src) || seen_src.contains(src) {
            continue;
        }
        if let Some(series) = matched_series {
            if let Some(canon_name) = series.names.get(src) {
                seen_src.insert(src.to_string());
                out.push(GlossaryEntry {
                    src: src.to_string(),
                    dst: canon_name.clone(),
                    note: format!("auto-lock:fandom:{}", series.id),
                });
                continue;
            }
        }
        if cand.count >= 2 {
            let rom = title_case_romaji_name(src);
            if rom.chars().count() >= 2 {
                seen_src.insert(src.to_string());
                out.push(GlossaryEntry {
                    src: src.to_string(),
                    dst: rom,
                    note: "auto-lock:katakana".to_string(),
                });
            }
        }
    }

    // Если тайтл опознан — добавляем каноничные термины фандома (например, `生臭坊主` -> `corrupt priest`)
    if let Some(series) = matched_series {
        let mut sorted_terms: Vec<(&String, &String)> = series.terms.iter().collect();
        sorted_terms.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        for (term_jp, term_en) in sorted_terms {
            let s = term_jp.trim();
            if s.is_empty() || user_srcs.contains(s) || seen_src.contains(s) {
                continue;
            }
            seen_src.insert(s.to_string());
            out.push(GlossaryEntry {
                src: s.to_string(),
                dst: term_en.clone(),
                note: format!("auto-lock:fandom-term:{}", series.id),
            });
        }
    }

    out
}
