use crate::canon::{match_canon, CanonCategory, CanonDb};
use crate::names::{
    extract_katakana_name_candidates, infer_fandom_series_from_candidates, match_fandom_series,
    resolve_name_lock, FandomDb,
};
use miko_core::{GlossaryEntry, TextBlock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChapterAnalysis {
    pub series_hint: Option<String>,
    pub matched_fandom_id: Option<String>,
    pub name_lock: Vec<GlossaryEntry>,
    pub canon_hit_ids: Vec<String>,
    pub nsfw_line_ratio: f32,
    pub nsfw_explicit_lines: usize,
    pub repeated_lines: Vec<(String, usize)>,
    pub total_lines: usize,
}

impl ChapterAnalysis {
    /// Выполняет мгновенный локальный анализ всех распознанных блоков главы без вызова LLM (< 5 мс).
    pub fn build(
        blocks: &[TextBlock],
        series_hint: Option<&str>,
        canon: &CanonDb,
        fandom: &FandomDb,
    ) -> Self {
        let lines: Vec<String> = blocks
            .iter()
            .filter(|b| !b.is_deleted && !b.original_text.trim().is_empty())
            .map(|b| b.original_text.trim().to_string())
            .collect();
        let total_lines = lines.len();
        let joined_chapter = lines.join("\n");

        let cands = extract_katakana_name_candidates(&lines);
        let matched_series = match_fandom_series(fandom, series_hint)
            .or_else(|| infer_fandom_series_from_candidates(fandom, &cands));

        let mut name_lock = resolve_name_lock(&cands, series_hint, fandom, &[]);

        // Подхватываем кандзи-имена из распознанного фандома, если они реально встретились в тексте главы
        if let Some(series) = matched_series {
            let mut existing_srcs: HashSet<String> =
                name_lock.iter().map(|g| g.src.clone()).collect();
            let mut kanji_names: Vec<(&String, &String)> = series.names.iter().collect();
            kanji_names.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
            for (jp_name, en_name) in kanji_names {
                if !existing_srcs.contains(jp_name.as_str())
                    && joined_chapter.contains(jp_name.as_str())
                {
                    existing_srcs.insert(jp_name.clone());
                    name_lock.push(GlossaryEntry {
                        src: jp_name.clone(),
                        dst: en_name.clone(),
                        note: format!("auto-lock:fandom:{}", series.id),
                    });
                }
            }
            // Оставляем в `name_lock` главы только те фандом-термины, которые реально есть на страницах главы
            if !lines.is_empty() {
                name_lock.retain(|g| {
                    !g.note.starts_with("auto-lock:fandom-term:")
                        || joined_chapter.contains(g.src.as_str())
                });
            }
        }

        let hits = match_canon(canon, &lines);
        let canon_hit_ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();

        let mut nsfw_explicit_lines = 0usize;
        for line in &lines {
            let is_explicit = canon.entries.iter().any(|e| {
                matches!(e.category, CanonCategory::Anatomy | CanonCategory::Act)
                    && e.matches_jp(line)
            });
            if is_explicit {
                nsfw_explicit_lines += 1;
            }
        }
        let nsfw_line_ratio = if total_lines > 0 {
            (nsfw_explicit_lines as f32) / (total_lines as f32)
        } else {
            0.0
        };

        let mut freq: HashMap<String, usize> = HashMap::new();
        for line in &lines {
            if line.chars().count() >= 2 {
                *freq.entry(line.clone()).or_insert(0) += 1;
            }
        }
        let mut repeated_lines: Vec<(String, usize)> =
            freq.into_iter().filter(|(_, c)| *c >= 2).collect();
        repeated_lines.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        Self {
            series_hint: series_hint
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
            matched_fandom_id: matched_series.map(|s| s.id.clone()),
            name_lock,
            canon_hit_ids,
            nsfw_line_ratio,
            nsfw_explicit_lines,
            repeated_lines,
            total_lines,
        }
    }

    /// Возвращает краткую сводку главы (`<= 600` символов) для системного промпта.
    pub fn chapter_brief(&self, max_chars: usize) -> String {
        let cap = max_chars.min(600);
        if cap == 0 {
            return String::new();
        }
        let is_heavy_nsfw = self.nsfw_line_ratio >= 0.12 || self.nsfw_explicit_lines >= 8;
        let char_names: Vec<&GlossaryEntry> = self
            .name_lock
            .iter()
            .filter(|g| !g.note.starts_with("auto-lock:fandom-term:"))
            .take(10)
            .collect();

        if self.series_hint.is_none() && char_names.is_empty() && !is_heavy_nsfw {
            return String::new();
        }

        let mut parts = Vec::new();
        let mut head = String::from("CHAPTER BRIEF: ");
        match (&self.series_hint, is_heavy_nsfw) {
            (Some(h), true) => {
                head.push_str(&format!(
                    "parody/series \"{h}\"; explicit adult content throughout."
                ));
            }
            (Some(h), false) => {
                head.push_str(&format!("series/title \"{h}\"."));
            }
            (None, true) => {
                head.push_str("explicit adult manga chapter throughout.");
            }
            (None, false) => {
                head.push_str("recurring character names locked across the chapter.");
            }
        }
        parts.push(head);

        if !char_names.is_empty() {
            let pairs: Vec<String> = char_names
                .iter()
                .map(|g| format!("{}={}", g.src, g.dst))
                .collect();
            parts.push(format!("Locked names: {}.", pairs.join(", ")));
        }

        let full = parts.join("\n");
        if full.chars().count() <= cap {
            full
        } else {
            let mut truncated: String = full.chars().take(cap.saturating_sub(1)).collect();
            truncated.push('…');
            truncated
        }
    }
}
