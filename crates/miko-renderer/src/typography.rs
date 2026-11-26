use regex::Regex;
use std::sync::OnceLock;

fn multi_space_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[ \t]{2,}").expect("space regex"))
}

fn crossbar_i_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\bi\b").expect("crossbar i regex"))
}

fn leading_dot_word_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\.([A-Za-zА-Яа-яЁё])").expect("leading dot regex"))
}

fn broken_hesitation_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b(But|Uh|Uhh|Um|Ah|Oh|No|Wait|Well|Hmm)\.\s+([a-zа-яё])")
            .expect("hesitation regex")
    })
}

/// Нормализует комиксную типографику: длинное тире, многоточие, артефакты `.?`/`.!`, пробелы и правило Crossbar-I.
pub fn format_comic_typography(text: &str, uppercase: bool) -> String {
    let mut s = text
        .replace("\r\n", "\n")
        .replace("...", "…")
        .replace("..", "…")
        .replace(".?", "…?")
        .replace(".!", "…!")
        .replace("--", "—");
    s = multi_space_re().replace_all(&s, " ").trim().to_string();
    s = leading_dot_word_re().replace_all(&s, "…$1").to_string();
    s = broken_hesitation_re().replace_all(&s, "$1… $2").to_string();
    if uppercase {
        s = s.to_uppercase();
    } else if s.chars().any(|c| c.is_ascii_alphabetic()) {
        // В английских комиксах одиночное местоимение 'i' всегда пишется с заглавной I (учитывая … и —)
        s = crossbar_i_re().replace_all(&s, "I").to_string();
    }
    s
}

/// Для узких вертикальных бабблов крика (`tategaki`, `cav_h >= cav_w * 2`) мягко поджимает
/// сверхдлинные серии одинаковых букв (`Whaaaaaaaaaat?!` -> `Whaaaat?!`, `Nooooooo!!!` -> `Noooo!!!`),
/// полностью сохраняя экспрессию крика, но позволяя набрать его крупным читаемым кеглем!
pub fn compact_scream_for_narrow_bubble(text: &str, cav_w: usize, cav_h: usize) -> String {
    if cav_w >= 185 || cav_h < cav_w.saturating_mul(2) {
        return text.to_string();
    }
    let max_repeat = if cav_w < 95 { 3usize } else { 4usize };
    let mut out = String::with_capacity(text.len());
    let mut prev_low: Option<char> = None;
    let mut run_len = 0usize;

    for ch in text.chars() {
        if ch.is_alphabetic() {
            let low = ch.to_lowercase().next().unwrap_or(ch);
            if prev_low == Some(low) {
                run_len += 1;
                if run_len <= max_repeat {
                    out.push(ch);
                }
            } else {
                prev_low = Some(low);
                run_len = 1;
                out.push(ch);
            }
        } else {
            prev_low = None;
            run_len = 0;
            out.push(ch);
        }
    }
    out
}

#[inline]
fn is_vowel(ch: char) -> bool {
    matches!(
        ch.to_ascii_lowercase(),
        'a' | 'e' | 'i'
            | 'o'
            | 'u'
            | 'y'
            | 'а'
            | 'е'
            | 'ё'
            | 'и'
            | 'о'
            | 'у'
            | 'ы'
            | 'э'
            | 'ю'
            | 'я'
            | 'А'
            | 'Е'
            | 'Ё'
            | 'И'
            | 'О'
            | 'У'
            | 'Ы'
            | 'Э'
            | 'Ю'
            | 'Я'
    )
}

#[inline]
fn is_silent_sign(ch: char) -> bool {
    matches!(ch, 'ь' | 'ъ' | 'Ь' | 'Ъ' | 'й' | 'Й')
}

fn max_char_run(chars: &[char]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    let mut prev: Option<char> = None;
    for &c in chars {
        let low = c.to_lowercase().next().unwrap_or(c);
        if prev == Some(low) {
            cur += 1;
        } else {
            prev = Some(low);
            cur = 1;
        }
        best = best.max(cur);
    }
    best
}

/// Слоговой и дефисный перенос длинных слов (RU и EN) с отделением краевых знаков препинания
/// (`Whaaaaat?!`, `matter-of-factly.`, `smartphone.`) и защитой односложных слов от разрыва.
pub fn hyphenate_word(word: &str) -> Vec<String> {
    if word.is_empty() || word.contains('\'') || word.contains('’') {
        return vec![word.to_string()];
    }

    // 1. Составные слова с собственным дефисом ("matter-of-factly", "twenty-nine") бьём строго по их дефисам!
    if word.contains('-') && !word.starts_with('-') && !word.ends_with('-') && !word.contains("--") {
        let hy_parts: Vec<&str> = word.split('-').filter(|p| !p.is_empty()).collect();
        if hy_parts.len() >= 2 && hy_parts.iter().all(|p| p.chars().count() >= 2) {
            return hy_parts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    if i + 1 < hy_parts.len() {
                        format!("{p}-")
                    } else {
                        (*p).to_string()
                    }
                })
                .collect();
        }
    }

    // 2. Отделяем ведущую и хвостовую пунктуацию ("Whaaaaaat?!" -> "", "Whaaaaaat", "?!")
    let all_chars: Vec<char> = word.chars().collect();
    let Some(first_alpha) = all_chars.iter().position(|c| c.is_alphabetic()) else {
        return vec![word.to_string()];
    };
    let Some(last_alpha_rev) = all_chars.iter().rev().position(|c| c.is_alphabetic()) else {
        return vec![word.to_string()];
    };
    let last_alpha = all_chars.len() - 1 - last_alpha_rev;
    if last_alpha < first_alpha {
        return vec![word.to_string()];
    }

    let lead: String = all_chars[..first_alpha].iter().collect();
    let core_chars: &[char] = &all_chars[first_alpha..=last_alpha];
    let trail: String = all_chars[(last_alpha + 1)..].iter().collect();

    if !core_chars.iter().all(|c| c.is_alphabetic()) {
        return vec![word.to_string()];
    }

    let n = core_chars.len();
    let is_scream_stretch = max_char_run(core_chars) >= 4;
    let min_len = if is_scream_stretch { 7 } else { 8 };
    if n < min_len {
        return vec![word.to_string()];
    }

    // Запрещаем разрывать односложные слова (обе половины обязаны содержать хотя бы одну гласную)
    let mut cut_points: Vec<usize> = Vec::new();
    for i in 3..=(n - 3) {
        let left_has_vowel = core_chars[..i].iter().any(|&c| is_vowel(c));
        let right_has_vowel = core_chars[i..].iter().any(|&c| is_vowel(c));
        if !left_has_vowel || !right_has_vowel {
            continue;
        }
        // Не отрываем немое окончание "-e" / "-ed" / "-es" без собственной слоговой гласной
        let right_tail: String = core_chars[i..].iter().collect::<String>().to_ascii_lowercase();
        if matches!(right_tail.as_str(), "e" | "ed" | "es" | "s") {
            continue;
        }
        if is_vowel(core_chars[i - 1]) && !is_silent_sign(core_chars[i]) {
            // Если после гласной идут 2+ согласные подряд (`DIS-TOR-TION`, `UN-DER-STAND`),
            // делим слог между первой и второй согласной (кроме неделимых диграфов `th`, `sh`, `ch`, `ph`, `ck`)!
            let cut_idx = if i + 1 <= n - 3
                && !is_vowel(core_chars[i])
                && !is_vowel(core_chars[i + 1])
                && !is_silent_sign(core_chars[i + 1])
            {
                let pair = format!(
                    "{}{}",
                    core_chars[i].to_ascii_lowercase(),
                    core_chars[i + 1].to_ascii_lowercase()
                );
                if matches!(pair.as_str(), "th" | "sh" | "ch" | "ph" | "ck" | "wh" | "gh") {
                    i
                } else {
                    i + 1
                }
            } else {
                i
            };
            let rt_check: String = core_chars[cut_idx..]
                .iter()
                .collect::<String>()
                .to_ascii_lowercase();
            if !matches!(rt_check.as_str(), "e" | "ed" | "es" | "s")
                && cut_points.last().is_none_or(|&prev| cut_idx - prev >= 3)
            {
                cut_points.push(cut_idx);
            }
        }
    }

    if cut_points.is_empty() {
        let mid = n / 2;
        let left_v = core_chars[..mid].iter().any(|&c| is_vowel(c));
        let right_v = core_chars[mid..].iter().any(|&c| is_vowel(c));
        if mid >= 3 && n - mid >= 3 && left_v && right_v {
            cut_points.push(mid);
        } else {
            return vec![word.to_string()];
        }
    }

    let mut parts = Vec::with_capacity(cut_points.len() + 1);
    let mut start = 0usize;
    for (idx, cp) in cut_points.into_iter().enumerate() {
        let chunk: String = core_chars[start..cp].iter().collect();
        if idx == 0 {
            parts.push(format!("{lead}{chunk}"));
        } else {
            parts.push(chunk);
        }
        start = cp;
    }
    let last_chunk: String = core_chars[start..].iter().collect();
    if parts.is_empty() {
        parts.push(format!("{lead}{last_chunk}{trail}"));
    } else {
        parts.push(format!("{last_chunk}{trail}"));
    }
    parts
}

/// DP-разбивка текста на строки с ромбовидным профилем (края уже середины, без висячих коротких слов).
pub fn auto_diamond_wrap_lines<F: Fn(&str) -> f32>(
    text: &str,
    max_width_px: f32,
    measure_width: F,
) -> Vec<String> {
    let clean = text.trim();
    if clean.is_empty() {
        return Vec::new();
    }
    let max_w = max_width_px.max(24.0);
    let mut raw_lines = Vec::new();

    for para in clean.lines() {
        let p_trim = para.trim();
        if p_trim.is_empty() {
            continue;
        }
        let mut tokens: Vec<String> = Vec::new();
        for w in p_trim.split_whitespace() {
            if measure_width(w) <= max_w {
                tokens.push(w.to_string());
            } else {
                let parts = hyphenate_word(w);
                if parts.len() <= 1 {
                    tokens.push(w.to_string());
                } else {
                for (idx, part) in parts.iter().enumerate() {
                    if idx + 1 < parts.len() {
                        if part.ends_with('-') {
                            tokens.push(part.clone());
                        } else {
                            tokens.push(format!("{part}-"));
                        }
                    } else {
                        tokens.push(part.clone());
                    }
                }
                }
            }
        }

        if tokens.is_empty() {
            continue;
        }
        if tokens.len() == 1 {
            raw_lines.push(tokens[0].clone());
            continue;
        }

        let n = tokens.len();
        let join_span = |i: usize, j: usize| -> String {
            let mut out = String::new();
            for k in i..j {
                if !out.is_empty() {
                    if !out.ends_with('-') {
                        out.push(' ');
                    }
                }
                out.push_str(&tokens[k]);
            }
            out
        };

        let mut span_w = vec![vec![0.0f32; n + 1]; n];
        for i in 0..n {
            for j in (i + 1)..=n {
                let line = join_span(i, j);
                span_w[i][j] = measure_width(&line);
            }
        }

        // Целевая ширина ромба — стараемся выравнивать строки ближе к ~82% от доступной ширины
        let target_w = max_w * 0.82;
        let mut dp = vec![f32::INFINITY; n + 1];
        let mut next_idx = vec![n; n + 1];
        dp[n] = 0.0;

        for i in (0..n).rev() {
            for j in (i + 1)..=n {
                let w = span_w[i][j];
                if j > i + 1 && w > max_w {
                    break;
                }
                let overflow_pen = if w > max_w { (w - max_w) * 60.0 + 4000.0 } else { 0.0 };
                let slack = (target_w - w).abs();
                let orphan_pen = if j == n && (j - i) == 1 && tokens[i].chars().count() <= 3 && i > 0 {
                    65.0
                } else {
                    0.0
                };
                let cost = slack * slack * 0.015 + 18.0 + overflow_pen + orphan_pen + dp[j];
                if cost < dp[i] {
                    dp[i] = cost;
                    next_idx[i] = j;
                }
            }
        }

        let mut cur = 0usize;
        while cur < n {
            let nxt = next_idx[cur].max(cur + 1).min(n);
            raw_lines.push(join_span(cur, nxt));
            cur = nxt;
        }
    }

    raw_lines
}
