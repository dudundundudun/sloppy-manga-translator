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

/// Нормализует комиксную типографику: длинное тире, многоточие, пробелы, регистр и правило Crossbar-I.
pub fn format_comic_typography(text: &str, uppercase: bool) -> String {
    let mut s = text
        .replace("\r\n", "\n")
        .replace("...", "…")
        .replace("--", "—");
    s = multi_space_re().replace_all(&s, " ").trim().to_string();
    if uppercase {
        s = s.to_uppercase();
    } else if s.chars().any(|c| c.is_ascii_alphabetic()) {
        // В английских комиксах одиночное местоимение 'i' всегда пишется с заглавной I (учитывая … и —)
        s = crossbar_i_re().replace_all(&s, "I").to_string();
    }
    s
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

/// Слоговой перенос длинных слов (>= 8 букв, минимум 3 буквы на фрагмент) для RU и EN.
pub fn hyphenate_word(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 8 || !chars.iter().all(|c| c.is_alphabetic()) {
        return vec![word.to_string()];
    }

    let mut cut_points: Vec<usize> = Vec::new();
    let n = chars.len();
    for i in 3..=(n - 3) {
        if is_vowel(chars[i - 1]) && !is_silent_sign(chars[i]) {
            if cut_points.last().is_none_or(|&prev| i - prev >= 3) {
                cut_points.push(i);
            }
        }
    }

    if cut_points.is_empty() {
        let mid = n / 2;
        if mid >= 3 && n - mid >= 3 {
            cut_points.push(mid);
        } else {
            return vec![word.to_string()];
        }
    }

    let mut parts = Vec::with_capacity(cut_points.len() + 1);
    let mut start = 0usize;
    for cp in cut_points {
        parts.push(chars[start..cp].iter().collect());
        start = cp;
    }
    parts.push(chars[start..].iter().collect());
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
                            tokens.push(format!("{part}-"));
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
