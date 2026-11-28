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

/// Разлепляет склеенные нейросетью точки внутри реплик без пробела (`"Mmm.such a.strange.feeling."` -> `"Mmm… such a… strange… feeling."`,
/// `"Something's.c-coming."` -> `"Something's… c-coming."`), не трогая аббревиатуры (`U.S.A.`, `e.g.`) и домены/расширения.
fn split_glued_sentence_dots(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    if len < 4 || !chars.contains(&'.') {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len() + 8);
    let mut i = 0usize;
    while i < len {
        let ch = chars[i];
        if ch == '.' && i > 0 && i + 1 < len {
            let prev = chars[i - 1];
            let next = chars[i + 1];
            let prev_ok = prev.is_alphabetic() || matches!(prev, '\'' | '’');
            let next_ok = next.is_alphabetic();
            if prev_ok && next_ok {
                let mut left_alpha = 0usize;
                let mut l = i;
                while l > 0 {
                    let c = chars[l - 1];
                    if c.is_alphabetic() {
                        left_alpha += 1;
                        l -= 1;
                    } else if matches!(c, '\'' | '’' | '-') {
                        l -= 1;
                    } else {
                        break;
                    }
                }

                let mut right_word = String::new();
                let mut right_alpha = 0usize;
                let mut r = i + 1;
                while r < len {
                    let c = chars[r];
                    if c.is_alphabetic() {
                        right_alpha += 1;
                        right_word.push(c.to_ascii_lowercase());
                        r += 1;
                    } else if matches!(c, '\'' | '’' | '-') {
                        r += 1;
                    } else {
                        break;
                    }
                }

                let is_domain_ext = matches!(
                    right_word.as_str(),
                    "com" | "net" | "org" | "jp" | "ru" | "io" | "co" | "png" | "jpg" | "webp" | "exe"
                );
                if (left_alpha >= 2 || right_alpha >= 2) && !is_domain_ext {
                    out.push('…');
                    out.push(' ');
                    i += 1;
                    continue;
                }
            }
        }
        out.push(ch);
        i += 1;
    }
    out
}

fn trailing_dangling_conj_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)(,\s*(?:but|and|or|so|но|а|да))\.$").expect("dangling conj regex")
    })
}

/// Исправляет оборванные точки внутри реплики перед строчным словом (`"Mmm. It feels. strange."` -> `"Mmm. It feels… strange."`),
/// не затрагивая стандартные сокращения (`Mr.`, `Dr.`, `etc.`).
fn repair_mid_sentence_periods(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    if len < 5 || !chars.contains(&'.') {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len() + 4);
    let mut i = 0usize;
    while i < len {
        let ch = chars[i];
        if ch == '.' && i >= 2 && i + 2 < len && chars[i - 1].is_alphabetic() {
            let prev_is_dot = chars[i - 1] == '.' || (i >= 2 && chars[i - 2] == '.');
            let mut j = i + 1;
            let mut saw_space = false;
            while j < len && matches!(chars[j], ' ' | '\t') {
                saw_space = true;
                j += 1;
            }
            if !prev_is_dot
                && saw_space
                && j < len
                && chars[j].is_alphabetic()
                && chars[j].is_lowercase()
            {
                let mut w_start = i;
                while w_start > 0 && chars[w_start - 1].is_alphabetic() {
                    w_start -= 1;
                }
                let word: String = chars[w_start..i].iter().collect();
                let word_len = word.chars().count();
                let is_abbrev = matches!(
                    word.to_ascii_lowercase().as_str(),
                    "mr" | "mrs" | "ms" | "dr" | "st" | "vs" | "etc"
                );
                if word_len >= 2 && !is_abbrev {
                    out.push('…');
                    i += 1;
                    continue;
                }
            }
        }
        out.push(ch);
        i += 1;
    }
    out
}

/// Нормализует комиксную типографику: длинное тире, многоточие, артефакты `.?`/`.!`, склеенные точки ИИ, пробелы и правило Crossbar-I.
pub fn format_comic_typography(text: &str, uppercase: bool) -> String {
    let mut s = text
        .replace("\r\n", "\n")
        .replace("...", "…")
        .replace("..", "…")
        .replace(".?", "…?")
        .replace(".!", "…!")
        .replace("--", "—");
    s = split_glued_sentence_dots(&s);
    s = multi_space_re().replace_all(&s, " ").trim().to_string();
    s = leading_dot_word_re().replace_all(&s, "…$1").to_string();
    s = broken_hesitation_re().replace_all(&s, "$1… $2").to_string();
    s = repair_mid_sentence_periods(&s);
    let replaced_conj = trailing_dangling_conj_re().replace(&s, "$1…");
    if replaced_conj != s {
        s = replaced_conj.into_owned();
    }
    if s == "I." || s == "i." {
        s = "I…".to_string();
    } else if s == "Я." || s == "я." {
        s = "Я…".to_string();
    }
    if uppercase {
        s = s.to_uppercase();
    } else if s.chars().any(|c| c.is_ascii_alphabetic()) {
        // В английских комиксах одиночное местоимение 'i' всегда пишется с заглавной I (учитывая … и —)
        s = crossbar_i_re().replace_all(&s, "I").to_string();
    }
    s
}

/// Для узких вертикальных бабблов крика (`tategaki`, `cav_h >= cav_w * 2`) мягко поджимает
/// сверхдлинные серии одинаковых букв (`Whaaaaaaaaaat?!` -> `Whaaaat?!`, `Sluuurp` в 65px -> `Sluurp`),
/// полностью сохраняя экспрессию крика, но позволяя набрать его крупным читаемым кеглем!
pub fn compact_scream_for_narrow_bubble(text: &str, cav_w: usize, cav_h: usize) -> String {
    if cav_w >= 185 || cav_h < cav_w.saturating_mul(2) {
        return text.to_string();
    }
    let max_repeat = if cav_w < 78 && cav_h * 10 >= cav_w * 22 {
        2usize
    } else if cav_w < 100 {
        3usize
    } else {
        4usize
    };
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

/// Проверяет, содержит ли английский хвост слова хотя бы одну реально произносимую слоговую гласную
/// (исключая немое окончание `-e`, а также `-ed` и `-es` без слогообразующей основы).
fn has_spoken_english_vowel(chars: &[char]) -> bool {
    if chars.is_empty() {
        return false;
    }
    let low: String = chars.iter().map(|c| c.to_ascii_lowercase()).collect();
    if matches!(low.as_str(), "e" | "ed" | "es" | "s" | "d") {
        return false;
    }
    let stem = if let Some(base) = low.strip_suffix("ed") {
        if base.ends_with('t') || base.ends_with('d') {
            return true;
        }
        base
    } else if let Some(base) = low.strip_suffix("es") {
        if base.ends_with('s')
            || base.ends_with('z')
            || base.ends_with('x')
            || base.ends_with("ch")
            || base.ends_with("sh")
        {
            return true;
        }
        base
    } else if let Some(base) = low.strip_suffix('e') {
        if base.ends_with('l') && base.len() >= 2 {
            return true;
        }
        base
    } else {
        low.as_str()
    };
    stem.chars().any(is_vowel)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrapToken {
    pub text: String,
    /// Следующий токен — продолжение этого же слова (без пробела между ними).
    pub glue_next: bool,
    /// При разрыве строки сразу после этого токена в конец строки добавляется дефис `'-'`.
    pub soft_hyphen_after: bool,
}

/// Разбивает одно слово на токены укладки (с мягкими слоговыми переносами или жёсткими дефисами составных слов).
pub fn tokenize_word_for_wrap(word: &str, allow_hyphenate: bool) -> Vec<WrapToken> {
    if !allow_hyphenate {
        return vec![WrapToken {
            text: word.to_string(),
            glue_next: false,
            soft_hyphen_after: false,
        }];
    }
    let parts = hyphenate_word(word);
    if parts.len() <= 1 {
        return vec![WrapToken {
            text: word.to_string(),
            glue_next: false,
            soft_hyphen_after: false,
        }];
    }
    let mut out = Vec::with_capacity(parts.len());
    for (idx, part) in parts.into_iter().enumerate() {
        let is_last = idx + 1 == out.capacity();
        if is_last {
            out.push(WrapToken {
                text: part,
                glue_next: false,
                soft_hyphen_after: false,
            });
        } else if part.ends_with('-') {
            // Составное слово со своим собственным дефисом ("matter-of-factly") — дефис сохраняется и внутри строки
            out.push(WrapToken {
                text: part,
                glue_next: true,
                soft_hyphen_after: false,
            });
        } else {
            // Мягкий слоговой перенос: внутри одной строки склеивается чисто без '-', дефис появляется ТОЛЬКО на конце строки!
            out.push(WrapToken {
                text: part,
                glue_next: true,
                soft_hyphen_after: true,
            });
        }
    }
    out
}

/// Собирает срез токенов `tokens[i..j]` в строку: мягкий дефис ставится ТОЛЬКО если строка обрывается посреди слова!
pub fn join_wrap_tokens(tokens: &[WrapToken], i: usize, j: usize) -> String {
    if i >= j || i >= tokens.len() {
        return String::new();
    }
    let end = j.min(tokens.len());
    let mut out = String::new();
    for k in i..end {
        let tok = &tokens[k];
        out.push_str(&tok.text);
        let is_line_end = k + 1 == end;
        if is_line_end {
            if tok.soft_hyphen_after {
                out.push('-');
            }
        } else if !tok.glue_next {
            out.push(' ');
        }
    }
    out
}

/// Слоговой и дефисный перенос длинных слов (RU и EN) с отделением краевых знаков препинания
/// (`Whaaaaat?!`, `matter-of-factly.`, `smartphone.`) и защитой односложных слов и дифтонгов от разрыва.
pub fn hyphenate_word(word: &str) -> Vec<String> {
    if word.is_empty() {
        return vec![word.to_string()];
    }

    // Разрешаем слоговой перенос длинных слов с апострофным окончанием ("Something's" -> ["Some", "thing's"], "EVERYTHING'S!"),
    // оставляя короткие стяжения ("don't", "can't", "I'm", "you're") неделимыми!
    if word.contains('\'') || word.contains('’') {
        for apos in ['\'', '’'] {
            if let Some(pos) = word.rfind(apos) {
                let stem = &word[..pos];
                let suffix = &word[pos..]; // напр. "'s", "'ll", "'ve!"
                let suffix_alpha: String = suffix
                    .chars()
                    .filter(|c| c.is_alphabetic())
                    .map(|c| c.to_ascii_lowercase())
                    .collect();
                let stem_alpha = stem.chars().filter(|c| c.is_alphabetic()).count();
                if matches!(suffix_alpha.as_str(), "s" | "d" | "ll" | "ve" | "re")
                    && stem_alpha >= 7
                    && !stem.contains('\'')
                    && !stem.contains('’')
                {
                    let mut parts = hyphenate_word(stem);
                    if parts.len() >= 2 {
                        if let Some(last) = parts.last_mut() {
                            last.push_str(suffix);
                        }
                        return parts;
                    }
                }
            }
        }
        return vec![word.to_string()];
    }

    // 1. Составные слова с собственным дефисом ("matter-of-factly", "twenty-nine", "great-granddaughter"):
    // бьём по их дефисам, а сверхдлинные половинки (>= 10 букв, вроде "granddaughter") дополнительно делим по слогам!
    if word.contains('-') && !word.starts_with('-') && !word.ends_with('-') && !word.contains("--") {
        let hy_parts: Vec<&str> = word.split('-').filter(|p| !p.is_empty()).collect();
        if hy_parts.len() >= 2 && hy_parts.iter().all(|p| p.chars().count() >= 2) {
            let mut out = Vec::new();
            for (i, p) in hy_parts.iter().enumerate() {
                let is_last = i + 1 == hy_parts.len();
                let alpha_cnt = p.chars().filter(|c| c.is_alphabetic()).count();
                if alpha_cnt >= 10 {
                    let sub = hyphenate_word(p);
                    if sub.len() >= 2 {
                        let sub_len = sub.len();
                        for (si, sp) in sub.into_iter().enumerate() {
                            if si + 1 == sub_len && !is_last {
                                out.push(format!("{sp}-"));
                            } else {
                                out.push(sp);
                            }
                        }
                        continue;
                    }
                }
                if !is_last {
                    out.push(format!("{p}-"));
                } else {
                    out.push((*p).to_string());
                }
            }
            return out;
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
    let is_scream_stretch = max_char_run(core_chars) >= 3;
    let min_len = if is_scream_stretch { 6 } else { 7 };
    if n < min_len {
        return vec![word.to_string()];
    }

    let is_latin = core_chars.iter().all(|c| c.is_ascii_alphabetic());

    // Для растянутых криков ("Whaaaaaat?!") разрешаем безопасный разрыв посередине длинного повтора
    if is_scream_stretch {
        let mid = n / 2;
        if mid >= 3 && n - mid >= 3 {
            let first_chunk: String = core_chars[..mid].iter().collect();
            let second_chunk: String = core_chars[mid..].iter().collect();
            return vec![
                format!("{lead}{first_chunk}"),
                format!("{second_chunk}{trail}"),
            ];
        }
    }

    // Запрещаем разрывать односложные слова (обе половины обязаны содержать слоговую гласную)
    let mut cut_points: Vec<usize> = Vec::new();
    if is_latin {
        let low_core: String = core_chars.iter().map(|c| c.to_ascii_lowercase()).collect();

        // 1. Чистые морфологические префиксы английского языка (>= 3 букв)
        for pref in [
            "under", "inter", "super", "trans", "counter", "ultra", "extra", "intro", "retro",
            "hyper", "cyber", "over", "anti", "auto", "semi", "tele", "micro", "multi", "mega",
            "giga", "kilo", "tera", "para", "meta", "abso", "dis", "mis", "pre", "pro", "sub",
            "out", "non",
        ] {
            let plen = pref.len();
            if plen >= 3
                && n >= plen + 4
                && low_core.starts_with(pref)
                && has_spoken_english_vowel(&core_chars[plen..])
            {
                cut_points.push(plen);
                break;
            }
        }

        // 2. Слогораздел по стечениям согласных (VC-CV: `break-fast`, `impor-tant`, `cir-cum-stances`, `black-out`),
        // без разрыва одиночной интервокальной согласной (`coo-perate`, `tho-usand`, `appe-ared`) и без разрыва дифтонгов!
        for i in 2..=(n - 3) {
            if !is_vowel(core_chars[i - 1]) || is_vowel(core_chars[i]) {
                continue;
            }
            if i + 1 >= n || is_vowel(core_chars[i + 1]) {
                continue;
            }
            let pair = format!(
                "{}{}",
                core_chars[i].to_ascii_lowercase(),
                core_chars[i + 1].to_ascii_lowercase()
            );
            let cut_idx = if pair == "ck" {
                i + 2
            } else if matches!(pair.as_str(), "th" | "sh" | "ch" | "ph" | "wh" | "gh") {
                let left_low = &low_core[..i];
                if matches!(left_low, "any" | "some" | "every") {
                    i
                } else {
                    i + 2
                }
            } else {
                i + 1
            };

            if cut_idx >= 3
                && cut_idx <= n - 3
                && core_chars[..cut_idx].iter().any(|&c| is_vowel(c))
                && has_spoken_english_vowel(&core_chars[cut_idx..])
                && cut_points.last().is_none_or(|&prev| cut_idx >= prev + 3)
            {
                cut_points.push(cut_idx);
            }
        }

        // 3. Продуктивные английские суффиксы (`unbeliev-able`, `insane-ly`, `care-lessly`, `apart-ment`)
        for suf in [
            "lessly", "fully", "ingly", "ation", "ition", "able", "ible", "tion", "sion", "ment",
            "ness", "less", "ward", "ship", "ing", "ly",
        ] {
            let slen = suf.len();
            if n >= slen + 4 && low_core.ends_with(suf) {
                let c_idx = n - slen;
                if c_idx >= 3
                    && core_chars[..c_idx].iter().any(|&c| is_vowel(c))
                    && has_spoken_english_vowel(&core_chars[c_idx..])
                {
                    if let Some(&last) = cut_points.last() {
                        if c_idx >= last + 3 {
                            cut_points.push(c_idx);
                        }
                    } else {
                        cut_points.push(c_idx);
                    }
                    break;
                }
            }
        }
    } else {
        for i in 3..=(n - 3) {
            let left_has_vowel = core_chars[..i].iter().any(|&c| is_vowel(c));
            let right_has_vowel = core_chars[i..].iter().any(|&c| is_vowel(c));
            if !left_has_vowel || !right_has_vowel {
                continue;
            }
            if is_vowel(core_chars[i - 1]) && !is_silent_sign(core_chars[i]) {
                let cut_idx = if i + 1 <= n - 3
                    && !is_vowel(core_chars[i])
                    && !is_vowel(core_chars[i + 1])
                    && !is_silent_sign(core_chars[i + 1])
                {
                    i + 1
                } else {
                    i
                };
                if core_chars[cut_idx..].iter().any(|&c| is_vowel(c))
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
            if mid >= 3 && n - mid >= 3 && left_v && right_v && !is_silent_sign(core_chars[mid]) {
                cut_points.push(mid);
            }
        }
    }

    if cut_points.is_empty() {
        return vec![word.to_string()];
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

/// DP-разбивка текста на строки с ромбовидным профилем (края уже середины, без висячих коротких слов и без дефисов внутри строки).
pub fn auto_diamond_wrap_lines<F: Fn(&str) -> f32>(
    text: &str,
    max_width_px: f32,
    measure_width: F,
) -> Vec<String> {
    auto_diamond_wrap_lines_opts(text, max_width_px, true, measure_width)
}

/// DP-разбивка текста на строки с управлением мягкими слоговыми переносами (`allow_soft_hyphen`).
pub fn auto_diamond_wrap_lines_opts<F: Fn(&str) -> f32>(
    text: &str,
    max_width_px: f32,
    allow_soft_hyphen: bool,
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
        let mut tokens: Vec<WrapToken> = Vec::new();
        for w in p_trim.split_whitespace() {
            let has_compound = w.contains('-') && !w.contains("--");
            let need_split = has_compound || (allow_soft_hyphen && measure_width(w) > max_w);
            tokens.extend(tokenize_word_for_wrap(w, need_split));
        }

        if tokens.is_empty() {
            continue;
        }
        if tokens.len() == 1 {
            raw_lines.push(tokens[0].text.clone());
            continue;
        }

        let n = tokens.len();
        let mut span_w = vec![vec![0.0f32; n + 1]; n];
        for i in 0..n {
            for j in (i + 1)..=n {
                let line = join_wrap_tokens(&tokens, i, j);
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
                let hy_pen = if j < n && tokens[j - 1].soft_hyphen_after {
                    120.0
                } else {
                    0.0
                };
                let orphan_pen =
                    if j == n && (j - i) == 1 && tokens[i].text.chars().count() <= 3 && i > 0 {
                        65.0
                    } else {
                        0.0
                    };
                let cost = slack * slack * 0.015 + 18.0 + overflow_pen + hy_pen + orphan_pen + dp[j];
                if cost < dp[i] {
                    dp[i] = cost;
                    next_idx[i] = j;
                }
            }
        }

        let mut cur = 0usize;
        while cur < n {
            let nxt = next_idx[cur].max(cur + 1).min(n);
            raw_lines.push(join_wrap_tokens(&tokens, cur, nxt));
            cur = nxt;
        }
    }

    raw_lines
}
