use crate::typography::{auto_diamond_wrap_lines, hyphenate_word};
use miko_cleaner::distance_transform_l2;

#[derive(Debug, Clone)]
pub struct BubbleLayout {
    pub font_size: i32,
    pub scale_x: f32,
    pub lines: Vec<String>,
    pub line_centers_x: Vec<f32>,
    pub center_xy: [i32; 2],
    pub line_height: f32,
    pub max_line_w: f32,
}

/// Вычисляет `[left_x, right_x]` самого широкого непрерывного сегмента полости для каждой строки `y`.
pub fn compute_scanline_chords(mask: &[u8], w: usize, h: usize) -> Vec<Option<(usize, usize)>> {
    let mut out = vec![None; h];
    if w == 0 || h == 0 || mask.len() < w * h {
        return out;
    }
    let mid_x = w as isize / 2;
    for y in 0..h {
        let row = y * w;
        let mut best_run: Option<(usize, usize)> = None;
        let mut best_score = isize::MIN;
        let mut x = 0usize;
        while x < w {
            if mask[row + x] == 0 {
                x += 1;
                continue;
            }
            let start = x;
            while x < w && mask[row + x] > 0 {
                x += 1;
            }
            let end = x;
            let run_w = (end - start) as isize;
            let dist_mid = if start as isize <= mid_x && mid_x <= end as isize {
                0
            } else {
                (start as isize - mid_x).abs().min((end as isize - mid_x).abs())
            };
            let score = run_w * 4 - dist_mid;
            if score > best_score {
                best_score = score;
                best_run = Some((start, end));
            }
        }
        out[y] = best_run;
    }
    out
}

/// Находит оптимальный центр текстового блока `(box_w x box_h)` внутри маски полости через 2D интегральное изображение.
pub fn find_optimal_box_center(
    mask: &[u8],
    w: usize,
    h: usize,
    box_w: usize,
    box_h: usize,
) -> [i32; 2] {
    if w == 0 || h == 0 || mask.len() < w * h {
        return [0, 0];
    }

    let mut sum_x = 0usize;
    let mut sum_y = 0usize;
    let mut area = 0usize;
    let iw = w + 1;
    let mut integ = vec![0u32; iw * (h + 1)];

    for y in 0..h {
        let mut row_acc = 0u32;
        for x in 0..w {
            let is_valid = mask[y * w + x] > 0;
            if is_valid {
                sum_x += x;
                sum_y += y;
                area += 1;
            } else {
                row_acc += 1;
            }
            integ[(y + 1) * iw + (x + 1)] = integ[y * iw + (x + 1)] + row_acc;
        }
    }

    let centroid = if area > 0 {
        [(sum_x / area) as i32, (sum_y / area) as i32]
    } else {
        [(w / 2) as i32, (h / 2) as i32]
    };

    let bw = box_w.clamp(1, w);
    let bh = box_h.clamp(1, h);
    let mut best_dist = i64::MAX;
    let mut best_center: Option<[i32; 2]> = None;
    let max_y_drift = ((h as i32) * 20 / 100).max(16);

    for y in 0..=(h - bh) {
        let cy = (y + bh / 2) as i32;
        if (cy - centroid[1]).abs() > max_y_drift && (h - bh) > 24 {
            continue;
        }
        for x in 0..=(w - bw) {
            let obstacles = integ[(y + bh) * iw + (x + bw)] + integ[y * iw + x]
                - integ[y * iw + (x + bw)]
                - integ[(y + bh) * iw + x];
            if obstacles == 0 {
                let cx = (x + bw / 2) as i32;
                let dx = (cx - centroid[0]) as i64;
                let dy = (cy - centroid[1]) as i64;
                let d2 = dx * dx + dy * dy * 4;
                if d2 < best_dist {
                    best_dist = d2;
                    best_center = Some([cx, cy]);
                }
            }
        }
    }

    if let Some(c) = best_center {
        return c;
    }

    // Фоллбэк: самая глубокая точка внутри полости по L2 Distance Transform,
    // зажатая так, чтобы текстовый блок `(bw x bh)` не вылезал за верх/низ/края узкого баббла!
    if area > 0 {
        let dt = distance_transform_l2(mask, w, h);
        let mut max_d = -1.0f32;
        let mut peak = centroid;
        for y in 0..h {
            for x in 0..w {
                let d = dt[y * w + x];
                if d > max_d {
                    max_d = d;
                    peak = [x as i32, y as i32];
                }
            }
        }
        let half_w = (bw / 2) as i32;
        let half_h = (bh / 2) as i32;
        let lo_x = half_w.min(w as i32 / 2);
        let hi_x = ((w as i32) - half_w).max(lo_x);
        let lo_y = (half_h + 2).min(h as i32 / 2);
        let hi_y = ((h as i32) - half_h - 2).max(lo_y);
        // По вертикали в узких высоких бабблах держимся ближе к центроиду полости, а не к случайному расширению у края
        let blended_y = (peak[1] + centroid[1] * 2) / 3;
        return [peak[0].clamp(lo_x, hi_x), blended_y.clamp(lo_y, hi_y)];
    }

    centroid
}

/// Вычисляет строгое пересечение горизонтальных хорд на диапазоне скан-линий `[y0, y1)`.
fn band_strict_interval(
    chords: &[Option<(usize, usize)>],
    y0: usize,
    y1: usize,
) -> Option<(usize, usize)> {
    if y0 >= y1 || y1 > chords.len() {
        return None;
    }
    let mut xl = 0usize;
    let mut xr = usize::MAX;
    for ch_opt in &chords[y0..y1] {
        let (l, r) = (*ch_opt)?;
        xl = xl.max(l);
        xr = xr.min(r);
        if xl >= xr {
            return None;
        }
    }
    if xr == usize::MAX {
        None
    } else {
        Some((xl, xr))
    }
}

fn line_ink_band_y(
    y_start: usize,
    k: usize,
    line_h: f32,
    fs_f: f32,
    h: usize,
) -> (usize, usize) {
    let top_trim = (fs_f * 0.08).max(0.0);
    let bot_trim = (fs_f * 0.08).max(0.0);
    let y0_raw = y_start as f32 + (k as f32) * line_h + top_trim;
    let y1_raw = y_start as f32 + ((k + 1) as f32) * line_h - bot_trim;
    let ly0 = (y0_raw.floor().max(0.0) as usize).min(h.saturating_sub(1));
    let ly1 = (y1_raw.ceil().max(0.0) as usize).clamp(ly0 + 1, h);
    (ly0, ly1)
}

/// Проверяет, что каждая строка с учётом реального интервала `[xl, xr]` и вертикальных границ
/// помещается внутрь полости баббла, и возвращает сбалансированные центры `X` по строкам.
#[allow(clippy::too_many_arguments)]
fn validate_lines_in_chords<F: Fn(&str, f32, f32) -> f32>(
    lines: &[String],
    chords: &[Option<(usize, usize)>],
    h: usize,
    y_start: usize,
    line_h: f32,
    fs_f: f32,
    scale_x: f32,
    side_pad: f32,
    ref_w: usize,
    measure_scaled: &F,
) -> Option<Vec<f32>> {
    if lines.is_empty() || h == 0 {
        return None;
    }
    let total_h = (lines.len() as f32 * line_h).ceil() as usize;
    let win_y0 = y_start.min(h.saturating_sub(1));
    let win_y1 = (y_start + total_h).clamp(win_y0 + 1, h);
    let mut band_centers: Vec<f32> = Vec::with_capacity(win_y1 - win_y0);
    for ch_opt in &chords[win_y0..win_y1] {
        if let Some((l, r)) = *ch_opt {
            band_centers.push((l + r) as f32 * 0.5);
        }
    }
    let global_cx = if band_centers.is_empty() {
        return None;
    } else {
        band_centers.sort_by(|a, b| a.total_cmp(b));
        band_centers[band_centers.len() / 2]
    };

    let max_step = (ref_w as f32 * 0.35).max(26.0);
    let mut line_cxs: Vec<f32> = Vec::with_capacity(lines.len());
    for (k, line) in lines.iter().enumerate() {
        let lw = measure_scaled(line, fs_f, scale_x);
        let (ly0, ly1) = line_ink_band_y(y_start, k, line_h, fs_f, h);
        let (xl, xr) = band_strict_interval(chords, ly0, ly1)?;
        let avail = xr.saturating_sub(xl) as f32;
        if lw + 2.0 * side_pad > avail {
            return None;
        }
        let min_cx = xl as f32 + lw * 0.5 + side_pad;
        let max_cx = xr as f32 - lw * 0.5 - side_pad;
        let cx_k = if min_cx <= max_cx {
            global_cx.clamp(min_cx, max_cx)
        } else {
            (xl + xr) as f32 * 0.5
        };
        if let Some(&prev_cx) = line_cxs.last() {
            if (cx_k - prev_cx).abs() > max_step {
                return None;
            }
        }
        line_cxs.push(cx_k);
    }
    Some(line_cxs)
}

/// Chord-aware 2D-DP разбиение текста под сужающиеся верхние/нижние хорды овала с сохранением вертикальной центровки.
#[allow(clippy::too_many_arguments)]
fn wrap_lines_to_chords<F: Fn(&str, f32, f32) -> f32>(
    text: &str,
    chords: &[Option<(usize, usize)>],
    h: usize,
    cav_top: usize,
    cav_bot: usize,
    line_h: f32,
    fs_f: f32,
    scale_x: f32,
    side_pad: f32,
    ref_w: usize,
    measure_scaled: &F,
) -> Option<(Vec<String>, Vec<f32>, usize)> {
    let clean = text.trim();
    if clean.is_empty() || clean.lines().count() > 1 {
        return None;
    }

    let words: Vec<&str> = clean.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }

    let cav_h = cav_bot.saturating_sub(cav_top);
    let max_possible_k = (cav_h as f32 / line_h).floor() as usize;
    if max_possible_k == 0 {
        return None;
    }

    let mut tokens: Vec<String> = Vec::new();
    for w in words {
        let w_len = w.chars().count();
        if w_len >= 8 || measure_scaled(w, fs_f, scale_x) > (ref_w as f32 * 0.72) {
            let parts = hyphenate_word(w);
            if parts.len() > 1 {
                for (idx, part) in parts.iter().enumerate() {
                    if idx + 1 < parts.len() && !part.ends_with('-') {
                        tokens.push(format!("{part}-"));
                    } else {
                        tokens.push(part.clone());
                    }
                }
                continue;
            }
        }
        tokens.push(w.to_string());
    }

    let n = tokens.len();
    let join_span = |i: usize, j: usize| -> String {
        let mut out = String::new();
        for k in i..j {
            if !out.is_empty() && !out.ends_with('-') {
                out.push(' ');
            }
            out.push_str(&tokens[k]);
        }
        out
    };

    let mut span_w = vec![vec![0.0f32; n + 1]; n];
    for i in 0..n {
        for j in (i + 1)..=n {
            let s = join_span(i, j);
            span_w[i][j] = measure_scaled(&s, fs_f, scale_x);
        }
    }

    let k_max = max_possible_k.min(n);
    let shift_cap = ((cav_h * 14) / 100).max(6);
    let mut best_cost = f32::INFINITY;
    let mut best_res: Option<(Vec<String>, Vec<f32>, usize)> = None;

    for k_lines in 1..=k_max {
        let k_total_h = (k_lines as f32 * line_h).ceil() as usize;
        if k_total_h > cav_h {
            break;
        }
        let k_free = cav_h - k_total_h;
        let k_min_margin = (k_free * 32) / 100;
        let k_ideal = cav_top + k_free / 2;
        let k_lo = k_ideal
            .saturating_sub(shift_cap)
            .max(cav_top + k_min_margin);
        let k_hi = (k_ideal + shift_cap)
            .min(cav_bot.saturating_sub(k_total_h + k_min_margin))
            .max(k_lo);

        for &dy in &[0isize, -4, 4, -8, 8, -12, 12] {
            let y_start = (k_ideal as isize + dy).clamp(k_lo as isize, k_hi as isize) as usize;
            if dy != 0 && (y_start == k_ideal || (dy.unsigned_abs() > shift_cap)) {
                continue;
            }

            let mut limits = Vec::with_capacity(k_lines);
            let mut valid_k = true;
            for k in 0..k_lines {
                let (ly0, ly1) = line_ink_band_y(y_start, k, line_h, fs_f, h);
                let Some((xl, xr)) = band_strict_interval(chords, ly0, ly1) else {
                    valid_k = false;
                    break;
                };
                let avail = xr.saturating_sub(xl) as f32 - 2.0 * side_pad;
                if avail <= fs_f * 0.7 {
                    valid_k = false;
                    break;
                }
                limits.push(avail);
            }
            if !valid_k {
                continue;
            }

            let inf = 1.0e12f32;
            let mut dp = vec![vec![inf; k_lines + 1]; n + 1];
            let mut parent = vec![vec![usize::MAX; k_lines + 1]; n + 1];
            dp[0][0] = 0.0;

            for k in 1..=k_lines {
                let lim = limits[k - 1];
                let rel_pos = ((k as f32 - 0.5) / (k_lines as f32) - 0.5).abs() * 2.0;
                let target_fill = 0.88 - 0.14 * rel_pos.powf(1.4);

                for i in 1..=n {
                    for j in (0..i).rev() {
                        let w = span_w[j][i];
                        if w > lim {
                            if i > j + 1 {
                                break;
                            }
                            continue;
                        }
                        if dp[j][k - 1] >= inf * 0.5 {
                            continue;
                        }
                        let fill = (w / lim.max(1.0)).clamp(0.0, 1.0);
                        let slack_pen = (1.0 - fill).powi(2) * 45.0;
                        let shape_pen = (fill - target_fill).powi(2) * 28.0;
                        let hy_pen = if tokens[i - 1].ends_with('-') { 8.0 } else { 0.0 };
                        let orphan_pen = if k == k_lines
                            && (i - j) == 1
                            && tokens[j].chars().count() <= 3
                            && j > 0
                        {
                            35.0
                        } else {
                            0.0
                        };
                        let cost = dp[j][k - 1] + slack_pen + shape_pen + hy_pen + orphan_pen;
                        if cost < dp[i][k] {
                            dp[i][k] = cost;
                            parent[i][k] = j;
                        }
                    }
                }
            }

            if dp[n][k_lines] >= inf * 0.5 {
                continue;
            }

            let mut lines = Vec::with_capacity(k_lines);
            let mut cur = n;
            let mut ok_chain = true;
            for k in (1..=k_lines).rev() {
                let prev = parent[cur][k];
                if prev >= cur || prev == usize::MAX {
                    ok_chain = false;
                    break;
                }
                lines.push(join_span(prev, cur));
                cur = prev;
            }
            if !ok_chain || cur != 0 {
                continue;
            }
            lines.reverse();

            if let Some(cxs) = validate_lines_in_chords(
                &lines,
                chords,
                h,
                y_start,
                line_h,
                fs_f,
                scale_x,
                side_pad,
                ref_w,
                measure_scaled,
            ) {
                let shift_pen = (dy.abs() as f32) * 0.8;
                let total_cost = dp[n][k_lines] + (k_lines as f32) * 14.0 + shift_pen;
                if total_cost < best_cost {
                    best_cost = total_cost;
                    best_res = Some((lines, cxs, y_start));
                }
            }
        }
    }

    best_res
}

/// Подбирает максимальный кегль в `[min_fs..=max_fs]` и оптимальную геометрию переноса по хордам полости баббла.
pub fn solve_bubble_layout<F: Fn(&str, f32) -> f32>(
    text: &str,
    cavity_mask: &[u8],
    w: usize,
    h: usize,
    min_fs: i32,
    max_fs: i32,
    fixed_fs: Option<i32>,
    measure: F,
) -> BubbleLayout {
    solve_bubble_layout_scaled(
        text,
        cavity_mask,
        w,
        h,
        min_fs,
        max_fs,
        fixed_fs,
        |s, fs, sx| measure(s, fs) * sx,
    )
}

/// Расширенный решатель вёрстки с поддержкой анаморфного горизонтального микро-сжатия (`0.88..=1.0`)
/// для узких японских вертикальных бабблов и поиском лучшего вертикального окна по реальным хордам.
pub fn solve_bubble_layout_scaled<F: Fn(&str, f32, f32) -> f32>(
    text: &str,
    cavity_mask: &[u8],
    w: usize,
    h: usize,
    min_fs: i32,
    max_fs: i32,
    fixed_fs: Option<i32>,
    measure_scaled: F,
) -> BubbleLayout {
    solve_bubble_layout_with_options(
        text,
        cavity_mask,
        w,
        h,
        min_fs,
        max_fs,
        fixed_fs,
        true,
        measure_scaled,
    )
}

/// Полная версия решателя вёрстки с управлением режимом переполнения (`allow_emergency_shrink`).
#[allow(clippy::too_many_arguments)]
pub fn solve_bubble_layout_with_options<F: Fn(&str, f32, f32) -> f32>(
    text: &str,
    cavity_mask: &[u8],
    w: usize,
    h: usize,
    min_fs: i32,
    max_fs: i32,
    fixed_fs: Option<i32>,
    allow_emergency_shrink: bool,
    measure_scaled: F,
) -> BubbleLayout {
    let chords = compute_scanline_chords(cavity_mask, w, h);
    let mut valid_rows: Vec<(usize, usize)> = Vec::new();
    for (y, ch_opt) in chords.iter().enumerate() {
        if let Some((l, r)) = ch_opt {
            valid_rows.push((y, r.saturating_sub(*l)));
        }
    }

    let (cav_top, cav_bot, ref_w) = if valid_rows.is_empty() {
        (0usize, h.max(1), w.max(1))
    } else {
        let top = valid_rows.first().unwrap().0;
        let bot = valid_rows.last().unwrap().0 + 1;
        let mut widths: Vec<usize> = valid_rows.iter().map(|&(_, cw)| cw).collect();
        widths.sort_unstable();
        // Берём 70-й перцентиль хорд вместо единственной средней строки, чтобы случайная перетяжка не роняла кегль!
        let p70_idx = ((widths.len() * 70) / 100).min(widths.len() - 1);
        let rw = widths[p70_idx].max(16);
        (top, bot, rw)
    };
    let cav_h = (cav_bot - cav_top).max(16);
    let allow_condense = cav_h * 100 >= ref_w * 155 || ref_w < 130;
    let side_pad = 1.5f32;

    let try_fs_with_scale = |fs: i32, scale_x: f32| -> (bool, BubbleLayout) {
        let fs_f = fs as f32;
        let line_h = (fs_f * 1.18).max(10.0);
        let target_wrap_w = (ref_w as f32 * 0.90).max(20.0);
        let mut lines =
            auto_diamond_wrap_lines(text, target_wrap_w, |s| measure_scaled(s, fs_f, scale_x));
        let mut total_h = (lines.len() as f32 * line_h).ceil() as usize;

        let mut fits_chords = false;
        let mut chosen_y_start = (cav_top + cav_h / 2).saturating_sub(total_h / 2);
        let mut line_centers_x: Vec<f32> = Vec::new();

        if total_h <= cav_h && !lines.is_empty() {
            let free_v = cav_h - total_h;
            let min_margin = (free_v * 32) / 100;
            let ideal_start = cav_top + free_v / 2;
            let shift_cap = ((cav_h * 14) / 100).max(6);
            let min_start = ideal_start
                .saturating_sub(shift_cap)
                .max(cav_top + min_margin);
            let max_start = (ideal_start + shift_cap)
                .min(cav_bot.saturating_sub(total_h + min_margin))
                .max(min_start);

            if let Some(cxs) = validate_lines_in_chords(
                &lines,
                &chords,
                h,
                ideal_start,
                line_h,
                fs_f,
                scale_x,
                side_pad,
                ref_w,
                &measure_scaled,
            ) {
                fits_chords = true;
                chosen_y_start = ideal_start;
                line_centers_x = cxs;
            } else {
                let max_shift = ideal_start
                    .saturating_sub(min_start)
                    .max(max_start.saturating_sub(ideal_start));
                let mut delta = 4usize;
                while delta <= max_shift {
                    if ideal_start >= min_start + delta {
                        if let Some(cxs) = validate_lines_in_chords(
                            &lines,
                            &chords,
                            h,
                            ideal_start - delta,
                            line_h,
                            fs_f,
                            scale_x,
                            side_pad,
                            ref_w,
                            &measure_scaled,
                        ) {
                            fits_chords = true;
                            chosen_y_start = ideal_start - delta;
                            line_centers_x = cxs;
                            break;
                        }
                    }
                    if ideal_start + delta <= max_start {
                        if let Some(cxs) = validate_lines_in_chords(
                            &lines,
                            &chords,
                            h,
                            ideal_start + delta,
                            line_h,
                            fs_f,
                            scale_x,
                            side_pad,
                            ref_w,
                            &measure_scaled,
                        ) {
                            fits_chords = true;
                            chosen_y_start = ideal_start + delta;
                            line_centers_x = cxs;
                            break;
                        }
                    }
                    delta += 4;
                }
            }
        }

        // Если ровный ромб не поместился по узким верхним/нижним хордам овала,
        // пересчитываем перенос строк строго под геометрию каждой скан-линии вокруг центра!
        if !fits_chords {
            if let Some((dp_lines, dp_cxs, dp_y_start)) = wrap_lines_to_chords(
                text,
                &chords,
                h,
                cav_top,
                cav_bot,
                line_h,
                fs_f,
                scale_x,
                side_pad,
                ref_w,
                &measure_scaled,
            ) {
                let dp_total_h = (dp_lines.len() as f32 * line_h).ceil() as usize;
                if dp_total_h <= cav_h {
                    lines = dp_lines;
                    total_h = dp_total_h;
                    chosen_y_start = dp_y_start;
                    line_centers_x = dp_cxs;
                    fits_chords = true;
                }
            }
        }

        let max_line_w = lines
            .iter()
            .map(|l| measure_scaled(l, fs_f, scale_x))
            .fold(0.0f32, f32::max);

        let box_w = (max_line_w.ceil() as usize).max(4).min(w.max(1));
        let box_h = total_h.max(4).min(h.max(1));
        let mut center_xy = find_optimal_box_center(cavity_mask, w, h, box_w, box_h);
        if fits_chords && !line_centers_x.is_empty() {
            let cy_chord = (chosen_y_start + total_h / 2) as i32;
            let avg_cx =
                (line_centers_x.iter().sum::<f32>() / (line_centers_x.len() as f32)).round() as i32;
            center_xy = [avg_cx, cy_chord];
        } else if line_centers_x.len() != lines.len() {
            line_centers_x = vec![center_xy[0] as f32; lines.len()];
        }

        (
            fits_chords,
            BubbleLayout {
                font_size: fs,
                scale_x,
                lines,
                line_centers_x,
                center_xy,
                line_height: line_h,
                max_line_w,
            },
        )
    };

    let try_fs = |fs: i32| -> (bool, BubbleLayout) {
        let (ok1, lay1) = try_fs_with_scale(fs, 1.0);
        if ok1 || !allow_condense {
            return (ok1, lay1);
        }
        for &sx in &[0.93f32, 0.88f32] {
            let (ok_s, lay_s) = try_fs_with_scale(fs, sx);
            if ok_s {
                return (true, lay_s);
            }
        }
        (false, lay1)
    };

    if let Some(fs_fix) = fixed_fs {
        let (_, lay) = try_fs(fs_fix.clamp(6, 220));
        return lay;
    }

    let lo_init = min_fs.clamp(8, 120);
    let hi_init = max_fs.clamp(lo_init, 200);
    let mut lo = lo_init;
    let mut hi = hi_init;
    let mut best_layout: Option<BubbleLayout> = None;

    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let (ok, lay) = try_fs(mid);
        if ok {
            best_layout = Some(lay);
            lo = mid + 1;
        } else {
            hi = mid.saturating_sub(1);
            if mid == 0 {
                break;
            }
        }
    }

    if let Some(lay) = best_layout {
        return lay;
    }

    if !allow_emergency_shrink {
        return try_fs(lo_init).1;
    }

    // Если даже `min_fs` чуть шире узкого баббла (например, узкий крик `Shut uuuup!` на `002.webp`),
    // аккуратно пробуем спуститься до 9px, чтобы слова не залезали на чёрный контур баббла!
    let emergency_lo = lo_init.min(9).max(8);
    for fs_try in (emergency_lo..lo_init).rev() {
        let (ok, lay) = try_fs(fs_try);
        if ok {
            return lay;
        }
    }

    try_fs(emergency_lo).1
}

/// Двухпроходная гармонизация кеглей на уровне страницы: гасит аномально огромные шрифты в коротких репликах.
pub fn harmonize_page_font_sizes(raw_sizes: &[(usize, i32, usize, bool)]) -> Vec<i32> {
    let mut out: Vec<i32> = raw_sizes.iter().map(|&(_, fs, _, _)| fs).collect();
    let mut weighted: Vec<(i32, f32)> = raw_sizes
        .iter()
        .filter(|&&(_, _, chars, fixed)| !fixed && chars > 0)
        .map(|&(_, fs, chars, _)| (fs, (chars as f32).sqrt().max(1.0)))
        .collect();

    if weighted.len() < 2 {
        return out;
    }

    weighted.sort_by_key(|&(fs, _)| fs);
    let total_w: f32 = weighted.iter().map(|&(_, w)| w).sum();
    let half = total_w * 0.5;
    let mut acc = 0.0f32;
    let mut base_fs = weighted[weighted.len() / 2].0 as f32;
    for &(fs, w) in &weighted {
        acc += w;
        if acc >= half {
            base_fs = fs as f32;
            break;
        }
    }

    let cap_hi = (base_fs * 1.25).round() as i32;
    for (pos, &(_, raw_fs, _, fixed)) in raw_sizes.iter().enumerate() {
        if !fixed {
            out[pos] = raw_fs.min(cap_hi.max(10));
        }
    }
    out
}
