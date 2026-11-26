use crate::typography::auto_diamond_wrap_lines;
use miko_cleaner::distance_transform_l2;

#[derive(Debug, Clone)]
pub struct BubbleLayout {
    pub font_size: i32,
    pub scale_x: f32,
    pub lines: Vec<String>,
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

    for y in 0..=(h - bh) {
        for x in 0..=(w - bw) {
            let obstacles = integ[(y + bh) * iw + (x + bw)] + integ[y * iw + x]
                - integ[y * iw + (x + bw)]
                - integ[(y + bh) * iw + x];
            if obstacles == 0 {
                let cx = (x + bw / 2) as i32;
                let cy = (y + bh / 2) as i32;
                let dx = (cx - centroid[0]) as i64;
                let dy = (cy - centroid[1]) as i64;
                let d2 = dx * dx + dy * dy;
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

    let try_fs_with_scale = |fs: i32, scale_x: f32| -> (bool, BubbleLayout) {
        let fs_f = fs as f32;
        let line_h = (fs_f * 1.18).max(10.0);
        let target_wrap_w = (ref_w as f32 * 0.93).max(20.0);
        let lines = auto_diamond_wrap_lines(text, target_wrap_w, |s| measure_scaled(s, fs_f, scale_x));
        let total_h = (lines.len() as f32 * line_h).ceil() as usize;
        let max_line_w = lines
            .iter()
            .map(|l| measure_scaled(l, fs_f, scale_x))
            .fold(0.0f32, f32::max);

        let fits_h = total_h <= cav_h && !lines.is_empty();
        let mut fits_chords = false;
        let mut chosen_y_start = (cav_top + cav_h / 2).saturating_sub(total_h / 2);

        if fits_h {
            let ideal_start = cav_top + (cav_h - total_h) / 2;
            let min_start = cav_top;
            let max_start = cav_bot.saturating_sub(total_h).max(min_start);

            let check_at = |y_start: usize| -> bool {
                for (k, line) in lines.iter().enumerate() {
                    let lw = measure_scaled(line, fs_f, scale_x);
                    let ly0 = (y_start + (k as f32 * line_h).floor() as usize).clamp(0, h - 1);
                    let ly1 = (y_start + ((k + 1) as f32 * line_h).ceil() as usize).clamp(ly0 + 1, h);
                    let mut band_widths: Vec<usize> = Vec::with_capacity(ly1 - ly0);
                    for y in ly0..ly1 {
                        if let Some((l, r)) = chords[y] {
                            band_widths.push(r.saturating_sub(l));
                        } else {
                            band_widths.push(0);
                        }
                    }
                    if band_widths.is_empty() {
                        return false;
                    }
                    band_widths.sort_unstable();
                    // 20-й перцентиль полосы строки (устойчив к 1-2px ступеньке скругления на краю глифа)
                    let p20 = band_widths[(band_widths.len() * 20) / 100];
                    if p20 == 0 || lw > (p20 as f32 + 2.5) {
                        return false;
                    }
                }
                true
            };

            if check_at(ideal_start) {
                fits_chords = true;
                chosen_y_start = ideal_start;
            } else {
                let max_shift = (ideal_start - min_start).max(max_start - ideal_start);
                let mut delta = 4usize;
                while delta <= max_shift {
                    if ideal_start >= min_start + delta && check_at(ideal_start - delta) {
                        fits_chords = true;
                        chosen_y_start = ideal_start - delta;
                        break;
                    }
                    if ideal_start + delta <= max_start && check_at(ideal_start + delta) {
                        fits_chords = true;
                        chosen_y_start = ideal_start + delta;
                        break;
                    }
                    delta += 4;
                }
            }
        }

        let box_w = (max_line_w.ceil() as usize).max(4).min(w.max(1));
        let box_h = total_h.max(4).min(h.max(1));
        let mut center_xy = find_optimal_box_center(cavity_mask, w, h, box_w, box_h);
        if fits_chords {
            let cy_chord = (chosen_y_start + total_h / 2) as i32;
            let cy_row = (cy_chord as usize).min(h.saturating_sub(1));
            if let Some((l, r)) = chords[cy_row] {
                center_xy = [((l + r) / 2) as i32, cy_chord];
            }
        }

        (
            fits_chords,
            BubbleLayout {
                font_size: fs,
                scale_x,
                lines,
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
