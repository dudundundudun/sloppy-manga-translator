#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CcComponent {
    pub label: u32,
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
    pub area: usize,
}

#[derive(Debug, Clone)]
pub struct CcResult {
    pub width: usize,
    pub height: usize,
    pub labels: Vec<u32>,
    pub stats: Vec<CcComponent>,
}

/// Порог Отсу за один линейный проход по 256-бинной гистограмме.
pub fn otsu_threshold(gray: &[u8]) -> u8 {
    if gray.is_empty() {
        return 127;
    }
    let mut hist = [0u64; 256];
    let mut total_sum = 0.0f64;
    for &px in gray {
        hist[px as usize] += 1;
        total_sum += px as f64;
    }
    let total = gray.len() as f64;
    let mut sum_b = 0.0f64;
    let mut w_b = 0.0f64;
    let mut max_var = -1.0f64;
    let mut sum_best_t = 0usize;
    let mut count_best_t = 0usize;

    for (t, &cnt) in hist.iter().enumerate() {
        w_b += cnt as f64;
        if w_b <= 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f <= 0.0 {
            break;
        }
        sum_b += (t as f64) * (cnt as f64);
        let m_b = sum_b / w_b;
        let m_f = (total_sum - sum_b) / w_f;
        let diff = m_b - m_f;
        let var_between = w_b * w_f * diff * diff;
        if var_between > max_var + 1e-6 {
            max_var = var_between;
            sum_best_t = t;
            count_best_t = 1;
        } else if (var_between - max_var).abs() <= 1e-6 {
            sum_best_t += t;
            count_best_t += 1;
        }
    }
    if count_best_t > 0 {
        (sum_best_t / count_best_t) as u8
    } else {
        127
    }
}

/// 8-связные компоненты связности на Union-Find без OpenCV.
pub fn connected_components(binary: &[u8], width: usize, height: usize) -> CcResult {
    let n = width.saturating_mul(height);
    let mut labels = vec![0u32; n];
    if n == 0 || binary.len() < n {
        return CcResult {
            width,
            height,
            labels,
            stats: Vec::new(),
        };
    }

    let mut parent: Vec<u32> = vec![0];
    let find = |parent: &mut Vec<u32>, mut x: u32| -> u32 {
        while parent[x as usize] != x {
            let p = parent[x as usize];
            parent[x as usize] = parent[p as usize];
            x = p;
        }
        x
    };
    let union = |parent: &mut Vec<u32>, a: u32, b: u32| -> u32 {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            ra
        } else if ra < rb {
            parent[rb as usize] = ra;
            ra
        } else {
            parent[ra as usize] = rb;
            rb
        }
    };

    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            if binary[row + x] == 0 {
                continue;
            }
            let mut nbrs = [0u32; 4];
            let mut nb_cnt = 0usize;
            if x > 0 && labels[row + x - 1] > 0 {
                nbrs[nb_cnt] = labels[row + x - 1];
                nb_cnt += 1;
            }
            if y > 0 {
                let prev = (y - 1) * width;
                if x > 0 && labels[prev + x - 1] > 0 {
                    nbrs[nb_cnt] = labels[prev + x - 1];
                    nb_cnt += 1;
                }
                if labels[prev + x] > 0 {
                    nbrs[nb_cnt] = labels[prev + x];
                    nb_cnt += 1;
                }
                if x + 1 < width && labels[prev + x + 1] > 0 {
                    nbrs[nb_cnt] = labels[prev + x + 1];
                    nb_cnt += 1;
                }
            }
            if nb_cnt == 0 {
                let next_id = parent.len() as u32;
                parent.push(next_id);
                labels[row + x] = next_id;
            } else {
                let mut min_lbl = nbrs[0];
                for &k in &nbrs[1..nb_cnt] {
                    min_lbl = union(&mut parent, min_lbl, k);
                }
                labels[row + x] = min_lbl;
            }
        }
    }

    let mut remap = vec![0u32; parent.len()];
    let mut final_count = 0u32;
    for i in 1..parent.len() {
        let r = find(&mut parent, i as u32);
        if r == i as u32 {
            final_count += 1;
            remap[i] = final_count;
        }
    }
    for i in 1..parent.len() {
        let r = find(&mut parent, i as u32);
        remap[i] = remap[r as usize];
    }

    let mut min_x = vec![usize::MAX; final_count as usize];
    let mut min_y = vec![usize::MAX; final_count as usize];
    let mut max_x = vec![0usize; final_count as usize];
    let mut max_y = vec![0usize; final_count as usize];
    let mut area = vec![0usize; final_count as usize];

    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let raw_l = labels[row + x];
            if raw_l == 0 {
                continue;
            }
            let fl = remap[raw_l as usize];
            labels[row + x] = fl;
            let idx = (fl - 1) as usize;
            min_x[idx] = min_x[idx].min(x);
            min_y[idx] = min_y[idx].min(y);
            max_x[idx] = max_x[idx].max(x);
            max_y[idx] = max_y[idx].max(y);
            area[idx] += 1;
        }
    }

    let stats = (0..final_count as usize)
        .map(|i| CcComponent {
            label: (i + 1) as u32,
            left: min_x[i],
            top: min_y[i],
            width: max_x[i] - min_x[i] + 1,
            height: max_y[i] - min_y[i] + 1,
            area: area[i],
        })
        .collect();

    CcResult {
        width,
        height,
        labels,
        stats,
    }
}

/// Точное 1D преобразование квадрата расстояния Фельзеншвальба–Хаттенлохера за O(N).
fn dt_1d(f: &[f32], d: &mut [f32], v: &mut [isize], z: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k: usize = 0;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    for q in 1..n {
        let qf = q as f32;
        let fq = f[q];
        loop {
            let vk = v[k] as usize;
            let vk_f = vk as f32;
            let denom = 2.0 * (qf - vk_f);
            let s = ((fq + qf * qf) - (f[vk] + vk_f * vk_f)) / denom;
            if s <= z[k] && k > 0 {
                k -= 1;
            } else {
                k += 1;
                v[k] = q as isize;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
        }
    }

    let mut k_scan: usize = 0;
    for q in 0..n {
        let qf = q as f32;
        while z[k_scan + 1] < qf {
            k_scan += 1;
        }
        let vk = v[k_scan] as usize;
        let dx = qf - (vk as f32);
        d[q] = dx * dx + f[vk];
    }
}

/// Точный линейный O(N) L2 Distance Transform (расстояние до ближайшего нулевого пикселя фона).
pub fn distance_transform_l2(binary: &[u8], width: usize, height: usize) -> Vec<f32> {
    let n = width.saturating_mul(height);
    if n == 0 || binary.len() < n {
        return vec![0.0; n];
    }
    let inf = 1.0e12f32;
    let mut grid: Vec<f32> = binary[..n]
        .iter()
        .map(|&px| if px == 0 { 0.0 } else { inf })
        .collect();

    let max_dim = width.max(height);
    let mut f_buf = vec![0.0f32; max_dim];
    let mut d_buf = vec![0.0f32; max_dim];
    let mut v_buf = vec![0isize; max_dim];
    let mut z_buf = vec![0.0f32; max_dim + 1];

    // Проход по столбцам
    for x in 0..width {
        for y in 0..height {
            f_buf[y] = grid[y * width + x];
        }
        dt_1d(
            &f_buf[..height],
            &mut d_buf[..height],
            &mut v_buf,
            &mut z_buf,
        );
        for y in 0..height {
            grid[y * width + x] = d_buf[y];
        }
    }

    // Проход по строкам + sqrt
    for y in 0..height {
        let row = y * width;
        f_buf[..width].copy_from_slice(&grid[row..row + width]);
        dt_1d(&f_buf[..width], &mut d_buf[..width], &mut v_buf, &mut z_buf);
        for x in 0..width {
            grid[row + x] = d_buf[x].sqrt();
        }
    }

    grid
}

/// Кубический сплайн Эрмита H(u) для замыкания разорванной стенки баббла по концевым касательным.
pub fn hermite_wall_bridge(
    p_a: [f32; 2],
    t_a: [f32; 2],
    p_b: [f32; 2],
    t_b: [f32; 2],
    num_pts: usize,
) -> Vec<[i32; 2]> {
    let steps = num_pts.max(4);
    let dx = p_b[0] - p_a[0];
    let dy = p_b[1] - p_a[1];
    let chord = (dx * dx + dy * dy).sqrt();
    let mut out = Vec::with_capacity(steps);
    for i in 0..steps {
        let u = (i as f32) / ((steps - 1) as f32);
        let u2 = u * u;
        let u3 = u2 * u;
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        let x = h00 * p_a[0] + h10 * (chord * t_a[0]) + h01 * p_b[0] + h11 * (chord * t_b[0]);
        let y = h00 * p_a[1] + h10 * (chord * t_a[1]) + h01 * p_b[1] + h11 * (chord * t_b[1]);
        out.push([x.round() as i32, y.round() as i32]);
    }
    out
}

/// Растеризует произвольный полигон в бинарную маску (`0`/`255`) методом scanline even-odd.
pub fn fill_polygon(poly: &[[i32; 2]], width: usize, height: usize) -> Vec<u8> {
    let n = width.saturating_mul(height);
    let mut mask = vec![0u8; n];
    if n == 0 || poly.len() < 3 {
        return mask;
    }

    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    for &[_, y] in poly {
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let y0 = min_y.clamp(0, height as i32) as usize;
    let y1 = max_y.saturating_add(1).clamp(0, height as i32) as usize;
    if y0 >= y1 {
        return mask;
    }

    let mut xs = Vec::with_capacity(poly.len());
    let m = poly.len();
    for y in y0..y1 {
        xs.clear();
        let yf = y as f32 + 0.5;
        for i in 0..m {
            let [x_a, y_a] = poly[i];
            let [x_b, y_b] = poly[(i + 1) % m];
            let ya = y_a as f32;
            let yb = y_b as f32;
            if (ya <= yf && yb > yf) || (yb <= yf && ya > yf) {
                let t = (yf - ya) / (yb - ya);
                let xi = x_a as f32 + t * ((x_b - x_a) as f32);
                xs.push(xi);
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for pair in xs.chunks_exact(2) {
            let sx = (pair[0].ceil() as i32).clamp(0, width as i32) as usize;
            let ex = (pair[1].floor() as i32)
                .saturating_add(1)
                .clamp(0, width as i32) as usize;
            if sx < ex {
                let row = y * width;
                for x in sx..ex {
                    mask[row + x] = 255;
                }
            }
        }
    }
    mask
}

/// Рисует заполненный круг радиуса `radius` в одноканальной маске.
pub fn draw_circle(mask: &mut [u8], w: usize, h: usize, cx: i32, cy: i32, radius: i32, val: u8) {
    let n = w.saturating_mul(h);
    if w == 0 || h == 0 || mask.len() < n {
        return;
    }
    if radius <= 0 {
        if cx >= 0 && cy >= 0 && (cx as usize) < w && (cy as usize) < h {
            mask[(cy as usize) * w + (cx as usize)] = val;
        }
        return;
    }
    let cx64 = cx as i64;
    let cy64 = cy as i64;
    let r64 = radius as i64;
    let w64 = w as i64;
    let h64 = h as i64;
    if cx64 + r64 < 0 || cx64 - r64 >= w64 || cy64 + r64 < 0 || cy64 - r64 >= h64 {
        return;
    }

    let r2 = r64 * r64;
    let y0 = (cy64 - r64).clamp(0, h64) as usize;
    let y1 = (cy64 + r64 + 1).clamp(0, h64) as usize;
    let x0 = (cx64 - r64).clamp(0, w64) as usize;
    let x1 = (cx64 + r64 + 1).clamp(0, w64) as usize;
    for y in y0..y1 {
        let dy = y as i64 - cy64;
        let dy2 = dy * dy;
        let row = y * w;
        for x in x0..x1 {
            let dx = x as i64 - cx64;
            if dx * dx + dy2 <= r2 {
                mask[row + x] = val;
            }
        }
    }
}

/// Рисует ломаную линию толщиной `radius` (через круговую кисть).
pub fn draw_polyline(mask: &mut [u8], w: usize, h: usize, pts: &[[i32; 2]], radius: i32, val: u8) {
    if pts.is_empty() {
        return;
    }
    if pts.len() == 1 {
        draw_circle(mask, w, h, pts[0][0], pts[0][1], radius, val);
        return;
    }
    let max_steps = (w.max(h) as i64 * 4).max(64);
    for seg in pts.windows(2) {
        let [x0, y0] = seg[0];
        let [x1, y1] = seg[1];
        let steps = (x1 as i64 - x0 as i64)
            .abs()
            .max((y1 as i64 - y0 as i64).abs())
            .clamp(1, max_steps) as usize;
        for s in 0..=steps {
            let t = (s as f32) / (steps as f32);
            let px = (x0 as f32 + t * ((x1 as f32) - (x0 as f32))).round() as i32;
            let py = (y0 as f32 + t * ((y1 as f32) - (y0 as f32))).round() as i32;
            draw_circle(mask, w, h, px, py, radius, val);
        }
    }
}

/// Точная евклидова дилатация круглым ядром радиуса `radius` за O(N) через L2 Distance Transform.
pub fn dilate_ellipse(mask: &[u8], w: usize, h: usize, radius: usize) -> Vec<u8> {
    let n = w.saturating_mul(h);
    if n == 0 || mask.len() < n {
        return vec![0u8; n];
    }
    if radius == 0 {
        return mask[..n]
            .iter()
            .map(|&p| if p > 0 { 255 } else { 0 })
            .collect();
    }
    if !mask[..n].iter().any(|&p| p > 0) {
        return vec![0u8; n];
    }
    let inv: Vec<u8> = mask[..n]
        .iter()
        .map(|&p| if p > 0 { 0 } else { 255 })
        .collect();
    let dt = distance_transform_l2(&inv, w, h);
    let r = radius as f32 + 1e-3;
    dt.into_iter()
        .map(|d| if d <= r { 255 } else { 0 })
        .collect()
}

/// Точная евклидова эрозия круглым ядром радиуса `radius` за O(N) через L2 Distance Transform.
pub fn erode_ellipse(mask: &[u8], w: usize, h: usize, radius: usize) -> Vec<u8> {
    let n = w.saturating_mul(h);
    if n == 0 || mask.len() < n {
        return vec![0u8; n];
    }
    if radius == 0 {
        return mask[..n]
            .iter()
            .map(|&p| if p > 0 { 255 } else { 0 })
            .collect();
    }
    if !mask[..n].iter().any(|&p| p == 0) {
        return vec![255u8; n];
    }
    let dt = distance_transform_l2(&mask[..n], w, h);
    let r = radius as f32 + 1e-3;
    dt.into_iter()
        .map(|d| if d >= r { 255 } else { 0 })
        .collect()
}

/// Вычисляет поканальную медиану RGB и среднее стандартное отклонение по пикселям `selector > 0`.
pub fn compute_rgb_median_and_std(rgb: &[u8], selector: &[u8]) -> Option<([u8; 3], f32)> {
    let n = selector.len();
    if n == 0 || rgb.len() < n * 3 {
        return None;
    }
    let mut hr = [0u64; 256];
    let mut hg = [0u64; 256];
    let mut hb = [0u64; 256];
    let mut count = 0u64;
    let mut sum = [0.0f64; 3];
    let mut sum_sq = [0.0f64; 3];

    for (i, &sel) in selector.iter().enumerate() {
        if sel == 0 {
            continue;
        }
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        hr[r as usize] += 1;
        hg[g as usize] += 1;
        hb[b as usize] += 1;
        sum[0] += r as f64;
        sum[1] += g as f64;
        sum[2] += b as f64;
        sum_sq[0] += (r as f64) * (r as f64);
        sum_sq[1] += (g as f64) * (g as f64);
        sum_sq[2] += (b as f64) * (b as f64);
        count += 1;
    }
    if count == 0 {
        return None;
    }

    let hist_median = |hist: &[u64; 256]| -> u8 {
        let target = (count + 1) / 2;
        let mut acc = 0u64;
        for (val, &c) in hist.iter().enumerate() {
            acc += c;
            if acc >= target {
                return val as u8;
            }
        }
        255
    };

    let cf = count as f64;
    let std_c = (0..3)
        .map(|c| {
            let mean = sum[c] / cf;
            (sum_sq[c] / cf - mean * mean).max(0.0).sqrt()
        })
        .sum::<f64>()
        / 3.0;

    Some((
        [hist_median(&hr), hist_median(&hg), hist_median(&hb)],
        std_c as f32,
    ))
}

/// Гармонический солвер Лапласа (Red-Black Gauss-Seidel SOR) для бесшовного инпейнтинга бабблов и градиентов.
pub fn inpaint_laplace_sor(
    rgb: &mut [u8],
    mask: &[u8],
    w: usize,
    h: usize,
    max_iters: usize,
    omega: f32,
) {
    let n = w.saturating_mul(h);
    if n == 0 || rgb.len() < n * 3 || mask.len() < n {
        return;
    }

    let mut red_pts: Vec<(usize, usize)> = Vec::new();
    let mut black_pts: Vec<(usize, usize)> = Vec::new();
    let mut bound_sum = [0.0f64; 3];
    let mut bound_cnt = 0u64;
    let mut global_sum = [0.0f64; 3];
    let mut global_cnt = 0u64;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if mask[idx] > 0 {
                if (x + y) % 2 == 0 {
                    red_pts.push((x, y));
                } else {
                    black_pts.push((x, y));
                }
            } else {
                let r = rgb[idx * 3] as f64;
                let g = rgb[idx * 3 + 1] as f64;
                let b = rgb[idx * 3 + 2] as f64;
                global_sum[0] += r;
                global_sum[1] += g;
                global_sum[2] += b;
                global_cnt += 1;

                let touches_mask = (x > 0 && mask[idx - 1] > 0)
                    || (x + 1 < w && mask[idx + 1] > 0)
                    || (y > 0 && mask[idx - w] > 0)
                    || (y + 1 < h && mask[idx + w] > 0);
                if touches_mask {
                    bound_sum[0] += r;
                    bound_sum[1] += g;
                    bound_sum[2] += b;
                    bound_cnt += 1;
                }
            }
        }
    }

    if red_pts.is_empty() && black_pts.is_empty() {
        return;
    }

    let init_rgb = if bound_cnt > 0 {
        [
            (bound_sum[0] / bound_cnt as f64) as f32,
            (bound_sum[1] / bound_cnt as f64) as f32,
            (bound_sum[2] / bound_cnt as f64) as f32,
        ]
    } else if global_cnt > 0 {
        [
            (global_sum[0] / global_cnt as f64) as f32,
            (global_sum[1] / global_cnt as f64) as f32,
            (global_sum[2] / global_cnt as f64) as f32,
        ]
    } else {
        [255.0f32, 255.0, 255.0]
    };

    let mut fr = vec![0.0f32; n];
    let mut fg = vec![0.0f32; n];
    let mut fb = vec![0.0f32; n];
    for i in 0..n {
        if mask[i] > 0 {
            fr[i] = init_rgb[0];
            fg[i] = init_rgb[1];
            fb[i] = init_rgb[2];
        } else {
            fr[i] = rgb[i * 3] as f32;
            fg[i] = rgb[i * 3 + 1] as f32;
            fb[i] = rgb[i * 3 + 2] as f32;
        }
    }

    let w_sor = omega.clamp(1.0, 1.92);
    let iters = max_iters.max(1);

    let step_color =
        |pts: &[(usize, usize)], fr: &mut [f32], fg: &mut [f32], fb: &mut [f32]| -> f32 {
            let mut max_delta = 0.0f32;
            for &(x, y) in pts {
                let idx = y * w + x;
                let l = if x > 0 { idx - 1 } else { idx };
                let r = if x + 1 < w { idx + 1 } else { idx };
                let u = if y > 0 { idx - w } else { idx };
                let d = if y + 1 < h { idx + w } else { idx };

                let avg_r = 0.25 * (fr[l] + fr[r] + fr[u] + fr[d]);
                let avg_g = 0.25 * (fg[l] + fg[r] + fg[u] + fg[d]);
                let avg_b = 0.25 * (fb[l] + fb[r] + fb[u] + fb[d]);

                let dr = w_sor * (avg_r - fr[idx]);
                let dg = w_sor * (avg_g - fg[idx]);
                let db = w_sor * (avg_b - fb[idx]);

                fr[idx] = (fr[idx] + dr).clamp(0.0, 255.0);
                fg[idx] = (fg[idx] + dg).clamp(0.0, 255.0);
                fb[idx] = (fb[idx] + db).clamp(0.0, 255.0);

                max_delta = max_delta.max(dr.abs()).max(dg.abs()).max(db.abs());
            }
            max_delta
        };

    for _ in 0..iters {
        let d_red = step_color(&red_pts, &mut fr, &mut fg, &mut fb);
        let d_blk = step_color(&black_pts, &mut fr, &mut fg, &mut fb);
        if d_red.max(d_blk) < 0.25 {
            break;
        }
    }

    for &(x, y) in red_pts.iter().chain(black_pts.iter()) {
        let idx = y * w + x;
        rgb[idx * 3] = fr[idx].round().clamp(0.0, 255.0) as u8;
        rgb[idx * 3 + 1] = fg[idx].round().clamp(0.0, 255.0) as u8;
        rgb[idx * 3 + 2] = fb[idx].round().clamp(0.0, 255.0) as u8;
    }
}
