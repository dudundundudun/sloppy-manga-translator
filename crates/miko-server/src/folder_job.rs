use crate::AppState;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use image::imageops::crop_imm;
use miko_cleaner::{compute_clean_state_hash, CleanOptions, RustCleanerEngine};
use miko_core::models::normalize_clean_bbox;
use miko_core::security::reject_unc_path;
use miko_core::{FolderJobState, TextBlock};
use miko_ocr::{OcrOptions, RustOcrEngine};
use miko_renderer::RenderOptions;
use miko_translator::{kana_to_romaji, merge_glossaries, TranslateBatchRequest};
use regex::RegexBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering as AtomicOrd;
use std::sync::Arc;
use std::time::{Instant, UNIX_EPOCH};

const MAX_JOB_LOGS: usize = 80;

#[inline]
fn is_supported_img(p: &Path) -> bool {
    p.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp" | "bmp"))
}

fn file_mtime_secs(p: &Path) -> i64 {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn unix_ts_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Естественное сравнение имён файлов (`2.png < 10.png`).
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let ba = a.as_bytes();
    let bb = b.as_bytes();
    let (mut ia, mut ib) = (0usize, 0usize);

    while ia < ba.len() && ib < bb.len() {
        let da = ba[ia].is_ascii_digit();
        let db = bb[ib].is_ascii_digit();
        if da && db {
            let za = ia;
            let zb = ib;
            while ia < ba.len() && ba[ia].is_ascii_digit() {
                ia += 1;
            }
            while ib < bb.len() && bb[ib].is_ascii_digit() {
                ib += 1;
            }
            let sa = std::str::from_utf8(&ba[za..ia]).unwrap_or("").trim_start_matches('0');
            let sb = std::str::from_utf8(&bb[zb..ib]).unwrap_or("").trim_start_matches('0');
            let cmp = sa.len().cmp(&sb.len()).then_with(|| sa.cmp(sb));
            if cmp != Ordering::Equal {
                return cmp;
            }
            // Если числовые значения равны (`02` vs `2`), короче = меньше по количеству ведущих нулей
            let raw_cmp = (ia - za).cmp(&(ib - zb));
            if raw_cmp != Ordering::Equal {
                return raw_cmp;
            }
        } else {
            let ca = ba[ia].to_ascii_lowercase();
            let cb = bb[ib].to_ascii_lowercase();
            if ca != cb {
                return ca.cmp(&cb);
            }
            ia += 1;
            ib += 1;
        }
    }
    ba.len().cmp(&bb.len())
}

pub fn collect_folder_images(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && is_supported_img(p)
                && !matches!(
                    p.file_name().and_then(|s| s.to_str()),
                    Some("mascot.png" | "mascot_fox.png" | "mascot_fox_sprites.png")
                )
        })
        .collect();
    files.sort_by(|a, b| {
        let na = a.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let nb = b.file_name().and_then(|s| s.to_str()).unwrap_or("");
        natural_cmp(na, nb)
    });
    files
}

/// Убирает суффикс вывода (`_RU`/`_EN`/`_ROMAJI`), если пользователь передал уже переведённую папку.
pub fn unwrap_source_folder(dir: &Path) -> PathBuf {
    let Some(name) = dir.file_name().and_then(|s| s.to_str()) else {
        return dir.to_path_buf();
    };
    for sfx in ["_ROMAJI", "_romaji", "_RU", "_ru", "_EN", "_en"] {
        if name.ends_with(sfx) && name.len() > sfx.len() {
            let stem = &name[..name.len() - sfx.len()];
            return dir.parent().unwrap_or(dir).join(stem);
        }
    }
    dir.to_path_buf()
}

pub fn output_dir_for_folder(source_dir: &Path, target_lang: &str) -> PathBuf {
    let base = unwrap_source_folder(source_dir);
    let name = base
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("manga");
    let lang_up = match target_lang.trim().to_ascii_lowercase().as_str() {
        "en" => "EN",
        "romaji" => "ROMAJI",
        _ => "RU",
    };
    base.parent()
        .unwrap_or(&base)
        .join(format!("{name}_{lang_up}"))
}

pub fn manifest_path(output_dir: &Path) -> PathBuf {
    output_dir.join(".translation_state.json")
}

pub fn load_manifest(output_dir: &Path) -> Value {
    let p = manifest_path(output_dir);
    let Ok(raw) = std::fs::read_to_string(&p) else {
        return json!({ "pages": {}, "phrase_cache": {} });
    };
    let mut val = serde_json::from_str::<Value>(&raw)
        .unwrap_or_else(|_| json!({ "pages": {}, "phrase_cache": {} }));
    if !val.get("pages").is_some_and(|v| v.is_object()) {
        val["pages"] = json!({});
    }
    if !val.get("phrase_cache").is_some_and(|v| v.is_object()) {
        val["phrase_cache"] = json!({});
    }
    val
}

pub fn save_manifest(output_dir: &Path, manifest: &Value) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let target = manifest_path(output_dir);
    let tmp = output_dir.join(".translation_state.tmp.json");
    let body = serde_json::to_vec_pretty(manifest)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&tmp, body)?;
    if target.exists() {
        let _ = std::fs::remove_file(&target);
    }
    std::fs::rename(&tmp, &target)?;
    Ok(())
}

pub fn push_folder_state(state: &AppState, mutator: impl FnOnce(&mut FolderJobState)) {
    {
        let mut job = state.folder_job.write().unwrap_or_else(|e| e.into_inner());
        mutator(&mut job);
        job.current = job.done;
        if job.source_dir.is_empty() && !job.folder_path.is_empty() {
            job.source_dir = job.folder_path.clone();
        }
        if job.logs.len() > MAX_JOB_LOGS {
            let drain_n = job.logs.len() - MAX_JOB_LOGS;
            job.logs.drain(0..drain_n);
        }
    }
    let next_seq = state.folder_tx.borrow().wrapping_add(1);
    let _ = state.folder_tx.send(next_seq);
}

fn time_hms_str() -> String {
    let s = unix_ts_now().rem_euclid(86_400);
    let h = s / 3600;
    let m = (s % 3600) / 60;
    let sec = s % 60;
    format!("{h:02}:{m:02}:{sec:02}")
}

fn push_live_event_obj(
    live: &mut Value,
    run_id: &str,
    stage: &str,
    stage_label: &str,
    model: &str,
    step_elapsed_sec: u64,
    expected_keys: usize,
    received_keys: usize,
    cloud_only: bool,
    event_text: Option<&str>,
) {
    let mut events = live
        .get("events")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if let Some(txt) = event_text {
        events.push(json!({
            "time_str": time_hms_str(),
            "text": txt,
        }));
        if events.len() > 60 {
            let d = events.len() - 60;
            events.drain(0..d);
        }
    }
    *live = json!({
        "run_id": run_id,
        "stage": stage,
        "stage_label": stage_label,
        "model": model,
        "batch_index": 1,
        "batches_total": 1,
        "step_elapsed_sec": step_elapsed_sec,
        "expected_keys": expected_keys,
        "received_keys": received_keys,
        "cloud_only": cloud_only,
        "events": events,
    });
}

#[derive(Debug, Clone)]
pub struct BatchFolderOptions {
    pub source_dir: PathBuf,
    pub output_dir: PathBuf,
    pub images: Vec<PathBuf>,
    pub target_lang: String,
    pub model_override: Option<String>,
    pub detect_free_text: bool,
    pub text_filter_mode: String,
    pub skip_existing: bool,
    pub force_retranslate: bool,
    pub uppercase: bool,
    pub min_font_size: i32,
    pub max_font_size: i32,
    pub mock_text: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchFolderReport {
    pub status: String,
    pub total: usize,
    pub processed: usize,
    pub skipped: usize,
    pub total_ms: u128,
    pub output_dir: String,
}

/// Полный пакетный проход по главе: `OCR -> Translator -> Cleaner -> Renderer -> .translation_state.json + SQLite`.
pub async fn process_chapter_folder_sync(
    state: Arc<AppState>,
    opts: BatchFolderOptions,
) -> Result<BatchFolderReport, String> {
    let t_batch = Instant::now();
    std::fs::create_dir_all(&opts.output_dir).map_err(|e| e.to_string())?;
    let _ = state.allowed_roots.register_root(&opts.source_dir);
    let _ = state.allowed_roots.register_root(&opts.output_dir);

    state.folder_stop_flag.store(false, AtomicOrd::SeqCst);
    state.folder_pause_flag.store(false, AtomicOrd::SeqCst);

    let total = opts.images.len();
    let cfg = state.config.read().unwrap_or_else(|e| e.into_inner()).clone();
    let run_id = format!("rs_{}", unix_ts_now() % 100_000);
    let mut manifest = load_manifest(&opts.output_dir);
    manifest["source_dir"] = json!(opts.source_dir.to_string_lossy());
    manifest["target_lang"] = json!(opts.target_lang);

    let init_model = opts
        .model_override
        .clone()
        .unwrap_or_else(|| cfg.selected_model.clone());
    let cloud_only = cfg.cloud_only_no_local_fallback;

    push_folder_state(&state, |j| {
        j.status = "running".to_string();
        j.total = total;
        j.done = 0;
        j.elapsed_sec = 0;
        j.current_file.clear();
        j.phase = "ocr".to_string();
        j.active_model = init_model.clone();
        j.folder_path = opts.source_dir.to_string_lossy().to_string();
        j.source_dir = opts.source_dir.to_string_lossy().to_string();
        j.output_dir = opts.output_dir.to_string_lossy().to_string();
        j.target_lang = opts.target_lang.clone();
        j.message = format!("Запуск перевода главы ({total} стр.)");
        j.logs.push(format!("Старт главы: {total} изображений"));
        j.live = json!({});
        push_live_event_obj(
            &mut j.live,
            &run_id,
            "ocr",
            &format!("Старт главы ({total} стр.)"),
            &init_model,
            0,
            0,
            0,
            cloud_only,
            Some(&format!("Старт Rust-конвейера ({total} стр.)")),
        );
    });

    let scope = format!(
        "folder:{}",
        opts.source_dir.to_string_lossy().trim().to_ascii_lowercase()
    );
    let global_g = state.db.load_glossary("global").unwrap_or(cfg.glossary.clone());
    let folder_g = state.db.load_glossary(&scope).unwrap_or_default();
    let merged_glossary = merge_glossaries(&global_g, &folder_g);

    let mut processed = 0usize;
    let mut skipped = 0usize;
    let mut failed_pages = 0usize;
    let mut stopped_early = false;
    let mut rolling_context: Vec<String> = Vec::new();

    for (idx, img_path) in opts.images.iter().enumerate() {
        while state.folder_pause_flag.load(AtomicOrd::SeqCst)
            && !state.folder_stop_flag.load(AtomicOrd::SeqCst)
        {
            push_folder_state(&state, |j| {
                j.phase = "paused".to_string();
                j.message = format!("Пауза на странице {}/{total}", idx + 1);
            });
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }
        if state.folder_stop_flag.load(AtomicOrd::SeqCst) {
            stopped_early = true;
            break;
        }

        let fname = img_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("page.png")
            .to_string();
        let dest_path = opts.output_dir.join(&fname);

        if opts.skip_existing && !opts.force_retranslate && dest_path.is_file() {
            skipped += 1;
            let el = t_batch.elapsed().as_secs();
            push_folder_state(&state, |j| {
                j.done = idx + 1;
                j.elapsed_sec = el;
                j.current_file = fname.clone();
                j.phase = "skipped".to_string();
                j.message = format!("Пропущена (уже готова): {fname}");
                push_live_event_obj(
                    &mut j.live,
                    &run_id,
                    "render",
                    &format!("Пропущена {}/{total}: {fname}", idx + 1),
                    &j.active_model.clone(),
                    0,
                    0,
                    0,
                    cloud_only,
                    None,
                );
            });
            continue;
        }

        let t_step = Instant::now();
        let el_ocr = t_batch.elapsed().as_secs();
        push_folder_state(&state, |j| {
            j.elapsed_sec = el_ocr;
            j.current_file = fname.clone();
            j.phase = "ocr".to_string();
            j.message = format!("OCR: {fname} ({}/{total})", idx + 1);
            push_live_event_obj(
                &mut j.live,
                &run_id,
                "ocr",
                &format!("OCR {}/{total}: {fname}", idx + 1),
                &j.active_model.clone(),
                0,
                0,
                0,
                cloud_only,
                None,
            );
        });

        let rgb = match image::open(img_path) {
            Ok(im) => im.to_rgb8(),
            Err(e) => {
                failed_pages += 1;
                let err_msg = format!("[ОШИБКА ЧТЕНИЯ] {fname}: {e} — страница пропущена");
                tracing::error!("{}", err_msg);
                let el_err = t_batch.elapsed().as_secs();
                push_folder_state(&state, |j| {
                    j.elapsed_sec = el_err;
                    j.current_file = fname.clone();
                    j.phase = "page_error".to_string();
                    j.message = err_msg.clone();
                    j.logs.push(err_msg.clone());
                    push_live_event_obj(
                        &mut j.live,
                        &run_id,
                        "ocr",
                        &err_msg,
                        &j.active_model.clone(),
                        0,
                        0,
                        0,
                        cloud_only,
                        Some(&err_msg),
                    );
                });
                continue;
            }
        };

        // 1. OCR
        let st_ocr = state.clone();
        let rgb_ocr = rgb.clone();
        let det_free = opts.detect_free_text;
        let tf_mode = opts.text_filter_mode.clone();
        let ocr_task_res = tokio::task::spawn_blocking(move || {
            let mut g = st_ocr.ocr_engine.lock().unwrap_or_else(|e| e.into_inner());
            if g.is_none() {
                let eng = RustOcrEngine::load(&st_ocr.base_dir, true).map_err(|e| e.to_string())?;
                *g = Some(eng);
            }
            g.as_mut()
                .expect("ocr init")
                .process_rgb_image(
                    &rgb_ocr,
                    &OcrOptions {
                        detect_free_text: det_free,
                        text_filter_mode: tf_mode,
                        ..Default::default()
                    },
                )
                .map_err(|e| e.to_string())
        })
        .await;

        let ocr_res = match ocr_task_res {
            Ok(Ok(r)) => r,
            Ok(Err(err)) => {
                failed_pages += 1;
                let err_msg = format!("[ОШИБКА OCR] {fname}: {err} — страница пропущена");
                tracing::error!("{}", err_msg);
                let el_err = t_batch.elapsed().as_secs();
                push_folder_state(&state, |j| {
                    j.elapsed_sec = el_err;
                    j.current_file = fname.clone();
                    j.phase = "page_error".to_string();
                    j.message = err_msg.clone();
                    j.logs.push(err_msg.clone());
                    push_live_event_obj(
                        &mut j.live,
                        &run_id,
                        "ocr",
                        &err_msg,
                        &j.active_model.clone(),
                        0,
                        0,
                        0,
                        cloud_only,
                        Some(&err_msg),
                    );
                });
                continue;
            }
            Err(join_err) => {
                failed_pages += 1;
                let err_msg = format!("[ПАНИКА OCR] {fname}: {join_err} — страница пропущена");
                tracing::error!("{}", err_msg);
                let el_err = t_batch.elapsed().as_secs();
                push_folder_state(&state, |j| {
                    j.elapsed_sec = el_err;
                    j.current_file = fname.clone();
                    j.phase = "page_error".to_string();
                    j.message = err_msg.clone();
                    j.logs.push(err_msg.clone());
                    push_live_event_obj(
                        &mut j.live,
                        &run_id,
                        "ocr",
                        &err_msg,
                        &j.active_model.clone(),
                        0,
                        0,
                        0,
                        cloud_only,
                        Some(&err_msg),
                    );
                });
                continue;
            }
        };

        let mut blocks = ocr_res.blocks;
        let mut active_model = opts
            .model_override
            .clone()
            .unwrap_or_else(|| cfg.selected_model.clone());
        let mut detected_lang = "ja".to_string();

        // 2. Перевод батча (или --mock-text для оффлайн E2E бенчмарка)
        if !blocks.is_empty() {
            let b_cnt = blocks.len();
            let el_tr = t_batch.elapsed().as_secs();
            let st_tr = t_step.elapsed().as_secs();
            push_folder_state(&state, |j| {
                j.elapsed_sec = el_tr;
                j.phase = "llm".to_string();
                j.message = format!("Перевод ({b_cnt} бл.): {fname}");
                push_live_event_obj(
                    &mut j.live,
                    &run_id,
                    "llm",
                    &format!("Перевод ({b_cnt} фраз): {fname}"),
                    &active_model,
                    st_tr,
                    b_cnt,
                    0,
                    cloud_only,
                    None,
                );
            });
            if let Some(ref mock) = opts.mock_text {
                for b in &mut blocks {
                    b.translated_text = mock.clone();
                    if b.romaji_text.trim().is_empty() {
                        b.romaji_text = kana_to_romaji(&b.original_text);
                    }
                }
                active_model = "mock-rust".to_string();
            } else {
                let tr_res = match state
                    .translator
                    .translate_blocks(
                        TranslateBatchRequest {
                            blocks,
                            target_lang: opts.target_lang.clone(),
                            model: active_model.clone(),
                            fallback_model: cfg.local_fallback_model_multilang.clone(),
                            cloud_only_no_local_fallback: cfg.cloud_only_no_local_fallback,
                            cloud_url: cfg.unsloth_url.clone(),
                            cloud_api_key: cfg.unsloth_api_key.clone(),
                            glossary: merged_glossary.clone(),
                            context_lines: rolling_context.clone(),
                            anti_censorship: cfg.anti_censorship_prompt,
                            use_cache: !opts.force_retranslate,
                        },
                        Some(&state.db),
                    )
                    .await
                {
                    Ok(res) => res,
                    Err(err) => {
                        failed_pages += 1;
                        let warn_msg = format!(
                            "[ОШИБКА LLM] {fname}: не удалось перевести бабблы ({err}) — страница НЕ помечена готовой"
                        );
                        tracing::warn!("{}", warn_msg);
                        let el_err = t_batch.elapsed().as_secs();
                        push_folder_state(&state, |j| {
                            j.elapsed_sec = el_err;
                            j.current_file = fname.clone();
                            j.phase = "llm_error".to_string();
                            j.message = warn_msg.clone();
                            j.logs.push(warn_msg.clone());
                            push_live_event_obj(
                                &mut j.live,
                                &run_id,
                                "llm",
                                &warn_msg,
                                &active_model,
                                0,
                                b_cnt,
                                0,
                                cloud_only,
                                Some(&warn_msg),
                            );
                        });
                        continue;
                    }
                };
                blocks = tr_res.blocks;
                active_model = tr_res.active_model;
                detected_lang = tr_res.detected_lang;
                for b in &blocks {
                    let o = b.original_text.trim();
                    let t = b.translated_text.trim();
                    if !b.is_deleted && !o.is_empty() && !t.is_empty() {
                        rolling_context.push(format!("{o} => {t}"));
                    }
                }
                if rolling_context.len() > 10 {
                    let drain_n = rolling_context.len() - 10;
                    rolling_context.drain(0..drain_n);
                }
            }
        }

        // 3. Очистка + Рендер + Запись на диск
        let el_cl = t_batch.elapsed().as_secs();
        let st_cl = t_step.elapsed().as_secs();
        let tr_cnt = blocks.len();
        push_folder_state(&state, |j| {
            j.elapsed_sec = el_cl;
            j.phase = "clean".to_string();
            j.active_model = active_model.clone();
            j.message = format!("Очистка и вёрстка: {fname}");
            push_live_event_obj(
                &mut j.live,
                &run_id,
                "clean",
                &format!("Очистка+Рендер: {fname}"),
                &active_model,
                st_cl,
                tr_cnt,
                tr_cnt,
                cloud_only,
                None,
            );
        });

        let st_cr = state.clone();
        let src_p_clone = img_path.clone();
        let dest_p_clone = dest_path.clone();
        let up = opts.uppercase;
        let min_fs = opts.min_font_size;
        let max_fs = opts.max_font_size;
        let overflow_mode = cfg.overflow_mode.clone();
        let mtime = file_mtime_secs(img_path);

        let clean_render_task = tokio::task::spawn_blocking(move || -> Result<(Vec<TextBlock>, u128), String> {
            let t_p = Instant::now();
            if blocks.is_empty() {
                std::fs::copy(&src_p_clone, &dest_p_clone)
                    .map_err(|e| format!("Ошибка копирования пустой страницы: {e}"))?;
                return Ok((blocks, t_p.elapsed().as_millis()));
            }
            let mut cguard = st_cr.cleaner_engine.lock().unwrap_or_else(|e| e.into_inner());
            if cguard.is_none() {
                *cguard = Some(RustCleanerEngine::load(&st_cr.base_dir, true));
            }
            let src_str = src_p_clone.to_string_lossy().to_string();
            let clean_res = cguard
                .as_mut()
                .expect("cleaner init")
                .clean_image(
                    &rgb,
                    &blocks,
                    &CleanOptions::default(),
                    Some(&src_str),
                    mtime,
                )
                .map_err(|e| e.to_string())?;

            let render_res = st_cr
                .renderer
                .render_page(
                    &clean_res.cleaned,
                    &mut blocks,
                    &RenderOptions {
                        min_font_size: min_fs,
                        max_font_size: max_fs,
                        uppercase: up,
                        overflow_mode,
                        default_font_path: None,
                    },
                )
                .map_err(|e| e.to_string())?;

            render_res
                .rendered
                .save(&dest_p_clone)
                .map_err(|e| format!("Ошибка сохранения {dest_p_clone:?}: {e}"))?;

            let hash = compute_clean_state_hash(
                &blocks,
                &src_str,
                mtime,
                9,
                "rust_harmonic_v3",
                "balanced",
            );
            let _ = st_cr.db.save_page_state(&src_str, &hash, &blocks);
            Ok((blocks, t_p.elapsed().as_millis()))
        })
        .await;

        let (blocks_after, page_ms) = match clean_render_task {
            Ok(Ok(pair)) => pair,
            Ok(Err(err)) => {
                failed_pages += 1;
                let err_msg = format!("[ОШИБКА РЕНДЕРА] {fname}: {err} — страница пропущена");
                tracing::error!("{}", err_msg);
                let el_err = t_batch.elapsed().as_secs();
                push_folder_state(&state, |j| {
                    j.elapsed_sec = el_err;
                    j.current_file = fname.clone();
                    j.phase = "page_error".to_string();
                    j.message = err_msg.clone();
                    j.logs.push(err_msg.clone());
                    push_live_event_obj(
                        &mut j.live,
                        &run_id,
                        "clean",
                        &err_msg,
                        &active_model,
                        0,
                        tr_cnt,
                        0,
                        cloud_only,
                        Some(&err_msg),
                    );
                });
                continue;
            }
            Err(join_err) => {
                failed_pages += 1;
                let err_msg = format!("[ПАНИКА КЛИНЕРА/РЕНДЕРА] {fname}: {join_err} — страница пропущена");
                tracing::error!("{}", err_msg);
                let el_err = t_batch.elapsed().as_secs();
                push_folder_state(&state, |j| {
                    j.elapsed_sec = el_err;
                    j.current_file = fname.clone();
                    j.phase = "page_error".to_string();
                    j.message = err_msg.clone();
                    j.logs.push(err_msg.clone());
                    push_live_event_obj(
                        &mut j.live,
                        &run_id,
                        "clean",
                        &err_msg,
                        &active_model,
                        0,
                        tr_cnt,
                        0,
                        cloud_only,
                        Some(&err_msg),
                    );
                });
                continue;
            }
        };

        // Обновляем phrase_cache + pages в manifest
        if let Some(pc) = manifest.get_mut("phrase_cache").and_then(|v| v.as_object_mut()) {
            let lang_low = opts.target_lang.to_ascii_lowercase();
            for b in &blocks_after {
                let orig = b.original_text.trim();
                let tr = b.translated_text.trim();
                let rom = b.romaji_text.trim();
                if !orig.is_empty() && !tr.is_empty() {
                    pc.insert(format!("{lang_low}||{orig}"), json!(tr));
                }
                if !orig.is_empty() && !rom.is_empty() {
                    pc.insert(format!("romaji||{orig}"), json!(rom));
                }
            }
        }
        let free_cnt = blocks_after.iter().filter(|b| b.is_free_text).count();
        manifest["pages"][&fname] = json!({
            "blocks_count": blocks_after.len(),
            "free_text_count": free_cnt,
            "detected_lang": detected_lang,
            "model": active_model,
            "uppercase": opts.uppercase,
            "min_font_size": opts.min_font_size,
            "max_font_size": opts.max_font_size,
            "updated_at": unix_ts_now(),
            "blocks": blocks_after,
        });
        save_manifest(&opts.output_dir, &manifest).map_err(|e| e.to_string())?;

        processed += 1;
        let el_done = t_batch.elapsed().as_secs();
        let ev_msg = format!(
            "[OK] {fname}: {} блоков, рендер+клин={page_ms} ms",
            blocks_after.len()
        );
        push_folder_state(&state, |j| {
            j.done = idx + 1;
            j.elapsed_sec = el_done;
            j.current_file = fname.clone();
            j.phase = "render".to_string();
            j.message = format!("Готово {}/{total}: {fname} ({page_ms} ms)", idx + 1);
            j.logs.push(ev_msg.clone());
            push_live_event_obj(
                &mut j.live,
                &run_id,
                "render",
                &format!("Готово {}/{total}: {fname}", idx + 1),
                &active_model,
                0,
                blocks_after.len(),
                blocks_after.len(),
                cloud_only,
                Some(&ev_msg),
            );
        });
    }

    let final_status = if stopped_early { "stopped" } else { "completed" }.to_string();
    let total_ms = t_batch.elapsed().as_millis();
    let total_el = t_batch.elapsed().as_secs();
    push_folder_state(&state, |j| {
        j.status = final_status.clone();
        j.elapsed_sec = total_el;
        j.phase = if stopped_early { "stopped" } else { "done" }.to_string();
        j.message = if failed_pages > 0 {
            format!(
                "{}: готово {processed}, ошибок страниц {failed_pages}, пропущено {skipped} за {total_ms} ms",
                if stopped_early { "Остановлено" } else { "Глава завершена с пропусками" }
            )
        } else {
            format!(
                "{}: обработано {processed}, пропущено {skipped} за {total_ms} ms",
                if stopped_early { "Остановлено" } else { "Глава завершена" }
            )
        };
        let msg_clone = j.message.clone();
        j.logs.push(msg_clone.clone());
        let stg = if stopped_early { "stopped" } else { "done" };
        push_live_event_obj(
            &mut j.live,
            &run_id,
            stg,
            &msg_clone,
            &j.active_model.clone(),
            0,
            0,
            0,
            cloud_only,
            Some(&msg_clone),
        );
    });

    Ok(BatchFolderReport {
        status: final_status,
        total,
        processed,
        skipped,
        total_ms,
        output_dir: opts.output_dir.to_string_lossy().to_string(),
    })
}

#[derive(Debug, Deserialize)]
pub struct TranslateFolderPayload {
    pub folder_path: String,
    pub target_lang: Option<String>,
    pub model: Option<String>,
    pub ollama_model: Option<String>,
    pub selected_files: Option<Vec<String>>,
    pub skip_existing: Option<bool>,
    pub force_retranslate: Option<bool>,
    pub detect_free_text: Option<bool>,
    pub text_filter_mode: Option<String>,
    pub uppercase: Option<bool>,
    pub min_font_size: Option<u32>,
    pub max_font_size: Option<u32>,
}

pub async fn translate_folder(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<TranslateFolderPayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.folder_path), "folder_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let already_running = state
        .folder_job
        .read()
        .is_ok_and(|j| matches!(j.status.as_str(), "running" | "starting" | "stopping"));
    if already_running {
        return (
            StatusCode::CONFLICT,
            Json(json!({"detail": "Перевод папки уже выполняется"})),
        )
            .into_response();
    }

    let source_dir = unwrap_source_folder(Path::new(payload.folder_path.trim()));
    if !state.allowed_roots.is_within_allowed_roots(&source_dir) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Папка вне разрешённых директорий"})),
        )
            .into_response();
    }
    if !source_dir.is_dir() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Папка главы не найдена"})),
        )
            .into_response();
    }

    let mut images = collect_folder_images(&source_dir);
    if let Some(ref sel) = payload.selected_files {
        let wanted: std::collections::HashSet<String> =
            sel.iter().map(|s| s.trim().to_string()).collect();
        images.retain(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| wanted.contains(n))
        });
    }
    if images.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "В папке не найдено подходящих изображений"})),
        )
            .into_response();
    }

    let t_lang = payload
        .target_lang
        .unwrap_or_else(|| "ru".to_string())
        .trim()
        .to_ascii_lowercase();
    let output_dir = output_dir_for_folder(&source_dir, &t_lang);
    let _ = state.allowed_roots.register_root(&source_dir);
    let _ = state.allowed_roots.register_root(&output_dir);

    let skip_existing = payload.skip_existing.unwrap_or(true);
    let force_retranslate = payload.force_retranslate.unwrap_or(false);
    let already_done = if skip_existing && !force_retranslate {
        images
            .iter()
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| output_dir.join(n).is_file())
            })
            .count()
    } else {
        0
    };

    let cfg = state.config.read().unwrap_or_else(|e| e.into_inner()).clone();
    let total_images = images.len();
    let batch_opts = BatchFolderOptions {
        source_dir: source_dir.clone(),
        output_dir: output_dir.clone(),
        images,
        target_lang: t_lang.clone(),
        model_override: payload.ollama_model.or(payload.model),
        detect_free_text: payload.detect_free_text.unwrap_or(cfg.detect_free_text),
        text_filter_mode: payload.text_filter_mode.unwrap_or(cfg.text_filter_mode),
        skip_existing,
        force_retranslate,
        uppercase: payload.uppercase.unwrap_or(cfg.uppercase),
        min_font_size: payload.min_font_size.unwrap_or(cfg.min_font_size) as i32,
        max_font_size: payload.max_font_size.unwrap_or(cfg.max_font_size) as i32,
        mock_text: None,
    };

    let st_bg = state.clone();
    tokio::spawn(async move {
        if let Err(err) = process_chapter_folder_sync(st_bg.clone(), batch_opts).await {
            tracing::error!("Ошибка фонового перевода папки: {}", err);
            push_folder_state(&st_bg, |j| {
                j.status = "error".to_string();
                j.phase = "error".to_string();
                j.message = err.clone();
                j.logs.push(format!("[ERR] {err}"));
            });
        }
    });

    Json(json!({
        "status": "started",
        "source_dir": source_dir.to_string_lossy(),
        "output_dir": output_dir.to_string_lossy(),
        "total_images": total_images,
        "already_translated": already_done,
        "remaining_images": total_images.saturating_sub(already_done),
        "target_lang": t_lang,
    }))
    .into_response()
}

pub async fn folder_stop(State(state): State<Arc<AppState>>) -> Json<Value> {
    let active = state
        .folder_job
        .read()
        .is_ok_and(|j| matches!(j.status.as_str(), "running" | "stopping"));
    if !active {
        return Json(json!({
            "status": "idle",
            "message": "Пакетный перевод сейчас не выполняется."
        }));
    }
    state.folder_stop_flag.store(true, AtomicOrd::SeqCst);
    state.folder_pause_flag.store(false, AtomicOrd::SeqCst);
    push_folder_state(&state, |j| {
        j.status = "stopping".to_string();
        j.phase = "stopping".to_string();
        j.message = "Остановка и сохранение прогресса...".to_string();
    });
    Json(json!({
        "status": "stopping",
        "message": "Запрос на остановку отправлен."
    }))
}

#[derive(Debug, Deserialize)]
pub struct FolderControlPayload {
    pub action: String,
    pub model: Option<String>,
    pub cloud_only: Option<bool>,
}

pub async fn folder_control(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<FolderControlPayload>,
) -> Response {
    let act = payload.action.trim().to_ascii_lowercase();
    match act.as_str() {
        "pause" => {
            state.folder_pause_flag.store(true, AtomicOrd::SeqCst);
            push_folder_state(&state, |j| {
                j.phase = "paused".to_string();
                j.message = "Пауза после текущей страницы".to_string();
            });
            Json(json!({"status": "ok", "action": act})).into_response()
        }
        "resume" => {
            state.folder_pause_flag.store(false, AtomicOrd::SeqCst);
            push_folder_state(&state, |j| {
                j.phase = "running".to_string();
                j.message = "Возобновление пакетного перевода".to_string();
            });
            Json(json!({"status": "ok", "action": act})).into_response()
        }
        "toggle_cloud_only" => {
            let new_val = {
                let mut cfg = state.config.write().unwrap_or_else(|e| e.into_inner());
                let v = payload
                    .cloud_only
                    .unwrap_or(!cfg.cloud_only_no_local_fallback);
                cfg.cloud_only_no_local_fallback = v;
                state.persist_config_if_enabled(&cfg);
                v
            };
            push_folder_state(&state, |j| {
                j.logs.push(format!(
                    "Только облако -> {}",
                    if new_val { "ВКЛ" } else { "ВЫКЛ" }
                ));
            });
            Json(json!({"status": "ok", "action": act, "cloud_only": new_val})).into_response()
        }
        "switch_model" => {
            let Some(m) = payload.model.as_deref().map(str::trim).filter(|s| !s.is_empty()) else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"detail": "Не указана модель для переключения"})),
                )
                    .into_response();
            };
            if let Ok(mut cfg) = state.config.write() {
                cfg.selected_model = m.to_string();
                state.persist_config_if_enabled(&cfg);
            }
            push_folder_state(&state, |j| {
                j.active_model = m.to_string();
                j.logs.push(format!("Смена модели -> {m}"));
            });
            Json(json!({"status": "ok", "action": act, "active_model": m})).into_response()
        }
        _ => (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": format!("Неизвестное действие: {act}")})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct PageBlocksPayload {
    pub source_path: String,
    pub source_dir: Option<String>,
    pub target_lang: Option<String>,
    pub force_rescan: Option<bool>,
    pub detect_free_text: Option<bool>,
    pub text_filter_mode: Option<String>,
}

pub async fn page_blocks(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<PageBlocksPayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.source_path), "source_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    if let Some(ref sd) = payload.source_dir {
        if let Err(err) = reject_unc_path(Some(sd), "source_dir") {
            return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()})))
                .into_response();
        }
    }
    let src_p = PathBuf::from(payload.source_path.trim());
    if !src_p.is_file() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Файл страницы не найден"})),
        )
            .into_response();
    }
    if !state.allowed_roots.is_within_allowed_roots(&src_p) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Путь вне разрешённых директорий"})),
        )
            .into_response();
    }

    let source_dir = payload
        .source_dir
        .as_deref()
        .map(|s| unwrap_source_folder(Path::new(s.trim())))
        .unwrap_or_else(|| unwrap_source_folder(src_p.parent().unwrap_or(Path::new("."))));
    let t_lang = payload
        .target_lang
        .unwrap_or_else(|| "ru".to_string())
        .trim()
        .to_ascii_lowercase();
    let output_dir = output_dir_for_folder(&source_dir, &t_lang);
    let fname = src_p
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("page.png")
        .to_string();

    let mut manifest = load_manifest(&output_dir);
    let force_rescan = payload.force_rescan.unwrap_or(false);

    if !force_rescan {
        if let Some(raw_blocks) = manifest["pages"][&fname]["blocks"].as_array() {
            if let Ok(mut blocks) = serde_json::from_value::<Vec<TextBlock>>(Value::Array(raw_blocks.clone())) {
                for b in &mut blocks {
                    normalize_clean_bbox(b);
                    if b.romaji_text.trim().is_empty() && !b.original_text.trim().is_empty() {
                        b.romaji_text = kana_to_romaji(b.original_text.trim());
                    }
                }
                return Json(json!({
                    "status": "ok",
                    "cached": true,
                    "source_path": src_p.to_string_lossy(),
                    "blocks_count": blocks.len(),
                    "detected_lang": manifest["pages"][&fname]["detected_lang"].as_str().unwrap_or("ja"),
                    "blocks": blocks,
                }))
                .into_response();
            }
        }
        let src_str = src_p.to_string_lossy().to_string();
        if let Ok(Some((_, mut blocks))) = state.db.load_page_state(&src_str) {
            for b in &mut blocks {
                normalize_clean_bbox(b);
                if b.romaji_text.trim().is_empty() && !b.original_text.trim().is_empty() {
                    b.romaji_text = kana_to_romaji(b.original_text.trim());
                }
            }
            return Json(json!({
                "status": "ok",
                "cached": true,
                "source_path": src_str,
                "blocks_count": blocks.len(),
                "detected_lang": "ja",
                "blocks": blocks,
            }))
            .into_response();
        }
    }

    // OCR скан исходника
    let st_ocr = state.clone();
    let src_clone = src_p.clone();
    let cfg = state.config.read().unwrap_or_else(|e| e.into_inner()).clone();
    let det_free = payload.detect_free_text.unwrap_or(cfg.detect_free_text);
    let tf_mode = payload.text_filter_mode.unwrap_or(cfg.text_filter_mode);

    let ocr_res = tokio::task::spawn_blocking(move || -> Result<Vec<TextBlock>, String> {
        let rgb = image::open(&src_clone)
            .map_err(|e| format!("Ошибка чтения изображения: {e}"))?
            .to_rgb8();
        let mut g = st_ocr.ocr_engine.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_none() {
            let eng = RustOcrEngine::load(&st_ocr.base_dir, true).map_err(|e| e.to_string())?;
            *g = Some(eng);
        }
        let res = g
            .as_mut()
            .expect("ocr init")
            .process_rgb_image(
                &rgb,
                &OcrOptions {
                    detect_free_text: det_free,
                    text_filter_mode: tf_mode,
                    ..Default::default()
                },
            )
            .map_err(|e| e.to_string())?;
        Ok(res.blocks)
    })
    .await;

    match ocr_res {
        Ok(Ok(mut blocks)) => {
            for b in &mut blocks {
                normalize_clean_bbox(b);
                if b.romaji_text.trim().is_empty() && !b.original_text.trim().is_empty() {
                    b.romaji_text = kana_to_romaji(b.original_text.trim());
                }
            }
            manifest["pages"][&fname] = json!({
                "blocks_count": blocks.len(),
                "free_text_count": blocks.iter().filter(|b| b.is_free_text).count(),
                "detected_lang": "ja",
                "updated_at": unix_ts_now(),
                "blocks": blocks,
            });
            let _ = save_manifest(&output_dir, &manifest);
            Json(json!({
                "status": "ok",
                "cached": false,
                "source_path": src_p.to_string_lossy(),
                "blocks_count": blocks.len(),
                "detected_lang": "ja",
                "blocks": blocks,
            }))
            .into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e.to_string()})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct PageRomajiPayload {
    pub blocks: Vec<TextBlock>,
}

pub async fn page_romaji(Json(payload): Json<PageRomajiPayload>) -> Json<Value> {
    let mut blocks = payload.blocks;
    for b in &mut blocks {
        if !b.original_text.trim().is_empty() {
            b.romaji_text = kana_to_romaji(b.original_text.trim());
        }
    }
    Json(json!({
        "status": "ok",
        "blocks": blocks,
    }))
}

#[derive(Debug, Deserialize)]
pub struct TranslateManualRegionPayload {
    pub source_path: String,
    pub bbox: [i32; 4],
    pub target_lang: Option<String>,
    pub text: Option<String>,
    pub model: Option<String>,
}

pub async fn translate_manual_region(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<TranslateManualRegionPayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.source_path), "source_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let src_p = PathBuf::from(payload.source_path.trim());
    if !src_p.is_file() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Файл изображения не найден"})),
        )
            .into_response();
    }
    if !state.allowed_roots.is_within_allowed_roots(&src_p) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Путь вне разрешённых директорий"})),
        )
            .into_response();
    }

    let supplied_text = payload
        .text
        .as_deref()
        .map(str::trim)
        .unwrap_or("")
        .to_string();

    let orig_text = if !supplied_text.is_empty() {
        supplied_text
    } else {
        let st_ocr = state.clone();
        let src_c = src_p.clone();
        let bb = payload.bbox;
        let ocr_out = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let rgb = image::open(&src_c).map_err(|e| e.to_string())?.to_rgb8();
            let (iw, ih) = (rgb.width() as i32, rgb.height() as i32);
            let x1 = bb[0].clamp(0, iw - 1) as u32;
            let y1 = bb[1].clamp(0, ih - 1) as u32;
            let x2 = bb[2].clamp(x1 as i32 + 1, iw) as u32;
            let y2 = bb[3].clamp(y1 as i32 + 1, ih) as u32;
            let crop = crop_imm(&rgb, x1, y1, x2 - x1, y2 - y1).to_image();
            let mut g = st_ocr.ocr_engine.lock().unwrap_or_else(|e| e.into_inner());
            if g.is_none() {
                let eng = RustOcrEngine::load(&st_ocr.base_dir, true).map_err(|e| e.to_string())?;
                *g = Some(eng);
            }
            let res = g
                .as_mut()
                .expect("ocr init")
                .process_rgb_image(&crop, &OcrOptions::default())
                .map_err(|e| e.to_string())?;
            let joined = res
                .blocks
                .into_iter()
                .map(|b| b.original_text)
                .filter(|s| !s.trim().is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            Ok(joined)
        })
        .await;
        match ocr_out {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"detail": e})),
                )
                    .into_response()
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"detail": e.to_string()})),
                )
                    .into_response()
            }
        }
    };

    let cfg = state.config.read().unwrap_or_else(|e| e.into_inner()).clone();
    let t_lang = payload.target_lang.unwrap_or_else(|| "ru".to_string());
    let model = payload.model.unwrap_or(cfg.selected_model);
    let glossary = state.db.load_glossary("global").unwrap_or(cfg.glossary);

    let translated_text = if orig_text.trim().is_empty() {
        String::new()
    } else {
        state
            .translator
            .translate_single_text(
                &orig_text,
                &t_lang,
                &model,
                &cfg.unsloth_url,
                &cfg.unsloth_api_key,
                &glossary,
                cfg.anti_censorship_prompt,
                Some(&state.db),
            )
            .await
            .unwrap_or_else(|_| orig_text.clone())
    };

    let mut blk = TextBlock {
        id: 1,
        bbox: payload.bbox,
        clean_bbox: Some(payload.bbox),
        original_text: orig_text.clone(),
        romaji_text: kana_to_romaji(&orig_text),
        translated_text,
        is_free_text: true,
        ..Default::default()
    };
    normalize_clean_bbox(&mut blk);

    Json(json!({
        "status": "ok",
        "block": blk,
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct FolderFindReplacePayload {
    pub folder_path: String,
    pub target_lang: Option<String>,
    pub find_text: String,
    pub replace_text: Option<String>,
    pub match_case: Option<bool>,
    pub search_in: Option<String>,
    pub dry_run: Option<bool>,
}

pub async fn folder_find_replace(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<FolderFindReplacePayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.folder_path), "folder_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let find_q = payload.find_text.trim();
    if find_q.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Поле 'find_text' не может быть пустым"})),
        )
            .into_response();
    }
    let source_dir = unwrap_source_folder(Path::new(payload.folder_path.trim()));
    if !state.allowed_roots.is_within_allowed_roots(&source_dir) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Папка вне разрешённых директорий"})),
        )
            .into_response();
    }

    let t_lang = payload
        .target_lang
        .unwrap_or_else(|| "ru".to_string())
        .trim()
        .to_ascii_lowercase();
    let output_dir = output_dir_for_folder(&source_dir, &t_lang);
    let mut manifest = load_manifest(&output_dir);
    let dry_run = payload.dry_run.unwrap_or(true);
    let match_case = payload.match_case.unwrap_or(false);
    let search_orig = payload
        .search_in
        .as_deref()
        .is_some_and(|s| s.eq_ignore_ascii_case("original"));
    let repl = payload.replace_text.as_deref().unwrap_or("");

    let Ok(re) = RegexBuilder::new(&regex::escape(find_q))
        .case_insensitive(!match_case)
        .build()
    else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Некорректный шаблон поиска"})),
        )
            .into_response();
    };

    let t0 = Instant::now();
    let mut matches = Vec::new();
    let mut affected_pages: Vec<String> = Vec::new();

    if let Some(pages_obj) = manifest.get_mut("pages").and_then(|v| v.as_object_mut()) {
        let mut page_keys: Vec<String> = pages_obj.keys().cloned().collect();
        page_keys.sort_by(|a, b| natural_cmp(a, b));

        for key in page_keys {
            let Some(page_val) = pages_obj.get_mut(&key) else {
                continue;
            };
            let Some(blocks_val) = page_val.get("blocks").cloned() else {
                continue;
            };
            let Ok(mut blocks) = serde_json::from_value::<Vec<TextBlock>>(blocks_val) else {
                continue;
            };
            let mut page_changed = false;
            for blk in &mut blocks {
                let target_field = if search_orig {
                    &blk.original_text
                } else {
                    &blk.translated_text
                };
                if re.is_match(target_field) {
                    let preview_after = re.replace_all(target_field, repl).to_string();
                    matches.push(json!({
                        "page": key,
                        "block_id": blk.id,
                        "before": target_field,
                        "after": preview_after,
                    }));
                    if !dry_run {
                        if search_orig {
                            blk.original_text = preview_after;
                        } else {
                            blk.translated_text = preview_after;
                        }
                        page_changed = true;
                    }
                }
            }
            if !dry_run && page_changed {
                page_val["blocks"] = json!(blocks);
                page_val["updated_at"] = json!(unix_ts_now());
                affected_pages.push(key);
            }
        }
    }

    let mut rerendered_count = 0usize;
    if !dry_run && !affected_pages.is_empty() {
        if let Err(e) = save_manifest(&output_dir, &manifest) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"detail": e.to_string()})),
            )
                .into_response();
        }
        let cfg = state.config.read().unwrap_or_else(|e| e.into_inner()).clone();
        for rel_key in &affected_pages {
            let src_p = source_dir.join(rel_key);
            if !src_p.is_file() {
                continue;
            }
            let Ok(mut blocks) =
                serde_json::from_value::<Vec<TextBlock>>(manifest["pages"][rel_key]["blocks"].clone())
            else {
                continue;
            };
            let dest_p = output_dir.join(rel_key);
            let st_cr = state.clone();
            let src_c = src_p.clone();
            let dest_c = dest_p.clone();
            let up = cfg.uppercase;
            let min_fs = cfg.min_font_size as i32;
            let max_fs = cfg.max_font_size as i32;
            let ov = cfg.overflow_mode.clone();

            let rr = tokio::task::spawn_blocking(move || -> Result<(), String> {
                let rgb = image::open(&src_c).map_err(|e| e.to_string())?.to_rgb8();
                let src_str = src_c.to_string_lossy().to_string();
                let mtime = file_mtime_secs(&src_c);
                let mut cg = st_cr.cleaner_engine.lock().unwrap_or_else(|e| e.into_inner());
                if cg.is_none() {
                    *cg = Some(RustCleanerEngine::load(&st_cr.base_dir, true));
                }
                let clean_res = cg
                    .as_mut()
                    .expect("cleaner init")
                    .clean_image(&rgb, &blocks, &CleanOptions::default(), Some(&src_str), mtime)
                    .map_err(|e| e.to_string())?;
                let render_res = st_cr
                    .renderer
                    .render_page(
                        &clean_res.cleaned,
                        &mut blocks,
                        &RenderOptions {
                            min_font_size: min_fs,
                            max_font_size: max_fs,
                            uppercase: up,
                            overflow_mode: ov,
                            default_font_path: None,
                        },
                    )
                    .map_err(|e| e.to_string())?;
                if let Some(par) = dest_c.parent() {
                    let _ = std::fs::create_dir_all(par);
                }
                render_res.rendered.save(&dest_c).map_err(|e| e.to_string())?;
                let hash = compute_clean_state_hash(
                    &blocks,
                    &src_str,
                    mtime,
                    9,
                    "rust_harmonic_v3",
                    "balanced",
                );
                let _ = st_cr.db.save_page_state(&src_str, &hash, &blocks);
                Ok(())
            })
            .await;

            if matches!(rr, Ok(Ok(()))) {
                rerendered_count += 1;
            }
        }
    }

    Json(json!({
        "status": "ok",
        "dry_run": dry_run,
        "matches_count": matches.len(),
        "affected_pages_count": affected_pages.len(),
        "rerendered_pages_count": rerendered_count,
        "processing_time_sec": (t0.elapsed().as_secs_f64() * 1000.0).round() / 1000.0,
        "matches": matches,
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct FolderQuery {
    pub folder_path: String,
    pub target_lang: Option<String>,
}

pub async fn folder_report(
    State(state): State<Arc<AppState>>,
    Query(q): Query<FolderQuery>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&q.folder_path), "folder_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let source_dir = unwrap_source_folder(Path::new(q.folder_path.trim()));
    if !state.allowed_roots.is_within_allowed_roots(&source_dir) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Папка вне разрешённых директорий"})),
        )
            .into_response();
    }
    let t_lang = q.target_lang.unwrap_or_else(|| "ru".to_string());
    let output_dir = output_dir_for_folder(&source_dir, &t_lang);
    let manifest = load_manifest(&output_dir);

    let mut total_blocks = 0usize;
    let mut free_blocks = 0usize;
    let mut pages_summary = Vec::new();
    if let Some(pages) = manifest["pages"].as_object() {
        let mut keys: Vec<&String> = pages.keys().collect();
        keys.sort_by(|a, b| natural_cmp(a, b));
        for k in keys {
            let p = &pages[k];
            let bc = p["blocks_count"].as_u64().unwrap_or(0) as usize;
            let fc = p["free_text_count"].as_u64().unwrap_or(0) as usize;
            total_blocks += bc;
            free_blocks += fc;
            pages_summary.push(json!({
                "page": k,
                "blocks_count": bc,
                "free_text_count": fc,
                "model": p["model"],
            }));
        }
    }

    Json(json!({
        "status": "ok",
        "source_dir": source_dir.to_string_lossy(),
        "output_dir": output_dir.to_string_lossy(),
        "translated_pages": pages_summary.len(),
        "total_blocks": total_blocks,
        "free_text_blocks": free_blocks,
        "pages": pages_summary,
    }))
    .into_response()
}

pub async fn chapter_book(
    State(state): State<Arc<AppState>>,
    Query(q): Query<FolderQuery>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&q.folder_path), "folder_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let source_dir = unwrap_source_folder(Path::new(q.folder_path.trim()));
    if !state.allowed_roots.is_within_allowed_roots(&source_dir) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Папка вне разрешённых директорий"})),
        )
            .into_response();
    }
    let t_lang = q.target_lang.unwrap_or_else(|| "ru".to_string());
    let output_dir = output_dir_for_folder(&source_dir, &t_lang);
    let manifest = load_manifest(&output_dir);

    let mut book_pages = Vec::new();
    let mut script_lines = Vec::new();
    if let Some(pages) = manifest["pages"].as_object() {
        let mut keys: Vec<&String> = pages.keys().collect();
        keys.sort_by(|a, b| natural_cmp(a, b));
        for k in keys {
            script_lines.push(format!("=== {k} ==="));
            let blocks = pages[k]["blocks"].clone();
            if let Some(arr) = blocks.as_array() {
                for b in arr {
                    let tr = b["translated_text"].as_str().unwrap_or("").trim();
                    let orig = b["original_text"].as_str().unwrap_or("").trim();
                    if !tr.is_empty() || !orig.is_empty() {
                        script_lines.push(format!("[{}] {} || {}", b["id"], orig, tr));
                    }
                }
            }
            book_pages.push(json!({
                "page": k,
                "blocks": blocks,
            }));
        }
    }

    Json(json!({
        "status": "ok",
        "source_dir": source_dir.to_string_lossy(),
        "output_dir": output_dir.to_string_lossy(),
        "pages": book_pages,
        "script_text": script_lines.join("\n"),
    }))
    .into_response()
}

/// Возвращает полный совместимый с `web_studio.html` список страниц главы и кэшированных блоков.
pub fn inspect_folder_pages_json(state: &AppState, raw_dir: &Path, target_lang: &str) -> Value {
    let t_lang = match target_lang.trim().to_ascii_lowercase().as_str() {
        "ru" => "ru",
        "romaji" => "romaji",
        _ => "en",
    };
    let _ = state.allowed_roots.register_root(raw_dir);
    let source_dir = unwrap_source_folder(raw_dir);
    let output_dir = output_dir_for_folder(&source_dir, t_lang);
    let _ = state.allowed_roots.register_root(&source_dir);
    let _ = state.allowed_roots.register_root(&output_dir);

    let images = collect_folder_images(&source_dir);
    let manifest = load_manifest(&output_dir);
    let pages_meta = manifest.get("pages").and_then(|v| v.as_object());
    let phrase_cache = manifest.get("phrase_cache").and_then(|v| v.as_object());

    let mut pages_info = Vec::with_capacity(images.len());
    let mut translated_count = 0usize;

    for (idx, img_path) in images.iter().enumerate() {
        let filename = img_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("page.png")
            .to_string();
        let dest_path = output_dir.join(&filename);
        let dest_size = std::fs::metadata(&dest_path).map(|m| m.len()).unwrap_or(0);
        let dest_exists = dest_path.is_file() && dest_size > 128;

        let meta_opt = pages_meta.and_then(|m| m.get(&filename)).and_then(|v| v.as_object());
        let has_saved_blocks = meta_opt
            .and_then(|m| m.get("blocks"))
            .is_some_and(|v| v.is_array());

        let mut enriched_blocks: Vec<TextBlock> = Vec::new();
        if let Some(raw_arr) = meta_opt.and_then(|m| m.get("blocks")).and_then(|v| v.as_array()) {
            if let Ok(mut parsed) = serde_json::from_value::<Vec<TextBlock>>(Value::Array(raw_arr.clone())) {
                for b in &mut parsed {
                    normalize_clean_bbox(b);
                    let tr_s = b.translated_text.trim().to_string();
                    if b.is_deleted || (t_lang != "romaji" && tr_s.is_empty()) {
                        b.is_deleted = true;
                        b.translated_text.clear();
                    } else {
                        b.is_deleted = false;
                        b.translated_text = tr_s;
                    }
                    let orig_s = b.original_text.trim().to_string();
                    if !orig_s.is_empty() && b.romaji_text.trim().is_empty() {
                        let cached_rom = phrase_cache
                            .and_then(|pc| pc.get(&format!("romaji||{orig_s}")))
                            .and_then(|v| v.as_str())
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(str::to_string);
                        b.romaji_text = cached_rom.unwrap_or_else(|| kana_to_romaji(&orig_s));
                    }
                }
                enriched_blocks = parsed;
            }
        }

        let has_valid_translations = if has_saved_blocks {
            let raw_total = meta_opt
                .and_then(|m| m.get("blocks"))
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            raw_total == 0
                || enriched_blocks
                    .iter()
                    .any(|b| !b.is_deleted && !b.translated_text.trim().is_empty())
        } else {
            false
        };

        let is_translated = (dest_exists && (!has_saved_blocks || has_valid_translations))
            || (has_saved_blocks && has_valid_translations);
        if is_translated {
            translated_count += 1;
        }
        let blocks_count = meta_opt
            .and_then(|m| m.get("blocks_count"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .or_else(|| {
                if enriched_blocks.is_empty() {
                    None
                } else {
                    Some(enriched_blocks.len())
                }
            });
        let free_text_count = meta_opt
            .and_then(|m| m.get("free_text_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        pages_info.push(json!({
            "index": idx + 1,
            "rel_path": &filename,
            "filename": &filename,
            "source_path": img_path.to_string_lossy(),
            "dest_path": dest_path.to_string_lossy(),
            "dest_exists": dest_exists,
            "is_translated": is_translated,
            "blocks_count": blocks_count,
            "free_text_count": free_text_count,
            "detected_lang": meta_opt.and_then(|m| m.get("detected_lang")).cloned().unwrap_or(Value::Null),
            "updated_at": meta_opt.and_then(|m| m.get("updated_at")).cloned().unwrap_or(Value::Null),
            "has_saved_blocks": has_saved_blocks,
            "blocks": enriched_blocks,
        }));
    }

    let folder_glossary = state
        .db
        .load_glossary(&format!(
            "folder:{}",
            source_dir.to_string_lossy().trim().to_ascii_lowercase()
        ))
        .unwrap_or_default();

    json!({
        "status": "ok",
        "source_dir": source_dir.to_string_lossy(),
        "output_dir": output_dir.to_string_lossy(),
        "target_lang": t_lang,
        "total_pages": images.len(),
        "translated_pages": translated_count,
        "remaining_pages": images.len().saturating_sub(translated_count),
        "folder_glossary": folder_glossary,
        "pages": pages_info,
    })
}

struct PickerBusyGuard(Arc<std::sync::atomic::AtomicBool>);

impl Drop for PickerBusyGuard {
    fn drop(&mut self) {
        self.0.store(false, AtomicOrd::SeqCst);
    }
}

enum NativePickOutcome {
    Selected(PathBuf),
    Cancelled,
    Error(String),
}

fn run_windows_dialog(ps_script: &str) -> NativePickOutcome {
    #[cfg(windows)]
    {
        let out = std::process::Command::new("powershell")
            .args(["-NoProfile", "-STA", "-Command", ps_script])
            .output();
        match out {
            Ok(o) => {
                if !o.status.success() {
                    let err = String::from_utf8_lossy(&o.stderr).trim().to_string();
                    return NativePickOutcome::Error(if err.is_empty() {
                        "Системный диалог вернул ошибку".to_string()
                    } else {
                        err
                    });
                }
                let picked = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if picked.is_empty() {
                    NativePickOutcome::Cancelled
                } else {
                    NativePickOutcome::Selected(PathBuf::from(picked))
                }
            }
            Err(e) => NativePickOutcome::Error(e.to_string()),
        }
    }
    #[cfg(not(windows))]
    {
        let _ = ps_script;
        NativePickOutcome::Error("Нативный диалог выбора поддерживается только в Windows".to_string())
    }
}

fn pick_folder_blocking() -> NativePickOutcome {
    let script = r#"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.Opacity = 0
$owner.ShowInTaskbar = $false
$owner.StartPosition = 'CenterScreen'
$owner.Size = New-Object System.Drawing.Size(1, 1)
$owner.Show()
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = 'Select manga folder'
$dlg.UseDescriptionForTitle = $true
$dlg.ShowNewFolderButton = $true
$res = $dlg.ShowDialog($owner)
$owner.Close()
$owner.Dispose()
if ($res -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($dlg.SelectedPath) }
"#;
    run_windows_dialog(script)
}

fn pick_file_blocking() -> NativePickOutcome {
    let script = r#"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.Opacity = 0
$owner.ShowInTaskbar = $false
$owner.StartPosition = 'CenterScreen'
$owner.Size = New-Object System.Drawing.Size(1, 1)
$owner.Show()
$dlg = New-Object System.Windows.Forms.OpenFileDialog
$dlg.Title = 'Select manga image'
$dlg.Filter = 'Images|*.png;*.jpg;*.jpeg;*.webp;*.bmp|All Files|*.*'
$res = $dlg.ShowDialog($owner)
$owner.Close()
$owner.Dispose()
if ($res -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($dlg.FileName) }
"#;
    run_windows_dialog(script)
}

fn open_in_explorer(target: &Path) -> Result<(), String> {
    #[cfg(windows)]
    {
        std::process::Command::new("explorer.exe")
            .arg(target)
            .spawn()
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let _ = target;
        Err("Открытие проводника поддерживается только в Windows".to_string())
    }
}

#[derive(Debug, Deserialize)]
pub struct FolderInspectPayload {
    pub folder_path: String,
    pub target_lang: Option<String>,
}

pub async fn folder_inspect(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<FolderInspectPayload>,
) -> Response {
    if let Err(err) = reject_unc_path(Some(&payload.folder_path), "folder_path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let raw_dir = PathBuf::from(payload.folder_path.trim());
    let src_dir = unwrap_source_folder(&raw_dir);
    if !src_dir.is_dir() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": format!("Папка не найдена: {}", src_dir.display())})),
        )
            .into_response();
    }
    if !state.allowed_roots.is_within_allowed_roots(&raw_dir)
        && !state.allowed_roots.is_within_allowed_roots(&src_dir)
    {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Сначала выберите папку через системный диалог выбора папки."})),
        )
            .into_response();
    }
    let t_lang = payload.target_lang.unwrap_or_else(|| "en".to_string());
    Json(inspect_folder_pages_json(&state, &raw_dir, &t_lang)).into_response()
}

#[derive(Debug, Deserialize)]
pub struct PickFolderQuery {
    pub target_lang: Option<String>,
}

pub async fn pick_folder(
    State(state): State<Arc<AppState>>,
    Query(q): Query<PickFolderQuery>,
) -> Response {
    if state.picker_busy.swap(true, AtomicOrd::SeqCst) {
        return Json(json!({
            "status": "busy",
            "message": "Окно выбора папки уже открыто — проверьте панель задач Windows",
        }))
        .into_response();
    }
    let guard = PickerBusyGuard(state.picker_busy.clone());
    let res = tokio::task::spawn_blocking(move || {
        let _g = guard;
        pick_folder_blocking()
    })
    .await;

    match res {
        Ok(NativePickOutcome::Selected(dir_p)) => {
            if !dir_p.is_dir() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"detail": format!("Папка не найдена: {}", dir_p.display())})),
                )
                    .into_response();
            }
            let t_lang = q.target_lang.unwrap_or_else(|| "en".to_string());
            Json(inspect_folder_pages_json(&state, &dir_p, &t_lang)).into_response()
        }
        Ok(NativePickOutcome::Cancelled) => Json(json!({"status": "cancelled"})).into_response(),
        Ok(NativePickOutcome::Error(msg)) => {
            Json(json!({"status": "error", "message": msg})).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e.to_string()})),
        )
            .into_response(),
    }
}

pub async fn pick_file(State(state): State<Arc<AppState>>) -> Response {
    if state.picker_busy.swap(true, AtomicOrd::SeqCst) {
        return Json(json!({
            "status": "busy",
            "message": "Окно выбора файла уже открыто — проверьте панель задач Windows",
        }))
        .into_response();
    }
    let guard = PickerBusyGuard(state.picker_busy.clone());
    let res = tokio::task::spawn_blocking(move || {
        let _g = guard;
        pick_file_blocking()
    })
    .await;

    match res {
        Ok(NativePickOutcome::Selected(file_p)) => {
            if !file_p.is_file() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"detail": format!("Файл не найден: {}", file_p.display())})),
                )
                    .into_response();
            }
            let parent = file_p.parent().unwrap_or(Path::new("."));
            let _ = state.allowed_roots.register_root(parent);
            let fname = file_p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            Json(json!({
                "status": "ok",
                "file_path": file_p.to_string_lossy(),
                "filename": fname,
                "parent_dir": parent.to_string_lossy(),
            }))
            .into_response()
        }
        Ok(NativePickOutcome::Cancelled) => Json(json!({"status": "cancelled"})).into_response(),
        Ok(NativePickOutcome::Error(msg)) => {
            Json(json!({"status": "error", "message": msg})).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e.to_string()})),
        )
            .into_response(),
    }
}

pub async fn pick_web_save_folder(State(state): State<Arc<AppState>>) -> Response {
    let current_dir = state
        .config
        .read()
        .map(|c| c.web_save_dir.clone())
        .unwrap_or_default();
    if state.picker_busy.swap(true, AtomicOrd::SeqCst) {
        return Json(json!({
            "status": "busy",
            "message": "Окно выбора папки уже открыто — проверьте панель задач Windows",
            "web_save_dir": current_dir,
        }))
        .into_response();
    }
    let guard = PickerBusyGuard(state.picker_busy.clone());
    let res = tokio::task::spawn_blocking(move || {
        let _g = guard;
        pick_folder_blocking()
    })
    .await;

    match res {
        Ok(NativePickOutcome::Selected(dir_p)) => {
            let _ = std::fs::create_dir_all(&dir_p);
            let _ = state.allowed_roots.register_root(&dir_p);
            let dir_str = dir_p.to_string_lossy().to_string();
            let auto_save = {
                let mut cfg = state.config.write().unwrap_or_else(|e| e.into_inner());
                cfg.web_save_dir = dir_str.clone();
                cfg.auto_save_web_pages
            };
            Json(json!({
                "status": "ok",
                "web_save_dir": dir_str,
                "auto_save_web_pages": auto_save,
            }))
            .into_response()
        }
        Ok(NativePickOutcome::Cancelled) => {
            Json(json!({"status": "cancelled", "web_save_dir": current_dir})).into_response()
        }
        Ok(NativePickOutcome::Error(msg)) => {
            Json(json!({"status": "error", "message": msg, "web_save_dir": current_dir}))
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e.to_string()})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct OpenFolderQuery {
    pub path: Option<String>,
}

pub async fn open_folder(
    State(state): State<Arc<AppState>>,
    Query(q): Query<OpenFolderQuery>,
) -> Response {
    let raw = q.path.as_deref().unwrap_or("").trim();
    if raw.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Путь к папке не указан"})),
        )
            .into_response();
    }
    if let Err(err) = reject_unc_path(Some(raw), "path") {
        return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()}))).into_response();
    }
    let p = Path::new(raw);
    if !state.allowed_roots.is_within_allowed_roots(p) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"detail": "Открытие директории вне разрешённых корней запрещено."})),
        )
            .into_response();
    }
    if !p.exists() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "Папка еще не создана"})),
        )
            .into_response();
    }
    if !p.is_dir() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"detail": "Разрешено открывать только папки, а не файлы"})),
        )
            .into_response();
    }
    match open_in_explorer(p) {
        Ok(()) => Json(json!({"status": "ok"})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e})),
        )
            .into_response(),
    }
}

pub async fn open_web_save_folder(
    State(state): State<Arc<AppState>>,
    Query(q): Query<OpenFolderQuery>,
) -> Response {
    let default_dir = state
        .config
        .read()
        .map(|c| PathBuf::from(&c.web_save_dir))
        .unwrap_or_else(|_| state.base_dir.join("Saved_Manga"));

    let target = if let Some(raw) = q.path.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        if let Err(err) = reject_unc_path(Some(raw), "path") {
            return (StatusCode::BAD_REQUEST, Json(json!({"detail": err.to_string()})))
                .into_response();
        }
        let p = PathBuf::from(raw);
        if !state.allowed_roots.is_within_allowed_roots(&p)
            && !p.starts_with(&default_dir)
        {
            return (
                StatusCode::FORBIDDEN,
                Json(json!({"detail": "Доступ к директории вне разрешённых корней запрещён."})),
            )
                .into_response();
        }
        p
    } else {
        default_dir
    };

    let _ = std::fs::create_dir_all(&target);
    let _ = state.allowed_roots.register_root(&target);
    match open_in_explorer(&target) {
        Ok(()) => Json(json!({"status": "ok", "web_save_dir": target.to_string_lossy()}))
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": e})),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_sort_and_output_dir_unwrap() {
        let mut names = vec!["10.png", "2.png", "1.png", "ch1_p12.png", "ch1_p2.png"];
        names.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(names, vec!["1.png", "2.png", "10.png", "ch1_p2.png", "ch1_p12.png"]);

        let out = output_dir_for_folder(Path::new("/manga/ch1_RU"), "en");
        assert_eq!(out, PathBuf::from("/manga/ch1_EN"));
    }
}
