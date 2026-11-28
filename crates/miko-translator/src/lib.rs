pub mod canon;
pub mod chapter;
pub mod client;
pub mod names;
pub mod prompts;
pub mod qa_lint;

pub use canon::{
    build_canon_prompt_block, builtin_canon_en, canon_entry_to_override, canon_hits_fingerprint,
    match_canon, match_pure_moan_en, merge_overrides, validate_canon_override, CanonCategory,
    CanonDb, CanonEntry,
};
pub use chapter::ChapterAnalysis;
pub use client::{
    strip_model_prefix, RustTranslator, TranslateBatchRequest, TranslateBatchResult,
    TranslatorError,
};
pub use names::{
    builtin_fandom_db, extract_katakana_name_candidates, infer_fandom_series_from_candidates,
    match_fandom_series, resolve_name_lock, FandomDb, FandomSeries, NameCandidate,
};
pub use prompts::{
    build_translation_prompts, build_translation_prompts_with_config,
    compose_context_with_series_hint, compute_translation_cache_key,
    compute_translation_cache_key_with_prompts, detect_source_language,
    extract_series_hint_from_path, filter_matching_glossary, has_foreign_script_leak,
    is_refusal_response, kana_to_romaji, parse_flat_json_translations,
    parse_flat_json_translations_for_lang, restore_source_ellipses, target_lang_prompt_label,
    PromptConfig, DEFAULT_ANTI_CENSORSHIP_PREAMBLE, DEFAULT_SYSTEM_STYLE_PROMPT,
};
pub use qa_lint::{lint_pair, QaIssue, QaKind, QaMode, QaResult};

use miko_core::GlossaryEntry;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct BreakerState {
    consecutive_failures: u32,
    open_until: Option<Instant>,
}

/// Circuit Breaker для облачных и локальных LLM-эндпоинтов.
pub struct EndpointCircuitBreaker {
    failure_threshold: u32,
    cooldown: Duration,
    inner: Mutex<BreakerState>,
}

impl EndpointCircuitBreaker {
    pub fn new(failure_threshold: u32, cooldown: Duration) -> Self {
        Self {
            failure_threshold: failure_threshold.max(1),
            cooldown,
            inner: Mutex::new(BreakerState {
                consecutive_failures: 0,
                open_until: None,
            }),
        }
    }

    pub fn is_open(&self) -> bool {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(until) = guard.open_until {
            if Instant::now() < until {
                return true;
            }
            guard.open_until = None;
            guard.consecutive_failures = 0;
        }
        false
    }

    pub fn record_failure(&self) -> bool {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.consecutive_failures += 1;
        if guard.consecutive_failures >= self.failure_threshold {
            guard.open_until = Some(Instant::now() + self.cooldown);
            return true;
        }
        false
    }

    pub fn record_success(&self) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard.consecutive_failures = 0;
        guard.open_until = None;
    }
}

pub fn normalize_glossary_entries(entries: &[GlossaryEntry]) -> Vec<GlossaryEntry> {
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut out: Vec<GlossaryEntry> = Vec::new();
    for e in entries {
        let src = e.src.trim().to_string();
        let dst = e.dst.trim().to_string();
        if src.is_empty() || dst.is_empty() {
            continue;
        }
        let item = GlossaryEntry {
            src: src.clone(),
            dst,
            note: e.note.trim().to_string(),
        };
        if let Some(&idx) = seen.get(&src) {
            out[idx] = item;
        } else {
            seen.insert(src, out.len());
            out.push(item);
        }
    }
    out
}

pub fn merge_glossaries(
    global_list: &[GlossaryEntry],
    folder_list: &[GlossaryEntry],
) -> Vec<GlossaryEntry> {
    let mut combined = normalize_glossary_entries(global_list);
    let mut index_map: HashMap<String, usize> = combined
        .iter()
        .enumerate()
        .map(|(i, e)| (e.src.clone(), i))
        .collect();
    for item in normalize_glossary_entries(folder_list) {
        if let Some(&idx) = index_map.get(&item.src) {
            combined[idx] = item;
        } else {
            index_map.insert(item.src.clone(), combined.len());
            combined.push(item);
        }
    }
    combined
}

pub fn export_glossary_to_sakura_text(entries: &[GlossaryEntry]) -> String {
    entries
        .iter()
        .map(|e| {
            if e.note.trim().is_empty() {
                format!("{}->{}", e.src, e.dst)
            } else {
                format!("{}->{} # {}", e.src, e.dst, e.note.trim())
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_and_glossary_merge() {
        let cb = EndpointCircuitBreaker::new(2, Duration::from_secs(60));
        assert!(!cb.is_open());
        assert!(!cb.record_failure());
        assert!(cb.record_failure());
        assert!(cb.is_open());
        cb.record_success();
        assert!(!cb.is_open());

        let g1 = vec![GlossaryEntry {
            src: "美咲".into(),
            dst: "Misaki".into(),
            note: "".into(),
        }];
        let g2 = vec![GlossaryEntry {
            src: "美咲".into(),
            dst: "Мисаки".into(),
            note: "ру".into(),
        }];
        let merged = merge_glossaries(&g1, &g2);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].dst, "Мисаки");
        assert!(export_glossary_to_sakura_text(&merged).contains("美咲->Мисаки # ру"));
    }

    #[test]
    fn test_language_detect_romaji_and_json_parser() {
        assert_eq!(detect_source_language(&["今日はいい天気！".into()]), "ja");
        assert_eq!(detect_source_language(&["안녕하세요".into()]), "ko");
        assert_eq!(detect_source_language(&["你好世界".into()]), "zh");
        assert_eq!(detect_source_language(&["Hello world".into()]), "en");

        assert_eq!(kana_to_romaji("きょうは"), "kyouha");
        assert_eq!(kana_to_romaji("ストップ！"), "sutoppu!");

        let g = vec![
            GlossaryEntry {
                src: "太郎".into(),
                dst: "Таро".into(),
                note: String::new(),
            },
            GlossaryEntry {
                src: "花子".into(),
                dst: "Ханако".into(),
                note: String::new(),
            },
        ];
        let matched = filter_matching_glossary(&g, &["太郎が来た".into()]);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].dst, "Таро");

        let k1 = compute_translation_cache_key("太郎が来た", "ru", "m1", &g);
        let k2 = compute_translation_cache_key("太郎が来た", "ru", "m1", &[g[0].clone()]);
        assert_eq!(k1, k2);

        let raw_llm = "<think>reasoning</think>\n```json\n{\"0\": \"Таро пришёл! [shape: oval]\", \"1\": \"I cannot translate this\", \"2\": \"I can’t translate sexualized content involving a minor.\"}\n```";
        let keys = vec!["0".to_string(), "1".to_string(), "2".to_string()];
        let parsed = parse_flat_json_translations(raw_llm, &keys);
        assert_eq!(parsed.get("0").map(String::as_str), Some("Таро пришёл!"));
        assert!(!parsed.contains_key("1"));
        assert!(
            !parsed.contains_key("2"),
            "curly apostrophe refusal must be stripped"
        );

        assert_eq!(
            strip_model_prefix("unsloth::deepseek-v4-flash"),
            (Some("unsloth"), "deepseek-v4-flash")
        );
        assert_eq!(
            strip_model_prefix("ollama::shisa-v2.1-14b:Q4_K_M"),
            (Some("ollama"), "shisa-v2.1-14b:Q4_K_M")
        );
        assert_eq!(strip_model_prefix("deepseek-chat"), (None, "deepseek-chat"));
    }

    #[tokio::test]
    async fn test_mock_openai_translate_and_sqlite_cache() {
        use std::sync::{Arc, Mutex};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
        let addr = listener.local_addr().unwrap();
        let captured_req = Arc::new(Mutex::new(String::new()));
        let cap_clone = captured_req.clone();

        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = vec![0u8; 8192];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                *cap_clone.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).to_string();
                let body = r#"{"choices":[{"message":{"content":"{\"0\":\"Привет, мир!\",\"1\":\"Погнали!\"}"}}]}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes()).await;
            }
        });

        let db = miko_core::StudioDb::open_in_memory().expect("db");
        let tr = RustTranslator::new(Some("http://127.0.0.1:1".into()));
        let b1 = miko_core::TextBlock {
            id: 1,
            bbox: [10, 10, 200, 120],
            original_text: "こんにちは".into(),
            ..Default::default()
        };
        let b2 = miko_core::TextBlock {
            id: 2,
            bbox: [10, 140, 200, 250],
            original_text: "行こうぜ".into(),
            ..Default::default()
        };

        let req = TranslateBatchRequest {
            blocks: vec![b1, b2],
            target_lang: "ru".into(),
            model: "unsloth::deepseek-v4-flash".into(),
            cloud_url: format!("http://{addr}/v1"),
            cloud_api_key: "sk-test".into(),
            ..Default::default()
        };

        let res1 = tr
            .translate_blocks(req.clone(), Some(&db))
            .await
            .expect("tr1");
        assert_eq!(res1.api_calls, 1);
        assert_eq!(res1.cache_hits, 0);
        assert_eq!(res1.active_model, "deepseek-v4-flash");
        assert!(captured_req
            .lock()
            .unwrap()
            .contains("\"model\":\"deepseek-v4-flash\""));
        assert!(!captured_req.lock().unwrap().contains("unsloth::"));
        assert_eq!(res1.blocks[0].translated_text, "Привет, мир!");
        assert_eq!(res1.blocks[1].translated_text, "Погнали!");

        // Второй вызов с теми же фразами должен отдать 2/2 из SQLite кэша с 0 API-вызовов
        let res2 = tr.translate_blocks(req, Some(&db)).await.expect("tr2");
        assert_eq!(res2.api_calls, 0);
        assert_eq!(res2.cache_hits, 2);
        assert_eq!(res2.blocks[0].translated_text, "Привет, мир!");
    }

    #[tokio::test]
    async fn test_cloud_partial_censorship_falls_back_to_local_ollama_only_for_censored_keys() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        // 1. Mock Cloud: переводит "0", но цензурит "1" с кривым апострофом ’
        let cloud_listener = TcpListener::bind("127.0.0.1:0").await.expect("bind cloud");
        let cloud_addr = cloud_listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = cloud_listener.accept().await {
                let mut buf = vec![0u8; 8192];
                let _ = sock.read(&mut buf).await;
                let body = "{\"choices\":[{\"message\":{\"content\":\"{\\\"0\\\":\\\"So cute.\\\",\\\"1\\\":\\\"I can\\u2019t translate sexualized content involving a minor.\\\"}\"}}]}";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes()).await;
            }
        });

        // 2. Mock Local Ollama: отдаёт список моделей (/api/tags) и допереводит зацензуренный ключ (/api/chat)
        let ollama_listener = TcpListener::bind("127.0.0.1:0").await.expect("bind ollama");
        let ollama_addr = ollama_listener.local_addr().unwrap();
        tokio::spawn(async move {
            for _ in 0..2 {
                if let Ok((mut sock, _)) = ollama_listener.accept().await {
                    let mut buf = vec![0u8; 8192];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req_str = String::from_utf8_lossy(&buf[..n]);
                    let body = if req_str.starts_with("GET /api/tags") {
                        r#"{"models":[{"name":"huihui_ai/qwen3.5-abliterated:9b"}]}"#
                    } else {
                        r#"{"message":{"content":"{\"1\":\"I was stripped naked.\"}"}}"#
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                }
            }
        });

        let db = miko_core::StudioDb::open_in_memory().expect("db");
        let tr = RustTranslator::new(Some(format!("http://{ollama_addr}")));
        let req = TranslateBatchRequest {
            blocks: vec![
                miko_core::TextBlock {
                    id: 1,
                    bbox: [10, 10, 150, 120],
                    original_text: "可愛いなぁ".into(),
                    ..Default::default()
                },
                miko_core::TextBlock {
                    id: 2,
                    bbox: [10, 140, 150, 260],
                    original_text: "私は裸にされていました".into(),
                    ..Default::default()
                },
            ],
            target_lang: "en".into(),
            model: "unsloth::deepseek-v4-flash".into(),
            fallback_model: "ollama::huihui_ai/qwen3.5-abliterated:9b".into(),
            cloud_url: format!("http://{cloud_addr}/v1"),
            cloud_api_key: "sk-test".into(),
            ..Default::default()
        };

        let res = tr
            .translate_blocks(req, Some(&db))
            .await
            .expect("fallback ok");
        assert_eq!(res.blocks[0].translated_text, "So cute.");
        assert_eq!(res.blocks[1].translated_text, "I was stripped naked.");
        assert!(!res.blocks[1].translated_text.contains("translate"));
    }

    #[tokio::test]
    async fn test_intra_batch_dedup_and_micro_retry_eliminates_empty_bubbles() {
        use std::sync::{Arc, Mutex};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
        let addr = listener.local_addr().unwrap();
        let reqs: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let reqs_clone = reqs.clone();

        tokio::spawn(async move {
            for call_idx in 0..2 {
                if let Ok((mut sock, _)) = listener.accept().await {
                    let mut buf = vec![0u8; 16384];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let raw_req = String::from_utf8_lossy(&buf[..n]).to_string();
                    reqs_clone.lock().unwrap().push(raw_req);
                    // Первый запрос: намеренно опускает ключ "2", второй (micro_retry): допереводит "2"
                    let body = if call_idx == 0 {
                        r#"{"choices":[{"message":{"content":"{\"0\":\"Wait, really?\",\"1\":\"Then, from this moment on.\"}"}}]}"#
                    } else {
                        r#"{"choices":[{"message":{"content":"{\"2\":\"I.\"}"}}]}"#
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                }
            }
        });

        let db = miko_core::StudioDb::open_in_memory().expect("db");
        let tr = RustTranslator::new(Some("http://127.0.0.1:1".into()));
        let mk_blk = |id: i64, txt: &str| miko_core::TextBlock {
            id,
            bbox: [10, 10, 170, 290],
            original_text: txt.into(),
            ..Default::default()
        };
        let req = TranslateBatchRequest {
            blocks: vec![
                mk_blk(1, "いや〜セックスは人生で最も気持ちいいことなのに"),
                mk_blk(2, "いや〜セックスは人生で最も気持ちいいことなのに"),
                mk_blk(3, "それじゃあ今から…"),
                mk_blk(4, "いや〜セックスは人生で最も気持ちいいことなのに"),
                mk_blk(5, "私…"),
                mk_blk(6, "いや〜セックスは人生で最も気持ちいいことなのに"),
            ],
            target_lang: "en".into(),
            model: "unsloth::deepseek-v4-flash".into(),
            cloud_only_no_local_fallback: true,
            cloud_url: format!("http://{addr}/v1"),
            cloud_api_key: "sk-test".into(),
            ..Default::default()
        };

        let res = tr
            .translate_blocks(req, Some(&db))
            .await
            .expect("dedup + retry ok");
        assert_eq!(res.api_calls, 2);
        assert_eq!(res.untranslated_count, 0);
        for b in &res.blocks {
            assert!(
                !b.translated_text.trim().is_empty(),
                "block {} remained empty",
                b.id
            );
        }
        assert_eq!(res.blocks[0].translated_text, "Wait, really?");
        assert_eq!(res.blocks[1].translated_text, "Wait, really?");
        assert_eq!(res.blocks[3].translated_text, "Wait, really?");
        assert_eq!(res.blocks[5].translated_text, "Wait, really?");
        assert_eq!(res.blocks[2].translated_text, "Then, from this moment on…");
        assert_eq!(res.blocks[4].translated_text, "I…");

        let captured = reqs.lock().unwrap();
        assert_eq!(captured.len(), 2);
        // Первый запрос должен содержать только 3 уникальных ключа ("0","1","2"), а не 6!
        assert!(!captured[0].contains("\\\"3\\\""));
    }

    #[tokio::test]
    async fn test_foreign_script_devanagari_leak_rejected_and_retried() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        assert!(has_foreign_script_leak("Schluuर्प", "en"));
        assert!(!has_foreign_script_leak("OK~ ♥", "ru"));
        let keys = vec!["0".to_string()];
        let rejected =
            parse_flat_json_translations_for_lang(r#"{"0":"Schluuर्प"}"#, &keys, "en", false);
        assert!(rejected.is_empty(), "Devanagari leak must be rejected");

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            for call_idx in 0..2 {
                if let Ok((mut sock, _)) = listener.accept().await {
                    let mut buf = vec![0u8; 16384];
                    let _ = sock.read(&mut buf).await;
                    let body = if call_idx == 0 {
                        r#"{"choices":[{"message":{"content":"{\"0\":\"That feels great\",\"1\":\"Schluuर्प\"}"}}]}"#
                    } else {
                        r#"{"choices":[{"message":{"content":"{\"1\":\"Sluuurp\"}"}}]}"#
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                }
            }
        });

        let db = miko_core::StudioDb::open_in_memory().expect("db");
        let tr = RustTranslator::new(Some("http://127.0.0.1:1".into()));
        let req = TranslateBatchRequest {
            blocks: vec![
                miko_core::TextBlock {
                    id: 1,
                    bbox: [10, 10, 150, 200],
                    original_text: "気持ちいい".into(),
                    ..Default::default()
                },
                miko_core::TextBlock {
                    id: 2,
                    bbox: [10, 210, 150, 400],
                    original_text: "ちゅちる".into(),
                    ..Default::default()
                },
            ],
            target_lang: "en".into(),
            model: "unsloth::deepseek-v4-flash".into(),
            cloud_only_no_local_fallback: true,
            cloud_url: format!("http://{addr}/v1"),
            cloud_api_key: "sk-test".into(),
            ..Default::default()
        };

        let res = tr
            .translate_blocks(req, Some(&db))
            .await
            .expect("foreign leak retried");
        assert_eq!(res.api_calls, 2);
        assert_eq!(res.blocks[0].translated_text, "That feels great");
        assert_eq!(res.blocks[1].translated_text, "Sluuurp");
    }

    #[test]
    fn test_custom_prompts_contract_and_cache_key_invalidation() {
        let g = vec![GlossaryEntry {
            src: "七香".into(),
            dst: "Нанака".into(),
            note: "героиня".into(),
        }];
        let items = vec![("0".to_string(), "七香、行こう…".to_string(), Some(40))];

        let default_cfg = PromptConfig::default();
        let custom_sys_cfg = PromptConfig {
            anti_censorship: true,
            anti_censorship_custom_prompt: "[CUSTOM 18+ HEADER]".into(),
            custom_system_prompt: "Переводи дерзким разговорным сленгом на {{LANG}}.".into(),
            custom_context_prompt: "Действие происходит в Киото.".into(),
            ..Default::default()
        };

        let (sys_def, _user_def) =
            build_translation_prompts_with_config("ru", &items, &g, &[], &default_cfg);
        assert!(sys_def.contains("[EXTENDED FICTION LOCALIZATION CONTEXT]"));
        assert!(sys_def.contains("MANDATORY OUTPUT CONTRACT:"));
        assert!(sys_def.contains("七香 => Нанака"));

        let (sys_cust, user_cust) =
            build_translation_prompts_with_config("ru", &items, &g, &[], &custom_sys_cfg);
        assert!(sys_cust.contains("[CUSTOM 18+ HEADER]"));
        assert!(!sys_cust.contains("[EXTENDED FICTION LOCALIZATION CONTEXT]"));
        assert!(sys_cust.contains("Переводи дерзким разговорным сленгом на Russian"));
        assert!(sys_cust.contains("SERIES / CHAPTER CONTEXT NOTES:\nДействие происходит в Киото."));
        // Защищённый JSON-контракт и глоссарий обязаны сохраняться при любом кастомном промпте!
        assert!(sys_cust.contains("MANDATORY OUTPUT CONTRACT:"));
        assert!(sys_cust.contains("Return ONLY a flat valid JSON object"));
        assert!(sys_cust.contains("七香 => Нанака (героиня)"));
        assert!(user_cust.contains("\"0\": \"七香、行こう…\""));

        // Ключ кэша ОБЯЗАН меняться при изменении любого поля промпта или флага антицензуры
        let k_def = compute_translation_cache_key_with_prompts(
            "七香、行こう…",
            "ru",
            "m1",
            &g,
            &default_cfg,
        );
        let k_no_ac = compute_translation_cache_key_with_prompts(
            "七香、行こう…",
            "ru",
            "m1",
            &g,
            &PromptConfig::new_default(false),
        );
        let k_cust = compute_translation_cache_key_with_prompts(
            "七香、行こう…",
            "ru",
            "m1",
            &g,
            &custom_sys_cfg,
        );
        assert_ne!(
            k_def, k_no_ac,
            "toggling anti_censorship must invalidate cache key"
        );
        assert_ne!(
            k_def, k_cust,
            "editing custom prompt must invalidate cache key"
        );
    }

    #[test]
    fn test_series_hint_extraction_and_internal_ellipsis_restoration() {
        use std::path::Path;

        let p = Path::new(
            r"C:\Users\Alexey\Documents\MANGA TEST TRANSLATIONS\[ProudBanana] Kuon no Ai - Mirrored Lotus (Sousou no Frieren) [Decensored] [Digital]",
        );
        let hint = extract_series_hint_from_path(p).expect("series hint");
        assert_eq!(
            hint,
            "[ProudBanana] Kuon no Ai - Mirrored Lotus (Sousou no Frieren)"
        );

        let composed = compose_context_with_series_hint("", Some(p));
        assert!(composed.contains("Sousou no Frieren"));
        assert!(composed.contains("canonical character"));

        // Подпапка `ch01` должна подниматься к родительской папке тайтла, а Temp-папка — игнорироваться
        let sub_ch = Path::new(r"D:\Manga\Sousou no Frieren [RAW]\Chapter 05");
        assert_eq!(
            extract_series_hint_from_path(sub_ch).as_deref(),
            Some("Sousou no Frieren")
        );
        let temp_p = Path::new(r"C:\Users\Alexey\AppData\Local\Temp\.tmpAbCdEf\ch1");
        assert!(extract_series_hint_from_path(temp_p).is_none());

        // Восстановление внутренних пауз-многоточий из реального лога (стр. 015, 016, 019, 027)
        assert_eq!(
            restore_source_ellipses(
                "聞きにくいだけど…昨日はその…平気なのかい?",
                "This is hard to ask, but. about yesterday. are you okay?"
            ),
            "This is hard to ask, but… about yesterday… are you okay?"
        );
        assert_eq!(
            restore_source_ellipses("そ…そこは…", "Th. there…"),
            "Th… there…"
        );
        assert_eq!(
            restore_source_ellipses("んんっ…変な…感じ…", "Mmm. This feels. strange…"),
            "Mmm… This feels… strange…"
        );
        assert_eq!(
            restore_source_ellipses("やめるんだ…フリーレン", "Stop. Frieren"),
            "Stop… Frieren"
        );
        // Обычные два предложения без исходного `…` не должны трогаться!
        assert_eq!(
            restore_source_ellipses(
                "後ろの初めてはもらいましたよここまでキツキツとは",
                "I got your first time from behind. I didn't expect you to be this tight."
            ),
            "I got your first time from behind. I didn't expect you to be this tight."
        );
    }

    #[tokio::test]
    async fn test_intra_chunk_context_passed_to_local_fallback_for_character_names() {
        use std::sync::{Arc, Mutex};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        // Облако переводит "0" с каноничным именем Frieren, но цензурит "1"
        let cloud_listener = TcpListener::bind("127.0.0.1:0").await.expect("bind cloud");
        let cloud_addr = cloud_listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = cloud_listener.accept().await {
                let mut buf = vec![0u8; 8192];
                let _ = sock.read(&mut buf).await;
                let body = "{\"choices\":[{\"message\":{\"content\":\"{\\\"0\\\":\\\"Hey, Frieren.\\\",\\\"1\\\":\\\"I cannot translate explicit sexual content.\\\"}\"}}]}";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes()).await;
            }
        });

        // Локальная Ollama должна получить в системном промпте свежий контекст "なぁフリーレン => Hey, Frieren." из этого же чанка!
        let ollama_listener = TcpListener::bind("127.0.0.1:0").await.expect("bind ollama");
        let ollama_addr = ollama_listener.local_addr().unwrap();
        let captured_ollama_req = Arc::new(Mutex::new(String::new()));
        let cap_clone = captured_ollama_req.clone();
        tokio::spawn(async move {
            for _ in 0..2 {
                if let Ok((mut sock, _)) = ollama_listener.accept().await {
                    let mut buf = vec![0u8; 16384];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req_str = String::from_utf8_lossy(&buf[..n]).to_string();
                    let body = if req_str.starts_with("GET /api/tags") {
                        r#"{"models":[{"name":"huihui-qwen3.5:9b"}]}"#
                    } else {
                        *cap_clone.lock().unwrap() = req_str;
                        r#"{"message":{"content":"{\"1\":\"Frieren is getting used to it.\"}"}}"#
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                }
            }
        });

        let db = miko_core::StudioDb::open_in_memory().expect("db");
        let tr = RustTranslator::new(Some(format!("http://{ollama_addr}")));
        let req = TranslateBatchRequest {
            blocks: vec![
                miko_core::TextBlock {
                    id: 1,
                    bbox: [10, 10, 150, 120],
                    original_text: "なぁフリーレン".into(),
                    ..Default::default()
                },
                miko_core::TextBlock {
                    id: 2,
                    bbox: [10, 140, 150, 260],
                    original_text: "フリーレンも馴染んできましたね".into(),
                    ..Default::default()
                },
            ],
            target_lang: "en".into(),
            model: "unsloth::deepseek-v4-flash".into(),
            fallback_model: "ollama::huihui-qwen3.5:9b".into(),
            cloud_url: format!("http://{cloud_addr}/v1"),
            cloud_api_key: "sk-test".into(),
            ..Default::default()
        };

        let res = tr.translate_blocks(req, Some(&db)).await.expect("ok");
        assert_eq!(res.blocks[0].translated_text, "Hey, Frieren.");
        assert_eq!(
            res.blocks[1].translated_text,
            "Frieren is getting used to it."
        );
        let ollama_payload = captured_ollama_req.lock().unwrap().clone();
        assert!(
            ollama_payload.contains("Hey, Frieren."),
            "Local fallback must receive same-chunk cloud translations in RECENT CONTEXT: {ollama_payload}"
        );
    }

    #[test]
    fn test_canon_loads_and_matches_kuon_fixtures() {
        let db = builtin_canon_en();
        assert!(!db.entries.is_empty(), "builtin canon DB must not be empty");

        let hits = match_canon(
            db,
            &[
                "ほほう〜絶頂しましたね".to_string(),
                "ヒンメルのちんちんキレイでかわいいね".to_string(),
                "イクッ!!".to_string(),
            ],
        );
        let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        assert!(ids.contains(&"climax_zecchou"));
        assert!(ids.contains(&"moan_hohou"));
        assert!(ids.contains(&"anatomy_chinchin"));
        assert!(ids.contains(&"climax_iku"));

        // expand_kana=false guard: обычные слова на хирагане НЕ должны ложно триггерить `climax_iku` / `act_yaru`
        let safe_hits = match_canon(
            db,
            &[
                "いっしょに行こう！".to_string(),
                "いっぱい食べよう".to_string(),
                "やる気が出た".to_string(),
            ],
        );
        let safe_ids: Vec<&str> = safe_hits.iter().map(|h| h.id.as_str()).collect();
        assert!(
            !safe_ids.contains(&"climax_iku"),
            "hiragana いっしょ/いっぱい/行く must not trigger climax_iku: {safe_ids:?}"
        );
        assert!(!safe_ids.contains(&"act_yaru"));
    }

    #[test]
    fn test_canon_prompt_block_only_matched_terms_and_cache_key_changes_with_canon_hits() {
        let items = vec![(
            "0".to_string(),
            "ヒンメルのちんちんキレイでかわいいね".to_string(),
            Some(50),
        )];
        let cfg_on = PromptConfig::default();
        let mut cfg_off = PromptConfig::default();
        cfg_off.adult_canon_en = false;

        let (sys_en, _) = build_translation_prompts_with_config("en", &items, &[], &[], &cfg_on);
        assert!(sys_en.contains("ADULT MANGA CANON"));
        assert!(sys_en.contains("ちんちん"));
        assert!(sys_en.contains("\"dick\""));
        assert!(
            !sys_en.contains("フェラ"),
            "unmatched canon entries must NOT bloat the prompt"
        );

        let (sys_ru, _) = build_translation_prompts_with_config("ru", &items, &[], &[], &cfg_on);
        assert!(
            !sys_ru.contains("ADULT MANGA CANON"),
            "English adult canon must not inject into Russian prompts"
        );

        let k_on = compute_translation_cache_key_with_prompts(
            "ヒンメルのちんちんキレイでかわいいね",
            "en",
            "m1",
            &[],
            &cfg_on,
        );
        let k_off = compute_translation_cache_key_with_prompts(
            "ヒンメルのちんちんキレイでかわいいね",
            "en",
            "m1",
            &[],
            &cfg_off,
        );
        assert_ne!(k_on, k_off, "canon hit fingerprint must alter EN cache key");

        let mut cfg_ov = PromptConfig::default();
        cfg_ov.canon_overrides = vec![miko_core::CanonOverride {
            id: "anatomy_chinchin".into(),
            category: "anatomy".into(),
            jp: vec!["ちんちん".into()],
            expand_kana: true,
            jp_regex: None,
            en: "cock".into(),
            en_alts: vec![],
            forbid_en: vec![],
            required_en_any: vec![],
            auto_fix: vec![],
            note: String::new(),
            priority: 95,
            enabled: true,
        }];
        let k_ov = compute_translation_cache_key_with_prompts(
            "ヒンメルのちんちんキレイでかわいいね",
            "en",
            "m1",
            &[],
            &cfg_ov,
        );
        assert_ne!(
            k_on, k_ov,
            "user canon override must invalidate matched line cache key"
        );
    }

    #[test]
    fn test_katakana_name_extraction_with_stoplist_and_frieren_fandom_resolve() {
        let lines = vec![
            "勇者ヒンメル一行".to_string(),
            "よかったですねヒンメル!".to_string(),
            "ハイターはわかるが何でアイゼンまで乗り気なんだ!?".to_string(),
            "フリーレンのセックスとフェラとエルフのビキニ".to_string(),
            "ドキドキ…ジュポジュポ…".to_string(),
            "生臭坊主め！".to_string(),
        ];
        let cands = extract_katakana_name_candidates(&lines);
        let cand_names: Vec<&str> = cands.iter().map(|c| c.jp.as_str()).collect();
        assert!(cand_names.contains(&"ヒンメル"));
        assert!(cand_names.contains(&"ハイター"));
        assert!(cand_names.contains(&"アイゼン"));
        assert!(cand_names.contains(&"フリーレン"));
        assert!(!cand_names.contains(&"セックス"));
        assert!(!cand_names.contains(&"フェラ"));
        assert!(!cand_names.contains(&"エルフ"));
        assert!(!cand_names.contains(&"ビキニ"));
        assert!(!cand_names.contains(&"ドキドキ"));
        assert!(!cand_names.contains(&"ジュポジュポ"));

        let fandom = builtin_fandom_db();
        let locked = resolve_name_lock(
            &cands,
            Some("[ProudBanana] Kuon no Ai (Sousou no Frieren)"),
            fandom,
            &[],
        );
        let map: HashMap<&str, &str> = locked
            .iter()
            .map(|g| (g.src.as_str(), g.dst.as_str()))
            .collect();
        assert_eq!(map.get("ヒンメル"), Some(&"Himmel"));
        assert_eq!(map.get("ハイター"), Some(&"Heiter"));
        assert_eq!(map.get("アイゼン"), Some(&"Eisen"));
        assert_eq!(map.get("フリーレン"), Some(&"Frieren"));
        assert_eq!(map.get("生臭坊主"), Some(&"corrupt priest"));
    }

    #[test]
    fn test_lint_fixes_name_drift_peak_junk_wet_and_grammar_anchor_igai() {
        let canon = builtin_canon_en();
        let glossary = vec![
            GlossaryEntry {
                src: "ヒンメル".into(),
                dst: "Himmel".into(),
                note: "auto-lock:fandom:frieren".into(),
            },
            GlossaryEntry {
                src: "ハイター".into(),
                dst: "Heiter".into(),
                note: "auto-lock:fandom:frieren".into(),
            },
            GlossaryEntry {
                src: "アイゼン".into(),
                dst: "Eisen".into(),
                note: "auto-lock:fandom:frieren".into(),
            },
            GlossaryEntry {
                src: "生臭坊主".into(),
                dst: "corrupt priest".into(),
                note: "auto-lock:fandom-term:frieren".into(),
            },
        ];

        // 1. Name drift: Himm's / Himm / Himmle / the bleach / Rough, monk!
        let r1 = lint_pair(
            "勇者ヒンメル一行",
            "The Hero Himm's Party",
            &glossary,
            canon,
        );
        assert_eq!(r1.text, "The Hero Himmel's Party");
        assert!(!r1.needs_retry);

        let r2 = lint_pair("どうもヒンメル", "Hello, Himmle!", &glossary, canon);
        assert_eq!(r2.text, "Hello, Himmel!");
        assert!(!r2.needs_retry);

        let r3 = lint_pair(
            "ハイターはわかるが何でアイゼンまで乗り気なんだ!?",
            "I get the bleach, but why is Eisen into it too!?",
            &glossary,
            canon,
        );
        assert_eq!(r3.text, "I get Heiter, but why is Eisen into it too!?");
        assert!(!r3.needs_retry);

        let r_monk = lint_pair("生臭坊主", "Rough, monk!", &glossary, canon);
        assert_eq!(r_monk.text, "Corrupt priest!");
        assert!(!r_monk.needs_retry);

        // 2. Adult canon auto-fixes (peak, Hohoho, junk, nut, big bulk, let it rip, cunnilingus, speaker inversion, Wet!!, sexy monk)
        let r_peak = lint_pair(
            "ほほう〜絶頂しましたね",
            "Hohoho~ You reached the peak!",
            &glossary,
            canon,
        );
        assert_eq!(r_peak.text, "Oh my~ You came!");
        assert!(!r_peak.needs_retry);

        let r_junk = lint_pair(
            "ヒンメルのちんちんキレイでかわいいね",
            "Himmel's junk is so clean and cute!",
            &glossary,
            canon,
        );
        assert_eq!(r_junk.text, "Himmel's dick is so clean and cute!");
        assert!(!r_junk.needs_retry);

        let r_nut = lint_pair("おちんちんの味…", "The taste of a nut…", &glossary, canon);
        assert_eq!(r_nut.text, "The taste of a dick…");
        assert!(!r_nut.needs_retry);

        let r_bulk = lint_pair(
            "アイゼンのデカブツもすんなり入れるようになりましたし",
            "Even Eisen's big bulk slides in easily",
            &glossary,
            canon,
        );
        assert_eq!(r_bulk.text, "Even Eisen's huge cock slides in easily");

        let r_rip = lint_pair("出しちゃえ♥", "Let it rip♥", &glossary, canon);
        assert_eq!(r_rip.text, "Cum for me♥");

        let r_fera = lint_pair(
            "…フェラもますます上達していますよ",
            "…her cunnilingus has gotten better",
            &glossary,
            canon,
        );
        assert_eq!(r_fera.text, "…her blowjobs has gotten better");

        let r_inv1 = lint_pair(
            "中が締め付けてきて実に気持ちがいいです",
            "My insides are being squeezed…",
            &glossary,
            canon,
        );
        assert_eq!(r_inv1.text, "Your insides are squeezing me…");

        let r_inv2 = lint_pair(
            "私はこの無防備な穴をいただくとしましょうか",
            "I'll take this defenseless hole of mine!",
            &glossary,
            canon,
        );
        assert_eq!(r_inv2.text, "I'll take this defenseless hole!");

        let r_wet = lint_pair("イクッ!!", "Wet!!", &glossary, canon);
        assert_eq!(r_wet.text, "I'm cumming!!");

        let r_ero = lint_pair("エロ坊主め", "You sexy monk!", &glossary, canon);
        assert_eq!(r_ero.text, "You pervy priest!");

        // 3. Grammar anchor `以外`: lost exclusion MUST flag needs_retry (even if ", but" is present!)
        let r_igai_bad = lint_pair(
            "こうしてヒンメル以外は一晩中ヤリまくりだった",
            "Thus, Himmle and the rest fucked all night long, but stayed awake",
            &glossary,
            canon,
        );
        assert_eq!(
            r_igai_bad.text,
            "Thus, Himmel and the rest fucked all night long, but stayed awake"
        );
        assert!(
            r_igai_bad.needs_retry,
            "lost 以外 must trigger retry even when conjunction ', but' exists"
        );

        let r_igai_ok = lint_pair(
            "こうしてヒンメル以外は一晩中ヤリまくりだった",
            "And so, everyone except Himmel fucked all night long",
            &glossary,
            canon,
        );
        assert!(!r_igai_ok.needs_retry);
    }

    #[tokio::test]
    async fn test_moan_table_bypasses_llm_and_linter_micro_retry_replaces_bad_line() {
        use std::sync::{Arc, Mutex};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        // 1. Детерминированные стоны обходят LLM (0 запросов к API)
        let db = miko_core::StudioDb::open_in_memory().expect("db");
        let tr = RustTranslator::new(Some("http://127.0.0.1:1".into()));
        let moan_req = TranslateBatchRequest {
            blocks: vec![
                miko_core::TextBlock {
                    id: 1,
                    bbox: [10, 10, 100, 120],
                    original_text: "ふう…".into(),
                    ..Default::default()
                },
                miko_core::TextBlock {
                    id: 2,
                    bbox: [10, 130, 100, 240],
                    original_text: "あんっ!!".into(),
                    ..Default::default()
                },
            ],
            target_lang: "en".into(),
            cloud_only_no_local_fallback: true,
            cloud_url: "http://127.0.0.1:1/v1".into(),
            use_cache: false,
            ..Default::default()
        };
        let moan_res = tr
            .translate_blocks(moan_req, Some(&db))
            .await
            .expect("moans ok");
        assert_eq!(moan_res.api_calls, 0);
        assert_eq!(moan_res.blocks[0].translated_text, "Phew…");
        assert_eq!(moan_res.blocks[1].translated_text, "Ahn!!");

        // 2. Post-QA Linter делает 1 микро-переперевод с подсказкой `QA CORRECTIONS`, когда потерян якорь `以外`
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
        let addr = listener.local_addr().unwrap();
        let reqs: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let reqs_clone = reqs.clone();

        tokio::spawn(async move {
            for call_idx in 0..2 {
                if let Ok((mut sock, _)) = listener.accept().await {
                    let mut buf = vec![0u8; 16384];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    reqs_clone
                        .lock()
                        .unwrap()
                        .push(String::from_utf8_lossy(&buf[..n]).to_string());
                    let body = if call_idx == 0 {
                        r#"{"choices":[{"message":{"content":"{\"0\":\"Thus, Himmle and the rest fucked all night long\"}"}}]}"#
                    } else {
                        r#"{"choices":[{"message":{"content":"{\"0\":\"And so, everyone except Himmel fucked all night long\"}"}}]}"#
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                }
            }
        });

        let qa_req = TranslateBatchRequest {
            blocks: vec![miko_core::TextBlock {
                id: 1,
                bbox: [10, 10, 200, 180],
                original_text: "こうしてヒンメル以外は一晩中ヤリまくりだった".into(),
                ..Default::default()
            }],
            target_lang: "en".into(),
            model: "unsloth::deepseek-v4-flash".into(),
            cloud_only_no_local_fallback: true,
            cloud_url: format!("http://{addr}/v1"),
            cloud_api_key: "sk-test".into(),
            glossary: vec![GlossaryEntry {
                src: "ヒンメル".into(),
                dst: "Himmel".into(),
                note: "auto-lock".into(),
            }],
            ..Default::default()
        };
        let qa_res = tr
            .translate_blocks(qa_req, Some(&db))
            .await
            .expect("qa retry ok");
        assert_eq!(qa_res.api_calls, 2);
        assert_eq!(qa_res.qa_retried, 1);
        assert_eq!(
            qa_res.blocks[0].translated_text,
            "And so, everyone except Himmel fucked all night long"
        );
        let captured = reqs.lock().unwrap();
        assert!(
            captured[1].contains("QA CORRECTIONS"),
            "second call must include QA CORRECTIONS hint: {}",
            captured[1]
        );
    }
}

pub fn export_glossary_to_tsv(entries: &[GlossaryEntry]) -> String {
    entries
        .iter()
        .map(|e| format!("{}\t{}\t{}", e.src, e.dst, e.note.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}
