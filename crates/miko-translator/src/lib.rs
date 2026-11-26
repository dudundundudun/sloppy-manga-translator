pub mod client;
pub mod prompts;

pub use client::{
    strip_model_prefix, RustTranslator, TranslateBatchRequest, TranslateBatchResult,
    TranslatorError,
};
pub use prompts::{
    build_translation_prompts, compute_translation_cache_key, detect_source_language,
    filter_matching_glossary, is_refusal_response, kana_to_romaji, parse_flat_json_translations,
};

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
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
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
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        guard.consecutive_failures += 1;
        if guard.consecutive_failures >= self.failure_threshold {
            guard.open_until = Some(Instant::now() + self.cooldown);
            return true;
        }
        false
    }

    pub fn record_success(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.consecutive_failures = 0;
            guard.open_until = None;
        }
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

pub fn merge_glossaries(global_list: &[GlossaryEntry], folder_list: &[GlossaryEntry]) -> Vec<GlossaryEntry> {
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
        assert!(!parsed.contains_key("2"), "curly apostrophe refusal must be stripped");

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

        let res1 = tr.translate_blocks(req.clone(), Some(&db)).await.expect("tr1");
        assert_eq!(res1.api_calls, 1);
        assert_eq!(res1.cache_hits, 0);
        assert_eq!(res1.active_model, "deepseek-v4-flash");
        assert!(captured_req.lock().unwrap().contains("\"model\":\"deepseek-v4-flash\""));
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

        let res = tr.translate_blocks(req, Some(&db)).await.expect("fallback ok");
        assert_eq!(res.blocks[0].translated_text, "So cute.");
        assert_eq!(res.blocks[1].translated_text, "I was stripped naked.");
        assert!(!res.blocks[1].translated_text.contains("translate"));
    }
}

pub fn export_glossary_to_tsv(entries: &[GlossaryEntry]) -> String {
    entries
        .iter()
        .map(|e| format!("{}\t{}\t{}", e.src, e.dst, e.note.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}
