use crate::prompts::{
    build_translation_prompts, compute_translation_cache_key, detect_source_language,
    is_refusal_response, kana_to_romaji, parse_flat_json_translations_for_lang,
    restore_source_ellipses,
};
use crate::EndpointCircuitBreaker;
use miko_core::{GlossaryEntry, StudioDb, TextBlock};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TranslatorError {
    #[error("Ошибка HTTP-запроса к LLM: {0}")]
    Http(#[from] reqwest::Error),
    #[error("LLM-сервер вернул статус {status}: {body}")]
    ApiStatus { status: u16, body: String },
    #[error("Цензурный отказ облачной модели")]
    Refused,
    #[error("Не удалось получить переводы (сервер недоступен или вернул отказ)")]
    UnavailableOrRefused,
}

/// Отделяет внутренний префикс провайдера (`unsloth::`, `cloud::`, `openai::`, `ollama::`) от имени модели.
pub fn strip_model_prefix(raw: &str) -> (Option<&str>, &str) {
    let trimmed = raw.trim();
    for pref in ["unsloth", "cloud", "openai", "ollama"] {
        if let Some(rest) = trimmed.strip_prefix(pref) {
            if let Some(bare) = rest.strip_prefix("::") {
                return (Some(pref), bare.trim());
            }
        }
    }
    (None, trimmed)
}

#[derive(Debug, Clone)]
pub struct TranslateBatchRequest {
    pub blocks: Vec<TextBlock>,
    pub target_lang: String,
    pub model: String,
    pub fallback_model: String,
    pub cloud_only_no_local_fallback: bool,
    pub cloud_url: String,
    pub cloud_api_key: String,
    pub glossary: Vec<GlossaryEntry>,
    pub context_lines: Vec<String>,
    pub anti_censorship: bool,
    pub use_cache: bool,
}

impl Default for TranslateBatchRequest {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            target_lang: "ru".to_string(),
            model: "unsloth::deepseek-v4-flash".to_string(),
            fallback_model: "auto".to_string(),
            cloud_only_no_local_fallback: false,
            cloud_url: String::new(),
            cloud_api_key: String::new(),
            glossary: Vec::new(),
            context_lines: Vec::new(),
            anti_censorship: true,
            use_cache: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TranslateBatchResult {
    pub blocks: Vec<TextBlock>,
    pub active_model: String,
    pub detected_lang: String,
    pub cache_hits: usize,
    pub api_calls: usize,
    pub elapsed_ms: u128,
    pub censored_fallback: bool,
    pub untranslated_count: usize,
}

/// Нативный async HTTP-переводчик манги (OpenAI-compatible Cloud + Local Ollama + SQLite WAL).
pub struct RustTranslator {
    http_cloud: reqwest::Client,
    http_local: reqwest::Client,
    ollama_base_url: String,
    cloud_breaker: EndpointCircuitBreaker,
    ollama_breaker: EndpointCircuitBreaker,
    /// Если облако цензурит 2 батча подряд в 18+ сцене — на 3 минуты сразу идём в локальную модель без потери 15с на запрос.
    censorship_breaker: EndpointCircuitBreaker,
}

#[derive(Deserialize)]
struct OllamaTagsResp {
    models: Option<Vec<OllamaTagItem>>,
}

#[derive(Deserialize)]
struct OllamaTagItem {
    name: String,
}

impl RustTranslator {
    pub fn new(ollama_base_url: Option<String>) -> Self {
        let base = ollama_base_url
            .unwrap_or_else(|| "http://127.0.0.1:11434".to_string())
            .trim_end_matches('/')
            .to_string();

        let http_cloud = reqwest::Client::builder()
            .timeout(Duration::from_secs(45))
            .connect_timeout(Duration::from_secs(12))
            .pool_idle_timeout(Duration::from_secs(20))
            .tcp_keepalive(Duration::from_secs(15))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let http_local = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(180))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            http_cloud,
            http_local,
            ollama_base_url: base,
            cloud_breaker: EndpointCircuitBreaker::new(3, Duration::from_secs(45)),
            ollama_breaker: EndpointCircuitBreaker::new(3, Duration::from_secs(30)),
            censorship_breaker: EndpointCircuitBreaker::new(2, Duration::from_secs(180)),
        }
    }

    /// Сбрасывает счётчик 18+ цензуры в начале новой главы.
    pub fn reset_censorship_bypass(&self) {
        self.censorship_breaker.record_success();
    }

    pub fn ollama_base_url(&self) -> &str {
        &self.ollama_base_url
    }

    /// Возвращает список установленных моделей в локальной Ollama (пустой вектор, если выключена).
    pub async fn list_ollama_models(&self) -> Vec<String> {
        let url = format!("{}/api/tags", self.ollama_base_url);
        let Ok(resp) = self
            .http_local
            .get(url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
        else {
            return Vec::new();
        };
        if !resp.status().is_success() {
            return Vec::new();
        }
        let Ok(parsed) = resp.json::<OllamaTagsResp>().await else {
            return Vec::new();
        };
        parsed
            .models
            .unwrap_or_default()
            .into_iter()
            .map(|m| m.name)
            .collect()
    }

    async fn call_cloud_openai(
        &self,
        cloud_url: &str,
        api_key: &str,
        model: &str,
        sys_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, TranslatorError> {
        let endpoint = format!("{}/chat/completions", cloud_url.trim_end_matches('/'));
        let body = json!({
            "model": model,
            "temperature": 0.25,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        });

        let mut last_err: Option<TranslatorError> = None;
        for attempt in 0..2usize {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            let mut req = self.http_cloud.post(&endpoint).json(&body);
            if !api_key.trim().is_empty() {
                req = req.bearer_auth(api_key.trim());
            }
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(TranslatorError::Http(e));
                    continue;
                }
            };
            let status = resp.status().as_u16();
            if !resp.status().is_success() {
                let txt = resp.text().await.unwrap_or_default();
                let is_transient = matches!(status, 408 | 429 | 500 | 502 | 503 | 504);
                let err = TranslatorError::ApiStatus { status, body: txt };
                if is_transient && attempt == 0 {
                    last_err = Some(err);
                    continue;
                }
                return Err(err);
            }
            let val: Value = match resp.json().await {
                Ok(v) => v,
                Err(e) => {
                    last_err = Some(TranslatorError::Http(e));
                    continue;
                }
            };
            let content = val["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            // Если ответ вообще не похож на JSON-объект и содержит цензурный отказ — сигнализируем Refused без размыкания circuit breaker!
            if (!content.contains('{') || !content.contains('}')) && is_refusal_response(&content) {
                return Err(TranslatorError::Refused);
            }
            return Ok(content);
        }
        Err(last_err.unwrap_or(TranslatorError::UnavailableOrRefused))
    }

    async fn call_ollama_chat(
        &self,
        model: &str,
        sys_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, TranslatorError> {
        let endpoint = format!("{}/api/chat", self.ollama_base_url);
        let body = json!({
            "model": model,
            "stream": false,
            "think": false,
            "format": "json",
            "options": {
                "temperature": 0.25,
                "num_ctx": 8192,
                "num_predict": 2560
            },
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        });
        let resp = self.http_local.post(endpoint).json(&body).send().await?;
        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(TranslatorError::ApiStatus { status, body: txt });
        }
        let val: Value = resp.json().await?;
        Ok(val["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    /// Проверяет локальную Ollama и при необходимости автоматически поднимает `ollama serve` в фоне.
    pub async fn ensure_ollama_models(&self) -> Vec<String> {
        let initial = self.list_ollama_models().await;
        if !initial.is_empty() {
            return initial;
        }
        // Если порт 11434 вообще не отвечает, пробуем тихо запустить фоновый демон `ollama serve`
        let url = format!("{}/api/tags", self.ollama_base_url);
        let reachable = self
            .http_local
            .get(&url)
            .timeout(Duration::from_millis(800))
            .send()
            .await
            .is_ok();
        if reachable {
            return initial;
        }

        let mut cmd = std::process::Command::new("ollama");
        cmd.arg("serve")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        if cmd.spawn().is_ok() {
            for _ in 0..10 {
                tokio::time::sleep(Duration::from_millis(450)).await;
                let tags = self.list_ollama_models().await;
                if !tags.is_empty() {
                    return tags;
                }
            }
        }
        self.list_ollama_models().await
    }

    fn resolve_cloud_model(requested: &str) -> &str {
        let (prefix, bare) = strip_model_prefix(requested);
        if matches!(prefix, Some("ollama")) || bare.is_empty() || bare.eq_ignore_ascii_case("auto") {
            "deepseek-v4-flash"
        } else {
            bare
        }
    }

    fn pick_best_ollama_tag(tags: &[String], preferred_fallback: &str) -> Option<String> {
        let (_, pref_bare) = strip_model_prefix(preferred_fallback);
        if !pref_bare.is_empty() && !pref_bare.eq_ignore_ascii_case("auto") {
            if let Some(exact) = tags.iter().find(|t| t.eq_ignore_ascii_case(pref_bare)) {
                return Some(exact.clone());
            }
            let short_pref = pref_bare.split('/').next_back().unwrap_or(pref_bare);
            if !short_pref.is_empty() {
                if let Some(hit) = tags
                    .iter()
                    .find(|t| t.to_ascii_lowercase().contains(&short_pref.to_ascii_lowercase()))
                {
                    return Some(hit.clone());
                }
            }
        }
        for pref in [
            "abliterated",
            "uncensored",
            "heretic",
            "shisa",
            "gemma4",
            "translategemma",
            "gemma3",
            "qwen3.5",
            "qwen3.6",
            "qwen3",
            "sugoi",
            "aya",
        ] {
            if let Some(hit) = tags.iter().find(|t| {
                let low = t.to_ascii_lowercase();
                low.contains(pref) && !low.contains("-vl") && !low.contains("glm-ocr")
            }) {
                return Some(hit.clone());
            }
        }
        tags.iter()
            .find(|t| {
                let low = t.to_ascii_lowercase();
                !low.contains("-vl") && !low.contains("glm-ocr")
            })
            .or_else(|| tags.first())
            .cloned()
    }

    fn is_pure_kana_ascii(text: &str) -> bool {
        text.chars().all(|c| {
            let cp = c as u32;
            (0x3040..=0x30FF).contains(&cp)
                || c.is_ascii()
                || matches!(c, '。' | '、' | '！' | '？' | '「' | '」' | '・' | '…')
        })
    }

    /// Переводит список блоков с детекцией языка, SQLite-кэшем, батчингом и Cloud->Ollama фоллбэком.
    pub async fn translate_blocks(
        &self,
        req: TranslateBatchRequest,
        db: Option<&StudioDb>,
    ) -> Result<TranslateBatchResult, TranslatorError> {
        let t0 = Instant::now();
        let mut blocks = req.blocks;
        let t_lang = req.target_lang.trim().to_ascii_lowercase();
        let all_texts: Vec<String> = blocks
            .iter()
            .filter(|b| !b.is_deleted && !b.original_text.trim().is_empty())
            .map(|b| b.original_text.clone())
            .collect();
        let detected_lang = detect_source_language(&all_texts).to_string();

        let (req_prefix, req_bare) = strip_model_prefix(&req.model);
        let is_explicit_ollama = matches!(req_prefix, Some("ollama"));
        let has_cloud =
            !is_explicit_ollama && !req.cloud_url.trim().is_empty() && !self.cloud_breaker.is_open();
        let cloud_model = Self::resolve_cloud_model(&req.model).to_string();
        let mut active_model = if has_cloud {
            cloud_model.clone()
        } else if !req_bare.is_empty() && !req_bare.eq_ignore_ascii_case("auto") && !req_bare.starts_with("deepseek") {
            req_bare.to_string()
        } else {
            "sugoi-14b-ultra".to_string()
        };

        let mut cache_hits = 0usize;
        let mut api_calls = 0usize;
        let mut censored_fallback = false;
        let mut missing_indices: Vec<usize> = Vec::new();

        for (idx, blk) in blocks.iter_mut().enumerate() {
            if blk.is_deleted {
                continue;
            }
            let src = blk.original_text.trim().to_string();
            if src.is_empty() {
                continue;
            }

            // Всегда заполняем быстрый romaji для японской каны, если поле пустое
            if blk.romaji_text.trim().is_empty() && Self::is_pure_kana_ascii(&src) {
                blk.romaji_text = kana_to_romaji(&src);
            }

            if t_lang == "romaji" && Self::is_pure_kana_ascii(&src) {
                blk.translated_text = kana_to_romaji(&src);
                blk.romaji_text = blk.translated_text.clone();
                cache_hits += 1;
                continue;
            }

            if req.use_cache {
                if let Some(store) = db {
                    let ckey =
                        compute_translation_cache_key(&src, &t_lang, &active_model, &req.glossary);
                    if let Ok(Some(cached)) = store.get_cached_translation(&ckey) {
                        if !cached.trim().is_empty() {
                            blk.translated_text = cached;
                            cache_hits += 1;
                            continue;
                        }
                    }
                }
            }

            missing_indices.push(idx);
        }

        let mut untranslated_count = 0usize;
        if !missing_indices.is_empty() {
            // Внутрипачковая дедупликация одинаковых фраз (например, на CG-вариациях одной сцены):
            // отправляем в LLM только уникальные реплики с минимальным бюджетом символов и раздаём ответ всем дублям.
            struct UniqueBatchItem {
                src: String,
                budget: Option<usize>,
                block_indices: Vec<usize>,
            }
            let mut unique_items: Vec<UniqueBatchItem> = Vec::new();
            let mut unique_pos: HashMap<String, usize> = HashMap::new();
            for &b_idx in &missing_indices {
                let b = &blocks[b_idx];
                let src = b.original_text.trim().to_string();
                let bw = (b.bbox[2] - b.bbox[0]).max(20) as usize;
                let bh = (b.bbox[3] - b.bbox[1]).max(20) as usize;
                let raw_budget = (bw * bh) / 380;
                let blk_budget = if bw < 78 && bh >= bw * 2 {
                    raw_budget.clamp(14, 48)
                } else {
                    raw_budget.clamp(18, 95)
                };
                if let Some(&pos) = unique_pos.get(&src) {
                    unique_items[pos].block_indices.push(b_idx);
                    unique_items[pos].budget = Some(
                        unique_items[pos]
                            .budget
                            .map_or(blk_budget, |prev| prev.min(blk_budget)),
                    );
                } else {
                    unique_pos.insert(src.clone(), unique_items.len());
                    unique_items.push(UniqueBatchItem {
                        src,
                        budget: Some(blk_budget),
                        block_indices: vec![b_idx],
                    });
                }
            }

            let mut rolling_ctx = req.context_lines.clone();
            let chunk_size = 30usize;
            let total_chunks = unique_items.len().div_ceil(chunk_size).max(1);
            for (chunk_idx, chunk) in unique_items.chunks(chunk_size).enumerate() {
                let mut items = Vec::with_capacity(chunk.len());
                let mut expected_keys = Vec::with_capacity(chunk.len());
                for (pos, u_item) in chunk.iter().enumerate() {
                    let k = pos.to_string();
                    expected_keys.push(k.clone());
                    items.push((k, u_item.src.clone(), u_item.budget));
                }

                let (sys_p, user_p) = build_translation_prompts(
                    &t_lang,
                    &items,
                    &req.glossary,
                    &rolling_ctx,
                    req.anti_censorship,
                );

                let allow_local = is_explicit_ollama || !req.cloud_only_no_local_fallback;
                let bypass_censored_cloud = allow_local && self.censorship_breaker.is_open();
                if bypass_censored_cloud {
                    censored_fallback = true;
                }

                let mut parsed_map: HashMap<String, (String, String)> = HashMap::new();
                let mut cloud_got_some = false;
                let mut cloud_censored_chunk = false;
                if has_cloud && !self.cloud_breaker.is_open() && !bypass_censored_cloud {
                    api_calls += 1;
                    match self
                        .call_cloud_openai(
                            &req.cloud_url,
                            &req.cloud_api_key,
                            &cloud_model,
                            &sys_p,
                            &user_p,
                        )
                        .await
                    {
                        Ok(raw) => {
                            let m = parse_flat_json_translations_for_lang(
                                &raw,
                                &expected_keys,
                                &t_lang,
                                false,
                            );
                            let raw_has_refusal = is_refusal_response(&raw);
                            if !m.is_empty() {
                                cloud_got_some = true;
                                self.cloud_breaker.record_success();
                                if m.len() == expected_keys.len() {
                                    self.censorship_breaker.record_success();
                                } else {
                                    if raw_has_refusal {
                                        cloud_censored_chunk = true;
                                    }
                                    if allow_local {
                                        // Облако вырезало часть 18+ реплик в батче
                                        censored_fallback = true;
                                        self.censorship_breaker.record_failure();
                                    }
                                }
                                active_model = cloud_model.clone();
                                for (k, v) in m {
                                    parsed_map.insert(k, (v, cloud_model.clone()));
                                }
                            } else if !raw_has_refusal {
                                self.cloud_breaker.record_failure();
                            } else {
                                cloud_censored_chunk = true;
                                censored_fallback = true;
                                if allow_local && self.censorship_breaker.record_failure() {
                                    tracing::info!(
                                        "Облако ({}) зацензурило 2 страницы подряд — включаю прямой 18+ режим через локальную ИИ на 3 мин",
                                        cloud_model
                                    );
                                } else {
                                    tracing::info!(
                                        "Облако ({}) зацензурило батч — мгновенное переключение на локальную модель",
                                        cloud_model
                                    );
                                }
                            }
                        }
                        Err(TranslatorError::Refused) => {
                            cloud_censored_chunk = true;
                            censored_fallback = true;
                            if allow_local && self.censorship_breaker.record_failure() {
                                tracing::info!(
                                    "Облако ({}) вернуло 2 отказа цензуры подряд — прямой переход на локальную ИИ на 3 мин",
                                    cloud_model
                                );
                            } else {
                                tracing::info!(
                                    "Облако ({}) вернуло отказ цензуры — переключаемся на локальную модель",
                                    cloud_model
                                );
                            }
                        }
                        Err(err) => {
                            tracing::warn!("Облачный переводчик не ответил: {}", err);
                            self.cloud_breaker.record_failure();
                        }
                    }
                }

                // Кешируем резолв локальной модели на уровне чанка, чтобы не дергать Ollama повторно при отсутствии моделей
                let mut ollama_model_resolved: Option<Option<String>> = None;

                // Шаг 2: Если все или часть реплик не переведены (например, облако зацензурило отдельные 18+ фразы в батче),
                // мгновенно допереводим недостающие ключи через локальную модель!
                if parsed_map.len() < expected_keys.len()
                    && allow_local
                    && !self.ollama_breaker.is_open()
                {
                    let ollama_model_opt = if is_explicit_ollama && !req_bare.is_empty() {
                        let _ = self.ensure_ollama_models().await;
                        Some(req_bare.to_string())
                    } else {
                        let tags = self.ensure_ollama_models().await;
                        Self::pick_best_ollama_tag(&tags, &req.fallback_model)
                    };
                    ollama_model_resolved = Some(ollama_model_opt.clone());

                    if let Some(ollama_model) = ollama_model_opt {
                        let mut fb_items = Vec::new();
                        let mut fb_keys = Vec::new();
                        for (k, src, budget) in &items {
                            if !parsed_map.contains_key(k) {
                                fb_keys.push(k.clone());
                                fb_items.push((k.clone(), src.clone(), *budget));
                            }
                        }
                        if !fb_items.is_empty() {
                            let (fb_sys, fb_user) = build_translation_prompts(
                                &t_lang,
                                &fb_items,
                                &req.glossary,
                                &rolling_ctx,
                                true,
                            );
                            api_calls += 1;
                            match self
                                .call_ollama_chat(&ollama_model, &fb_sys, &fb_user)
                                .await
                            {
                                Ok(raw) => {
                                    let m = parse_flat_json_translations_for_lang(
                                        &raw,
                                        &fb_keys,
                                        &t_lang,
                                        false,
                                    );
                                    if !m.is_empty() {
                                        self.ollama_breaker.record_success();
                                        active_model = ollama_model.clone();
                                        for (k, v) in m {
                                            parsed_map.insert(k, (v, ollama_model.clone()));
                                        }
                                    } else {
                                        self.ollama_breaker.record_failure();
                                    }
                                }
                                Err(err) => {
                                    tracing::warn!("Ollama ({}) не ответила: {}", ollama_model, err);
                                    self.ollama_breaker.record_failure();
                                }
                            }
                        }
                    }
                }

                // Шаг 3 (micro_retry): если после основного прохода и фоллбэка часть ключей всё ещё пропала
                // (обрыв JSON на локальной модели или единичный пропуск/утечка алфавита), добиваем микро-пачками по 6 фраз.
                if parsed_map.len() < expected_keys.len() {
                    let missing_items: Vec<(String, String, Option<usize>)> = items
                        .iter()
                        .filter(|(k, _, _)| !parsed_map.contains_key(k))
                        .cloned()
                        .collect();

                    for sub in missing_items.chunks(6) {
                        let sub_keys: Vec<String> = sub.iter().map(|(k, _, _)| k.clone()).collect();
                        if has_cloud
                            && !self.cloud_breaker.is_open()
                            && !bypass_censored_cloud
                            && !cloud_censored_chunk
                            && cloud_got_some
                        {
                            let (mr_sys, mr_user) = build_translation_prompts(
                                &t_lang,
                                sub,
                                &req.glossary,
                                &rolling_ctx,
                                req.anti_censorship,
                            );
                            api_calls += 1;
                            match self
                                .call_cloud_openai(
                                    &req.cloud_url,
                                    &req.cloud_api_key,
                                    &cloud_model,
                                    &mr_sys,
                                    &mr_user,
                                )
                                .await
                            {
                                Ok(raw) => {
                                    let m = parse_flat_json_translations_for_lang(
                                        &raw,
                                        &sub_keys,
                                        &t_lang,
                                        false,
                                    );
                                    if !m.is_empty() {
                                        self.cloud_breaker.record_success();
                                        for (k, v) in m {
                                            parsed_map.insert(k, (v, cloud_model.clone()));
                                        }
                                    } else if is_refusal_response(&raw) {
                                        censored_fallback = true;
                                    }
                                }
                                Err(TranslatorError::Refused) => {
                                    censored_fallback = true;
                                }
                                Err(err) => {
                                    tracing::warn!("Облачный micro_retry не ответил: {}", err);
                                    self.cloud_breaker.record_failure();
                                }
                            }
                        }

                        let still_missing: Vec<(String, String, Option<usize>)> = sub
                            .iter()
                            .filter(|(k, _, _)| !parsed_map.contains_key(k))
                            .cloned()
                            .collect();
                        if !still_missing.is_empty()
                            && allow_local
                            && !self.ollama_breaker.is_open()
                        {
                            let ollama_model_opt = match ollama_model_resolved {
                                Some(ref cached) => cached.clone(),
                                None => {
                                    let resolved = if is_explicit_ollama && !req_bare.is_empty() {
                                        let _ = self.ensure_ollama_models().await;
                                        Some(req_bare.to_string())
                                    } else {
                                        let tags = self.ensure_ollama_models().await;
                                        Self::pick_best_ollama_tag(&tags, &req.fallback_model)
                                    };
                                    ollama_model_resolved = Some(resolved.clone());
                                    resolved
                                }
                            };

                            if let Some(ollama_model) = ollama_model_opt {
                                let still_keys: Vec<String> =
                                    still_missing.iter().map(|(k, _, _)| k.clone()).collect();
                                let (fb_sys, fb_user) = build_translation_prompts(
                                    &t_lang,
                                    &still_missing,
                                    &req.glossary,
                                    &rolling_ctx,
                                    true,
                                );
                                api_calls += 1;
                                match self
                                    .call_ollama_chat(&ollama_model, &fb_sys, &fb_user)
                                    .await
                                {
                                    Ok(raw) => {
                                        let mut m = parse_flat_json_translations_for_lang(
                                            &raw,
                                            &still_keys,
                                            &t_lang,
                                            false,
                                        );
                                        for k in &still_keys {
                                            if !m.contains_key(k) {
                                                let rec = parse_flat_json_translations_for_lang(
                                                    &raw,
                                                    std::slice::from_ref(k),
                                                    &t_lang,
                                                    true,
                                                );
                                                if let Some((rk, rv)) = rec.into_iter().next() {
                                                    m.insert(rk, rv);
                                                }
                                            }
                                        }
                                        if !m.is_empty() {
                                            self.ollama_breaker.record_success();
                                            active_model = ollama_model.clone();
                                            for (k, v) in m {
                                                parsed_map.insert(k, (v, ollama_model.clone()));
                                            }
                                        } else {
                                            self.ollama_breaker.record_failure();
                                        }
                                    }
                                    Err(err) => {
                                        tracing::warn!(
                                            "Ollama micro_retry ({}) не ответила: {}",
                                            ollama_model,
                                            err
                                        );
                                        self.ollama_breaker.record_failure();
                                    }
                                }
                            }
                        }
                    }
                }

                let chunk_got = parsed_map.len();
                let chunk_exp = expected_keys.len();
                if chunk_got < chunk_exp {
                    tracing::warn!(
                        "Чанк {}/{} ({}): переведено {}/{} уникальных фраз (активная модель: {})",
                        chunk_idx + 1,
                        total_chunks,
                        t_lang,
                        chunk_got,
                        chunk_exp,
                        active_model
                    );
                } else {
                    tracing::info!(
                        "Чанк {}/{} ({}): переведено {}/{} уникальных фраз (модель: {})",
                        chunk_idx + 1,
                        total_chunks,
                        t_lang,
                        chunk_got,
                        chunk_exp,
                        active_model
                    );
                }

                for (pos, u_item) in chunk.iter().enumerate() {
                    let k = pos.to_string();
                    if let Some((tr, used_model)) = parsed_map.get(&k) {
                        let final_tr = restore_source_ellipses(&u_item.src, tr);
                        for &b_idx in &u_item.block_indices {
                            blocks[b_idx].translated_text = final_tr.clone();
                        }
                        if !final_tr.trim().is_empty() {
                            rolling_ctx.push(format!("{} => {}", u_item.src, final_tr));
                            if rolling_ctx.len() > 12 {
                                let d = rolling_ctx.len() - 12;
                                rolling_ctx.drain(0..d);
                            }
                        }
                        if let Some(store) = db {
                            // Кэшируем и под запрошенной моделью (если сработал фоллбэк на цензуру, чтобы повторно не биться в облако), и под самой моделью
                            for cache_m in [&active_model, used_model, &cloud_model] {
                                if cache_m.trim().is_empty() {
                                    continue;
                                }
                                let ckey = compute_translation_cache_key(
                                    &u_item.src,
                                    &t_lang,
                                    cache_m,
                                    &req.glossary,
                                );
                                let _ = store.put_cached_translation(
                                    &ckey,
                                    &u_item.src,
                                    &t_lang,
                                    &final_tr,
                                    used_model,
                                );
                            }
                        }
                    } else if t_lang == "romaji" {
                        let rom = kana_to_romaji(&u_item.src);
                        for &b_idx in &u_item.block_indices {
                            blocks[b_idx].translated_text = rom.clone();
                        }
                    }
                }
            }

            // Защита от пустых белых страниц со статусом «Готово»:
            // если ни один из запрошенных блоков на странице так и не получил перевод, возвращаем явную ошибку!
            let translated_now = missing_indices
                .iter()
                .filter(|&&i| !blocks[i].translated_text.trim().is_empty())
                .count();
            untranslated_count = missing_indices.len().saturating_sub(translated_now);
            if translated_now == 0 && t_lang != "romaji" {
                return Err(TranslatorError::UnavailableOrRefused);
            }
        }

        Ok(TranslateBatchResult {
            blocks,
            active_model,
            detected_lang,
            cache_hits,
            api_calls,
            elapsed_ms: t0.elapsed().as_millis(),
            censored_fallback,
            untranslated_count,
        })
    }

    pub async fn translate_single_text(
        &self,
        text: &str,
        target_lang: &str,
        model: &str,
        cloud_url: &str,
        cloud_api_key: &str,
        glossary: &[GlossaryEntry],
        anti_censorship: bool,
        db: Option<&StudioDb>,
    ) -> Result<String, TranslatorError> {
        let blk = TextBlock {
            id: 1,
            bbox: [0, 0, 200, 120],
            original_text: text.to_string(),
            ..Default::default()
        };
        let res = self
            .translate_blocks(
                TranslateBatchRequest {
                    blocks: vec![blk],
                    target_lang: target_lang.to_string(),
                    model: model.to_string(),
                    cloud_url: cloud_url.to_string(),
                    cloud_api_key: cloud_api_key.to_string(),
                    glossary: glossary.to_vec(),
                    context_lines: Vec::new(),
                    anti_censorship,
                    use_cache: true,
                    ..Default::default()
                },
                db,
            )
            .await?;
        let translated = res
            .blocks
            .into_iter()
            .next()
            .map(|b| b.translated_text)
            .unwrap_or_default();
        if translated.trim().is_empty() && !text.trim().is_empty() {
            return Err(TranslatorError::UnavailableOrRefused);
        }
        Ok(translated)
    }
}
