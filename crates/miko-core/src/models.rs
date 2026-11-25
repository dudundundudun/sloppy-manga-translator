use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MaskStroke {
    pub x: i32,
    pub y: i32,
    pub r: i32,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    pub id: i64,
    pub bbox: [i32; 4],
    #[serde(default)]
    pub clean_bbox: Option<[i32; 4]>,
    #[serde(default)]
    pub text_bbox: Option<[i32; 4]>,
    #[serde(default)]
    pub bubble_polygon: Option<Vec<[i32; 2]>>,
    #[serde(default)]
    pub bbox_modified: bool,
    #[serde(default)]
    pub clean_bbox_modified: bool,
    #[serde(default)]
    pub bbox_resized: bool,
    #[serde(default)]
    pub is_merged_bubble: bool,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default)]
    pub original_text: String,
    #[serde(default)]
    pub romaji_text: String,
    #[serde(default)]
    pub translated_text: String,
    #[serde(default)]
    pub is_free_text: bool,
    #[serde(default)]
    pub is_deleted: bool,
    #[serde(default)]
    pub font_size: Option<i32>,
    #[serde(default)]
    pub rendered_font_size: Option<i32>,
    #[serde(default)]
    pub font_family: Option<String>,
    #[serde(default)]
    pub text_color: Option<String>,
    #[serde(default)]
    pub stroke_color: Option<String>,
    #[serde(default)]
    pub stroke_width: Option<i32>,
    #[serde(default = "default_align")]
    pub align: String,
    #[serde(default)]
    pub angle: f32,
    #[serde(default = "default_style_preset")]
    pub style_preset: String,
    #[serde(default = "default_bg_style")]
    pub bg_style: String,
    #[serde(default)]
    pub bg_color: Option<String>,
    #[serde(default)]
    pub bg_padding: Option<i32>,
    #[serde(default)]
    pub overflow_mode: Option<String>,
    #[serde(default)]
    pub safe_bbox: Option<[i32; 4]>,
    #[serde(default)]
    pub bubble_bbox: Option<[i32; 4]>,
    #[serde(default)]
    pub lir_bbox: Option<[i32; 4]>,
    #[serde(default)]
    pub mask_dilate: Option<u8>,
    #[serde(default)]
    pub custom_mask_strokes: Option<Vec<MaskStroke>>,
    #[serde(default)]
    pub skip_reason: Option<String>,
}

fn default_confidence() -> f32 {
    1.0
}

fn default_align() -> String {
    "center".to_string()
}

fn default_style_preset() -> String {
    "dialogue".to_string()
}

fn default_bg_style() -> String {
    "none".to_string()
}

impl Default for TextBlock {
    fn default() -> Self {
        Self {
            id: 1,
            bbox: [0, 0, 0, 0],
            clean_bbox: None,
            text_bbox: None,
            bubble_polygon: None,
            bbox_modified: false,
            clean_bbox_modified: false,
            bbox_resized: false,
            is_merged_bubble: false,
            confidence: 1.0,
            original_text: String::new(),
            romaji_text: String::new(),
            translated_text: String::new(),
            is_free_text: false,
            is_deleted: false,
            font_size: None,
            rendered_font_size: None,
            font_family: None,
            text_color: None,
            stroke_color: None,
            stroke_width: None,
            align: default_align(),
            angle: 0.0,
            style_preset: default_style_preset(),
            bg_style: default_bg_style(),
            bg_color: None,
            bg_padding: None,
            overflow_mode: None,
            safe_bbox: None,
            bubble_bbox: None,
            lir_bbox: None,
            mask_dilate: None,
            custom_mask_strokes: None,
            skip_reason: None,
        }
    }
}

/// Нормализует `clean_bbox`, `bubble_polygon`, `mask_dilate` и флаги модификации,
/// гарантируя, что сдвиг `bbox` не сбрасывает `clean_bbox_modified` и полигон.
pub fn normalize_clean_bbox(block: &mut TextBlock) {
    let cb = match block.clean_bbox {
        Some(c) => c,
        None => block.bbox,
    };
    block.clean_bbox = Some(cb);
    block.bbox_modified = block.bbox_modified || (cb != block.bbox);

    let cw = cb[2] - cb[0];
    let ch = cb[3] - cb[1];
    let nw = block.bbox[2] - block.bbox[0];
    let nh = block.bbox[3] - block.bbox[1];
    block.bbox_resized = block.bbox_resized || (nw - cw).abs() >= 3 || (nh - ch).abs() >= 3;

    if block.is_free_text || block.clean_bbox_modified {
        block.bubble_polygon = None;
    } else if let Some(ref mut poly) = block.bubble_polygon {
        if poly.len() > 256 {
            poly.truncate(256);
        }
        if poly.len() < 3 {
            block.bubble_polygon = None;
        }
    }

    if let Some(d) = block.mask_dilate {
        block.mask_dilate = if d == 0 { None } else { Some(d.clamp(1, 28)) };
    }

    if let Some(ref mut strokes) = block.custom_mask_strokes {
        for s in strokes.iter_mut() {
            s.r = s.r.clamp(1, 80);
            s.mode = if s.mode.trim().eq_ignore_ascii_case("erase") {
                "erase".to_string()
            } else {
                "add".to_string()
            };
        }
        if strokes.is_empty() {
            block.custom_mask_strokes = None;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GlossaryEntry {
    pub src: String,
    pub dst: String,
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub selected_model: String,
    pub unsloth_url: String,
    #[serde(skip_serializing)]
    pub unsloth_api_key: String,
    pub auto_save_web_pages: bool,
    pub auto_export_chapter_book: bool,
    pub web_save_dir: String,
    pub text_filter_mode: String,
    pub skip_sighs: bool,
    pub skip_sfx: bool,
    pub detect_free_text: bool,
    pub min_text_length: u32,
    pub min_font_size: u32,
    pub max_font_size: u32,
    pub uppercase: bool,
    pub overflow_mode: String,
    pub batch_pages: u32,
    pub translation_profile: String,
    pub pipeline_mode: String,
    pub use_image_context: bool,
    pub local_fallback_model_multilang: String,
    pub anti_censorship_prompt: bool,
    pub preflight_nsfw_split: bool,
    pub cloud_only_no_local_fallback: bool,
    pub cloud_fallback_model: String,
    pub ultra_proofread_mode: String,
    pub ui_settings: serde_json::Value,
    pub glossary: Vec<GlossaryEntry>,
}

impl AppConfig {
    pub fn default_with_base(base_dir: &Path) -> Self {
        Self {
            selected_model: "auto".to_string(),
            unsloth_url: String::new(),
            unsloth_api_key: String::new(),
            auto_save_web_pages: true,
            auto_export_chapter_book: false,
            web_save_dir: base_dir.join("Saved_Manga").to_string_lossy().to_string(),
            text_filter_mode: "dialogue_only".to_string(),
            skip_sighs: true,
            skip_sfx: true,
            detect_free_text: false,
            min_text_length: 2,
            min_font_size: 12,
            max_font_size: 32,
            uppercase: false,
            overflow_mode: "shrink".to_string(),
            batch_pages: 15,
            translation_profile: "balanced".to_string(),
            pipeline_mode: "balanced".to_string(),
            use_image_context: false,
            local_fallback_model_multilang: "auto".to_string(),
            anti_censorship_prompt: false,
            preflight_nsfw_split: true,
            cloud_only_no_local_fallback: false,
            cloud_fallback_model: "auto".to_string(),
            ultra_proofread_mode: "selective".to_string(),
            ui_settings: serde_json::json!({}),
            glossary: Vec::new(),
        }
    }

    /// Читает безопасные дефолты из config.json без раскрытия секретных ключей наружу.
    pub fn load_defaults_from_config_json(base_dir: &Path) -> Self {
        let mut cfg = Self::default_with_base(base_dir);
        let cfg_path = base_dir.join("config.json");
        let Ok(raw) = std::fs::read_to_string(cfg_path) else {
            return cfg;
        };
        let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) else {
            return cfg;
        };
        let Some(obj) = val.as_object() else {
            return cfg;
        };

        let pick_str = |k: &str| obj.get(k).and_then(|v| v.as_str()).map(str::to_string);
        let pick_bool = |k: &str| obj.get(k).and_then(|v| v.as_bool());
        let pick_u32 = |k: &str| {
            obj.get(k)
                .and_then(|v| v.as_u64())
                .and_then(|n| u32::try_from(n).ok())
        };

        if let Some(v) = pick_str("selected_model") {
            cfg.selected_model = v;
        }
        if let Some(v) = pick_str("unsloth_url") {
            cfg.unsloth_url = v;
        }
        if let Some(v) = pick_str("unsloth_api_key") {
            cfg.unsloth_api_key = v;
        }
        if let Some(v) = pick_bool("auto_save_web_pages") {
            cfg.auto_save_web_pages = v;
        }
        if let Some(v) = pick_bool("auto_export_chapter_book") {
            cfg.auto_export_chapter_book = v;
        }
        if let Some(v) = pick_str("web_save_dir") {
            if !v.trim().is_empty() {
                cfg.web_save_dir = v;
            }
        }
        if let Some(v) = pick_str("text_filter_mode") {
            cfg.text_filter_mode = v;
        }
        if let Some(v) = pick_bool("skip_sighs") {
            cfg.skip_sighs = v;
        }
        if let Some(v) = pick_bool("skip_sfx") {
            cfg.skip_sfx = v;
        }
        if let Some(v) = pick_bool("detect_free_text") {
            cfg.detect_free_text = v;
        }
        if let Some(v) = pick_u32("min_text_length") {
            cfg.min_text_length = v.clamp(1, 50);
        }
        if let Some(v) = pick_u32("min_font_size") {
            cfg.min_font_size = v.clamp(6, 200);
        }
        if let Some(v) = pick_u32("max_font_size") {
            cfg.max_font_size = v.clamp(8, 300);
        }
        if let Some(v) = pick_bool("uppercase") {
            cfg.uppercase = v;
        }
        if let Some(v) = pick_str("overflow_mode") {
            cfg.overflow_mode = v;
        }
        if let Some(v) = pick_u32("batch_pages") {
            cfg.batch_pages = v.clamp(1, 200);
        }
        if let Some(v) = pick_str("translation_profile") {
            cfg.translation_profile = v;
        }
        if let Some(v) = pick_str("pipeline_mode") {
            cfg.pipeline_mode = v;
        }
        if let Some(v) = pick_bool("use_image_context") {
            cfg.use_image_context = v;
        }
        if let Some(v) = pick_str("local_fallback_model_multilang") {
            cfg.local_fallback_model_multilang = v;
        }
        if let Some(v) = pick_bool("anti_censorship_prompt") {
            cfg.anti_censorship_prompt = v;
        }
        if let Some(v) = pick_bool("preflight_nsfw_split") {
            cfg.preflight_nsfw_split = v;
        }
        if let Some(v) = pick_bool("cloud_only_no_local_fallback") {
            cfg.cloud_only_no_local_fallback = v;
        }
        if let Some(v) = pick_str("cloud_fallback_model") {
            cfg.cloud_fallback_model = v;
        }
        if let Some(v) = pick_str("ultra_proofread_mode") {
            cfg.ultra_proofread_mode = v;
        }
        if let Some(v) = obj.get("ui_settings") {
            if v.is_object() {
                cfg.ui_settings = v.clone();
            }
        }
        if let Some(arr) = obj.get("glossary").and_then(|v| v.as_array()) {
            let mut g_out = Vec::new();
            for item in arr {
                if let Ok(entry) = serde_json::from_value::<GlossaryEntry>(item.clone()) {
                    let src = entry.src.trim().to_string();
                    let dst = entry.dst.trim().to_string();
                    if !src.is_empty() && !dst.is_empty() {
                        g_out.push(GlossaryEntry {
                            src,
                            dst,
                            note: entry.note.trim().to_string(),
                        });
                    }
                }
            }
            cfg.glossary = g_out;
        }
        cfg
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderJobState {
    pub status: String,
    pub total: usize,
    pub done: usize,
    pub current_file: String,
    pub phase: String,
    pub active_model: String,
    pub folder_path: String,
    pub output_dir: String,
    pub message: String,
    pub logs: Vec<String>,
}

impl Default for FolderJobState {
    fn default() -> Self {
        Self {
            status: "idle".to_string(),
            total: 0,
            done: 0,
            current_file: String::new(),
            phase: "idle".to_string(),
            active_model: String::new(),
            folder_path: String::new(),
            output_dir: String::new(),
            message: "Готово к работе (Rust-ядро v7.0)".to_string(),
            logs: Vec::new(),
        }
    }
}
