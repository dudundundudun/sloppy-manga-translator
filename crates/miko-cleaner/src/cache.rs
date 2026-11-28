use miko_core::TextBlock;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Serialize)]
struct CleanHashBlockSpec {
    coords: [i32; 4],
    is_free: bool,
    mask_dilate: u8,
    poly_sig: u64,
    strokes: Vec<(i32, i32, i32, String)>,
}

#[derive(Serialize)]
struct CleanHashPayload<'a> {
    src_file: &'a str,
    mtime: i64,
    dilate_size: u8,
    inpaint_backend: &'a str,
    pipeline_mode: &'a str,
    clean_ver: &'a str,
    active_clean_blocks: Vec<CleanHashBlockSpec>,
}

fn effective_clean_bbox(block: &TextBlock) -> [i32; 4] {
    let mut cb = block.clean_bbox.unwrap_or(block.bbox);
    if !block.clean_bbox_modified {
        if let Some(tb) = block.text_bbox {
            if (tb[2] - tb[0]) >= 8 && (tb[3] - tb[1]) >= 8 {
                cb[0] = cb[0].min(tb[0]);
                cb[1] = cb[1].min(tb[1]);
                cb[2] = cb[2].max(tb[2]);
                cb[3] = cb[3].max(tb[3]);
            }
        }
    }
    cb
}

fn block_poly_sig(block: &TextBlock) -> u64 {
    if block.is_free_text || block.clean_bbox_modified {
        return 0;
    }
    let Some(ref poly) = block.bubble_polygon else {
        return 0;
    };
    if poly.len() < 3 {
        return 0;
    }
    let p0 = poly[0];
    let pm = poly[poly.len() / 2];
    ((poly.len() as u64) << 16) ^ (((p0[0] & 0xFF) as u64) << 8) ^ ((pm[1] & 0xFF) as u64)
}

fn is_block_skipped(block: &TextBlock, skip_untranslated: bool) -> bool {
    if block.is_deleted
        || (block.translated_text.trim().is_empty() && block.original_text.trim().is_empty())
    {
        return true;
    }
    if skip_untranslated && block.translated_text.trim().is_empty() {
        let has_strokes = block
            .custom_mask_strokes
            .as_ref()
            .is_some_and(|s| !s.is_empty());
        if !has_strokes {
            return true;
        }
    }
    false
}

/// Вычисляет детерминированный SHA-256 ключ CleanCache (инвариантен к сдвигу рамки перевода `bbox`).
pub fn compute_clean_state_hash(
    blocks: &[TextBlock],
    src_file: &str,
    src_mtime: i64,
    dilate_size: u8,
    inpaint_backend: &str,
    pipeline_mode: &str,
) -> String {
    compute_clean_state_hash_opts(
        blocks,
        src_file,
        src_mtime,
        dilate_size,
        inpaint_backend,
        pipeline_mode,
        false,
    )
}

/// Вычисляет SHA-256 ключ CleanCache с явным флагом `skip_untranslated`,
/// чтобы допереведённые позже бабблы автоматически сбрасывали старый кэш.
pub fn compute_clean_state_hash_opts(
    blocks: &[TextBlock],
    src_file: &str,
    src_mtime: i64,
    dilate_size: u8,
    inpaint_backend: &str,
    pipeline_mode: &str,
    skip_untranslated: bool,
) -> String {
    let mut specs: Vec<CleanHashBlockSpec> = Vec::new();
    for blk in blocks {
        if is_block_skipped(blk, skip_untranslated) {
            continue;
        }
        let coords = effective_clean_bbox(blk);
        let mut strokes: Vec<(i32, i32, i32, String)> = blk
            .custom_mask_strokes
            .as_ref()
            .map(|list| {
                list.iter()
                    .map(|s| {
                        (
                            s.x,
                            s.y,
                            s.r.clamp(1, 80),
                            if s.mode.eq_ignore_ascii_case("erase") {
                                "erase".to_string()
                            } else {
                                "add".to_string()
                            },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        strokes.sort();
        specs.push(CleanHashBlockSpec {
            coords,
            is_free: blk.is_free_text,
            mask_dilate: blk.mask_dilate.unwrap_or(0),
            poly_sig: block_poly_sig(blk),
            strokes,
        });
    }

    specs.sort_by(|a, b| {
        a.coords
            .cmp(&b.coords)
            .then(a.is_free.cmp(&b.is_free))
            .then(a.mask_dilate.cmp(&b.mask_dilate))
            .then(a.poly_sig.cmp(&b.poly_sig))
            .then(a.strokes.cmp(&b.strokes))
    });

    let norm_src = src_file.replace('\\', "/").to_ascii_lowercase();
    let payload = CleanHashPayload {
        src_file: &norm_src,
        mtime: src_mtime,
        dilate_size,
        inpaint_backend,
        pipeline_mode,
        clean_ver: "v7.0-rust-clean",
        active_clean_blocks: specs,
    };
    let encoded = serde_json::to_vec(&payload).unwrap_or_default();
    let digest = Sha256::digest(&encoded);
    hex::encode(digest)[..24].to_string()
}
