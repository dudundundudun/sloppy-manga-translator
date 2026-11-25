pub mod models;
pub mod security;
pub mod storage;

pub use models::{
    normalize_clean_bbox, AppConfig, FolderJobState, GlossaryEntry, MaskStroke, TextBlock,
};
pub use security::{
    is_blocked_ip, is_filesystem_root, is_trusted_host_header, is_trusted_local_origin,
    is_trusted_referer, mask_api_key, reject_unc_path, AllowedRoots, SecurityError,
    PRIVILEGED_ENDPOINTS,
};
pub use storage::{StorageError, StudioDb};

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
    use std::path::Path;

    #[test]
    fn test_security_guards_and_masking() {
        assert_eq!(mask_api_key(Some("sk-1234567890abcd")), "sk-12...abcd");
        assert_eq!(mask_api_key(Some("short")), "***");
        assert_eq!(mask_api_key(Some("  ")), "");

        assert!(is_trusted_local_origin(Some("http://127.0.0.1:8000")));
        assert!(is_trusted_local_origin(Some("chrome-extension://abcdef123")));
        assert!(!is_trusted_local_origin(Some("https://evil.example.com")));
        assert!(!is_trusted_local_origin(Some("null")));

        assert!(is_trusted_host_header(Some("localhost:8000")));
        assert!(is_trusted_host_header(Some("[::1]:8000")));
        assert!(!is_trusted_host_header(Some("evil.com:8000")));

        assert!(is_trusted_referer(Some("http://127.0.0.1:8000/ui")));
        assert!(!is_trusted_referer(Some("http://attacker.local/page")));

        assert!(reject_unc_path(Some("//server/share"), "p").is_err());
        assert!(reject_unc_path(Some("\\\\server\\share"), "p").is_err());
        assert!(reject_unc_path(Some("abc\0def"), "p").is_err());
        assert!(reject_unc_path(Some("C:\\Manga\\ch01"), "p").is_ok());

        assert!(is_blocked_ip(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))));
        assert!(is_blocked_ip(IpAddr::V4(Ipv4Addr::new(100, 64, 1, 2))));
        assert!(is_blocked_ip(IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254))));
        assert!(is_blocked_ip(IpAddr::V6(Ipv6Addr::LOCALHOST)));
        assert!(!is_blocked_ip(IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))));
    }

    #[test]
    fn test_normalize_clean_bbox_preserves_polygon_on_drag() {
        let poly = vec![[10, 10], [80, 10], [80, 80], [10, 80]];
        let mut blk = TextBlock {
            id: 1,
            bbox: [25, 30, 95, 100],
            clean_bbox: Some([10, 10, 80, 80]),
            bubble_polygon: Some(poly.clone()),
            mask_dilate: Some(12),
            ..Default::default()
        };
        normalize_clean_bbox(&mut blk);
        assert!(blk.bbox_modified);
        assert!(!blk.clean_bbox_modified);
        assert_eq!(blk.bubble_polygon, Some(poly));

        blk.clean_bbox_modified = true;
        normalize_clean_bbox(&mut blk);
        assert!(blk.bubble_polygon.is_none());
    }

    #[test]
    fn test_sqlite_wal_storage_roundtrip() {
        let db = StudioDb::open_in_memory().expect("db");
        db.put_cached_translation("k1", "こんにちは", "ru", "Привет", "grok-4.7")
            .expect("put");
        assert_eq!(
            db.get_cached_translation("k1").expect("get"),
            Some("Привет".to_string())
        );
        db.replace_glossary(
            "global",
            &[GlossaryEntry {
                src: "美咲".into(),
                dst: "Мисаки".into(),
                note: "героиня".into(),
            }],
        )
        .expect("glossary");
        assert_eq!(db.load_glossary("global").expect("load").len(), 1);
        db.save_page_state("p1", "hash123", &[TextBlock::default()])
            .expect("save page");
        let loaded = db.load_page_state("p1").expect("load page").unwrap();
        assert_eq!(loaded.0, "hash123");
        assert_eq!(loaded.1.len(), 1);
        let _ = Path::new(".");
    }
}
