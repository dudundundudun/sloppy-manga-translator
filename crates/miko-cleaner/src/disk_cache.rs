use image::{ImageFormat, RgbImage};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Гибридный RAM LRU + lossless WebP дисковый кэш очищенных страниц (чистый Rust `image-webp`).
pub struct CleanDiskCache {
    cache_dir: PathBuf,
    max_ram_entries: usize,
    max_disk_files: usize,
    ram: VecDeque<(String, RgbImage)>,
}

impl CleanDiskCache {
    pub fn new(cache_dir: PathBuf, max_ram_entries: usize, max_disk_files: usize) -> Self {
        Self {
            cache_dir,
            max_ram_entries: max_ram_entries.max(1),
            max_disk_files: max_disk_files.max(1),
            ram: VecDeque::new(),
        }
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn clear_ram(&mut self) {
        self.ram.clear();
    }

    fn entry_path(&self, key: &str) -> PathBuf {
        let safe: String = key
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        self.cache_dir.join(format!("clean_{safe}.webp"))
    }

    fn insert_ram(&mut self, key: String, img: RgbImage) {
        if let Some(pos) = self.ram.iter().position(|(k, _)| k == &key) {
            self.ram.remove(pos);
        }
        self.ram.push_back((key, img));
        while self.ram.len() > self.max_ram_entries {
            self.ram.pop_front();
        }
    }

    pub fn get(&mut self, key: &str) -> Option<RgbImage> {
        if let Some(pos) = self.ram.iter().position(|(k, _)| k == key) {
            let (k, img) = self.ram.remove(pos)?;
            let cloned = img.clone();
            self.ram.push_back((k, img));
            return Some(cloned);
        }

        let path = self.entry_path(key);
        if !path.is_file() {
            return None;
        }
        let img = image::open(&path).ok()?.to_rgb8();
        self.insert_ram(key.to_string(), img.clone());
        Some(img)
    }

    pub fn put(&mut self, key: &str, img: &RgbImage) -> std::io::Result<()> {
        self.insert_ram(key.to_string(), img.clone());
        std::fs::create_dir_all(&self.cache_dir)?;

        let final_path = self.entry_path(key);
        let tmp_path = self.cache_dir.join(format!("clean_{key}.tmp.webp"));
        img.save_with_format(&tmp_path, ImageFormat::WebP)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        if final_path.exists() {
            let _ = std::fs::remove_file(&final_path);
        }
        std::fs::rename(&tmp_path, &final_path)?;

        self.prune_disk();
        Ok(())
    }

    fn prune_disk(&self) {
        let Ok(rd) = std::fs::read_dir(&self.cache_dir) else {
            return;
        };
        let mut files: Vec<(SystemTime, PathBuf)> = rd
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                if !p.is_file() {
                    return None;
                }
                let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
                if !ext.eq_ignore_ascii_case("webp") {
                    return None;
                }
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if name.ends_with(".tmp.webp") || !name.starts_with("clean_") {
                    return None;
                }
                let mtime = e
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                Some((mtime, p))
            })
            .collect();

        if files.len() <= self.max_disk_files {
            return;
        }
        files.sort_by_key(|(t, _)| *t);
        let to_remove = files.len() - self.max_disk_files;
        for (_, p) in files.into_iter().take(to_remove) {
            let _ = std::fs::remove_file(p);
        }
    }
}
