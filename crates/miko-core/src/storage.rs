use crate::models::{GlossaryEntry, TextBlock};
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Mutex;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Ошибка SQLite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Ошибка JSON сериализации: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Блокировка БД повреждена")]
    LockPoisoned,
}

#[derive(Debug, Clone, Default)]
pub struct StorageMetrics {
    pub translation_cache_rows: usize,
    pub glossary_rows: usize,
    pub page_state_rows: usize,
}

/// Потокобезопасное локальное хранилище состояния, глоссария и кэша переводов на SQLite WAL.
pub struct StudioDb {
    conn: Mutex<Connection>,
}

impl StudioDb {
    pub fn open(db_path: &Path) -> Result<Self, StorageError> {
        if let Some(parent) = db_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(db_path)?;
        Self::init_conn(conn)
    }

    pub fn open_in_memory() -> Result<Self, StorageError> {
        let conn = Connection::open_in_memory()?;
        Self::init_conn(conn)
    }

    fn init_conn(conn: Connection) -> Result<Self, StorageError> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS translation_cache (
                 cache_key TEXT PRIMARY KEY,
                 source_text TEXT NOT NULL,
                 target_lang TEXT NOT NULL,
                 translated_text TEXT NOT NULL,
                 model_id TEXT NOT NULL DEFAULT '',
                 updated_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS glossary_entries (
                 scope TEXT NOT NULL,
                 src TEXT NOT NULL,
                 dst TEXT NOT NULL,
                 note TEXT NOT NULL DEFAULT '',
                 PRIMARY KEY (scope, src)
             );
             CREATE TABLE IF NOT EXISTS page_states (
                 page_key TEXT PRIMARY KEY,
                 clean_state_hash TEXT NOT NULL DEFAULT '',
                 blocks_json TEXT NOT NULL,
                 updated_at INTEGER NOT NULL
             );",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn unix_ts() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }

    pub fn get_cached_translation(&self, cache_key: &str) -> Result<Option<String>, StorageError> {
        let guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = guard.prepare(
            "SELECT translated_text FROM translation_cache WHERE cache_key = ?1",
        )?;
        let mut rows = stmt.query(params![cache_key])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub fn put_cached_translation(
        &self,
        cache_key: &str,
        source_text: &str,
        target_lang: &str,
        translated_text: &str,
        model_id: &str,
    ) -> Result<(), StorageError> {
        let guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        guard.execute(
            "INSERT INTO translation_cache (cache_key, source_text, target_lang, translated_text, model_id, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(cache_key) DO UPDATE SET
               translated_text = excluded.translated_text,
               model_id = excluded.model_id,
               updated_at = excluded.updated_at",
            params![
                cache_key,
                source_text,
                target_lang,
                translated_text,
                model_id,
                Self::unix_ts()
            ],
        )?;
        Ok(())
    }

    pub fn replace_glossary(&self, scope: &str, entries: &[GlossaryEntry]) -> Result<(), StorageError> {
        let mut guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let tx = guard.transaction()?;
        tx.execute("DELETE FROM glossary_entries WHERE scope = ?1", params![scope])?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO glossary_entries (scope, src, dst, note) VALUES (?1, ?2, ?3, ?4)",
            )?;
            for e in entries {
                let s = e.src.trim();
                let d = e.dst.trim();
                if s.is_empty() || d.is_empty() {
                    continue;
                }
                stmt.execute(params![scope, s, d, e.note.trim()])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn load_glossary(&self, scope: &str) -> Result<Vec<GlossaryEntry>, StorageError> {
        let guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = guard.prepare(
            "SELECT src, dst, note FROM glossary_entries WHERE scope = ?1 ORDER BY rowid ASC",
        )?;
        let mut rows = stmt.query(params![scope])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(GlossaryEntry {
                src: row.get(0)?,
                dst: row.get(1)?,
                note: row.get(2)?,
            });
        }
        Ok(out)
    }

    pub fn save_page_state(
        &self,
        page_key: &str,
        clean_state_hash: &str,
        blocks: &[TextBlock],
    ) -> Result<(), StorageError> {
        let blocks_json = serde_json::to_string(blocks)?;
        let guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        guard.execute(
            "INSERT INTO page_states (page_key, clean_state_hash, blocks_json, updated_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(page_key) DO UPDATE SET
               clean_state_hash = excluded.clean_state_hash,
               blocks_json = excluded.blocks_json,
               updated_at = excluded.updated_at",
            params![page_key, clean_state_hash, blocks_json, Self::unix_ts()],
        )?;
        Ok(())
    }

    pub fn load_page_state(&self, page_key: &str) -> Result<Option<(String, Vec<TextBlock>)>, StorageError> {
        let guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = guard.prepare(
            "SELECT clean_state_hash, blocks_json FROM page_states WHERE page_key = ?1",
        )?;
        let mut rows = stmt.query(params![page_key])?;
        if let Some(row) = rows.next()? {
            let hash: String = row.get(0)?;
            let raw_json: String = row.get(1)?;
            let blocks: Vec<TextBlock> = serde_json::from_str(&raw_json)?;
            Ok(Some((hash, blocks)))
        } else {
            Ok(None)
        }
    }

    pub fn purge_refusal_cache<F: Fn(&str) -> bool>(&self, is_bad: F) -> Result<usize, StorageError> {
        let guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let mut bad_keys: Vec<String> = Vec::new();
        {
            let mut stmt =
                guard.prepare("SELECT cache_key, translated_text FROM translation_cache")?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let k: String = row.get(0)?;
                let txt: String = row.get(1)?;
                if txt.trim().is_empty() || is_bad(&txt) {
                    bad_keys.push(k);
                }
            }
        }
        for k in &bad_keys {
            guard.execute("DELETE FROM translation_cache WHERE cache_key = ?1", params![k])?;
        }
        Ok(bad_keys.len())
    }

    pub fn metrics(&self) -> Result<StorageMetrics, StorageError> {
        let guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let t_rows: i64 =
            guard.query_row("SELECT COUNT(*) FROM translation_cache", [], |r| r.get(0))?;
        let g_rows: i64 =
            guard.query_row("SELECT COUNT(*) FROM glossary_entries", [], |r| r.get(0))?;
        let p_rows: i64 =
            guard.query_row("SELECT COUNT(*) FROM page_states", [], |r| r.get(0))?;
        Ok(StorageMetrics {
            translation_cache_rows: t_rows.max(0) as usize,
            glossary_rows: g_rows.max(0) as usize,
            page_state_rows: p_rows.max(0) as usize,
        })
    }
}
