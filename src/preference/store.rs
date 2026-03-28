//! Standalone preference pair store for DPO/RLHF training data.

use crate::types::IfranError;
use crate::types::TenantId;
use crate::types::error::Result;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A preference pair for DPO training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencePair {
    pub id: Uuid,
    pub prompt: String,
    pub chosen: String,
    pub rejected: String,
    pub source: String,
    pub score_delta: Option<f64>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

pub struct PreferenceStore {
    conn: Connection,
}

impl PreferenceStore {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let s = Self { conn };
        s.migrate()?;
        Ok(s)
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| IfranError::StorageError(e.to_string()))?;
        let s = Self { conn };
        s.migrate()?;
        Ok(s)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS preference_pairs (
                id           TEXT PRIMARY KEY,
                tenant_id    TEXT NOT NULL DEFAULT 'default',
                prompt       TEXT NOT NULL,
                chosen       TEXT NOT NULL,
                rejected     TEXT NOT NULL,
                source       TEXT NOT NULL DEFAULT 'manual',
                score_delta  REAL,
                created_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prefs_tenant ON preference_pairs(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_prefs_source ON preference_pairs(source);",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Add a preference pair.
    pub fn add(&self, pair: &PreferencePair, tenant_id: &TenantId) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO preference_pairs (id, tenant_id, prompt, chosen, rejected, source, score_delta, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    pair.id.to_string(),
                    tenant_id.0,
                    pair.prompt,
                    pair.chosen,
                    pair.rejected,
                    pair.source,
                    pair.score_delta,
                    pair.created_at.to_rfc3339()
                ],
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// List preference pairs for a tenant.
    pub fn list(&self, tenant_id: &TenantId, limit: u32) -> Result<Vec<PreferencePair>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, prompt, chosen, rejected, source, score_delta, created_at
             FROM preference_pairs WHERE tenant_id = ?1 ORDER BY created_at DESC LIMIT ?2",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let rows = stmt
            .query_map(params![tenant_id.0, limit as i64], row_to_pair)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(rows)
    }

    /// Export all pairs in DPO format (JSON lines).
    pub fn export_dpo(&self, tenant_id: &TenantId) -> Result<Vec<serde_json::Value>> {
        let mut stmt = self
            .conn
            .prepare("SELECT prompt, chosen, rejected FROM preference_pairs WHERE tenant_id = ?1")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let rows = stmt
            .query_map(params![tenant_id.0], |row| {
                Ok(serde_json::json!({
                    "prompt": row.get::<_, String>(0)?,
                    "chosen": row.get::<_, String>(1)?,
                    "rejected": row.get::<_, String>(2)?,
                }))
            })
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(rows)
    }

    /// Count pairs for a tenant.
    pub fn count(&self, tenant_id: &TenantId) -> Result<u64> {
        let c: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM preference_pairs WHERE tenant_id = ?1",
                params![tenant_id.0],
                |row| row.get(0),
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(c as u64)
    }

    /// Add multiple pairs in a batch, wrapped in a transaction for atomicity.
    pub fn add_batch(&self, pairs: &[PreferencePair], tenant_id: &TenantId) -> Result<u64> {
        self.conn
            .execute_batch("BEGIN")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let mut count = 0u64;
        for pair in pairs {
            if let Err(e) = self.add(pair, tenant_id) {
                let _ = self.conn.execute_batch("ROLLBACK");
                return Err(e);
            }
            count += 1;
        }
        self.conn
            .execute_batch("COMMIT")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(count)
    }
}

fn row_to_pair(row: &rusqlite::Row) -> rusqlite::Result<PreferencePair> {
    let id_str: String = row.get(0)?;
    let created_str: String = row.get(6)?;
    Ok(PreferencePair {
        id: Uuid::parse_str(&id_str).unwrap_or_default(),
        prompt: row.get(1)?,
        chosen: row.get(2)?,
        rejected: row.get(3)?,
        source: row.get(4)?,
        score_delta: row.get(5)?,
        created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> TenantId {
        TenantId::default_tenant()
    }

    fn make_pair(prompt: &str) -> PreferencePair {
        PreferencePair {
            id: Uuid::new_v4(),
            prompt: prompt.into(),
            chosen: "Good answer.".into(),
            rejected: "Bad answer.".into(),
            source: "manual".into(),
            score_delta: Some(0.3),
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn add_and_list() {
        let s = PreferenceStore::open_in_memory().unwrap();
        s.add(&make_pair("What is Rust?"), &t()).unwrap();
        s.add(&make_pair("What is Python?"), &t()).unwrap();
        let pairs = s.list(&t(), 10).unwrap();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn export_dpo() {
        let s = PreferenceStore::open_in_memory().unwrap();
        s.add(&make_pair("Q1"), &t()).unwrap();
        s.add(&make_pair("Q2"), &t()).unwrap();
        let export = s.export_dpo(&t()).unwrap();
        assert_eq!(export.len(), 2);
        assert!(export[0]["chosen"].is_string());
        assert!(export[0]["rejected"].is_string());
    }

    #[test]
    fn count_pairs() {
        let s = PreferenceStore::open_in_memory().unwrap();
        assert_eq!(s.count(&t()).unwrap(), 0);
        s.add(&make_pair("Q"), &t()).unwrap();
        assert_eq!(s.count(&t()).unwrap(), 1);
    }

    #[test]
    fn batch_add() {
        let s = PreferenceStore::open_in_memory().unwrap();
        let pairs = vec![make_pair("Q1"), make_pair("Q2"), make_pair("Q3")];
        let count = s.add_batch(&pairs, &t()).unwrap();
        assert_eq!(count, 3);
        assert_eq!(s.count(&t()).unwrap(), 3);
    }

    #[test]
    fn tenant_isolation() {
        let s = PreferenceStore::open_in_memory().unwrap();
        s.add(&make_pair("Q"), &TenantId("a".into())).unwrap();
        s.add(&make_pair("Q"), &TenantId("b".into())).unwrap();
        assert_eq!(s.count(&TenantId("a".into())).unwrap(), 1);
    }

    #[test]
    fn list_respects_limit() {
        let s = PreferenceStore::open_in_memory().unwrap();
        for i in 0..5 {
            s.add(&make_pair(&format!("Q{i}")), &t()).unwrap();
        }
        let pairs = s.list(&t(), 2).unwrap();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn export_empty() {
        let s = PreferenceStore::open_in_memory().unwrap();
        let export = s.export_dpo(&t()).unwrap();
        assert!(export.is_empty());
    }

    #[test]
    fn pair_fields_roundtrip() {
        let s = PreferenceStore::open_in_memory().unwrap();
        let mut pair = make_pair("Tell me about Rust");
        pair.chosen = "Rust is a systems programming language.".into();
        pair.rejected = "I don't know.".into();
        pair.source = "human_eval".into();
        pair.score_delta = Some(0.75);
        s.add(&pair, &t()).unwrap();

        let pairs = s.list(&t(), 10).unwrap();
        assert_eq!(pairs.len(), 1);
        let fetched = &pairs[0];
        assert_eq!(fetched.prompt, "Tell me about Rust");
        assert_eq!(fetched.chosen, "Rust is a systems programming language.");
        assert_eq!(fetched.rejected, "I don't know.");
        assert_eq!(fetched.source, "human_eval");
        assert_eq!(fetched.score_delta, Some(0.75));
    }
}
