//! Dataset curation with deduplication and version tracking.

use rusqlite::{Connection, params};
use synapse_types::SynapseError;
use synapse_types::TenantId;
use synapse_types::dataset::{CuratedDataset, DatasetId};
use synapse_types::error::Result;
use uuid::Uuid;

pub struct DatasetCurator {
    conn: Connection,
}

impl DatasetCurator {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let c = Self { conn };
        c.migrate()?;
        Ok(c)
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let c = Self { conn };
        c.migrate()?;
        Ok(c)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS curated_datasets (
                id           TEXT PRIMARY KEY,
                tenant_id    TEXT NOT NULL DEFAULT 'default',
                name         TEXT NOT NULL,
                source_path  TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                format       TEXT NOT NULL,
                version      INTEGER NOT NULL DEFAULT 1,
                fingerprint  TEXT NOT NULL,
                created_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dataset_hashes (
                dataset_id   TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                PRIMARY KEY (dataset_id, content_hash),
                FOREIGN KEY (dataset_id) REFERENCES curated_datasets(id)
            );
            CREATE INDEX IF NOT EXISTS idx_datasets_tenant ON curated_datasets(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_datasets_name ON curated_datasets(name);",
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Register a new curated dataset.
    pub fn register(&self, dataset: &CuratedDataset, tenant_id: &TenantId) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO curated_datasets (id, tenant_id, name, source_path, sample_count, format, version, fingerprint, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    dataset.id.to_string(),
                    tenant_id.0,
                    dataset.name,
                    dataset.source_path,
                    dataset.sample_count as i64,
                    dataset.format,
                    dataset.version as i64,
                    dataset.fingerprint,
                    dataset.created_at.to_rfc3339()
                ],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Get a dataset by ID.
    pub fn get(&self, id: DatasetId, tenant_id: &TenantId) -> Result<CuratedDataset> {
        self.conn
            .query_row(
                "SELECT id, name, source_path, sample_count, format, version, fingerprint, created_at
             FROM curated_datasets WHERE id = ?1 AND tenant_id = ?2",
                params![id.to_string(), tenant_id.0],
                row_to_dataset,
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    SynapseError::StorageError(format!("Dataset {id} not found"))
                }
                other => SynapseError::StorageError(other.to_string()),
            })
    }

    /// List datasets for a tenant.
    pub fn list(&self, tenant_id: &TenantId) -> Result<Vec<CuratedDataset>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, name, source_path, sample_count, format, version, fingerprint, created_at
             FROM curated_datasets WHERE tenant_id = ?1 ORDER BY created_at DESC",
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let rows = stmt
            .query_map(params![tenant_id.0], row_to_dataset)
            .map_err(|e| SynapseError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(rows)
    }

    /// Check if a content hash already exists for deduplication.
    pub fn is_duplicate(&self, dataset_id: DatasetId, content_hash: &str) -> Result<bool> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM dataset_hashes WHERE dataset_id = ?1 AND content_hash = ?2",
                params![dataset_id.to_string(), content_hash],
                |row| row.get(0),
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(count > 0)
    }

    /// Record a content hash for deduplication.
    pub fn record_hash(&self, dataset_id: DatasetId, content_hash: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO dataset_hashes (dataset_id, content_hash) VALUES (?1, ?2)",
                params![dataset_id.to_string(), content_hash],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Count unique samples in a dataset (by hash).
    pub fn unique_count(&self, dataset_id: DatasetId) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM dataset_hashes WHERE dataset_id = ?1",
                params![dataset_id.to_string()],
                |row| row.get(0),
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(count as u64)
    }
}

fn row_to_dataset(row: &rusqlite::Row) -> rusqlite::Result<CuratedDataset> {
    let id_str: String = row.get(0)?;
    let created_str: String = row.get(7)?;
    Ok(CuratedDataset {
        id: Uuid::parse_str(&id_str).unwrap_or_default(),
        name: row.get(1)?,
        source_path: row.get(2)?,
        sample_count: row.get::<_, i64>(3)? as u64,
        format: row.get(4)?,
        version: row.get::<_, i64>(5)? as u32,
        fingerprint: row.get(6)?,
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

    fn make_dataset(name: &str) -> CuratedDataset {
        CuratedDataset {
            id: Uuid::new_v4(),
            name: name.into(),
            source_path: "/data/train.jsonl".into(),
            sample_count: 1000,
            format: "jsonl".into(),
            version: 1,
            fingerprint: "abc123".into(),
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn register_and_get() {
        let c = DatasetCurator::open_in_memory().unwrap();
        let d = make_dataset("train-v1");
        c.register(&d, &t()).unwrap();
        let fetched = c.get(d.id, &t()).unwrap();
        assert_eq!(fetched.name, "train-v1");
    }

    #[test]
    fn list_datasets() {
        let c = DatasetCurator::open_in_memory().unwrap();
        c.register(&make_dataset("d1"), &t()).unwrap();
        c.register(&make_dataset("d2"), &t()).unwrap();
        assert_eq!(c.list(&t()).unwrap().len(), 2);
    }

    #[test]
    fn deduplication() {
        let c = DatasetCurator::open_in_memory().unwrap();
        let d = make_dataset("data");
        c.register(&d, &t()).unwrap();

        assert!(!c.is_duplicate(d.id, "hash1").unwrap());
        c.record_hash(d.id, "hash1").unwrap();
        assert!(c.is_duplicate(d.id, "hash1").unwrap());
        assert!(!c.is_duplicate(d.id, "hash2").unwrap());
    }

    #[test]
    fn unique_count() {
        let c = DatasetCurator::open_in_memory().unwrap();
        let d = make_dataset("data");
        c.register(&d, &t()).unwrap();

        c.record_hash(d.id, "h1").unwrap();
        c.record_hash(d.id, "h2").unwrap();
        c.record_hash(d.id, "h1").unwrap(); // duplicate, ignored
        assert_eq!(c.unique_count(d.id).unwrap(), 2);
    }

    #[test]
    fn tenant_isolation() {
        let c = DatasetCurator::open_in_memory().unwrap();
        let t1 = TenantId("a".into());
        let t2 = TenantId("b".into());
        c.register(&make_dataset("d"), &t1).unwrap();
        c.register(&make_dataset("d"), &t2).unwrap();
        assert_eq!(c.list(&t1).unwrap().len(), 1);
    }
}
