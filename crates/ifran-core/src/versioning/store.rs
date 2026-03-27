//! SQLite storage for model versions.

use ifran_types::error::Result;
use ifran_types::versioning::{ModelVersion, ModelVersionId};
use ifran_types::{IfranError, TenantId};
use rusqlite::{Connection, params};
use uuid::Uuid;

pub struct VersionStore {
    conn: Connection,
}

impl VersionStore {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| IfranError::StorageError(e.to_string()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS model_versions (
                id                TEXT PRIMARY KEY,
                tenant_id         TEXT NOT NULL DEFAULT 'default',
                model_family      TEXT NOT NULL,
                version_tag       TEXT NOT NULL,
                model_id          TEXT,
                training_job_id   TEXT,
                parent_version_id TEXT,
                consumer          TEXT,
                notes             TEXT,
                created_at        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_versions_tenant ON model_versions(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_versions_family ON model_versions(model_family);
            CREATE INDEX IF NOT EXISTS idx_versions_consumer ON model_versions(consumer);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_versions_family_tag ON model_versions(tenant_id, model_family, version_tag);",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Create a new model version.
    pub fn create(&self, version: &ModelVersion, tenant_id: &TenantId) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO model_versions (id, tenant_id, model_family, version_tag, model_id,
             training_job_id, parent_version_id, consumer, notes, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    version.id.to_string(),
                    tenant_id.0,
                    version.model_family,
                    version.version_tag,
                    version.model_id.map(|id| id.to_string()),
                    version.training_job_id.map(|id| id.to_string()),
                    version.parent_version_id.map(|id| id.to_string()),
                    version.consumer,
                    version.notes,
                    version.created_at.to_rfc3339(),
                ],
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Get a version by ID.
    pub fn get(&self, id: ModelVersionId, tenant_id: &TenantId) -> Result<ModelVersion> {
        self.conn
            .query_row(
                "SELECT id, model_family, version_tag, model_id, training_job_id,
             parent_version_id, consumer, notes, created_at
             FROM model_versions WHERE id = ?1 AND tenant_id = ?2",
                params![id.to_string(), tenant_id.0],
                row_to_version,
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => IfranError::ModelNotFound(id.to_string()),
                other => IfranError::StorageError(other.to_string()),
            })
    }

    /// List versions for a model family.
    pub fn list_by_family(
        &self,
        family: &str,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<ModelVersion>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, model_family, version_tag, model_id, training_job_id,
             parent_version_id, consumer, notes, created_at
             FROM model_versions WHERE model_family = ?1 AND tenant_id = ?2
             ORDER BY created_at DESC LIMIT ?3 OFFSET ?4",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let versions = stmt
            .query_map(params![family, tenant_id.0, limit, offset], row_to_version)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(versions)
    }

    /// List all versions for a tenant.
    pub fn list(&self, tenant_id: &TenantId, limit: u32, offset: u32) -> Result<Vec<ModelVersion>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, model_family, version_tag, model_id, training_job_id,
             parent_version_id, consumer, notes, created_at
             FROM model_versions WHERE tenant_id = ?1
             ORDER BY created_at DESC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let versions = stmt
            .query_map(params![tenant_id.0, limit, offset], row_to_version)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(versions)
    }

    /// Get the latest version for a model family.
    pub fn latest(&self, family: &str, tenant_id: &TenantId) -> Result<ModelVersion> {
        self.conn
            .query_row(
                "SELECT id, model_family, version_tag, model_id, training_job_id,
             parent_version_id, consumer, notes, created_at
             FROM model_versions WHERE model_family = ?1 AND tenant_id = ?2
             ORDER BY created_at DESC LIMIT 1",
                params![family, tenant_id.0],
                row_to_version,
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    IfranError::ModelNotFound(format!("No versions for {family}"))
                }
                other => IfranError::StorageError(other.to_string()),
            })
    }

    /// Get the version lineage (chain of parent versions).
    pub fn get_lineage(
        &self,
        id: ModelVersionId,
        tenant_id: &TenantId,
    ) -> Result<Vec<ModelVersion>> {
        let mut chain = Vec::new();
        let mut current = Some(id);
        while let Some(vid) = current {
            match self.get(vid, tenant_id) {
                Ok(v) => {
                    current = v.parent_version_id;
                    chain.push(v);
                }
                Err(_) => break,
            }
        }
        Ok(chain)
    }
}

fn row_to_version(row: &rusqlite::Row) -> rusqlite::Result<ModelVersion> {
    let id_str: String = row.get(0)?;
    let model_id_str: Option<String> = row.get(3)?;
    let training_id_str: Option<String> = row.get(4)?;
    let parent_str: Option<String> = row.get(5)?;
    let created_str: String = row.get(8)?;

    Ok(ModelVersion {
        id: Uuid::parse_str(&id_str).unwrap_or_default(),
        model_family: row.get(1)?,
        version_tag: row.get(2)?,
        model_id: model_id_str.and_then(|s| Uuid::parse_str(&s).ok()),
        training_job_id: training_id_str.and_then(|s| Uuid::parse_str(&s).ok()),
        parent_version_id: parent_str.and_then(|s| Uuid::parse_str(&s).ok()),
        consumer: row.get(6)?,
        notes: row.get(7)?,
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

    fn make_version(family: &str, tag: &str) -> ModelVersion {
        ModelVersion {
            id: Uuid::new_v4(),
            model_family: family.into(),
            version_tag: tag.into(),
            model_id: None,
            training_job_id: None,
            parent_version_id: None,
            consumer: None,
            notes: None,
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn create_and_get() {
        let store = VersionStore::open_in_memory().unwrap();
        let v = make_version("llama-8b", "v1");
        store.create(&v, &t()).unwrap();
        let fetched = store.get(v.id, &t()).unwrap();
        assert_eq!(fetched.model_family, "llama-8b");
        assert_eq!(fetched.version_tag, "v1");
    }

    #[test]
    fn list_by_family() {
        let store = VersionStore::open_in_memory().unwrap();
        store.create(&make_version("llama-8b", "v1"), &t()).unwrap();
        store.create(&make_version("llama-8b", "v2"), &t()).unwrap();
        store
            .create(&make_version("mistral-7b", "v1"), &t())
            .unwrap();

        let llama = store.list_by_family("llama-8b", &t(), 100, 0).unwrap();
        assert_eq!(llama.len(), 2);
    }

    #[test]
    fn latest_version() {
        let store = VersionStore::open_in_memory().unwrap();
        let v1 = make_version("llama-8b", "v1");
        store.create(&v1, &t()).unwrap();
        // Small delay to ensure different timestamps
        let v2 = make_version("llama-8b", "v2");
        store.create(&v2, &t()).unwrap();

        let latest = store.latest("llama-8b", &t()).unwrap();
        assert_eq!(latest.version_tag, "v2");
    }

    #[test]
    fn version_lineage() {
        let store = VersionStore::open_in_memory().unwrap();
        let v1 = make_version("llama-8b", "v1");
        store.create(&v1, &t()).unwrap();

        let mut v2 = make_version("llama-8b", "v2");
        v2.parent_version_id = Some(v1.id);
        store.create(&v2, &t()).unwrap();

        let mut v3 = make_version("llama-8b", "v3");
        v3.parent_version_id = Some(v2.id);
        store.create(&v3, &t()).unwrap();

        let chain = store.get_lineage(v3.id, &t()).unwrap();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].version_tag, "v3");
        assert_eq!(chain[1].version_tag, "v2");
        assert_eq!(chain[2].version_tag, "v1");
    }

    #[test]
    fn tenant_isolation() {
        let store = VersionStore::open_in_memory().unwrap();
        let t1 = TenantId("t1".into());
        let t2 = TenantId("t2".into());

        store.create(&make_version("m", "v1"), &t1).unwrap();
        store.create(&make_version("m", "v1"), &t2).unwrap();

        assert_eq!(store.list(&t1, 100, 0).unwrap().len(), 1);
        assert_eq!(store.list(&t2, 100, 0).unwrap().len(), 1);
    }

    #[test]
    fn get_not_found() {
        let store = VersionStore::open_in_memory().unwrap();
        assert!(store.get(Uuid::new_v4(), &t()).is_err());
    }

    #[test]
    fn latest_not_found() {
        let store = VersionStore::open_in_memory().unwrap();
        assert!(store.latest("nonexistent", &t()).is_err());
    }

    #[test]
    fn with_consumer_and_notes() {
        let store = VersionStore::open_in_memory().unwrap();
        let mut v = make_version("llama-8b", "v1");
        v.consumer = Some("support-bot".into());
        v.notes = Some("Fine-tuned for customer support".into());
        store.create(&v, &t()).unwrap();

        let fetched = store.get(v.id, &t()).unwrap();
        assert_eq!(fetched.consumer, Some("support-bot".into()));
    }

    #[test]
    fn duplicate_family_tag_fails() {
        let store = VersionStore::open_in_memory().unwrap();
        store.create(&make_version("llama-8b", "v1"), &t()).unwrap();
        // Same tenant + family + tag should violate the unique index.
        let result = store.create(&make_version("llama-8b", "v1"), &t());
        assert!(result.is_err());
    }

    #[test]
    fn list_all_tenants_isolated() {
        let store = VersionStore::open_in_memory().unwrap();
        let t1 = TenantId("t1".into());
        let t2 = TenantId("t2".into());

        store.create(&make_version("m", "v1"), &t1).unwrap();
        store.create(&make_version("m", "v2"), &t1).unwrap();
        store.create(&make_version("m", "v1"), &t2).unwrap();

        // list returns only the tenant's data
        let t1_versions = store.list(&t1, 100, 0).unwrap();
        let t2_versions = store.list(&t2, 100, 0).unwrap();
        assert_eq!(t1_versions.len(), 2);
        assert_eq!(t2_versions.len(), 1);
    }

    #[test]
    fn get_lineage_single_node() {
        let store = VersionStore::open_in_memory().unwrap();
        let v = make_version("llama-8b", "v1");
        store.create(&v, &t()).unwrap();

        // A node with no parent returns a chain of just itself.
        let chain = store.get_lineage(v.id, &t()).unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].version_tag, "v1");
    }
}
