//! SQLite model metadata database.
//!
//! Provides CRUD operations for the local model catalog. Each model entry
//! maps to a [`ModelInfo`] and tracks where the model files live on disk.

use crate::types::IfranError;
use crate::types::PagedResult;
use crate::types::TenantId;
use crate::types::error::Result;
use crate::types::model::{ModelFormat, ModelId, ModelInfo, QuantLevel};
use chrono::{DateTime, Utc};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use std::path::Path;
use uuid::Uuid;

/// Handle to the SQLite model catalog.
pub struct ModelDatabase {
    pool: Pool<SqliteConnectionManager>,
}

impl ModelDatabase {
    /// Open (or create) the catalog database at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let manager = SqliteConnectionManager::file(path).with_init(|conn| {
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
            Ok(())
        });
        let pool = Pool::builder()
            .max_size(4)
            .build(manager)
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let db = Self { pool };
        db.migrate()?;
        Ok(db)
    }

    /// Open an in-memory database (useful for tests).
    #[cfg(test)]
    pub fn open_memory() -> Result<Self> {
        let manager = SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
            Ok(())
        });
        let pool = Pool::builder()
            .max_size(1)
            .build(manager)
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let db = Self { pool };
        db.migrate()?;
        Ok(db)
    }

    /// Run schema migrations.
    fn migrate(&self) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS models (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    repo_id     TEXT,
                    format      TEXT NOT NULL,
                    quant       TEXT NOT NULL,
                    size_bytes  INTEGER NOT NULL,
                    parameter_count INTEGER,
                    architecture TEXT,
                    license     TEXT,
                    local_path  TEXT NOT NULL,
                    sha256      TEXT,
                    pulled_at   TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
                CREATE INDEX IF NOT EXISTS idx_models_repo ON models(repo_id);",
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;

        // Add tenant_id column (idempotent — ignore if already exists)
        let _ = conn.execute_batch(
            "ALTER TABLE models ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default';",
        );
        conn.execute_batch("CREATE INDEX IF NOT EXISTS idx_models_tenant ON models(tenant_id);")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(())
    }

    /// Insert a new model into the catalog.
    pub fn insert(&self, model: &ModelInfo, tenant_id: &TenantId) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let format_str =
            serde_json::to_string(&model.format).map_err(|e| IfranError::Other(e.to_string()))?;
        let quant_str =
            serde_json::to_string(&model.quant).map_err(|e| IfranError::Other(e.to_string()))?;
        conn.execute(
            "INSERT INTO models (id, name, repo_id, format, quant, size_bytes,
                    parameter_count, architecture, license, local_path, sha256, pulled_at, tenant_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                model.id.to_string(),
                model.name,
                model.repo_id,
                format_str.trim_matches('"'),
                quant_str.trim_matches('"'),
                model.size_bytes as i64,
                model.parameter_count.map(|v| v as i64),
                model.architecture,
                model.license,
                model.local_path,
                model.sha256,
                model.pulled_at.to_rfc3339(),
                tenant_id.0,
            ],
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Get a model by its UUID.
    pub fn get(&self, id: ModelId, tenant_id: &TenantId) -> Result<ModelInfo> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.query_row(
            "SELECT * FROM models WHERE id = ?1 AND tenant_id = ?2",
            params![id.to_string(), tenant_id.0],
            row_to_model,
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => IfranError::ModelNotFound(id.to_string()),
            other => IfranError::StorageError(other.to_string()),
        })
    }

    /// Find a model by name (exact match).
    pub fn get_by_name(&self, name: &str, tenant_id: &TenantId) -> Result<ModelInfo> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.query_row(
            "SELECT * FROM models WHERE name = ?1 AND tenant_id = ?2",
            params![name, tenant_id.0],
            row_to_model,
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => IfranError::ModelNotFound(name.to_string()),
            other => IfranError::StorageError(other.to_string()),
        })
    }

    /// List models with pagination.
    pub fn list(
        &self,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<ModelInfo>> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let total: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM models WHERE tenant_id = ?1",
                params![tenant_id.0],
                |row| row.get::<_, i64>(0),
            )
            .map(|c| c as usize)
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let mut stmt = conn
            .prepare(
                "SELECT * FROM models WHERE tenant_id = ?1 ORDER BY pulled_at DESC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let items = stmt
            .query_map(params![tenant_id.0, limit, offset], row_to_model)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(PagedResult { items, total })
    }

    /// Update an existing model entry.
    pub fn update(&self, model: &ModelInfo, tenant_id: &TenantId) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let format_str =
            serde_json::to_string(&model.format).map_err(|e| IfranError::Other(e.to_string()))?;
        let quant_str =
            serde_json::to_string(&model.quant).map_err(|e| IfranError::Other(e.to_string()))?;
        let rows = conn
            .execute(
                "UPDATE models SET name = ?2, repo_id = ?3, format = ?4, quant = ?5,
                    size_bytes = ?6, parameter_count = ?7, architecture = ?8,
                    license = ?9, local_path = ?10, sha256 = ?11, pulled_at = ?12
                 WHERE id = ?1 AND tenant_id = ?13",
                params![
                    model.id.to_string(),
                    model.name,
                    model.repo_id,
                    format_str.trim_matches('"'),
                    quant_str.trim_matches('"'),
                    model.size_bytes as i64,
                    model.parameter_count.map(|v| v as i64),
                    model.architecture,
                    model.license,
                    model.local_path,
                    model.sha256,
                    model.pulled_at.to_rfc3339(),
                    tenant_id.0,
                ],
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        if rows == 0 {
            return Err(IfranError::ModelNotFound(model.id.to_string()));
        }
        Ok(())
    }

    /// Delete a model from the catalog by ID.
    pub fn delete(&self, id: ModelId, tenant_id: &TenantId) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let rows = conn
            .execute(
                "DELETE FROM models WHERE id = ?1 AND tenant_id = ?2",
                params![id.to_string(), tenant_id.0],
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        if rows == 0 {
            return Err(IfranError::ModelNotFound(id.to_string()));
        }
        Ok(())
    }

    /// Count total models in the catalog.
    pub fn count(&self, tenant_id: &TenantId) -> Result<usize> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.query_row(
            "SELECT COUNT(*) FROM models WHERE tenant_id = ?1",
            params![tenant_id.0],
            |row| row.get::<_, i64>(0),
        )
        .map(|c| c as usize)
        .map_err(|e| IfranError::StorageError(e.to_string()))
    }
}

impl crate::storage::traits::ModelStore for ModelDatabase {
    fn insert(&self, model: &ModelInfo, tenant_id: &TenantId) -> Result<()> {
        self.insert(model, tenant_id)
    }

    fn get(&self, id: ModelId, tenant_id: &TenantId) -> Result<ModelInfo> {
        self.get(id, tenant_id)
    }

    fn get_by_name(&self, name: &str, tenant_id: &TenantId) -> Result<ModelInfo> {
        self.get_by_name(name, tenant_id)
    }

    fn list(
        &self,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<ModelInfo>> {
        self.list(tenant_id, limit, offset)
    }

    fn update(&self, model: &ModelInfo, tenant_id: &TenantId) -> Result<()> {
        self.update(model, tenant_id)
    }

    fn delete(&self, id: ModelId, tenant_id: &TenantId) -> Result<()> {
        self.delete(id, tenant_id)
    }

    fn count(&self, tenant_id: &TenantId) -> Result<usize> {
        self.count(tenant_id)
    }
}

/// Map a SQLite row to a ModelInfo.
fn row_to_model(row: &rusqlite::Row) -> rusqlite::Result<ModelInfo> {
    let id_str: String = row.get(0)?;
    let format_str: String = row.get(3)?;
    let quant_str: String = row.get(4)?;
    let pulled_str: String = row.get(11)?;

    let id = Uuid::parse_str(&id_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;

    let format: ModelFormat = crate::storage::deserialize_quoted(&format_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e))
    })?;

    let quant: QuantLevel = crate::storage::deserialize_quoted(&quant_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(e))
    })?;

    let pulled_at: DateTime<Utc> = DateTime::parse_from_rfc3339(&pulled_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(11, rusqlite::types::Type::Text, Box::new(e))
        })?;

    Ok(ModelInfo {
        id,
        name: row.get(1)?,
        repo_id: row.get(2)?,
        format,
        quant,
        size_bytes: row.get::<_, i64>(5)? as u64,
        parameter_count: row.get::<_, Option<i64>>(6)?.map(|v| v as u64),
        architecture: row.get(7)?,
        license: row.get(8)?,
        local_path: row.get(9)?,
        sha256: row.get(10)?,
        pulled_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_model() -> ModelInfo {
        ModelInfo {
            id: Uuid::new_v4(),
            name: "llama-3.1-8b".into(),
            repo_id: Some("meta-llama/Llama-3.1-8B-Instruct".into()),
            format: ModelFormat::Gguf,
            quant: QuantLevel::Q4KM,
            size_bytes: 4_800_000_000,
            parameter_count: Some(8_000_000_000),
            architecture: Some("llama".into()),
            license: Some("llama3.1".into()),
            local_path: "/home/user/.ifran/models/llama-3.1-8b/model.gguf".into(),
            sha256: Some("abc123".into()),
            pulled_at: Utc::now(),
        }
    }

    #[test]
    fn insert_and_get() {
        let db = ModelDatabase::open_memory().unwrap();
        let model = sample_model();
        let tenant = TenantId::default_tenant();
        db.insert(&model, &tenant).unwrap();
        let fetched = db.get(model.id, &tenant).unwrap();
        assert_eq!(fetched.name, model.name);
        assert_eq!(fetched.format, model.format);
        assert_eq!(fetched.quant, model.quant);
    }

    #[test]
    fn get_by_name() {
        let db = ModelDatabase::open_memory().unwrap();
        let model = sample_model();
        let tenant = TenantId::default_tenant();
        db.insert(&model, &tenant).unwrap();
        let fetched = db.get_by_name("llama-3.1-8b", &tenant).unwrap();
        assert_eq!(fetched.id, model.id);
    }

    #[test]
    fn list_models() {
        let db = ModelDatabase::open_memory().unwrap();
        let tenant = TenantId::default_tenant();
        assert_eq!(db.list(&tenant, 100, 0).unwrap().items.len(), 0);
        db.insert(&sample_model(), &tenant).unwrap();
        assert_eq!(db.list(&tenant, 100, 0).unwrap().items.len(), 1);
    }

    #[test]
    fn update_model() {
        let db = ModelDatabase::open_memory().unwrap();
        let mut model = sample_model();
        let tenant = TenantId::default_tenant();
        db.insert(&model, &tenant).unwrap();
        model.name = "llama-3.1-8b-updated".into();
        db.update(&model, &tenant).unwrap();
        let fetched = db.get(model.id, &tenant).unwrap();
        assert_eq!(fetched.name, "llama-3.1-8b-updated");
    }

    #[test]
    fn delete_model() {
        let db = ModelDatabase::open_memory().unwrap();
        let model = sample_model();
        let tenant = TenantId::default_tenant();
        db.insert(&model, &tenant).unwrap();
        assert_eq!(db.count(&tenant).unwrap(), 1);
        db.delete(model.id, &tenant).unwrap();
        assert_eq!(db.count(&tenant).unwrap(), 0);
    }

    #[test]
    fn get_not_found() {
        let db = ModelDatabase::open_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let result = db.get(Uuid::new_v4(), &tenant);
        assert!(matches!(result, Err(IfranError::ModelNotFound(_))));
    }

    #[test]
    fn delete_not_found() {
        let db = ModelDatabase::open_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let result = db.delete(Uuid::new_v4(), &tenant);
        assert!(matches!(result, Err(IfranError::ModelNotFound(_))));
    }

    #[test]
    fn count_with_tenant() {
        let db = ModelDatabase::open_memory().unwrap();
        let tenant_a = TenantId("tenant-a".into());
        let tenant_b = TenantId("tenant-b".into());

        let mut m1 = sample_model();
        m1.name = "model-a".into();
        db.insert(&m1, &tenant_a).unwrap();

        let mut m2 = sample_model();
        m2.name = "model-b".into();
        db.insert(&m2, &tenant_a).unwrap();

        let mut m3 = sample_model();
        m3.name = "model-c".into();
        db.insert(&m3, &tenant_b).unwrap();

        assert_eq!(db.count(&tenant_a).unwrap(), 2);
        assert_eq!(db.count(&tenant_b).unwrap(), 1);
    }

    #[test]
    fn get_by_name_wrong_tenant() {
        let db = ModelDatabase::open_memory().unwrap();
        let tenant_a = TenantId("tenant-a".into());
        let tenant_b = TenantId("tenant-b".into());

        let model = sample_model();
        db.insert(&model, &tenant_a).unwrap();

        // Should not find it under a different tenant
        let result = db.get_by_name("llama-3.1-8b", &tenant_b);
        assert!(matches!(result, Err(IfranError::ModelNotFound(_))));
    }

    #[test]
    fn update_wrong_tenant() {
        let db = ModelDatabase::open_memory().unwrap();
        let tenant_a = TenantId("tenant-a".into());
        let tenant_b = TenantId("tenant-b".into());

        let mut model = sample_model();
        db.insert(&model, &tenant_a).unwrap();

        model.name = "updated-name".into();
        let result = db.update(&model, &tenant_b);
        assert!(matches!(result, Err(IfranError::ModelNotFound(_))));
    }
}
