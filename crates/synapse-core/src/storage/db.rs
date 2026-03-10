//! SQLite model metadata database.
//!
//! Provides CRUD operations for the local model catalog. Each model entry
//! maps to a [`ModelInfo`] and tracks where the model files live on disk.

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use std::path::Path;
use synapse_types::error::Result;
use synapse_types::model::{ModelFormat, ModelId, ModelInfo, QuantLevel};
use synapse_types::SynapseError;
use uuid::Uuid;

/// Handle to the SQLite model catalog.
pub struct ModelDatabase {
    conn: Connection,
}

impl ModelDatabase {
    /// Open (or create) the catalog database at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    /// Open an in-memory database (useful for tests).
    #[cfg(test)]
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    /// Run schema migrations.
    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
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
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Insert a new model into the catalog.
    pub fn insert(&self, model: &ModelInfo) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO models (id, name, repo_id, format, quant, size_bytes,
                    parameter_count, architecture, license, local_path, sha256, pulled_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    model.id.to_string(),
                    model.name,
                    model.repo_id,
                    serde_json::to_string(&model.format).unwrap().trim_matches('"'),
                    serde_json::to_string(&model.quant).unwrap().trim_matches('"'),
                    model.size_bytes as i64,
                    model.parameter_count.map(|v| v as i64),
                    model.architecture,
                    model.license,
                    model.local_path,
                    model.sha256,
                    model.pulled_at.to_rfc3339(),
                ],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Get a model by its UUID.
    pub fn get(&self, id: ModelId) -> Result<ModelInfo> {
        self.conn
            .query_row("SELECT * FROM models WHERE id = ?1", params![id.to_string()], |row| {
                row_to_model(row)
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    SynapseError::ModelNotFound(id.to_string())
                }
                other => SynapseError::StorageError(other.to_string()),
            })
    }

    /// Find a model by name (exact match).
    pub fn get_by_name(&self, name: &str) -> Result<ModelInfo> {
        self.conn
            .query_row("SELECT * FROM models WHERE name = ?1", params![name], |row| {
                row_to_model(row)
            })
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    SynapseError::ModelNotFound(name.to_string())
                }
                other => SynapseError::StorageError(other.to_string()),
            })
    }

    /// List all models in the catalog.
    pub fn list(&self) -> Result<Vec<ModelInfo>> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM models ORDER BY pulled_at DESC")
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        let models = stmt
            .query_map([], |row| row_to_model(row))
            .map_err(|e| SynapseError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        Ok(models)
    }

    /// Update an existing model entry.
    pub fn update(&self, model: &ModelInfo) -> Result<()> {
        let rows = self
            .conn
            .execute(
                "UPDATE models SET name = ?2, repo_id = ?3, format = ?4, quant = ?5,
                    size_bytes = ?6, parameter_count = ?7, architecture = ?8,
                    license = ?9, local_path = ?10, sha256 = ?11, pulled_at = ?12
                 WHERE id = ?1",
                params![
                    model.id.to_string(),
                    model.name,
                    model.repo_id,
                    serde_json::to_string(&model.format).unwrap().trim_matches('"'),
                    serde_json::to_string(&model.quant).unwrap().trim_matches('"'),
                    model.size_bytes as i64,
                    model.parameter_count.map(|v| v as i64),
                    model.architecture,
                    model.license,
                    model.local_path,
                    model.sha256,
                    model.pulled_at.to_rfc3339(),
                ],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        if rows == 0 {
            return Err(SynapseError::ModelNotFound(model.id.to_string()));
        }
        Ok(())
    }

    /// Delete a model from the catalog by ID.
    pub fn delete(&self, id: ModelId) -> Result<()> {
        let rows = self
            .conn
            .execute("DELETE FROM models WHERE id = ?1", params![id.to_string()])
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        if rows == 0 {
            return Err(SynapseError::ModelNotFound(id.to_string()));
        }
        Ok(())
    }

    /// Count total models in the catalog.
    pub fn count(&self) -> Result<usize> {
        self.conn
            .query_row("SELECT COUNT(*) FROM models", [], |row| {
                row.get::<_, i64>(0)
            })
            .map(|c| c as usize)
            .map_err(|e| SynapseError::StorageError(e.to_string()))
    }
}

/// Map a SQLite row to a ModelInfo.
fn row_to_model(row: &rusqlite::Row) -> rusqlite::Result<ModelInfo> {
    let id_str: String = row.get(0)?;
    let format_str: String = row.get(3)?;
    let quant_str: String = row.get(4)?;
    let pulled_str: String = row.get(11)?;

    let id = Uuid::parse_str(&id_str)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e)))?;

    let format: ModelFormat = serde_json::from_str(&format!("\"{format_str}\""))
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e)))?;

    let quant: QuantLevel = serde_json::from_str(&format!("\"{quant_str}\""))
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(e)))?;

    let pulled_at: DateTime<Utc> = DateTime::parse_from_rfc3339(&pulled_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(11, rusqlite::types::Type::Text, Box::new(e)))?;

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
            local_path: "/home/user/.synapse/models/llama-3.1-8b/model.gguf".into(),
            sha256: Some("abc123".into()),
            pulled_at: Utc::now(),
        }
    }

    #[test]
    fn insert_and_get() {
        let db = ModelDatabase::open_memory().unwrap();
        let model = sample_model();
        db.insert(&model).unwrap();
        let fetched = db.get(model.id).unwrap();
        assert_eq!(fetched.name, model.name);
        assert_eq!(fetched.format, model.format);
        assert_eq!(fetched.quant, model.quant);
    }

    #[test]
    fn get_by_name() {
        let db = ModelDatabase::open_memory().unwrap();
        let model = sample_model();
        db.insert(&model).unwrap();
        let fetched = db.get_by_name("llama-3.1-8b").unwrap();
        assert_eq!(fetched.id, model.id);
    }

    #[test]
    fn list_models() {
        let db = ModelDatabase::open_memory().unwrap();
        assert_eq!(db.list().unwrap().len(), 0);
        db.insert(&sample_model()).unwrap();
        assert_eq!(db.list().unwrap().len(), 1);
    }

    #[test]
    fn update_model() {
        let db = ModelDatabase::open_memory().unwrap();
        let mut model = sample_model();
        db.insert(&model).unwrap();
        model.name = "llama-3.1-8b-updated".into();
        db.update(&model).unwrap();
        let fetched = db.get(model.id).unwrap();
        assert_eq!(fetched.name, "llama-3.1-8b-updated");
    }

    #[test]
    fn delete_model() {
        let db = ModelDatabase::open_memory().unwrap();
        let model = sample_model();
        db.insert(&model).unwrap();
        assert_eq!(db.count().unwrap(), 1);
        db.delete(model.id).unwrap();
        assert_eq!(db.count().unwrap(), 0);
    }

    #[test]
    fn get_not_found() {
        let db = ModelDatabase::open_memory().unwrap();
        let result = db.get(Uuid::new_v4());
        assert!(matches!(result, Err(SynapseError::ModelNotFound(_))));
    }

    #[test]
    fn delete_not_found() {
        let db = ModelDatabase::open_memory().unwrap();
        let result = db.delete(Uuid::new_v4());
        assert!(matches!(result, Err(SynapseError::ModelNotFound(_))));
    }
}
