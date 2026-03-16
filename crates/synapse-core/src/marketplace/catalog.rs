//! Marketplace catalog — SQLite-backed index of published models.

use rusqlite::{Connection, params};
use synapse_types::SynapseError;
use synapse_types::TenantId;
use synapse_types::error::Result;
use synapse_types::marketplace::{MarketplaceEntry, MarketplaceQuery};

/// Local marketplace catalog backed by SQLite.
pub struct MarketplaceCatalog {
    conn: Connection,
}

impl MarketplaceCatalog {
    /// Open (or create) the marketplace database.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let catalog = Self { conn };
        catalog.migrate()?;
        Ok(catalog)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS marketplace_entries (
                    model_name          TEXT PRIMARY KEY,
                    description         TEXT,
                    format              TEXT NOT NULL,
                    quant               TEXT NOT NULL,
                    size_bytes          INTEGER NOT NULL,
                    parameter_count     INTEGER,
                    architecture        TEXT,
                    publisher_instance  TEXT NOT NULL,
                    download_url        TEXT NOT NULL,
                    sha256              TEXT,
                    tags                TEXT NOT NULL DEFAULT '[]',
                    published_at        TEXT NOT NULL,
                    eval_scores         TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_marketplace_publisher
                    ON marketplace_entries(publisher_instance);",
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        // Add tenant_id column (idempotent — ignore if already exists)
        let _ = self.conn.execute_batch(
            "ALTER TABLE marketplace_entries ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default';",
        );
        self.conn
            .execute_batch(
                "CREATE INDEX IF NOT EXISTS idx_marketplace_tenant ON marketplace_entries(tenant_id);",
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        Ok(())
    }

    /// Publish a model to the marketplace.
    pub fn publish(&self, entry: &MarketplaceEntry, tenant_id: &TenantId) -> Result<()> {
        let format_str = serde_json::to_string(&entry.format)
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let quant_str = serde_json::to_string(&entry.quant)
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let tags_str = serde_json::to_string(&entry.tags)
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let eval_str = entry
            .eval_scores
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        self.conn
            .execute(
                "INSERT OR REPLACE INTO marketplace_entries
                    (model_name, description, format, quant, size_bytes, parameter_count,
                     architecture, publisher_instance, download_url, sha256, tags,
                     published_at, eval_scores, tenant_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                params![
                    entry.model_name,
                    entry.description,
                    format_str.trim_matches('"'),
                    quant_str.trim_matches('"'),
                    entry.size_bytes as i64,
                    entry.parameter_count.map(|v| v as i64),
                    entry.architecture,
                    entry.publisher_instance,
                    entry.download_url,
                    entry.sha256,
                    tags_str,
                    entry.published_at.to_rfc3339(),
                    eval_str,
                    tenant_id.0,
                ],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Get a marketplace entry by model name.
    pub fn get(&self, model_name: &str, tenant_id: &TenantId) -> Result<MarketplaceEntry> {
        self.conn
            .query_row(
                "SELECT * FROM marketplace_entries WHERE model_name = ?1 AND tenant_id = ?2",
                params![model_name, tenant_id.0],
                row_to_entry,
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => SynapseError::MarketplaceError(format!(
                    "Model '{model_name}' not found in marketplace"
                )),
                other => SynapseError::StorageError(other.to_string()),
            })
    }

    /// Search the marketplace catalog.
    pub fn search(
        &self,
        query: &MarketplaceQuery,
        tenant_id: &TenantId,
    ) -> Result<Vec<MarketplaceEntry>> {
        // Build parameterized WHERE clause to prevent SQL injection
        let mut conditions = Vec::new();
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut param_idx = 1;

        // Always filter by tenant_id
        conditions.push(format!("tenant_id = ?{param_idx}"));
        param_values.push(Box::new(tenant_id.0.clone()));
        param_idx += 1;

        if let Some(ref search) = query.search {
            conditions.push(format!(
                "(model_name LIKE ?{pi1} OR description LIKE ?{pi2})",
                pi1 = param_idx,
                pi2 = param_idx + 1
            ));
            let pattern = format!("%{search}%");
            param_values.push(Box::new(pattern.clone()));
            param_values.push(Box::new(pattern));
            param_idx += 2;
        }
        if let Some(ref format) = query.format {
            conditions.push(format!("format = ?{param_idx}"));
            let f = serde_json::to_string(format)
                .map_err(|e| SynapseError::StorageError(e.to_string()))?;
            param_values.push(Box::new(f.trim_matches('"').to_string()));
            param_idx += 1;
        }
        if let Some(max_size) = query.max_size_bytes {
            conditions.push(format!("size_bytes <= ?{param_idx}"));
            param_values.push(Box::new(max_size as i64));
            param_idx += 1;
        }

        let _ = param_idx; // suppress unused warning

        let mut sql = String::from("SELECT * FROM marketplace_entries");
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(" ORDER BY published_at DESC");

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        let params: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let entries = stmt
            .query_map(params.as_slice(), row_to_entry)
            .map_err(|e| SynapseError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        Ok(entries)
    }

    /// List all marketplace entries.
    pub fn list(&self, tenant_id: &TenantId) -> Result<Vec<MarketplaceEntry>> {
        self.search(&MarketplaceQuery::default(), tenant_id)
    }

    /// Unpublish a model.
    pub fn unpublish(&self, model_name: &str, tenant_id: &TenantId) -> Result<()> {
        let rows = self
            .conn
            .execute(
                "DELETE FROM marketplace_entries WHERE model_name = ?1 AND tenant_id = ?2",
                params![model_name, tenant_id.0],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        if rows == 0 {
            return Err(SynapseError::MarketplaceError(format!(
                "Model '{model_name}' not found in marketplace"
            )));
        }
        Ok(())
    }

    /// Count entries.
    pub fn count(&self, tenant_id: &TenantId) -> Result<usize> {
        self.conn
            .query_row(
                "SELECT COUNT(*) FROM marketplace_entries WHERE tenant_id = ?1",
                params![tenant_id.0],
                |row| row.get::<_, i64>(0),
            )
            .map(|c| c as usize)
            .map_err(|e| SynapseError::StorageError(e.to_string()))
    }
}

fn row_to_entry(row: &rusqlite::Row) -> rusqlite::Result<MarketplaceEntry> {
    use synapse_types::model::{ModelFormat, QuantLevel};

    let format_str: String = row.get(2)?;
    let quant_str: String = row.get(3)?;
    let tags_str: String = row.get(10)?;
    let published_str: String = row.get(11)?;
    let eval_str: Option<String> = row.get(12)?;

    let format: ModelFormat = serde_json::from_str(&format!("\"{format_str}\"")).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(2, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let quant: QuantLevel = serde_json::from_str(&format!("\"{quant_str}\"")).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let tags: Vec<String> = serde_json::from_str(&tags_str).unwrap_or_default();
    let published_at = chrono::DateTime::parse_from_rfc3339(&published_str)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(11, rusqlite::types::Type::Text, Box::new(e))
        })?;
    let eval_scores = eval_str.and_then(|s| serde_json::from_str(&s).ok());

    Ok(MarketplaceEntry {
        model_name: row.get(0)?,
        description: row.get(1)?,
        format,
        quant,
        size_bytes: row.get::<_, i64>(4)? as u64,
        parameter_count: row.get::<_, Option<i64>>(5)?.map(|v| v as u64),
        architecture: row.get(6)?,
        publisher_instance: row.get(7)?,
        download_url: row.get(8)?,
        sha256: row.get(9)?,
        tags,
        published_at,
        eval_scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use synapse_types::model::{ModelFormat, QuantLevel};

    fn test_catalog() -> MarketplaceCatalog {
        let conn = Connection::open_in_memory().unwrap();
        let catalog = MarketplaceCatalog { conn };
        catalog.migrate().unwrap();
        catalog
    }

    fn sample_entry() -> MarketplaceEntry {
        MarketplaceEntry {
            model_name: "llama-3.1-8b-q4km".into(),
            description: Some("Llama 3.1 8B quantized".into()),
            format: ModelFormat::Gguf,
            quant: QuantLevel::Q4KM,
            size_bytes: 4_800_000_000,
            parameter_count: Some(8_000_000_000),
            architecture: Some("llama".into()),
            publisher_instance: "node-1".into(),
            download_url: "http://node-1:8420/marketplace/download/llama-3.1-8b-q4km".into(),
            sha256: Some("abc123".into()),
            tags: vec!["chat".into(), "instruct".into()],
            published_at: Utc::now(),
            eval_scores: None,
        }
    }

    #[test]
    fn publish_and_get() {
        let catalog = test_catalog();
        let tenant = TenantId::default_tenant();
        let entry = sample_entry();
        catalog.publish(&entry, &tenant).unwrap();
        let fetched = catalog.get("llama-3.1-8b-q4km", &tenant).unwrap();
        assert_eq!(fetched.publisher_instance, "node-1");
        assert_eq!(fetched.size_bytes, 4_800_000_000);
    }

    #[test]
    fn search_by_name() {
        let catalog = test_catalog();
        let tenant = TenantId::default_tenant();
        catalog.publish(&sample_entry(), &tenant).unwrap();
        let results = catalog
            .search(
                &MarketplaceQuery {
                    search: Some("llama".into()),
                    ..Default::default()
                },
                &tenant,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn unpublish() {
        let catalog = test_catalog();
        let tenant = TenantId::default_tenant();
        catalog.publish(&sample_entry(), &tenant).unwrap();
        assert_eq!(catalog.count(&tenant).unwrap(), 1);
        catalog.unpublish("llama-3.1-8b-q4km", &tenant).unwrap();
        assert_eq!(catalog.count(&tenant).unwrap(), 0);
    }

    #[test]
    fn list_empty() {
        let catalog = test_catalog();
        let tenant = TenantId::default_tenant();
        assert!(catalog.list(&tenant).unwrap().is_empty());
    }
}
