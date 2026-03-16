//! SQLite-backed tenant store for multi-tenant deployments.
//!
//! Maps API keys (hashed with BLAKE3) to tenant IDs. In single-tenant mode
//! this store is not used — the auth middleware falls back to `SYNAPSE_API_KEY`.

use rusqlite::{Connection, params};
use synapse_types::SynapseError;
use synapse_types::TenantId;
use synapse_types::error::Result;
use uuid::Uuid;

/// A tenant record from the database.
#[derive(Debug, Clone)]
pub struct TenantRecord {
    pub id: TenantId,
    pub name: String,
    pub enabled: bool,
    pub created_at: String,
}

/// Manages tenant storage in SQLite.
pub struct TenantStore {
    conn: Connection,
}

impl TenantStore {
    /// Open (or create) the tenant database at the given path.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Open an in-memory database (useful for tests).
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| SynapseError::StorageError(e.to_string()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS tenants (
                    id           TEXT PRIMARY KEY,
                    name         TEXT NOT NULL,
                    api_key_hash TEXT NOT NULL UNIQUE,
                    created_at   TEXT NOT NULL,
                    enabled      INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_tenants_key ON tenants(api_key_hash);",
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Create a new tenant. Returns the record and the raw API key (shown once).
    pub fn create_tenant(&self, name: &str) -> Result<(TenantRecord, String)> {
        let id = TenantId(Uuid::new_v4().to_string());
        let raw_key = format!("syn_{}", Uuid::new_v4().to_string().replace('-', ""));
        let key_hash = hash_key(&raw_key);
        let now = chrono::Utc::now().to_rfc3339();

        self.conn
            .execute(
                "INSERT INTO tenants (id, name, api_key_hash, created_at, enabled)
                 VALUES (?1, ?2, ?3, ?4, 1)",
                params![id.0, name, key_hash, now],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        let record = TenantRecord {
            id,
            name: name.to_string(),
            enabled: true,
            created_at: now,
        };

        Ok((record, raw_key))
    }

    /// Resolve a tenant by hashing the provided API key and looking it up.
    pub fn resolve_by_key(&self, raw_key: &str) -> Result<TenantRecord> {
        let key_hash = hash_key(raw_key);
        self.conn
            .query_row(
                "SELECT id, name, enabled, created_at FROM tenants WHERE api_key_hash = ?1",
                params![key_hash],
                |row| {
                    let enabled: i64 = row.get(2)?;
                    Ok(TenantRecord {
                        id: TenantId(row.get(0)?),
                        name: row.get(1)?,
                        enabled: enabled != 0,
                        created_at: row.get(3)?,
                    })
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    SynapseError::TenantNotFound("Invalid API key".into())
                }
                other => SynapseError::StorageError(other.to_string()),
            })
    }

    /// List all tenants.
    pub fn list_tenants(&self) -> Result<Vec<TenantRecord>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, enabled, created_at FROM tenants ORDER BY created_at DESC")
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        let records = stmt
            .query_map([], |row| {
                let enabled: i64 = row.get(2)?;
                Ok(TenantRecord {
                    id: TenantId(row.get(0)?),
                    name: row.get(1)?,
                    enabled: enabled != 0,
                    created_at: row.get(3)?,
                })
            })
            .map_err(|e| SynapseError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        Ok(records)
    }

    /// Disable a tenant (soft delete).
    pub fn disable_tenant(&self, id: &TenantId) -> Result<()> {
        let rows = self
            .conn
            .execute(
                "UPDATE tenants SET enabled = 0 WHERE id = ?1",
                params![id.0],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        if rows == 0 {
            return Err(SynapseError::TenantNotFound(id.to_string()));
        }
        Ok(())
    }

    /// Enable a previously disabled tenant.
    pub fn enable_tenant(&self, id: &TenantId) -> Result<()> {
        let rows = self
            .conn
            .execute(
                "UPDATE tenants SET enabled = 1 WHERE id = ?1",
                params![id.0],
            )
            .map_err(|e| SynapseError::StorageError(e.to_string()))?;

        if rows == 0 {
            return Err(SynapseError::TenantNotFound(id.to_string()));
        }
        Ok(())
    }
}

/// Hash an API key with BLAKE3 for storage/lookup.
fn hash_key(key: &str) -> String {
    blake3::hash(key.as_bytes()).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_resolve_tenant() {
        let store = TenantStore::open_in_memory().unwrap();
        let (record, raw_key) = store.create_tenant("Acme Corp").unwrap();
        assert_eq!(record.name, "Acme Corp");
        assert!(record.enabled);
        assert!(raw_key.starts_with("syn_"));

        let resolved = store.resolve_by_key(&raw_key).unwrap();
        assert_eq!(resolved.id, record.id);
        assert_eq!(resolved.name, "Acme Corp");
        assert!(resolved.enabled);
    }

    #[test]
    fn resolve_invalid_key() {
        let store = TenantStore::open_in_memory().unwrap();
        let result = store.resolve_by_key("nonexistent");
        assert!(matches!(result, Err(SynapseError::TenantNotFound(_))));
    }

    #[test]
    fn list_tenants() {
        let store = TenantStore::open_in_memory().unwrap();
        assert!(store.list_tenants().unwrap().is_empty());

        store.create_tenant("Tenant A").unwrap();
        store.create_tenant("Tenant B").unwrap();
        assert_eq!(store.list_tenants().unwrap().len(), 2);
    }

    #[test]
    fn disable_and_enable_tenant() {
        let store = TenantStore::open_in_memory().unwrap();
        let (record, raw_key) = store.create_tenant("Test").unwrap();

        store.disable_tenant(&record.id).unwrap();
        let resolved = store.resolve_by_key(&raw_key).unwrap();
        assert!(!resolved.enabled);

        store.enable_tenant(&record.id).unwrap();
        let resolved = store.resolve_by_key(&raw_key).unwrap();
        assert!(resolved.enabled);
    }

    #[test]
    fn disable_nonexistent() {
        let store = TenantStore::open_in_memory().unwrap();
        let result = store.disable_tenant(&TenantId("nonexistent".into()));
        assert!(result.is_err());
    }

    #[test]
    fn hash_key_deterministic() {
        let h1 = hash_key("test-key");
        let h2 = hash_key("test-key");
        assert_eq!(h1, h2);

        let h3 = hash_key("different-key");
        assert_ne!(h1, h3);
    }

    #[test]
    fn create_tenant_key_format() {
        let store = TenantStore::open_in_memory().unwrap();
        let (_, raw_key) = store.create_tenant("Test").unwrap();
        assert!(raw_key.starts_with("syn_"));
        assert_eq!(raw_key.len(), 36); // "syn_" + 32 hex chars
    }
}
