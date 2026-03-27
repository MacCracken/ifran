//! SQLite storage for pipeline lineage graphs.

use ifran_types::error::Result;
use ifran_types::lineage::{LineageId, LineageNode, PipelineStage};
use ifran_types::{IfranError, PagedResult, TenantId};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use uuid::Uuid;

pub struct LineageStore {
    pool: Pool<SqliteConnectionManager>,
}

impl LineageStore {
    pub fn open(path: &std::path::Path) -> Result<Self> {
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
        let store = Self { pool };
        store.migrate()?;
        Ok(store)
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let manager = SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
            Ok(())
        });
        let pool = Pool::builder()
            .max_size(1)
            .build(manager)
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let store = Self { pool };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS lineage_nodes (
                id           TEXT PRIMARY KEY,
                tenant_id    TEXT NOT NULL DEFAULT 'default',
                stage        TEXT NOT NULL,
                name         TEXT NOT NULL,
                artifact_ref TEXT NOT NULL,
                metadata     TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lineage_edges (
                parent_id TEXT NOT NULL,
                child_id  TEXT NOT NULL,
                PRIMARY KEY (parent_id, child_id),
                FOREIGN KEY (parent_id) REFERENCES lineage_nodes(id),
                FOREIGN KEY (child_id) REFERENCES lineage_nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_lineage_tenant ON lineage_nodes(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_lineage_stage ON lineage_nodes(stage);
            CREATE INDEX IF NOT EXISTS idx_lineage_artifact ON lineage_nodes(artifact_ref);
            CREATE INDEX IF NOT EXISTS idx_edges_child ON lineage_edges(child_id);",
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Record a new lineage node with its parent relationships.
    pub fn record(&self, node: &LineageNode, tenant_id: &TenantId) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let metadata_str = node.metadata.to_string();
        conn.execute(
            "INSERT INTO lineage_nodes (id, tenant_id, stage, name, artifact_ref, metadata, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                node.id.to_string(),
                tenant_id.0,
                node.stage.to_string(),
                node.name,
                node.artifact_ref,
                metadata_str,
                node.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;

        for parent_id in &node.parent_ids {
            conn.execute(
                "INSERT INTO lineage_edges (parent_id, child_id) VALUES (?1, ?2)",
                params![parent_id.to_string(), node.id.to_string()],
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        }

        Ok(())
    }

    /// Get a lineage node by ID.
    pub fn get(&self, id: LineageId, tenant_id: &TenantId) -> Result<LineageNode> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let node = conn
            .query_row(
                "SELECT id, stage, name, artifact_ref, metadata, created_at
             FROM lineage_nodes WHERE id = ?1 AND tenant_id = ?2",
                params![id.to_string(), tenant_id.0],
                |row| {
                    let id_str: String = row.get(0)?;
                    let stage_str: String = row.get(1)?;
                    let metadata_str: String = row.get(4)?;
                    let created_str: String = row.get(5)?;
                    Ok((
                        id_str,
                        stage_str,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        metadata_str,
                        created_str,
                    ))
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    IfranError::StorageError(format!("Lineage node {id} not found"))
                }
                other => IfranError::StorageError(other.to_string()),
            })?;

        let parent_ids = Self::get_parent_ids_with_conn(&conn, id)?;

        Ok(LineageNode {
            id: Uuid::parse_str(&node.0).unwrap_or_default(),
            stage: serde_json::from_str(&format!("\"{}\"", node.1))
                .unwrap_or(PipelineStage::Dataset),
            name: node.2,
            artifact_ref: node.3,
            parent_ids,
            metadata: serde_json::from_str(&node.4).unwrap_or_default(),
            created_at: chrono::DateTime::parse_from_rfc3339(&node.5)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(|_| chrono::Utc::now()),
        })
    }

    fn get_parent_ids_with_conn(
        conn: &rusqlite::Connection,
        child_id: LineageId,
    ) -> Result<Vec<LineageId>> {
        let mut stmt = conn
            .prepare("SELECT parent_id FROM lineage_edges WHERE child_id = ?1")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let ids = stmt
            .query_map(params![child_id.to_string()], |row| {
                let id_str: String = row.get(0)?;
                Ok(id_str)
            })
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|s| Uuid::parse_str(&s).ok())
            .collect();

        Ok(ids)
    }

    /// Default maximum ancestry traversal depth.
    pub const DEFAULT_MAX_ANCESTRY_DEPTH: u32 = 10_000;

    /// Get the ancestry chain for a node (walk up the graph).
    ///
    /// `max_depth` limits how many levels to traverse. `None` uses the
    /// default limit of 10,000 nodes to prevent OOM on deep/wide DAGs.
    pub fn get_ancestry(
        &self,
        id: LineageId,
        tenant_id: &TenantId,
        max_depth: Option<u32>,
    ) -> Result<Vec<LineageNode>> {
        let limit = max_depth.unwrap_or(Self::DEFAULT_MAX_ANCESTRY_DEPTH) as usize;
        let mut result = Vec::new();
        let mut to_visit = vec![id];
        let mut visited = std::collections::HashSet::new();

        while let Some(current) = to_visit.pop() {
            if visited.len() >= limit {
                break;
            }
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            match self.get(current, tenant_id) {
                Ok(node) => {
                    to_visit.extend(&node.parent_ids);
                    result.push(node);
                }
                Err(_) => continue,
            }
        }

        Ok(result)
    }

    /// List all lineage nodes for a tenant, optionally filtered by stage.
    pub fn list(
        &self,
        tenant_id: &TenantId,
        stage: Option<PipelineStage>,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<LineageNode>> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let total: usize = match stage {
            Some(ref s) => conn.query_row(
                "SELECT COUNT(*) FROM lineage_nodes WHERE tenant_id = ?1 AND stage = ?2",
                params![tenant_id.0, s.to_string()],
                |row| row.get::<_, i64>(0),
            ),
            None => conn.query_row(
                "SELECT COUNT(*) FROM lineage_nodes WHERE tenant_id = ?1",
                params![tenant_id.0],
                |row| row.get::<_, i64>(0),
            ),
        }
        .map(|c| c as usize)
        .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let (sql, query_params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match stage {
            Some(s) => (
                "SELECT id FROM lineage_nodes WHERE tenant_id = ?1 AND stage = ?2 ORDER BY created_at DESC LIMIT ?3 OFFSET ?4".into(),
                vec![Box::new(tenant_id.0.clone()), Box::new(s.to_string()), Box::new(limit), Box::new(offset)],
            ),
            None => (
                "SELECT id FROM lineage_nodes WHERE tenant_id = ?1 ORDER BY created_at DESC LIMIT ?2 OFFSET ?3"
                    .into(),
                vec![Box::new(tenant_id.0.clone()), Box::new(limit), Box::new(offset)],
            ),
        };

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            query_params.iter().map(|p| p.as_ref()).collect();

        let ids: Vec<LineageId> = stmt
            .query_map(param_refs.as_slice(), |row| {
                let id_str: String = row.get(0)?;
                Ok(id_str)
            })
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|s| Uuid::parse_str(&s).ok())
            .collect();

        drop(stmt);
        drop(conn);

        let mut items = Vec::new();
        for id in ids {
            if let Ok(node) = self.get(id, tenant_id) {
                items.push(node);
            }
        }
        Ok(PagedResult { items, total })
    }

    /// Find lineage nodes by artifact reference.
    pub fn find_by_artifact(
        &self,
        artifact_ref: &str,
        tenant_id: &TenantId,
    ) -> Result<Vec<LineageNode>> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT id FROM lineage_nodes WHERE artifact_ref = ?1 AND tenant_id = ?2")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let ids: Vec<LineageId> = stmt
            .query_map(params![artifact_ref, tenant_id.0], |row| {
                let id_str: String = row.get(0)?;
                Ok(id_str)
            })
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .filter_map(|r| r.ok())
            .filter_map(|s| Uuid::parse_str(&s).ok())
            .collect();

        drop(stmt);
        drop(conn);

        let mut nodes = Vec::new();
        for id in ids {
            if let Ok(node) = self.get(id, tenant_id) {
                nodes.push(node);
            }
        }
        Ok(nodes)
    }
}

impl crate::storage::traits::LineageStore for LineageStore {
    fn record(&self, node: &LineageNode, tenant_id: &TenantId) -> Result<()> {
        self.record(node, tenant_id)
    }

    fn get(&self, id: Uuid, tenant_id: &TenantId) -> Result<LineageNode> {
        self.get(id, tenant_id)
    }

    fn get_ancestry(
        &self,
        id: Uuid,
        tenant_id: &TenantId,
        max_depth: Option<u32>,
    ) -> Result<Vec<LineageNode>> {
        self.get_ancestry(id, tenant_id, max_depth)
    }

    fn list(
        &self,
        tenant_id: &TenantId,
        stage: Option<PipelineStage>,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<LineageNode>> {
        self.list(tenant_id, stage, limit, offset)
    }

    fn find_by_artifact(
        &self,
        artifact_ref: &str,
        tenant_id: &TenantId,
    ) -> Result<Vec<LineageNode>> {
        self.find_by_artifact(artifact_ref, tenant_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ifran_types::lineage::PipelineStage;

    fn default_tenant() -> TenantId {
        TenantId::default_tenant()
    }

    fn make_node(
        stage: PipelineStage,
        name: &str,
        artifact: &str,
        parents: Vec<LineageId>,
    ) -> LineageNode {
        LineageNode {
            id: Uuid::new_v4(),
            stage,
            name: name.into(),
            artifact_ref: artifact.into(),
            parent_ids: parents,
            metadata: serde_json::json!({}),
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn record_and_get() {
        let store = LineageStore::open_in_memory().unwrap();
        let node = make_node(
            PipelineStage::Dataset,
            "train.jsonl",
            "/data/train.jsonl",
            vec![],
        );
        store.record(&node, &default_tenant()).unwrap();

        let fetched = store.get(node.id, &default_tenant()).unwrap();
        assert_eq!(fetched.name, "train.jsonl");
        assert_eq!(fetched.stage, PipelineStage::Dataset);
        assert!(fetched.parent_ids.is_empty());
    }

    #[test]
    fn record_with_parents() {
        let store = LineageStore::open_in_memory().unwrap();
        let dataset = make_node(PipelineStage::Dataset, "data", "/data", vec![]);
        store.record(&dataset, &default_tenant()).unwrap();

        let training = make_node(PipelineStage::Training, "lora-1", "job-1", vec![dataset.id]);
        store.record(&training, &default_tenant()).unwrap();

        let fetched = store.get(training.id, &default_tenant()).unwrap();
        assert_eq!(fetched.parent_ids.len(), 1);
        assert_eq!(fetched.parent_ids[0], dataset.id);
    }

    #[test]
    fn get_ancestry() {
        let store = LineageStore::open_in_memory().unwrap();
        let t = default_tenant();

        let dataset = make_node(PipelineStage::Dataset, "data", "/data", vec![]);
        store.record(&dataset, &t).unwrap();

        let training = make_node(PipelineStage::Training, "train", "job-1", vec![dataset.id]);
        store.record(&training, &t).unwrap();

        let eval = make_node(
            PipelineStage::Evaluation,
            "eval",
            "eval-1",
            vec![training.id],
        );
        store.record(&eval, &t).unwrap();

        let ancestry = store.get_ancestry(eval.id, &t, None).unwrap();
        assert_eq!(ancestry.len(), 3); // eval -> training -> dataset
    }

    #[test]
    fn list_by_stage() {
        let store = LineageStore::open_in_memory().unwrap();
        let t = default_tenant();

        store
            .record(&make_node(PipelineStage::Dataset, "d1", "d1", vec![]), &t)
            .unwrap();
        store
            .record(&make_node(PipelineStage::Training, "t1", "t1", vec![]), &t)
            .unwrap();
        store
            .record(&make_node(PipelineStage::Dataset, "d2", "d2", vec![]), &t)
            .unwrap();

        let datasets = store
            .list(&t, Some(PipelineStage::Dataset), 100, 0)
            .unwrap();
        assert_eq!(datasets.items.len(), 2);
        assert_eq!(datasets.total, 2);

        let all = store.list(&t, None, 100, 0).unwrap();
        assert_eq!(all.items.len(), 3);
        assert_eq!(all.total, 3);
    }

    #[test]
    fn find_by_artifact() {
        let store = LineageStore::open_in_memory().unwrap();
        let t = default_tenant();

        store
            .record(
                &make_node(PipelineStage::Training, "run", "job-42", vec![]),
                &t,
            )
            .unwrap();
        let found = store.find_by_artifact("job-42", &t).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].artifact_ref, "job-42");

        let not_found = store.find_by_artifact("nonexistent", &t).unwrap();
        assert!(not_found.is_empty());
    }

    #[test]
    fn tenant_isolation() {
        let store = LineageStore::open_in_memory().unwrap();
        let t1 = TenantId("tenant-1".into());
        let t2 = TenantId("tenant-2".into());

        store
            .record(&make_node(PipelineStage::Dataset, "d1", "d1", vec![]), &t1)
            .unwrap();
        store
            .record(&make_node(PipelineStage::Dataset, "d2", "d2", vec![]), &t2)
            .unwrap();

        assert_eq!(store.list(&t1, None, 100, 0).unwrap().items.len(), 1);
        assert_eq!(store.list(&t2, None, 100, 0).unwrap().items.len(), 1);
    }

    #[test]
    fn get_not_found() {
        let store = LineageStore::open_in_memory().unwrap();
        let result = store.get(Uuid::new_v4(), &default_tenant());
        assert!(result.is_err());
    }

    #[test]
    fn record_duplicate_id_fails() {
        let store = LineageStore::open_in_memory().unwrap();
        let node = make_node(PipelineStage::Dataset, "d1", "/d1", vec![]);
        store.record(&node, &default_tenant()).unwrap();
        // Inserting the same node again should fail (PRIMARY KEY constraint).
        let result = store.record(&node, &default_tenant());
        assert!(result.is_err());
    }

    #[test]
    fn get_ancestry_with_diamond() {
        let store = LineageStore::open_in_memory().unwrap();
        let t = default_tenant();

        // A -> B, A -> C, B -> D, C -> D  (diamond shape)
        let a = make_node(PipelineStage::Dataset, "A", "a", vec![]);
        store.record(&a, &t).unwrap();

        let b = make_node(PipelineStage::Training, "B", "b", vec![a.id]);
        store.record(&b, &t).unwrap();

        let c = make_node(PipelineStage::Training, "C", "c", vec![a.id]);
        store.record(&c, &t).unwrap();

        let d = make_node(PipelineStage::Evaluation, "D", "d", vec![b.id, c.id]);
        store.record(&d, &t).unwrap();

        let ancestry = store.get_ancestry(d.id, &t, None).unwrap();
        // D, B, C, A — all 4 nodes reachable, each visited once.
        assert_eq!(ancestry.len(), 4);
    }

    #[test]
    fn find_by_artifact_multiple_matches() {
        let store = LineageStore::open_in_memory().unwrap();
        let t = default_tenant();

        let shared_ref = "shared-artifact";
        store
            .record(
                &make_node(PipelineStage::Training, "run-1", shared_ref, vec![]),
                &t,
            )
            .unwrap();
        store
            .record(
                &make_node(PipelineStage::Training, "run-2", shared_ref, vec![]),
                &t,
            )
            .unwrap();
        store
            .record(
                &make_node(PipelineStage::Training, "run-3", "other-artifact", vec![]),
                &t,
            )
            .unwrap();

        let found = store.find_by_artifact(shared_ref, &t).unwrap();
        assert_eq!(found.len(), 2);
    }
}
