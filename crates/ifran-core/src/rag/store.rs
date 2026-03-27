use ifran_types::IfranError;
use ifran_types::PagedResult;
use ifran_types::TenantId;
use ifran_types::error::Result;
use ifran_types::rag::{ChunkInfo, DocumentId, DocumentInfo, RagPipelineConfig, RagPipelineId};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use uuid::Uuid;

pub struct RagStore {
    pool: Pool<SqliteConnectionManager>,
}

impl RagStore {
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
            "CREATE TABLE IF NOT EXISTS rag_pipelines (
                    id          TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id          TEXT PRIMARY KEY,
                    pipeline_id TEXT NOT NULL,
                    filename    TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL,
                    FOREIGN KEY (pipeline_id) REFERENCES rag_pipelines(id)
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id          TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    embedding   BLOB NOT NULL,
                    position    INTEGER NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES rag_documents(id)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON rag_chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_documents_pipeline ON rag_documents(pipeline_id);",
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;

        // Add tenant_id column to rag_pipelines (idempotent — ignore if already exists)
        let _ = conn.execute_batch(
            "ALTER TABLE rag_pipelines ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default';",
        );
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_rag_pipelines_tenant ON rag_pipelines(tenant_id);",
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(())
    }

    pub fn create_pipeline(
        &self,
        id: RagPipelineId,
        config: &RagPipelineConfig,
        tenant_id: &TenantId,
    ) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let config_json =
            serde_json::to_string(config).map_err(|e| IfranError::StorageError(e.to_string()))?;
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO rag_pipelines (id, config_json, created_at, tenant_id) VALUES (?1, ?2, ?3, ?4)",
            params![id.to_string(), config_json, now, tenant_id.0],
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    pub fn get_pipeline(
        &self,
        id: RagPipelineId,
        tenant_id: &TenantId,
    ) -> Result<RagPipelineConfig> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT config_json FROM rag_pipelines WHERE id = ?1 AND tenant_id = ?2")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        stmt.query_row(params![id.to_string(), tenant_id.0], |row| {
            let json: String = row.get(0)?;
            serde_json::from_str(&json).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })
        .map_err(|e| IfranError::StorageError(e.to_string()))
    }

    pub fn list_pipelines(
        &self,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<(RagPipelineId, RagPipelineConfig)>> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let total: usize =
            conn.query_row(
                "SELECT COUNT(*) FROM rag_pipelines WHERE tenant_id = ?1",
                params![tenant_id.0],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))? as usize;

        let mut stmt = conn
            .prepare("SELECT id, config_json FROM rag_pipelines WHERE tenant_id = ?1 ORDER BY created_at DESC LIMIT ?2 OFFSET ?3")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let items = stmt
            .query_map(params![tenant_id.0, limit, offset], |row| {
                let id_str: String = row.get(0)?;
                let json: String = row.get(1)?;
                let id = Uuid::parse_str(&id_str).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
                let config: RagPipelineConfig = serde_json::from_str(&json).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        1,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
                Ok((id, config))
            })
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(PagedResult { items, total })
    }

    pub fn delete_pipeline(&self, id: RagPipelineId, tenant_id: &TenantId) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        // Verify the pipeline belongs to this tenant before cascading deletes
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM rag_pipelines WHERE id = ?1 AND tenant_id = ?2",
                params![id.to_string(), tenant_id.0],
                |row| row.get::<_, i64>(0).map(|c| c > 0),
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        if !exists {
            return Err(IfranError::StorageError(format!("Pipeline {id} not found")));
        }

        // Delete chunks for all documents in this pipeline
        conn.execute(
            "DELETE FROM rag_chunks WHERE document_id IN (SELECT id FROM rag_documents WHERE pipeline_id = ?1)",
            params![id.to_string()],
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
        // Delete documents
        conn.execute(
            "DELETE FROM rag_documents WHERE pipeline_id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
        // Delete pipeline
        conn.execute(
            "DELETE FROM rag_pipelines WHERE id = ?1 AND tenant_id = ?2",
            params![id.to_string(), tenant_id.0],
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    pub fn insert_document(&self, doc: &DocumentInfo) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.execute(
            "INSERT INTO rag_documents (id, pipeline_id, filename, chunk_count, ingested_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                doc.id.to_string(),
                doc.pipeline_id.to_string(),
                doc.filename,
                doc.chunk_count as i64,
                doc.ingested_at.to_rfc3339(),
            ],
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    pub fn insert_chunk(&self, chunk: &ChunkInfo) -> Result<()> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        // Serialize embedding as bytes (4 bytes per f32)
        let embedding_bytes: Vec<u8> = chunk
            .embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        conn.execute(
            "INSERT INTO rag_chunks (id, document_id, content, embedding, position)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                chunk.id.to_string(),
                chunk.document_id.to_string(),
                chunk.content,
                embedding_bytes,
                chunk.position as i64,
            ],
        )
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(())
    }

    pub fn get_chunks_for_document(&self, document_id: DocumentId) -> Result<Vec<ChunkInfo>> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, document_id, content, embedding, position FROM rag_chunks WHERE document_id = ?1 ORDER BY position",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map(params![document_id.to_string()], row_to_chunk)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(results)
    }

    /// Get all chunks for a pipeline (for similarity search).
    pub fn get_all_chunks_for_pipeline(
        &self,
        pipeline_id: RagPipelineId,
    ) -> Result<Vec<ChunkInfo>> {
        let conn = self
            .pool
            .get()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT c.id, c.document_id, c.content, c.embedding, c.position
                 FROM rag_chunks c
                 JOIN rag_documents d ON c.document_id = d.id
                 WHERE d.pipeline_id = ?1",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map(params![pipeline_id.to_string()], row_to_chunk)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(results)
    }
}

impl crate::storage::traits::RagStore for RagStore {
    fn create_pipeline(
        &self,
        id: RagPipelineId,
        config: &RagPipelineConfig,
        tenant_id: &TenantId,
    ) -> Result<()> {
        self.create_pipeline(id, config, tenant_id)
    }

    fn get_pipeline(&self, id: RagPipelineId, tenant_id: &TenantId) -> Result<RagPipelineConfig> {
        self.get_pipeline(id, tenant_id)
    }

    fn list_pipelines(
        &self,
        tenant_id: &TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<PagedResult<(RagPipelineId, RagPipelineConfig)>> {
        self.list_pipelines(tenant_id, limit, offset)
    }

    fn delete_pipeline(&self, id: RagPipelineId, tenant_id: &TenantId) -> Result<()> {
        self.delete_pipeline(id, tenant_id)
    }

    fn insert_document(&self, doc: &DocumentInfo) -> Result<()> {
        self.insert_document(doc)
    }

    fn insert_chunk(&self, chunk: &ChunkInfo) -> Result<()> {
        self.insert_chunk(chunk)
    }

    fn get_chunks_for_document(&self, document_id: DocumentId) -> Result<Vec<ChunkInfo>> {
        self.get_chunks_for_document(document_id)
    }

    fn get_all_chunks_for_pipeline(&self, pipeline_id: RagPipelineId) -> Result<Vec<ChunkInfo>> {
        self.get_all_chunks_for_pipeline(pipeline_id)
    }
}

fn row_to_chunk(row: &rusqlite::Row) -> rusqlite::Result<ChunkInfo> {
    let id_str: String = row.get(0)?;
    let doc_id_str: String = row.get(1)?;
    let content: String = row.get(2)?;
    let embedding_bytes: Vec<u8> = row.get(3)?;
    let position: i64 = row.get(4)?;

    let id = Uuid::parse_str(&id_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let doc_id = Uuid::parse_str(&doc_id_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(1, rusqlite::types::Type::Text, Box::new(e))
    })?;

    // Deserialize embedding from bytes
    let embedding: Vec<f32> = embedding_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(ChunkInfo {
        id,
        document_id: doc_id,
        content,
        embedding,
        position: position as usize,
    })
}

/// Compute cosine similarity between two vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> RagPipelineConfig {
        RagPipelineConfig {
            name: "test-pipeline".into(),
            chunk_size: 512,
            chunk_overlap: 64,
            embedding_model: "test-model".into(),
            similarity_top_k: 5,
        }
    }

    fn sample_document(pipeline_id: RagPipelineId) -> DocumentInfo {
        DocumentInfo {
            id: Uuid::new_v4(),
            pipeline_id,
            filename: "test.txt".into(),
            chunk_count: 3,
            ingested_at: chrono::Utc::now(),
        }
    }

    fn sample_chunk(document_id: DocumentId, position: usize) -> ChunkInfo {
        ChunkInfo {
            id: Uuid::new_v4(),
            document_id,
            content: format!("Chunk content at position {position}"),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            position,
        }
    }

    #[test]
    fn create_and_get_pipeline() {
        let store = RagStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let id = Uuid::new_v4();
        let config = sample_config();
        store.create_pipeline(id, &config, &tenant).unwrap();

        let got = store.get_pipeline(id, &tenant).unwrap();
        assert_eq!(got.name, config.name);
        assert_eq!(got.chunk_size, config.chunk_size);
        assert_eq!(got.embedding_model, config.embedding_model);
    }

    #[test]
    fn list_pipelines_empty() {
        let store = RagStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let list = store.list_pipelines(&tenant, 100, 0).unwrap();
        assert!(list.items.is_empty());
    }

    #[test]
    fn list_pipelines_multiple() {
        let store = RagStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        store
            .create_pipeline(id1, &sample_config(), &tenant)
            .unwrap();
        store
            .create_pipeline(id2, &sample_config(), &tenant)
            .unwrap();

        let list = store.list_pipelines(&tenant, 100, 0).unwrap();
        assert_eq!(list.items.len(), 2);
        assert_eq!(list.total, 2);
    }

    #[test]
    fn delete_pipeline_cascades() {
        let store = RagStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let pipeline_id = Uuid::new_v4();
        store
            .create_pipeline(pipeline_id, &sample_config(), &tenant)
            .unwrap();

        let doc = sample_document(pipeline_id);
        store.insert_document(&doc).unwrap();
        let chunk = sample_chunk(doc.id, 0);
        store.insert_chunk(&chunk).unwrap();

        store.delete_pipeline(pipeline_id, &tenant).unwrap();

        // Pipeline should be gone
        assert!(store.get_pipeline(pipeline_id, &tenant).is_err());
        // Chunks should be gone
        let chunks = store.get_chunks_for_document(doc.id).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn delete_pipeline_not_found() {
        let store = RagStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let result = store.delete_pipeline(Uuid::new_v4(), &tenant);
        assert!(result.is_err());
    }

    #[test]
    fn insert_document_and_chunks() {
        let store = RagStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let pipeline_id = Uuid::new_v4();
        store
            .create_pipeline(pipeline_id, &sample_config(), &tenant)
            .unwrap();

        let doc = sample_document(pipeline_id);
        store.insert_document(&doc).unwrap();

        let chunk0 = sample_chunk(doc.id, 0);
        let chunk1 = sample_chunk(doc.id, 1);
        let chunk2 = sample_chunk(doc.id, 2);
        store.insert_chunk(&chunk0).unwrap();
        store.insert_chunk(&chunk2).unwrap(); // Insert out of order
        store.insert_chunk(&chunk1).unwrap();

        let chunks = store.get_chunks_for_document(doc.id).unwrap();
        assert_eq!(chunks.len(), 3);
        // Should be ordered by position
        assert_eq!(chunks[0].position, 0);
        assert_eq!(chunks[1].position, 1);
        assert_eq!(chunks[2].position, 2);
    }

    #[test]
    fn get_all_chunks_for_pipeline() {
        let store = RagStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let pipeline_id = Uuid::new_v4();
        store
            .create_pipeline(pipeline_id, &sample_config(), &tenant)
            .unwrap();

        let doc1 = sample_document(pipeline_id);
        let doc2 = sample_document(pipeline_id);
        store.insert_document(&doc1).unwrap();
        store.insert_document(&doc2).unwrap();

        store.insert_chunk(&sample_chunk(doc1.id, 0)).unwrap();
        store.insert_chunk(&sample_chunk(doc1.id, 1)).unwrap();
        store.insert_chunk(&sample_chunk(doc2.id, 0)).unwrap();

        let all = store.get_all_chunks_for_pipeline(pipeline_id).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn embedding_blob_roundtrip() {
        let store = RagStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let pipeline_id = Uuid::new_v4();
        store
            .create_pipeline(pipeline_id, &sample_config(), &tenant)
            .unwrap();

        let doc = sample_document(pipeline_id);
        store.insert_document(&doc).unwrap();

        let original_embedding = vec![1.0_f32, -0.5, 0.0, 1.23456, f32::MIN, f32::MAX];
        let chunk = ChunkInfo {
            id: Uuid::new_v4(),
            document_id: doc.id,
            content: "test".into(),
            embedding: original_embedding.clone(),
            position: 0,
        };
        store.insert_chunk(&chunk).unwrap();

        let chunks = store.get_chunks_for_document(doc.id).unwrap();
        assert_eq!(chunks[0].embedding, original_embedding);
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn cosine_similarity_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }
}
