use crate::types::error::Result;
use crate::types::rag::*;
use uuid::Uuid;

use super::chunker::chunk_text;
use super::store::{RagStore, cosine_similarity};

pub struct RagPipeline<'a> {
    store: &'a RagStore,
    pipeline_id: RagPipelineId,
    config: RagPipelineConfig,
}

impl<'a> RagPipeline<'a> {
    pub fn new(store: &'a RagStore, pipeline_id: RagPipelineId, config: RagPipelineConfig) -> Self {
        Self {
            store,
            pipeline_id,
            config,
        }
    }

    /// Ingest a document: chunk text, compute embeddings via provided function, store.
    pub fn ingest_document(
        &self,
        filename: &str,
        content: &str,
        embed_fn: impl Fn(&str) -> Vec<f32>,
    ) -> Result<DocumentInfo> {
        let chunks = chunk_text(content, self.config.chunk_size, self.config.chunk_overlap);
        let doc_id = Uuid::new_v4();
        let now = chrono::Utc::now();

        let doc = DocumentInfo {
            id: doc_id,
            pipeline_id: self.pipeline_id,
            filename: filename.to_string(),
            chunk_count: chunks.len(),
            ingested_at: now,
        };
        self.store.insert_document(&doc)?;

        for (i, chunk_content) in chunks.iter().enumerate() {
            let embedding = embed_fn(chunk_content);
            let chunk = ChunkInfo {
                id: Uuid::new_v4(),
                document_id: doc_id,
                content: chunk_content.clone(),
                embedding,
                position: i,
            };
            self.store.insert_chunk(&chunk)?;
        }

        Ok(doc)
    }

    /// Query: embed query, find similar chunks, return top-k.
    pub fn query(
        &self,
        query: &str,
        top_k: Option<usize>,
        embed_fn: impl Fn(&str) -> Vec<f32>,
    ) -> Result<Vec<RagSource>> {
        let query_embedding = embed_fn(query);
        let k = top_k.unwrap_or(self.config.similarity_top_k);

        let all_chunks = self.store.get_all_chunks_for_pipeline(self.pipeline_id)?;

        let mut scored: Vec<(f32, &ChunkInfo)> = all_chunks
            .iter()
            .map(|c| (cosine_similarity(&query_embedding, &c.embedding), c))
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(score, chunk)| RagSource {
                document_id: chunk.document_id,
                chunk_content: chunk.content.clone(),
                score,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::super::store::RagStore;
    use super::*;

    fn setup() -> (RagStore, RagPipelineId, RagPipelineConfig) {
        let store = RagStore::open_in_memory().unwrap();
        let pipeline_id = Uuid::new_v4();
        let config = RagPipelineConfig {
            name: "test-pipeline".into(),
            chunk_size: 50,
            chunk_overlap: 10,
            embedding_model: "mock".into(),
            similarity_top_k: 3,
        };
        let tenant = crate::types::TenantId::default_tenant();
        store
            .create_pipeline(pipeline_id, &config, &tenant)
            .unwrap();
        (store, pipeline_id, config)
    }

    /// Simple mock embedding: returns a vector based on the first char.
    fn mock_embed(text: &str) -> Vec<f32> {
        let first = text.bytes().next().unwrap_or(0) as f32;
        vec![first, first * 0.5, first * 0.1]
    }

    #[test]
    fn ingest_document_creates_chunks() {
        let (store, pipeline_id, config) = setup();
        let pipeline = RagPipeline::new(&store, pipeline_id, config);

        let content = "This is a test document with enough text to be split into multiple chunks by the chunker.";
        let doc = pipeline
            .ingest_document("test.txt", content, mock_embed)
            .unwrap();

        assert_eq!(doc.filename, "test.txt");
        assert!(doc.chunk_count > 0);

        let chunks = store.get_chunks_for_document(doc.id).unwrap();
        assert_eq!(chunks.len(), doc.chunk_count);
    }

    #[test]
    fn query_returns_ranked_results() {
        let (store, pipeline_id, config) = setup();
        let pipeline = RagPipeline::new(&store, pipeline_id, config);

        // Ingest with a deterministic embedding function
        let embed_fn = |text: &str| -> Vec<f32> {
            if text.contains("rust") {
                vec![1.0, 0.0, 0.0]
            } else if text.contains("python") {
                vec![0.0, 1.0, 0.0]
            } else {
                vec![0.5, 0.5, 0.0]
            }
        };

        // Insert documents with enough content to create chunks
        pipeline
            .ingest_document("rust.txt", "rust is great", embed_fn)
            .unwrap();
        pipeline
            .ingest_document("python.txt", "python is nice", embed_fn)
            .unwrap();

        // Query for "rust" - should rank rust chunk higher
        let query_embed = |_: &str| -> Vec<f32> { vec![1.0, 0.0, 0.0] };
        let results = pipeline.query("rust", None, query_embed).unwrap();

        assert!(!results.is_empty());
        // First result should be the rust chunk (highest similarity)
        assert!(results[0].chunk_content.contains("rust"));
    }

    #[test]
    fn query_respects_top_k() {
        let (store, pipeline_id, config) = setup();
        let pipeline = RagPipeline::new(&store, pipeline_id, config);

        // Ingest multiple short documents
        for i in 0..5 {
            pipeline
                .ingest_document(
                    &format!("doc{i}.txt"),
                    &format!("document number {i} content"),
                    mock_embed,
                )
                .unwrap();
        }

        let results = pipeline.query("test", Some(2), mock_embed).unwrap();
        assert!(results.len() <= 2);
    }

    #[test]
    fn empty_pipeline_returns_no_results() {
        let (store, pipeline_id, config) = setup();
        let pipeline = RagPipeline::new(&store, pipeline_id, config);

        let results = pipeline.query("anything", None, mock_embed).unwrap();
        assert!(results.is_empty());
    }
}
