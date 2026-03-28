use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type RagPipelineId = Uuid;
pub type DocumentId = Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagPipelineConfig {
    pub name: String,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    pub embedding_model: String,
    #[serde(default = "default_similarity_top_k")]
    pub similarity_top_k: usize,
}

fn default_chunk_size() -> usize {
    512
}
fn default_chunk_overlap() -> usize {
    64
}
fn default_similarity_top_k() -> usize {
    5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInfo {
    pub id: DocumentId,
    pub pipeline_id: RagPipelineId,
    pub filename: String,
    pub chunk_count: usize,
    pub ingested_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    pub id: Uuid,
    pub document_id: DocumentId,
    pub content: String,
    pub embedding: Vec<f32>,
    pub position: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagQuery {
    pub query: String,
    pub pipeline_id: RagPipelineId,
    pub top_k: Option<usize>,
    #[serde(default = "default_include_sources")]
    pub include_sources: bool,
}

fn default_include_sources() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagResult {
    pub answer: String,
    pub sources: Vec<RagSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagSource {
    pub document_id: DocumentId,
    pub chunk_content: String,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rag_pipeline_config_serde_roundtrip() {
        let config = RagPipelineConfig {
            name: "test-pipeline".into(),
            chunk_size: 1024,
            chunk_overlap: 128,
            embedding_model: "text-embedding-3-small".into(),
            similarity_top_k: 10,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: RagPipelineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.name, back.name);
        assert_eq!(config.chunk_size, back.chunk_size);
        assert_eq!(config.chunk_overlap, back.chunk_overlap);
        assert_eq!(config.embedding_model, back.embedding_model);
        assert_eq!(config.similarity_top_k, back.similarity_top_k);
    }

    #[test]
    fn rag_pipeline_config_defaults() {
        let json = r#"{"name":"minimal","embedding_model":"test-model"}"#;
        let config: RagPipelineConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.name, "minimal");
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.chunk_overlap, 64);
        assert_eq!(config.similarity_top_k, 5);
        assert_eq!(config.embedding_model, "test-model");
    }

    #[test]
    fn document_info_serde_roundtrip() {
        let doc = DocumentInfo {
            id: Uuid::new_v4(),
            pipeline_id: Uuid::new_v4(),
            filename: "test.pdf".into(),
            chunk_count: 42,
            ingested_at: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&doc).unwrap();
        let back: DocumentInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(doc.id, back.id);
        assert_eq!(doc.pipeline_id, back.pipeline_id);
        assert_eq!(doc.filename, back.filename);
        assert_eq!(doc.chunk_count, back.chunk_count);
    }

    #[test]
    fn chunk_info_serde_roundtrip() {
        let chunk = ChunkInfo {
            id: Uuid::new_v4(),
            document_id: Uuid::new_v4(),
            content: "Hello world".into(),
            embedding: vec![0.1, 0.2, 0.3],
            position: 0,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let back: ChunkInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk.id, back.id);
        assert_eq!(chunk.document_id, back.document_id);
        assert_eq!(chunk.content, back.content);
        assert_eq!(chunk.embedding, back.embedding);
        assert_eq!(chunk.position, back.position);
    }

    #[test]
    fn rag_query_serde_roundtrip_with_top_k() {
        let query = RagQuery {
            query: "What is Rust?".into(),
            pipeline_id: Uuid::new_v4(),
            top_k: Some(10),
            include_sources: true,
        };
        let json = serde_json::to_string(&query).unwrap();
        let back: RagQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(query.query, back.query);
        assert_eq!(query.pipeline_id, back.pipeline_id);
        assert_eq!(query.top_k, back.top_k);
        assert_eq!(query.include_sources, back.include_sources);
    }

    #[test]
    fn rag_query_serde_roundtrip_without_top_k() {
        let id = Uuid::new_v4();
        let json = format!(r#"{{"query":"test","pipeline_id":"{}"}}"#, id);
        let back: RagQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(back.query, "test");
        assert_eq!(back.pipeline_id, id);
        assert!(back.top_k.is_none());
        assert!(back.include_sources); // default
    }

    #[test]
    fn rag_result_serde_roundtrip() {
        let result = RagResult {
            answer: "Rust is a systems programming language.".into(),
            sources: vec![
                RagSource {
                    document_id: Uuid::new_v4(),
                    chunk_content: "Rust is a systems language...".into(),
                    score: 0.95,
                },
                RagSource {
                    document_id: Uuid::new_v4(),
                    chunk_content: "Rust focuses on safety...".into(),
                    score: 0.87,
                },
            ],
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: RagResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.answer, back.answer);
        assert_eq!(result.sources.len(), back.sources.len());
        assert_eq!(result.sources[0].score, back.sources[0].score);
        assert_eq!(
            result.sources[1].chunk_content,
            back.sources[1].chunk_content
        );
    }

    #[test]
    fn default_values_validation() {
        assert_eq!(default_chunk_size(), 512);
        assert_eq!(default_chunk_overlap(), 64);
        assert_eq!(default_similarity_top_k(), 5);
        assert!(default_include_sources());
    }
}
