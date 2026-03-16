//! REST handlers for RAG pipeline management.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::Deserialize;
use synapse_types::rag::{RagPipelineConfig, RagPipelineId, RagQuery, RagSource};

use synapse_types::TenantId;

use crate::middleware::validation::validate_filename;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct IngestRequest {
    pub filename: String,
    pub content: String,
}

/// Stub embedding function — produces a deterministic vector from text.
/// In production, this would call an actual embedding model.
fn stub_embed(text: &str) -> Vec<f32> {
    // Simple hash-based embedding for development/testing
    let mut embedding = vec![0.0f32; 64];
    for (i, byte) in text.bytes().enumerate() {
        embedding[i % 64] += byte as f32 / 255.0;
    }
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut embedding {
            *v /= norm;
        }
    }
    embedding
}

/// POST /rag/pipelines
pub async fn create_pipeline(
    State(state): State<AppState>,
    Json(config): Json<RagPipelineConfig>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized".into(),
    ))?;

    let id = uuid::Uuid::new_v4();
    let s = store.lock().await;
    let tenant = TenantId::default_tenant();
    s.create_pipeline(id, &config, &tenant)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "id": id.to_string(), "name": config.name })),
    ))
}

/// GET /rag/pipelines
pub async fn list_pipelines(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized".into(),
    ))?;

    let s = store.lock().await;
    let tenant = TenantId::default_tenant();
    let pipelines = s
        .list_pipelines(&tenant)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let data: Vec<serde_json::Value> = pipelines
        .iter()
        .map(|(id, config)| {
            serde_json::json!({
                "id": id.to_string(),
                "name": config.name,
                "chunk_size": config.chunk_size,
                "embedding_model": config.embedding_model,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({ "data": data })))
}

/// GET /rag/pipelines/{id}
pub async fn get_pipeline(
    State(state): State<AppState>,
    Path(id): Path<RagPipelineId>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized".into(),
    ))?;

    let s = store.lock().await;
    let tenant = TenantId::default_tenant();
    let config = s
        .get_pipeline(id, &tenant)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(serde_json::json!({
        "id": id.to_string(),
        "name": config.name,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embedding_model": config.embedding_model,
        "similarity_top_k": config.similarity_top_k,
    })))
}

/// DELETE /rag/pipelines/{id}
pub async fn delete_pipeline(
    State(state): State<AppState>,
    Path(id): Path<RagPipelineId>,
) -> Result<StatusCode, (StatusCode, String)> {
    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized".into(),
    ))?;

    let s = store.lock().await;
    let tenant = TenantId::default_tenant();
    s.delete_pipeline(id, &tenant)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

/// POST /rag/pipelines/{id}/ingest
pub async fn ingest_document(
    State(state): State<AppState>,
    Path(id): Path<RagPipelineId>,
    Json(req): Json<IngestRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    validate_filename(&req.filename)?;

    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized".into(),
    ))?;

    let s = store.lock().await;
    let tenant = TenantId::default_tenant();
    let config = s
        .get_pipeline(id, &tenant)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let pipeline = synapse_core::rag::pipeline::RagPipeline::new(&s, id, config);
    let doc = pipeline
        .ingest_document(&req.filename, &req.content, stub_embed)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "document_id": doc.id.to_string(),
            "chunks": doc.chunk_count,
        })),
    ))
}

/// POST /rag/query
pub async fn query(
    State(state): State<AppState>,
    Json(req): Json<RagQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized".into(),
    ))?;

    let s = store.lock().await;
    let tenant = TenantId::default_tenant();
    let config = s
        .get_pipeline(req.pipeline_id, &tenant)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let pipeline = synapse_core::rag::pipeline::RagPipeline::new(&s, req.pipeline_id, config);
    let sources: Vec<RagSource> = pipeline
        .query(&req.query, req.top_k, stub_embed)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let source_json: Vec<serde_json::Value> = sources
        .iter()
        .map(|s| {
            serde_json::json!({
                "document_id": s.document_id.to_string(),
                "chunk_content": s.chunk_content,
                "score": s.score,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "sources": source_json,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_request_deserialize() {
        let json = r#"{"filename": "doc.txt", "content": "hello world"}"#;
        let req: IngestRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.filename, "doc.txt");
        assert_eq!(req.content, "hello world");
    }

    #[test]
    fn stub_embed_produces_normalized_vector() {
        let embedding = stub_embed("hello world");
        assert_eq!(embedding.len(), 64);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn stub_embed_different_texts_differ() {
        let a = stub_embed("hello");
        let b = stub_embed("world");
        assert_ne!(a, b);
    }

    #[test]
    fn stub_embed_same_text_same_result() {
        let a = stub_embed("hello");
        let b = stub_embed("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn rag_query_deserialize() {
        let json = r#"{"query": "what is rust?", "pipeline_id": "00000000-0000-0000-0000-000000000001", "top_k": 3, "include_sources": true}"#;
        let q: RagQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.query, "what is rust?");
        assert_eq!(q.top_k, Some(3));
        assert!(q.include_sources);
    }

    #[test]
    fn rag_pipeline_config_deserialize_with_defaults() {
        let json = r#"{"name": "test", "embedding_model": "nomic-embed"}"#;
        let config: RagPipelineConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.name, "test");
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.chunk_overlap, 64);
        assert_eq!(config.similarity_top_k, 5);
    }
}
