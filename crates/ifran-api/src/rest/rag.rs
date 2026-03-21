//! REST handlers for RAG pipeline management.

use std::sync::Arc;

use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::Deserialize;
use ifran_backends::{InferenceBackend, ModelHandle, hash_text_to_embedding};
use ifran_types::rag::{RagPipelineConfig, RagPipelineId, RagQuery, RagSource};

use ifran_types::TenantId;

use super::pagination::{PaginatedResponse, PaginationQuery};
use crate::middleware::validation::validate_filename;
use crate::state::AppState;

/// Default embedding dimensionality when using the inference-backed embedder.
const EMBEDDING_DIMS: usize = 384;

/// Boxed embedding function returned by [`make_embed_fn`].
type EmbedFn = Box<dyn Fn(&str) -> Vec<f32> + Send>;

#[derive(Deserialize)]
pub struct IngestRequest {
    pub filename: String,
    pub content: String,
}

/// Fallback embedding function used when no inference backend is available.
/// Produces a deterministic, normalised vector from raw text.
fn fallback_embed(text: &str) -> Vec<f32> {
    hash_text_to_embedding(text, EMBEDDING_DIMS)
}

/// Resolve the embedding model's backend and handle from `AppState`.
///
/// Returns `None` when no matching model is loaded (callers fall back to
/// [`fallback_embed`]).
async fn resolve_embedding_backend(
    state: &AppState,
    embedding_model: &str,
    tenant_id: &TenantId,
) -> Option<(Arc<dyn InferenceBackend>, ModelHandle)> {
    let loaded = state.model_manager.list_loaded(Some(tenant_id)).await;
    let loaded_model = loaded.iter().find(|m| m.model_name == embedding_model)?;

    let backend = state.backends.get(&ifran_types::backend::BackendId(
        loaded_model.backend_id.clone(),
    ))?;

    let handle = ModelHandle(loaded_model.handle.clone());
    Some((backend, handle))
}

/// Build an embedding closure that delegates to a loaded inference backend.
///
/// When the backend and handle are available the closure calls
/// [`InferenceBackend::embed`] via `block_in_place` (safe because the
/// Axum handler runs on a multi-threaded Tokio runtime).  When no backend
/// is available it falls back to [`fallback_embed`].
fn make_embed_fn(backend: Option<(Arc<dyn InferenceBackend>, ModelHandle)>) -> EmbedFn {
    match backend {
        Some((backend, handle)) => {
            let rt = tokio::runtime::Handle::current();
            Box::new(move |text: &str| {
                let text = text.to_string();
                let backend = backend.clone();
                let handle = handle.clone();
                tokio::task::block_in_place(|| {
                    rt.block_on(async {
                        backend
                            .embed(&handle, &text, EMBEDDING_DIMS)
                            .await
                            .unwrap_or_else(|e| {
                                tracing::warn!(error = %e, "Backend embed failed, using fallback");
                                fallback_embed(&text)
                            })
                    })
                })
            })
        }
        None => {
            tracing::debug!("No embedding backend loaded, using fallback embedder");
            Box::new(|text: &str| fallback_embed(text))
        }
    }
}

/// POST /rag/pipelines
pub async fn create_pipeline(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Json(config): Json<RagPipelineConfig>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized. Check storage configuration and database path.".into(),
    ))?;

    let id = uuid::Uuid::new_v4();
    let s = store.lock().await;
    s.create_pipeline(id, &config, &tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "id": id.to_string(), "name": config.name })),
    ))
}

/// GET /rag/pipelines
pub async fn list_pipelines(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Query(page): Query<PaginationQuery>,
) -> Result<Json<PaginatedResponse<serde_json::Value>>, (StatusCode, String)> {
    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized. Check storage configuration and database path.".into(),
    ))?;

    let s = store.lock().await;
    let pipelines = s
        .list_pipelines(&tenant_id)
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

    Ok(Json(PaginatedResponse::from_slice(&data, &page, |item| {
        item.clone()
    })))
}

/// GET /rag/pipelines/{id}
pub async fn get_pipeline(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<RagPipelineId>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized. Check storage configuration and database path.".into(),
    ))?;

    let s = store.lock().await;
    let config = s.get_pipeline(id, &tenant_id).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            format!("{}. Use GET /rag/pipelines to list available pipelines.", e),
        )
    })?;

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
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<RagPipelineId>,
) -> Result<StatusCode, (StatusCode, String)> {
    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized. Check storage configuration and database path.".into(),
    ))?;

    let s = store.lock().await;
    s.delete_pipeline(id, &tenant_id).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            format!("{}. Use GET /rag/pipelines to list available pipelines.", e),
        )
    })?;

    Ok(StatusCode::NO_CONTENT)
}

/// POST /rag/pipelines/{id}/ingest
pub async fn ingest_document(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Path(id): Path<RagPipelineId>,
    Json(req): Json<IngestRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    validate_filename(&req.filename)?;

    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized. Check storage configuration and database path.".into(),
    ))?;

    // Peek at the pipeline config to learn the embedding model name *before*
    // we hold the store lock for the duration of ingestion.
    let embedding_model = {
        let s = store.lock().await;
        let config = s.get_pipeline(id, &tenant_id).map_err(|e| {
            (
                StatusCode::NOT_FOUND,
                format!("{}. Use GET /rag/pipelines to list available pipelines.", e),
            )
        })?;
        config.embedding_model.clone()
    };

    // Resolve backend outside the store lock so the async lookup doesn't
    // block other store operations.
    let backend_info = resolve_embedding_backend(&state, &embedding_model, &tenant_id).await;
    let embed_fn = make_embed_fn(backend_info);

    let s = store.lock().await;
    let config = s.get_pipeline(id, &tenant_id).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            format!("{}. Use GET /rag/pipelines to list available pipelines.", e),
        )
    })?;

    let pipeline = ifran_core::rag::pipeline::RagPipeline::new(&s, id, config);
    let doc = pipeline
        .ingest_document(&req.filename, &req.content, embed_fn.as_ref())
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
    Extension(tenant_id): Extension<TenantId>,
    Json(req): Json<RagQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = state.rag_store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "RAG store not initialized. Check storage configuration and database path.".into(),
    ))?;

    // Peek at the pipeline config to learn the embedding model name.
    let embedding_model = {
        let s = store.lock().await;
        let config = s.get_pipeline(req.pipeline_id, &tenant_id).map_err(|e| {
            (
                StatusCode::NOT_FOUND,
                format!("{}. Use GET /rag/pipelines to list available pipelines.", e),
            )
        })?;
        config.embedding_model.clone()
    };

    // Resolve backend outside the store lock.
    let backend_info = resolve_embedding_backend(&state, &embedding_model, &tenant_id).await;
    let embed_fn = make_embed_fn(backend_info);

    let s = store.lock().await;
    let config = s.get_pipeline(req.pipeline_id, &tenant_id).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            format!("{}. Use GET /rag/pipelines to list available pipelines.", e),
        )
    })?;

    let pipeline = ifran_core::rag::pipeline::RagPipeline::new(&s, req.pipeline_id, config);
    let sources: Vec<RagSource> = pipeline
        .query(&req.query, req.top_k, embed_fn.as_ref())
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
    fn fallback_embed_produces_normalized_vector() {
        let embedding = fallback_embed("hello world");
        assert_eq!(embedding.len(), EMBEDDING_DIMS);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn fallback_embed_different_texts_differ() {
        let a = fallback_embed("hello");
        let b = fallback_embed("world");
        assert_ne!(a, b);
    }

    #[test]
    fn fallback_embed_same_text_same_result() {
        let a = fallback_embed("hello");
        let b = fallback_embed("hello");
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
