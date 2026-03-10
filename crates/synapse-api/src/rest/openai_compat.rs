//! OpenAI-compatible API endpoints for drop-in replacement compatibility.
//!
//! Implements the subset of the OpenAI API used by most clients:
//! - POST /v1/chat/completions
//! - GET /v1/models

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::Json;
use crate::state::AppState;
use serde::Deserialize;
use synapse_types::inference::InferenceRequest;

/// OpenAI-style message.
#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// POST /v1/chat/completions request body.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> u32 {
    512
}

/// GET /v1/models — list models in OpenAI format.
pub async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let db = state.db.lock().await;
    let models = db.list().unwrap_or_default();

    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.name,
                "object": "model",
                "created": m.pulled_at.timestamp(),
                "owned_by": "synapse",
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": data,
    }))
}

/// POST /v1/chat/completions — OpenAI-compatible chat completions.
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let loaded = state.model_manager.list_loaded().await;
    let loaded_model = loaded
        .first()
        .ok_or((StatusCode::BAD_REQUEST, "No model loaded".into()))?;

    let backend = state
        .backends
        .get(&synapse_types::backend::BackendId(loaded_model.backend_id.clone()))
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "Backend not available".into()))?;

    let handle = synapse_backends::ModelHandle(loaded_model.handle.clone());

    // Extract system prompt and user messages
    let system_prompt = body
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.clone());

    let prompt = body
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let req = InferenceRequest {
        prompt,
        max_tokens: Some(body.max_tokens),
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: None,
        stop_sequences: None,
        system_prompt,
    };

    if body.stream {
        return stream_response(backend, handle, req, &body.model).await;
    }

    // Non-streaming response
    let resp = backend
        .infer(&handle, &req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let json = serde_json::json!({
        "id": response_id,
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": resp.text,
            },
            "finish_reason": match resp.finish_reason {
                synapse_types::inference::FinishReason::Stop => "stop",
                synapse_types::inference::FinishReason::MaxTokens => "length",
                synapse_types::inference::FinishReason::StopSequence => "stop",
            },
        }],
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
    });

    Ok(Json(json).into_response())
}

async fn stream_response(
    backend: std::sync::Arc<dyn synapse_backends::InferenceBackend>,
    handle: synapse_backends::ModelHandle,
    req: InferenceRequest,
    model: &str,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let mut rx = backend
        .infer_stream(&handle, req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let model = model.to_string();
    let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let stream = async_stream::stream! {
        while let Some(chunk) = rx.recv().await {
            if chunk.done {
                let data = serde_json::json!({
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                });
                yield Ok(Event::default().data(data.to_string()));
                yield Ok(Event::default().data("[DONE]"));
                break;
            }
            let data = serde_json::json!({
                "id": response_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk.text}, "finish_reason": serde_json::Value::Null}],
            });
            yield Ok::<_, std::convert::Infallible>(Event::default().data(data.to_string()));
        }
    };

    Ok(Sse::new(stream).into_response())
}
