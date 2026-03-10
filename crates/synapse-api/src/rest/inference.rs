//! REST handlers for inference requests (generate with optional streaming).

use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use serde::Deserialize;
use synapse_types::inference::InferenceRequest;

/// POST /inference request body.
#[derive(Debug, Deserialize)]
pub struct InferenceBody {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> u32 {
    512
}

/// POST /inference — run inference, returning the full response.
pub async fn inference(
    State(state): State<AppState>,
    Json(body): Json<InferenceBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let loaded = state.model_manager.list_loaded().await;
    let loaded_model = loaded.iter().find(|m| m.backend_id == "llamacpp").ok_or((
        StatusCode::BAD_REQUEST,
        "No model loaded. Load a model first.".into(),
    ))?;

    let backend = state
        .backends
        .get(&synapse_types::backend::BackendId("llamacpp".into()))
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Backend not available".into(),
        ))?;

    let handle = synapse_backends::ModelHandle(loaded_model.handle.clone());
    let req = InferenceRequest {
        prompt: body.prompt,
        max_tokens: Some(body.max_tokens),
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: body.top_k,
        stop_sequences: None,
        system_prompt: body.system_prompt,
    };

    let resp = backend
        .infer(&handle, &req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({
        "text": resp.text,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
        "finish_reason": format!("{:?}", resp.finish_reason).to_lowercase(),
    })))
}

/// POST /inference/stream — run inference with SSE streaming.
pub async fn inference_stream(
    State(state): State<AppState>,
    Json(body): Json<InferenceBody>,
) -> Result<
    Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>,
    (StatusCode, String),
> {
    let loaded = state.model_manager.list_loaded().await;
    let loaded_model = loaded
        .iter()
        .find(|m| m.backend_id == "llamacpp")
        .ok_or((StatusCode::BAD_REQUEST, "No model loaded".into()))?;

    let backend = state
        .backends
        .get(&synapse_types::backend::BackendId("llamacpp".into()))
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Backend not available".into(),
        ))?;

    let handle = synapse_backends::ModelHandle(loaded_model.handle.clone());
    let req = InferenceRequest {
        prompt: body.prompt,
        max_tokens: Some(body.max_tokens),
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: body.top_k,
        stop_sequences: None,
        system_prompt: body.system_prompt,
    };

    let mut rx = backend
        .infer_stream(&handle, req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let stream = async_stream::stream! {
        while let Some(chunk) = rx.recv().await {
            if chunk.done {
                yield Ok(Event::default().data("[DONE]"));
                break;
            }
            let data = serde_json::json!({ "text": chunk.text }).to_string();
            yield Ok(Event::default().data(data));
        }
    };

    Ok(Sse::new(stream))
}
