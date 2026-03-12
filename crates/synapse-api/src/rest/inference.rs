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
    let loaded_model = loaded
        .iter()
        .find(|m| m.model_name == body.model)
        .or_else(|| loaded.first())
        .ok_or((
            StatusCode::BAD_REQUEST,
            "No model loaded. Load a model first.".into(),
        ))?;

    let backend = state
        .backends
        .get(&synapse_types::backend::BackendId(
            loaded_model.backend_id.clone(),
        ))
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
        .find(|m| m.model_name == body.model)
        .or_else(|| loaded.first())
        .ok_or((StatusCode::BAD_REQUEST, "No model loaded".into()))?;

    let backend = state
        .backends
        .get(&synapse_types::backend::BackendId(
            loaded_model.backend_id.clone(),
        ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use synapse_core::config::*;

    fn test_state(tmp: &tempfile::TempDir) -> AppState {
        let config = SynapseConfig {
            server: ServerConfig {
                bind: "127.0.0.1:0".into(),
                grpc_bind: "127.0.0.1:0".into(),
            },
            storage: StorageConfig {
                models_dir: tmp.path().join("models"),
                database: tmp.path().join("test.db"),
                cache_dir: tmp.path().join("cache"),
            },
            backends: BackendsConfig {
                default: "llamacpp".into(),
                enabled: vec!["llamacpp".into()],
            },
            training: TrainingConfig {
                executor: "subprocess".into(),
                trainer_image: None,
                max_concurrent_jobs: 2,
                checkpoints_dir: tmp.path().join("checkpoints"),
            },
            bridge: BridgeConfig {
                sy_endpoint: None,
                enabled: false,
                heartbeat_interval_secs: 10,
            },
            hardware: HardwareConfig {
                gpu_memory_reserve_mb: 512,
            },
        };
        AppState::new(config).unwrap()
    }

    #[test]
    fn inference_body_deserialize() {
        let json = r#"{
            "model": "llama-7b",
            "prompt": "Hello, world!",
            "max_tokens": 256,
            "temperature": 0.8,
            "top_p": 0.95,
            "system_prompt": "Be helpful."
        }"#;
        let body: InferenceBody = serde_json::from_str(json).unwrap();
        assert_eq!(body.model, "llama-7b");
        assert_eq!(body.prompt, "Hello, world!");
        assert_eq!(body.max_tokens, 256);
        assert_eq!(body.temperature, Some(0.8));
        assert_eq!(body.top_p, Some(0.95));
        assert_eq!(body.system_prompt, Some("Be helpful.".into()));
        assert!(!body.stream);
    }

    #[test]
    fn inference_body_defaults() {
        let json = r#"{"model": "test", "prompt": "hi"}"#;
        let body: InferenceBody = serde_json::from_str(json).unwrap();
        assert_eq!(body.max_tokens, 512);
        assert!(!body.stream);
        assert_eq!(body.temperature, None);
        assert_eq!(body.top_p, None);
        assert_eq!(body.top_k, None);
        assert_eq!(body.system_prompt, None);
    }

    #[test]
    fn inference_body_with_stream() {
        let json = r#"{"model": "test", "prompt": "hi", "stream": true}"#;
        let body: InferenceBody = serde_json::from_str(json).unwrap();
        assert!(body.stream);
    }

    #[tokio::test]
    async fn inference_no_model_loaded() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let body = InferenceBody {
            model: "test-model".into(),
            prompt: "Hello".into(),
            max_tokens: 100,
            temperature: None,
            top_p: None,
            top_k: None,
            system_prompt: None,
            stream: false,
        };

        let result = inference(State(state), Json(body)).await;
        let err = result.unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.contains("No model loaded"));
    }

    #[tokio::test]
    async fn inference_stream_no_model_loaded() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let body = InferenceBody {
            model: "test-model".into(),
            prompt: "Hello".into(),
            max_tokens: 100,
            temperature: None,
            top_p: None,
            top_k: None,
            system_prompt: None,
            stream: true,
        };

        let result = inference_stream(State(state), Json(body)).await;
        let err = result.unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }
}
