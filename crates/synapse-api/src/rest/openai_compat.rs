//! OpenAI-compatible API endpoints for drop-in replacement compatibility.
//!
//! Implements the subset of the OpenAI API used by most clients:
//! - POST /v1/chat/completions
//! - GET /v1/models

use crate::middleware::validation::{validate_model_name, validate_prompt_length};
use crate::state::AppState;
use axum::Json;
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::{Event, Sse};
use serde::Deserialize;
use synapse_types::TenantId;
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
pub async fn list_models(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let db = state.db.lock().await;
    let models = db
        .list(&tenant_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

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

    Ok(Json(serde_json::json!({
        "object": "list",
        "data": data,
    })))
}

/// POST /v1/chat/completions — OpenAI-compatible chat completions.
pub async fn chat_completions(
    State(state): State<AppState>,
    Extension(tenant_id): Extension<TenantId>,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    validate_model_name(&body.model)?;
    // Validate all message contents against prompt length limit
    for msg in &body.messages {
        validate_prompt_length(&msg.content, state.config.security.max_prompt_length)?;
    }

    let loaded = state.model_manager.list_loaded(Some(&tenant_id)).await;
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
        sensitivity: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use axum::extract::Extension;
    use synapse_core::config::*;
    use synapse_core::storage::db::ModelDatabase;
    use synapse_types::TenantId;
    use synapse_types::model::{ModelFormat, ModelInfo, QuantLevel};

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
                telemetry_interval_secs: 0,
            },
            security: SecurityConfig::default(),
            budget: BudgetConfig::default(),
            fleet: FleetConfig::default(),
        };
        AppState::new(config).unwrap()
    }

    #[test]
    fn chat_message_deserialize() {
        let json = r#"{"role": "user", "content": "Hello!"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello!");
    }

    #[test]
    fn chat_completion_request_deserialize() {
        let json = r#"{
            "model": "llama-7b",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"}
            ],
            "max_tokens": 256,
            "temperature": 0.7
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama-7b");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.temperature, Some(0.7));
        assert!(!req.stream);
    }

    #[test]
    fn chat_completion_request_defaults() {
        let json = r#"{"model": "test", "messages": [{"role": "user", "content": "hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 512);
        assert!(!req.stream);
        assert_eq!(req.temperature, None);
        assert_eq!(req.top_p, None);
    }

    #[test]
    fn chat_completion_request_with_stream() {
        let json =
            r#"{"model": "t", "messages": [{"role": "user", "content": "h"}], "stream": true}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream);
    }

    #[tokio::test]
    async fn list_models_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let Json(json) = list_models(State(state), Extension(TenantId::default_tenant()))
            .await
            .unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn list_models_with_data() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let db = ModelDatabase::open(&tmp.path().join("test.db")).unwrap();
        let model = ModelInfo {
            id: uuid::Uuid::new_v4(),
            name: "test-model".into(),
            repo_id: None,
            format: ModelFormat::Gguf,
            quant: QuantLevel::Q4KM,
            size_bytes: 1000,
            parameter_count: None,
            architecture: None,
            license: None,
            local_path: "/tmp/model.gguf".into(),
            sha256: None,
            pulled_at: chrono::Utc::now(),
        };
        db.insert(&model, &synapse_types::TenantId::default_tenant())
            .unwrap();
        drop(db);

        let Json(json) = list_models(State(state), Extension(TenantId::default_tenant()))
            .await
            .unwrap();
        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["id"], "test-model");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(data[0]["owned_by"], "synapse");
        assert!(data[0]["created"].is_number());
    }

    #[tokio::test]
    async fn chat_completions_no_model_loaded() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);

        let body = ChatCompletionRequest {
            model: "test".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            }],
            max_tokens: 100,
            temperature: None,
            top_p: None,
            stream: false,
        };

        let result = chat_completions(
            State(state),
            Extension(TenantId::default_tenant()),
            Json(body),
        )
        .await;
        let err = result.unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.contains("No model loaded"));
    }
}
