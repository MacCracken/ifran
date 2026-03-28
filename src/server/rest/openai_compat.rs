//! OpenAI-compatible API endpoints for drop-in replacement compatibility.
//!
//! Implements the subset of the OpenAI API used by most clients:
//! - POST /v1/chat/completions
//! - GET /v1/models

use crate::server::middleware::validation::{validate_model_name, validate_prompt_length};
use crate::server::state::AppState;
use crate::types::TenantId;
use crate::types::inference::InferenceRequest;
use axum::Json;
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::{Event, Sse};
use serde::Deserialize;

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
    let paged = state
        .db
        .list(&tenant_id, 1000, 0)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut data: Vec<serde_json::Value> = paged
        .items
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.name,
                "object": "model",
                "created": m.pulled_at.timestamp(),
                "owned_by": "ifran",
            })
        })
        .collect();

    // Merge models from hoosh provider registry
    if let Some(providers) = &state.hoosh_providers {
        let mut seen: std::collections::HashSet<String> = data
            .iter()
            .filter_map(|d| d["id"].as_str().map(String::from))
            .collect();

        for provider in providers.all() {
            if let Ok(models) = provider.list_models().await {
                for m in models {
                    // Avoid duplicates — skip models already listed from local DB
                    if seen.insert(m.id.clone()) {
                        data.push(serde_json::json!({
                            "id": m.id,
                            "object": "model",
                            "created": 0,
                            "owned_by": m.provider,
                        }));
                    }
                }
            }
        }
    }

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
    // Validate each message content against prompt length limit
    for msg in &body.messages {
        validate_prompt_length(&msg.content, state.config.security.max_prompt_length)?;
    }

    // Validate combined message length against prompt length limit
    let total_content_length: usize = body.messages.iter().map(|m| m.content.len()).sum();
    if total_content_length > state.config.security.max_prompt_length {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Combined message length {} exceeds maximum {}",
                total_content_length, state.config.security.max_prompt_length
            ),
        ));
    }

    // Extract system prompt and user messages
    let system_prompt = body
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.clone());

    let prompt = match body
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
    {
        Some(content) if !content.is_empty() => content,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                "At least one non-empty user message is required".into(),
            ));
        }
    };

    // Try local model path first
    let loaded = state.model_manager.list_loaded(Some(&tenant_id)).await;
    let loaded_model = loaded.iter().find(|m| m.model_name == body.model);

    if let Some(loaded_model) = loaded_model {
        // Local model path — use existing backend
        let backend = state
            .backends
            .get(&crate::types::backend::BackendId(
                loaded_model.backend_id.clone(),
            ))
            .ok_or((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Backend not available".into(),
            ))?;

        let handle = crate::backends::ModelHandle(loaded_model.handle.clone());

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
                    crate::types::inference::FinishReason::Stop => "stop",
                    crate::types::inference::FinishReason::MaxTokens => "length",
                    crate::types::inference::FinishReason::StopSequence => "stop",
                },
            }],
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
        });

        return Ok(Json(json).into_response());
    }

    // Fallback: route through hoosh if model is not loaded locally
    let (router, providers) = match (&state.hoosh_router, &state.hoosh_providers) {
        (Some(r), Some(p)) => (r, p),
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "Model '{}' is not loaded. Load it first with POST /v1/models or 'ifran pull {}'",
                    body.model, body.model
                ),
            ));
        }
    };

    let route = router.select(&body.model).ok_or((
        StatusCode::BAD_REQUEST,
        format!(
            "Model '{}' is not loaded and no hoosh provider matches",
            body.model
        ),
    ))?;

    let provider = providers.get(route.provider, &route.base_url).ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        format!(
            "Hoosh provider '{}' at {} is not registered",
            route.provider, route.base_url
        ),
    ))?;

    // Convert messages to hoosh format
    let hoosh_messages: Vec<hoosh::inference::Message> = body
        .messages
        .iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => hoosh::inference::Role::System,
                "assistant" => hoosh::inference::Role::Assistant,
                "tool" => hoosh::inference::Role::Tool,
                _ => hoosh::inference::Role::User,
            };
            hoosh::inference::Message::new(role, &m.content)
        })
        .collect();

    let hoosh_req = hoosh::InferenceRequest {
        model: body.model.clone(),
        prompt,
        system: system_prompt,
        messages: hoosh_messages,
        max_tokens: Some(body.max_tokens),
        temperature: body.temperature.map(|t| t as f64),
        top_p: body.top_p.map(|p| p as f64),
        stream: body.stream,
        ..Default::default()
    };

    tracing::info!(
        model = %body.model,
        provider = %route.provider,
        "Routing chat completion through hoosh fallback"
    );

    if body.stream {
        // Hoosh streaming fallback
        let mut rx = provider.infer_stream(hoosh_req).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Hoosh streaming failed: {e}"),
            )
        })?;

        let model = body.model.clone();
        let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        let stream = async_stream::stream! {
            while let Some(chunk_result) = rx.recv().await {
                match chunk_result {
                    Ok(text) => {
                        if text.is_empty() {
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
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": serde_json::Value::Null}],
                        });
                        yield Ok::<_, std::convert::Infallible>(Event::default().data(data.to_string()));
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "Hoosh stream chunk error");
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                }
            }
        };

        return Ok(Sse::new(stream).into_response());
    }

    // Non-streaming hoosh fallback
    let hoosh_resp = provider.infer(&hoosh_req).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Hoosh inference failed: {e}"),
        )
    })?;

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
                "content": hoosh_resp.text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": hoosh_resp.usage.prompt_tokens,
            "completion_tokens": hoosh_resp.usage.completion_tokens,
            "total_tokens": hoosh_resp.usage.total_tokens,
        },
    });

    Ok(Json(json).into_response())
}

async fn stream_response(
    backend: std::sync::Arc<dyn crate::backends::InferenceBackend>,
    handle: crate::backends::ModelHandle,
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
    use crate::config::*;
    use crate::server::state::AppState;
    use crate::storage::db::ModelDatabase;
    use crate::types::TenantId;
    use crate::types::model::{ModelFormat, ModelInfo, QuantLevel};
    use axum::extract::Extension;

    fn test_state(tmp: &tempfile::TempDir) -> AppState {
        let config = IfranConfig {
            server: ServerConfig {
                bind: "127.0.0.1:0".into(),
                grpc_bind: "127.0.0.1:0".into(),
                ws_bind: None,
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
                job_eviction_ttl_secs: 86400,
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
        db.insert(&model, &crate::types::TenantId::default_tenant())
            .unwrap();
        drop(db);

        let Json(json) = list_models(State(state), Extension(TenantId::default_tenant()))
            .await
            .unwrap();
        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["id"], "test-model");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(data[0]["owned_by"], "ifran");
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
        // With hoosh fallback, the error may come from hoosh routing or model-not-loaded
        assert!(err.0 == StatusCode::BAD_REQUEST || err.0 == StatusCode::INTERNAL_SERVER_ERROR);
    }
}
