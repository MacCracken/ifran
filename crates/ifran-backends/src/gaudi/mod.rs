//! Intel Gaudi backend integration.
//!
//! HTTP client to a running optimum-habana or vLLM-HPU serving process.
//! Intel Gaudi uses Habana Labs HPU accelerators and exposes an
//! OpenAI-compatible API that this backend wraps.

use async_trait::async_trait;
use ifran_types::IfranError;
use ifran_types::backend::{
    AcceleratorType, BackendCapabilities, BackendId, BackendLocality, DeviceConfig,
};
use ifran_types::error::Result;
use ifran_types::inference::{
    FinishReason, InferenceRequest, InferenceResponse, StreamChunk, TokenUsage,
};
use ifran_types::model::{ModelFormat, ModelManifest};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn};

use crate::traits::{InferenceBackend, ModelHandle};

/// Intel Gaudi backend that proxies to a running Gaudi serving process.
pub struct GaudiBackend {
    /// Base URL of the Gaudi serving process.
    base_url: String,
    /// HTTP client.
    client: reqwest::Client,
    /// Loaded models (handle -> model name).
    loaded: Arc<RwLock<HashMap<String, String>>>,
}

impl GaudiBackend {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            base_url: base_url.unwrap_or_else(|| "http://127.0.0.1:8004".into()),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            loaded: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl InferenceBackend for GaudiBackend {
    fn id(&self) -> BackendId {
        BackendId("gaudi".into())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            accelerators: vec![AcceleratorType::Gaudi],
            max_context_length: Some(131072),
            supports_streaming: true,
            supports_embeddings: false,
            supports_vision: false,
            locality: BackendLocality::Local,
        }
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::SafeTensors, ModelFormat::PyTorch]
    }

    async fn load_model(
        &self,
        manifest: &ModelManifest,
        _device: &DeviceConfig,
    ) -> Result<ModelHandle> {
        // Gaudi serving loads models at server startup. We register the model
        // name and verify it's available via the /v1/models endpoint.
        let model_name = manifest
            .info
            .repo_id
            .as_deref()
            .unwrap_or(&manifest.info.name);

        let url = format!("{}/v1/models", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| IfranError::BackendError(format!("Cannot reach Gaudi server: {e}")))?;

        if !resp.status().is_success() {
            return Err(IfranError::BackendError(
                "Gaudi server returned error on /v1/models".into(),
            ));
        }

        let handle_id = format!("gaudi-{}", model_name.replace('/', "-"));
        info!(handle = %handle_id, model = %model_name, "Registered model with Gaudi backend");

        self.loaded
            .write()
            .await
            .insert(handle_id.clone(), model_name.to_string());
        Ok(ModelHandle(handle_id))
    }

    async fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        let mut loaded = self.loaded.write().await;
        if loaded.remove(&handle.0).is_some() {
            info!(handle = %handle.0, "Unregistered model from Gaudi backend");
            Ok(())
        } else {
            Err(IfranError::ModelNotFound(handle.0))
        }
    }

    async fn infer(
        &self,
        handle: &ModelHandle,
        req: &InferenceRequest,
    ) -> Result<InferenceResponse> {
        let loaded = self.loaded.read().await;
        let model_name = loaded
            .get(&handle.0)
            .ok_or_else(|| IfranError::ModelNotFound(handle.0.clone()))?;

        let url = format!("{}/v1/chat/completions", self.base_url);
        let messages = build_messages(req);
        let body = serde_json::json!({
            "model": model_name,
            "messages": messages,
            "max_tokens": req.max_tokens.unwrap_or(512),
            "temperature": req.temperature.unwrap_or(0.7),
            "top_p": req.top_p.unwrap_or(0.9),
            "stream": false,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| IfranError::BackendError(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(IfranError::BackendError(format!("Gaudi error: {text}")));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| IfranError::BackendError(e.to_string()))?;

        parse_openai_response(&json)
    }

    async fn infer_stream(
        &self,
        handle: &ModelHandle,
        req: InferenceRequest,
    ) -> Result<mpsc::Receiver<StreamChunk>> {
        let loaded = self.loaded.read().await;
        let model_name = loaded
            .get(&handle.0)
            .ok_or_else(|| IfranError::ModelNotFound(handle.0.clone()))?
            .clone();

        let url = format!("{}/v1/chat/completions", self.base_url);
        let messages = build_messages(&req);
        let body = serde_json::json!({
            "model": model_name,
            "messages": messages,
            "max_tokens": req.max_tokens.unwrap_or(512),
            "temperature": req.temperature.unwrap_or(0.7),
            "top_p": req.top_p.unwrap_or(0.9),
            "stream": true,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| IfranError::BackendError(e.to_string()))?;

        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            use futures::StreamExt;
            const MAX_BUFFER_SIZE: usize = 1024 * 1024; // 1 MB
            let mut stream = resp.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = stream.next().await {
                if tx.is_closed() {
                    break;
                }
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        warn!("Gaudi stream error: {e}");
                        break;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                if buffer.len() > MAX_BUFFER_SIZE {
                    warn!("Gaudi stream buffer exceeded {MAX_BUFFER_SIZE} bytes, aborting");
                    break;
                }

                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            let _ = tx
                                .send(StreamChunk {
                                    text: String::new(),
                                    done: true,
                                    usage: None,
                                })
                                .await;
                            return;
                        }

                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                            let text = json["choices"][0]["delta"]["content"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();

                            if !text.is_empty() {
                                let _ = tx
                                    .send(StreamChunk {
                                        text,
                                        done: false,
                                        usage: None,
                                    })
                                    .await;
                            }
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

fn build_messages(req: &InferenceRequest) -> Vec<serde_json::Value> {
    let mut messages = Vec::new();
    if let Some(ref system) = req.system_prompt {
        messages.push(serde_json::json!({
            "role": "system",
            "content": system,
        }));
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": &req.prompt,
    }));
    messages
}

fn parse_openai_response(json: &serde_json::Value) -> Result<InferenceResponse> {
    let text = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let finish_reason = match json["choices"][0]["finish_reason"].as_str() {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::MaxTokens,
        _ => FinishReason::Stop,
    };

    let usage = TokenUsage {
        prompt_tokens: json["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
        total_tokens: json["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
    };

    Ok(InferenceResponse {
        text,
        usage,
        finish_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id() {
        let backend = GaudiBackend::new(None);
        assert_eq!(backend.id().0, "gaudi");
    }

    #[test]
    fn default_url() {
        let backend = GaudiBackend::new(None);
        assert_eq!(backend.base_url, "http://127.0.0.1:8004");
    }

    #[test]
    fn custom_url() {
        let backend = GaudiBackend::new(Some("http://gaudi-server:8004".into()));
        assert_eq!(backend.base_url, "http://gaudi-server:8004");
    }

    #[test]
    fn capabilities() {
        let backend = GaudiBackend::new(None);
        let caps = backend.capabilities();
        assert!(caps.supports_streaming);
        assert!(!caps.supports_embeddings);
        assert!(!caps.supports_vision);
        assert_eq!(caps.max_context_length, Some(131072));
        assert!(caps.accelerators.contains(&AcceleratorType::Gaudi));
        assert!(!caps.accelerators.contains(&AcceleratorType::Cpu));
        assert_eq!(caps.locality, BackendLocality::Local);
    }

    #[test]
    fn formats() {
        let backend = GaudiBackend::new(None);
        assert!(
            backend
                .supported_formats()
                .contains(&ModelFormat::SafeTensors)
        );
        assert!(backend.supported_formats().contains(&ModelFormat::PyTorch));
    }

    #[test]
    fn build_messages_user_only() {
        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };
        let msgs = build_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn build_messages_with_system() {
        let req = InferenceRequest {
            prompt: "Hi".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: Some("Be helpful.".into()),
            sensitivity: None,
        };
        let msgs = build_messages(&req);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
    }

    #[test]
    fn parse_response_stop() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "Hello!");
        assert_eq!(resp.usage.prompt_tokens, 5);
        assert_eq!(resp.usage.total_tokens, 8);
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn parse_response_length() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "truncated"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 512, "total_tokens": 522}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert!(matches!(resp.finish_reason, FinishReason::MaxTokens));
    }

    #[test]
    fn parse_response_missing_fields() {
        let json = serde_json::json!({});
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "");
        assert_eq!(resp.usage.total_tokens, 0);
    }

    #[tokio::test]
    async fn load_model_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/v1/models")
            .with_status(200)
            .with_body(r#"{"data": [{"id": "test-model"}]}"#)
            .create_async()
            .await;

        let backend = GaudiBackend::new(Some(server.url()));
        let manifest = ifran_types::model::ModelManifest {
            info: ifran_types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test-model".into(),
                repo_id: Some("org/test-model".into()),
                format: ModelFormat::SafeTensors,
                quant: ifran_types::model::QuantLevel::None,
                size_bytes: 1000,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/model".into(),
                sha256: None,
                pulled_at: chrono::Utc::now(),
            },
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
        };
        let device = DeviceConfig {
            accelerator: AcceleratorType::Gaudi,
            device_ids: vec![0],
            memory_limit_mb: None,
        };

        let handle = backend.load_model(&manifest, &device).await.unwrap();
        assert_eq!(handle.0, "gaudi-org-test-model");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn load_model_server_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/v1/models")
            .with_status(500)
            .create_async()
            .await;

        let backend = GaudiBackend::new(Some(server.url()));
        let manifest = ifran_types::model::ModelManifest {
            info: ifran_types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test".into(),
                repo_id: None,
                format: ModelFormat::SafeTensors,
                quant: ifran_types::model::QuantLevel::None,
                size_bytes: 1000,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/model".into(),
                sha256: None,
                pulled_at: chrono::Utc::now(),
            },
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
        };
        let device = DeviceConfig {
            accelerator: AcceleratorType::Gaudi,
            device_ids: vec![0],
            memory_limit_mb: None,
        };

        let result = backend.load_model(&manifest, &device).await;
        assert!(matches!(result, Err(IfranError::BackendError(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn unload_model_success() {
        let backend = GaudiBackend::new(None);
        backend
            .loaded
            .write()
            .await
            .insert("gaudi-test".into(), "test-model".into());

        backend
            .unload_model(ModelHandle("gaudi-test".into()))
            .await
            .unwrap();
        assert!(backend.loaded.read().await.is_empty());
    }

    #[tokio::test]
    async fn unload_model_not_found() {
        let backend = GaudiBackend::new(None);
        let result = backend
            .unload_model(ModelHandle("nonexistent".into()))
            .await;
        assert!(matches!(result, Err(IfranError::ModelNotFound(_))));
    }

    #[tokio::test]
    async fn infer_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "choices": [{"message": {"content": "Hi there!"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
                }"#,
            )
            .create_async()
            .await;

        let backend = GaudiBackend::new(Some(server.url()));
        backend
            .loaded
            .write()
            .await
            .insert("gaudi-test".into(), "test-model".into());

        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: Some(100),
            temperature: Some(0.5),
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };

        let resp = backend
            .infer(&ModelHandle("gaudi-test".into()), &req)
            .await
            .unwrap();
        assert_eq!(resp.text, "Hi there!");
        assert_eq!(resp.usage.prompt_tokens, 8);
        assert_eq!(resp.usage.completion_tokens, 4);
        assert_eq!(resp.usage.total_tokens, 12);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn infer_model_not_loaded() {
        let backend = GaudiBackend::new(None);
        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };
        let result = backend
            .infer(&ModelHandle("nonexistent".into()), &req)
            .await;
        assert!(matches!(result, Err(IfranError::ModelNotFound(_))));
    }

    #[tokio::test]
    async fn infer_server_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(500)
            .with_body("HPU out of memory")
            .create_async()
            .await;

        let backend = GaudiBackend::new(Some(server.url()));
        backend
            .loaded
            .write()
            .await
            .insert("gaudi-test".into(), "model".into());

        let req = InferenceRequest {
            prompt: "Hello".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };
        let result = backend.infer(&ModelHandle("gaudi-test".into()), &req).await;
        assert!(matches!(result, Err(IfranError::BackendError(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn health_check_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/health")
            .with_status(200)
            .create_async()
            .await;

        let backend = GaudiBackend::new(Some(server.url()));
        let healthy = backend.health_check().await.unwrap();
        assert!(healthy);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn health_check_server_down() {
        let backend = GaudiBackend::new(Some("http://127.0.0.1:1".into()));
        let healthy = backend.health_check().await.unwrap();
        assert!(!healthy);
    }
}
