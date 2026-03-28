//! TPU backend integration.
//!
//! HTTP client to a running JAX/PJRT serving process or vLLM with TPU support.
//! The serving endpoint exposes an OpenAI-compatible API that this backend wraps,
//! targeting Google Cloud TPU v5+ accelerators for high-throughput inference.

use crate::types::IfranError;
use crate::types::backend::{
    AcceleratorType, BackendCapabilities, BackendId, BackendLocality, DeviceConfig,
};
use crate::types::error::Result;
use crate::types::inference::{InferenceRequest, InferenceResponse, StreamChunk};
use crate::types::model::{ModelFormat, ModelManifest};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::info;

use crate::backends::openai_compat::{
    build_openai_messages, parse_openai_response, stream_openai_sse,
};
use crate::backends::traits::{InferenceBackend, ModelHandle};

/// TPU backend that proxies to a running JAX/PJRT serving process.
pub struct TpuBackend {
    /// Base URL of the TPU serving process.
    base_url: String,
    /// HTTP client.
    client: reqwest::Client,
    /// Loaded models (handle → model name).
    loaded: Arc<RwLock<HashMap<String, String>>>,
}

impl TpuBackend {
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            base_url: base_url.unwrap_or_else(|| "http://127.0.0.1:8001".into()),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            loaded: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl InferenceBackend for TpuBackend {
    fn id(&self) -> BackendId {
        BackendId("tpu".into())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            accelerators: vec![AcceleratorType::Tpu],
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
        // The TPU serving process loads models at startup. We register the
        // model name and verify it's available via the /v1/models endpoint.
        let model_name = manifest
            .info
            .repo_id
            .as_deref()
            .unwrap_or(&manifest.info.name);

        let url = format!("{}/v1/models", self.base_url);
        let resp = self.client.get(&url).send().await.map_err(|e| {
            IfranError::BackendError(format!("Cannot reach TPU serving process: {e}"))
        })?;

        if !resp.status().is_success() {
            return Err(IfranError::BackendError(
                "TPU serving process returned error on /v1/models".into(),
            ));
        }

        let handle_id = format!("tpu-{}", model_name.replace('/', "-"));
        info!(handle = %handle_id, model = %model_name, "Registered model with TPU backend");

        self.loaded
            .write()
            .await
            .insert(handle_id.clone(), model_name.to_string());
        Ok(ModelHandle(handle_id))
    }

    async fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        let mut loaded = self.loaded.write().await;
        if loaded.remove(&handle.0).is_some() {
            info!(handle = %handle.0, "Unregistered model from TPU backend");
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
        let messages = build_openai_messages(req);
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
            return Err(IfranError::BackendError(format!("TPU error: {text}")));
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
        let messages = build_openai_messages(&req);
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
            let _ = stream_openai_sse(resp, tx).await;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::inference::FinishReason;

    #[test]
    fn backend_id() {
        let backend = TpuBackend::new(None);
        assert_eq!(backend.id().0, "tpu");
    }

    #[test]
    fn default_url() {
        let backend = TpuBackend::new(None);
        assert_eq!(backend.base_url, "http://127.0.0.1:8001");
    }

    #[test]
    fn custom_url() {
        let backend = TpuBackend::new(Some("http://tpu-host:9000".into()));
        assert_eq!(backend.base_url, "http://tpu-host:9000");
    }

    #[test]
    fn capabilities() {
        let backend = TpuBackend::new(None);
        let caps = backend.capabilities();
        assert!(caps.supports_streaming);
        assert!(!caps.supports_embeddings);
        assert!(!caps.supports_vision);
        assert_eq!(caps.max_context_length, Some(131072));
        assert!(caps.accelerators.contains(&AcceleratorType::Tpu));
        assert!(!caps.accelerators.contains(&AcceleratorType::Cuda));
        assert!(!caps.accelerators.contains(&AcceleratorType::Cpu));
        assert_eq!(caps.locality, BackendLocality::Local);
    }

    #[test]
    fn supports_safetensors() {
        let backend = TpuBackend::new(None);
        assert!(
            backend
                .supported_formats()
                .contains(&ModelFormat::SafeTensors)
        );
    }

    #[test]
    fn supports_pytorch() {
        let backend = TpuBackend::new(None);
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
        let msgs = build_openai_messages(&req);
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
        let msgs = build_openai_messages(&req);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
    }

    #[test]
    fn parse_openai_response_stop() {
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
    fn parse_openai_response_length() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "truncated"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 512, "total_tokens": 522}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert!(matches!(resp.finish_reason, FinishReason::MaxTokens));
    }

    #[test]
    fn parse_openai_response_missing_fields() {
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

        let backend = TpuBackend::new(Some(server.url()));
        let manifest = crate::types::model::ModelManifest {
            info: crate::types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test-model".into(),
                repo_id: Some("org/test-model".into()),
                format: ModelFormat::SafeTensors,
                quant: crate::types::model::QuantLevel::None,
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
            accelerator: AcceleratorType::Tpu,
            device_ids: vec![0],
            memory_limit_mb: None,
        };

        let handle = backend.load_model(&manifest, &device).await.unwrap();
        assert_eq!(handle.0, "tpu-org-test-model");
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

        let backend = TpuBackend::new(Some(server.url()));
        let manifest = crate::types::model::ModelManifest {
            info: crate::types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test".into(),
                repo_id: None,
                format: ModelFormat::SafeTensors,
                quant: crate::types::model::QuantLevel::None,
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
            accelerator: AcceleratorType::Tpu,
            device_ids: vec![0],
            memory_limit_mb: None,
        };

        let result = backend.load_model(&manifest, &device).await;
        assert!(matches!(result, Err(IfranError::BackendError(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn unload_model_success() {
        let backend = TpuBackend::new(None);
        backend
            .loaded
            .write()
            .await
            .insert("tpu-test".into(), "test-model".into());

        backend
            .unload_model(ModelHandle("tpu-test".into()))
            .await
            .unwrap();
        assert!(backend.loaded.read().await.is_empty());
    }

    #[tokio::test]
    async fn unload_model_not_found() {
        let backend = TpuBackend::new(None);
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

        let backend = TpuBackend::new(Some(server.url()));
        backend
            .loaded
            .write()
            .await
            .insert("tpu-test".into(), "test-model".into());

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
            .infer(&ModelHandle("tpu-test".into()), &req)
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
        let backend = TpuBackend::new(None);
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
            .with_body("TPU out of memory")
            .create_async()
            .await;

        let backend = TpuBackend::new(Some(server.url()));
        backend
            .loaded
            .write()
            .await
            .insert("tpu-test".into(), "model".into());

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
        let result = backend.infer(&ModelHandle("tpu-test".into()), &req).await;
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

        let backend = TpuBackend::new(Some(server.url()));
        let healthy = backend.health_check().await.unwrap();
        assert!(healthy);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn health_check_server_down() {
        let backend = TpuBackend::new(Some("http://127.0.0.1:1".into()));
        let healthy = backend.health_check().await.unwrap();
        assert!(!healthy);
    }
}
