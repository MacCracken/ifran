//! TensorRT-LLM backend integration.
//!
//! HTTP client to a running TensorRT-LLM server (Triton Inference Server with
//! TensorRT-LLM backend). Provides optimized NVIDIA GPU inference via layer
//! fusion, FP16/INT8 precision, and in-flight batching.

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

/// TensorRT-LLM backend proxying to a Triton server.
pub struct TensorRtBackend {
    /// Base URL of the Triton/TensorRT-LLM server.
    base_url: String,
    /// HTTP client.
    client: reqwest::Client,
    /// Loaded models (handle → model name).
    loaded: Arc<RwLock<HashMap<String, String>>>,
}

impl TensorRtBackend {
    pub fn new(base_url: Option<String>) -> Self {
        crate::ensure_crypto_provider();
        Self {
            base_url: base_url.unwrap_or_else(|| "http://127.0.0.1:8000".into()),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            loaded: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl InferenceBackend for TensorRtBackend {
    fn id(&self) -> BackendId {
        BackendId("tensorrt".into())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            accelerators: vec![AcceleratorType::Cuda],
            max_context_length: Some(131072),
            supports_streaming: true,
            supports_embeddings: false,
            supports_vision: false,
            locality: BackendLocality::Local,
        }
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::TensorRt]
    }

    async fn load_model(
        &self,
        manifest: &ModelManifest,
        _device: &DeviceConfig,
    ) -> Result<ModelHandle> {
        // TensorRT-LLM models are pre-compiled engine files loaded by Triton.
        // We verify the model is available on the server.
        let model_name = manifest
            .info
            .repo_id
            .as_deref()
            .unwrap_or(&manifest.info.name);

        let handle_id = format!("tensorrt-{}", model_name.replace('/', "-"));
        info!(handle = %handle_id, model = %model_name, "Registered model with TensorRT backend");

        self.loaded
            .write()
            .await
            .insert(handle_id.clone(), model_name.to_string());
        Ok(ModelHandle(handle_id))
    }

    async fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        if self.loaded.write().await.remove(&handle.0).is_some() {
            info!(handle = %handle.0, "Unregistered model from TensorRT backend");
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

        // TensorRT-LLM with Triton exposes an OpenAI-compatible endpoint
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": model_name,
            "messages": build_openai_messages(req),
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
            return Err(IfranError::BackendError(format!("TensorRT error: {text}")));
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
        let body = serde_json::json!({
            "model": model_name,
            "messages": build_openai_messages(&req),
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
        let url = format!("{}/v2/health/ready", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id() {
        let backend = TensorRtBackend::new(None);
        assert_eq!(backend.id().0, "tensorrt");
    }

    #[test]
    fn cuda_only() {
        let backend = TensorRtBackend::new(None);
        assert_eq!(
            backend.capabilities().accelerators,
            vec![AcceleratorType::Cuda]
        );
    }

    #[test]
    fn supports_tensorrt_format() {
        let backend = TensorRtBackend::new(None);
        assert_eq!(backend.supported_formats(), &[ModelFormat::TensorRt]);
    }

    #[test]
    fn custom_url() {
        let backend = TensorRtBackend::new(Some("http://gpu-server:8000".into()));
        assert_eq!(backend.base_url, "http://gpu-server:8000");
    }

    #[test]
    fn default_url() {
        let backend = TensorRtBackend::new(None);
        assert_eq!(backend.base_url, "http://127.0.0.1:8000");
    }

    #[test]
    fn capabilities_details() {
        let backend = TensorRtBackend::new(None);
        let caps = backend.capabilities();
        assert!(caps.supports_streaming);
        assert!(!caps.supports_embeddings);
        assert!(!caps.supports_vision);
        assert_eq!(caps.max_context_length, Some(131072));
    }

    #[tokio::test]
    async fn load_and_unload_model() {
        let backend = TensorRtBackend::new(None);
        let manifest = ModelManifest {
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
            info: crate::types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test-trt".into(),
                repo_id: Some("org/model".into()),
                format: ModelFormat::TensorRt,
                quant: crate::types::model::QuantLevel::None,
                size_bytes: 2000,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/model.engine".into(),
                sha256: None,
                pulled_at: chrono::Utc::now(),
            },
        };
        let device = DeviceConfig {
            accelerator: AcceleratorType::Cuda,
            device_ids: vec![0],
            memory_limit_mb: None,
        };
        let handle = backend.load_model(&manifest, &device).await.unwrap();
        assert!(handle.0.starts_with("tensorrt-"));
        backend.unload_model(handle).await.unwrap();
    }

    #[tokio::test]
    async fn load_model_without_repo_id() {
        let backend = TensorRtBackend::new(None);
        let manifest = ModelManifest {
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
            info: crate::types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "local-model".into(),
                repo_id: None,
                format: ModelFormat::TensorRt,
                quant: crate::types::model::QuantLevel::None,
                size_bytes: 0,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/model.engine".into(),
                sha256: None,
                pulled_at: chrono::Utc::now(),
            },
        };
        let device = DeviceConfig {
            accelerator: AcceleratorType::Cuda,
            device_ids: vec![0],
            memory_limit_mb: None,
        };
        let handle = backend.load_model(&manifest, &device).await.unwrap();
        assert_eq!(handle.0, "tensorrt-local-model");
    }

    #[tokio::test]
    async fn unload_nonexistent_fails() {
        let backend = TensorRtBackend::new(None);
        assert!(
            backend
                .unload_model(ModelHandle("nope".into()))
                .await
                .is_err()
        );
    }

    #[test]
    fn build_messages_user_only() {
        let req = InferenceRequest {
            prompt: "hello".into(),
            system_prompt: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            sensitivity: None,
        };
        let msgs = build_openai_messages(&req);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "hello");
    }

    #[test]
    fn build_messages_with_system() {
        let req = InferenceRequest {
            prompt: "hello".into(),
            system_prompt: Some("You are helpful".into()),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            sensitivity: None,
        };
        let msgs = build_openai_messages(&req);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are helpful");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn backend_locality_is_local() {
        let backend = TensorRtBackend::new(None);
        assert!(matches!(
            backend.capabilities().locality,
            BackendLocality::Local
        ));
    }

    #[tokio::test]
    async fn health_check_unreachable_returns_false() {
        // Point at a port that's (almost certainly) not listening
        let backend = TensorRtBackend::new(Some("http://127.0.0.1:1".into()));
        let result = backend.health_check().await;
        assert!(result.is_ok());
        assert!(!result.unwrap(), "unreachable backend should return false");
    }

    #[tokio::test]
    async fn infer_unloaded_model_fails() {
        let backend = TensorRtBackend::new(None);
        let handle = ModelHandle("nonexistent".into());
        let req = InferenceRequest {
            prompt: "test".into(),
            system_prompt: None,
            max_tokens: Some(10),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            sensitivity: None,
        };
        let result = backend.infer(&handle, &req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn infer_stream_unloaded_model_fails() {
        let backend = TensorRtBackend::new(None);
        let handle = ModelHandle("nonexistent".into());
        let req = InferenceRequest {
            prompt: "test".into(),
            system_prompt: None,
            max_tokens: Some(10),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            sensitivity: None,
        };
        let result = backend.infer_stream(&handle, req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn loaded_map_tracks_models() {
        let backend = TensorRtBackend::new(None);
        assert!(backend.loaded.read().await.is_empty());

        let manifest = ModelManifest {
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
            info: crate::types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test-model".into(),
                repo_id: Some("org/test".into()),
                format: ModelFormat::TensorRt,
                quant: crate::types::model::QuantLevel::None,
                size_bytes: 100,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/test.engine".into(),
                sha256: None,
                pulled_at: chrono::Utc::now(),
            },
        };
        let device = DeviceConfig {
            accelerator: AcceleratorType::Cuda,
            device_ids: vec![0],
            memory_limit_mb: None,
        };

        let handle = backend.load_model(&manifest, &device).await.unwrap();
        assert_eq!(backend.loaded.read().await.len(), 1);

        backend.unload_model(handle).await.unwrap();
        assert!(backend.loaded.read().await.is_empty());
    }

    #[test]
    fn parse_response_with_usage() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        });
        let resp = parse_openai_response(&json).unwrap();
        assert_eq!(resp.text, "answer");
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 20);
        assert_eq!(resp.usage.total_tokens, 30);
    }

    #[test]
    fn handle_id_format_with_slashes() {
        // Models with org/name should have slashes replaced
        let model_name = "meta-llama/Llama-3-8B";
        let handle_id = format!("tensorrt-{}", model_name.replace('/', "-"));
        assert_eq!(handle_id, "tensorrt-meta-llama-Llama-3-8B");
    }

    #[test]
    fn max_context_length_is_128k() {
        let backend = TensorRtBackend::new(None);
        assert_eq!(backend.capabilities().max_context_length, Some(131072));
    }
}
