//! WebAssembly backend for browser-based inference.
//!
//! Provides a `WasmBackend` implementing the `InferenceBackend` trait that
//! targets browser-side execution via WebAssembly. Uses a pluggable
//! `WasmRuntime` trait for the actual execution — mock in tests, real wasm
//! runtime in the browser.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use synapse_types::SynapseError;
use synapse_types::backend::{
    AcceleratorType, BackendCapabilities, BackendId, BackendLocality, DeviceConfig,
};
use synapse_types::error::Result;
use synapse_types::inference::{
    FinishReason, InferenceRequest, InferenceResponse, StreamChunk, TokenUsage,
};
use synapse_types::model::{ModelFormat, ModelManifest};
use tokio::sync::{RwLock, mpsc};
use tracing::info;

use crate::traits::{InferenceBackend, ModelHandle};

/// Trait for pluggable WebAssembly runtime execution.
/// Allows mock implementations in tests and real wasm runtimes in the browser.
pub trait WasmRuntime: Send + Sync {
    fn infer(&self, handle: &str, prompt: &str) -> std::result::Result<String, String>;
}

/// Default stub runtime that returns placeholder responses.
pub struct StubWasmRuntime;

impl WasmRuntime for StubWasmRuntime {
    fn infer(&self, handle: &str, _prompt: &str) -> std::result::Result<String, String> {
        Ok(format!(
            "[wasm-stub:{handle}] inference not available in server mode"
        ))
    }
}

/// Metadata for a loaded model in the wasm backend.
struct LoadedModel {
    #[allow(dead_code)]
    model_path: String,
}

/// WebAssembly backend for browser-based inference.
pub struct WasmBackend {
    loaded: Arc<RwLock<HashMap<String, LoadedModel>>>,
    runtime: Arc<dyn WasmRuntime>,
}

impl WasmBackend {
    pub fn new() -> Self {
        Self {
            loaded: Arc::new(RwLock::new(HashMap::new())),
            runtime: Arc::new(StubWasmRuntime),
        }
    }

    pub fn with_runtime(runtime: Arc<dyn WasmRuntime>) -> Self {
        Self {
            loaded: Arc::new(RwLock::new(HashMap::new())),
            runtime,
        }
    }
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceBackend for WasmBackend {
    fn id(&self) -> BackendId {
        BackendId("wasm".into())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            accelerators: vec![AcceleratorType::Cpu],
            max_context_length: Some(4096),
            supports_streaming: false,
            supports_embeddings: false,
            supports_vision: false,
            locality: BackendLocality::Local,
        }
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Gguf, ModelFormat::Onnx]
    }

    async fn load_model(
        &self,
        manifest: &ModelManifest,
        _device: &DeviceConfig,
    ) -> Result<ModelHandle> {
        let model_path = &manifest.info.local_path;
        let handle_id = format!("wasm-{}", manifest.info.id);
        info!(
            handle = %handle_id,
            model = %model_path,
            "Loading model with WebAssembly backend"
        );

        self.loaded.write().await.insert(
            handle_id.clone(),
            LoadedModel {
                model_path: model_path.clone(),
            },
        );

        Ok(ModelHandle(handle_id))
    }

    async fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        if self.loaded.write().await.remove(&handle.0).is_some() {
            info!(handle = %handle.0, "Unloaded WebAssembly model");
            Ok(())
        } else {
            Err(SynapseError::ModelNotFound(handle.0))
        }
    }

    async fn infer(
        &self,
        handle: &ModelHandle,
        req: &InferenceRequest,
    ) -> Result<InferenceResponse> {
        let loaded = self.loaded.read().await;
        if !loaded.contains_key(&handle.0) {
            return Err(SynapseError::ModelNotFound(handle.0.clone()));
        }

        let text = self
            .runtime
            .infer(&handle.0, &req.prompt)
            .map_err(SynapseError::BackendError)?;

        Ok(InferenceResponse {
            text,
            usage: TokenUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
            finish_reason: FinishReason::Stop,
        })
    }

    async fn infer_stream(
        &self,
        handle: &ModelHandle,
        _req: InferenceRequest,
    ) -> Result<mpsc::Receiver<StreamChunk>> {
        let loaded = self.loaded.read().await;
        if !loaded.contains_key(&handle.0) {
            return Err(SynapseError::ModelNotFound(handle.0.clone()));
        }

        Err(SynapseError::BackendError(
            "WebAssembly backend does not support streaming".into(),
        ))
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRuntime;
    impl WasmRuntime for MockRuntime {
        fn infer(&self, _handle: &str, prompt: &str) -> std::result::Result<String, String> {
            Ok(format!("mock response to: {prompt}"))
        }
    }

    fn test_manifest() -> ModelManifest {
        ModelManifest {
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
            info: synapse_types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test-wasm".into(),
                repo_id: None,
                format: ModelFormat::Gguf,
                quant: synapse_types::model::QuantLevel::None,
                size_bytes: 1000,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/model.gguf".into(),
                sha256: None,
                pulled_at: chrono::Utc::now(),
            },
        }
    }

    fn test_device() -> DeviceConfig {
        DeviceConfig {
            accelerator: AcceleratorType::Cpu,
            device_ids: vec![0],
            memory_limit_mb: None,
        }
    }

    #[test]
    fn backend_id() {
        let backend = WasmBackend::new();
        assert_eq!(backend.id().0, "wasm");
    }

    #[test]
    fn supported_formats_gguf_and_onnx() {
        let backend = WasmBackend::new();
        let formats = backend.supported_formats();
        assert_eq!(formats.len(), 2);
        assert!(formats.contains(&ModelFormat::Gguf));
        assert!(formats.contains(&ModelFormat::Onnx));
    }

    #[test]
    fn capabilities_details() {
        let backend = WasmBackend::new();
        let caps = backend.capabilities();
        assert!(!caps.supports_streaming);
        assert!(!caps.supports_embeddings);
        assert!(!caps.supports_vision);
        assert_eq!(caps.max_context_length, Some(4096));
        assert_eq!(caps.accelerators, vec![AcceleratorType::Cpu]);
    }

    #[test]
    fn default_constructor() {
        let backend = WasmBackend::default();
        assert_eq!(backend.id().0, "wasm");
    }

    #[tokio::test]
    async fn health_check_ok() {
        let backend = WasmBackend::new();
        assert!(backend.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn load_and_unload_model() {
        let backend = WasmBackend::new();
        let manifest = test_manifest();
        let device = test_device();
        let handle = backend.load_model(&manifest, &device).await.unwrap();
        assert!(handle.0.starts_with("wasm-"));
        backend.unload_model(handle).await.unwrap();
    }

    #[tokio::test]
    async fn unload_nonexistent_fails() {
        let backend = WasmBackend::new();
        assert!(
            backend
                .unload_model(ModelHandle("nope".into()))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn infer_with_mock_runtime() {
        let backend = WasmBackend::with_runtime(Arc::new(MockRuntime));
        let manifest = test_manifest();
        let device = test_device();
        let handle = backend.load_model(&manifest, &device).await.unwrap();

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
        let resp = backend.infer(&handle, &req).await.unwrap();
        assert_eq!(resp.text, "mock response to: hello");
    }

    #[tokio::test]
    async fn infer_without_load_fails() {
        let backend = WasmBackend::new();
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
        assert!(
            backend
                .infer(&ModelHandle("nope".into()), &req)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn stream_not_supported() {
        let backend = WasmBackend::new();
        let manifest = test_manifest();
        let device = test_device();
        let handle = backend.load_model(&manifest, &device).await.unwrap();

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
        assert!(backend.infer_stream(&handle, req).await.is_err());
    }

    #[tokio::test]
    async fn stub_runtime_returns_placeholder() {
        let backend = WasmBackend::new();
        let manifest = test_manifest();
        let device = test_device();
        let handle = backend.load_model(&manifest, &device).await.unwrap();

        let req = InferenceRequest {
            prompt: "test".into(),
            system_prompt: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            sensitivity: None,
        };
        let resp = backend.infer(&handle, &req).await.unwrap();
        assert!(resp.text.contains("wasm-stub"));
        assert!(resp.text.contains("inference not available"));
    }
}
