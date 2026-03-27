//! ONNX Runtime backend integration.
//!
//! Wraps the ONNX Runtime via the `ort` crate for inference on ONNX-exported
//! models. Supports CPU, CUDA, TensorRT, DirectML, and CoreML execution
//! providers.

use async_trait::async_trait;
use ifran_types::IfranError;
use ifran_types::backend::{
    AcceleratorType, BackendCapabilities, BackendId, BackendLocality, DeviceConfig,
};
use ifran_types::error::Result;
use ifran_types::inference::{InferenceRequest, InferenceResponse, StreamChunk};
use ifran_types::model::{ModelFormat, ModelManifest};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::info;

use crate::traits::{InferenceBackend, ModelHandle};

/// Metadata for a loaded ONNX model.
struct LoadedOnnx {
    #[allow(dead_code)]
    model_path: String,
}

/// ONNX Runtime backend.
pub struct OnnxBackend {
    loaded: Arc<RwLock<HashMap<String, LoadedOnnx>>>,
}

impl OnnxBackend {
    pub fn new() -> Self {
        Self {
            loaded: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceBackend for OnnxBackend {
    fn id(&self) -> BackendId {
        BackendId("onnx".into())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            accelerators: vec![AcceleratorType::Cpu, AcceleratorType::Cuda],
            max_context_length: None,
            supports_streaming: false,
            supports_embeddings: true,
            supports_vision: true,
            locality: BackendLocality::Local,
        }
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Onnx]
    }

    async fn load_model(
        &self,
        manifest: &ModelManifest,
        _device: &DeviceConfig,
    ) -> Result<ModelHandle> {
        let model_path = &manifest.info.local_path;

        // In a full implementation:
        // 1. Create an ort::Session with the appropriate execution provider
        // 2. Load the ONNX model file
        // 3. Configure input/output bindings

        let handle_id = format!("onnx-{}", manifest.info.id);
        info!(handle = %handle_id, model = %model_path, "Loading ONNX model");

        self.loaded.write().await.insert(
            handle_id.clone(),
            LoadedOnnx {
                model_path: model_path.clone(),
            },
        );

        Ok(ModelHandle(handle_id))
    }

    async fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        if self.loaded.write().await.remove(&handle.0).is_some() {
            info!(handle = %handle.0, "Unloaded ONNX model");
            Ok(())
        } else {
            Err(IfranError::ModelNotFound(handle.0))
        }
    }

    async fn infer(
        &self,
        handle: &ModelHandle,
        _req: &InferenceRequest,
    ) -> Result<InferenceResponse> {
        let loaded = self.loaded.read().await;
        if !loaded.contains_key(&handle.0) {
            return Err(IfranError::ModelNotFound(handle.0.clone()));
        }

        Err(IfranError::BackendError(
            "ONNX inference not yet wired — requires ort crate dependency".into(),
        ))
    }

    async fn infer_stream(
        &self,
        handle: &ModelHandle,
        _req: InferenceRequest,
    ) -> Result<mpsc::Receiver<StreamChunk>> {
        let loaded = self.loaded.read().await;
        if !loaded.contains_key(&handle.0) {
            return Err(IfranError::ModelNotFound(handle.0.clone()));
        }

        // ONNX Runtime doesn't natively support streaming generation,
        // but we could implement token-by-token generation with manual KV cache.
        Err(IfranError::BackendError(
            "ONNX streaming not supported — use non-streaming inference".into(),
        ))
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id() {
        let backend = OnnxBackend::new();
        assert_eq!(backend.id().0, "onnx");
    }

    #[test]
    fn supports_onnx_format() {
        let backend = OnnxBackend::new();
        assert_eq!(backend.supported_formats(), &[ModelFormat::Onnx]);
    }

    #[test]
    fn no_streaming() {
        let backend = OnnxBackend::new();
        assert!(!backend.capabilities().supports_streaming);
    }

    #[test]
    fn default_constructor() {
        let backend = OnnxBackend::default();
        assert_eq!(backend.id().0, "onnx");
    }

    #[test]
    fn capabilities_details() {
        let backend = OnnxBackend::new();
        let caps = backend.capabilities();
        assert!(caps.supports_embeddings);
        assert!(caps.supports_vision);
        assert_eq!(caps.max_context_length, None);
    }

    #[tokio::test]
    async fn health_check_ok() {
        let backend = OnnxBackend::new();
        assert!(backend.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn load_and_unload_model() {
        let backend = OnnxBackend::new();
        let manifest = ModelManifest {
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
            info: ifran_types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test-onnx".into(),
                repo_id: None,
                format: ModelFormat::Onnx,
                quant: ifran_types::model::QuantLevel::None,
                size_bytes: 500,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/model.onnx".into(),
                sha256: None,
                pulled_at: chrono::Utc::now(),
            },
        };
        let device = DeviceConfig {
            accelerator: AcceleratorType::Cpu,
            device_ids: vec![0],
            memory_limit_mb: None,
        };
        let handle = backend.load_model(&manifest, &device).await.unwrap();
        assert!(handle.0.starts_with("onnx-"));
        backend.unload_model(handle).await.unwrap();
    }

    #[tokio::test]
    async fn unload_nonexistent_fails() {
        let backend = OnnxBackend::new();
        assert!(
            backend
                .unload_model(ModelHandle("nope".into()))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn infer_not_implemented() {
        let backend = OnnxBackend::new();
        let manifest = ModelManifest {
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
            info: ifran_types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test".into(),
                repo_id: None,
                format: ModelFormat::Onnx,
                quant: ifran_types::model::QuantLevel::None,
                size_bytes: 0,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/m.onnx".into(),
                sha256: None,
                pulled_at: chrono::Utc::now(),
            },
        };
        let device = DeviceConfig {
            accelerator: AcceleratorType::Cpu,
            device_ids: vec![0],
            memory_limit_mb: None,
        };
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
        assert!(backend.infer(&handle, &req).await.is_err());
        assert!(backend.infer_stream(&handle, req).await.is_err());
    }
}
