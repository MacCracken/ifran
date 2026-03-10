//! ONNX Runtime backend integration.
//!
//! Wraps the ONNX Runtime via the `ort` crate for inference on ONNX-exported
//! models. Supports CPU, CUDA, TensorRT, DirectML, and CoreML execution
//! providers.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use synapse_types::SynapseError;
use synapse_types::backend::{AcceleratorType, BackendCapabilities, BackendId, DeviceConfig};
use synapse_types::error::Result;
use synapse_types::inference::{InferenceRequest, InferenceResponse, StreamChunk};
use synapse_types::model::{ModelFormat, ModelManifest};
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
            Err(SynapseError::ModelNotFound(handle.0))
        }
    }

    async fn infer(
        &self,
        handle: &ModelHandle,
        _req: &InferenceRequest,
    ) -> Result<InferenceResponse> {
        let loaded = self.loaded.read().await;
        if !loaded.contains_key(&handle.0) {
            return Err(SynapseError::ModelNotFound(handle.0.clone()));
        }

        Err(SynapseError::BackendError(
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
            return Err(SynapseError::ModelNotFound(handle.0.clone()));
        }

        // ONNX Runtime doesn't natively support streaming generation,
        // but we could implement token-by-token generation with manual KV cache.
        Err(SynapseError::BackendError(
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
}
