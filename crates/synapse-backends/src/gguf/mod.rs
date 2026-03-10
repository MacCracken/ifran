//! GGUF backend — direct GGUF model loading via Candle.
//!
//! Loads GGUF-quantized models directly in-process using candle-gguf,
//! without requiring an external server like llama.cpp. Useful for
//! lightweight or embedded deployments.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use synapse_types::backend::{AcceleratorType, BackendCapabilities, BackendId, DeviceConfig};
use synapse_types::error::Result;
use synapse_types::inference::{InferenceRequest, InferenceResponse, StreamChunk};
use synapse_types::model::{ModelFormat, ModelManifest};
use synapse_types::SynapseError;
use tokio::sync::{mpsc, RwLock};
use tracing::info;

use crate::traits::{InferenceBackend, ModelHandle};

/// Metadata for a loaded GGUF model.
struct LoadedGguf {
    #[allow(dead_code)]
    model_path: String,
}

/// GGUF backend for direct in-process GGUF inference.
pub struct GgufBackend {
    loaded: Arc<RwLock<HashMap<String, LoadedGguf>>>,
}

impl GgufBackend {
    pub fn new() -> Self {
        Self {
            loaded: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for GgufBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceBackend for GgufBackend {
    fn id(&self) -> BackendId {
        BackendId("gguf".into())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            accelerators: vec![AcceleratorType::Cpu, AcceleratorType::Cuda],
            max_context_length: Some(32768),
            supports_streaming: true,
            supports_embeddings: false,
            supports_vision: false,
        }
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Gguf]
    }

    async fn load_model(
        &self,
        manifest: &ModelManifest,
        _device: &DeviceConfig,
    ) -> Result<ModelHandle> {
        let model_path = &manifest.info.local_path;

        // In a full implementation:
        // 1. Parse the GGUF file header to read model architecture
        // 2. Load quantized tensors into memory
        // 3. Build the computation graph with candle-gguf
        // 4. Initialize KV cache

        let handle_id = format!("gguf-{}", manifest.info.id);
        info!(handle = %handle_id, model = %model_path, "Loading GGUF model directly");

        self.loaded.write().await.insert(
            handle_id.clone(),
            LoadedGguf {
                model_path: model_path.clone(),
            },
        );

        Ok(ModelHandle(handle_id))
    }

    async fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        if self.loaded.write().await.remove(&handle.0).is_some() {
            info!(handle = %handle.0, "Unloaded GGUF model");
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
            "GGUF direct inference not yet wired — requires candle-gguf dependency".into(),
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

        Err(SynapseError::BackendError(
            "GGUF direct streaming not yet wired — requires candle-gguf dependency".into(),
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
        let backend = GgufBackend::new();
        assert_eq!(backend.id().0, "gguf");
    }

    #[test]
    fn supports_gguf_format() {
        let backend = GgufBackend::new();
        assert_eq!(backend.supported_formats(), &[ModelFormat::Gguf]);
    }

    #[tokio::test]
    async fn health_check_ok() {
        let backend = GgufBackend::new();
        assert!(backend.health_check().await.unwrap());
    }
}
