//! GGUF backend — direct GGUF model loading via Candle.
//!
//! Loads GGUF-quantized models directly in-process using candle-gguf,
//! without requiring an external server like llama.cpp. Useful for
//! lightweight or embedded deployments.

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
            locality: BackendLocality::Local,
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
            return Err(IfranError::ModelNotFound(handle.0.clone()));
        }

        Err(IfranError::BackendError(
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

    #[test]
    fn default_constructor() {
        let backend = GgufBackend::default();
        assert_eq!(backend.id().0, "gguf");
    }

    #[test]
    fn capabilities_details() {
        let backend = GgufBackend::new();
        let caps = backend.capabilities();
        assert!(caps.accelerators.contains(&AcceleratorType::Cpu));
        assert!(caps.accelerators.contains(&AcceleratorType::Cuda));
        assert_eq!(caps.max_context_length, Some(32768));
        assert!(caps.supports_streaming);
        assert!(!caps.supports_embeddings);
        assert!(!caps.supports_vision);
    }

    #[tokio::test]
    async fn load_and_unload_model() {
        let backend = GgufBackend::new();
        let manifest = ModelManifest {
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
            info: ifran_types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test-model".into(),
                repo_id: None,
                format: ModelFormat::Gguf,
                quant: ifran_types::model::QuantLevel::Q4KM,
                size_bytes: 1000,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/model.gguf".into(),
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
        assert!(handle.0.starts_with("gguf-"));

        // Unload should succeed
        backend.unload_model(handle).await.unwrap();
    }

    #[tokio::test]
    async fn unload_nonexistent_fails() {
        let backend = GgufBackend::new();
        let handle = ModelHandle("nonexistent".into());
        assert!(backend.unload_model(handle).await.is_err());
    }

    #[tokio::test]
    async fn infer_not_implemented() {
        let backend = GgufBackend::new();
        let manifest = ModelManifest {
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
            info: ifran_types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test".into(),
                repo_id: None,
                format: ModelFormat::Gguf,
                quant: ifran_types::model::QuantLevel::None,
                size_bytes: 0,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/m.gguf".into(),
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
        // Infer should return error (not yet wired)
        assert!(backend.infer(&handle, &req).await.is_err());
    }

    #[tokio::test]
    async fn infer_stream_not_implemented() {
        let backend = GgufBackend::new();
        let manifest = ModelManifest {
            context_length: None,
            gpu_layers: None,
            tensor_split: None,
            info: ifran_types::model::ModelInfo {
                id: uuid::Uuid::new_v4(),
                name: "test".into(),
                repo_id: None,
                format: ModelFormat::Gguf,
                quant: ifran_types::model::QuantLevel::None,
                size_bytes: 0,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/m.gguf".into(),
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
        assert!(backend.infer_stream(&handle, req).await.is_err());
    }

    #[tokio::test]
    async fn infer_nonexistent_model_fails() {
        let backend = GgufBackend::new();
        let handle = ModelHandle("nonexistent".into());
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
