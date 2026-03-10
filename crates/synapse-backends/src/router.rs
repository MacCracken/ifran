//! Smart backend selection and routing.
//!
//! The router holds all registered backends and selects the best one for a
//! given model based on format compatibility, hardware availability, and
//! user preference.

use crate::traits::InferenceBackend;
use dashmap::DashMap;
use std::sync::Arc;
use synapse_types::backend::BackendId;
use synapse_types::model::ModelFormat;

/// Registry that holds all available backends and routes requests to the most
/// appropriate one.
pub struct BackendRouter {
    backends: DashMap<BackendId, Arc<dyn InferenceBackend>>,
    /// User-configured default backend (from config).
    default_backend: Option<BackendId>,
}

impl BackendRouter {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            backends: DashMap::new(),
            default_backend: None,
        }
    }

    /// Create a router with a preferred default backend.
    pub fn with_default(default: &str) -> Self {
        Self {
            backends: DashMap::new(),
            default_backend: Some(BackendId(default.to_string())),
        }
    }

    /// Register a backend with the router.
    pub fn register(&self, backend: Arc<dyn InferenceBackend>) {
        let id = backend.id();
        self.backends.insert(id, backend);
    }

    /// Remove a backend from the router.
    pub fn unregister(&self, id: &BackendId) {
        self.backends.remove(id);
    }

    /// Get a specific backend by its identifier.
    pub fn get(&self, id: &BackendId) -> Option<Arc<dyn InferenceBackend>> {
        self.backends.get(id).map(|entry| entry.value().clone())
    }

    /// List all registered backend identifiers.
    pub fn list_backends(&self) -> Vec<BackendId> {
        self.backends
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Select the best backend for a given model format.
    ///
    /// Selection priority:
    /// 1. User-specified backend (if provided and registered)
    /// 2. Configured default backend (if it supports the format)
    /// 3. First backend that supports the format
    /// 4. Any backend (fallback)
    pub fn select(
        &self,
        format: ModelFormat,
        preferred: Option<&str>,
    ) -> Option<Arc<dyn InferenceBackend>> {
        // 1. Explicit preference
        if let Some(name) = preferred {
            let id = BackendId(name.to_string());
            if let Some(b) = self.get(&id) {
                return Some(b);
            }
        }

        // 2. Configured default, if it supports the format
        if let Some(ref default_id) = self.default_backend {
            if let Some(b) = self.get(default_id) {
                if b.supported_formats().contains(&format) {
                    return Some(b);
                }
            }
        }

        // 3. First backend that supports the format
        for entry in self.backends.iter() {
            if entry.value().supported_formats().contains(&format) {
                return Some(entry.value().clone());
            }
        }

        // 4. Any backend
        self.backends.iter().next().map(|e| e.value().clone())
    }

    /// Select a backend for a format, returning the BackendId too.
    pub fn select_with_id(
        &self,
        format: ModelFormat,
        preferred: Option<&str>,
    ) -> Option<(BackendId, Arc<dyn InferenceBackend>)> {
        self.select(format, preferred).map(|b| (b.id(), b))
    }

    /// Check if any backend supports a given format.
    pub fn supports_format(&self, format: ModelFormat) -> bool {
        self.backends
            .iter()
            .any(|entry| entry.value().supported_formats().contains(&format))
    }

    /// Number of registered backends.
    pub fn count(&self) -> usize {
        self.backends.len()
    }
}

impl Default for BackendRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use synapse_types::backend::{AcceleratorType, BackendCapabilities, DeviceConfig};
    use synapse_types::inference::{
        FinishReason, InferenceRequest, InferenceResponse, StreamChunk, TokenUsage,
    };
    use synapse_types::model::ModelManifest;
    use tokio::sync::mpsc;

    struct MockBackend {
        name: &'static str,
        formats: Vec<ModelFormat>,
    }

    #[async_trait]
    impl InferenceBackend for MockBackend {
        fn id(&self) -> BackendId {
            BackendId(self.name.into())
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                accelerators: vec![AcceleratorType::Cpu],
                max_context_length: None,
                supports_streaming: false,
                supports_embeddings: false,
                supports_vision: false,
            }
        }
        fn supported_formats(&self) -> &[ModelFormat] {
            &self.formats
        }
        async fn load_model(
            &self,
            _: &ModelManifest,
            _: &DeviceConfig,
        ) -> synapse_types::error::Result<crate::ModelHandle> {
            Ok(crate::ModelHandle("mock".into()))
        }
        async fn unload_model(&self, _: crate::ModelHandle) -> synapse_types::error::Result<()> {
            Ok(())
        }
        async fn infer(
            &self,
            _: &crate::ModelHandle,
            _: &InferenceRequest,
        ) -> synapse_types::error::Result<InferenceResponse> {
            Ok(InferenceResponse {
                text: "mock".into(),
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
            _: &crate::ModelHandle,
            _: InferenceRequest,
        ) -> synapse_types::error::Result<mpsc::Receiver<StreamChunk>> {
            let (_, rx) = mpsc::channel(1);
            Ok(rx)
        }
        async fn health_check(&self) -> synapse_types::error::Result<bool> {
            Ok(true)
        }
    }

    #[test]
    fn select_by_format() {
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "gguf-be",
            formats: vec![ModelFormat::Gguf],
        }));
        router.register(Arc::new(MockBackend {
            name: "st-be",
            formats: vec![ModelFormat::SafeTensors],
        }));

        let selected = router.select(ModelFormat::Gguf, None).unwrap();
        assert_eq!(selected.id().0, "gguf-be");

        let selected = router.select(ModelFormat::SafeTensors, None).unwrap();
        assert_eq!(selected.id().0, "st-be");
    }

    #[test]
    fn select_explicit_preference() {
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "a",
            formats: vec![ModelFormat::Gguf],
        }));
        router.register(Arc::new(MockBackend {
            name: "b",
            formats: vec![ModelFormat::Gguf],
        }));

        let selected = router.select(ModelFormat::Gguf, Some("b")).unwrap();
        assert_eq!(selected.id().0, "b");
    }

    #[test]
    fn select_default_backend() {
        let router = BackendRouter::with_default("preferred");
        router.register(Arc::new(MockBackend {
            name: "other",
            formats: vec![ModelFormat::Gguf],
        }));
        router.register(Arc::new(MockBackend {
            name: "preferred",
            formats: vec![ModelFormat::Gguf],
        }));

        let selected = router.select(ModelFormat::Gguf, None).unwrap();
        assert_eq!(selected.id().0, "preferred");
    }

    #[test]
    fn supports_format_check() {
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "a",
            formats: vec![ModelFormat::Gguf],
        }));
        assert!(router.supports_format(ModelFormat::Gguf));
        assert!(!router.supports_format(ModelFormat::Onnx));
    }
}
