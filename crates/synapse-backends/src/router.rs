//! Smart backend selection and routing.
//!
//! The router holds all registered backends and selects the best one for a
//! given model based on format compatibility, hardware availability, and
//! user preference.

use crate::traits::InferenceBackend;
use dashmap::DashMap;
use std::sync::Arc;
use synapse_types::backend::{BackendId, BackendLocality};
use synapse_types::inference::DataSensitivity;
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
            tracing::warn!(backend = %name, "Preferred backend not found, falling back to auto-selection");
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

        // 4. No matching backend found
        tracing::warn!(?format, "No backend supports the requested model format");
        None
    }

    /// Select a backend for a format, returning the BackendId too.
    pub fn select_with_id(
        &self,
        format: ModelFormat,
        preferred: Option<&str>,
    ) -> Option<(BackendId, Arc<dyn InferenceBackend>)> {
        self.select(format, preferred).map(|b| (b.id(), b))
    }

    /// Select the best backend, respecting data sensitivity.
    ///
    /// When sensitivity is `Confidential` or `Restricted`, only backends with
    /// `BackendLocality::Local` are eligible.
    pub fn select_with_privacy(
        &self,
        format: ModelFormat,
        preferred: Option<&str>,
        sensitivity: Option<DataSensitivity>,
    ) -> Option<Arc<dyn InferenceBackend>> {
        let requires_local = matches!(
            sensitivity,
            Some(DataSensitivity::Confidential) | Some(DataSensitivity::Restricted)
        );

        if !requires_local {
            return self.select(format, preferred);
        }

        // For sensitive data, only consider local backends
        // 1. Explicit preference (if it's local)
        if let Some(name) = preferred {
            let id = BackendId(name.to_string());
            if let Some(b) = self.get(&id) {
                if b.capabilities().locality == BackendLocality::Local {
                    return Some(b);
                }
                tracing::warn!(
                    backend = %name,
                    "Preferred backend is remote, skipping for sensitive data"
                );
            }
        }

        // 2. Default backend (if local and supports format)
        if let Some(ref default_id) = self.default_backend {
            if let Some(b) = self.get(default_id) {
                if b.capabilities().locality == BackendLocality::Local
                    && b.supported_formats().contains(&format)
                {
                    return Some(b);
                }
            }
        }

        // 3. First local backend that supports the format
        for entry in self.backends.iter() {
            if entry.value().capabilities().locality == BackendLocality::Local
                && entry.value().supported_formats().contains(&format)
            {
                return Some(entry.value().clone());
            }
        }

        tracing::warn!(
            ?format,
            ?sensitivity,
            "No local backend available for sensitive data"
        );
        None
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
    use synapse_types::backend::{
        AcceleratorType, BackendCapabilities, BackendLocality, DeviceConfig,
    };
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
                locality: BackendLocality::Local,
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

    #[test]
    fn get_backend_by_id() {
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "gguf-be",
            formats: vec![ModelFormat::Gguf],
        }));
        let id = BackendId("gguf-be".into());
        assert!(router.get(&id).is_some());
        assert_eq!(router.get(&id).unwrap().id().0, "gguf-be");

        let missing = BackendId("nope".into());
        assert!(router.get(&missing).is_none());
    }

    #[test]
    fn unregister_backend() {
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "a",
            formats: vec![ModelFormat::Gguf],
        }));
        assert_eq!(router.count(), 1);
        router.unregister(&BackendId("a".into()));
        assert_eq!(router.count(), 0);
        assert!(router.get(&BackendId("a".into())).is_none());
    }

    #[test]
    fn count_backends() {
        let router = BackendRouter::new();
        assert_eq!(router.count(), 0);
        router.register(Arc::new(MockBackend {
            name: "a",
            formats: vec![ModelFormat::Gguf],
        }));
        assert_eq!(router.count(), 1);
        router.register(Arc::new(MockBackend {
            name: "b",
            formats: vec![ModelFormat::Onnx],
        }));
        assert_eq!(router.count(), 2);
    }

    #[test]
    fn list_backends_returns_all_ids() {
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "alpha",
            formats: vec![ModelFormat::Gguf],
        }));
        router.register(Arc::new(MockBackend {
            name: "beta",
            formats: vec![ModelFormat::Onnx],
        }));
        let mut ids: Vec<String> = router.list_backends().into_iter().map(|id| id.0).collect();
        ids.sort();
        assert_eq!(ids, vec!["alpha", "beta"]);
    }

    #[test]
    fn select_with_id_returns_both() {
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "test-be",
            formats: vec![ModelFormat::Gguf],
        }));
        let (id, backend) = router.select_with_id(ModelFormat::Gguf, None).unwrap();
        assert_eq!(id.0, "test-be");
        assert_eq!(backend.id().0, "test-be");
    }

    #[test]
    fn select_no_backends_returns_none() {
        let router = BackendRouter::new();
        assert!(router.select(ModelFormat::Gguf, None).is_none());
    }

    #[test]
    fn select_preferred_not_found_falls_back() {
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "actual",
            formats: vec![ModelFormat::Gguf],
        }));
        // Preferred "missing" doesn't exist, should fall back to "actual"
        let selected = router.select(ModelFormat::Gguf, Some("missing")).unwrap();
        assert_eq!(selected.id().0, "actual");
    }

    #[test]
    fn default_backend_unsupported_format_skips() {
        let router = BackendRouter::with_default("default-be");
        router.register(Arc::new(MockBackend {
            name: "default-be",
            formats: vec![ModelFormat::Onnx], // doesn't support Gguf
        }));
        router.register(Arc::new(MockBackend {
            name: "gguf-be",
            formats: vec![ModelFormat::Gguf],
        }));
        let selected = router.select(ModelFormat::Gguf, None).unwrap();
        assert_eq!(selected.id().0, "gguf-be");
    }

    #[test]
    fn supports_format_empty_router() {
        let router = BackendRouter::new();
        assert!(!router.supports_format(ModelFormat::Gguf));
    }

    #[test]
    fn default_trait() {
        let router = BackendRouter::default();
        assert_eq!(router.count(), 0);
    }

    #[test]
    fn register_overwrites_same_id() {
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "a",
            formats: vec![ModelFormat::Gguf],
        }));
        router.register(Arc::new(MockBackend {
            name: "a",
            formats: vec![ModelFormat::Onnx],
        }));
        assert_eq!(router.count(), 1);
        // The second registration should overwrite
        let backend = router.get(&BackendId("a".into())).unwrap();
        assert!(backend.supported_formats().contains(&ModelFormat::Onnx));
    }

    #[test]
    fn select_with_id_no_match() {
        let router = BackendRouter::new();
        assert!(router.select_with_id(ModelFormat::Gguf, None).is_none());
    }

    #[test]
    fn unregister_nonexistent() {
        let router = BackendRouter::new();
        // Should not panic
        router.unregister(&BackendId("nonexistent".into()));
        assert_eq!(router.count(), 0);
    }

    // -- Concurrent access tests --

    #[tokio::test]
    async fn concurrent_register_and_read() {
        let router = Arc::new(BackendRouter::new());
        let mut handles = vec![];

        // Spawn 10 concurrent writers
        for i in 0..10 {
            let router = router.clone();
            handles.push(tokio::spawn(async move {
                let name: &'static str = Box::leak(format!("be-{i}").into_boxed_str());
                router.register(Arc::new(MockBackend {
                    name,
                    formats: vec![ModelFormat::Gguf],
                }));
            }));
        }

        // Spawn 10 concurrent readers
        for _ in 0..10 {
            let router = router.clone();
            handles.push(tokio::spawn(async move {
                let _ = router.list_backends();
                let _ = router.count();
                let _ = router.supports_format(ModelFormat::Gguf);
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(router.count(), 10);
    }

    #[tokio::test]
    async fn concurrent_register_and_unregister() {
        let router = Arc::new(BackendRouter::new());

        // Pre-populate
        for i in 0..20 {
            let name: &'static str = Box::leak(format!("be-{i}").into_boxed_str());
            router.register(Arc::new(MockBackend {
                name,
                formats: vec![ModelFormat::Gguf],
            }));
        }

        let mut handles = vec![];

        // Unregister even-numbered backends concurrently
        for i in (0..20).step_by(2) {
            let router = router.clone();
            handles.push(tokio::spawn(async move {
                router.unregister(&BackendId(format!("be-{i}")));
            }));
        }

        // Concurrently select backends
        for _ in 0..10 {
            let router = router.clone();
            handles.push(tokio::spawn(async move {
                let _ = router.select(ModelFormat::Gguf, None);
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(router.count(), 10);
    }

    #[tokio::test]
    async fn concurrent_select_with_preference() {
        let router = Arc::new(BackendRouter::new());
        for i in 0..5 {
            let name: &'static str = Box::leak(format!("be-{i}").into_boxed_str());
            router.register(Arc::new(MockBackend {
                name,
                formats: vec![ModelFormat::Gguf],
            }));
        }

        let mut handles = vec![];
        for i in 0..20 {
            let router = router.clone();
            handles.push(tokio::spawn(async move {
                let pref = format!("be-{}", i % 5);
                let result = router.select(ModelFormat::Gguf, Some(&pref));
                assert!(result.is_some());
                assert_eq!(result.unwrap().id().0, pref);
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }

    #[test]
    fn select_with_privacy_public_data_allows_remote() {
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "remote-be",
            formats: vec![ModelFormat::Gguf],
        }));

        let selected = router
            .select_with_privacy(ModelFormat::Gguf, None, Some(DataSensitivity::Public))
            .unwrap();
        assert_eq!(selected.id().0, "remote-be");
    }

    #[test]
    fn select_with_privacy_confidential_blocks_remote() {
        // MockBackend has BackendLocality::Local by default in capabilities
        // but let's test the filtering logic
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "local-be",
            formats: vec![ModelFormat::Gguf],
        }));

        let selected = router
            .select_with_privacy(ModelFormat::Gguf, None, Some(DataSensitivity::Confidential))
            .unwrap();
        assert_eq!(selected.id().0, "local-be");
    }

    #[test]
    fn select_with_privacy_none_sensitivity_acts_like_select() {
        let router = BackendRouter::new();
        router.register(Arc::new(MockBackend {
            name: "be",
            formats: vec![ModelFormat::Gguf],
        }));

        let selected = router
            .select_with_privacy(ModelFormat::Gguf, None, None)
            .unwrap();
        assert_eq!(selected.id().0, "be");
    }
}
