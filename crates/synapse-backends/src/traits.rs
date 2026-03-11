//! Core trait definitions for pluggable inference backends.

use async_trait::async_trait;
use synapse_types::{
    backend::{BackendCapabilities, BackendId, DeviceConfig},
    inference::{InferenceRequest, InferenceResponse, StreamChunk},
    model::{ModelFormat, ModelManifest},
};
use tokio::sync::mpsc;

/// Handle to a loaded model within a backend.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ModelHandle(pub String);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_handle_equality() {
        let a = ModelHandle("llamacpp-8430".into());
        let b = ModelHandle("llamacpp-8430".into());
        let c = ModelHandle("llamacpp-8431".into());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn model_handle_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ModelHandle("a".into()));
        set.insert(ModelHandle("a".into()));
        assert_eq!(set.len(), 1);
        set.insert(ModelHandle("b".into()));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn model_handle_clone() {
        let h = ModelHandle("test".into());
        let h2 = h.clone();
        assert_eq!(h, h2);
    }

    #[test]
    fn model_handle_debug() {
        let h = ModelHandle("test-handle".into());
        let debug = format!("{h:?}");
        assert!(debug.contains("test-handle"));
    }
}

/// The core trait that all inference backends must implement.
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    fn id(&self) -> BackendId;
    fn capabilities(&self) -> BackendCapabilities;
    fn supported_formats(&self) -> &[ModelFormat];
    async fn load_model(
        &self,
        manifest: &ModelManifest,
        device: &DeviceConfig,
    ) -> synapse_types::error::Result<ModelHandle>;
    async fn unload_model(&self, handle: ModelHandle) -> synapse_types::error::Result<()>;
    async fn infer(
        &self,
        handle: &ModelHandle,
        req: &InferenceRequest,
    ) -> synapse_types::error::Result<InferenceResponse>;
    async fn infer_stream(
        &self,
        handle: &ModelHandle,
        req: InferenceRequest,
    ) -> synapse_types::error::Result<mpsc::Receiver<StreamChunk>>;
    async fn health_check(&self) -> synapse_types::error::Result<bool>;
}
