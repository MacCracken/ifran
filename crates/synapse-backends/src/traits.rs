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
