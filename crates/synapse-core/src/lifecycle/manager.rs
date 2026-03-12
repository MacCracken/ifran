//! Model load/unload/swap orchestration.
//!
//! The [`ModelManager`] coordinates loading models into backends, tracking
//! which models are currently loaded, and unloading them when needed. It
//! checks memory budgets before loading and maintains a mapping of
//! model IDs to backend handles.

use crate::hardware::detect;
use crate::lifecycle::memory;
use std::collections::HashMap;
use std::sync::Arc;
use synapse_types::SynapseError;
use synapse_types::backend::{AcceleratorType, DeviceConfig};
use synapse_types::error::Result;
use synapse_types::model::{ModelId, ModelManifest};
use tokio::sync::RwLock;

/// Tracks a loaded model and its backend handle.
#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub model_id: ModelId,
    pub model_name: String,
    pub handle: String,
    pub backend_id: String,
    pub vram_used_mb: u64,
}

/// Manages model loading/unloading across backends.
pub struct ModelManager {
    loaded: Arc<RwLock<HashMap<ModelId, LoadedModel>>>,
    gpu_reserve_mb: u64,
}

impl ModelManager {
    /// Create a new model manager.
    ///
    /// `gpu_reserve_mb` is the amount of VRAM to always keep free.
    pub fn new(gpu_reserve_mb: u64) -> Self {
        Self {
            loaded: Arc::new(RwLock::new(HashMap::new())),
            gpu_reserve_mb,
        }
    }

    /// Check memory budget and record a model as loaded.
    ///
    /// This performs the budget check but does NOT call backend.load_model() —
    /// that's the caller's responsibility. This separation allows the manager
    /// to be backend-agnostic.
    pub async fn prepare_load(&self, manifest: &ModelManifest) -> Result<DeviceConfig> {
        let hardware = detect::detect()?;

        let total_layers = 32; // TODO: read from GGUF metadata
        let estimate =
            memory::estimate_gguf(manifest.info.size_bytes, manifest.gpu_layers, total_layers);

        memory::check_budget(&hardware, &estimate, self.gpu_reserve_mb)?;

        // Determine device config based on available hardware
        let (accelerator, device_ids) = if hardware.has_gpu() {
            let kind = hardware.best_accelerator().ok_or_else(|| {
                SynapseError::HardwareError(
                    "GPU detected but no accelerator type could be determined".into(),
                )
            })?;
            let acc = match kind {
                detect::AcceleratorKind::Cuda => AcceleratorType::Cuda,
                detect::AcceleratorKind::Rocm => AcceleratorType::Rocm,
            };
            let ids: Vec<u32> = hardware.gpus.iter().map(|g| g.index as u32).collect();
            (acc, ids)
        } else {
            (AcceleratorType::Cpu, vec![])
        };

        Ok(DeviceConfig {
            accelerator,
            device_ids,
            memory_limit_mb: Some(estimate.vram_mb + estimate.ram_mb),
        })
    }

    /// Register a model as loaded.
    pub async fn register_loaded(
        &self,
        model_id: ModelId,
        model_name: String,
        handle: String,
        backend_id: String,
        vram_used_mb: u64,
    ) {
        let mut loaded = self.loaded.write().await;
        loaded.insert(
            model_id,
            LoadedModel {
                model_id,
                model_name,
                handle,
                backend_id,
                vram_used_mb,
            },
        );
    }

    /// Unregister a model (after the backend has unloaded it).
    pub async fn unregister(&self, model_id: &ModelId) -> Option<LoadedModel> {
        let mut loaded = self.loaded.write().await;
        loaded.remove(model_id)
    }

    /// Get info about a loaded model.
    pub async fn get_loaded(&self, model_id: &ModelId) -> Option<LoadedModel> {
        let loaded = self.loaded.read().await;
        loaded.get(model_id).cloned()
    }

    /// List all currently loaded models.
    pub async fn list_loaded(&self) -> Vec<LoadedModel> {
        let loaded = self.loaded.read().await;
        loaded.values().cloned().collect()
    }

    /// Total VRAM currently used by loaded models.
    pub async fn total_vram_used(&self) -> u64 {
        let loaded = self.loaded.read().await;
        loaded.values().map(|m| m.vram_used_mb).sum()
    }

    /// Check if a specific model is loaded.
    pub async fn is_loaded(&self, model_id: &ModelId) -> bool {
        let loaded = self.loaded.read().await;
        loaded.contains_key(model_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn register_and_list() {
        let manager = ModelManager::new(512);
        assert!(manager.list_loaded().await.is_empty());

        let id = uuid::Uuid::new_v4();
        manager
            .register_loaded(
                id,
                "test-model".into(),
                "handle-1".into(),
                "llamacpp".into(),
                4000,
            )
            .await;

        assert!(manager.is_loaded(&id).await);
        assert_eq!(manager.list_loaded().await.len(), 1);
        assert_eq!(manager.total_vram_used().await, 4000);
    }

    #[tokio::test]
    async fn get_loaded() {
        let manager = ModelManager::new(512);
        let id = uuid::Uuid::new_v4();
        manager
            .register_loaded(
                id,
                "test-model".into(),
                "handle-1".into(),
                "llamacpp".into(),
                4000,
            )
            .await;

        let loaded = manager.get_loaded(&id).await;
        assert!(loaded.is_some());
        let model = loaded.unwrap();
        assert_eq!(model.handle, "handle-1");
        assert_eq!(model.backend_id, "llamacpp");
        assert_eq!(model.vram_used_mb, 4000);
    }

    #[tokio::test]
    async fn get_loaded_not_found() {
        let manager = ModelManager::new(512);
        assert!(manager.get_loaded(&uuid::Uuid::new_v4()).await.is_none());
    }

    #[tokio::test]
    async fn multiple_models() {
        let manager = ModelManager::new(512);
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        manager
            .register_loaded(id1, "model-1".into(), "h1".into(), "llamacpp".into(), 2000)
            .await;
        manager
            .register_loaded(id2, "model-2".into(), "h2".into(), "ollama".into(), 3000)
            .await;

        assert_eq!(manager.list_loaded().await.len(), 2);
        assert_eq!(manager.total_vram_used().await, 5000);
        assert!(manager.is_loaded(&id1).await);
        assert!(manager.is_loaded(&id2).await);
    }

    #[tokio::test]
    async fn unregister() {
        let manager = ModelManager::new(512);
        let id = uuid::Uuid::new_v4();
        manager
            .register_loaded(
                id,
                "test-model".into(),
                "handle-1".into(),
                "llamacpp".into(),
                4000,
            )
            .await;

        let removed = manager.unregister(&id).await;
        assert!(removed.is_some());
        assert!(!manager.is_loaded(&id).await);
        assert_eq!(manager.total_vram_used().await, 0);
    }
}
