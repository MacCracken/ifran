//! Smart backend selection and routing.
//!
//! The router is responsible for choosing the best available backend for a given
//! inference request based on model format, device availability, load, and
//! backend capabilities.

use crate::traits::InferenceBackend;
use dashmap::DashMap;
use std::sync::Arc;
use synapse_types::backend::BackendId;

/// Registry that holds all available backends and routes requests to the most
/// appropriate one.
pub struct BackendRouter {
    backends: DashMap<BackendId, Arc<dyn InferenceBackend>>,
}

impl BackendRouter {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            backends: DashMap::new(),
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
        self.backends.iter().map(|entry| entry.key().clone()).collect()
    }
}

impl Default for BackendRouter {
    fn default() -> Self {
        Self::new()
    }
}
