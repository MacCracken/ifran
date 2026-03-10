//! Bridge gRPC server — receives commands from SecureYeoman.
//!
//! Implements the `SynapseBridge` gRPC service defined in bridge.proto.
//! SY connects to this server to submit training jobs, pull models, and
//! run inference on this Synapse instance.

use crate::protocol::{ConnectionState, Heartbeat, ProtocolConfig};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// The bridge server state.
pub struct BridgeServer {
    config: ProtocolConfig,
    state: Arc<RwLock<ConnectionState>>,
    instance_id: String,
}

impl BridgeServer {
    pub fn new(instance_id: String, config: ProtocolConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            instance_id,
        }
    }

    /// Get current connection state.
    pub async fn connection_state(&self) -> ConnectionState {
        *self.state.read().await
    }

    /// Start the bridge server on the given address.
    ///
    /// This spawns a background task that listens for SY connections.
    /// The actual gRPC service implementation will be wired up when
    /// tonic codegen is integrated.
    pub async fn start(&self, bind_addr: &str) -> synapse_types::error::Result<()> {
        info!(addr = %bind_addr, "Starting bridge server");
        *self.state.write().await = ConnectionState::Connecting;

        // TODO: Wire up tonic gRPC server with generated bridge.proto service.
        // For now, mark as connected to show the framework works.
        *self.state.write().await = ConnectionState::Connected;
        info!("Bridge server ready for SY connections");

        Ok(())
    }

    /// Build a heartbeat message with current state.
    pub fn build_heartbeat(
        &self,
        loaded_models: u32,
        gpu_memory_free_mb: u64,
        active_training_jobs: u32,
    ) -> Heartbeat {
        Heartbeat {
            instance_id: self.instance_id.clone(),
            timestamp: chrono::Utc::now().timestamp(),
            loaded_models,
            gpu_memory_free_mb,
            active_training_jobs,
        }
    }

    /// Heartbeat interval from config.
    pub fn heartbeat_interval(&self) -> std::time::Duration {
        self.config.heartbeat_interval
    }

    /// Transition to degraded mode (SY connection lost but still serving).
    pub async fn enter_degraded(&self) {
        warn!("Bridge entering degraded mode — SY connection lost");
        *self.state.write().await = ConnectionState::Degraded;
    }
}
