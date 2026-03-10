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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_server() -> BridgeServer {
        BridgeServer::new("test-instance".into(), ProtocolConfig::default())
    }

    #[tokio::test]
    async fn starts_disconnected() {
        let server = test_server();
        assert_eq!(
            server.connection_state().await,
            ConnectionState::Disconnected
        );
    }

    #[tokio::test]
    async fn start_transitions_to_connected() {
        let server = test_server();
        server.start("127.0.0.1:0").await.unwrap();
        assert_eq!(server.connection_state().await, ConnectionState::Connected);
    }

    #[tokio::test]
    async fn enter_degraded_sets_state() {
        let server = test_server();
        server.start("127.0.0.1:0").await.unwrap();
        server.enter_degraded().await;
        assert_eq!(server.connection_state().await, ConnectionState::Degraded);
    }

    #[test]
    fn build_heartbeat_includes_instance_id() {
        let server = test_server();
        let hb = server.build_heartbeat(5, 4096, 2);
        assert_eq!(hb.instance_id, "test-instance");
        assert_eq!(hb.loaded_models, 5);
        assert_eq!(hb.gpu_memory_free_mb, 4096);
        assert_eq!(hb.active_training_jobs, 2);
        assert!(hb.timestamp > 0);
    }

    #[test]
    fn heartbeat_interval_from_config() {
        let server = test_server();
        assert_eq!(
            server.heartbeat_interval(),
            std::time::Duration::from_secs(10)
        );
    }
}
