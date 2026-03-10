//! Bridge gRPC client — calls back to SecureYeoman.
//!
//! Implements the client side of the `YeomanBridge` gRPC service defined
//! in bridge.proto. Synapse uses this to request GPU allocations, report
//! progress, and register completed models with SY.

use crate::protocol::{Capabilities, ConnectionState, ProtocolConfig};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Client for calling back to SecureYeoman.
pub struct BridgeClient {
    endpoint: String,
    config: ProtocolConfig,
    state: Arc<RwLock<ConnectionState>>,
    reconnect_count: Arc<RwLock<u32>>,
}

impl BridgeClient {
    pub fn new(endpoint: String, config: ProtocolConfig) -> Self {
        Self {
            endpoint,
            config,
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            reconnect_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Attempt to connect to SY.
    pub async fn connect(&self) -> synapse_types::error::Result<()> {
        info!(endpoint = %self.endpoint, "Connecting to SecureYeoman");
        *self.state.write().await = ConnectionState::Connecting;

        // TODO: Establish tonic gRPC channel to SY endpoint.
        // For now, simulate connection.
        *self.state.write().await = ConnectionState::Connected;
        *self.reconnect_count.write().await = 0;
        info!("Connected to SecureYeoman");
        Ok(())
    }

    /// Reconnect with exponential backoff.
    pub async fn reconnect(&self) -> synapse_types::error::Result<()> {
        let mut count = self.reconnect_count.write().await;
        if *count >= self.config.max_reconnect_attempts {
            warn!("Max reconnect attempts reached, entering degraded mode");
            *self.state.write().await = ConnectionState::Degraded;
            return Err(synapse_types::SynapseError::BridgeError(
                "Max reconnect attempts reached".into(),
            ));
        }

        *count += 1;
        let delay = self.config.reconnect_delay * (*count).min(6);
        info!(attempt = *count, delay_secs = delay.as_secs(), "Reconnecting to SY");
        drop(count);

        tokio::time::sleep(delay).await;
        self.connect().await
    }

    /// Current connection state.
    pub async fn connection_state(&self) -> ConnectionState {
        *self.state.read().await
    }

    /// Send a capabilities announcement to SY.
    pub async fn announce(&self, capabilities: Capabilities) -> synapse_types::error::Result<()> {
        info!(
            instance = %capabilities.instance_id,
            gpus = capabilities.gpu_count,
            "Announcing capabilities to SY"
        );
        // TODO: Call YeomanBridge.RegisterCompletedModel or a dedicated announce RPC.
        Ok(())
    }

    /// Request GPU allocation from SY.
    pub async fn request_gpu(&self, memory_mb: u64, count: u32) -> synapse_types::error::Result<Vec<u32>> {
        info!(memory_mb, count, "Requesting GPU allocation from SY");
        // TODO: Call YeomanBridge.RequestGpuAllocation.
        // For now, return empty (local-only mode).
        Ok(Vec::new())
    }

    /// Report training progress to SY.
    pub async fn report_progress(
        &self,
        job_id: &str,
        status: &str,
        step: u64,
        loss: f64,
    ) -> synapse_types::error::Result<()> {
        // TODO: Call YeomanBridge.ReportProgress streaming RPC.
        tracing::debug!(job_id, status, step, loss, "Reporting progress to SY");
        Ok(())
    }
}
