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
        info!(
            attempt = *count,
            delay_secs = delay.as_secs(),
            "Reconnecting to SY"
        );
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
    pub async fn request_gpu(
        &self,
        memory_mb: u64,
        count: u32,
    ) -> synapse_types::error::Result<Vec<u32>> {
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

    /// Request SY to coordinate a remote worker assignment for distributed training.
    pub async fn request_worker_assignment(
        &self,
        job_id: &str,
        rank: u32,
        endpoint: &str,
        device_ids: &[u32],
    ) -> synapse_types::error::Result<()> {
        info!(
            job_id,
            rank,
            endpoint,
            device_count = device_ids.len(),
            "Requesting worker assignment from SY"
        );
        // TODO: Call SynapseBridge.RequestWorkerAssignment RPC.
        // SY will route this to the appropriate Synapse node.
        Ok(())
    }

    /// Notify SY that a checkpoint is ready for synchronization.
    pub async fn sync_checkpoint(
        &self,
        job_id: &str,
        rank: u32,
        checkpoint_path: &str,
    ) -> synapse_types::error::Result<()> {
        info!(
            job_id,
            rank, checkpoint_path, "Notifying SY of checkpoint ready for sync"
        );
        // TODO: Call SynapseBridge.SyncCheckpoint RPC.
        // SY coordinates checkpoint transfer between nodes.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_client() -> BridgeClient {
        BridgeClient::new("http://127.0.0.1:9420".into(), ProtocolConfig::default())
    }

    #[tokio::test]
    async fn starts_disconnected() {
        let client = test_client();
        assert_eq!(
            client.connection_state().await,
            ConnectionState::Disconnected
        );
    }

    #[tokio::test]
    async fn connect_transitions_to_connected() {
        let client = test_client();
        client.connect().await.unwrap();
        assert_eq!(client.connection_state().await, ConnectionState::Connected);
    }

    #[tokio::test]
    async fn reconnect_exceeds_max_attempts() {
        let config = ProtocolConfig {
            max_reconnect_attempts: 0,
            reconnect_delay: std::time::Duration::from_millis(1),
            ..ProtocolConfig::default()
        };
        let client = BridgeClient::new("http://127.0.0.1:9420".into(), config);
        let result = client.reconnect().await;
        assert!(result.is_err());
        assert_eq!(client.connection_state().await, ConnectionState::Degraded);
    }

    #[tokio::test]
    async fn announce_succeeds() {
        let client = test_client();
        client.connect().await.unwrap();
        let caps = Capabilities {
            instance_id: "test".into(),
            version: "1.0".into(),
            gpu_count: 1,
            total_gpu_memory_mb: 8192,
            supported_methods: vec!["lora".into()],
        };
        client.announce(caps).await.unwrap();
    }

    #[tokio::test]
    async fn request_gpu_returns_empty() {
        let client = test_client();
        client.connect().await.unwrap();
        let gpus = client.request_gpu(4096, 1).await.unwrap();
        assert!(gpus.is_empty());
    }

    #[tokio::test]
    async fn report_progress_succeeds() {
        let client = test_client();
        client
            .report_progress("job-1", "running", 100, 0.5)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn request_worker_assignment_succeeds() {
        let client = test_client();
        client.connect().await.unwrap();
        client
            .request_worker_assignment("job-1", 1, "http://node-2:9000", &[0, 1])
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn sync_checkpoint_succeeds() {
        let client = test_client();
        client.connect().await.unwrap();
        client
            .sync_checkpoint("job-1", 0, "/tmp/checkpoints/step-100")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn connect_resets_reconnect_count() {
        let config = ProtocolConfig {
            max_reconnect_attempts: 5,
            reconnect_delay: std::time::Duration::from_millis(1),
            ..ProtocolConfig::default()
        };
        let client = BridgeClient::new("http://127.0.0.1:9420".into(), config);

        // Bump reconnect count
        *client.reconnect_count.write().await = 3;
        assert_eq!(*client.reconnect_count.read().await, 3);

        // Connect should reset it
        client.connect().await.unwrap();
        assert_eq!(*client.reconnect_count.read().await, 0);
    }

    #[tokio::test]
    async fn connect_transitions_through_connecting() {
        let client = test_client();
        assert_eq!(
            client.connection_state().await,
            ConnectionState::Disconnected
        );
        client.connect().await.unwrap();
        // After connect(), should be Connected (Connecting is transient)
        assert_eq!(client.connection_state().await, ConnectionState::Connected);
    }

    #[tokio::test]
    async fn report_progress_works_without_connect() {
        // report_progress doesn't check connection state (stub)
        let client = test_client();
        let result = client.report_progress("job-1", "running", 0, 0.0).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn request_gpu_returns_empty_in_stub() {
        let client = test_client();
        let gpus = client.request_gpu(16384, 4).await.unwrap();
        assert!(gpus.is_empty());
    }

    #[tokio::test]
    async fn request_gpu_edge_cases() {
        let client = test_client();
        // Zero memory, zero count
        let gpus = client.request_gpu(0, 0).await.unwrap();
        assert!(gpus.is_empty());
        // Max values
        let gpus = client.request_gpu(u64::MAX, u32::MAX).await.unwrap();
        assert!(gpus.is_empty());
    }

    #[tokio::test]
    async fn report_progress_extreme_values() {
        let client = test_client();
        client
            .report_progress("", "running", u64::MAX, f64::INFINITY)
            .await
            .unwrap();
        client
            .report_progress("job-1", "", 0, 0.0)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn worker_assignment_empty_devices() {
        let client = test_client();
        client
            .request_worker_assignment("job-1", 0, "http://localhost:9000", &[])
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn sync_checkpoint_empty_path() {
        let client = test_client();
        client.sync_checkpoint("job-1", 0, "").await.unwrap();
    }

    #[tokio::test]
    async fn announce_multiple_times() {
        let client = test_client();
        client.connect().await.unwrap();
        for i in 0..3 {
            let caps = Capabilities {
                instance_id: format!("test-{i}"),
                version: "1.0".into(),
                gpu_count: i,
                total_gpu_memory_mb: 8192,
                supported_methods: vec![],
            };
            client.announce(caps).await.unwrap();
        }
        assert_eq!(client.connection_state().await, ConnectionState::Connected);
    }

    #[tokio::test]
    async fn reconnect_succeeds_within_limit() {
        let config = ProtocolConfig {
            max_reconnect_attempts: 3,
            reconnect_delay: std::time::Duration::from_millis(1),
            ..ProtocolConfig::default()
        };
        let client = BridgeClient::new("http://127.0.0.1:9420".into(), config);
        // First reconnect should succeed (attempt 1 <= 3)
        client.reconnect().await.unwrap();
        assert_eq!(client.connection_state().await, ConnectionState::Connected);
        // reconnect_count should have been reset by successful connect
        assert_eq!(*client.reconnect_count.read().await, 0);
    }
}
