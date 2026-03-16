//! Bridge gRPC client — calls back to SecureYeoman.
//!
//! Implements the client side of the `YeomanBridge` gRPC service defined
//! in bridge.proto. Synapse uses this to request GPU allocations, report
//! progress, and register completed models with SY.

use crate::protocol::{Capabilities, ConnectionState, ProtocolConfig};
use std::sync::Arc;
use synapse_types::bridge::{
    GpuRequest, ModelRegistration, ProgressUpdate, ScaleRequest, ScaleResponse,
    yeoman_bridge_client::YeomanBridgeClient,
};
use tokio::sync::RwLock;
use tonic::transport::Channel;
use tracing::{info, warn};

/// Client for calling back to SecureYeoman.
pub struct BridgeClient {
    endpoint: String,
    config: ProtocolConfig,
    state: Arc<RwLock<ConnectionState>>,
    reconnect_count: Arc<RwLock<u32>>,
    grpc_client: Arc<RwLock<Option<YeomanBridgeClient<Channel>>>>,
}

impl BridgeClient {
    pub fn new(endpoint: String, config: ProtocolConfig) -> Self {
        Self {
            endpoint,
            config,
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            reconnect_count: Arc::new(RwLock::new(0)),
            grpc_client: Arc::new(RwLock::new(None)),
        }
    }

    /// Attempt to connect to SY.
    pub async fn connect(&self) -> synapse_types::error::Result<()> {
        info!(endpoint = %self.endpoint, "Connecting to SecureYeoman");
        *self.state.write().await = ConnectionState::Connecting;

        match Channel::from_shared(self.endpoint.clone())
            .map_err(|e| synapse_types::SynapseError::BridgeError(e.to_string()))?
            .connect()
            .await
        {
            Ok(channel) => {
                let client = YeomanBridgeClient::new(channel);
                *self.grpc_client.write().await = Some(client);
                *self.state.write().await = ConnectionState::Connected;
                *self.reconnect_count.write().await = 0;
                info!("Connected to SecureYeoman");
                Ok(())
            }
            Err(e) => {
                warn!(error = %e, "Failed to connect to SecureYeoman, entering degraded mode");
                *self.state.write().await = ConnectionState::Degraded;
                *self.reconnect_count.write().await = 0;
                Ok(())
            }
        }
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

    /// Get a clone of the gRPC client, if connected.
    async fn get_client(
        &self,
    ) -> synapse_types::error::Result<Option<YeomanBridgeClient<Channel>>> {
        Ok(self.grpc_client.read().await.clone())
    }

    /// Send a capabilities announcement to SY.
    pub async fn announce(&self, capabilities: Capabilities) -> synapse_types::error::Result<()> {
        info!(
            instance = %capabilities.instance_id,
            gpus = capabilities.gpu_count,
            "Announcing capabilities to SY"
        );

        if let Some(mut client) = self.get_client().await? {
            let request = tonic::Request::new(ModelRegistration {
                model_name: capabilities.instance_id,
                model_path: String::new(),
                base_model: format!("gpus:{}", capabilities.gpu_count),
                training_method: capabilities.supported_methods.join(","),
            });
            client
                .register_completed_model(request)
                .await
                .map_err(|e| {
                    synapse_types::SynapseError::BridgeError(format!(
                        "RegisterCompletedModel RPC failed: {e}"
                    ))
                })?;
        } else {
            warn!("No gRPC connection — announce skipped (degraded mode)");
        }

        Ok(())
    }

    /// Request GPU allocation from SY.
    pub async fn request_gpu(
        &self,
        memory_mb: u64,
        count: u32,
    ) -> synapse_types::error::Result<Vec<u32>> {
        info!(memory_mb, count, "Requesting GPU allocation from SY");

        if let Some(mut client) = self.get_client().await? {
            let request = tonic::Request::new(GpuRequest {
                memory_mb,
                gpu_count: count,
            });
            let response = client.request_gpu_allocation(request).await.map_err(|e| {
                synapse_types::SynapseError::BridgeError(format!(
                    "RequestGpuAllocation RPC failed: {e}"
                ))
            })?;
            let alloc = response.into_inner();
            if alloc.granted {
                Ok(alloc.device_ids)
            } else {
                Ok(Vec::new())
            }
        } else {
            warn!("No gRPC connection — returning empty GPU allocation (degraded mode)");
            Ok(Vec::new())
        }
    }

    /// Report training progress to SY.
    pub async fn report_progress(
        &self,
        job_id: &str,
        status: &str,
        step: u64,
        loss: f64,
    ) -> synapse_types::error::Result<()> {
        tracing::debug!(job_id, status, step, loss, "Reporting progress to SY");

        if let Some(mut client) = self.get_client().await? {
            let update = ProgressUpdate {
                job_id: job_id.to_string(),
                status: status.to_string(),
                loss,
                step,
            };
            let stream = tokio_stream::once(update);
            client
                .report_progress(tonic::Request::new(stream))
                .await
                .map_err(|e| {
                    synapse_types::SynapseError::BridgeError(format!(
                        "ReportProgress RPC failed: {e}"
                    ))
                })?;
        } else {
            tracing::debug!("No gRPC connection — progress report skipped (degraded mode)");
        }

        Ok(())
    }

    /// Request SY to scale out additional instances.
    pub async fn request_scale_out(
        &self,
        additional: u32,
        reason: &str,
    ) -> synapse_types::error::Result<ScaleResponse> {
        info!(additional, reason, "Requesting scale-out from SY");

        if let Some(mut client) = self.get_client().await? {
            let request = tonic::Request::new(ScaleRequest {
                additional_instances: additional,
                reason: reason.to_string(),
            });
            let response = client.request_scale_out(request).await.map_err(|e| {
                synapse_types::SynapseError::BridgeError(format!("RequestScaleOut RPC failed: {e}"))
            })?;
            Ok(response.into_inner())
        } else {
            warn!("No gRPC connection — scale-out request denied (degraded mode)");
            Ok(ScaleResponse {
                approved: false,
                instance_endpoints: Vec::new(),
            })
        }
    }

    /// Register a completed model with SY.
    pub async fn register_model(
        &self,
        name: &str,
        path: &str,
        base_model: &str,
        method: &str,
    ) -> synapse_types::error::Result<()> {
        info!(name, path, base_model, method, "Registering model with SY");

        if let Some(mut client) = self.get_client().await? {
            let request = tonic::Request::new(ModelRegistration {
                model_name: name.to_string(),
                model_path: path.to_string(),
                base_model: base_model.to_string(),
                training_method: method.to_string(),
            });
            client
                .register_completed_model(request)
                .await
                .map_err(|e| {
                    synapse_types::SynapseError::BridgeError(format!(
                        "RegisterCompletedModel RPC failed: {e}"
                    ))
                })?;
        } else {
            warn!("No gRPC connection — model registration skipped (degraded mode)");
        }

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
    async fn connect_enters_degraded_when_no_server() {
        let client = test_client();
        client.connect().await.unwrap();
        // No server running, so should enter degraded mode
        assert_eq!(client.connection_state().await, ConnectionState::Degraded);
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
    async fn announce_degraded_mode() {
        let client = test_client();
        // No connection established — should succeed in degraded mode
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
    async fn request_gpu_returns_empty_without_connection() {
        let client = test_client();
        let gpus = client.request_gpu(4096, 1).await.unwrap();
        assert!(gpus.is_empty());
    }

    #[tokio::test]
    async fn report_progress_succeeds_without_connection() {
        let client = test_client();
        let result = client.report_progress("job-1", "running", 100, 0.5).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn request_worker_assignment_succeeds() {
        let client = test_client();
        client
            .request_worker_assignment("job-1", 1, "http://node-2:9000", &[0, 1])
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn sync_checkpoint_succeeds() {
        let client = test_client();
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

        // Connect should reset it (even if it enters degraded mode)
        client.connect().await.unwrap();
        assert_eq!(*client.reconnect_count.read().await, 0);
    }

    #[tokio::test]
    async fn report_progress_works_without_connect() {
        let client = test_client();
        let result = client.report_progress("job-1", "running", 0, 0.0).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn request_gpu_returns_empty_in_degraded() {
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
        client.report_progress("job-1", "", 0, 0.0).await.unwrap();
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
    }

    #[tokio::test]
    async fn reconnect_succeeds_within_limit() {
        let config = ProtocolConfig {
            max_reconnect_attempts: 3,
            reconnect_delay: std::time::Duration::from_millis(1),
            ..ProtocolConfig::default()
        };
        let client = BridgeClient::new("http://127.0.0.1:9420".into(), config);
        // First reconnect should succeed (attempt 1 <= 3), though it enters degraded
        client.reconnect().await.unwrap();
        // reconnect_count should have been reset by connect
        assert_eq!(*client.reconnect_count.read().await, 0);
    }

    #[tokio::test]
    async fn request_scale_out_without_connection() {
        let client = test_client();
        let response = client.request_scale_out(2, "load spike").await.unwrap();
        assert!(!response.approved);
        assert!(response.instance_endpoints.is_empty());
    }

    #[tokio::test]
    async fn register_model_without_connection() {
        let client = test_client();
        client
            .register_model("my-model", "/models/my-model", "llama-3", "lora")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn grpc_client_is_none_initially() {
        let client = test_client();
        assert!(client.get_client().await.unwrap().is_none());
    }
}
