//! Bridge gRPC server — receives commands from SecureYeoman.
//!
//! Implements the `IfranBridge` gRPC service defined in bridge.proto.
//! SY connects to this server to submit training jobs, pull models, and
//! run inference on this Ifran instance.

use crate::protocol::{ConnectionState, Heartbeat, ProtocolConfig};
use ifran_backends::BackendRouter;
use ifran_core::lifecycle::manager::ModelManager;
use ifran_train::job::manager::JobManager;
use ifran_types::TenantId;
use ifran_types::bridge::ifran_bridge_server::{IfranBridge, IfranBridgeServer};
use ifran_types::bridge::{
    InferenceRequest, InferenceResponse, JobStatusRequest, JobStatusUpdate, PullModelRequest,
    PullProgress, StreamChunk, TrainingJobRequest, TrainingJobResponse,
};
use ifran_types::training::TrainingJobConfig;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_stream::Stream;
use tonic::{Request, Response, Status};
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
    #[inline]
    pub async fn connection_state(&self) -> ConnectionState {
        *self.state.read().await
    }

    /// Start the bridge server on the given address.
    ///
    /// This spawns a background task that listens for SY connections
    /// and serves the `IfranBridge` gRPC service.
    pub async fn start(
        &self,
        bind_addr: &str,
        service: IfranBridgeService,
    ) -> ifran_types::error::Result<()> {
        info!(addr = %bind_addr, "Starting bridge server");
        *self.state.write().await = ConnectionState::Connecting;

        let addr = bind_addr
            .parse()
            .map_err(|e| ifran_types::IfranError::BridgeError(format!("Invalid address: {e}")))?;

        let state = self.state.clone();

        tokio::spawn(async move {
            let result = tonic::transport::Server::builder()
                .add_service(IfranBridgeServer::new(service))
                .serve(addr)
                .await;

            if let Err(e) = result {
                warn!(error = %e, "Bridge gRPC server exited with error");
                *state.write().await = ConnectionState::Degraded;
            }
        });

        *self.state.write().await = ConnectionState::Connected;
        info!("Bridge server ready for SY connections");

        Ok(())
    }

    /// Build a heartbeat message with current state.
    #[must_use]
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
    #[must_use]
    #[inline]
    pub fn heartbeat_interval(&self) -> std::time::Duration {
        self.config.heartbeat_interval
    }

    /// Transition to degraded mode (SY connection lost but still serving).
    pub async fn enter_degraded(&self) {
        warn!("Bridge entering degraded mode — SY connection lost");
        *self.state.write().await = ConnectionState::Degraded;
    }
}

// ---------------------------------------------------------------------------
// IfranBridge gRPC service implementation
// ---------------------------------------------------------------------------

/// Implements the `IfranBridge` tonic service trait.
///
/// Holds shared references to the job manager, backend router, and model
/// manager so incoming gRPC calls can be dispatched to the right subsystem.
pub struct IfranBridgeService {
    job_manager: Arc<JobManager>,
    backend_router: Arc<BackendRouter>,
    model_manager: Arc<ModelManager>,
}

impl IfranBridgeService {
    pub fn new(
        job_manager: Arc<JobManager>,
        backend_router: Arc<BackendRouter>,
        model_manager: Arc<ModelManager>,
    ) -> Self {
        Self {
            job_manager,
            backend_router,
            model_manager,
        }
    }
}

type GrpcStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

#[tonic::async_trait]
impl IfranBridge for IfranBridgeService {
    /// Submit a new training job.
    ///
    /// Parses `config_json` from the request into a `TrainingJobConfig`,
    /// creates the job, starts it, and returns the job ID.
    async fn submit_training_job(
        &self,
        request: Request<TrainingJobRequest>,
    ) -> Result<Response<TrainingJobResponse>, Status> {
        let req = request.into_inner();
        info!(
            base_model = %req.base_model,
            method = %req.method,
            "Received SubmitTrainingJob"
        );

        // Parse the full config from JSON.
        let config: TrainingJobConfig = serde_json::from_str(&req.config_json)
            .map_err(|e| Status::invalid_argument(format!("Invalid config_json: {e}")))?;

        // Create and start the job (bridge operates as system-level).
        let job_id = self
            .job_manager
            .create_job(config, TenantId::default_tenant())
            .await
            .map_err(|e| Status::internal(format!("Failed to create job: {e}")))?;

        self.job_manager
            .start_job(job_id, &TenantId::default_tenant())
            .await
            .map_err(|e| Status::internal(format!("Failed to start job: {e}")))?;

        Ok(Response::new(TrainingJobResponse {
            job_id: job_id.to_string(),
        }))
    }

    type GetJobStatusStream = GrpcStream<JobStatusUpdate>;

    /// Stream job status updates until the job reaches a terminal state.
    ///
    /// Polls the job manager every 2 seconds and yields the current status.
    async fn get_job_status(
        &self,
        request: Request<JobStatusRequest>,
    ) -> Result<Response<Self::GetJobStatusStream>, Status> {
        let req = request.into_inner();
        let job_id: uuid::Uuid = req
            .job_id
            .parse()
            .map_err(|e| Status::invalid_argument(format!("Invalid job_id: {e}")))?;

        let job_manager = self.job_manager.clone();
        // Bridge operates as system-level
        let tenant = TenantId::default_tenant();

        let (tx, rx) = tokio::sync::mpsc::channel(16);

        tokio::spawn(async move {
            loop {
                let state = match job_manager.get_job(job_id, &tenant).await {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx
                            .send(Err(Status::not_found(format!("Job not found: {e}"))))
                            .await;
                        return;
                    }
                };

                let is_terminal = state.is_terminal();

                let update = JobStatusUpdate {
                    status: format!("{:?}", state.status),
                    step: state.current_step,
                    loss: state.current_loss.unwrap_or(0.0),
                    epoch: state.current_epoch,
                };

                if tx.send(Ok(update)).await.is_err() {
                    return; // receiver dropped
                }

                if is_terminal {
                    return;
                }

                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    type PullModelStream = GrpcStream<PullProgress>;

    /// Pull a model — streams progress back to the caller.
    ///
    /// Currently accepts the request and reports status transitions.
    /// Actual HuggingFace resolution and download will be wired in once
    /// the full pull pipeline (DB, HTTP client, config) is available.
    async fn pull_model(
        &self,
        request: Request<PullModelRequest>,
    ) -> Result<Response<Self::PullModelStream>, Status> {
        let req = request.into_inner();
        info!(model = %req.model_name, quant = %req.quant, "Received PullModel request");

        let (tx, rx) = tokio::sync::mpsc::channel(16);

        tokio::spawn(async move {
            // Report starting
            let _ = tx
                .send(Ok(PullProgress {
                    downloaded_bytes: 0,
                    total_bytes: 0,
                    state: "resolving".into(),
                }))
                .await;

            // TODO: Wire in actual HuggingFace resolution and download.
            // For now, report that pull was accepted and needs to be
            // completed through the REST API.
            let _ = tx
                .send(Ok(PullProgress {
                    downloaded_bytes: 0,
                    total_bytes: 0,
                    state: "accepted".into(),
                }))
                .await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    /// Run a single inference request against a loaded model.
    async fn run_inference(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>, Status> {
        let req = request.into_inner();
        if req.prompt.len() > 500_000 {
            return Err(Status::invalid_argument("Prompt too long (max 500KB)"));
        }
        info!(model = %req.model, "Received RunInference request");

        let loaded = self.model_manager.list_loaded(None).await;
        let loaded_model = loaded
            .iter()
            .find(|m| m.model_name == req.model)
            .or_else(|| loaded.first())
            .ok_or_else(|| Status::failed_precondition("No model loaded"))?;

        let backend = self
            .backend_router
            .get(&ifran_types::backend::BackendId(
                loaded_model.backend_id.clone(),
            ))
            .ok_or_else(|| Status::internal("Backend not available"))?;

        let handle = ifran_backends::ModelHandle(loaded_model.handle.clone());
        let inference_req = ifran_types::inference::InferenceRequest {
            prompt: req.prompt,
            max_tokens: Some(req.max_tokens),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };

        let resp = backend
            .infer(&handle, &inference_req)
            .await
            .map_err(|e| Status::internal(format!("Inference failed: {e}")))?;

        Ok(Response::new(InferenceResponse { text: resp.text }))
    }

    type StreamInferenceStream = GrpcStream<StreamChunk>;

    /// Stream inference tokens from a loaded model.
    async fn stream_inference(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<Self::StreamInferenceStream>, Status> {
        let req = request.into_inner();
        if req.prompt.len() > 500_000 {
            return Err(Status::invalid_argument("Prompt too long (max 500KB)"));
        }
        info!(model = %req.model, "Received StreamInference request");

        let loaded = self.model_manager.list_loaded(None).await;
        let loaded_model = loaded
            .iter()
            .find(|m| m.model_name == req.model)
            .or_else(|| loaded.first())
            .ok_or_else(|| Status::failed_precondition("No model loaded"))?;

        let backend = self
            .backend_router
            .get(&ifran_types::backend::BackendId(
                loaded_model.backend_id.clone(),
            ))
            .ok_or_else(|| Status::internal("Backend not available"))?;

        let handle = ifran_backends::ModelHandle(loaded_model.handle.clone());
        let inference_req = ifran_types::inference::InferenceRequest {
            prompt: req.prompt,
            max_tokens: Some(req.max_tokens),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };

        let mut rx = backend
            .infer_stream(&handle, inference_req)
            .await
            .map_err(|e| Status::internal(format!("Stream setup failed: {e}")))?;

        let (tx, grpc_rx) = tokio::sync::mpsc::channel(32);

        tokio::spawn(async move {
            while let Some(chunk) = rx.recv().await {
                let proto_chunk = StreamChunk {
                    text: chunk.text,
                    done: chunk.done,
                };
                if tx.send(Ok(proto_chunk)).await.is_err() {
                    break; // receiver dropped
                }
                if chunk.done {
                    break;
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(grpc_rx);
        Ok(Response::new(Box::pin(stream)))
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
    async fn enter_degraded_sets_state() {
        let server = test_server();
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

    #[test]
    fn custom_heartbeat_interval() {
        let config = ProtocolConfig {
            heartbeat_interval: std::time::Duration::from_secs(30),
            ..ProtocolConfig::default()
        };
        let server = BridgeServer::new("test".into(), config);
        assert_eq!(
            server.heartbeat_interval(),
            std::time::Duration::from_secs(30)
        );
    }

    #[test]
    fn build_heartbeat_has_positive_timestamp() {
        let server = test_server();
        let hb = server.build_heartbeat(0, 0, 0);
        assert!(hb.timestamp > 0);
    }

    #[test]
    fn build_heartbeat_zero_values() {
        let server = test_server();
        let hb = server.build_heartbeat(0, 0, 0);
        assert_eq!(hb.loaded_models, 0);
        assert_eq!(hb.gpu_memory_free_mb, 0);
        assert_eq!(hb.active_training_jobs, 0);
    }

    #[tokio::test]
    async fn enter_degraded_from_disconnected() {
        let server = test_server();
        assert_eq!(
            server.connection_state().await,
            ConnectionState::Disconnected
        );
        server.enter_degraded().await;
        assert_eq!(server.connection_state().await, ConnectionState::Degraded);
    }

    #[test]
    fn build_heartbeat_max_values() {
        let server = test_server();
        let hb = server.build_heartbeat(u32::MAX, u64::MAX, u32::MAX);
        assert_eq!(hb.loaded_models, u32::MAX);
        assert_eq!(hb.gpu_memory_free_mb, u64::MAX);
        assert_eq!(hb.active_training_jobs, u32::MAX);
    }

    #[test]
    fn build_heartbeat_instance_id_matches() {
        let server = BridgeServer::new("my-node-42".into(), ProtocolConfig::default());
        let hb = server.build_heartbeat(1, 2048, 0);
        assert_eq!(hb.instance_id, "my-node-42");
    }

    #[test]
    fn build_heartbeat_timestamp_monotonic() {
        let server = test_server();
        let hb1 = server.build_heartbeat(0, 0, 0);
        let hb2 = server.build_heartbeat(0, 0, 0);
        assert!(hb2.timestamp >= hb1.timestamp);
    }

    #[test]
    fn empty_instance_id() {
        let server = BridgeServer::new("".into(), ProtocolConfig::default());
        let hb = server.build_heartbeat(0, 0, 0);
        assert_eq!(hb.instance_id, "");
    }

    // -- IfranBridgeService tests --

    use ifran_train::executor::ExecutorKind;
    use ifran_types::training::*;

    fn test_config_json() -> String {
        let config = TrainingJobConfig {
            base_model: "test-model".into(),
            dataset: DatasetConfig {
                path: "/tmp/data.jsonl".into(),
                format: DatasetFormat::Jsonl,
                split: None,
                max_samples: Some(100),
            },
            method: TrainingMethod::Lora,
            hyperparams: HyperParams {
                learning_rate: 2e-4,
                epochs: 1,
                batch_size: 4,
                gradient_accumulation_steps: 1,
                warmup_steps: 0,
                weight_decay: 0.0,
                max_seq_length: 512,
            },
            output_name: None,
            lora: None,
            max_steps: None,
            time_budget_secs: None,
        };
        serde_json::to_string(&config).unwrap()
    }

    fn test_service() -> IfranBridgeService {
        let job_manager = Arc::new(JobManager::new(ExecutorKind::Subprocess, None, 4));
        let backend_router = Arc::new(BackendRouter::new());
        let model_manager = Arc::new(ModelManager::new(512));
        IfranBridgeService::new(job_manager, backend_router, model_manager)
    }

    #[tokio::test]
    async fn submit_training_job_success() {
        let svc = test_service();
        let req = Request::new(TrainingJobRequest {
            base_model: "test-model".into(),
            dataset_path: "/tmp/data.jsonl".into(),
            method: "lora".into(),
            config_json: test_config_json(),
        });

        let resp = svc.submit_training_job(req).await.unwrap();
        let job_id = resp.into_inner().job_id;

        // Should be a valid UUID
        assert!(job_id.parse::<uuid::Uuid>().is_ok());
    }

    #[tokio::test]
    async fn submit_training_job_invalid_config() {
        let svc = test_service();
        let req = Request::new(TrainingJobRequest {
            base_model: "test-model".into(),
            dataset_path: "/tmp/data.jsonl".into(),
            method: "lora".into(),
            config_json: "not valid json".into(),
        });

        let err = svc.submit_training_job(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn submit_training_job_invalid_hyperparams() {
        let svc = test_service();
        // learning_rate = 0 is invalid
        let config_json = r#"{
            "base_model": "test",
            "dataset": {"path": "/tmp/data.jsonl", "format": "jsonl"},
            "method": "lora",
            "hyperparams": {
                "learning_rate": 0.0,
                "epochs": 1,
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 0,
                "weight_decay": 0.0,
                "max_seq_length": 512
            }
        }"#;
        let req = Request::new(TrainingJobRequest {
            base_model: "test".into(),
            dataset_path: "/tmp/data.jsonl".into(),
            method: "lora".into(),
            config_json: config_json.into(),
        });

        let err = svc.submit_training_job(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::Internal);
    }

    #[tokio::test]
    async fn get_job_status_invalid_uuid() {
        let svc = test_service();
        let req = Request::new(JobStatusRequest {
            job_id: "not-a-uuid".into(),
        });

        match svc.get_job_status(req).await {
            Err(status) => assert_eq!(status.code(), tonic::Code::InvalidArgument),
            Ok(_) => panic!("Expected InvalidArgument error"),
        }
    }

    #[tokio::test]
    async fn get_job_status_not_found() {
        use tokio_stream::StreamExt;

        let svc = test_service();
        let job_id = uuid::Uuid::new_v4().to_string();
        let req = Request::new(JobStatusRequest { job_id });

        let resp = svc.get_job_status(req).await.unwrap();
        let mut stream = resp.into_inner();

        // First poll should yield a not-found error
        let result = stream.next().await.unwrap();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_job_status_streams_updates() {
        use tokio_stream::StreamExt;

        let svc = test_service();

        // Create a job first
        let req = Request::new(TrainingJobRequest {
            base_model: "test-model".into(),
            dataset_path: "/tmp/data.jsonl".into(),
            method: "lora".into(),
            config_json: test_config_json(),
        });
        let resp = svc.submit_training_job(req).await.unwrap();
        let job_id = resp.into_inner().job_id;

        // Stream status
        let req = Request::new(JobStatusRequest {
            job_id: job_id.clone(),
        });
        let resp = svc.get_job_status(req).await.unwrap();
        let mut stream = resp.into_inner();

        // Should get at least one update (Running status)
        let update = stream.next().await.unwrap().unwrap();
        assert!(!update.status.is_empty());
    }

    #[tokio::test]
    async fn pull_model_streams_progress() {
        use tokio_stream::StreamExt;

        let svc = test_service();
        let req = Request::new(PullModelRequest {
            model_name: "test-model".into(),
            quant: "q4_k_m".into(),
        });

        let resp = svc.pull_model(req).await.unwrap();
        let mut stream = resp.into_inner();

        // First message should be "resolving"
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.state, "resolving");
    }

    #[tokio::test]
    async fn run_inference_no_model_loaded() {
        let svc = test_service();
        let req = Request::new(InferenceRequest {
            model: "test".into(),
            prompt: "Hello".into(),
            max_tokens: 100,
        });

        let err = svc.run_inference(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[tokio::test]
    async fn stream_inference_no_model_loaded() {
        let svc = test_service();
        let req = Request::new(InferenceRequest {
            model: "test".into(),
            prompt: "Hello".into(),
            max_tokens: 100,
        });

        match svc.stream_inference(req).await {
            Err(status) => assert_eq!(status.code(), tonic::Code::FailedPrecondition),
            Ok(_) => panic!("Expected FailedPrecondition error"),
        }
    }

    #[test]
    fn service_construction() {
        let svc = test_service();
        // Just verify it constructs without panic
        assert!(Arc::strong_count(&svc.job_manager) >= 1);
    }
}
