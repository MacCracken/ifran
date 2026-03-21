//! gRPC service implementation for high-performance inference and cluster communication.
//!
//! Implements the `IfranService` trait generated from `proto/ifran.proto`.
//! Mirrors the REST API logic but over gRPC for lower-latency programmatic access.

use crate::state::AppState;
use std::pin::Pin;
use ifran_types::ifran_proto::ifran_service_server::IfranService;
use ifran_types::ifran_proto::{
    GpuInfo, InferenceRequest, InferenceResponse, ListModelsRequest, ListModelsResponse,
    LoadModelRequest, LoadModelResponse, ModelInfo as ProtoModelInfo, PullModelRequest,
    PullProgress, StatusRequest, StatusResponse, StreamChunk, UnloadModelRequest,
    UnloadModelResponse,
};
use tokio_stream::Stream;
use tonic::{Request, Response, Status};
use tracing::info;

/// gRPC service backed by shared [`AppState`].
pub struct IfranGrpcService {
    state: AppState,
}

impl IfranGrpcService {
    /// Create a new service instance wrapping the shared application state.
    pub fn new(state: AppState) -> Self {
        Self { state }
    }
}

type GrpcStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

#[tonic::async_trait]
impl IfranService for IfranGrpcService {
    // -----------------------------------------------------------------------
    // GetStatus
    // -----------------------------------------------------------------------

    async fn get_status(
        &self,
        _request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let loaded = self.state.model_manager.list_loaded(None).await;
        let loaded_names: Vec<String> = loaded.iter().map(|m| m.model_name.clone()).collect();

        let gpus = ifran_core::hardware::detect::detect()
            .map(|hw| {
                hw.gpus
                    .iter()
                    .map(|g| GpuInfo {
                        id: g.index as u32,
                        name: g.name.clone(),
                        memory_total_mb: g.memory_total_mb,
                        memory_used_mb: g.memory_total_mb.saturating_sub(g.memory_free_mb),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(Response::new(StatusResponse {
            loaded_models: loaded_names,
            gpus,
        }))
    }

    // -----------------------------------------------------------------------
    // ListModels
    // -----------------------------------------------------------------------

    async fn list_models(
        &self,
        _request: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        let db = self.state.db.lock().await;
        let models = db
            .list(&ifran_types::TenantId::default_tenant())
            .map_err(|e| Status::internal(format!("Database error: {e}")))?;

        let proto_models = models
            .into_iter()
            .map(|m| ProtoModelInfo {
                id: m.id.to_string(),
                name: m.name,
                format: format!("{:?}", m.format).to_lowercase(),
                quant: serde_json::to_value(m.quant)
                    .ok()
                    .and_then(|v| v.as_str().map(String::from))
                    .unwrap_or_default(),
                size_bytes: m.size_bytes,
                local_path: m.local_path,
            })
            .collect();

        Ok(Response::new(ListModelsResponse {
            models: proto_models,
        }))
    }

    // -----------------------------------------------------------------------
    // Infer
    // -----------------------------------------------------------------------

    async fn infer(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>, Status> {
        let req = request.into_inner();
        info!(handle = %req.handle, "gRPC Infer request");

        let loaded = self.state.model_manager.list_loaded(None).await;
        let loaded_model = loaded
            .iter()
            .find(|m| m.handle == req.handle || m.model_name == req.handle)
            .or_else(|| loaded.first())
            .ok_or_else(|| Status::failed_precondition("No model loaded"))?;

        let backend = self
            .state
            .backends
            .get(&ifran_types::backend::BackendId(
                loaded_model.backend_id.clone(),
            ))
            .ok_or_else(|| Status::internal("Backend not available"))?;

        let handle = ifran_backends::ModelHandle(loaded_model.handle.clone());
        let inference_req = ifran_types::inference::InferenceRequest {
            prompt: req.prompt,
            max_tokens: if req.max_tokens > 0 {
                Some(req.max_tokens)
            } else {
                None
            },
            temperature: if req.temperature > 0.0 {
                Some(req.temperature)
            } else {
                None
            },
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

        Ok(Response::new(InferenceResponse {
            text: resp.text,
            prompt_tokens: resp.usage.prompt_tokens,
            completion_tokens: resp.usage.completion_tokens,
        }))
    }

    // -----------------------------------------------------------------------
    // InferStream
    // -----------------------------------------------------------------------

    type InferStreamStream = GrpcStream<StreamChunk>;

    async fn infer_stream(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<Self::InferStreamStream>, Status> {
        let req = request.into_inner();
        info!(handle = %req.handle, "gRPC InferStream request");

        let loaded = self.state.model_manager.list_loaded(None).await;
        let loaded_model = loaded
            .iter()
            .find(|m| m.handle == req.handle || m.model_name == req.handle)
            .or_else(|| loaded.first())
            .ok_or_else(|| Status::failed_precondition("No model loaded"))?;

        let backend = self
            .state
            .backends
            .get(&ifran_types::backend::BackendId(
                loaded_model.backend_id.clone(),
            ))
            .ok_or_else(|| Status::internal("Backend not available"))?;

        let handle = ifran_backends::ModelHandle(loaded_model.handle.clone());
        let inference_req = ifran_types::inference::InferenceRequest {
            prompt: req.prompt,
            max_tokens: if req.max_tokens > 0 {
                Some(req.max_tokens)
            } else {
                None
            },
            temperature: if req.temperature > 0.0 {
                Some(req.temperature)
            } else {
                None
            },
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
                    break;
                }
                if chunk.done {
                    break;
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(grpc_rx);
        Ok(Response::new(Box::pin(stream)))
    }

    // -----------------------------------------------------------------------
    // Stubs — PullModel, LoadModel, UnloadModel
    // -----------------------------------------------------------------------

    type PullModelStream = GrpcStream<PullProgress>;

    async fn pull_model(
        &self,
        _request: Request<PullModelRequest>,
    ) -> Result<Response<Self::PullModelStream>, Status> {
        Err(Status::unimplemented(
            "PullModel is not yet implemented via gRPC — use the REST API",
        ))
    }

    async fn load_model(
        &self,
        _request: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        Err(Status::unimplemented(
            "LoadModel is not yet implemented via gRPC — use the REST API",
        ))
    }

    async fn unload_model(
        &self,
        _request: Request<UnloadModelRequest>,
    ) -> Result<Response<UnloadModelResponse>, Status> {
        Err(Status::unimplemented(
            "UnloadModel is not yet implemented via gRPC — use the REST API",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ifran_core::config::*;

    fn test_state(tmp: &tempfile::TempDir) -> AppState {
        let config = IfranConfig {
            server: ServerConfig {
                bind: "127.0.0.1:0".into(),
                grpc_bind: "127.0.0.1:0".into(),
            },
            storage: StorageConfig {
                models_dir: tmp.path().join("models"),
                database: tmp.path().join("test.db"),
                cache_dir: tmp.path().join("cache"),
            },
            backends: BackendsConfig {
                default: "llamacpp".into(),
                enabled: vec!["llamacpp".into()],
            },
            training: TrainingConfig {
                executor: "subprocess".into(),
                trainer_image: None,
                max_concurrent_jobs: 2,
                checkpoints_dir: tmp.path().join("checkpoints"),
                job_eviction_ttl_secs: 86400,
            },
            bridge: BridgeConfig {
                sy_endpoint: None,
                enabled: false,
                heartbeat_interval_secs: 10,
            },
            hardware: HardwareConfig {
                gpu_memory_reserve_mb: 512,
                telemetry_interval_secs: 0,
            },
            security: SecurityConfig::default(),
            budget: BudgetConfig::default(),
            fleet: FleetConfig::default(),
        };
        AppState::new(config).unwrap()
    }

    #[tokio::test]
    async fn get_status_returns_empty_when_no_models() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let svc = IfranGrpcService::new(state);

        let resp = svc
            .get_status(Request::new(StatusRequest {}))
            .await
            .unwrap();
        let status = resp.into_inner();
        assert!(status.loaded_models.is_empty());
    }

    #[tokio::test]
    async fn list_models_empty_db() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let svc = IfranGrpcService::new(state);

        let resp = svc
            .list_models(Request::new(ListModelsRequest {}))
            .await
            .unwrap();
        assert!(resp.into_inner().models.is_empty());
    }

    #[tokio::test]
    async fn infer_fails_with_no_model() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let svc = IfranGrpcService::new(state);

        let err = svc
            .infer(Request::new(InferenceRequest {
                handle: "nonexistent".into(),
                prompt: "Hello".into(),
                max_tokens: 100,
                temperature: 0.0,
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[tokio::test]
    async fn infer_stream_fails_with_no_model() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let svc = IfranGrpcService::new(state);

        match svc
            .infer_stream(Request::new(InferenceRequest {
                handle: "nonexistent".into(),
                prompt: "Hello".into(),
                max_tokens: 100,
                temperature: 0.0,
            }))
            .await
        {
            Err(status) => assert_eq!(status.code(), tonic::Code::FailedPrecondition),
            Ok(_) => panic!("Expected FailedPrecondition error"),
        }
    }

    #[tokio::test]
    async fn pull_model_returns_unimplemented() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let svc = IfranGrpcService::new(state);

        match svc
            .pull_model(Request::new(PullModelRequest {
                model_name: "test".into(),
                quant: "q4_k_m".into(),
                source: String::new(),
            }))
            .await
        {
            Err(status) => assert_eq!(status.code(), tonic::Code::Unimplemented),
            Ok(_) => panic!("Expected Unimplemented error"),
        }
    }

    #[tokio::test]
    async fn load_model_returns_unimplemented() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let svc = IfranGrpcService::new(state);

        let err = svc
            .load_model(Request::new(LoadModelRequest {
                model_id: "test".into(),
                backend: "llamacpp".into(),
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::Unimplemented);
    }

    #[tokio::test]
    async fn unload_model_returns_unimplemented() {
        let tmp = tempfile::TempDir::new().unwrap();
        let state = test_state(&tmp);
        let svc = IfranGrpcService::new(state);

        let err = svc
            .unload_model(Request::new(UnloadModelRequest {
                handle: "test".into(),
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::Unimplemented);
    }
}
