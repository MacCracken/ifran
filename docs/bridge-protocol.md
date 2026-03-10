# SY ↔ Synapse Bridge Protocol

Synapse and SecureYeoman maintain a cyclic relationship via gRPC bidirectional streaming.

## Protocol Overview

Two gRPC services define the bridge:

### SynapseBridge (SY → Synapse)
SY sends commands to Synapse:
- `SubmitTrainingJob` — delegate a training job
- `GetJobStatus` — stream job progress updates
- `PullModel` — pull a model with progress streaming
- `RunInference` / `StreamInference` — run inference

### YeomanBridge (Synapse → SY)
Synapse calls back to SY:
- `RequestGpuAllocation` — request GPU resources for a job
- `ReportProgress` — stream training/download progress
- `RequestScaleOut` — request additional Synapse instances
- `RegisterCompletedModel` — notify SY of a newly trained model

## Connection Lifecycle

1. Synapse reads `SY_GRPC_ENDPOINT` from config/env
2. Connects to SY and registers itself
3. Heartbeats flow every 10 seconds in both directions
4. If SY is unavailable, Synapse operates in degraded mode (no orchestration)
5. Reconnection is automatic with exponential backoff

## Discovery

SY instances can be discovered via:
- Environment variable (`SY_GRPC_ENDPOINT`)
- Config file (`synapse.toml` → `bridge.sy_endpoint`)
- mDNS (future)
