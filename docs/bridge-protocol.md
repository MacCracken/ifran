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

1. Synapse reads the SY endpoint from config or `SY_ENDPOINT` env var
2. Connects to SY and announces capabilities (GPU count, supported methods)
3. Heartbeats flow every 10 seconds with instance status (loaded models, free VRAM, active jobs)
4. If SY is unavailable, Synapse operates in degraded mode (no orchestration)
5. Reconnection is automatic with exponential backoff (max 10 attempts)

## Protocol Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `heartbeat_interval` | 10s | Time between heartbeats |
| `heartbeat_timeout` | 30s | Max time to wait for heartbeat response |
| `reconnect_delay` | 5s | Base delay between reconnect attempts |
| `max_reconnect_attempts` | 10 | Max reconnects before entering degraded mode |

## Discovery

SY endpoint is resolved in order:
1. `bridge.sy_endpoint` in `synapse.toml`
2. `SY_ENDPOINT` environment variable
3. `http://127.0.0.1:9420` (well-known local address)

## REST API

Bridge status and management is exposed via REST endpoints for monitoring and manual control:

- `GET /bridge/status` — connection state, endpoint, heartbeat config
- `POST /bridge/connect` — manually trigger SY connection + capability announcement
- `POST /bridge/heartbeat` — send a one-off heartbeat (debugging)

Bridge status is also included in the `/system/status` response.

## Auto-Initialization

When `bridge.enabled = true` in config, synapse-server automatically:

1. Discovers the SY endpoint (config → env → well-known)
2. Connects the bridge client to SY
3. Starts the bridge gRPC server on the configured `grpc_bind` address
4. Spawns a background heartbeat task
5. Reports training job lifecycle events (start, progress, cancel, completion) to SY
6. Coordinates distributed training worker assignments and checkpoint sync via SY

## Training Integration

The bridge is wired into the training and distributed training handlers:

- **Job start**: Reports `running` status to SY via `ReportProgress`
- **Job cancel**: Reports `cancelled` status to SY
- **Worker assignment**: Calls `RequestWorkerAssignment` on SY for cross-node coordination
- **Worker completion**: Calls `SyncCheckpoint` for checkpoint transfer, reports `completed` when all workers finish

## Proto Definitions

The shared contract lives in `proto/bridge.proto`. SY generates its TypeScript client from the same proto files.
