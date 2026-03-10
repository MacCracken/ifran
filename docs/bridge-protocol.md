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

## Proto Definitions

The shared contract lives in `proto/bridge.proto`. SY generates its TypeScript client from the same proto files.
