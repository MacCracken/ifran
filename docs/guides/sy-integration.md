# SecureYeoman Integration Guide

## Overview

Synapse and SecureYeoman (SY) have a cyclic relationship:
- SY orchestrates Synapse instances (scheduling, scaling, resource allocation)
- Synapse calls back to SY (progress reporting, GPU requests, model registration)

## Setup

### 1. Configure the bridge

In `synapse.toml`:

```toml
[bridge]
sy_endpoint = "http://your-sy-host:9000"
enabled = true
heartbeat_interval_secs = 10
```

Or via environment variable:

```bash
export SY_GRPC_ENDPOINT="http://your-sy-host:9000"
```

### 2. Start Synapse

```bash
synapse serve
```

Synapse will connect to SY and register itself. Heartbeats maintain the connection.

## Communication Flow

### SY → Synapse (SynapseBridge service)

| RPC | Description |
|-----|-------------|
| `SubmitTrainingJob` | Delegate a training job |
| `GetJobStatus` | Stream job progress updates |
| `PullModel` | Pull a model with progress |
| `RunInference` | Single inference request |
| `StreamInference` | Streaming inference |

### Synapse → SY (YeomanBridge service)

| RPC | Description |
|-----|-------------|
| `RequestGpuAllocation` | Request GPU resources |
| `ReportProgress` | Stream training/download progress |
| `RequestScaleOut` | Request additional instances |
| `RegisterCompletedModel` | Notify of newly trained model |

## Degraded Mode

If SY is unavailable, Synapse operates independently:
- Inference and local training continue normally
- Orchestration features (scaling, GPU allocation) are disabled
- Automatic reconnection with exponential backoff

## Proto Definitions

The shared contract lives in `proto/bridge.proto`. SY generates its TypeScript client from the same proto files.
