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
sy_endpoint = "http://your-sy-host:9420"
enabled = true
heartbeat_interval_secs = 10
```

Or via environment variable:

```bash
export SY_ENDPOINT="http://your-sy-host:9420"
```

### 2. Start Synapse

```bash
synapse serve
```

Synapse will connect to SY, announce its capabilities (GPU count, supported training methods), and begin sending heartbeats with status updates.

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

## Endpoint Discovery

The SY endpoint is resolved in order:
1. `bridge.sy_endpoint` in config
2. `SY_ENDPOINT` environment variable
3. `http://127.0.0.1:9420` (well-known local address)

## Degraded Mode

If SY is unavailable, Synapse operates independently:
- Inference and local training continue normally
- Orchestration features (scaling, GPU allocation) are disabled
- Automatic reconnection with exponential backoff (base 5s, max 10 attempts)
- After max attempts, enters degraded mode until manually reconnected

## Agnosticos Deployment

When installed via `pkg install synapse` on Agnosticos, Synapse automatically registers as a capability provider:

```bash
agnosticos capability register synapse \
    --type llm-inference \
    --endpoint http://127.0.0.1:8420 \
    --grpc-endpoint http://127.0.0.1:8421 \
    --capabilities 'model-pull,inference,training,openai-compat'
```

SY discovers Synapse instances through the Agnosticos agent-runtime capability registry.

## Proto Definitions

The shared contract lives in `proto/bridge.proto`. SY generates its TypeScript client from the same proto files.
