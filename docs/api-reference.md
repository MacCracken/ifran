# API Reference

Synapse exposes a REST API (default `:8420`) and a gRPC API (default `:8421`).

## Authentication

### Single-Tenant (Default)

Set `SYNAPSE_API_KEY` to require Bearer token authentication:

```bash
export SYNAPSE_API_KEY=your-secret-token
synapse serve
```

All endpoints except `/health` require the header:

```
Authorization: Bearer your-secret-token
```

When `SYNAPSE_API_KEY` is unset, the API is open (no auth required).

### Multi-Tenant

Enable in config with `multi_tenant = true` under `[security]`. Each tenant gets its own API key and isolated resources (models, jobs, evals, pipelines, etc.).

```bash
# Set admin key for tenant management
export SYNAPSE_ADMIN_KEY=your-admin-secret
synapse serve
```

Create tenants via the admin API:

```bash
curl -X POST http://localhost:8420/admin/tenants \
  -H "Authorization: Bearer $SYNAPSE_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp"}'
```

The response includes a `syn_*` API key (shown once). Tenants use this key as their Bearer token for all regular endpoints. Resources are automatically scoped to the tenant.

## REST Endpoints

### System
- `GET /health` — liveness probe, returns `"ok"` (always unauthenticated)
- `GET /system/status` — system info: version, loaded models, registered backends, hardware (CPU, GPUs)

### Models
- `GET /models` — list all models in the local catalog
- `GET /models/:id` — get a specific model by UUID or name
- `DELETE /models/:id` — remove a model from catalog and disk

### Inference
- `POST /inference` — run inference (blocking, returns full response)
- `POST /inference/stream` — run inference with SSE streaming

#### POST /inference request body
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Hello, how are you?",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "system_prompt": "You are a helpful assistant.",
  "stream": false
}
```

### OpenAI-Compatible
- `POST /v1/chat/completions` — OpenAI-compatible chat endpoint (streaming + non-streaming)
- `GET /v1/models` — list models in OpenAI format

#### POST /v1/chat/completions request body
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": true
}
```

Streaming responses use SSE with `data: {...}` lines followed by `data: [DONE]`.

### Training
- `POST /training/jobs` — create (and optionally start) a training job
- `GET /training/jobs` — list all training jobs
- `GET /training/jobs/:id` — get job status and progress
- `POST /training/jobs/:id/cancel` — cancel a running or queued job

#### POST /training/jobs request body
```json
{
  "base_model": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": {
    "path": "./my-dataset.jsonl",
    "format": "jsonl"
  },
  "method": "lora",
  "hyperparams": {
    "learning_rate": 2e-4,
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_seq_length": 2048
  },
  "lora": {
    "rank": 16,
    "alpha": 32.0,
    "dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
  },
  "auto_start": true
}
```

#### Training job response
```json
{
  "id": "uuid",
  "status": "running",
  "current_step": 150,
  "total_steps": 7500,
  "current_epoch": 0.6,
  "current_loss": 1.23,
  "progress_percent": 2.0,
  "error": null,
  "created_at": "2026-03-10T12:00:00Z",
  "started_at": "2026-03-10T12:00:01Z",
  "completed_at": null
}
```

### Evaluation
- `POST /eval/runs` — create an evaluation run
- `GET /eval/runs` — list all evaluation runs
- `GET /eval/runs/:id` — get evaluation run status and results

#### POST /eval/runs request body
```json
{
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "benchmarks": ["mmlu", "hellaswag", "perplexity", "humaneval", "custom"],
  "sample_limit": 100
}
```

Supported benchmarks: `mmlu` (multiple-choice), `hellaswag` (sentence completion), `humaneval` (code generation), `perplexity` (loss-based), `custom` (exact/contains match). Each benchmark requires a JSONL dataset with format-specific fields.

### Marketplace
- `GET /marketplace/search?q=<query>` — search published models (local + remote peers)
- `GET /marketplace/entries` — list all locally published models
- `POST /marketplace/publish` — publish a local model to the marketplace
- `DELETE /marketplace/entries/:name` — unpublish a model
- `GET /marketplace/download/:name` — download a published model file
- `POST /marketplace/pull` — pull a model from a remote peer

#### POST /marketplace/publish request body
```json
{
  "model_name": "my-fine-tuned-model"
}
```

#### POST /marketplace/pull request body
```json
{
  "model_name": "remote-model",
  "peer_url": "http://peer-node:8420",
  "sha256": "optional-checksum-for-verification"
}
```

### Distributed Training
- `POST /training/distributed/jobs` — create a distributed training job
- `GET /training/distributed/jobs` — list all distributed jobs
- `GET /training/distributed/jobs/:id` — get distributed job status
- `POST /training/distributed/jobs/:id/workers` — assign a worker
- `POST /training/distributed/jobs/:id/start` — start the job (all workers must be assigned)
- `POST /training/distributed/jobs/:id/workers/:rank/complete` — mark a worker as completed
- `POST /training/distributed/jobs/:id/fail` — fail the job
- `POST /training/distributed/jobs/:id/aggregate` — trigger checkpoint aggregation

#### POST /training/distributed/jobs request body
```json
{
  "base_model": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": {
    "path": "./my-dataset.jsonl",
    "format": "jsonl"
  },
  "method": "lora",
  "hyperparams": {
    "learning_rate": 2e-4,
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_seq_length": 2048
  },
  "world_size": 2,
  "strategy": "data_parallel"
}
```

#### POST /training/distributed/jobs/:id/workers request body
```json
{
  "rank": 0,
  "instance_id": "node-1",
  "endpoint": "http://node-1:9000",
  "device_ids": [0, 1]
}
```

#### POST /training/distributed/jobs/:id/aggregate request body
```json
{
  "output_dir": "/path/to/output",
  "method": "average"
}
```

Supported aggregation methods: `average`, `weighted_average`.

### Bridge (SY Integration)
- `GET /bridge/status` — bridge connection status (enabled, client/server state, SY endpoint)
- `POST /bridge/connect` — manually connect to SY and announce capabilities
- `POST /bridge/heartbeat` — send a one-off heartbeat to SY (debugging)

Bridge status is also included in `GET /system/status` under the `bridge` key.

When `bridge.enabled = true` in config, the server automatically:
1. Connects to SY on startup (with reconnect on failure)
2. Starts the bridge gRPC server for incoming SY commands
3. Spawns a background heartbeat task at the configured interval
4. Reports training job progress and completions to SY
5. Coordinates distributed training worker assignments via SY

### Tenant Admin (Multi-Tenant Only)

Only available when `multi_tenant = true`. Protected by `SYNAPSE_ADMIN_KEY` (separate from tenant API keys).

- `POST /admin/tenants` — create a new tenant, returns API key (shown once)
- `GET /admin/tenants` — list all tenants
- `DELETE /admin/tenants/:id` — disable a tenant (soft delete)

#### POST /admin/tenants request body
```json
{
  "name": "Acme Corp"
}
```

#### POST /admin/tenants response
```json
{
  "tenant": {
    "id": "uuid",
    "name": "Acme Corp",
    "enabled": true,
    "created_at": "2026-03-15T00:00:00Z"
  },
  "api_key": "syn_a1b2c3..."
}
```

## gRPC Services

See proto files in `proto/` for full message definitions:
- `synapse.proto` — core model and inference service
- `training.proto` — training job management
- `bridge.proto` — SY↔Synapse bridge protocol
