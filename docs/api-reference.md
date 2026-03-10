# API Reference

Synapse exposes a REST API (default `:8420`) and a gRPC API (default `:8421`).

## Authentication

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

## gRPC Services

See proto files in `proto/` for full message definitions:
- `synapse.proto` — core model and inference service
- `training.proto` — training job management
- `bridge.proto` — SY↔Synapse bridge protocol
