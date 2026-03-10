# API Reference

Synapse exposes a REST API (default `:8420`) and a gRPC API (default `:8421`).

## REST Endpoints

### System
- `GET /health` — liveness probe, returns `"ok"`
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

### Training (planned)
- `POST /training/jobs` — create a training job
- `GET /training/jobs` — list training jobs
- `GET /training/jobs/:id` — get job status
- `DELETE /training/jobs/:id` — cancel a job

## gRPC Services

See proto files in `proto/` for full message definitions:
- `synapse.proto` — core model and inference service
- `training.proto` — training job management
- `bridge.proto` — SY↔Synapse bridge protocol
