# API Reference

Synapse exposes a REST API (default `:8420`) and a gRPC API (default `:8421`).

## REST Endpoints

### Models
- `GET /models` — list local models
- `POST /models/pull` — pull a model from a registry
- `POST /models/:id/load` — load a model into a backend
- `DELETE /models/:id` — remove a model

### Inference
- `POST /inference` — run inference (blocking)
- `POST /inference/stream` — run inference (SSE streaming)

### OpenAI-Compatible
- `POST /v1/chat/completions` — OpenAI-compatible chat endpoint
- `GET /v1/models` — list available models

### Training
- `POST /training/jobs` — create a training job
- `GET /training/jobs` — list training jobs
- `GET /training/jobs/:id` — get job status
- `DELETE /training/jobs/:id` — cancel a job

### System
- `GET /health` — health check
- `GET /system/status` — system status (GPUs, loaded models, memory)

## gRPC Services

See proto files in `proto/` for full message definitions:
- `synapse.proto` — core model and inference service
- `training.proto` — training job management
- `bridge.proto` — SY↔Synapse bridge protocol
