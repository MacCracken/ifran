# Synapse Development Roadmap

> Versioning: CalVer `YYYY.M.D` / `YYYY.M.D-N` for patches

## MVP — Target: v2026.4.x

The minimum viable product delivers a working CLI and API server that can pull models, run inference through llama.cpp, and expose an OpenAI-compatible endpoint.

### Phase 1: Foundation ✓
- [x] Project scaffold (workspace, crate layout, proto definitions)
- [x] `synapse-types`: finalize core data structures
- [x] `synapse-core/config`: TOML config loading with defaults
- [x] `synapse-core/storage/db`: SQLite model catalog (create, read, update, delete)
- [x] `synapse-core/storage/layout`: filesystem layout for `~/.synapse/models/`
- [x] `synapse-core/hardware/detect`: GPU detection (CUDA via NVML, ROCm, CPU fallback)

### Phase 2: Model Pulling ✓
- [x] `synapse-core/registry/huggingface`: HuggingFace Hub API — search, resolve GGUF files
- [x] `synapse-core/pull/downloader`: chunked parallel HTTP download with resume
- [x] `synapse-core/pull/verifier`: SHA-256 integrity check
- [x] `synapse-core/pull/progress`: broadcast channel for progress events
- [x] `synapse-core/registry/scanner`: scan local filesystem for existing models
- [x] `synapse-cli pull` command
- [x] `synapse-cli list` command
- [x] `synapse-cli rm` command

### Phase 3: Inference — llama.cpp Backend ✓
- [x] `synapse-backends/traits`: finalize `InferenceBackend` trait
- [x] `synapse-backends/llamacpp`: llama.cpp via `llama-server` subprocess (OpenAI-compatible API)
- [x] `synapse-core/lifecycle/manager`: model load/unload orchestration
- [x] `synapse-core/lifecycle/memory`: VRAM budget checks before loading
- [x] `synapse-cli run` command (interactive prompt with streaming)
- [x] `synapse-cli serve` command (start API server)

### Phase 4: API Server ✓
- [x] `synapse-api/rest/router`: Axum router setup with all route groups + CORS + telemetry
- [x] `synapse-api/rest/models`: `GET /models`, `GET /models/:id`, `DELETE /models/:id`
- [x] `synapse-api/rest/inference`: `POST /inference` + `POST /inference/stream` (SSE)
- [x] `synapse-api/rest/openai_compat`: `POST /v1/chat/completions` + `GET /v1/models` (OpenAI-compatible)
- [x] `synapse-api/rest/system`: `GET /health`, `GET /system/status`
- [x] `synapse-api/middleware/telemetry`: request tracing with `tower-http`

**MVP milestone: a user can `synapse pull`, `synapse run`, and hit a local OpenAI-compatible API.** ✓

---

## v1 — Target: v2026.6.x

Full-featured release with training, all backends, SY bridge, and desktop app.

### Phase 5: Additional Backends
- [ ] `synapse-backends/candle`: HuggingFace Candle (pure Rust) for SafeTensors
- [ ] `synapse-backends/ollama`: HTTP client to Ollama API
- [ ] `synapse-backends/gguf`: direct GGUF loading (via candle-gguf)
- [ ] `synapse-backends/onnx`: ONNX Runtime via `ort` crate
- [ ] `synapse-backends/vllm`: HTTP client to vLLM server
- [ ] `synapse-backends/tensorrt`: TensorRT-LLM integration
- [ ] `synapse-backends/router`: smart backend auto-selection based on model format + hardware

### Phase 6: Training
- [ ] `synapse-train/executor/docker`: launch training containers
- [ ] `synapse-train/executor/subprocess`: spawn Python training scripts
- [ ] `synapse-train/job/manager`: job lifecycle (create, start, stop, resume)
- [ ] `synapse-train/job/scheduler`: priority queue with GPU-aware scheduling
- [ ] `synapse-train/dataset/loader`: JSONL, Parquet, HuggingFace datasets
- [ ] `synapse-train/dataset/validator`: schema validation, quality checks
- [ ] `synapse-train/methods/lora`: LoRA/QLoRA orchestration
- [ ] `synapse-train/methods/full`: full parameter fine-tuning
- [ ] `synapse-train/methods/dpo`: Direct Preference Optimization
- [ ] `synapse-train/checkpoint/store`: checkpoint save/load/prune
- [ ] `synapse-train/checkpoint/merger`: merge LoRA adapters into base model
- [ ] `synapse-cli train` command
- [ ] Training REST/gRPC endpoints in `synapse-api`
- [ ] Python training scripts (SFT, DPO, RLHF) bundled in Docker image

### Phase 7: SecureYeoman Bridge
- [ ] `synapse-bridge/server`: gRPC server receiving SY commands
- [ ] `synapse-bridge/client`: gRPC client calling back to SY
- [ ] `synapse-bridge/protocol`: heartbeat, reconnection, degraded mode
- [ ] `synapse-bridge/discovery`: SY instance discovery (env, config, mDNS)
- [ ] SY-side integration: Synapse as a managed service in SecureYeoman orchestrator
- [ ] Bidirectional job delegation (SY delegates training, Synapse requests GPU scaling)

### Phase 8: Desktop Application
- [ ] Tauri v2 scaffold with Svelte frontend
- [ ] Model management UI (browse, pull, delete with progress)
- [ ] Chat interface with streaming output
- [ ] Training dashboard with live loss curves and GPU metrics
- [ ] System monitor (GPU utilization, VRAM, loaded models)
- [ ] Settings page (backends, storage paths, bridge config)

### Phase 9: Agnosticos Integration
- [ ] systemd service unit (`synapse.service`)
- [ ] Agnosticos package spec (`synapse.pkg.toml`)
- [ ] Agent-runtime registration as capability provider
- [ ] `/etc/synapse/synapse.toml` system-level config
- [ ] Model storage at `/var/lib/synapse/models/`

### Phase 10: Polish & Release
- [ ] Integration test suite
- [ ] API documentation
- [ ] Docker multi-arch images (amd64 + arm64)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] CHANGELOG for v1 release
- [ ] Security audit (API auth, model integrity, bridge mTLS)

**v1 milestone: full product — pull, infer, train, orchestrate, desktop + web + CLI.**

---

## Post-v1 Considerations
- Model marketplace / shared registry between Synapse instances
- Distributed training across multiple Synapse nodes (via SY orchestration)
- Model evaluation benchmarks (automated quality scoring)
- Prompt management and versioning
- RAG pipeline integration
- WebAssembly builds for browser-based inference
- RLHF annotation UI in desktop app
- Multi-tenant support for shared deployments
