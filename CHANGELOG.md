# Changelog

All notable changes to Synapse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows CalVer: YYYY.M.D / YYYY.M.D-N for patches.

## [2026.3.10] — (in progress)

### Added
- `synapse-api/rest/router`: Full Axum router with model, inference, system, and OpenAI-compatible routes + CORS + telemetry
- `synapse-api/rest/models`: Model CRUD endpoints — `GET /models`, `GET /models/:id`, `DELETE /models/:id`
- `synapse-api/rest/inference`: Inference endpoints — `POST /inference` (full) and `POST /inference/stream` (SSE)
- `synapse-api/rest/openai_compat`: OpenAI drop-in replacement — `POST /v1/chat/completions` (streaming + non-streaming) and `GET /v1/models`
- `synapse-api/rest/system`: System endpoints — `GET /health` and `GET /system/status` (hardware, loaded models, backends)
- `synapse-api/middleware/telemetry`: Request tracing middleware via `tower-http`
- `synapse-api/state`: Shared application state with config, DB, backend router, and model manager
- `synapse-cli serve`: Now uses the full API router instead of minimal stub
- `synapse-backends/llamacpp`: llama.cpp backend via `llama-server` subprocess — load/unload models, inference, streaming via OpenAI-compatible API
- `synapse-core/lifecycle/manager`: Model load/unload orchestration with backend-agnostic handle tracking
- `synapse-core/lifecycle/memory`: VRAM/RAM budget estimation and pre-load memory checks with GPU/CPU fallback
- `synapse-cli run`: Interactive inference session with streaming output
- `synapse-cli serve`: Minimal API server with `/health` and `/v1/models` endpoints
- `synapse-core/registry/huggingface`: HuggingFace Hub API client — model info, GGUF file resolution by quant level, search
- `synapse-core/pull/downloader`: Chunked HTTP downloader with resume support via `.part` files and Range headers
- `synapse-core/pull/verifier`: SHA-256 and BLAKE3 integrity verification with auto-detection
- `synapse-core/pull/progress`: Broadcast-channel progress tracking for multi-consumer updates (CLI, API, desktop)
- `synapse-core/registry/scanner`: Local filesystem scanner for model files (GGUF, SafeTensors, ONNX, PyTorch, TensorRT)
- `synapse-cli pull`: Full model pull command with HuggingFace resolution, progress bar, integrity check, and catalog registration
- `synapse-cli list`: Table-formatted model listing from local catalog
- `synapse-cli rm`: Model removal with confirmation prompt, disk cleanup, and catalog deletion
- `synapse-core/storage/db`: SQLite model catalog with full CRUD operations (insert, get, get_by_name, list, update, delete, count) and schema migrations
- `synapse-core/storage/layout`: Filesystem layout manager for `~/.synapse/` directory structure with model slug generation, directory creation, and cleanup
- `synapse-core/hardware/detect`: Hardware detection — NVIDIA GPUs via nvidia-smi, AMD ROCm GPUs via sysfs, CPU info from /proc; unified `SystemHardware` snapshot for backend selection
- CalVer versioning via workspace — all crates inherit from `VERSION` file
- 36 unit tests across storage, pull, registry, hardware, and lifecycle modules
- Initial project scaffold with 8-crate Cargo workspace
- Protobuf definitions for core, bridge, and training services
- Pluggable inference backend trait system with 7 backend stubs (llama.cpp, Candle, Ollama, vLLM, GGUF, ONNX, TensorRT)
- Training orchestration framework with Docker/subprocess/native executors
- CLI structure with pull, list, run, serve, train, status, remove commands
- REST API (Axum) + gRPC (tonic) server structure
- SY<->Synapse bidirectional gRPC bridge
- Documentation: architecture, roadmap, 6 ADRs, 3 guides, API reference
- CI/CD pipeline (GitHub Actions): build, quality, security, test, benchmarks, docs, container, license
- Release pipeline with SBOM generation and multi-arch container publishing
- Dependency update automation (weekly cargo update PRs)
- Docker support: server, dev, trainer, release containers
- Agnosticos integration: systemd service, package spec
- Governance: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, SUPPORT
- Configuration: .audit.toml, osv-scanner.toml, .editorconfig, .gitattributes
