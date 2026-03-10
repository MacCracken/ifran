# Changelog

All notable changes to Synapse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows CalVer: YYYY.M.D / YYYY.M.D-N for patches.

## [2026.3.10] — (in progress)

### Added
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
- 28 unit tests across storage, pull, registry, and hardware modules
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
