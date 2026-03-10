# Synapse

LLM controller for pulling, managing, and training language models. A self-contained product with CLI, REST/gRPC API, and desktop interfaces.

## Overview

Synapse is a Rust-based tool that provides:

- **Model pulling** from HuggingFace, OCI registries, and direct URLs
- **Inference** through pluggable backends (llama.cpp, Candle, Ollama, vLLM, ONNX, TensorRT)
- **Training** orchestration (LoRA, QLoRA, full fine-tune, DPO, RLHF, distillation)
- **OpenAI-compatible API** for drop-in replacement
- **Desktop app** via Tauri + Svelte
- **Bidirectional integration** with SecureYeoman orchestrator

## Quick Start

```bash
# Pull a model
synapse pull meta-llama/Llama-3.1-8B-Instruct --quant q4_k_m

# Run interactive chat
synapse run meta-llama/Llama-3.1-8B-Instruct

# Start the API server
synapse serve

# List local models
synapse list
```

## Building

```bash
# Prerequisites: Rust stable, protoc
./scripts/setup-dev.sh

# Build
make build

# Run tests
make test

# Dev server with auto-reload
make dev
```

## Project Structure

```
crates/
├── synapse-types      # Shared types + protobuf codegen
├── synapse-core       # Model registry, pull engine, lifecycle, hardware
├── synapse-backends   # Pluggable inference backends (trait-based)
├── synapse-train      # Training orchestration
├── synapse-api        # Axum REST + tonic gRPC server
├── synapse-bridge     # SY↔Synapse bidirectional gRPC
├── synapse-cli        # CLI (binary: synapse)
└── synapse-desktop    # Tauri v2 + Svelte (desktop app)
```

## Configuration

Synapse reads from `synapse.toml`. See [deploy/synapse.toml.example](deploy/synapse.toml.example) for all options.

Default storage: `~/.synapse/` (models, database, cache, checkpoints).

## Documentation

- [Architecture](docs/architecture.md)
- [Inference Backends](docs/backends.md)
- [Training](docs/training.md)
- [API Reference](docs/api-reference.md)
- [SY Bridge Protocol](docs/bridge-protocol.md)
- [Development Guide](docs/development/README.md)
- [Roadmap](docs/development/roadmap.md)
- [ADRs](docs/adr/)

## Related Projects

- **[Agnosticos](https://github.com/MacCracken/agnosticos)** — Target operating system (Rust)
- **[SecureYeoman](https://github.com/MacCracken/secureyeoman)** — Orchestrator with cyclic integration (TS/Bun)

## Versioning

CalVer: `YYYY.M.D` for releases, `YYYY.M.D-N` for patches.

## License

MIT
