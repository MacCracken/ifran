# Ifran

LLM controller for pulling, managing, and training language models. A self-contained product with CLI, REST/gRPC API, and desktop interfaces.

## Overview

Ifran is a Rust-based tool that provides:

- **Model pulling** from HuggingFace with resume, integrity verification, and catalog tracking
- **Inference** through 15 pluggable backends (llama.cpp, Candle, Ollama, vLLM, GGUF, ONNX, TensorRT, TPU, Gaudi, Inferentia, OneAPI, Qualcomm AI 100, Metal, Vulkan, AMD XDNA)
- **Hardware acceleration** with auto-detection for CUDA, ROCm, Metal, Vulkan, TPU, Gaudi, Inferentia, OneAPI, Qualcomm AI, and AMD XDNA
- **Training** orchestration (LoRA, QLoRA, full fine-tune, DPO, RLHF, distillation) with distributed training support
- **Evaluation** benchmarking (MMLU, HellaSwag, HumanEval, perplexity, custom)
- **Experiments** — autonomous hyperparameter sweeps with grid/random/Bayesian search
- **Fleet management** for multi-node deployments with health monitoring
- **Multi-tenancy** with per-tenant API keys, resource isolation, and GPU budget enforcement
- **Marketplace** for model publishing, discovery, and peer-to-peer sharing
- **RAG pipelines** for document ingestion and retrieval-augmented generation
- **RLHF annotation** sessions with DPO export
- **Lineage tracking** for full pipeline provenance (dataset → training → evaluation → deployment)
- **OpenAI-compatible API** for drop-in replacement (`/v1/chat/completions`)
- **Desktop app** via Tauri v2 + SvelteKit
- **Bidirectional integration** with SecureYeoman orchestrator
- **Agnosticos integration** — runs as a systemd service with capability registration

## Quick Start

```bash
# Pull a model
ifran pull meta-llama/Llama-3.1-8B-Instruct --quant q4_k_m

# Run interactive chat
ifran run meta-llama/Llama-3.1-8B-Instruct

# Start the API server
ifran serve

# List local models
ifran list

# Start a training job
ifran train --base-model meta-llama/Llama-3.1-8B-Instruct --dataset ./data.jsonl --method lora
```

## Installation

### From Source

```bash
git clone git@github.com:MacCracken/ifran.git
cd ifran
cargo build --release
```

Binaries: `target/release/ifran` (CLI), `target/release/ifran-api` (server).

### On Agnosticos

```bash
pkg install ifran
systemctl start ifran
```

Installs to `/usr/local/bin/`, config at `/etc/ifran/ifran.toml`, data at `/var/lib/ifran/`.

## Configuration

Ifran discovers config in this order:

1. `IFRAN_CONFIG` environment variable
2. `~/.ifran/ifran.toml` (user config)
3. `/etc/ifran/ifran.toml` (system config, Agnosticos)
4. Built-in defaults

See [deploy/ifran.toml.example](deploy/ifran.toml.example) for all options.

Default storage: `~/.ifran/` (models, database, cache, checkpoints).

### Authentication

Set `IFRAN_API_KEY` to enable Bearer token authentication on all API endpoints (except `/health`):

```bash
export IFRAN_API_KEY=your-secret-token
ifran serve
```

Without `IFRAN_API_KEY`, the API is open (suitable for local development).

## Project Structure

```
crates/
├── ifran-types      # Shared types + protobuf codegen
├── ifran-core       # Model registry, pull engine, lifecycle, hardware
├── ifran-backends   # Pluggable inference backends (trait-based)
├── ifran-train      # Training orchestration
├── ifran-api        # Axum REST + tonic gRPC server
├── ifran-bridge     # SY↔Ifran bidirectional gRPC
├── ifran-cli        # CLI (binary: ifran)
└── ifran-desktop    # Tauri v2 + SvelteKit (desktop app)
```

## Documentation

- [Architecture](docs/architecture.md)
- [Inference Backends](docs/backends.md)
- [Hardware Acceleration](docs/hardware-acceleration.md)
- [Training](docs/training.md)
- [Evaluation Guide](docs/evaluation-guide.md)
- [Fleet Management](docs/fleet-management.md)
- [Multi-Tenancy](docs/multi-tenancy.md)
- [API Reference](docs/api-reference.md)
- [CLI Reference](docs/cli-reference.md)
- [SY Bridge Protocol](docs/bridge-protocol.md)
- [Getting Started](docs/guides/getting-started.md)
- [Training Guide](docs/guides/training-guide.md)
- [Development Guide](docs/development/README.md)
- [Roadmap](docs/development/roadmap.md)
- [ADRs](docs/adr/)

## Related Projects

- **[Agnosticos](https://github.com/MacCracken/agnosticos)** — Target operating system (Rust)
- **[SecureYeoman](https://github.com/MacCracken/secureyeoman)** — Orchestrator with cyclic integration (TS/Bun)

## Testing

1,406 tests across 7 crates (~73% coverage). CI runs per-package test matrix with coverage via cargo-tarpaulin.

```bash
cargo test --workspace
```

## Versioning

CalVer: `YYYY.M.D` for releases, `YYYY.M.D-N` for patches.

## License

AGPL-3.0
