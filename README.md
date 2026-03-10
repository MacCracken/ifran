# Synapse

LLM controller for pulling, managing, and training language models. A self-contained product with CLI, REST/gRPC API, and desktop interfaces.

## Overview

Synapse is a Rust-based tool that provides:

- **Model pulling** from HuggingFace with resume, integrity verification, and catalog tracking
- **Inference** through 7 pluggable backends (llama.cpp, Candle, Ollama, vLLM, GGUF, ONNX, TensorRT)
- **Training** orchestration (LoRA, QLoRA, full fine-tune, DPO, RLHF, distillation)
- **OpenAI-compatible API** for drop-in replacement (`/v1/chat/completions`)
- **Desktop app** via Tauri v2 + SvelteKit
- **Bidirectional integration** with SecureYeoman orchestrator
- **Agnosticos integration** — runs as a systemd service with capability registration

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

# Start a training job
synapse train --base-model meta-llama/Llama-3.1-8B-Instruct --dataset ./data.jsonl --method lora
```

## Installation

### From Source

```bash
git clone git@github.com:MacCracken/synapse.git
cd synapse
cargo build --release
```

Binaries: `target/release/synapse` (CLI), `target/release/synapse-api` (server).

### On Agnosticos

```bash
pkg install synapse
systemctl start synapse
```

Installs to `/usr/local/bin/`, config at `/etc/synapse/synapse.toml`, data at `/var/lib/synapse/`.

## Configuration

Synapse discovers config in this order:

1. `SYNAPSE_CONFIG` environment variable
2. `~/.synapse/synapse.toml` (user config)
3. `/etc/synapse/synapse.toml` (system config, Agnosticos)
4. Built-in defaults

See [deploy/synapse.toml.example](deploy/synapse.toml.example) for all options.

Default storage: `~/.synapse/` (models, database, cache, checkpoints).

### Authentication

Set `SYNAPSE_API_KEY` to enable Bearer token authentication on all API endpoints (except `/health`):

```bash
export SYNAPSE_API_KEY=your-secret-token
synapse serve
```

Without `SYNAPSE_API_KEY`, the API is open (suitable for local development).

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
└── synapse-desktop    # Tauri v2 + SvelteKit (desktop app)
```

## Documentation

- [Architecture](docs/architecture.md)
- [Inference Backends](docs/backends.md)
- [Training](docs/training.md)
- [API Reference](docs/api-reference.md)
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

132 tests across all modules (~45% coverage). CI runs per-package test matrix with coverage via cargo-tarpaulin.

```bash
cargo test --workspace
```

## Versioning

CalVer: `YYYY.M.D` for releases, `YYYY.M.D-N` for patches.

## License

MIT
