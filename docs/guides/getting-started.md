# Getting Started with Synapse

## Prerequisites

- **Rust** stable toolchain (install via [rustup](https://rustup.rs))
- **protoc** (protobuf compiler)
- **Docker** (optional, for training executor)
- **CUDA toolkit** (optional, for GPU inference)

## Installation

### From Source

```bash
git clone git@github.com:MacCracken/synapse.git
cd synapse
./scripts/setup-dev.sh
make release
```

The binaries will be at:
- `target/release/synapse` — CLI
- `target/release/synapse-api` — API server

### On Agnosticos

```bash
pkg install synapse
```

## First Steps

### 1. Pull a model

```bash
synapse pull meta-llama/Llama-3.1-8B-Instruct --quant q4_k_m
```

This downloads from HuggingFace Hub, verifies integrity, and registers in the local catalog.

### 2. Chat with it

```bash
synapse run meta-llama/Llama-3.1-8B-Instruct
```

### 3. Start the API server

```bash
synapse serve
```

The server exposes:
- REST API at `http://localhost:8420`
- gRPC at `localhost:8421`
- OpenAI-compatible endpoint at `http://localhost:8420/v1/chat/completions`

### 4. Use the desktop app

```bash
cd crates/synapse-desktop
cargo tauri dev
```

## Configuration

Create `~/.synapse/synapse.toml` or copy from `deploy/synapse.toml.example`.

Key settings:
- `storage.models_dir` — where models are stored
- `backends.default` — which inference backend to use
- `bridge.sy_endpoint` — SecureYeoman connection (if using orchestration)

## Next Steps

- [Add a new backend](../backends.md)
- [Start a training job](./training-guide.md)
- [Connect to SecureYeoman](./sy-integration.md)
