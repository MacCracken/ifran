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
cargo build --release
```

The binaries will be at:
- `target/release/synapse` — CLI
- `target/release/synapse-api` — API server

### On Agnosticos

```bash
pkg install synapse
systemctl enable --now synapse
```

This installs:
- `/usr/local/bin/synapse` and `/usr/local/bin/synapse-server`
- `/etc/synapse/synapse.toml` (system config)
- `/var/lib/synapse/` (models, database, checkpoints, cache)
- `synapse.service` (systemd unit with security hardening)

## First Steps

### 1. Pull a model

```bash
synapse pull meta-llama/Llama-3.1-8B-Instruct --quant q4_k_m
```

This resolves the model on HuggingFace Hub, finds the matching GGUF file for the requested quantization, downloads it with a progress bar, verifies SHA-256 integrity, and registers it in the local SQLite catalog.

Downloads support resume — if interrupted, re-running the same command picks up where it left off via HTTP Range requests and `.part` files.

Set `HF_TOKEN` for gated models:
```bash
export HF_TOKEN=hf_xxxxx
synapse pull meta-llama/Llama-3.1-8B-Instruct --quant q4_k_m
```

### 2. List models

```bash
synapse list
```

Shows all locally registered models with name, format, quantization, size, and pull date.

### 3. Remove a model

```bash
synapse rm meta-llama/Llama-3.1-8B-Instruct
synapse rm <model-name> -y   # skip confirmation
```

Removes the model files from disk and deletes it from the catalog.

### 4. Chat with it

```bash
synapse run meta-llama/Llama-3.1-8B-Instruct
```

### 5. Start the API server

```bash
synapse serve
```

The server exposes:
- REST API at `http://localhost:8420`
- gRPC at `localhost:8421`
- OpenAI-compatible endpoint at `http://localhost:8420/v1/chat/completions`

### 6. Enable authentication

```bash
export SYNAPSE_API_KEY=your-secret-token
synapse serve
```

All endpoints except `/health` will require `Authorization: Bearer your-secret-token`.

### 7. Use the desktop app

```bash
cd crates/synapse-desktop
cargo tauri dev
```

## Configuration

Synapse discovers config in this order:

1. `SYNAPSE_CONFIG` environment variable (explicit path)
2. `~/.synapse/synapse.toml` (user config)
3. `/etc/synapse/synapse.toml` (system config, Agnosticos)
4. Built-in defaults

Create `~/.synapse/synapse.toml` or copy from `deploy/synapse.toml.example`.

Key settings:
- `server.bind` — REST API address (default: `0.0.0.0:8420`)
- `server.grpc_bind` — gRPC address (default: `0.0.0.0:8421`)
- `storage.models_dir` — where models are stored (default: `~/.synapse/models/`)
- `storage.database` — SQLite catalog path (default: `~/.synapse/synapse.db`)
- `backends.default` — which inference backend to use (default: `llamacpp`)
- `training.executor` — `docker` or `subprocess` (default: `docker`)
- `bridge.sy_endpoint` — SecureYeoman connection (if using orchestration)
- `hardware.gpu_memory_reserve_mb` — VRAM to keep free (default: 512 MB)

## Storage Layout

Synapse stores all data under `~/.synapse/` by default (or `/var/lib/synapse/` on Agnosticos):

```text
~/.synapse/
├── synapse.toml        # Configuration file
├── synapse.db          # SQLite model catalog
├── cache/              # Temporary download cache
├── checkpoints/        # Training checkpoints
└── models/
    ├── llama-3.1-8b-instruct-q4km/
    │   └── model.gguf
    └── mistral-7b-q5km/
        └── model.gguf
```

Models are tracked in a SQLite catalog (`synapse.db`) and stored in slugified directories under `models/`.

## Next Steps

- [Inference Backends](../backends.md) — configure and add backends
- [Training Guide](./training-guide.md) — fine-tune models
- [SY Integration](./sy-integration.md) — connect to SecureYeoman
- [API Reference](../api-reference.md) — full endpoint documentation
