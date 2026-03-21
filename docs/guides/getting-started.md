# Getting Started with Ifran

## Prerequisites

- **Rust** stable toolchain (install via [rustup](https://rustup.rs))
- **protoc** (protobuf compiler)
- **Docker** (optional, for training executor)
- **CUDA toolkit** (optional, for GPU inference)

## Installation

### From Source

```bash
git clone git@github.com:MacCracken/ifran.git
cd ifran
cargo build --release
```

The binaries will be at:
- `target/release/ifran` — CLI
- `target/release/ifran-api` — API server

### On Agnosticos

```bash
pkg install ifran
systemctl enable --now ifran
```

This installs:
- `/usr/local/bin/ifran` and `/usr/local/bin/ifran-server`
- `/etc/ifran/ifran.toml` (system config)
- `/var/lib/ifran/` (models, database, checkpoints, cache)
- `ifran.service` (systemd unit with security hardening)

## First Steps

### 1. Pull a model

```bash
ifran pull meta-llama/Llama-3.1-8B-Instruct --quant q4_k_m
```

This resolves the model on HuggingFace Hub, finds the matching GGUF file for the requested quantization, downloads it with a progress bar, verifies SHA-256 integrity, and registers it in the local SQLite catalog.

Downloads support resume — if interrupted, re-running the same command picks up where it left off via HTTP Range requests and `.part` files.

Set `HF_TOKEN` for gated models:
```bash
export HF_TOKEN=hf_xxxxx
ifran pull meta-llama/Llama-3.1-8B-Instruct --quant q4_k_m
```

### 2. List models

```bash
ifran list
```

Shows all locally registered models with name, format, quantization, size, and pull date.

### 3. Remove a model

```bash
ifran rm meta-llama/Llama-3.1-8B-Instruct
ifran rm <model-name> -y   # skip confirmation
```

Removes the model files from disk and deletes it from the catalog.

### 4. Chat with it

```bash
ifran run meta-llama/Llama-3.1-8B-Instruct
```

### 5. Start the API server

```bash
ifran serve
```

The server exposes:
- REST API at `http://localhost:8420`
- gRPC at `localhost:8421`
- OpenAI-compatible endpoint at `http://localhost:8420/v1/chat/completions`

### 6. Enable authentication

```bash
export IFRAN_API_KEY=your-secret-token
ifran serve
```

All endpoints except `/health` will require `Authorization: Bearer your-secret-token`.

### 7. Use the desktop app

```bash
cd crates/ifran-desktop
cargo tauri dev
```

## Configuration

Ifran discovers config in this order:

1. `IFRAN_CONFIG` environment variable (explicit path)
2. `~/.ifran/ifran.toml` (user config)
3. `/etc/ifran/ifran.toml` (system config, Agnosticos)
4. Built-in defaults

Create `~/.ifran/ifran.toml` or copy from `deploy/ifran.toml.example`.

Key settings:
- `server.bind` — REST API address (default: `0.0.0.0:8420`)
- `server.grpc_bind` — gRPC address (default: `0.0.0.0:8421`)
- `storage.models_dir` — where models are stored (default: `~/.ifran/models/`)
- `storage.database` — SQLite catalog path (default: `~/.ifran/ifran.db`)
- `backends.default` — which inference backend to use (default: `llamacpp`)
- `training.executor` — `docker` or `subprocess` (default: `docker`)
- `bridge.sy_endpoint` — SecureYeoman connection (if using orchestration)
- `hardware.gpu_memory_reserve_mb` — VRAM to keep free (default: 512 MB)

## Storage Layout

Ifran stores all data under `~/.ifran/` by default (or `/var/lib/ifran/` on Agnosticos):

```text
~/.ifran/
├── ifran.toml        # Configuration file
├── ifran.db          # SQLite model catalog
├── cache/              # Temporary download cache
├── checkpoints/        # Training checkpoints
└── models/
    ├── llama-3.1-8b-instruct-q4km/
    │   └── model.gguf
    └── mistral-7b-q5km/
        └── model.gguf
```

Models are tracked in a SQLite catalog (`ifran.db`) and stored in slugified directories under `models/`.

## Next Steps

- [Inference Backends](../backends.md) — configure and add backends
- [Training Guide](./training-guide.md) — fine-tune models
- [SY Integration](./sy-integration.md) — connect to SecureYeoman
- [API Reference](../api-reference.md) — full endpoint documentation
