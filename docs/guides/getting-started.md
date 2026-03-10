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
synapse remove meta-llama/Llama-3.1-8B-Instruct
synapse remove <model-name> -y   # skip confirmation
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

### 6. Use the desktop app

```bash
cd crates/synapse-desktop
cargo tauri dev
```

## Storage Layout

Synapse stores all data under `~/.synapse/` by default:

```text
~/.synapse/
├── synapse.toml        # Configuration file
├── synapse.db          # SQLite model catalog
├── cache/              # Temporary download cache
├── checkpoints/        # Training checkpoints
└── models/
    ├── llama-3.1-8b-instruct-q4km/
    │   ├── model.gguf
    │   └── metadata.json
    └── mistral-7b-q5km/
        ├── model.gguf
        └── metadata.json
```

Models are tracked in a SQLite catalog (`synapse.db`) and stored in slugified directories under `models/`.

## Configuration

Create `~/.synapse/synapse.toml` or copy from `deploy/synapse.toml.example`.

Key settings:
- `storage.models_dir` — where models are stored (default: `~/.synapse/models/`)
- `storage.database` — SQLite catalog path (default: `~/.synapse/synapse.db`)
- `backends.default` — which inference backend to use
- `bridge.sy_endpoint` — SecureYeoman connection (if using orchestration)
- `hardware.gpu_memory_reserve_mb` — VRAM to keep free (default: 512 MB)

## Next Steps

- [Add a new backend](../backends.md)
- [Start a training job](./training-guide.md)
- [Connect to SecureYeoman](./sy-integration.md)
