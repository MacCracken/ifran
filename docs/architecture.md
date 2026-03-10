# Synapse Architecture

## Overview

Synapse is a Rust-based LLM controller for pulling, managing, and training language models. It provides CLI, REST/gRPC API, and desktop interfaces. It integrates bidirectionally with SecureYeoman (orchestrator) and runs as a system service on Agnosticos.

## Workspace Layout

```
synapse/
├── crates/
│   ├── synapse-types      # Shared types, protobuf codegen
│   ├── synapse-core       # Model registry, pull engine, lifecycle, hardware
│   ├── synapse-backends   # Pluggable inference backends (trait-based)
│   ├── synapse-train      # Training orchestration
│   ├── synapse-api        # Axum REST + tonic gRPC server
│   ├── synapse-bridge     # SY<->Synapse bidirectional gRPC
│   ├── synapse-cli        # CLI binary
│   └── synapse-desktop    # Tauri v2 + SvelteKit
├── proto/                 # Protobuf definitions (source of truth)
├── docker/                # Dockerfiles (server, dev, trainer, release)
├── deploy/                # systemd, config examples, Agnosticos pkg
│   ├── synapse.service    # systemd unit with security hardening
│   ├── synapse.toml.example
│   └── agnosticos/
│       └── synapse.pkg.toml
├── docs/                  # Documentation, ADRs, guides
└── scripts/               # Build, test, dev setup
```

## Crate Dependency Graph

```
synapse-types          (leaf — no internal deps)
       ↑
synapse-core           (→ types)
       ↑
synapse-backends       (→ types, core)
       ↑
synapse-train          (→ types, core)
       ↑
synapse-bridge         (→ types)
       ↑
synapse-api            (→ types, core, backends, train, bridge)
       ↑
synapse-cli            (→ types, core, api)
synapse-desktop        (→ types, core, backends, train)
```

## Key Design Decisions

### Backend Pluggability
All inference backends implement the `InferenceBackend` trait with dynamic dispatch via `Arc<dyn InferenceBackend>`. Feature flags gate heavy native dependencies so builds without CUDA/TensorRT still compile. The `BackendRouter` auto-selects backends by model format, hardware, and user preference.

### Model Pulling
Multi-source registry client with adapters for HuggingFace Hub (with GGUF quant resolution), direct URLs, and local filesystem. Downloads are chunked, resumable via `.part` files and HTTP Range headers, and integrity-verified (SHA-256 / BLAKE3).

### Training
Orchestrated via subprocess/Docker — no embedded Python runtime. Training scripts (`train_sft.py`, `train_full.py`, `train_dpo.py`, `train_rlhf.py`, `train_distill.py`) run in containers or as child processes, with logs streamed and checkpoints monitored from Rust. The `JobManager` enforces concurrent job limits.

### SY Bridge
gRPC bidirectional streaming via `tonic`. Synapse acts as both server (receiving from SY) and client (calling back for GPU allocation, scaling, progress reporting). Heartbeats maintain connection health. Degrades gracefully if SY is unavailable.

### Configuration Discovery
Config is resolved in order: `SYNAPSE_CONFIG` env → `~/.synapse/synapse.toml` → `/etc/synapse/synapse.toml` → built-in defaults. This supports both user-level development and system-level Agnosticos deployments.

### Authentication
Optional Bearer token auth via `SYNAPSE_API_KEY` environment variable. When unset, the API is open. The `/health` endpoint is always unauthenticated for load balancer probes.

### Agnosticos Integration
Runs as a `systemd` `Type=notify` service with security hardening (`ProtectSystem=strict`, `PrivateTmp`, `NoNewPrivileges`). Registers with the agent-runtime as a capability provider via `synapse.pkg.toml` hooks. Ships as an Agnosticos package with dedicated `synapse` system user.

### TLS
Uses `rustls` (via reqwest's `rustls-tls` feature) instead of OpenSSL. This avoids cross-compilation issues for aarch64 and removes the system OpenSSL dependency.
