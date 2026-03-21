# Ifran Architecture

## Overview

Ifran is a Rust-based LLM controller for pulling, managing, and training language models. It provides CLI, REST/gRPC API, and desktop interfaces. It integrates bidirectionally with SecureYeoman (orchestrator) and runs as a system service on Agnosticos.

## Workspace Layout

```
ifran/
├── crates/
│   ├── ifran-types      # Shared types, protobuf codegen
│   ├── ifran-core       # Model registry, pull engine, lifecycle, hardware
│   ├── ifran-backends   # Pluggable inference backends (trait-based)
│   ├── ifran-train      # Training orchestration
│   ├── ifran-api        # Axum REST + tonic gRPC server
│   ├── ifran-bridge     # SY<->Ifran bidirectional gRPC
│   ├── ifran-cli        # CLI binary
│   └── ifran-desktop    # Tauri v2 + SvelteKit
├── proto/                 # Protobuf definitions (source of truth)
├── docker/                # Dockerfiles (server, dev, trainer, release)
├── deploy/                # systemd, config examples, Agnosticos pkg
│   ├── ifran.service    # systemd unit with security hardening
│   ├── ifran.toml.example
│   └── agnosticos/
│       └── ifran.pkg.toml
├── docs/                  # Documentation, ADRs, guides
└── scripts/               # Build, test, dev setup
```

## Crate Dependency Graph

```
ifran-types          (leaf — no internal deps)
       ↑
ifran-core           (→ types)
       ↑
ifran-backends       (→ types, core)
       ↑
ifran-train          (→ types, core)
       ↑
ifran-bridge         (→ types)
       ↑
ifran-api            (→ types, core, backends, train, bridge)
       ↑
ifran-cli            (→ types, core, api)
ifran-desktop        (→ types, core, backends, train)
```

## Key Design Decisions

### Backend Pluggability
All inference backends implement the `InferenceBackend` trait with dynamic dispatch via `Arc<dyn InferenceBackend>`. Feature flags gate heavy native dependencies so builds without CUDA/TensorRT still compile. The `BackendRouter` auto-selects backends by model format, hardware, and user preference.

### Model Pulling
Multi-source registry client with adapters for HuggingFace Hub (with GGUF quant resolution), direct URLs, and local filesystem. Downloads are chunked, resumable via `.part` files and HTTP Range headers, and integrity-verified (SHA-256 / BLAKE3).

### Training
Orchestrated via subprocess/Docker — no embedded Python runtime. Training scripts (`train_sft.py`, `train_full.py`, `train_dpo.py`, `train_rlhf.py`, `train_distill.py`) run in containers or as child processes, with logs streamed and checkpoints monitored from Rust. The `JobManager` enforces concurrent job limits.

### SY Bridge
gRPC bidirectional streaming via `tonic`. Ifran acts as both server (receiving from SY) and client (calling back for GPU allocation, scaling, progress reporting). Heartbeats maintain connection health. Degrades gracefully if SY is unavailable.

### Configuration Discovery
Config is resolved in order: `IFRAN_CONFIG` env → `~/.ifran/ifran.toml` → `/etc/ifran/ifran.toml` → built-in defaults. This supports both user-level development and system-level Agnosticos deployments.

### Authentication
Optional Bearer token auth via `IFRAN_API_KEY` environment variable. When unset, the API is open. The `/health` endpoint is always unauthenticated for load balancer probes.

### Agnosticos Integration
Runs as a `systemd` `Type=notify` service with security hardening (`ProtectSystem=strict`, `PrivateTmp`, `NoNewPrivileges`). Registers with the agent-runtime as a capability provider via `ifran.pkg.toml` hooks. Ships as an Agnosticos package with dedicated `ifran` system user.

### TLS
Uses `rustls` (via reqwest's `rustls-tls` feature) instead of OpenSSL. This avoids cross-compilation issues for aarch64 and removes the system OpenSSL dependency.
