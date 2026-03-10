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
│   └── synapse-desktop    # Tauri v2 + Svelte
├── proto/                 # Protobuf definitions (source of truth)
├── docker/                # Dockerfiles + compose
├── deploy/                # systemd, config examples, agnosticos pkg
├── docs/                  # Documentation
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
All inference backends implement the `InferenceBackend` trait with dynamic dispatch via `Arc<dyn InferenceBackend>`. Feature flags gate heavy native dependencies so builds without CUDA/TensorRT still compile.

### Model Pulling
Multi-source registry client with adapters for HuggingFace Hub, OCI registries (Ollama-compatible), direct URLs, and local filesystem. Downloads are chunked, parallel, resumable, and integrity-verified.

### Training
Orchestrated via subprocess/Docker — no embedded Python runtime. Training scripts run in containers or as child processes, with logs streamed and checkpoints monitored from Rust.

### SY Bridge
gRPC bidirectional streaming via `tonic`. Synapse acts as both server (receiving from SY) and client (calling back for GPU allocation, scaling, progress reporting). Heartbeats maintain connection health. Degrades gracefully if SY is unavailable.

### Agnosticos Integration
Runs as a systemd `Type=notify` service. Registers with the agent-runtime as a capability provider. Ships as an Agnosticos package.
