# ADR-003: Training via Subprocess/Docker Executors

**Status**: Accepted
**Date**: 2026-03-09

## Context

LLM training requires Python libraries (PyTorch, Transformers, PEFT, Unsloth, TRL). Embedding a Python runtime in Rust adds complexity and fragility.

## Decision

Orchestrate training as external processes:
1. **Docker executor** (default): launches containers with the `synapse-trainer` image
2. **Subprocess executor**: spawns Python scripts directly
3. **Native executor** (future): in-process Rust training via candle/burn for small models

Training scripts are bundled in `crates/synapse-train/src/scripts/`. The Rust code manages job lifecycle, streams logs, and monitors checkpoints.

## Consequences

- Clean separation between orchestration (Rust) and training (Python)
- Follows SecureYeoman's `FinetuneManager` pattern (Docker container executor)
- Training environment is reproducible via Docker images
- Slight overhead from process spawning (irrelevant for training job duration)
- Requires Docker or Python on the host for training features
