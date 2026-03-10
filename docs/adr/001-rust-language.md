# ADR-001: Rust as Primary Language

**Status**: Accepted
**Date**: 2026-03-09

## Context

Synapse needs to manage LLM inference and training with high performance, low overhead, and native hardware access. The primary target platform (Agnosticos) is a Rust-focused operating system.

## Decision

Use Rust as the primary language for all Synapse crates. The Cargo workspace pattern follows Agnosticos conventions.

## Consequences

- Native performance for model loading, memory management, and inference orchestration
- Direct FFI to llama.cpp and other C/C++ inference runtimes
- Consistent toolchain with Agnosticos (shared patterns, types, build system)
- Steeper learning curve than Go/TS but justified by performance requirements and ecosystem alignment
- Python training scripts run as subprocesses/containers rather than being embedded
