# Contributing to Synapse

Thank you for your interest in contributing to Synapse. This document provides guidelines and information to help you get started.

## Project Overview

Synapse is a Rust-based LLM controller that provides a unified interface for managing, serving, and training large language models. The project is organized as a Cargo workspace containing 8 crates:

| Crate | Purpose |
|---|---|
| `synapse-types` | Shared type definitions, error types, and data models |
| `synapse-core` | Core engine: model registry, download manager, configuration |
| `synapse-backends` | Pluggable inference backend trait and implementations |
| `synapse-train` | Training orchestration with Docker/subprocess/native executors |
| `synapse-api` | REST (Axum) and gRPC (tonic) server |
| `synapse-bridge` | Bidirectional gRPC bridge (SY <-> Synapse) |
| `synapse-cli` | Command-line interface |
| `synapse-desktop` | Tauri-based desktop application shell |

## Prerequisites

- **Rust** (stable toolchain) -- install via [rustup](https://rustup.rs/)
- **protoc** (Protocol Buffers compiler) -- required for gRPC code generation
- **cmake** -- required by some native dependencies
- **Docker** (optional) -- needed for containerized training jobs and integration tests
- **CUDA toolkit** (optional) -- needed only if building GPU-accelerated backends

## Getting Started

```bash
# Clone the repository
git clone https://github.com/MacCracken/synapse.git
cd synapse

# Install build dependencies (Linux)
./scripts/install-build-deps.sh

# Or run the dev setup script
./scripts/setup-dev.sh

# Build all crates
make build

# Run the test suite
make test
```

## Git Workflow

We use a **Simplified Git Flow** model:

```
main (stable releases)
 └── develop (integration branch)
      ├── feature/add-ollama-backend
      ├── bugfix/fix-download-retry
      └── docs/update-architecture
```

- **main** -- always releasable; tags correspond to CalVer releases.
- **develop** -- integration branch where feature branches merge.
- **feature/\*** -- new functionality branched from develop.

### Branch Naming

Use the following prefixes:

| Prefix | Use |
|---|---|
| `feature/` | New features or enhancements |
| `bugfix/` | Bug fixes |
| `docs/` | Documentation-only changes |
| `refactor/` | Code restructuring without behavior change |
| `security/` | Security patches or hardening |
| `chore/` | Build, CI, tooling, dependency updates |

### Conventional Commits

All commit messages must follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

**Scopes:** `types`, `core`, `backends`, `train`, `api`, `bridge`, `cli`, `desktop`

Examples:

```
feat(backends): add llama.cpp GGUF quantization support
fix(core): handle interrupted downloads with resume
docs(api): document OpenAI-compatible endpoint mapping
chore(ci): add SBOM generation to release workflow
```

## Pull Request Process

1. Branch from `develop` using the naming convention above.
2. Make your changes, ensuring all checks pass locally.
3. Open a PR targeting `develop`.
4. Fill out the PR template completely.
5. All CI checks must pass before merge.
6. At least one approving review is required.

## Coding Standards

### Rust

- Format with `rustfmt` (default configuration, 100 character line length).
- Pass `clippy` with `-D warnings` (all warnings are errors).
- Maximum line length: 100 characters.
- Write doc comments (`///`) for all public items.
- Use `thiserror` for error types, `anyhow` only in binaries.
- Prefer strong typing over stringly-typed interfaces.

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

### TypeScript / Svelte

- Format with Prettier (2-space indent).
- Lint with ESLint.

### Python

- Format with Black (4-space indent).
- Lint with Ruff.

## Testing

- **Unit tests** live alongside source code in each crate (`#[cfg(test)]` modules).
- **Integration tests** live alongside each crate (e.g. `crates/synapse-api/tests/`).
- Coverage target: **45%** minimum (CI enforced, goal: 80%).
- CI runs tests per-package in parallel.

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p synapse-core

# Run with coverage (requires cargo-tarpaulin)
cargo tarpaulin --workspace --out html
```

## Documentation

- Update files in `docs/` for any architectural or user-facing changes.
- Add an **Architecture Decision Record** (ADR) in `docs/adr/` for significant design decisions.
- Keep the README and CHANGELOG up to date.

## Security

- **Never** commit secrets, API keys, or credentials.
- Validate all inputs at API boundaries.
- Run `cargo audit` before submitting PRs that update dependencies.
- See [SECURITY.md](SECURITY.md) for the full security policy and reporting instructions.

## Release Process

Synapse uses **Calendar Versioning** (CalVer):

- Format: `YYYY.M.D` for releases (e.g., `2026.3.9`)
- Format: `YYYY.M.D-N` for same-day patch releases (e.g., `2026.3.9-1`)

Releases are created through the `release-automation` GitHub Actions workflow, which handles:

1. Version bumping across all workspace crates.
2. CHANGELOG generation.
3. Binary builds for supported platforms.
4. Container image publishing.
5. GitHub Release creation with artifacts.

## Crate Development Guide

When working on a specific crate, keep these responsibilities in mind:

- **synapse-types** -- Keep this crate dependency-free where possible. Changes here affect the entire workspace.
- **synapse-core** -- The engine. Manages model registry, downloads, and configuration. Does not depend on any specific backend.
- **synapse-backends** -- Implement the `InferenceBackend` trait for new backends. Each backend is feature-gated.
- **synapse-train** -- Orchestrates training jobs. Executor implementations (Docker, subprocess, native) live here.
- **synapse-api** -- HTTP and gRPC endpoints. Delegates to core and backends. OpenAI-compatible API surface.
- **synapse-bridge** -- Handles bidirectional communication with SY. Protocol defined in `proto/`.
- **synapse-cli** -- Thin layer over core functionality. Keep business logic out of this crate.
- **synapse-desktop** -- Tauri shell. Frontend lives in `crates/synapse-desktop/src/`.

---

Thank you for contributing to Synapse.
