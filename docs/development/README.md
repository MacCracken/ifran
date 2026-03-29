# Development Guide

## Prerequisites

- Rust stable (via `rustup`)
- `protoc` (protobuf compiler)
- Docker (optional, for training executor)
- CUDA toolkit (optional, for GPU backends)

## Quick Start

```bash
# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Check formatting + clippy
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
```

## Testing

132 tests across 7 crates. CI runs per-package test matrix with coverage via cargo-tarpaulin.

```bash
# All tests
cargo test --workspace

# Single crate
cargo test -p ifran-core

# Coverage (requires cargo-tarpaulin)
cargo tarpaulin --workspace --out html
```

Current coverage: ~73%. CI threshold: 65%. See [roadmap.md](./roadmap.md) for the coverage improvement plan.

## Project Structure

See [architecture.md](../architecture.md) for the full workspace layout and design decisions.

See [roadmap.md](./roadmap.md) for remaining work.

## Configuration

Config is auto-discovered: `IFRAN_CONFIG` env → `~/.ifran/ifran.toml` → `/etc/ifran/ifran.toml` → defaults.

See `deploy/ifran.toml.example` for all options.

## Versioning

Semver format: `MAJOR.MINOR.PATCH` for releases, with optional pre-release suffix.

## CI/CD

GitHub Actions workflows:
- **Build** — x86_64 native + aarch64 cross-compilation (via `cross`)
- **Quality** — `cargo fmt --check` + `cargo clippy -D warnings`
- **Security** — Trivy scan, `cargo audit`, `cargo deny`
- **Tests** — per-package matrix (7 jobs in parallel)
- **Coverage** — `cargo tarpaulin` with 65% threshold
- **Docs** — verify required files exist, `cargo doc --no-deps`
- **Container** — Docker build verification
- **License** — MIT license check

## Related Projects

- **Agnosticos** (`../agnosticos/`) — Target operating system
- **SecureYeoman** (`../secureyeoman/`) — Orchestrator with cyclic integration
