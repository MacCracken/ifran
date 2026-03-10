# Development Guide

## Prerequisites

- Rust stable (via `rustup`)
- `protoc` (protobuf compiler)
- Docker (for training executor)
- CUDA toolkit (optional, for GPU backends)

## Quick Start

```bash
# Setup dev dependencies
./scripts/setup-dev.sh

# Build all crates
make build

# Run tests
make test

# Start dev server with auto-reload
make dev

# Lint + format + test
make check
```

## Project Structure

See [architecture.md](../architecture.md) for the full workspace layout and design decisions.

See [roadmap.md](./roadmap.md) for the development roadmap and phased delivery plan.

## Versioning

CalVer format: `YYYY.M.D` for releases, `YYYY.M.D-N` for patches.

Matches the convention used by Agnosticos and SecureYeoman.

## Related Projects

- **Agnosticos** (`../agnosticos/`) — Target operating system
- **SecureYeoman** (`../secureyeoman/`) — Orchestrator with cyclic integration
