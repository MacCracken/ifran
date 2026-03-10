# ADR-008: Agnosticos System Integration

## Status
Accepted

## Context
Synapse needs to run as a managed system service on Agnosticos, the target operating system. This requires proper packaging, service management, security hardening, and capability registration with the Agnosticos agent-runtime.

## Decision
Integrate Synapse as a first-class Agnosticos capability provider:

1. **systemd service** (`deploy/synapse.service`) with `Type=notify`, security hardening (`ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, `NoNewPrivileges`), GPU device access, and a dedicated `synapse` system user.

2. **Agnosticos package spec** (`deploy/agnosticos/synapse.pkg.toml`) with pre/post install hooks for user creation, directory setup, systemd enablement, and capability registration.

3. **Config auto-discovery** — `SynapseConfig::discover()` resolves config in order: `SYNAPSE_CONFIG` env → `~/.synapse/synapse.toml` → `/etc/synapse/synapse.toml` → defaults. This supports both user-level and system-level operation.

4. **System paths** — when installed via package, data lives at `/var/lib/synapse/` (models, database, cache, checkpoints) instead of `~/.synapse/`.

## Consequences
- Synapse installs cleanly on Agnosticos with `pkg install synapse`
- Runs under a dedicated system user with minimal privileges
- Registers automatically as an `llm-inference` capability provider
- SY discovers Synapse instances through the capability registry
- Config discovery chain means the same binary works for development (`~/.synapse/`) and production (`/etc/synapse/`)
