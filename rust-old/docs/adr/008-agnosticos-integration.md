# ADR-008: Agnosticos System Integration

## Status
Accepted

## Context
Ifran needs to run as a managed system service on Agnosticos, the target operating system. This requires proper packaging, service management, security hardening, and capability registration with the Agnosticos agent-runtime.

## Decision
Integrate Ifran as a first-class Agnosticos capability provider:

1. **systemd service** (`deploy/ifran.service`) with `Type=notify`, security hardening (`ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, `NoNewPrivileges`), GPU device access, and a dedicated `ifran` system user.

2. **Agnosticos package spec** (`deploy/agnosticos/ifran.pkg.toml`) with pre/post install hooks for user creation, directory setup, systemd enablement, and capability registration.

3. **Config auto-discovery** — `IfranConfig::discover()` resolves config in order: `IFRAN_CONFIG` env → `~/.ifran/ifran.toml` → `/etc/ifran/ifran.toml` → defaults. This supports both user-level and system-level operation.

4. **System paths** — when installed via package, data lives at `/var/lib/ifran/` (models, database, cache, checkpoints) instead of `~/.ifran/`.

## Consequences
- Ifran installs cleanly on Agnosticos with `pkg install ifran`
- Runs under a dedicated system user with minimal privileges
- Registers automatically as an `llm-inference` capability provider
- SY discovers Ifran instances through the capability registry
- Config discovery chain means the same binary works for development (`~/.ifran/`) and production (`/etc/ifran/`)
