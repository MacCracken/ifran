# ADR-006: Tauri for Desktop Application

**Status**: Accepted
**Date**: 2026-03-09

## Context

Synapse needs a desktop application for model management, chat interface, training dashboards, and system monitoring. The backend is Rust.

## Decision

Use Tauri v2 with a Svelte frontend. The Tauri backend calls synapse-core directly via Rust function calls (not HTTP), since they compile together in the same binary.

## Consequences

- Small binary size compared to Electron (~10MB vs ~100MB+)
- Native Rust backend integration (no IPC overhead for core operations)
- Svelte provides reactive UI with minimal bundle size
- Follows SecureYeoman's existing Tauri v2 pattern
- WebView rendering may have minor platform inconsistencies (acceptable)
- Web version can share the Svelte frontend, hitting the REST API instead of Tauri commands
