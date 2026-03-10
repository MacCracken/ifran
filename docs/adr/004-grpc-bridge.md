# ADR-004: gRPC for SY Bridge

**Status**: Accepted
**Date**: 2026-03-09

## Context

Synapse and SecureYeoman have a cyclic relationship: SY orchestrates Synapse instances, and Synapse calls back to SY for resource allocation and progress reporting. The protocol must support bidirectional streaming and be strongly typed.

## Decision

Use gRPC via `tonic` for the bridge protocol. Two services define the contract:
- `SynapseBridge`: SY → Synapse (commands)
- `YeomanBridge`: Synapse → SY (callbacks)

Proto definitions in `proto/bridge.proto` are the shared source of truth.

## Consequences

- Strongly typed contract between projects via protobuf
- Bidirectional streaming for real-time progress updates
- `tonic` is already used in Agnosticos, consistent toolchain
- SY (TypeScript) can generate a client from the same proto files
- Both services degrade gracefully if the other is unavailable
