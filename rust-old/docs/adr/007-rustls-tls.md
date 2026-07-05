# ADR-007: rustls over OpenSSL

## Status
Accepted

## Context
Cross-compiling for aarch64 failed because `openssl-sys` requires target-architecture OpenSSL development headers (`libssl-dev`). Installing cross-compile OpenSSL headers is fragile and adds CI complexity. The `cross` tool handles most cross-compilation needs but OpenSSL remains a persistent pain point.

## Decision
Use `rustls` (via reqwest's `rustls-tls` feature) instead of OpenSSL for all TLS operations. The workspace-level reqwest dependency uses `default-features = false` with explicit `rustls-tls` feature.

## Consequences
- aarch64 cross-compilation works without additional system dependencies
- No runtime dependency on system OpenSSL libraries
- Pure Rust TLS stack — consistent behavior across platforms
- Slightly larger binary size (bundled root certificates via `webpki-roots`)
- Some edge cases with older TLS servers may behave differently than OpenSSL
