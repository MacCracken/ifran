# ADR-005: Semver Versioning

**Status**: Superseded (was CalVer, now Semver)
**Date**: 2026-03-09
**Updated**: 2026-03-28

## Context

Consistent versioning across the project ecosystem reduces confusion and aligns release tracking. CalVer was previously used but caused issues with crates.io compatibility and standard Cargo tooling expectations.

## Decision

Use Semantic Versioning: `MAJOR.MINOR.PATCH` for releases, with optional pre-release suffix (e.g., `1.0.0-rc1`). The VERSION file at repo root is the single source of truth.

## Consequences

- Compatible with crates.io and Cargo semver expectations
- Clear signaling of breaking changes via major version bumps
- Pre-release versions use standard semver suffixes
- VERSION file at repo root is the single source of truth
