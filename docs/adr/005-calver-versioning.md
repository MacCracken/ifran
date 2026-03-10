# ADR-005: CalVer Versioning

**Status**: Accepted
**Date**: 2026-03-09

## Context

Consistent versioning across the project ecosystem (Agnosticos, SecureYeoman, Synapse) reduces confusion and aligns release tracking.

## Decision

Use Calendar Versioning: `YYYY.M.D` for releases, `YYYY.M.D-N` for patches (where N is a patch counter). Matches the convention in Agnosticos and SecureYeoman.

## Consequences

- Version immediately communicates release date
- No ambiguity about "major" vs "minor" — every release is date-stamped
- Patch releases use `-N` suffix for same-day hotfixes
- VERSION file at repo root is the single source of truth
