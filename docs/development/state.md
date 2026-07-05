# ifran — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**1.3.0 (the Rust line's version — unchanged until the first Cyrius cut; the
completed port ships as 2.0.0 per the goonj/naad precedent).** **M0 DONE
2026-07-04**: `cyrius port` ran — the 53.6k-line Rust tree (+ Cargo/Cross/
cargo-config/osv strays) lives at `rust-old/` (reference oracle, eventual
dismissal); skeleton builds + 2/2 scaffold tests; manifest corrected
(`${file:VERSION}`, AGPL license matched to the LICENSE file — relicensing
flagged as a maintainer call). Disposition map + milestone tracker:
[`port-ledger.md`](port-ledger.md). **Next: M1 — the job core** (`ifran run
<job.cyml>` drives a real sibling binary end-to-end and persists the run
record). NOT porting: `backends/` (dead broker), server/lineage/marketplace/
fleet/rag/hardware (owned homes) — see the ledger.

## Toolchain

- **Cyrius pin**: `6.4.3` (in `cyrius.cyml [package].cyrius`)

## Source

- Rust reference: 53612 lines at `rust-old/` (frozen, do not edit).
- Cyrius port: scaffold only — `src/main.cyr` stub.

## Tests

_Replace with parity test status once tests land._

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, vec, str, syscalls, io, args, assert

## Consumers

_None yet._

## Next

See [`roadmap.md`](roadmap.md). The first milestone is typically Rust→Cyrius surface parity for the 53612-line subset.
