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
flagged as a maintainer call). Disposition map + milestone tracker: [`port-ledger.md`](port-ledger.md).
**M1 (the job core) ✅ DONE same day**: jobspec (bayan CYML) + executor (own
fork+pipe+execve with exit-code capture + stderr merge) + patra run store +
CLI (`run`/`runs`); suite **18/18**; **proof met** — `ifran run
examples/lora-demo.cyml` drove the real `anukulana gpt2-lora` end-to-end
(33.6 s, exit 0, training log captured, run recorded). **M2 (checkpoint/model store) ✅ DONE same day**: operator key management
(`keys init/show` — getrandom → sigil ed25519, 0600 secret, no-clobber) + the
store (`store add/ls/verify` — tula structural validation, Ed25519 verify with
honest per-artifact status, sigil-sha256 content addressing + dedup,
tamper-detecting verify); suite **29/29**; **proof met** — keys → `gpt2-tula`
job (80 s) → both artifacts ingested + verified. **Next: M3 — datasets.** NOT porting: `backends/` (dead broker), server/lineage/marketplace/
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
