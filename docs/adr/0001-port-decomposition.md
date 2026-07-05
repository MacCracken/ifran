# ADR 0001 — The port decomposes: one thin control plane, no math, no broker

**Status:** accepted (2026-07-04, maintainer) · **Shipped:** 2.0.0

## Context

The Rust ifran (1.3.x, 53.6k lines) was a monolith: a 15-engine inference-
backend broker, an axum REST server (~21 groups), training methods shelled to
Python, plus lineage/marketplace/fleet/RAG/hardware subsystems. Porting it
line-for-line would have carried a boundary the AGNOS ecosystem had already
dissolved into owned homes.

## Decision

1. **The Rust monolith boundary does NOT carry across.** Everything an
   existing home owns leaves at the port boundary: lineage→itihas,
   serving→hoosh, marketplace→mela, fleet→seema, RAG→mneme,
   GPU→ai-hwaccel/mabda, weight codec→tula.
2. **`backends/` is dead**, not ported — the murti re-derivation (agnosticos
   `planning/murti.md`) found the external-engine broker anti-sovereign;
   local inference is rosnet/tentib via hoosh, foreign engines go to mehman.
3. **The remainder is ONE repo** — job manager, model store, datasets,
   sweeps, evals, preferences — with internal module boundaries on the
   candidate seams and **second-consumer-gated extraction** (the model store
   is the named first candidate).
4. **No training math, ever** — the sovereign siblings own it; ifran drives
   their binaries as child processes. The Rust tree's fake primitives (the
   caller-hash "dedup", the contains-rate "perplexity") were explicitly not
   ported; the M3 dedup is real.
5. **CLI-first** (cmdit); the REST server is not carried; a bote-MCP surface
   opens when SecureYeoman re-wires off its HTTP proxy.

## Consequences

~1.9k lines of Cyrius replace the orchestration core; the port completed in
two days with per-milestone proofs. The full disposition map + milestone
record live in [`../development/port-ledger.md`](../development/port-ledger.md);
the Rust line (and its 10 Rust-era ADRs, relocated to `rust-old/docs/adr/`)
is held at `rust-old/` until ~2.1/2.2.
