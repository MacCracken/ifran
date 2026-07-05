# ifran — Claude Code Instructions

> **Core rule**: this file is **preferences, process, and procedures** —
> durable rules that change rarely. Volatile state (current version,
> module line counts, port progress, test counts, consumers) lives in
> [`docs/development/state.md`](docs/development/state.md).
> Do not inline state here.

## Project Identity

**ifran** (عرفان — *gnosis*) — the AGNOS **training control plane**: job
manager/scheduler, tula+sigil checkpoint store, dataset curation, sweep runner,
eval runner, preference/annotation store — **thin over the sovereign ML
siblings** (attn11/tarka/tentib/prajna/anukūlana own ALL the math; ifran drives
their binaries as jobs). **The Cyrius port SHIPPED as 2.0.0 (2026-07-05)**;
the Rust line (53.6k lines) is preserved at `rust-old/` (reference-only,
original AGPL, held until ~2.1/2.2).

- **Type**: Binary (CLI on cmdit) + control-plane services; completed Rust→Cyrius port
- **License**: GPL-3.0-only (relicensed at the port cut per the ecosystem
  policy — GPL-3.0-only everywhere, AGPL reserved for desktop apps; the Rust
  line in `rust-old/` keeps its original AGPL)
- **Language**: Cyrius (toolchain pinned in `cyrius.cyml [package].cyrius`)
- **Version**: `VERSION` at the project root is the source of truth — do not inline the number here
- **Standards**: [First-Party Standards](https://github.com/MacCracken/agnosticos/blob/main/docs/development/applications/first-party-standards.md) · [First-Party Documentation](https://github.com/MacCracken/agnosticos/blob/main/docs/development/applications/first-party-documentation.md)

## Goal

Own **the orchestration around a training run — never the math**: submit/run/
track jobs that drive the sovereign ML siblings' binaries, store their
checkpoints (tula format, sigil-signed — real key management lands HERE),
curate datasets, sweep hyperparameters, run evals, hold preference data. Thin
over existing homes (lineage→itihas, serving→hoosh, marketplace→mela,
fleet→seema, GPU→mabda/ai-hwaccel); the Rust `backends/` broker is DEAD by
design (murti re-derivation) and does not port. Decomposition + disposition
map: `docs/development/port-ledger.md` + agnosticos `planning/ifran-port.md`.

## Current State

> Volatile state lives in [`docs/development/state.md`](docs/development/state.md) —
> port progress, surface parity, in-flight work. Refreshed every release.

This file (`CLAUDE.md`) is durable rules.

## The port (complete)

Scaffolded with `cyrius port`; shipped 2.0.0. The bar was **parity of
PURPOSE, not line parity** — most of the Rust tree deliberately did not port
(ADR 0001: dead broker, owned homes, fake primitives). `rust-old/` is
historical reference only — do not modify; removal ~2.1/2.2.

## Quick Start

```sh
cyrius deps                              # patra/sigil/tula/cmdit + stdlib
cyrius build src/main.cyr build/ifran
cyrius test tests/ifran.tcyr             # the suite — all against /tmp
```

Golden path + workspace model: `docs/guides/getting-started.md` +
`examples/`.

## Key Principles

- **Orchestration, never math** — anything that computes a gradient/loss/
  metric belongs in a sibling; ifran spawns, records, stores. If a change
  smells like training science, it's in the wrong repo.
- **Exit codes ARE the contract** — `run` = the child's, `eval` = the gate
  (frozen in `docs/api.md`); usage errors = 2.
- **The 2.x surface is FROZEN** (`docs/api.md` + `STABILITY.md`) — additions
  are additive minors; schema changes ship with in-place migrations (the M4
  `ALTER TABLE` precedent).
- **Delegate parsing** — tula/bayan/patra own their formats; ifran should
  parse no non-operator bytes itself (the audit's standing posture). Any new
  foreign-input parser follows the anukūlana rule: wrap-safe bounds +
  null-checked allocs + fuzz in the same cut.
- **Patra/bayan gotchas are recorded** — CHANGELOG "Porting notes" (0-based
  binds, no implicit rowid, type-keyword column names, toml_get returns Str,
  map_set is string-keyed, print is 2-arg). Read before touching the stores.
- Test after every change; ONE change at a time; build with `cyrius build`;
  `var buf[N]` local = N **bytes**.

## Rules (Hard Constraints)

- **Do not commit or push** — the user handles all git operations
- **Never use `gh` CLI** — use `curl` to the GitHub API if needed
- Do not modify `rust-old/` — historical reference (removal ~2.1/2.2 is its own confirmed step)
- Do not skip tests before claiming changes work
- Do not modify `lib/` files (vendored stdlib / dep symlinks)
- Do not hardcode toolchain versions in CI YAML — `cyrius = "X.Y.Z"` in `cyrius.cyml` is the source of truth

## Documentation

- [`docs/api.md`](docs/api.md) + [`STABILITY.md`](STABILITY.md) — the frozen 2.x surface
- [`docs/cli-reference.md`](docs/cli-reference.md) · [`docs/guides/getting-started.md`](docs/guides/getting-started.md) · [`examples/`](examples/)
- [`docs/adr/`](docs/adr/) — decisions (0001 = the port decomposition)
- [`docs/development/state.md`](docs/development/state.md) — live state
- [`docs/development/roadmap.md`](docs/development/roadmap.md) — the post-2.0 lanes
- [`docs/development/port-ledger.md`](docs/development/port-ledger.md) — the port record (historical)
- [`SECURITY.md`](SECURITY.md) · [`docs/audit/`](docs/audit/) · [`docs/benchmarks.md`](docs/benchmarks.md)

