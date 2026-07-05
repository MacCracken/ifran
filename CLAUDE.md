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
their binaries as jobs). Cyrius port of 53.6k lines of Rust (preserved at
`rust-old/`, reference-only, eventual dismissal).

- **Type**: Port (Rust → Cyrius); binary + control-plane services
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

## Scaffolding

Project was scaffolded with `cyrius port`. Original Rust at `rust-old/` is the reference oracle — do not modify it; cross-check the port against it.

## Quick Start

```sh
cyrius deps                              # resolve dependencies
cyrius build src/main.cyr build/ifran    # compile
cyrius test                              # run tests/*.tcyr
```

## Key Principles

- **Cross-check against `rust-old/`** — the port's correctness bar is "matches what Rust did". Diverge only with an ADR.
- **Correctness over cleverness** — if the Cyrius behavior diverges silently from Rust, the bugs win
- Test after every change, not after the feature is "done"
- ONE change at a time — never bundle unrelated changes
- Build with `cyrius build`, not raw `cat file | cc5` — the manifest auto-resolves deps
- Source files only need project includes — stdlib auto-resolves from `cyrius.cyml`
- `var buf[N]` = N **bytes**, not N entries

## Rules (Hard Constraints)

- **Do not commit or push** — the user handles all git operations
- **Never use `gh` CLI** — use `curl` to the GitHub API if needed
- Do not modify `rust-old/` — it's the parity oracle
- Do not skip tests before claiming changes work
- Do not modify `lib/` files (vendored stdlib / dep symlinks)
- Do not hardcode toolchain versions in CI YAML — `cyrius = "X.Y.Z"` in `cyrius.cyml` is the source of truth

## Documentation

- [`docs/adr/`](docs/adr/) — Architecture Decision Records (*why X over Y?*)
- [`docs/architecture/`](docs/architecture/) — Non-obvious constraints
- [`docs/guides/`](docs/guides/) — Task-oriented how-tos
- [`docs/examples/`](docs/examples/) — Runnable examples
- [`docs/development/state.md`](docs/development/state.md) — Live state
- [`docs/development/roadmap.md`](docs/development/roadmap.md) — Milestones through v1.0

