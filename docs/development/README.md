# Development Guide

## Prerequisites

- The **cyrius** toolchain (pin in `cyrius.cyml`; install per the cyrius repo)
- Sibling repos checked out beside ifran for local dep resolution
  (`../patra`, `../sigil`, `../tula`, `../cmdit` — `cyrius deps` uses
  `path=` first, git+tag in CI)

## Loop

```sh
cyrius deps                               # after manifest changes
cyrius build src/main.cyr build/ifran
cyrius test tests/ifran.tcyr              # the suite (77) — runs against /tmp
cyrius lint src/<file>.cyr                # keep 0 warns; lines <= 120
```

Definition of done per bite: suite green · lint/fmt clean · CHANGELOG
`[Unreleased]` entry · `state.md` refreshed. Version bumps are the
maintainer's (cut mechanics: `VERSION` + the `IFRAN_VERSION` literal in
`src/main.cyr`).

## Where things live

- Decisions → [`../adr/`](../adr/) · port history →
  [`port-ledger.md`](port-ledger.md) · sequencing →
  [`roadmap.md`](roadmap.md) · live state → [`state.md`](state.md)
- The frozen surface → [`../api.md`](../api.md); breaking it = 3.0.0
  (see `STABILITY.md`)
- `rust-old/` is reference-only — never modify; removal ~2.1/2.2
