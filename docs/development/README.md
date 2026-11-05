# Development Guide

## Prerequisites

- The **cyrius** toolchain (pin in `cyrius.cyml`; install per the cyrius repo)
- Sibling repos beside ifran (`../patra`, `../sigil`, `../tula`,
  `../cmdit`) are optional. `cyrius deps` / `build` resolve each dep's git
  tag, the same as CI (cyrius >= 6.7.6). `CYRIUS_LOCAL=1` or `--local`
  builds the checkouts instead, from `build/local-deps/`, leaving `lib/` and
  `cyrius.lock` untouched.

## Loop

```sh
cyrius deps                               # after manifest changes
cyrius build src/main.cyr build/ifran
cyrius test tests/ifran.tcyr              # the suite (133) — runs against /tmp
cyrius lint <file>                        # src/ + tests/: keep 0 warns; lines <= 120
```

Definition of done per bite: suite green · lint/fmt clean · CHANGELOG
`[Unreleased]` entry · `state.md` refreshed. Version bumps are the
maintainer's (cut mechanics: `VERSION` + the `IFRAN_VERSION` literal in
`src/main.cyr`, both set by `./scripts/version-set.sh X.Y.Z`).
`./scripts/test.sh` runs the full gate (lock guard, suite, bench, fuzz).

## Where things live

- Decisions → [`../adr/`](../adr/) · port history →
  [`port-ledger.md`](port-ledger.md) · sequencing →
  [`roadmap.md`](roadmap.md) · live state → [`state.md`](state.md)
- The frozen surface → [`../api.md`](../api.md); breaking it = 3.0.0
  (see `STABILITY.md`)
- `rust-old/` is reference-only — never modify; removal ~2.1/2.2
