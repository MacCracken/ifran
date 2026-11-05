# Contributing to ifran

Thank you for your interest in contributing to ifran. This document covers how
the project is laid out, how to build and test it, and what a change needs
before it can merge.

## Project Overview

ifran (عرفان) is the AGNOS **training control plane**: a Cyrius CLI that runs
and records training jobs, keeps a signed checkpoint store, curates datasets,
runs sweeps and evals, and holds preference data. It orchestrates; the
sovereign ML siblings (attn11, tarka, tentib, prajna, anukūlana) own the math.
Start with the [README](README.md), the frozen 2.x surface in
[`docs/api.md`](docs/api.md), and
[ADR 0001](docs/adr/0001-port-decomposition.md).

| Path | Purpose |
|---|---|
| `src/main.cyr` | CLI entry: verb dispatch on cmdit |
| `src/jobspec.cyr` | Job spec parsing (bayan TOML) |
| `src/run.cyr`, `src/runstore.cyr` | Executor (fork/exec, capture, timeout) + run records (patra) |
| `src/keys.cyr`, `src/store.cyr` | Operator keys + the signed model store (sigil, tula) |
| `src/dataset.cyr` | Datasets: ingest, dedup, validate |
| `src/sweep.cyr` | Sweeps: grid/random expansion, budgets, leaderboard |
| `src/eval.cyr` | Eval runner + the benchmark store |
| `src/pref.cyr` | Preference store + JSONL export |
| `tests/ifran.{tcyr,bcyr,fcyr}` | Test suite, bench harness, fuzz harness |
| `lib/` | Vendored stdlib + deps, written by `cyrius deps` — never edit by hand |

## Prerequisites

- **cyrius**, at the version pinned in `cyrius.cyml` (`[package] cyrius`).
  `./scripts/setup-dev.sh` checks it and prints the install command if needed.
- **Linux** — the executor and patra are Linux-first.
- Sibling checkouts (`../patra`, `../sigil`, `../tula`, `../cmdit`) are
  optional: builds resolve the pinned git tags, and `CYRIUS_LOCAL=1` (or
  `--local`) builds against the checkouts instead.

## Getting Started

```bash
git clone https://github.com/MacCracken/ifran.git
cd ifran

# Toolchain matches the pin; lib/ + cyrius.lock match the pinned tags
./scripts/setup-dev.sh

# cyrius deps + cyrius build -> build/ifran
./scripts/build.sh

# The full gate: lock guard, suite, bench, fuzz
./scripts/test.sh
```

The golden path (dataset → job → sweep → eval → store → preferences) is in
[`docs/guides/getting-started.md`](docs/guides/getting-started.md), with runnable
specs in [`examples/`](examples/).

## Git Workflow

- **main** is always releasable; tags are semver releases.
- Branch from `main` with a prefix:

| Prefix | Use |
|---|---|
| `feature/` | New features or enhancements |
| `bugfix/` | Bug fixes |
| `docs/` | Documentation-only changes |
| `refactor/` | Code restructuring without behavior change |
| `security/` | Security patches or hardening |
| `chore/` | Build, CI, tooling, dependency updates |

### Commit Messages

[Conventional Commits](https://www.conventionalcommits.org/) are preferred:

```
<type>(<scope>): <description>
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

**Scopes:** `cli`, `run`, `store`, `keys`, `dataset`, `sweep`, `eval`, `pref`, `deps`, `ci`

```
fix(sweep): carry the template's timeout_s into every combo
chore(deps): move to cyrius 6.7.18
```

## Pull Request Process

1. Branch from `main` using the naming convention above.
2. Run `./scripts/test.sh` locally; it must pass.
3. Open a PR against `main`.
4. CI (`.github/workflows/ci.yml`) must pass: lock guard, build, tests.
5. At least one approving review is required.

## Coding Standards

- **Format:** cyrfmt-canonical. `cyrius fmt --check <file>` reports;
  `cyrius fmt <file>` rewrites in place.
- **Lint:** `cyrius lint <file>` with 0 warnings, for `src/` and `tests/`;
  lines ≤ 120.
- **Orchestration, never math.** Anything that computes a gradient, loss or
  metric belongs in a sibling; ifran spawns, records and stores.
- **Delegate parsing** to tula/bayan/patra. A new parser of foreign input ships
  wrap-safe bounds, null-checked allocations and a fuzz harness in the same
  change.
- **Read the CHANGELOG "Porting notes"** before touching the stores (patra and
  bayan gotchas: 0-based binds, no implicit rowid, the TOML getters return a
  `Str`, …).
- **The 2.x surface is frozen** (`docs/api.md` + `STABILITY.md`). Additions
  are additive minors; schema changes ship with in-place migrations; exit codes
  are the contract (`run` = the child's, `eval` = the gate, usage errors = 2).

Optional Claude Code hooks: `scripts/hook-quality.sh` (PostToolUse on
Write|Edit: format-check and lint) and
`scripts/hook-release-gate.sh` (PreToolUse on Bash: blocks `git commit` until
the gate passes). Wire them in `.claude/settings.json` to use them.

## Testing

- `cyrius test tests/ifran.tcyr` — the suite; everything runs against `/tmp`.
- `cyrius bench tests/ifran.bcyr` — the bench harness;
  `./scripts/bench-history.sh [label]` appends a run to `benchmarks/history.csv`.
- `cyrius fuzz tests/ifran.fcyr` — the fuzz harness.
- `assert` and `bench` are `[test] stdlib` leaves: they reach only test, bench
  and fuzz compiles. Each test file includes what it uses (`lib/assert.cyr`,
  `lib/bench.cyr`) so it also compiles outside the test scope, e.g. in an
  editor's language server.
- Test after every change. A bug fix lands with a test that fails without it.

## Dependencies

Pins live in `cyrius.cyml`; what to watch at each one is in
[`docs/development/dependency-watch.md`](docs/development/dependency-watch.md).
To move a pin: `cyrius lib sync --full`, then `cyrius deps`, then
`cyrius deps --locked` to confirm `lib/` and `cyrius.lock` are exactly what the
tags resolve. Commit `lib/` and `cyrius.lock` with the pin.

## Documentation

- Update `docs/` for any user-facing change, and add an ADR in `docs/adr/` for
  a significant design decision.
- Every change gets a CHANGELOG entry; `docs/development/state.md` is refreshed
  each release.

## Security

- **Never** commit secrets, credentials or operator signing keys.
- See [SECURITY.md](SECURITY.md) for the security policy and how to report a
  vulnerability.

## Release Process

ifran uses **Semantic Versioning**: `MAJOR.MINOR.PATCH`, or
`MAJOR.MINOR.PATCH-pre` for pre-releases.

1. `./scripts/version-set.sh X.Y.Z` sets `VERSION` and the `src/main.cyr`
   version literal (the `release-automation` workflow runs the same script,
   then commits and tags).
2. Write the CHANGELOG entry and refresh `docs/development/state.md`.
3. Pushing the tag runs `.github/workflows/release.yml`: the CI gate, a
   version-consistency check, the build, a source tarball + `SHA256SUMS`, and
   the GitHub release.

## License

ifran is GPL-3.0-only; contributions are accepted under the same license.

---

Thank you for contributing to ifran.
