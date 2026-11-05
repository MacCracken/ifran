# Dependency Watch

Pinned versions, upgrade paths, and gates for ifran's deps. (The Rust-era
watch — redis/tokio/axum — retired with the port; see `rust-old/`.)

## Pins (cyrius.cyml)

| Dep | Pin | Role | Watch |
|---|---|---|---|
| cyrius | 6.7.9 | toolchain | bump opportunistically per release discipline; check its CHANGELOG + `docs/development/ecosystem-migration.md` first |
| patra | 1.16.0 | run/model/dataset/pref tables | dialect gotchas recorded in CHANGELOG porting notes (0-based binds, no implicit rowid, type-keyword column names). Pins 6.7.5 (the version the 6.7.9 toolchain bundles); 1.16.1+ pins 6.7.10 |
| sigil | 3.13.15 | ed25519 (keys, tula verify) + sha256 (content addressing) | tula's unresolved-symbol supplier — include BEFORE tula. Pins exactly 6.7.9; the toolchain bundles an older 3.13.11, hence the `lib/ shadows` build warning |
| tula | 1.1.0 | artifact format (validate/verify) | format v1 frozen; nothing to chase. Pins exactly 6.7.9, with the same sigil (3.13.15) as ifran; 1.1.1 pins 6.7.18 |
| cmdit | 1.4.0 | CLI (verbs/help/version/exit codes) | API frozen at 1.0; append-only. Pins exactly 6.7.9; 1.4.1+ pins 6.7.18 |

Git first (cyrius ≥ 6.7.6): every default `cyrius deps` / `build` resolves the
tag, so a pin move re-vendors in place (`cyrius lib sync --full`, then
`cyrius deps`) and `cyrius deps --locked` confirms `lib/` + `cyrius.lock`
match the tags. The `path = "../<dep>"` lines apply only in local mode
(`CYRIUS_LOCAL=1` / `--local`).

Stdlib set: see `[deps] stdlib` in `cyrius.cyml` (bayan for spec TOML +
JSONL, process/syscalls for the executor, chrono, hashmap for dedup, random
for keygen entropy).

## Sibling binaries (runtime, not build deps)

Jobs execute whatever `bin` names — the proofs used attn11 / tarka /
anukūlana. ifran has **no build dependency** on any sibling; a missing binary
is a recorded exit-127 run.

## Gates

- **agnos target**: blocked on process/patra agnos support (their arcs, not
  ours) — Linux-first by design for now.
- **bote** (Lane 4): becomes a dep only when the MCP interface opens.
