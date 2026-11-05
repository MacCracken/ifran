# Dependency Watch

Pinned versions, upgrade paths, and gates for ifran's deps. (The Rust-era
watch — redis/tokio/axum — retired with the port; see `rust-old/`.)

## Pins (cyrius.cyml)

| Dep | Pin | Role | Watch |
|---|---|---|---|
| cyrius | 6.6.20 | toolchain | bump opportunistically per release discipline; check its CHANGELOG + `docs/development/ecosystem-migration.md` first |
| patra | 1.15.2 | run/model/dataset/pref tables | dialect gotchas recorded in CHANGELOG porting notes (0-based binds, no implicit rowid, type-keyword column names). Newest 6.6-line tag; 1.16.0+ pins 6.7.x |
| sigil | 3.13.10 | ed25519 (keys, tula verify) + sha256 (content addressing) | tula's unresolved-symbol supplier — include BEFORE tula. Newest 6.6-line tag; 3.13.11+ pins 6.7.x |
| tula | 1.0.3 | artifact format (validate/verify) | format v1 frozen; nothing to chase. Pins exactly 6.6.20, with the same sigil (3.13.10) as ifran |
| cmdit | 1.2.6 | CLI (verbs/help/version/exit codes) | API frozen at 1.0; append-only. Pins exactly 6.6.20; 1.3.0+ pins 6.7.x |

On 6.6.x the `path = "../<dep>"` lines override `tag`: every compiling verb
re-resolves first and copies the sibling's working tree, not the pinned tag,
and there is no switch to skip that. Re-vendor and verify at a pin move from a
staged copy outside `~/Repos` (look for `N commit-pinned (N re-verified)` in
the `cyrius deps` output). From cyrius 6.7.6 on, `path` only applies in local
mode (roadmap note).

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
