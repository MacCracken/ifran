# Dependency Watch

Pinned versions, upgrade paths, and gates for ifran's deps. (The Rust-era
watch — redis/tokio/axum — retired with the port; see `rust-old/`.)

## Pins (cyrius.cyml)

| Dep | Pin | Role | Watch |
|---|---|---|---|
| cyrius | 6.4.3 | toolchain | bump opportunistically per release discipline; check its CHANGELOG first |
| patra | 1.12.8 | run/model/dataset/pref tables | dialect gotchas recorded in CHANGELOG porting notes (0-based binds, no implicit rowid, type-keyword column names) |
| sigil | 3.10.0 | ed25519 (keys, tula verify) + sha256 (content addressing) | tula's unresolved-symbol supplier — include BEFORE tula |
| tula | 1.0.0 | artifact format (validate/verify) | format v1 frozen; nothing to chase |
| cmdit | 1.1.0 | CLI (verbs/help/version/exit codes) | API frozen at 1.0; append-only |

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
