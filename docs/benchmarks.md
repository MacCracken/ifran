# ifran — Benchmarks

> 2.0 capture (2026-07-05). Box: AMD Ryzen 7 5800H (Zen 3), x86_64 Linux,
> cycc 6.4.3. A control plane's honest number is its OVERHEAD — the work it
> adds around a job — plus the end-to-end wall of the real workflows it drove.

## Orchestration overhead

20 full `ifran run` round-trips of `/bin/echo` (spec parse → insert-as-running
→ fork/pipe/execve → capture → log write → record update) vs 20 bare
`/bin/echo` invocations:

| | total | per run |
|---|---|---|
| `ifran run` | 27 ms | **~1.35 ms** |
| bare `/bin/echo` | 15 ms | ~0.75 ms |
| **overhead** | | **~0.6 ms/job** |

Negligible against real workloads (the cheapest real job below is ~7 s).

## Real workflows driven (measured during the M-proofs)

| Workflow | Wall | What ifran added |
|---|---|---|
| attn11 train (`--preset --steps 30`, 51 KB corpus) | 31.7 s | spawn/capture/record + `{dataset}` resolution |
| attn11 3-combo steps-sweep (10/20/40) | 75.2 s | grid expansion + 3 tagged runs (durations scaled with steps: 10.8/21.5/42.9 s) |
| anukūlana HF-fidelity oracle as an eval | 7.2 s | gate + `maxrel=0.000001049` extracted into the benchmark store |
| anukūlana `gpt2-tula` (full QLoRA round-trip) | 80.0 s | run record + the two artifacts then ingested |
| `store add` of the 63.8 MB NF4 checkpoint | ~1 s | tula validate + Ed25519 verify + sha256 + copy |
| dataset dedup (51 KB, 567 lines) | <0.1 s | 567 line-hashes + rewrite |
| tarka full gate suite as a job | 4.6 s | spawn/capture/record |

Suite: 77 checks in ~4 s (`cyrius test tests/ifran.tcyr`).

## Honest framing

ifran adds bookkeeping, not compute — the siblings own the math and their own
performance stories. Re-capture on toolchain/hardware change.
