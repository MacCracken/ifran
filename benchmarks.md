# Benchmarks

3-point trend tracking: baseline / previous / latest.

## P(-1) Scaffold Hardening — 2026-03-28

| Benchmark | Baseline | Post-Audit | Delta |
|-----------|----------|------------|-------|
| cosine_similarity_768d | 1,718 ns | 1,738 ns | +1.2% (noise) |
| cosine_similarity_1536d | 3,525 ns | 3,551 ns | +0.7% (noise) |
| estimate_gguf_7b | 7 ns | 4 ns | **-43%** |
| estimate_gguf_70b | 5 ns | 4 ns | **-20%** |
| cache_insert_100 | 27,836 ns | 17,709 ns | **-36.4%** |
| cache_insert_with_eviction | 50,438 ns | 24,586 ns | **-51.3%** |
| cache_touch_hit | 41 ns | 42 ns | +2.4% (noise) |
| score_exact_match_1000 | 10,080 ns | 10,672 ns | +5.9% (noise) |
| score_contains_match_1000 | 4,521 ns | 4,729 ns | +4.6% (noise) |
| score_mmlu_500 | 70,407 ns | 73,349 ns | +4.2% (noise) |

### Key wins

- **cache_insert_100**: 36% faster from O(1) `current_bytes` tracking (was O(n) recomputation)
- **cache_insert_with_eviction**: 51% faster from same running-total optimization
- **estimate_gguf**: 20-43% faster from `#[inline]` annotations on hot-path functions

### Methodology

All benchmarks run via `./scripts/bench-history.sh` using Criterion 0.5. Results appended to `benchmarks/history.csv`.
