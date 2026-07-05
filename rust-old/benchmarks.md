# Benchmarks

3-point trend tracking: baseline / previous / latest.

## 2026.3.28 — P(-1) Scaffold Hardening + Flat Crate Restructure

| Benchmark | Workspace Baseline | Post-Audit | Flat Crate | Key Change |
|-----------|--------------------|------------|------------|------------|
| cosine_similarity_768d | 1,718 ns | 1,738 ns | 1,725 ns | stable |
| cosine_similarity_1536d | 3,525 ns | 3,551 ns | 3,521 ns | stable |
| estimate_gguf_7b | 7 ns | 4 ns | 4 ns | `#[inline]` |
| estimate_gguf_70b | 5 ns | 4 ns | 4 ns | `#[inline]` |
| cache_insert_100 | 27,836 ns | 17,709 ns | 17,077 ns | **-39%** O(1) running total |
| cache_insert_with_eviction | 50,438 ns | 24,586 ns | 23,367 ns | **-54%** O(1) running total |
| cache_touch_hit | 41 ns | 42 ns | 40 ns | stable |
| score_exact_match_1000 | 10,080 ns | 10,672 ns | 9,227 ns | **-8%** |
| score_contains_match_1000 | 4,521 ns | 4,729 ns | 4,482 ns | stable |
| score_mmlu_500 | 70,407 ns | 73,349 ns | 81,240 ns | noise |

### Key wins

- **cache_insert_100**: 39% faster from O(1) `current_bytes` tracking (was O(n) recomputation per insert)
- **cache_insert_with_eviction**: 54% faster from same running-total optimization
- **estimate_gguf**: 43% faster from `#[inline]` annotations on hot-path functions
- **Flat crate restructure**: No performance regression — inlining opportunities may have slightly improved single-crate LTO

### Methodology

All benchmarks run via `./scripts/bench-history.sh` using Criterion 0.5. Results appended to `benchmarks/history.csv`. Numbers are median `ns/iter` from 100+ iterations.
