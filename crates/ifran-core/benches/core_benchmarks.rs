use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_cosine_similarity(c: &mut Criterion) {
    let a: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..768).map(|i| (i as f32 * 0.02).cos()).collect();

    c.bench_function("cosine_similarity_768d", |bencher| {
        bencher.iter(|| ifran_core::rag::store::cosine_similarity(black_box(&a), black_box(&b)));
    });

    let a_large: Vec<f32> = (0..1536).map(|i| (i as f32 * 0.01).sin()).collect();
    let b_large: Vec<f32> = (0..1536).map(|i| (i as f32 * 0.02).cos()).collect();

    c.bench_function("cosine_similarity_1536d", |bencher| {
        bencher.iter(|| {
            ifran_core::rag::store::cosine_similarity(black_box(&a_large), black_box(&b_large))
        });
    });
}

fn bench_memory_estimate(c: &mut Criterion) {
    c.bench_function("estimate_gguf_7b", |bencher| {
        bencher.iter(|| {
            ifran_core::lifecycle::memory::estimate_gguf(
                black_box(4_000_000_000),
                black_box(Some(32)),
                black_box(32),
            )
        });
    });

    c.bench_function("estimate_gguf_70b", |bencher| {
        bencher.iter(|| {
            ifran_core::lifecycle::memory::estimate_gguf(
                black_box(40_000_000_000),
                black_box(Some(60)),
                black_box(80),
            )
        });
    });
}

fn bench_cache_operations(c: &mut Criterion) {
    c.bench_function("cache_insert_100", |bencher| {
        bencher.iter(|| {
            let mut cache = ifran_core::storage::cache::ModelCache::new(1_000_000_000);
            for i in 0..100 {
                cache.insert(format!("model-{i}"), 1_000_000);
            }
            black_box(&cache);
        });
    });

    c.bench_function("cache_insert_with_eviction", |bencher| {
        bencher.iter(|| {
            let mut cache = ifran_core::storage::cache::ModelCache::new(50_000_000);
            for i in 0..100 {
                cache.insert(format!("model-{i}"), 1_000_000);
            }
            black_box(&cache);
        });
    });

    c.bench_function("cache_touch_hit", |bencher| {
        let mut cache = ifran_core::storage::cache::ModelCache::new(1_000_000_000);
        for i in 0..100 {
            cache.insert(format!("model-{i}"), 1_000_000);
        }
        bencher.iter(|| {
            black_box(cache.touch(black_box("model-50")));
        });
    });
}

fn bench_eval_scoring(c: &mut Criterion) {
    let predictions: Vec<(String, String)> = (0..1000)
        .map(|i| {
            if i % 3 == 0 {
                (format!("answer {i}"), format!("answer {i}"))
            } else {
                (format!("wrong {i}"), format!("answer {i}"))
            }
        })
        .collect();

    c.bench_function("score_exact_match_1000", |bencher| {
        bencher.iter(|| ifran_core::eval::benchmarks::score_exact_match(black_box(&predictions)));
    });

    c.bench_function("score_contains_match_1000", |bencher| {
        bencher
            .iter(|| ifran_core::eval::benchmarks::score_contains_match(black_box(&predictions)));
    });

    let mmlu_predictions: Vec<(String, String)> = (0..500)
        .map(|i| {
            let letter = match i % 4 {
                0 => "A",
                1 => "B",
                2 => "C",
                _ => "D",
            };
            (format!("The answer is {letter}"), letter.to_string())
        })
        .collect();

    c.bench_function("score_mmlu_500", |bencher| {
        bencher.iter(|| ifran_core::eval::benchmarks::score_mmlu(black_box(&mmlu_predictions)));
    });
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_memory_estimate,
    bench_cache_operations,
    bench_eval_scoring,
);
criterion_main!(benches);
