//! Concurrent operation tests — verify correctness under parallel access.

use ifran::audit;
use ifran::backends::health;
use ifran::fleet::manager::{FleetManager, RegisterNodeRequest};
use ifran::storage::cache::ModelCache;
use ifran::train::executor::ExecutorKind;
use ifran::train::job::manager::JobManager;
use ifran::types::TenantId;
use ifran::types::training::*;
use std::sync::Arc;

/// Test concurrent model cache insertions don't lose entries or corrupt state.
#[tokio::test]
async fn cache_concurrent_insert() {
    let cache = Arc::new(tokio::sync::Mutex::new(ModelCache::new(100_000_000)));
    let mut handles = Vec::new();

    for t in 0..10 {
        let cache = cache.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..100 {
                let mut c = cache.lock().await;
                c.insert(format!("t{t}-model-{i}"), 1000);
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let c = cache.lock().await;
    // All 1000 entries should be present (cache has room for all)
    assert_eq!(c.total_bytes(), 1000 * 1000);
}

/// Test concurrent fleet node registration doesn't lose nodes.
#[tokio::test]
async fn fleet_concurrent_registration() {
    let fm = Arc::new(FleetManager::with_defaults());
    let mut handles = Vec::new();

    for i in 0..20 {
        let fm = fm.clone();
        handles.push(tokio::spawn(async move {
            fm.register(RegisterNodeRequest {
                id: format!("node-{i}"),
                endpoint: format!("http://node-{i}:8420"),
                gpu_count: 2,
                total_gpu_memory_mb: 16384,
            })
            .await
            .unwrap();
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let stats = fm.stats().await;
    assert_eq!(stats.total_nodes, 20);
}

/// Test concurrent fleet heartbeats don't panic or corrupt state.
#[tokio::test]
async fn fleet_concurrent_heartbeats() {
    let fm = Arc::new(FleetManager::with_defaults());

    // Register nodes first
    for i in 0..5 {
        fm.register(RegisterNodeRequest {
            id: format!("node-{i}"),
            endpoint: format!("http://node-{i}:8420"),
            gpu_count: 1,
            total_gpu_memory_mb: 8192,
        })
        .await
        .unwrap();
    }

    // Concurrent heartbeats from all nodes
    let mut handles = Vec::new();
    for round in 0..10 {
        for node in 0..5 {
            let fm = fm.clone();
            handles.push(tokio::spawn(async move {
                fm.heartbeat(
                    &format!("node-{node}"),
                    Some(50.0 + round as f32),
                    Some(4096),
                    Some(65.0),
                )
                .await
                .unwrap();
            }));
        }
    }

    for h in handles {
        h.await.unwrap();
    }

    // All nodes should still be online
    let stats = fm.stats().await;
    assert_eq!(stats.online, 5);
}

/// Test concurrent job creation doesn't exceed max_concurrent limit.
#[tokio::test]
async fn job_manager_concurrent_submit() {
    let jm = Arc::new(JobManager::new(
        ExecutorKind::Subprocess,
        None,
        3, // max 3 concurrent
    ));

    let mut handles = Vec::new();
    for i in 0..10 {
        let jm = jm.clone();
        handles.push(tokio::spawn(async move {
            let config = TrainingJobConfig {
                base_model: format!("model-{i}"),
                dataset: DatasetConfig {
                    path: "/data/train.jsonl".into(),
                    format: DatasetFormat::Jsonl,
                    split: None,
                    max_samples: None,
                },
                method: TrainingMethod::Lora,
                hyperparams: HyperParams {
                    learning_rate: 2e-4,
                    epochs: 1,
                    batch_size: 4,
                    gradient_accumulation_steps: 1,
                    warmup_steps: 0,
                    weight_decay: 0.0,
                    max_seq_length: 512,
                },
                output_name: None,
                lora: None,
                max_steps: None,
                time_budget_secs: None,
            };
            let tenant = TenantId::default_tenant();
            let job_id = jm.create_job(config, tenant.clone()).await.unwrap();
            // start_job may succeed or fail (executor won't find python3)
            // but it should never panic
            let _ = jm.start_job(job_id, &tenant).await;
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    // Should not have panicked; max_concurrent is respected
    assert!(jm.running_count().await <= 3);
}

/// Test concurrent audit chain recording produces valid chain.
#[tokio::test]
async fn audit_chain_concurrent_recording() {
    let chain = Arc::new(audit::AuditChain::new(b"concurrent-test-key!!", 1000));
    let mut handles = Vec::new();

    for t in 0..10 {
        let chain = chain.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..20 {
                chain
                    .record(
                        &format!("actor-{t}"),
                        audit::AuditAction::AdminAction {
                            action: format!("concurrent-{t}-{i}"),
                            details: "test".into(),
                        },
                    )
                    .await;
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    // All 200 entries should be recorded
    assert_eq!(chain.len().await, 200);

    // Chain integrity must be valid despite concurrent writes
    assert!(
        chain.verify().await.is_none(),
        "Audit chain integrity violated under concurrent writes"
    );
}

/// Test concurrent backend health tracking doesn't corrupt ring buffers.
#[tokio::test]
async fn health_tracker_concurrent_recording() {
    let tracker = Arc::new(health::BackendHealthTracker::new(
        health::HealthConfig::default(),
    ));

    let mut handles = Vec::new();
    for t in 0..10 {
        let tracker = tracker.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..50 {
                tracker.record("backend-1", i % 3 != 0).await; // 67% success rate
                tracker.record(&format!("thread-backend-{t}"), true).await;
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    // backend-1 should be degraded or unhealthy (33% failure rate)
    let status = tracker.status("backend-1").await;
    assert_ne!(status, health::HealthStatus::Healthy);

    // Other backends should be healthy (100% success)
    for t in 0..10 {
        let status = tracker.status(&format!("thread-backend-{t}")).await;
        assert_eq!(status, health::HealthStatus::Healthy);
    }
}

/// Test concurrent cache insert + eviction doesn't corrupt running total.
#[tokio::test]
async fn cache_concurrent_insert_with_eviction() {
    let cache = Arc::new(tokio::sync::Mutex::new(ModelCache::new(50_000))); // small capacity
    let mut handles = Vec::new();

    for t in 0..5 {
        let cache = cache.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..100 {
                let mut c = cache.lock().await;
                c.insert(format!("t{t}-m{i}"), 1000);
            }
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let mut c = cache.lock().await;
    // Running total must match actual sum
    let actual: u64 = (0..500)
        .filter_map(|n| {
            let t = n / 100;
            let i = n % 100;
            if c.touch(&format!("t{t}-m{i}")) {
                Some(1000u64)
            } else {
                None
            }
        })
        .sum();
    assert_eq!(c.total_bytes(), actual);
}
