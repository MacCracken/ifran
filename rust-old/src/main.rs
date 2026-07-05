/// Binary entrypoint for the ifran-server.
use ifran::config::IfranConfig;
use ifran::server::state::AppState;

#[tokio::main]
async fn main() {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

    ifran::server::middleware::telemetry::init_tracing();

    let config = IfranConfig::discover();

    // Enforce auth-required: refuse to start without an API key
    if config.security.auth_required {
        let has_key = std::env::var("IFRAN_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .is_some();
        if !has_key {
            eprintln!("ERROR: security.auth_required is true but IFRAN_API_KEY is not set.");
            eprintln!("Set IFRAN_API_KEY or set [security] auth_required = false in config.");
            std::process::exit(1);
        }
    }

    // Verify encrypted storage requirement if configured
    if config.security.require_encrypted_storage {
        ifran::storage::encryption::verify_encryption_requirement(&config.storage.models_dir, true)
            .expect("Encrypted storage check failed");
    }

    let bind_addr = config.server.bind.clone();
    let bridge_enabled = config.bridge.enabled;
    let grpc_bind = config.server.grpc_bind.clone();

    let state = AppState::new(config).expect("Failed to initialize application state");

    // Start background eviction of terminal training jobs
    state
        .job_manager
        .start_eviction_loop(state.config.training.job_eviction_ttl_secs);

    // If bridge is enabled, connect to SY and start heartbeat
    if bridge_enabled {
        if let Some(client) = &state.bridge_client {
            let client = client.clone();
            tokio::spawn(async move {
                match client.connect().await {
                    Ok(()) => tracing::info!("Connected to SecureYeoman"),
                    Err(e) => tracing::warn!("Failed to connect to SY (will retry): {e}"),
                }
            });
        }

        if let Some(server) = &state.bridge_server {
            let server = server.clone();
            let addr = grpc_bind.clone();
            let grpc_service = ifran::bridge::server::IfranBridgeService::new(
                state.job_manager.clone(),
                state.backends.clone(),
                state.model_manager.clone(),
            );
            tokio::spawn(async move {
                match server.start(&addr, grpc_service).await {
                    Ok(()) => tracing::info!("Bridge server started on {addr}"),
                    Err(e) => tracing::warn!("Failed to start bridge server: {e}"),
                }
            });
        }

        // Spawn heartbeat task
        if let (Some(server), Some(client)) =
            (state.bridge_server.clone(), state.bridge_client.clone())
        {
            let model_manager = state.model_manager.clone();
            let job_manager = state.job_manager.clone();
            let interval = server.heartbeat_interval();
            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(interval);
                loop {
                    ticker.tick().await;
                    let loaded = model_manager.list_loaded(None).await.len() as u32;
                    let active = job_manager.running_count().await as u32;
                    let gpu_free = ifran::hardware::detect::detect()
                        .ok()
                        .map(|hw| hw.gpus.iter().map(|g| g.memory_free_mb).sum::<u64>())
                        .unwrap_or(0);
                    let hb = server.build_heartbeat(loaded, gpu_free, active);
                    tracing::debug!(
                        instance = %hb.instance_id,
                        models = hb.loaded_models,
                        jobs = hb.active_training_jobs,
                        "Heartbeat sent"
                    );
                    if let Err(e) = client.report_progress("heartbeat", "alive", 0, 0.0).await {
                        tracing::warn!(error = %e, "Failed to send heartbeat to SY");
                    }
                }
            });
        }
    }

    // Self-register with fleet manager if fleet mode is enabled
    if state.config.fleet.enabled {
        let hardware = ifran::hardware::detect::detect().ok();
        let (gpu_count, gpu_mem) = hardware
            .as_ref()
            .map(|hw| (hw.gpus.len(), hw.total_gpu_memory_mb()))
            .unwrap_or((0, 0));

        let instance_id = std::env::var("IFRAN_INSTANCE_ID").unwrap_or_else(|_| {
            state
                .config
                .server
                .bind
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' {
                        c
                    } else {
                        '-'
                    }
                })
                .collect()
        });

        let _ = state
            .fleet_manager
            .register(ifran::fleet::manager::RegisterNodeRequest {
                id: instance_id,
                endpoint: format!("http://{}", state.config.server.bind),
                gpu_count,
                total_gpu_memory_mb: gpu_mem,
            })
            .await;

        tracing::info!(
            gpus = gpu_count,
            gpu_memory_mb = gpu_mem,
            "Self-registered with fleet manager"
        );
    }

    let state_for_cleanup = state.clone();
    let app = ifran::server::router::build(state);

    tracing::info!("Starting ifran-server on {bind_addr}");

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .expect("Failed to bind address");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("Server error");

    // Cleanup
    if let Some(tl) = &state_for_cleanup.telemetry {
        tl.stop();
    }
    state_for_cleanup.fleet_manager.stop();
    tracing::info!("Shutdown complete");
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    tracing::info!("Shutdown signal received, draining connections...");
}
