/// Binary entrypoint for the synapse-server.
use synapse_api::state::AppState;
use synapse_core::config::SynapseConfig;

#[tokio::main]
async fn main() {
    synapse_api::middleware::telemetry::init_tracing();

    let config = SynapseConfig::discover();

    // Enforce auth-required: refuse to start without an API key
    if config.security.auth_required {
        let has_key = std::env::var("SYNAPSE_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .is_some();
        if !has_key {
            eprintln!("ERROR: security.auth_required is true but SYNAPSE_API_KEY is not set.");
            eprintln!("Set SYNAPSE_API_KEY or set [security] auth_required = false in config.");
            std::process::exit(1);
        }
    }

    // Verify encrypted storage requirement if configured
    if config.security.require_encrypted_storage {
        synapse_core::storage::encryption::verify_encryption_requirement(
            &config.storage.models_dir,
            true,
        )
        .expect("Encrypted storage check failed");
    }

    let bind_addr = config.server.bind.clone();
    let bridge_enabled = config.bridge.enabled;
    let grpc_bind = config.server.grpc_bind.clone();

    let state = AppState::new(config).expect("Failed to initialize application state");

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
            let grpc_service = synapse_bridge::server::SynapseBridgeService::new(
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
                    let gpu_free = synapse_core::hardware::detect::detect()
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
                    // Report as progress to keep SY informed
                    if let Err(e) = client.report_progress("heartbeat", "alive", 0, 0.0).await {
                        tracing::warn!(error = %e, "Failed to send heartbeat to SY");
                    }
                }
            });
        }
    }

    // Self-register with fleet manager if fleet mode is enabled
    if state.config.fleet.enabled {
        let hardware = synapse_core::hardware::detect::detect().ok();
        let (gpu_count, gpu_mem) = hardware
            .as_ref()
            .map(|hw| (hw.gpus.len(), hw.total_gpu_memory_mb()))
            .unwrap_or((0, 0));

        let instance_id = std::env::var("SYNAPSE_INSTANCE_ID").unwrap_or_else(|_| {
            // Sanitize bind address to valid node id (alphanumeric + hyphens)
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
            .register(synapse_core::fleet::manager::RegisterNodeRequest {
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

    let app = synapse_api::router::build(state);

    tracing::info!("Starting synapse-server on {bind_addr}");

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .expect("Failed to bind address");

    axum::serve(listener, app).await.expect("Server error");
}
