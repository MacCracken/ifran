/// Run interactive inference on a specified model.
use std::io::{self, BufRead, Write};
use synapse_backends::InferenceBackend;
use synapse_backends::llamacpp::LlamaCppBackend;
use synapse_core::config::SynapseConfig;
use synapse_core::lifecycle::manager::ModelManager;
use synapse_core::storage::db::ModelDatabase;
use synapse_types::SynapseError;
use synapse_types::error::Result;
use synapse_types::inference::InferenceRequest;
use synapse_types::model::ModelManifest;

pub async fn execute(model: &str) -> Result<()> {
    let config = SynapseConfig::discover();
    let db = ModelDatabase::open(&config.storage.database)?;

    let tenant = synapse_types::TenantId::default_tenant();
    let model_info = db.get_by_name(model, &tenant).or_else(|_| {
        uuid::Uuid::parse_str(model)
            .map_err(|_| SynapseError::ModelNotFound(model.to_string()))
            .and_then(|id| db.get(id, &tenant))
    })?;

    let manifest = ModelManifest {
        info: model_info.clone(),
        context_length: Some(4096),
        gpu_layers: None,
        tensor_split: None,
    };

    eprintln!("Loading {}...", model_info.name);

    let manager = ModelManager::new(config.hardware.gpu_memory_reserve_mb);
    let device = manager.prepare_load(&manifest).await?;

    let backend = LlamaCppBackend::new(None);
    let handle = backend.load_model(&manifest, &device).await?;

    manager
        .register_loaded(
            model_info.id,
            model_info.name.clone(),
            handle.0.clone(),
            "llamacpp".into(),
            0,
            synapse_types::TenantId::default_tenant(),
        )
        .await;

    eprintln!("Model loaded. Type your message (Ctrl+D to quit).\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break; // EOF
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let req = InferenceRequest {
            prompt: input.to_string(),
            max_tokens: Some(512),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: None,
            stop_sequences: None,
            system_prompt: None,
            sensitivity: None,
        };

        // Use streaming for interactive feel
        match backend.infer_stream(&handle, req).await {
            Ok(mut rx) => {
                while let Some(chunk) = rx.recv().await {
                    if chunk.done {
                        break;
                    }
                    print!("{}", chunk.text);
                    stdout.flush()?;
                }
                println!("\n");
            }
            Err(e) => {
                eprintln!("Error: {e}\n");
            }
        }
    }

    eprintln!("Shutting down...");
    backend.unload_model(handle).await?;
    Ok(())
}
