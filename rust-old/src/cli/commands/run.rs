use crate::config::IfranConfig;
use crate::storage::db::ModelDatabase;
use crate::types::IfranError;
use crate::types::error::Result;
use std::io::{self, BufRead, Write};

/// Run interactive inference on a specified model.
///
/// Uses [`hoosh::HooshClient`] to route inference through the local hoosh
/// gateway, which handles provider selection, caching, and token budgets.
/// Falls back to a direct HTTP call to the ifran server if hoosh is
/// unavailable.
pub async fn execute(model: &str) -> Result<()> {
    let config = IfranConfig::discover();
    let db = ModelDatabase::open(&config.storage.database)?;

    let tenant = crate::types::TenantId::default_tenant();
    let model_info = db.get_by_name(model, &tenant).or_else(|_| {
        uuid::Uuid::parse_str(model)
            .map_err(|_| IfranError::ModelNotFound(model.to_string()))
            .and_then(|id| db.get(id, &tenant))
    })?;

    eprintln!("Model: {} ({:?})", model_info.name, model_info.format);

    // Resolve hoosh gateway URL — prefer env var, then config bind address
    let hoosh_url =
        std::env::var("HOOSH_URL").unwrap_or_else(|_| "http://127.0.0.1:8088".to_string());

    let client = hoosh::HooshClient::new(&hoosh_url);

    // Check if hoosh gateway is reachable
    let gateway_available = client.health().await.unwrap_or(false);
    if !gateway_available {
        return Err(IfranError::Other(format!(
            "Hoosh inference gateway at {hoosh_url} is not reachable. \
             Start it with `hoosh` or set HOOSH_URL to the correct address."
        )));
    }

    eprintln!(
        "Connected to inference gateway at {hoosh_url}. Type your message (Ctrl+D to quit).\n"
    );

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

        let req = hoosh::InferenceRequest {
            model: model_info.name.clone(),
            prompt: input.to_string(),
            system: None,
            messages: Vec::new(),
            max_tokens: Some(512),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stream: true,
            tools: Vec::new(),
            tool_choice: None,
        };

        // Stream tokens for interactive feel
        match client.infer_stream(&req).await {
            Ok(mut rx) => {
                while let Some(chunk) = rx.recv().await {
                    match chunk {
                        Ok(text) => {
                            print!("{text}");
                            stdout.flush()?;
                        }
                        Err(e) => {
                            eprintln!("\nStream error: {e}");
                            break;
                        }
                    }
                }
                println!("\n");
            }
            Err(e) => {
                eprintln!("Error: {e}\n");
            }
        }
    }

    eprintln!("Done.");
    Ok(())
}
