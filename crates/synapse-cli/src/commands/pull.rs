/// Pull a model from a remote registry to local storage.
use indicatif::{ProgressBar, ProgressStyle};
use synapse_core::config::SynapseConfig;
use synapse_core::pull::downloader::{self, DownloadRequest};
use synapse_core::pull::progress::ProgressTracker;
use synapse_core::registry::huggingface::HfClient;
use synapse_core::storage::db::ModelDatabase;
use synapse_core::storage::layout::StorageLayout;
use synapse_types::error::Result;
use synapse_types::model::{ModelFormat, ModelInfo, QuantLevel};
use synapse_types::registry::DownloadState;

pub async fn execute(model: &str, quant: Option<&str>) -> Result<()> {
    let config = SynapseConfig::discover();
    let layout = StorageLayout::new(
        config
            .storage
            .models_dir
            .parent()
            .unwrap_or(&config.storage.models_dir),
    );
    layout.ensure_dirs()?;

    let db = ModelDatabase::open(&config.storage.database)?;

    // Check if already pulled
    if let Ok(existing) = db.get_by_name(model) {
        eprintln!(
            "Model '{}' already exists (id: {})",
            existing.name, existing.id
        );
        eprintln!("  Path: {}", existing.local_path);
        return Ok(());
    }

    let http_client = downloader::build_client()?;
    let hf = HfClient::from_env(http_client.clone());

    eprintln!("Resolving {model}...");
    let gguf_file = hf.resolve_gguf(model, quant).await?;

    let quant_str = quant.unwrap_or("none");
    let slug = StorageLayout::slugify(model, quant_str);
    let dest = layout.model_file(&slug, &gguf_file.filename);

    eprintln!(
        "Downloading {} ({})",
        gguf_file.filename,
        format_bytes(gguf_file.file_size().unwrap_or(0)),
    );

    let progress = ProgressTracker::new(64);
    let mut rx = progress.subscribe();

    let download_url = HfClient::download_url(model, &gguf_file.filename);
    let request = DownloadRequest {
        url: download_url,
        dest: dest.clone(),
        model_name: model.to_string(),
        expected_sha256: gguf_file.sha256().map(|s| s.to_string()),
    };

    // Set up progress bar
    let pb = ProgressBar::new(gguf_file.file_size().unwrap_or(0));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    // Spawn progress bar updater
    let pb_clone = pb.clone();
    let progress_task = tokio::spawn(async move {
        while let Ok(event) = rx.recv().await {
            match event.state {
                DownloadState::Downloading => {
                    if let Some(total) = event.total_bytes {
                        pb_clone.set_length(total);
                    }
                    pb_clone.set_position(event.downloaded_bytes);
                }
                DownloadState::Verifying => {
                    pb_clone.set_message("Verifying...");
                }
                DownloadState::Complete => {
                    pb_clone.finish_with_message("done");
                    break;
                }
                DownloadState::Failed => {
                    pb_clone.abandon_with_message("failed");
                    break;
                }
                _ => {}
            }
        }
    });

    // Run the download
    downloader::download(&http_client, &request, &progress).await?;
    let _ = progress_task.await;

    // Get file size from disk
    let file_meta = std::fs::metadata(&dest)?;

    // Parse quant level
    let quant_level = parse_quant(quant_str);

    // Register in catalog
    let model_info = ModelInfo {
        id: uuid::Uuid::new_v4(),
        name: model.to_string(),
        repo_id: Some(model.to_string()),
        format: ModelFormat::Gguf,
        quant: quant_level,
        size_bytes: file_meta.len(),
        parameter_count: None,
        architecture: None,
        license: None,
        local_path: dest.to_string_lossy().to_string(),
        sha256: gguf_file.sha256().map(|s| s.to_string()),
        pulled_at: chrono::Utc::now(),
    };

    db.insert(&model_info)?;
    eprintln!("Model registered: {} ({})", model_info.name, model_info.id);

    Ok(())
}

fn parse_quant(s: &str) -> QuantLevel {
    match s.to_lowercase().replace('_', "").as_str() {
        "f32" => QuantLevel::F32,
        "f16" => QuantLevel::F16,
        "bf16" => QuantLevel::Bf16,
        "q80" => QuantLevel::Q8_0,
        "q6k" => QuantLevel::Q6K,
        "q5km" => QuantLevel::Q5KM,
        "q5ks" => QuantLevel::Q5KS,
        "q4km" => QuantLevel::Q4KM,
        "q4ks" => QuantLevel::Q4KS,
        "q40" => QuantLevel::Q4_0,
        "q3km" => QuantLevel::Q3KM,
        "q3ks" => QuantLevel::Q3KS,
        "q2k" => QuantLevel::Q2K,
        "iq4xs" => QuantLevel::Iq4Xs,
        "iq3xxs" => QuantLevel::Iq3Xxs,
        _ => QuantLevel::None,
    }
}

fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1_000_000_000;
    const MB: u64 = 1_000_000;
    const KB: u64 = 1_000;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}
