/// Pull a model from a remote registry to local storage.
use indicatif::{ProgressBar, ProgressStyle};
use ifran_core::config::IfranConfig;
use ifran_core::pull::downloader::{self, DownloadRequest};
use ifran_core::pull::progress::ProgressTracker;
use ifran_core::registry::huggingface::HfClient;
use ifran_core::storage::db::ModelDatabase;
use ifran_core::storage::layout::StorageLayout;
use ifran_types::error::Result;
use ifran_types::model::{ModelFormat, ModelInfo, QuantLevel};
use ifran_types::registry::DownloadState;

pub async fn execute(model: &str, quant: Option<&str>) -> Result<()> {
    let config = IfranConfig::discover();
    let layout = StorageLayout::new(
        config
            .storage
            .models_dir
            .parent()
            .unwrap_or(&config.storage.models_dir),
    );
    layout.ensure_dirs()?;

    let db = ModelDatabase::open(&config.storage.database)?;

    let tenant = ifran_types::TenantId::default_tenant();

    // Check if already pulled
    if let Ok(existing) = db.get_by_name(model, &tenant) {
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

    db.insert(&model_info, &tenant)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_quant_all_variants() {
        assert_eq!(parse_quant("f32"), QuantLevel::F32);
        assert_eq!(parse_quant("f16"), QuantLevel::F16);
        assert_eq!(parse_quant("bf16"), QuantLevel::Bf16);
        assert_eq!(parse_quant("q8_0"), QuantLevel::Q8_0);
        assert_eq!(parse_quant("q6k"), QuantLevel::Q6K);
        assert_eq!(parse_quant("q5_k_m"), QuantLevel::Q5KM);
        assert_eq!(parse_quant("q5_k_s"), QuantLevel::Q5KS);
        assert_eq!(parse_quant("q4_k_m"), QuantLevel::Q4KM);
        assert_eq!(parse_quant("q4_k_s"), QuantLevel::Q4KS);
        assert_eq!(parse_quant("q4_0"), QuantLevel::Q4_0);
        assert_eq!(parse_quant("q3_k_m"), QuantLevel::Q3KM);
        assert_eq!(parse_quant("q3_k_s"), QuantLevel::Q3KS);
        assert_eq!(parse_quant("q2k"), QuantLevel::Q2K);
        assert_eq!(parse_quant("iq4_xs"), QuantLevel::Iq4Xs);
        assert_eq!(parse_quant("iq3_xxs"), QuantLevel::Iq3Xxs);
    }

    #[test]
    fn parse_quant_case_insensitive() {
        assert_eq!(parse_quant("F16"), QuantLevel::F16);
        assert_eq!(parse_quant("Q4_K_M"), QuantLevel::Q4KM);
        assert_eq!(parse_quant("BF16"), QuantLevel::Bf16);
    }

    #[test]
    fn parse_quant_unknown_returns_none() {
        assert_eq!(parse_quant("unknown"), QuantLevel::None);
        assert_eq!(parse_quant(""), QuantLevel::None);
        assert_eq!(parse_quant("none"), QuantLevel::None);
    }

    #[test]
    fn format_bytes_gb() {
        assert_eq!(format_bytes(4_700_000_000), "4.7 GB");
        assert_eq!(format_bytes(1_000_000_000), "1.0 GB");
    }

    #[test]
    fn format_bytes_mb() {
        assert_eq!(format_bytes(500_000_000), "500.0 MB");
        assert_eq!(format_bytes(1_000_000), "1.0 MB");
    }

    #[test]
    fn format_bytes_kb() {
        assert_eq!(format_bytes(1_500), "1.5 KB");
        assert_eq!(format_bytes(1_000), "1.0 KB");
    }

    #[test]
    fn format_bytes_bytes() {
        assert_eq!(format_bytes(999), "999 B");
        assert_eq!(format_bytes(0), "0 B");
    }
}
