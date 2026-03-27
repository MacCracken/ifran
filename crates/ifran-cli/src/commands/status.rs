/// Show status of running models, jobs, and hardware.
use crate::output;
use ifran_core::config::IfranConfig;
use ifran_core::hardware::detect;
use ifran_core::storage::db::ModelDatabase;
use ifran_types::error::Result;

pub async fn execute() -> Result<()> {
    let config = IfranConfig::discover();

    // Hardware
    output::header("Hardware");
    if let Ok(hw) = detect::detect() {
        output::kv("CPU", &hw.cpu.model_name);
        output::kv(
            "Cores",
            &format!(
                "{} physical, {} logical",
                hw.cpu.physical_cores, hw.cpu.logical_cores
            ),
        );
        output::kv(
            "Memory",
            &format!(
                "{} / {} MB available",
                output::format_size(hw.cpu.available_memory_mb * 1024 * 1024),
                output::format_size(hw.cpu.total_memory_mb * 1024 * 1024),
            ),
        );

        if hw.gpus.is_empty() {
            output::kv("GPU", &"none detected (CPU-only mode)");
        } else {
            for gpu in &hw.gpus {
                output::kv(
                    &format!("GPU {}", gpu.index),
                    &format!(
                        "{} [{:?}] — {} total, {} free",
                        gpu.name,
                        gpu.accelerator,
                        output::format_size(gpu.memory_total_mb * 1024 * 1024),
                        output::format_size(gpu.memory_free_mb * 1024 * 1024),
                    ),
                );
            }
        }
    } else {
        output::warn("Hardware detection failed");
    }

    eprintln!();

    // Models in catalog
    output::header("Models");
    let db = ModelDatabase::open(&config.storage.database)?;
    let tenant = ifran_types::TenantId::default_tenant();
    let paged = db.list(&tenant, 1000, 0)?;
    output::kv("Catalog", &format!("{} model(s)", paged.items.len()));

    eprintln!();
    output::header("Server");
    output::kv("API bind", &config.server.bind);
    output::kv("gRPC bind", &config.server.grpc_bind);

    Ok(())
}
