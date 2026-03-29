/// Show status of running models, jobs, and hardware.
use crate::cli::output;
use crate::config::IfranConfig;
use crate::hardware::detect;
use crate::storage::db::ModelDatabase;
use crate::types::error::Result;

/// Format a cores display string from physical and logical core counts.
#[must_use]
fn format_cores(physical: usize, logical: usize) -> String {
    format!("{physical} physical, {logical} logical")
}

/// Format a memory display string from available and total sizes (in MB).
#[must_use]
fn format_memory(available_mb: u64, total_mb: u64) -> String {
    format!(
        "{} / {} available",
        output::format_size(available_mb * 1024 * 1024),
        output::format_size(total_mb * 1024 * 1024),
    )
}

/// Format a GPU info display string.
#[must_use]
fn format_gpu(name: &str, accelerator: &str, total_mb: u64, free_mb: u64) -> String {
    format!(
        "{} [{}] — {} total, {} free",
        name,
        accelerator,
        output::format_size(total_mb * 1024 * 1024),
        output::format_size(free_mb * 1024 * 1024),
    )
}

/// Format a model count display string.
#[must_use]
fn format_catalog_count(count: usize) -> String {
    format!("{count} model(s)")
}

pub async fn execute() -> Result<()> {
    let config = IfranConfig::discover();

    // Hardware
    output::header("Hardware");
    if let Ok(hw) = detect::detect() {
        output::kv("CPU", &hw.cpu.model_name);
        output::kv(
            "Cores",
            &format_cores(hw.cpu.physical_cores, hw.cpu.logical_cores),
        );
        output::kv(
            "Memory",
            &format_memory(hw.cpu.available_memory_mb, hw.cpu.total_memory_mb),
        );

        if hw.gpus.is_empty() {
            output::kv("GPU", &"none detected (CPU-only mode)");
        } else {
            for gpu in &hw.gpus {
                output::kv(
                    &format!("GPU {}", gpu.index),
                    &format_gpu(
                        &gpu.name,
                        &format!("{:?}", gpu.accelerator),
                        gpu.memory_total_mb,
                        gpu.memory_free_mb,
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
    let tenant = crate::types::TenantId::default_tenant();
    let paged = db.list(&tenant, 1000, 0)?;
    output::kv("Catalog", &format_catalog_count(paged.items.len()));

    eprintln!();
    output::header("Server");
    output::kv("API bind", &config.server.bind);
    output::kv("gRPC bind", &config.server.grpc_bind);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_cores_display() {
        assert_eq!(format_cores(8, 16), "8 physical, 16 logical");
        assert_eq!(format_cores(1, 1), "1 physical, 1 logical");
        assert_eq!(format_cores(0, 0), "0 physical, 0 logical");
    }

    #[test]
    fn format_memory_display() {
        // 8192 MB available, 16384 MB total
        let result = format_memory(8192, 16384);
        assert!(result.contains("available"));
        assert!(result.contains("8.6 GB")); // 8192 * 1024 * 1024 = 8589934592
        assert!(result.contains("17.2 GB")); // 16384 * 1024 * 1024 = 17179869184
    }

    #[test]
    fn format_memory_small() {
        let result = format_memory(512, 1024);
        assert!(result.contains("available"));
    }

    #[test]
    fn format_gpu_display() {
        let result = format_gpu("RTX 4090", "Cuda", 24576, 20000);
        assert!(result.contains("RTX 4090"));
        assert!(result.contains("[Cuda]"));
        assert!(result.contains("total"));
        assert!(result.contains("free"));
    }

    #[test]
    fn format_gpu_zero_memory() {
        let result = format_gpu("Test GPU", "Vulkan", 0, 0);
        assert!(result.contains("Test GPU"));
        assert!(result.contains("[Vulkan]"));
    }

    #[test]
    fn format_catalog_count_display() {
        assert_eq!(format_catalog_count(0), "0 model(s)");
        assert_eq!(format_catalog_count(1), "1 model(s)");
        assert_eq!(format_catalog_count(42), "42 model(s)");
    }
}
