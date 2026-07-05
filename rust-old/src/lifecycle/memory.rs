//! VRAM/RAM budget estimation.
//!
//! Estimates memory requirements for loading a model and checks whether
//! sufficient GPU/system memory is available before loading.

use crate::hardware::detect::SystemHardware;
use crate::types::IfranError;
use crate::types::error::Result;

/// Estimated memory requirements for a model.
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Estimated GPU VRAM needed in MB.
    pub vram_mb: u64,
    /// Estimated system RAM needed in MB (for CPU layers / KV cache overflow).
    pub ram_mb: u64,
}

/// Estimate memory needed to load a GGUF model.
///
/// Rough heuristic: file size on disk is close to the VRAM needed,
/// plus ~20% overhead for KV cache and runtime buffers.
#[must_use]
#[inline]
pub fn estimate_gguf(
    file_size_bytes: u64,
    gpu_layers: Option<u32>,
    total_layers: u32,
) -> MemoryEstimate {
    let file_mb = file_size_bytes.saturating_add(1024 * 1024 - 1) / (1024 * 1024);
    let overhead = file_mb / 5; // ~20% for KV cache + buffers
    let total_mb = file_mb.saturating_add(overhead);

    let gpu_fraction = match gpu_layers {
        Some(layers) => {
            if total_layers == 0 {
                1.0
            } else {
                (layers as f64 / total_layers as f64).min(1.0)
            }
        }
        None => 1.0, // Default: offload everything to GPU
    };

    let vram_mb = (total_mb as f64 * gpu_fraction) as u64;
    let ram_mb = (total_mb as f64 * (1.0 - gpu_fraction)) as u64;

    MemoryEstimate { vram_mb, ram_mb }
}

/// Check whether the system has enough memory to load a model.
///
/// `reserve_mb` is the amount of GPU memory to keep free (from config).
pub fn check_budget(
    hardware: &SystemHardware,
    estimate: &MemoryEstimate,
    reserve_mb: u64,
) -> Result<()> {
    // Check GPU memory
    if estimate.vram_mb > 0 {
        if !hardware.has_gpu() {
            // No GPU — everything goes to RAM
            let total_ram_needed = estimate.vram_mb.saturating_add(estimate.ram_mb);
            if total_ram_needed > hardware.cpu.available_memory_mb {
                return Err(IfranError::InsufficientMemory {
                    required_mb: total_ram_needed,
                    available_mb: hardware.cpu.available_memory_mb,
                });
            }
        } else {
            let available_vram = hardware.free_gpu_memory_mb().saturating_sub(reserve_mb);
            if estimate.vram_mb > available_vram {
                return Err(IfranError::InsufficientMemory {
                    required_mb: estimate.vram_mb,
                    available_mb: available_vram,
                });
            }
        }
    }

    // Check system RAM
    if estimate.ram_mb > 0 && estimate.ram_mb > hardware.cpu.available_memory_mb {
        return Err(IfranError::InsufficientMemory {
            required_mb: estimate.ram_mb,
            available_mb: hardware.cpu.available_memory_mb,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::detect::{AcceleratorKind, CpuInfo, GpuDevice};

    fn mock_hardware(gpu_free_mb: u64, ram_available_mb: u64) -> SystemHardware {
        SystemHardware {
            cpu: CpuInfo {
                model_name: "Test CPU".into(),
                physical_cores: 4,
                logical_cores: 8,
                total_memory_mb: ram_available_mb * 2,
                available_memory_mb: ram_available_mb,
            },
            gpus: if gpu_free_mb > 0 {
                vec![GpuDevice {
                    index: 0,
                    name: "Test GPU".into(),
                    accelerator: AcceleratorKind::Cuda,
                    memory_total_mb: gpu_free_mb * 2,
                    memory_free_mb: gpu_free_mb,
                    compute_capability: Some((8, 9)),
                }]
            } else {
                vec![]
            },
        }
    }

    #[test]
    fn estimate_full_gpu() {
        let est = estimate_gguf(5_000_000_000, None, 32);
        assert!(est.vram_mb > 5000);
        assert_eq!(est.ram_mb, 0);
    }

    #[test]
    fn estimate_partial_offload() {
        let est = estimate_gguf(5_000_000_000, Some(16), 32);
        assert!(est.vram_mb > 0);
        assert!(est.ram_mb > 0);
    }

    #[test]
    fn budget_ok() {
        let hw = mock_hardware(16000, 32000);
        let est = MemoryEstimate {
            vram_mb: 6000,
            ram_mb: 0,
        };
        check_budget(&hw, &est, 512).unwrap();
    }

    #[test]
    fn budget_insufficient_vram() {
        let hw = mock_hardware(4000, 32000);
        let est = MemoryEstimate {
            vram_mb: 6000,
            ram_mb: 0,
        };
        let result = check_budget(&hw, &est, 512);
        assert!(matches!(result, Err(IfranError::InsufficientMemory { .. })));
    }

    #[test]
    fn budget_cpu_fallback() {
        let hw = mock_hardware(0, 32000);
        let est = MemoryEstimate {
            vram_mb: 6000,
            ram_mb: 0,
        };
        check_budget(&hw, &est, 512).unwrap(); // Falls back to RAM
    }

    #[test]
    fn budget_cpu_fallback_insufficient() {
        let hw = mock_hardware(0, 4000);
        let est = MemoryEstimate {
            vram_mb: 6000,
            ram_mb: 0,
        };
        let result = check_budget(&hw, &est, 512);
        assert!(matches!(result, Err(IfranError::InsufficientMemory { .. })));
    }
}
