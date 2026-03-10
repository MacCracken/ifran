//! GPU/NPU detection.
//!
//! Detects available compute accelerators by probing for CUDA (NVML),
//! ROCm (sysfs), and CPU capabilities. Returns a unified [`SystemHardware`]
//! snapshot used for backend selection and VRAM budget checks.

use std::path::Path;
use synapse_types::error::Result;

/// A detected compute device.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub index: usize,
    pub name: String,
    pub accelerator: AcceleratorKind,
    pub memory_total_mb: u64,
    pub memory_free_mb: u64,
    pub compute_capability: Option<(u32, u32)>,
}

/// Kind of accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceleratorKind {
    Cuda,
    Rocm,
}

/// CPU information.
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub model_name: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub total_memory_mb: u64,
    pub available_memory_mb: u64,
}

/// Snapshot of detected system hardware.
#[derive(Debug, Clone)]
pub struct SystemHardware {
    pub cpu: CpuInfo,
    pub gpus: Vec<GpuDevice>,
}

impl SystemHardware {
    /// True if any GPU is available.
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }

    /// Total GPU memory across all devices.
    pub fn total_gpu_memory_mb(&self) -> u64 {
        self.gpus.iter().map(|g| g.memory_total_mb).sum()
    }

    /// Total free GPU memory across all devices.
    pub fn free_gpu_memory_mb(&self) -> u64 {
        self.gpus.iter().map(|g| g.memory_free_mb).sum()
    }

    /// Best available accelerator kind, or None for CPU-only.
    pub fn best_accelerator(&self) -> Option<AcceleratorKind> {
        // Prefer CUDA over ROCm
        if self
            .gpus
            .iter()
            .any(|g| g.accelerator == AcceleratorKind::Cuda)
        {
            Some(AcceleratorKind::Cuda)
        } else if self
            .gpus
            .iter()
            .any(|g| g.accelerator == AcceleratorKind::Rocm)
        {
            Some(AcceleratorKind::Rocm)
        } else {
            None
        }
    }
}

/// Detect all available hardware on this system.
pub fn detect() -> Result<SystemHardware> {
    let cpu = detect_cpu()?;
    let mut gpus = Vec::new();

    gpus.extend(detect_nvidia()?);
    gpus.extend(detect_rocm()?);

    Ok(SystemHardware { cpu, gpus })
}

/// Detect CPU info from /proc.
fn detect_cpu() -> Result<CpuInfo> {
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let model_name = read_cpu_model().unwrap_or_else(|| "Unknown CPU".into());
    let physical_cores = read_physical_cores().unwrap_or(logical_cores);
    let (total_memory_mb, available_memory_mb) = read_memory_info().unwrap_or((0, 0));

    Ok(CpuInfo {
        model_name,
        physical_cores,
        logical_cores,
        total_memory_mb,
        available_memory_mb,
    })
}

/// Read CPU model name from /proc/cpuinfo.
fn read_cpu_model() -> Option<String> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in content.lines() {
        if line.starts_with("model name") {
            return line.split(':').nth(1).map(|s| s.trim().to_string());
        }
    }
    None
}

/// Read physical core count from /proc/cpuinfo.
fn read_physical_cores() -> Option<usize> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut core_ids = std::collections::HashSet::new();
    for line in content.lines() {
        if line.starts_with("core id") {
            if let Some(id) = line.split(':').nth(1) {
                core_ids.insert(id.trim().to_string());
            }
        }
    }
    if core_ids.is_empty() {
        None
    } else {
        Some(core_ids.len())
    }
}

/// Read total and available memory from /proc/meminfo.
fn read_memory_info() -> Option<(u64, u64)> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = 0u64;
    let mut available_kb = 0u64;

    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = parse_meminfo_value(line)?;
        } else if line.starts_with("MemAvailable:") {
            available_kb = parse_meminfo_value(line)?;
        }
    }
    Some((total_kb / 1024, available_kb / 1024))
}

fn parse_meminfo_value(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse().ok()
}

/// Detect NVIDIA GPUs via nvidia-smi (shells out for portability without
/// linking NVML at compile time).
fn detect_nvidia() -> Result<Vec<GpuDevice>> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Ok(Vec::new()), // nvidia-smi not available
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 5 {
            continue;
        }

        let index: usize = parts[0].parse().unwrap_or(0);
        let name = parts[1].to_string();
        let mem_total: u64 = parts[2].parse().unwrap_or(0);
        let mem_free: u64 = parts[3].parse().unwrap_or(0);
        let compute_cap = parse_compute_capability(parts[4]);

        gpus.push(GpuDevice {
            index,
            name,
            accelerator: AcceleratorKind::Cuda,
            memory_total_mb: mem_total,
            memory_free_mb: mem_free,
            compute_capability: compute_cap,
        });
    }

    Ok(gpus)
}

fn parse_compute_capability(s: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() == 2 {
        Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
    } else {
        None
    }
}

/// Detect AMD ROCm GPUs via sysfs.
fn detect_rocm() -> Result<Vec<GpuDevice>> {
    let drm = Path::new("/sys/class/drm");
    if !drm.exists() {
        return Ok(Vec::new());
    }

    let mut gpus = Vec::new();
    let mut index = 0usize;

    // Look for renderD128, renderD129, etc. which are AMD GPU render nodes
    for entry in std::fs::read_dir(drm).into_iter().flatten().flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }

        let device_dir = entry.path().join("device");

        // Check if it's an AMD GPU by looking for the amdgpu driver
        let driver_link = device_dir.join("driver");
        let driver_name = std::fs::read_link(&driver_link)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));

        if driver_name.as_deref() != Some("amdgpu") {
            continue;
        }

        let gpu_name = read_sysfs_string(&device_dir.join("product_name"))
            .or_else(|| read_sysfs_string(&device_dir.join("pp_features")))
            .unwrap_or_else(|| format!("AMD GPU {index}"));

        let mem_total = read_sysfs_u64(&device_dir.join("mem_info_vram_total"))
            .map(|b| b / (1024 * 1024))
            .unwrap_or(0);
        let mem_used = read_sysfs_u64(&device_dir.join("mem_info_vram_used"))
            .map(|b| b / (1024 * 1024))
            .unwrap_or(0);

        gpus.push(GpuDevice {
            index,
            name: gpu_name,
            accelerator: AcceleratorKind::Rocm,
            memory_total_mb: mem_total,
            memory_free_mb: mem_total.saturating_sub(mem_used),
            compute_capability: None,
        });
        index += 1;
    }

    Ok(gpus)
}

fn read_sysfs_string(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

fn read_sysfs_u64(path: &Path) -> Option<u64> {
    read_sysfs_string(path)?.parse().ok()
}

impl std::fmt::Display for SystemHardware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "CPU: {} ({} cores, {} threads, {} MB RAM)",
            self.cpu.model_name,
            self.cpu.physical_cores,
            self.cpu.logical_cores,
            self.cpu.total_memory_mb,
        )?;
        if self.gpus.is_empty() {
            writeln!(f, "GPU: none detected (CPU-only mode)")?;
        } else {
            for gpu in &self.gpus {
                writeln!(
                    f,
                    "GPU {}: {} [{:?}] — {} MB total, {} MB free{}",
                    gpu.index,
                    gpu.name,
                    gpu.accelerator,
                    gpu.memory_total_mb,
                    gpu.memory_free_mb,
                    gpu.compute_capability
                        .map(|(maj, min)| format!(", compute {maj}.{min}"))
                        .unwrap_or_default(),
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_valid_cpu() {
        let hw = detect().unwrap();
        assert!(hw.cpu.logical_cores > 0);
    }

    #[test]
    fn no_gpu_is_handled() {
        // This test just verifies the code doesn't panic when no GPU is present
        let hw = detect().unwrap();
        // has_gpu may be true or false depending on the machine
        let _ = hw.has_gpu();
        let _ = hw.total_gpu_memory_mb();
    }

    #[test]
    fn parse_compute_cap() {
        assert_eq!(parse_compute_capability("8.9"), Some((8, 9)));
        assert_eq!(parse_compute_capability("bad"), None);
        assert_eq!(parse_compute_capability(""), None);
    }

    #[test]
    fn display_impl() {
        let hw = detect().unwrap();
        let output = format!("{hw}");
        assert!(output.contains("CPU:"));
    }
}
