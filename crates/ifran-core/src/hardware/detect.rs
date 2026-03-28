//! GPU/NPU detection.
//!
//! Detects available compute accelerators by probing for CUDA (NVML),
//! ROCm (sysfs), and CPU capabilities. Returns a unified [`SystemHardware`]
//! snapshot used for backend selection and VRAM budget checks.
//!
//! When the `ai-hwaccel` feature is enabled, detection is delegated to the
//! `ai-hwaccel` crate which provides more comprehensive hardware discovery
//! (13 backend families including Apple ANE, Intel NPU, and richer metadata).
//! The results are converted back into ifran's [`SystemHardware`] types so
//! the rest of the codebase is unaffected.

use ifran_types::error::Result;
#[cfg(not(feature = "ai-hwaccel"))]
use std::path::Path;

/// Re-export the full `ai-hwaccel` registry when the feature is enabled.
/// Callers that want the richer API (quantization suggestions, sharding plans,
/// accelerator profiles) can use this directly.
#[cfg(feature = "ai-hwaccel")]
pub use ai_hwaccel;

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
#[non_exhaustive]
pub enum AcceleratorKind {
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    Tpu,
    Gaudi,
    Inferentia,
    OneApi,
    QualcommAi,
    AmdXdna,
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
        let priority = [
            AcceleratorKind::Cuda,
            AcceleratorKind::Tpu,
            AcceleratorKind::Gaudi,
            AcceleratorKind::Rocm,
            AcceleratorKind::Inferentia,
            AcceleratorKind::OneApi,
            AcceleratorKind::Metal,
            AcceleratorKind::Vulkan,
            AcceleratorKind::QualcommAi,
            AcceleratorKind::AmdXdna,
        ];
        priority
            .into_iter()
            .find(|kind| self.gpus.iter().any(|g| g.accelerator == *kind))
    }
}

/// Detect all available hardware on this system.
///
/// When the `ai-hwaccel` feature is enabled, delegates to `ai_hwaccel::AcceleratorRegistry`
/// for richer, more comprehensive detection. The results are converted back into ifran's
/// own types so all downstream code continues to work unchanged.
pub fn detect() -> Result<SystemHardware> {
    #[cfg(feature = "ai-hwaccel")]
    {
        detect_via_hwaccel()
    }
    #[cfg(not(feature = "ai-hwaccel"))]
    {
        detect_builtin()
    }
}

/// Get the full `ai-hwaccel` [`AcceleratorRegistry`] for callers that want
/// the richer API (quantization suggestions, sharding plans, profiles, etc.).
///
/// Only available when the `ai-hwaccel` feature is enabled.
#[cfg(feature = "ai-hwaccel")]
pub fn detect_registry() -> ai_hwaccel::AcceleratorRegistry {
    ai_hwaccel::AcceleratorRegistry::detect()
}

/// Detection via `ai-hwaccel` crate — converts its rich types back to ifran's types.
#[cfg(feature = "ai-hwaccel")]
fn detect_via_hwaccel() -> Result<SystemHardware> {
    let registry = ai_hwaccel::AcceleratorRegistry::detect();

    // Log any detection warnings
    for w in registry.warnings() {
        tracing::warn!("ai-hwaccel detection warning: {w}");
    }

    // Extract CPU info from the CPU profile
    let cpu_profile = registry
        .available()
        .iter()
        .find(|p| matches!(p.accelerator, ai_hwaccel::AcceleratorType::Cpu))
        .copied();

    let cpu = if let Some(profile) = cpu_profile {
        // ai-hwaccel gives us memory; fill in CPU details from /proc as before
        let model_name = read_cpu_model().unwrap_or_else(|| "Unknown CPU".into());
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let physical_cores = read_physical_cores().unwrap_or(logical_cores);
        let total_memory_mb = profile.memory_bytes / (1024 * 1024);
        let available_memory_mb = read_memory_info()
            .map(|(_, avail)| avail)
            .unwrap_or(total_memory_mb);
        CpuInfo {
            model_name,
            physical_cores,
            logical_cores,
            total_memory_mb,
            available_memory_mb,
        }
    } else {
        detect_cpu()?
    };

    // Convert non-CPU profiles to GpuDevice
    let gpus: Vec<GpuDevice> = registry
        .available()
        .iter()
        .filter(|p| !matches!(p.accelerator, ai_hwaccel::AcceleratorType::Cpu))
        .enumerate()
        .map(|(i, p)| {
            let accelerator = hwaccel_to_kind(&p.accelerator);
            let memory_total_mb = p.memory_bytes / (1024 * 1024);
            let compute_capability = p
                .compute_capability
                .as_ref()
                .and_then(|s| parse_compute_capability(s));
            GpuDevice {
                index: i,
                name: p.accelerator.to_string(),
                accelerator,
                memory_total_mb,
                memory_free_mb: memory_total_mb, // ai-hwaccel reports total; free not tracked
                compute_capability,
            }
        })
        .collect();

    Ok(SystemHardware { cpu, gpus })
}

/// Map ai-hwaccel's AcceleratorType to ifran's AcceleratorKind.
#[cfg(feature = "ai-hwaccel")]
fn hwaccel_to_kind(accel: &ai_hwaccel::AcceleratorType) -> AcceleratorKind {
    use ai_hwaccel::AcceleratorType as AT;
    match accel {
        AT::CudaGpu { .. } => AcceleratorKind::Cuda,
        AT::RocmGpu { .. } => AcceleratorKind::Rocm,
        AT::MetalGpu => AcceleratorKind::Metal,
        AT::VulkanGpu { .. } => AcceleratorKind::Vulkan,
        AT::Tpu { .. } => AcceleratorKind::Tpu,
        AT::Gaudi { .. } => AcceleratorKind::Gaudi,
        AT::AwsNeuron { .. } => AcceleratorKind::Inferentia,
        AT::IntelOneApi { .. } => AcceleratorKind::OneApi,
        AT::QualcommAi100 { .. } => AcceleratorKind::QualcommAi,
        AT::AmdXdnaNpu { .. } => AcceleratorKind::AmdXdna,
        // ai-hwaccel knows about NPUs that ifran doesn't have a kind for yet;
        // map them to the closest match or fall through to Vulkan as a generic compute device.
        AT::IntelNpu | AT::AppleNpu => AcceleratorKind::Vulkan,
        AT::Cpu => AcceleratorKind::Cuda, // unreachable — CPU is filtered above
        _ => AcceleratorKind::Vulkan,     // future-proof for new ai-hwaccel variants
    }
}

/// Built-in detection (used when ai-hwaccel feature is not enabled).
#[cfg(not(feature = "ai-hwaccel"))]
fn detect_builtin() -> Result<SystemHardware> {
    let cpu = detect_cpu()?;
    let mut gpus = Vec::new();

    gpus.extend(detect_nvidia()?);
    gpus.extend(detect_rocm()?);
    gpus.extend(detect_tpu()?);
    gpus.extend(detect_metal()?);
    gpus.extend(detect_vulkan()?);
    gpus.extend(detect_gaudi()?);
    gpus.extend(detect_inferentia()?);
    gpus.extend(detect_oneapi()?);
    gpus.extend(detect_qualcomm()?);
    gpus.extend(detect_xdna()?);

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

fn parse_compute_capability(s: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() == 2 {
        Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Built-in per-backend detection (compiled out when ai-hwaccel handles it)
// ---------------------------------------------------------------------------
#[cfg(not(feature = "ai-hwaccel"))]
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

/// Detect AMD ROCm GPUs via sysfs.
#[cfg(not(feature = "ai-hwaccel"))]
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

/// Detect Google TPU devices via /dev/accel* (libtpu device nodes).
#[cfg(not(feature = "ai-hwaccel"))]
fn detect_tpu() -> Result<Vec<GpuDevice>> {
    let dev = Path::new("/dev");
    if !dev.exists() {
        return Ok(Vec::new());
    }

    let mut gpus = Vec::new();
    let mut index = 0usize;

    for entry in std::fs::read_dir(dev).into_iter().flatten().flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if !name_str.starts_with("accel") {
            continue;
        }

        // Verify it's a TPU by checking for the Google vendor in sysfs
        let accel_path = Path::new("/sys/class/accel").join(&*name_str);
        let device_dir = accel_path.join("device");

        let vendor = read_sysfs_string(&device_dir.join("vendor")).unwrap_or_default();
        // Google vendor ID is 0x1ae0
        if vendor != "0x1ae0" {
            continue;
        }

        let device_name = read_sysfs_string(&device_dir.join("product_name"))
            .unwrap_or_else(|| format!("Google TPU {index}"));

        // TPU HBM detection via sysfs (if available)
        let mem_total = read_sysfs_u64(&device_dir.join("mem_info_total"))
            .map(|b| b / (1024 * 1024))
            .unwrap_or(16384); // Default 16GB HBM per chip
        let mem_used = read_sysfs_u64(&device_dir.join("mem_info_used"))
            .map(|b| b / (1024 * 1024))
            .unwrap_or(0);

        gpus.push(GpuDevice {
            index,
            name: device_name,
            accelerator: AcceleratorKind::Tpu,
            memory_total_mb: mem_total,
            memory_free_mb: mem_total.saturating_sub(mem_used),
            compute_capability: None,
        });
        index += 1;
    }

    Ok(gpus)
}

/// Detect Apple Metal GPUs via system_profiler (macOS only).
#[cfg(not(feature = "ai-hwaccel"))]
fn detect_metal() -> Result<Vec<GpuDevice>> {
    let output = std::process::Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Ok(Vec::new()),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = match serde_json::from_str(&stdout) {
        Ok(v) => v,
        Err(_) => return Ok(Vec::new()),
    };

    let mut gpus = Vec::new();
    if let Some(displays) = json["SPDisplaysDataType"].as_array() {
        for (i, display) in displays.iter().enumerate() {
            let name = display["sppci_model"]
                .as_str()
                .unwrap_or("Apple GPU")
                .to_string();

            // VRAM is reported as a string like "16 GB"
            let vram_str = display["spdisplays_vram"]
                .as_str()
                .or_else(|| display["spdisplays_mtlgpufamilysupport"].as_str())
                .unwrap_or("0 MB");
            let mem_total = parse_vram_string(vram_str);

            gpus.push(GpuDevice {
                index: i,
                name,
                accelerator: AcceleratorKind::Metal,
                memory_total_mb: mem_total,
                memory_free_mb: mem_total, // macOS unified memory — report full
                compute_capability: None,
            });
        }
    }

    Ok(gpus)
}

#[cfg(not(feature = "ai-hwaccel"))]
fn parse_vram_string(s: &str) -> u64 {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() >= 2 {
        let value: u64 = parts[0].parse().unwrap_or(0);
        match parts[1].to_lowercase().as_str() {
            "gb" => value * 1024,
            "mb" => value,
            "tb" => value * 1024 * 1024,
            _ => value,
        }
    } else {
        0
    }
}

/// Detect Vulkan-capable GPUs via vulkaninfo.
#[cfg(not(feature = "ai-hwaccel"))]
fn detect_vulkan() -> Result<Vec<GpuDevice>> {
    let output = std::process::Command::new("vulkaninfo")
        .args(["--summary"])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Ok(Vec::new()),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();
    let mut index = 0usize;

    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("deviceName") {
            let name = trimmed
                .split('=')
                .nth(1)
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| format!("Vulkan GPU {index}"));

            gpus.push(GpuDevice {
                index,
                name,
                accelerator: AcceleratorKind::Vulkan,
                memory_total_mb: 0, // Vulkan doesn't easily report total VRAM from summary
                memory_free_mb: 0,
                compute_capability: None,
            });
            index += 1;
        }
    }

    Ok(gpus)
}

/// Detect Intel Gaudi (Habana Labs) accelerators via hl-smi.
#[cfg(not(feature = "ai-hwaccel"))]
fn detect_gaudi() -> Result<Vec<GpuDevice>> {
    let output = std::process::Command::new("hl-smi")
        .args([
            "--query-aip=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Ok(Vec::new()),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }

        let index: usize = parts[0].parse().unwrap_or(0);
        let name = parts[1].to_string();
        let mem_total: u64 = parts[2].parse().unwrap_or(0);
        let mem_free: u64 = parts[3].parse().unwrap_or(0);

        gpus.push(GpuDevice {
            index,
            name,
            accelerator: AcceleratorKind::Gaudi,
            memory_total_mb: mem_total,
            memory_free_mb: mem_free,
            compute_capability: None,
        });
    }

    Ok(gpus)
}

/// Detect AWS Inferentia/Trainium devices via neuron-ls.
#[cfg(not(feature = "ai-hwaccel"))]
fn detect_inferentia() -> Result<Vec<GpuDevice>> {
    let output = std::process::Command::new("neuron-ls")
        .args(["--json-output"])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Ok(Vec::new()),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = match serde_json::from_str(&stdout) {
        Ok(v) => v,
        Err(_) => return Ok(Vec::new()),
    };

    let mut gpus = Vec::new();

    if let Some(devices) = json.as_array() {
        for (i, device) in devices.iter().enumerate() {
            let model = device["model"].as_str().unwrap_or("Neuron Device");
            let nc_count = device["nc_count"].as_u64().unwrap_or(1);
            let memory_per_nc = device["memory_per_nc_mb"].as_u64().unwrap_or(8192);
            let mem_total = nc_count * memory_per_nc;

            gpus.push(GpuDevice {
                index: i,
                name: model.to_string(),
                accelerator: AcceleratorKind::Inferentia,
                memory_total_mb: mem_total,
                memory_free_mb: mem_total, // neuron-ls doesn't report used memory
                compute_capability: None,
            });
        }
    }

    Ok(gpus)
}

/// Detect Intel Arc / Data Center GPU Max via xpu-smi (oneAPI).
#[cfg(not(feature = "ai-hwaccel"))]
fn detect_oneapi() -> Result<Vec<GpuDevice>> {
    let output = std::process::Command::new("xpu-smi")
        .args(["discovery", "--dump", "1,2,18,19"])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Ok(Vec::new()),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        // Skip header line
        if line.starts_with("DeviceId") || line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }

        let index: usize = parts[0].parse().unwrap_or(0);
        let name = parts[1].to_string();
        let mem_total: u64 = parts[2].parse().unwrap_or(0);
        let mem_free: u64 = parts[3].parse().unwrap_or(0);

        gpus.push(GpuDevice {
            index,
            name,
            accelerator: AcceleratorKind::OneApi,
            memory_total_mb: mem_total,
            memory_free_mb: mem_free,
            compute_capability: None,
        });
    }

    Ok(gpus)
}

/// Detect Qualcomm Cloud AI 100 accelerators via /dev/qaic*.
#[cfg(not(feature = "ai-hwaccel"))]
fn detect_qualcomm() -> Result<Vec<GpuDevice>> {
    let dev = Path::new("/dev");
    if !dev.exists() {
        return Ok(Vec::new());
    }

    let mut gpus = Vec::new();
    let mut index = 0usize;

    for entry in std::fs::read_dir(dev).into_iter().flatten().flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if !name_str.starts_with("qaic") {
            continue;
        }

        // Try to get device info from sysfs
        let sysfs_path = Path::new("/sys/class/qaic").join(&*name_str).join("device");
        let device_name = read_sysfs_string(&sysfs_path.join("product_name"))
            .unwrap_or_else(|| format!("Qualcomm Cloud AI 100 #{index}"));

        let mem_total = read_sysfs_u64(&sysfs_path.join("mem_total"))
            .map(|b| b / (1024 * 1024))
            .unwrap_or(32768); // Default 32GB DDR

        gpus.push(GpuDevice {
            index,
            name: device_name,
            accelerator: AcceleratorKind::QualcommAi,
            memory_total_mb: mem_total,
            memory_free_mb: mem_total,
            compute_capability: None,
        });
        index += 1;
    }

    Ok(gpus)
}

/// Detect AMD XDNA (Ryzen AI) NPU via sysfs.
#[cfg(not(feature = "ai-hwaccel"))]
fn detect_xdna() -> Result<Vec<GpuDevice>> {
    let accel = Path::new("/sys/class/accel");
    if !accel.exists() {
        return Ok(Vec::new());
    }

    let mut gpus = Vec::new();
    let mut index = 0usize;

    for entry in std::fs::read_dir(accel).into_iter().flatten().flatten() {
        let device_dir = entry.path().join("device");
        let driver_link = device_dir.join("driver");

        let driver_name = std::fs::read_link(&driver_link)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));

        if driver_name.as_deref() != Some("amdxdna") {
            continue;
        }

        let device_name = read_sysfs_string(&device_dir.join("product_name"))
            .unwrap_or_else(|| format!("AMD Ryzen AI NPU {index}"));

        // XDNA NPUs typically have limited on-chip SRAM, report shared system memory
        let mem_total = read_sysfs_u64(&device_dir.join("mem_info_total"))
            .map(|b| b / (1024 * 1024))
            .unwrap_or(0);

        gpus.push(GpuDevice {
            index,
            name: device_name,
            accelerator: AcceleratorKind::AmdXdna,
            memory_total_mb: mem_total,
            memory_free_mb: mem_total,
            compute_capability: None,
        });
        index += 1;
    }

    Ok(gpus)
}

#[cfg(not(feature = "ai-hwaccel"))]
fn read_sysfs_string(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

#[cfg(not(feature = "ai-hwaccel"))]
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
