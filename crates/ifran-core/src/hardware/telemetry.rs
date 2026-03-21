//! Periodic GPU health monitoring.
//!
//! Polls GPU utilization, temperature, and memory at a configurable interval,
//! storing the latest readings for API exposure and proactive OOM prevention.

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, watch};

/// A single GPU telemetry reading.
#[derive(Debug, Clone, Serialize)]
pub struct GpuTelemetry {
    pub device_index: u32,
    pub name: String,
    pub utilization_pct: f32,
    pub temperature_c: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub timestamp: DateTime<Utc>,
}

/// Configuration for the telemetry loop.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub interval: Duration,
    pub enabled: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(10),
            enabled: true,
        }
    }
}

/// Background loop that periodically polls GPU metrics.
pub struct TelemetryLoop {
    latest: Arc<RwLock<Vec<GpuTelemetry>>>,
    cancel_tx: watch::Sender<bool>,
}

impl TelemetryLoop {
    /// Start the telemetry polling loop.
    pub fn start(config: TelemetryConfig) -> Self {
        let latest = Arc::new(RwLock::new(Vec::new()));
        let (cancel_tx, cancel_rx) = watch::channel(false);

        if config.enabled {
            let latest_clone = latest.clone();
            let interval = config.interval;
            tokio::spawn(async move {
                Self::poll_loop(latest_clone, interval, cancel_rx).await;
            });
        }

        Self { latest, cancel_tx }
    }

    /// Get the most recent telemetry readings.
    pub async fn latest(&self) -> Vec<GpuTelemetry> {
        self.latest.read().await.clone()
    }

    /// Stop the polling loop.
    pub fn stop(&self) {
        let _ = self.cancel_tx.send(true);
    }

    async fn poll_loop(
        latest: Arc<RwLock<Vec<GpuTelemetry>>>,
        interval: Duration,
        mut cancel_rx: watch::Receiver<bool>,
    ) {
        loop {
            let readings = Self::probe_gpus().await;
            *latest.write().await = readings;

            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                _ = cancel_rx.changed() => break,
            }
        }
    }

    async fn probe_gpus() -> Vec<GpuTelemetry> {
        let mut results = Vec::new();

        // Try NVIDIA first
        if let Ok(nvidia) = probe_nvidia().await {
            results.extend(nvidia);
        }

        // Try ROCm
        if let Ok(rocm) = probe_rocm().await {
            results.extend(rocm);
        }

        results
    }
}

impl Drop for TelemetryLoop {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Probe NVIDIA GPUs via nvidia-smi for runtime telemetry.
async fn probe_nvidia() -> ifran_types::error::Result<Vec<GpuTelemetry>> {
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,utilization.gpu,temperature.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await;

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Ok(Vec::new()),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let now = Utc::now();
    let mut readings = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 6 {
            tracing::debug!(line, "Skipping malformed nvidia-smi telemetry line");
            continue;
        }

        readings.push(GpuTelemetry {
            device_index: parts[0].parse().unwrap_or(0),
            name: parts[1].to_string(),
            utilization_pct: parts[2].parse().unwrap_or(0.0),
            temperature_c: parts[3].parse().unwrap_or(0.0),
            memory_used_mb: parts[4].parse().unwrap_or(0),
            memory_total_mb: parts[5].parse().unwrap_or(0),
            timestamp: now,
        });
    }

    Ok(readings)
}

/// Probe AMD ROCm GPUs via sysfs for runtime telemetry.
async fn probe_rocm() -> ifran_types::error::Result<Vec<GpuTelemetry>> {
    use std::path::Path;

    let drm = Path::new("/sys/class/drm");
    if !drm.exists() {
        return Ok(Vec::new());
    }

    let now = Utc::now();
    let mut readings = Vec::new();
    let mut index = 0u32;

    let entries = match std::fs::read_dir(drm) {
        Ok(e) => e,
        Err(_) => return Ok(Vec::new()),
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }

        let device_dir = entry.path().join("device");
        let driver_link = device_dir.join("driver");
        let driver_name = std::fs::read_link(&driver_link)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));

        if driver_name.as_deref() != Some("amdgpu") {
            continue;
        }

        let gpu_name = read_sysfs_string(&device_dir.join("product_name"))
            .unwrap_or_else(|| format!("AMD GPU {index}"));

        let utilization_pct = read_sysfs_string(&device_dir.join("gpu_busy_percent"))
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);

        // hwmon temperature (millidegrees C)
        let temperature_c = find_hwmon_temp(&device_dir).unwrap_or(0.0);

        let mem_total = read_sysfs_u64(&device_dir.join("mem_info_vram_total"))
            .map(|b| b / (1024 * 1024))
            .unwrap_or(0);
        let mem_used = read_sysfs_u64(&device_dir.join("mem_info_vram_used"))
            .map(|b| b / (1024 * 1024))
            .unwrap_or(0);

        readings.push(GpuTelemetry {
            device_index: index,
            name: gpu_name,
            utilization_pct,
            temperature_c,
            memory_used_mb: mem_used,
            memory_total_mb: mem_total,
            timestamp: now,
        });
        index += 1;
    }

    Ok(readings)
}

fn read_sysfs_string(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

fn read_sysfs_u64(path: &std::path::Path) -> Option<u64> {
    read_sysfs_string(path)?.parse().ok()
}

fn find_hwmon_temp(device_dir: &std::path::Path) -> Option<f32> {
    let hwmon_dir = device_dir.join("hwmon");
    let entries = std::fs::read_dir(&hwmon_dir).ok()?;
    for entry in entries.flatten() {
        let temp_path = entry.path().join("temp1_input");
        if let Some(millidegrees) = read_sysfs_u64(&temp_path) {
            return Some(millidegrees as f32 / 1000.0);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_config_default() {
        let config = TelemetryConfig::default();
        assert!(config.enabled);
        assert_eq!(config.interval, Duration::from_secs(10));
    }

    #[test]
    fn gpu_telemetry_serializes() {
        let t = GpuTelemetry {
            device_index: 0,
            name: "RTX 4090".into(),
            utilization_pct: 85.0,
            temperature_c: 72.0,
            memory_used_mb: 20000,
            memory_total_mb: 24576,
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("RTX 4090"));
        assert!(json.contains("85"));
    }

    #[tokio::test]
    async fn telemetry_loop_start_stop() {
        let config = TelemetryConfig {
            interval: Duration::from_millis(50),
            enabled: true,
        };
        let tl = TelemetryLoop::start(config);
        // Give it a moment to run
        tokio::time::sleep(Duration::from_millis(100)).await;
        let readings = tl.latest().await;
        // May be empty if no GPU, but should not panic
        let _ = readings;
        tl.stop();
    }

    #[tokio::test]
    async fn telemetry_loop_disabled() {
        let config = TelemetryConfig {
            interval: Duration::from_millis(50),
            enabled: false,
        };
        let tl = TelemetryLoop::start(config);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let readings = tl.latest().await;
        assert!(readings.is_empty());
    }
}
