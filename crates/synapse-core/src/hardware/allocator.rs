//! Device memory allocator for GPU scheduling.
//!
//! Manages GPU device assignment across concurrent inference and training jobs.
//! Tracks per-device memory usage and supports fair allocation when multiple
//! jobs compete for the same devices.

use std::collections::HashMap;
use synapse_types::SynapseError;
use synapse_types::error::Result;
use tokio::sync::RwLock;

/// Unique identifier for an allocation.
pub type AllocationId = uuid::Uuid;

/// A GPU device allocation for a job.
#[derive(Debug, Clone)]
pub struct Allocation {
    pub id: AllocationId,
    pub device_ids: Vec<u32>,
    pub memory_mb: u64,
    pub memory_per_device_mb: u64,
    pub job_label: String,
}

/// Per-device tracking state.
#[derive(Debug, Clone)]
struct DeviceState {
    pub index: u32,
    pub total_memory_mb: u64,
    pub allocated_mb: u64,
}

impl DeviceState {
    fn available_mb(&self) -> u64 {
        self.total_memory_mb.saturating_sub(self.allocated_mb)
    }
}

/// GPU device allocator with fair scheduling.
pub struct DeviceAllocator {
    devices: RwLock<Vec<DeviceState>>,
    allocations: RwLock<HashMap<AllocationId, Allocation>>,
    reserve_mb: u64,
}

impl DeviceAllocator {
    /// Create a new allocator from detected hardware.
    ///
    /// `reserve_mb` is kept free on each device for system use.
    pub fn new(reserve_mb: u64) -> Self {
        Self {
            devices: RwLock::new(Vec::new()),
            allocations: RwLock::new(HashMap::new()),
            reserve_mb,
        }
    }

    /// Initialize device inventory from hardware detection.
    pub async fn init_from_hardware(&self, gpus: &[crate::hardware::detect::GpuDevice]) {
        let mut devices = self.devices.write().await;
        devices.clear();
        for gpu in gpus {
            devices.push(DeviceState {
                index: gpu.index as u32,
                total_memory_mb: gpu.memory_total_mb,
                allocated_mb: 0,
            });
        }
    }

    /// Allocate devices for a job.
    ///
    /// Selects the `count` devices with the most available memory that
    /// can each satisfy the per-device memory requirement.
    pub async fn allocate(
        &self,
        memory_per_device_mb: u64,
        count: u32,
        job_label: &str,
    ) -> Result<Allocation> {
        let mut devices = self.devices.write().await;
        let allocations_guard = &mut *self.allocations.write().await;

        // Find eligible devices (enough free memory after reserve)
        let required = memory_per_device_mb + self.reserve_mb;
        let mut candidates: Vec<(usize, u64)> = devices
            .iter()
            .enumerate()
            .filter(|(_, d)| d.available_mb() >= required)
            .map(|(i, d)| (i, d.available_mb()))
            .collect();

        // Sort by most available memory (fair: prefer least-loaded devices)
        candidates.sort_by(|a, b| b.1.cmp(&a.1));

        if (candidates.len() as u32) < count {
            return Err(SynapseError::HardwareError(format!(
                "Not enough devices: need {count} with {memory_per_device_mb}MB free, \
                 only {} eligible",
                candidates.len()
            )));
        }

        let selected: Vec<usize> = candidates
            .iter()
            .take(count as usize)
            .map(|(i, _)| *i)
            .collect();
        let device_ids: Vec<u32> = selected.iter().map(|&i| devices[i].index).collect();

        // Commit allocation
        for &i in &selected {
            devices[i].allocated_mb += memory_per_device_mb;
        }

        let alloc = Allocation {
            id: uuid::Uuid::new_v4(),
            device_ids,
            memory_mb: memory_per_device_mb * count as u64,
            memory_per_device_mb,
            job_label: job_label.to_string(),
        };

        allocations_guard.insert(alloc.id, alloc.clone());
        Ok(alloc)
    }

    /// Release a previously allocated set of devices.
    pub async fn deallocate(&self, allocation_id: AllocationId) -> Result<()> {
        let mut allocations = self.allocations.write().await;
        let alloc = allocations.remove(&allocation_id).ok_or_else(|| {
            SynapseError::HardwareError(format!("Allocation {allocation_id} not found"))
        })?;

        let per_device_mb = alloc.memory_per_device_mb;

        let mut devices = self.devices.write().await;
        for &dev_id in &alloc.device_ids {
            if let Some(device) = devices.iter_mut().find(|d| d.index == dev_id) {
                device.allocated_mb = device.allocated_mb.saturating_sub(per_device_mb);
            }
        }

        Ok(())
    }

    /// List current allocations.
    pub async fn list_allocations(&self) -> Vec<Allocation> {
        self.allocations.read().await.values().cloned().collect()
    }

    /// Get available memory per device (after allocations and reserve).
    pub async fn available_memory(&self) -> Vec<(u32, u64)> {
        self.devices
            .read()
            .await
            .iter()
            .map(|d| (d.index, d.available_mb().saturating_sub(self.reserve_mb)))
            .collect()
    }

    /// Total number of tracked devices.
    pub async fn device_count(&self) -> usize {
        self.devices.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::detect::{AcceleratorKind, GpuDevice};

    fn mock_gpus(count: usize, memory_mb: u64) -> Vec<GpuDevice> {
        (0..count)
            .map(|i| GpuDevice {
                index: i,
                name: format!("GPU {i}"),
                accelerator: AcceleratorKind::Cuda,
                memory_total_mb: memory_mb,
                memory_free_mb: memory_mb,
                compute_capability: Some((8, 0)),
            })
            .collect()
    }

    #[tokio::test]
    async fn init_from_hardware() {
        let alloc = DeviceAllocator::new(512);
        alloc.init_from_hardware(&mock_gpus(2, 8192)).await;
        assert_eq!(alloc.device_count().await, 2);
    }

    #[tokio::test]
    async fn allocate_single_device() {
        let alloc = DeviceAllocator::new(512);
        alloc.init_from_hardware(&mock_gpus(2, 8192)).await;

        let a = alloc.allocate(4000, 1, "inference").await.unwrap();
        assert_eq!(a.device_ids.len(), 1);
        assert_eq!(a.memory_mb, 4000);
    }

    #[tokio::test]
    async fn allocate_multiple_devices() {
        let alloc = DeviceAllocator::new(512);
        alloc.init_from_hardware(&mock_gpus(4, 16384)).await;

        let a = alloc.allocate(8000, 2, "training").await.unwrap();
        assert_eq!(a.device_ids.len(), 2);
    }

    #[tokio::test]
    async fn allocate_fails_insufficient_memory() {
        let alloc = DeviceAllocator::new(512);
        alloc.init_from_hardware(&mock_gpus(1, 4096)).await;

        let result = alloc.allocate(4000, 1, "big-model").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn allocate_fails_insufficient_devices() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(1, 8192)).await;

        let result = alloc.allocate(1000, 3, "distributed").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn deallocate_frees_memory() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(1, 8192)).await;

        let a = alloc.allocate(4000, 1, "job-1").await.unwrap();
        assert_eq!(alloc.available_memory().await[0].1, 4192);

        alloc.deallocate(a.id).await.unwrap();
        assert_eq!(alloc.available_memory().await[0].1, 8192);
    }

    #[tokio::test]
    async fn deallocate_not_found() {
        let alloc = DeviceAllocator::new(0);
        let result = alloc.deallocate(uuid::Uuid::new_v4()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn fair_scheduling_prefers_least_loaded() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(2, 8192)).await;

        // First allocation gets device with most free memory (either one)
        let a1 = alloc.allocate(4000, 1, "job-1").await.unwrap();
        // Second allocation should pick the OTHER device (more free memory)
        let a2 = alloc.allocate(4000, 1, "job-2").await.unwrap();
        assert_ne!(a1.device_ids[0], a2.device_ids[0]);
    }

    #[tokio::test]
    async fn list_allocations() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(2, 8192)).await;

        assert!(alloc.list_allocations().await.is_empty());
        alloc.allocate(1000, 1, "job-1").await.unwrap();
        alloc.allocate(1000, 1, "job-2").await.unwrap();
        assert_eq!(alloc.list_allocations().await.len(), 2);
    }

    #[tokio::test]
    async fn no_devices_allocation_fails() {
        let alloc = DeviceAllocator::new(0);
        let result = alloc.allocate(1000, 1, "job").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn reserve_reduces_available() {
        let alloc = DeviceAllocator::new(1000);
        alloc.init_from_hardware(&mock_gpus(1, 8192)).await;
        let avail = alloc.available_memory().await;
        assert_eq!(avail[0].1, 7192); // 8192 - 1000 reserve
    }
}
