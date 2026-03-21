//! Device memory allocator for GPU scheduling.
//!
//! Manages GPU device assignment across concurrent inference and training jobs.
//! Tracks per-device memory usage and supports fair allocation when multiple
//! jobs compete for the same devices.

use std::collections::HashMap;
use std::sync::Arc;
use ifran_types::IfranError;
use ifran_types::error::Result;
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
    index: u32,
    total_memory_mb: u64,
    allocated_mb: u64,
    compute_capability: Option<(u32, u32)>,
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
    event_bus: Option<Arc<super::events::GpuEventBus>>,
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
            event_bus: None,
        }
    }

    /// Create a new allocator with an event bus for GPU lifecycle notifications.
    pub fn with_event_bus(reserve_mb: u64, event_bus: Arc<super::events::GpuEventBus>) -> Self {
        Self {
            devices: RwLock::new(Vec::new()),
            allocations: RwLock::new(HashMap::new()),
            reserve_mb,
            event_bus: Some(event_bus),
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
                compute_capability: gpu.compute_capability,
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
        min_compute_capability: Option<(u32, u32)>,
    ) -> Result<Allocation> {
        let mut devices = self.devices.write().await;
        let allocations_guard = &mut *self.allocations.write().await;

        // Find eligible devices (enough free memory after reserve)
        let required = memory_per_device_mb + self.reserve_mb;
        let mut candidates: Vec<(usize, u64)> = devices
            .iter()
            .enumerate()
            .filter(|(_, d)| {
                d.available_mb() >= required
                    && min_compute_capability
                        .is_none_or(|min| d.compute_capability.is_some_and(|cc| cc >= min))
            })
            .map(|(i, d)| (i, d.available_mb()))
            .collect();

        // Sort by most available memory (fair: prefer least-loaded devices)
        candidates.sort_by(|a, b| b.1.cmp(&a.1));

        if (candidates.len() as u32) < count {
            return Err(IfranError::HardwareError(format!(
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

        let total_memory = memory_per_device_mb
            .checked_mul(count as u64)
            .ok_or_else(|| IfranError::HardwareError("Memory calculation overflow".into()))?;

        let alloc = Allocation {
            id: uuid::Uuid::new_v4(),
            device_ids,
            memory_mb: total_memory,
            memory_per_device_mb,
            job_label: job_label.to_string(),
        };

        allocations_guard.insert(alloc.id, alloc.clone());

        if let Some(ref bus) = self.event_bus {
            bus.emit_allocated(
                alloc.id,
                alloc.device_ids.clone(),
                alloc.memory_mb,
                job_label,
            );
        }

        Ok(alloc)
    }

    /// Release a previously allocated set of devices.
    pub async fn deallocate(&self, allocation_id: AllocationId) -> Result<()> {
        // Acquire locks in the same order as allocate(): devices first, then allocations.
        let mut devices = self.devices.write().await;
        let mut allocations = self.allocations.write().await;

        let alloc = allocations.remove(&allocation_id).ok_or_else(|| {
            IfranError::HardwareError(format!("Allocation {allocation_id} not found"))
        })?;

        let per_device_mb = alloc.memory_per_device_mb;
        let device_ids = alloc.device_ids.clone();

        for &dev_id in &device_ids {
            if let Some(device) = devices.iter_mut().find(|d| d.index == dev_id) {
                device.allocated_mb = device.allocated_mb.saturating_sub(per_device_mb);
            }
        }

        // Drop locks before emitting events to avoid holding them unnecessarily.
        drop(allocations);
        drop(devices);

        if let Some(ref bus) = self.event_bus {
            bus.emit_released(allocation_id, device_ids);
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

        let a = alloc.allocate(4000, 1, "inference", None).await.unwrap();
        assert_eq!(a.device_ids.len(), 1);
        assert_eq!(a.memory_mb, 4000);
    }

    #[tokio::test]
    async fn allocate_multiple_devices() {
        let alloc = DeviceAllocator::new(512);
        alloc.init_from_hardware(&mock_gpus(4, 16384)).await;

        let a = alloc.allocate(8000, 2, "training", None).await.unwrap();
        assert_eq!(a.device_ids.len(), 2);
    }

    #[tokio::test]
    async fn allocate_fails_insufficient_memory() {
        let alloc = DeviceAllocator::new(512);
        alloc.init_from_hardware(&mock_gpus(1, 4096)).await;

        let result = alloc.allocate(4000, 1, "big-model", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn allocate_fails_insufficient_devices() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(1, 8192)).await;

        let result = alloc.allocate(1000, 3, "distributed", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn deallocate_frees_memory() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(1, 8192)).await;

        let a = alloc.allocate(4000, 1, "job-1", None).await.unwrap();
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
        let a1 = alloc.allocate(4000, 1, "job-1", None).await.unwrap();
        // Second allocation should pick the OTHER device (more free memory)
        let a2 = alloc.allocate(4000, 1, "job-2", None).await.unwrap();
        assert_ne!(a1.device_ids[0], a2.device_ids[0]);
    }

    #[tokio::test]
    async fn list_allocations() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(2, 8192)).await;

        assert!(alloc.list_allocations().await.is_empty());
        alloc.allocate(1000, 1, "job-1", None).await.unwrap();
        alloc.allocate(1000, 1, "job-2", None).await.unwrap();
        assert_eq!(alloc.list_allocations().await.len(), 2);
    }

    #[tokio::test]
    async fn no_devices_allocation_fails() {
        let alloc = DeviceAllocator::new(0);
        let result = alloc.allocate(1000, 1, "job", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn reserve_reduces_available() {
        let alloc = DeviceAllocator::new(1000);
        alloc.init_from_hardware(&mock_gpus(1, 8192)).await;
        let avail = alloc.available_memory().await;
        assert_eq!(avail[0].1, 7192); // 8192 - 1000 reserve
    }

    #[tokio::test]
    async fn allocate_all_devices() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(4, 8192)).await;
        // Request all 4 devices
        let a = alloc.allocate(4000, 4, "full-use", None).await.unwrap();
        assert_eq!(a.device_ids.len(), 4);
        assert_eq!(a.memory_mb, 4000 * 4);
        assert_eq!(alloc.list_allocations().await.len(), 1);
    }

    #[tokio::test]
    async fn deallocate_then_reallocate() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(1, 8192)).await;

        // Allocate nearly all memory
        let a1 = alloc.allocate(7000, 1, "job-1", None).await.unwrap();
        // Should not fit another 7000
        assert!(alloc.allocate(7000, 1, "job-2", None).await.is_err());

        // Free and re-allocate
        alloc.deallocate(a1.id).await.unwrap();
        let a2 = alloc.allocate(7000, 1, "job-2", None).await.unwrap();
        assert_eq!(a2.device_ids.len(), 1);
    }

    #[tokio::test]
    async fn concurrent_allocate_and_deallocate() {
        let alloc = std::sync::Arc::new(DeviceAllocator::new(0));
        alloc.init_from_hardware(&mock_gpus(8, 16384)).await;

        let mut handles = vec![];
        // 8 concurrent allocations of 1 device each
        for i in 0..8 {
            let alloc = alloc.clone();
            handles.push(tokio::spawn(async move {
                let a = alloc
                    .allocate(1000, 1, &format!("job-{i}"), None)
                    .await
                    .unwrap();
                a.id
            }));
        }
        let mut ids = vec![];
        for h in handles {
            ids.push(h.await.unwrap());
        }
        assert_eq!(alloc.list_allocations().await.len(), 8);

        // Deallocate all concurrently
        let mut handles = vec![];
        for id in ids {
            let alloc = alloc.clone();
            handles.push(tokio::spawn(async move {
                alloc.deallocate(id).await.unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert!(alloc.list_allocations().await.is_empty());
    }

    #[tokio::test]
    async fn available_memory_after_multiple_allocations() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(2, 8192)).await;

        alloc.allocate(2000, 1, "job-1", None).await.unwrap();
        alloc.allocate(3000, 1, "job-2", None).await.unwrap();

        let avail = alloc.available_memory().await;
        // One device got 2000 allocated, the other got 3000 (fair scheduling picks least-loaded)
        let mut mems: Vec<u64> = avail.iter().map(|(_, m)| *m).collect();
        mems.sort();
        assert_eq!(mems, vec![5192, 6192]);
    }

    #[tokio::test]
    async fn allocate_memory_overflow_detection() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(2, u64::MAX)).await;
        // u64::MAX * 2 would overflow
        let result = alloc.allocate(u64::MAX, 2, "overflow-job", None).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("overflow"),
            "Expected overflow error, got: {err_msg}"
        );
    }

    #[tokio::test]
    async fn allocate_filters_by_compute_capability() {
        let alloc = DeviceAllocator::new(0);
        let gpus = vec![
            GpuDevice {
                index: 0,
                name: "Old GPU".into(),
                accelerator: AcceleratorKind::Cuda,
                memory_total_mb: 8192,
                memory_free_mb: 8192,
                compute_capability: Some((7, 0)),
            },
            GpuDevice {
                index: 1,
                name: "New GPU".into(),
                accelerator: AcceleratorKind::Cuda,
                memory_total_mb: 8192,
                memory_free_mb: 8192,
                compute_capability: Some((8, 9)),
            },
        ];
        alloc.init_from_hardware(&gpus).await;

        // Require compute 8.0+ — only device 1 qualifies
        let a = alloc
            .allocate(4000, 1, "bf16-model", Some((8, 0)))
            .await
            .unwrap();
        assert_eq!(a.device_ids, vec![1]);
    }

    #[tokio::test]
    async fn allocate_no_device_meets_compute_requirement() {
        let alloc = DeviceAllocator::new(0);
        let gpus = vec![GpuDevice {
            index: 0,
            name: "Old GPU".into(),
            accelerator: AcceleratorKind::Cuda,
            memory_total_mb: 8192,
            memory_free_mb: 8192,
            compute_capability: Some((7, 0)),
        }];
        alloc.init_from_hardware(&gpus).await;

        let result = alloc.allocate(4000, 1, "needs-ampere", Some((8, 0))).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn allocate_none_compute_cap_skips_filter() {
        let alloc = DeviceAllocator::new(0);
        alloc.init_from_hardware(&mock_gpus(2, 8192)).await;
        // None means no filtering
        let a = alloc.allocate(4000, 1, "any-gpu", None).await.unwrap();
        assert_eq!(a.device_ids.len(), 1);
    }

    #[tokio::test]
    async fn event_bus_receives_allocate_and_deallocate() {
        let bus = Arc::new(crate::hardware::events::GpuEventBus::new(16));
        let alloc = DeviceAllocator::with_event_bus(0, bus.clone());
        alloc.init_from_hardware(&mock_gpus(1, 8192)).await;

        let mut rx = bus.subscribe();

        let a = alloc.allocate(4000, 1, "test-job", None).await.unwrap();
        let event = rx.try_recv().unwrap();
        match event {
            crate::hardware::events::GpuEvent::Allocated { job_label, .. } => {
                assert_eq!(job_label, "test-job");
            }
            _ => panic!("Expected Allocated event"),
        }

        alloc.deallocate(a.id).await.unwrap();
        let event = rx.try_recv().unwrap();
        match event {
            crate::hardware::events::GpuEvent::Released { .. } => {}
            _ => panic!("Expected Released event"),
        }
    }
}
