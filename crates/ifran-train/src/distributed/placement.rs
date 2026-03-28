//! Pluggable placement policies for distributing training workers across nodes.
//!
//! Policies decide which nodes receive which workers based on GPU availability,
//! cost, and locality constraints.

use ifran_types::IfranError;
use ifran_types::distributed::WorkerAssignment;
use ifran_types::error::Result;

/// Resource snapshot for a fleet node, used for placement decisions.
#[derive(Debug, Clone)]
pub struct NodeResources {
    pub node_id: String,
    pub endpoint: String,
    pub available_gpu_ids: Vec<u32>,
    pub available_gpu_memory_mb: u64,
    pub gpu_utilization_pct: Option<f32>,
    pub cost_per_gpu_hour: Option<f64>,
}

/// Trait for pluggable placement policies.
pub trait PlacementPolicy: Send + Sync {
    #[must_use]
    fn name(&self) -> &str;
    fn place(
        &self,
        world_size: u32,
        gpus_per_worker: u32,
        nodes: &[NodeResources],
    ) -> Result<Vec<WorkerAssignment>>;
}

/// Pack workers onto the fewest nodes possible to maximize GPU locality (NVLink, same bus).
pub struct GpuAffinityPolicy;

impl PlacementPolicy for GpuAffinityPolicy {
    fn name(&self) -> &str {
        "gpu_affinity"
    }

    fn place(
        &self,
        world_size: u32,
        gpus_per_worker: u32,
        nodes: &[NodeResources],
    ) -> Result<Vec<WorkerAssignment>> {
        if gpus_per_worker == 0 {
            return Err(IfranError::DistributedError(
                "gpus_per_worker must be > 0".into(),
            ));
        }

        // Sort nodes by most available GPUs (pack onto fewest nodes)
        let mut sorted: Vec<&NodeResources> = nodes.iter().collect();
        sorted.sort_by(|a, b| b.available_gpu_ids.len().cmp(&a.available_gpu_ids.len()));

        let mut assignments = Vec::new();
        let mut rank = 0u32;

        for node in &sorted {
            if rank >= world_size {
                break;
            }

            let available = &node.available_gpu_ids;
            let mut gpu_offset = 0;

            while rank < world_size && gpu_offset + gpus_per_worker as usize <= available.len() {
                let device_ids: Vec<u32> =
                    available[gpu_offset..gpu_offset + gpus_per_worker as usize].to_vec();
                assignments.push(WorkerAssignment {
                    rank,
                    instance_id: node.node_id.clone(),
                    endpoint: node.endpoint.clone(),
                    device_ids,
                });
                rank += 1;
                gpu_offset += gpus_per_worker as usize;
            }
        }

        if rank < world_size {
            return Err(IfranError::DistributedError(format!(
                "Not enough GPU slots: need {} workers x {} GPUs, only placed {rank}",
                world_size, gpus_per_worker
            )));
        }

        Ok(assignments)
    }
}

/// Spread workers evenly across nodes.
pub struct BalancedPolicy;

impl PlacementPolicy for BalancedPolicy {
    fn name(&self) -> &str {
        "balanced"
    }

    fn place(
        &self,
        world_size: u32,
        gpus_per_worker: u32,
        nodes: &[NodeResources],
    ) -> Result<Vec<WorkerAssignment>> {
        if gpus_per_worker == 0 {
            return Err(IfranError::DistributedError(
                "gpus_per_worker must be > 0".into(),
            ));
        }

        if nodes.is_empty() {
            return Err(IfranError::DistributedError(
                "No nodes available for placement".into(),
            ));
        }

        let eligible: Vec<&NodeResources> = nodes
            .iter()
            .filter(|n| n.available_gpu_ids.len() >= gpus_per_worker as usize)
            .collect();

        if eligible.is_empty() {
            return Err(IfranError::DistributedError(format!(
                "No nodes have {} available GPUs",
                gpus_per_worker
            )));
        }

        let mut assignments = Vec::new();
        let mut rank = 0u32;
        let mut node_gpu_offsets: Vec<usize> = vec![0; eligible.len()];

        // Round-robin across eligible nodes
        while rank < world_size {
            let node_idx = rank as usize % eligible.len();
            let node = eligible[node_idx];
            let offset = node_gpu_offsets[node_idx];

            if offset + gpus_per_worker as usize > node.available_gpu_ids.len() {
                return Err(IfranError::DistributedError(format!(
                    "Node {} exhausted GPU slots",
                    node.node_id
                )));
            }

            let device_ids =
                node.available_gpu_ids[offset..offset + gpus_per_worker as usize].to_vec();
            assignments.push(WorkerAssignment {
                rank,
                instance_id: node.node_id.clone(),
                endpoint: node.endpoint.clone(),
                device_ids,
            });

            node_gpu_offsets[node_idx] += gpus_per_worker as usize;
            rank += 1;
        }

        Ok(assignments)
    }
}

/// Prefer cheapest nodes first, then fill more expensive ones.
pub struct CostAwarePolicy;

impl PlacementPolicy for CostAwarePolicy {
    fn name(&self) -> &str {
        "cost_aware"
    }

    fn place(
        &self,
        world_size: u32,
        gpus_per_worker: u32,
        nodes: &[NodeResources],
    ) -> Result<Vec<WorkerAssignment>> {
        if gpus_per_worker == 0 {
            return Err(IfranError::DistributedError(
                "gpus_per_worker must be > 0".into(),
            ));
        }

        // Sort by cost (cheapest first, unknown cost goes last)
        let mut sorted: Vec<&NodeResources> = nodes.iter().collect();
        sorted.sort_by(|a, b| {
            let cost_a = a.cost_per_gpu_hour.unwrap_or(f64::MAX);
            let cost_b = b.cost_per_gpu_hour.unwrap_or(f64::MAX);
            cost_a
                .partial_cmp(&cost_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut assignments = Vec::new();
        let mut rank = 0u32;

        for node in &sorted {
            if rank >= world_size {
                break;
            }

            let available = &node.available_gpu_ids;
            let mut gpu_offset = 0;

            while rank < world_size && gpu_offset + gpus_per_worker as usize <= available.len() {
                let device_ids =
                    available[gpu_offset..gpu_offset + gpus_per_worker as usize].to_vec();
                assignments.push(WorkerAssignment {
                    rank,
                    instance_id: node.node_id.clone(),
                    endpoint: node.endpoint.clone(),
                    device_ids,
                });
                rank += 1;
                gpu_offset += gpus_per_worker as usize;
            }
        }

        if rank < world_size {
            return Err(IfranError::DistributedError(format!(
                "Not enough GPU slots for {world_size} workers"
            )));
        }

        Ok(assignments)
    }
}

/// Create a placement policy from the enum variant.
pub fn policy_from_kind(
    kind: ifran_types::distributed::PlacementPolicyKind,
) -> Box<dyn PlacementPolicy> {
    use ifran_types::distributed::PlacementPolicyKind;
    match kind {
        PlacementPolicyKind::GpuAffinity => Box::new(GpuAffinityPolicy),
        PlacementPolicyKind::Balanced => Box::new(BalancedPolicy),
        PlacementPolicyKind::CostAware => Box::new(CostAwarePolicy),
        _ => Box::new(BalancedPolicy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nodes() -> Vec<NodeResources> {
        vec![
            NodeResources {
                node_id: "node-1".into(),
                endpoint: "http://node-1:8420".into(),
                available_gpu_ids: vec![0, 1, 2, 3],
                available_gpu_memory_mb: 96000,
                gpu_utilization_pct: Some(10.0),
                cost_per_gpu_hour: Some(2.0),
            },
            NodeResources {
                node_id: "node-2".into(),
                endpoint: "http://node-2:8420".into(),
                available_gpu_ids: vec![0, 1],
                available_gpu_memory_mb: 48000,
                gpu_utilization_pct: Some(50.0),
                cost_per_gpu_hour: Some(1.0),
            },
            NodeResources {
                node_id: "node-3".into(),
                endpoint: "http://node-3:8420".into(),
                available_gpu_ids: vec![0, 1, 2, 3],
                available_gpu_memory_mb: 96000,
                gpu_utilization_pct: Some(5.0),
                cost_per_gpu_hour: Some(3.0),
            },
        ]
    }

    #[test]
    fn gpu_affinity_packs_onto_fewest_nodes() {
        let policy = GpuAffinityPolicy;
        let nodes = make_nodes();
        let assignments = policy.place(4, 1, &nodes).unwrap();

        assert_eq!(assignments.len(), 4);
        // Should pack onto node-1 and node-3 (4 GPUs each) before node-2
        let node_ids: Vec<&str> = assignments.iter().map(|a| a.instance_id.as_str()).collect();
        // First 4 should be from the same node or two nodes with 4 GPUs
        assert!(
            node_ids.iter().filter(|&&n| n == "node-1").count() == 4
                || node_ids.iter().filter(|&&n| n == "node-3").count() == 4
        );
    }

    #[test]
    fn balanced_spreads_evenly() {
        let policy = BalancedPolicy;
        let nodes = make_nodes();
        let assignments = policy.place(3, 1, &nodes).unwrap();

        assert_eq!(assignments.len(), 3);
        // Round-robin: each node gets one worker
        let mut node_ids: Vec<&str> = assignments.iter().map(|a| a.instance_id.as_str()).collect();
        node_ids.sort();
        assert_eq!(node_ids, vec!["node-1", "node-2", "node-3"]);
    }

    #[test]
    fn cost_aware_prefers_cheapest() {
        let policy = CostAwarePolicy;
        let nodes = make_nodes();
        let assignments = policy.place(3, 1, &nodes).unwrap();

        assert_eq!(assignments.len(), 3);
        // node-2 is cheapest ($1/hr) so ranks 0-1 should go there (2 GPUs)
        assert_eq!(assignments[0].instance_id, "node-2");
        assert_eq!(assignments[1].instance_id, "node-2");
        // node-1 is next cheapest ($2/hr)
        assert_eq!(assignments[2].instance_id, "node-1");
    }

    #[test]
    fn multi_gpu_per_worker() {
        let policy = GpuAffinityPolicy;
        let nodes = make_nodes();
        let assignments = policy.place(2, 2, &nodes).unwrap();

        assert_eq!(assignments.len(), 2);
        for a in &assignments {
            assert_eq!(a.device_ids.len(), 2);
        }
    }

    #[test]
    fn insufficient_gpus_fails() {
        let policy = GpuAffinityPolicy;
        let nodes = make_nodes(); // total 10 GPUs
        let result = policy.place(20, 1, &nodes);
        assert!(result.is_err());
    }

    #[test]
    fn empty_nodes_fails() {
        let policy = BalancedPolicy;
        let result = policy.place(1, 1, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn ranks_are_sequential() {
        let policy = BalancedPolicy;
        let nodes = make_nodes();
        let assignments = policy.place(4, 1, &nodes).unwrap();
        for (i, a) in assignments.iter().enumerate() {
            assert_eq!(a.rank, i as u32);
        }
    }

    #[test]
    fn gpus_per_worker_zero_fails() {
        let nodes = make_nodes();
        assert!(GpuAffinityPolicy.place(2, 0, &nodes).is_err());
        assert!(BalancedPolicy.place(2, 0, &nodes).is_err());
        assert!(CostAwarePolicy.place(2, 0, &nodes).is_err());
    }

    #[test]
    fn world_size_zero_returns_empty() {
        let nodes = make_nodes();
        let result = GpuAffinityPolicy.place(0, 1, &nodes).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn policy_from_kind_works() {
        use ifran_types::distributed::PlacementPolicyKind;
        let p = policy_from_kind(PlacementPolicyKind::GpuAffinity);
        assert_eq!(p.name(), "gpu_affinity");
        let p = policy_from_kind(PlacementPolicyKind::Balanced);
        assert_eq!(p.name(), "balanced");
        let p = policy_from_kind(PlacementPolicyKind::CostAware);
        assert_eq!(p.name(), "cost_aware");
    }
}
