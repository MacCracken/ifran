//! Fleet node management for multi-node Synapse deployments.
//!
//! Tracks node registration, heartbeats, and health state transitions.
//! Nodes transition: Online -> Suspect (no heartbeat for suspect_timeout)
//!                         -> Offline (no heartbeat for offline_timeout)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use synapse_types::SynapseError;
use tokio::sync::{RwLock, watch};

/// Unique node identifier.
pub type NodeId = String;

/// Health state of a fleet node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeHealth {
    Online,
    Suspect,
    Offline,
}

/// GPU information reported by a fleet node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeGpuInfo {
    pub gpu_count: usize,
    pub total_gpu_memory_mb: u64,
    pub gpu_utilization_pct: Option<f32>,
    pub gpu_memory_used_mb: Option<u64>,
    pub gpu_temperature_c: Option<f32>,
}

/// A registered fleet node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetNode {
    pub id: NodeId,
    pub endpoint: String,
    pub health: NodeHealth,
    pub gpu_info: NodeGpuInfo,
    pub last_heartbeat: DateTime<Utc>,
    pub registered_at: DateTime<Utc>,
}

/// Aggregate fleet statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetStats {
    pub total_nodes: usize,
    pub online: usize,
    pub suspect: usize,
    pub offline: usize,
    pub total_gpus: usize,
    pub total_gpu_memory_mb: u64,
}

/// Registration request for a new node.
#[derive(Debug, Clone, Deserialize)]
pub struct RegisterNodeRequest {
    pub id: NodeId,
    pub endpoint: String,
    pub gpu_count: usize,
    pub total_gpu_memory_mb: u64,
}

/// Manages a fleet of Synapse nodes.
pub struct FleetManager {
    nodes: Arc<RwLock<HashMap<NodeId, FleetNode>>>,
    suspect_timeout: Duration,
    offline_timeout: Duration,
    cancel_tx: watch::Sender<bool>,
}

impl FleetManager {
    /// Create a new fleet manager.
    pub fn new(suspect_timeout: Duration, offline_timeout: Duration) -> Self {
        let (cancel_tx, _) = watch::channel(false);
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            suspect_timeout,
            offline_timeout,
            cancel_tx,
        }
    }

    /// Create with default timeouts (30s suspect, 90s offline).
    pub fn with_defaults() -> Self {
        Self::new(Duration::from_secs(30), Duration::from_secs(90))
    }

    /// Start the periodic health check loop.
    pub fn start_health_check_loop(&self, interval: Duration) {
        let nodes = self.nodes.clone();
        let suspect = self.suspect_timeout;
        let offline = self.offline_timeout;
        let mut cancel_rx = self.cancel_tx.subscribe();

        tokio::spawn(async move {
            loop {
                Self::check_health_inner(&nodes, suspect, offline).await;
                tokio::select! {
                    _ = tokio::time::sleep(interval) => {}
                    _ = cancel_rx.changed() => break,
                }
            }
        });
    }

    /// Register a new node or update an existing one.
    pub async fn register(
        &self,
        req: RegisterNodeRequest,
    ) -> synapse_types::error::Result<FleetNode> {
        // Validate id: 1-128 chars, alphanumeric + hyphens only
        if req.id.is_empty() || req.id.len() > 128 {
            return Err(SynapseError::ValidationError(
                "id must be 1-128 characters".into(),
            ));
        }
        if !req
            .id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            return Err(SynapseError::ValidationError(
                "id must contain only alphanumeric characters and hyphens".into(),
            ));
        }

        // Validate endpoint
        if !req.endpoint.starts_with("http://") && !req.endpoint.starts_with("https://") {
            return Err(SynapseError::ValidationError(
                "endpoint must start with http:// or https://".into(),
            ));
        }

        // Validate gpu_count
        if req.gpu_count > 64 {
            return Err(SynapseError::ValidationError(
                "gpu_count must be <= 64".into(),
            ));
        }

        // Validate total_gpu_memory_mb
        if req.total_gpu_memory_mb > 10_000_000 {
            return Err(SynapseError::ValidationError(
                "total_gpu_memory_mb must be <= 10,000,000 (10 TB)".into(),
            ));
        }

        let now = Utc::now();
        let node = FleetNode {
            id: req.id.clone(),
            endpoint: req.endpoint,
            health: NodeHealth::Online,
            gpu_info: NodeGpuInfo {
                gpu_count: req.gpu_count,
                total_gpu_memory_mb: req.total_gpu_memory_mb,
                gpu_utilization_pct: None,
                gpu_memory_used_mb: None,
                gpu_temperature_c: None,
            },
            last_heartbeat: now,
            registered_at: now,
        };
        self.nodes.write().await.insert(req.id, node.clone());
        Ok(node)
    }

    /// Process a heartbeat from a node.
    pub async fn heartbeat(
        &self,
        node_id: &str,
        gpu_utilization_pct: Option<f32>,
        gpu_memory_used_mb: Option<u64>,
        gpu_temperature_c: Option<f32>,
    ) -> synapse_types::error::Result<()> {
        // Validate telemetry values
        if let Some(pct) = gpu_utilization_pct {
            if pct.is_nan() || !(0.0..=100.0).contains(&pct) {
                return Err(SynapseError::ValidationError(
                    "gpu_utilization_pct must be 0.0..=100.0 and not NaN".into(),
                ));
            }
        }
        if let Some(temp) = gpu_temperature_c {
            if temp.is_nan() || !(-50.0..=250.0).contains(&temp) {
                return Err(SynapseError::ValidationError(
                    "gpu_temperature_c must be -50.0..=250.0 and not NaN".into(),
                ));
            }
        }

        let mut nodes = self.nodes.write().await;
        let node = nodes
            .get_mut(node_id)
            .ok_or_else(|| SynapseError::HardwareError(format!("Node {node_id} not found")))?;

        node.health = NodeHealth::Online;
        node.last_heartbeat = Utc::now();
        node.gpu_info.gpu_utilization_pct = gpu_utilization_pct;
        node.gpu_info.gpu_memory_used_mb = gpu_memory_used_mb;
        node.gpu_info.gpu_temperature_c = gpu_temperature_c;

        Ok(())
    }

    /// Run health checks on all nodes.
    pub async fn check_health(&self) {
        Self::check_health_inner(&self.nodes, self.suspect_timeout, self.offline_timeout).await;
    }

    async fn check_health_inner(
        nodes: &Arc<RwLock<HashMap<NodeId, FleetNode>>>,
        suspect_timeout: Duration,
        offline_timeout: Duration,
    ) {
        let now = Utc::now();
        let mut nodes = nodes.write().await;
        for node in nodes.values_mut() {
            let elapsed = now
                .signed_duration_since(node.last_heartbeat)
                .to_std()
                .unwrap_or(Duration::MAX);

            if elapsed >= offline_timeout {
                node.health = NodeHealth::Offline;
            } else if elapsed >= suspect_timeout {
                node.health = NodeHealth::Suspect;
            }
        }

        // Evict nodes that have been offline for too long (2x offline_timeout)
        let eviction_threshold = offline_timeout * 2;
        let before = nodes.len();
        nodes.retain(|id, node| {
            let elapsed = now
                .signed_duration_since(node.last_heartbeat)
                .to_std()
                .unwrap_or(Duration::MAX);
            if elapsed >= eviction_threshold {
                tracing::info!(node_id = %id, elapsed_secs = elapsed.as_secs(), "evicting offline node from fleet");
                false
            } else {
                true
            }
        });
        let evicted = before - nodes.len();
        if evicted > 0 {
            tracing::info!(count = evicted, "evicted offline nodes from fleet");
        }
    }

    /// Evict nodes that have been offline longer than `2 * offline_timeout`.
    /// Returns the number of evicted nodes.
    pub async fn evict_offline(&self) -> usize {
        let now = Utc::now();
        let eviction_threshold = self.offline_timeout * 2;
        let mut nodes = self.nodes.write().await;
        let before = nodes.len();
        nodes.retain(|id, node| {
            let elapsed = now
                .signed_duration_since(node.last_heartbeat)
                .to_std()
                .unwrap_or(Duration::MAX);
            if elapsed >= eviction_threshold {
                tracing::info!(node_id = %id, elapsed_secs = elapsed.as_secs(), "evicting offline node from fleet");
                false
            } else {
                true
            }
        });
        let evicted = before - nodes.len();
        if evicted > 0 {
            tracing::info!(count = evicted, "evicted offline nodes from fleet");
        }
        evicted
    }

    /// List all registered nodes.
    pub async fn list_nodes(&self) -> Vec<FleetNode> {
        self.nodes.read().await.values().cloned().collect()
    }

    /// Get a specific node.
    pub async fn get_node(&self, node_id: &str) -> Option<FleetNode> {
        self.nodes.read().await.get(node_id).cloned()
    }

    /// Remove a node from the fleet.
    pub async fn remove(&self, node_id: &str) -> bool {
        self.nodes.write().await.remove(node_id).is_some()
    }

    /// Get aggregate fleet statistics.
    pub async fn stats(&self) -> FleetStats {
        let nodes = self.nodes.read().await;
        let mut stats = FleetStats {
            total_nodes: nodes.len(),
            online: 0,
            suspect: 0,
            offline: 0,
            total_gpus: 0,
            total_gpu_memory_mb: 0,
        };

        for node in nodes.values() {
            match node.health {
                NodeHealth::Online => stats.online += 1,
                NodeHealth::Suspect => stats.suspect += 1,
                NodeHealth::Offline => stats.offline += 1,
            }
            stats.total_gpus += node.gpu_info.gpu_count;
            stats.total_gpu_memory_mb += node.gpu_info.total_gpu_memory_mb;
        }

        stats
    }

    /// Stop the health check loop.
    pub fn stop(&self) {
        let _ = self.cancel_tx.send(true);
    }
}

impl Drop for FleetManager {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_req(id: &str) -> RegisterNodeRequest {
        RegisterNodeRequest {
            id: id.into(),
            endpoint: format!("http://{id}:8420"),
            gpu_count: 2,
            total_gpu_memory_mb: 48000,
        }
    }

    #[tokio::test]
    async fn register_and_list() {
        let fm = FleetManager::with_defaults();
        fm.register(make_req("node-1")).await.unwrap();
        fm.register(make_req("node-2")).await.unwrap();

        let nodes = fm.list_nodes().await;
        assert_eq!(nodes.len(), 2);
    }

    #[tokio::test]
    async fn register_is_idempotent() {
        let fm = FleetManager::with_defaults();
        fm.register(make_req("node-1")).await.unwrap();
        fm.register(make_req("node-1")).await.unwrap();
        assert_eq!(fm.list_nodes().await.len(), 1);
    }

    #[tokio::test]
    async fn heartbeat_updates_health() {
        let fm = FleetManager::with_defaults();
        fm.register(make_req("node-1")).await.unwrap();

        fm.heartbeat("node-1", Some(75.0), Some(20000), Some(68.0))
            .await
            .unwrap();

        let node = fm.get_node("node-1").await.unwrap();
        assert_eq!(node.health, NodeHealth::Online);
        assert_eq!(node.gpu_info.gpu_utilization_pct, Some(75.0));
        assert_eq!(node.gpu_info.gpu_memory_used_mb, Some(20000));
        assert_eq!(node.gpu_info.gpu_temperature_c, Some(68.0));
    }

    #[tokio::test]
    async fn heartbeat_unknown_node() {
        let fm = FleetManager::with_defaults();
        assert!(fm.heartbeat("unknown", None, None, None).await.is_err());
    }

    #[tokio::test]
    async fn health_transitions() {
        let fm = FleetManager::new(Duration::from_millis(50), Duration::from_millis(100));
        fm.register(make_req("node-1")).await.unwrap();

        // Initially online
        let node = fm.get_node("node-1").await.unwrap();
        assert_eq!(node.health, NodeHealth::Online);

        // Wait past suspect timeout
        tokio::time::sleep(Duration::from_millis(60)).await;
        fm.check_health().await;
        let node = fm.get_node("node-1").await.unwrap();
        assert_eq!(node.health, NodeHealth::Suspect);

        // Wait past offline timeout
        tokio::time::sleep(Duration::from_millis(50)).await;
        fm.check_health().await;
        let node = fm.get_node("node-1").await.unwrap();
        assert_eq!(node.health, NodeHealth::Offline);
    }

    #[tokio::test]
    async fn heartbeat_resets_to_online() {
        let fm = FleetManager::new(Duration::from_millis(50), Duration::from_millis(100));
        fm.register(make_req("node-1")).await.unwrap();

        tokio::time::sleep(Duration::from_millis(60)).await;
        fm.check_health().await;
        assert_eq!(
            fm.get_node("node-1").await.unwrap().health,
            NodeHealth::Suspect
        );

        // Heartbeat resets to Online
        fm.heartbeat("node-1", None, None, None).await.unwrap();
        assert_eq!(
            fm.get_node("node-1").await.unwrap().health,
            NodeHealth::Online
        );
    }

    #[tokio::test]
    async fn remove_node() {
        let fm = FleetManager::with_defaults();
        fm.register(make_req("node-1")).await.unwrap();
        assert!(fm.remove("node-1").await);
        assert!(!fm.remove("node-1").await); // already removed
        assert!(fm.list_nodes().await.is_empty());
    }

    #[tokio::test]
    async fn stats() {
        let fm = FleetManager::new(Duration::from_millis(50), Duration::from_millis(100));
        fm.register(make_req("node-1")).await.unwrap();
        fm.register(make_req("node-2")).await.unwrap();

        let s = fm.stats().await;
        assert_eq!(s.total_nodes, 2);
        assert_eq!(s.online, 2);
        assert_eq!(s.total_gpus, 4);
        assert_eq!(s.total_gpu_memory_mb, 96000);

        // Make one suspect
        tokio::time::sleep(Duration::from_millis(60)).await;
        fm.heartbeat("node-1", None, None, None).await.unwrap();
        fm.check_health().await;

        let s = fm.stats().await;
        assert_eq!(s.online, 1);
        assert_eq!(s.suspect, 1);
    }

    #[tokio::test]
    async fn get_nonexistent_node() {
        let fm = FleetManager::with_defaults();
        assert!(fm.get_node("nope").await.is_none());
    }

    #[tokio::test]
    async fn fleet_stats_serializes() {
        let stats = FleetStats {
            total_nodes: 3,
            online: 2,
            suspect: 1,
            offline: 0,
            total_gpus: 6,
            total_gpu_memory_mb: 144000,
        };
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"total_nodes\":3"));
    }
}
