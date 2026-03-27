//! Fleet node management for multi-node Ifran deployments.
//!
//! Tracks node registration, heartbeats, and health state transitions
//! using majra's HeartbeatTracker for the Online/Suspect/Offline FSM.

use chrono::{DateTime, Utc};
use ifran_types::IfranError;
use majra::heartbeat::{HeartbeatConfig, HeartbeatTracker, Status};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
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

impl From<Status> for NodeHealth {
    fn from(s: Status) -> Self {
        match s {
            Status::Online => NodeHealth::Online,
            Status::Suspect => NodeHealth::Suspect,
            Status::Offline => NodeHealth::Offline,
            _ => NodeHealth::Offline,
        }
    }
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

/// Ifran-specific metadata stored alongside majra's heartbeat state.
#[derive(Debug, Clone)]
struct NodeMeta {
    endpoint: String,
    gpu_info: NodeGpuInfo,
    last_heartbeat: DateTime<Utc>,
    registered_at: DateTime<Utc>,
}

/// Manages a fleet of Ifran nodes.
///
/// Delegates the Online/Suspect/Offline FSM to majra's HeartbeatTracker.
/// Keeps GPU-specific metadata, validation, eviction, and stats in Ifran.
pub struct FleetManager {
    tracker: Arc<RwLock<HeartbeatTracker>>,
    meta: Arc<RwLock<HashMap<NodeId, NodeMeta>>>,
    offline_timeout: Duration,
    cancel_tx: watch::Sender<bool>,
}

impl FleetManager {
    /// Create a new fleet manager.
    pub fn new(suspect_timeout: Duration, offline_timeout: Duration) -> Self {
        let (cancel_tx, _) = watch::channel(false);
        Self {
            tracker: Arc::new(RwLock::new(HeartbeatTracker::new(HeartbeatConfig {
                suspect_after: suspect_timeout,
                offline_after: offline_timeout,
                eviction_policy: None,
            }))),
            meta: Arc::new(RwLock::new(HashMap::new())),
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
        let tracker = self.tracker.clone();
        let meta = self.meta.clone();
        let offline_timeout = self.offline_timeout;
        let mut cancel_rx = self.cancel_tx.subscribe();

        tokio::spawn(async move {
            loop {
                Self::do_health_check(&tracker, &meta, offline_timeout).await;
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
    ) -> ifran_types::error::Result<FleetNode> {
        // Validate id: 1-128 chars, alphanumeric + hyphens only
        if req.id.is_empty() || req.id.len() > 128 {
            return Err(IfranError::ValidationError(
                "id must be 1-128 characters".into(),
            ));
        }
        if !req
            .id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            return Err(IfranError::ValidationError(
                "id must contain only alphanumeric characters and hyphens".into(),
            ));
        }

        // Validate endpoint
        if !req.endpoint.starts_with("http://") && !req.endpoint.starts_with("https://") {
            return Err(IfranError::ValidationError(
                "endpoint must start with http:// or https://".into(),
            ));
        }

        // Validate gpu_count
        if req.gpu_count > 64 {
            return Err(IfranError::ValidationError(
                "gpu_count must be <= 64".into(),
            ));
        }

        // Validate total_gpu_memory_mb
        if req.total_gpu_memory_mb > 10_000_000 {
            return Err(IfranError::ValidationError(
                "total_gpu_memory_mb must be <= 10,000,000 (10 TB)".into(),
            ));
        }

        let now = Utc::now();

        // Register with majra heartbeat tracker
        self.tracker
            .write()
            .await
            .register(&req.id, serde_json::json!({}));

        // Store Ifran-specific metadata
        let node_meta = NodeMeta {
            endpoint: req.endpoint.clone(),
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

        let node = FleetNode {
            id: req.id.clone(),
            endpoint: req.endpoint,
            health: NodeHealth::Online,
            gpu_info: node_meta.gpu_info.clone(),
            last_heartbeat: now,
            registered_at: now,
        };

        self.meta.write().await.insert(req.id, node_meta);
        Ok(node)
    }

    /// Process a heartbeat from a node.
    pub async fn heartbeat(
        &self,
        node_id: &str,
        gpu_utilization_pct: Option<f32>,
        gpu_memory_used_mb: Option<u64>,
        gpu_temperature_c: Option<f32>,
    ) -> ifran_types::error::Result<()> {
        // Validate telemetry values
        if let Some(pct) = gpu_utilization_pct {
            if pct.is_nan() || !(0.0..=100.0).contains(&pct) {
                return Err(IfranError::ValidationError(
                    "gpu_utilization_pct must be 0.0..=100.0 and not NaN".into(),
                ));
            }
        }
        if let Some(temp) = gpu_temperature_c {
            if temp.is_nan() || !(-50.0..=250.0).contains(&temp) {
                return Err(IfranError::ValidationError(
                    "gpu_temperature_c must be -50.0..=250.0 and not NaN".into(),
                ));
            }
        }

        // Update majra heartbeat (resets to Online)
        let found = self.tracker.write().await.heartbeat(node_id);
        if !found {
            return Err(IfranError::HardwareError(format!(
                "Node {node_id} not found"
            )));
        }

        // Update GPU metadata
        let mut meta = self.meta.write().await;
        if let Some(m) = meta.get_mut(node_id) {
            m.last_heartbeat = Utc::now();
            m.gpu_info.gpu_utilization_pct = gpu_utilization_pct;
            m.gpu_info.gpu_memory_used_mb = gpu_memory_used_mb;
            m.gpu_info.gpu_temperature_c = gpu_temperature_c;
        }

        Ok(())
    }

    /// Run health checks on all nodes, transitioning statuses and evicting
    /// nodes that have been offline for 2x offline_timeout.
    pub async fn check_health(&self) {
        Self::do_health_check(&self.tracker, &self.meta, self.offline_timeout).await;
    }

    /// Internal health check logic, usable from both the public method and
    /// the spawned background loop.
    async fn do_health_check(
        tracker: &RwLock<HeartbeatTracker>,
        meta: &RwLock<HashMap<NodeId, NodeMeta>>,
        offline_timeout: Duration,
    ) {
        // Let majra update statuses based on elapsed time
        tracker.write().await.update_statuses();

        // Evict nodes that have been offline too long
        let eviction_threshold = offline_timeout * 2;
        let now = Utc::now();
        let mut meta = meta.write().await;
        let mut tracker = tracker.write().await;

        let mut evicted = Vec::new();
        meta.retain(|id, m| {
            let elapsed = now
                .signed_duration_since(m.last_heartbeat)
                .to_std()
                .unwrap_or(Duration::MAX);
            if elapsed >= eviction_threshold {
                tracing::info!(node_id = %id, elapsed_secs = elapsed.as_secs(), "evicting offline node from fleet");
                evicted.push(id.clone());
                false
            } else {
                true
            }
        });

        for id in &evicted {
            tracker.deregister(id);
        }

        if !evicted.is_empty() {
            tracing::info!(count = evicted.len(), "evicted offline nodes from fleet");
        }
    }

    /// Evict nodes that have been offline longer than `2 * offline_timeout`.
    /// Returns the number of evicted nodes.
    pub async fn evict_offline(&self) -> usize {
        let eviction_threshold = self.offline_timeout * 2;
        let now = Utc::now();
        let mut meta = self.meta.write().await;
        let mut tracker = self.tracker.write().await;
        let before = meta.len();

        let mut evicted_ids = Vec::new();
        meta.retain(|id, m| {
            let elapsed = now
                .signed_duration_since(m.last_heartbeat)
                .to_std()
                .unwrap_or(Duration::MAX);
            if elapsed >= eviction_threshold {
                tracing::info!(node_id = %id, elapsed_secs = elapsed.as_secs(), "evicting offline node from fleet");
                evicted_ids.push(id.clone());
                false
            } else {
                true
            }
        });

        for id in &evicted_ids {
            tracker.deregister(id);
        }

        let evicted = before - meta.len();
        if evicted > 0 {
            tracing::info!(count = evicted, "evicted offline nodes from fleet");
        }
        evicted
    }

    /// List all registered nodes.
    pub async fn list_nodes(&self) -> Vec<FleetNode> {
        let tracker = self.tracker.read().await;
        let meta = self.meta.read().await;

        meta.iter()
            .map(|(id, m)| {
                let health = tracker
                    .get(id)
                    .map(|s| NodeHealth::from(s.status))
                    .unwrap_or(NodeHealth::Offline);

                FleetNode {
                    id: id.clone(),
                    endpoint: m.endpoint.clone(),
                    health,
                    gpu_info: m.gpu_info.clone(),
                    last_heartbeat: m.last_heartbeat,
                    registered_at: m.registered_at,
                }
            })
            .collect()
    }

    /// Get a specific node.
    pub async fn get_node(&self, node_id: &str) -> Option<FleetNode> {
        let tracker = self.tracker.read().await;
        let meta = self.meta.read().await;

        let m = meta.get(node_id)?;
        let health = tracker
            .get(node_id)
            .map(|s| NodeHealth::from(s.status))
            .unwrap_or(NodeHealth::Offline);

        Some(FleetNode {
            id: node_id.to_string(),
            endpoint: m.endpoint.clone(),
            health,
            gpu_info: m.gpu_info.clone(),
            last_heartbeat: m.last_heartbeat,
            registered_at: m.registered_at,
        })
    }

    /// Remove a node from the fleet.
    pub async fn remove(&self, node_id: &str) -> bool {
        self.tracker.write().await.deregister(node_id);
        self.meta.write().await.remove(node_id).is_some()
    }

    /// Get aggregate fleet statistics.
    pub async fn stats(&self) -> FleetStats {
        let tracker = self.tracker.read().await;
        let meta = self.meta.read().await;

        let mut stats = FleetStats {
            total_nodes: meta.len(),
            online: 0,
            suspect: 0,
            offline: 0,
            total_gpus: 0,
            total_gpu_memory_mb: 0,
        };

        for (id, m) in meta.iter() {
            let health = tracker
                .get(id)
                .map(|s| NodeHealth::from(s.status))
                .unwrap_or(NodeHealth::Offline);

            match health {
                NodeHealth::Online => stats.online += 1,
                NodeHealth::Suspect => stats.suspect += 1,
                NodeHealth::Offline => stats.offline += 1,
            }
            stats.total_gpus += m.gpu_info.gpu_count;
            stats.total_gpu_memory_mb += m.gpu_info.total_gpu_memory_mb;
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
