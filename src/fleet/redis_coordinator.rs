//! Redis-backed fleet coordination using majra's Redis primitives.
//!
//! When enabled, runs alongside the in-memory FleetManager to provide
//! cross-instance node discovery and heartbeat propagation via Redis.
//! This allows multiple ifran-server instances to share fleet state.

use crate::types::IfranError;
use crate::types::error::Result;
use majra::redis_backend::{RedisHeartbeatTracker, RedisPubSub, RedisRateLimiter};
use std::sync::Arc;

/// Redis fleet coordinator — wraps majra's Redis-backed primitives
/// for multi-instance fleet coordination.
pub struct RedisCoordinator {
    /// Redis-backed heartbeat tracker for cross-instance node health.
    pub heartbeat_tracker: Arc<RedisHeartbeatTracker>,
    /// Redis-backed pub/sub for cross-instance event propagation.
    pub pubsub: Arc<RedisPubSub>,
    /// Redis-backed rate limiter for distributed rate limiting.
    pub rate_limiter: Arc<RedisRateLimiter>,
    client: redis::Client,
}

impl RedisCoordinator {
    /// Create a new Redis coordinator from a connection URL.
    pub fn new(redis_url: &str, rate: f64, burst: usize) -> Result<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| IfranError::ConfigError(format!("Redis connection failed: {e}")))?;

        let heartbeat_tracker = Arc::new(RedisHeartbeatTracker::new(
            client.clone(),
            "ifran:fleet:hb:",
            60, // TTL in seconds — nodes must heartbeat within this window
        ));

        let pubsub = Arc::new(RedisPubSub::new(client.clone(), "ifran:events:"));

        let rate_limiter = Arc::new(RedisRateLimiter::new(
            client.clone(),
            rate,
            burst,
            "ifran:ratelimit:",
        ));

        Ok(Self {
            heartbeat_tracker,
            pubsub,
            rate_limiter,
            client,
        })
    }

    /// Get the underlying Redis client for custom operations.
    #[must_use]
    pub fn client(&self) -> &redis::Client {
        &self.client
    }

    /// Register a node heartbeat via Redis (cross-instance visible).
    pub async fn heartbeat(&self, node_id: &str, metadata: &serde_json::Value) -> Result<()> {
        self.heartbeat_tracker
            .heartbeat(node_id, metadata)
            .await
            .map_err(|e| IfranError::StorageError(format!("Redis heartbeat failed: {e}")))?;
        Ok(())
    }

    /// Register a node in Redis with JSON metadata.
    pub async fn register_node(&self, node_id: &str, metadata: &serde_json::Value) -> Result<()> {
        self.heartbeat_tracker
            .register(node_id, metadata)
            .await
            .map_err(|e| IfranError::StorageError(format!("Redis register failed: {e}")))?;
        Ok(())
    }

    /// Deregister a node from Redis.
    pub async fn deregister_node(&self, node_id: &str) -> Result<()> {
        self.heartbeat_tracker
            .deregister(node_id)
            .await
            .map_err(|e| IfranError::StorageError(format!("Redis deregister failed: {e}")))?;
        Ok(())
    }

    /// Publish an event to Redis pub/sub for cross-instance propagation.
    pub async fn publish_event<T: serde::Serialize>(&self, topic: &str, payload: &T) -> Result<()> {
        self.pubsub
            .publish(topic, payload)
            .await
            .map_err(|e| IfranError::StorageError(format!("Redis publish failed: {e}")))?;
        Ok(())
    }

    /// Check rate limit via Redis (distributed across instances).
    pub async fn check_rate_limit(&self, key: &str) -> Result<bool> {
        self.rate_limiter
            .check(key)
            .await
            .map_err(|e| IfranError::StorageError(format!("Redis rate limit failed: {e}")))
    }
}
