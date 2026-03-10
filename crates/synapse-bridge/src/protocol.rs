//! Wire protocol definitions for Synapse↔SY communication.
//!
//! Defines the heartbeat, connection state, and message types
//! independent of the gRPC transport layer.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Connection state between Synapse and SecureYeoman.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Degraded,
}

/// Configuration for the bridge protocol.
#[derive(Debug, Clone)]
pub struct ProtocolConfig {
    pub heartbeat_interval: Duration,
    pub heartbeat_timeout: Duration,
    pub reconnect_delay: Duration,
    pub max_reconnect_attempts: u32,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(10),
            heartbeat_timeout: Duration::from_secs(30),
            reconnect_delay: Duration::from_secs(5),
            max_reconnect_attempts: 10,
        }
    }
}

/// A heartbeat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heartbeat {
    pub instance_id: String,
    pub timestamp: i64,
    pub loaded_models: u32,
    pub gpu_memory_free_mb: u64,
    pub active_training_jobs: u32,
}

/// Capability announcement sent on connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    pub instance_id: String,
    pub version: String,
    pub gpu_count: u32,
    pub total_gpu_memory_mb: u64,
    pub supported_methods: Vec<String>,
}
