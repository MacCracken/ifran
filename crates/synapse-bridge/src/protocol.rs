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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_config_defaults() {
        let cfg = ProtocolConfig::default();
        assert_eq!(cfg.heartbeat_interval, Duration::from_secs(10));
        assert_eq!(cfg.heartbeat_timeout, Duration::from_secs(30));
        assert_eq!(cfg.reconnect_delay, Duration::from_secs(5));
        assert_eq!(cfg.max_reconnect_attempts, 10);
    }

    #[test]
    fn connection_state_equality() {
        assert_eq!(ConnectionState::Disconnected, ConnectionState::Disconnected);
        assert_ne!(ConnectionState::Connected, ConnectionState::Degraded);
    }

    #[test]
    fn heartbeat_serialization() {
        let hb = Heartbeat {
            instance_id: "test-1".into(),
            timestamp: 1234567890,
            loaded_models: 3,
            gpu_memory_free_mb: 8192,
            active_training_jobs: 1,
        };
        let json = serde_json::to_string(&hb).unwrap();
        let parsed: Heartbeat = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.instance_id, "test-1");
        assert_eq!(parsed.loaded_models, 3);
    }

    #[test]
    fn capabilities_serialization() {
        let caps = Capabilities {
            instance_id: "node-1".into(),
            version: "2026.3.10".into(),
            gpu_count: 2,
            total_gpu_memory_mb: 16384,
            supported_methods: vec!["lora".into(), "dpo".into()],
        };
        let json = serde_json::to_string(&caps).unwrap();
        let parsed: Capabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.gpu_count, 2);
        assert_eq!(parsed.supported_methods.len(), 2);
    }

    #[test]
    fn connection_state_all_variants_distinct() {
        let states = [
            ConnectionState::Disconnected,
            ConnectionState::Connecting,
            ConnectionState::Connected,
            ConnectionState::Degraded,
        ];
        for (i, a) in states.iter().enumerate() {
            for (j, b) in states.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn connection_state_is_copy() {
        let state = ConnectionState::Connected;
        let copy = state;
        assert_eq!(state, copy);
    }

    #[test]
    fn protocol_config_custom_values() {
        let cfg = ProtocolConfig {
            heartbeat_interval: Duration::from_secs(5),
            heartbeat_timeout: Duration::from_secs(15),
            reconnect_delay: Duration::from_secs(2),
            max_reconnect_attempts: 3,
        };
        assert_eq!(cfg.heartbeat_interval, Duration::from_secs(5));
        assert_eq!(cfg.max_reconnect_attempts, 3);
    }

    #[test]
    fn heartbeat_roundtrip_preserves_all_fields() {
        let hb = Heartbeat {
            instance_id: "node-42".into(),
            timestamp: 9999999999,
            loaded_models: 0,
            gpu_memory_free_mb: 0,
            active_training_jobs: 10,
        };
        let json = serde_json::to_string(&hb).unwrap();
        let parsed: Heartbeat = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.instance_id, "node-42");
        assert_eq!(parsed.timestamp, 9999999999);
        assert_eq!(parsed.loaded_models, 0);
        assert_eq!(parsed.gpu_memory_free_mb, 0);
        assert_eq!(parsed.active_training_jobs, 10);
    }

    #[test]
    fn capabilities_roundtrip_preserves_all_fields() {
        let caps = Capabilities {
            instance_id: "synapse-node".into(),
            version: "2026.3.12".into(),
            gpu_count: 0,
            total_gpu_memory_mb: 0,
            supported_methods: vec![],
        };
        let json = serde_json::to_string(&caps).unwrap();
        let parsed: Capabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.instance_id, "synapse-node");
        assert_eq!(parsed.version, "2026.3.12");
        assert_eq!(parsed.gpu_count, 0);
        assert!(parsed.supported_methods.is_empty());
    }

    #[test]
    fn connection_state_debug_format() {
        assert_eq!(
            format!("{:?}", ConnectionState::Disconnected),
            "Disconnected"
        );
        assert_eq!(format!("{:?}", ConnectionState::Connecting), "Connecting");
        assert_eq!(format!("{:?}", ConnectionState::Connected), "Connected");
        assert_eq!(format!("{:?}", ConnectionState::Degraded), "Degraded");
    }
}
