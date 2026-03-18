use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendId(pub String);

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Hardware accelerator type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceleratorType {
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    Cpu,
}

/// Capabilities reported by a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub accelerators: Vec<AcceleratorType>,
    pub max_context_length: Option<u32>,
    pub supports_streaming: bool,
    pub supports_embeddings: bool,
    pub supports_vision: bool,
    pub locality: BackendLocality,
}

/// Whether a backend runs locally or calls a remote API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendLocality {
    /// Runs on local hardware, data never leaves the machine.
    Local,
    /// Calls an external API.
    Remote,
}

/// Device configuration for model loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub accelerator: AcceleratorType,
    pub device_ids: Vec<u32>,
    pub memory_limit_mb: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_id_display() {
        let id = BackendId("llamacpp".into());
        assert_eq!(id.to_string(), "llamacpp");
    }

    #[test]
    fn backend_id_equality() {
        let a = BackendId("ollama".into());
        let b = BackendId("ollama".into());
        let c = BackendId("vllm".into());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn backend_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BackendId("a".into()));
        set.insert(BackendId("a".into()));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn accelerator_type_serde_roundtrip() {
        let types = [
            AcceleratorType::Cuda,
            AcceleratorType::Rocm,
            AcceleratorType::Metal,
            AcceleratorType::Vulkan,
            AcceleratorType::Cpu,
        ];
        for t in &types {
            let json = serde_json::to_string(t).unwrap();
            let back: AcceleratorType = serde_json::from_str(&json).unwrap();
            assert_eq!(*t, back);
        }
    }

    #[test]
    fn backend_capabilities_serde() {
        let caps = BackendCapabilities {
            accelerators: vec![AcceleratorType::Cuda, AcceleratorType::Cpu],
            max_context_length: Some(8192),
            supports_streaming: true,
            supports_embeddings: false,
            supports_vision: true,
            locality: BackendLocality::Local,
        };
        let json = serde_json::to_string(&caps).unwrap();
        let back: BackendCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(back.accelerators.len(), 2);
        assert_eq!(back.max_context_length, Some(8192));
        assert!(back.supports_streaming);
        assert!(!back.supports_embeddings);
        assert!(back.supports_vision);
    }

    #[test]
    fn device_config_serde() {
        let cfg = DeviceConfig {
            accelerator: AcceleratorType::Cuda,
            device_ids: vec![0, 1],
            memory_limit_mb: Some(16384),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: DeviceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.device_ids, vec![0, 1]);
        assert_eq!(back.memory_limit_mb, Some(16384));
    }

    #[test]
    fn backend_id_serde_roundtrip() {
        let id = BackendId("test-backend".into());
        let json = serde_json::to_string(&id).unwrap();
        let back: BackendId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn device_config_no_memory_limit() {
        let cfg = DeviceConfig {
            accelerator: AcceleratorType::Cpu,
            device_ids: vec![],
            memory_limit_mb: None,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: DeviceConfig = serde_json::from_str(&json).unwrap();
        assert!(back.memory_limit_mb.is_none());
        assert!(back.device_ids.is_empty());
    }

    #[test]
    fn accelerator_type_invalid_json() {
        let result = serde_json::from_str::<AcceleratorType>("\"invalid\"");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_accelerator() -> impl Strategy<Value = AcceleratorType> {
        prop_oneof![
            Just(AcceleratorType::Cuda),
            Just(AcceleratorType::Rocm),
            Just(AcceleratorType::Metal),
            Just(AcceleratorType::Vulkan),
            Just(AcceleratorType::Cpu),
        ]
    }

    proptest! {
        #[test]
        fn accelerator_type_roundtrips(acc in arb_accelerator()) {
            let json = serde_json::to_string(&acc).unwrap();
            let back: AcceleratorType = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(acc, back);
        }

        #[test]
        fn backend_id_roundtrips(name in "[a-z][a-z0-9-]{0,30}") {
            let id = BackendId(name.clone());
            let json = serde_json::to_string(&id).unwrap();
            let back: BackendId = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(id, back);
        }

        #[test]
        fn device_config_roundtrips(
            acc in arb_accelerator(),
            n_devices in 0usize..8,
            mem in proptest::option::of(0u64..100_000),
        ) {
            let device_ids: Vec<u32> = (0..n_devices as u32).collect();
            let cfg = DeviceConfig {
                accelerator: acc,
                device_ids: device_ids.clone(),
                memory_limit_mb: mem,
            };
            let json = serde_json::to_string(&cfg).unwrap();
            let back: DeviceConfig = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(cfg.device_ids, back.device_ids);
            prop_assert_eq!(cfg.memory_limit_mb, back.memory_limit_mb);
        }
    }
}
