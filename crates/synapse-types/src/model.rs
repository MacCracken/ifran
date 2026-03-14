use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a model in the local catalog.
pub type ModelId = Uuid;

/// Supported model file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
    Onnx,
    TensorRt,
    PyTorch,
    Bin,
}

/// Quantization level for GGUF models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantLevel {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q6K,
    Q5KM,
    Q5KS,
    Q4KM,
    Q4KS,
    Q4_0,
    Q3KM,
    Q3KS,
    Q2K,
    Iq4Xs,
    Iq3Xxs,
    None,
}

/// Model metadata stored in the local catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: ModelId,
    pub name: String,
    pub repo_id: Option<String>,
    pub format: ModelFormat,
    pub quant: QuantLevel,
    pub size_bytes: u64,
    pub parameter_count: Option<u64>,
    pub architecture: Option<String>,
    pub license: Option<String>,
    pub local_path: String,
    pub sha256: Option<String>,
    pub pulled_at: chrono::DateTime<chrono::Utc>,
}

/// Manifest describing a model to be loaded by a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub info: ModelInfo,
    pub context_length: Option<u32>,
    pub gpu_layers: Option<u32>,
    pub tensor_split: Option<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_format_serde_roundtrip() {
        let formats = [
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            ModelFormat::Onnx,
            ModelFormat::TensorRt,
            ModelFormat::PyTorch,
            ModelFormat::Bin,
        ];
        for fmt in &formats {
            let json = serde_json::to_string(fmt).unwrap();
            let back: ModelFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(*fmt, back);
        }
    }

    #[test]
    fn model_format_json_values() {
        assert_eq!(
            serde_json::to_string(&ModelFormat::Gguf).unwrap(),
            "\"gguf\""
        );
        assert_eq!(
            serde_json::to_string(&ModelFormat::SafeTensors).unwrap(),
            "\"safetensors\""
        );
        assert_eq!(
            serde_json::to_string(&ModelFormat::Onnx).unwrap(),
            "\"onnx\""
        );
        assert_eq!(
            serde_json::to_string(&ModelFormat::TensorRt).unwrap(),
            "\"tensorrt\""
        );
        assert_eq!(
            serde_json::to_string(&ModelFormat::PyTorch).unwrap(),
            "\"pytorch\""
        );
        assert_eq!(serde_json::to_string(&ModelFormat::Bin).unwrap(), "\"bin\"");
    }

    #[test]
    fn quant_level_serde_roundtrip() {
        let quants = [
            QuantLevel::F32,
            QuantLevel::F16,
            QuantLevel::Bf16,
            QuantLevel::Q8_0,
            QuantLevel::Q6K,
            QuantLevel::Q5KM,
            QuantLevel::Q5KS,
            QuantLevel::Q4KM,
            QuantLevel::Q4KS,
            QuantLevel::Q4_0,
            QuantLevel::Q3KM,
            QuantLevel::Q3KS,
            QuantLevel::Q2K,
            QuantLevel::Iq4Xs,
            QuantLevel::Iq3Xxs,
            QuantLevel::None,
        ];
        for q in &quants {
            let json = serde_json::to_string(q).unwrap();
            let back: QuantLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(*q, back);
        }
    }

    #[test]
    fn model_info_serde_roundtrip() {
        let info = ModelInfo {
            id: Uuid::new_v4(),
            name: "test-model".into(),
            repo_id: Some("org/model".into()),
            format: ModelFormat::Gguf,
            quant: QuantLevel::Q4KM,
            size_bytes: 4_000_000_000,
            parameter_count: Some(7_000_000_000),
            architecture: Some("llama".into()),
            license: Some("MIT".into()),
            local_path: "/models/test.gguf".into(),
            sha256: Some("abc123".into()),
            pulled_at: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: ModelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info.id, back.id);
        assert_eq!(info.name, back.name);
        assert_eq!(info.format, back.format);
        assert_eq!(info.quant, back.quant);
        assert_eq!(info.size_bytes, back.size_bytes);
    }

    #[test]
    fn model_manifest_serde_roundtrip() {
        let manifest = ModelManifest {
            info: ModelInfo {
                id: Uuid::new_v4(),
                name: "test".into(),
                repo_id: None,
                format: ModelFormat::Gguf,
                quant: QuantLevel::None,
                size_bytes: 100,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/test.gguf".into(),
                sha256: None,
                pulled_at: chrono::Utc::now(),
            },
            context_length: Some(4096),
            gpu_layers: Some(32),
            tensor_split: Some(vec![0.5, 0.5]),
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let back: ModelManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest.context_length, back.context_length);
        assert_eq!(manifest.gpu_layers, back.gpu_layers);
        assert_eq!(manifest.tensor_split, back.tensor_split);
    }

    #[test]
    fn model_format_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ModelFormat::Gguf);
        set.insert(ModelFormat::Gguf);
        assert_eq!(set.len(), 1);
        set.insert(ModelFormat::Onnx);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn quant_level_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(QuantLevel::Q4KM);
        set.insert(QuantLevel::Q4KM);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn model_format_invalid_json() {
        let result = serde_json::from_str::<ModelFormat>("\"invalid\"");
        assert!(result.is_err());
    }

    #[test]
    fn quant_level_invalid_json() {
        let result = serde_json::from_str::<QuantLevel>("\"invalid\"");
        assert!(result.is_err());
    }

    #[test]
    fn model_info_missing_required_fields() {
        let result = serde_json::from_str::<ModelInfo>(r#"{"name":"test"}"#);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_model_format() -> impl Strategy<Value = ModelFormat> {
        prop_oneof![
            Just(ModelFormat::Gguf),
            Just(ModelFormat::SafeTensors),
            Just(ModelFormat::Onnx),
            Just(ModelFormat::TensorRt),
            Just(ModelFormat::PyTorch),
            Just(ModelFormat::Bin),
        ]
    }

    fn arb_quant_level() -> impl Strategy<Value = QuantLevel> {
        prop_oneof![
            Just(QuantLevel::F32),
            Just(QuantLevel::F16),
            Just(QuantLevel::Bf16),
            Just(QuantLevel::Q8_0),
            Just(QuantLevel::Q6K),
            Just(QuantLevel::Q5KM),
            Just(QuantLevel::Q5KS),
            Just(QuantLevel::Q4KM),
            Just(QuantLevel::Q4KS),
            Just(QuantLevel::Q4_0),
            Just(QuantLevel::Q3KM),
            Just(QuantLevel::Q3KS),
            Just(QuantLevel::Q2K),
            Just(QuantLevel::Iq4Xs),
            Just(QuantLevel::Iq3Xxs),
            Just(QuantLevel::None),
        ]
    }

    proptest! {
        #[test]
        fn model_format_roundtrips(fmt in arb_model_format()) {
            let json = serde_json::to_string(&fmt).unwrap();
            let back: ModelFormat = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(fmt, back);
        }

        #[test]
        fn quant_level_roundtrips(q in arb_quant_level()) {
            let json = serde_json::to_string(&q).unwrap();
            let back: QuantLevel = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(q, back);
        }

        #[test]
        fn model_info_roundtrips(
            name in "[a-z][a-z0-9-]{0,20}",
            size in 0u64..100_000_000_000u64,
            fmt in arb_model_format(),
            q in arb_quant_level(),
        ) {
            let info = ModelInfo {
                id: Uuid::new_v4(),
                name: name.clone(),
                repo_id: None,
                format: fmt,
                quant: q,
                size_bytes: size,
                parameter_count: None,
                architecture: None,
                license: None,
                local_path: "/tmp/test".into(),
                sha256: None,
                pulled_at: chrono::Utc::now(),
            };
            let json = serde_json::to_string(&info).unwrap();
            let back: ModelInfo = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(info.name, back.name);
            prop_assert_eq!(info.format, back.format);
            prop_assert_eq!(info.quant, back.quant);
            prop_assert_eq!(info.size_bytes, back.size_bytes);
        }
    }
}
