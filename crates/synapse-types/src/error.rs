use thiserror::Error;

#[derive(Debug, Error)]
pub enum SynapseError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Backend not found: {0}")]
    BackendNotFound(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Download failed: {0}")]
    DownloadError(String),

    #[error("Integrity check failed: expected {expected}, got {actual}")]
    IntegrityError { expected: String, actual: String },

    #[error("Insufficient memory: need {required_mb}MB, available {available_mb}MB")]
    InsufficientMemory { required_mb: u64, available_mb: u64 },

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Bridge error: {0}")]
    BridgeError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Hardware error: {0}")]
    HardwareError(String),

    #[error("Evaluation error: {0}")]
    EvalError(String),

    #[error("Marketplace error: {0}")]
    MarketplaceError(String),

    #[error("Distributed training error: {0}")]
    DistributedError(String),

    #[error("RAG pipeline error: {0}")]
    RagError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Tenant not found: {0}")]
    TenantNotFound(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, SynapseError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_model_not_found() {
        let e = SynapseError::ModelNotFound("llama-7b".into());
        assert_eq!(e.to_string(), "Model not found: llama-7b");
    }

    #[test]
    fn error_display_backend_not_found() {
        let e = SynapseError::BackendNotFound("vllm".into());
        assert_eq!(e.to_string(), "Backend not found: vllm");
    }

    #[test]
    fn error_display_integrity() {
        let e = SynapseError::IntegrityError {
            expected: "abc".into(),
            actual: "def".into(),
        };
        assert_eq!(
            e.to_string(),
            "Integrity check failed: expected abc, got def"
        );
    }

    #[test]
    fn error_display_insufficient_memory() {
        let e = SynapseError::InsufficientMemory {
            required_mb: 8000,
            available_mb: 4000,
        };
        assert_eq!(
            e.to_string(),
            "Insufficient memory: need 8000MB, available 4000MB"
        );
    }

    #[test]
    fn error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let e: SynapseError = io_err.into();
        assert!(e.to_string().contains("file missing"));
    }

    #[test]
    fn all_error_variants_display() {
        let errors: Vec<SynapseError> = vec![
            SynapseError::ModelNotFound("x".into()),
            SynapseError::BackendNotFound("x".into()),
            SynapseError::BackendError("x".into()),
            SynapseError::DownloadError("x".into()),
            SynapseError::IntegrityError {
                expected: "a".into(),
                actual: "b".into(),
            },
            SynapseError::InsufficientMemory {
                required_mb: 1,
                available_mb: 0,
            },
            SynapseError::TrainingError("x".into()),
            SynapseError::BridgeError("x".into()),
            SynapseError::ConfigError("x".into()),
            SynapseError::StorageError("x".into()),
            SynapseError::HardwareError("x".into()),
            SynapseError::EvalError("x".into()),
            SynapseError::MarketplaceError("x".into()),
            SynapseError::DistributedError("x".into()),
            SynapseError::RagError("x".into()),
            SynapseError::ValidationError("x".into()),
            SynapseError::TenantNotFound("x".into()),
            SynapseError::Unauthorized("x".into()),
            SynapseError::Other("x".into()),
        ];
        for e in errors {
            assert!(!e.to_string().is_empty());
        }
    }

    #[test]
    fn error_display_backend_error() {
        let e = SynapseError::BackendError("timeout".into());
        assert_eq!(e.to_string(), "Backend error: timeout");
    }

    #[test]
    fn error_display_download() {
        let e = SynapseError::DownloadError("connection refused".into());
        assert_eq!(e.to_string(), "Download failed: connection refused");
    }

    #[test]
    fn error_display_training() {
        let e = SynapseError::TrainingError("OOM".into());
        assert_eq!(e.to_string(), "Training error: OOM");
    }

    #[test]
    fn error_display_bridge() {
        let e = SynapseError::BridgeError("disconnected".into());
        assert_eq!(e.to_string(), "Bridge error: disconnected");
    }

    #[test]
    fn error_display_config() {
        let e = SynapseError::ConfigError("missing field".into());
        assert_eq!(e.to_string(), "Configuration error: missing field");
    }

    #[test]
    fn error_display_storage() {
        let e = SynapseError::StorageError("db locked".into());
        assert_eq!(e.to_string(), "Storage error: db locked");
    }

    #[test]
    fn error_display_hardware() {
        let e = SynapseError::HardwareError("no GPU".into());
        assert_eq!(e.to_string(), "Hardware error: no GPU");
    }

    #[test]
    fn error_display_eval() {
        let e = SynapseError::EvalError("benchmark failed".into());
        assert_eq!(e.to_string(), "Evaluation error: benchmark failed");
    }

    #[test]
    fn error_display_marketplace() {
        let e = SynapseError::MarketplaceError("not published".into());
        assert_eq!(e.to_string(), "Marketplace error: not published");
    }

    #[test]
    fn error_display_distributed() {
        let e = SynapseError::DistributedError("worker lost".into());
        assert_eq!(e.to_string(), "Distributed training error: worker lost");
    }

    #[test]
    fn error_display_rag() {
        let e = SynapseError::RagError("embedding failed".into());
        assert_eq!(e.to_string(), "RAG pipeline error: embedding failed");
    }

    #[test]
    fn error_display_tenant_not_found() {
        let e = SynapseError::TenantNotFound("acme".into());
        assert_eq!(e.to_string(), "Tenant not found: acme");
    }

    #[test]
    fn error_display_unauthorized() {
        let e = SynapseError::Unauthorized("bad token".into());
        assert_eq!(e.to_string(), "Unauthorized: bad token");
    }

    #[test]
    fn error_display_other() {
        let e = SynapseError::Other("unknown".into());
        assert_eq!(e.to_string(), "unknown");
    }

    #[test]
    fn error_debug_format() {
        let e = SynapseError::ModelNotFound("test".into());
        let debug = format!("{:?}", e);
        assert!(debug.contains("ModelNotFound"));
    }

    #[test]
    fn error_from_io_kind_preserved() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e: SynapseError = io_err.into();
        assert!(e.to_string().contains("denied"));
    }

    #[test]
    fn error_integrity_includes_both_hashes() {
        let e = SynapseError::IntegrityError {
            expected: "aaa111".into(),
            actual: "bbb222".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("aaa111"));
        assert!(msg.contains("bbb222"));
    }

    #[test]
    fn error_insufficient_memory_includes_values() {
        let e = SynapseError::InsufficientMemory {
            required_mb: 16384,
            available_mb: 8192,
        };
        let msg = e.to_string();
        assert!(msg.contains("16384"));
        assert!(msg.contains("8192"));
    }
}
