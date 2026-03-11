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
            SynapseError::Other("x".into()),
        ];
        for e in errors {
            assert!(!e.to_string().is_empty());
        }
    }
}
