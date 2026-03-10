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

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, SynapseError>;
