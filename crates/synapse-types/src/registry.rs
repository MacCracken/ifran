use serde::{Deserialize, Serialize};

/// Source from which a model can be pulled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegistrySource {
    HuggingFace { repo_id: String },
    Oci { registry: String, repository: String },
    DirectUrl { url: String },
    LocalPath { path: String },
}

/// Status of an active download.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadStatus {
    pub source: RegistrySource,
    pub total_bytes: Option<u64>,
    pub downloaded_bytes: u64,
    pub speed_bytes_per_sec: u64,
    pub state: DownloadState,
}

/// Download state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadState {
    Queued,
    Downloading,
    Verifying,
    Complete,
    Failed,
    Paused,
}
