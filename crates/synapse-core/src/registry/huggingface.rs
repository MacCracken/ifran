//! HuggingFace Hub API client.
//!
//! Resolves model repo IDs to downloadable GGUF file URLs. Supports:
//! - Listing files in a repo
//! - Filtering by format (GGUF) and quantization level
//! - Resolving the best file to download for a given quant preference

use reqwest::Client;
use serde::Deserialize;
use synapse_types::SynapseError;
use synapse_types::error::Result;

const HF_API_BASE: &str = "https://huggingface.co/api";

/// Metadata for a file in a HuggingFace repo.
#[derive(Debug, Clone, Deserialize)]
pub struct HfFile {
    #[serde(rename = "rfilename")]
    pub filename: String,
    pub size: Option<u64>,
    #[serde(default)]
    pub lfs: Option<HfLfs>,
}

/// LFS metadata (contains the SHA-256 hash).
#[derive(Debug, Clone, Deserialize)]
pub struct HfLfs {
    pub sha256: Option<String>,
    pub size: Option<u64>,
}

impl HfFile {
    /// The SHA-256 hash from LFS metadata, if available.
    pub fn sha256(&self) -> Option<&str> {
        self.lfs.as_ref()?.sha256.as_deref()
    }

    /// File size, preferring LFS size over top-level.
    pub fn file_size(&self) -> Option<u64> {
        self.lfs.as_ref().and_then(|l| l.size).or(self.size)
    }

    /// True if this file is a GGUF model file.
    pub fn is_gguf(&self) -> bool {
        self.filename.ends_with(".gguf")
    }
}

/// Model info from the HuggingFace API.
#[derive(Debug, Clone, Deserialize)]
pub struct HfModelInfo {
    #[serde(rename = "modelId")]
    pub model_id: String,
    pub sha: Option<String>,
    pub author: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub siblings: Vec<HfFile>,
}

/// HuggingFace Hub client.
pub struct HfClient {
    client: Client,
    token: Option<String>,
}

impl HfClient {
    /// Create a new client, optionally with an API token.
    pub fn new(client: Client, token: Option<String>) -> Self {
        Self { client, token }
    }

    /// Create a client that reads the token from `HF_TOKEN` env var.
    pub fn from_env(client: Client) -> Self {
        let token = std::env::var("HF_TOKEN").ok();
        Self::new(client, token)
    }

    /// Fetch model info including file listing.
    pub async fn model_info(&self, repo_id: &str) -> Result<HfModelInfo> {
        let url = format!("{HF_API_BASE}/models/{repo_id}");
        let mut req = self.client.get(&url);
        if let Some(ref token) = self.token {
            req = req.bearer_auth(token);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| SynapseError::DownloadError(e.to_string()))?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(SynapseError::ModelNotFound(repo_id.to_string()));
        }
        if !resp.status().is_success() {
            return Err(SynapseError::DownloadError(format!(
                "HuggingFace API returned HTTP {}",
                resp.status()
            )));
        }

        resp.json::<HfModelInfo>()
            .await
            .map_err(|e| SynapseError::DownloadError(e.to_string()))
    }

    /// List all GGUF files in a repo.
    pub async fn list_gguf_files(&self, repo_id: &str) -> Result<Vec<HfFile>> {
        let info = self.model_info(repo_id).await?;
        Ok(info.siblings.into_iter().filter(|f| f.is_gguf()).collect())
    }

    /// Find the best GGUF file matching a quantization substring (e.g. "Q4_K_M").
    ///
    /// Matching is case-insensitive on the filename.
    pub async fn resolve_gguf(&self, repo_id: &str, quant_filter: Option<&str>) -> Result<HfFile> {
        let files = self.list_gguf_files(repo_id).await?;

        if files.is_empty() {
            return Err(SynapseError::ModelNotFound(format!(
                "No GGUF files found in {repo_id}"
            )));
        }

        if let Some(quant) = quant_filter {
            let quant_upper = quant.to_uppercase();
            if let Some(matched) = files
                .iter()
                .find(|f| f.filename.to_uppercase().contains(&quant_upper))
            {
                return Ok(matched.clone());
            }
            return Err(SynapseError::ModelNotFound(format!(
                "No GGUF file matching '{quant}' in {repo_id}. Available: {}",
                files
                    .iter()
                    .map(|f| f.filename.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }

        // No filter — return the first (or a middle-of-the-road) GGUF file
        Ok(files.into_iter().next().unwrap())
    }

    /// Build the download URL for a file in a repo.
    pub fn download_url(repo_id: &str, filename: &str) -> String {
        format!("https://huggingface.co/{repo_id}/resolve/main/{filename}")
    }
}

/// Search for models on HuggingFace Hub.
pub async fn search(client: &Client, query: &str, limit: usize) -> Result<Vec<HfModelInfo>> {
    let url = format!(
        "{HF_API_BASE}/models?search={}&limit={}&filter=gguf",
        urlencoded(query),
        limit
    );

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| SynapseError::DownloadError(e.to_string()))?;

    resp.json::<Vec<HfModelInfo>>()
        .await
        .map_err(|e| SynapseError::DownloadError(e.to_string()))
}

/// Minimal percent-encoding for query params.
fn urlencoded(s: &str) -> String {
    s.replace(' ', "%20").replace('/', "%2F")
}
