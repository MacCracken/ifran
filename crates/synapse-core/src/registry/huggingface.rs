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
    base_url: String,
}

impl HfClient {
    /// Create a new client, optionally with an API token.
    pub fn new(client: Client, token: Option<String>) -> Self {
        Self {
            client,
            token,
            base_url: HF_API_BASE.to_string(),
        }
    }

    /// Create a client that reads the token from `HF_TOKEN` env var.
    pub fn from_env(client: Client) -> Self {
        let token = std::env::var("HF_TOKEN").ok();
        Self::new(client, token)
    }

    /// Override the API base URL (useful for testing).
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    /// Fetch model info including file listing.
    pub async fn model_info(&self, repo_id: &str) -> Result<HfModelInfo> {
        let url = format!("{}/models/{repo_id}", self.base_url);
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

        // No filter — return the first GGUF file
        // Safety: files is guaranteed non-empty by the check above
        files.into_iter().next().ok_or_else(|| {
            SynapseError::ModelNotFound(format!("No GGUF files found in {repo_id}"))
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    // -- Pure function / data structure tests --

    #[test]
    fn hf_file_is_gguf() {
        let f = HfFile {
            filename: "model-Q4_K_M.gguf".into(),
            size: Some(1000),
            lfs: None,
        };
        assert!(f.is_gguf());

        let f2 = HfFile {
            filename: "config.json".into(),
            size: Some(100),
            lfs: None,
        };
        assert!(!f2.is_gguf());
    }

    #[test]
    fn hf_file_sha256_from_lfs() {
        let f = HfFile {
            filename: "model.gguf".into(),
            size: None,
            lfs: Some(HfLfs {
                sha256: Some("abc123".into()),
                size: Some(5000),
            }),
        };
        assert_eq!(f.sha256(), Some("abc123"));
    }

    #[test]
    fn hf_file_sha256_none_without_lfs() {
        let f = HfFile {
            filename: "model.gguf".into(),
            size: Some(1000),
            lfs: None,
        };
        assert_eq!(f.sha256(), None);
    }

    #[test]
    fn hf_file_size_prefers_lfs() {
        let f = HfFile {
            filename: "model.gguf".into(),
            size: Some(100),
            lfs: Some(HfLfs {
                sha256: None,
                size: Some(5000),
            }),
        };
        assert_eq!(f.file_size(), Some(5000));
    }

    #[test]
    fn hf_file_size_falls_back_to_top_level() {
        let f = HfFile {
            filename: "model.gguf".into(),
            size: Some(100),
            lfs: Some(HfLfs {
                sha256: None,
                size: None,
            }),
        };
        assert_eq!(f.file_size(), Some(100));
    }

    #[test]
    fn hf_file_size_none() {
        let f = HfFile {
            filename: "model.gguf".into(),
            size: None,
            lfs: None,
        };
        assert_eq!(f.file_size(), None);
    }

    #[test]
    fn download_url_format() {
        let url = HfClient::download_url("TheBloke/Llama-2-7B-GGUF", "llama-2-7b.Q4_K_M.gguf");
        assert_eq!(
            url,
            "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
        );
    }

    #[test]
    fn urlencoded_spaces_and_slashes() {
        assert_eq!(urlencoded("hello world"), "hello%20world");
        assert_eq!(urlencoded("a/b"), "a%2Fb");
        assert_eq!(urlencoded("no special"), "no%20special");
    }

    #[test]
    fn hf_model_info_deserialize() {
        let json = r#"{
            "modelId": "test-org/test-model",
            "sha": "abc123",
            "author": "test-org",
            "tags": ["gguf", "llama"],
            "siblings": [
                {"rfilename": "model.gguf", "size": 1000},
                {"rfilename": "config.json", "size": 50}
            ]
        }"#;
        let info: HfModelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.model_id, "test-org/test-model");
        assert_eq!(info.sha, Some("abc123".into()));
        assert_eq!(info.author, Some("test-org".into()));
        assert_eq!(info.tags, vec!["gguf", "llama"]);
        assert_eq!(info.siblings.len(), 2);
        assert!(info.siblings[0].is_gguf());
        assert!(!info.siblings[1].is_gguf());
    }

    #[test]
    fn hf_model_info_deserialize_minimal() {
        let json = r#"{"modelId": "org/model"}"#;
        let info: HfModelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.model_id, "org/model");
        assert_eq!(info.sha, None);
        assert_eq!(info.author, None);
        assert!(info.tags.is_empty());
        assert!(info.siblings.is_empty());
    }

    #[test]
    fn hf_file_with_lfs_deserialize() {
        let json = r#"{
            "rfilename": "model.gguf",
            "size": 1000,
            "lfs": {"sha256": "deadbeef", "size": 5000}
        }"#;
        let f: HfFile = serde_json::from_str(json).unwrap();
        assert_eq!(f.filename, "model.gguf");
        assert_eq!(f.sha256(), Some("deadbeef"));
        assert_eq!(f.file_size(), Some(5000));
    }

    // -- Mock HTTP tests --

    #[tokio::test]
    async fn model_info_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/models/test-org/test-model")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "modelId": "test-org/test-model",
                    "sha": "abc",
                    "siblings": [
                        {"rfilename": "model-Q4_K_M.gguf", "size": 4000000000},
                        {"rfilename": "model-Q8_0.gguf", "size": 8000000000},
                        {"rfilename": "config.json", "size": 200}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let client = HfClient::new(Client::new(), None).with_base_url(server.url());
        let info = client.model_info("test-org/test-model").await.unwrap();
        assert_eq!(info.model_id, "test-org/test-model");
        assert_eq!(info.siblings.len(), 3);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn model_info_not_found() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/models/nonexistent/model")
            .with_status(404)
            .create_async()
            .await;

        let client = HfClient::new(Client::new(), None).with_base_url(server.url());
        let result = client.model_info("nonexistent/model").await;
        assert!(matches!(result, Err(SynapseError::ModelNotFound(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn model_info_server_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/models/org/model")
            .with_status(500)
            .create_async()
            .await;

        let client = HfClient::new(Client::new(), None).with_base_url(server.url());
        let result = client.model_info("org/model").await;
        assert!(matches!(result, Err(SynapseError::DownloadError(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn list_gguf_files_filters() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/models/org/model")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "modelId": "org/model",
                    "siblings": [
                        {"rfilename": "model.gguf", "size": 1000},
                        {"rfilename": "config.json", "size": 50},
                        {"rfilename": "weights.safetensors", "size": 2000}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let client = HfClient::new(Client::new(), None).with_base_url(server.url());
        let files = client.list_gguf_files("org/model").await.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].filename, "model.gguf");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn resolve_gguf_with_quant_filter() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/models/org/model")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "modelId": "org/model",
                    "siblings": [
                        {"rfilename": "model-Q4_K_M.gguf", "size": 4000},
                        {"rfilename": "model-Q8_0.gguf", "size": 8000}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let client = HfClient::new(Client::new(), None).with_base_url(server.url());
        let file = client
            .resolve_gguf("org/model", Some("q4_k_m"))
            .await
            .unwrap();
        assert_eq!(file.filename, "model-Q4_K_M.gguf");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn resolve_gguf_no_filter_returns_first() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/models/org/model")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "modelId": "org/model",
                    "siblings": [
                        {"rfilename": "model-Q4_K_M.gguf", "size": 4000},
                        {"rfilename": "model-Q8_0.gguf", "size": 8000}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let client = HfClient::new(Client::new(), None).with_base_url(server.url());
        let file = client.resolve_gguf("org/model", None).await.unwrap();
        assert_eq!(file.filename, "model-Q4_K_M.gguf");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn resolve_gguf_no_matching_quant() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/models/org/model")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "modelId": "org/model",
                    "siblings": [
                        {"rfilename": "model-Q4_K_M.gguf", "size": 4000}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let client = HfClient::new(Client::new(), None).with_base_url(server.url());
        let result = client.resolve_gguf("org/model", Some("Q8_0")).await;
        assert!(matches!(result, Err(SynapseError::ModelNotFound(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn resolve_gguf_no_gguf_files() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/models/org/model")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "modelId": "org/model",
                    "siblings": [
                        {"rfilename": "config.json", "size": 50}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let client = HfClient::new(Client::new(), None).with_base_url(server.url());
        let result = client.resolve_gguf("org/model", None).await;
        assert!(matches!(result, Err(SynapseError::ModelNotFound(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn model_info_with_auth_token() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/models/org/private-model")
            .match_header("authorization", "Bearer test-token-123")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"modelId": "org/private-model", "siblings": []}"#)
            .create_async()
            .await;

        let client =
            HfClient::new(Client::new(), Some("test-token-123".into())).with_base_url(server.url());
        let info = client.model_info("org/private-model").await.unwrap();
        assert_eq!(info.model_id, "org/private-model");
        mock.assert_async().await;
    }
}
