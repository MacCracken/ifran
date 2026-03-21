//! OCI registry client (Ollama-compatible).
//!
//! Pulls model manifests from Docker Registry API v2 endpoints.
//! Supports both Docker Hub and custom registries.

use reqwest::Client;
use serde::Deserialize;
use ifran_types::IfranError;
use ifran_types::error::Result;

const DEFAULT_REGISTRY: &str = "https://registry.ollama.ai";

/// An OCI manifest layer descriptor.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OciLayer {
    pub digest: String,
    pub size: u64,
    pub media_type: String,
}

/// A minimal OCI image manifest (v2 schema 2).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OciManifest {
    pub schema_version: u32,
    #[serde(default)]
    pub media_type: Option<String>,
    #[serde(default)]
    pub config: Option<OciLayer>,
    #[serde(default)]
    pub layers: Vec<OciLayer>,
}

/// Client for pulling model manifests from an OCI / Docker v2 registry.
pub struct OciClient {
    client: Client,
    registry_url: String,
}

impl OciClient {
    /// Create a new OCI client pointed at the default Ollama registry.
    pub fn new(client: Client) -> Self {
        Self {
            client,
            registry_url: DEFAULT_REGISTRY.to_string(),
        }
    }

    /// Create a client pointed at a custom registry URL.
    pub fn with_registry(client: Client, registry_url: String) -> Self {
        Self {
            client,
            registry_url,
        }
    }

    /// Override the registry URL (useful for testing).
    pub fn set_registry_url(&mut self, url: String) {
        self.registry_url = url;
    }

    /// Fetch the manifest for `name:tag` (e.g. `library/llama2:latest`).
    pub async fn get_manifest(&self, name: &str, tag: &str) -> Result<OciManifest> {
        let url = format!("{}/v2/{}/manifests/{}", self.registry_url, name, tag);

        let resp = self
            .client
            .get(&url)
            .header(
                "Accept",
                "application/vnd.docker.distribution.manifest.v2+json",
            )
            .send()
            .await
            .map_err(|e| IfranError::DownloadError(format!("OCI request failed: {e}")))?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(IfranError::ModelNotFound(format!("{name}:{tag}")));
        }
        if !resp.status().is_success() {
            return Err(IfranError::DownloadError(format!(
                "OCI registry returned HTTP {}",
                resp.status()
            )));
        }

        resp.json::<OciManifest>()
            .await
            .map_err(|e| IfranError::DownloadError(format!("Failed to parse OCI manifest: {e}")))
    }

    /// Build the blob download URL for a given digest.
    pub fn blob_url(&self, name: &str, digest: &str) -> String {
        format!("{}/v2/{}/blobs/{}", self.registry_url, name, digest)
    }

    /// Find the largest layer in a manifest (typically the model weights).
    pub fn find_model_layer(manifest: &OciManifest) -> Option<&OciLayer> {
        manifest.layers.iter().max_by_key(|l| l.size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oci_layer_deserialize() {
        let json = r#"{
            "digest": "sha256:abcdef1234567890",
            "size": 4000000000,
            "mediaType": "application/vnd.ollama.image.model"
        }"#;
        let layer: OciLayer = serde_json::from_str(json).unwrap();
        assert_eq!(layer.digest, "sha256:abcdef1234567890");
        assert_eq!(layer.size, 4_000_000_000);
        assert_eq!(layer.media_type, "application/vnd.ollama.image.model");
    }

    #[test]
    fn oci_manifest_deserialize() {
        let json = r#"{
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "digest": "sha256:config123",
                "size": 500,
                "mediaType": "application/vnd.docker.container.image.v1+json"
            },
            "layers": [
                {
                    "digest": "sha256:layer1",
                    "size": 100,
                    "mediaType": "application/vnd.ollama.image.template"
                },
                {
                    "digest": "sha256:layer2",
                    "size": 4000000000,
                    "mediaType": "application/vnd.ollama.image.model"
                }
            ]
        }"#;
        let manifest: OciManifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.schema_version, 2);
        assert_eq!(manifest.layers.len(), 2);
        assert!(manifest.config.is_some());
    }

    #[test]
    fn find_model_layer_picks_largest() {
        let manifest = OciManifest {
            schema_version: 2,
            media_type: None,
            config: None,
            layers: vec![
                OciLayer {
                    digest: "sha256:small".into(),
                    size: 100,
                    media_type: "text/plain".into(),
                },
                OciLayer {
                    digest: "sha256:big".into(),
                    size: 4_000_000_000,
                    media_type: "application/vnd.ollama.image.model".into(),
                },
                OciLayer {
                    digest: "sha256:medium".into(),
                    size: 5000,
                    media_type: "text/plain".into(),
                },
            ],
        };
        let layer = OciClient::find_model_layer(&manifest).unwrap();
        assert_eq!(layer.digest, "sha256:big");
        assert_eq!(layer.size, 4_000_000_000);
    }

    #[test]
    fn find_model_layer_empty() {
        let manifest = OciManifest {
            schema_version: 2,
            media_type: None,
            config: None,
            layers: vec![],
        };
        assert!(OciClient::find_model_layer(&manifest).is_none());
    }

    #[test]
    fn blob_url_format() {
        let client = OciClient::new(Client::new());
        let url = client.blob_url("library/llama2", "sha256:abc123");
        assert_eq!(
            url,
            "https://registry.ollama.ai/v2/library/llama2/blobs/sha256:abc123"
        );
    }

    #[test]
    fn custom_registry_url() {
        let client = OciClient::with_registry(Client::new(), "https://my-registry.io".into());
        let url = client.blob_url("org/model", "sha256:def456");
        assert_eq!(
            url,
            "https://my-registry.io/v2/org/model/blobs/sha256:def456"
        );
    }

    #[tokio::test]
    async fn get_manifest_not_found() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/v2/library/nomodel/manifests/latest")
            .with_status(404)
            .create_async()
            .await;

        let client = OciClient::with_registry(Client::new(), server.url());
        let result = client.get_manifest("library/nomodel", "latest").await;
        assert!(matches!(result, Err(IfranError::ModelNotFound(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn get_manifest_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/v2/library/llama2/manifests/latest")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "schemaVersion": 2,
                    "layers": [
                        {"digest": "sha256:abc", "size": 4000000000, "mediaType": "application/vnd.ollama.image.model"}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let client = OciClient::with_registry(Client::new(), server.url());
        let manifest = client
            .get_manifest("library/llama2", "latest")
            .await
            .unwrap();
        assert_eq!(manifest.schema_version, 2);
        assert_eq!(manifest.layers.len(), 1);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn get_manifest_server_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/v2/library/llama2/manifests/latest")
            .with_status(500)
            .create_async()
            .await;

        let client = OciClient::with_registry(Client::new(), server.url());
        let result = client.get_manifest("library/llama2", "latest").await;
        assert!(matches!(result, Err(IfranError::DownloadError(_))));
        mock.assert_async().await;
    }
}
