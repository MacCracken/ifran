//! Direct URL download.
//!
//! Resolves a direct URL to a downloadable resource, performing HEAD requests
//! to determine file size and content type before downloading.

use reqwest::Client;
use ifran_types::IfranError;
use ifran_types::error::Result;

/// Metadata about a direct download URL.
#[derive(Debug, Clone)]
pub struct DownloadInfo {
    /// The resolved URL (after any redirects from HEAD).
    pub url: String,
    /// Content length in bytes, if reported by the server.
    pub content_length: Option<u64>,
    /// Content type header value, if present.
    pub content_type: Option<String>,
    /// The inferred filename from the URL path.
    pub filename: String,
    /// Whether the server supports range requests.
    pub accepts_ranges: bool,
}

/// Resolve a URL by performing a HEAD request to gather metadata.
pub async fn resolve(client: &Client, url: &str) -> Result<DownloadInfo> {
    let resp = client
        .head(url)
        .send()
        .await
        .map_err(|e| IfranError::DownloadError(format!("HEAD request failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(IfranError::DownloadError(format!(
            "URL returned HTTP {}",
            resp.status()
        )));
    }

    let content_length = resp
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());

    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let accepts_ranges = resp
        .headers()
        .get(reqwest::header::ACCEPT_RANGES)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("bytes"));

    let final_url = resp.url().to_string();
    let filename = extract_filename(&final_url);

    Ok(DownloadInfo {
        url: final_url,
        content_length,
        content_type,
        filename,
        accepts_ranges,
    })
}

/// Extract a filename from a URL path.
///
/// Falls back to "download" if no path segment looks like a filename.
fn extract_filename(url: &str) -> String {
    url.split('?')
        .next()
        .unwrap_or(url)
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty() && s.contains('.'))
        .unwrap_or("download")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_filename_from_url() {
        assert_eq!(
            extract_filename("https://example.com/models/llama-7b.gguf"),
            "llama-7b.gguf"
        );
    }

    #[test]
    fn extract_filename_with_query() {
        assert_eq!(
            extract_filename("https://example.com/model.gguf?token=abc"),
            "model.gguf"
        );
    }

    #[test]
    fn extract_filename_no_extension() {
        assert_eq!(
            extract_filename("https://example.com/models/latest"),
            "download"
        );
    }

    #[test]
    fn extract_filename_empty_path() {
        assert_eq!(extract_filename("https://example.com/"), "download");
    }

    #[test]
    fn extract_filename_root() {
        // "example.com" contains a dot so it looks like a filename;
        // in practice direct download URLs always have a path.
        assert_eq!(extract_filename("https://example.com"), "example.com");
    }

    #[tokio::test]
    async fn resolve_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("HEAD", "/models/llama.gguf")
            .with_status(200)
            .with_header("content-length", "4000000000")
            .with_header("content-type", "application/octet-stream")
            .with_header("accept-ranges", "bytes")
            .create_async()
            .await;

        let client = Client::new();
        let info = resolve(&client, &format!("{}/models/llama.gguf", server.url()))
            .await
            .unwrap();
        assert_eq!(info.content_length, Some(4_000_000_000));
        assert_eq!(
            info.content_type.as_deref(),
            Some("application/octet-stream")
        );
        assert_eq!(info.filename, "llama.gguf");
        assert!(info.accepts_ranges);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn resolve_not_found() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("HEAD", "/missing.gguf")
            .with_status(404)
            .create_async()
            .await;

        let client = Client::new();
        let result = resolve(&client, &format!("{}/missing.gguf", server.url())).await;
        assert!(matches!(result, Err(IfranError::DownloadError(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn resolve_no_content_length() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("HEAD", "/model.gguf")
            .with_status(200)
            .create_async()
            .await;

        let client = Client::new();
        let info = resolve(&client, &format!("{}/model.gguf", server.url()))
            .await
            .unwrap();
        assert_eq!(info.content_length, None);
        assert_eq!(info.content_type, None);
        assert!(!info.accepts_ranges);
        mock.assert_async().await;
    }
}
