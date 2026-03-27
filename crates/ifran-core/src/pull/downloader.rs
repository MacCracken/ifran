//! Chunked HTTP downloader with resume support.
//!
//! Downloads a file from a URL to a local path. Supports:
//! - Resume via HTTP `Range` header (checks for partial `.part` file)
//! - Progress callbacks via [`ProgressTracker`]
//! - Configurable timeouts

use crate::pull::progress::{ProgressEvent, ProgressTracker};
use ifran_types::IfranError;
use ifran_types::error::Result;
use ifran_types::registry::DownloadState;
use reqwest::Client;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

/// Configuration for a download.
pub struct DownloadRequest {
    pub url: String,
    pub dest: PathBuf,
    pub model_name: String,
    pub expected_sha256: Option<String>,
}

/// Download a file from a URL, supporting resume via `.part` files.
///
/// The file is first downloaded to `<dest>.part`, then renamed to `<dest>` on
/// success. If a `.part` file already exists, the download resumes from where
/// it left off (if the server supports `Range` requests).
pub async fn download(
    client: &Client,
    request: &DownloadRequest,
    progress: &ProgressTracker,
) -> Result<()> {
    let part_path = PathBuf::from(format!("{}.part", request.dest.display()));

    // Check existing partial download size for resume
    let existing_len = if part_path.exists() {
        tokio::fs::metadata(&part_path)
            .await
            .map(|m| m.len())
            .unwrap_or(0)
    } else {
        if let Some(parent) = part_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        0
    };

    // Build request with optional Range header for resume
    let mut req = client.get(&request.url);
    if existing_len > 0 {
        req = req.header("Range", format!("bytes={existing_len}-"));
    }

    let response = req
        .send()
        .await
        .map_err(|e| IfranError::DownloadError(e.to_string()))?;

    if !response.status().is_success() && response.status().as_u16() != 206 {
        return Err(IfranError::DownloadError(format!(
            "HTTP {}: {}",
            response.status(),
            request.url
        )));
    }

    let is_resumed = response.status().as_u16() == 206;
    let content_length = response.content_length();
    let total_bytes = content_length.map(|cl| cl + if is_resumed { existing_len } else { 0 });

    progress.send(ProgressEvent {
        model_name: request.model_name.clone(),
        state: DownloadState::Downloading,
        downloaded_bytes: existing_len,
        total_bytes,
        speed_bytes_per_sec: 0,
        message: if is_resumed {
            Some(format!("Resuming from {} bytes", existing_len))
        } else {
            None
        },
    });

    // Open file for append (resume) or create
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(is_resumed)
        .write(true)
        .truncate(!is_resumed)
        .open(&part_path)
        .await?;

    let mut downloaded = existing_len;
    let mut stream = response.bytes_stream();
    let mut last_progress = std::time::Instant::now();
    let mut bytes_since_last = 0u64;

    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| IfranError::DownloadError(e.to_string()))?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        bytes_since_last += chunk.len() as u64;

        // Emit progress at most every 100ms
        let elapsed = last_progress.elapsed();
        if elapsed.as_millis() >= 100 {
            let speed = (bytes_since_last as f64 / elapsed.as_secs_f64()) as u64;
            progress.send(ProgressEvent {
                model_name: request.model_name.clone(),
                state: DownloadState::Downloading,
                downloaded_bytes: downloaded,
                total_bytes,
                speed_bytes_per_sec: speed,
                message: None,
            });
            last_progress = std::time::Instant::now();
            bytes_since_last = 0;
        }
    }

    file.flush().await?;
    drop(file);

    // Verify integrity if expected hash is provided
    if let Some(ref expected) = request.expected_sha256 {
        progress.emit(
            &request.model_name,
            DownloadState::Verifying,
            "Verifying SHA-256 integrity...",
        );
        if let Err(e) = crate::pull::verifier::verify_file(
            &part_path,
            expected,
            crate::pull::verifier::HashAlgorithm::Sha256,
        ) {
            // Clean up corrupted partial file so next attempt starts fresh
            let _ = tokio::fs::remove_file(&part_path).await;
            return Err(e);
        }
    }

    // Rename .part to final destination
    tokio::fs::rename(&part_path, &request.dest).await?;

    progress.send(ProgressEvent {
        model_name: request.model_name.clone(),
        state: DownloadState::Complete,
        downloaded_bytes: downloaded,
        total_bytes,
        speed_bytes_per_sec: 0,
        message: Some("Download complete".to_string()),
    });

    Ok(())
}

/// Build a reusable reqwest client with sensible defaults.
pub fn build_client() -> Result<Client> {
    Client::builder()
        .user_agent(format!("ifran/{}", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(3600)) // 1 hour max for large models
        .build()
        .map_err(|e| IfranError::DownloadError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pull::progress::ProgressTracker;

    #[test]
    fn build_client_succeeds() {
        let client = build_client();
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn download_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/test-file.bin")
            .with_status(200)
            .with_body(b"file content here")
            .create_async()
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("output.bin");

        let request = DownloadRequest {
            url: format!("{}/test-file.bin", server.url()),
            dest: dest.clone(),
            model_name: "test-model".into(),
            expected_sha256: None,
        };

        let progress = ProgressTracker::default();
        let client = Client::new();
        download(&client, &request, &progress).await.unwrap();

        assert!(dest.exists());
        let content = tokio::fs::read(&dest).await.unwrap();
        assert_eq!(content, b"file content here");

        // .part file should be cleaned up
        assert!(!PathBuf::from(format!("{}.part", dest.display())).exists());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn download_creates_parent_dirs() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/file.bin")
            .with_status(200)
            .with_body(b"data")
            .create_async()
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("nested").join("dir").join("file.bin");

        let request = DownloadRequest {
            url: format!("{}/file.bin", server.url()),
            dest: dest.clone(),
            model_name: "test".into(),
            expected_sha256: None,
        };

        let progress = ProgressTracker::default();
        download(&Client::new(), &request, &progress).await.unwrap();
        assert!(dest.exists());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn download_http_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/missing.bin")
            .with_status(404)
            .create_async()
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let request = DownloadRequest {
            url: format!("{}/missing.bin", server.url()),
            dest: tmp.path().join("output.bin"),
            model_name: "test".into(),
            expected_sha256: None,
        };

        let progress = ProgressTracker::default();
        let result = download(&Client::new(), &request, &progress).await;
        assert!(matches!(result, Err(IfranError::DownloadError(_))));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn download_with_sha256_verification() {
        let content = b"verifiable content";
        let expected_hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(content);
            format!("{:x}", hasher.finalize())
        };

        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/verified.bin")
            .with_status(200)
            .with_body(content)
            .create_async()
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("verified.bin");

        let request = DownloadRequest {
            url: format!("{}/verified.bin", server.url()),
            dest: dest.clone(),
            model_name: "test".into(),
            expected_sha256: Some(expected_hash),
        };

        let progress = ProgressTracker::default();
        download(&Client::new(), &request, &progress).await.unwrap();
        assert!(dest.exists());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn download_bad_sha256_fails() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/bad.bin")
            .with_status(200)
            .with_body(b"some content")
            .create_async()
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let request = DownloadRequest {
            url: format!("{}/bad.bin", server.url()),
            dest: tmp.path().join("bad.bin"),
            model_name: "test".into(),
            expected_sha256: Some("wrong_hash".into()),
        };

        let progress = ProgressTracker::default();
        let result = download(&Client::new(), &request, &progress).await;
        assert!(result.is_err());
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn download_resume_from_partial() {
        // Create a partial file first
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("resumed.bin");
        let part_path = tmp.path().join("resumed.bin.part");
        tokio::fs::write(&part_path, b"first_").await.unwrap();

        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/resumed.bin")
            .match_header("range", "bytes=6-")
            .with_status(206)
            .with_body(b"second")
            .create_async()
            .await;

        let request = DownloadRequest {
            url: format!("{}/resumed.bin", server.url()),
            dest: dest.clone(),
            model_name: "test".into(),
            expected_sha256: None,
        };

        let progress = ProgressTracker::default();
        download(&Client::new(), &request, &progress).await.unwrap();

        let content = tokio::fs::read(&dest).await.unwrap();
        assert_eq!(content, b"first_second");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn download_progress_events() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/progress.bin")
            .with_status(200)
            .with_body(b"data")
            .create_async()
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let request = DownloadRequest {
            url: format!("{}/progress.bin", server.url()),
            dest: tmp.path().join("progress.bin"),
            model_name: "progress-test".into(),
            expected_sha256: None,
        };

        let progress = ProgressTracker::default();
        let mut rx = progress.subscribe();

        download(&Client::new(), &request, &progress).await.unwrap();

        // Should receive at least the initial and completion events
        let first = rx.recv().await.unwrap();
        assert_eq!(first.model_name, "progress-test");
        assert_eq!(first.state, DownloadState::Downloading);

        // Drain to find completion event
        let mut found_complete = false;
        while let Ok(event) = rx.try_recv() {
            if event.state == DownloadState::Complete {
                found_complete = true;
                assert_eq!(event.message, Some("Download complete".into()));
            }
        }
        assert!(found_complete);
        mock.assert_async().await;
    }
}
