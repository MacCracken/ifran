//! Trust and verification layer for marketplace models.
//!
//! Provides configurable trust policies for models downloaded from
//! remote marketplace peers. Leverages existing SHA-256/BLAKE3
//! verification infrastructure.

use std::path::Path;

use ifran_types::IfranError;
use ifran_types::error::Result;
use ifran_types::marketplace::MarketplaceEntry;

/// Trust level assigned to a marketplace model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    /// No verification performed or possible.
    Untrusted,
    /// SHA-256 checksum verified against entry metadata.
    ChecksumVerified,
    /// Publisher is in the trusted publishers list.
    TrustedPublisher,
}

/// Configurable trust policy for marketplace operations.
#[derive(Debug, Clone)]
pub struct TrustPolicy {
    /// Require SHA-256 checksum to be present on entries before download.
    pub require_checksum: bool,
    /// List of trusted publisher instance IDs.
    pub trusted_publishers: Vec<String>,
    /// Maximum file size (bytes) allowed for download. None = unlimited.
    pub max_download_size: Option<u64>,
}

impl Default for TrustPolicy {
    fn default() -> Self {
        Self {
            require_checksum: true,
            trusted_publishers: Vec::new(),
            max_download_size: None,
        }
    }
}

/// Verify a marketplace entry against a trust policy before download.
///
/// Returns the trust level that would be achieved if the download succeeds.
pub fn verify_entry(entry: &MarketplaceEntry, policy: &TrustPolicy) -> Result<TrustLevel> {
    // Check size limit
    if let Some(max_size) = policy.max_download_size {
        if entry.size_bytes > max_size {
            return Err(IfranError::MarketplaceError(format!(
                "Model '{}' size ({} bytes) exceeds policy limit ({} bytes)",
                entry.model_name, entry.size_bytes, max_size
            )));
        }
    }

    // Check checksum requirement
    if policy.require_checksum && entry.sha256.is_none() {
        return Err(IfranError::MarketplaceError(format!(
            "Model '{}' has no SHA-256 checksum and policy requires one",
            entry.model_name
        )));
    }

    // Determine trust level
    if policy
        .trusted_publishers
        .contains(&entry.publisher_instance)
    {
        Ok(TrustLevel::TrustedPublisher)
    } else if entry.sha256.is_some() {
        Ok(TrustLevel::ChecksumVerified)
    } else {
        Ok(TrustLevel::Untrusted)
    }
}

/// Verify a downloaded file against a marketplace entry's checksum.
///
/// Returns `ChecksumVerified` on success, or an error if verification fails.
pub fn verify_download(path: &Path, entry: &MarketplaceEntry) -> Result<TrustLevel> {
    match &entry.sha256 {
        Some(expected) => {
            crate::pull::verifier::verify_file(
                path,
                expected,
                crate::pull::verifier::HashAlgorithm::Sha256,
            )?;
            Ok(TrustLevel::ChecksumVerified)
        }
        None => Ok(TrustLevel::Untrusted),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use ifran_types::model::{ModelFormat, QuantLevel};

    fn sample_entry(sha256: Option<&str>, publisher: &str) -> MarketplaceEntry {
        MarketplaceEntry {
            model_name: "test-model".into(),
            description: None,
            format: ModelFormat::Gguf,
            quant: QuantLevel::Q4KM,
            size_bytes: 1_000_000,
            parameter_count: None,
            architecture: None,
            publisher_instance: publisher.into(),
            download_url: "http://node-1:8420/marketplace/download/test-model".into(),
            sha256: sha256.map(String::from),
            tags: Vec::new(),
            published_at: Utc::now(),
            eval_scores: None,
        }
    }

    #[test]
    fn entry_with_checksum_passes() {
        let entry = sample_entry(Some("abc123"), "node-1");
        let policy = TrustPolicy::default();
        let level = verify_entry(&entry, &policy).unwrap();
        assert_eq!(level, TrustLevel::ChecksumVerified);
    }

    #[test]
    fn entry_without_checksum_rejected_by_policy() {
        let entry = sample_entry(None, "node-1");
        let policy = TrustPolicy {
            require_checksum: true,
            ..Default::default()
        };
        assert!(verify_entry(&entry, &policy).is_err());
    }

    #[test]
    fn entry_without_checksum_allowed_when_not_required() {
        let entry = sample_entry(None, "node-1");
        let policy = TrustPolicy {
            require_checksum: false,
            ..Default::default()
        };
        let level = verify_entry(&entry, &policy).unwrap();
        assert_eq!(level, TrustLevel::Untrusted);
    }

    #[test]
    fn trusted_publisher_gets_higher_trust() {
        let entry = sample_entry(Some("abc123"), "node-1");
        let policy = TrustPolicy {
            trusted_publishers: vec!["node-1".into()],
            ..Default::default()
        };
        let level = verify_entry(&entry, &policy).unwrap();
        assert_eq!(level, TrustLevel::TrustedPublisher);
    }

    #[test]
    fn size_limit_enforced() {
        let entry = sample_entry(Some("abc123"), "node-1");
        let policy = TrustPolicy {
            max_download_size: Some(100), // 100 bytes, entry is 1MB
            ..Default::default()
        };
        assert!(verify_entry(&entry, &policy).is_err());
    }

    #[test]
    fn trust_level_ordering() {
        assert!(TrustLevel::Untrusted < TrustLevel::ChecksumVerified);
        assert!(TrustLevel::ChecksumVerified < TrustLevel::TrustedPublisher);
    }

    #[test]
    fn verify_download_without_checksum_returns_untrusted() {
        let entry = sample_entry(None, "node-1");
        let level = verify_download(Path::new("/nonexistent"), &entry).unwrap();
        assert_eq!(level, TrustLevel::Untrusted);
    }

    #[test]
    fn verify_download_with_checksum_on_real_file() {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"hello world").unwrap();
        tmp.flush().unwrap();

        let entry = sample_entry(
            Some("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"),
            "node-1",
        );
        let level = verify_download(tmp.path(), &entry).unwrap();
        assert_eq!(level, TrustLevel::ChecksumVerified);
    }

    #[test]
    fn verify_download_with_wrong_checksum_fails() {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"hello world").unwrap();
        tmp.flush().unwrap();

        let entry = sample_entry(Some("badhash"), "node-1");
        assert!(verify_download(tmp.path(), &entry).is_err());
    }
}
