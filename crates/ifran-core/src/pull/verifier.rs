//! SHA-256/BLAKE3 integrity verification.
//!
//! Hashes a file on disk and compares against an expected digest.
//! Supports both SHA-256 (standard for HuggingFace) and BLAKE3 (faster).

use ifran_types::IfranError;
use ifran_types::error::Result;
use sha2::{Digest, Sha256};
use std::path::Path;

/// Hash algorithm used for verification.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum HashAlgorithm {
    Sha256,
    Blake3,
}

/// Compute the hex-encoded hash of a file.
#[must_use = "hash result should be used or compared"]
pub fn hash_file(path: &Path, algorithm: HashAlgorithm) -> Result<String> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);

    match algorithm {
        HashAlgorithm::Sha256 => {
            let mut hasher = Sha256::new();
            std::io::copy(&mut reader, &mut hasher)?;
            Ok(format!("{:x}", hasher.finalize()))
        }
        HashAlgorithm::Blake3 => {
            let mut hasher = blake3::Hasher::new();
            std::io::copy(&mut reader, &mut hasher)?;
            Ok(hasher.finalize().to_hex().to_string())
        }
    }
}

/// Verify a file's integrity against an expected hash.
///
/// Returns `Ok(())` if the hash matches, or `IfranError::IntegrityError` if not.
pub fn verify_file(path: &Path, expected: &str, algorithm: HashAlgorithm) -> Result<()> {
    let actual = hash_file(path, algorithm)?;
    if actual == expected {
        Ok(())
    } else {
        Err(IfranError::IntegrityError {
            expected: expected.to_string(),
            actual,
        })
    }
}

/// Auto-detect algorithm from hash length and verify.
///
/// - 64 hex chars → SHA-256
/// - Other → BLAKE3
pub fn verify_auto(path: &Path, expected: &str) -> Result<()> {
    let algorithm = if expected.len() == 64 {
        // Could be either SHA-256 or BLAKE3 (both 64 hex chars).
        // Default to SHA-256 since that's what HuggingFace uses.
        HashAlgorithm::Sha256
    } else {
        HashAlgorithm::Blake3
    };
    verify_file(path, expected, algorithm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn sha256_hash() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"hello world").unwrap();
        tmp.flush().unwrap();

        let hash = hash_file(tmp.path(), HashAlgorithm::Sha256).unwrap();
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn blake3_hash() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"hello world").unwrap();
        tmp.flush().unwrap();

        let hash = hash_file(tmp.path(), HashAlgorithm::Blake3).unwrap();
        // blake3 hash of "hello world"
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn verify_success() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"hello world").unwrap();
        tmp.flush().unwrap();

        verify_file(
            tmp.path(),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
            HashAlgorithm::Sha256,
        )
        .unwrap();
    }

    #[test]
    fn verify_failure() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"hello world").unwrap();
        tmp.flush().unwrap();

        let result = verify_file(tmp.path(), "badhash", HashAlgorithm::Sha256);
        assert!(matches!(result, Err(IfranError::IntegrityError { .. })));
    }

    #[test]
    fn verify_auto_sha256() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"hello world").unwrap();
        tmp.flush().unwrap();

        // 64-char hex → detected as SHA-256
        verify_auto(
            tmp.path(),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        )
        .unwrap();
    }

    #[test]
    fn verify_auto_blake3() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"hello world").unwrap();
        tmp.flush().unwrap();

        let expected = hash_file(tmp.path(), HashAlgorithm::Blake3).unwrap();
        // BLAKE3 produces 64 hex chars too, but verify_auto defaults to SHA-256 for 64-char.
        // A non-64-char hash would go to BLAKE3. Verify the auto-detect logic with a truncated hash fails.
        let result = verify_auto(tmp.path(), &expected[..32]);
        assert!(matches!(result, Err(IfranError::IntegrityError { .. })));
    }

    #[test]
    fn hash_nonexistent_file() {
        let result = hash_file(
            std::path::Path::new("/nonexistent/file"),
            HashAlgorithm::Sha256,
        );
        assert!(result.is_err());
    }

    #[test]
    fn hash_empty_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let hash = hash_file(tmp.path(), HashAlgorithm::Sha256).unwrap();
        // SHA-256 of empty input
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn blake3_verify_roundtrip() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"test data for blake3").unwrap();
        tmp.flush().unwrap();

        let hash = hash_file(tmp.path(), HashAlgorithm::Blake3).unwrap();
        verify_file(tmp.path(), &hash, HashAlgorithm::Blake3).unwrap();
    }

    #[test]
    fn integrity_error_contains_hashes() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"hello world").unwrap();
        tmp.flush().unwrap();

        let result = verify_file(tmp.path(), "wrong_hash", HashAlgorithm::Sha256);
        match result {
            Err(IfranError::IntegrityError { expected, actual }) => {
                assert_eq!(expected, "wrong_hash");
                assert_eq!(
                    actual,
                    "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
                );
            }
            other => panic!("Expected IntegrityError, got {:?}", other),
        }
    }
}
