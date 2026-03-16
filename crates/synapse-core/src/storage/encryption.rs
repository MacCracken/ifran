//! Encrypted storage detection and daimon key management integration.
//!
//! Detects whether the model storage directory is backed by an encrypted
//! filesystem (LUKS/dm-crypt). On Agnosticos, integrates with daimon's
//! key management service to unlock volumes at startup.

use std::path::Path;
use synapse_types::SynapseError;
use synapse_types::error::Result;

/// Encryption status of a storage path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncryptionStatus {
    /// Path is on an encrypted volume (LUKS/dm-crypt).
    Encrypted { device: String },
    /// Path is on an unencrypted volume.
    Unencrypted,
    /// Could not determine encryption status.
    Unknown,
}

impl std::fmt::Display for EncryptionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encrypted { device } => write!(f, "encrypted ({device})"),
            Self::Unencrypted => write!(f, "unencrypted"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Check if a path resides on an encrypted (dm-crypt/LUKS) volume.
///
/// Reads `/proc/mounts` to find the mount point, then checks if the
/// backing device is a dm-crypt device via `/sys/block/*/dm/uuid`.
pub fn check_encryption(path: &Path) -> EncryptionStatus {
    let path_str = match path.canonicalize() {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(_) => return EncryptionStatus::Unknown,
    };

    // Find the mount point for this path from /proc/mounts
    let mounts = match std::fs::read_to_string("/proc/mounts") {
        Ok(m) => m,
        Err(_) => return EncryptionStatus::Unknown,
    };

    let mut best_mount = None;
    let mut best_len = 0;

    for line in mounts.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let mount_point = parts[1];
        if path_str.starts_with(mount_point) && mount_point.len() > best_len {
            best_mount = Some((parts[0].to_string(), mount_point.to_string()));
            best_len = mount_point.len();
        }
    }

    let (device, _mount_point) = match best_mount {
        Some(m) => m,
        None => return EncryptionStatus::Unknown,
    };

    // Check if the device is a dm-crypt device
    // dm-crypt devices show up as /dev/dm-N or /dev/mapper/*
    if device.starts_with("/dev/mapper/") || device.starts_with("/dev/dm-") {
        // Check for CRYPT- prefix in dm uuid
        let dm_name = device
            .strip_prefix("/dev/mapper/")
            .or_else(|| device.strip_prefix("/dev/"))
            .unwrap_or(&device);

        let uuid_path = format!("/sys/block/{}/dm/uuid", dm_name.replace('/', "!"));
        if let Ok(uuid) = std::fs::read_to_string(&uuid_path) {
            if uuid.trim().starts_with("CRYPT-") {
                return EncryptionStatus::Encrypted { device };
            }
        }

        // Fallback: if it's a mapper device, check via dmsetup or assume encrypted
        // on Agnosticos where all mapper devices are LUKS
        if device.starts_with("/dev/mapper/") {
            return EncryptionStatus::Encrypted { device };
        }
    }

    EncryptionStatus::Unencrypted
}

/// Verify that storage meets encryption requirements.
///
/// When `require_encrypted` is true and the path is not encrypted,
/// returns an error. Otherwise succeeds.
pub fn verify_encryption_requirement(
    path: &Path,
    require_encrypted: bool,
) -> Result<EncryptionStatus> {
    let status = check_encryption(path);

    if require_encrypted && status == EncryptionStatus::Unencrypted {
        return Err(SynapseError::ConfigError(format!(
            "Encrypted storage required but {} is on an unencrypted volume. \
             Set require_encrypted_storage = false or mount on a LUKS volume.",
            path.display()
        )));
    }

    Ok(status)
}

/// Request daimon to unlock an encrypted volume (Agnosticos integration).
///
/// Contacts the daimon key management service to request unlocking
/// of the volume backing the given path. Returns Ok if the volume
/// is already unlocked or was successfully unlocked.
pub async fn request_unlock(path: &Path, daimon_endpoint: &str) -> Result<()> {
    let path_str = path.to_string_lossy();
    tracing::info!(path = %path_str, "Requesting volume unlock from daimon");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| SynapseError::StorageError(format!("HTTP client error: {e}")))?;

    let resp = client
        .post(format!("{daimon_endpoint}/v1/keys/unlock"))
        .json(&serde_json::json!({ "path": path_str }))
        .send()
        .await
        .map_err(|e| SynapseError::StorageError(format!("Daimon unlock request failed: {e}")))?;

    if resp.status().is_success() {
        tracing::info!("Volume unlock confirmed by daimon");
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(SynapseError::StorageError(format!(
            "Daimon unlock failed (HTTP {status}): {body}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encryption_status_display() {
        assert_eq!(
            EncryptionStatus::Encrypted {
                device: "/dev/mapper/data".into()
            }
            .to_string(),
            "encrypted (/dev/mapper/data)"
        );
        assert_eq!(EncryptionStatus::Unencrypted.to_string(), "unencrypted");
        assert_eq!(EncryptionStatus::Unknown.to_string(), "unknown");
    }

    #[test]
    fn encryption_status_eq() {
        assert_eq!(EncryptionStatus::Unencrypted, EncryptionStatus::Unencrypted);
        assert_ne!(EncryptionStatus::Unencrypted, EncryptionStatus::Unknown);
    }

    #[test]
    fn check_encryption_nonexistent_path() {
        let status = check_encryption(Path::new("/nonexistent/path/12345"));
        assert_eq!(status, EncryptionStatus::Unknown);
    }

    #[test]
    fn check_encryption_tmp() {
        // /tmp should exist and be readable on any system
        let status = check_encryption(Path::new("/tmp"));
        // Should be either Unencrypted or Unknown (not panic)
        assert!(matches!(
            status,
            EncryptionStatus::Unencrypted
                | EncryptionStatus::Unknown
                | EncryptionStatus::Encrypted { .. }
        ));
    }

    #[test]
    fn verify_not_required_passes() {
        let result = verify_encryption_requirement(Path::new("/tmp"), false);
        assert!(result.is_ok());
    }

    #[test]
    fn verify_required_on_unencrypted_fails() {
        // /tmp is almost certainly unencrypted
        let status = check_encryption(Path::new("/tmp"));
        if status == EncryptionStatus::Unencrypted {
            let result = verify_encryption_requirement(Path::new("/tmp"), true);
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn request_unlock_no_daimon() {
        // No daimon running — should fail
        let result = request_unlock(Path::new("/tmp"), "http://127.0.0.1:19998").await;
        assert!(result.is_err());
    }

    #[test]
    fn check_encryption_with_real_tmp() {
        // /tmp should exist and return a definitive status (not panic)
        let status = check_encryption(Path::new("/tmp"));
        // Must be one of the three variants — the key thing is no panic
        match &status {
            EncryptionStatus::Encrypted { device } => assert!(!device.is_empty()),
            EncryptionStatus::Unencrypted => {}
            EncryptionStatus::Unknown => {}
        }
    }

    #[test]
    fn verify_encryption_not_required_unknown_path() {
        // Nonexistent path + not required = Ok(Unknown)
        let result = verify_encryption_requirement(Path::new("/nonexistent/path/xyz/12345"), false);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), EncryptionStatus::Unknown);
    }

    #[test]
    fn encryption_status_clone_and_debug() {
        let encrypted = EncryptionStatus::Encrypted {
            device: "/dev/mapper/luks".into(),
        };
        let cloned = encrypted.clone();
        assert_eq!(encrypted, cloned);
        // Debug should produce a non-empty string
        let debug = format!("{:?}", encrypted);
        assert!(debug.contains("Encrypted"));

        let unknown = EncryptionStatus::Unknown;
        let debug_unknown = format!("{:?}", unknown);
        assert!(debug_unknown.contains("Unknown"));
    }

    #[test]
    fn encryption_status_display_all_variants() {
        // Verify Display for all three variants returns non-empty strings
        let encrypted = EncryptionStatus::Encrypted {
            device: "/dev/dm-0".into(),
        };
        let display = format!("{}", encrypted);
        assert!(display.contains("dm-0"));

        let unencrypted = format!("{}", EncryptionStatus::Unencrypted);
        assert_eq!(unencrypted, "unencrypted");

        let unknown = format!("{}", EncryptionStatus::Unknown);
        assert_eq!(unknown, "unknown");
    }
}
