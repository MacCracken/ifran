use serde::{Deserialize, Serialize};

/// Source from which a model can be pulled.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegistrySource {
    HuggingFace {
        repo_id: String,
    },
    Oci {
        registry: String,
        repository: String,
    },
    DirectUrl {
        url: String,
    },
    LocalPath {
        path: String,
    },
    Marketplace {
        instance_url: String,
        model_name: String,
    },
}

/// Download state machine.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DownloadState {
    Queued,
    Downloading,
    Verifying,
    Complete,
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_source_huggingface_serde() {
        let src = RegistrySource::HuggingFace {
            repo_id: "meta-llama/Llama-3.1-8B".into(),
        };
        let json = serde_json::to_string(&src).unwrap();
        let back: RegistrySource = serde_json::from_str(&json).unwrap();
        if let RegistrySource::HuggingFace { repo_id } = back {
            assert_eq!(repo_id, "meta-llama/Llama-3.1-8B");
        } else {
            panic!("expected HuggingFace variant");
        }
    }

    #[test]
    fn registry_source_all_variants_serde() {
        let sources = vec![
            RegistrySource::HuggingFace {
                repo_id: "x".into(),
            },
            RegistrySource::Oci {
                registry: "ghcr.io".into(),
                repository: "org/model".into(),
            },
            RegistrySource::DirectUrl {
                url: "https://example.com/model.gguf".into(),
            },
            RegistrySource::LocalPath {
                path: "/tmp/model.gguf".into(),
            },
            RegistrySource::Marketplace {
                instance_url: "http://peer:8420".into(),
                model_name: "test".into(),
            },
        ];
        for src in &sources {
            let json = serde_json::to_string(src).unwrap();
            let _: RegistrySource = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn download_state_serde_roundtrip() {
        let states = [
            DownloadState::Queued,
            DownloadState::Downloading,
            DownloadState::Verifying,
            DownloadState::Complete,
            DownloadState::Failed,
        ];
        for s in &states {
            let json = serde_json::to_string(s).unwrap();
            let back: DownloadState = serde_json::from_str(&json).unwrap();
            assert_eq!(*s, back);
        }
    }
}
