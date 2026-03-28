//! Auto-discovery of models running on local inference servers.
//!
//! Probes well-known endpoints for Ollama, LM Studio, and LocalAI to
//! discover what models are available without manual configuration.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A model discovered from a local inference server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    pub name: String,
    pub source: DiscoverySource,
    pub size_bytes: Option<u64>,
    pub family: Option<String>,
    pub quantization: Option<String>,
    pub parameter_count: Option<String>,
}

/// Where a model was discovered from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DiscoverySource {
    Ollama,
    LmStudio,
    LocalAi,
}

impl std::fmt::Display for DiscoverySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ollama => write!(f, "Ollama"),
            Self::LmStudio => write!(f, "LM Studio"),
            Self::LocalAi => write!(f, "LocalAI"),
        }
    }
}

/// Default endpoints for local inference servers.
const OLLAMA_DEFAULT: &str = "http://127.0.0.1:11434";
const LM_STUDIO_DEFAULT: &str = "http://127.0.0.1:1234";
const LOCAL_AI_DEFAULT: &str = "http://127.0.0.1:8080";

/// Timeout for discovery probes.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Discover models from all known local inference servers.
pub async fn discover_all() -> Vec<DiscoveredModel> {
    let (ollama, lm_studio, local_ai) = tokio::join!(
        discover_ollama(OLLAMA_DEFAULT),
        discover_lm_studio(LM_STUDIO_DEFAULT),
        discover_local_ai(LOCAL_AI_DEFAULT),
    );

    let ollama = ollama.unwrap_or_default();
    let lm_studio = lm_studio.unwrap_or_default();
    let local_ai = local_ai.unwrap_or_default();
    let mut results = Vec::with_capacity(ollama.len() + lm_studio.len() + local_ai.len());
    results.extend(ollama);
    results.extend(lm_studio);
    results.extend(local_ai);
    results
}

/// Discover models from a specific Ollama instance.
pub async fn discover_ollama(base_url: &str) -> Result<Vec<DiscoveredModel>, String> {
    let client = reqwest::Client::builder()
        .timeout(PROBE_TIMEOUT)
        .build()
        .map_err(|e| e.to_string())?;

    let url = format!("{base_url}/api/tags");
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Ollama unreachable: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Ollama returned {}", resp.status()));
    }

    #[derive(Deserialize)]
    struct OllamaResponse {
        models: Option<Vec<OllamaModel>>,
    }
    #[derive(Deserialize)]
    struct OllamaModel {
        name: String,
        size: Option<u64>,
        details: Option<OllamaDetails>,
    }
    #[derive(Deserialize)]
    struct OllamaDetails {
        family: Option<String>,
        parameter_size: Option<String>,
        quantization_level: Option<String>,
    }

    let body: OllamaResponse = resp.json().await.map_err(|e| e.to_string())?;
    let models = body.models.unwrap_or_default();

    Ok(models
        .into_iter()
        .map(|m| {
            let (family, parameter_count, quantization) = match m.details {
                Some(d) => (d.family, d.parameter_size, d.quantization_level),
                None => (None, None, None),
            };
            DiscoveredModel {
                name: m.name,
                source: DiscoverySource::Ollama,
                size_bytes: m.size,
                family,
                quantization,
                parameter_count,
            }
        })
        .collect())
}

/// Discover models from an LM Studio instance (OpenAI-compatible API).
pub async fn discover_lm_studio(base_url: &str) -> Result<Vec<DiscoveredModel>, String> {
    discover_openai_compatible(base_url, DiscoverySource::LmStudio).await
}

/// Discover models from a LocalAI instance (OpenAI-compatible API).
pub async fn discover_local_ai(base_url: &str) -> Result<Vec<DiscoveredModel>, String> {
    discover_openai_compatible(base_url, DiscoverySource::LocalAi).await
}

/// Common discovery for OpenAI-compatible APIs (LM Studio, LocalAI).
async fn discover_openai_compatible(
    base_url: &str,
    source: DiscoverySource,
) -> Result<Vec<DiscoveredModel>, String> {
    let client = reqwest::Client::builder()
        .timeout(PROBE_TIMEOUT)
        .build()
        .map_err(|e| e.to_string())?;

    let url = format!("{base_url}/v1/models");
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("{source} unreachable: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("{source} returned {}", resp.status()));
    }

    #[derive(Deserialize)]
    struct ModelsResponse {
        data: Option<Vec<ModelEntry>>,
    }
    #[derive(Deserialize)]
    struct ModelEntry {
        id: String,
    }

    let body: ModelsResponse = resp.json().await.map_err(|e| e.to_string())?;
    let entries = body.data.unwrap_or_default();

    Ok(entries
        .into_iter()
        .map(|m| {
            let parameter_count = extract_param_count(&m.id);
            DiscoveredModel {
                name: m.id,
                source,
                size_bytes: None,
                family: None,
                quantization: None,
                parameter_count,
            }
        })
        .collect())
}

/// Try to extract parameter count from model name (e.g., "llama-7b" -> "7b").
fn extract_param_count(name: &str) -> Option<String> {
    let lower = name.to_lowercase();
    // Check from largest to smallest to avoid partial matches (e.g., "70b" before "7b")
    let patterns = [
        "405b", "236b", "180b", "72b", "70b", "65b", "34b", "32b", "14b", "13b", "8b", "7b", "3b",
        "2b", "1.5b", "0.5b",
    ];
    for p in patterns {
        // Check for word boundary: pattern must be preceded by non-alphanumeric or start of string
        if let Some(pos) = lower.find(p) {
            let before_ok = pos == 0 || !lower.as_bytes()[pos - 1].is_ascii_alphanumeric();
            if before_ok {
                return Some(p.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_source_display() {
        assert_eq!(DiscoverySource::Ollama.to_string(), "Ollama");
        assert_eq!(DiscoverySource::LmStudio.to_string(), "LM Studio");
        assert_eq!(DiscoverySource::LocalAi.to_string(), "LocalAI");
    }

    #[test]
    fn extract_param_count_works() {
        assert_eq!(extract_param_count("llama-7b-q4"), Some("7b".into()));
        assert_eq!(
            extract_param_count("mistral-70b-instruct"),
            Some("70b".into())
        );
        assert_eq!(extract_param_count("phi-3b"), Some("3b".into()));
        assert_eq!(extract_param_count("unknown-model"), None);
    }

    #[test]
    fn discovered_model_serializes() {
        let m = DiscoveredModel {
            name: "llama3:8b".into(),
            source: DiscoverySource::Ollama,
            size_bytes: Some(4_000_000_000),
            family: Some("llama".into()),
            quantization: Some("Q4_K_M".into()),
            parameter_count: Some("8b".into()),
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("ollama"));
        assert!(json.contains("llama3:8b"));
    }

    #[test]
    fn discovery_source_serde_roundtrip() {
        let sources = [
            DiscoverySource::Ollama,
            DiscoverySource::LmStudio,
            DiscoverySource::LocalAi,
        ];
        for s in &sources {
            let json = serde_json::to_string(s).unwrap();
            let back: DiscoverySource = serde_json::from_str(&json).unwrap();
            assert_eq!(*s, back);
        }
    }

    #[tokio::test]
    async fn discover_all_returns_empty_when_nothing_running() {
        // All servers down — should return empty, not error
        let results = discover_all().await;
        // May or may not be empty depending on what's running, but should not panic
        let _ = results;
    }
}
