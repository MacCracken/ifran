//! SY instance discovery — resolves SecureYeoman endpoints.
//!
//! Checks in order: explicit config, environment variable, daimon service
//! registry, well-known local address.

use ifran_types::error::Result;

/// Default SecureYeoman endpoint (well-known local address).
const DEFAULT_SY_ENDPOINT: &str = "http://127.0.0.1:9420";

/// Daimon service registry base URL.
/// Override with `DAIMON_ENDPOINT` environment variable.
const DEFAULT_DAIMON_ENDPOINT: &str = "http://127.0.0.1:9400";

/// A discovered SY endpoint.
#[derive(Debug, Clone)]
pub struct SyEndpoint {
    pub address: String,
    pub discovered_via: DiscoveryMethod,
}

#[derive(Debug, Clone, Copy)]
pub enum DiscoveryMethod {
    Config,
    Environment,
    DaimonRegistry,
    WellKnown,
}

/// Discover the SY endpoint (sync).
///
/// Priority: explicit config → `SY_ENDPOINT` env → well-known localhost:9420.
///
/// For daimon service registry support, use [`discover_async`].
pub fn discover(config_endpoint: Option<&str>) -> Result<SyEndpoint> {
    // 1. Explicit config
    if let Some(ep) = config_endpoint {
        if !ep.is_empty() {
            return Ok(SyEndpoint {
                address: ep.to_string(),
                discovered_via: DiscoveryMethod::Config,
            });
        }
    }

    // 2. Environment variable
    if let Ok(ep) = std::env::var("SY_ENDPOINT") {
        if !ep.is_empty() {
            return Ok(SyEndpoint {
                address: ep,
                discovered_via: DiscoveryMethod::Environment,
            });
        }
    }

    // 3. Well-known local address
    Ok(SyEndpoint {
        address: DEFAULT_SY_ENDPOINT.to_string(),
        discovered_via: DiscoveryMethod::WellKnown,
    })
}

/// Async discovery that can query daimon's service registry.
///
/// Priority: explicit config → `SY_ENDPOINT` env → daimon registry → well-known localhost:9420.
pub async fn discover_async(config_endpoint: Option<&str>) -> Result<SyEndpoint> {
    // 1. Explicit config
    if let Some(ep) = config_endpoint {
        if !ep.is_empty() {
            return Ok(SyEndpoint {
                address: ep.to_string(),
                discovered_via: DiscoveryMethod::Config,
            });
        }
    }

    // 2. SY_ENDPOINT env
    if let Ok(ep) = std::env::var("SY_ENDPOINT") {
        if !ep.is_empty() {
            return Ok(SyEndpoint {
                address: ep,
                discovered_via: DiscoveryMethod::Environment,
            });
        }
    }

    // 3. Daimon service registry
    if let Some(ep) = query_daimon_registry().await {
        return Ok(SyEndpoint {
            address: ep,
            discovered_via: DiscoveryMethod::DaimonRegistry,
        });
    }

    // 4. Well-known fallback
    Ok(SyEndpoint {
        address: DEFAULT_SY_ENDPOINT.to_string(),
        discovered_via: DiscoveryMethod::WellKnown,
    })
}

/// Query daimon's service registry for SecureYeoman endpoint.
/// Returns `None` if daimon is not available or SY is not registered.
///
/// The daimon endpoint is configurable via `DAIMON_ENDPOINT` env var,
/// defaulting to `http://127.0.0.1:9400` (standard Agnosticos location).
async fn query_daimon_registry() -> Option<String> {
    let base =
        std::env::var("DAIMON_ENDPOINT").unwrap_or_else(|_| DEFAULT_DAIMON_ENDPOINT.to_string());
    let url = format!("{base}/v1/discover?service=secureyeoman");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?;

    let resp = client.get(&url).send().await.ok()?;

    if !resp.status().is_success() {
        return None;
    }

    let body: serde_json::Value = resp.json().await.ok()?;
    body["endpoint"].as_str().map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_takes_priority() {
        let ep = discover(Some("http://sy.local:9420")).unwrap();
        assert_eq!(ep.address, "http://sy.local:9420");
        assert!(matches!(ep.discovered_via, DiscoveryMethod::Config));
    }

    #[test]
    fn falls_back_to_well_known() {
        let ep = discover(None).unwrap();
        assert_eq!(ep.address, "http://127.0.0.1:9420");
        assert!(matches!(ep.discovered_via, DiscoveryMethod::WellKnown));
    }

    #[test]
    fn empty_config_falls_through() {
        let ep = discover(Some("")).unwrap();
        // Empty string should be treated as no config
        assert!(matches!(
            ep.discovered_via,
            DiscoveryMethod::Environment | DiscoveryMethod::WellKnown
        ));
    }

    #[test]
    fn sy_endpoint_debug_format() {
        let ep = SyEndpoint {
            address: "http://test:9420".into(),
            discovered_via: DiscoveryMethod::Config,
        };
        let debug = format!("{:?}", ep);
        assert!(debug.contains("http://test:9420"));
        assert!(debug.contains("Config"));
    }

    #[test]
    fn discovery_method_is_copy() {
        let method = DiscoveryMethod::WellKnown;
        let copy = method;
        assert!(matches!(copy, DiscoveryMethod::WellKnown));
    }

    #[test]
    fn config_with_various_urls() {
        let ep = discover(Some("http://10.0.0.1:9420")).unwrap();
        assert_eq!(ep.address, "http://10.0.0.1:9420");
        assert!(matches!(ep.discovered_via, DiscoveryMethod::Config));

        let ep = discover(Some("https://sy.prod.example.com")).unwrap();
        assert_eq!(ep.address, "https://sy.prod.example.com");
    }

    #[test]
    fn none_config_without_env_falls_to_well_known() {
        // Clear env var to ensure well-known fallback
        let prev = std::env::var("SY_ENDPOINT").ok();
        unsafe { std::env::remove_var("SY_ENDPOINT") };

        let ep = discover(None).unwrap();
        assert_eq!(ep.address, "http://127.0.0.1:9420");
        assert!(matches!(ep.discovered_via, DiscoveryMethod::WellKnown));

        // Restore
        if let Some(val) = prev {
            unsafe { std::env::set_var("SY_ENDPOINT", val) };
        }
    }

    #[test]
    fn sy_endpoint_clone() {
        let ep = SyEndpoint {
            address: "http://test:9420".into(),
            discovered_via: DiscoveryMethod::Config,
        };
        let cloned = ep.clone();
        assert_eq!(ep.address, cloned.address);
    }

    #[test]
    fn all_discovery_methods_debug() {
        assert!(format!("{:?}", DiscoveryMethod::Config).contains("Config"));
        assert!(format!("{:?}", DiscoveryMethod::Environment).contains("Environment"));
        assert!(format!("{:?}", DiscoveryMethod::DaimonRegistry).contains("DaimonRegistry"));
        assert!(format!("{:?}", DiscoveryMethod::WellKnown).contains("WellKnown"));
    }

    #[test]
    fn daimon_registry_variant_is_copy() {
        let method = DiscoveryMethod::DaimonRegistry;
        let copy = method;
        assert!(matches!(copy, DiscoveryMethod::DaimonRegistry));
    }

    #[tokio::test]
    async fn async_config_takes_priority() {
        let ep = discover_async(Some("http://sy.local:9420")).await.unwrap();
        assert_eq!(ep.address, "http://sy.local:9420");
        assert!(matches!(ep.discovered_via, DiscoveryMethod::Config));
    }

    #[tokio::test]
    async fn async_empty_config_falls_through() {
        let ep = discover_async(Some("")).await.unwrap();
        assert!(matches!(
            ep.discovered_via,
            DiscoveryMethod::Environment | DiscoveryMethod::WellKnown
        ));
    }

    #[tokio::test]
    async fn async_no_daimon_falls_to_well_known() {
        // No daimon running — should fall through to well-known
        let prev = std::env::var("SY_ENDPOINT").ok();
        unsafe { std::env::remove_var("SY_ENDPOINT") };

        let ep = discover_async(None).await.unwrap();
        assert_eq!(ep.address, DEFAULT_SY_ENDPOINT);
        assert!(matches!(ep.discovered_via, DiscoveryMethod::WellKnown));

        if let Some(val) = prev {
            unsafe { std::env::set_var("SY_ENDPOINT", val) };
        }
    }

    #[tokio::test]
    async fn query_daimon_returns_none_when_unavailable() {
        let result = query_daimon_registry().await;
        assert!(result.is_none());
    }
}
