//! SY instance discovery — resolves SecureYeoman endpoints.
//!
//! Checks in order: explicit config, environment variable, well-known
//! local address.

use synapse_types::error::Result;

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
    WellKnown,
}

/// Discover the SY endpoint.
///
/// Priority: explicit config → `SY_ENDPOINT` env → localhost:9420.
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
        address: "http://127.0.0.1:9420".to_string(),
        discovered_via: DiscoveryMethod::WellKnown,
    })
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
        assert!(format!("{:?}", DiscoveryMethod::WellKnown).contains("WellKnown"));
    }
}
