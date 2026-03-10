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
    if let Some(ep) = config_endpoint
        && !ep.is_empty()
    {
        return Ok(SyEndpoint {
            address: ep.to_string(),
            discovered_via: DiscoveryMethod::Config,
        });
    }

    // 2. Environment variable
    if let Ok(ep) = std::env::var("SY_ENDPOINT")
        && !ep.is_empty()
    {
        return Ok(SyEndpoint {
            address: ep,
            discovered_via: DiscoveryMethod::Environment,
        });
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
}
