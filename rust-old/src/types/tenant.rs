use serde::{Deserialize, Serialize};

/// Identifies a tenant in multi-tenant deployments.
///
/// In single-tenant mode, all resources belong to the `"default"` tenant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub String);

impl TenantId {
    /// The default tenant used in single-tenant mode.
    #[must_use]
    #[inline]
    pub fn default_tenant() -> Self {
        Self("default".into())
    }

    /// Whether this is the default (single-tenant) tenant.
    #[must_use]
    #[inline]
    pub fn is_default(&self) -> bool {
        self.0 == "default"
    }
}

impl AsRef<str> for TenantId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TenantId {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tenant() {
        let t = TenantId::default_tenant();
        assert_eq!(t.0, "default");
        assert!(t.is_default());
    }

    #[test]
    fn custom_tenant() {
        let t = TenantId("acme".into());
        assert!(!t.is_default());
        assert_eq!(t.to_string(), "acme");
    }

    #[test]
    fn serde_roundtrip() {
        let t = TenantId("tenant-1".into());
        let json = serde_json::to_string(&t).unwrap();
        let back: TenantId = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn hash_and_eq() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(TenantId("a".into()), 1);
        map.insert(TenantId("b".into()), 2);
        assert_eq!(map[&TenantId("a".into())], 1);
    }

    #[test]
    fn display() {
        let t = TenantId("org-42".into());
        assert_eq!(format!("{t}"), "org-42");
    }
}
