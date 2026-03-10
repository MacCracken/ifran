//! Marketplace resolver — searches for models across marketplace instances.

use synapse_types::error::Result;
use synapse_types::marketplace::{MarketplaceEntry, MarketplaceQuery};

/// Resolves models from marketplace sources.
///
/// MVP: searches the local catalog only. Future: queries remote instances
/// via REST or gRPC, aggregates results.
pub struct MarketplaceResolver {
    /// Known peer instance URLs.
    peers: Vec<String>,
}

impl MarketplaceResolver {
    pub fn new() -> Self {
        Self { peers: Vec::new() }
    }

    /// Add a peer instance URL for remote queries.
    pub fn add_peer(&mut self, url: String) {
        if !self.peers.contains(&url) {
            self.peers.push(url);
        }
    }

    /// List known peers.
    pub fn peers(&self) -> &[String] {
        &self.peers
    }

    /// Search remote peers for models.
    ///
    /// MVP: Returns empty — remote search is a future feature.
    /// This will query `GET /marketplace/search` on each peer.
    pub async fn search_remote(&self, _query: &MarketplaceQuery) -> Result<Vec<MarketplaceEntry>> {
        // TODO: Query each peer's REST API and aggregate results.
        Ok(Vec::new())
    }
}

impl Default for MarketplaceResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_peer_deduplicates() {
        let mut resolver = MarketplaceResolver::new();
        resolver.add_peer("http://node-2:8420".into());
        resolver.add_peer("http://node-2:8420".into());
        assert_eq!(resolver.peers().len(), 1);
    }

    #[tokio::test]
    async fn remote_search_returns_empty_for_now() {
        let resolver = MarketplaceResolver::new();
        let results = resolver
            .search_remote(&MarketplaceQuery::default())
            .await
            .unwrap();
        assert!(results.is_empty());
    }
}
