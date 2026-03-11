//! Marketplace resolver — searches for models across marketplace instances.

use synapse_types::error::Result;
use synapse_types::marketplace::{MarketplaceEntry, MarketplaceQuery};

/// Resolves models from marketplace sources.
///
/// Searches the local catalog and queries remote Synapse instances
/// via their REST API, aggregating results.
pub struct MarketplaceResolver {
    /// Known peer instance URLs.
    peers: Vec<String>,
    client: reqwest::Client,
}

impl MarketplaceResolver {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();
        Self {
            peers: Vec::new(),
            client,
        }
    }

    /// Add a peer instance URL for remote queries.
    pub fn add_peer(&mut self, url: String) {
        if !self.peers.contains(&url) {
            self.peers.push(url);
        }
    }

    /// Remove a peer instance URL.
    pub fn remove_peer(&mut self, url: &str) {
        self.peers.retain(|p| p != url);
    }

    /// List known peers.
    pub fn peers(&self) -> &[String] {
        &self.peers
    }

    /// Search remote peers for models.
    ///
    /// Queries `GET /marketplace/search` on each peer, aggregates results,
    /// and deduplicates by model_name (first occurrence wins).
    pub async fn search_remote(&self, query: &MarketplaceQuery) -> Result<Vec<MarketplaceEntry>> {
        let mut all_entries: Vec<MarketplaceEntry> = Vec::new();
        let mut seen_names = std::collections::HashSet::new();

        for peer in &self.peers {
            match self.query_peer(peer, query).await {
                Ok(entries) => {
                    for entry in entries {
                        if seen_names.insert(entry.model_name.clone()) {
                            all_entries.push(entry);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(peer = %peer, error = %e, "Failed to query marketplace peer");
                }
            }
        }

        Ok(all_entries)
    }

    /// Query a single peer's marketplace search endpoint.
    async fn query_peer(
        &self,
        peer_url: &str,
        query: &MarketplaceQuery,
    ) -> Result<Vec<MarketplaceEntry>> {
        let url = format!("{}/marketplace/search", peer_url.trim_end_matches('/'));

        let mut req = self.client.get(&url);

        if let Some(ref search) = query.search {
            req = req.query(&[("q", search.as_str())]);
        }
        if let Some(ref format) = query.format {
            let f = serde_json::to_string(format)
                .unwrap_or_default()
                .trim_matches('"')
                .to_string();
            req = req.query(&[("format", f.as_str())]);
        }
        if let Some(max_size) = query.max_size_bytes {
            req = req.query(&[("max_size", &max_size.to_string())]);
        }

        let response = req.send().await.map_err(|e| {
            synapse_types::SynapseError::MarketplaceError(format!(
                "Failed to reach peer {peer_url}: {e}"
            ))
        })?;

        if !response.status().is_success() {
            return Err(synapse_types::SynapseError::MarketplaceError(format!(
                "Peer {peer_url} returned HTTP {}",
                response.status()
            )));
        }

        let entries: Vec<MarketplaceEntry> = response.json().await.map_err(|e| {
            synapse_types::SynapseError::MarketplaceError(format!(
                "Invalid response from peer {peer_url}: {e}"
            ))
        })?;

        Ok(entries)
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

    #[test]
    fn remove_peer() {
        let mut resolver = MarketplaceResolver::new();
        resolver.add_peer("http://node-2:8420".into());
        resolver.add_peer("http://node-3:8420".into());
        resolver.remove_peer("http://node-2:8420");
        assert_eq!(resolver.peers().len(), 1);
        assert_eq!(resolver.peers()[0], "http://node-3:8420");
    }

    #[tokio::test]
    async fn remote_search_with_no_peers_returns_empty() {
        let resolver = MarketplaceResolver::new();
        let results = resolver
            .search_remote(&MarketplaceQuery::default())
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn remote_search_logs_unreachable_peers() {
        let mut resolver = MarketplaceResolver::new();
        resolver.add_peer("http://127.0.0.1:1".into()); // unreachable
        let results = resolver
            .search_remote(&MarketplaceQuery::default())
            .await
            .unwrap();
        assert!(results.is_empty()); // gracefully returns empty
    }
}
