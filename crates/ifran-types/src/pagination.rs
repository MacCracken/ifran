//! Pagination primitives for paginated store queries.

use serde::{Deserialize, Serialize};

/// Result of a paginated query from a store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedResult<T> {
    /// Items in this page.
    pub items: Vec<T>,
    /// Total number of items matching the query (across all pages).
    pub total: usize,
}

impl<T> PagedResult<T> {
    /// Create an empty paged result.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            items: Vec::new(),
            total: 0,
        }
    }
}
