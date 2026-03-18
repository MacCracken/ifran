//! Shared pagination types for consistent list endpoint responses.

use serde::{Deserialize, Serialize};

/// Default page size for list endpoints.
pub const DEFAULT_LIMIT: u32 = 50;
/// Maximum page size for list endpoints.
pub const MAX_LIMIT: u32 = 1000;

/// Standard pagination query parameters.
#[derive(Debug, Deserialize)]
pub struct PaginationQuery {
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
}

fn default_limit() -> u32 {
    DEFAULT_LIMIT
}

impl PaginationQuery {
    /// Clamp limit to [1, MAX_LIMIT].
    pub fn safe_limit(&self) -> u32 {
        self.limit.clamp(1, MAX_LIMIT)
    }
}

/// Standard pagination metadata in responses.
#[derive(Debug, Serialize)]
pub struct PaginationInfo {
    pub total: usize,
    pub limit: u32,
    pub offset: u32,
}

/// A paginated list response wrapper.
#[derive(Debug, Serialize)]
pub struct PaginatedResponse<T: Serialize> {
    pub data: Vec<T>,
    pub pagination: PaginationInfo,
}

impl<T: Serialize> PaginatedResponse<T> {
    /// Create a paginated response from a full list, applying offset and limit.
    pub fn from_vec(items: Vec<T>, limit: u32, offset: u32) -> Self {
        let total = items.len();
        let safe_limit = limit.clamp(1, MAX_LIMIT) as usize;
        let safe_offset = (offset as usize).min(total);
        let data: Vec<T> = items
            .into_iter()
            .skip(safe_offset)
            .take(safe_limit)
            .collect();
        Self {
            data,
            pagination: PaginationInfo {
                total,
                limit: safe_limit as u32,
                offset: safe_offset as u32,
            },
        }
    }

    /// Create a paginated response when the backend already did the slicing.
    pub fn pre_sliced(data: Vec<T>, total: usize, limit: u32, offset: u32) -> Self {
        Self {
            data,
            pagination: PaginationInfo {
                total,
                limit,
                offset,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pagination_query_defaults() {
        let json = "{}";
        let q: PaginationQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.limit, DEFAULT_LIMIT);
        assert_eq!(q.offset, 0);
    }

    #[test]
    fn pagination_query_custom() {
        let json = r#"{"limit": 10, "offset": 20}"#;
        let q: PaginationQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.limit, 10);
        assert_eq!(q.offset, 20);
    }

    #[test]
    fn safe_limit_clamps() {
        let q = PaginationQuery {
            limit: 5000,
            offset: 0,
        };
        assert_eq!(q.safe_limit(), MAX_LIMIT);

        let q = PaginationQuery {
            limit: 0,
            offset: 0,
        };
        assert_eq!(q.safe_limit(), 1);
    }

    #[test]
    fn paginated_response_from_vec() {
        let items: Vec<u32> = (0..100).collect();
        let resp = PaginatedResponse::from_vec(items, 10, 20);
        assert_eq!(resp.data.len(), 10);
        assert_eq!(resp.data[0], 20);
        assert_eq!(resp.data[9], 29);
        assert_eq!(resp.pagination.total, 100);
        assert_eq!(resp.pagination.limit, 10);
        assert_eq!(resp.pagination.offset, 20);
    }

    #[test]
    fn paginated_response_offset_past_end() {
        let items: Vec<u32> = (0..5).collect();
        let resp = PaginatedResponse::from_vec(items, 10, 100);
        assert!(resp.data.is_empty());
        assert_eq!(resp.pagination.total, 5);
    }

    #[test]
    fn paginated_response_serializes() {
        let resp = PaginatedResponse::from_vec(vec!["a", "b", "c"], 2, 0);
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
        assert_eq!(json["pagination"]["total"], 3);
        assert_eq!(json["pagination"]["limit"], 2);
        assert_eq!(json["pagination"]["offset"], 0);
    }

    #[test]
    fn paginated_response_pre_sliced() {
        let resp = PaginatedResponse::pre_sliced(vec![1, 2, 3], 100, 3, 0);
        assert_eq!(resp.data.len(), 3);
        assert_eq!(resp.pagination.total, 100);
    }
}
