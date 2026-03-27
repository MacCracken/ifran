//! LRU model cache with size-based eviction.
//!
//! Tracks cached model files on disk and evicts the least-recently-used
//! entries when the total cache size exceeds a configured limit.

use std::collections::HashMap;
use std::time::Instant;

/// Metadata for a cached model entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Unique key (e.g. model slug or digest).
    pub key: String,
    /// Size of the cached file in bytes.
    pub size_bytes: u64,
    /// When the entry was last accessed.
    pub last_accessed: Instant,
}

/// An LRU cache that tracks model files and their sizes.
///
/// When the total size exceeds `max_bytes`, the least-recently-used entries
/// are evicted until the cache is within budget.
pub struct ModelCache {
    entries: HashMap<String, CacheEntry>,
    max_bytes: u64,
}

impl ModelCache {
    /// Create a new cache with the given size limit in bytes.
    pub fn new(max_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            max_bytes,
        }
    }

    /// Total bytes currently used by cached entries.
    #[inline]
    pub fn total_bytes(&self) -> u64 {
        self.entries.values().map(|e| e.size_bytes).sum()
    }

    /// Number of cached entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Record a cache hit (updates last-accessed time).
    ///
    /// Returns `true` if the key was found.
    #[inline]
    pub fn touch(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = Instant::now();
            true
        } else {
            false
        }
    }

    /// Check if a key exists in the cache.
    #[inline]
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Insert an entry, evicting LRU entries if needed to stay within budget.
    ///
    /// Returns the keys of any evicted entries so the caller can delete
    /// the corresponding files from disk.
    pub fn insert(&mut self, key: String, size_bytes: u64) -> Vec<String> {
        // If the item already exists, remove it first so we re-add with fresh timestamp
        self.entries.remove(&key);

        let mut evicted = Vec::new();

        // Evict LRU entries until we have room
        while self.total_bytes() + size_bytes > self.max_bytes && !self.entries.is_empty() {
            if let Some(lru_key) = self.find_lru() {
                self.entries.remove(&lru_key);
                evicted.push(lru_key);
            } else {
                break;
            }
        }

        self.entries.insert(
            key.clone(),
            CacheEntry {
                key,
                size_bytes,
                last_accessed: Instant::now(),
            },
        );

        evicted
    }

    /// Remove an entry by key.
    #[inline]
    pub fn remove(&mut self, key: &str) -> Option<CacheEntry> {
        self.entries.remove(key)
    }

    /// Find the least-recently-used key.
    #[inline]
    fn find_lru(&self) -> Option<String> {
        self.entries
            .values()
            .min_by_key(|e| e.last_accessed)
            .map(|e| e.key.clone())
    }

    /// List all cache keys, ordered by last access (oldest first).
    pub fn keys_by_age(&self) -> Vec<String> {
        let mut entries: Vec<_> = self.entries.values().collect();
        entries.sort_by_key(|e| e.last_accessed);
        entries.iter().map(|e| e.key.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_cache_is_empty() {
        let cache = ModelCache::new(1_000_000);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.total_bytes(), 0);
    }

    #[test]
    fn insert_and_lookup() {
        let mut cache = ModelCache::new(10_000);
        let evicted = cache.insert("model-a".into(), 5000);
        assert!(evicted.is_empty());
        assert!(cache.contains("model-a"));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_bytes(), 5000);
    }

    #[test]
    fn insert_multiple() {
        let mut cache = ModelCache::new(20_000);
        cache.insert("a".into(), 5000);
        cache.insert("b".into(), 5000);
        cache.insert("c".into(), 5000);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.total_bytes(), 15_000);
    }

    #[test]
    fn eviction_on_overflow() {
        let mut cache = ModelCache::new(10_000);
        cache.insert("old".into(), 6000);
        // This insert should evict "old" to make room
        let evicted = cache.insert("new".into(), 6000);
        assert_eq!(evicted, vec!["old"]);
        assert!(!cache.contains("old"));
        assert!(cache.contains("new"));
        assert_eq!(cache.total_bytes(), 6000);
    }

    #[test]
    fn eviction_multiple() {
        let mut cache = ModelCache::new(10_000);
        cache.insert("a".into(), 4000);
        // Touch "a" so "b" becomes oldest after insertion
        std::thread::sleep(std::time::Duration::from_millis(5));
        cache.insert("b".into(), 4000);
        std::thread::sleep(std::time::Duration::from_millis(5));
        // Touch "b" so "a" is oldest
        cache.touch("b");
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Inserting 6000 bytes needs 10000 total but we have 8000 used
        // Should evict "a" (oldest) to free 4000, still need more, evict "b"
        let evicted = cache.insert("c".into(), 6000);
        // "a" should be evicted first since it's oldest
        assert!(evicted.contains(&"a".to_string()));
        assert!(cache.contains("c"));
    }

    #[test]
    fn touch_updates_access() {
        let mut cache = ModelCache::new(10_000);
        cache.insert("a".into(), 3000);
        std::thread::sleep(std::time::Duration::from_millis(5));
        cache.insert("b".into(), 3000);
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Touch "a" so it's no longer the oldest
        cache.touch("a");

        let keys = cache.keys_by_age();
        assert_eq!(keys[0], "b"); // b is oldest now
        assert_eq!(keys[1], "a");
    }

    #[test]
    fn touch_nonexistent_returns_false() {
        let mut cache = ModelCache::new(10_000);
        assert!(!cache.touch("nonexistent"));
    }

    #[test]
    fn remove_entry() {
        let mut cache = ModelCache::new(10_000);
        cache.insert("a".into(), 5000);
        let removed = cache.remove("a");
        assert!(removed.is_some());
        assert!(!cache.contains("a"));
        assert_eq!(cache.total_bytes(), 0);
    }

    #[test]
    fn remove_nonexistent() {
        let mut cache = ModelCache::new(10_000);
        assert!(cache.remove("nope").is_none());
    }

    #[test]
    fn reinsert_same_key() {
        let mut cache = ModelCache::new(10_000);
        cache.insert("a".into(), 5000);
        cache.insert("a".into(), 3000);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_bytes(), 3000);
    }

    #[test]
    fn keys_by_age_ordering() {
        let mut cache = ModelCache::new(100_000);
        cache.insert("first".into(), 100);
        std::thread::sleep(std::time::Duration::from_millis(5));
        cache.insert("second".into(), 100);
        std::thread::sleep(std::time::Duration::from_millis(5));
        cache.insert("third".into(), 100);

        let keys = cache.keys_by_age();
        assert_eq!(keys, vec!["first", "second", "third"]);
    }
}
