//! Response caching for deterministic inference requests.
//!
//! This module provides an LRU cache for caching responses to deterministic
//! requests (temperature=0 or low temperature). This can significantly improve
//! latency and throughput for repeated requests.
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::cache::{ResponseCache, CacheConfig};
//! use std::time::Duration;
//!
//! let config = CacheConfig {
//!     max_entries: 1000,
//!     max_memory_bytes: 100 * 1024 * 1024, // 100MB
//!     ttl: Duration::from_secs(3600),
//!     cacheable_temp_max: 0.0,
//! };
//!
//! let cache = ResponseCache::new(config);
//!
//! // Check for cached response
//! if let Some(response) = cache.get(&key) {
//!     return response;
//! }
//!
//! // Cache new response
//! cache.put(key, response);
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Configuration for the response cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,

    /// Maximum memory usage in bytes (approximate).
    pub max_memory_bytes: usize,

    /// Time-to-live for cache entries.
    pub ttl: Duration,

    /// Maximum temperature value for cacheable requests.
    /// Requests with temperature <= this value are cacheable.
    pub cacheable_temp_max: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_memory_bytes: 100 * 1024 * 1024, // 100MB
            ttl: Duration::from_secs(3600),      // 1 hour
            cacheable_temp_max: 0.0,             // Only temp=0
        }
    }
}

impl CacheConfig {
    /// Creates a new cache config with the given max entries.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            ..Default::default()
        }
    }

    /// Creates a config that disables caching.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            max_entries: 0,
            max_memory_bytes: 0,
            ttl: Duration::ZERO,
            cacheable_temp_max: -1.0,
        }
    }

    /// Sets the TTL for cache entries.
    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Sets the maximum cacheable temperature.
    #[must_use]
    pub fn with_cacheable_temp(mut self, temp: f32) -> Self {
        self.cacheable_temp_max = temp;
        self
    }

    /// Returns true if caching is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.max_entries > 0 && self.ttl > Duration::ZERO
    }
}

/// Cache key for identifying unique requests.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Model identifier.
    model: String,
    /// Hash of the request content.
    content_hash: u64,
    /// Max tokens requested.
    max_tokens: Option<u32>,
}

impl CacheKey {
    /// Creates a new cache key from request parameters.
    #[must_use]
    pub fn new(model: &str, content_hash: u64, max_tokens: Option<u32>) -> Self {
        Self {
            model: model.to_string(),
            content_hash,
            max_tokens,
        }
    }

    /// Creates a cache key from a chat completion request.
    #[must_use]
    pub fn from_chat_request(
        model: &str,
        messages: &[impl AsRef<str>],
        max_tokens: Option<u32>,
    ) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for msg in messages {
            msg.as_ref().hash(&mut hasher);
        }
        Self::new(model, hasher.finish(), max_tokens)
    }

    /// Creates a cache key from a completion request.
    #[must_use]
    pub fn from_completion_request(model: &str, prompt: &str, max_tokens: Option<u32>) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        prompt.hash(&mut hasher);
        Self::new(model, hasher.finish(), max_tokens)
    }

    /// Returns the model for this cache key.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Cached response data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    /// The cached response body.
    pub body: String,
    /// Approximate size in bytes.
    pub size_bytes: usize,
    /// When the entry was created.
    #[serde(skip)]
    created_at: Option<Instant>,
    /// Number of times this entry was accessed.
    #[serde(skip)]
    access_count: u64,
}

impl CachedResponse {
    /// Creates a new cached response.
    #[must_use]
    pub fn new(body: String) -> Self {
        let size_bytes = body.len();
        Self {
            body,
            size_bytes,
            created_at: Some(Instant::now()),
            access_count: 0,
        }
    }

    /// Returns the age of this entry.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.map_or(Duration::ZERO, |t| t.elapsed())
    }

    /// Returns true if this entry has expired.
    #[must_use]
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.age() > ttl
    }

    /// Returns the number of times this entry has been accessed.
    #[must_use]
    pub fn access_count(&self) -> u64 {
        self.access_count
    }

    /// Increments the access count and returns the new value.
    pub fn record_access(&mut self) -> u64 {
        self.access_count += 1;
        self.access_count
    }
}

/// Cache entry with metadata.
struct CacheEntry {
    response: CachedResponse,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
}

impl CacheEntry {
    fn new(response: CachedResponse) -> Self {
        let now = Instant::now();
        Self {
            response,
            created_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    fn size_bytes(&self) -> usize {
        self.response.size_bytes
    }
}

/// Cache metrics for monitoring.
#[derive(Debug, Default)]
pub struct CacheMetrics {
    /// Total cache hits.
    hits: AtomicU64,
    /// Total cache misses.
    misses: AtomicU64,
    /// Total evictions.
    evictions: AtomicU64,
    /// Total expired entries removed.
    expirations: AtomicU64,
    /// Current entry count.
    entry_count: AtomicU64,
    /// Current memory usage (approximate).
    memory_bytes: AtomicU64,
}

impl CacheMetrics {
    /// Creates new cache metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of cache hits.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Returns the number of cache misses.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Returns the cache hit ratio (0.0 - 1.0).
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.hits() as f64;
        let total = hits + self.misses() as f64;
        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }

    /// Returns the number of evictions.
    #[must_use]
    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    /// Returns the number of expirations.
    #[must_use]
    pub fn expirations(&self) -> u64 {
        self.expirations.load(Ordering::Relaxed)
    }

    /// Returns the current entry count.
    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.entry_count.load(Ordering::Relaxed)
    }

    /// Returns the current memory usage.
    #[must_use]
    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes.load(Ordering::Relaxed)
    }

    /// Renders Prometheus-format metrics.
    #[must_use]
    pub fn render_prometheus(&self) -> String {
        format!(
            r#"# HELP infernum_cache_hits_total Total cache hits
# TYPE infernum_cache_hits_total counter
infernum_cache_hits_total {}

# HELP infernum_cache_misses_total Total cache misses
# TYPE infernum_cache_misses_total counter
infernum_cache_misses_total {}

# HELP infernum_cache_hit_ratio Cache hit ratio
# TYPE infernum_cache_hit_ratio gauge
infernum_cache_hit_ratio {:.4}

# HELP infernum_cache_evictions_total Total evictions
# TYPE infernum_cache_evictions_total counter
infernum_cache_evictions_total {}

# HELP infernum_cache_entries Current cache entry count
# TYPE infernum_cache_entries gauge
infernum_cache_entries {}

# HELP infernum_cache_memory_bytes Current cache memory usage
# TYPE infernum_cache_memory_bytes gauge
infernum_cache_memory_bytes {}
"#,
            self.hits(),
            self.misses(),
            self.hit_ratio(),
            self.evictions(),
            self.entry_count(),
            self.memory_bytes(),
        )
    }

    fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    fn record_expiration(&self) {
        self.expirations.fetch_add(1, Ordering::Relaxed);
    }

    fn set_entry_count(&self, count: u64) {
        self.entry_count.store(count, Ordering::Relaxed);
    }

    fn set_memory_bytes(&self, bytes: u64) {
        self.memory_bytes.store(bytes, Ordering::Relaxed);
    }
}

/// Cache result indicating hit or miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheResult {
    /// Cache hit.
    Hit,
    /// Cache miss.
    Miss,
}

impl CacheResult {
    /// Returns the cache header value.
    #[must_use]
    pub fn header_value(&self) -> &'static str {
        match self {
            Self::Hit => "HIT",
            Self::Miss => "MISS",
        }
    }
}

/// Response cache with LRU eviction.
pub struct ResponseCache {
    cache: RwLock<HashMap<CacheKey, CacheEntry>>,
    config: CacheConfig,
    metrics: CacheMetrics,
}

impl ResponseCache {
    /// Creates a new response cache with the given configuration.
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache: RwLock::new(HashMap::with_capacity(config.max_entries)),
            config,
            metrics: CacheMetrics::new(),
        }
    }

    /// Creates a response cache with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Creates a disabled cache.
    #[must_use]
    pub fn disabled() -> Self {
        Self::new(CacheConfig::disabled())
    }

    /// Returns true if caching is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.is_enabled()
    }

    /// Returns true if a request with the given temperature is cacheable.
    #[must_use]
    pub fn is_cacheable_temp(&self, temperature: f32) -> bool {
        temperature <= self.config.cacheable_temp_max
    }

    /// Gets a cached response if available and not expired.
    pub fn get(&self, key: &CacheKey) -> Option<CachedResponse> {
        if !self.is_enabled() {
            return None;
        }

        let mut cache = self.cache.write().ok()?;

        if let Some(entry) = cache.get_mut(key) {
            if entry.is_expired(self.config.ttl) {
                // Entry expired, remove it
                cache.remove(key);
                self.metrics.record_expiration();
                self.metrics.record_miss();
                self.update_metrics(&cache);
                return None;
            }

            entry.touch();
            self.metrics.record_hit();
            return Some(entry.response.clone());
        }

        self.metrics.record_miss();
        None
    }

    /// Stores a response in the cache.
    pub fn put(&self, key: CacheKey, response: CachedResponse) {
        if !self.is_enabled() {
            return;
        }

        let Ok(mut cache) = self.cache.write() else {
            return;
        };

        // Check if key already exists
        if cache.contains_key(&key) {
            cache.insert(key, CacheEntry::new(response));
            self.update_metrics(&cache);
            return;
        }

        // Check entry limit - evict if at capacity
        while cache.len() >= self.config.max_entries {
            self.evict_one_lru(&mut cache);
        }

        // Check memory limit before inserting
        let new_size = response.size_bytes;
        let current_memory: usize = cache.values().map(|e| e.size_bytes()).sum();

        while current_memory + new_size > self.config.max_memory_bytes && !cache.is_empty() {
            self.evict_one_lru(&mut cache);
        }

        // Insert new entry
        cache.insert(key, CacheEntry::new(response));
        self.update_metrics(&cache);
    }

    /// Removes a specific entry from the cache.
    pub fn remove(&self, key: &CacheKey) -> Option<CachedResponse> {
        let Ok(mut cache) = self.cache.write() else {
            return None;
        };

        let entry = cache.remove(key);
        self.update_metrics(&cache);
        entry.map(|e| e.response)
    }

    /// Clears all entries from the cache.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
            self.update_metrics(&cache);
        }
    }

    /// Returns the current number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.read().map_or(0, |c| c.len())
    }

    /// Returns true if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the cache metrics.
    #[must_use]
    pub fn metrics(&self) -> &CacheMetrics {
        &self.metrics
    }

    /// Returns the cache configuration.
    #[must_use]
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Removes expired entries from the cache.
    pub fn cleanup_expired(&self) {
        let Ok(mut cache) = self.cache.write() else {
            return;
        };

        let before = cache.len();
        cache.retain(|_, entry| !entry.is_expired(self.config.ttl));
        let removed = before - cache.len();

        for _ in 0..removed {
            self.metrics.record_expiration();
        }

        self.update_metrics(&cache);
    }

    // Internal methods

    fn evict_one_lru(&self, cache: &mut HashMap<CacheKey, CacheEntry>) {
        if cache.is_empty() {
            return;
        }

        // Find the LRU entry (oldest last_accessed time)
        let lru_key = cache
            .iter()
            .min_by_key(|(_, v)| v.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            cache.remove(&key);
            self.metrics.record_eviction();
        }
    }

    fn update_metrics(&self, cache: &HashMap<CacheKey, CacheEntry>) {
        self.metrics.set_entry_count(cache.len() as u64);

        let total_bytes: usize = cache.values().map(|e| e.size_bytes()).sum();
        self.metrics.set_memory_bytes(total_bytes as u64);
    }
}

impl std::fmt::Debug for ResponseCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponseCache")
            .field("enabled", &self.is_enabled())
            .field("entries", &self.len())
            .field("config", &self.config)
            .finish()
    }
}

/// Header name for cache status.
pub const CACHE_HEADER: &str = "x-cache";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.max_entries, 1000);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_cache_config_disabled() {
        let config = CacheConfig::disabled();
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_cache_config_builder() {
        let config = CacheConfig::new(500)
            .with_ttl(Duration::from_secs(300))
            .with_cacheable_temp(0.1);

        assert_eq!(config.max_entries, 500);
        assert_eq!(config.ttl, Duration::from_secs(300));
        assert!((config.cacheable_temp_max - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_cache_key_from_chat() {
        let key1 = CacheKey::from_chat_request("gpt-4", &["Hello", "World"], Some(100));
        let key2 = CacheKey::from_chat_request("gpt-4", &["Hello", "World"], Some(100));
        let key3 = CacheKey::from_chat_request("gpt-4", &["Hello", "Different"], Some(100));

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_key_from_completion() {
        let key1 = CacheKey::from_completion_request("llama", "Hello world", Some(50));
        let key2 = CacheKey::from_completion_request("llama", "Hello world", Some(50));
        let key3 = CacheKey::from_completion_request("llama", "Different", Some(50));

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_key_different_max_tokens() {
        let key1 = CacheKey::from_completion_request("llama", "Hello", Some(50));
        let key2 = CacheKey::from_completion_request("llama", "Hello", Some(100));

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_cached_response_age() {
        let response = CachedResponse::new("test".to_string());
        std::thread::sleep(Duration::from_millis(10));
        assert!(response.age() >= Duration::from_millis(10));
    }

    #[test]
    fn test_cached_response_expiry() {
        let response = CachedResponse::new("test".to_string());
        assert!(!response.is_expired(Duration::from_secs(60)));
        assert!(response.is_expired(Duration::ZERO));
    }

    #[test]
    fn test_cache_put_and_get() {
        let cache = ResponseCache::with_defaults();
        let key = CacheKey::from_completion_request("test", "hello", None);
        let response = CachedResponse::new("response".to_string());

        cache.put(key.clone(), response);

        let cached = cache.get(&key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().body, "response");
    }

    #[test]
    fn test_cache_miss() {
        let cache = ResponseCache::with_defaults();
        let key = CacheKey::from_completion_request("test", "hello", None);

        assert!(cache.get(&key).is_none());
        assert_eq!(cache.metrics().misses(), 1);
    }

    #[test]
    fn test_cache_hit_metrics() {
        let cache = ResponseCache::with_defaults();
        let key = CacheKey::from_completion_request("test", "hello", None);

        cache.put(key.clone(), CachedResponse::new("response".to_string()));
        let _ = cache.get(&key);

        assert_eq!(cache.metrics().hits(), 1);
    }

    #[test]
    fn test_cache_remove() {
        let cache = ResponseCache::with_defaults();
        let key = CacheKey::from_completion_request("test", "hello", None);

        cache.put(key.clone(), CachedResponse::new("response".to_string()));
        assert!(!cache.is_empty());

        let removed = cache.remove(&key);
        assert!(removed.is_some());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_clear() {
        let cache = ResponseCache::with_defaults();

        for i in 0..10 {
            let key = CacheKey::from_completion_request("test", &format!("hello{}", i), None);
            cache.put(key, CachedResponse::new("response".to_string()));
        }

        assert_eq!(cache.len(), 10);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_ttl_expiry() {
        let config = CacheConfig::new(100).with_ttl(Duration::from_millis(50));
        let cache = ResponseCache::new(config);

        let key = CacheKey::from_completion_request("test", "hello", None);
        cache.put(key.clone(), CachedResponse::new("response".to_string()));

        // Should be in cache
        assert!(cache.get(&key).is_some());

        // Wait for expiry
        std::thread::sleep(Duration::from_millis(100));

        // Should be expired now
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.metrics().expirations(), 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let config = CacheConfig {
            max_entries: 3,
            max_memory_bytes: 1024 * 1024,
            ttl: Duration::from_secs(3600),
            cacheable_temp_max: 0.0,
        };
        let cache = ResponseCache::new(config);

        // Add 3 entries
        for i in 0..3 {
            let key = CacheKey::from_completion_request("test", &format!("msg{}", i), None);
            cache.put(key, CachedResponse::new("response".to_string()));
            std::thread::sleep(Duration::from_millis(10));
        }

        assert_eq!(cache.len(), 3);

        // Access msg1 to make it recently used
        let key1 = CacheKey::from_completion_request("test", "msg1", None);
        let _ = cache.get(&key1);

        // Add new entry - should evict msg0 (LRU)
        let key3 = CacheKey::from_completion_request("test", "msg3", None);
        cache.put(key3.clone(), CachedResponse::new("new".to_string()));

        assert_eq!(cache.len(), 3);

        // msg0 should be evicted
        let key0 = CacheKey::from_completion_request("test", "msg0", None);
        assert!(cache.get(&key0).is_none());

        // msg1 should still be there (was accessed)
        assert!(cache.get(&key1).is_some());
    }

    #[test]
    fn test_cache_disabled() {
        let cache = ResponseCache::disabled();

        assert!(!cache.is_enabled());

        let key = CacheKey::from_completion_request("test", "hello", None);
        cache.put(key.clone(), CachedResponse::new("response".to_string()));

        // Should not store anything
        assert!(cache.is_empty());
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cacheable_temp() {
        let config = CacheConfig::new(100).with_cacheable_temp(0.1);
        let cache = ResponseCache::new(config);

        assert!(cache.is_cacheable_temp(0.0));
        assert!(cache.is_cacheable_temp(0.1));
        assert!(!cache.is_cacheable_temp(0.2));
        assert!(!cache.is_cacheable_temp(1.0));
    }

    #[test]
    fn test_cache_metrics_prometheus() {
        let cache = ResponseCache::with_defaults();

        let key = CacheKey::from_completion_request("test", "hello", None);
        cache.put(key.clone(), CachedResponse::new("response".to_string()));
        let _ = cache.get(&key);
        let _ = cache.get(&CacheKey::from_completion_request("test", "miss", None));

        let output = cache.metrics().render_prometheus();

        assert!(output.contains("infernum_cache_hits_total 1"));
        assert!(output.contains("infernum_cache_misses_total 1"));
        assert!(output.contains("infernum_cache_hit_ratio 0.5"));
    }

    #[test]
    fn test_cache_result_header() {
        assert_eq!(CacheResult::Hit.header_value(), "HIT");
        assert_eq!(CacheResult::Miss.header_value(), "MISS");
    }

    #[test]
    fn test_cache_memory_limit() {
        let config = CacheConfig {
            max_entries: 1000,
            max_memory_bytes: 100, // Very small
            ttl: Duration::from_secs(3600),
            cacheable_temp_max: 0.0,
        };
        let cache = ResponseCache::new(config);

        // Add entry that takes up memory
        let key1 = CacheKey::from_completion_request("test", "msg1", None);
        cache.put(
            key1.clone(),
            CachedResponse::new("x".repeat(50)), // 50 bytes
        );

        assert_eq!(cache.len(), 1);

        // Add another entry that exceeds limit
        let key2 = CacheKey::from_completion_request("test", "msg2", None);
        cache.put(
            key2.clone(),
            CachedResponse::new("y".repeat(60)), // 60 bytes - total would be 110
        );

        // Should have evicted to fit
        assert!(cache.metrics().evictions() > 0);
    }

    #[test]
    fn test_cleanup_expired() {
        let config = CacheConfig::new(100).with_ttl(Duration::from_millis(50));
        let cache = ResponseCache::new(config);

        // Add entries
        for i in 0..5 {
            let key = CacheKey::from_completion_request("test", &format!("msg{}", i), None);
            cache.put(key, CachedResponse::new("response".to_string()));
        }

        assert_eq!(cache.len(), 5);

        // Wait for expiry
        std::thread::sleep(Duration::from_millis(100));

        // Cleanup
        cache.cleanup_expired();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.metrics().expirations(), 5);
    }

    #[test]
    fn test_cache_debug() {
        let cache = ResponseCache::with_defaults();
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("ResponseCache"));
        assert!(debug_str.contains("enabled"));
    }

    #[test]
    fn test_hit_ratio_empty() {
        let metrics = CacheMetrics::new();
        assert_eq!(metrics.hit_ratio(), 0.0);
    }
}
