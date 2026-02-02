//! Request deduplication for coalescing identical in-flight requests.
//!
//! This module provides request deduplication to avoid redundant inference
//! when multiple identical requests arrive concurrently. Only one request
//! is actually processed, and the result is shared with all waiters.
//!
//! # Example
//!
//! ```ignore
//! use infernum_server::dedup::{RequestDeduplicator, DeduplicatorConfig};
//!
//! let dedup = RequestDeduplicator::new(DeduplicatorConfig::default());
//!
//! // Try to acquire - returns None if another request is already in-flight
//! if let Some(handle) = dedup.try_acquire(&request_hash) {
//!     // We're responsible for processing - do inference
//!     let result = do_inference(&request).await;
//!
//!     // Complete the request, waking all waiters
//!     handle.complete(result);
//! } else {
//!     // Wait for the in-flight request to complete
//!     let result = dedup.wait(&request_hash).await;
//! }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::openai::{ChatCompletionRequest, CompletionRequest};

/// Default maximum number of in-flight requests to track.
pub const DEFAULT_MAX_IN_FLIGHT: usize = 1000;

/// Default TTL for in-flight entries (guards against stuck requests).
pub const DEFAULT_IN_FLIGHT_TTL: Duration = Duration::from_secs(300);

/// Configuration for the request deduplicator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicatorConfig {
    /// Whether deduplication is enabled.
    pub enabled: bool,

    /// Maximum number of in-flight requests to track.
    pub max_in_flight: usize,

    /// TTL for in-flight entries (guards against stuck requests).
    pub in_flight_ttl: Duration,

    /// Only deduplicate deterministic requests (temperature <= threshold).
    pub require_deterministic: bool,

    /// Maximum temperature for request to be considered deterministic.
    pub deterministic_temp_max: f32,
}

impl Default for DeduplicatorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_in_flight: DEFAULT_MAX_IN_FLIGHT,
            in_flight_ttl: DEFAULT_IN_FLIGHT_TTL,
            require_deterministic: true,
            deterministic_temp_max: 0.0,
        }
    }
}

impl DeduplicatorConfig {
    /// Creates a disabled configuration.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Builder method to set enabled state.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Builder method to set max in-flight requests.
    pub fn with_max_in_flight(mut self, max: usize) -> Self {
        self.max_in_flight = max;
        self
    }

    /// Builder method to set in-flight TTL.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.in_flight_ttl = ttl;
        self
    }

    /// Builder method to set deterministic requirement.
    pub fn with_require_deterministic(mut self, require: bool) -> Self {
        self.require_deterministic = require;
        self
    }
}

/// A hash of a request used for deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestHash(u64);

impl RequestHash {
    /// Creates a new request hash from a u64 value.
    pub fn new(hash: u64) -> Self {
        Self(hash)
    }

    /// Creates a request hash from a chat completion request.
    pub fn from_chat_request(request: &ChatCompletionRequest) -> Self {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();

        // Hash model
        request.model.hash(&mut hasher);

        // Hash messages
        for msg in &request.messages {
            msg.role.hash(&mut hasher);
            msg.content.hash(&mut hasher);
            if let Some(name) = &msg.name {
                name.hash(&mut hasher);
            }
        }

        // Hash key parameters that affect output
        request.max_tokens.hash(&mut hasher);
        request.stop.hash(&mut hasher);

        // Hash temperature if present (for determinism check)
        if let Some(temp) = request.temperature {
            // Use bits representation for consistent hashing
            temp.to_bits().hash(&mut hasher);
        }

        Self(hasher.finish())
    }

    /// Creates a request hash from a completion request.
    pub fn from_completion_request(request: &CompletionRequest) -> Self {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();

        // Hash model
        request.model.hash(&mut hasher);

        // Hash prompt
        request.prompt.hash(&mut hasher);

        // Hash key parameters
        request.max_tokens.hash(&mut hasher);
        request.stop.hash(&mut hasher);

        // Hash temperature if present
        if let Some(temp) = request.temperature {
            temp.to_bits().hash(&mut hasher);
        }

        Self(hasher.finish())
    }

    /// Returns the raw hash value.
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for RequestHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// Result that can be shared between deduplicated requests.
#[derive(Debug, Clone)]
pub enum DeduplicatedResult<T> {
    /// The result was computed by this request.
    Computed(T),

    /// The result was shared from another request.
    Shared(T),

    /// The original request failed or was cancelled.
    Failed(String),
}

impl<T> DeduplicatedResult<T> {
    /// Returns true if this result was shared from another request.
    pub fn is_shared(&self) -> bool {
        matches!(self, Self::Shared(_))
    }

    /// Returns the inner value if successful.
    pub fn into_inner(self) -> Result<T, String> {
        match self {
            Self::Computed(v) | Self::Shared(v) => Ok(v),
            Self::Failed(e) => Err(e),
        }
    }

    /// Returns a reference to the inner value if successful.
    pub fn as_ref(&self) -> Result<&T, &String> {
        match self {
            Self::Computed(v) | Self::Shared(v) => Ok(v),
            Self::Failed(e) => Err(e),
        }
    }
}

/// Internal state for an in-flight request.
struct InFlightRequest<T: Clone + Send + Sync + 'static> {
    /// When this request started.
    started_at: Instant,

    /// Broadcast sender for result distribution.
    sender: broadcast::Sender<DeduplicatedResult<T>>,

    /// Number of waiters.
    waiter_count: AtomicU64,
}

/// Handle for the request that is responsible for computing the result.
pub struct ComputeHandle<T: Clone + Send + Sync + 'static> {
    hash: RequestHash,
    sender: broadcast::Sender<DeduplicatedResult<T>>,
    completed: bool,
}

impl<T: Clone + Send + Sync + 'static> ComputeHandle<T> {
    /// Completes the request with a successful result.
    pub fn complete(mut self, result: T) {
        self.completed = true;
        // Ignore send errors - waiters may have timed out
        let _ = self.sender.send(DeduplicatedResult::Computed(result));
    }

    /// Fails the request with an error message.
    pub fn fail(mut self, error: String) {
        self.completed = true;
        let _ = self.sender.send(DeduplicatedResult::Failed(error));
    }

    /// Returns the request hash.
    pub fn hash(&self) -> RequestHash {
        self.hash
    }
}

impl<T: Clone + Send + Sync + 'static> Drop for ComputeHandle<T> {
    fn drop(&mut self) {
        if !self.completed {
            // Request was dropped without completing - notify waiters
            let _ = self
                .sender
                .send(DeduplicatedResult::Failed("Request cancelled".to_string()));
        }
    }
}

/// Metrics for the request deduplicator.
#[derive(Debug, Default)]
pub struct DeduplicatorMetrics {
    /// Number of deduplicated requests (shared results).
    pub dedupe_hits: AtomicU64,

    /// Number of unique requests (no deduplication).
    pub dedupe_misses: AtomicU64,

    /// Current number of in-flight requests.
    pub in_flight_count: AtomicU64,

    /// Number of requests that timed out waiting.
    pub wait_timeouts: AtomicU64,

    /// Number of requests that failed.
    pub failures: AtomicU64,
}

impl DeduplicatorMetrics {
    /// Creates new metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the deduplication hit ratio.
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.dedupe_hits.load(Ordering::Relaxed);
        let misses = self.dedupe_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Renders metrics in Prometheus format.
    pub fn render_prometheus(&self) -> String {
        let mut output = String::with_capacity(512);

        output.push_str("# HELP infernum_dedup_hits_total Deduplicated request hits\n");
        output.push_str("# TYPE infernum_dedup_hits_total counter\n");
        output.push_str(&format!(
            "infernum_dedup_hits_total {}\n",
            self.dedupe_hits.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP infernum_dedup_misses_total Deduplicated request misses\n");
        output.push_str("# TYPE infernum_dedup_misses_total counter\n");
        output.push_str(&format!(
            "infernum_dedup_misses_total {}\n",
            self.dedupe_misses.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP infernum_dedup_in_flight Current in-flight requests\n");
        output.push_str("# TYPE infernum_dedup_in_flight gauge\n");
        output.push_str(&format!(
            "infernum_dedup_in_flight {}\n",
            self.in_flight_count.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP infernum_dedup_wait_timeouts_total Wait timeouts\n");
        output.push_str("# TYPE infernum_dedup_wait_timeouts_total counter\n");
        output.push_str(&format!(
            "infernum_dedup_wait_timeouts_total {}\n",
            self.wait_timeouts.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP infernum_dedup_failures_total Request failures\n");
        output.push_str("# TYPE infernum_dedup_failures_total counter\n");
        output.push_str(&format!(
            "infernum_dedup_failures_total {}\n",
            self.failures.load(Ordering::Relaxed)
        ));

        output.push_str("# HELP infernum_dedup_hit_ratio Deduplication hit ratio\n");
        output.push_str("# TYPE infernum_dedup_hit_ratio gauge\n");
        output.push_str(&format!("infernum_dedup_hit_ratio {}\n", self.hit_ratio()));

        output
    }
}

/// Request deduplicator for coalescing identical in-flight requests.
pub struct RequestDeduplicator<T: Clone + Send + Sync + 'static> {
    config: DeduplicatorConfig,
    in_flight: RwLock<HashMap<RequestHash, Arc<InFlightRequest<T>>>>,
    metrics: DeduplicatorMetrics,
}

impl<T: Clone + Send + Sync + 'static> RequestDeduplicator<T> {
    /// Creates a new request deduplicator with the given configuration.
    pub fn new(config: DeduplicatorConfig) -> Self {
        Self {
            config,
            in_flight: RwLock::new(HashMap::new()),
            metrics: DeduplicatorMetrics::new(),
        }
    }

    /// Creates a disabled deduplicator.
    pub fn disabled() -> Self {
        Self::new(DeduplicatorConfig::disabled())
    }

    /// Returns whether deduplication is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Checks if a chat request is eligible for deduplication.
    pub fn is_chat_request_eligible(&self, request: &ChatCompletionRequest) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Streaming requests cannot be deduplicated
        if request.stream.unwrap_or(false) {
            return false;
        }

        // Check determinism requirement
        if self.config.require_deterministic {
            let temp = request.temperature.unwrap_or(1.0);
            if temp > self.config.deterministic_temp_max {
                return false;
            }
        }

        true
    }

    /// Checks if a completion request is eligible for deduplication.
    pub fn is_completion_request_eligible(&self, request: &CompletionRequest) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Streaming requests cannot be deduplicated
        if request.stream.unwrap_or(false) {
            return false;
        }

        // Check determinism requirement
        if self.config.require_deterministic {
            let temp = request.temperature.unwrap_or(1.0);
            if temp > self.config.deterministic_temp_max {
                return false;
            }
        }

        true
    }

    /// Tries to acquire a compute handle for a request.
    ///
    /// Returns `Some(handle)` if this request should compute the result,
    /// or `None` if another request is already in-flight for this hash.
    pub fn try_acquire(&self, hash: RequestHash) -> Option<ComputeHandle<T>> {
        if !self.config.enabled {
            return Some(self.create_handle_disabled(hash));
        }

        // Clean up expired entries first
        self.cleanup_expired();

        let mut in_flight = self.in_flight.write();

        // Check if already in-flight
        if in_flight.contains_key(&hash) {
            // Record dedup hit
            self.metrics.dedupe_hits.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Check capacity
        if in_flight.len() >= self.config.max_in_flight {
            // At capacity - just process without deduplication
            self.metrics.dedupe_misses.fetch_add(1, Ordering::Relaxed);
            return Some(self.create_handle_disabled(hash));
        }

        // Create new in-flight entry
        let (sender, _) = broadcast::channel(1);
        let entry = Arc::new(InFlightRequest {
            started_at: Instant::now(),
            sender: sender.clone(),
            waiter_count: AtomicU64::new(0),
        });

        in_flight.insert(hash, entry);
        self.metrics.in_flight_count.fetch_add(1, Ordering::Relaxed);
        self.metrics.dedupe_misses.fetch_add(1, Ordering::Relaxed);

        Some(ComputeHandle {
            hash,
            sender,
            completed: false,
        })
    }

    /// Waits for an in-flight request to complete.
    ///
    /// Returns `None` if no in-flight request exists for this hash.
    pub async fn wait(&self, hash: &RequestHash) -> Option<DeduplicatedResult<T>> {
        let receiver = {
            let in_flight = self.in_flight.read();
            let entry = in_flight.get(hash)?;
            entry.waiter_count.fetch_add(1, Ordering::Relaxed);
            entry.sender.subscribe()
        };

        match tokio::time::timeout(self.config.in_flight_ttl, self.receive(receiver)).await {
            Ok(result) => Some(result),
            Err(_) => {
                self.metrics.wait_timeouts.fetch_add(1, Ordering::Relaxed);
                Some(DeduplicatedResult::Failed("Wait timeout".to_string()))
            }
        }
    }

    /// Helper to receive result and convert to Shared variant.
    async fn receive(
        &self,
        mut receiver: broadcast::Receiver<DeduplicatedResult<T>>,
    ) -> DeduplicatedResult<T> {
        match receiver.recv().await {
            Ok(DeduplicatedResult::Computed(v)) => DeduplicatedResult::Shared(v),
            Ok(DeduplicatedResult::Shared(v)) => DeduplicatedResult::Shared(v),
            Ok(DeduplicatedResult::Failed(e)) => {
                self.metrics.failures.fetch_add(1, Ordering::Relaxed);
                DeduplicatedResult::Failed(e)
            }
            Err(_) => {
                self.metrics.failures.fetch_add(1, Ordering::Relaxed);
                DeduplicatedResult::Failed("Channel closed".to_string())
            }
        }
    }

    /// Removes an entry from the in-flight map.
    pub fn remove(&self, hash: &RequestHash) {
        let mut in_flight = self.in_flight.write();
        if in_flight.remove(hash).is_some() {
            self.metrics.in_flight_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Returns the current number of in-flight requests.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.read().len()
    }

    /// Returns the metrics.
    pub fn metrics(&self) -> &DeduplicatorMetrics {
        &self.metrics
    }

    /// Creates a handle that doesn't participate in deduplication.
    fn create_handle_disabled(&self, hash: RequestHash) -> ComputeHandle<T> {
        let (sender, _) = broadcast::channel(1);
        ComputeHandle {
            hash,
            sender,
            completed: false,
        }
    }

    /// Cleans up expired in-flight entries.
    fn cleanup_expired(&self) {
        let now = Instant::now();
        let ttl = self.config.in_flight_ttl;

        let mut in_flight = self.in_flight.write();
        let initial_len = in_flight.len();

        in_flight.retain(|_, entry| now.duration_since(entry.started_at) < ttl);

        let removed = initial_len - in_flight.len();
        if removed > 0 {
            self.metrics
                .in_flight_count
                .fetch_sub(removed as u64, Ordering::Relaxed);
        }
    }
}

impl<T: Clone + Send + Sync + 'static> fmt::Debug for RequestDeduplicator<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestDeduplicator")
            .field("enabled", &self.config.enabled)
            .field("in_flight_count", &self.in_flight_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DeduplicatorConfig::default();

        assert!(config.enabled);
        assert_eq!(config.max_in_flight, DEFAULT_MAX_IN_FLIGHT);
        assert_eq!(config.in_flight_ttl, DEFAULT_IN_FLIGHT_TTL);
        assert!(config.require_deterministic);
        assert_eq!(config.deterministic_temp_max, 0.0);
    }

    #[test]
    fn test_config_disabled() {
        let config = DeduplicatorConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_config_builder() {
        let config = DeduplicatorConfig::default()
            .with_enabled(false)
            .with_max_in_flight(500)
            .with_ttl(Duration::from_secs(60))
            .with_require_deterministic(false);

        assert!(!config.enabled);
        assert_eq!(config.max_in_flight, 500);
        assert_eq!(config.in_flight_ttl, Duration::from_secs(60));
        assert!(!config.require_deterministic);
    }

    #[test]
    fn test_request_hash_new() {
        let hash = RequestHash::new(12345);
        assert_eq!(hash.as_u64(), 12345);
    }

    #[test]
    fn test_request_hash_display() {
        let hash = RequestHash::new(0x1234567890ABCDEF);
        assert_eq!(hash.to_string(), "1234567890abcdef");
    }

    #[test]
    fn test_request_hash_equality() {
        let hash1 = RequestHash::new(100);
        let hash2 = RequestHash::new(100);
        let hash3 = RequestHash::new(200);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    // Helper function to create a chat request for testing
    fn make_chat_request(model: &str, content: &str, temp: f32) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![crate::openai::ChatMessage {
                role: "user".to_string(),
                content: content.to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: Some(temp),
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            max_tokens: Some(100),
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            response_format: None,
        }
    }

    // Helper function to create a completion request for testing
    fn make_completion_request(model: &str, prompt: &str, temp: f32) -> CompletionRequest {
        CompletionRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            temperature: Some(temp),
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            max_tokens: Some(100),
            logprobs: None,
            echo: None,
            suffix: None,
            presence_penalty: None,
            frequency_penalty: None,
        }
    }

    #[test]
    fn test_request_hash_from_chat_request() {
        let request1 = make_chat_request("llama", "Hello", 0.0);
        let request2 = make_chat_request("llama", "Hello", 0.0);
        let request3 = make_chat_request("llama", "World", 0.0); // Different content

        let hash1 = RequestHash::from_chat_request(&request1);
        let hash2 = RequestHash::from_chat_request(&request2);
        let hash3 = RequestHash::from_chat_request(&request3);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_request_hash_from_completion_request() {
        let request1 = make_completion_request("llama", "Hello", 0.0);
        let request2 = make_completion_request("llama", "Hello", 0.0);

        let hash1 = RequestHash::from_completion_request(&request1);
        let hash2 = RequestHash::from_completion_request(&request2);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_deduplicated_result_computed() {
        let result: DeduplicatedResult<String> = DeduplicatedResult::Computed("test".to_string());

        assert!(!result.is_shared());
        assert_eq!(result.as_ref(), Ok(&"test".to_string()));
        assert_eq!(result.into_inner(), Ok("test".to_string()));
    }

    #[test]
    fn test_deduplicated_result_shared() {
        let result: DeduplicatedResult<String> = DeduplicatedResult::Shared("test".to_string());

        assert!(result.is_shared());
        assert_eq!(result.as_ref(), Ok(&"test".to_string()));
        assert_eq!(result.into_inner(), Ok("test".to_string()));
    }

    #[test]
    fn test_deduplicated_result_failed() {
        let result: DeduplicatedResult<String> = DeduplicatedResult::Failed("error".to_string());

        assert!(!result.is_shared());
        assert_eq!(result.as_ref(), Err(&"error".to_string()));
        assert_eq!(result.into_inner(), Err("error".to_string()));
    }

    #[test]
    fn test_deduplicator_disabled() {
        let dedup: RequestDeduplicator<String> = RequestDeduplicator::disabled();

        assert!(!dedup.is_enabled());
    }

    #[test]
    fn test_deduplicator_new() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());

        assert!(dedup.is_enabled());
        assert_eq!(dedup.in_flight_count(), 0);
    }

    #[test]
    fn test_try_acquire_first_request() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());
        let hash = RequestHash::new(100);

        let handle = dedup.try_acquire(hash);

        assert!(handle.is_some());
        assert_eq!(dedup.in_flight_count(), 1);
    }

    #[test]
    fn test_try_acquire_duplicate_returns_none() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());
        let hash = RequestHash::new(100);

        let handle1 = dedup.try_acquire(hash);
        let handle2 = dedup.try_acquire(hash);

        assert!(handle1.is_some());
        assert!(handle2.is_none());
        assert_eq!(dedup.in_flight_count(), 1);

        // Record that we got a hit
        assert_eq!(dedup.metrics().dedupe_hits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_try_acquire_different_hashes() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());

        let handle1 = dedup.try_acquire(RequestHash::new(100));
        let handle2 = dedup.try_acquire(RequestHash::new(200));

        assert!(handle1.is_some());
        assert!(handle2.is_some());
        assert_eq!(dedup.in_flight_count(), 2);
    }

    #[test]
    fn test_compute_handle_complete() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());
        let hash = RequestHash::new(100);

        let handle = dedup.try_acquire(hash).expect("should acquire");
        handle.complete("result".to_string());

        // Entry should still be in map (cleanup happens on next access)
        assert_eq!(dedup.in_flight_count(), 1);
    }

    #[test]
    fn test_compute_handle_fail() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());
        let hash = RequestHash::new(100);

        let handle = dedup.try_acquire(hash).expect("should acquire");
        handle.fail("error".to_string());

        assert_eq!(dedup.in_flight_count(), 1);
    }

    #[test]
    fn test_remove() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());
        let hash = RequestHash::new(100);

        let _handle = dedup.try_acquire(hash).expect("should acquire");
        assert_eq!(dedup.in_flight_count(), 1);

        dedup.remove(&hash);
        assert_eq!(dedup.in_flight_count(), 0);
    }

    // Helper to create a chat request with specific stream setting
    fn make_chat_request_with_stream(
        model: &str,
        temp: f32,
        stream: Option<bool>,
    ) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![],
            temperature: Some(temp),
            top_p: None,
            n: None,
            stream,
            stop: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            user: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            response_format: None,
        }
    }

    // Helper to create a completion request with specific stream setting
    fn make_completion_request_with_stream(
        model: &str,
        prompt: &str,
        temp: f32,
        stream: Option<bool>,
    ) -> CompletionRequest {
        CompletionRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            temperature: Some(temp),
            top_p: None,
            n: None,
            stream,
            stop: None,
            max_tokens: None,
            logprobs: None,
            echo: None,
            suffix: None,
            presence_penalty: None,
            frequency_penalty: None,
        }
    }

    #[test]
    fn test_is_chat_request_eligible_disabled() {
        let dedup: RequestDeduplicator<String> = RequestDeduplicator::disabled();
        let request = make_chat_request_with_stream("llama", 0.0, None);

        assert!(!dedup.is_chat_request_eligible(&request));
    }

    #[test]
    fn test_is_chat_request_eligible_streaming() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());
        let request = make_chat_request_with_stream("llama", 0.0, Some(true));

        assert!(!dedup.is_chat_request_eligible(&request));
    }

    #[test]
    fn test_is_chat_request_eligible_non_deterministic() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());
        let request = make_chat_request_with_stream("llama", 0.7, Some(false)); // Non-deterministic

        assert!(!dedup.is_chat_request_eligible(&request));
    }

    #[test]
    fn test_is_chat_request_eligible_valid() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());
        let request = make_chat_request_with_stream("llama", 0.0, Some(false));

        assert!(dedup.is_chat_request_eligible(&request));
    }

    #[test]
    fn test_is_completion_request_eligible() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());

        let eligible = make_completion_request_with_stream("llama", "Hello", 0.0, None);
        let not_eligible = make_completion_request_with_stream("llama", "Hello", 0.7, Some(true));

        assert!(dedup.is_completion_request_eligible(&eligible));
        assert!(!dedup.is_completion_request_eligible(&not_eligible));
    }

    #[test]
    fn test_metrics_new() {
        let metrics = DeduplicatorMetrics::new();

        assert_eq!(metrics.dedupe_hits.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.dedupe_misses.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.in_flight_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_hit_ratio_empty() {
        let metrics = DeduplicatorMetrics::new();
        assert_eq!(metrics.hit_ratio(), 0.0);
    }

    #[test]
    fn test_metrics_hit_ratio() {
        let metrics = DeduplicatorMetrics::new();
        metrics.dedupe_hits.store(30, Ordering::Relaxed);
        metrics.dedupe_misses.store(70, Ordering::Relaxed);

        assert!((metrics.hit_ratio() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_metrics_prometheus() {
        let metrics = DeduplicatorMetrics::new();
        metrics.dedupe_hits.store(10, Ordering::Relaxed);
        metrics.dedupe_misses.store(90, Ordering::Relaxed);

        let output = metrics.render_prometheus();

        assert!(output.contains("infernum_dedup_hits_total 10"));
        assert!(output.contains("infernum_dedup_misses_total 90"));
        assert!(output.contains("infernum_dedup_hit_ratio 0.1"));
    }

    #[test]
    fn test_deduplicator_max_in_flight() {
        let config = DeduplicatorConfig::default().with_max_in_flight(2);
        let dedup: RequestDeduplicator<String> = RequestDeduplicator::new(config);

        // Fill up to capacity
        let _h1 = dedup.try_acquire(RequestHash::new(1));
        let _h2 = dedup.try_acquire(RequestHash::new(2));

        // At capacity - should still return handle but not participate in dedup
        let h3 = dedup.try_acquire(RequestHash::new(3));
        assert!(h3.is_some());

        // Original entries still in map
        assert_eq!(dedup.in_flight_count(), 2);
    }

    #[test]
    fn test_deduplicator_debug() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());

        let debug_str = format!("{:?}", dedup);
        assert!(debug_str.contains("RequestDeduplicator"));
        assert!(debug_str.contains("enabled: true"));
    }

    #[tokio::test]
    async fn test_wait_returns_none_when_not_in_flight() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());
        let hash = RequestHash::new(100);

        let result = dedup.wait(&hash).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_wait_receives_result() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());
        let hash = RequestHash::new(100);

        let handle = dedup.try_acquire(hash).expect("should acquire");

        // Spawn waiter
        let dedup_clone = Arc::new(dedup);
        let dedup_wait = Arc::clone(&dedup_clone);

        let waiter = tokio::spawn(async move { dedup_wait.wait(&hash).await });

        // Give waiter time to subscribe
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Complete the request
        handle.complete("result".to_string());

        let result = waiter.await.expect("join").expect("result");
        assert!(result.is_shared());
        assert_eq!(result.into_inner(), Ok("result".to_string()));
    }

    #[tokio::test]
    async fn test_compute_handle_drop_notifies_waiters() {
        let dedup: RequestDeduplicator<String> =
            RequestDeduplicator::new(DeduplicatorConfig::default());
        let hash = RequestHash::new(100);

        let handle = dedup.try_acquire(hash).expect("should acquire");

        // Spawn waiter
        let dedup_clone = Arc::new(dedup);
        let dedup_wait = Arc::clone(&dedup_clone);

        let waiter = tokio::spawn(async move { dedup_wait.wait(&hash).await });

        // Give waiter time to subscribe
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Drop handle without completing
        drop(handle);

        let result = waiter.await.expect("join").expect("result");
        assert!(matches!(result, DeduplicatedResult::Failed(_)));
    }
}
