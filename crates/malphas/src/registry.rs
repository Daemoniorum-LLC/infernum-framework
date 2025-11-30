//! Model registry for tracking available models.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use abaddon::Engine;
use dashmap::DashMap;
use infernum_core::ModelId;
use parking_lot::RwLock;

/// Model cost configuration.
#[derive(Debug, Clone)]
pub struct ModelCost {
    /// Cost per input token in USD (e.g., 0.000001 = $0.001 per 1K tokens).
    pub input_token_cost: f64,
    /// Cost per output token in USD.
    pub output_token_cost: f64,
    /// Fixed cost per request in USD.
    pub request_cost: f64,
}

impl Default for ModelCost {
    fn default() -> Self {
        Self {
            input_token_cost: 0.0,
            output_token_cost: 0.0,
            request_cost: 0.0,
        }
    }
}

impl ModelCost {
    /// Creates a new cost configuration.
    #[must_use]
    pub fn new(input_token_cost: f64, output_token_cost: f64) -> Self {
        Self {
            input_token_cost,
            output_token_cost,
            request_cost: 0.0,
        }
    }

    /// Calculates the cost for a request.
    #[must_use]
    pub fn calculate(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        self.request_cost
            + (input_tokens as f64 * self.input_token_cost)
            + (output_tokens as f64 * self.output_token_cost)
    }
}

/// Latency statistics for a model.
#[derive(Debug, Default)]
pub struct LatencyStats {
    /// Sum of all latencies in microseconds.
    total_latency_us: AtomicU64,
    /// Number of completed requests.
    request_count: AtomicU64,
    /// Recent latencies for percentile calculation (circular buffer).
    recent_latencies: RwLock<Vec<u64>>,
    /// Maximum buffer size.
    max_buffer_size: usize,
}

impl LatencyStats {
    /// Creates new latency stats.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_latency_us: AtomicU64::new(0),
            request_count: AtomicU64::new(0),
            recent_latencies: RwLock::new(Vec::with_capacity(1000)),
            max_buffer_size: 1000,
        }
    }

    /// Records a latency measurement.
    pub fn record(&self, duration: Duration) {
        let us = duration.as_micros() as u64;
        self.total_latency_us.fetch_add(us, Ordering::Relaxed);
        self.request_count.fetch_add(1, Ordering::Relaxed);

        let mut recent = self.recent_latencies.write();
        if recent.len() >= self.max_buffer_size {
            recent.remove(0);
        }
        recent.push(us);
    }

    /// Returns the average latency in milliseconds.
    #[must_use]
    pub fn average_latency_ms(&self) -> f64 {
        let count = self.request_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let total = self.total_latency_us.load(Ordering::Relaxed);
        (total as f64 / count as f64) / 1000.0
    }

    /// Returns the P50 latency in milliseconds.
    #[must_use]
    pub fn p50_latency_ms(&self) -> f64 {
        self.percentile_latency_ms(50)
    }

    /// Returns the P99 latency in milliseconds.
    #[must_use]
    pub fn p99_latency_ms(&self) -> f64 {
        self.percentile_latency_ms(99)
    }

    /// Returns the specified percentile latency in milliseconds.
    #[must_use]
    pub fn percentile_latency_ms(&self, percentile: u8) -> f64 {
        let recent = self.recent_latencies.read();
        if recent.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<_> = recent.clone();
        sorted.sort_unstable();

        let idx = ((percentile as f64 / 100.0) * (sorted.len() - 1) as f64) as usize;
        sorted.get(idx).copied().unwrap_or(0) as f64 / 1000.0
    }

    /// Returns the request count.
    #[must_use]
    pub fn request_count(&self) -> u64 {
        self.request_count.load(Ordering::Relaxed)
    }
}

/// A registered model in the system.
pub struct RegisteredModel {
    /// Model identifier.
    pub id: ModelId,
    /// The inference engine.
    pub engine: Arc<Engine>,
    /// Number of active requests.
    pub active_requests: AtomicU32,
    /// Latency statistics.
    pub latency_stats: LatencyStats,
    /// Cost configuration.
    pub cost: RwLock<ModelCost>,
    /// Model capabilities/tags.
    pub capabilities: RwLock<Vec<String>>,
    /// Whether the model is healthy.
    pub healthy: std::sync::atomic::AtomicBool,
    /// Last health check timestamp.
    pub last_health_check: RwLock<Option<Instant>>,
}

impl RegisteredModel {
    /// Creates a new registered model.
    pub fn new(id: ModelId, engine: Arc<Engine>) -> Self {
        Self {
            id,
            engine,
            active_requests: AtomicU32::new(0),
            latency_stats: LatencyStats::new(),
            cost: RwLock::new(ModelCost::default()),
            capabilities: RwLock::new(Vec::new()),
            healthy: std::sync::atomic::AtomicBool::new(true),
            last_health_check: RwLock::new(None),
        }
    }

    /// Sets the cost configuration.
    pub fn set_cost(&self, cost: ModelCost) {
        *self.cost.write() = cost;
    }

    /// Adds a capability tag.
    pub fn add_capability(&self, capability: impl Into<String>) {
        self.capabilities.write().push(capability.into());
    }

    /// Checks if the model has a specific capability.
    #[must_use]
    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.read().iter().any(|c| c == capability)
    }

    /// Marks the start of a request (for tracking).
    pub fn start_request(&self) {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Marks the end of a request with latency.
    pub fn end_request(&self, duration: Duration) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.latency_stats.record(duration);
    }

    /// Calculates estimated cost for a request.
    #[must_use]
    pub fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        self.cost.read().calculate(input_tokens, output_tokens)
    }

    /// Returns current load factor (0.0-1.0+).
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        // Simple load factor based on active requests
        // Can be enhanced with GPU utilization, memory usage, etc.
        let active = self.active_requests.load(Ordering::Relaxed) as f64;
        active / 10.0 // Assume ~10 concurrent requests is full load
    }

    /// Returns whether the model is available for requests.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }
}

/// Registry of available models.
pub struct ModelRegistry {
    models: DashMap<String, Arc<RegisteredModel>>,
}

impl ModelRegistry {
    /// Creates a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            models: DashMap::new(),
        }
    }

    /// Registers a model.
    pub fn register(&self, model_id: impl Into<String>, engine: Arc<Engine>) {
        let id = model_id.into();
        let registered = Arc::new(RegisteredModel::new(ModelId::new(&id), engine));
        self.models.insert(id, registered);
    }

    /// Registers a model with cost configuration.
    pub fn register_with_cost(
        &self,
        model_id: impl Into<String>,
        engine: Arc<Engine>,
        cost: ModelCost,
    ) {
        let id = model_id.into();
        let registered = Arc::new(RegisteredModel::new(ModelId::new(&id), engine));
        registered.set_cost(cost);
        self.models.insert(id, registered);
    }

    /// Unregisters a model.
    pub fn unregister(&self, model_id: &str) {
        self.models.remove(model_id);
    }

    /// Gets a model by ID.
    #[must_use]
    pub fn get(&self, model_id: &str) -> Option<Arc<RegisteredModel>> {
        self.models.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Lists all registered models.
    #[must_use]
    pub fn list(&self) -> Vec<String> {
        self.models.iter().map(|r| r.key().clone()).collect()
    }

    /// Lists all available (healthy) models.
    #[must_use]
    pub fn list_available(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|r| r.is_available())
            .map(|r| r.key().clone())
            .collect()
    }

    /// Returns all registered models.
    #[must_use]
    pub fn all(&self) -> Vec<Arc<RegisteredModel>> {
        self.models.iter().map(|r| Arc::clone(&r)).collect()
    }

    /// Returns the number of registered models.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns true if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Finds models matching a capability.
    #[must_use]
    pub fn find_by_capability(&self, capability: &str) -> Vec<Arc<RegisteredModel>> {
        self.models
            .iter()
            .filter(|r| r.has_capability(capability))
            .map(|r| Arc::clone(&r))
            .collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_stats() {
        let stats = LatencyStats::new();

        stats.record(Duration::from_millis(10));
        stats.record(Duration::from_millis(20));
        stats.record(Duration::from_millis(30));

        assert_eq!(stats.request_count(), 3);
        assert!((stats.average_latency_ms() - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_model_cost() {
        let cost = ModelCost::new(0.001, 0.002);
        let total = cost.calculate(1000, 500);
        // 1000 * 0.001 + 500 * 0.002 = 1.0 + 1.0 = 2.0
        assert!((total - 2.0).abs() < 0.001);
    }
}
