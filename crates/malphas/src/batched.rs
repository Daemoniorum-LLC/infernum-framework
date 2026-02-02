//! Batched inference service that wires the scheduler to the inference engine.
//!
//! This module provides continuous batching for efficient GPU utilization by
//! collecting multiple requests and processing them together.
//!
//! ## Architecture
//!
//! ```text
//!   Request 1 ─┐
//!   Request 2 ─┼─► BatchScheduler ─► BatchProcessor ─► Engine
//!   Request 3 ─┘         │                  │
//!                   Priority Queue     Batch Formation
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use malphas::batched::{BatchedInferenceService, BatchedServiceConfig};
//! use abaddon::Engine;
//!
//! let engine = Engine::new(config).await?;
//! let service = BatchedInferenceService::new(engine, BatchedServiceConfig::default());
//!
//! // Start the batch processing loop
//! service.start().await;
//!
//! // Submit requests (they'll be batched automatically)
//! let response = service.generate(request, Priority::Normal).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot, RwLock};

use abaddon::{Engine, InferenceEngine};
use infernum_core::{GenerateRequest, GenerateResponse, RequestId, Result};

use crate::scheduler::{BatchScheduler, Priority, SchedulerConfig, SchedulerStats};
use crate::thermal::{ThermalManager, ThermalState};

/// Configuration for the batched inference service.
#[derive(Debug, Clone)]
pub struct BatchedServiceConfig {
    /// Scheduler configuration.
    pub scheduler: SchedulerConfig,
    /// Enable thermal-aware batch sizing.
    pub thermal_aware: bool,
    /// Minimum batch interval (even if batch is ready).
    pub min_batch_interval: Duration,
    /// Maximum in-flight batches.
    pub max_in_flight_batches: usize,
    /// Channel buffer size for responses.
    pub response_channel_size: usize,
}

impl Default for BatchedServiceConfig {
    fn default() -> Self {
        Self {
            scheduler: SchedulerConfig::default(),
            thermal_aware: true,
            min_batch_interval: Duration::from_millis(10),
            max_in_flight_batches: 4,
            response_channel_size: 1024,
        }
    }
}

impl BatchedServiceConfig {
    /// Creates a configuration optimized for low latency.
    #[must_use]
    pub fn low_latency() -> Self {
        Self {
            scheduler: SchedulerConfig {
                max_batch_size: 8,
                max_wait_time: Duration::from_millis(10),
                ..Default::default()
            },
            min_batch_interval: Duration::from_millis(5),
            ..Default::default()
        }
    }

    /// Creates a configuration optimized for high throughput.
    #[must_use]
    pub fn high_throughput() -> Self {
        Self {
            scheduler: SchedulerConfig {
                max_batch_size: 64,
                max_batch_tokens: 32768,
                max_wait_time: Duration::from_millis(100),
                ..Default::default()
            },
            min_batch_interval: Duration::from_millis(20),
            max_in_flight_batches: 8,
            ..Default::default()
        }
    }
}

/// Pending request waiting for a response.
struct PendingRequest {
    response_tx: oneshot::Sender<Result<GenerateResponse>>,
}

/// Statistics for the batched inference service.
#[derive(Debug, Clone, Default)]
pub struct BatchedServiceStats {
    /// Scheduler statistics.
    pub scheduler: SchedulerStats,
    /// Total inference time in milliseconds.
    pub total_inference_time_ms: u64,
    /// Number of batches processed.
    pub batches_processed: u64,
    /// Average batch inference time.
    pub avg_batch_time_ms: f64,
    /// Current thermal state.
    pub thermal_state: Option<ThermalState>,
    /// Current effective batch size (after thermal adjustment).
    pub effective_batch_size: usize,
}

/// A batched inference service that processes requests efficiently.
pub struct BatchedInferenceService {
    /// The inference engine.
    engine: Arc<Engine>,
    /// The batch scheduler.
    scheduler: Arc<BatchScheduler>,
    /// Configuration.
    config: BatchedServiceConfig,
    /// Pending requests awaiting responses.
    pending: Arc<Mutex<HashMap<RequestId, PendingRequest>>>,
    /// Thermal manager for adaptive batch sizing.
    thermal: Option<Arc<ThermalManager>>,
    /// Service statistics.
    stats: Arc<Mutex<BatchedServiceStats>>,
    /// Shutdown signal.
    shutdown: Arc<RwLock<bool>>,
    /// Request submission channel.
    request_tx: mpsc::Sender<(GenerateRequest, Priority, oneshot::Sender<Result<GenerateResponse>>)>,
    /// Request receiver (moved to processor task).
    request_rx: Arc<Mutex<Option<mpsc::Receiver<(GenerateRequest, Priority, oneshot::Sender<Result<GenerateResponse>>)>>>>,
}

impl BatchedInferenceService {
    /// Creates a new batched inference service.
    #[must_use]
    pub fn new(engine: Engine, config: BatchedServiceConfig) -> Self {
        let scheduler = Arc::new(BatchScheduler::new(config.scheduler.clone()));
        let (request_tx, request_rx) = mpsc::channel(config.response_channel_size);

        Self {
            engine: Arc::new(engine),
            scheduler,
            config,
            pending: Arc::new(Mutex::new(HashMap::new())),
            thermal: None,
            stats: Arc::new(Mutex::new(BatchedServiceStats::default())),
            shutdown: Arc::new(RwLock::new(false)),
            request_tx,
            request_rx: Arc::new(Mutex::new(Some(request_rx))),
        }
    }

    /// Creates a new service with thermal management.
    #[must_use]
    pub fn with_thermal(engine: Engine, config: BatchedServiceConfig, thermal: ThermalManager) -> Self {
        let mut service = Self::new(engine, config);
        service.thermal = Some(Arc::new(thermal));
        service
    }

    /// Returns the scheduler.
    #[must_use]
    pub fn scheduler(&self) -> &BatchScheduler {
        &self.scheduler
    }

    /// Returns service statistics.
    #[must_use]
    pub fn stats(&self) -> BatchedServiceStats {
        let mut stats = self.stats.lock().clone();
        stats.scheduler = self.scheduler.stats();
        // Note: thermal_state is updated in the batch processor loop
        stats
    }

    /// Returns service statistics with async thermal state.
    pub async fn stats_async(&self) -> BatchedServiceStats {
        let mut stats = self.stats.lock().clone();
        stats.scheduler = self.scheduler.stats();
        if let Some(thermal) = &self.thermal {
            stats.thermal_state = Some(thermal.state().await);
        }
        stats
    }

    /// Submits a request for batched processing.
    ///
    /// Returns a future that resolves when the request is processed.
    pub async fn generate(
        &self,
        request: GenerateRequest,
        priority: Priority,
    ) -> Result<GenerateResponse> {
        let (response_tx, response_rx) = oneshot::channel();

        // Send request through channel
        self.request_tx
            .send((request, priority, response_tx))
            .await
            .map_err(|_| infernum_core::Error::internal("Service shutdown"))?;

        // Wait for response
        response_rx
            .await
            .map_err(|_| infernum_core::Error::internal("Request cancelled"))?
    }

    /// Starts the batch processing loop.
    ///
    /// This spawns a background task that continuously:
    /// 1. Collects requests from the scheduler
    /// 2. Forms optimal batches
    /// 3. Processes batches through the engine
    /// 4. Dispatches responses to waiting callers
    pub async fn start(&self) {
        // Take ownership of the receiver
        let request_rx = self.request_rx.lock().take();
        let Some(mut request_rx) = request_rx else {
            tracing::warn!("BatchedInferenceService already started");
            return;
        };

        let engine = Arc::clone(&self.engine);
        let scheduler = Arc::clone(&self.scheduler);
        let pending = Arc::clone(&self.pending);
        let thermal = self.thermal.clone();
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);
        let shutdown = Arc::clone(&self.shutdown);

        // Spawn request receiver task
        let scheduler_clone = Arc::clone(&scheduler);
        let pending_clone = Arc::clone(&pending);
        tokio::spawn(async move {
            while let Some((request, priority, response_tx)) = request_rx.recv().await {
                let request_id = request.request_id.clone();

                // Store the response channel
                pending_clone.lock().insert(
                    request_id.clone(),
                    PendingRequest { response_tx },
                );

                // Enqueue in scheduler
                if !scheduler_clone.enqueue(request, priority) {
                    // Queue full - immediately respond with error
                    if let Some(pending_req) = pending_clone.lock().remove(&request_id) {
                        let _ = pending_req.response_tx.send(Err(
                            infernum_core::Error::internal("Request queue full"),
                        ));
                    }
                }
            }
        });

        // Spawn batch processor task
        tokio::spawn(async move {
            let mut last_batch_time = std::time::Instant::now();

            loop {
                // Check shutdown
                if *shutdown.read().await {
                    tracing::info!("BatchedInferenceService shutting down");
                    break;
                }

                // Wait for requests or timeout
                let has_requests = scheduler
                    .wait_for_requests_timeout(config.min_batch_interval)
                    .await;

                if !has_requests && scheduler.is_empty() {
                    continue;
                }

                // Enforce minimum batch interval
                let elapsed = last_batch_time.elapsed();
                if elapsed < config.min_batch_interval {
                    tokio::time::sleep(config.min_batch_interval - elapsed).await;
                }

                // Get effective batch size (thermal-adjusted)
                let effective_batch_size = if let Some(ref thermal) = thermal {
                    let max_batch = config.scheduler.max_batch_size as u32;
                    thermal.recommended_batch_size(max_batch).await as usize
                } else {
                    config.scheduler.max_batch_size
                };

                // Update stats
                stats.lock().effective_batch_size = effective_batch_size;

                // Dequeue a batch
                let batch = scheduler.dequeue_batch();
                if batch.is_empty() {
                    continue;
                }

                let batch_size = batch.len();
                let batch_start = std::time::Instant::now();

                tracing::debug!(
                    batch_size = batch_size,
                    effective_max = effective_batch_size,
                    "Processing batch"
                );

                // Process batch using parallel batch API
                // This uses async parallelism for overlapping tokenization/response assembly
                let request_ids: Vec<_> = batch.iter().map(|r| r.request_id.clone()).collect();
                let results = engine.generate_batch(batch).await;

                // Dispatch results to waiting callers
                for (request_id, result) in request_ids.into_iter().zip(results) {
                    // Complete the request in scheduler
                    let generated_tokens = result
                        .as_ref()
                        .map(|r| {
                            r.choices.first().map(|c| c.text.len() as u32 / 4).unwrap_or(0)
                        })
                        .unwrap_or(0);
                    scheduler.complete_request(&request_id, generated_tokens);

                    // Send response to waiting caller
                    if let Some(pending_req) = pending.lock().remove(&request_id) {
                        let _ = pending_req.response_tx.send(result);
                    }
                }

                // Update batch statistics
                let batch_time = batch_start.elapsed().as_millis() as u64;
                {
                    let mut s = stats.lock();
                    s.batches_processed += 1;
                    s.total_inference_time_ms += batch_time;
                    s.avg_batch_time_ms = s.total_inference_time_ms as f64 / s.batches_processed as f64;
                }

                last_batch_time = std::time::Instant::now();
            }
        });

        tracing::info!("BatchedInferenceService started");
    }

    /// Gracefully shuts down the service.
    pub async fn shutdown(&self) {
        *self.shutdown.write().await = true;
        tracing::info!("BatchedInferenceService shutdown requested");
    }

    /// Returns true if the service is running.
    pub async fn is_running(&self) -> bool {
        !*self.shutdown.read().await
    }
}

/// Builder for BatchedInferenceService.
pub struct BatchedInferenceServiceBuilder {
    config: BatchedServiceConfig,
    thermal: Option<ThermalManager>,
}

impl BatchedInferenceServiceBuilder {
    /// Creates a new builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: BatchedServiceConfig::default(),
            thermal: None,
        }
    }

    /// Sets the scheduler configuration.
    #[must_use]
    pub fn scheduler_config(mut self, config: SchedulerConfig) -> Self {
        self.config.scheduler = config;
        self
    }

    /// Sets the maximum batch size.
    #[must_use]
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.config.scheduler.max_batch_size = size;
        self
    }

    /// Sets the maximum wait time before processing a batch.
    #[must_use]
    pub fn max_wait_time(mut self, duration: Duration) -> Self {
        self.config.scheduler.max_wait_time = duration;
        self
    }

    /// Enables continuous batching.
    #[must_use]
    pub fn continuous_batching(mut self, enabled: bool) -> Self {
        self.config.scheduler.continuous_batching = enabled;
        self
    }

    /// Enables thermal-aware batch sizing.
    #[must_use]
    pub fn thermal_aware(mut self, enabled: bool) -> Self {
        self.config.thermal_aware = enabled;
        self
    }

    /// Sets the thermal manager.
    #[must_use]
    pub fn with_thermal(mut self, thermal: ThermalManager) -> Self {
        self.thermal = Some(thermal);
        self.config.thermal_aware = true;
        self
    }

    /// Builds the service with the given engine.
    #[must_use]
    pub fn build(self, engine: Engine) -> BatchedInferenceService {
        if let Some(thermal) = self.thermal {
            BatchedInferenceService::with_thermal(engine, self.config, thermal)
        } else {
            BatchedInferenceService::new(engine, self.config)
        }
    }
}

impl Default for BatchedInferenceServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = BatchedServiceConfig::default();
        assert_eq!(config.scheduler.max_batch_size, 32);
        assert!(config.thermal_aware);
    }

    #[test]
    fn test_config_low_latency() {
        let config = BatchedServiceConfig::low_latency();
        assert_eq!(config.scheduler.max_batch_size, 8);
        assert_eq!(config.scheduler.max_wait_time, Duration::from_millis(10));
    }

    #[test]
    fn test_config_high_throughput() {
        let config = BatchedServiceConfig::high_throughput();
        assert_eq!(config.scheduler.max_batch_size, 64);
        assert_eq!(config.scheduler.max_wait_time, Duration::from_millis(100));
    }

    #[test]
    fn test_builder_defaults() {
        let builder = BatchedInferenceServiceBuilder::new();
        assert_eq!(builder.config.scheduler.max_batch_size, 32);
    }

    #[test]
    fn test_builder_customization() {
        let builder = BatchedInferenceServiceBuilder::new()
            .max_batch_size(16)
            .max_wait_time(Duration::from_millis(25))
            .continuous_batching(true)
            .thermal_aware(false);

        assert_eq!(builder.config.scheduler.max_batch_size, 16);
        assert_eq!(builder.config.scheduler.max_wait_time, Duration::from_millis(25));
        assert!(builder.config.scheduler.continuous_batching);
        assert!(!builder.config.thermal_aware);
    }

    #[test]
    fn test_stats_default() {
        let stats = BatchedServiceStats::default();
        assert_eq!(stats.batches_processed, 0);
        assert_eq!(stats.total_inference_time_ms, 0);
    }
}
