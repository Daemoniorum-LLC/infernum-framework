//! Simple request batching service for continuous batching.
//!
//! This module provides request batching that queues incoming requests and
//! processes them in batches using the engine's generate_batch API.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info};

use abaddon::engine::InferenceEngine;
use infernum_core::request::GenerateRequest;
use infernum_core::response::GenerateResponse;
use infernum_core::Error;

/// Configuration for the request batcher.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum number of requests per batch.
    pub max_batch_size: usize,
    /// Maximum time to wait for batch formation.
    pub max_wait_time: Duration,
    /// Minimum time between batches (prevents thrashing).
    pub min_batch_interval: Duration,
    /// Maximum queue size before rejecting requests.
    pub max_queue_size: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_wait_time: Duration::from_millis(50),
            min_batch_interval: Duration::from_millis(10),
            max_queue_size: 100,
        }
    }
}

/// Statistics for the batcher.
#[derive(Debug, Default)]
pub struct BatcherStats {
    /// Total number of requests submitted to the batcher.
    pub requests_submitted: AtomicU64,
    /// Total number of requests that completed successfully.
    pub requests_completed: AtomicU64,
    /// Total number of requests rejected (e.g., queue full).
    pub requests_rejected: AtomicU64,
    /// Total number of batches processed.
    pub batches_processed: AtomicU64,
    /// Cumulative batch processing time in milliseconds.
    pub total_batch_time_ms: AtomicU64,
}

impl BatcherStats {
    /// Returns the average batch processing time in milliseconds.
    pub fn avg_batch_time_ms(&self) -> f64 {
        let batches = self.batches_processed.load(Ordering::Relaxed);
        if batches == 0 {
            return 0.0;
        }
        self.total_batch_time_ms.load(Ordering::Relaxed) as f64 / batches as f64
    }
}

/// A pending request with its response channel.
struct PendingRequest {
    request: GenerateRequest,
    response_tx: oneshot::Sender<Result<GenerateResponse, Error>>,
    submitted_at: Instant,
}

/// Handle for submitting requests to the batcher.
#[derive(Clone)]
pub struct BatcherHandle {
    request_tx: mpsc::Sender<PendingRequest>,
    stats: Arc<BatcherStats>,
    config: BatcherConfig,
}

impl BatcherHandle {
    /// Submits a request and returns a receiver for the response.
    pub async fn submit(
        &self,
        request: GenerateRequest,
    ) -> Result<oneshot::Receiver<Result<GenerateResponse, Error>>, Error> {
        let (response_tx, response_rx) = oneshot::channel();

        let pending = PendingRequest {
            request,
            response_tx,
            submitted_at: Instant::now(),
        };

        self.request_tx
            .send(pending)
            .await
            .map_err(|_| Error::internal("Batcher channel closed"))?;

        self.stats
            .requests_submitted
            .fetch_add(1, Ordering::Relaxed);
        Ok(response_rx)
    }

    /// Returns current statistics.
    pub fn stats(&self) -> &BatcherStats {
        &self.stats
    }

    /// Returns the configuration.
    pub fn config(&self) -> &BatcherConfig {
        &self.config
    }
}

/// The request batcher service.
pub struct RequestBatcher {
    config: BatcherConfig,
    stats: Arc<BatcherStats>,
    shutdown: Arc<AtomicBool>,
}

impl RequestBatcher {
    /// Creates a new request batcher with the given configuration.
    pub fn new(config: BatcherConfig) -> Self {
        Self {
            config,
            stats: Arc::new(BatcherStats::default()),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Starts the batcher and returns a handle for submitting requests.
    ///
    /// The batcher runs a background task that collects requests into batches
    /// and processes them using the engine's generate_batch API.
    pub fn start<E: InferenceEngine + Send + Sync + 'static>(
        self,
        engine: Arc<E>,
    ) -> BatcherHandle {
        let (request_tx, mut request_rx) =
            mpsc::channel::<PendingRequest>(self.config.max_queue_size);

        let config = self.config.clone();
        let stats = self.stats.clone();
        let shutdown = self.shutdown.clone();

        // Spawn the batch processing task
        tokio::spawn(async move {
            let mut pending_batch: Vec<PendingRequest> = Vec::with_capacity(config.max_batch_size);
            let mut last_batch_time = Instant::now();

            info!(
                max_batch_size = config.max_batch_size,
                max_wait_ms = config.max_wait_time.as_millis(),
                "Request batcher started"
            );

            loop {
                // Check for shutdown
                if shutdown.load(Ordering::Relaxed) {
                    info!("Request batcher shutting down");
                    // Drain remaining requests with errors
                    for pending in pending_batch.drain(..) {
                        let _ = pending
                            .response_tx
                            .send(Err(Error::internal("Server shutting down")));
                    }
                    break;
                }

                // Wait for requests with timeout
                let timeout = if pending_batch.is_empty() {
                    // No pending requests, wait longer
                    Duration::from_secs(1)
                } else {
                    // Have pending requests, use max wait time
                    config.max_wait_time
                };

                match tokio::time::timeout(timeout, request_rx.recv()).await {
                    Ok(Some(request)) => {
                        pending_batch.push(request);
                    },
                    Ok(None) => {
                        // Channel closed, shutdown
                        info!("Request channel closed, shutting down batcher");
                        break;
                    },
                    Err(_) => {
                        // Timeout - process batch if we have requests
                    },
                }

                // Check if we should process the batch
                let should_process = !pending_batch.is_empty()
                    && (pending_batch.len() >= config.max_batch_size
                        || pending_batch
                            .first()
                            .map(|r| r.submitted_at.elapsed() >= config.max_wait_time)
                            .unwrap_or(false));

                if !should_process {
                    continue;
                }

                // Enforce minimum batch interval
                let since_last = last_batch_time.elapsed();
                if since_last < config.min_batch_interval {
                    tokio::time::sleep(config.min_batch_interval - since_last).await;
                }

                // Take the batch
                let batch: Vec<_> = pending_batch
                    .drain(..pending_batch.len().min(config.max_batch_size))
                    .collect();

                let batch_size = batch.len();
                let batch_start = Instant::now();

                debug!(batch_size = batch_size, "Processing request batch");

                // Extract requests and channels
                let (requests, channels): (Vec<_>, Vec<_>) = batch
                    .into_iter()
                    .map(|p| (p.request, p.response_tx))
                    .unzip();

                // Process batch
                let results = engine.generate_batch(requests).await;

                // Send results back
                for (result, response_tx) in results.into_iter().zip(channels) {
                    let _ = response_tx.send(result);
                    stats.requests_completed.fetch_add(1, Ordering::Relaxed);
                }

                // Update stats
                let batch_time = batch_start.elapsed().as_millis() as u64;
                stats.batches_processed.fetch_add(1, Ordering::Relaxed);
                stats
                    .total_batch_time_ms
                    .fetch_add(batch_time, Ordering::Relaxed);

                debug!(
                    batch_size = batch_size,
                    batch_time_ms = batch_time,
                    "Batch completed"
                );

                last_batch_time = Instant::now();
            }
        });

        BatcherHandle {
            request_tx,
            stats: self.stats,
            config: self.config,
        }
    }

    /// Signals the batcher to shut down.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}
