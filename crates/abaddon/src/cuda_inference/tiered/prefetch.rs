//! Asynchronous prefetching for tiered memory system.
//!
//! Provides background prefetching to hide memory transfer latency during inference.
//! Uses a dedicated thread pool to load upcoming layers before they're needed.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::{bounded, Receiver, Sender, TryRecvError};

use super::stats::TieredStats;

/// Request to prefetch a layer.
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    /// Layer index to prefetch.
    pub layer_idx: usize,

    /// Priority (higher = more urgent).
    pub priority: f32,

    /// Requested at this time.
    pub requested_at: Instant,
}

/// Result of a prefetch operation.
#[derive(Debug)]
pub enum PrefetchResult {
    /// Layer is now ready in the target tier.
    Ready { layer_idx: usize, duration_ms: f64 },
    /// Prefetch was cancelled (e.g., layer already accessed).
    Cancelled { layer_idx: usize },
    /// Prefetch failed.
    Failed { layer_idx: usize, error: String },
}

/// Prefetch controller for managing background layer loading.
///
/// Coordinates prefetching across a thread pool to overlap computation
/// with memory transfers.
pub struct PrefetchController {
    /// Request sender.
    request_tx: Sender<PrefetchRequest>,

    /// Result receiver.
    result_rx: Receiver<PrefetchResult>,

    /// Shutdown signal.
    shutdown: Arc<AtomicBool>,

    /// Worker handles (Option to allow taking in wait()).
    workers: Mutex<Vec<thread::JoinHandle<()>>>,

    /// Layers currently being prefetched.
    in_flight: Arc<Mutex<HashSet<usize>>>,

    /// Statistics.
    stats: Arc<TieredStats>,

    /// Prefetch depth (how many layers ahead to prefetch).
    prefetch_depth: usize,

    /// Current layer position (for calculating prefetch targets).
    current_layer: AtomicUsize,
}

impl PrefetchController {
    /// Create a new prefetch controller.
    ///
    /// # Arguments
    /// * `num_workers` - Number of worker threads
    /// * `prefetch_depth` - How many layers ahead to prefetch
    /// * `stats` - Statistics tracker
    pub fn new(num_workers: usize, prefetch_depth: usize, stats: Arc<TieredStats>) -> Self {
        let (request_tx, request_rx) = bounded(64);
        let (result_tx, result_rx) = bounded(64);
        let shutdown = Arc::new(AtomicBool::new(false));
        let in_flight = Arc::new(Mutex::new(HashSet::new()));

        let mut workers = Vec::with_capacity(num_workers);

        for worker_id in 0..num_workers {
            let rx = request_rx.clone();
            let tx = result_tx.clone();
            let shutdown = shutdown.clone();
            let in_flight = in_flight.clone();

            let handle = thread::Builder::new()
                .name(format!("prefetch-{}", worker_id))
                .spawn(move || {
                    Self::worker_loop(worker_id, rx, tx, shutdown, in_flight);
                })
                .expect("Failed to spawn prefetch worker");

            workers.push(handle);
        }

        Self {
            request_tx,
            result_rx,
            shutdown,
            workers: Mutex::new(workers),
            in_flight,
            stats,
            prefetch_depth,
            current_layer: AtomicUsize::new(0),
        }
    }

    /// Worker thread main loop.
    fn worker_loop(
        worker_id: usize,
        rx: Receiver<PrefetchRequest>,
        tx: Sender<PrefetchResult>,
        shutdown: Arc<AtomicBool>,
        in_flight: Arc<Mutex<HashSet<usize>>>,
    ) {
        tracing::debug!(worker_id, "Prefetch worker started");

        while !shutdown.load(Ordering::Relaxed) {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(request) => {
                    let start = Instant::now();
                    let layer_idx = request.layer_idx;

                    // Check if still in flight (might have been accessed already)
                    {
                        let flight = in_flight.lock().unwrap();
                        if !flight.contains(&layer_idx) {
                            let _ = tx.send(PrefetchResult::Cancelled { layer_idx });
                            continue;
                        }
                    }

                    // Perform prefetch
                    // In a real implementation, this would:
                    // 1. Read from NVMe or decompress HCT
                    // 2. Upload to RAM (pinned memory)
                    // 3. Optionally upload to VRAM

                    // Simulate work for now
                    let result = Self::do_prefetch(layer_idx);

                    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

                    // Remove from in-flight
                    {
                        let mut flight = in_flight.lock().unwrap();
                        flight.remove(&layer_idx);
                    }

                    match result {
                        Ok(()) => {
                            let _ = tx.send(PrefetchResult::Ready {
                                layer_idx,
                                duration_ms,
                            });
                        },
                        Err(e) => {
                            let _ = tx.send(PrefetchResult::Failed {
                                layer_idx,
                                error: e,
                            });
                        },
                    }
                },
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => continue,
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => break,
            }
        }

        tracing::debug!(worker_id, "Prefetch worker stopped");
    }

    /// Perform actual prefetch operation.
    fn do_prefetch(layer_idx: usize) -> Result<(), String> {
        // Placeholder - actual implementation would load from disk
        tracing::trace!(layer_idx, "Prefetching layer");
        Ok(())
    }

    /// Request prefetch of a specific layer.
    pub fn request(&self, layer_idx: usize, priority: f32) -> bool {
        // Check if already in flight
        {
            let mut flight = self.in_flight.lock().unwrap();
            if flight.contains(&layer_idx) {
                return false;
            }
            flight.insert(layer_idx);
        }

        let request = PrefetchRequest {
            layer_idx,
            priority,
            requested_at: Instant::now(),
        };

        if self.request_tx.try_send(request).is_ok() {
            self.stats.record_prefetch_request();
            true
        } else {
            // Queue full, remove from in-flight
            let mut flight = self.in_flight.lock().unwrap();
            flight.remove(&layer_idx);
            false
        }
    }

    /// Request prefetch of layers ahead of current position.
    pub fn prefetch_ahead(&self, current: usize, num_layers: usize) {
        self.current_layer.store(current, Ordering::Relaxed);

        for offset in 1..=self.prefetch_depth {
            let target = current + offset;
            if target >= num_layers {
                break;
            }

            // Priority decreases with distance from current layer
            let priority = 1.0 - (offset as f32 / self.prefetch_depth as f32);
            self.request(target, priority);
        }
    }

    /// Check if a layer is currently being prefetched.
    pub fn is_in_flight(&self, layer_idx: usize) -> bool {
        let flight = self.in_flight.lock().unwrap();
        flight.contains(&layer_idx)
    }

    /// Cancel prefetch of a specific layer.
    pub fn cancel(&self, layer_idx: usize) {
        let mut flight = self.in_flight.lock().unwrap();
        flight.remove(&layer_idx);
    }

    /// Drain completed results.
    pub fn poll_results(&self) -> Vec<PrefetchResult> {
        let mut results = Vec::new();
        loop {
            match self.result_rx.try_recv() {
                Ok(result) => {
                    if let PrefetchResult::Ready { .. } = &result {
                        self.stats.record_prefetch_hit();
                    }
                    results.push(result);
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        results
    }

    /// Get number of layers currently in flight.
    pub fn in_flight_count(&self) -> usize {
        let flight = self.in_flight.lock().unwrap();
        flight.len()
    }

    /// Shutdown the prefetcher.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Wait for all workers to finish.
    pub fn wait(&self) {
        self.shutdown();
        let mut workers = self.workers.lock().unwrap();
        for handle in workers.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for PrefetchController {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Try to join workers if possible
        if let Ok(mut workers) = self.workers.lock() {
            for handle in workers.drain(..) {
                let _ = handle.join();
            }
        }
    }
}

/// Prefetch scheduler that determines which layers to prefetch.
///
/// Uses access patterns and model structure to predict future layer needs.
pub struct PrefetchScheduler {
    /// Recent access history.
    access_history: VecDeque<usize>,

    /// History size limit.
    history_limit: usize,

    /// Whether to use sequential prediction.
    use_sequential: bool,

    /// Whether to use frequency-based prediction.
    use_frequency: bool,
}

impl PrefetchScheduler {
    /// Create a new prefetch scheduler.
    pub fn new(history_limit: usize) -> Self {
        Self {
            access_history: VecDeque::with_capacity(history_limit),
            history_limit,
            use_sequential: true,
            use_frequency: false,
        }
    }

    /// Record a layer access.
    pub fn record_access(&mut self, layer_idx: usize) {
        if self.access_history.len() >= self.history_limit {
            self.access_history.pop_front();
        }
        self.access_history.push_back(layer_idx);
    }

    /// Predict next layers to prefetch.
    ///
    /// Returns (layer_idx, priority) pairs.
    pub fn predict(&self, current: usize, num_layers: usize, depth: usize) -> Vec<(usize, f32)> {
        let mut predictions = Vec::new();

        if self.use_sequential {
            // Simple sequential prediction
            for offset in 1..=depth {
                let target = current + offset;
                if target < num_layers {
                    let priority = 1.0 - (offset as f32 / depth as f32);
                    predictions.push((target, priority));
                }
            }
        }

        // Could add frequency-based prediction here

        predictions
    }

    /// Check if access pattern is sequential.
    pub fn is_sequential(&self) -> bool {
        if self.access_history.len() < 3 {
            return true; // Assume sequential until proven otherwise
        }

        let last_few: Vec<_> = self.access_history.iter().rev().take(5).collect();
        for window in last_few.windows(2) {
            if let [a, b] = window {
                if **a != **b + 1 && **a + 1 != **b {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_controller_basic() {
        let stats = Arc::new(TieredStats::new());
        let controller = PrefetchController::new(1, 2, stats);

        // Request prefetch
        assert!(controller.request(5, 0.8));
        assert!(controller.is_in_flight(5));

        // Duplicate request should fail
        assert!(!controller.request(5, 0.8));

        controller.shutdown();
    }

    #[test]
    fn test_prefetch_scheduler_sequential() {
        let mut scheduler = PrefetchScheduler::new(10);

        scheduler.record_access(0);
        scheduler.record_access(1);
        scheduler.record_access(2);

        assert!(scheduler.is_sequential());

        let predictions = scheduler.predict(2, 10, 3);
        assert_eq!(predictions.len(), 3);
        assert_eq!(predictions[0].0, 3);
        assert_eq!(predictions[1].0, 4);
        assert_eq!(predictions[2].0, 5);
    }
}
