//! Asynchronous Fragment Streaming for Progressive Quality Inference
//!
//! Handles background streaming of fragments from RAM to VRAM while
//! GPU compute is active, hiding memory transfer latency.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use haagenti::holotensor::QualityCurve;

use super::memory::{FragmentId, HoloMemoryManager, MemoryTier};
use super::{HoloInferenceError, Result};

/// Priority level for streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    /// Low priority - background streaming.
    Low = 0,
    /// Normal priority - standard streaming.
    Normal = 1,
    /// High priority - needed soon.
    High = 2,
    /// Critical priority - blocking on this.
    Critical = 3,
}

/// A streaming request for a fragment.
#[derive(Debug, Clone)]
pub struct StreamRequest {
    /// Fragment to stream.
    pub fragment_id: FragmentId,
    /// Priority level.
    pub priority: StreamPriority,
    /// Target tier (usually VRAM).
    pub target_tier: MemoryTier,
    /// Request timestamp.
    pub requested_at: Instant,
}

impl StreamRequest {
    /// Create new stream request.
    pub fn new(fragment_id: FragmentId, priority: StreamPriority) -> Self {
        Self {
            fragment_id,
            priority,
            target_tier: MemoryTier::Vram,
            requested_at: Instant::now(),
        }
    }

    /// Age of request in milliseconds.
    pub fn age_ms(&self) -> u64 {
        self.requested_at.elapsed().as_millis() as u64
    }
}

/// Statistics for streaming operations.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// Total requests submitted.
    pub requests_submitted: usize,
    /// Requests completed.
    pub requests_completed: usize,
    /// Requests cancelled.
    pub requests_cancelled: usize,
    /// Total bytes transferred.
    pub bytes_transferred: usize,
    /// Total transfer time (microseconds).
    pub transfer_time_us: u64,
    /// Average transfer speed (bytes/second).
    pub avg_speed_bps: f64,
    /// Peak transfer speed (bytes/second).
    pub peak_speed_bps: f64,
    /// Current queue depth.
    pub queue_depth: usize,
    /// Times streaming was paused.
    pub pause_count: usize,
}

impl StreamStats {
    /// Calculate effective bandwidth utilization.
    pub fn bandwidth_utilization(&self, max_bandwidth_bps: f64) -> f32 {
        if max_bandwidth_bps <= 0.0 {
            return 0.0;
        }
        (self.avg_speed_bps / max_bandwidth_bps) as f32
    }
}

/// Stream manager for async RAM→VRAM transfers.
///
/// Coordinates background streaming of holographic fragments while
/// GPU is busy with compute, effectively hiding memory latency.
///
/// ## Pipelining Strategy
///
/// ```text
/// Time →
/// ┌─────────┬─────────┬─────────┬─────────┐
/// │ Layer N │ Layer N+1 Layer N+2 Layer N+3│  GPU Compute
/// │ COMPUTE │ COMPUTE │ COMPUTE │ COMPUTE │
/// └─────────┴─────────┴─────────┴─────────┘
///      ┌─────────┬─────────┬─────────┐
///      │ Layer   │ Layer   │ Layer   │       RAM→VRAM Stream
///      │ N+1     │ N+2     │ N+3     │       (hidden behind compute)
///      │ STREAM  │ STREAM  │ STREAM  │
///      └─────────┴─────────┴─────────┘
/// ```
pub struct StreamManager {
    /// Memory manager reference.
    memory: Arc<HoloMemoryManager>,

    /// Request queue (priority sorted).
    queue: Mutex<VecDeque<StreamRequest>>,

    /// In-flight transfers.
    in_flight: RwLock<Vec<FragmentId>>,

    /// Whether streaming is paused (during heavy compute).
    paused: AtomicBool,

    /// Whether manager is running.
    running: AtomicBool,

    /// Statistics.
    stats: RwLock<StreamStats>,

    /// Bytes transferred counter.
    bytes_transferred: AtomicUsize,

    /// Transfer time counter (microseconds).
    transfer_time_us: AtomicU64,

    /// Maximum concurrent transfers.
    max_concurrent: usize,

    /// Current layer being processed (for priority hints).
    current_layer: AtomicUsize,
}

impl StreamManager {
    /// Create new stream manager.
    pub fn new(memory: Arc<HoloMemoryManager>, max_concurrent: usize) -> Self {
        Self {
            memory,
            queue: Mutex::new(VecDeque::new()),
            in_flight: RwLock::new(Vec::new()),
            paused: AtomicBool::new(false),
            running: AtomicBool::new(true),
            stats: RwLock::new(StreamStats::default()),
            bytes_transferred: AtomicUsize::new(0),
            transfer_time_us: AtomicU64::new(0),
            max_concurrent,
            current_layer: AtomicUsize::new(0),
        }
    }

    /// Submit a stream request.
    pub fn submit(&self, request: StreamRequest) -> Result<()> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| HoloInferenceError::Cuda("queue lock poisoned".to_string()))?;

        // Insert by priority (higher priority first).
        let insert_pos = queue
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(queue.len());

        queue.insert(insert_pos, request);

        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.requests_submitted += 1;
            stats.queue_depth = queue.len();
        }

        Ok(())
    }

    /// Submit multiple requests at once.
    pub fn submit_batch(&self, requests: Vec<StreamRequest>) -> Result<()> {
        for request in requests {
            self.submit(request)?;
        }
        Ok(())
    }

    /// Request streaming for a layer's fragments.
    pub fn request_layer(
        &self,
        layer: usize,
        num_fragments: u16,
        priority: StreamPriority,
    ) -> Result<()> {
        let requests: Vec<_> = (0..num_fragments)
            .map(|i| StreamRequest::new(FragmentId::new(layer, 0, i), priority))
            .collect();
        self.submit_batch(requests)
    }

    /// Request streaming for next N layers (prefetch).
    pub fn prefetch_layers(
        &self,
        start_layer: usize,
        count: usize,
        fragments_per_layer: u16,
    ) -> Result<()> {
        for layer in start_layer..(start_layer + count) {
            // Priority decreases with distance from current layer
            let priority = match layer - start_layer {
                0 => StreamPriority::High,
                1 => StreamPriority::Normal,
                _ => StreamPriority::Low,
            };
            self.request_layer(layer, fragments_per_layer, priority)?;
        }
        Ok(())
    }

    /// Quality-aware prefetch using Haagenti's QualityCurve.
    ///
    /// Prioritizes fragments that give the highest quality improvement per bandwidth unit.
    /// Layers below min_quality get Critical priority, while layers already above target_quality
    /// get Low priority.
    ///
    /// # Arguments
    /// * `layers` - Slice of (layer_index, loaded_fragments, total_fragments)
    /// * `min_quality` - Minimum acceptable quality (layers below this get Critical)
    /// * `target_quality` - Target quality (layers above this get Low)
    /// * `fragments_per_layer` - Number of fragments to prefetch per layer
    pub fn prefetch_quality_aware(
        &self,
        layers: &[(usize, u16, u16)],
        min_quality: f32,
        target_quality: f32,
        fragments_per_layer: u16,
    ) -> Result<()> {
        let quality_curve = QualityCurve::default();

        // Sort layers by quality gap (highest gap first = most priority)
        let mut layer_priorities: Vec<(usize, StreamPriority, u16)> = layers
            .iter()
            .map(|&(layer, loaded, total)| {
                // Predict current quality
                let current_quality = quality_curve.predict(loaded, total);

                // Determine priority based on quality gap
                let priority = if current_quality < min_quality {
                    // Layer is below minimum - critical!
                    StreamPriority::Critical
                } else if current_quality < target_quality * 0.8 {
                    // Layer significantly below target
                    StreamPriority::High
                } else if current_quality < target_quality {
                    // Layer approaching target
                    StreamPriority::Normal
                } else {
                    // Layer at or above target
                    StreamPriority::Low
                };

                // Calculate how many more fragments to load (up to fragments_per_layer)
                let needed = quality_curve.fragments_for_quality(target_quality, total);
                let to_load = (needed.saturating_sub(loaded)).min(fragments_per_layer);

                (layer, priority, to_load)
            })
            .filter(|&(_, _, to_load)| to_load > 0)
            .collect();

        // Sort by priority (Critical first)
        layer_priorities.sort_by(|a, b| b.1.cmp(&a.1));

        // Submit requests in priority order
        for (layer, priority, count) in layer_priorities {
            self.request_layer(layer, count, priority)?;
        }

        Ok(())
    }

    /// Set current layer (for priority adjustment).
    pub fn set_current_layer(&self, layer: usize) {
        self.current_layer.store(layer, Ordering::Relaxed);
    }

    /// Pause streaming (during heavy GPU compute).
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Relaxed);
        if let Ok(mut stats) = self.stats.write() {
            stats.pause_count += 1;
        }
    }

    /// Resume streaming.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
    }

    /// Check if paused.
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    /// Check if running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Stop the stream manager.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Get next request to process.
    pub fn next_request(&self) -> Option<StreamRequest> {
        if self.is_paused() || !self.is_running() {
            return None;
        }

        // Check if we have room for more in-flight
        if let Ok(in_flight) = self.in_flight.read() {
            if in_flight.len() >= self.max_concurrent {
                return None;
            }
        }

        let mut queue = self.queue.lock().ok()?;
        queue.pop_front()
    }

    /// Mark request as in-flight.
    pub fn mark_in_flight(&self, id: FragmentId) {
        if let Ok(mut in_flight) = self.in_flight.write() {
            in_flight.push(id);
        }
    }

    /// Mark request as completed.
    pub fn complete(&self, id: FragmentId, bytes: usize, duration: Duration) {
        // Remove from in-flight
        if let Ok(mut in_flight) = self.in_flight.write() {
            in_flight.retain(|i| *i != id);
        }

        // Update counters
        self.bytes_transferred.fetch_add(bytes, Ordering::Relaxed);
        self.transfer_time_us
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);

        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            stats.requests_completed += 1;
            stats.bytes_transferred += bytes;
            stats.transfer_time_us += duration.as_micros() as u64;

            // Calculate speeds
            if stats.transfer_time_us > 0 {
                stats.avg_speed_bps =
                    (stats.bytes_transferred as f64 * 1_000_000.0) / stats.transfer_time_us as f64;
            }

            let instant_speed = (bytes as f64 * 1_000_000.0) / duration.as_micros() as f64;
            if instant_speed > stats.peak_speed_bps {
                stats.peak_speed_bps = instant_speed;
            }

            if let Ok(queue) = self.queue.lock() {
                stats.queue_depth = queue.len();
            }
        }
    }

    /// Cancel a pending request.
    pub fn cancel(&self, id: &FragmentId) {
        if let Ok(mut queue) = self.queue.lock() {
            queue.retain(|r| r.fragment_id != *id);
        }
        if let Ok(mut stats) = self.stats.write() {
            stats.requests_cancelled += 1;
        }
    }

    /// Clear all pending requests.
    pub fn clear(&self) {
        if let Ok(mut queue) = self.queue.lock() {
            let cancelled = queue.len();
            queue.clear();
            if let Ok(mut stats) = self.stats.write() {
                stats.requests_cancelled += cancelled;
                stats.queue_depth = 0;
            }
        }
    }

    /// Get queue depth.
    pub fn queue_depth(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Get number of in-flight transfers.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.read().map(|f| f.len()).unwrap_or(0)
    }

    /// Get statistics.
    pub fn stats(&self) -> StreamStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Check if a fragment needs streaming.
    pub fn needs_streaming(&self, id: &FragmentId) -> bool {
        // Check if already in VRAM
        if self.memory.is_in_vram(id) {
            return false;
        }

        // Check if already in queue or in-flight
        if let Ok(queue) = self.queue.lock() {
            if queue.iter().any(|r| r.fragment_id == *id) {
                return false;
            }
        }

        if let Ok(in_flight) = self.in_flight.read() {
            if in_flight.contains(id) {
                return false;
            }
        }

        true
    }

    /// Estimate time to complete queue (milliseconds).
    pub fn estimated_completion_time_ms(&self) -> u64 {
        let stats = self.stats();
        if stats.avg_speed_bps <= 0.0 {
            return 0;
        }

        let queue_depth = self.queue_depth();
        let avg_fragment_size = if stats.requests_completed > 0 {
            stats.bytes_transferred / stats.requests_completed
        } else {
            1024 * 1024 // Assume 1MB default
        };

        let remaining_bytes = queue_depth * avg_fragment_size;
        ((remaining_bytes as f64 / stats.avg_speed_bps) * 1000.0) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::super::memory::MemoryConfig;
    use super::*;

    fn create_test_manager() -> StreamManager {
        let memory = Arc::new(HoloMemoryManager::new(MemoryConfig::default()));
        StreamManager::new(memory, 4)
    }

    #[test]
    fn test_submit_request() {
        let manager = create_test_manager();
        let request = StreamRequest::new(FragmentId::new(0, 0, 0), StreamPriority::Normal);

        manager.submit(request).unwrap();
        assert_eq!(manager.queue_depth(), 1);
    }

    #[test]
    fn test_priority_ordering() {
        let manager = create_test_manager();

        // Submit low priority first
        manager
            .submit(StreamRequest::new(
                FragmentId::new(0, 0, 0),
                StreamPriority::Low,
            ))
            .unwrap();

        // Submit high priority
        manager
            .submit(StreamRequest::new(
                FragmentId::new(0, 0, 1),
                StreamPriority::High,
            ))
            .unwrap();

        // High priority should come first
        let next = manager.next_request().unwrap();
        assert_eq!(next.fragment_id.fragment_index, 1);
        assert_eq!(next.priority, StreamPriority::High);
    }

    #[test]
    fn test_pause_resume() {
        let manager = create_test_manager();

        manager
            .submit(StreamRequest::new(
                FragmentId::new(0, 0, 0),
                StreamPriority::Normal,
            ))
            .unwrap();

        // Pause should prevent getting next request
        manager.pause();
        assert!(manager.is_paused());
        assert!(manager.next_request().is_none());

        // Resume should allow getting request
        manager.resume();
        assert!(!manager.is_paused());
        assert!(manager.next_request().is_some());
    }

    #[test]
    fn test_completion_tracking() {
        let manager = create_test_manager();

        let id = FragmentId::new(0, 0, 0);
        manager.mark_in_flight(id);
        assert_eq!(manager.in_flight_count(), 1);

        manager.complete(id, 1024 * 1024, Duration::from_millis(10));
        assert_eq!(manager.in_flight_count(), 0);

        let stats = manager.stats();
        assert_eq!(stats.requests_completed, 1);
        assert_eq!(stats.bytes_transferred, 1024 * 1024);
    }

    #[test]
    fn test_prefetch_layers() {
        let manager = create_test_manager();

        manager.prefetch_layers(5, 3, 16).unwrap();

        // Should have 3 layers * 16 fragments = 48 requests
        assert_eq!(manager.queue_depth(), 48);
    }

    #[test]
    fn test_max_concurrent_limit() {
        let manager = create_test_manager();

        // Add 10 requests
        for i in 0..10 {
            manager
                .submit(StreamRequest::new(
                    FragmentId::new(0, 0, i),
                    StreamPriority::Normal,
                ))
                .unwrap();
        }

        // Get 4 (max_concurrent) requests
        for i in 0..4 {
            let req = manager.next_request().unwrap();
            manager.mark_in_flight(req.fragment_id);
        }

        // 5th should return None (at limit)
        assert!(manager.next_request().is_none());
        assert_eq!(manager.in_flight_count(), 4);
    }
}
