//! Request batch scheduling with priority queues and continuous batching.

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use infernum_core::{GenerateRequest, RequestId};
use parking_lot::Mutex;
use tokio::sync::Notify;

/// Priority levels for request scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Background processing, lowest priority.
    Background = 0,
    /// Normal user requests.
    Normal = 1,
    /// Higher priority for paying customers.
    High = 2,
    /// Real-time requirements.
    Realtime = 3,
    /// System-level urgent requests.
    Critical = 4,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// A queued request awaiting processing.
pub struct QueuedRequest {
    /// The request.
    pub request: GenerateRequest,
    /// Priority level.
    pub priority: Priority,
    /// Timestamp when queued.
    pub queued_at: Instant,
    /// Estimated token count for the prompt.
    pub estimated_prompt_tokens: usize,
    /// Requested max tokens.
    pub max_tokens: u32,
}

impl PartialEq for QueuedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request.request_id == other.request.request_id
    }
}

impl Eq for QueuedRequest {}

impl PartialOrd for QueuedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then older requests first
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.queued_at.cmp(&self.queued_at),
            other => other,
        }
    }
}

/// Configuration for the batch scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of requests in a batch.
    pub max_batch_size: usize,
    /// Maximum total tokens in a batch.
    pub max_batch_tokens: usize,
    /// Maximum time to wait before processing a partial batch.
    pub max_wait_time: Duration,
    /// Enable continuous batching (add requests to running batches).
    pub continuous_batching: bool,
    /// Maximum queue size before rejecting requests.
    pub max_queue_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_batch_tokens: 16384,
            max_wait_time: Duration::from_millis(50),
            continuous_batching: true,
            max_queue_size: 1000,
        }
    }
}

/// Statistics about scheduler performance.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total requests processed.
    pub total_requests: u64,
    /// Total batches processed.
    pub total_batches: u64,
    /// Average batch size.
    pub avg_batch_size: f64,
    /// Average wait time in milliseconds.
    pub avg_wait_time_ms: f64,
    /// Number of rejected requests (queue full).
    pub rejected_requests: u64,
    /// Current queue depth.
    pub current_queue_depth: usize,
}

/// Scheduler for batching requests with priority queue support.
pub struct BatchScheduler {
    /// Priority queue for pending requests.
    queue: Mutex<BinaryHeap<QueuedRequest>>,
    /// Configuration.
    config: SchedulerConfig,
    /// Statistics.
    stats: Mutex<SchedulerStats>,
    /// Notify when new requests arrive.
    notify: Notify,
    /// Active request tracking.
    active_requests: Mutex<HashMap<RequestId, ActiveRequest>>,
}

/// Information about an active (in-progress) request.
struct ActiveRequest {
    started_at: Instant,
    generated_tokens: u32,
    max_tokens: u32,
}

impl BatchScheduler {
    /// Creates a new batch scheduler with the given configuration.
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            queue: Mutex::new(BinaryHeap::new()),
            config,
            stats: Mutex::new(SchedulerStats::default()),
            notify: Notify::new(),
            active_requests: Mutex::new(HashMap::new()),
        }
    }

    /// Enqueues a request with the given priority.
    ///
    /// # Returns
    ///
    /// Returns `true` if the request was queued, `false` if the queue is full.
    pub fn enqueue(&self, request: GenerateRequest, priority: Priority) -> bool {
        let mut queue = self.queue.lock();

        if queue.len() >= self.config.max_queue_size {
            self.stats.lock().rejected_requests += 1;
            return false;
        }

        let estimated_tokens = Self::estimate_prompt_tokens(&request);
        let max_tokens = request.sampling.max_tokens;

        queue.push(QueuedRequest {
            request,
            priority,
            queued_at: Instant::now(),
            estimated_prompt_tokens: estimated_tokens,
            max_tokens,
        });

        drop(queue);
        self.notify.notify_one();
        true
    }

    /// Estimates the number of tokens in a prompt.
    fn estimate_prompt_tokens(request: &GenerateRequest) -> usize {
        match &request.prompt {
            infernum_core::request::PromptInput::Text(s) => s.len() / 4, // Rough estimate
            infernum_core::request::PromptInput::Messages(msgs) => {
                msgs.iter().map(|m| m.content.len() / 4).sum()
            }
            infernum_core::request::PromptInput::Tokens(tokens) => tokens.len(),
        }
    }

    /// Dequeues a batch of requests ready for processing.
    pub fn dequeue_batch(&self) -> Vec<GenerateRequest> {
        let mut queue = self.queue.lock();
        let mut batch = Vec::new();
        let mut total_tokens = 0;
        let now = Instant::now();
        let mut to_requeue = Vec::new();

        // Pop requests from the priority queue
        while let Some(queued) = queue.pop() {
            let request_tokens = queued.estimated_prompt_tokens + queued.max_tokens as usize;

            // Check batch size limit
            if batch.len() >= self.config.max_batch_size {
                to_requeue.push(queued);
                break;
            }

            // Check token limit
            if total_tokens + request_tokens > self.config.max_batch_tokens && !batch.is_empty() {
                to_requeue.push(queued);
                break;
            }

            // If batch is empty and oldest hasn't waited long enough, stop
            if batch.is_empty() && queued.queued_at.elapsed() < self.config.max_wait_time {
                to_requeue.push(queued);
                break;
            }

            total_tokens += request_tokens;

            // Track active request
            self.active_requests.lock().insert(
                queued.request.request_id.clone(),
                ActiveRequest {
                    started_at: Instant::now(),
                    generated_tokens: 0,
                    max_tokens: queued.max_tokens,
                },
            );

            batch.push(queued.request);
        }

        // Requeue requests that couldn't fit
        for req in to_requeue {
            queue.push(req);
        }

        // Update stats
        if !batch.is_empty() {
            let mut stats = self.stats.lock();
            stats.total_batches += 1;
            stats.total_requests += batch.len() as u64;
            let total_batches = stats.total_batches as f64;
            stats.avg_batch_size =
                (stats.avg_batch_size * (total_batches - 1.0) + batch.len() as f64) / total_batches;
            stats.current_queue_depth = queue.len();
        }

        batch
    }

    /// Waits for requests to be available.
    pub async fn wait_for_requests(&self) {
        self.notify.notified().await;
    }

    /// Waits for requests with a timeout.
    pub async fn wait_for_requests_timeout(&self, timeout: Duration) -> bool {
        tokio::select! {
            _ = self.notify.notified() => true,
            _ = tokio::time::sleep(timeout) => false,
        }
    }

    /// Marks a request as completed.
    pub fn complete_request(&self, request_id: &RequestId, generated_tokens: u32) {
        if let Some(active) = self.active_requests.lock().remove(request_id) {
            let wait_time = active.started_at.elapsed().as_millis() as f64;
            let mut stats = self.stats.lock();
            let total = stats.total_requests as f64;
            stats.avg_wait_time_ms =
                (stats.avg_wait_time_ms * (total - 1.0) + wait_time) / total;
        }
    }

    /// Returns current scheduler statistics.
    #[must_use]
    pub fn stats(&self) -> SchedulerStats {
        let mut stats = self.stats.lock().clone();
        stats.current_queue_depth = self.queue.lock().len();
        stats
    }

    /// Returns the current queue length.
    #[must_use]
    pub fn queue_len(&self) -> usize {
        self.queue.lock().len()
    }

    /// Returns true if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.lock().is_empty()
    }

    /// Returns the number of active requests.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active_requests.lock().len()
    }

    /// Clears all pending requests.
    pub fn clear(&self) {
        self.queue.lock().clear();
    }

    /// Returns the scheduler configuration.
    #[must_use]
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
}

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new(SchedulerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernum_core::SamplingParams;

    fn make_request(id: &str) -> GenerateRequest {
        GenerateRequest {
            request_id: RequestId::new(id),
            prompt: infernum_core::request::PromptInput::Text("test".to_string()),
            model: None,
            sampling: SamplingParams::default(),
        }
    }

    #[test]
    fn test_priority_ordering() {
        let scheduler = BatchScheduler::default();

        scheduler.enqueue(make_request("low"), Priority::Background);
        scheduler.enqueue(make_request("high"), Priority::High);
        scheduler.enqueue(make_request("normal"), Priority::Normal);

        // Wait for max_wait_time to pass
        std::thread::sleep(Duration::from_millis(60));

        let batch = scheduler.dequeue_batch();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].request_id.as_str(), "high");
        assert_eq!(batch[1].request_id.as_str(), "normal");
        assert_eq!(batch[2].request_id.as_str(), "low");
    }

    #[test]
    fn test_batch_size_limit() {
        let config = SchedulerConfig {
            max_batch_size: 2,
            max_wait_time: Duration::from_millis(0),
            ..Default::default()
        };
        let scheduler = BatchScheduler::new(config);

        for i in 0..5 {
            scheduler.enqueue(make_request(&format!("req{}", i)), Priority::Normal);
        }

        let batch = scheduler.dequeue_batch();
        assert_eq!(batch.len(), 2);
        assert_eq!(scheduler.queue_len(), 3);
    }

    #[test]
    fn test_queue_full_rejection() {
        let config = SchedulerConfig {
            max_queue_size: 2,
            ..Default::default()
        };
        let scheduler = BatchScheduler::new(config);

        assert!(scheduler.enqueue(make_request("1"), Priority::Normal));
        assert!(scheduler.enqueue(make_request("2"), Priority::Normal));
        assert!(!scheduler.enqueue(make_request("3"), Priority::Normal));

        assert_eq!(scheduler.stats().rejected_requests, 1);
    }
}
