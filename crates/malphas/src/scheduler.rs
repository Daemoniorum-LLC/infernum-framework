//! Request batch scheduling.

use std::collections::VecDeque;
use std::sync::Arc;

use infernum_core::{GenerateRequest, RequestId};
use parking_lot::Mutex;

/// A queued request awaiting processing.
pub struct QueuedRequest {
    /// The request.
    pub request: GenerateRequest,
    /// Priority (higher = more urgent).
    pub priority: u32,
    /// Timestamp when queued.
    pub queued_at: std::time::Instant,
}

/// Scheduler for batching requests.
pub struct BatchScheduler {
    queue: Mutex<VecDeque<QueuedRequest>>,
    max_batch_size: u32,
    max_wait_time_ms: u64,
}

impl BatchScheduler {
    /// Creates a new batch scheduler.
    #[must_use]
    pub fn new(max_batch_size: u32, max_wait_time_ms: u64) -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            max_batch_size,
            max_wait_time_ms,
        }
    }

    /// Enqueues a request.
    pub fn enqueue(&self, request: GenerateRequest, priority: u32) {
        let queued = QueuedRequest {
            request,
            priority,
            queued_at: std::time::Instant::now(),
        };
        self.queue.lock().push_back(queued);
    }

    /// Dequeues a batch of requests ready for processing.
    #[must_use]
    pub fn dequeue_batch(&self) -> Vec<GenerateRequest> {
        let mut queue = self.queue.lock();
        let mut batch = Vec::new();
        let now = std::time::Instant::now();

        while let Some(queued) = queue.front() {
            // Check if we've hit the batch size limit
            if batch.len() >= self.max_batch_size as usize {
                break;
            }

            // Check if the oldest request has waited long enough
            let waited_ms = queued.queued_at.elapsed().as_millis() as u64;
            if batch.is_empty() && waited_ms < self.max_wait_time_ms {
                // Not ready yet
                break;
            }

            if let Some(queued) = queue.pop_front() {
                batch.push(queued.request);
            }
        }

        batch
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
}

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new(32, 50)
    }
}
