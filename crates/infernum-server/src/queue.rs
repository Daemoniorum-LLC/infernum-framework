//! Priority-based request queuing with fairness.
//!
//! This module implements a multi-priority request queue that prevents
//! starvation of lower-priority requests while still respecting priorities.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      RequestQueue                           │
//! │  ┌──────────────────────────────────────────────────────┐  │
//! │  │  Priority Queues (4 levels)                          │  │
//! │  │  ┌──────────┬──────────┬──────────┬──────────┐       │  │
//! │  │  │ Critical │   High   │  Normal  │Background│       │  │
//! │  │  │ weight=4 │ weight=2 │ weight=1 │ weight=1 │       │  │
//! │  │  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘       │  │
//! │  │       └──────────┴──────────┴──────────┘             │  │
//! │  │                      ↓                                │  │
//! │  │            Weighted Fair Scheduler                    │  │
//! │  │                      ↓                                │  │
//! │  │              Starvation Check                         │  │
//! │  └──────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Fairness Model
//!
//! The queue uses weighted fair queuing (WFQ) to balance between:
//! - **Priority**: Higher priority requests are preferred
//! - **Starvation Prevention**: Requests waiting too long get promoted
//! - **Throughput**: Batching-friendly ordering when possible
//!
//! # Example
//!
//! ```rust,ignore
//! use infernum_server::queue::{RequestQueue, QueueConfig, QueuedRequest};
//!
//! let config = QueueConfig::default();
//! let queue = RequestQueue::new(config);
//!
//! // Enqueue requests
//! queue.enqueue(request1)?;
//! queue.enqueue(request2)?;
//!
//! // Dequeue with fairness
//! if let Some(request) = queue.dequeue() {
//!     // Process request
//! }
//! ```

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use tokio::sync::Notify;

use crate::batching::BatchPriority;

/// Number of priority levels.
pub const NUM_PRIORITY_LEVELS: usize = 4;

/// Configuration for the request queue.
#[derive(Debug, Clone)]
pub struct QueueConfig {
    /// Maximum total queue depth across all priorities.
    pub max_queue_depth: usize,

    /// Maximum queue depth per priority level.
    pub max_per_priority: usize,

    /// Weights for each priority level (higher = more bandwidth).
    /// Order: [Background, Normal, High, Critical]
    pub priority_weights: [f32; NUM_PRIORITY_LEVELS],

    /// Time after which a request is considered starving.
    pub starvation_timeout: Duration,

    /// Promote starving requests to this priority.
    pub starvation_promotion_priority: BatchPriority,

    /// Enable weighted fair queuing.
    pub enable_wfq: bool,

    /// Maximum wait time before rejecting new requests.
    pub max_wait_time: Duration,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_queue_depth: 1000,
            max_per_priority: 500,
            // Critical gets 4x bandwidth, High gets 2x, Normal and Background get 1x
            priority_weights: [1.0, 1.0, 2.0, 4.0],
            starvation_timeout: Duration::from_secs(30),
            starvation_promotion_priority: BatchPriority::High,
            enable_wfq: true,
            max_wait_time: Duration::from_secs(60),
        }
    }
}

impl QueueConfig {
    /// Creates a new queue configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method for max queue depth.
    pub fn with_max_queue_depth(mut self, depth: usize) -> Self {
        self.max_queue_depth = depth;
        self
    }

    /// Builder method for max per priority.
    pub fn with_max_per_priority(mut self, max: usize) -> Self {
        self.max_per_priority = max;
        self
    }

    /// Builder method for priority weights.
    pub fn with_priority_weights(mut self, weights: [f32; NUM_PRIORITY_LEVELS]) -> Self {
        self.priority_weights = weights;
        self
    }

    /// Builder method for starvation timeout.
    pub fn with_starvation_timeout(mut self, timeout: Duration) -> Self {
        self.starvation_timeout = timeout;
        self
    }

    /// Builder method for WFQ.
    pub fn with_wfq(mut self, enabled: bool) -> Self {
        self.enable_wfq = enabled;
        self
    }

    /// Builder method for max wait time.
    pub fn with_max_wait_time(mut self, timeout: Duration) -> Self {
        self.max_wait_time = timeout;
        self
    }

    /// Returns the weight for a priority level.
    pub fn weight_for(&self, priority: BatchPriority) -> f32 {
        self.priority_weights[priority.as_level() as usize]
    }
}

/// A request waiting in the queue.
#[derive(Debug)]
pub struct QueuedRequest {
    /// Unique request ID.
    pub id: String,

    /// Request priority.
    pub priority: BatchPriority,

    /// Original priority (before any promotions).
    pub original_priority: BatchPriority,

    /// When the request was enqueued.
    pub enqueued_at: Instant,

    /// Estimated tokens (prompt + max output).
    pub estimated_tokens: usize,

    /// Model ID.
    pub model: String,

    /// Whether this request is streamable.
    pub streamable: bool,

    /// Arbitrary payload (typically serialized request).
    payload: Vec<u8>,
}

impl QueuedRequest {
    /// Creates a new queued request.
    pub fn new(
        id: impl Into<String>,
        priority: BatchPriority,
        model: impl Into<String>,
        estimated_tokens: usize,
    ) -> Self {
        Self {
            id: id.into(),
            priority,
            original_priority: priority,
            enqueued_at: Instant::now(),
            estimated_tokens,
            model: model.into(),
            streamable: false,
            payload: Vec::new(),
        }
    }

    /// Sets the payload.
    pub fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }

    /// Sets the streamable flag.
    pub fn with_streamable(mut self, streamable: bool) -> Self {
        self.streamable = streamable;
        self
    }

    /// Returns the payload.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Takes the payload, leaving an empty vec.
    pub fn take_payload(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.payload)
    }

    /// Returns time spent waiting in queue.
    pub fn wait_time(&self) -> Duration {
        self.enqueued_at.elapsed()
    }

    /// Returns true if this request is starving.
    pub fn is_starving(&self, timeout: Duration) -> bool {
        self.wait_time() > timeout
    }

    /// Promotes the request to a higher priority.
    pub fn promote(&mut self, new_priority: BatchPriority) {
        if new_priority > self.priority {
            self.priority = new_priority;
        }
    }
}

/// Error type for queue operations.
#[derive(Debug, Clone)]
pub enum QueueError {
    /// Queue is full.
    QueueFull {
        /// Current queue depth.
        current: usize,
        /// Maximum queue depth.
        max: usize,
    },

    /// Priority queue is full.
    PriorityQueueFull {
        /// The priority level.
        priority: BatchPriority,
        /// Current count at this priority.
        current: usize,
        /// Maximum at this priority.
        max: usize,
    },

    /// Request would wait too long.
    WouldExceedMaxWait {
        /// Estimated wait time.
        estimated_wait: Duration,
        /// Maximum wait time.
        max_wait: Duration,
    },

    /// Queue is shutting down.
    ShuttingDown,
}

impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull { current, max } => {
                write!(f, "Queue full: {}/{}", current, max)
            }
            Self::PriorityQueueFull {
                priority,
                current,
                max,
            } => {
                write!(
                    f,
                    "Priority queue {} full: {}/{}",
                    priority, current, max
                )
            }
            Self::WouldExceedMaxWait {
                estimated_wait,
                max_wait,
            } => {
                write!(
                    f,
                    "Would exceed max wait: {:?} > {:?}",
                    estimated_wait, max_wait
                )
            }
            Self::ShuttingDown => write!(f, "Queue is shutting down"),
        }
    }
}

impl std::error::Error for QueueError {}

/// State of the request queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueState {
    /// Queue is accepting requests.
    Open,
    /// Queue is draining (not accepting new requests).
    Draining,
    /// Queue is closed.
    Closed,
}

impl fmt::Display for QueueState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open => write!(f, "open"),
            Self::Draining => write!(f, "draining"),
            Self::Closed => write!(f, "closed"),
        }
    }
}

/// Priority-based request queue with fairness.
pub struct RequestQueue {
    /// Configuration.
    config: QueueConfig,

    /// Current state.
    state: RwLock<QueueState>,

    /// Priority queues (one per priority level).
    queues: [Mutex<VecDeque<QueuedRequest>>; NUM_PRIORITY_LEVELS],

    /// Weighted fair queue credits (for WFQ).
    credits: [AtomicU64; NUM_PRIORITY_LEVELS],

    /// Metrics.
    metrics: QueueMetrics,

    /// Notify when items are enqueued.
    notify_enqueue: Notify,

    /// Notify when items are dequeued.
    notify_dequeue: Notify,
}

impl RequestQueue {
    /// Creates a new request queue.
    pub fn new(config: QueueConfig) -> Self {
        Self {
            config,
            state: RwLock::new(QueueState::Open),
            queues: [
                Mutex::new(VecDeque::new()),
                Mutex::new(VecDeque::new()),
                Mutex::new(VecDeque::new()),
                Mutex::new(VecDeque::new()),
            ],
            credits: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            metrics: QueueMetrics::new(),
            notify_enqueue: Notify::new(),
            notify_dequeue: Notify::new(),
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &QueueConfig {
        &self.config
    }

    /// Returns the current state.
    pub fn state(&self) -> QueueState {
        *self.state.read()
    }

    /// Returns the metrics.
    pub fn metrics(&self) -> &QueueMetrics {
        &self.metrics
    }

    /// Returns the total queue depth.
    pub fn len(&self) -> usize {
        self.queues
            .iter()
            .map(|q| q.lock().len())
            .sum()
    }

    /// Returns true if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queues.iter().all(|q| q.lock().is_empty())
    }

    /// Returns the depth at a specific priority level.
    pub fn len_at_priority(&self, priority: BatchPriority) -> usize {
        self.queues[priority.as_level() as usize].lock().len()
    }

    /// Enqueues a request.
    pub fn enqueue(&self, request: QueuedRequest) -> Result<(), QueueError> {
        // Check state
        if *self.state.read() != QueueState::Open {
            return Err(QueueError::ShuttingDown);
        }

        let priority_idx = request.priority.as_level() as usize;

        // Check total queue depth
        let total_depth = self.len();
        if total_depth >= self.config.max_queue_depth {
            self.metrics.record_rejected(request.priority);
            return Err(QueueError::QueueFull {
                current: total_depth,
                max: self.config.max_queue_depth,
            });
        }

        // Check per-priority limit
        let priority_depth = self.queues[priority_idx].lock().len();
        if priority_depth >= self.config.max_per_priority {
            self.metrics.record_rejected(request.priority);
            return Err(QueueError::PriorityQueueFull {
                priority: request.priority,
                current: priority_depth,
                max: self.config.max_per_priority,
            });
        }

        // Enqueue
        self.queues[priority_idx].lock().push_back(request);
        self.metrics.record_enqueued(BatchPriority::from_level(priority_idx as u8));
        self.notify_enqueue.notify_one();

        Ok(())
    }

    /// Dequeues the next request using weighted fair queuing.
    pub fn dequeue(&self) -> Option<QueuedRequest> {
        // Check for starving requests first
        if let Some(request) = self.dequeue_starving() {
            self.metrics.record_starvation_promotion();
            return Some(request);
        }

        if self.config.enable_wfq {
            self.dequeue_wfq()
        } else {
            self.dequeue_priority()
        }
    }

    /// Dequeues using strict priority ordering.
    fn dequeue_priority(&self) -> Option<QueuedRequest> {
        // Try from highest to lowest priority
        for priority in [
            BatchPriority::Critical,
            BatchPriority::High,
            BatchPriority::Normal,
            BatchPriority::Background,
        ] {
            let idx = priority.as_level() as usize;
            let mut queue = self.queues[idx].lock();
            if let Some(request) = queue.pop_front() {
                drop(queue);
                self.metrics.record_dequeued(priority);
                self.metrics.record_wait_time(request.wait_time());
                self.notify_dequeue.notify_one();
                return Some(request);
            }
        }
        None
    }

    /// Dequeues using weighted fair queuing.
    fn dequeue_wfq(&self) -> Option<QueuedRequest> {
        // Add credits based on weights
        for (idx, weight) in self.config.priority_weights.iter().enumerate() {
            let credit_add = (*weight * 100.0) as u64;
            self.credits[idx].fetch_add(credit_add, Ordering::Relaxed);
        }

        // Find the queue with highest credits that has items
        let mut best_idx: Option<usize> = None;
        let mut best_credits: u64 = 0;

        for idx in 0..NUM_PRIORITY_LEVELS {
            if !self.queues[idx].lock().is_empty() {
                let credits = self.credits[idx].load(Ordering::Relaxed);
                if credits > best_credits {
                    best_credits = credits;
                    best_idx = Some(idx);
                }
            }
        }

        if let Some(idx) = best_idx {
            let mut queue = self.queues[idx].lock();
            if let Some(request) = queue.pop_front() {
                drop(queue);

                // Deduct credits (cost = estimated tokens)
                let cost = request.estimated_tokens.max(1) as u64;
                self.credits[idx].fetch_sub(cost.min(best_credits), Ordering::Relaxed);

                let priority = BatchPriority::from_level(idx as u8);
                self.metrics.record_dequeued(priority);
                self.metrics.record_wait_time(request.wait_time());
                self.notify_dequeue.notify_one();
                return Some(request);
            }
        }

        None
    }

    /// Dequeues a starving request if any.
    fn dequeue_starving(&self) -> Option<QueuedRequest> {
        // Check lower priorities for starving requests
        for priority in [BatchPriority::Background, BatchPriority::Normal] {
            let idx = priority.as_level() as usize;
            let mut queue = self.queues[idx].lock();

            // Find first starving request
            if let Some(pos) = queue
                .iter()
                .position(|r| r.is_starving(self.config.starvation_timeout))
            {
                if let Some(mut request) = queue.remove(pos) {
                    drop(queue);
                    request.promote(self.config.starvation_promotion_priority);
                    self.metrics.record_dequeued(priority);
                    self.metrics.record_wait_time(request.wait_time());
                    self.notify_dequeue.notify_one();
                    return Some(request);
                }
            }
        }
        None
    }

    /// Dequeues up to `n` requests.
    pub fn dequeue_batch(&self, n: usize) -> Vec<QueuedRequest> {
        let mut batch = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(request) = self.dequeue() {
                batch.push(request);
            } else {
                break;
            }
        }
        batch
    }

    /// Peeks at the next request without removing it.
    pub fn peek(&self) -> Option<PeekResult> {
        // Check starving first
        for priority in [BatchPriority::Background, BatchPriority::Normal] {
            let idx = priority.as_level() as usize;
            let queue = self.queues[idx].lock();
            if let Some(request) = queue.iter().find(|r| r.is_starving(self.config.starvation_timeout)) {
                return Some(PeekResult {
                    id: request.id.clone(),
                    priority: request.priority,
                    wait_time: request.wait_time(),
                    is_starving: true,
                });
            }
        }

        // Check by priority
        for priority in [
            BatchPriority::Critical,
            BatchPriority::High,
            BatchPriority::Normal,
            BatchPriority::Background,
        ] {
            let idx = priority.as_level() as usize;
            let queue = self.queues[idx].lock();
            if let Some(request) = queue.front() {
                return Some(PeekResult {
                    id: request.id.clone(),
                    priority: request.priority,
                    wait_time: request.wait_time(),
                    is_starving: false,
                });
            }
        }

        None
    }

    /// Removes a specific request by ID.
    pub fn remove(&self, request_id: &str) -> Option<QueuedRequest> {
        for (idx, queue_mutex) in self.queues.iter().enumerate() {
            let mut queue = queue_mutex.lock();
            if let Some(pos) = queue.iter().position(|r| r.id == request_id) {
                if let Some(request) = queue.remove(pos) {
                    drop(queue);
                    let priority = BatchPriority::from_level(idx as u8);
                    self.metrics.record_cancelled(priority);
                    self.notify_dequeue.notify_one();
                    return Some(request);
                }
            }
        }
        None
    }

    /// Starts draining the queue (no new requests accepted).
    pub fn start_drain(&self) {
        *self.state.write() = QueueState::Draining;
    }

    /// Closes the queue and cancels all pending requests.
    pub fn close(&self) -> Vec<QueuedRequest> {
        *self.state.write() = QueueState::Closed;

        let mut cancelled = Vec::new();
        for queue_mutex in &self.queues {
            let mut queue = queue_mutex.lock();
            cancelled.extend(queue.drain(..));
        }

        self.notify_enqueue.notify_waiters();
        self.notify_dequeue.notify_waiters();

        cancelled
    }

    /// Waits for an item to be enqueued.
    pub async fn wait_for_enqueue(&self) {
        self.notify_enqueue.notified().await;
    }

    /// Waits for an item to be dequeued (space available).
    pub async fn wait_for_dequeue(&self) {
        self.notify_dequeue.notified().await;
    }

    /// Returns a snapshot of queue statistics.
    pub fn stats(&self) -> QueueStats {
        let depths: [usize; NUM_PRIORITY_LEVELS] = [
            self.queues[0].lock().len(),
            self.queues[1].lock().len(),
            self.queues[2].lock().len(),
            self.queues[3].lock().len(),
        ];

        let oldest_wait = self.oldest_wait_time();

        QueueStats {
            state: self.state(),
            total_depth: depths.iter().sum(),
            depths_by_priority: depths,
            oldest_wait_time: oldest_wait,
            enqueued: self.metrics.enqueued(),
            dequeued: self.metrics.dequeued(),
            rejected: self.metrics.rejected(),
            cancelled: self.metrics.cancelled(),
            starvation_promotions: self.metrics.starvation_promotions(),
        }
    }

    /// Returns the oldest wait time across all queues.
    fn oldest_wait_time(&self) -> Option<Duration> {
        let mut oldest: Option<Duration> = None;

        for queue_mutex in &self.queues {
            let queue = queue_mutex.lock();
            if let Some(request) = queue.front() {
                let wait = request.wait_time();
                oldest = Some(oldest.map_or(wait, |o| o.max(wait)));
            }
        }

        oldest
    }
}

impl fmt::Debug for RequestQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestQueue")
            .field("state", &self.state())
            .field("total_depth", &self.len())
            .field("config", &self.config)
            .finish()
    }
}

/// Result from peeking at the queue.
#[derive(Debug, Clone)]
pub struct PeekResult {
    /// Request ID.
    pub id: String,
    /// Current priority.
    pub priority: BatchPriority,
    /// Time spent waiting.
    pub wait_time: Duration,
    /// Whether this request is starving.
    pub is_starving: bool,
}

/// Statistics snapshot for the queue.
#[derive(Debug, Clone)]
pub struct QueueStats {
    /// Current state.
    pub state: QueueState,
    /// Total queue depth.
    pub total_depth: usize,
    /// Depth by priority [Background, Normal, High, Critical].
    pub depths_by_priority: [usize; NUM_PRIORITY_LEVELS],
    /// Oldest wait time.
    pub oldest_wait_time: Option<Duration>,
    /// Total enqueued.
    pub enqueued: u64,
    /// Total dequeued.
    pub dequeued: u64,
    /// Total rejected.
    pub rejected: u64,
    /// Total cancelled.
    pub cancelled: u64,
    /// Total starvation promotions.
    pub starvation_promotions: u64,
}

/// Metrics for the request queue.
#[derive(Debug)]
pub struct QueueMetrics {
    /// Enqueued counts by priority.
    enqueued: [AtomicU64; NUM_PRIORITY_LEVELS],

    /// Dequeued counts by priority.
    dequeued: [AtomicU64; NUM_PRIORITY_LEVELS],

    /// Rejected counts by priority.
    rejected: [AtomicU64; NUM_PRIORITY_LEVELS],

    /// Cancelled counts by priority.
    cancelled: [AtomicU64; NUM_PRIORITY_LEVELS],

    /// Starvation promotions.
    starvation_promotions: AtomicU64,

    /// Total wait time in nanoseconds (for averaging).
    total_wait_ns: AtomicU64,

    /// Total requests processed (for averaging).
    total_processed: AtomicU64,
}

impl QueueMetrics {
    /// Creates new metrics.
    pub fn new() -> Self {
        Self {
            enqueued: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            dequeued: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            rejected: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            cancelled: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            starvation_promotions: AtomicU64::new(0),
            total_wait_ns: AtomicU64::new(0),
            total_processed: AtomicU64::new(0),
        }
    }

    /// Records an enqueue.
    pub fn record_enqueued(&self, priority: BatchPriority) {
        self.enqueued[priority.as_level() as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// Records a dequeue.
    pub fn record_dequeued(&self, priority: BatchPriority) {
        self.dequeued[priority.as_level() as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// Records a rejection.
    pub fn record_rejected(&self, priority: BatchPriority) {
        self.rejected[priority.as_level() as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// Records a cancellation.
    pub fn record_cancelled(&self, priority: BatchPriority) {
        self.cancelled[priority.as_level() as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// Records a starvation promotion.
    pub fn record_starvation_promotion(&self) {
        self.starvation_promotions.fetch_add(1, Ordering::Relaxed);
    }

    /// Records wait time.
    pub fn record_wait_time(&self, wait: Duration) {
        self.total_wait_ns
            .fetch_add(wait.as_nanos() as u64, Ordering::Relaxed);
        self.total_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns total enqueued.
    pub fn enqueued(&self) -> u64 {
        self.enqueued.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// Returns enqueued by priority.
    pub fn enqueued_by_priority(&self, priority: BatchPriority) -> u64 {
        self.enqueued[priority.as_level() as usize].load(Ordering::Relaxed)
    }

    /// Returns total dequeued.
    pub fn dequeued(&self) -> u64 {
        self.dequeued.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// Returns dequeued by priority.
    pub fn dequeued_by_priority(&self, priority: BatchPriority) -> u64 {
        self.dequeued[priority.as_level() as usize].load(Ordering::Relaxed)
    }

    /// Returns total rejected.
    pub fn rejected(&self) -> u64 {
        self.rejected.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// Returns rejected by priority.
    pub fn rejected_by_priority(&self, priority: BatchPriority) -> u64 {
        self.rejected[priority.as_level() as usize].load(Ordering::Relaxed)
    }

    /// Returns total cancelled.
    pub fn cancelled(&self) -> u64 {
        self.cancelled.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// Returns starvation promotions.
    pub fn starvation_promotions(&self) -> u64 {
        self.starvation_promotions.load(Ordering::Relaxed)
    }

    /// Returns average wait time.
    pub fn avg_wait_time(&self) -> Duration {
        let total_ns = self.total_wait_ns.load(Ordering::Relaxed);
        let count = self.total_processed.load(Ordering::Relaxed);
        if count > 0 {
            Duration::from_nanos(total_ns / count)
        } else {
            Duration::ZERO
        }
    }

    /// Renders metrics in Prometheus format.
    pub fn prometheus(&self) -> String {
        let mut output = String::new();
        let priorities = ["background", "normal", "high", "critical"];

        // Queue depth by priority (would need to be passed in for actual depth)
        output.push_str("# HELP infernum_queue_enqueued_total Requests enqueued\n");
        output.push_str("# TYPE infernum_queue_enqueued_total counter\n");
        for (idx, name) in priorities.iter().enumerate() {
            output.push_str(&format!(
                "infernum_queue_enqueued_total{{priority=\"{}\"}} {}\n",
                name,
                self.enqueued[idx].load(Ordering::Relaxed)
            ));
        }

        output.push_str("# HELP infernum_queue_dequeued_total Requests dequeued\n");
        output.push_str("# TYPE infernum_queue_dequeued_total counter\n");
        for (idx, name) in priorities.iter().enumerate() {
            output.push_str(&format!(
                "infernum_queue_dequeued_total{{priority=\"{}\"}} {}\n",
                name,
                self.dequeued[idx].load(Ordering::Relaxed)
            ));
        }

        output.push_str("# HELP infernum_queue_rejected_total Requests rejected\n");
        output.push_str("# TYPE infernum_queue_rejected_total counter\n");
        for (idx, name) in priorities.iter().enumerate() {
            output.push_str(&format!(
                "infernum_queue_rejected_total{{priority=\"{}\"}} {}\n",
                name,
                self.rejected[idx].load(Ordering::Relaxed)
            ));
        }

        output.push_str("# HELP infernum_queue_cancelled_total Requests cancelled\n");
        output.push_str("# TYPE infernum_queue_cancelled_total counter\n");
        output.push_str(&format!(
            "infernum_queue_cancelled_total {}\n",
            self.cancelled()
        ));

        output.push_str(
            "# HELP infernum_queue_starvation_promotions_total Starvation promotions\n",
        );
        output.push_str("# TYPE infernum_queue_starvation_promotions_total counter\n");
        output.push_str(&format!(
            "infernum_queue_starvation_promotions_total {}\n",
            self.starvation_promotions()
        ));

        output.push_str("# HELP infernum_queue_avg_wait_seconds Average wait time\n");
        output.push_str("# TYPE infernum_queue_avg_wait_seconds gauge\n");
        output.push_str(&format!(
            "infernum_queue_avg_wait_seconds {:.6}\n",
            self.avg_wait_time().as_secs_f64()
        ));

        output
    }
}

impl Default for QueueMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_config_default() {
        let config = QueueConfig::default();

        assert_eq!(config.max_queue_depth, 1000);
        assert_eq!(config.max_per_priority, 500);
        assert_eq!(config.starvation_timeout, Duration::from_secs(30));
        assert!(config.enable_wfq);
    }

    #[test]
    fn test_queue_config_builder() {
        let config = QueueConfig::new()
            .with_max_queue_depth(500)
            .with_max_per_priority(100)
            .with_starvation_timeout(Duration::from_secs(60))
            .with_wfq(false);

        assert_eq!(config.max_queue_depth, 500);
        assert_eq!(config.max_per_priority, 100);
        assert_eq!(config.starvation_timeout, Duration::from_secs(60));
        assert!(!config.enable_wfq);
    }

    #[test]
    fn test_queue_config_weights() {
        let config = QueueConfig::default();

        assert_eq!(config.weight_for(BatchPriority::Background), 1.0);
        assert_eq!(config.weight_for(BatchPriority::Normal), 1.0);
        assert_eq!(config.weight_for(BatchPriority::High), 2.0);
        assert_eq!(config.weight_for(BatchPriority::Critical), 4.0);
    }

    #[test]
    fn test_queued_request_new() {
        let request = QueuedRequest::new("req-1", BatchPriority::Normal, "llama", 100);

        assert_eq!(request.id, "req-1");
        assert_eq!(request.priority, BatchPriority::Normal);
        assert_eq!(request.estimated_tokens, 100);
        assert!(!request.streamable);
    }

    #[test]
    fn test_queued_request_with_payload() {
        let request = QueuedRequest::new("req-1", BatchPriority::High, "llama", 50)
            .with_payload(vec![1, 2, 3])
            .with_streamable(true);

        assert_eq!(request.payload(), &[1, 2, 3]);
        assert!(request.streamable);
    }

    #[test]
    fn test_queued_request_promote() {
        let mut request = QueuedRequest::new("req-1", BatchPriority::Normal, "llama", 100);

        request.promote(BatchPriority::High);
        assert_eq!(request.priority, BatchPriority::High);
        assert_eq!(request.original_priority, BatchPriority::Normal);

        // Can't demote
        request.promote(BatchPriority::Normal);
        assert_eq!(request.priority, BatchPriority::High);
    }

    #[test]
    fn test_queue_error_display() {
        let err = QueueError::QueueFull {
            current: 100,
            max: 100,
        };
        assert!(err.to_string().contains("Queue full"));

        let err = QueueError::PriorityQueueFull {
            priority: BatchPriority::High,
            current: 50,
            max: 50,
        };
        assert!(err.to_string().contains("Priority queue"));

        let err = QueueError::ShuttingDown;
        assert!(err.to_string().contains("shutting down"));
    }

    #[test]
    fn test_queue_state_display() {
        assert_eq!(QueueState::Open.to_string(), "open");
        assert_eq!(QueueState::Draining.to_string(), "draining");
        assert_eq!(QueueState::Closed.to_string(), "closed");
    }

    #[test]
    fn test_request_queue_new() {
        let config = QueueConfig::default();
        let queue = RequestQueue::new(config);

        assert_eq!(queue.state(), QueueState::Open);
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_request_queue_enqueue_dequeue() {
        let config = QueueConfig::default();
        let queue = RequestQueue::new(config);

        let request = QueuedRequest::new("req-1", BatchPriority::Normal, "llama", 100);
        queue.enqueue(request).unwrap();

        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());

        let dequeued = queue.dequeue();
        assert!(dequeued.is_some());
        assert_eq!(dequeued.unwrap().id, "req-1");
        assert!(queue.is_empty());
    }

    #[test]
    fn test_request_queue_priority_ordering() {
        let config = QueueConfig::new().with_wfq(false); // Disable WFQ for strict priority
        let queue = RequestQueue::new(config);

        // Enqueue in reverse priority order
        queue
            .enqueue(QueuedRequest::new("low", BatchPriority::Background, "m", 10))
            .unwrap();
        queue
            .enqueue(QueuedRequest::new("normal", BatchPriority::Normal, "m", 10))
            .unwrap();
        queue
            .enqueue(QueuedRequest::new("high", BatchPriority::High, "m", 10))
            .unwrap();
        queue
            .enqueue(QueuedRequest::new("critical", BatchPriority::Critical, "m", 10))
            .unwrap();

        // Should dequeue in priority order
        assert_eq!(queue.dequeue().unwrap().id, "critical");
        assert_eq!(queue.dequeue().unwrap().id, "high");
        assert_eq!(queue.dequeue().unwrap().id, "normal");
        assert_eq!(queue.dequeue().unwrap().id, "low");
    }

    #[test]
    fn test_request_queue_max_depth() {
        let config = QueueConfig::new().with_max_queue_depth(2);
        let queue = RequestQueue::new(config);

        queue
            .enqueue(QueuedRequest::new("req-1", BatchPriority::Normal, "m", 10))
            .unwrap();
        queue
            .enqueue(QueuedRequest::new("req-2", BatchPriority::Normal, "m", 10))
            .unwrap();

        let result = queue.enqueue(QueuedRequest::new("req-3", BatchPriority::Normal, "m", 10));
        assert!(matches!(result, Err(QueueError::QueueFull { .. })));
    }

    #[test]
    fn test_request_queue_max_per_priority() {
        let config = QueueConfig::new()
            .with_max_queue_depth(100)
            .with_max_per_priority(1);
        let queue = RequestQueue::new(config);

        queue
            .enqueue(QueuedRequest::new("req-1", BatchPriority::High, "m", 10))
            .unwrap();

        let result = queue.enqueue(QueuedRequest::new("req-2", BatchPriority::High, "m", 10));
        assert!(matches!(result, Err(QueueError::PriorityQueueFull { .. })));

        // Different priority should work
        queue
            .enqueue(QueuedRequest::new("req-3", BatchPriority::Normal, "m", 10))
            .unwrap();
    }

    #[test]
    fn test_request_queue_len_at_priority() {
        let config = QueueConfig::default();
        let queue = RequestQueue::new(config);

        queue
            .enqueue(QueuedRequest::new("req-1", BatchPriority::High, "m", 10))
            .unwrap();
        queue
            .enqueue(QueuedRequest::new("req-2", BatchPriority::High, "m", 10))
            .unwrap();
        queue
            .enqueue(QueuedRequest::new("req-3", BatchPriority::Normal, "m", 10))
            .unwrap();

        assert_eq!(queue.len_at_priority(BatchPriority::High), 2);
        assert_eq!(queue.len_at_priority(BatchPriority::Normal), 1);
        assert_eq!(queue.len_at_priority(BatchPriority::Critical), 0);
    }

    #[test]
    fn test_request_queue_dequeue_batch() {
        let config = QueueConfig::default();
        let queue = RequestQueue::new(config);

        for i in 0..5 {
            queue
                .enqueue(QueuedRequest::new(
                    format!("req-{}", i),
                    BatchPriority::Normal,
                    "m",
                    10,
                ))
                .unwrap();
        }

        let batch = queue.dequeue_batch(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_request_queue_peek() {
        let config = QueueConfig::default();
        let queue = RequestQueue::new(config);

        assert!(queue.peek().is_none());

        queue
            .enqueue(QueuedRequest::new("req-1", BatchPriority::Normal, "m", 10))
            .unwrap();

        let peek = queue.peek();
        assert!(peek.is_some());
        assert_eq!(peek.unwrap().id, "req-1");

        // Peek doesn't remove
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_request_queue_remove() {
        let config = QueueConfig::default();
        let queue = RequestQueue::new(config);

        queue
            .enqueue(QueuedRequest::new("req-1", BatchPriority::Normal, "m", 10))
            .unwrap();
        queue
            .enqueue(QueuedRequest::new("req-2", BatchPriority::Normal, "m", 10))
            .unwrap();

        let removed = queue.remove("req-1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, "req-1");
        assert_eq!(queue.len(), 1);

        // Remove non-existent
        let removed = queue.remove("req-999");
        assert!(removed.is_none());
    }

    #[test]
    fn test_request_queue_drain_close() {
        let config = QueueConfig::default();
        let queue = RequestQueue::new(config);

        queue
            .enqueue(QueuedRequest::new("req-1", BatchPriority::Normal, "m", 10))
            .unwrap();

        queue.start_drain();
        assert_eq!(queue.state(), QueueState::Draining);

        // Can't enqueue while draining
        let result = queue.enqueue(QueuedRequest::new("req-2", BatchPriority::Normal, "m", 10));
        assert!(matches!(result, Err(QueueError::ShuttingDown)));

        // Can still dequeue
        assert!(queue.dequeue().is_some());

        // Close returns remaining
        queue
            .enqueue(QueuedRequest::new("req-3", BatchPriority::Normal, "m", 10))
            .unwrap_err(); // Still draining

        let cancelled = queue.close();
        assert_eq!(queue.state(), QueueState::Closed);
        assert!(cancelled.is_empty()); // Already dequeued
    }

    #[test]
    fn test_request_queue_stats() {
        let config = QueueConfig::default();
        let queue = RequestQueue::new(config);

        queue
            .enqueue(QueuedRequest::new("req-1", BatchPriority::High, "m", 10))
            .unwrap();
        queue
            .enqueue(QueuedRequest::new("req-2", BatchPriority::Normal, "m", 10))
            .unwrap();

        let stats = queue.stats();

        assert_eq!(stats.state, QueueState::Open);
        assert_eq!(stats.total_depth, 2);
        assert_eq!(stats.depths_by_priority[2], 1); // High
        assert_eq!(stats.depths_by_priority[1], 1); // Normal
    }

    #[test]
    fn test_queue_metrics_new() {
        let metrics = QueueMetrics::new();

        assert_eq!(metrics.enqueued(), 0);
        assert_eq!(metrics.dequeued(), 0);
        assert_eq!(metrics.rejected(), 0);
    }

    #[test]
    fn test_queue_metrics_record() {
        let metrics = QueueMetrics::new();

        metrics.record_enqueued(BatchPriority::High);
        metrics.record_enqueued(BatchPriority::High);
        metrics.record_enqueued(BatchPriority::Normal);
        metrics.record_dequeued(BatchPriority::High);
        metrics.record_rejected(BatchPriority::Critical);

        assert_eq!(metrics.enqueued(), 3);
        assert_eq!(metrics.enqueued_by_priority(BatchPriority::High), 2);
        assert_eq!(metrics.dequeued(), 1);
        assert_eq!(metrics.rejected(), 1);
    }

    #[test]
    fn test_queue_metrics_wait_time() {
        let metrics = QueueMetrics::new();

        metrics.record_wait_time(Duration::from_millis(100));
        metrics.record_wait_time(Duration::from_millis(200));

        let avg = metrics.avg_wait_time();
        assert!(avg >= Duration::from_millis(140) && avg <= Duration::from_millis(160));
    }

    #[test]
    fn test_queue_metrics_prometheus() {
        let metrics = QueueMetrics::new();
        metrics.record_enqueued(BatchPriority::Normal);
        metrics.record_starvation_promotion();

        let output = metrics.prometheus();

        assert!(output.contains("infernum_queue_enqueued_total"));
        assert!(output.contains("infernum_queue_starvation_promotions_total 1"));
    }

    #[test]
    fn test_wfq_fairness() {
        let config = QueueConfig::new()
            .with_wfq(true)
            .with_priority_weights([1.0, 1.0, 2.0, 4.0]);
        let queue = RequestQueue::new(config);

        // Add many requests at each priority
        for i in 0..10 {
            queue
                .enqueue(QueuedRequest::new(
                    format!("critical-{}", i),
                    BatchPriority::Critical,
                    "m",
                    10,
                ))
                .unwrap();
            queue
                .enqueue(QueuedRequest::new(
                    format!("normal-{}", i),
                    BatchPriority::Normal,
                    "m",
                    10,
                ))
                .unwrap();
        }

        // Dequeue 10 and count by priority
        let mut critical_count = 0;
        let mut normal_count = 0;

        for _ in 0..10 {
            if let Some(req) = queue.dequeue() {
                if req.id.starts_with("critical") {
                    critical_count += 1;
                } else {
                    normal_count += 1;
                }
            }
        }

        // With 4:1 weight ratio, critical should get more
        assert!(
            critical_count > normal_count,
            "Critical: {}, Normal: {}",
            critical_count,
            normal_count
        );
    }

    #[test]
    fn test_starvation_prevention() {
        // Use very short starvation timeout for testing
        let config = QueueConfig::new()
            .with_wfq(false)
            .with_starvation_timeout(Duration::from_millis(1));
        let queue = RequestQueue::new(config);

        // Add a normal priority request
        queue
            .enqueue(QueuedRequest::new("normal", BatchPriority::Normal, "m", 10))
            .unwrap();

        // Wait for it to become starving
        std::thread::sleep(Duration::from_millis(5));

        // Add a critical request
        queue
            .enqueue(QueuedRequest::new("critical", BatchPriority::Critical, "m", 10))
            .unwrap();

        // The starving normal request should be dequeued first
        let first = queue.dequeue().unwrap();
        assert_eq!(first.id, "normal");
        assert_eq!(first.priority, BatchPriority::High); // Promoted
    }
}
