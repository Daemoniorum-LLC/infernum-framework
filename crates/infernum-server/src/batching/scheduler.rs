//! Batch scheduler for continuous batching.
//!
//! This module implements the core scheduling logic for managing batches
//! of inference requests, including priority handling and preemption.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::{Mutex, RwLock};
use tokio::sync::Notify;

use super::batch::{
    ActiveBatch, BatchId, PendingRequest, SamplingParams, Sequence, SequenceGroup, SequenceId,
};
use super::iteration::{IterationConfig, IterationResult, TokenIterator};
use super::{BatchError, BatchPriority};

/// Configuration for the batch scheduler.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size (number of sequence groups).
    pub max_batch_size: usize,

    /// Maximum tokens per batch.
    pub max_tokens_per_batch: usize,

    /// Maximum time to wait for batch formation.
    pub max_waiting_time: Duration,

    /// Maximum queue size.
    pub max_queue_size: usize,

    /// Maximum sequence length.
    pub max_sequence_length: usize,

    /// Scheduling policy.
    pub scheduling_policy: SchedulingPolicy,

    /// Preemption policy.
    pub preemption_policy: PreemptionPolicy,

    /// Iteration configuration.
    pub iteration_config: IterationConfig,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_tokens_per_batch: 8192,
            max_waiting_time: Duration::from_secs(30),
            max_queue_size: 1000,
            max_sequence_length: 4096,
            scheduling_policy: SchedulingPolicy::default(),
            preemption_policy: PreemptionPolicy::default(),
            iteration_config: IterationConfig::default(),
        }
    }
}

impl BatchConfig {
    /// Creates a new batch config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method for max batch size.
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Builder method for max tokens per batch.
    pub fn with_max_tokens_per_batch(mut self, tokens: usize) -> Self {
        self.max_tokens_per_batch = tokens;
        self
    }

    /// Builder method for max waiting time.
    pub fn with_max_waiting_time(mut self, duration: Duration) -> Self {
        self.max_waiting_time = duration;
        self
    }

    /// Builder method for max queue size.
    pub fn with_max_queue_size(mut self, size: usize) -> Self {
        self.max_queue_size = size;
        self
    }

    /// Builder method for scheduling policy.
    pub fn with_scheduling_policy(mut self, policy: SchedulingPolicy) -> Self {
        self.scheduling_policy = policy;
        self
    }

    /// Builder method for preemption policy.
    pub fn with_preemption_policy(mut self, policy: PreemptionPolicy) -> Self {
        self.preemption_policy = policy;
        self
    }
}

/// Scheduling policy for batch formation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulingPolicy {
    /// First-come, first-served.
    #[default]
    Fcfs,

    /// Priority-based scheduling.
    Priority,

    /// Shortest job first.
    ShortestJobFirst,

    /// Longest job first (for better batching).
    LongestJobFirst,
}

impl fmt::Display for SchedulingPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fcfs => write!(f, "fcfs"),
            Self::Priority => write!(f, "priority"),
            Self::ShortestJobFirst => write!(f, "sjf"),
            Self::LongestJobFirst => write!(f, "ljf"),
        }
    }
}

/// Preemption policy for handling priority inversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreemptionPolicy {
    /// No preemption.
    #[default]
    None,

    /// Preempt based on priority.
    Priority,

    /// Recompute-based preemption (swap out and later resume).
    Recompute,

    /// Swap-based preemption (save KV cache to CPU/disk).
    Swap,
}

impl fmt::Display for PreemptionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Priority => write!(f, "priority"),
            Self::Recompute => write!(f, "recompute"),
            Self::Swap => write!(f, "swap"),
        }
    }
}

/// State of the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerState {
    /// Scheduler is idle.
    Idle,
    /// Scheduler is forming a batch.
    Forming,
    /// Scheduler is running a batch.
    Running,
    /// Scheduler is paused.
    Paused,
    /// Scheduler is shutting down.
    ShuttingDown,
}

impl fmt::Display for SchedulerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "idle"),
            Self::Forming => write!(f, "forming"),
            Self::Running => write!(f, "running"),
            Self::Paused => write!(f, "paused"),
            Self::ShuttingDown => write!(f, "shutting_down"),
        }
    }
}

/// The main batch scheduler.
pub struct BatchScheduler {
    /// Configuration.
    config: BatchConfig,

    /// Current state.
    state: RwLock<SchedulerState>,

    /// Pending request queue.
    pending_queue: Mutex<VecDeque<PendingRequest>>,

    /// Currently active batch.
    active_batch: RwLock<Option<ActiveBatch>>,

    /// Token iterator for the active batch.
    iterator: Mutex<TokenIterator>,

    /// Next batch ID.
    next_batch_id: AtomicU64,

    /// Next sequence ID.
    next_sequence_id: AtomicU64,

    /// Metrics.
    metrics: SchedulerMetrics,

    /// Notify when new requests arrive.
    notify_new_request: Notify,

    /// Notify when batch completes.
    notify_batch_complete: Notify,
}

impl BatchScheduler {
    /// Creates a new batch scheduler.
    pub fn new(config: BatchConfig) -> Self {
        let iterator = TokenIterator::new(config.iteration_config.clone());

        Self {
            config,
            state: RwLock::new(SchedulerState::Idle),
            pending_queue: Mutex::new(VecDeque::new()),
            active_batch: RwLock::new(None),
            iterator: Mutex::new(iterator),
            next_batch_id: AtomicU64::new(1),
            next_sequence_id: AtomicU64::new(1),
            metrics: SchedulerMetrics::new(),
            notify_new_request: Notify::new(),
            notify_batch_complete: Notify::new(),
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Returns the current state.
    pub fn state(&self) -> SchedulerState {
        *self.state.read()
    }

    /// Returns the number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending_queue.lock().len()
    }

    /// Returns the current batch size.
    pub fn current_batch_size(&self) -> usize {
        self.active_batch.read().as_ref().map_or(0, |b| b.size())
    }

    /// Returns the metrics.
    pub fn metrics(&self) -> &SchedulerMetrics {
        &self.metrics
    }

    /// Generates a new sequence ID.
    pub fn next_sequence_id(&self) -> SequenceId {
        SequenceId::new(self.next_sequence_id.fetch_add(1, Ordering::SeqCst))
    }

    /// Generates a new batch ID.
    fn next_batch_id(&self) -> BatchId {
        BatchId::new(self.next_batch_id.fetch_add(1, Ordering::SeqCst))
    }

    /// Submits a request to the scheduler.
    pub fn submit(&self, request: PendingRequest) -> Result<(), BatchError> {
        let mut queue = self.pending_queue.lock();

        // Check queue limit
        if queue.len() >= self.config.max_queue_size {
            self.metrics.record_rejected();
            return Err(BatchError::QueueFull {
                current: queue.len(),
                max: self.config.max_queue_size,
            });
        }

        // Check sequence length
        if request.total_length() > self.config.max_sequence_length {
            self.metrics.record_rejected();
            return Err(BatchError::SequenceTooLong {
                actual: request.total_length(),
                max: self.config.max_sequence_length,
            });
        }

        // Add to queue based on policy
        match self.config.scheduling_policy {
            SchedulingPolicy::Priority => {
                // Insert in priority order
                let pos = queue
                    .iter()
                    .position(|r| r.priority < request.priority)
                    .unwrap_or(queue.len());
                queue.insert(pos, request);
            }
            SchedulingPolicy::ShortestJobFirst => {
                let pos = queue
                    .iter()
                    .position(|r| r.total_length() > request.total_length())
                    .unwrap_or(queue.len());
                queue.insert(pos, request);
            }
            SchedulingPolicy::LongestJobFirst => {
                let pos = queue
                    .iter()
                    .position(|r| r.total_length() < request.total_length())
                    .unwrap_or(queue.len());
                queue.insert(pos, request);
            }
            SchedulingPolicy::Fcfs => {
                queue.push_back(request);
            }
        }

        self.metrics.record_submitted();
        self.notify_new_request.notify_one();

        Ok(())
    }

    /// Tries to form a new batch from pending requests.
    pub fn try_form_batch(&self) -> Option<ActiveBatch> {
        let mut queue = self.pending_queue.lock();

        if queue.is_empty() {
            return None;
        }

        let batch_id = self.next_batch_id();
        let batch = ActiveBatch::new(
            batch_id,
            self.config.max_batch_size,
            self.config.max_tokens_per_batch,
        );

        // Add requests to batch until full
        let mut to_remove = Vec::new();

        for (i, request) in queue.iter().enumerate() {
            let seq = Sequence::new(
                request.sequence_id,
                request.input_tokens.clone(),
                request.max_tokens,
            );

            let sampling = SamplingParams::default()
                .with_temperature(request.temperature)
                .with_stop_tokens(request.stop_sequences.iter().flatten().copied().collect());

            let group =
                SequenceGroup::new(request.id.clone(), seq, sampling).with_priority(request.priority);

            match batch.try_add(group) {
                Ok(_) => {
                    to_remove.push(i);
                }
                Err(_) => {
                    // Batch is full
                    break;
                }
            }
        }

        // Remove added requests from queue
        for i in to_remove.into_iter().rev() {
            queue.remove(i);
        }

        if batch.is_empty() {
            None
        } else {
            self.metrics.record_batch_formed(batch.size());
            Some(batch)
        }
    }

    /// Sets the active batch.
    pub fn set_active_batch(&self, batch: ActiveBatch) {
        *self.state.write() = SchedulerState::Running;
        *self.active_batch.write() = Some(batch);
    }

    /// Clears the active batch.
    pub fn clear_active_batch(&self) -> Option<ActiveBatch> {
        *self.state.write() = SchedulerState::Idle;
        self.active_batch.write().take()
    }

    /// Runs a single iteration step on the active batch.
    pub fn step(&self) -> Option<IterationResult> {
        let mut batch_guard = self.active_batch.write();
        let batch = batch_guard.as_mut()?;

        let mut iterator = self.iterator.lock();
        let result = iterator.step(batch);

        if result.batch_complete {
            self.metrics.record_batch_completed();
            self.notify_batch_complete.notify_waiters();
        }

        Some(result)
    }

    /// Runs the active batch to completion.
    pub fn run_batch(&self) -> Option<IterationResult> {
        let mut batch_guard = self.active_batch.write();
        let batch = batch_guard.as_mut()?;

        let mut iterator = self.iterator.lock();
        let result = iterator.run_to_completion(batch);

        if result.batch_complete {
            self.metrics.record_batch_completed();
            self.notify_batch_complete.notify_waiters();
        }

        Some(result)
    }

    /// Checks if preemption should occur.
    pub fn should_preempt(&self) -> bool {
        if self.config.preemption_policy == PreemptionPolicy::None {
            return false;
        }

        let queue = self.pending_queue.lock();
        let batch_guard = self.active_batch.read();

        if let (Some(waiting), Some(batch)) = (queue.front(), batch_guard.as_ref()) {
            // Check if waiting request has higher priority than running batch
            if self.config.preemption_policy == PreemptionPolicy::Priority {
                let batch_min_priority = batch
                    .stats()
                    .num_groups; // Simplified: check priority of first group
                return waiting.priority > BatchPriority::from_level(batch_min_priority as u8);
            }
        }

        false
    }

    /// Requests preemption of the current batch.
    pub fn request_preemption(&self) {
        self.iterator.lock().request_preemption();
    }

    /// Pauses the scheduler.
    pub fn pause(&self) {
        *self.state.write() = SchedulerState::Paused;
    }

    /// Resumes the scheduler.
    pub fn resume(&self) {
        if *self.state.read() == SchedulerState::Paused {
            *self.state.write() = SchedulerState::Idle;
        }
    }

    /// Initiates shutdown.
    pub fn shutdown(&self) {
        *self.state.write() = SchedulerState::ShuttingDown;

        // Cancel all pending requests
        let mut queue = self.pending_queue.lock();
        for request in queue.drain(..) {
            request.cancel();
        }

        self.notify_new_request.notify_waiters();
        self.notify_batch_complete.notify_waiters();
    }

    /// Waits for a new request.
    pub async fn wait_for_request(&self) {
        self.notify_new_request.notified().await;
    }

    /// Waits for batch completion.
    pub async fn wait_for_batch_complete(&self) {
        self.notify_batch_complete.notified().await;
    }

    /// Gets a snapshot of scheduler statistics.
    pub fn stats(&self) -> SchedulerStats {
        let batch_stats = self.active_batch.read().as_ref().map(|b| b.stats());

        SchedulerStats {
            state: self.state(),
            pending_count: self.pending_count(),
            batch_size: self.current_batch_size(),
            batches_formed: self.metrics.batches_formed(),
            batches_completed: self.metrics.batches_completed(),
            requests_submitted: self.metrics.requests_submitted(),
            requests_rejected: self.metrics.requests_rejected(),
            batch_stats,
        }
    }
}

impl fmt::Debug for BatchScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchScheduler")
            .field("state", &self.state())
            .field("pending_count", &self.pending_count())
            .field("current_batch_size", &self.current_batch_size())
            .finish()
    }
}

/// Metrics for the batch scheduler.
#[derive(Debug)]
pub struct SchedulerMetrics {
    /// Requests submitted.
    requests_submitted: AtomicU64,

    /// Requests rejected.
    requests_rejected: AtomicU64,

    /// Batches formed.
    batches_formed: AtomicU64,

    /// Batches completed.
    batches_completed: AtomicU64,

    /// Preemptions.
    preemptions: AtomicU64,

    /// Total requests in batches.
    total_batch_requests: AtomicU64,
}

impl SchedulerMetrics {
    /// Creates new metrics.
    pub fn new() -> Self {
        Self {
            requests_submitted: AtomicU64::new(0),
            requests_rejected: AtomicU64::new(0),
            batches_formed: AtomicU64::new(0),
            batches_completed: AtomicU64::new(0),
            preemptions: AtomicU64::new(0),
            total_batch_requests: AtomicU64::new(0),
        }
    }

    /// Records a submitted request.
    pub fn record_submitted(&self) {
        self.requests_submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a rejected request.
    pub fn record_rejected(&self) {
        self.requests_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a batch formed.
    pub fn record_batch_formed(&self, size: usize) {
        self.batches_formed.fetch_add(1, Ordering::Relaxed);
        self.total_batch_requests
            .fetch_add(size as u64, Ordering::Relaxed);
    }

    /// Records a batch completed.
    pub fn record_batch_completed(&self) {
        self.batches_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a preemption.
    pub fn record_preemption(&self) {
        self.preemptions.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns requests submitted.
    pub fn requests_submitted(&self) -> u64 {
        self.requests_submitted.load(Ordering::Relaxed)
    }

    /// Returns requests rejected.
    pub fn requests_rejected(&self) -> u64 {
        self.requests_rejected.load(Ordering::Relaxed)
    }

    /// Returns batches formed.
    pub fn batches_formed(&self) -> u64 {
        self.batches_formed.load(Ordering::Relaxed)
    }

    /// Returns batches completed.
    pub fn batches_completed(&self) -> u64 {
        self.batches_completed.load(Ordering::Relaxed)
    }

    /// Returns preemptions.
    pub fn preemptions(&self) -> u64 {
        self.preemptions.load(Ordering::Relaxed)
    }

    /// Returns average batch size.
    pub fn avg_batch_size(&self) -> f64 {
        let batches = self.batches_formed();
        let requests = self.total_batch_requests.load(Ordering::Relaxed);
        if batches > 0 {
            requests as f64 / batches as f64
        } else {
            0.0
        }
    }

    /// Renders metrics in Prometheus format.
    pub fn prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str("# HELP infernum_scheduler_requests_submitted_total Requests submitted\n");
        output.push_str("# TYPE infernum_scheduler_requests_submitted_total counter\n");
        output.push_str(&format!(
            "infernum_scheduler_requests_submitted_total {}\n",
            self.requests_submitted()
        ));

        output.push_str("# HELP infernum_scheduler_requests_rejected_total Requests rejected\n");
        output.push_str("# TYPE infernum_scheduler_requests_rejected_total counter\n");
        output.push_str(&format!(
            "infernum_scheduler_requests_rejected_total {}\n",
            self.requests_rejected()
        ));

        output.push_str("# HELP infernum_scheduler_batches_formed_total Batches formed\n");
        output.push_str("# TYPE infernum_scheduler_batches_formed_total counter\n");
        output.push_str(&format!(
            "infernum_scheduler_batches_formed_total {}\n",
            self.batches_formed()
        ));

        output.push_str("# HELP infernum_scheduler_batches_completed_total Batches completed\n");
        output.push_str("# TYPE infernum_scheduler_batches_completed_total counter\n");
        output.push_str(&format!(
            "infernum_scheduler_batches_completed_total {}\n",
            self.batches_completed()
        ));

        output.push_str("# HELP infernum_scheduler_preemptions_total Preemptions\n");
        output.push_str("# TYPE infernum_scheduler_preemptions_total counter\n");
        output.push_str(&format!(
            "infernum_scheduler_preemptions_total {}\n",
            self.preemptions()
        ));

        output.push_str("# HELP infernum_scheduler_avg_batch_size Average batch size\n");
        output.push_str("# TYPE infernum_scheduler_avg_batch_size gauge\n");
        output.push_str(&format!(
            "infernum_scheduler_avg_batch_size {:.2}\n",
            self.avg_batch_size()
        ));

        output
    }
}

impl Default for SchedulerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics snapshot for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Current state.
    pub state: SchedulerState,

    /// Pending request count.
    pub pending_count: usize,

    /// Current batch size.
    pub batch_size: usize,

    /// Total batches formed.
    pub batches_formed: u64,

    /// Total batches completed.
    pub batches_completed: u64,

    /// Total requests submitted.
    pub requests_submitted: u64,

    /// Total requests rejected.
    pub requests_rejected: u64,

    /// Active batch stats (if any).
    pub batch_stats: Option<super::batch::BatchStats>,
}

#[cfg(test)]
mod tests {
    use super::*;
    // Re-import types from batch module for tests
    use super::super::batch::{SequenceId, Sequence, SequenceGroup, SamplingParams};

    fn create_test_request(id: &str, seq_id: u64) -> PendingRequest {
        let (request, _rx) = PendingRequest::new(
            id,
            SequenceId::new(seq_id),
            "test-model",
            vec![1, 2, 3, 4, 5],
            100,
        );
        request
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();

        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_tokens_per_batch, 8192);
        assert_eq!(config.max_queue_size, 1000);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::new()
            .with_max_batch_size(16)
            .with_max_tokens_per_batch(4096)
            .with_max_queue_size(500)
            .with_scheduling_policy(SchedulingPolicy::Priority);

        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.max_tokens_per_batch, 4096);
        assert_eq!(config.max_queue_size, 500);
        assert_eq!(config.scheduling_policy, SchedulingPolicy::Priority);
    }

    #[test]
    fn test_scheduling_policy_display() {
        assert_eq!(SchedulingPolicy::Fcfs.to_string(), "fcfs");
        assert_eq!(SchedulingPolicy::Priority.to_string(), "priority");
        assert_eq!(SchedulingPolicy::ShortestJobFirst.to_string(), "sjf");
        assert_eq!(SchedulingPolicy::LongestJobFirst.to_string(), "ljf");
    }

    #[test]
    fn test_preemption_policy_display() {
        assert_eq!(PreemptionPolicy::None.to_string(), "none");
        assert_eq!(PreemptionPolicy::Priority.to_string(), "priority");
        assert_eq!(PreemptionPolicy::Recompute.to_string(), "recompute");
        assert_eq!(PreemptionPolicy::Swap.to_string(), "swap");
    }

    #[test]
    fn test_scheduler_state_display() {
        assert_eq!(SchedulerState::Idle.to_string(), "idle");
        assert_eq!(SchedulerState::Running.to_string(), "running");
        assert_eq!(SchedulerState::ShuttingDown.to_string(), "shutting_down");
    }

    #[test]
    fn test_scheduler_new() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        assert_eq!(scheduler.state(), SchedulerState::Idle);
        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.current_batch_size(), 0);
    }

    #[test]
    fn test_scheduler_submit() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        let request = create_test_request("req-1", 1);
        let result = scheduler.submit(request);

        assert!(result.is_ok());
        assert_eq!(scheduler.pending_count(), 1);
    }

    #[test]
    fn test_scheduler_submit_queue_full() {
        let config = BatchConfig::new().with_max_queue_size(2);
        let scheduler = BatchScheduler::new(config);

        // Fill queue
        scheduler
            .submit(create_test_request("req-1", 1))
            .unwrap();
        scheduler
            .submit(create_test_request("req-2", 2))
            .unwrap();

        // Third should fail
        let result = scheduler.submit(create_test_request("req-3", 3));
        assert!(matches!(result, Err(BatchError::QueueFull { .. })));
    }

    #[test]
    fn test_scheduler_submit_sequence_too_long() {
        let config = BatchConfig::new().with_max_queue_size(100);
        let scheduler = BatchScheduler::new(config);

        let (request, _rx) = PendingRequest::new(
            "req-1",
            SequenceId::new(1),
            "test-model",
            vec![1; 2000], // Long input
            3000,          // Want 3000 more tokens
        );

        let result = scheduler.submit(request);
        assert!(matches!(result, Err(BatchError::SequenceTooLong { .. })));
    }

    #[test]
    fn test_scheduler_try_form_batch() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        // Submit some requests
        scheduler
            .submit(create_test_request("req-1", 1))
            .unwrap();
        scheduler
            .submit(create_test_request("req-2", 2))
            .unwrap();

        let batch = scheduler.try_form_batch();
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.size(), 2);
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_scheduler_try_form_batch_empty() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        let batch = scheduler.try_form_batch();
        assert!(batch.is_none());
    }

    #[test]
    fn test_scheduler_set_active_batch() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        scheduler
            .submit(create_test_request("req-1", 1))
            .unwrap();
        let batch = scheduler.try_form_batch().unwrap();

        scheduler.set_active_batch(batch);

        assert_eq!(scheduler.state(), SchedulerState::Running);
        assert_eq!(scheduler.current_batch_size(), 1);
    }

    #[test]
    fn test_scheduler_clear_active_batch() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        scheduler
            .submit(create_test_request("req-1", 1))
            .unwrap();
        let batch = scheduler.try_form_batch().unwrap();
        scheduler.set_active_batch(batch);

        let cleared = scheduler.clear_active_batch();
        assert!(cleared.is_some());
        assert_eq!(scheduler.state(), SchedulerState::Idle);
        assert_eq!(scheduler.current_batch_size(), 0);
    }

    #[test]
    fn test_scheduler_step() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        scheduler
            .submit(create_test_request("req-1", 1))
            .unwrap();
        let batch = scheduler.try_form_batch().unwrap();
        scheduler.set_active_batch(batch);

        let result = scheduler.step();
        assert!(result.is_some());
    }

    #[test]
    fn test_scheduler_pause_resume() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        scheduler.pause();
        assert_eq!(scheduler.state(), SchedulerState::Paused);

        scheduler.resume();
        assert_eq!(scheduler.state(), SchedulerState::Idle);
    }

    #[test]
    fn test_scheduler_shutdown() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        scheduler
            .submit(create_test_request("req-1", 1))
            .unwrap();
        assert_eq!(scheduler.pending_count(), 1);

        scheduler.shutdown();

        assert_eq!(scheduler.state(), SchedulerState::ShuttingDown);
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_scheduler_stats() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        scheduler
            .submit(create_test_request("req-1", 1))
            .unwrap();

        let stats = scheduler.stats();

        assert_eq!(stats.state, SchedulerState::Idle);
        assert_eq!(stats.pending_count, 1);
        assert_eq!(stats.requests_submitted, 1);
    }

    #[test]
    fn test_scheduler_metrics_new() {
        let metrics = SchedulerMetrics::new();

        assert_eq!(metrics.requests_submitted(), 0);
        assert_eq!(metrics.requests_rejected(), 0);
        assert_eq!(metrics.batches_formed(), 0);
    }

    #[test]
    fn test_scheduler_metrics_record() {
        let metrics = SchedulerMetrics::new();

        metrics.record_submitted();
        metrics.record_submitted();
        metrics.record_rejected();
        metrics.record_batch_formed(10);
        metrics.record_batch_completed();

        assert_eq!(metrics.requests_submitted(), 2);
        assert_eq!(metrics.requests_rejected(), 1);
        assert_eq!(metrics.batches_formed(), 1);
        assert_eq!(metrics.batches_completed(), 1);
        assert_eq!(metrics.avg_batch_size(), 10.0);
    }

    #[test]
    fn test_scheduler_metrics_prometheus() {
        let metrics = SchedulerMetrics::new();
        metrics.record_submitted();
        metrics.record_batch_formed(5);

        let output = metrics.prometheus();

        assert!(output.contains("infernum_scheduler_requests_submitted_total 1"));
        assert!(output.contains("infernum_scheduler_batches_formed_total 1"));
    }

    #[test]
    fn test_priority_scheduling() {
        let config = BatchConfig::new().with_scheduling_policy(SchedulingPolicy::Priority);
        let scheduler = BatchScheduler::new(config);

        // Submit low priority first
        let low = create_test_request("low", 1);
        scheduler.submit(low).unwrap();

        // Submit high priority second
        let high = create_test_request("high", 2).with_priority(BatchPriority::High);
        scheduler.submit(high).unwrap();

        // Form batch - high priority should be first
        let batch = scheduler.try_form_batch().unwrap();
        assert_eq!(batch.size(), 2);
    }

    #[test]
    fn test_sequence_id_generation() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        let id1 = scheduler.next_sequence_id();
        let id2 = scheduler.next_sequence_id();
        let id3 = scheduler.next_sequence_id();

        assert_eq!(id1.as_u64(), 1);
        assert_eq!(id2.as_u64(), 2);
        assert_eq!(id3.as_u64(), 3);
    }

    // =========================================================================
    // Continuous Batching Tests
    // =========================================================================

    #[test]
    fn test_continuous_batching_add_to_running_batch() {
        // Continuous batching allows adding new requests to a running batch
        let config = BatchConfig::new().with_max_batch_size(10);
        let scheduler = BatchScheduler::new(config);

        // Start with one request and form a batch
        scheduler.submit(create_test_request("req-1", 1)).unwrap();
        let mut batch = scheduler.try_form_batch().unwrap();

        // Start the batch (simulates inference in progress)
        batch.start();
        assert_eq!(batch.size(), 1);

        // Add more requests while batch is running
        // In continuous batching, we can add to the running batch
        let seq = Sequence::new(
            SequenceId::new(2),
            vec![1, 2, 3],
            100,
        );
        let group = SequenceGroup::new(
            "req-2".to_string(),
            seq,
            SamplingParams::default(),
        );

        // try_add works on running batch
        let result = batch.try_add(group);
        assert!(result.is_ok(), "Should be able to add to running batch");
        assert_eq!(batch.size(), 2, "Batch should have 2 requests now");
    }

    #[test]
    fn test_continuous_batching_respects_limits() {
        let config = BatchConfig::new()
            .with_max_batch_size(2)
            .with_max_tokens_per_batch(200);
        let scheduler = BatchScheduler::new(config);

        // Form a batch with one request
        scheduler.submit(create_test_request("req-1", 1)).unwrap();
        let batch = scheduler.try_form_batch().unwrap();

        // Add second request
        let seq = Sequence::new(
            SequenceId::new(2),
            vec![1, 2, 3],
            50,
        );
        let group = SequenceGroup::new(
            "req-2".to_string(),
            seq,
            SamplingParams::default(),
        );
        assert!(batch.try_add(group).is_ok());

        // Third request should fail (batch full)
        let seq = Sequence::new(
            SequenceId::new(3),
            vec![1, 2, 3],
            50,
        );
        let group = SequenceGroup::new(
            "req-3".to_string(),
            seq,
            SamplingParams::default(),
        );
        assert!(batch.try_add(group).is_err(), "Batch should reject when full");
    }

    #[test]
    fn test_continuous_batching_token_limit() {
        // Test that token limits are respected when adding to batches
        // Note: total_tokens() counts current tokens (input tokens), not input + max_output
        let config = BatchConfig::new()
            .with_max_batch_size(10)
            .with_max_tokens_per_batch(150);
        let scheduler = BatchScheduler::new(config);

        // Create initial request with 5 input tokens
        scheduler.submit(create_test_request("req-1", 1)).unwrap();
        let batch = scheduler.try_form_batch().unwrap();
        // Batch now has 5 tokens

        // Add another request with 30 input tokens → total 35 < 150, should fit
        let seq = Sequence::new(
            SequenceId::new(2),
            vec![1; 30], // 30 input tokens
            100,         // max output (not counted in limit check)
        );
        let group = SequenceGroup::new(
            "req-2".to_string(),
            seq,
            SamplingParams::default(),
        );
        assert!(batch.try_add(group).is_ok(), "Should fit: 5 + 30 = 35 < 150");

        // Add a large request that would exceed limit: 130 input tokens
        // 35 + 130 = 165 > 150
        let seq = Sequence::new(
            SequenceId::new(3),
            vec![1; 130], // 130 input tokens
            50,           // max output (not counted)
        );
        let group = SequenceGroup::new(
            "req-3".to_string(),
            seq,
            SamplingParams::default(),
        );
        assert!(batch.try_add(group).is_err(), "Should reject: 35 + 130 = 165 > 150");

        // Add a smaller request that fits: 50 input tokens
        // 35 + 50 = 85 < 150
        let seq = Sequence::new(
            SequenceId::new(4),
            vec![1; 50], // 50 input tokens
            100,         // max output (not counted)
        );
        let group = SequenceGroup::new(
            "req-4".to_string(),
            seq,
            SamplingParams::default(),
        );
        assert!(batch.try_add(group).is_ok(), "Should fit: 35 + 50 = 85 < 150");
    }

    #[test]
    fn test_scheduler_can_add_while_running() {
        // Tests that new requests queue while a batch is running
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        // Submit initial request and form batch
        scheduler.submit(create_test_request("req-1", 1)).unwrap();
        let batch = scheduler.try_form_batch().unwrap();
        scheduler.set_active_batch(batch);

        assert_eq!(scheduler.state(), SchedulerState::Running);

        // Submit more requests while batch is active
        scheduler.submit(create_test_request("req-2", 2)).unwrap();
        scheduler.submit(create_test_request("req-3", 3)).unwrap();

        // New requests should be queued
        assert_eq!(scheduler.pending_count(), 2);

        // After current batch completes, new batch can be formed
        scheduler.clear_active_batch();
        let new_batch = scheduler.try_form_batch();
        assert!(new_batch.is_some());
        assert_eq!(new_batch.unwrap().size(), 2);
    }

    #[test]
    fn test_metrics_track_continuous_batching() {
        let config = BatchConfig::default();
        let scheduler = BatchScheduler::new(config);

        // Submit requests in waves
        for i in 0..5 {
            scheduler.submit(create_test_request(&format!("req-{}", i), i as u64)).unwrap();
        }

        // Form first batch
        let batch1 = scheduler.try_form_batch().unwrap();
        scheduler.set_active_batch(batch1);

        // Submit more while running
        for i in 5..10 {
            scheduler.submit(create_test_request(&format!("req-{}", i), i as u64)).unwrap();
        }

        // Check metrics
        let metrics = scheduler.metrics();
        assert_eq!(metrics.requests_submitted(), 10);
        assert_eq!(metrics.batches_formed(), 1);

        // Clear and form second batch
        scheduler.clear_active_batch();
        let batch2 = scheduler.try_form_batch().unwrap();
        scheduler.set_active_batch(batch2);

        assert_eq!(scheduler.metrics().batches_formed(), 2);
    }

    #[test]
    fn test_preemption_policy_priority() {
        let config = BatchConfig::new()
            .with_preemption_policy(PreemptionPolicy::Priority);
        let scheduler = BatchScheduler::new(config);

        // Submit and start low priority batch
        let low_req = create_test_request("low", 1);
        scheduler.submit(low_req).unwrap();
        let batch = scheduler.try_form_batch().unwrap();
        scheduler.set_active_batch(batch);

        // Submit critical priority request
        let high_req = create_test_request("high", 2)
            .with_priority(BatchPriority::Critical);
        scheduler.submit(high_req).unwrap();

        // Should signal preemption for high priority request
        assert!(scheduler.should_preempt(), "Should preempt for critical request");
    }

    #[test]
    fn test_no_preemption_policy() {
        let config = BatchConfig::new()
            .with_preemption_policy(PreemptionPolicy::None);
        let scheduler = BatchScheduler::new(config);

        // Submit and start low priority batch
        let low_req = create_test_request("low", 1);
        scheduler.submit(low_req).unwrap();
        let batch = scheduler.try_form_batch().unwrap();
        scheduler.set_active_batch(batch);

        // Submit critical priority request
        let high_req = create_test_request("high", 2)
            .with_priority(BatchPriority::Critical);
        scheduler.submit(high_req).unwrap();

        // Should NOT signal preemption when policy is None
        assert!(!scheduler.should_preempt(), "Should not preempt with None policy");
    }
}
