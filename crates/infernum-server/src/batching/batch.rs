//! Batch and sequence management for continuous batching.
//!
//! This module provides types for managing batches of requests and their
//! associated sequences during token-by-token generation.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tokio::sync::oneshot;

use super::BatchPriority;

/// Unique identifier for a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchId(u64);

impl BatchId {
    /// Creates a new batch ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw ID value.
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for BatchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "batch-{}", self.0)
    }
}

/// Unique identifier for a sequence within a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(u64);

impl SequenceId {
    /// Creates a new sequence ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw ID value.
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for SequenceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seq-{}", self.0)
    }
}

/// State of a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchState {
    /// Batch is being formed.
    Forming,
    /// Batch is actively being processed.
    Running,
    /// Batch is paused (preempted).
    Paused,
    /// Batch has completed.
    Completed,
    /// Batch was aborted.
    Aborted,
}

impl fmt::Display for BatchState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forming => write!(f, "forming"),
            Self::Running => write!(f, "running"),
            Self::Paused => write!(f, "paused"),
            Self::Completed => write!(f, "completed"),
            Self::Aborted => write!(f, "aborted"),
        }
    }
}

/// State of a sequence within a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    /// Waiting to start generation.
    Waiting,
    /// Actively generating tokens.
    Running,
    /// Generation paused (waiting for batch slot).
    Paused,
    /// Finished generation (hit stop condition).
    Finished,
    /// Preempted and swapped out.
    Swapped,
    /// Aborted due to error.
    Aborted,
}

impl fmt::Display for SequenceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Waiting => write!(f, "waiting"),
            Self::Running => write!(f, "running"),
            Self::Paused => write!(f, "paused"),
            Self::Finished => write!(f, "finished"),
            Self::Swapped => write!(f, "swapped"),
            Self::Aborted => write!(f, "aborted"),
        }
    }
}

/// State of a pending request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Waiting in queue.
    Queued,
    /// Being processed.
    Processing,
    /// Completed successfully.
    Completed,
    /// Cancelled by client.
    Cancelled,
    /// Failed with error.
    Failed,
}

impl fmt::Display for RequestState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Processing => write!(f, "processing"),
            Self::Completed => write!(f, "completed"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// A pending request waiting to be batched.
#[derive(Debug)]
pub struct PendingRequest {
    /// Unique request ID.
    pub id: String,

    /// Sequence ID for this request.
    pub sequence_id: SequenceId,

    /// Priority level.
    pub priority: BatchPriority,

    /// Input token IDs.
    pub input_tokens: Vec<u32>,

    /// Maximum tokens to generate.
    pub max_tokens: usize,

    /// When the request was queued.
    pub queued_at: Instant,

    /// Current state.
    pub state: RequestState,

    /// Estimated memory requirement in bytes.
    pub estimated_memory: u64,

    /// Model ID.
    pub model: String,

    /// Stop sequences.
    pub stop_sequences: Vec<Vec<u32>>,

    /// Temperature for sampling.
    pub temperature: f32,

    /// Channel for sending the result.
    result_tx: Option<oneshot::Sender<Result<Vec<u32>, String>>>,
}

impl PendingRequest {
    /// Creates a new pending request.
    pub fn new(
        id: impl Into<String>,
        sequence_id: SequenceId,
        model: impl Into<String>,
        input_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> (Self, oneshot::Receiver<Result<Vec<u32>, String>>) {
        let (tx, rx) = oneshot::channel();

        let request = Self {
            id: id.into(),
            sequence_id,
            priority: BatchPriority::default(),
            input_tokens,
            max_tokens,
            queued_at: Instant::now(),
            state: RequestState::Queued,
            estimated_memory: 0,
            model: model.into(),
            stop_sequences: Vec::new(),
            temperature: 1.0,
            result_tx: Some(tx),
        };

        (request, rx)
    }

    /// Sets the priority.
    pub fn with_priority(mut self, priority: BatchPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Sets the stop sequences.
    pub fn with_stop_sequences(mut self, stops: Vec<Vec<u32>>) -> Self {
        self.stop_sequences = stops;
        self
    }

    /// Returns the time spent waiting in queue.
    pub fn wait_time(&self) -> Duration {
        self.queued_at.elapsed()
    }

    /// Returns the prompt length.
    pub fn prompt_length(&self) -> usize {
        self.input_tokens.len()
    }

    /// Returns the total sequence length (prompt + max output).
    pub fn total_length(&self) -> usize {
        self.input_tokens.len() + self.max_tokens
    }

    /// Completes the request with a result.
    pub fn complete(mut self, output_tokens: Vec<u32>) {
        self.state = RequestState::Completed;
        if let Some(tx) = self.result_tx.take() {
            let _ = tx.send(Ok(output_tokens));
        }
    }

    /// Fails the request with an error.
    pub fn fail(mut self, error: String) {
        self.state = RequestState::Failed;
        if let Some(tx) = self.result_tx.take() {
            let _ = tx.send(Err(error));
        }
    }

    /// Cancels the request.
    pub fn cancel(mut self) {
        self.state = RequestState::Cancelled;
        if let Some(tx) = self.result_tx.take() {
            let _ = tx.send(Err("Request cancelled".to_string()));
        }
    }
}

/// A group of sequences for the same request (e.g., beam search).
#[derive(Debug)]
pub struct SequenceGroup {
    /// Request ID.
    pub request_id: String,

    /// Sequences in this group.
    pub sequences: Vec<Sequence>,

    /// Priority level.
    pub priority: BatchPriority,

    /// When the group was created.
    pub created_at: Instant,

    /// When processing started.
    pub started_at: Option<Instant>,

    /// Sampling parameters.
    pub sampling_params: SamplingParams,
}

impl SequenceGroup {
    /// Creates a new sequence group with a single sequence.
    pub fn new(
        request_id: impl Into<String>,
        sequence: Sequence,
        sampling_params: SamplingParams,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            sequences: vec![sequence],
            priority: BatchPriority::default(),
            created_at: Instant::now(),
            started_at: None,
            sampling_params,
        }
    }

    /// Sets the priority.
    pub fn with_priority(mut self, priority: BatchPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Marks the group as started.
    pub fn start(&mut self) {
        if self.started_at.is_none() {
            self.started_at = Some(Instant::now());
        }
    }

    /// Returns the number of sequences.
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Returns the number of running sequences.
    pub fn num_running(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| s.state == SequenceState::Running)
            .count()
    }

    /// Returns the number of finished sequences.
    pub fn num_finished(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| s.state == SequenceState::Finished)
            .count()
    }

    /// Returns true if all sequences are finished.
    pub fn is_finished(&self) -> bool {
        self.sequences
            .iter()
            .all(|s| matches!(s.state, SequenceState::Finished | SequenceState::Aborted))
    }

    /// Returns the maximum sequence length in the group.
    pub fn max_sequence_length(&self) -> usize {
        self.sequences
            .iter()
            .map(|s| s.current_length())
            .max()
            .unwrap_or(0)
    }

    /// Returns the total tokens across all sequences.
    pub fn total_tokens(&self) -> usize {
        self.sequences.iter().map(|s| s.current_length()).sum()
    }
}

/// A single sequence being generated.
#[derive(Debug)]
pub struct Sequence {
    /// Sequence ID.
    pub id: SequenceId,

    /// Current state.
    pub state: SequenceState,

    /// Token IDs (prompt + generated).
    pub tokens: Vec<u32>,

    /// Number of prompt tokens.
    pub prompt_length: usize,

    /// Maximum tokens to generate.
    pub max_tokens: usize,

    /// Log probabilities for generated tokens.
    pub logprobs: Option<Vec<f32>>,

    /// KV cache block IDs.
    pub cache_blocks: Vec<usize>,

    /// Cumulative log probability.
    pub cumulative_logprob: f64,

    /// Finish reason if finished.
    pub finish_reason: Option<FinishReason>,
}

impl Sequence {
    /// Creates a new sequence.
    pub fn new(id: SequenceId, prompt_tokens: Vec<u32>, max_tokens: usize) -> Self {
        let prompt_length = prompt_tokens.len();
        Self {
            id,
            state: SequenceState::Waiting,
            tokens: prompt_tokens,
            prompt_length,
            max_tokens,
            logprobs: None,
            cache_blocks: Vec::new(),
            cumulative_logprob: 0.0,
            finish_reason: None,
        }
    }

    /// Enables logprob collection.
    pub fn with_logprobs(mut self) -> Self {
        self.logprobs = Some(Vec::new());
        self
    }

    /// Returns the current length (prompt + generated).
    pub fn current_length(&self) -> usize {
        self.tokens.len()
    }

    /// Returns the number of generated tokens.
    pub fn num_generated(&self) -> usize {
        self.tokens.len().saturating_sub(self.prompt_length)
    }

    /// Returns the remaining tokens to generate.
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.num_generated())
    }

    /// Returns true if at max length.
    pub fn is_at_max_length(&self) -> bool {
        self.num_generated() >= self.max_tokens
    }

    /// Returns the generated tokens.
    pub fn generated_tokens(&self) -> &[u32] {
        &self.tokens[self.prompt_length..]
    }

    /// Appends a token with optional logprob.
    pub fn append_token(&mut self, token: u32, logprob: Option<f32>) {
        self.tokens.push(token);
        if let (Some(ref mut probs), Some(prob)) = (&mut self.logprobs, logprob) {
            probs.push(prob);
            self.cumulative_logprob += prob as f64;
        }
    }

    /// Marks the sequence as finished.
    pub fn finish(&mut self, reason: FinishReason) {
        self.state = SequenceState::Finished;
        self.finish_reason = Some(reason);
    }

    /// Marks the sequence as aborted.
    pub fn abort(&mut self) {
        self.state = SequenceState::Aborted;
        self.finish_reason = Some(FinishReason::Aborted);
    }
}

/// Reason for sequence completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit a stop token.
    Stop,
    /// Reached maximum length.
    Length,
    /// Hit a stop sequence.
    StopSequence,
    /// Aborted due to error.
    Aborted,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stop => write!(f, "stop"),
            Self::Length => write!(f, "length"),
            Self::StopSequence => write!(f, "stop_sequence"),
            Self::Aborted => write!(f, "aborted"),
        }
    }
}

/// Sampling parameters for generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for softmax.
    pub temperature: f32,

    /// Top-p (nucleus) sampling.
    pub top_p: f32,

    /// Top-k sampling.
    pub top_k: i32,

    /// Repetition penalty.
    pub repetition_penalty: f32,

    /// Presence penalty.
    pub presence_penalty: f32,

    /// Frequency penalty.
    pub frequency_penalty: f32,

    /// Stop token IDs.
    pub stop_tokens: Vec<u32>,

    /// Stop sequences (as token IDs).
    pub stop_sequences: Vec<Vec<u32>>,

    /// Whether to return logprobs.
    pub logprobs: bool,

    /// Number of top logprobs to return.
    pub top_logprobs: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1, // Disabled
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_tokens: Vec::new(),
            stop_sequences: Vec::new(),
            logprobs: false,
            top_logprobs: 0,
        }
    }
}

impl SamplingParams {
    /// Creates greedy sampling parameters.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Builder method for temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Builder method for top_p.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Builder method for top_k.
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Builder method for stop tokens.
    pub fn with_stop_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.stop_tokens = tokens;
        self
    }

    /// Returns true if this is greedy (deterministic) sampling.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 || self.temperature < 1e-6
    }
}

/// An entry in an active batch.
#[derive(Debug)]
pub struct BatchEntry {
    /// Sequence group.
    pub group: SequenceGroup,

    /// Position in the batch.
    pub position: usize,

    /// When added to the batch.
    pub added_at: Instant,
}

impl BatchEntry {
    /// Creates a new batch entry.
    pub fn new(group: SequenceGroup, position: usize) -> Self {
        Self {
            group,
            position,
            added_at: Instant::now(),
        }
    }

    /// Returns the time in batch.
    pub fn time_in_batch(&self) -> Duration {
        self.added_at.elapsed()
    }
}

/// An active batch being processed.
#[derive(Debug)]
pub struct ActiveBatch {
    /// Batch ID.
    pub id: BatchId,

    /// Current state.
    pub state: BatchState,

    /// Entries in the batch.
    entries: RwLock<Vec<BatchEntry>>,

    /// When the batch was created.
    pub created_at: Instant,

    /// When the batch started running.
    pub started_at: Option<Instant>,

    /// Total tokens processed.
    tokens_processed: AtomicU64,

    /// Maximum batch size.
    max_size: usize,

    /// Maximum tokens per batch.
    max_tokens: usize,
}

impl ActiveBatch {
    /// Creates a new active batch.
    pub fn new(id: BatchId, max_size: usize, max_tokens: usize) -> Self {
        Self {
            id,
            state: BatchState::Forming,
            entries: RwLock::new(Vec::with_capacity(max_size)),
            created_at: Instant::now(),
            started_at: None,
            tokens_processed: AtomicU64::new(0),
            max_size,
            max_tokens,
        }
    }

    /// Tries to add a sequence group to the batch.
    pub fn try_add(&self, group: SequenceGroup) -> Result<usize, SequenceGroup> {
        let mut entries = self.entries.write();

        // Check size limit
        if entries.len() >= self.max_size {
            return Err(group);
        }

        // Check token limit
        let current_tokens: usize = entries.iter().map(|e| e.group.total_tokens()).sum();
        let new_tokens = group.total_tokens();

        if current_tokens + new_tokens > self.max_tokens {
            return Err(group);
        }

        let position = entries.len();
        entries.push(BatchEntry::new(group, position));
        Ok(position)
    }

    /// Removes a sequence group by request ID.
    pub fn remove(&self, request_id: &str) -> Option<SequenceGroup> {
        let mut entries = self.entries.write();
        if let Some(pos) = entries
            .iter()
            .position(|e| e.group.request_id == request_id)
        {
            let entry = entries.remove(pos);
            // Update positions
            for (i, e) in entries.iter_mut().enumerate() {
                e.position = i;
            }
            Some(entry.group)
        } else {
            None
        }
    }

    /// Returns the current batch size.
    pub fn size(&self) -> usize {
        self.entries.read().len()
    }

    /// Returns true if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Returns true if the batch is full.
    pub fn is_full(&self) -> bool {
        self.entries.read().len() >= self.max_size
    }

    /// Returns the total tokens in the batch.
    pub fn total_tokens(&self) -> usize {
        self.entries
            .read()
            .iter()
            .map(|e| e.group.total_tokens())
            .sum()
    }

    /// Starts the batch.
    pub fn start(&mut self) {
        self.state = BatchState::Running;
        self.started_at = Some(Instant::now());
    }

    /// Pauses the batch.
    pub fn pause(&mut self) {
        self.state = BatchState::Paused;
    }

    /// Resumes the batch.
    pub fn resume(&mut self) {
        self.state = BatchState::Running;
    }

    /// Completes the batch.
    pub fn complete(&mut self) {
        self.state = BatchState::Completed;
    }

    /// Aborts the batch.
    pub fn abort(&mut self) {
        self.state = BatchState::Aborted;
    }

    /// Records tokens processed.
    pub fn record_tokens(&self, count: u64) {
        self.tokens_processed.fetch_add(count, Ordering::Relaxed);
    }

    /// Returns total tokens processed.
    pub fn tokens_processed(&self) -> u64 {
        self.tokens_processed.load(Ordering::Relaxed)
    }

    /// Returns the number of finished sequence groups.
    pub fn num_finished(&self) -> usize {
        self.entries
            .read()
            .iter()
            .filter(|e| e.group.is_finished())
            .count()
    }

    /// Returns true if all sequences are finished.
    pub fn all_finished(&self) -> bool {
        let entries = self.entries.read();
        !entries.is_empty() && entries.iter().all(|e| e.group.is_finished())
    }

    /// Gets a snapshot of the batch stats.
    pub fn stats(&self) -> BatchStats {
        let entries = self.entries.read();
        let num_sequences: usize = entries.iter().map(|e| e.group.num_sequences()).sum();
        let num_running: usize = entries.iter().map(|e| e.group.num_running()).sum();
        let num_finished: usize = entries.iter().map(|e| e.group.num_finished()).sum();

        BatchStats {
            batch_id: self.id,
            state: self.state,
            num_groups: entries.len(),
            num_sequences,
            num_running,
            num_finished,
            total_tokens: self.total_tokens(),
            tokens_processed: self.tokens_processed(),
            created_at: self.created_at,
            started_at: self.started_at,
        }
    }

    /// Iterates over sequence groups.
    pub fn for_each_group<F>(&self, mut f: F)
    where
        F: FnMut(&SequenceGroup),
    {
        let entries = self.entries.read();
        for entry in entries.iter() {
            f(&entry.group);
        }
    }

    /// Iterates over sequence groups mutably.
    pub fn for_each_group_mut<F>(&self, mut f: F)
    where
        F: FnMut(&mut SequenceGroup),
    {
        let mut entries = self.entries.write();
        for entry in entries.iter_mut() {
            f(&mut entry.group);
        }
    }
}

/// Statistics for a batch.
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Batch ID.
    pub batch_id: BatchId,

    /// Current state.
    pub state: BatchState,

    /// Number of sequence groups.
    pub num_groups: usize,

    /// Total number of sequences.
    pub num_sequences: usize,

    /// Number of running sequences.
    pub num_running: usize,

    /// Number of finished sequences.
    pub num_finished: usize,

    /// Total tokens in the batch.
    pub total_tokens: usize,

    /// Tokens processed so far.
    pub tokens_processed: u64,

    /// When the batch was created.
    pub created_at: Instant,

    /// When the batch started.
    pub started_at: Option<Instant>,
}

impl BatchStats {
    /// Returns the batch duration.
    pub fn duration(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Returns the running duration (if started).
    pub fn running_duration(&self) -> Option<Duration> {
        self.started_at.map(|s| s.elapsed())
    }

    /// Returns tokens per second.
    pub fn tokens_per_second(&self) -> Option<f64> {
        self.running_duration().map(|d| {
            let secs = d.as_secs_f64();
            if secs > 0.0 {
                self.tokens_processed as f64 / secs
            } else {
                0.0
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_id() {
        let id = BatchId::new(42);
        assert_eq!(id.as_u64(), 42);
        assert_eq!(id.to_string(), "batch-42");
    }

    #[test]
    fn test_sequence_id() {
        let id = SequenceId::new(123);
        assert_eq!(id.as_u64(), 123);
        assert_eq!(id.to_string(), "seq-123");
    }

    #[test]
    fn test_batch_state_display() {
        assert_eq!(BatchState::Forming.to_string(), "forming");
        assert_eq!(BatchState::Running.to_string(), "running");
        assert_eq!(BatchState::Paused.to_string(), "paused");
        assert_eq!(BatchState::Completed.to_string(), "completed");
        assert_eq!(BatchState::Aborted.to_string(), "aborted");
    }

    #[test]
    fn test_sequence_state_display() {
        assert_eq!(SequenceState::Waiting.to_string(), "waiting");
        assert_eq!(SequenceState::Running.to_string(), "running");
        assert_eq!(SequenceState::Finished.to_string(), "finished");
    }

    #[test]
    fn test_pending_request_new() {
        let seq_id = SequenceId::new(1);
        let tokens = vec![1, 2, 3, 4, 5];
        let (request, _rx) = PendingRequest::new("req-1", seq_id, "llama", tokens.clone(), 100);

        assert_eq!(request.id, "req-1");
        assert_eq!(request.sequence_id, seq_id);
        assert_eq!(request.prompt_length(), 5);
        assert_eq!(request.total_length(), 105);
        assert_eq!(request.state, RequestState::Queued);
    }

    #[test]
    fn test_pending_request_complete() {
        let seq_id = SequenceId::new(1);
        let (request, rx) = PendingRequest::new("req-1", seq_id, "llama", vec![1, 2], 10);

        request.complete(vec![100, 101, 102]);

        let result = rx.blocking_recv();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().unwrap(), vec![100, 101, 102]);
    }

    #[test]
    fn test_pending_request_fail() {
        let seq_id = SequenceId::new(1);
        let (request, rx) = PendingRequest::new("req-1", seq_id, "llama", vec![1, 2], 10);

        request.fail("Out of memory".to_string());

        let result = rx.blocking_recv();
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn test_sequence_new() {
        let seq = Sequence::new(SequenceId::new(1), vec![10, 20, 30], 50);

        assert_eq!(seq.prompt_length, 3);
        assert_eq!(seq.max_tokens, 50);
        assert_eq!(seq.current_length(), 3);
        assert_eq!(seq.num_generated(), 0);
        assert_eq!(seq.remaining_tokens(), 50);
        assert!(!seq.is_at_max_length());
    }

    #[test]
    fn test_sequence_append_token() {
        let mut seq = Sequence::new(SequenceId::new(1), vec![10, 20], 5).with_logprobs();

        seq.append_token(100, Some(-0.5));
        seq.append_token(101, Some(-0.3));

        assert_eq!(seq.current_length(), 4);
        assert_eq!(seq.num_generated(), 2);
        assert_eq!(seq.generated_tokens(), &[100, 101]);
        assert_eq!(seq.logprobs.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_sequence_finish() {
        let mut seq = Sequence::new(SequenceId::new(1), vec![10], 5);

        seq.finish(FinishReason::Stop);

        assert_eq!(seq.state, SequenceState::Finished);
        assert_eq!(seq.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_sequence_group_new() {
        let seq = Sequence::new(SequenceId::new(1), vec![1, 2, 3], 10);
        let params = SamplingParams::default();
        let group = SequenceGroup::new("req-1", seq, params);

        assert_eq!(group.request_id, "req-1");
        assert_eq!(group.num_sequences(), 1);
        assert!(!group.is_finished());
    }

    #[test]
    fn test_sequence_group_is_finished() {
        let mut seq = Sequence::new(SequenceId::new(1), vec![1], 10);
        seq.finish(FinishReason::Length);

        let params = SamplingParams::default();
        let group = SequenceGroup::new("req-1", seq, params);

        assert!(group.is_finished());
        assert_eq!(group.num_finished(), 1);
    }

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();

        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 1.0);
        assert_eq!(params.top_k, -1);
        assert!(!params.is_greedy());
    }

    #[test]
    fn test_sampling_params_greedy() {
        let params = SamplingParams::greedy();

        assert_eq!(params.temperature, 0.0);
        assert!(params.is_greedy());
    }

    #[test]
    fn test_sampling_params_builder() {
        let params = SamplingParams::default()
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_top_k(50)
            .with_stop_tokens(vec![2]);

        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.top_k, 50);
        assert_eq!(params.stop_tokens, vec![2]);
    }

    #[test]
    fn test_active_batch_new() {
        let batch = ActiveBatch::new(BatchId::new(1), 32, 4096);

        assert_eq!(batch.id, BatchId::new(1));
        assert_eq!(batch.state, BatchState::Forming);
        assert!(batch.is_empty());
        assert!(!batch.is_full());
    }

    #[test]
    fn test_active_batch_try_add() {
        let batch = ActiveBatch::new(BatchId::new(1), 32, 4096);

        let seq = Sequence::new(SequenceId::new(1), vec![1, 2, 3], 10);
        let group = SequenceGroup::new("req-1", seq, SamplingParams::default());

        let result = batch.try_add(group);
        assert!(result.is_ok());
        assert_eq!(batch.size(), 1);
    }

    #[test]
    fn test_active_batch_size_limit() {
        let batch = ActiveBatch::new(BatchId::new(1), 2, 4096);

        // Add two groups
        for i in 0..2 {
            let seq = Sequence::new(SequenceId::new(i), vec![1, 2], 10);
            let group = SequenceGroup::new(format!("req-{}", i), seq, SamplingParams::default());
            let _ = batch.try_add(group);
        }

        assert!(batch.is_full());

        // Try to add one more
        let seq = Sequence::new(SequenceId::new(99), vec![1, 2], 10);
        let group = SequenceGroup::new("req-overflow", seq, SamplingParams::default());
        let result = batch.try_add(group);
        assert!(result.is_err());
    }

    #[test]
    fn test_active_batch_remove() {
        let batch = ActiveBatch::new(BatchId::new(1), 32, 4096);

        let seq = Sequence::new(SequenceId::new(1), vec![1, 2], 10);
        let group = SequenceGroup::new("req-1", seq, SamplingParams::default());
        let _ = batch.try_add(group);

        assert_eq!(batch.size(), 1);

        let removed = batch.remove("req-1");
        assert!(removed.is_some());
        assert_eq!(batch.size(), 0);
    }

    #[test]
    fn test_active_batch_state_transitions() {
        let mut batch = ActiveBatch::new(BatchId::new(1), 32, 4096);

        assert_eq!(batch.state, BatchState::Forming);

        batch.start();
        assert_eq!(batch.state, BatchState::Running);
        assert!(batch.started_at.is_some());

        batch.pause();
        assert_eq!(batch.state, BatchState::Paused);

        batch.resume();
        assert_eq!(batch.state, BatchState::Running);

        batch.complete();
        assert_eq!(batch.state, BatchState::Completed);
    }

    #[test]
    fn test_active_batch_stats() {
        let batch = ActiveBatch::new(BatchId::new(1), 32, 4096);

        let seq = Sequence::new(SequenceId::new(1), vec![1, 2, 3], 10);
        let group = SequenceGroup::new("req-1", seq, SamplingParams::default());
        let _ = batch.try_add(group);

        batch.record_tokens(50);

        let stats = batch.stats();
        assert_eq!(stats.batch_id, BatchId::new(1));
        assert_eq!(stats.num_groups, 1);
        assert_eq!(stats.num_sequences, 1);
        assert_eq!(stats.tokens_processed, 50);
    }

    #[test]
    fn test_finish_reason_display() {
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::Length.to_string(), "length");
        assert_eq!(FinishReason::StopSequence.to_string(), "stop_sequence");
        assert_eq!(FinishReason::Aborted.to_string(), "aborted");
    }

    #[test]
    fn test_batch_entry() {
        let seq = Sequence::new(SequenceId::new(1), vec![1, 2], 10);
        let group = SequenceGroup::new("req-1", seq, SamplingParams::default());
        let entry = BatchEntry::new(group, 0);

        assert_eq!(entry.position, 0);
        assert!(entry.time_in_batch() < std::time::Duration::from_secs(1));
    }
}
