//! Continuous batching for high-throughput inference.
//!
//! Implements iteration-level scheduling where:
//! - New requests can join at any iteration
//! - Completed requests free their slots immediately
//! - Prefill and decode can be mixed in the same batch
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │                    BatchScheduler                     │
//! │  ┌────────────────────────────────────────────────┐  │
//! │  │  Waiting Queue    │  Active Slots  │  Complete │  │
//! │  │  [Req 5, Req 6]   │  [0][1][2][3]  │  [Req 1]  │  │
//! │  └────────────────────────────────────────────────┘  │
//! │                         ↓                            │
//! │  ┌────────────────────────────────────────────────┐  │
//! │  │              Slot-based KV Cache               │  │
//! │  │  [Slot 0: Req 2] [Slot 1: Req 3] [Slot 2: Req 4] │
//! │  └────────────────────────────────────────────────┘  │
//! └──────────────────────────────────────────────────────┘
//! ```

use std::collections::VecDeque;
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use super::arch::ModelConfig;
use super::tensor::{GpuDType, GpuTensor};
use super::InferenceError;

/// Request state in the batch scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Waiting to be scheduled.
    Waiting,
    /// In prefill phase (processing prompt).
    Prefill,
    /// In decode phase (generating tokens).
    Decode,
    /// Request completed.
    Complete,
    /// Request failed or cancelled.
    Failed,
}

/// A single inference request.
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request ID.
    pub id: u64,

    /// Input token IDs.
    pub input_ids: Vec<u32>,

    /// Generated output tokens.
    pub output_ids: Vec<u32>,

    /// Maximum tokens to generate.
    pub max_tokens: usize,

    /// Current state.
    pub state: RequestState,

    /// Assigned slot index (if active).
    pub slot_idx: Option<usize>,

    /// Current position in generation.
    pub position: usize,

    /// Prompt length (for separating prefill from decode).
    pub prompt_len: usize,
}

impl Request {
    /// Create a new request.
    pub fn new(id: u64, input_ids: Vec<u32>, max_tokens: usize) -> Self {
        let prompt_len = input_ids.len();
        Self {
            id,
            input_ids,
            output_ids: Vec::new(),
            max_tokens,
            state: RequestState::Waiting,
            slot_idx: None,
            position: 0,
            prompt_len,
        }
    }

    /// Check if generation is complete.
    pub fn is_complete(&self) -> bool {
        matches!(self.state, RequestState::Complete | RequestState::Failed)
    }

    /// Get total length (prompt + generated).
    pub fn total_len(&self) -> usize {
        self.prompt_len + self.output_ids.len()
    }
}

/// A batch slot with KV cache allocation.
#[derive(Debug)]
pub struct BatchSlot {
    /// Slot index.
    pub index: usize,

    /// Request ID using this slot (None if free).
    pub request_id: Option<u64>,

    /// Current sequence length in KV cache.
    pub seq_len: usize,

    /// Maximum sequence length for this slot.
    pub max_seq_len: usize,
}

impl BatchSlot {
    /// Create a new empty slot.
    pub fn new(index: usize, max_seq_len: usize) -> Self {
        Self {
            index,
            request_id: None,
            seq_len: 0,
            max_seq_len,
        }
    }

    /// Check if slot is free.
    pub fn is_free(&self) -> bool {
        self.request_id.is_none()
    }

    /// Assign a request to this slot.
    pub fn assign(&mut self, request_id: u64) {
        self.request_id = Some(request_id);
        self.seq_len = 0;
    }

    /// Release the slot.
    pub fn release(&mut self) {
        self.request_id = None;
        self.seq_len = 0;
    }
}

/// Slot-based KV cache for continuous batching.
///
/// Unlike single-sequence KV cache, this manages independent slots
/// that can be allocated/freed dynamically.
pub struct SlottedKvCache {
    /// Model configuration.
    #[allow(dead_code)]
    config: ModelConfig,

    /// CUDA device.
    #[allow(dead_code)]
    device: Arc<CudaDevice>,

    /// Key cache [num_slots, num_layers, max_seq_len, kv_heads, head_dim].
    #[allow(dead_code)]
    keys: GpuTensor,

    /// Value cache [num_slots, num_layers, max_seq_len, kv_heads, head_dim].
    #[allow(dead_code)]
    values: GpuTensor,

    /// Slot metadata.
    slots: Vec<BatchSlot>,

    /// Number of slots.
    num_slots: usize,

    /// Maximum sequence length per slot.
    #[allow(dead_code)]
    max_seq_len: usize,
}

impl SlottedKvCache {
    /// Create a new slotted KV cache.
    pub fn new(
        config: &ModelConfig,
        num_slots: usize,
        max_seq_len: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Self, InferenceError> {
        let num_layers = config.num_layers;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;

        // 5D cache: [num_slots, num_layers, max_seq_len, kv_heads, head_dim]
        let cache_shape = vec![num_slots, num_layers, max_seq_len, num_kv_heads, head_dim];

        let keys = GpuTensor::zeros(cache_shape.clone(), GpuDType::F16, Arc::clone(&device))?;
        let values = GpuTensor::zeros(cache_shape, GpuDType::F16, Arc::clone(&device))?;

        let slots = (0..num_slots)
            .map(|i| BatchSlot::new(i, max_seq_len))
            .collect();

        let cache_bytes = keys.size_bytes() + values.size_bytes();
        tracing::info!(
            cache_mb = cache_bytes as f64 / 1024.0 / 1024.0,
            num_slots = num_slots,
            max_seq_len = max_seq_len,
            "Created slotted KV cache"
        );

        Ok(Self {
            config: config.clone(),
            device,
            keys,
            values,
            slots,
            num_slots,
            max_seq_len,
        })
    }

    /// Get a free slot.
    pub fn allocate_slot(&mut self) -> Option<usize> {
        for slot in &mut self.slots {
            if slot.is_free() {
                return Some(slot.index);
            }
        }
        None
    }

    /// Assign a request to a slot.
    pub fn assign_slot(&mut self, slot_idx: usize, request_id: u64) -> Result<(), InferenceError> {
        if slot_idx >= self.num_slots {
            return Err(InferenceError::InvalidParam(format!(
                "Slot index {} out of range",
                slot_idx
            )));
        }
        self.slots[slot_idx].assign(request_id);
        Ok(())
    }

    /// Release a slot.
    pub fn release_slot(&mut self, slot_idx: usize) -> Result<(), InferenceError> {
        if slot_idx >= self.num_slots {
            return Err(InferenceError::InvalidParam(format!(
                "Slot index {} out of range",
                slot_idx
            )));
        }
        self.slots[slot_idx].release();
        Ok(())
    }

    /// Update sequence length for a slot.
    pub fn update_seq_len(&mut self, slot_idx: usize, seq_len: usize) {
        if slot_idx < self.num_slots {
            self.slots[slot_idx].seq_len = seq_len;
        }
    }

    /// Get sequence length for a slot.
    pub fn seq_len(&self, slot_idx: usize) -> usize {
        self.slots.get(slot_idx).map_or(0, |s| s.seq_len)
    }

    /// Get number of free slots.
    pub fn num_free_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.is_free()).count()
    }

    /// Get number of active slots.
    pub fn num_active_slots(&self) -> usize {
        self.num_slots - self.num_free_slots()
    }

    /// Get slot information.
    pub fn slot(&self, idx: usize) -> Option<&BatchSlot> {
        self.slots.get(idx)
    }

    /// Get all active slot indices.
    pub fn active_slots(&self) -> Vec<usize> {
        self.slots
            .iter()
            .filter(|s| !s.is_free())
            .map(|s| s.index)
            .collect()
    }
}

/// Batch scheduler for continuous batching.
pub struct BatchScheduler {
    /// Waiting queue of requests.
    waiting: VecDeque<Request>,

    /// Active requests (mapped by request ID).
    active: Vec<Option<Request>>,

    /// Completed requests.
    completed: Vec<Request>,

    /// Slotted KV cache.
    kv_cache: SlottedKvCache,

    /// Maximum batch size.
    #[allow(dead_code)]
    max_batch_size: usize,

    /// Next request ID.
    next_request_id: u64,
}

impl BatchScheduler {
    /// Create a new batch scheduler.
    pub fn new(
        config: &ModelConfig,
        max_batch_size: usize,
        max_seq_len: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Self, InferenceError> {
        let kv_cache = SlottedKvCache::new(config, max_batch_size, max_seq_len, device)?;

        Ok(Self {
            waiting: VecDeque::new(),
            active: vec![None; max_batch_size],
            completed: Vec::new(),
            kv_cache,
            max_batch_size,
            next_request_id: 0,
        })
    }

    /// Add a new request to the waiting queue.
    pub fn add_request(&mut self, input_ids: Vec<u32>, max_tokens: usize) -> u64 {
        let id = self.next_request_id;
        self.next_request_id += 1;

        let request = Request::new(id, input_ids, max_tokens);
        self.waiting.push_back(request);

        id
    }

    /// Schedule waiting requests to available slots.
    pub fn schedule(&mut self) -> Result<(), InferenceError> {
        while !self.waiting.is_empty() {
            if let Some(slot_idx) = self.kv_cache.allocate_slot() {
                let mut request = self.waiting.pop_front().unwrap();
                let request_id = request.id;

                request.state = RequestState::Prefill;
                request.slot_idx = Some(slot_idx);

                self.kv_cache.assign_slot(slot_idx, request_id)?;
                self.active[slot_idx] = Some(request);
            } else {
                // No free slots
                break;
            }
        }
        Ok(())
    }

    /// Get requests ready for prefill (prompt processing).
    pub fn get_prefill_requests(&self) -> Vec<(usize, &Request)> {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(idx, r)| {
                r.as_ref().and_then(|req| {
                    if req.state == RequestState::Prefill {
                        Some((idx, req))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Get requests ready for decode (token generation).
    pub fn get_decode_requests(&self) -> Vec<(usize, &Request)> {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(idx, r)| {
                r.as_ref().and_then(|req| {
                    if req.state == RequestState::Decode {
                        Some((idx, req))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Mark a request as transitioned from prefill to decode.
    pub fn transition_to_decode(&mut self, slot_idx: usize) -> Result<(), InferenceError> {
        if let Some(ref mut request) = self.active[slot_idx] {
            request.state = RequestState::Decode;
            request.position = request.prompt_len;
        }
        Ok(())
    }

    /// Add a generated token to a request.
    pub fn add_token(&mut self, slot_idx: usize, token: u32) -> Result<bool, InferenceError> {
        if let Some(ref mut request) = self.active[slot_idx] {
            request.output_ids.push(token);
            request.position += 1;

            // Update KV cache sequence length
            self.kv_cache.update_seq_len(slot_idx, request.total_len());

            // Check if generation is complete
            let is_complete = request.output_ids.len() >= request.max_tokens;
            if is_complete {
                request.state = RequestState::Complete;
            }

            return Ok(is_complete);
        }
        Ok(false)
    }

    /// Complete a request and move it to completed queue.
    pub fn complete_request(&mut self, slot_idx: usize) -> Result<Option<Request>, InferenceError> {
        if let Some(request) = self.active[slot_idx].take() {
            self.kv_cache.release_slot(slot_idx)?;
            self.completed.push(request);
            return Ok(self.completed.last().cloned());
        }
        Ok(None)
    }

    /// Get completed requests.
    pub fn get_completed(&mut self) -> Vec<Request> {
        std::mem::take(&mut self.completed)
    }

    /// Check if there are any active or waiting requests.
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || self.active.iter().any(|r| r.is_some())
    }

    /// Get statistics.
    pub fn stats(&self) -> BatchStats {
        let prefill_count = self
            .active
            .iter()
            .filter(|r| matches!(r, Some(req) if req.state == RequestState::Prefill))
            .count();
        let decode_count = self
            .active
            .iter()
            .filter(|r| matches!(r, Some(req) if req.state == RequestState::Decode))
            .count();

        BatchStats {
            waiting: self.waiting.len(),
            prefill: prefill_count,
            decode: decode_count,
            completed: self.completed.len(),
            free_slots: self.kv_cache.num_free_slots(),
        }
    }
}

/// Batch statistics.
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Requests waiting to be scheduled.
    pub waiting: usize,
    /// Requests in prefill phase.
    pub prefill: usize,
    /// Requests in decode phase.
    pub decode: usize,
    /// Completed requests not yet collected.
    pub completed: usize,
    /// Free KV cache slots.
    pub free_slots: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_inference::arch::{Activation, ModelArch};

    fn test_config() -> ModelConfig {
        ModelConfig {
            arch: ModelArch::Llama,
            vocab_size: 32000,
            hidden_size: 576,
            intermediate_size: 1536,
            num_layers: 4,
            num_attention_heads: 9,
            num_kv_heads: 3,
            head_dim: 64,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            mlp_bias: false,
            hidden_act: Activation::SiLU,
            tie_word_embeddings: true,
            sliding_window: None,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
        }
    }

    #[test]
    fn test_request_creation() {
        let request = Request::new(1, vec![1, 2, 3, 4], 100);
        assert_eq!(request.id, 1);
        assert_eq!(request.prompt_len, 4);
        assert_eq!(request.state, RequestState::Waiting);
        assert!(!request.is_complete());
    }

    #[test]
    fn test_batch_slot() {
        let mut slot = BatchSlot::new(0, 1024);
        assert!(slot.is_free());

        slot.assign(42);
        assert!(!slot.is_free());
        assert_eq!(slot.request_id, Some(42));

        slot.release();
        assert!(slot.is_free());
    }
}
