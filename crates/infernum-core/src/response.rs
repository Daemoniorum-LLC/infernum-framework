//! Response types for inference operations.

use serde::{Deserialize, Serialize};

use crate::types::{FinishReason, ModelId, RequestId, Usage};

/// Response from text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// Request identifier.
    pub request_id: RequestId,

    /// Model used for generation.
    pub model: ModelId,

    /// Generated completions.
    pub choices: Vec<Choice>,

    /// Token usage statistics.
    pub usage: Usage,

    /// Time to first token in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token_ms: Option<f64>,

    /// Total generation time in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time_ms: Option<f64>,
}

/// A single completion choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    /// Index of this choice.
    pub index: u32,

    /// Generated text.
    pub text: String,

    /// Reason generation stopped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Log probabilities (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
}

/// Log probability information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbs {
    /// Token strings.
    pub tokens: Vec<String>,

    /// Log probabilities for each token.
    pub token_logprobs: Vec<f32>,

    /// Top log probabilities at each position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<Vec<TopLogProb>>>,
}

/// Top log probability entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogProb {
    /// Token string.
    pub token: String,

    /// Log probability.
    pub logprob: f32,
}

/// Information about a single generated token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    /// Token ID.
    pub id: u32,

    /// Token text.
    pub text: String,

    /// Log probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob: Option<f32>,

    /// Whether this is a special token.
    #[serde(default)]
    pub special: bool,
}

/// Response from embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    /// Request identifier.
    pub request_id: RequestId,

    /// Model used for embeddings.
    pub model: ModelId,

    /// Generated embeddings.
    pub data: Vec<Embedding>,

    /// Token usage statistics.
    pub usage: Usage,
}

/// A single embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Index of this embedding.
    pub index: u32,

    /// Embedding vector.
    pub embedding: EmbeddingData,
}

/// Embedding data in different formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingData {
    /// Float vector.
    Float(Vec<f32>),

    /// Base64 encoded binary.
    Base64(String),
}

impl EmbeddingData {
    /// Returns the embedding as a float vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is base64 encoded and decoding fails.
    pub fn as_floats(&self) -> Result<Vec<f32>, &'static str> {
        match self {
            Self::Float(v) => Ok(v.clone()),
            Self::Base64(_) => Err("base64 decoding not implemented"),
        }
    }

    /// Returns the dimensionality of the embedding.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        match self {
            Self::Float(v) => v.len(),
            Self::Base64(s) => {
                // Each f32 is 4 bytes, base64 encoding is ~4/3 ratio
                (s.len() * 3) / 16
            },
        }
    }
}
