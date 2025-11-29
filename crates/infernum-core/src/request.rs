//! Request types for inference operations.

use serde::{Deserialize, Serialize};

use crate::sampling::SamplingParams;
use crate::types::{Message, ModelId, RequestId};

/// Input format for prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    /// Raw text prompt.
    Text(String),
    /// Chat messages (will be formatted according to model's chat template).
    Messages(Vec<Message>),
    /// Pre-tokenized input.
    Tokens(Vec<u32>),
}

impl From<String> for PromptInput {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for PromptInput {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<Vec<Message>> for PromptInput {
    fn from(messages: Vec<Message>) -> Self {
        Self::Messages(messages)
    }
}

/// Request for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Unique request identifier.
    #[serde(default)]
    pub request_id: RequestId,

    /// Model to use for generation.
    #[serde(default)]
    pub model: Option<ModelId>,

    /// Input prompt.
    pub prompt: PromptInput,

    /// Sampling parameters.
    #[serde(default)]
    pub sampling: SamplingParams,

    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,

    /// Whether to echo the prompt in the response.
    #[serde(default)]
    pub echo: bool,

    /// Number of completions to generate.
    #[serde(default = "default_n")]
    pub n: u32,

    /// Include log probabilities for top tokens.
    #[serde(default)]
    pub logprobs: Option<u32>,
}

fn default_n() -> u32 {
    1
}

impl GenerateRequest {
    /// Creates a new generation request with the given prompt.
    #[must_use]
    pub fn new(prompt: impl Into<PromptInput>) -> Self {
        Self {
            request_id: RequestId::new(),
            model: None,
            prompt: prompt.into(),
            sampling: SamplingParams::default(),
            stream: false,
            echo: false,
            n: 1,
            logprobs: None,
        }
    }

    /// Creates a chat completion request.
    #[must_use]
    pub fn chat(messages: Vec<Message>) -> Self {
        Self::new(PromptInput::Messages(messages))
    }

    /// Sets the model to use.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<ModelId>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the sampling parameters.
    #[must_use]
    pub fn with_sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = sampling;
        self
    }

    /// Enables streaming.
    #[must_use]
    pub fn with_stream(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Sets the number of completions.
    #[must_use]
    pub fn with_n(mut self, n: u32) -> Self {
        self.n = n;
        self
    }

    /// Enables log probabilities.
    #[must_use]
    pub fn with_logprobs(mut self, top_logprobs: u32) -> Self {
        self.logprobs = Some(top_logprobs);
        self
    }
}

/// Request for generating embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    /// Unique request identifier.
    #[serde(default)]
    pub request_id: RequestId,

    /// Model to use for embeddings.
    #[serde(default)]
    pub model: Option<ModelId>,

    /// Input texts to embed.
    pub input: EmbedInput,

    /// Encoding format for the embeddings.
    #[serde(default)]
    pub encoding_format: EncodingFormat,

    /// Dimensionality for the embeddings (if model supports it).
    #[serde(default)]
    pub dimensions: Option<u32>,
}

/// Input format for embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbedInput {
    /// Single text.
    Single(String),
    /// Multiple texts.
    Multiple(Vec<String>),
}

impl From<String> for EmbedInput {
    fn from(s: String) -> Self {
        Self::Single(s)
    }
}

impl From<&str> for EmbedInput {
    fn from(s: &str) -> Self {
        Self::Single(s.to_string())
    }
}

impl From<Vec<String>> for EmbedInput {
    fn from(v: Vec<String>) -> Self {
        Self::Multiple(v)
    }
}

impl EmbedInput {
    /// Returns the inputs as a slice of strings.
    #[must_use]
    pub fn as_texts(&self) -> Vec<&str> {
        match self {
            Self::Single(s) => vec![s.as_str()],
            Self::Multiple(v) => v.iter().map(String::as_str).collect(),
        }
    }
}

/// Encoding format for embeddings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    /// 32-bit floating point.
    #[default]
    Float,
    /// Base64 encoded binary.
    Base64,
}

impl EmbedRequest {
    /// Creates a new embedding request.
    #[must_use]
    pub fn new(input: impl Into<EmbedInput>) -> Self {
        Self {
            request_id: RequestId::new(),
            model: None,
            input: input.into(),
            encoding_format: EncodingFormat::Float,
            dimensions: None,
        }
    }

    /// Sets the model to use.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<ModelId>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the encoding format.
    #[must_use]
    pub fn with_encoding_format(mut self, format: EncodingFormat) -> Self {
        self.encoding_format = format;
        self
    }

    /// Sets the output dimensions.
    #[must_use]
    pub fn with_dimensions(mut self, dims: u32) -> Self {
        self.dimensions = Some(dims);
        self
    }
}
