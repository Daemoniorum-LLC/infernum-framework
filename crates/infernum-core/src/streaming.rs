//! Streaming types for real-time token generation.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::response::TokenInfo;
use crate::types::{FinishReason, ModelId, RequestId, Usage};

/// A chunk in a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Request identifier.
    pub request_id: RequestId,

    /// Model used for generation.
    pub model: ModelId,

    /// Choice updates.
    pub choices: Vec<StreamChoice>,

    /// Usage (only present in final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A choice update in a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    /// Index of this choice.
    pub index: u32,

    /// Delta content.
    pub delta: StreamDelta,

    /// Finish reason (only present when done).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// Delta content in a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDelta {
    /// New text content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Token information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<TokenInfo>,
}

impl StreamDelta {
    /// Creates a delta with text content.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            token: None,
        }
    }

    /// Creates a delta with token information.
    #[must_use]
    pub fn token(token: TokenInfo) -> Self {
        Self {
            content: Some(token.text.clone()),
            token: Some(token),
        }
    }

    /// Creates an empty delta (for finish signals).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            content: None,
            token: None,
        }
    }
}

/// A stream of generated tokens.
pub struct TokenStream {
    inner: Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>,
}

impl TokenStream {
    /// Creates a new `TokenStream` from a boxed stream.
    pub fn new<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<StreamChunk>> + Send + 'static,
    {
        Self {
            inner: Box::pin(stream),
        }
    }

    /// Creates an empty stream.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(futures::stream::empty())
    }

    /// Creates a stream from a single chunk.
    #[must_use]
    pub fn once(chunk: StreamChunk) -> Self {
        Self::new(futures::stream::once(async move { Ok(chunk) }))
    }

    /// Collects all chunks into a vector.
    ///
    /// # Errors
    ///
    /// Returns an error if any chunk fails.
    pub async fn collect(self) -> Result<Vec<StreamChunk>> {
        use futures::StreamExt;
        let mut chunks = Vec::new();
        let mut stream = self;
        while let Some(result) = stream.next().await {
            chunks.push(result?);
        }
        Ok(chunks)
    }

    /// Collects all text into a single string.
    ///
    /// # Errors
    ///
    /// Returns an error if any chunk fails.
    pub async fn collect_text(self) -> Result<String> {
        let chunks = self.collect().await?;
        let mut text = String::new();
        for chunk in chunks {
            for choice in chunk.choices {
                if let Some(content) = choice.delta.content {
                    text.push_str(&content);
                }
            }
        }
        Ok(text)
    }
}

impl Stream for TokenStream {
    type Item = Result<StreamChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// Builder for creating stream chunks.
#[derive(Debug)]
pub struct StreamChunkBuilder {
    request_id: RequestId,
    model: ModelId,
    choices: Vec<StreamChoice>,
    usage: Option<Usage>,
}

impl StreamChunkBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new(request_id: RequestId, model: ModelId) -> Self {
        Self {
            request_id,
            model,
            choices: Vec::new(),
            usage: None,
        }
    }

    /// Adds a choice with text content.
    #[must_use]
    pub fn text(mut self, index: u32, content: impl Into<String>) -> Self {
        self.choices.push(StreamChoice {
            index,
            delta: StreamDelta::text(content),
            finish_reason: None,
        });
        self
    }

    /// Adds a choice with a finish reason.
    #[must_use]
    pub fn finish(mut self, index: u32, reason: FinishReason) -> Self {
        self.choices.push(StreamChoice {
            index,
            delta: StreamDelta::empty(),
            finish_reason: Some(reason),
        });
        self
    }

    /// Sets the usage (for final chunk).
    #[must_use]
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Builds the chunk.
    #[must_use]
    pub fn build(self) -> StreamChunk {
        StreamChunk {
            request_id: self.request_id,
            model: self.model,
            choices: self.choices,
            usage: self.usage,
        }
    }
}
