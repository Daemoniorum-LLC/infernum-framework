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

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // StreamDelta tests
    // ==========================================================================

    #[test]
    fn test_stream_delta_text() {
        let delta = StreamDelta::text("Hello");
        assert_eq!(delta.content, Some("Hello".to_string()));
        assert!(delta.token.is_none());
    }

    #[test]
    fn test_stream_delta_token() {
        let token_info = TokenInfo {
            id: 123,
            text: "world".to_string(),
            logprob: Some(-0.5),
            special: false,
        };

        let delta = StreamDelta::token(token_info.clone());
        assert_eq!(delta.content, Some("world".to_string()));
        assert!(delta.token.is_some());
        assert_eq!(delta.token.as_ref().unwrap().id, 123);
    }

    #[test]
    fn test_stream_delta_empty() {
        let delta = StreamDelta::empty();
        assert!(delta.content.is_none());
        assert!(delta.token.is_none());
    }

    #[test]
    fn test_stream_delta_clone() {
        let delta = StreamDelta::text("clone me");
        let cloned = delta.clone();
        assert_eq!(cloned.content, delta.content);
    }

    #[test]
    fn test_stream_delta_serialization() {
        let delta = StreamDelta::text("serialize");
        let json = serde_json::to_string(&delta).expect("serialize");
        assert!(json.contains("serialize"));

        let parsed: StreamDelta = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.content, Some("serialize".to_string()));
    }

    // ==========================================================================
    // StreamChoice tests
    // ==========================================================================

    #[test]
    fn test_stream_choice_basic() {
        let choice = StreamChoice {
            index: 0,
            delta: StreamDelta::text("content"),
            finish_reason: None,
        };

        assert_eq!(choice.index, 0);
        assert_eq!(choice.delta.content, Some("content".to_string()));
        assert!(choice.finish_reason.is_none());
    }

    #[test]
    fn test_stream_choice_with_finish_reason() {
        let choice = StreamChoice {
            index: 0,
            delta: StreamDelta::empty(),
            finish_reason: Some(FinishReason::Stop),
        };

        assert_eq!(choice.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_stream_choice_serialization() {
        let choice = StreamChoice {
            index: 1,
            delta: StreamDelta::text("test"),
            finish_reason: None,
        };

        let json = serde_json::to_string(&choice).expect("serialize");
        assert!(json.contains("\"index\":1"));

        let parsed: StreamChoice = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.index, 1);
    }

    // ==========================================================================
    // StreamChunk tests
    // ==========================================================================

    #[test]
    fn test_stream_chunk_basic() {
        let chunk = StreamChunk {
            request_id: RequestId::new(),
            model: ModelId::new("test-model"),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta::text("hello"),
                finish_reason: None,
            }],
            usage: None,
        };

        assert_eq!(chunk.model.0, "test-model");
        assert_eq!(chunk.choices.len(), 1);
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn test_stream_chunk_with_usage() {
        let chunk = StreamChunk {
            request_id: RequestId::new(),
            model: ModelId::new("model"),
            choices: vec![],
            usage: Some(Usage::new(100, 50)),
        };

        assert!(chunk.usage.is_some());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
    }

    #[test]
    fn test_stream_chunk_clone() {
        let chunk = StreamChunk {
            request_id: RequestId::new(),
            model: ModelId::new("clone-model"),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta::text("clone"),
                finish_reason: None,
            }],
            usage: None,
        };

        let cloned = chunk.clone();
        assert_eq!(cloned.model.0, chunk.model.0);
        assert_eq!(cloned.choices.len(), chunk.choices.len());
    }

    // ==========================================================================
    // StreamChunkBuilder tests
    // ==========================================================================

    #[test]
    fn test_stream_chunk_builder_basic() {
        let request_id = RequestId::new();
        let model = ModelId::new("builder-model");

        let chunk = StreamChunkBuilder::new(request_id.clone(), model.clone())
            .text(0, "Hello")
            .build();

        assert_eq!(chunk.request_id, request_id);
        assert_eq!(chunk.model.0, "builder-model");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_stream_chunk_builder_multiple_choices() {
        let chunk = StreamChunkBuilder::new(RequestId::new(), ModelId::new("model"))
            .text(0, "Choice 1")
            .text(1, "Choice 2")
            .build();

        assert_eq!(chunk.choices.len(), 2);
        assert_eq!(chunk.choices[0].index, 0);
        assert_eq!(chunk.choices[1].index, 1);
    }

    #[test]
    fn test_stream_chunk_builder_with_finish() {
        let chunk = StreamChunkBuilder::new(RequestId::new(), ModelId::new("model"))
            .text(0, "Final token")
            .finish(0, FinishReason::Stop)
            .build();

        assert_eq!(chunk.choices.len(), 2);
        assert_eq!(chunk.choices[1].finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_stream_chunk_builder_with_usage() {
        let usage = Usage::new(200, 100);
        let chunk = StreamChunkBuilder::new(RequestId::new(), ModelId::new("model"))
            .usage(usage)
            .build();

        assert!(chunk.usage.is_some());
        assert_eq!(chunk.usage.as_ref().unwrap().total_tokens, 300);
    }

    // ==========================================================================
    // TokenStream tests
    // ==========================================================================

    #[tokio::test]
    async fn test_token_stream_empty() {
        let stream = TokenStream::empty();
        let chunks = stream.collect().await.expect("collect");
        assert!(chunks.is_empty());
    }

    #[tokio::test]
    async fn test_token_stream_once() {
        let chunk = StreamChunkBuilder::new(RequestId::new(), ModelId::new("model"))
            .text(0, "single chunk")
            .build();

        let stream = TokenStream::once(chunk);
        let chunks = stream.collect().await.expect("collect");

        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].choices[0].delta.content,
            Some("single chunk".to_string())
        );
    }

    #[tokio::test]
    async fn test_token_stream_collect_text() {
        use futures::stream;

        let chunks = vec![
            Ok(StreamChunkBuilder::new(RequestId::new(), ModelId::new("m"))
                .text(0, "Hello ")
                .build()),
            Ok(StreamChunkBuilder::new(RequestId::new(), ModelId::new("m"))
                .text(0, "World!")
                .build()),
        ];

        let stream = TokenStream::new(stream::iter(chunks));
        let text = stream.collect_text().await.expect("collect text");

        assert_eq!(text, "Hello World!");
    }

    #[tokio::test]
    async fn test_token_stream_collect_multiple_choices() {
        use futures::stream;

        let chunks = vec![
            Ok(StreamChunkBuilder::new(RequestId::new(), ModelId::new("m"))
                .text(0, "A")
                .text(1, "X")
                .build()),
            Ok(StreamChunkBuilder::new(RequestId::new(), ModelId::new("m"))
                .text(0, "B")
                .text(1, "Y")
                .build()),
        ];

        let stream = TokenStream::new(stream::iter(chunks));
        let text = stream.collect_text().await.expect("collect text");

        // All choices' text concatenated
        assert_eq!(text, "AXBY");
    }

    #[tokio::test]
    async fn test_token_stream_as_stream() {
        use futures::StreamExt;

        let chunk = StreamChunkBuilder::new(RequestId::new(), ModelId::new("model"))
            .text(0, "streaming")
            .build();

        let mut stream = TokenStream::once(chunk);

        let first = stream.next().await;
        assert!(first.is_some());
        assert!(first.unwrap().is_ok());

        let second = stream.next().await;
        assert!(second.is_none());
    }
}
