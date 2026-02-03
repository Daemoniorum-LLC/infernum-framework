//! Mock implementations for testing
//!
//! Provides mock versions of core Infernum traits that can be configured
//! to return specific responses or simulate error conditions.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use infernum_core::response::{Choice, GenerateResponse};
use infernum_core::types::{FinishReason, ModelId, RequestId, Usage};
use infernum_core::GenerateRequest;

/// Configuration for mock inference behavior
#[derive(Debug, Clone)]
pub struct MockConfig {
    /// Default response text for generate requests
    pub default_response: String,
    /// Latency to simulate (milliseconds)
    pub latency_ms: u64,
    /// Whether to simulate errors
    pub should_fail: bool,
    /// Error message when failing
    pub error_message: String,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self {
            default_response: "This is a mock response from the test engine.".to_string(),
            latency_ms: 0,
            should_fail: false,
            error_message: "Mock error".to_string(),
        }
    }
}

/// Mock inference engine for testing
///
/// # Example
///
/// ```rust,ignore
/// use test_utils::MockInferenceEngine;
///
/// let engine = MockInferenceEngine::new();
/// // Use engine.generate() in tests
/// ```
#[derive(Debug, Clone)]
pub struct MockInferenceEngine {
    config: Arc<RwLock<MockConfig>>,
    call_count: Arc<RwLock<u64>>,
    last_request: Arc<RwLock<Option<GenerateRequest>>>,
}

impl MockInferenceEngine {
    /// Create a new mock engine with default configuration
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(MockConfig::default())),
            call_count: Arc::new(RwLock::new(0)),
            last_request: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a mock engine that will fail with the given error
    pub fn failing(error_message: &str) -> Self {
        let mut config = MockConfig::default();
        config.should_fail = true;
        config.error_message = error_message.to_string();

        Self {
            config: Arc::new(RwLock::new(config)),
            call_count: Arc::new(RwLock::new(0)),
            last_request: Arc::new(RwLock::new(None)),
        }
    }

    /// Configure the mock to return a specific response
    pub async fn set_response(&self, response: &str) {
        let mut config = self.config.write().await;
        config.default_response = response.to_string();
    }

    /// Configure simulated latency
    pub async fn set_latency(&self, ms: u64) {
        let mut config = self.config.write().await;
        config.latency_ms = ms;
    }

    /// Get the number of times generate was called
    pub async fn call_count(&self) -> u64 {
        *self.call_count.read().await
    }

    /// Get the last request that was made
    pub async fn last_request(&self) -> Option<GenerateRequest> {
        self.last_request.read().await.clone()
    }

    /// Simulate a generate request
    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, MockError> {
        // Record the call
        {
            let mut count = self.call_count.write().await;
            *count += 1;
        }
        {
            let mut last = self.last_request.write().await;
            *last = Some(request.clone());
        }

        let config = self.config.read().await;

        // Simulate latency
        if config.latency_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(config.latency_ms)).await;
        }

        // Check for configured failure
        if config.should_fail {
            return Err(MockError::SimulatedError(config.error_message.clone()));
        }

        // Return mock response matching the actual GenerateResponse structure
        Ok(GenerateResponse {
            request_id: RequestId::new(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            model: ModelId::from("mock-model"),
            choices: vec![Choice {
                index: 0,
                text: config.default_response.clone(),
                message: None,
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
            time_to_first_token_ms: Some(10.0),
            total_time_ms: Some(50.0),
        })
    }
}

impl Default for MockInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur in mock implementations
#[derive(Debug, Clone, thiserror::Error)]
pub enum MockError {
    /// A simulated error for testing error handling
    #[error("Simulated error: {0}")]
    SimulatedError(String),

    /// Engine not loaded
    #[error("Model not loaded")]
    ModelNotLoaded,

    /// Context exceeded
    #[error("Context length exceeded")]
    ContextExceeded,
}

/// Mock vector store for RAG testing
#[derive(Debug, Clone)]
pub struct MockVectorStore {
    documents: Arc<RwLock<HashMap<String, MockDocument>>>,
    query_count: Arc<RwLock<u64>>,
}

/// A mock document in the vector store
#[derive(Debug, Clone)]
pub struct MockDocument {
    /// Document ID
    pub id: String,
    /// Document content
    pub content: String,
    /// Similarity score (for controlling search results)
    pub score: f32,
}

impl MockVectorStore {
    /// Create a new empty mock vector store
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
            query_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Add a document to the store
    pub async fn add_document(&self, id: &str, content: &str, score: f32) {
        let doc = MockDocument {
            id: id.to_string(),
            content: content.to_string(),
            score,
        };
        let mut docs = self.documents.write().await;
        docs.insert(id.to_string(), doc);
    }

    /// Query the store (returns documents sorted by score)
    pub async fn query(&self, _query: &str, top_k: usize) -> Vec<MockDocument> {
        {
            let mut count = self.query_count.write().await;
            *count += 1;
        }

        let docs = self.documents.read().await;
        let mut results: Vec<_> = docs.values().cloned().collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Get query count
    pub async fn query_count(&self) -> u64 {
        *self.query_count.read().await
    }
}

impl Default for MockVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernum_core::SamplingParams;

    #[tokio::test]
    async fn test_mock_engine_default_response() {
        let engine = MockInferenceEngine::new();
        let request = GenerateRequest::new("Hello")
            .with_sampling(SamplingParams::default().with_max_tokens(100));

        let response = engine.generate(request).await.expect("should succeed");
        assert!(!response.choices.is_empty());
        assert!(!response.choices[0].text.is_empty());
        assert_eq!(engine.call_count().await, 1);
    }

    #[tokio::test]
    async fn test_mock_engine_failing() {
        let engine = MockInferenceEngine::failing("Test error");
        let request = GenerateRequest::new("Hello")
            .with_sampling(SamplingParams::default().with_max_tokens(100));

        let result = engine.generate(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_vector_store() {
        let store = MockVectorStore::new();
        store.add_document("doc1", "First document", 0.9).await;
        store.add_document("doc2", "Second document", 0.7).await;

        let results = store.query("test query", 2).await;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1"); // Highest score first
    }
}
