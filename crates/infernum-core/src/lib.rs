//! # Infernum Core
//!
//! Core types and traits for the Infernum ecosystem.
//!
//! This crate provides the foundational abstractions used across all Infernum components:
//! - Common error types
//! - Request/response structures
//! - Model metadata and configuration
//! - Shared traits for inference, embedding, and streaming

#![warn(missing_docs)]

pub mod edge;
pub mod error;
pub mod model;
pub mod perf;
pub mod request;
pub mod response;
pub mod sampling;
pub mod streaming;
pub mod types;

pub use edge::{
    CacheEntry, CacheStats, EdgeConfig, EdgeConfigError, EdgeTarget, LightweightContext,
    MemoryUsage, ModelCache, QuantizationLevel,
};
pub use error::{Error, Result};
pub use model::{ModelArchitecture, ModelMetadata, ModelSource};
pub use perf::{MemoryTracker, ObjectPool, PoolStats, StringPool};
pub use request::{EmbedRequest, GenerateRequest, PromptInput};
pub use response::{EmbedResponse, GenerateResponse, TokenInfo};
pub use sampling::{GrammarConstraint, SamplingParams};
pub use streaming::TokenStream;
pub use types::*; // Includes ModelFamily, ModelId, Role, Message, ToolCall, ToolDefinition, etc.

#[cfg(test)]
mod tests {
    use super::*;

    // === Error Type Tests ===

    #[test]
    fn test_error_model_not_found() {
        let err = Error::ModelNotFound {
            model_id: "test-model".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("test-model"));
    }

    #[test]
    fn test_error_unsupported_architecture() {
        let err = Error::UnsupportedArchitecture {
            architecture: "unknown-arch".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("unknown-arch"));
    }

    #[test]
    fn test_result_ok() {
        let result: Result<i32> = Ok(42);
        assert_eq!(result.expect("ok"), 42);
    }

    #[test]
    fn test_result_err() {
        let result: Result<i32> = Err(Error::ModelNotFound {
            model_id: "missing".to_string(),
        });
        assert!(result.is_err());
    }

    // === Request Type Tests ===

    #[test]
    fn test_generate_request_new() {
        let request = GenerateRequest::new("Hello, world!");
        let debug = format!("{:?}", request);
        assert!(!debug.is_empty());
    }

    #[test]
    fn test_generate_request_with_model() {
        let request = GenerateRequest::new("test").with_model(ModelId::from("llama-3b"));
        assert!(request.model.is_some());
    }

    #[test]
    fn test_generate_request_with_sampling() {
        let params = SamplingParams::default().with_max_tokens(100);
        let request = GenerateRequest::new("test").with_sampling(params);
        assert_eq!(request.sampling.max_tokens, 100);
    }

    #[test]
    fn test_embed_request_new() {
        let request = EmbedRequest::new("test embedding");
        let debug = format!("{:?}", request);
        assert!(!debug.is_empty());
    }

    // === Sampling Params Tests ===

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.max_tokens, 256);
    }

    #[test]
    fn test_sampling_params_greedy() {
        let params = SamplingParams::greedy();
        assert_eq!(params.temperature, 0.0);
    }

    #[test]
    fn test_sampling_params_balanced() {
        let params = SamplingParams::balanced();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
    }

    #[test]
    fn test_sampling_params_creative() {
        let params = SamplingParams::creative();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_k, 50);
    }

    #[test]
    fn test_sampling_params_with_temperature() {
        let params = SamplingParams::default().with_temperature(0.7);
        assert_eq!(params.temperature, 0.7);
    }

    #[test]
    fn test_sampling_params_with_max_tokens() {
        let params = SamplingParams::default().with_max_tokens(512);
        assert_eq!(params.max_tokens, 512);
    }

    #[test]
    fn test_sampling_params_builder_chain() {
        let params = SamplingParams::default()
            .with_temperature(0.8)
            .with_top_p(0.95)
            .with_max_tokens(1024);

        assert_eq!(params.temperature, 0.8);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.max_tokens, 1024);
    }

    // === Edge Module Tests ===

    #[test]
    fn test_edge_target_wasm() {
        let target = EdgeTarget::Wasm;
        assert!(target.max_model_size() > 0);
        assert!(!target.has_gpu());
    }

    #[test]
    fn test_edge_target_ios() {
        let target = EdgeTarget::Ios;
        assert!(target.has_gpu());
        assert!(target.max_context_length() > 0);
    }

    #[test]
    fn test_quantization_level_none() {
        let level = QuantizationLevel::None;
        assert_eq!(level.compression_ratio(), 1.0);
        assert_eq!(level.quality_factor(), 1.0);
    }

    #[test]
    fn test_quantization_level_q4_k_m() {
        let level = QuantizationLevel::Q4_K_M;
        assert!(level.compression_ratio() > 1.0);
        assert!(level.quality_factor() < 1.0);
    }

    #[test]
    fn test_edge_config_for_target() {
        let config = EdgeConfig::for_target(EdgeTarget::Wasm);
        assert_eq!(config.target, EdgeTarget::Wasm);
    }

    #[test]
    fn test_edge_config_validate() {
        let config = EdgeConfig::for_target(EdgeTarget::Ios);
        assert!(config.validate().is_ok());
    }

    // === Types Module Tests ===

    #[test]
    fn test_model_id_from_string() {
        let id = ModelId::from("test-model");
        assert_eq!(id.0, "test-model");
    }

    #[test]
    fn test_request_id_new() {
        let id = RequestId::new();
        // UUID format check
        let id_str = id.0.to_string();
        assert!(id_str.len() >= 32);
    }

    #[test]
    fn test_request_id_unique() {
        let id1 = RequestId::new();
        let id2 = RequestId::new();
        assert_ne!(id1.0, id2.0);
    }

    #[test]
    fn test_usage_struct() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn test_role_variants() {
        assert!(matches!(Role::User, Role::User));
        assert!(matches!(Role::Assistant, Role::Assistant));
        assert!(matches!(Role::System, Role::System));
        assert!(matches!(Role::Tool, Role::Tool));
    }

    #[test]
    fn test_finish_reason_variants() {
        assert!(matches!(FinishReason::Stop, FinishReason::Stop));
        assert!(matches!(FinishReason::Length, FinishReason::Length));
    }

    #[test]
    fn test_message_construction() {
        let msg = Message {
            role: Role::User,
            content: "Hello!".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };
        assert_eq!(msg.content, "Hello!");
        assert!(matches!(msg.role, Role::User));
    }

    #[test]
    fn test_message_with_name() {
        let msg = Message {
            role: Role::Assistant,
            content: "Response".to_string(),
            name: Some("assistant-1".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        assert_eq!(msg.name, Some("assistant-1".to_string()));
    }

    // === Response Module Tests ===

    #[test]
    fn test_token_info_construction() {
        let info = TokenInfo {
            id: 123,
            text: "hello".to_string(),
            logprob: Some(-0.5),
            special: false,
        };
        assert_eq!(info.id, 123);
        assert_eq!(info.text, "hello");
    }

    // === Performance Module Tests ===

    #[test]
    fn test_memory_tracker_new() {
        let tracker = MemoryTracker::new();
        // Just verify it creates successfully
        let debug = format!("{:?}", tracker);
        assert!(!debug.is_empty());
    }

    #[test]
    fn test_string_pool_acquire_release() {
        let pool = StringPool::new(16, 4096);
        let mut buffer = pool.acquire();
        buffer.push_str("hello");
        pool.release(buffer);

        let stats = pool.stats();
        assert!(stats.hits + stats.misses > 0);
    }

    #[test]
    fn test_pool_stats_default() {
        let stats = PoolStats {
            pool_size: 0,
            hits: 0,
            misses: 0,
        };
        assert_eq!(stats.pool_size, 0);
    }

    // === Prompt Input Tests ===

    #[test]
    fn test_prompt_input_text() {
        let input = PromptInput::Text("Hello, world!".to_string());
        assert!(matches!(input, PromptInput::Text(_)));
    }

    #[test]
    fn test_prompt_input_messages() {
        let messages = vec![Message {
            role: Role::User,
            content: "Hi".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let input = PromptInput::Messages(messages);
        assert!(matches!(input, PromptInput::Messages(_)));
    }

    // === Memory Usage and Cache Stats Tests ===

    #[test]
    fn test_memory_usage_struct() {
        let usage = MemoryUsage {
            used: 1024,
            available: 2048,
            total: 4096,
            utilization: 0.25,
        };
        assert_eq!(usage.used, 1024);
        assert_eq!(usage.total, 4096);
    }

    #[test]
    fn test_cache_stats_struct() {
        let stats = CacheStats {
            entry_count: 10,
            total_size: 1024,
            max_size: 2048,
            utilization: 0.5,
        };
        assert_eq!(stats.entry_count, 10);
        assert_eq!(stats.utilization, 0.5);
    }

    // === Serialization Tests ===

    #[test]
    fn test_role_serialization() {
        let role = Role::User;
        let json = serde_json::to_string(&role).expect("serialize");
        let parsed: Role = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(parsed, Role::User));
    }

    #[test]
    fn test_finish_reason_serialization() {
        let reason = FinishReason::Stop;
        let json = serde_json::to_string(&reason).expect("serialize");
        let parsed: FinishReason = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(parsed, FinishReason::Stop));
    }

    #[test]
    fn test_edge_target_serialization() {
        let target = EdgeTarget::Wasm;
        let json = serde_json::to_string(&target).expect("serialize");
        let parsed: EdgeTarget = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(parsed, EdgeTarget::Wasm));
    }

    #[test]
    fn test_quantization_level_serialization() {
        let level = QuantizationLevel::Q4_K_M;
        let json = serde_json::to_string(&level).expect("serialize");
        let parsed: QuantizationLevel = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(parsed, QuantizationLevel::Q4_K_M));
    }

    #[test]
    fn test_sampling_params_serialization() {
        let params = SamplingParams::balanced();
        let json = serde_json::to_string(&params).expect("serialize");
        let parsed: SamplingParams = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.temperature, 0.7);
    }
}
