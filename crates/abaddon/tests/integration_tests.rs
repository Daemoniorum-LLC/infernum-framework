//! Integration tests for Abaddon inference engine.

use abaddon::models::Architecture;

#[test]
fn test_architecture_detection_llama() {
    // Test Llama architecture detection
    let config_json = r#"{
        "model_type": "llama",
        "hidden_size": 2048,
        "num_hidden_layers": 22
    }"#;

    let temp_dir = std::env::temp_dir();
    let config_path = temp_dir.join("test_llama_config.json");
    std::fs::write(&config_path, config_json).unwrap();

    let arch = Architecture::detect_from_config(&config_path).unwrap();
    assert_eq!(arch, Architecture::Llama);
    assert_eq!(arch.name(), "Llama");
    assert!(arch.is_llama_compatible());
    assert!(arch.notes().is_none());

    std::fs::remove_file(&config_path).ok();
}

#[test]
fn test_architecture_detection_qwen2() {
    // Test Qwen2 architecture detection
    let config_json = r#"{
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 1536,
        "num_hidden_layers": 28
    }"#;

    let temp_dir = std::env::temp_dir();
    let config_path = temp_dir.join("test_qwen2_config.json");
    std::fs::write(&config_path, config_json).unwrap();

    let arch = Architecture::detect_from_config(&config_path).unwrap();
    assert_eq!(arch, Architecture::Qwen2);
    assert_eq!(arch.name(), "Qwen2");
    assert!(arch.is_llama_compatible());
    assert!(arch.notes().is_some());
    assert!(arch.notes().unwrap().contains("RoPE theta"));

    std::fs::remove_file(&config_path).ok();
}

#[test]
fn test_architecture_detection_mistral() {
    // Test Mistral architecture detection
    let config_json = r#"{
        "architectures": ["MistralForCausalLM"],
        "hidden_size": 4096
    }"#;

    let temp_dir = std::env::temp_dir();
    let config_path = temp_dir.join("test_mistral_config.json");
    std::fs::write(&config_path, config_json).unwrap();

    let arch = Architecture::detect_from_config(&config_path).unwrap();
    assert_eq!(arch, Architecture::Mistral);
    assert_eq!(arch.name(), "Mistral");
    assert!(arch.is_llama_compatible());

    std::fs::remove_file(&config_path).ok();
}

#[test]
fn test_architecture_detection_unknown() {
    // Test unknown architecture
    let config_json = r#"{
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"]
    }"#;

    let temp_dir = std::env::temp_dir();
    let config_path = temp_dir.join("test_unknown_config.json");
    std::fs::write(&config_path, config_json).unwrap();

    let arch = Architecture::detect_from_config(&config_path).unwrap();
    assert_eq!(arch, Architecture::Unknown);
    assert_eq!(arch.name(), "Unknown");
    assert!(!arch.is_llama_compatible());
    assert!(arch.notes().is_some());

    std::fs::remove_file(&config_path).ok();
}

#[cfg(feature = "integration")]
mod model_tests {
    use abaddon::AbaddonEngine;
    use beleth::SamplingParams;

    #[test]
    #[ignore] // Requires model download
    fn test_tinyllama_tokenization() {
        // Test that tokenizer works correctly
        let engine = AbaddonEngine::new("TinyLlama/TinyLlama-1.1B-Chat-v1.0", None)
            .expect("Failed to create engine");

        let test_text = "Hello world";

        // Generate with very short max tokens to test tokenizer
        let params = SamplingParams {
            max_tokens: 1,
            temperature: 0.0, // Greedy for deterministic output
            ..Default::default()
        };

        let result = engine.generate(test_text, &params);
        assert!(result.is_ok(), "Generation should not error");

        let (tokens, text) = result.unwrap();
        assert!(!tokens.is_empty(), "Should generate at least one token");
        assert!(!text.is_empty(), "Should generate some text");
    }

    #[test]
    #[ignore] // Requires model download
    fn test_qwen2_tokenization() {
        // Test that Qwen2 tokenizer works correctly
        let engine = AbaddonEngine::new("Qwen/Qwen2.5-1.5B-Instruct", None)
            .expect("Failed to create engine");

        let test_text = "Hello world";

        // Generate with very short max tokens to test tokenizer
        let params = SamplingParams {
            max_tokens: 1,
            temperature: 0.0, // Greedy for deterministic output
            ..Default::default()
        };

        let result = engine.generate(test_text, &params);
        assert!(result.is_ok(), "Generation should not error");

        let (tokens, text) = result.unwrap();
        assert!(!tokens.is_empty(), "Should generate at least one token");
        assert!(!text.is_empty(), "Should generate some text");
    }
}
