//! Debugging tests to isolate generation issues.

#[cfg(test)]
mod debug_tests {
    use abaddon::AbaddonEngine;
    use beleth::SamplingParams;

    #[test]
    #[ignore] // Run with: cargo test --package abaddon debug_tokenization -- --ignored
    fn debug_tokenization() {
        // Test that tokenization works correctly
        let engine = AbaddonEngine::new("TinyLlama/TinyLlama-1.1B-Chat-v1.0", None)
            .expect("Failed to create engine");

        let test_text = "Hello world";
        println!("\n=== TOKENIZATION TEST ===");
        println!("Input text: {:?}", test_text);

        // We need to access the tokenizer through the engine
        // For now, let's just test generation with max_tokens=1
        let params = SamplingParams {
            max_tokens: 1,
            temperature: 0.0, // Greedy sampling for determinism
            ..Default::default()
        };

        match engine.generate(test_text, &params) {
            Ok((tokens, text)) => {
                println!("Generated tokens: {:?}", tokens);
                println!("Generated text: {:?}", text);
                println!("Number of tokens: {}", tokens.len());

                assert!(!tokens.is_empty(), "Should generate at least one token");
                assert!(!text.is_empty(), "Should generate some text");

                // Check if it's the repetitive garbage pattern
                if text.len() == 1 && text[0].len() <= 5 {
                    println!("WARNING: Generated very short text: {:?}", text[0]);
                }
            }
            Err(e) => {
                panic!("Generation failed: {:?}", e);
            }
        }
    }

    #[test]
    #[ignore]
    fn debug_qwen2_tokenization() {
        let engine = AbaddonEngine::new("Qwen/Qwen2.5-1.5B-Instruct", None)
            .expect("Failed to create engine");

        let test_text = "Hello world";
        println!("\n=== QWEN2 TOKENIZATION TEST ===");
        println!("Input text: {:?}", test_text);

        let params = SamplingParams {
            max_tokens: 1,
            temperature: 0.0,
            ..Default::default()
        };

        match engine.generate(test_text, &params) {
            Ok((tokens, text)) => {
                println!("Generated tokens: {:?}", tokens);
                println!("Generated text: {:?}", text);
                println!("Number of tokens: {}", tokens.len());

                assert!(!tokens.is_empty(), "Should generate at least one token");

                // Check for the "uto" pattern
                if text.len() > 0 && text[0].contains("uto") {
                    println!("WARNING: Detected 'uto' repetition pattern!");
                }
            }
            Err(e) => {
                panic!("Generation failed: {:?}", e);
            }
        }
    }

    #[test]
    #[ignore]
    fn debug_multiple_tokens() {
        let engine = AbaddonEngine::new("TinyLlama/TinyLlama-1.1B-Chat-v1.0", None)
            .expect("Failed to create engine");

        let test_text = "The capital of France is";
        println!("\n=== MULTIPLE TOKEN GENERATION TEST ===");
        println!("Input text: {:?}", test_text);

        let params = SamplingParams {
            max_tokens: 5,
            temperature: 0.0, // Greedy for reproducibility
            ..Default::default()
        };

        match engine.generate(test_text, &params) {
            Ok((tokens, text)) => {
                println!("Generated {} tokens: {:?}", tokens.len(), tokens);
                println!("Generated text: {:?}", text);

                // Print each token individually
                for (i, (token_id, token_text)) in tokens.iter().zip(text.iter()).enumerate() {
                    println!("  Token {}: ID={}, Text={:?}", i, token_id, token_text);
                }

                // Check if all tokens are the same (repetition bug)
                if tokens.len() > 1 {
                    let all_same = tokens.windows(2).all(|w| w[0] == w[1]);
                    if all_same {
                        println!("ERROR: All generated tokens are identical! Token ID: {}", tokens[0]);
                    }
                }

                // Check if all text is the same
                if text.len() > 1 {
                    let all_same_text = text.windows(2).all(|w| w[0] == w[1]);
                    if all_same_text {
                        println!("ERROR: All generated text is identical! Text: {:?}", text[0]);
                    }
                }
            }
            Err(e) => {
                panic!("Generation failed: {:?}", e);
            }
        }
    }

    #[test]
    #[ignore]
    fn debug_qwen2_multiple_tokens() {
        let engine = AbaddonEngine::new("Qwen/Qwen2.5-1.5B-Instruct", None)
            .expect("Failed to create engine");

        let test_text = "The capital of France is";
        println!("\n=== QWEN2 MULTIPLE TOKEN TEST ===");
        println!("Input text: {:?}", test_text);

        let params = SamplingParams {
            max_tokens: 5,
            temperature: 0.0,
            ..Default::default()
        };

        match engine.generate(test_text, &params) {
            Ok((tokens, text)) => {
                println!("Generated {} tokens: {:?}", tokens.len(), tokens);
                println!("Generated text: {:?}", text);

                for (i, (token_id, token_text)) in tokens.iter().zip(text.iter()).enumerate() {
                    println!("  Token {}: ID={}, Text={:?}", i, token_id, token_text);
                }

                // Check for repetition
                if tokens.len() > 1 {
                    let all_same = tokens.windows(2).all(|w| w[0] == w[1]);
                    if all_same {
                        println!("ERROR: All tokens identical! Token ID: {}", tokens[0]);
                    }
                }
            }
            Err(e) => {
                panic!("Generation failed: {:?}", e);
            }
        }
    }
}
