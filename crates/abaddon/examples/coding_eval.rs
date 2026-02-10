//! Comprehensive coding evaluation: Original vs 70% HCT compressed
//! Tests progressively harder coding tasks

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;

use abaddon::hct_sequential::load_hct_directory_sequential;
use abaddon::models::{Llama, LlamaConfig};
use anyhow::Result;

fn get_config() -> LlamaConfig {
    LlamaConfig {
        hidden_size: 2048,
        intermediate_size: 8192,
        vocab_size: 128256,
        num_hidden_layers: 16,
        num_attention_heads: 32,
        num_key_value_heads: Some(8),
        rms_norm_eps: 1e-5,
        rope_theta: 500000.0,
        max_position_embeddings: 131072,
        tie_word_embeddings: true,
        bos_token_id: Some(128000),
        eos_token_id: Some(128001),
        rope_scaling: None,
    }
}

fn load_safetensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let file_content = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&file_content)?;
    let mut tensors = HashMap::new();
    for name in st.names() {
        let st_tensor = st.tensor(name)?;
        let shape: Vec<usize> = st_tensor.shape().to_vec();
        let data = st_tensor.data();
        let tensor = match st_tensor.dtype() {
            safetensors::Dtype::BF16 => {
                let halfs: Vec<half::bf16> = data
                    .chunks_exact(2)
                    .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                Tensor::from_vec(floats, shape.as_slice(), device)?
            },
            safetensors::Dtype::F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(floats, shape.as_slice(), device)?
            },
            safetensors::Dtype::F16 => {
                let halfs: Vec<half::f16> = data
                    .chunks_exact(2)
                    .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                Tensor::from_vec(floats, shape.as_slice(), device)?
            },
            _ => continue,
        };
        tensors.insert(name.to_string(), tensor);
    }
    Ok(tensors)
}

fn load_hybrid(
    hct_dir: &Path,
    safetensors_path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = load_hct_directory_sequential(hct_dir, device, dtype)?;
    let file_content = std::fs::read(safetensors_path)?;
    let st = SafeTensors::deserialize(&file_content)?;
    for name in st.names() {
        let tensor_name = name.to_string();
        if !tensors.contains_key(&tensor_name) {
            let st_tensor = st.tensor(&tensor_name)?;
            let shape: Vec<usize> = st_tensor.shape().to_vec();
            let data = st_tensor.data();
            let tensor = match st_tensor.dtype() {
                safetensors::Dtype::BF16 => {
                    let halfs: Vec<half::bf16> = data
                        .chunks_exact(2)
                        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect();
                    let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                    Tensor::from_vec(floats, shape.as_slice(), device)?
                },
                safetensors::Dtype::F32 => {
                    let floats: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();
                    Tensor::from_vec(floats, shape.as_slice(), device)?
                },
                safetensors::Dtype::F16 => {
                    let halfs: Vec<half::f16> = data
                        .chunks_exact(2)
                        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect();
                    let floats: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();
                    Tensor::from_vec(floats, shape.as_slice(), device)?
                },
                _ => continue,
            };
            tensors.insert(tensor_name, tensor);
        }
    }
    Ok(tensors)
}

struct ModelPair {
    original: Llama,
    compressed: Llama,
    tokenizer: tokenizers::Tokenizer,
}

impl ModelPair {
    fn new(device: &Device, dtype: DType) -> Result<Self> {
        let safetensors_path = Path::new("/home/crook/models/llama-3.2-1b/model.safetensors");
        let hct_dir = Path::new("/home/crook/models/llama-3.2-1b-hct-70pct");
        let tokenizer_path = Path::new("/home/crook/models/llama-3.2-1b/tokenizer.json");

        println!("Loading tokenizer...");
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

        println!("Loading original model...");
        let orig_tensors = load_safetensors(safetensors_path, device)?;
        let orig_vb = VarBuilder::from_tensors(orig_tensors, dtype, device);
        let original = Llama::load(get_config(), orig_vb)?;

        println!("Loading 70% compressed model...");
        let comp_tensors = load_hybrid(hct_dir, safetensors_path, device, dtype)?;
        let comp_vb = VarBuilder::from_tensors(comp_tensors, dtype, device);
        let compressed = Llama::load(get_config(), comp_vb)?;

        Ok(Self {
            original,
            compressed,
            tokenizer,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        device: &Device,
        use_compressed: bool,
    ) -> Result<(String, Vec<u32>)> {
        // Encode first before borrowing model mutably
        let mut tokens = self.encode(prompt)?;
        let prompt_len = tokens.len();

        let model = if use_compressed {
            &mut self.compressed
        } else {
            &mut self.original
        };

        // Clear KV cache before each generation
        model.clear_cache();

        // Prefill: process entire prompt
        let input = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;
        let mut next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

        if next_token != 128001 && next_token != 128009 {
            tokens.push(next_token);

            // Autoregressive: generate one token at a time
            for _ in 1..max_tokens {
                let start_pos = tokens.len() - 1;
                let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
                let logits = model.forward(&input, start_pos)?;
                let last_logits = logits.i((0, 0, ..))?;

                // Greedy decoding
                next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

                // Stop on EOS
                if next_token == 128001 || next_token == 128009 {
                    break;
                }

                tokens.push(next_token);
            }
        }

        let generated_tokens = tokens[prompt_len..].to_vec();
        let output = self.decode(&tokens)?;

        Ok((output, generated_tokens))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn token_match_rate(orig: &[u32], comp: &[u32]) -> f32 {
    let min_len = orig.len().min(comp.len());
    if min_len == 0 {
        return 0.0;
    }
    let matches = orig.iter().zip(comp.iter()).filter(|(a, b)| a == b).count();
    matches as f32 / min_len as f32
}

struct TestResult {
    name: String,
    prompt_tokens: usize,
    orig_output: String,
    comp_output: String,
    orig_tokens: Vec<u32>,
    comp_tokens: Vec<u32>,
    token_match: f32,
    passed: bool,
}

impl TestResult {
    fn print(&self) {
        let status = if self.passed { "PASS" } else { "FAIL" };
        println!(
            "\n{} [{}] - {} prompt tokens",
            status, self.name, self.prompt_tokens
        );
        println!("  Token match rate: {:.1}%", self.token_match * 100.0);
        println!("  Original ({} tokens):", self.orig_tokens.len());
        println!("    {}", truncate(&self.orig_output, 200));
        println!("  Compressed ({} tokens):", self.comp_tokens.len());
        println!("    {}", truncate(&self.comp_output, 200));
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

fn main() -> Result<()> {
    println!("=== Coding Evaluation: Original vs 70% HCT ===\n");

    let device = Device::Cpu;
    let dtype = DType::F32;

    let mut models = ModelPair::new(&device, dtype)?;
    let mut results: Vec<TestResult> = Vec::new();

    // ============================================
    // LEVEL 1: Simple Code Completion
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("LEVEL 1: Simple Code Completion");
    println!("{}", "=".repeat(60));

    let level1_prompts = [
        ("Print hello", "```python\nprint(\""),
        ("For loop", "```python\nfor i in range(10):\n    "),
        ("Function def", "```python\ndef add(a, b):\n    return "),
        ("List comprehension", "```python\nsquares = [x**2 for x in "),
        ("Import statement", "```python\nimport "),
    ];

    for (name, prompt) in &level1_prompts {
        let (orig_out, orig_tokens) = models.generate(prompt, 20, &device, false)?;
        let (comp_out, comp_tokens) = models.generate(prompt, 20, &device, true)?;
        let match_rate = token_match_rate(&orig_tokens, &comp_tokens);

        results.push(TestResult {
            name: format!("L1: {}", name),
            prompt_tokens: models.encode(prompt)?.len(),
            orig_output: orig_out,
            comp_output: comp_out,
            orig_tokens,
            comp_tokens,
            token_match: match_rate,
            passed: match_rate >= 0.5,
        });
        results.last().unwrap().print();
    }

    // ============================================
    // LEVEL 2: Function Implementation
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("LEVEL 2: Function Implementation");
    println!("{}", "=".repeat(60));

    let level2_prompts = [
        ("Factorial", "```python\ndef factorial(n):\n    \"\"\"Return factorial of n.\"\"\"\n    "),
        ("Fibonacci", "```python\ndef fibonacci(n):\n    \"\"\"Return nth fibonacci number.\"\"\"\n    "),
        ("Is Prime", "```python\ndef is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n    "),
        ("Reverse String", "```python\ndef reverse_string(s):\n    \"\"\"Return reversed string.\"\"\"\n    return "),
        ("Find Max", "```python\ndef find_max(lst):\n    \"\"\"Return maximum element in list.\"\"\"\n    "),
    ];

    for (name, prompt) in &level2_prompts {
        let (orig_out, orig_tokens) = models.generate(prompt, 50, &device, false)?;
        let (comp_out, comp_tokens) = models.generate(prompt, 50, &device, true)?;
        let match_rate = token_match_rate(&orig_tokens, &comp_tokens);

        results.push(TestResult {
            name: format!("L2: {}", name),
            prompt_tokens: models.encode(prompt)?.len(),
            orig_output: orig_out,
            comp_output: comp_out,
            orig_tokens,
            comp_tokens,
            token_match: match_rate,
            passed: match_rate >= 0.4,
        });
        results.last().unwrap().print();
    }

    // ============================================
    // LEVEL 3: Bug Fixing
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("LEVEL 3: Bug Fixing");
    println!("{}", "=".repeat(60));

    let level3_prompts = [
        (
            "Off by one",
            r#"# Bug: off-by-one error
def sum_range(n):
    total = 0
    for i in range(n):  # Should include n
        total += i
    return total

# Fixed version:
def sum_range_fixed(n):
    total = 0
    for i in range("#,
        ),
        (
            "Missing return",
            r#"# Bug: missing return statement
def double(x):
    result = x * 2

# Fixed version:
def double_fixed(x):
    result = x * 2
    "#,
        ),
        (
            "Wrong operator",
            r#"# Bug: using + instead of *
def multiply(a, b):
    return a + b

# Fixed version:
def multiply_fixed(a, b):
    return "#,
        ),
    ];

    for (name, prompt) in &level3_prompts {
        let (orig_out, orig_tokens) = models.generate(prompt, 30, &device, false)?;
        let (comp_out, comp_tokens) = models.generate(prompt, 30, &device, true)?;
        let match_rate = token_match_rate(&orig_tokens, &comp_tokens);

        results.push(TestResult {
            name: format!("L3: {}", name),
            prompt_tokens: models.encode(prompt)?.len(),
            orig_output: orig_out,
            comp_output: comp_out,
            orig_tokens,
            comp_tokens,
            token_match: match_rate,
            passed: match_rate >= 0.3,
        });
        results.last().unwrap().print();
    }

    // ============================================
    // LEVEL 4: Algorithm Implementation
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("LEVEL 4: Algorithm Implementation");
    println!("{}", "=".repeat(60));

    let level4_prompts = [
        (
            "Binary Search",
            r#"```python
def binary_search(arr, target):
    """Return index of target in sorted array, or -1 if not found."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = "#,
        ),
        (
            "Merge Sort",
            r#"```python
def merge_sort(arr):
    """Sort array using merge sort."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return "#,
        ),
        (
            "BFS",
            r#"```python
from collections import deque

def bfs(graph, start):
    """Breadth-first search from start node."""
    visited = set()
    queue = deque([start])
    while queue:
        node = "#,
        ),
    ];

    for (name, prompt) in &level4_prompts {
        let (orig_out, orig_tokens) = models.generate(prompt, 60, &device, false)?;
        let (comp_out, comp_tokens) = models.generate(prompt, 60, &device, true)?;
        let match_rate = token_match_rate(&orig_tokens, &comp_tokens);

        results.push(TestResult {
            name: format!("L4: {}", name),
            prompt_tokens: models.encode(prompt)?.len(),
            orig_output: orig_out,
            comp_output: comp_out,
            orig_tokens,
            comp_tokens,
            token_match: match_rate,
            passed: match_rate >= 0.25,
        });
        results.last().unwrap().print();
    }

    // ============================================
    // LEVEL 5: Complex Multi-Step
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("LEVEL 5: Complex Multi-Step Reasoning");
    println!("{}", "=".repeat(60));

    let level5_prompts = [
        (
            "Class implementation",
            r#"```python
class Stack:
    """Stack data structure with push, pop, peek, is_empty."""

    def __init__(self):
        self.items = []

    def push(self, item):
        "#,
        ),
        (
            "Decorator",
            r#"```python
def memoize(func):
    """Decorator that caches function results."""
    cache = {}
    def wrapper(*args):
        if args not in cache:
            "#,
        ),
        (
            "Context manager",
            r#"```python
class FileManager:
    """Context manager for file operations."""

    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        "#,
        ),
    ];

    for (name, prompt) in &level5_prompts {
        let (orig_out, orig_tokens) = models.generate(prompt, 80, &device, false)?;
        let (comp_out, comp_tokens) = models.generate(prompt, 80, &device, true)?;
        let match_rate = token_match_rate(&orig_tokens, &comp_tokens);

        results.push(TestResult {
            name: format!("L5: {}", name),
            prompt_tokens: models.encode(prompt)?.len(),
            orig_output: orig_out,
            comp_output: comp_out,
            orig_tokens,
            comp_tokens,
            token_match: match_rate,
            passed: match_rate >= 0.2,
        });
        results.last().unwrap().print();
    }

    // ============================================
    // SUMMARY
    // ============================================
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));

    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let avg_match: f32 = results.iter().map(|r| r.token_match).sum::<f32>() / total as f32;

    println!("\nTests passed: {}/{}", passed, total);
    println!("Average token match rate: {:.1}%", avg_match * 100.0);

    // By level
    for level in 1..=5 {
        let level_results: Vec<_> = results
            .iter()
            .filter(|r| r.name.starts_with(&format!("L{}", level)))
            .collect();
        let level_passed = level_results.iter().filter(|r| r.passed).count();
        let level_avg: f32 =
            level_results.iter().map(|r| r.token_match).sum::<f32>() / level_results.len() as f32;
        println!(
            "  Level {}: {}/{} passed, {:.1}% avg match",
            level,
            level_passed,
            level_results.len(),
            level_avg * 100.0
        );
    }

    if passed == total {
        println!("\n*** ALL TESTS PASSED - 70% retention is viable for coding! ***");
    } else if passed as f32 / total as f32 >= 0.8 {
        println!("\n*** MOSTLY PASSED - 70% retention is usable with occasional divergence ***");
    } else if passed as f32 / total as f32 >= 0.5 {
        println!("\n*** MIXED RESULTS - Consider higher retention for complex tasks ***");
    } else {
        println!("\n*** INSUFFICIENT - 70% retention not recommended for coding ***");
    }

    Ok(())
}
