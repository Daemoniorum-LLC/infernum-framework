//! Parameterized coding evaluation - test any retention level
//! Usage: cargo run --release --example coding_eval_param -- --retention 80

use std::collections::HashMap;
use std::env;
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
    fn new(retention_pct: u32, device: &Device, dtype: DType) -> Result<Self> {
        let safetensors_path = Path::new("/home/crook/models/llama-3.2-1b/model.safetensors");
        let hct_dir_str = format!("/home/crook/models/llama-3.2-1b-hct-{}pct", retention_pct);
        let hct_dir = Path::new(&hct_dir_str);
        let tokenizer_path = Path::new("/home/crook/models/llama-3.2-1b/tokenizer.json");

        if !hct_dir.exists() {
            anyhow::bail!(
                "HCT directory not found: {}. Run compression first.",
                hct_dir_str
            );
        }

        println!("Loading tokenizer...");
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

        println!("Loading original model...");
        let orig_tensors = load_safetensors(safetensors_path, device)?;
        let orig_vb = VarBuilder::from_tensors(orig_tensors, dtype, device);
        let original = Llama::load(get_config(), orig_vb)?;

        println!(
            "Loading {}% compressed model from {}...",
            retention_pct, hct_dir_str
        );
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
        let mut tokens = self.encode(prompt)?;
        let prompt_len = tokens.len();

        let model = if use_compressed {
            &mut self.compressed
        } else {
            &mut self.original
        };
        model.clear_cache();

        // Prefill
        let input = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;
        let mut next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

        if next_token != 128001 && next_token != 128009 {
            tokens.push(next_token);

            // Autoregressive
            for _ in 1..max_tokens {
                let start_pos = tokens.len() - 1;
                let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
                let logits = model.forward(&input, start_pos)?;
                let last_logits = logits.i((0, 0, ..))?;
                next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

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
    orig_tokens: Vec<u32>,
    comp_tokens: Vec<u32>,
    token_match: f32,
    passed: bool,
}

fn main() -> Result<()> {
    // Parse --retention argument
    let args: Vec<String> = env::args().collect();
    let retention_pct: u32 = args
        .iter()
        .position(|a| a == "--retention")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);

    println!("=== Coding Evaluation: {}% Retention ===\n", retention_pct);

    let device = Device::Cpu;
    let dtype = DType::F32;

    let mut models = ModelPair::new(retention_pct, &device, dtype)?;
    let mut results: Vec<TestResult> = Vec::new();

    // All test prompts organized by level
    let tests = [
        // Level 1: Simple completion (threshold: 50%)
        (1, "Print hello", "```python\nprint(\"", 20, 0.50),
        (1, "For loop", "```python\nfor i in range(10):\n    ", 20, 0.50),
        (1, "Function def", "```python\ndef add(a, b):\n    return ", 20, 0.50),
        (1, "List comprehension", "```python\nsquares = [x**2 for x in ", 20, 0.50),
        (1, "Import statement", "```python\nimport ", 20, 0.50),
        // Level 2: Function implementation (threshold: 40%)
        (2, "Factorial", "```python\ndef factorial(n):\n    \"\"\"Return factorial of n.\"\"\"\n    ", 50, 0.40),
        (2, "Fibonacci", "```python\ndef fibonacci(n):\n    \"\"\"Return nth fibonacci number.\"\"\"\n    ", 50, 0.40),
        (2, "Is Prime", "```python\ndef is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n    ", 50, 0.40),
        (2, "Reverse String", "```python\ndef reverse_string(s):\n    \"\"\"Return reversed string.\"\"\"\n    return ", 50, 0.40),
        (2, "Find Max", "```python\ndef find_max(lst):\n    \"\"\"Return maximum element in list.\"\"\"\n    ", 50, 0.40),
        // Level 3: Bug fixing (threshold: 30%)
        (3, "Off by one", "# Bug: off-by-one error\ndef sum_range(n):\n    total = 0\n    for i in range(n):  # Should include n\n        total += i\n    return total\n\n# Fixed version:\ndef sum_range_fixed(n):\n    total = 0\n    for i in range(", 30, 0.30),
        (3, "Missing return", "# Bug: missing return statement\ndef double(x):\n    result = x * 2\n\n# Fixed version:\ndef double_fixed(x):\n    result = x * 2\n    ", 30, 0.30),
        (3, "Wrong operator", "# Bug: using + instead of *\ndef multiply(a, b):\n    return a + b\n\n# Fixed version:\ndef multiply_fixed(a, b):\n    return ", 30, 0.30),
        // Level 4: Algorithm (threshold: 25%)
        (4, "Binary Search", "```python\ndef binary_search(arr, target):\n    \"\"\"Return index of target in sorted array, or -1 if not found.\"\"\"\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = ", 60, 0.25),
        (4, "Merge Sort", "```python\ndef merge_sort(arr):\n    \"\"\"Sort array using merge sort.\"\"\"\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return ", 60, 0.25),
        (4, "BFS", "```python\nfrom collections import deque\n\ndef bfs(graph, start):\n    \"\"\"Breadth-first search from start node.\"\"\"\n    visited = set()\n    queue = deque([start])\n    while queue:\n        node = ", 60, 0.25),
        // Level 5: Complex (threshold: 20%)
        (5, "Class implementation", "```python\nclass Stack:\n    \"\"\"Stack data structure with push, pop, peek, is_empty.\"\"\"\n\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        ", 80, 0.20),
        (5, "Decorator", "```python\ndef memoize(func):\n    \"\"\"Decorator that caches function results.\"\"\"\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            ", 80, 0.20),
        (5, "Context manager", "```python\nclass FileManager:\n    \"\"\"Context manager for file operations.\"\"\"\n\n    def __init__(self, filename, mode):\n        self.filename = filename\n        self.mode = mode\n        self.file = None\n\n    def __enter__(self):\n        ", 80, 0.20),
    ];

    let mut level_results: HashMap<u32, Vec<(bool, f32)>> = HashMap::new();

    for (level, name, prompt, max_tokens, threshold) in tests.iter() {
        print!("Testing L{}: {}...", level, name);

        let (_, orig_tokens) = models.generate(prompt, *max_tokens, &device, false)?;
        let (_, comp_tokens) = models.generate(prompt, *max_tokens, &device, true)?;
        let match_rate = token_match_rate(&orig_tokens, &comp_tokens);
        let passed = match_rate >= *threshold;

        let status = if passed { "PASS" } else { "FAIL" };
        println!(" {} ({:.1}%)", status, match_rate * 100.0);

        results.push(TestResult {
            name: format!("L{}: {}", level, name),
            orig_tokens,
            comp_tokens,
            token_match: match_rate,
            passed,
        });

        level_results
            .entry(*level)
            .or_default()
            .push((passed, match_rate));
    }

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY - {}% RETENTION", retention_pct);
    println!("{}", "=".repeat(60));

    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let avg_match: f32 = results.iter().map(|r| r.token_match).sum::<f32>() / total as f32;

    println!("\nTests passed: {}/{}", passed, total);
    println!("Average token match rate: {:.1}%", avg_match * 100.0);

    for level in 1..=5 {
        if let Some(level_data) = level_results.get(&level) {
            let level_passed = level_data.iter().filter(|(p, _)| *p).count();
            let level_avg: f32 =
                level_data.iter().map(|(_, m)| m).sum::<f32>() / level_data.len() as f32;
            println!(
                "  Level {}: {}/{} passed, {:.1}% avg match",
                level,
                level_passed,
                level_data.len(),
                level_avg * 100.0
            );
        }
    }

    let pass_rate = passed as f32 / total as f32;
    println!("\n");
    if pass_rate >= 0.95 {
        println!(
            "*** EXCELLENT - {}% retention is VIABLE for coding! ***",
            retention_pct
        );
    } else if pass_rate >= 0.80 {
        println!(
            "*** GOOD - {}% retention is USABLE with occasional divergence ***",
            retention_pct
        );
    } else if pass_rate >= 0.50 {
        println!(
            "*** MARGINAL - {}% retention has significant divergence ***",
            retention_pct
        );
    } else {
        println!(
            "*** INSUFFICIENT - {}% retention NOT recommended for coding ***",
            retention_pct
        );
    }

    Ok(())
}
