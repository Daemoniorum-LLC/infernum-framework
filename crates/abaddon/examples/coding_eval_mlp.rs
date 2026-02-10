//! MLP-only coding evaluation - test selective compression
//! Usage: cargo run --release --example coding_eval_mlp -- --mlp-retention 45

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;

use abaddon::models::{Llama, LlamaConfig};
use anyhow::Result;
use haagenti::compressive::CompressiveSpectralDecoder;
use haagenti::holotensor::HoloFragment;

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

/// Decompress MLP-only HCT file (zstd + fragments format)
fn decompress_mlp_hct(data: &[u8], _width: usize, _height: usize) -> Result<Vec<f32>> {
    // Zstd decompress using reference implementation
    let decompressed = zstd::decode_all(std::io::Cursor::new(data))
        .map_err(|e| anyhow::anyhow!("Zstd decompress error: {}", e))?;

    // Parse fragments
    if decompressed.len() < 2 {
        anyhow::bail!("Data too short for fragment count");
    }

    let frag_count = u16::from_le_bytes([decompressed[0], decompressed[1]]) as usize;
    let mut offset = 2usize;
    let mut fragments = Vec::with_capacity(frag_count);

    // Debug: first time, print fragment info
    static PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let print_debug = !PRINTED.swap(true, std::sync::atomic::Ordering::Relaxed);

    for _ in 0..frag_count {
        if offset + 16 > decompressed.len() {
            anyhow::bail!("Data too short for fragment header");
        }

        let index = u16::from_le_bytes([decompressed[offset], decompressed[offset + 1]]);
        let flags = u16::from_le_bytes([decompressed[offset + 2], decompressed[offset + 3]]);
        let checksum = u64::from_le_bytes([
            decompressed[offset + 4],
            decompressed[offset + 5],
            decompressed[offset + 6],
            decompressed[offset + 7],
            decompressed[offset + 8],
            decompressed[offset + 9],
            decompressed[offset + 10],
            decompressed[offset + 11],
        ]);
        let data_len = u32::from_le_bytes([
            decompressed[offset + 12],
            decompressed[offset + 13],
            decompressed[offset + 14],
            decompressed[offset + 15],
        ]) as usize;
        offset += 16;

        if offset + data_len > decompressed.len() {
            anyhow::bail!("Data too short for fragment data");
        }

        let frag_data = decompressed[offset..offset + data_len].to_vec();
        offset += data_len;

        fragments.push(HoloFragment {
            index,
            flags,
            checksum,
            data: frag_data,
        });
    }

    // Decode with compressive decoder - add essentials first, then details
    let mut decoder = CompressiveSpectralDecoder::new();

    // Sort by index to ensure fragment 0 is first
    fragments.sort_by_key(|f| f.index);

    if print_debug {
        eprintln!(
            "Fragments: count={}, indices={:?}",
            fragments.len(),
            fragments.iter().map(|f| f.index).collect::<Vec<_>>()
        );
    }

    for fragment in &fragments {
        if fragment.index == 0 {
            decoder
                .add_essentials(fragment)
                .map_err(|e| anyhow::anyhow!("Add essentials error: {}", e))?;
        } else {
            decoder
                .add_detail(fragment)
                .map_err(|e| anyhow::anyhow!("Add detail error: {}", e))?;
        }
    }

    let result = decoder
        .reconstruct()
        .map_err(|e| anyhow::anyhow!("Reconstruct error: {}", e))?;

    Ok(result)
}

/// Load MLP-only compressed model:
/// - MLP tensors from HCT files (using underscore naming convention)
/// - Everything else from safetensors
fn load_mlp_hybrid(
    mlp_dir: &Path,
    safetensors_path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    // First load everything from safetensors
    let mut tensors = load_safetensors(safetensors_path, device)?;

    // Now override MLP tensors with HCT decompressed versions
    let entries = fs::read_dir(mlp_dir)?;
    let mut loaded = 0;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "hct") {
            // Convert filename back to tensor name
            // model_layers_0_mlp_gate_proj_weight.hct -> model.layers.0.mlp.gate_proj.weight
            let filename = path.file_stem().unwrap().to_str().unwrap();
            // Replace underscores with dots, but preserve _proj patterns
            let tensor_name = filename
                .replace("_proj_", "_proj.")
                .replace("model_", "model.")
                .replace("layers_", "layers.")
                .replace("_mlp_", ".mlp.")
                .replace("_weight", ".weight")
                // Handle layer numbers
                .replace("0_", "0.")
                .replace("1_", "1.")
                .replace("2_", "2.")
                .replace("3_", "3.")
                .replace("4_", "4.")
                .replace("5_", "5.")
                .replace("6_", "6.")
                .replace("7_", "7.")
                .replace("8_", "8.")
                .replace("9_", "9.");

            // Get original shape from safetensors
            if let Some(orig_tensor) = tensors.get(&tensor_name) {
                let shape: Vec<usize> = orig_tensor.dims().to_vec();
                let (width, height) = if shape.len() == 2 {
                    (shape[1], shape[0])
                } else {
                    let total: usize = shape.iter().product();
                    let w = shape.last().copied().unwrap_or(1);
                    (w, total / w)
                };

                // Read and decompress HCT file
                let hct_data = fs::read(&path)?;
                let decompressed = decompress_mlp_hct(&hct_data, width, height)?;

                // Debug: check size and values
                let expected_size: usize = shape.iter().product();
                if decompressed.len() != expected_size {
                    eprintln!(
                        "Size mismatch for {}: got {} expected {}",
                        tensor_name,
                        decompressed.len(),
                        expected_size
                    );
                    continue;
                }

                // Check for NaN/Inf and compare with original
                let nan_count = decompressed.iter().filter(|x| x.is_nan()).count();
                let inf_count = decompressed.iter().filter(|x| x.is_infinite()).count();
                if nan_count > 0 || inf_count > 0 {
                    eprintln!(
                        "Bad values in {}: nan={} inf={}",
                        tensor_name, nan_count, inf_count
                    );
                }

                // Compare with original values
                let orig_vals: Vec<f32> = orig_tensor.flatten_all()?.to_vec1()?;
                let diff: f32 = decompressed
                    .iter()
                    .zip(orig_vals.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / expected_size as f32;
                let orig_mean = orig_vals.iter().sum::<f32>() / expected_size as f32;
                let decomp_mean = decompressed.iter().sum::<f32>() / expected_size as f32;
                // Check first 10 values
                if loaded == 0 {
                    eprintln!(
                        "First tensor {}: orig_mean={:.6}, decomp_mean={:.6}, mean_abs_diff={:.6}",
                        tensor_name, orig_mean, decomp_mean, diff
                    );
                    eprintln!("  First 5 orig:  {:?}", &orig_vals[..5]);
                    eprintln!("  First 5 decomp: {:?}", &decompressed[..5]);
                    // Check if maybe they're transposed - compare shapes
                    eprintln!("  Shape: {:?}, width={}, height={}", shape, width, height);
                }

                let tensor = Tensor::from_vec(decompressed, shape.as_slice(), device)?;
                let tensor = tensor.to_dtype(dtype)?;
                tensors.insert(tensor_name, tensor);
                loaded += 1;
            } else {
                eprintln!("Warning: No shape info for {}", tensor_name);
            }
        }
    }

    println!("Loaded {} MLP tensors from HCT", loaded);
    Ok(tensors)
}

struct ModelPair {
    original: Llama,
    compressed: Llama,
    tokenizer: tokenizers::Tokenizer,
}

impl ModelPair {
    fn new(mlp_retention_pct: u32, device: &Device, dtype: DType) -> Result<Self> {
        let safetensors_path = Path::new("/home/crook/models/llama-3.2-1b/model.safetensors");
        let mlp_dir_str = format!(
            "/home/crook/models/llama-3.2-1b-mlp-{}pct",
            mlp_retention_pct
        );
        let mlp_dir = Path::new(&mlp_dir_str);
        let tokenizer_path = Path::new("/home/crook/models/llama-3.2-1b/tokenizer.json");

        if !mlp_dir.exists() {
            anyhow::bail!(
                "MLP directory not found: {}. Run compress_selective first.",
                mlp_dir_str
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
            "Loading MLP-{}% compressed model from {}...",
            mlp_retention_pct, mlp_dir_str
        );
        let comp_tensors = load_mlp_hybrid(mlp_dir, safetensors_path, device, dtype)?;
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
        let tokens = self.encode(prompt)?;
        let prompt_len = tokens.len();

        let model = if use_compressed {
            &mut self.compressed
        } else {
            &mut self.original
        };
        model.clear_cache();

        let mut all_tokens = tokens.clone();

        // Prefill
        let input = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;
        let mut next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

        if next_token != 128001 && next_token != 128009 {
            all_tokens.push(next_token);

            // Autoregressive
            for _ in 1..max_tokens {
                let start_pos = all_tokens.len() - 1;
                let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
                let logits = model.forward(&input, start_pos)?;
                let last_logits = logits.i((0, 0, ..))?;
                next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

                if next_token == 128001 || next_token == 128009 {
                    break;
                }
                all_tokens.push(next_token);
            }
        }

        let generated_tokens = all_tokens[prompt_len..].to_vec();
        let output = self.decode(&all_tokens)?;
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
    // Parse --mlp-retention argument
    let args: Vec<String> = env::args().collect();
    let mlp_retention_pct: u32 = args
        .iter()
        .position(|a| a == "--mlp-retention")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(45);

    println!(
        "=== MLP-Only Coding Evaluation: {}% MLP Retention ===",
        mlp_retention_pct
    );
    println!("(Attention at 100%, only MLP compressed)\n");

    let device = Device::Cpu;
    let dtype = DType::F32;

    let mut models = ModelPair::new(mlp_retention_pct, &device, dtype)?;
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

        // Debug: show first test's actual tokens
        if *level == 1 && *name == "Print hello" {
            println!(
                "  Original tokens: {:?}",
                &orig_tokens[..orig_tokens.len().min(10)]
            );
            println!(
                "  Compressed tokens: {:?}",
                &comp_tokens[..comp_tokens.len().min(10)]
            );
        }

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
    println!("SUMMARY - MLP-ONLY {}% RETENTION", mlp_retention_pct);
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
            "*** EXCELLENT - MLP-{}% is VIABLE for coding! ***",
            mlp_retention_pct
        );
    } else if pass_rate >= 0.80 {
        println!(
            "*** GOOD - MLP-{}% is USABLE with occasional divergence ***",
            mlp_retention_pct
        );
    } else if pass_rate >= 0.50 {
        println!(
            "*** MARGINAL - MLP-{}% has significant divergence ***",
            mlp_retention_pct
        );
    } else {
        println!(
            "*** INSUFFICIENT - MLP-{}% NOT recommended for coding ***",
            mlp_retention_pct
        );
    }

    Ok(())
}
