//! Full model coding evaluation - test complete model compression
//! Usage: cargo run --release --example coding_eval_full -- --hct-dir /path/to/compressed

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

/// Decompress HCT file (zstd + fragments format)
fn decompress_hct(data: &[u8]) -> Result<Vec<f32>> {
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

    // Sort by index for proper reconstruction
    fragments.sort_by_key(|f| f.index);

    // Create decoder and add fragments
    let mut decoder = CompressiveSpectralDecoder::new();

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

/// Convert HCT filename to tensor name
fn filename_to_tensor_name(filename: &str) -> String {
    // Handle different tensor types:
    // model_embed_tokens_weight -> model.embed_tokens.weight
    // model_layers_0_self_attn_q_proj_weight -> model.layers.0.self_attn.q_proj.weight
    // model_layers_0_mlp_gate_proj_weight -> model.layers.0.mlp.gate_proj.weight
    // lm_head_weight -> lm_head.weight

    let result = filename
        // Preserve _proj patterns first
        .replace("_proj_", "_proj.")
        // Handle model prefix
        .replace("model_", "model.")
        // Handle layers
        .replace("layers_", "layers.")
        // Handle components
        .replace("_self_attn_", ".self_attn.")
        .replace("_mlp_", ".mlp.")
        .replace("_input_layernorm_", ".input_layernorm.")
        .replace("_post_attention_layernorm_", ".post_attention_layernorm.")
        // Handle embed_tokens
        .replace("embed_tokens_", "embed_tokens.")
        // Handle lm_head
        .replace("lm_head_", "lm_head.")
        // Handle norm
        .replace("_norm_", ".norm.")
        .replace("model.norm_", "model.norm.")
        // Handle weight suffix
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
        .replace("9_", "9.")
        // Handle numbers followed by letters (e.g., "10mlp" -> "10.mlp")
        .replace("10.", "10.")
        .replace("11.", "11.")
        .replace("12.", "12.")
        .replace("13.", "13.")
        .replace("14.", "14.")
        .replace("15.", "15.");

    result
}

/// Load fully compressed model from HCT files + safetensors for 1D tensors
fn load_full_compressed(
    hct_dir: &Path,
    safetensors_path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    // First load everything from safetensors (as fallback for 1D tensors)
    let mut tensors = load_safetensors(safetensors_path, device)?;

    // Now override with HCT decompressed versions
    let entries = fs::read_dir(hct_dir)?;
    let mut loaded = 0;
    let mut first = true;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "hct") {
            let filename = path.file_stem().unwrap().to_str().unwrap();
            let tensor_name = filename_to_tensor_name(filename);

            // Get original shape from safetensors
            if let Some(orig_tensor) = tensors.get(&tensor_name) {
                let shape: Vec<usize> = orig_tensor.dims().to_vec();
                let expected_size: usize = shape.iter().product();

                // Read and decompress HCT file
                let hct_data = fs::read(&path)?;
                match decompress_hct(&hct_data) {
                    Ok(decompressed) => {
                        if decompressed.len() != expected_size {
                            eprintln!(
                                "Size mismatch for {}: got {} expected {}",
                                tensor_name,
                                decompressed.len(),
                                expected_size
                            );
                            continue;
                        }

                        // Check for NaN/Inf
                        let nan_count = decompressed.iter().filter(|x| x.is_nan()).count();
                        let inf_count = decompressed.iter().filter(|x| x.is_infinite()).count();
                        if nan_count > 0 || inf_count > 0 {
                            eprintln!(
                                "Bad values in {}: nan={} inf={}",
                                tensor_name, nan_count, inf_count
                            );
                            continue;
                        }

                        // Debug: print first tensor info
                        if first {
                            let orig_vals: Vec<f32> = orig_tensor.flatten_all()?.to_vec1()?;
                            let diff: f32 = decompressed
                                .iter()
                                .zip(orig_vals.iter())
                                .map(|(a, b)| (a - b).abs())
                                .sum::<f32>()
                                / expected_size as f32;
                            eprintln!("First tensor {}: mean_abs_diff={:.6}", tensor_name, diff);
                            first = false;
                        }

                        let tensor = Tensor::from_vec(decompressed, shape.as_slice(), device)?;
                        let tensor = tensor.to_dtype(dtype)?;
                        tensors.insert(tensor_name, tensor);
                        loaded += 1;
                    },
                    Err(e) => {
                        eprintln!("Error decompressing {}: {}", tensor_name, e);
                    },
                }
            } else {
                // Try some common name variations
                let alt_name = tensor_name.replace("..", ".");
                if tensors.contains_key(&alt_name) {
                    eprintln!("Hint: {} might match {}", tensor_name, alt_name);
                }
            }
        }
    }

    println!("Loaded {} tensors from HCT", loaded);
    Ok(tensors)
}

struct ModelPair {
    original: Llama,
    compressed: Llama,
    tokenizer: tokenizers::Tokenizer,
}

impl ModelPair {
    fn new(hct_dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let safetensors_path = Path::new("/home/crook/models/llama-3.2-1b/model.safetensors");
        let tokenizer_path = Path::new("/home/crook/models/llama-3.2-1b/tokenizer.json");

        if !hct_dir.exists() {
            anyhow::bail!("HCT directory not found: {:?}", hct_dir);
        }

        println!("Loading tokenizer...");
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

        println!("Loading original model...");
        let orig_tensors = load_safetensors(safetensors_path, device)?;
        let orig_vb = VarBuilder::from_tensors(orig_tensors, dtype, device);
        let original = Llama::load(get_config(), orig_vb)?;

        println!("Loading compressed model from {:?}...", hct_dir);
        let comp_tensors = load_full_compressed(hct_dir, safetensors_path, device, dtype)?;
        let comp_vb = VarBuilder::from_tensors(comp_tensors, dtype, device);
        let compressed = Llama::load(get_config(), comp_vb)?;

        Ok(Self {
            original,
            compressed,
            tokenizer,
        })
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        use_compressed: bool,
    ) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();

        let device = Device::Cpu;
        let mut tokens = input_ids.clone();

        let model = if use_compressed {
            &mut self.compressed
        } else {
            &mut self.original
        };
        model.clear_cache();

        for _ in 0..max_tokens {
            let input = Tensor::new(&tokens[..], &device)?;
            let input = input.unsqueeze(0)?;

            let logits = model.forward(&input, tokens.len().saturating_sub(1))?;
            let logits = logits.i((.., logits.dim(1)? - 1, ..))?;

            let next_token = logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?[0];

            tokens.push(next_token);

            // Stop on EOS
            if next_token == 128001 || next_token == 128009 {
                break;
            }
        }

        Ok(tokens[input_ids.len()..].to_vec())
    }
}

#[derive(Debug)]
struct TestCase {
    name: &'static str,
    level: u8,
    prompt: &'static str,
    max_tokens: usize,
}

const TEST_CASES: &[TestCase] = &[
    // Level 1: Basic syntax
    TestCase { name: "Print hello", level: 1, prompt: "Complete the Python code:\n```python\nprint(\"Hello", max_tokens: 10 },
    TestCase { name: "For loop", level: 1, prompt: "Complete the Python code:\n```python\nfor i in range(5):\n    print(", max_tokens: 10 },
    TestCase { name: "Function def", level: 1, prompt: "Complete the Python code:\n```python\ndef greet(name):\n    return f\"Hello, {", max_tokens: 10 },
    TestCase { name: "List comprehension", level: 1, prompt: "Complete the Python code:\n```python\nsquares = [x**2 for x in", max_tokens: 10 },
    TestCase { name: "Import statement", level: 1, prompt: "Complete the Python code:\n```python\nimport json\ndata = json.loads(", max_tokens: 10 },

    // Level 2: Simple functions
    TestCase { name: "Factorial", level: 2, prompt: "Complete the Python function:\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n *", max_tokens: 20 },
    TestCase { name: "Fibonacci", level: 2, prompt: "Complete the Python function:\n```python\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) +", max_tokens: 15 },
    TestCase { name: "Is Prime", level: 2, prompt: "Complete the Python function:\n```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n %", max_tokens: 25 },
    TestCase { name: "Reverse String", level: 2, prompt: "Complete the Python function:\n```python\ndef reverse(s):\n    return s[", max_tokens: 10 },
    TestCase { name: "Find Max", level: 2, prompt: "Complete the Python function:\n```python\ndef find_max(arr):\n    return max(", max_tokens: 10 },

    // Level 3: Bug fixing
    TestCase { name: "Off by one", level: 3, prompt: "Fix the bug:\n```python\n# This should print 0,1,2,3,4 but has off-by-one error\nfor i in range(1, 5):\n    print(i)\n# Fixed:\nfor i in range(", max_tokens: 10 },
    TestCase { name: "Missing return", level: 3, prompt: "Fix the bug:\n```python\n# This function should return the sum\ndef add(a, b):\n    a + b\n# Fixed:\ndef add(a, b):\n    return", max_tokens: 10 },
    TestCase { name: "Wrong operator", level: 3, prompt: "Fix the bug:\n```python\n# This should multiply, not add\nresult = a + b\n# Fixed:\nresult = a *", max_tokens: 5 },

    // Level 4: Algorithms
    TestCase { name: "Binary Search", level: 4, prompt: "Complete the binary search:\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] <", max_tokens: 40 },
    TestCase { name: "Merge Sort", level: 4, prompt: "Complete merge sort:\n```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(", max_tokens: 30 },
    TestCase { name: "BFS", level: 4, prompt: "Complete BFS:\n```python\nfrom collections import deque\ndef bfs(graph, start):\n    visited = set()\n    queue = deque([start])\n    while queue:\n        node = queue.popleft()\n        if node not in visited:\n            visited.add(node)\n            for neighbor in", max_tokens: 40 },

    // Level 5: Complex patterns
    TestCase { name: "Class implementation", level: 5, prompt: "Complete the class:\n```python\nclass Stack:\n    def __init__(self):\n        self.items = []\n    \n    def push(self, item):\n        self.items.append(item)\n    \n    def pop(self):\n        return self.items.", max_tokens: 20 },
    TestCase { name: "Decorator", level: 5, prompt: "Complete the decorator:\n```python\ndef timer(func):\n    import time\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f\"Took {end - start:.2f}s\")\n        return", max_tokens: 20 },
    TestCase { name: "Context manager", level: 5, prompt: "Complete the context manager:\n```python\nclass FileManager:\n    def __init__(self, filename, mode):\n        self.filename = filename\n        self.mode = mode\n    \n    def __enter__(self):\n        self.file = open(self.filename, self.mode)\n        return self.file\n    \n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.file.", max_tokens: 20 },
];

#[derive(Debug)]
struct TestResult {
    name: String,
    orig_tokens: Vec<u32>,
    comp_tokens: Vec<u32>,
    match_rate: f32,
    passed: bool,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut hct_dir = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--hct-dir" | "-d" => {
                i += 1;
                hct_dir = args.get(i).cloned().unwrap_or_default();
            },
            "--help" | "-h" => {
                println!("Usage: coding_eval_full --hct-dir /path/to/compressed");
                return Ok(());
            },
            _ => {},
        }
        i += 1;
    }

    if hct_dir.is_empty() {
        eprintln!("Error: --hct-dir is required");
        std::process::exit(1);
    }

    let hct_path = Path::new(&hct_dir);
    let dir_name = hct_path.file_name().unwrap().to_str().unwrap();

    println!("=== Full Model Coding Evaluation: {} ===", dir_name);
    println!();

    let device = Device::Cpu;
    let dtype = DType::F32;

    let mut models = ModelPair::new(hct_path, &device, dtype)?;

    let mut results: Vec<TestResult> = Vec::new();

    for test in TEST_CASES {
        let orig_tokens = models.generate(test.prompt, test.max_tokens, false)?;
        let comp_tokens = models.generate(test.prompt, test.max_tokens, true)?;

        let min_len = orig_tokens.len().min(comp_tokens.len());
        let matches = orig_tokens
            .iter()
            .take(min_len)
            .zip(comp_tokens.iter().take(min_len))
            .filter(|(a, b)| a == b)
            .count();
        let match_rate = if min_len > 0 {
            matches as f32 / min_len as f32
        } else {
            0.0
        };
        let passed = match_rate >= 0.8;

        let result = TestResult {
            name: test.name.to_string(),
            orig_tokens: orig_tokens.clone(),
            comp_tokens: comp_tokens.clone(),
            match_rate,
            passed,
        };

        let status = if passed { "PASS" } else { "FAIL" };
        println!(
            "Testing L{}: {}... {} ({:.1}%)",
            test.level,
            test.name,
            status,
            match_rate * 100.0
        );

        if !passed {
            println!(
                "  Original tokens: {:?}",
                &orig_tokens[..orig_tokens.len().min(10)]
            );
            println!(
                "  Compressed tokens: {:?}",
                &comp_tokens[..comp_tokens.len().min(10)]
            );
        }

        results.push(result);
    }

    // Summary
    println!();
    println!("============================================================");
    println!("SUMMARY - {}", dir_name.to_uppercase());
    println!("============================================================");
    println!();

    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let avg_match: f32 = results.iter().map(|r| r.match_rate).sum::<f32>() / total as f32;

    println!("Tests passed: {}/{}", passed, total);
    println!("Average token match rate: {:.1}%", avg_match * 100.0);

    // Per-level summary
    for level in 1..=5 {
        let level_results: Vec<_> = results
            .iter()
            .enumerate()
            .filter(|(i, _)| TEST_CASES[*i].level == level)
            .map(|(_, r)| r)
            .collect();
        let level_passed = level_results.iter().filter(|r| r.passed).count();
        let level_avg: f32 =
            level_results.iter().map(|r| r.match_rate).sum::<f32>() / level_results.len() as f32;
        println!(
            "  Level {}: {}/{} passed, {:.1}% avg match",
            level,
            level_passed,
            level_results.len(),
            level_avg * 100.0
        );
    }

    println!();
    if passed == total {
        println!("*** EXCELLENT - All tests passed! ***");
    } else if avg_match >= 0.95 {
        println!("*** VIABLE - {} can be used for coding ***", dir_name);
    } else if avg_match >= 0.80 {
        println!("*** MARGINAL - {} has some divergence ***", dir_name);
    } else {
        println!(
            "*** INSUFFICIENT - {} NOT recommended for coding ***",
            dir_name
        );
    }

    Ok(())
}
