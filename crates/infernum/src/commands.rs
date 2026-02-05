//! CLI command implementations.

use std::io::{self, Write};
use std::sync::Arc;

use color_eyre::eyre::{eyre, Context, Result};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};

use abaddon::{Engine, EngineConfig, InferenceEngine};
use infernum_core::{GenerateRequest, Message, Role, SamplingParams};

/// Start the inference server.
#[cfg(feature = "server")]
#[allow(clippy::too_many_arguments)]
pub async fn serve(
    host: String,
    port: u16,
    model: Option<String>,
    _config: Option<String>,
    holo: bool,
    holo_min_quality: f32,
    holo_target_quality: f32,
) -> Result<()> {
    use infernum_server::{Server, ServerConfig};
    

    // Require a model - from args, env var, or prompt interactively
    let model = match model {
        Some(m) => m,
        None => {
            // Check env var before prompting
            if let Ok(env_model) = std::env::var("INFERNUM_MODEL") {
                if !env_model.is_empty() {
                    env_model
                } else {
                    prompt_for_model()?
                }
            } else {
                prompt_for_model()?
            }
        }
    };

    // Auto-detect HCT models and enable HoloTensor mode
    // INFERNUM_HCT_EAGER=1 disables lazy loading for HCT models that fit in VRAM
    let use_eager_hct = std::env::var("INFERNUM_HCT_EAGER")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    let holo = if use_eager_hct {
        holo // Don't auto-enable for HCT if eager mode requested
    } else {
        holo || is_hct_model(&model)
    };

    // Allow env vars to override quality settings
    let holo_min_quality = std::env::var("INFERNUM_HOLO_MIN_QUALITY")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(holo_min_quality);
    let holo_target_quality = std::env::var("INFERNUM_HOLO_TARGET_QUALITY")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(holo_target_quality);

    println!();
    if holo {
        println!("\x1b[1m🌀 Starting Infernum Server (HoloTensor Mode)\x1b[0m");
        println!("   Model: {} \x1b[36m[HCT]\x1b[0m", model);
        println!(
            "   Quality: {:.0}% → {:.0}%",
            holo_min_quality * 100.0,
            holo_target_quality * 100.0
        );
    } else {
        println!("\x1b[1m🚀 Starting Infernum Server\x1b[0m");
        println!("   Model: {}", model);
    }
    println!("   Address: http://{}:{}", host, port);
    println!();

    let addr = format!("{}:{}", host, port).parse()?;

    // Build model string with HoloTensor marker if enabled
    let model_source = if holo {
        // For HoloTensor, we pass the path with a special prefix that the server recognizes
        format!(
            "holo://{}?min={}&target={}",
            model, holo_min_quality, holo_target_quality
        )
    } else {
        model
    };

    // Get speculative decoding config from env vars
    let draft_model = std::env::var("INFERNUM_DRAFT_MODEL").ok().filter(|s| !s.is_empty());
    let speculative_tokens = std::env::var("INFERNUM_SPECULATIVE_TOKENS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(5);

    if draft_model.is_some() {
        tracing::info!(
            draft_model = ?draft_model,
            speculative_tokens = speculative_tokens,
            "Speculative decoding enabled"
        );
    }

    let config = ServerConfig {
        addr,
        cors: true,
        model: Some(model_source),
        draft_model,
        speculative_tokens,
        validation_limits: Default::default(),
        timeouts: Default::default(),
        queue: Default::default(),
    };

    let server = Server::new(config);
    server.run().await?;

    Ok(())
}

/// Check if a model path is an HCT (HoloTensor Compressed) directory.
/// Returns true if the path exists and contains .hct files.
fn is_hct_model(model: &str) -> bool {
    use std::path::Path;

    let path = Path::new(model);
    if !path.exists() || !path.is_dir() {
        return false;
    }

    // Check if directory contains .hct files
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "hct" {
                    tracing::info!("Auto-detected HCT model directory, enabling HoloTensor mode");
                    return true;
                }
            }
        }
    }

    false
}

/// Prompt user to select a model interactively.
fn prompt_for_model() -> Result<String> {
    use dialoguer::{theme::ColorfulTheme, Input, Select};
    use std::path::PathBuf;

    println!("\x1b[1m🤖 No model specified\x1b[0m\n");

    // Check for cached models
    let cache_dir = dirs::cache_dir()
        .map(|p| p.join("huggingface").join("hub"))
        .unwrap_or_else(|| PathBuf::from("~/.cache/huggingface/hub"));

    let mut cached_models: Vec<String> = Vec::new();
    if cache_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("models--") {
                    let model_name = name_str
                        .strip_prefix("models--")
                        .unwrap_or(&name_str)
                        .replace("--", "/");
                    cached_models.push(model_name);
                }
            }
        }
    }

    // Build options list
    let mut options: Vec<String> = Vec::new();

    // Add cached models first
    if !cached_models.is_empty() {
        for model in &cached_models {
            options.push(format!("{} (cached)", model));
        }
    }

    // Add suggested models
    let suggested = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ];

    for model in suggested {
        if !cached_models.contains(&model.to_string()) {
            options.push(format!("{} (will download)", model));
        }
    }

    options.push("Enter custom model ID...".to_string());

    // Display selection
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a model")
        .items(&options)
        .default(0)
        .interact()
        .map_err(|e| eyre!("Selection cancelled: {}", e))?;

    let selected = &options[selection];

    // Handle selection
    if selected.contains("Enter custom") {
        // Custom input
        let model: String = Input::with_theme(&ColorfulTheme::default())
            .with_prompt("Enter model ID (HuggingFace repo)")
            .interact_text()
            .map_err(|e| eyre!("Input cancelled: {}", e))?;
        Ok(model)
    } else {
        // Extract model name (remove status suffix)
        let model = selected.split(" (").next().unwrap_or(selected).to_string();
        Ok(model)
    }
}

/// Generate text from a prompt.
pub async fn generate(
    prompt: String,
    model: Option<String>,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
) -> Result<()> {
    let model_id = model.ok_or_else(|| {
        eyre!(
            "Model is required.\n\n\
         Options:\n  \
         1. Specify on command line: --model <model>\n  \
         2. Set a default: infernum config set-model <model>\n  \
         3. Set environment variable: INFERNUM_DEFAULT_MODEL=<model>\n\n\
         Example models:\n  \
         - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (small, fast)\n  \
         - meta-llama/Llama-3.2-3B-Instruct (requires HuggingFace login)"
        )
    })?;

    // Show loading indicator
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .context("invalid progress bar template")?,
    );
    spinner.set_message(format!("Loading model {}...", model_id));
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    // Create engine config
    let config = EngineConfig::builder()
        .model(&model_id)
        .build()
        .map_err(|e| eyre!("Failed to configure engine: {}", e))?;

    // Load the model
    let engine = Engine::new(config).await?;
    let engine = Arc::new(engine);

    spinner.finish_and_clear();
    println!("Model loaded: {}\n", engine.model_info().id);

    // Build sampling params
    let sampling = SamplingParams::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);

    // Create request
    let request = GenerateRequest::new(prompt.clone()).with_sampling(sampling);

    if stream {
        // Streaming generation
        print!("{}", prompt);
        io::stdout().flush()?;

        let token_stream = engine.generate_stream(request).await?;
        futures::pin_mut!(token_stream);

        while let Some(result) = token_stream.next().await {
            match result {
                Ok(chunk) => {
                    for choice in chunk.choices {
                        if let Some(content) = choice.delta.content {
                            print!("{}", content);
                            io::stdout().flush()?;
                        }
                    }
                },
                Err(e) => {
                    eprintln!("\nError during generation: {}", e);
                    break;
                },
            }
        }
        println!();
    } else {
        // Non-streaming generation
        let response = engine.generate(request).await?;

        for choice in response.choices {
            println!("{}", choice.text);
        }

        println!(
            "\n[Tokens: {} prompt, {} completion]",
            response.usage.prompt_tokens, response.usage.completion_tokens
        );
    }

    Ok(())
}

/// Generate embeddings.
pub async fn embed(text: String, model: Option<String>) -> Result<()> {
    let model_id = model.ok_or_else(|| eyre!("Model is required. Use --model <model>"))?;

    // Show loading indicator
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .context("invalid progress bar template")?,
    );
    spinner.set_message(format!("Loading embedding model {}...", model_id));
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    // Create engine config
    let config = EngineConfig::builder()
        .model(&model_id)
        .build()
        .map_err(|e| eyre!("Failed to configure engine: {}", e))?;

    // Load the model
    let engine = Engine::new(config).await?;

    spinner.finish_and_clear();
    println!("Embedding model loaded: {}\n", engine.model_info().id);

    // Create embed request
    let request = infernum_core::EmbedRequest::new(text.clone());

    // Generate embeddings
    let response = engine.embed(request).await?;

    println!("Text: \"{}\"", text);
    println!(
        "Dimensions: {}",
        response
            .data
            .first()
            .map(|e| {
                match &e.embedding {
                    infernum_core::response::EmbeddingData::Float(v) => v.len(),
                    infernum_core::response::EmbeddingData::Base64(_) => 0,
                }
            })
            .unwrap_or(0)
    );

    // Show first few dimensions
    if let Some(embedding) = response.data.first() {
        if let Ok(values) = embedding.embedding.as_floats() {
            let preview: Vec<_> = values.iter().take(5).collect();
            println!(
                "Embedding (first 5): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, ...]",
                preview.get(0).unwrap_or(&&0.0),
                preview.get(1).unwrap_or(&&0.0),
                preview.get(2).unwrap_or(&&0.0),
                preview.get(3).unwrap_or(&&0.0),
                preview.get(4).unwrap_or(&&0.0)
            );
        }
    }

    println!("\n[Tokens used: {}]", response.usage.total_tokens);

    Ok(())
}

/// List available models.
pub async fn model_list() -> Result<()> {
    use std::path::PathBuf;

    println!("\x1b[1m📦 Cached Models\x1b[0m\n");

    // Check HuggingFace cache directory
    let cache_dir = dirs::cache_dir()
        .map(|p| p.join("huggingface").join("hub"))
        .unwrap_or_else(|| PathBuf::from("~/.cache/huggingface/hub"));

    if cache_dir.exists() {
        let mut models: Vec<ModelCacheInfo> = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("models--") {
                    let model_name = name_str
                        .strip_prefix("models--")
                        .unwrap_or(&name_str)
                        .replace("--", "/");

                    let model_path = entry.path();
                    let info = get_model_cache_info(&model_name, &model_path);
                    models.push(info);
                }
            }
        }

        if models.is_empty() {
            println!("  \x1b[90m(No models cached yet)\x1b[0m");
        } else {
            // Sort by name
            models.sort_by(|a, b| a.name.cmp(&b.name));

            // Calculate column widths
            let max_name_len = models
                .iter()
                .map(|m| m.name.len())
                .max()
                .unwrap_or(30)
                .min(50);

            // Print header
            println!(
                "  {:<width$}  {:>10}  {:>12}  {}",
                "Model",
                "Size",
                "Context",
                "Architecture",
                width = max_name_len
            );
            println!(
                "  {:-<width$}  {:-<10}  {:-<12}  {:-<15}",
                "",
                "",
                "",
                "",
                width = max_name_len
            );

            // Print models
            for model in &models {
                let name_display = if model.name.len() > max_name_len {
                    format!("{}...", &model.name[..max_name_len - 3])
                } else {
                    model.name.clone()
                };

                println!(
                    "  {:<width$}  {:>10}  {:>12}  {}",
                    name_display,
                    model.size_str,
                    model.context_str,
                    model.architecture,
                    width = max_name_len
                );
            }

            // Summary
            let total_size: u64 = models.iter().map(|m| m.size_bytes).sum();
            println!();
            println!(
                "  \x1b[90m{} model(s), {} total\x1b[0m",
                models.len(),
                format_size(total_size)
            );
        }
    } else {
        println!("  \x1b[90m(No models cached yet)\x1b[0m");
    }

    println!();
    println!("\x1b[1mRecommended Models:\x1b[0m");
    println!("  TinyLlama/TinyLlama-1.1B-Chat-v1.0  - Fast, lightweight (~2GB)");
    println!("  meta-llama/Llama-3.2-1B-Instruct    - Balanced quality (~2GB)");
    println!("  meta-llama/Llama-3.2-3B-Instruct    - Higher quality (~6GB)");
    println!();
    println!("Use '\x1b[1minfernum model pull <model>\x1b[0m' to download a model.");

    Ok(())
}

/// Information about a cached model.
struct ModelCacheInfo {
    name: String,
    size_bytes: u64,
    size_str: String,
    context_str: String,
    architecture: String,
}

/// Get information about a cached model.
fn get_model_cache_info(name: &str, cache_path: &std::path::Path) -> ModelCacheInfo {
    // Calculate size
    let size_bytes = dir_size(cache_path).unwrap_or(0);
    let size_str = format_size(size_bytes);

    // Try to read config.json for more info
    let snapshots_dir = cache_path.join("snapshots");
    let (context_str, architecture) = if snapshots_dir.exists() {
        // Find first snapshot directory
        if let Ok(mut entries) = std::fs::read_dir(&snapshots_dir) {
            if let Some(Ok(snapshot)) = entries.next() {
                let config_path = snapshot.path().join("config.json");
                if config_path.exists() {
                    if let Ok(content) = std::fs::read_to_string(&config_path) {
                        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                            let ctx = config
                                .get("max_position_embeddings")
                                .and_then(|v| v.as_u64())
                                .map(|v| format!("{}K", v / 1024))
                                .unwrap_or_else(|| "-".to_string());

                            let arch = config
                                .get("architectures")
                                .and_then(|a| a.as_array())
                                .and_then(|a| a.first())
                                .and_then(|a| a.as_str())
                                .map(|s| {
                                    // Simplify architecture name
                                    s.replace("ForCausalLM", "").replace("Model", "")
                                })
                                .unwrap_or_else(|| "-".to_string());

                            return ModelCacheInfo {
                                name: name.to_string(),
                                size_bytes,
                                size_str,
                                context_str: ctx,
                                architecture: arch,
                            };
                        }
                    }
                }
            }
        }
        ("-".to_string(), "-".to_string())
    } else {
        ("-".to_string(), "-".to_string())
    };

    ModelCacheInfo {
        name: name.to_string(),
        size_bytes,
        size_str,
        context_str,
        architecture,
    }
}

/// Pull a model from HuggingFace.
pub async fn model_pull(model: String, revision: Option<String>) -> Result<()> {
    use hf_hub::api::sync::Api;

    println!("Downloading model: {}", model);
    if let Some(rev) = &revision {
        println!("Revision: {}", rev);
    }
    println!();

    let api = Api::new()?;
    let repo = if let Some(rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            model.clone(),
            hf_hub::RepoType::Model,
            rev,
        ))
    } else {
        api.model(model.clone())
    };

    // Required files - model won't work without these
    let required_files = ["config.json", "tokenizer.json"];

    // Optional files - may or may not exist depending on model
    let optional_files = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ];

    // Weight files - try single file first, then sharded
    #[allow(unused)]
    let weight_files = ["model.safetensors", "model.safetensors.index.json"];

    let mut downloaded: Vec<String> = Vec::new();
    let mut failed: Vec<String> = Vec::new();
    let mut has_weights = false;

    // Download required files
    println!("Downloading required files...");
    for file in required_files {
        print!("  {} ... ", file);
        io::stdout().flush()?;
        match repo.get(file) {
            Ok(_) => {
                println!("\x1b[32m✓\x1b[0m");
                downloaded.push(file.to_string());
            },
            Err(e) => {
                println!("\x1b[31m✗\x1b[0m ({})", e);
                failed.push(file.to_string());
            },
        }
    }

    // Download optional files (silently skip if not found)
    println!("\nDownloading optional files...");
    for file in optional_files {
        print!("  {} ... ", file);
        io::stdout().flush()?;
        match repo.get(file) {
            Ok(_) => {
                println!("\x1b[32m✓\x1b[0m");
                downloaded.push(file.to_string());
            },
            Err(_) => {
                println!("\x1b[33m-\x1b[0m (optional, skipped)");
            },
        }
    }

    // Download weight files
    println!("\nDownloading model weights...");

    // Try single safetensors file first
    print!("  model.safetensors ... ");
    io::stdout().flush()?;
    match repo.get("model.safetensors") {
        Ok(_) => {
            println!("\x1b[32m✓\x1b[0m");
            downloaded.push("model.safetensors".to_string());
            has_weights = true;
        },
        Err(_) => {
            println!("\x1b[33m-\x1b[0m (checking for sharded weights...)");

            // Try sharded format
            print!("  model.safetensors.index.json ... ");
            io::stdout().flush()?;
            match repo.get("model.safetensors.index.json") {
                Ok(index_path) => {
                    println!("\x1b[32m✓\x1b[0m");
                    downloaded.push("model.safetensors.index.json".to_string());

                    // Parse index to find shard files
                    if let Ok(index_content) = std::fs::read_to_string(&index_path) {
                        if let Ok(index_json) =
                            serde_json::from_str::<serde_json::Value>(&index_content)
                        {
                            if let Some(weight_map) =
                                index_json.get("weight_map").and_then(|w| w.as_object())
                            {
                                // Get unique shard files
                                let mut shard_files: Vec<String> = weight_map
                                    .values()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect();
                                shard_files.sort();
                                shard_files.dedup();

                                println!(
                                    "\n  Found {} weight shards to download:",
                                    shard_files.len()
                                );
                                let shard_progress = ProgressBar::new(shard_files.len() as u64);
                                shard_progress.set_style(
                                    ProgressStyle::default_bar()
                                        .template("  [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                                        .context("invalid progress bar template")?
                                        .progress_chars("#>-"),
                                );

                                let mut shard_errors = 0;
                                for shard in &shard_files {
                                    shard_progress.set_message(shard.clone());
                                    match repo.get(shard) {
                                        Ok(_) => {
                                            downloaded.push(shard.clone());
                                        },
                                        Err(_) => {
                                            shard_errors += 1;
                                        },
                                    }
                                    shard_progress.inc(1);
                                }
                                shard_progress.finish_and_clear();

                                if shard_errors == 0 {
                                    println!(
                                        "  \x1b[32mAll {} shards downloaded successfully.\x1b[0m",
                                        shard_files.len()
                                    );
                                    has_weights = true;
                                } else {
                                    println!(
                                        "  \x1b[31m{} of {} shards failed to download.\x1b[0m",
                                        shard_errors,
                                        shard_files.len()
                                    );
                                }
                            }
                        }
                    }
                },
                Err(_) => {
                    println!("\x1b[31m✗\x1b[0m");
                    failed.push("model weights".to_string());
                },
            }
        },
    }

    // Summary
    println!();
    if !failed.is_empty() {
        println!("\x1b[31mDownload incomplete!\x1b[0m");
        println!("Failed to download: {}", failed.join(", "));
        println!("\nThis model may require authentication. Try:");
        println!("  huggingface-cli login");
        return Err(eyre!("Some required files failed to download"));
    }

    if !has_weights {
        println!("\x1b[31mNo model weights found!\x1b[0m");
        println!("The model may use a format not yet supported.");
        return Err(eyre!("Could not find model weights"));
    }

    println!("\x1b[32mDownload complete!\x1b[0m");
    println!(
        "Downloaded {} files for model '{}'",
        downloaded.len(),
        model
    );
    println!(
        "\nUse 'infernum generate --model {}' to run inference.",
        model
    );

    Ok(())
}

/// Show model information.
pub async fn model_info(model: String) -> Result<()> {
    use hf_hub::api::sync::Api;

    println!("Model: {}\n", model);

    let api = Api::new()?;
    let repo = api.model(model.clone());

    // Try to get config.json
    match repo.get("config.json") {
        Ok(path) => {
            let config_str = std::fs::read_to_string(&path)?;
            let config: serde_json::Value = serde_json::from_str(&config_str)?;

            if let Some(arch) = config.get("architectures").and_then(|a| a.as_array()) {
                println!(
                    "Architecture: {}",
                    arch.first().and_then(|a| a.as_str()).unwrap_or("Unknown")
                );
            }
            if let Some(hidden) = config.get("hidden_size").and_then(|h| h.as_u64()) {
                println!("Hidden size: {}", hidden);
            }
            if let Some(layers) = config.get("num_hidden_layers").and_then(|l| l.as_u64()) {
                println!("Layers: {}", layers);
            }
            if let Some(heads) = config.get("num_attention_heads").and_then(|h| h.as_u64()) {
                println!("Attention heads: {}", heads);
            }
            if let Some(vocab) = config.get("vocab_size").and_then(|v| v.as_u64()) {
                println!("Vocabulary size: {}", vocab);
            }
            if let Some(ctx) = config
                .get("max_position_embeddings")
                .and_then(|c| c.as_u64())
            {
                println!("Max context length: {}", ctx);
            }

            println!("\nCache location: {:?}", path.parent().unwrap_or(&path));
        },
        Err(e) => {
            println!("Could not fetch model info: {}", e);
            println!("The model may need to be downloaded first.");
            println!("Use: infernum model pull {}", model);
        },
    }

    Ok(())
}

/// Remove a cached model.
pub async fn model_remove(model: String) -> Result<()> {
    use std::path::PathBuf;

    let cache_dir = dirs::cache_dir()
        .map(|p| p.join("huggingface").join("hub"))
        .unwrap_or_else(|| PathBuf::from("~/.cache/huggingface/hub"));

    let model_dir_name = format!("models--{}", model.replace("/", "--"));
    let model_path = cache_dir.join(&model_dir_name);

    if model_path.exists() {
        println!("Removing cached model: {}", model);
        println!("Path: {:?}", model_path);

        std::fs::remove_dir_all(&model_path)?;
        println!("\nModel removed successfully.");
    } else {
        println!("Model {} is not cached.", model);
        println!("Expected path: {:?}", model_path);
    }

    Ok(())
}

/// Convert a model to HoloTensor HCT format.
///
/// This enables 70B+ parameter models to run on 24GB VRAM via progressive
/// quality reconstruction using holographic tensor compression.
pub async fn model_convert(
    model: String,
    output: String,
    fragments: u32,
    max_rank: u32,
    min_quality: f32,
    verify: bool,
    cpu_only: bool,
    encoding: String,
) -> Result<()> {
    use abaddon::holotensor::HolographicEncoding;
    use std::path::PathBuf;

    // Auto-detect GPU unless cpu_only is set
    #[cfg(feature = "cuda")]
    let use_gpu = if cpu_only {
        false
    } else {
        // Check if CUDA is available
        match probe_cuda_runtime() {
            Ok(info) if info.device_count > 0 => {
                println!("\x1b[32m✓ GPU detected:\x1b[0m {} ({:.1} GB VRAM)",
                    info.devices[0].name, info.devices[0].memory_gb);
                true
            }
            _ => {
                println!("\x1b[33m○ No GPU detected, using CPU\x1b[0m");
                false
            }
        }
    };
    #[cfg(not(feature = "cuda"))]
    let use_gpu = false;

    // Parse encoding type
    let holo_encoding = match encoding.to_lowercase().as_str() {
        "spectral" | "dct" => HolographicEncoding::Spectral,
        "lrdf" | "svd" => HolographicEncoding::LowRankDistributed,
        "rph" | "random" => HolographicEncoding::RandomProjection,
        _ => {
            return Err(eyre!(
                "Invalid encoding '{}'. Valid options: lrdf, spectral, rph",
                encoding
            ));
        }
    };

    let encoding_name = match holo_encoding {
        HolographicEncoding::Spectral => "Spectral (DCT)",
        HolographicEncoding::LowRankDistributed => "LRDF (SVD)",
        HolographicEncoding::RandomProjection => "RPH (Random Projection)",
    };

    println!();
    println!("\x1b[1m🌀 HoloTensor Model Conversion\x1b[0m");
    println!("   Source: {}", model);
    println!("   Output: {}", output);
    println!("   Encoding: \x1b[36m{}\x1b[0m", encoding_name);
    println!("   Fragments: {} (more = higher max quality)", fragments);
    println!("   Max Rank: {} (higher = better approximation)", max_rank);
    println!("   Min Quality: {:.0}%", min_quality * 100.0);
    println!("   Verify: {}", if verify { "yes" } else { "no" });
    println!("   Compute: {}", if use_gpu { "\x1b[32mGPU (CUDA)\x1b[0m" } else { "CPU" });
    println!();

    // Create output directory
    let output_path = PathBuf::from(&output);
    std::fs::create_dir_all(&output_path)?;

    // Show progress
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .context("invalid progress bar template")?,
    );
    spinner.set_message(format!("Loading model {}...", model));
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    // Use the HoloModelConverter from abaddon
    use abaddon::holotensor::converter::{ConversionConfig, HoloModelConverter};

    let config = ConversionConfig {
        encoding: holo_encoding,
        num_fragments: fragments as u16,
        max_rank: max_rank as usize,
        verify_quality: verify,
        min_quality,
        parallel: true,  // Enable parallel tensor processing
        num_threads: 0,  // Use all available cores
        use_gpu,         // Use GPU if available
        ..ConversionConfig::default()
    };

    let converter = HoloModelConverter::new(config);

    spinner.set_message("Converting to HoloTensor format...");

    // Perform conversion
    match converter.convert_model(&model, &output_path).await {
        Ok(metadata) => {
            spinner.finish_and_clear();

            println!("\x1b[32m✓ Conversion complete!\x1b[0m");
            println!();
            println!("Model metadata:");
            println!("  Layers: {}", metadata.num_layers);
            println!("  Total fragments: {}", metadata.total_fragments);
            println!(
                "  Original size: {:.2} GB",
                metadata.original_size as f64 / 1_073_741_824.0
            );
            println!(
                "  HCT size: {:.2} GB",
                metadata.hct_size as f64 / 1_073_741_824.0
            );
            println!(
                "  Compression ratio: {:.2}x",
                metadata.original_size as f64 / metadata.hct_size as f64
            );

            if verify {
                println!();
                println!("Quality verification:");
                println!(
                    "  Min quality achieved: {:.1}%",
                    metadata.verified_quality * 100.0
                );
                if metadata.verified_quality >= min_quality {
                    println!("  \x1b[32m✓ Meets minimum quality threshold\x1b[0m");
                } else {
                    println!("  \x1b[33m⚠ Below minimum quality threshold\x1b[0m");
                }
            }

            println!();
            println!("To use this model:");
            println!(
                "  infernum serve --holo --model {}",
                output_path.display()
            );
        }
        Err(e) => {
            spinner.finish_and_clear();
            return Err(eyre!("Conversion failed: {}", e));
        }
    }

    Ok(())
}

/// Quantize a model to INT4/INT8 format.
pub async fn model_quantize(
    model: String,
    output: String,
    format: String,
    block_size: usize,
    verify: bool,
) -> Result<()> {
    use std::path::PathBuf;
    use abaddon::quantize::{Quantizer, QuantizeConfig, QuantizeFormat};
    use candle_core::{Device, DType};
    use safetensors::SafeTensors;

    // Parse format
    let quant_format = match format.as_str() {
        "int4-sym" | "int4" => QuantizeFormat::Int4Symmetric,
        "int4-asym" => QuantizeFormat::Int4Asymmetric,
        "int8-sym" | "int8" => QuantizeFormat::Int8Symmetric,
        "int8-asym" => QuantizeFormat::Int8Asymmetric,
        _ => return Err(eyre!(
            "Unknown format: {}. Valid formats: int4-sym, int4-asym, int8-sym, int8-asym",
            format
        )),
    };

    println!();
    println!("\x1b[1m🔢 INT4/INT8 Model Quantization\x1b[0m");
    println!("   Source: {}", model);
    println!("   Output: {}", output);
    println!("   Format: {} ({}-bit)", format, quant_format.bits());
    println!("   Block size: {} values per scale", block_size);
    println!("   Verify: {}", if verify { "yes" } else { "no" });
    println!();

    // Create output directory
    let output_path = PathBuf::from(&output);
    std::fs::create_dir_all(&output_path)?;

    // Show progress
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .context("invalid progress bar template")?,
    );
    spinner.set_message(format!("Loading model {}...", model));
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    // Resolve model path (HuggingFace or local)
    let model_path = if model.contains('/') && !PathBuf::from(&model).exists() {
        // HuggingFace model - download safetensors
        use hf_hub::{api::sync::Api, Repo, RepoType};
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model.clone(), RepoType::Model));

        // Find safetensors files
        let files: Vec<_> = repo.info()?
            .siblings
            .iter()
            .filter(|s| s.rfilename.ends_with(".safetensors"))
            .map(|s| s.rfilename.clone())
            .collect();

        if files.is_empty() {
            return Err(eyre!("No safetensors files found in model {}", model));
        }

        // Download all safetensors files
        let mut paths = Vec::new();
        for file in &files {
            spinner.set_message(format!("Downloading {}...", file));
            paths.push(repo.get(file)?);
        }
        paths
    } else {
        // Local path
        let path = PathBuf::from(&model);
        if path.is_file() {
            vec![path]
        } else {
            // Find all safetensors in directory
            std::fs::read_dir(&path)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
                .collect()
        }
    };

    if model_path.is_empty() {
        return Err(eyre!("No safetensors files found"));
    }

    spinner.set_message("Quantizing tensors...");

    // Create quantizer
    let config = QuantizeConfig {
        format: quant_format,
        block_size,
        activation_aware: false,
    };
    let quantizer = Quantizer::new(config);

    let mut total_original = 0u64;
    let mut total_quantized = 0u64;
    let mut tensor_count = 0usize;
    let mut total_snr = 0.0f64;

    // Process each safetensors file
    for (file_idx, source_path) in model_path.iter().enumerate() {
        let file_name = source_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model.safetensors");

        spinner.set_message(format!(
            "[{}/{}] Processing {}...",
            file_idx + 1,
            model_path.len(),
            file_name
        ));

        // Load safetensors
        let data = std::fs::read(source_path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        // Prepare quantized tensors
        let mut quantized_data: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
        let mut metadata: std::collections::HashMap<String, String> = std::collections::HashMap::new();

        for (name, tensor_view) in tensors.tensors() {
            let shape = tensor_view.shape().to_vec();
            let dtype = tensor_view.dtype();

            // Skip non-float tensors
            if !matches!(dtype, safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16) {
                // Copy as-is
                let tensor_data = tensor_view.data().to_vec();
                quantized_data.push((name.to_string(), tensor_data, shape));
                continue;
            }

            // Convert to candle tensor
            let device = Device::Cpu;
            let candle_dtype = match dtype {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F16 => DType::F16,
                safetensors::Dtype::BF16 => DType::BF16,
                _ => continue,
            };

            let tensor = candle_core::Tensor::from_raw_buffer(
                tensor_view.data(),
                candle_dtype,
                &shape,
                &device,
            )?;

            let original_size = tensor.elem_count() * 4; // Assume F32 storage
            total_original += original_size as u64;

            // Quantize
            let result = quantizer.quantize_tensor(&tensor)
                .map_err(|e| eyre!("Failed to quantize {}: {}", name, e))?;

            total_quantized += (result.data.len() + result.scales.len() * 2) as u64;
            tensor_count += 1;
            total_snr += result.stats.snr_db as f64;

            // Store quantized data (as packed bytes)
            quantized_data.push((name.to_string(), result.data.clone(), shape.clone()));

            // Store scales separately
            let scales_name = format!("{}.scales", name);
            let scales_bytes: Vec<u8> = result.scales.iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
            quantized_data.push((scales_name, scales_bytes, vec![result.scales.len()]));

            // Store zero points if asymmetric
            if let Some(zp) = &result.zero_points {
                let zp_name = format!("{}.zero_points", name);
                let zp_bytes: Vec<u8> = zp.iter().map(|&z| z as u8).collect();
                quantized_data.push((zp_name, zp_bytes, vec![zp.len()]));
            }

            // Add metadata
            metadata.insert(format!("{}.format", name), format.clone());
            metadata.insert(format!("{}.block_size", name), block_size.to_string());
        }

        // Write quantized safetensors
        let output_file = output_path.join(file_name);

        // Build tensors for safetensors
        use safetensors::tensor::TensorView;
        let tensors_for_save: Vec<(&str, TensorView)> = quantized_data.iter()
            .map(|(name, data, shape)| {
                TensorView::new(
                    safetensors::Dtype::U8,  // Store as bytes
                    shape.clone(),
                    data,
                ).map(|view| (name.as_str(), view))
                .map_err(|e| eyre!("failed to create tensor view for '{}': {}", name, e))
            })
            .collect::<Result<Vec<_>>>()?;

        safetensors::serialize_to_file(tensors_for_save, &Some(metadata.clone()), &output_file)?;
    }

    spinner.finish_and_clear();

    let compression = total_original as f64 / total_quantized as f64;
    let avg_snr = if tensor_count > 0 { total_snr / tensor_count as f64 } else { 0.0 };

    println!("\x1b[32m✓ Quantization complete!\x1b[0m");
    println!();
    println!("Results:");
    println!("  Tensors quantized: {}", tensor_count);
    println!(
        "  Original size: {:.2} GB",
        total_original as f64 / 1_073_741_824.0
    );
    println!(
        "  Quantized size: {:.2} GB",
        total_quantized as f64 / 1_073_741_824.0
    );
    println!("  Compression: {:.2}x", compression);
    println!("  Average SNR: {:.1} dB", avg_snr);

    if verify {
        println!();
        if avg_snr > 20.0 {
            println!("  \x1b[32m✓ Quality verified (SNR > 20 dB)\x1b[0m");
        } else {
            println!("  \x1b[33m⚠ Quality may be degraded (SNR < 20 dB)\x1b[0m");
        }
    }

    println!();
    println!("Quantized model saved to: {}", output_path.display());
    println!();
    println!("To use this model:");
    println!("  infernum serve --model {}", output_path.display());

    Ok(())
}

/// Start an interactive chat session.
pub async fn chat(model: Option<String>, system: Option<String>) -> Result<()> {
    let model_id = model.ok_or_else(|| {
        eyre!(
            "Model is required.\n\n\
         Options:\n  \
         1. Specify on command line: --model <model>\n  \
         2. Set a default: infernum config set-model <model>\n  \
         3. Set environment variable: INFERNUM_DEFAULT_MODEL=<model>\n\n\
         Example models:\n  \
         - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (small, fast)\n  \
         - meta-llama/Llama-3.2-3B-Instruct (requires HuggingFace login)"
        )
    })?;

    // Show loading indicator
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .context("invalid progress bar template")?,
    );
    spinner.set_message(format!("Loading model {}...", model_id));
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    // Create engine config
    let config = EngineConfig::builder()
        .model(&model_id)
        .build()
        .map_err(|e| eyre!("Failed to configure engine: {}", e))?;

    // Load the model
    let engine = Engine::new(config).await?;
    let engine = Arc::new(engine);

    spinner.finish_and_clear();

    println!("Infernum Interactive Chat");
    println!("========================");
    println!("Model: {}", engine.model_info().id);
    if let Some(sys) = &system {
        println!("System: {}", sys);
    }
    println!("\nCommands:");
    println!("  /help    - Show this help");
    println!("  /clear   - Clear conversation history");
    println!("  /history - Show conversation history");
    println!("  /save <file> - Save conversation to file");
    println!("  /load <file> - Load conversation from file");
    println!("  exit/quit    - End the session\n");

    // Initialize conversation history
    let mut messages: Vec<Message> = Vec::new();

    // Add system message if provided
    if let Some(system_prompt) = &system {
        messages.push(Message {
            role: Role::System,
            content: system_prompt.clone(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    loop {
        print!("\x1b[32mYou:\x1b[0m ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("\nGoodbye!");
            break;
        }

        // Handle commands
        if input.starts_with('/') {
            let parts: Vec<&str> = input.splitn(2, ' ').collect();
            let cmd = parts[0].to_lowercase();
            let arg = parts.get(1).map(|s| s.trim());

            match cmd.as_str() {
                "/help" => {
                    println!("\nCommands:");
                    println!("  /help    - Show this help");
                    println!("  /clear   - Clear conversation history");
                    println!("  /history - Show conversation history");
                    println!("  /save <file> - Save conversation to file");
                    println!("  /load <file> - Load conversation from file");
                    println!("  exit/quit    - End the session\n");
                    continue;
                },
                "/clear" => {
                    messages.clear();
                    if let Some(system_prompt) = &system {
                        messages.push(Message {
                            role: Role::System,
                            content: system_prompt.clone(),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                    println!("\nConversation cleared.\n");
                    continue;
                },
                "/history" => {
                    println!("\n--- Conversation History ---");
                    for (i, msg) in messages.iter().enumerate() {
                        let role_color = match msg.role {
                            Role::System => "\x1b[35m",    // magenta
                            Role::User => "\x1b[32m",      // green
                            Role::Assistant => "\x1b[34m", // blue
                            _ => "\x1b[0m",
                        };
                        let role_name = match msg.role {
                            Role::System => "System",
                            Role::User => "You",
                            Role::Assistant => "Assistant",
                            _ => "Unknown",
                        };
                        let preview = if msg.content.len() > 80 {
                            format!("{}...", &msg.content[..80])
                        } else {
                            msg.content.clone()
                        };
                        println!(
                            "{}[{}] {}:\x1b[0m {}",
                            role_color,
                            i + 1,
                            role_name,
                            preview
                        );
                    }
                    println!("--- {} messages ---\n", messages.len());
                    continue;
                },
                "/save" => {
                    if let Some(filename) = arg {
                        match save_chat_history(&messages, filename) {
                            Ok(()) => println!("\nConversation saved to '{}'\n", filename),
                            Err(e) => eprintln!("\nFailed to save: {}\n", e),
                        }
                    } else {
                        eprintln!("\nUsage: /save <filename>\n");
                    }
                    continue;
                },
                "/load" => {
                    if let Some(filename) = arg {
                        match load_chat_history(filename) {
                            Ok(loaded_messages) => {
                                messages = loaded_messages;
                                println!(
                                    "\nLoaded {} messages from '{}'\n",
                                    messages.len(),
                                    filename
                                );
                            },
                            Err(e) => eprintln!("\nFailed to load: {}\n", e),
                        }
                    } else {
                        eprintln!("\nUsage: /load <filename>\n");
                    }
                    continue;
                },
                _ => {
                    eprintln!(
                        "\nUnknown command: {}\nType /help for available commands.\n",
                        cmd
                    );
                    continue;
                },
            }
        }

        // Add user message
        messages.push(Message {
            role: Role::User,
            content: input.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });

        // Create request with conversation history
        let request = GenerateRequest::chat(messages.clone())
            .with_sampling(SamplingParams::default().with_max_tokens(1024));

        // Generate response with streaming
        print!("\n\x1b[34mAssistant:\x1b[0m ");
        io::stdout().flush()?;

        let mut response_text = String::new();

        match engine.generate_stream(request).await {
            Ok(token_stream) => {
                futures::pin_mut!(token_stream);

                while let Some(result) = token_stream.next().await {
                    match result {
                        Ok(chunk) => {
                            for choice in chunk.choices {
                                if let Some(content) = choice.delta.content {
                                    print!("{}", content);
                                    io::stdout().flush()?;
                                    response_text.push_str(&content);
                                }
                            }
                        },
                        Err(e) => {
                            eprintln!("\nError: {}", e);
                            break;
                        },
                    }
                }
            },
            Err(e) => {
                eprintln!("Error generating response: {}", e);
                // Remove the last user message on error
                messages.pop();
                continue;
            },
        }

        println!("\n");

        // Add assistant response to history
        if !response_text.is_empty() {
            messages.push(Message {
                role: Role::Assistant,
                content: response_text,
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    Ok(())
}

/// Run an autonomous agent with tools.
pub async fn agent(
    objective: String,
    model: Option<String>,
    system: Option<String>,
    max_iterations: u32,
    _verbose: bool,
    working_dir: Option<std::path::PathBuf>,
    code_tools: bool,
) -> Result<()> {
    use beleth::{Agent, ToolRegistry};

    // Get model
    let model_id = match model {
        Some(m) => m,
        None => {
            return Err(eyre!(
                "Model is required.\n\n\
                 Options:\n  \
                 1. Specify on command line: infernum agent \"task\" --model <model>\n  \
                 2. Set default model: infernum config set-model <model>\n\n\
                 Example:\n  \
                 infernum agent \"Calculate 23 * 47\" --model TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ));
        },
    };

    println!("\x1b[1m🤖 Infernum Agent\x1b[0m");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    // Resolve working directory (default to cwd) early so we can display it
    let wd = working_dir
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| ".".into()));

    println!("\x1b[1mObjective:\x1b[0m {}", objective);
    println!("\x1b[1mModel:\x1b[0m {}", model_id);
    println!("\x1b[1mMax iterations:\x1b[0m {}", max_iterations);
    if code_tools {
        println!("\x1b[1mWorking dir:\x1b[0m {}", wd.display());
    }
    println!();

    // Load model
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .context("invalid progress bar template")?,
    );
    spinner.set_message(format!("Loading model: {}", model_id));
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    let config = EngineConfig::builder()
        .model(&model_id)
        .build()
        .map_err(|e| eyre!("Failed to configure engine: {}", e))?;

    let engine = Engine::new(config).await?;
    let engine = Arc::new(engine);

    spinner.finish_and_clear();

    // Set up tools
    let tools = if code_tools {
        ToolRegistry::with_code_tools()
    } else {
        ToolRegistry::with_builtins()
    };

    println!("\x1b[1mAvailable tools:\x1b[0m");
    for tool in tools.tools() {
        println!("  • {} - {}", tool.name(), tool.description());
    }
    println!();

    // Create agent
    let mut agent = Agent::builder()
        .id("cli-agent")
        .max_iterations(max_iterations)
        .tools(tools)
        .engine(engine)
        .working_dir(&wd);

    // Set system prompt if provided
    if let Some(sys) = system {
        agent = agent.system_prompt(sys);
    } else {
        agent = agent.system_prompt(
            "You are a helpful AI assistant with access to tools. \
             Think step by step and use tools when needed to accomplish tasks. \
             Always explain your reasoning.",
        );
    }

    let mut agent = agent.build();

    println!("\x1b[33m⚡ Starting agent execution...\x1b[0m\n");

    // Run agent
    let result = agent.run(&objective).await;

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    match result {
        Ok(answer) => {
            println!("\x1b[32m✓ Agent completed\x1b[0m\n");
            println!("\x1b[1mFinal Answer:\x1b[0m");
            println!("{}", answer);
        },
        Err(e) => {
            println!("\x1b[31m✗ Agent failed\x1b[0m\n");
            println!("Error: {}", e);
            return Err(eyre!("Agent execution failed: {}", e));
        },
    }

    Ok(())
}

/// Display version information.
pub fn version() {
    println!("\x1b[1mInfernum {}\x1b[0m", env!("CARGO_PKG_VERSION"));
    println!("From the depths, intelligence rises.");
    println!();

    // Build information
    println!("\x1b[1mBuild Info:\x1b[0m");
    println!("  Rust Version:  {}", rustc_version());
    println!(
        "  Profile:       {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    println!("  Target:        {}", std::env::consts::ARCH);
    println!("  OS:            {}", std::env::consts::OS);
    println!();

    // Feature flags
    println!("\x1b[1mAccelerators:\x1b[0m");
    #[cfg(feature = "cuda")]
    println!("  CUDA:          \x1b[32menabled\x1b[0m");
    #[cfg(not(feature = "cuda"))]
    println!("  CUDA:          \x1b[33mdisabled\x1b[0m (rebuild with --features cuda)");

    #[cfg(feature = "metal")]
    println!("  Metal:         \x1b[32menabled\x1b[0m");
    #[cfg(not(feature = "metal"))]
    println!("  Metal:         \x1b[33mdisabled\x1b[0m (rebuild with --features metal)");

    println!("  CPU:           \x1b[32malways available\x1b[0m");
    println!();

    // Components
    println!("\x1b[1mComponents:\x1b[0m");
    println!("  Abaddon    - Inference Engine");
    println!("  Malphas    - Orchestration Layer");
    println!("  Stolas     - Knowledge Engine");
    println!("  Beleth     - Agent Framework");
    println!("  Asmodeus   - Adaptation Layer");
    println!("  Dantalion  - Observability");
    println!();

    println!("Daemoniorum, LLC - Building Tomorrow's Intelligence");
}

/// Returns rustc version string.
fn rustc_version() -> &'static str {
    env!("CARGO_PKG_RUST_VERSION")
}

/// Run system diagnostics.
pub fn doctor() {
    use std::path::PathBuf;

    println!("\x1b[1m🔍 Infernum System Diagnostics\x1b[0m");
    println!("================================\n");

    let mut issues: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // 1. Check Rust/build info
    println!("\x1b[1m[Build]\x1b[0m");
    println!("  Version:     {}", env!("CARGO_PKG_VERSION"));
    println!(
        "  Profile:     {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    check_ok("Build info");
    println!();

    // 2. Check compute backends
    println!("\x1b[1m[Compute Backends]\x1b[0m");

    #[cfg(feature = "cuda")]
    {
        println!("  CUDA:        \x1b[32m✓ compiled in\x1b[0m");
        // Probe CUDA runtime availability
        match probe_cuda_runtime() {
            Ok(info) => {
                println!("               Runtime: \x1b[32m✓ available\x1b[0m");
                println!("               Devices: {}", info.device_count);
                for (i, device) in info.devices.iter().enumerate() {
                    println!(
                        "                 GPU {}: {} ({:.1} GB)",
                        i, device.name, device.memory_gb
                    );
                }
            }
            Err(e) => {
                println!("               Runtime: \x1b[31m✗ unavailable\x1b[0m");
                warnings.push(format!("CUDA runtime not available: {}", e));
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("  CUDA:        \x1b[33m○ not compiled\x1b[0m");
        if std::env::consts::OS == "linux" {
            warnings.push("CUDA support not compiled. For NVIDIA GPUs, rebuild with: cargo build --features cuda".to_string());
        }
    }

    #[cfg(feature = "metal")]
    {
        println!("  Metal:       \x1b[32m✓ compiled in\x1b[0m");
    }
    #[cfg(not(feature = "metal"))]
    {
        println!("  Metal:       \x1b[33m○ not compiled\x1b[0m");
        if std::env::consts::OS == "macos" {
            warnings.push("Metal support not compiled. For Apple Silicon, rebuild with: cargo build --features metal".to_string());
        }
    }

    println!("  CPU:         \x1b[32m✓ always available\x1b[0m");
    println!();

    // 3. Check configuration
    println!("\x1b[1m[Configuration]\x1b[0m");
    let config_path = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("infernum")
        .join("config.toml");

    if config_path.exists() {
        println!("  Config file: \x1b[32m✓ found\x1b[0m");
        println!("               {}", config_path.display());

        // Try to parse it
        match std::fs::read_to_string(&config_path) {
            Ok(content) => {
                if toml::from_str::<toml::Value>(&content).is_ok() {
                    println!("  Syntax:      \x1b[32m✓ valid TOML\x1b[0m");
                } else {
                    println!("  Syntax:      \x1b[31m✗ invalid TOML\x1b[0m");
                    issues.push(format!(
                        "Config file has invalid TOML syntax: {}",
                        config_path.display()
                    ));
                }
            },
            Err(_) => {
                println!("  Syntax:      \x1b[31m✗ unreadable\x1b[0m");
                issues.push(format!(
                    "Cannot read config file: {}",
                    config_path.display()
                ));
            },
        }
    } else {
        println!("  Config file: \x1b[90m○ not found (using defaults)\x1b[0m");
        println!("               {}", config_path.display());
    }
    println!();

    // 4. Check HuggingFace cache
    println!("\x1b[1m[Model Cache]\x1b[0m");
    let cache_dir = dirs::cache_dir()
        .map(|p| p.join("huggingface").join("hub"))
        .unwrap_or_else(|| PathBuf::from("~/.cache/huggingface/hub"));

    if cache_dir.exists() {
        let model_count = std::fs::read_dir(&cache_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_name().to_string_lossy().starts_with("models--"))
                    .count()
            })
            .unwrap_or(0);

        println!("  Cache dir:   \x1b[32m✓ found\x1b[0m");
        println!("               {}", cache_dir.display());
        println!("  Models:      {} cached", model_count);

        // Check cache size
        if let Ok(size) = dir_size(&cache_dir) {
            println!("  Size:        {}", format_size(size));
        }
    } else {
        println!("  Cache dir:   \x1b[90m○ not created yet\x1b[0m");
        println!("               (will be created on first model download)");
    }
    println!();

    // 5. Check HuggingFace auth
    println!("\x1b[1m[HuggingFace]\x1b[0m");
    let hf_token_path = dirs::home_dir()
        .map(|h| h.join(".cache").join("huggingface").join("token"))
        .unwrap_or_else(|| PathBuf::from("~/.cache/huggingface/token"));

    if hf_token_path.exists() {
        println!("  Auth token:  \x1b[32m✓ found\x1b[0m");
        println!("               (can access gated models)");
    } else if std::env::var("HF_TOKEN").is_ok() || std::env::var("HUGGING_FACE_HUB_TOKEN").is_ok() {
        println!("  Auth token:  \x1b[32m✓ found in environment\x1b[0m");
    } else {
        println!("  Auth token:  \x1b[33m○ not found\x1b[0m");
        warnings.push(
            "HuggingFace not authenticated. Some models require login: huggingface-cli login"
                .to_string(),
        );
    }
    println!();

    // 6. Check system resources
    println!("\x1b[1m[System Resources]\x1b[0m");
    println!("  CPU cores:   {}", num_cpus::get());
    println!("  Architecture: {}", std::env::consts::ARCH);
    println!();

    // 7. Model recommendations based on detected hardware
    println!("\x1b[1m[Recommended Models]\x1b[0m");
    #[cfg(feature = "cuda")]
    {
        if let Ok(info) = probe_cuda_runtime() {
            if let Some(device) = info.devices.first() {
                let vram_gb = device.memory_gb;
                print_model_recommendations(vram_gb);
            } else {
                print_model_recommendations(0.0);
            }
        } else {
            print_model_recommendations(0.0);
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        print_model_recommendations(0.0);
    }
    println!();

    // Summary
    println!("\x1b[1m[Summary]\x1b[0m");
    if issues.is_empty() && warnings.is_empty() {
        println!("  \x1b[32m✓ All checks passed!\x1b[0m");
        println!("  Infernum is ready to use.");
    } else {
        if !issues.is_empty() {
            println!("\n  \x1b[31mIssues ({}):\x1b[0m", issues.len());
            for issue in &issues {
                println!("    • {}", issue);
            }
        }
        if !warnings.is_empty() {
            println!("\n  \x1b[33mWarnings ({}):\x1b[0m", warnings.len());
            for warning in &warnings {
                println!("    • {}", warning);
            }
        }
    }
    println!();

    // Quick start hint
    println!("\x1b[1mNext Steps:\x1b[0m");
    if issues.is_empty() && warnings.is_empty() {
        println!("  infernum setup              # First-time setup wizard");
        println!("  infernum chat               # Start chatting");
    } else {
        println!("  infernum setup              # Run setup wizard to configure");
        println!("  infernum doctor             # Re-run after fixing issues");
    }
}

/// Interactive first-time setup wizard.
///
/// Guides users through:
/// 1. Hardware detection (GPU/CPU, VRAM)
/// 2. Model selection based on hardware
/// 3. Model download
/// 4. Quick inference test
/// 5. Configuration save
///
/// Use `--yes` flag for non-interactive mode (accepts all defaults).
pub async fn setup(skip_detect: bool, skip_download: bool, skip_test: bool, auto_yes: bool) -> Result<()> {
    use dialoguer::{theme::ColorfulTheme, Confirm, Select};

    println!();
    println!("\x1b[1m🔧 Infernum Setup Wizard\x1b[0m");
    println!("========================\n");
    println!("This wizard will help you get started with Infernum.\n");

    // Step 1: Hardware Detection
    let (backend, vram_gb) = if skip_detect {
        println!("\x1b[90m[Skipping hardware detection]\x1b[0m\n");
        ("CPU".to_string(), 0.0)
    } else {
        println!("\x1b[1mStep 1: Hardware Detection\x1b[0m\n");
        detect_hardware()
    };

    // Step 2: Model Recommendation
    println!("\x1b[1mStep 2: Model Selection\x1b[0m\n");
    let recommended = recommend_model(vram_gb);

    println!("  Based on your hardware ({}):", backend);
    println!();
    for (i, (model, desc, size, vram)) in recommended.iter().enumerate() {
        let marker = if i == 0 { " \x1b[32m← Recommended\x1b[0m" } else { "" };
        println!(
            "  {}. \x1b[1m{}\x1b[0m{}",
            i + 1,
            model,
            marker
        );
        println!("     {} | ~{} download | {} VRAM", desc, size, vram);
        println!();
    }

    // Let user select (or auto-select first if --yes)
    let model_names: Vec<&str> = recommended.iter().map(|(m, _, _, _)| *m).collect();
    let selection = if auto_yes {
        println!("  \x1b[90m[Auto-selecting recommended model]\x1b[0m");
        0
    } else {
        Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Select a model to install")
            .items(&model_names)
            .default(0)
            .interact()
            .map_err(|e| eyre!("Selection cancelled: {}", e))?
    };

    let selected_model = model_names[selection].to_string();
    println!();
    println!("  Selected: \x1b[32m{}\x1b[0m\n", selected_model);

    // Step 3: Download Model
    if skip_download {
        println!("\x1b[90m[Skipping model download]\x1b[0m\n");
    } else {
        println!("\x1b[1mStep 3: Download Model\x1b[0m\n");

        let confirm = if auto_yes {
            true
        } else {
            Confirm::with_theme(&ColorfulTheme::default())
                .with_prompt(format!("Download {}?", selected_model))
                .default(true)
                .interact()
                .map_err(|e| eyre!("Confirmation cancelled: {}", e))?
        };

        if confirm {
            println!();
            model_pull(selected_model.clone(), None).await?;
            println!();
        } else {
            println!("\n  \x1b[33mSkipped download. You can download later with:\x1b[0m");
            println!("  infernum model pull {}\n", selected_model);
        }
    }

    // Step 4: Test Inference
    if skip_test {
        println!("\x1b[90m[Skipping inference test]\x1b[0m\n");
    } else {
        println!("\x1b[1mStep 4: Test Inference\x1b[0m\n");

        let confirm = if auto_yes {
            true
        } else {
            Confirm::with_theme(&ColorfulTheme::default())
                .with_prompt("Run a quick test to verify the model works?")
                .default(true)
                .interact()
                .map_err(|e| eyre!("Confirmation cancelled: {}", e))?
        };

        if confirm {
            println!("\n  Testing with prompt: \"What is 2+2?\"\n");

            match test_inference(&selected_model).await {
                Ok(response) => {
                    println!("  \x1b[32m✓ Model response:\x1b[0m {}\n", response.trim());
                }
                Err(e) => {
                    println!("  \x1b[31m✗ Test failed:\x1b[0m {}\n", e);
                    println!("  \x1b[33mThe model may still work - this could be a temporary issue.\x1b[0m\n");
                }
            }
        }
    }

    // Step 5: Save Configuration
    println!("\x1b[1mStep 5: Save Configuration\x1b[0m\n");

    let confirm = if auto_yes {
        true
    } else {
        Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt(format!(
                "Set {} as your default model?",
                selected_model
            ))
            .default(true)
            .interact()
            .map_err(|e| eyre!("Confirmation cancelled: {}", e))?
    };

    if confirm {
        let mut cfg = crate::config::Config::load();
        match cfg.set_default_model(&selected_model) {
            Ok(()) => {
                println!("\n  \x1b[32m✓ Configuration saved!\x1b[0m");
                println!(
                    "  Config file: {}",
                    crate::config::Config::config_path().display()
                );
            }
            Err(e) => {
                println!("\n  \x1b[31m✗ Failed to save config:\x1b[0m {}", e);
            }
        }
    }

    // Done!
    println!();
    println!("\x1b[1m🎉 Setup Complete!\x1b[0m\n");
    println!("You can now use Infernum:");
    println!();
    println!("  \x1b[36minfernum\x1b[0m                  # Start interactive chat");
    println!("  \x1b[36minfernum serve\x1b[0m            # Start API server");
    println!("  \x1b[36minfernum generate \"...\"\x1b[0m   # Generate text");
    println!();
    println!("For more options: \x1b[36minfernum --help\x1b[0m");
    println!();

    Ok(())
}

/// Detect hardware and return (backend name, VRAM in GB).
fn detect_hardware() -> (String, f64) {
    let mut backend = "CPU".to_string();
    let mut vram_gb = 0.0;

    #[cfg(feature = "cuda")]
    {
        match probe_cuda_runtime() {
            Ok(info) => {
                if let Some(device) = info.devices.first() {
                    backend = format!("CUDA - {} ({:.0} GB)", device.name, device.memory_gb);
                    vram_gb = device.memory_gb;
                    println!("  \x1b[32m✓ Found NVIDIA GPU:\x1b[0m {}", device.name);
                    println!("    VRAM: {:.1} GB", device.memory_gb);
                }
            }
            Err(_) => {
                println!("  \x1b[33m○ No CUDA GPU detected\x1b[0m");
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        // Metal doesn't expose VRAM easily, estimate based on Apple Silicon
        backend = "Metal (Apple Silicon)".to_string();
        // Assume 8GB unified memory as baseline
        vram_gb = 8.0;
        println!("  \x1b[32m✓ Apple Silicon detected\x1b[0m");
        println!("    Using Metal backend");
    }

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        println!("  \x1b[33m○ No GPU acceleration compiled in\x1b[0m");
        println!("    Running on CPU (slower)");
    }

    if vram_gb == 0.0 {
        println!("  \x1b[33m○ Running on CPU\x1b[0m");
        // For CPU, estimate based on system memory
        let sys_mem_gb = (num_cpus::get() as f64) * 2.0; // Rough heuristic
        vram_gb = sys_mem_gb.min(8.0); // Cap recommendations for CPU
    }

    println!();
    (backend, vram_gb)
}

/// Recommend models based on available VRAM.
/// Returns Vec of (model_id, description, download_size, vram_requirement).
///
/// With modern GPUs and HoloTensor, users can run much larger models than
/// traditional VRAM limits would suggest. HoloTensor enables 70B+ models
/// on 24GB VRAM via progressive quality inference.
fn recommend_model(vram_gb: f64) -> Vec<(&'static str, &'static str, &'static str, &'static str)> {
    let mut suitable: Vec<(&str, &str, &str, &str)> = Vec::new();

    // For 24GB+ VRAM, recommend the best models
    if vram_gb >= 24.0 {
        suitable.push((
            "Qwen/Qwen2.5-14B-Instruct",
            "Near-frontier quality",
            "28 GB",
            "~24 GB",
        ));
        suitable.push((
            "meta-llama/Llama-3.1-70B-Instruct",
            "Frontier-class (use --holo)",
            "140 GB",
            "~24 GB (HCT)",
        ));
        suitable.push((
            "Qwen/Qwen2.5-7B-Instruct",
            "Excellent quality, good speed",
            "15 GB",
            "~16 GB",
        ));
    } else if vram_gb >= 16.0 {
        suitable.push((
            "Qwen/Qwen2.5-7B-Instruct",
            "Excellent quality, good speed",
            "15 GB",
            "~16 GB",
        ));
        suitable.push((
            "meta-llama/Llama-3.1-8B-Instruct",
            "Meta's flagship 8B (gated)",
            "16 GB",
            "~18 GB",
        ));
        suitable.push((
            "Qwen/Qwen2.5-3B-Instruct",
            "Fast, capable instruct model",
            "6.5 GB",
            "~8 GB",
        ));
    } else if vram_gb >= 8.0 {
        suitable.push((
            "Qwen/Qwen2.5-3B-Instruct",
            "Fast, capable instruct model",
            "6.5 GB",
            "~8 GB",
        ));
        suitable.push((
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Lightweight but capable",
            "3.1 GB",
            "~4 GB",
        ));
    } else {
        // CPU or very low VRAM
        suitable.push((
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Lightweight but capable",
            "3.1 GB",
            "~4 GB",
        ));
        suitable.push((
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Ultra-fast, minimal resources",
            "2.2 GB",
            "~3 GB",
        ));
    }

    suitable
}

/// Print model recommendations for the doctor command.
fn print_model_recommendations(vram_gb: f64) {
    let recommendations = recommend_model(vram_gb);

    if recommendations.is_empty() {
        println!("  No specific recommendations available.");
        return;
    }

    for (i, (model, desc, size, vram)) in recommendations.iter().take(2).enumerate() {
        let marker = if i == 0 { " \x1b[32m← Best fit\x1b[0m" } else { "" };
        println!("  • \x1b[1m{}\x1b[0m{}", model, marker);
        println!("    {} | {} | {}", desc, size, vram);
    }
}

/// Run a quick inference test.
async fn test_inference(model_id: &str) -> Result<String> {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .context("invalid progress bar template")?,
    );
    spinner.set_message("Loading model...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));

    // Create engine
    let config = EngineConfig::builder()
        .model(model_id)
        .build()
        .map_err(|e| eyre!("Failed to configure engine: {}", e))?;

    let engine = Engine::new(config).await?;
    let engine = Arc::new(engine);

    spinner.set_message("Running inference...");

    // Simple test prompt
    let sampling = SamplingParams::greedy().with_max_tokens(10);
    let request = GenerateRequest::new("What is 2+2? Answer with just the number.")
        .with_sampling(sampling);

    let response = engine.generate(request).await?;

    spinner.finish_and_clear();

    // Get text from first choice
    let text = response
        .choices
        .first()
        .map(|c| c.text.clone())
        .unwrap_or_default();

    Ok(text)
}

/// Information about CUDA runtime.
#[cfg(feature = "cuda")]
struct CudaInfo {
    device_count: usize,
    devices: Vec<CudaDeviceInfo>,
}

/// Information about a CUDA device.
#[cfg(feature = "cuda")]
struct CudaDeviceInfo {
    name: String,
    memory_gb: f64,
}

/// Probes the CUDA runtime for availability and device information.
///
/// Uses abaddon's device enumeration which correctly detects VRAM based on
/// compute capability, avoiding the WSL memory allocation probe issues.
#[cfg(feature = "cuda")]
fn probe_cuda_runtime() -> std::result::Result<CudaInfo, String> {
    use abaddon::device::enumerate_devices;
    use infernum_core::DeviceType;

    // Use abaddon's device enumeration which has proper VRAM detection
    let all_devices = enumerate_devices();

    let cuda_devices: Vec<CudaDeviceInfo> = all_devices
        .into_iter()
        .filter_map(|dev| {
            if matches!(dev.device_type, DeviceType::Cuda { .. }) {
                Some(CudaDeviceInfo {
                    name: dev.name,
                    memory_gb: dev.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                })
            } else {
                None
            }
        })
        .collect();

    if cuda_devices.is_empty() {
        return Err("No CUDA devices found".to_string());
    }

    Ok(CudaInfo {
        device_count: cuda_devices.len(),
        devices: cuda_devices,
    })
}

/// Probes CUDA device memory capacity (legacy, kept for compatibility).
///
/// Note: This function is no longer used - probe_cuda_runtime now uses
/// abaddon's device enumeration which has proper VRAM detection.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn probe_cuda_memory(device: &candle_core::Device) -> Option<f64> {
    // Try to determine GPU memory by attempting allocations
    // This is a heuristic since candle doesn't expose memory info directly
    use candle_core::{DType, Tensor};

    // Try progressively larger allocations to estimate total memory
    let test_sizes: Vec<usize> = vec![
        1 * 1024 * 1024 * 1024,  // 1 GB
        2 * 1024 * 1024 * 1024,  // 2 GB
        4 * 1024 * 1024 * 1024,  // 4 GB
        8 * 1024 * 1024 * 1024,  // 8 GB
        16 * 1024 * 1024 * 1024, // 16 GB
        24 * 1024 * 1024 * 1024, // 24 GB
        48 * 1024 * 1024 * 1024, // 48 GB
        80 * 1024 * 1024 * 1024, // 80 GB
    ];

    let mut max_successful = 0usize;

    for size in test_sizes {
        // Try to allocate a tensor of this size (in f32 elements = size/4)
        let elements = size / 4;
        if Tensor::zeros(elements, DType::F32, device).is_ok() {
            max_successful = size;
        } else {
            break;
        }
    }

    if max_successful > 0 {
        Some(max_successful as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        None
    }
}

fn check_ok(name: &str) {
    println!("  {}: \x1b[32m✓\x1b[0m", name);
}

/// Calculate directory size recursively.
fn dir_size(path: &std::path::Path) -> std::io::Result<u64> {
    let mut size = 0;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                size += dir_size(&path)?;
            } else {
                size += entry.metadata()?.len();
            }
        }
    }
    Ok(size)
}

/// Format byte size to human readable string.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

// === Chat History Persistence ===

/// Serializable chat message for persistence.
#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableMessage {
    role: String,
    content: String,
}

impl From<&Message> for SerializableMessage {
    fn from(msg: &Message) -> Self {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        Self {
            role: role.to_string(),
            content: msg.content.clone(),
        }
    }
}

impl From<SerializableMessage> for Message {
    fn from(msg: SerializableMessage) -> Self {
        let role = match msg.role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        };
        Self {
            role,
            content: msg.content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// Saves chat history to a JSON file.
fn save_chat_history(messages: &[Message], filename: &str) -> Result<()> {
    use std::path::Path;

    // Validate filename
    if filename.is_empty() {
        return Err(eyre!("Filename cannot be empty"));
    }

    // Add .json extension if not present
    let filename = if !filename.ends_with(".json") {
        format!("{}.json", filename)
    } else {
        filename.to_string()
    };

    // Check if parent directory exists
    if let Some(parent) = Path::new(&filename).parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(eyre!(
                "Directory '{}' does not exist.\nCreate it first or use a different path.",
                parent.display()
            ));
        }
    }

    let serializable: Vec<SerializableMessage> = messages.iter().map(|m| m.into()).collect();
    let json = serde_json::to_string_pretty(&serializable)?;

    std::fs::write(&filename, &json)
        .map_err(|e| eyre!("Could not write to '{}': {}", filename, e))?;

    Ok(())
}

/// Loads chat history from a JSON file.
fn load_chat_history(filename: &str) -> Result<Vec<Message>> {
    use std::path::Path;

    // Validate filename
    if filename.is_empty() {
        return Err(eyre!("Filename cannot be empty"));
    }

    // Try with .json extension if file doesn't exist
    let path = Path::new(filename);
    let filename = if !path.exists() && !filename.ends_with(".json") {
        let with_ext = format!("{}.json", filename);
        if Path::new(&with_ext).exists() {
            with_ext
        } else {
            filename.to_string()
        }
    } else {
        filename.to_string()
    };

    // Check if file exists
    if !Path::new(&filename).exists() {
        return Err(eyre!(
            "File '{}' not found.\n\
             Use /save <filename> to save a conversation first.",
            filename
        ));
    }

    let content = std::fs::read_to_string(&filename)
        .map_err(|e| eyre!("Could not read '{}': {}", filename, e))?;

    let serializable: Vec<SerializableMessage> = serde_json::from_str(&content).map_err(|e| {
        eyre!(
            "Invalid chat history format in '{}'.\n\
             Expected JSON array of messages.\n\
             Error: {}",
            filename,
            e
        )
    })?;

    if serializable.is_empty() {
        return Err(eyre!(
            "Chat history in '{}' is empty.\n\
             Nothing to load.",
            filename
        ));
    }

    // Validate message roles
    for (i, msg) in serializable.iter().enumerate() {
        let valid_roles = ["system", "user", "assistant", "tool"];
        if !valid_roles.contains(&msg.role.as_str()) {
            eprintln!(
                "\x1b[33mWarning:\x1b[0m Unknown role '{}' at message {} (treating as 'user')",
                msg.role,
                i + 1
            );
        }
    }

    Ok(serializable.into_iter().map(|m| m.into()).collect())
}

// ============================================================================
// Studio Commands
// ============================================================================

/// Get the studio data directory.
/// Uses PAIMON_WORKSPACE env var if set, otherwise falls back to system data dir.
fn studio_dir() -> std::path::PathBuf {
    if let Ok(workspace) = std::env::var("PAIMON_WORKSPACE") {
        std::path::PathBuf::from(workspace)
    } else {
        dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("infernum")
            .join("studio")
    }
}

/// Show studio statistics.
pub async fn studio_stats() -> Result<()> {
    use paimon::{DatasetManager, ExperimentTracker, ModelRegistry, PromptStudio};

    println!("\x1b[1m📊 Infernum Studio\x1b[0m");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let base_dir = studio_dir();

    // Dataset stats
    let datasets = DatasetManager::new(base_dir.join("datasets"));
    let dataset_count = datasets.count().await;

    // Experiment stats
    let experiments = ExperimentTracker::new(base_dir.join("experiments"));
    let experiment_count = experiments.count().await;

    // Prompt stats
    let prompts = PromptStudio::new(base_dir.join("prompts"));
    let prompt_count = prompts.count().await;

    // Registry stats
    let registry = ModelRegistry::new(base_dir.join("registry"));
    let model_count = registry.count().await;

    println!("\x1b[1mResources:\x1b[0m");
    println!("  Datasets:     {}", dataset_count);
    println!("  Experiments:  {}", experiment_count);
    println!("  Prompts:      {}", prompt_count);
    println!("  Models:       {}", model_count);
    println!();

    println!("\x1b[1mAgent Familiars:\x1b[0m");
    println!("  🗃️  Data Curator     - Dataset quality analysis");
    println!("  🏋️  Training Coach   - Training run monitoring");
    println!("  📈 Eval Analyst     - Benchmark interpretation");
    println!("  🔧 Hyperparam Opt   - Parameter optimization");
    println!();

    println!("\x1b[1mStorage:\x1b[0m {}", base_dir.display());
    println!();

    println!("Use '\x1b[1minfernum studio <command> --help\x1b[0m' for more info.");

    Ok(())
}

// === Dataset Commands ===

/// List datasets.
pub async fn dataset_list() -> Result<()> {
    use paimon::DatasetManager;

    let manager = DatasetManager::new(studio_dir().join("datasets"));
    let datasets = manager.list().await;

    println!("\x1b[1m📁 Datasets\x1b[0m");
    println!();

    if datasets.is_empty() {
        println!("  \x1b[90m(No datasets yet)\x1b[0m");
        println!();
        println!("Create a dataset:");
        println!("  infernum studio dataset create <name>");
    } else {
        println!(
            "  {:<30}  {:>10}  {:>12}  {}",
            "Name", "Examples", "Created", "Description"
        );
        println!(
            "  {:-<30}  {:-<10}  {:-<12}  {:-<30}",
            "", "", "", ""
        );

        for dataset in &datasets {
            let desc = dataset
                .description
                .as_deref()
                .unwrap_or("-")
                .chars()
                .take(30)
                .collect::<String>();
            println!(
                "  {:<30}  {:>10}  {:>12}  {}",
                dataset.name,
                dataset.examples.len(),
                dataset.created_at.format("%Y-%m-%d"),
                desc
            );
        }
        println!();
        println!("  {} dataset(s)", datasets.len());
    }

    Ok(())
}

/// Create a dataset.
pub async fn dataset_create(name: String, description: Option<String>) -> Result<()> {
    use paimon::{DatasetConfig, DatasetManager};

    let manager = DatasetManager::new(studio_dir().join("datasets"));

    let mut config = DatasetConfig::new(&name);
    if let Some(desc) = &description {
        config = config.with_description(desc);
    }

    let dataset = manager.create(config, Vec::new()).await
        .map_err(|e| eyre!("Failed to create dataset: {}", e))?;

    println!("\x1b[32m✓\x1b[0m Dataset '{}' created (ID: {})", name, dataset.id);
    if let Some(desc) = description {
        println!("  Description: {}", desc);
    }
    println!();
    println!("Next steps:");
    println!("  infernum studio dataset import {} <file.jsonl>", dataset.id);

    Ok(())
}

/// Import data into a dataset.
pub async fn dataset_import(name: String, file: String) -> Result<()> {
    use paimon::{DatasetConfig, DatasetManager, Example};
    use std::io::BufRead;

    let manager = DatasetManager::new(studio_dir().join("datasets"));

    // Read JSONL file first
    let file_handle =
        std::fs::File::open(&file).map_err(|e| eyre!("Could not open '{}': {}", file, e))?;

    let reader = std::io::BufReader::new(file_handle);
    let mut examples = Vec::new();
    let mut errors = 0;

    println!("Importing from {}...", file);

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| eyre!("Error reading line {}: {}", line_num + 1, e))?;

        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<Example>(&line) {
            Ok(example) => {
                examples.push(example);
            },
            Err(e) => {
                eprintln!(
                    "  \x1b[33mWarning:\x1b[0m Line {}: {}",
                    line_num + 1,
                    e
                );
                errors += 1;
            },
        }
    }

    let imported = examples.len();

    // Create dataset with examples
    let config = DatasetConfig::new(&name);
    let _dataset = manager.create(config, examples).await
        .map_err(|e| eyre!("Failed to create dataset: {}", e))?;

    println!();
    println!("\x1b[32m✓\x1b[0m Created dataset '{}' with {} examples", name, imported);
    if errors > 0 {
        println!("  \x1b[33m{} lines skipped due to errors\x1b[0m", errors);
    }

    Ok(())
}

/// Show dataset info.
pub async fn dataset_info(name: String) -> Result<()> {
    use paimon::DatasetManager;

    let manager = DatasetManager::new(studio_dir().join("datasets"));
    let datasets = manager.list().await;

    // Find by name or ID
    let dataset = datasets.iter()
        .find(|d| d.name == name || d.id == name)
        .ok_or_else(|| eyre!("Dataset '{}' not found", name))?;

    println!("\x1b[1mDataset: {}\x1b[0m", dataset.name);
    println!("ID: {}", dataset.id);
    println!();

    if let Some(desc) = &dataset.description {
        println!("Description: {}", desc);
    }

    println!("Examples:    {}", dataset.examples.len());
    println!("Created:     {}", dataset.created_at.format("%Y-%m-%d %H:%M:%S"));

    if !dataset.examples.is_empty() {
        println!();
        println!("\x1b[1mSample (first 3):\x1b[0m");
        for (i, example) in dataset.examples.iter().take(3).enumerate() {
            let input_preview: String = example.input.chars().take(50).collect();
            let output_preview: String = example.output.chars().take(50).collect();
            println!("  [{}] Input:  {}...", i + 1, input_preview);
            println!("      Output: {}...", output_preview);
        }
    }

    Ok(())
}

/// Validate dataset.
pub async fn dataset_validate(name: String) -> Result<()> {
    use paimon::DatasetManager;

    let manager = DatasetManager::new(studio_dir().join("datasets"));
    let datasets = manager.list().await;

    // Find by name or ID
    let dataset = datasets.iter()
        .find(|d| d.name == name || d.id == name)
        .ok_or_else(|| eyre!("Dataset '{}' not found", name))?;

    println!("Validating dataset '{}'...", dataset.name);
    println!();

    let report = manager.validate(&dataset.id).await
        .map_err(|e| eyre!("Validation failed: {}", e))?;

    println!("\x1b[1mValidation Report:\x1b[0m");
    println!("  Quality score: {:.1}%", report.quality_score * 100.0);
    println!("  Passed:        {}", report.passed);
    println!();

    if report.issues.is_empty() {
        println!("\x1b[32m✓\x1b[0m No issues found!");
    } else {
        println!("\x1b[33mIssues ({}):\x1b[0m", report.issues.len());
        for issue in &report.issues {
            println!("  • [{:?}] {}", issue.severity, issue.message);
        }
    }

    if !report.suggestions.is_empty() {
        println!();
        println!("\x1b[1mSuggestions:\x1b[0m");
        for suggestion in &report.suggestions {
            println!("  → {}", suggestion);
        }
    }

    Ok(())
}

/// Analyze dataset with Data Curator agent.
pub async fn dataset_analyze(name: String) -> Result<()> {
    use paimon::{DataCuratorAgent, DatasetManager};

    let manager = DatasetManager::new(studio_dir().join("datasets"));
    let datasets = manager.list().await;

    // Find by name or ID
    let dataset = datasets.iter()
        .find(|d| d.name == name || d.id == name)
        .ok_or_else(|| eyre!("Dataset '{}' not found", name))?;

    println!("🗃️  Data Curator analyzing '{}'...", dataset.name);
    println!();

    // Load full dataset with examples
    let full_dataset = manager.get(&dataset.id).await
        .map_err(|e| eyre!("Failed to load dataset: {}", e))?;
    let examples = &full_dataset.examples;

    let curator = DataCuratorAgent::new(None);

    // Run quality check
    let quality_report = curator.quality_check(&examples).await
        .map_err(|e| eyre!("Quality check failed: {}", e))?;

    // Get augmentation suggestions
    let suggestions = curator.suggest_augmentations(&examples).await
        .map_err(|e| eyre!("Augmentation analysis failed: {}", e))?;

    // Calculate stats
    let avg_input_len: f32 = examples.iter().map(|e| e.input.len()).sum::<usize>() as f32
        / examples.len().max(1) as f32;
    let avg_output_len: f32 = examples.iter().map(|e| e.output.len()).sum::<usize>() as f32
        / examples.len().max(1) as f32;

    println!("\x1b[1mDataset Analysis:\x1b[0m");
    println!();
    println!("  Total examples:    {}", quality_report.total_examples);
    println!("  Average input len: {:.0} chars", avg_input_len);
    println!("  Average output len: {:.0} chars", avg_output_len);
    println!("  Quality score:     {:.1}%", quality_report.average_score * 100.0);
    println!("  High quality:      {}", quality_report.high_quality_count);
    println!("  Low quality:       {}", quality_report.low_quality_count);
    println!();

    if !quality_report.issues.is_empty() {
        println!("\x1b[33mIssues Detected:\x1b[0m");
        for issue in quality_report.issues.iter().take(10) {
            println!("  • [{}] {}", issue.example_id.chars().take(8).collect::<String>(), issue.description);
        }
        if quality_report.issues.len() > 10 {
            println!("  ... and {} more issues", quality_report.issues.len() - 10);
        }
        println!();
    }

    if !suggestions.is_empty() {
        println!("\x1b[1mRecommendations:\x1b[0m");
        for suggestion in &suggestions {
            println!("  → {} - {}", suggestion.name, suggestion.description);
            if suggestion.recommended_count > 0 {
                println!("    (recommended: {} examples)", suggestion.recommended_count);
            }
        }
    }

    Ok(())
}

// === Experiment Commands ===

/// List experiments.
pub async fn experiment_list() -> Result<()> {
    use paimon::ExperimentTracker;

    let tracker = ExperimentTracker::new(studio_dir().join("experiments"));
    let experiments = tracker.list_experiments().await;

    println!("\x1b[1m🧪 Experiments\x1b[0m");
    println!();

    if experiments.is_empty() {
        println!("  \x1b[90m(No experiments yet)\x1b[0m");
        println!();
        println!("Create an experiment:");
        println!("  infernum studio experiment create <name>");
    } else {
        println!(
            "  {:<30}  {:>6}  {:>12}  {}",
            "Name", "Runs", "Created", "Description"
        );
        println!(
            "  {:-<30}  {:-<6}  {:-<12}  {:-<30}",
            "", "", "", ""
        );

        for exp in &experiments {
            let desc = exp
                .config
                .description
                .as_deref()
                .unwrap_or("-")
                .chars()
                .take(30)
                .collect::<String>();
            println!(
                "  {:<30}  {:>6}  {:>12}  {}",
                exp.config.name,
                exp.runs.len(),
                exp.created_at.format("%Y-%m-%d"),
                desc
            );
        }
        println!();
        println!("  {} experiment(s)", experiments.len());
    }

    Ok(())
}

/// Create an experiment.
pub async fn experiment_create(name: String, description: Option<String>) -> Result<()> {
    use paimon::{ExperimentConfig, ExperimentTracker};

    let tracker = ExperimentTracker::new(studio_dir().join("experiments"));

    // ExperimentConfig requires base_model and dataset_id
    let mut config = ExperimentConfig::new(&name, "unknown", "unknown");
    if let Some(desc) = &description {
        config = config.with_description(desc);
    }

    let experiment = tracker.create_experiment(config).await
        .map_err(|e| eyre!("Failed to create experiment: {}", e))?;

    println!("\x1b[32m✓\x1b[0m Experiment '{}' created (ID: {})", name, experiment.id);

    Ok(())
}

/// Show experiment info.
pub async fn experiment_info(name: String) -> Result<()> {
    use paimon::ExperimentTracker;

    let tracker = ExperimentTracker::new(studio_dir().join("experiments"));
    let experiments = tracker.list_experiments().await;

    // Find by name or ID
    let experiment = experiments.iter()
        .find(|e| e.config.name == name || e.id == name)
        .ok_or_else(|| eyre!("Experiment '{}' not found", name))?;

    println!("\x1b[1mExperiment: {}\x1b[0m", experiment.config.name);
    println!("ID: {}", experiment.id);
    println!();

    if let Some(desc) = &experiment.config.description {
        println!("Description: {}", desc);
    }

    println!("Runs:        {}", experiment.runs.len());
    println!("Created:     {}", experiment.created_at.format("%Y-%m-%d %H:%M:%S"));
    println!("Base model:  {}", experiment.config.base_model);
    println!("Dataset:     {}", experiment.config.dataset_id);

    if !experiment.config.tags.is_empty() {
        println!("Tags:        {}", experiment.config.tags.join(", "));
    }

    Ok(())
}

/// List runs in an experiment.
pub async fn experiment_runs(name: String) -> Result<()> {
    use paimon::ExperimentTracker;

    let tracker = ExperimentTracker::new(studio_dir().join("experiments"));
    let experiments = tracker.list_experiments().await;

    // Find by name or ID
    let experiment = experiments.iter()
        .find(|e| e.config.name == name || e.id == name)
        .ok_or_else(|| eyre!("Experiment '{}' not found", name))?;

    println!("\x1b[1mRuns in '{}'\x1b[0m", experiment.config.name);
    println!();

    if experiment.runs.is_empty() {
        println!("  \x1b[90m(No runs yet)\x1b[0m");
    } else {
        println!(
            "  {:<20}  {:>10}  {:>8}  {:>12}",
            "Run ID", "Status", "Metrics", "Started"
        );
        println!(
            "  {:-<20}  {:-<10}  {:-<8}  {:-<12}",
            "", "", "", ""
        );

        for run in &experiment.runs {
            let status = format!("{:?}", run.status);
            let id_short = if run.id.len() > 20 { &run.id[..20] } else { &run.id };
            println!(
                "  {:<20}  {:>10}  {:>8}  {:>12}",
                id_short,
                status,
                run.metrics_history.len(),
                run.started_at.format("%Y-%m-%d")
            );
        }
    }

    Ok(())
}

/// Analyze experiment with Training Coach.
pub async fn experiment_analyze(name: String) -> Result<()> {
    use paimon::{ExperimentTracker, TrainingCoachAgent, TrainingMetrics};

    let tracker = ExperimentTracker::new(studio_dir().join("experiments"));
    let experiments = tracker.list_experiments().await;

    // Find by name or ID
    let experiment = experiments.iter()
        .find(|e| e.config.name == name || e.id == name)
        .ok_or_else(|| eyre!("Experiment '{}' not found", name))?;

    if experiment.runs.is_empty() {
        return Err(eyre!("No runs to analyze in experiment '{}'", name));
    }

    println!("🏋️  Training Coach analyzing '{}'...", experiment.config.name);
    println!();

    let coach = TrainingCoachAgent::new(None);

    // Analyze the latest run
    if let Some(run) = experiment.runs.last() {
        // Extract loss history from metrics_history
        // metrics_history is HashMap<String, Vec<(step, value)>>
        let loss_history: Vec<f32> = run
            .metrics_history
            .get("loss")
            .or_else(|| run.metrics_history.get("train_loss"))
            .map(|hist| hist.iter().map(|(_, v)| *v as f32).collect())
            .unwrap_or_default();

        // Count total metric points for epoch estimate
        let total_points: usize = run.metrics_history.values().map(|v| v.len()).sum();

        let metrics = TrainingMetrics {
            current_epoch: total_points.max(1) as u32,
            train_loss: loss_history.last().cloned(),
            val_loss: run
                .metrics_history
                .get("val_loss")
                .and_then(|hist| hist.last())
                .map(|(_, v)| *v as f32),
            loss_history,
            learning_rate: run
                .metrics_history
                .get("learning_rate")
                .and_then(|hist| hist.last())
                .map(|(_, v)| *v as f32),
            gradient_norm: None,
            tokens_per_second: None,
            gpu_memory_used: None,
        };

        let analysis = coach.analyze_run(&metrics).await
            .map_err(|e| eyre!("Analysis failed: {}", e))?;

        println!("\x1b[1mTraining Analysis:\x1b[0m");
        println!();
        println!(
            "  Health: {}",
            match analysis.health {
                paimon::RunHealth::Healthy => "\x1b[32mHealthy\x1b[0m",
                paimon::RunHealth::Warning => "\x1b[33mWarning\x1b[0m",
                paimon::RunHealth::Critical => "\x1b[31mCritical\x1b[0m",
            }
        );

        if analysis.should_stop {
            println!("  \x1b[31m⚠ Recommendation: Stop training\x1b[0m");
        }

        if let Some(remaining) = analysis.estimated_epochs_remaining {
            println!("  Estimated epochs remaining: {}", remaining);
        }

        if !analysis.insights.is_empty() {
            println!();
            println!("\x1b[1mInsights:\x1b[0m");
            for insight in &analysis.insights {
                println!("  • {}", insight);
            }
        }

        if !analysis.recommended_actions.is_empty() {
            println!();
            println!("\x1b[1mRecommended Actions:\x1b[0m");
            for action in &analysis.recommended_actions {
                println!("  → {}", action);
            }
        }
    }

    Ok(())
}

// === Prompt Commands ===

/// List prompts.
pub async fn prompt_list() -> Result<()> {
    use paimon::PromptStudio;

    let studio = PromptStudio::new(studio_dir().join("prompts"));
    let templates = studio.list_templates().await;

    println!("\x1b[1m📝 Prompt Templates\x1b[0m");
    println!();

    if templates.is_empty() {
        println!("  \x1b[90m(No templates yet)\x1b[0m");
        println!();
        println!("Create a template:");
        println!("  infernum studio prompt create <name> \"<content>\"");
    } else {
        println!(
            "  {:<30}  {:>8}  {:>12}  {}",
            "Name", "Versions", "Created", "Description"
        );
        println!(
            "  {:-<30}  {:-<8}  {:-<12}  {:-<30}",
            "", "", "", ""
        );

        for template in &templates {
            let desc = template
                .description
                .as_deref()
                .unwrap_or("-")
                .chars()
                .take(30)
                .collect::<String>();
            println!(
                "  {:<30}  {:>8}  {:>12}  {}",
                template.name,
                template.versions.len(),
                template.created_at.format("%Y-%m-%d"),
                desc
            );
        }
        println!();
        println!("  {} template(s)", templates.len());
    }

    Ok(())
}

/// Create a prompt template.
pub async fn prompt_create(
    name: String,
    content: String,
    description: Option<String>,
) -> Result<()> {
    use paimon::PromptStudio;

    let studio = PromptStudio::new(studio_dir().join("prompts"));

    // Handle stdin
    let content = if content == "-" {
        use std::io::Read;
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer)?;
        buffer
    } else {
        content
    };

    // Create template
    let mut template = studio.create_template(&name).await
        .map_err(|e| eyre!("Failed to create template: {}", e))?;
    if let Some(desc) = &description {
        template = template.with_description(desc);
    }

    // Add initial version with content
    studio.add_version(&template.id, &content, "Initial version").await
        .map_err(|e| eyre!("Failed to add version: {}", e))?;

    println!("\x1b[32m✓\x1b[0m Prompt template '{}' created (ID: {})", name, template.id);

    // Show detected variables
    let vars = template.variables();
    if !vars.is_empty() {
        println!("  Variables: {}", vars.join(", "));
    }

    Ok(())
}

/// Show a prompt template.
pub async fn prompt_show(name: String, version: Option<u32>) -> Result<()> {
    use paimon::PromptStudio;

    let studio = PromptStudio::new(studio_dir().join("prompts"));

    let template = studio
        .get_template_by_name(&name).await
        .ok_or_else(|| eyre!("Template '{}' not found", name))?;

    // Get the requested version or active version
    let version_to_show = if let Some(v) = version {
        template.versions.iter()
            .find(|pv| pv.version_number == v)
            .ok_or_else(|| eyre!("Version {} not found", v))?
    } else {
        template.active_version()
            .ok_or_else(|| eyre!("No active version found"))?
    };

    println!("\x1b[1mTemplate: {} (v{})\x1b[0m", template.name, version_to_show.version_number);
    println!("ID: {}", template.id);
    println!();

    if let Some(desc) = &template.description {
        println!("Description: {}", desc);
        println!();
    }

    println!("\x1b[1mContent:\x1b[0m");
    println!("{}", version_to_show.content);

    // Show variables
    let vars = version_to_show.variables();
    if !vars.is_empty() {
        println!();
        println!("\x1b[1mVariables:\x1b[0m {}", vars.join(", "));
    }

    Ok(())
}

/// Test a prompt template.
pub async fn prompt_test(name: String, input: Option<String>) -> Result<()> {
    use paimon::PromptStudio;

    let studio = PromptStudio::new(studio_dir().join("prompts"));

    let template = studio
        .get_template_by_name(&name).await
        .ok_or_else(|| eyre!("Template '{}' not found", name))?;

    let current = template.active_version()
        .ok_or_else(|| eyre!("No active version found"))?;

    println!("\x1b[1mTesting template: {} (v{})\x1b[0m", template.name, current.version_number);
    println!();

    // Parse input variables
    let variables: std::collections::HashMap<String, String> = if let Some(json) = input {
        serde_json::from_str(&json).map_err(|e| eyre!("Invalid JSON input: {}", e))?
    } else {
        std::collections::HashMap::new()
    };

    // Render template
    let rendered = current.render(&variables)
        .map_err(|e| eyre!("Render failed: {}", e))?;

    println!("\x1b[1mRendered:\x1b[0m");
    println!("{}", rendered);

    Ok(())
}

// === Registry Commands ===

/// List registered models.
pub async fn registry_list() -> Result<()> {
    use paimon::ModelRegistry;

    let registry = ModelRegistry::new(studio_dir().join("registry"));
    let models = registry.list_models();

    println!("\x1b[1m📦 Model Registry\x1b[0m");
    println!();

    if models.is_empty() {
        println!("  \x1b[90m(No models registered)\x1b[0m");
        println!();
        println!("Register a model:");
        println!("  infernum studio registry register <name> <path>");
    } else {
        println!(
            "  {:<30}  {:>8}  {:>12}  {}",
            "Name", "Version", "Stage", "Description"
        );
        println!(
            "  {:-<30}  {:-<8}  {:-<12}  {:-<30}",
            "", "", "", ""
        );

        for model in &models {
            let latest = model.latest_version();
            let version = latest.map(|v| format!("v{}", v.version)).unwrap_or_else(|| "-".to_string());
            let stage = latest
                .map(|v| v.stage.as_str().to_string())
                .unwrap_or_else(|| "-".to_string());
            let desc = model
                .description
                .as_deref()
                .unwrap_or("-")
                .chars()
                .take(30)
                .collect::<String>();

            println!(
                "  {:<30}  {:>8}  {:>12}  {}",
                model.name, version, stage, desc
            );
        }
        println!();
        println!("  {} model(s)", models.len());
    }

    Ok(())
}

/// Register a model.
pub async fn registry_register(
    name: String,
    path: String,
    description: Option<String>,
) -> Result<()> {
    use paimon::{Model, ModelMetadata, ModelRegistry};

    let registry = ModelRegistry::new(studio_dir().join("registry"));

    // Create a model
    let mut model = Model::new(&name, "unknown", "text-generation");
    if let Some(desc) = &description {
        model = model.with_description(desc);
    }

    // Create initial version with metadata
    let metadata = ModelMetadata::new();
    let version = model.create_version(metadata);
    let version_id = version.id.clone();

    // Register the model
    let model_id = registry.register_model(model)
        .map_err(|e| eyre!("Failed to register model: {}", e))?;

    println!("\x1b[32m✓\x1b[0m Model '{}' registered (ID: {})", name, model_id);
    println!("  Path: {}", path);
    println!("  Version: {}", version_id);

    Ok(())
}

/// Show model info.
pub async fn registry_info(name: String) -> Result<()> {
    use paimon::ModelRegistry;

    let registry = ModelRegistry::new(studio_dir().join("registry"));

    let model = registry
        .get_model_by_name(&name)
        .ok_or_else(|| eyre!("Model '{}' not found", name))?;

    println!("\x1b[1mModel: {}\x1b[0m", model.name);
    println!("ID: {}", model.id);
    println!();

    if let Some(desc) = &model.description {
        println!("Description: {}", desc);
    }

    println!("Base model:  {}", model.base_model);
    println!("Task type:   {}", model.task_type);
    println!("Versions:    {}", model.versions.len());
    println!("Created:     {}", model.created_at.format("%Y-%m-%d %H:%M:%S"));

    if !model.versions.is_empty() {
        println!();
        println!("\x1b[1mVersions:\x1b[0m");
        for version in &model.versions {
            println!(
                "  v{}: {} ({})",
                version.version,
                version.stage.as_str(),
                version.created_at.format("%Y-%m-%d")
            );
        }
    }

    Ok(())
}

/// Promote model to a stage.
pub async fn registry_promote(name: String, stage: String) -> Result<()> {
    use paimon::{ModelRegistry, ModelStage};

    let registry = ModelRegistry::new(studio_dir().join("registry"));

    let target_stage = match stage.to_lowercase().as_str() {
        "staging" => ModelStage::Staging,
        "production" => ModelStage::Production,
        "archived" => ModelStage::Archived,
        "development" => ModelStage::Development,
        _ => return Err(eyre!("Invalid stage: {}. Use: staging, production, archived", stage)),
    };

    let model = registry
        .get_model_by_name(&name)
        .ok_or_else(|| eyre!("Model '{}' not found", name))?;

    let latest_version = model
        .latest_version()
        .ok_or_else(|| eyre!("Model has no versions"))?
        .version;

    registry.transition_stage(&model.id, latest_version, target_stage, Some("CLI promotion".to_string()))
        .map_err(|e| eyre!("Transition failed: {}", e))?;

    println!(
        "\x1b[32m✓\x1b[0m Model '{}' v{} promoted to {}",
        name, latest_version, target_stage.as_str()
    );

    Ok(())
}

/// Get improvement roadmap from Eval Analyst.
pub async fn registry_roadmap(name: String) -> Result<()> {
    use paimon::{
        analyst::{BenchmarkResults, BenchmarkScore},
        EvalAnalystAgent, ModelRegistry,
    };

    let registry = ModelRegistry::new(studio_dir().join("registry"));

    let _model = registry
        .get_model_by_name(&name)
        .ok_or_else(|| eyre!("Model '{}' not found", name))?;

    println!("📈 Eval Analyst creating roadmap for '{}'...", name);
    println!();

    let analyst = EvalAnalystAgent::new(None);

    // Create mock benchmark results for demonstration
    // In production, this would load actual benchmark results
    let results = BenchmarkResults {
        model_name: name.clone(),
        benchmarks: vec![
            BenchmarkScore {
                name: "General".to_string(),
                score: 0.75,
                category: Some("Quality".to_string()),
                description: Some("Overall model quality".to_string()),
                test_cases: Some(100),
            },
            BenchmarkScore {
                name: "Accuracy".to_string(),
                score: 0.80,
                category: Some("Quality".to_string()),
                description: Some("Response accuracy".to_string()),
                test_cases: Some(50),
            },
        ],
        evaluated_at: chrono::Utc::now(),
    };

    let plan = analyst.improvement_roadmap(&results).await
        .map_err(|e| eyre!("Roadmap generation failed: {}", e))?;

    println!("\x1b[1m{}\x1b[0m", plan.title);
    println!();
    println!("{}", plan.summary);
    println!();

    println!("\x1b[1mImprovement Steps:\x1b[0m");
    for step in &plan.steps {
        println!();
        println!("  \x1b[1mStep {}:\x1b[0m {}", step.step, step.action);
        println!("    Rationale: {}", step.rationale);
        println!("    Impact: {}", step.impact);
    }

    println!();
    println!("\x1b[1mExpected Outcome:\x1b[0m {}", plan.expected_outcome);
    println!("\x1b[1mEstimated Effort:\x1b[0m {}", plan.estimated_effort);

    Ok(())
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // === format_size tests ===

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 bytes");
        assert_eq!(format_size(1), "1 bytes");
        assert_eq!(format_size(100), "100 bytes");
        assert_eq!(format_size(1023), "1023 bytes");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1536), "1.50 KB");
        assert_eq!(format_size(2048), "2.00 KB");
        assert_eq!(format_size(1024 * 1023), "1023.00 KB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(1024 * 1024 + 512 * 1024), "1.50 MB");
        assert_eq!(format_size(100 * 1024 * 1024), "100.00 MB");
    }

    #[test]
    fn test_format_size_gigabytes() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.00 GB");
        assert_eq!(format_size(10 * 1024 * 1024 * 1024), "10.00 GB");
    }

    // === dir_size tests ===

    #[test]
    fn test_dir_size_empty_dir() {
        let temp = TempDir::new().expect("temp dir");
        let size = dir_size(temp.path()).expect("dir_size");
        assert_eq!(size, 0);
    }

    #[test]
    fn test_dir_size_with_file() {
        let temp = TempDir::new().expect("temp dir");
        let file_path = temp.path().join("test.txt");
        std::fs::write(&file_path, "Hello, World!").expect("write file");

        let size = dir_size(temp.path()).expect("dir_size");
        assert_eq!(size, 13); // "Hello, World!" = 13 bytes
    }

    #[test]
    fn test_dir_size_nested_dirs() {
        let temp = TempDir::new().expect("temp dir");

        // Create nested structure
        let subdir = temp.path().join("subdir");
        std::fs::create_dir(&subdir).expect("create subdir");

        std::fs::write(temp.path().join("file1.txt"), "12345").expect("write");
        std::fs::write(subdir.join("file2.txt"), "67890").expect("write");

        let size = dir_size(temp.path()).expect("dir_size");
        assert_eq!(size, 10); // 5 + 5 = 10 bytes
    }

    #[test]
    fn test_dir_size_nonexistent_path() {
        let result = dir_size(&PathBuf::from("/nonexistent/path/12345"));
        assert!(result.is_ok()); // dir_size returns 0 for non-dirs
    }

    // === SerializableMessage conversion tests ===

    #[test]
    fn test_serializable_message_from_user() {
        let msg = Message {
            role: Role::User,
            content: "Hello!".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let serializable = SerializableMessage::from(&msg);
        assert_eq!(serializable.role, "user");
        assert_eq!(serializable.content, "Hello!");
    }

    #[test]
    fn test_serializable_message_from_system() {
        let msg = Message {
            role: Role::System,
            content: "You are a helpful assistant.".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let serializable = SerializableMessage::from(&msg);
        assert_eq!(serializable.role, "system");
    }

    #[test]
    fn test_serializable_message_from_assistant() {
        let msg = Message {
            role: Role::Assistant,
            content: "I can help with that.".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let serializable = SerializableMessage::from(&msg);
        assert_eq!(serializable.role, "assistant");
    }

    #[test]
    fn test_serializable_message_from_tool() {
        let msg = Message {
            role: Role::Tool,
            content: "{\"result\": 42}".to_string(),
            name: Some("calculator".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_123".to_string()),
        };

        let serializable = SerializableMessage::from(&msg);
        assert_eq!(serializable.role, "tool");
    }

    #[test]
    fn test_message_from_serializable() {
        let serializable = SerializableMessage {
            role: "user".to_string(),
            content: "Test message".to_string(),
        };

        let msg: Message = serializable.into();
        assert!(matches!(msg.role, Role::User));
        assert_eq!(msg.content, "Test message");
    }

    #[test]
    fn test_message_from_serializable_unknown_role() {
        let serializable = SerializableMessage {
            role: "unknown".to_string(),
            content: "Test".to_string(),
        };

        let msg: Message = serializable.into();
        // Unknown roles default to User
        assert!(matches!(msg.role, Role::User));
    }

    #[test]
    fn test_message_roundtrip() {
        let original = Message {
            role: Role::Assistant,
            content: "Roundtrip test!".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let serializable = SerializableMessage::from(&original);
        let json = serde_json::to_string(&serializable).expect("serialize");
        let parsed: SerializableMessage = serde_json::from_str(&json).expect("deserialize");
        let restored: Message = parsed.into();

        assert!(matches!(restored.role, Role::Assistant));
        assert_eq!(restored.content, "Roundtrip test!");
    }

    // === save_chat_history / load_chat_history tests ===

    #[test]
    fn test_save_chat_history_empty_filename() {
        let messages = vec![];
        let result = save_chat_history(&messages, "");
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("empty"));
    }

    #[test]
    fn test_save_chat_history_adds_json_extension() {
        let temp = TempDir::new().expect("temp dir");
        let filename = temp.path().join("chat");

        let messages = vec![Message {
            role: Role::User,
            content: "Hello".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        save_chat_history(&messages, filename.to_str().unwrap()).expect("save");

        // Check that .json was added
        let json_path = temp.path().join("chat.json");
        assert!(json_path.exists());
    }

    #[test]
    fn test_save_chat_history_preserves_json_extension() {
        let temp = TempDir::new().expect("temp dir");
        let filename = temp.path().join("chat.json");

        let messages = vec![Message {
            role: Role::User,
            content: "Hello".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        save_chat_history(&messages, filename.to_str().unwrap()).expect("save");

        // Check that only one .json exists (not .json.json)
        assert!(filename.exists());
        assert!(!temp.path().join("chat.json.json").exists());
    }

    #[test]
    fn test_save_chat_history_nonexistent_parent() {
        let messages = vec![];
        let result = save_chat_history(&messages, "/nonexistent/dir/chat.json");
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("does not exist"));
    }

    #[test]
    fn test_load_chat_history_empty_filename() {
        let result = load_chat_history("");
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("empty"));
    }

    #[test]
    fn test_load_chat_history_file_not_found() {
        let result = load_chat_history("/nonexistent/path/chat.json");
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("not found"));
    }

    #[test]
    fn test_load_chat_history_tries_json_extension() {
        let temp = TempDir::new().expect("temp dir");
        let json_path = temp.path().join("chat.json");

        // Create file with .json extension
        let content = r#"[{"role": "user", "content": "Hello"}]"#;
        std::fs::write(&json_path, content).expect("write");

        // Load without extension
        let filename = temp.path().join("chat");
        let messages = load_chat_history(filename.to_str().unwrap()).expect("load");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "Hello");
    }

    #[test]
    fn test_load_chat_history_invalid_json() {
        let temp = TempDir::new().expect("temp dir");
        let filename = temp.path().join("chat.json");

        std::fs::write(&filename, "not valid json").expect("write");

        let result = load_chat_history(filename.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("Invalid chat history"));
    }

    #[test]
    fn test_load_chat_history_empty_array() {
        let temp = TempDir::new().expect("temp dir");
        let filename = temp.path().join("chat.json");

        std::fs::write(&filename, "[]").expect("write");

        let result = load_chat_history(filename.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("empty"));
    }

    #[test]
    fn test_save_and_load_chat_history_roundtrip() {
        let temp = TempDir::new().expect("temp dir");
        let filename = temp.path().join("chat.json");

        let messages = vec![
            Message {
                role: Role::System,
                content: "You are helpful.".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: "Hello!".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: Role::Assistant,
                content: "Hi there!".to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        save_chat_history(&messages, filename.to_str().unwrap()).expect("save");
        let loaded = load_chat_history(filename.to_str().unwrap()).expect("load");

        assert_eq!(loaded.len(), 3);
        assert!(matches!(loaded[0].role, Role::System));
        assert!(matches!(loaded[1].role, Role::User));
        assert!(matches!(loaded[2].role, Role::Assistant));
        assert_eq!(loaded[0].content, "You are helpful.");
        assert_eq!(loaded[1].content, "Hello!");
        assert_eq!(loaded[2].content, "Hi there!");
    }

    // === studio_dir tests ===

    #[test]
    fn test_studio_dir_with_env_var() {
        std::env::set_var("PAIMON_WORKSPACE", "/custom/studio/path");
        let dir = studio_dir();
        assert_eq!(dir, PathBuf::from("/custom/studio/path"));
        std::env::remove_var("PAIMON_WORKSPACE");
    }

    #[test]
    fn test_studio_dir_default() {
        std::env::remove_var("PAIMON_WORKSPACE");
        let dir = studio_dir();
        // Should end with infernum/studio
        assert!(dir.ends_with("studio") || dir.ends_with("infernum/studio"));
    }

    // === rustc_version tests ===

    #[test]
    fn test_rustc_version_not_empty() {
        let version = rustc_version();
        assert!(!version.is_empty());
    }

    // === ModelCacheInfo tests ===

    #[test]
    fn test_model_cache_info_creation() {
        let temp = TempDir::new().expect("temp dir");
        let info = get_model_cache_info("test-model", temp.path());

        assert_eq!(info.name, "test-model");
        assert_eq!(info.size_bytes, 0);
        assert_eq!(info.size_str, "0 bytes");
        assert_eq!(info.context_str, "-");
        assert_eq!(info.architecture, "-");
    }

    #[test]
    fn test_model_cache_info_with_config() {
        let temp = TempDir::new().expect("temp dir");

        // Create snapshots directory with config.json
        let snapshot_dir = temp.path().join("snapshots").join("abc123");
        std::fs::create_dir_all(&snapshot_dir).expect("create dirs");

        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 4096
        });
        std::fs::write(
            snapshot_dir.join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .expect("write config");

        let info = get_model_cache_info("llama-model", temp.path());

        assert_eq!(info.name, "llama-model");
        assert_eq!(info.context_str, "4K");
        assert_eq!(info.architecture, "Llama");
    }

    #[test]
    fn test_model_cache_info_with_size() {
        let temp = TempDir::new().expect("temp dir");

        // Create a file to give size
        std::fs::write(temp.path().join("weights.bin"), vec![0u8; 1024]).expect("write");

        let info = get_model_cache_info("model-with-weights", temp.path());

        assert_eq!(info.size_bytes, 1024);
        assert_eq!(info.size_str, "1.00 KB");
    }

    // === check_ok tests ===

    #[test]
    fn test_check_ok_prints_correctly() {
        // This just verifies it doesn't panic
        check_ok("Test Check");
    }

    // === Chat history JSON format tests ===

    #[test]
    fn test_chat_history_json_format() {
        let temp = TempDir::new().expect("temp dir");
        let filename = temp.path().join("chat.json");

        let messages = vec![Message {
            role: Role::User,
            content: "Hello".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        save_chat_history(&messages, filename.to_str().unwrap()).expect("save");

        let content = std::fs::read_to_string(&filename).expect("read");
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&content).expect("parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["role"], "user");
        assert_eq!(parsed[0]["content"], "Hello");
    }

    #[test]
    fn test_load_chat_history_with_various_roles() {
        let temp = TempDir::new().expect("temp dir");
        let filename = temp.path().join("chat.json");

        let content = r#"[
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "tool", "content": "Tool result"}
        ]"#;
        std::fs::write(&filename, content).expect("write");

        let messages = load_chat_history(filename.to_str().unwrap()).expect("load");

        assert_eq!(messages.len(), 4);
        assert!(matches!(messages[0].role, Role::System));
        assert!(matches!(messages[1].role, Role::User));
        assert!(matches!(messages[2].role, Role::Assistant));
        assert!(matches!(messages[3].role, Role::Tool));
    }

    // === Edge case tests ===

    #[test]
    fn test_format_size_large_values() {
        // Test with realistic model sizes
        let size_2gb = 2 * 1024 * 1024 * 1024u64;
        let size_7gb = 7 * 1024 * 1024 * 1024u64;
        let size_70gb = 70 * 1024 * 1024 * 1024u64;

        assert_eq!(format_size(size_2gb), "2.00 GB");
        assert_eq!(format_size(size_7gb), "7.00 GB");
        assert_eq!(format_size(size_70gb), "70.00 GB");
    }

    #[test]
    fn test_dir_size_deeply_nested() {
        let temp = TempDir::new().expect("temp dir");

        // Create deeply nested structure
        let deep_path = temp
            .path()
            .join("a")
            .join("b")
            .join("c")
            .join("d")
            .join("e");
        std::fs::create_dir_all(&deep_path).expect("create dirs");
        std::fs::write(deep_path.join("file.txt"), "content").expect("write");

        let size = dir_size(temp.path()).expect("dir_size");
        assert_eq!(size, 7); // "content" = 7 bytes
    }

    #[test]
    fn test_save_chat_history_unicode_content() {
        let temp = TempDir::new().expect("temp dir");
        let filename = temp.path().join("chat.json");

        let messages = vec![Message {
            role: Role::User,
            content: "Hello 世界! 🎉 مرحبا".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        save_chat_history(&messages, filename.to_str().unwrap()).expect("save");
        let loaded = load_chat_history(filename.to_str().unwrap()).expect("load");

        assert_eq!(loaded[0].content, "Hello 世界! 🎉 مرحبا");
    }

    #[test]
    fn test_save_chat_history_special_chars_in_content() {
        let temp = TempDir::new().expect("temp dir");
        let filename = temp.path().join("chat.json");

        let messages = vec![Message {
            role: Role::Assistant,
            content: "Code: ```rust\nfn main() { println!(\"Hello\"); }\n```".to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        save_chat_history(&messages, filename.to_str().unwrap()).expect("save");
        let loaded = load_chat_history(filename.to_str().unwrap()).expect("load");

        assert!(loaded[0].content.contains("println!"));
        assert!(loaded[0].content.contains("```rust"));
    }

    #[test]
    fn test_save_chat_history_long_content() {
        let temp = TempDir::new().expect("temp dir");
        let filename = temp.path().join("chat.json");

        let long_content = "a".repeat(100_000);
        let messages = vec![Message {
            role: Role::User,
            content: long_content.clone(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        save_chat_history(&messages, filename.to_str().unwrap()).expect("save");
        let loaded = load_chat_history(filename.to_str().unwrap()).expect("load");

        assert_eq!(loaded[0].content.len(), 100_000);
    }

    #[test]
    fn test_model_cache_info_missing_config_fields() {
        let temp = TempDir::new().expect("temp dir");

        // Create snapshots directory with incomplete config.json
        let snapshot_dir = temp.path().join("snapshots").join("xyz789");
        std::fs::create_dir_all(&snapshot_dir).expect("create dirs");

        let config = serde_json::json!({
            "hidden_size": 4096
            // Missing architectures and max_position_embeddings
        });
        std::fs::write(
            snapshot_dir.join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .expect("write config");

        let info = get_model_cache_info("incomplete-model", temp.path());

        // Should fallback to defaults
        assert_eq!(info.context_str, "-");
        assert_eq!(info.architecture, "-");
    }

    #[test]
    fn test_model_cache_info_invalid_config_json() {
        let temp = TempDir::new().expect("temp dir");

        // Create snapshots directory with invalid JSON
        let snapshot_dir = temp.path().join("snapshots").join("invalid");
        std::fs::create_dir_all(&snapshot_dir).expect("create dirs");
        std::fs::write(snapshot_dir.join("config.json"), "not valid json").expect("write");

        let info = get_model_cache_info("invalid-config", temp.path());

        // Should fallback gracefully
        assert_eq!(info.context_str, "-");
        assert_eq!(info.architecture, "-");
    }
}
