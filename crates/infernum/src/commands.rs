//! CLI command implementations.

use std::io::{self, Write};
use std::sync::Arc;

use color_eyre::eyre::{eyre, Result};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};

use abaddon::{Engine, EngineConfig, InferenceEngine};
use infernum_core::{GenerateRequest, Message, Role, SamplingParams};

/// Start the inference server.
pub async fn serve(
    host: String,
    port: u16,
    model: Option<String>,
    _config: Option<String>,
) -> Result<()> {
    use infernum_server::{Server, ServerConfig};

    // Require a model - either from args or prompt interactively
    let model = match model {
        Some(m) => m,
        None => prompt_for_model()?,
    };

    println!();
    println!("\x1b[1m🚀 Starting Infernum Server\x1b[0m");
    println!("   Model: {}", model);
    println!("   Address: http://{}:{}", host, port);
    println!();

    let addr = format!("{}:{}", host, port).parse()?;
    let config = ServerConfig {
        addr,
        cors: true,
        model: Some(model),
        max_concurrent_requests: 64,
    };

    let server = Server::new(config);
    server.run().await?;

    Ok(())
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
            .unwrap(),
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
// TODO: Re-enable when embedding models are supported
#[allow(dead_code)]
pub async fn embed(text: String, model: Option<String>) -> Result<()> {
    let model_id = model.ok_or_else(|| eyre!("Model is required. Use --model <model>"))?;

    // Show loading indicator
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
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
                                        .unwrap()
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
            .unwrap(),
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
    println!("\x1b[1mObjective:\x1b[0m {}", objective);
    println!("\x1b[1mModel:\x1b[0m {}", model_id);
    println!("\x1b[1mMax iterations:\x1b[0m {}", max_iterations);
    println!();

    // Load model
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .expect("valid template"),
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
    let tools = ToolRegistry::with_builtins();

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
        .engine(engine);

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
        // TODO: Actually probe CUDA availability at runtime
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
    if issues.is_empty() {
        println!("\x1b[1mQuick Start:\x1b[0m");
        println!("  infernum config set-model TinyLlama/TinyLlama-1.1B-Chat-v1.0");
        println!("  infernum chat");
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
