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

    tracing::info!("Starting Infernum server...");

    let addr = format!("{}:{}", host, port).parse()?;
    let config = ServerConfig {
        addr,
        cors: true,
        model,
        max_concurrent_requests: 64,
    };

    let server = Server::new(config);
    server.run().await?;

    Ok(())
}

/// Generate text from a prompt.
pub async fn generate(
    prompt: String,
    model: Option<String>,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
) -> Result<()> {
    let model_id = model.ok_or_else(|| eyre!("Model is required. Use --model <model>"))?;

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
    let request = GenerateRequest::new(prompt.clone())
        .with_sampling(sampling);

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
                }
                Err(e) => {
                    eprintln!("\nError during generation: {}", e);
                    break;
                }
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
    println!("Dimensions: {}", response.data.first().map(|e| {
        match &e.embedding {
            infernum_core::response::EmbeddingData::Float(v) => v.len(),
            infernum_core::response::EmbeddingData::Base64(_) => 0,
        }
    }).unwrap_or(0));

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

    println!("Cached models:\n");

    // Check HuggingFace cache directory
    let cache_dir = dirs::cache_dir()
        .map(|p| p.join("huggingface").join("hub"))
        .unwrap_or_else(|| PathBuf::from("~/.cache/huggingface/hub"));

    if cache_dir.exists() {
        let mut found = false;
        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("models--") {
                    let model_name = name_str
                        .strip_prefix("models--")
                        .unwrap_or(&name_str)
                        .replace("--", "/");
                    println!("  {}", model_name);
                    found = true;
                }
            }
        }
        if !found {
            println!("  (No models cached yet)");
        }
    } else {
        println!("  (No models cached yet)");
    }

    println!("\nUse 'infernum model pull <model>' to download a model.");
    println!("Example: infernum model pull meta-llama/Llama-3.2-1B-Instruct");

    Ok(())
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

    // Download key files
    let files_to_download = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
    ];

    let progress = ProgressBar::new(files_to_download.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    for file in files_to_download {
        progress.set_message(format!("Downloading {}...", file));
        match repo.get(file) {
            Ok(path) => {
                tracing::debug!("Downloaded {} to {:?}", file, path);
            }
            Err(e) => {
                tracing::debug!("File {} not found or error: {}", file, e);
            }
        }
        progress.inc(1);
    }

    progress.finish_with_message("Download complete!");
    println!("\nModel {} is now cached.", model);
    println!("Use 'infernum generate --model {}' to run inference.", model);

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
                    arch.first()
                        .and_then(|a| a.as_str())
                        .unwrap_or("Unknown")
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
            if let Some(ctx) = config.get("max_position_embeddings").and_then(|c| c.as_u64()) {
                println!("Max context length: {}", ctx);
            }

            println!("\nCache location: {:?}", path.parent().unwrap_or(&path));
        }
        Err(e) => {
            println!("Could not fetch model info: {}", e);
            println!("The model may need to be downloaded first.");
            println!("Use: infernum model pull {}", model);
        }
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
    let model_id = model.ok_or_else(|| eyre!("Model is required. Use --model <model>"))?;

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
    println!("\nType 'exit' or 'quit' to end the session.");
    println!("Type '/clear' to clear conversation history.\n");

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

        if input.eq_ignore_ascii_case("/clear") {
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
                        }
                        Err(e) => {
                            eprintln!("\nError: {}", e);
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Error generating response: {}", e);
                // Remove the last user message on error
                messages.pop();
                continue;
            }
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

/// Display version information.
pub fn version() {
    println!("Infernum {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("From the depths, intelligence rises.");
    println!();
    println!("Components:");
    println!("  Abaddon    - Inference Engine");
    println!("  Malphas    - Orchestration Layer");
    println!("  Stolas     - Knowledge Engine");
    println!("  Beleth     - Agent Framework");
    println!("  Asmodeus   - Adaptation Layer");
    println!("  Dantalion  - Observability");
    println!();
    println!("Daemoniorum, LLC - Building Tomorrow's Intelligence");
}
