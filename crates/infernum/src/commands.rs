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
    let model_id = model.ok_or_else(|| eyre!(
        "Model is required.\n\n\
         Options:\n  \
         1. Specify on command line: --model <model>\n  \
         2. Set a default: infernum config set-model <model>\n  \
         3. Set environment variable: INFERNUM_DEFAULT_MODEL=<model>\n\n\
         Example models:\n  \
         - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (small, fast)\n  \
         - meta-llama/Llama-3.2-3B-Instruct (requires HuggingFace login)"
    ))?;

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

    // Required files - model won't work without these
    let required_files = ["config.json", "tokenizer.json"];

    // Optional files - may or may not exist depending on model
    let optional_files = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ];

    // Weight files - try single file first, then sharded
    let weight_files = [
        "model.safetensors",
        "model.safetensors.index.json",
    ];

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
            }
            Err(e) => {
                println!("\x1b[31m✗\x1b[0m ({})", e);
                failed.push(file.to_string());
            }
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
            }
            Err(_) => {
                println!("\x1b[33m-\x1b[0m (optional, skipped)");
            }
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
        }
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
                        if let Ok(index_json) = serde_json::from_str::<serde_json::Value>(&index_content) {
                            if let Some(weight_map) = index_json.get("weight_map").and_then(|w| w.as_object()) {
                                // Get unique shard files
                                let mut shard_files: Vec<String> = weight_map
                                    .values()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect();
                                shard_files.sort();
                                shard_files.dedup();

                                println!("\n  Found {} weight shards to download:", shard_files.len());
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
                                        }
                                        Err(_) => {
                                            shard_errors += 1;
                                        }
                                    }
                                    shard_progress.inc(1);
                                }
                                shard_progress.finish_and_clear();

                                if shard_errors == 0 {
                                    println!("  \x1b[32mAll {} shards downloaded successfully.\x1b[0m", shard_files.len());
                                    has_weights = true;
                                } else {
                                    println!("  \x1b[31m{} of {} shards failed to download.\x1b[0m", shard_errors, shard_files.len());
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    println!("\x1b[31m✗\x1b[0m");
                    failed.push("model weights".to_string());
                }
            }
        }
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
    println!("Downloaded {} files for model '{}'", downloaded.len(), model);
    println!("\nUse 'infernum generate --model {}' to run inference.", model);

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
    let model_id = model.ok_or_else(|| eyre!(
        "Model is required.\n\n\
         Options:\n  \
         1. Specify on command line: --model <model>\n  \
         2. Set a default: infernum config set-model <model>\n  \
         3. Set environment variable: INFERNUM_DEFAULT_MODEL=<model>\n\n\
         Example models:\n  \
         - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (small, fast)\n  \
         - meta-llama/Llama-3.2-3B-Instruct (requires HuggingFace login)"
    ))?;

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
                }
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
                }
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
                        println!("{}[{}] {}:\x1b[0m {}", role_color, i + 1, role_name, preview);
                    }
                    println!("--- {} messages ---\n", messages.len());
                    continue;
                }
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
                }
                "/load" => {
                    if let Some(filename) = arg {
                        match load_chat_history(filename) {
                            Ok(loaded_messages) => {
                                messages = loaded_messages;
                                println!("\nLoaded {} messages from '{}'\n", messages.len(), filename);
                            }
                            Err(e) => eprintln!("\nFailed to load: {}\n", e),
                        }
                    } else {
                        eprintln!("\nUsage: /load <filename>\n");
                    }
                    continue;
                }
                _ => {
                    eprintln!("\nUnknown command: {}\nType /help for available commands.\n", cmd);
                    continue;
                }
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
    let serializable: Vec<SerializableMessage> = messages.iter().map(|m| m.into()).collect();
    let json = serde_json::to_string_pretty(&serializable)?;
    std::fs::write(filename, json)?;
    Ok(())
}

/// Loads chat history from a JSON file.
fn load_chat_history(filename: &str) -> Result<Vec<Message>> {
    let content = std::fs::read_to_string(filename)?;
    let serializable: Vec<SerializableMessage> = serde_json::from_str(&content)?;
    Ok(serializable.into_iter().map(|m| m.into()).collect())
}
