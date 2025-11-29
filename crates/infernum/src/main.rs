//! # Infernum CLI
//!
//! *"From the depths, intelligence rises"*
//!
//! The main command-line interface for the Infernum ecosystem.

use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;

mod commands;

#[derive(Parser)]
#[command(name = "infernum")]
#[command(author = "Daemoniorum Engineering")]
#[command(version)]
#[command(about = "Blazingly fast LLM inference ecosystem", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info", global = true)]
    log_level: String,

    /// Enable JSON logging
    #[arg(long, global = true)]
    json_logs: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server
    Serve {
        /// Host to bind to
        #[arg(short = 'H', long, default_value = "0.0.0.0")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Model to load (HuggingFace repo ID or local path)
        #[arg(short, long)]
        model: Option<String>,

        /// Configuration file
        #[arg(short, long)]
        config: Option<String>,
    },

    /// Run inference on a prompt
    Generate {
        /// The prompt to generate from
        prompt: String,

        /// Model to use
        #[arg(short, long)]
        model: Option<String>,

        /// Maximum tokens to generate
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: u32,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Stream output
        #[arg(short, long)]
        stream: bool,
    },

    /// Generate embeddings
    Embed {
        /// Text to embed
        text: String,

        /// Model to use
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Manage models
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Start an interactive chat session
    Chat {
        /// Model to use
        #[arg(short, long)]
        model: Option<String>,

        /// System prompt
        #[arg(short, long)]
        system: Option<String>,
    },

    /// Display version and build info
    Version,
}

#[derive(Subcommand)]
enum ModelAction {
    /// List available models
    List,

    /// Download a model
    Pull {
        /// Model identifier (HuggingFace repo ID)
        model: String,

        /// Specific revision to download
        #[arg(short, long)]
        revision: Option<String>,
    },

    /// Show model information
    Info {
        /// Model identifier
        model: String,
    },

    /// Remove a cached model
    Remove {
        /// Model identifier
        model: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();

    // Initialize logging
    let telemetry_config = dantalion::TelemetryConfig::new("infernum")
        .with_log_level(&cli.log_level);

    let telemetry_config = if cli.json_logs {
        telemetry_config.with_json_logs()
    } else {
        telemetry_config
    };

    dantalion::init_logging(&telemetry_config);

    match cli.command {
        Commands::Serve {
            host,
            port,
            model,
            config,
        } => {
            commands::serve(host, port, model, config).await?;
        }

        Commands::Generate {
            prompt,
            model,
            max_tokens,
            temperature,
            stream,
        } => {
            commands::generate(prompt, model, max_tokens, temperature, stream).await?;
        }

        Commands::Embed { text, model } => {
            commands::embed(text, model).await?;
        }

        Commands::Model { action } => match action {
            ModelAction::List => commands::model_list().await?,
            ModelAction::Pull { model, revision } => commands::model_pull(model, revision).await?,
            ModelAction::Info { model } => commands::model_info(model).await?,
            ModelAction::Remove { model } => commands::model_remove(model).await?,
        },

        Commands::Chat { model, system } => {
            commands::chat(model, system).await?;
        }

        Commands::Version => {
            commands::version();
        }
    }

    Ok(())
}
