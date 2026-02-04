//! # Infernum CLI
//!
//! *"From the depths, intelligence rises"*
//!
//! The main command-line interface for the Infernum ecosystem.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use color_eyre::eyre::Result;
use std::io;

mod commands;
mod config;

#[derive(Parser)]
#[command(name = "infernum")]
#[command(author = "Daemoniorum Engineering")]
#[command(version)]
#[command(about = "Blazingly fast LLM inference ecosystem")]
#[command(long_about = "Blazingly fast LLM inference ecosystem.\n\n\
    Run without arguments to start interactive chat mode.\n\n\
    Examples:\n  \
    infernum                              # Start interactive chat\n  \
    infernum -m llama                     # Chat with specific model\n  \
    infernum serve --model llama          # Start API server\n  \
    infernum agent \"solve this puzzle\"   # Run autonomous agent")]
#[command(propagate_version = true)]
struct Cli {
    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info", global = true)]
    log_level: String,

    /// Enable JSON logging
    #[arg(long, global = true)]
    json_logs: bool,

    /// Model to use (for default interactive mode)
    #[arg(short, long, global = true)]
    model: Option<String>,

    /// System prompt (for default interactive mode)
    #[arg(short, long, global = true)]
    system: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server
    #[cfg(feature = "server")]
    Serve {
        /// Host to bind to
        #[arg(short = 'H', long, default_value = "0.0.0.0")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Model to load (HuggingFace repo ID, local path, or HCT directory)
        #[arg(short, long)]
        model: Option<String>,

        /// Configuration file
        #[arg(short, long)]
        config: Option<String>,

        /// Enable HoloTensor progressive quality inference.
        /// This enables 70B+ models on 24GB VRAM via holographic compression.
        /// The model must be a HoloTensor HCT directory.
        #[arg(long)]
        holo: bool,

        /// Minimum quality for HoloTensor inference (0.0-1.0).
        /// Start generating at this quality level. Default: 0.7
        #[arg(long, default_value = "0.7")]
        holo_min_quality: f32,

        /// Target quality for HoloTensor inference (0.0-1.0).
        /// Progressively improve to this quality during generation. Default: 0.95
        #[arg(long, default_value = "0.95")]
        holo_target_quality: f32,
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

    /// Check system configuration and dependencies
    Doctor,

    /// Interactive first-time setup wizard
    Setup {
        /// Skip hardware detection
        #[arg(long)]
        skip_detect: bool,

        /// Skip model download
        #[arg(long)]
        skip_download: bool,

        /// Skip inference test
        #[arg(long)]
        skip_test: bool,

        /// Accept defaults without prompting (non-interactive mode)
        #[arg(short, long)]
        yes: bool,
    },

    /// Run an autonomous agent with tools
    Agent {
        /// Objective or task for the agent
        objective: String,

        /// Model to use
        #[arg(short, long)]
        model: Option<String>,

        /// System prompt / persona
        #[arg(short, long)]
        system: Option<String>,

        /// Maximum reasoning iterations
        #[arg(long, default_value = "10")]
        max_iterations: u32,

        /// Enable verbose output (show reasoning)
        #[arg(short, long)]
        verbose: bool,

        /// Working directory for file tools (defaults to current directory)
        #[arg(short, long)]
        working_dir: Option<std::path::PathBuf>,

        /// Enable code tools (file I/O, shell, search) in addition to builtins
        #[arg(long)]
        code_tools: bool,
    },

    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Generate shell completions
    #[command(hide = true)]
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },

    /// LLM Studio - Dataset, experiment, and prompt management
    Studio {
        #[command(subcommand)]
        action: StudioAction,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,

    /// Set default model
    SetModel {
        /// Model identifier (HuggingFace repo ID or local path)
        model: String,
    },

    /// Clear default model
    ClearModel,

    /// Show config file path
    Path,
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

    /// Convert a model to HoloTensor HCT format.
    /// This enables 70B+ models on 24GB VRAM via progressive quality inference.
    Convert {
        /// Source model (HuggingFace repo ID or local path)
        model: String,

        /// Output directory for HCT files
        #[arg(short, long)]
        output: String,

        /// Number of fragments (more = higher max quality, larger files). Default: 64
        #[arg(long, default_value = "64")]
        fragments: u32,

        /// Maximum rank for SVD decomposition. Default: 256
        #[arg(long, default_value = "256")]
        max_rank: u32,

        /// Minimum quality threshold for verification. Default: 0.85
        #[arg(long, default_value = "0.85")]
        min_quality: f32,

        /// Verify quality after conversion
        #[arg(long, default_value = "true")]
        verify: bool,

        /// Force CPU-only conversion (disables GPU acceleration)
        #[arg(long)]
        cpu_only: bool,

        /// Encoding type: lrdf (default, SVD-based) or spectral (DCT-based)
        #[arg(long, default_value = "lrdf")]
        encoding: String,
    },

    /// Quantize a model to INT4 format for faster inference.
    /// Achieves ~4x memory reduction with minimal quality loss.
    Quantize {
        /// Source model (HuggingFace repo ID or local path)
        model: String,

        /// Output directory for quantized safetensors
        #[arg(short, long)]
        output: String,

        /// Quantization format: int4-sym, int4-asym, int8-sym, int8-asym
        #[arg(long, default_value = "int4-sym")]
        format: String,

        /// Block size for quantization (values per scale). Default: 128
        #[arg(long, default_value = "128")]
        block_size: usize,

        /// Verify quality after quantization
        #[arg(long, default_value = "true")]
        verify: bool,
    },
}

#[derive(Subcommand)]
enum StudioAction {
    /// Show studio statistics and overview
    Stats,

    /// Dataset management
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },

    /// Experiment tracking
    Experiment {
        #[command(subcommand)]
        action: ExperimentAction,
    },

    /// Prompt template management
    Prompt {
        #[command(subcommand)]
        action: PromptAction,
    },

    /// Model registry (for custom trained models)
    Registry {
        #[command(subcommand)]
        action: RegistryAction,
    },
}

#[derive(Subcommand)]
enum DatasetAction {
    /// List datasets
    List,

    /// Create a new dataset
    Create {
        /// Dataset name
        name: String,

        /// Dataset description
        #[arg(short, long)]
        description: Option<String>,
    },

    /// Import data from a file
    Import {
        /// Dataset name
        name: String,

        /// Path to JSONL file
        file: String,
    },

    /// Show dataset info
    Info {
        /// Dataset name
        name: String,
    },

    /// Validate dataset quality
    Validate {
        /// Dataset name
        name: String,
    },

    /// Analyze dataset with Data Curator agent
    Analyze {
        /// Dataset name
        name: String,
    },
}

#[derive(Subcommand)]
enum ExperimentAction {
    /// List experiments
    List,

    /// Create a new experiment
    Create {
        /// Experiment name
        name: String,

        /// Experiment description
        #[arg(short, long)]
        description: Option<String>,
    },

    /// Show experiment details
    Info {
        /// Experiment name
        name: String,
    },

    /// List runs in an experiment
    Runs {
        /// Experiment name
        name: String,
    },

    /// Analyze with Training Coach agent
    Analyze {
        /// Experiment name
        name: String,
    },
}

#[derive(Subcommand)]
enum PromptAction {
    /// List prompt templates
    List,

    /// Create a new prompt template
    Create {
        /// Template name
        name: String,

        /// Template content (use - for stdin)
        content: String,

        /// Template description
        #[arg(short, long)]
        description: Option<String>,
    },

    /// Show prompt template
    #[command(disable_version_flag = true)]
    Show {
        /// Template name
        name: String,

        /// Specific version
        #[arg(short, long)]
        version: Option<u32>,
    },

    /// Test a prompt template
    Test {
        /// Template name
        name: String,

        /// Test input (JSON object for variables)
        #[arg(short, long)]
        input: Option<String>,
    },
}

#[derive(Subcommand)]
enum RegistryAction {
    /// List registered models
    List,

    /// Register a new model
    Register {
        /// Model name
        name: String,

        /// Path to model artifacts
        path: String,

        /// Model description
        #[arg(short, long)]
        description: Option<String>,
    },

    /// Show model details
    Info {
        /// Model name
        name: String,
    },

    /// Promote model to next stage
    Promote {
        /// Model name
        name: String,

        /// Target stage (staging, production)
        stage: String,
    },

    /// Get improvement roadmap from Eval Analyst
    Roadmap {
        /// Model name
        name: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();

    // Initialize logging
    let telemetry_config =
        dantalion::TelemetryConfig::new("infernum").with_log_level(&cli.log_level);

    let telemetry_config = if cli.json_logs {
        telemetry_config.with_json_logs()
    } else {
        telemetry_config
    };

    dantalion::init_logging(&telemetry_config);

    // Load configuration for default values
    let cfg = config::Config::load();

    match cli.command {
        // Default: interactive chat mode when no subcommand is provided
        None => {
            // Use CLI args or config default model
            let model = cli.model.or(cfg.default_model.clone());
            let system = cli.system;
            commands::chat(model, system).await?;
        },

        #[cfg(feature = "server")]
        Some(Commands::Serve {
            host,
            port,
            model,
            config: config_file,
            holo,
            holo_min_quality,
            holo_target_quality,
        }) => {
            // Use config default model if not specified on command line
            let model = model.or(cli.model).or(cfg.default_model.clone());
            commands::serve(
                host,
                port,
                model,
                config_file,
                holo,
                holo_min_quality,
                holo_target_quality,
            )
            .await?;
        },

        Some(Commands::Generate {
            prompt,
            model,
            max_tokens,
            temperature,
            stream,
        }) => {
            // Use config default model if not specified on command line
            let model = model.or(cli.model).or(cfg.default_model.clone());
            commands::generate(prompt, model, max_tokens, temperature, stream).await?;
        },

        Some(Commands::Embed { text, model }) => {
            let model = model.or(cli.model).or(cfg.default_model.clone());
            commands::embed(text, model).await?;
        },

        Some(Commands::Model { action }) => match action {
            ModelAction::List => commands::model_list().await?,
            ModelAction::Pull { model, revision } => commands::model_pull(model, revision).await?,
            ModelAction::Info { model } => commands::model_info(model).await?,
            ModelAction::Remove { model } => commands::model_remove(model).await?,
            ModelAction::Convert {
                model,
                output,
                fragments,
                max_rank,
                min_quality,
                verify,
                cpu_only,
                encoding,
            } => {
                commands::model_convert(model, output, fragments, max_rank, min_quality, verify, cpu_only, encoding)
                    .await?;
            }
            ModelAction::Quantize {
                model,
                output,
                format,
                block_size,
                verify,
            } => {
                commands::model_quantize(model, output, format, block_size, verify).await?;
            }
        },

        Some(Commands::Chat { model, system }) => {
            // Use config default model if not specified on command line
            let model = model.or(cli.model).or(cfg.default_model.clone());
            let system = system.or(cli.system);
            commands::chat(model, system).await?;
        },

        Some(Commands::Version) => {
            commands::version();
        },

        Some(Commands::Doctor) => {
            commands::doctor();
        },

        Some(Commands::Setup {
            skip_detect,
            skip_download,
            skip_test,
            yes,
        }) => {
            commands::setup(skip_detect, skip_download, skip_test, yes).await?;
        },

        Some(Commands::Agent {
            objective,
            model,
            system,
            max_iterations,
            verbose,
            working_dir,
            code_tools,
        }) => {
            let model = model.or(cli.model).or(cfg.default_model.clone());
            let system = system.or(cli.system);
            commands::agent(objective, model, system, max_iterations, verbose, working_dir, code_tools).await?;
        },

        Some(Commands::Config { action }) => match action {
            ConfigAction::Show => {
                config::show_config();
            },
            ConfigAction::SetModel { model } => {
                let mut cfg = config::Config::load();
                match cfg.set_default_model(&model) {
                    Ok(()) => {
                        println!("Default model set to: {}", model);
                        println!(
                            "Config saved to: {}",
                            config::Config::config_path().display()
                        );
                    },
                    Err(e) => {
                        eprintln!("Failed to save config: {}", e);
                    },
                }
            },
            ConfigAction::ClearModel => {
                let mut cfg = config::Config::load();
                match cfg.clear_default_model() {
                    Ok(()) => {
                        println!("Default model cleared.");
                    },
                    Err(e) => {
                        eprintln!("Failed to save config: {}", e);
                    },
                }
            },
            ConfigAction::Path => {
                println!("{}", config::Config::config_path().display());
            },
        },

        Some(Commands::Completions { shell }) => {
            generate(shell, &mut Cli::command(), "infernum", &mut io::stdout());
        },

        Some(Commands::Studio { action }) => match action {
            StudioAction::Stats => {
                commands::studio_stats().await?;
            },
            StudioAction::Dataset { action } => match action {
                DatasetAction::List => commands::dataset_list().await?,
                DatasetAction::Create { name, description } => {
                    commands::dataset_create(name, description).await?;
                },
                DatasetAction::Import { name, file } => {
                    commands::dataset_import(name, file).await?;
                },
                DatasetAction::Info { name } => commands::dataset_info(name).await?,
                DatasetAction::Validate { name } => commands::dataset_validate(name).await?,
                DatasetAction::Analyze { name } => commands::dataset_analyze(name).await?,
            },
            StudioAction::Experiment { action } => match action {
                ExperimentAction::List => commands::experiment_list().await?,
                ExperimentAction::Create { name, description } => {
                    commands::experiment_create(name, description).await?;
                },
                ExperimentAction::Info { name } => commands::experiment_info(name).await?,
                ExperimentAction::Runs { name } => commands::experiment_runs(name).await?,
                ExperimentAction::Analyze { name } => commands::experiment_analyze(name).await?,
            },
            StudioAction::Prompt { action } => match action {
                PromptAction::List => commands::prompt_list().await?,
                PromptAction::Create {
                    name,
                    content,
                    description,
                } => {
                    commands::prompt_create(name, content, description).await?;
                },
                PromptAction::Show { name, version } => {
                    commands::prompt_show(name, version).await?;
                },
                PromptAction::Test { name, input } => {
                    commands::prompt_test(name, input).await?;
                },
            },
            StudioAction::Registry { action } => match action {
                RegistryAction::List => commands::registry_list().await?,
                RegistryAction::Register {
                    name,
                    path,
                    description,
                } => {
                    commands::registry_register(name, path, description).await?;
                },
                RegistryAction::Info { name } => commands::registry_info(name).await?,
                RegistryAction::Promote { name, stage } => {
                    commands::registry_promote(name, stage).await?;
                },
                RegistryAction::Roadmap { name } => commands::registry_roadmap(name).await?,
            },
        },
    }

    Ok(())
}
