//! Model loading utilities with HuggingFace Hub integration.

use std::path::{Path, PathBuf};

use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use infernum_core::{ModelSource, Result};
use tracing::{debug, info, warn};

/// Model loader for different sources.
pub struct ModelLoader {
    cache_dir: PathBuf,
    api: Api,
}

impl ModelLoader {
    /// Creates a new model loader with the given cache directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the HuggingFace API cannot be initialized.
    pub fn new(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let cache_dir = cache_dir.into();
        let api = Api::new().map_err(|e| infernum_core::Error::ModelLoad {
            message: format!("Failed to initialize HuggingFace API: {}", e),
        })?;

        Ok(Self { cache_dir, api })
    }

    /// Creates a model loader with the default cache directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the API cannot be initialized.
    pub fn default_cache() -> Result<Self> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("infernum")
            .join("models");
        Self::new(cache_dir)
    }

    /// Resolves a model source to local paths for all required files.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be resolved or downloaded.
    pub fn resolve(&self, source: &ModelSource) -> Result<ModelFiles> {
        match source {
            ModelSource::HuggingFace { repo_id, revision } => {
                self.resolve_huggingface(repo_id, revision.as_deref())
            }
            ModelSource::LocalPath { path } => self.resolve_local(path),
            ModelSource::Gguf { path } => self.resolve_gguf(path),
            ModelSource::S3 { bucket, key, region } => {
                self.resolve_s3(bucket, key, region.as_deref())
            }
        }
    }

    /// Resolves a HuggingFace model, downloading if necessary.
    fn resolve_huggingface(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<ModelFiles> {
        info!(repo_id, revision, "Resolving HuggingFace model");

        let repo = self.api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            revision.unwrap_or("main").to_string(),
        ));

        // Try to get config first to determine model type
        let config_path = repo.get("config.json").map_err(|e| {
            infernum_core::Error::ModelLoad {
                message: format!("Failed to download config.json: {}", e),
            }
        })?;

        debug!(?config_path, "Downloaded config.json");

        // Get tokenizer files
        let tokenizer_path = repo.get("tokenizer.json").ok();
        let tokenizer_config_path = repo.get("tokenizer_config.json").ok();

        // Try different weight file patterns
        let weights = self.resolve_weights(&repo, repo_id)?;

        Ok(ModelFiles {
            config: config_path,
            weights,
            tokenizer: tokenizer_path,
            tokenizer_config: tokenizer_config_path,
        })
    }

    /// Resolves weight files, trying different formats.
    fn resolve_weights(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        repo_id: &str,
    ) -> Result<WeightFiles> {
        // Try safetensors first (preferred)
        if let Ok(path) = repo.get("model.safetensors") {
            info!("Found single safetensors file");
            return Ok(WeightFiles::SingleSafetensors(path));
        }

        // Try sharded safetensors
        if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            info!("Found sharded safetensors");
            let shards = self.download_shards(repo, &index_path)?;
            return Ok(WeightFiles::ShardedSafetensors { index: index_path, shards });
        }

        // Try PyTorch format
        if let Ok(path) = repo.get("pytorch_model.bin") {
            warn!("Using PyTorch format - safetensors preferred for performance");
            return Ok(WeightFiles::PyTorch(path));
        }

        // Try sharded PyTorch
        if let Ok(index_path) = repo.get("pytorch_model.bin.index.json") {
            warn!("Using sharded PyTorch format");
            let shards = self.download_shards(repo, &index_path)?;
            return Ok(WeightFiles::ShardedPyTorch { index: index_path, shards });
        }

        Err(infernum_core::Error::ModelLoad {
            message: format!("No supported weight files found in {}", repo_id),
        })
    }

    /// Downloads sharded weight files based on index.
    fn download_shards(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        index_path: &Path,
    ) -> Result<Vec<PathBuf>> {
        let index_content = std::fs::read_to_string(index_path).map_err(|e| {
            infernum_core::Error::ModelLoad {
                message: format!("Failed to read index file: {}", e),
            }
        })?;

        let index: serde_json::Value = serde_json::from_str(&index_content).map_err(|e| {
            infernum_core::Error::ModelLoad {
                message: format!("Failed to parse index file: {}", e),
            }
        })?;

        // Extract unique shard filenames from weight_map
        let weight_map = index.get("weight_map").and_then(|w| w.as_object()).ok_or_else(|| {
            infernum_core::Error::ModelLoad {
                message: "Invalid index file: missing weight_map".to_string(),
            }
        })?;

        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(String::from)
            .collect();
        shard_names.sort();
        shard_names.dedup();

        info!(num_shards = shard_names.len(), "Downloading model shards");

        let mut shard_paths = Vec::new();
        for (i, shard_name) in shard_names.iter().enumerate() {
            debug!(shard = %shard_name, progress = format!("{}/{}", i + 1, shard_names.len()), "Downloading shard");
            let path = repo.get(shard_name).map_err(|e| {
                infernum_core::Error::ModelLoad {
                    message: format!("Failed to download shard {}: {}", shard_name, e),
                }
            })?;
            shard_paths.push(path);
        }

        Ok(shard_paths)
    }

    /// Resolves a local path.
    fn resolve_local(&self, path: &Path) -> Result<ModelFiles> {
        debug!(?path, "Resolving local model");

        if !path.exists() {
            return Err(infernum_core::Error::ModelNotFound {
                model_id: path.display().to_string(),
            });
        }

        // Check if it's a directory or single file
        if path.is_dir() {
            let config = path.join("config.json");
            if !config.exists() {
                return Err(infernum_core::Error::ModelLoad {
                    message: format!("config.json not found in {}", path.display()),
                });
            }

            let weights = if path.join("model.safetensors").exists() {
                WeightFiles::SingleSafetensors(path.join("model.safetensors"))
            } else if path.join("model.safetensors.index.json").exists() {
                let index = path.join("model.safetensors.index.json");
                let shards = self.find_local_shards(&index)?;
                WeightFiles::ShardedSafetensors { index, shards }
            } else if path.join("pytorch_model.bin").exists() {
                WeightFiles::PyTorch(path.join("pytorch_model.bin"))
            } else {
                return Err(infernum_core::Error::ModelLoad {
                    message: "No weight files found in directory".to_string(),
                });
            };

            Ok(ModelFiles {
                config,
                weights,
                tokenizer: Some(path.join("tokenizer.json")).filter(|p| p.exists()),
                tokenizer_config: Some(path.join("tokenizer_config.json")).filter(|p| p.exists()),
            })
        } else {
            // Single file - assume it's a GGUF or similar
            self.resolve_gguf(path)
        }
    }

    /// Finds shards from a local index file.
    fn find_local_shards(&self, index_path: &Path) -> Result<Vec<PathBuf>> {
        let parent = index_path.parent().ok_or_else(|| {
            infernum_core::Error::ModelLoad {
                message: "Invalid index path".to_string(),
            }
        })?;

        let index_content = std::fs::read_to_string(index_path).map_err(|e| {
            infernum_core::Error::ModelLoad {
                message: format!("Failed to read index: {}", e),
            }
        })?;

        let index: serde_json::Value = serde_json::from_str(&index_content).map_err(|e| {
            infernum_core::Error::ModelLoad {
                message: format!("Failed to parse index: {}", e),
            }
        })?;

        let weight_map = index.get("weight_map").and_then(|w| w.as_object()).ok_or_else(|| {
            infernum_core::Error::ModelLoad {
                message: "Invalid index: missing weight_map".to_string(),
            }
        })?;

        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(String::from)
            .collect();
        shard_names.sort();
        shard_names.dedup();

        Ok(shard_names.into_iter().map(|n| parent.join(n)).collect())
    }

    /// Resolves a GGUF file.
    fn resolve_gguf(&self, path: &Path) -> Result<ModelFiles> {
        debug!(?path, "Resolving GGUF model");

        if !path.exists() {
            return Err(infernum_core::Error::ModelNotFound {
                model_id: path.display().to_string(),
            });
        }

        // GGUF files are self-contained
        Ok(ModelFiles {
            config: path.to_path_buf(), // GGUF has embedded config
            weights: WeightFiles::Gguf(path.to_path_buf()),
            tokenizer: None, // GGUF has embedded tokenizer
            tokenizer_config: None,
        })
    }

    /// Resolves an S3 model.
    fn resolve_s3(
        &self,
        bucket: &str,
        key: &str,
        _region: Option<&str>,
    ) -> Result<ModelFiles> {
        info!(bucket, key, "S3 model resolution not yet implemented");

        Err(infernum_core::Error::ModelLoad {
            message: "S3 model loading not yet implemented".to_string(),
        })
    }

    /// Returns the cache directory.
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Clears the model cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache cannot be cleared.
    pub fn clear_cache(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }
}

/// Collection of files needed to load a model.
#[derive(Debug)]
pub struct ModelFiles {
    /// Path to config.json.
    pub config: PathBuf,
    /// Weight files.
    pub weights: WeightFiles,
    /// Optional tokenizer.json.
    pub tokenizer: Option<PathBuf>,
    /// Optional tokenizer_config.json.
    pub tokenizer_config: Option<PathBuf>,
}

/// Weight file formats.
#[derive(Debug)]
pub enum WeightFiles {
    /// Single safetensors file.
    SingleSafetensors(PathBuf),
    /// Sharded safetensors files.
    ShardedSafetensors {
        /// Index file.
        index: PathBuf,
        /// Shard files.
        shards: Vec<PathBuf>,
    },
    /// Single PyTorch file.
    PyTorch(PathBuf),
    /// Sharded PyTorch files.
    ShardedPyTorch {
        /// Index file.
        index: PathBuf,
        /// Shard files.
        shards: Vec<PathBuf>,
    },
    /// GGUF file (self-contained).
    Gguf(PathBuf),
}

impl WeightFiles {
    /// Returns all weight file paths.
    #[must_use]
    pub fn paths(&self) -> Vec<&Path> {
        match self {
            Self::SingleSafetensors(p) => vec![p.as_path()],
            Self::ShardedSafetensors { shards, .. } => shards.iter().map(PathBuf::as_path).collect(),
            Self::PyTorch(p) => vec![p.as_path()],
            Self::ShardedPyTorch { shards, .. } => shards.iter().map(PathBuf::as_path).collect(),
            Self::Gguf(p) => vec![p.as_path()],
        }
    }

    /// Returns true if this is a safetensors format.
    #[must_use]
    pub fn is_safetensors(&self) -> bool {
        matches!(self, Self::SingleSafetensors(_) | Self::ShardedSafetensors { .. })
    }

    /// Returns true if this is a GGUF format.
    #[must_use]
    pub fn is_gguf(&self) -> bool {
        matches!(self, Self::Gguf(_))
    }
}

/// Detects the model format from file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// SafeTensors format.
    SafeTensors,
    /// GGUF format.
    Gguf,
    /// PyTorch format.
    PyTorch,
    /// Unknown format.
    Unknown,
}

impl ModelFormat {
    /// Detects the format from a file path.
    #[must_use]
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => Self::SafeTensors,
            Some("gguf") => Self::Gguf,
            Some("pt") | Some("pth") | Some("bin") => Self::PyTorch,
            _ => Self::Unknown,
        }
    }
}

/// Model configuration loaded from config.json.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    /// Model architecture type.
    #[serde(default)]
    pub model_type: Option<String>,

    /// Architecture list (alternative).
    #[serde(default)]
    pub architectures: Option<Vec<String>>,

    /// Hidden size.
    #[serde(default)]
    pub hidden_size: Option<usize>,

    /// Intermediate size (FFN).
    #[serde(default)]
    pub intermediate_size: Option<usize>,

    /// Number of hidden layers.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,

    /// Number of attention heads.
    #[serde(default)]
    pub num_attention_heads: Option<usize>,

    /// Number of key-value heads (for GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Vocabulary size.
    #[serde(default)]
    pub vocab_size: Option<usize>,

    /// Maximum position embeddings.
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    /// RMS norm epsilon.
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,

    /// Rope theta.
    #[serde(default)]
    pub rope_theta: Option<f64>,

    /// Hidden activation function.
    #[serde(default)]
    pub hidden_act: Option<String>,

    /// Torch dtype.
    #[serde(default)]
    pub torch_dtype: Option<String>,

    /// Tie word embeddings.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,

    /// Beginning of sentence token ID.
    #[serde(default)]
    pub bos_token_id: Option<u32>,

    /// End of sentence token ID.
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,

    /// Padding token ID.
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

impl ModelConfig {
    /// Loads configuration from a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            infernum_core::Error::ModelLoad {
                message: format!("Failed to read config: {}", e),
            }
        })?;

        serde_json::from_str(&content).map_err(|e| {
            infernum_core::Error::ModelLoad {
                message: format!("Failed to parse config: {}", e),
            }
        })
    }

    /// Returns the model architecture name.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.model_type.as_deref().or_else(|| {
            self.architectures.as_ref().and_then(|a| a.first().map(String::as_str))
        })
    }

    /// Returns EOS token IDs.
    #[must_use]
    pub fn eos_token_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => {
                n.as_u64().map(|v| vec![v as u32]).unwrap_or_default()
            }
            Some(serde_json::Value::Array(arr)) => {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect()
            }
            _ => Vec::new(),
        }
    }
}
