//! Model loading utilities with HuggingFace Hub integration.

use std::path::{Path, PathBuf};

use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use infernum_core::{ModelSource, Result};
use tracing::{debug, info, warn};

use crate::models::llama::RopeScalingConfig;

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
            },
            ModelSource::LocalPath { path } => self.resolve_local(path),
            ModelSource::Gguf { path } => self.resolve_gguf(path),
            ModelSource::S3 {
                bucket,
                key,
                region,
            } => self.resolve_s3(bucket, key, region.as_deref()),
            ModelSource::HoloTensor {
                path,
                min_quality,
                target_quality,
            } => self.resolve_holotensor(path, *min_quality, *target_quality),
        }
    }

    /// Resolves a HuggingFace model, downloading if necessary.
    fn resolve_huggingface(&self, repo_id: &str, revision: Option<&str>) -> Result<ModelFiles> {
        info!(repo_id, revision, "Resolving HuggingFace model");

        let repo = self.api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            revision.unwrap_or("main").to_string(),
        ));

        // Try to get config first to determine model type
        let config_path = repo
            .get("config.json")
            .map_err(|e| infernum_core::Error::ModelLoad {
                message: format!("Failed to download config.json: {}", e),
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
            return Ok(WeightFiles::ShardedSafetensors {
                index: index_path,
                shards,
            });
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
            return Ok(WeightFiles::ShardedPyTorch {
                index: index_path,
                shards,
            });
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
        let index_content =
            std::fs::read_to_string(index_path).map_err(|e| infernum_core::Error::ModelLoad {
                message: format!("Failed to read index file: {}", e),
            })?;

        let index: serde_json::Value =
            serde_json::from_str(&index_content).map_err(|e| infernum_core::Error::ModelLoad {
                message: format!("Failed to parse index file: {}", e),
            })?;

        // Extract unique shard filenames from weight_map
        let weight_map = index
            .get("weight_map")
            .and_then(|w| w.as_object())
            .ok_or_else(|| infernum_core::Error::ModelLoad {
                message: "Invalid index file: missing weight_map".to_string(),
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
            let path = repo
                .get(shard_name)
                .map_err(|e| infernum_core::Error::ModelLoad {
                    message: format!("Failed to download shard {}: {}", shard_name, e),
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
        let parent = index_path
            .parent()
            .ok_or_else(|| infernum_core::Error::ModelLoad {
                message: "Invalid index path".to_string(),
            })?;

        let index_content =
            std::fs::read_to_string(index_path).map_err(|e| infernum_core::Error::ModelLoad {
                message: format!("Failed to read index: {}", e),
            })?;

        let index: serde_json::Value =
            serde_json::from_str(&index_content).map_err(|e| infernum_core::Error::ModelLoad {
                message: format!("Failed to parse index: {}", e),
            })?;

        let weight_map = index
            .get("weight_map")
            .and_then(|w| w.as_object())
            .ok_or_else(|| infernum_core::Error::ModelLoad {
                message: "Invalid index: missing weight_map".to_string(),
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

    /// Resolves a HoloTensor HCT model directory with progressive quality settings.
    fn resolve_holotensor(
        &self,
        path: &Path,
        min_quality: f32,
        target_quality: f32,
    ) -> Result<ModelFiles> {
        debug!(
            ?path,
            min_quality, target_quality, "Resolving HoloTensor model with progressive quality"
        );

        if !path.exists() {
            return Err(infernum_core::Error::ModelNotFound {
                model_id: path.display().to_string(),
            });
        }

        if !path.is_dir() {
            return Err(infernum_core::Error::ModelLoad {
                message: format!("HoloTensor path must be a directory: {}", path.display()),
            });
        }

        // Look for HCT files in the directory
        let hct_files: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().map(|e| e == "hct").unwrap_or(false) {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        if hct_files.is_empty() {
            return Err(infernum_core::Error::ModelLoad {
                message: format!("No .hct files found in {}", path.display()),
            });
        }

        // Look for config.json (required for model architecture info)
        let config = path.join("config.json");
        if !config.exists() {
            return Err(infernum_core::Error::ModelLoad {
                message: format!(
                    "config.json not found in HoloTensor directory: {}",
                    path.display()
                ),
            });
        }

        // Check if eager HCT loading is requested (for models that fit in VRAM)
        // INFERNUM_HCT_EAGER=1 uses fast sequential loading instead of lazy layer swapping
        let use_eager_hct = std::env::var("INFERNUM_HCT_EAGER")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        if use_eager_hct {
            info!(
                directory = %path.display(),
                file_count = hct_files.len(),
                "Using eager HCT loading (INFERNUM_HCT_EAGER=1) - fast sequential decompression"
            );

            Ok(ModelFiles {
                config,
                weights: WeightFiles::Hct {
                    directory: path.to_path_buf(),
                    files: hct_files,
                },
                tokenizer: Some(path.join("tokenizer.json")).filter(|p| p.exists()),
                tokenizer_config: Some(path.join("tokenizer_config.json")).filter(|p| p.exists()),
            })
        } else {
            // Use progressive HoloTensor loading for tiered memory management
            // VRAM budget: 22GB (2GB headroom on 24GB RTX cards)
            // RAM budget: 64GB
            let vram_budget = 22 * 1024 * 1024 * 1024;
            let ram_budget = 64 * 1024 * 1024 * 1024;

            info!(
                directory = %path.display(),
                file_count = hct_files.len(),
                min_quality = %min_quality,
                target_quality = %target_quality,
                vram_budget_gb = vram_budget / (1024 * 1024 * 1024),
                ram_budget_gb = ram_budget / (1024 * 1024 * 1024),
                "Configured HoloTensor progressive loading"
            );

            Ok(ModelFiles {
                config,
                weights: WeightFiles::HoloTensor {
                    directory: path.to_path_buf(),
                    min_quality,
                    target_quality,
                    vram_budget,
                    ram_budget,
                },
                tokenizer: Some(path.join("tokenizer.json")).filter(|p| p.exists()),
                tokenizer_config: Some(path.join("tokenizer_config.json")).filter(|p| p.exists()),
            })
        }
    }

    /// Resolves an S3 model by downloading to local cache.
    ///
    /// Supports both public S3 buckets (via HTTPS) and authenticated access
    /// via AWS credentials in environment variables.
    fn resolve_s3(&self, bucket: &str, key: &str, region: Option<&str>) -> Result<ModelFiles> {
        let region = region.unwrap_or("us-east-1");
        info!(bucket, key, region, "Resolving S3 model");

        // Create a cache directory for this S3 path
        let cache_key = format!("s3/{}/{}", bucket, key.replace('/', "_"));
        let model_cache_dir = self.cache_dir.join(&cache_key);
        std::fs::create_dir_all(&model_cache_dir)?;

        // Determine if we're loading a single file or a directory prefix
        if key.ends_with(".gguf") {
            // Single GGUF file
            let local_path = model_cache_dir.join(
                Path::new(key)
                    .file_name()
                    .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf")),
            );

            if !local_path.exists() {
                self.download_s3_file(bucket, key, region, &local_path)?;
            } else {
                debug!(?local_path, "Using cached S3 file");
            }

            return self.resolve_gguf(&local_path);
        }

        // Directory-style prefix - download standard model files
        // Download config.json (required)
        let config_key = format!("{}/config.json", key.trim_end_matches('/'));
        let config_path = model_cache_dir.join("config.json");

        if !config_path.exists() {
            self.download_s3_file(bucket, &config_key, region, &config_path)?;
        }

        // Download optional tokenizer files
        let tokenizer_key = format!("{}/tokenizer.json", key.trim_end_matches('/'));
        let tokenizer_path = model_cache_dir.join("tokenizer.json");
        let tokenizer = if !tokenizer_path.exists() {
            self.download_s3_file(bucket, &tokenizer_key, region, &tokenizer_path)
                .ok()
                .map(|_| tokenizer_path.clone())
        } else {
            Some(tokenizer_path)
        };

        let tokenizer_config_key = format!("{}/tokenizer_config.json", key.trim_end_matches('/'));
        let tokenizer_config_path = model_cache_dir.join("tokenizer_config.json");
        let tokenizer_config = if !tokenizer_config_path.exists() {
            self.download_s3_file(
                bucket,
                &tokenizer_config_key,
                region,
                &tokenizer_config_path,
            )
            .ok()
            .map(|_| tokenizer_config_path.clone())
        } else {
            Some(tokenizer_config_path)
        };

        // Try to download weight files in order of preference
        let weights = self.download_s3_weights(bucket, key, region, &model_cache_dir)?;

        Ok(ModelFiles {
            config: config_path,
            weights,
            tokenizer,
            tokenizer_config,
        })
    }

    /// Downloads a single file from S3.
    fn download_s3_file(
        &self,
        bucket: &str,
        key: &str,
        region: &str,
        local_path: &Path,
    ) -> Result<()> {
        // Construct S3 URL (works for public buckets)
        // Format: https://{bucket}.s3.{region}.amazonaws.com/{key}
        // Or path-style: https://s3.{region}.amazonaws.com/{bucket}/{key}
        let url = format!("https://{}.s3.{}.amazonaws.com/{}", bucket, region, key);

        info!(url = %url, "Downloading from S3");

        // Use blocking HTTP request with timeout via agent config
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(300)))
            .build()
            .new_agent();
        let response = agent
            .get(&url)
            .call()
            .map_err(|e: ureq::Error| {
                // Check if it's an auth error
                let msg = if e.to_string().contains("403") {
                    format!(
                        "S3 access denied for s3://{}/{}. Ensure the bucket is public or AWS credentials are configured.",
                        bucket, key
                    )
                } else if e.to_string().contains("404") {
                    format!("S3 object not found: s3://{}/{}", bucket, key)
                } else {
                    format!("Failed to download from S3: {}", e)
                };
                infernum_core::Error::ModelLoad { message: msg }
            })?;

        // Create parent directory if needed
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Stream the response to file
        let mut file = std::fs::File::create(local_path)?;
        let mut reader = response.into_body().into_reader();
        std::io::copy(&mut reader, &mut file).map_err(|e| infernum_core::Error::ModelLoad {
            message: format!("Failed to write S3 file: {}", e),
        })?;

        debug!(?local_path, "Downloaded S3 file");
        Ok(())
    }

    /// Downloads weight files from S3.
    fn download_s3_weights(
        &self,
        bucket: &str,
        key: &str,
        region: &str,
        cache_dir: &Path,
    ) -> Result<WeightFiles> {
        let key = key.trim_end_matches('/');

        // Try single safetensors
        let st_key = format!("{}/model.safetensors", key);
        let st_path = cache_dir.join("model.safetensors");
        if self
            .download_s3_file(bucket, &st_key, region, &st_path)
            .is_ok()
        {
            return Ok(WeightFiles::SingleSafetensors(st_path));
        }

        // Try sharded safetensors - first get the index
        let st_index_key = format!("{}/model.safetensors.index.json", key);
        let st_index_path = cache_dir.join("model.safetensors.index.json");
        if self
            .download_s3_file(bucket, &st_index_key, region, &st_index_path)
            .is_ok()
        {
            let shards =
                self.download_s3_shards_from_index(bucket, key, region, &st_index_path, cache_dir)?;
            return Ok(WeightFiles::ShardedSafetensors {
                index: st_index_path,
                shards,
            });
        }

        // Try single PyTorch file
        let pt_key = format!("{}/pytorch_model.bin", key);
        let pt_path = cache_dir.join("pytorch_model.bin");
        if self
            .download_s3_file(bucket, &pt_key, region, &pt_path)
            .is_ok()
        {
            warn!("Using PyTorch format from S3 - safetensors preferred");
            return Ok(WeightFiles::PyTorch(pt_path));
        }

        // Try sharded PyTorch
        let pt_index_key = format!("{}/pytorch_model.bin.index.json", key);
        let pt_index_path = cache_dir.join("pytorch_model.bin.index.json");
        if self
            .download_s3_file(bucket, &pt_index_key, region, &pt_index_path)
            .is_ok()
        {
            let shards =
                self.download_s3_shards_from_index(bucket, key, region, &pt_index_path, cache_dir)?;
            return Ok(WeightFiles::ShardedPyTorch {
                index: pt_index_path,
                shards,
            });
        }

        Err(infernum_core::Error::ModelLoad {
            message: format!("No supported weight files found in s3://{}/{}", bucket, key),
        })
    }

    /// Downloads shards listed in an index file from S3.
    fn download_s3_shards_from_index(
        &self,
        bucket: &str,
        key: &str,
        region: &str,
        index_path: &Path,
        cache_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        let key = key.trim_end_matches('/');

        let index_content =
            std::fs::read_to_string(index_path).map_err(|e| infernum_core::Error::ModelLoad {
                message: format!("Failed to read index: {}", e),
            })?;

        let index: serde_json::Value =
            serde_json::from_str(&index_content).map_err(|e| infernum_core::Error::ModelLoad {
                message: format!("Failed to parse index: {}", e),
            })?;

        let weight_map = index
            .get("weight_map")
            .and_then(|w| w.as_object())
            .ok_or_else(|| infernum_core::Error::ModelLoad {
                message: "Invalid index: missing weight_map".to_string(),
            })?;

        // Get unique shard names
        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(String::from)
            .collect();
        shard_names.sort();
        shard_names.dedup();

        info!(
            num_shards = shard_names.len(),
            "Downloading S3 model shards"
        );

        let mut shard_paths = Vec::new();
        for (i, shard_name) in shard_names.iter().enumerate() {
            let shard_key = format!("{}/{}", key, shard_name);
            let shard_path = cache_dir.join(shard_name);

            if !shard_path.exists() {
                debug!(
                    shard = %shard_name,
                    progress = format!("{}/{}", i + 1, shard_names.len()),
                    "Downloading shard from S3"
                );
                self.download_s3_file(bucket, &shard_key, region, &shard_path)?;
            } else {
                debug!(shard = %shard_name, "Using cached shard");
            }

            shard_paths.push(shard_path);
        }

        Ok(shard_paths)
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
    /// HCT compressed tensor directory.
    Hct {
        /// Directory containing .hct files.
        directory: PathBuf,
        /// List of .hct files found.
        files: Vec<PathBuf>,
    },
    /// HoloTensor progressive loading (for 405B+ models).
    ///
    /// Uses tiered memory management with VRAM/RAM/Disk placement
    /// and background quality improvement.
    HoloTensor {
        /// Directory containing .hct files.
        directory: PathBuf,
        /// Minimum quality for initial load (0.0-1.0).
        min_quality: f32,
        /// Target quality for background improvement (0.0-1.0).
        target_quality: f32,
        /// VRAM budget in bytes.
        vram_budget: u64,
        /// RAM budget in bytes.
        ram_budget: u64,
    },
}

impl WeightFiles {
    /// Returns all weight file paths.
    #[must_use]
    pub fn paths(&self) -> Vec<&Path> {
        match self {
            Self::SingleSafetensors(p) => vec![p.as_path()],
            Self::ShardedSafetensors { shards, .. } => {
                shards.iter().map(PathBuf::as_path).collect()
            },
            Self::PyTorch(p) => vec![p.as_path()],
            Self::ShardedPyTorch { shards, .. } => shards.iter().map(PathBuf::as_path).collect(),
            Self::Gguf(p) => vec![p.as_path()],
            Self::Hct { files, .. } => files.iter().map(PathBuf::as_path).collect(),
            Self::HoloTensor { directory, .. } => vec![directory.as_path()],
        }
    }

    /// Returns true if this is a safetensors format.
    #[must_use]
    pub fn is_safetensors(&self) -> bool {
        matches!(
            self,
            Self::SingleSafetensors(_) | Self::ShardedSafetensors { .. }
        )
    }

    /// Returns true if this is a GGUF format.
    #[must_use]
    pub fn is_gguf(&self) -> bool {
        matches!(self, Self::Gguf(_))
    }

    /// Returns true if this is an HCT compressed format.
    #[must_use]
    pub fn is_hct(&self) -> bool {
        matches!(self, Self::Hct { .. })
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
    /// HCT compressed tensor format.
    Hct,
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
            Some("hct") => Self::Hct,
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

    /// RoPE scaling configuration (for Llama 3.2+ extended context).
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

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
        let content =
            std::fs::read_to_string(path).map_err(|e| infernum_core::Error::ModelLoad {
                message: format!("Failed to read config: {}", e),
            })?;

        serde_json::from_str(&content).map_err(|e| infernum_core::Error::ModelLoad {
            message: format!("Failed to parse config: {}", e),
        })
    }

    /// Returns the model architecture name.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.model_type.as_deref().or_else(|| {
            self.architectures
                .as_ref()
                .and_then(|a| a.first().map(String::as_str))
        })
    }

    /// Returns EOS token IDs.
    #[must_use]
    pub fn eos_token_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => {
                n.as_u64().map(|v| vec![v as u32]).unwrap_or_default()
            },
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect(),
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // WeightFiles tests
    // ==========================================================================

    #[test]
    fn test_weight_files_single_safetensors_paths() {
        let path = PathBuf::from("/models/model.safetensors");
        let weights = WeightFiles::SingleSafetensors(path.clone());

        let paths = weights.paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], path.as_path());
    }

    #[test]
    fn test_weight_files_sharded_safetensors_paths() {
        let shards = vec![
            PathBuf::from("/models/model-00001.safetensors"),
            PathBuf::from("/models/model-00002.safetensors"),
        ];
        let weights = WeightFiles::ShardedSafetensors {
            index: PathBuf::from("/models/model.safetensors.index.json"),
            shards: shards.clone(),
        };

        let paths = weights.paths();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], shards[0].as_path());
        assert_eq!(paths[1], shards[1].as_path());
    }

    #[test]
    fn test_weight_files_pytorch_paths() {
        let path = PathBuf::from("/models/pytorch_model.bin");
        let weights = WeightFiles::PyTorch(path.clone());

        let paths = weights.paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], path.as_path());
    }

    #[test]
    fn test_weight_files_sharded_pytorch_paths() {
        let shards = vec![
            PathBuf::from("/models/pytorch_model-00001.bin"),
            PathBuf::from("/models/pytorch_model-00002.bin"),
        ];
        let weights = WeightFiles::ShardedPyTorch {
            index: PathBuf::from("/models/pytorch_model.bin.index.json"),
            shards: shards.clone(),
        };

        let paths = weights.paths();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_weight_files_gguf_paths() {
        let path = PathBuf::from("/models/model.gguf");
        let weights = WeightFiles::Gguf(path.clone());

        let paths = weights.paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], path.as_path());
    }

    #[test]
    fn test_weight_files_is_safetensors() {
        assert!(WeightFiles::SingleSafetensors(PathBuf::new()).is_safetensors());
        assert!(WeightFiles::ShardedSafetensors {
            index: PathBuf::new(),
            shards: vec![]
        }
        .is_safetensors());
        assert!(!WeightFiles::PyTorch(PathBuf::new()).is_safetensors());
        assert!(!WeightFiles::Gguf(PathBuf::new()).is_safetensors());
    }

    #[test]
    fn test_weight_files_is_gguf() {
        assert!(WeightFiles::Gguf(PathBuf::new()).is_gguf());
        assert!(!WeightFiles::SingleSafetensors(PathBuf::new()).is_gguf());
        assert!(!WeightFiles::PyTorch(PathBuf::new()).is_gguf());
    }

    // ==========================================================================
    // ModelFormat tests
    // ==========================================================================

    #[test]
    fn test_model_format_from_safetensors() {
        let format = ModelFormat::from_path(Path::new("model.safetensors"));
        assert_eq!(format, ModelFormat::SafeTensors);
    }

    #[test]
    fn test_model_format_from_gguf() {
        let format = ModelFormat::from_path(Path::new("model.gguf"));
        assert_eq!(format, ModelFormat::Gguf);
    }

    #[test]
    fn test_model_format_from_pytorch_bin() {
        let format = ModelFormat::from_path(Path::new("pytorch_model.bin"));
        assert_eq!(format, ModelFormat::PyTorch);
    }

    #[test]
    fn test_model_format_from_pytorch_pt() {
        let format = ModelFormat::from_path(Path::new("model.pt"));
        assert_eq!(format, ModelFormat::PyTorch);
    }

    #[test]
    fn test_model_format_from_pytorch_pth() {
        let format = ModelFormat::from_path(Path::new("model.pth"));
        assert_eq!(format, ModelFormat::PyTorch);
    }

    #[test]
    fn test_model_format_unknown() {
        let format = ModelFormat::from_path(Path::new("model.unknown"));
        assert_eq!(format, ModelFormat::Unknown);

        let format = ModelFormat::from_path(Path::new("model"));
        assert_eq!(format, ModelFormat::Unknown);
    }

    #[test]
    fn test_model_format_with_path() {
        let format = ModelFormat::from_path(Path::new("/path/to/models/llama.gguf"));
        assert_eq!(format, ModelFormat::Gguf);
    }

    // ==========================================================================
    // ModelConfig tests
    // ==========================================================================

    #[test]
    fn test_model_config_deserialize_minimal() {
        let json = r#"{}"#;
        let config: ModelConfig = serde_json::from_str(json).expect("deserialize");

        assert!(config.model_type.is_none());
        assert!(config.hidden_size.is_none());
        assert!(config.architecture().is_none());
    }

    #[test]
    fn test_model_config_deserialize_llama() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "bos_token_id": 1,
            "eos_token_id": 2
        }"#;

        let config: ModelConfig = serde_json::from_str(json).expect("deserialize");

        assert_eq!(config.model_type, Some("llama".to_string()));
        assert_eq!(config.hidden_size, Some(4096));
        assert_eq!(config.intermediate_size, Some(11008));
        assert_eq!(config.num_hidden_layers, Some(32));
        assert_eq!(config.num_attention_heads, Some(32));
        assert_eq!(config.num_key_value_heads, Some(32));
        assert_eq!(config.vocab_size, Some(32000));
        assert_eq!(config.max_position_embeddings, Some(4096));
        assert!((config.rms_norm_eps.unwrap() - 1e-6).abs() < 1e-10);
        assert!((config.rope_theta.unwrap() - 10000.0).abs() < 0.01);
        assert_eq!(config.hidden_act, Some("silu".to_string()));
        assert_eq!(config.bos_token_id, Some(1));
    }

    #[test]
    fn test_model_config_architecture_from_model_type() {
        let config = ModelConfig {
            model_type: Some("llama".to_string()),
            architectures: None,
            hidden_size: None,
            intermediate_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            vocab_size: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            rope_scaling: None,
            hidden_act: None,
            torch_dtype: None,
            tie_word_embeddings: None,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
        };

        assert_eq!(config.architecture(), Some("llama"));
    }

    #[test]
    fn test_model_config_architecture_from_architectures() {
        let config = ModelConfig {
            model_type: None,
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            hidden_size: None,
            intermediate_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            vocab_size: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            rope_scaling: None,
            hidden_act: None,
            torch_dtype: None,
            tie_word_embeddings: None,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
        };

        assert_eq!(config.architecture(), Some("LlamaForCausalLM"));
    }

    #[test]
    fn test_model_config_architecture_prefers_model_type() {
        let config = ModelConfig {
            model_type: Some("llama".to_string()),
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            hidden_size: None,
            intermediate_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            vocab_size: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            rope_scaling: None,
            hidden_act: None,
            torch_dtype: None,
            tie_word_embeddings: None,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
        };

        assert_eq!(config.architecture(), Some("llama"));
    }

    #[test]
    fn test_model_config_eos_token_ids_single() {
        let json = r#"{"eos_token_id": 2}"#;
        let config: ModelConfig = serde_json::from_str(json).expect("deserialize");

        let eos_ids = config.eos_token_ids();
        assert_eq!(eos_ids, vec![2]);
    }

    #[test]
    fn test_model_config_eos_token_ids_array() {
        let json = r#"{"eos_token_id": [2, 128001, 128009]}"#;
        let config: ModelConfig = serde_json::from_str(json).expect("deserialize");

        let eos_ids = config.eos_token_ids();
        assert_eq!(eos_ids, vec![2, 128001, 128009]);
    }

    #[test]
    fn test_model_config_eos_token_ids_missing() {
        let json = r#"{}"#;
        let config: ModelConfig = serde_json::from_str(json).expect("deserialize");

        let eos_ids = config.eos_token_ids();
        assert!(eos_ids.is_empty());
    }

    #[test]
    fn test_model_config_gqa() {
        // Test Grouped Query Attention config (e.g., Llama 2 70B)
        let json = r#"{
            "model_type": "llama",
            "num_attention_heads": 64,
            "num_key_value_heads": 8
        }"#;

        let config: ModelConfig = serde_json::from_str(json).expect("deserialize");

        assert_eq!(config.num_attention_heads, Some(64));
        assert_eq!(config.num_key_value_heads, Some(8));
        // GQA ratio would be 64/8 = 8
    }

    #[test]
    fn test_model_config_qwen2() {
        let json = r#"{
            "model_type": "qwen2",
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "vocab_size": 152064,
            "max_position_embeddings": 131072,
            "rope_theta": 1000000.0
        }"#;

        let config: ModelConfig = serde_json::from_str(json).expect("deserialize");

        assert_eq!(config.architecture(), Some("qwen2"));
        assert_eq!(config.hidden_size, Some(3584));
        assert!((config.rope_theta.unwrap() - 1_000_000.0).abs() < 0.01);
    }

    // ==========================================================================
    // ModelFiles tests
    // ==========================================================================

    #[test]
    fn test_model_files_debug() {
        let files = ModelFiles {
            config: PathBuf::from("/models/config.json"),
            weights: WeightFiles::Gguf(PathBuf::from("/models/model.gguf")),
            tokenizer: None,
            tokenizer_config: None,
        };

        let debug = format!("{:?}", files);
        assert!(debug.contains("ModelFiles"));
        assert!(debug.contains("config.json"));
    }

    #[test]
    fn test_model_files_with_tokenizer() {
        let files = ModelFiles {
            config: PathBuf::from("/models/config.json"),
            weights: WeightFiles::SingleSafetensors(PathBuf::from("/models/model.safetensors")),
            tokenizer: Some(PathBuf::from("/models/tokenizer.json")),
            tokenizer_config: Some(PathBuf::from("/models/tokenizer_config.json")),
        };

        assert!(files.tokenizer.is_some());
        assert!(files.tokenizer_config.is_some());
    }
}
