//! Model loading utilities.

use std::path::{Path, PathBuf};

use infernum_core::{ModelSource, Result};
use tracing::{debug, info};

/// Model loader for different sources.
pub struct ModelLoader {
    cache_dir: PathBuf,
}

impl ModelLoader {
    /// Creates a new model loader with the given cache directory.
    #[must_use]
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    /// Creates a model loader with the default cache directory.
    #[must_use]
    pub fn default_cache() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("infernum")
            .join("models");
        Self::new(cache_dir)
    }

    /// Resolves a model source to a local path, downloading if necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be resolved or downloaded.
    pub async fn resolve(&self, source: &ModelSource) -> Result<PathBuf> {
        match source {
            ModelSource::HuggingFace { repo_id, revision } => {
                self.resolve_huggingface(repo_id, revision.as_deref()).await
            }
            ModelSource::LocalPath { path } => self.resolve_local(path).await,
            ModelSource::Gguf { path } => self.resolve_local(path).await,
            ModelSource::S3 { bucket, key, region } => {
                self.resolve_s3(bucket, key, region.as_deref()).await
            }
        }
    }

    /// Resolves a HuggingFace model.
    async fn resolve_huggingface(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<PathBuf> {
        info!(repo_id, revision, "Resolving HuggingFace model");

        let revision = revision.unwrap_or("main");
        let cache_path = self.cache_dir.join("huggingface").join(repo_id).join(revision);

        if cache_path.exists() {
            debug!(?cache_path, "Model found in cache");
            return Ok(cache_path);
        }

        // TODO: Implement actual HuggingFace download using hf-hub crate
        // For now, return an error indicating download is not yet implemented
        Err(infernum_core::Error::ModelLoad {
            message: format!(
                "HuggingFace download not yet implemented. Expected path: {}",
                cache_path.display()
            ),
        })
    }

    /// Resolves a local path.
    async fn resolve_local(&self, path: &Path) -> Result<PathBuf> {
        debug!(?path, "Resolving local model");

        if !path.exists() {
            return Err(infernum_core::Error::ModelNotFound {
                model_id: path.display().to_string(),
            });
        }

        Ok(path.to_path_buf())
    }

    /// Resolves an S3 model.
    async fn resolve_s3(
        &self,
        bucket: &str,
        key: &str,
        _region: Option<&str>,
    ) -> Result<PathBuf> {
        info!(bucket, key, "Resolving S3 model");

        let cache_path = self.cache_dir.join("s3").join(bucket).join(key);

        if cache_path.exists() {
            debug!(?cache_path, "Model found in cache");
            return Ok(cache_path);
        }

        // TODO: Implement actual S3 download
        Err(infernum_core::Error::ModelLoad {
            message: format!(
                "S3 download not yet implemented. Expected path: {}",
                cache_path.display()
            ),
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
    pub async fn clear_cache(&self) -> Result<()> {
        if self.cache_dir.exists() {
            tokio::fs::remove_dir_all(&self.cache_dir).await?;
        }
        Ok(())
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
