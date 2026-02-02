//! Edge deployment support for lightweight inference.
//!
//! This module provides infrastructure for running Infernum in edge
//! environments including WASM, mobile, and offline scenarios.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Edge deployment target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeTarget {
    /// WebAssembly (browser or Node.js).
    Wasm,
    /// iOS (via Rust FFI).
    Ios,
    /// Android (via JNI).
    Android,
    /// Embedded Linux (Raspberry Pi, Jetson).
    EmbeddedLinux,
    /// Desktop with limited resources.
    LightweightDesktop,
}

impl EdgeTarget {
    /// Returns recommended maximum model size in bytes.
    pub fn max_model_size(&self) -> u64 {
        match self {
            Self::Wasm => 500 * 1024 * 1024,           // 500MB
            Self::Ios => 2 * 1024 * 1024 * 1024,       // 2GB
            Self::Android => 1024 * 1024 * 1024,      // 1GB
            Self::EmbeddedLinux => 4 * 1024 * 1024 * 1024, // 4GB
            Self::LightweightDesktop => 8 * 1024 * 1024 * 1024, // 8GB
        }
    }

    /// Returns recommended quantization level.
    pub fn recommended_quantization(&self) -> QuantizationLevel {
        match self {
            Self::Wasm => QuantizationLevel::Q4_0,
            Self::Ios => QuantizationLevel::Q4_K_M,
            Self::Android => QuantizationLevel::Q4_K_S,
            Self::EmbeddedLinux => QuantizationLevel::Q5_K_M,
            Self::LightweightDesktop => QuantizationLevel::Q8_0,
        }
    }

    /// Returns whether GPU acceleration is typically available.
    pub fn has_gpu(&self) -> bool {
        match self {
            Self::Wasm => false,    // WebGPU is emerging but not reliable
            Self::Ios => true,      // Metal
            Self::Android => true,  // Vulkan/OpenCL
            Self::EmbeddedLinux => false, // Usually CPU only
            Self::LightweightDesktop => true,
        }
    }

    /// Returns maximum recommended context length.
    pub fn max_context_length(&self) -> u32 {
        match self {
            Self::Wasm => 2048,
            Self::Ios => 4096,
            Self::Android => 4096,
            Self::EmbeddedLinux => 2048,
            Self::LightweightDesktop => 8192,
        }
    }
}

/// Quantization levels for model compression.
///
/// Names follow the GGUF/llama.cpp quantization naming convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum QuantizationLevel {
    /// No quantization (FP16/FP32).
    None,
    /// 8-bit quantization.
    Q8_0,
    /// 5-bit K-quant (medium).
    Q5_K_M,
    /// 5-bit K-quant (small).
    Q5_K_S,
    /// 4-bit K-quant (medium).
    Q4_K_M,
    /// 4-bit K-quant (small).
    Q4_K_S,
    /// 4-bit quantization.
    Q4_0,
    /// 3-bit quantization (aggressive).
    Q3_K_S,
    /// 2-bit quantization (extreme).
    Q2_K,
}

impl QuantizationLevel {
    /// Returns approximate compression ratio.
    pub fn compression_ratio(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Q8_0 => 2.0,
            Self::Q5_K_M => 3.2,
            Self::Q5_K_S => 3.5,
            Self::Q4_K_M => 4.0,
            Self::Q4_K_S => 4.5,
            Self::Q4_0 => 4.0,
            Self::Q3_K_S => 5.3,
            Self::Q2_K => 8.0,
        }
    }

    /// Returns quality impact (1.0 = no impact, lower = more degradation).
    pub fn quality_factor(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Q8_0 => 0.99,
            Self::Q5_K_M => 0.97,
            Self::Q5_K_S => 0.96,
            Self::Q4_K_M => 0.95,
            Self::Q4_K_S => 0.94,
            Self::Q4_0 => 0.93,
            Self::Q3_K_S => 0.88,
            Self::Q2_K => 0.80,
        }
    }
}

/// Configuration for edge deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    /// Target platform.
    pub target: EdgeTarget,
    /// Maximum memory usage in bytes.
    pub max_memory: u64,
    /// Maximum model size in bytes.
    pub max_model_size: u64,
    /// Preferred quantization level.
    pub quantization: QuantizationLevel,
    /// Maximum context length.
    pub max_context: u32,
    /// Enable offline mode.
    pub offline_mode: bool,
    /// Model cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Maximum cache size in bytes.
    pub max_cache_size: u64,
    /// Number of inference threads.
    pub num_threads: u32,
    /// Enable memory mapping for models.
    pub use_mmap: bool,
    /// Batch size for inference.
    pub batch_size: u32,
}

impl EdgeConfig {
    /// Creates configuration for a specific target.
    pub fn for_target(target: EdgeTarget) -> Self {
        Self {
            target,
            max_memory: match target {
                EdgeTarget::Wasm => 2 * 1024 * 1024 * 1024,
                EdgeTarget::Ios => 4 * 1024 * 1024 * 1024,
                EdgeTarget::Android => 3 * 1024 * 1024 * 1024,
                EdgeTarget::EmbeddedLinux => 2 * 1024 * 1024 * 1024,
                EdgeTarget::LightweightDesktop => 8 * 1024 * 1024 * 1024,
            },
            max_model_size: target.max_model_size(),
            quantization: target.recommended_quantization(),
            max_context: target.max_context_length(),
            offline_mode: false,
            cache_dir: None,
            max_cache_size: 5 * 1024 * 1024 * 1024, // 5GB default
            num_threads: match target {
                EdgeTarget::Wasm => 1, // WebWorker limitations
                EdgeTarget::Ios => 4,
                EdgeTarget::Android => 4,
                EdgeTarget::EmbeddedLinux => 4,
                EdgeTarget::LightweightDesktop => 8,
            },
            use_mmap: !matches!(target, EdgeTarget::Wasm),
            batch_size: 1,
        }
    }

    /// Creates WASM-specific configuration.
    pub fn wasm() -> Self {
        Self::for_target(EdgeTarget::Wasm)
    }

    /// Creates mobile-optimized configuration.
    pub fn mobile() -> Self {
        Self::for_target(EdgeTarget::Ios)
    }

    /// Enables offline mode with cache directory.
    pub fn with_offline_cache(mut self, cache_dir: PathBuf) -> Self {
        self.offline_mode = true;
        self.cache_dir = Some(cache_dir);
        self
    }

    /// Sets maximum memory.
    pub fn with_max_memory(mut self, bytes: u64) -> Self {
        self.max_memory = bytes;
        self
    }

    /// Sets quantization level.
    pub fn with_quantization(mut self, level: QuantizationLevel) -> Self {
        self.quantization = level;
        self
    }

    /// Validates configuration for the target.
    pub fn validate(&self) -> Result<(), EdgeConfigError> {
        if self.max_model_size > self.max_memory {
            return Err(EdgeConfigError::ModelTooLarge {
                model_size: self.max_model_size,
                available_memory: self.max_memory,
            });
        }

        if self.max_context > self.target.max_context_length() * 2 {
            return Err(EdgeConfigError::ContextTooLarge {
                requested: self.max_context,
                recommended: self.target.max_context_length(),
            });
        }

        Ok(())
    }
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self::for_target(EdgeTarget::LightweightDesktop)
    }
}

/// Edge configuration errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum EdgeConfigError {
    /// Model too large for available memory.
    #[error("Model size ({model_size} bytes) exceeds available memory ({available_memory} bytes)")]
    ModelTooLarge {
        /// Size of the model in bytes.
        model_size: u64,
        /// Available memory in bytes.
        available_memory: u64,
    },
    /// Context length too large for target.
    #[error("Context length {requested} exceeds recommended {recommended} for target")]
    ContextTooLarge {
        /// Requested context length.
        requested: u32,
        /// Recommended maximum context length.
        recommended: u32,
    },
}

/// Model cache entry metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Model identifier.
    pub model_id: String,
    /// File path in cache.
    pub path: PathBuf,
    /// File size in bytes.
    pub size: u64,
    /// Quantization level.
    pub quantization: QuantizationLevel,
    /// Cache timestamp.
    pub cached_at: chrono::DateTime<chrono::Utc>,
    /// Last access timestamp.
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    /// Access count.
    pub access_count: u64,
    /// Checksum (SHA256).
    pub checksum: String,
}

/// Offline model cache manager.
pub struct ModelCache {
    /// Cache directory.
    cache_dir: PathBuf,
    /// Maximum cache size.
    max_size: u64,
    /// Current cache size.
    current_size: RwLock<u64>,
    /// Cache entries.
    entries: RwLock<HashMap<String, CacheEntry>>,
    /// Cache index file.
    index_file: PathBuf,
}

impl ModelCache {
    /// Creates a new model cache.
    pub fn new(cache_dir: PathBuf, max_size: u64) -> std::io::Result<Self> {
        std::fs::create_dir_all(&cache_dir)?;

        let index_file = cache_dir.join("cache_index.json");
        let entries = if index_file.exists() {
            let content = std::fs::read_to_string(&index_file)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            HashMap::new()
        };

        let current_size: u64 = entries.values().map(|e: &CacheEntry| e.size).sum();

        Ok(Self {
            cache_dir,
            max_size,
            current_size: RwLock::new(current_size),
            entries: RwLock::new(entries),
            index_file,
        })
    }

    /// Checks if a model is cached.
    pub fn is_cached(&self, model_id: &str) -> bool {
        self.entries.read().contains_key(model_id)
    }

    /// Gets the cache path for a model.
    pub fn get_path(&self, model_id: &str) -> Option<PathBuf> {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(model_id) {
            entry.last_accessed = chrono::Utc::now();
            entry.access_count += 1;
            Some(entry.path.clone())
        } else {
            None
        }
    }

    /// Adds a model to the cache.
    pub fn add(&self, model_id: &str, source_path: &std::path::Path, quantization: QuantizationLevel) -> std::io::Result<PathBuf> {
        let metadata = std::fs::metadata(source_path)?;
        let size = metadata.len();

        // Ensure enough space
        self.ensure_space(size)?;

        // Copy to cache
        let filename = format!("{}_{:?}.bin", model_id, quantization);
        let cache_path = self.cache_dir.join(&filename);
        std::fs::copy(source_path, &cache_path)?;

        // Compute checksum
        let content = std::fs::read(&cache_path)?;
        let checksum = format!("{:x}", md5::compute(&content));

        // Add entry
        let now = chrono::Utc::now();
        let entry = CacheEntry {
            model_id: model_id.to_string(),
            path: cache_path.clone(),
            size,
            quantization,
            cached_at: now,
            last_accessed: now,
            access_count: 0,
            checksum,
        };

        {
            let mut entries = self.entries.write();
            entries.insert(model_id.to_string(), entry);
            *self.current_size.write() += size;
        }

        self.save_index()?;
        Ok(cache_path)
    }

    /// Removes a model from the cache.
    pub fn remove(&self, model_id: &str) -> std::io::Result<bool> {
        let entry = self.entries.write().remove(model_id);
        if let Some(entry) = entry {
            if entry.path.exists() {
                std::fs::remove_file(&entry.path)?;
            }
            *self.current_size.write() -= entry.size;
            self.save_index()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Ensures enough space for a new entry.
    fn ensure_space(&self, needed: u64) -> std::io::Result<()> {
        let current = *self.current_size.read();
        if current + needed <= self.max_size {
            return Ok(());
        }

        // Evict LRU entries
        let mut entries: Vec<_> = self.entries.read().values().cloned().collect();
        entries.sort_by_key(|e| e.last_accessed);

        let mut to_remove = Vec::new();
        let mut freed = 0u64;

        for entry in entries {
            if current + needed - freed <= self.max_size {
                break;
            }
            freed += entry.size;
            to_remove.push(entry.model_id.clone());
        }

        for model_id in to_remove {
            self.remove(&model_id)?;
        }

        Ok(())
    }

    /// Saves the cache index.
    fn save_index(&self) -> std::io::Result<()> {
        let entries = self.entries.read();
        let content = serde_json::to_string_pretty(&*entries)?;
        std::fs::write(&self.index_file, content)?;
        Ok(())
    }

    /// Returns cache statistics.
    pub fn stats(&self) -> CacheStats {
        let entries = self.entries.read();
        CacheStats {
            entry_count: entries.len(),
            total_size: *self.current_size.read(),
            max_size: self.max_size,
            utilization: *self.current_size.read() as f64 / self.max_size as f64,
        }
    }

    /// Lists all cached models.
    pub fn list(&self) -> Vec<CacheEntry> {
        self.entries.read().values().cloned().collect()
    }

    /// Clears the entire cache.
    pub fn clear(&self) -> std::io::Result<()> {
        let entries: Vec<_> = self.entries.read().keys().cloned().collect();
        for model_id in entries {
            self.remove(&model_id)?;
        }
        Ok(())
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached entries.
    pub entry_count: usize,
    /// Total size in bytes.
    pub total_size: u64,
    /// Maximum size in bytes.
    pub max_size: u64,
    /// Utilization (0.0-1.0).
    pub utilization: f64,
}

/// Lightweight inference context for edge environments.
pub struct LightweightContext {
    /// Edge configuration.
    pub config: EdgeConfig,
    /// Model cache (if offline mode enabled).
    pub cache: Option<Arc<ModelCache>>,
    /// Memory usage tracker.
    current_memory: RwLock<u64>,
}

impl LightweightContext {
    /// Creates a new lightweight context.
    pub fn new(config: EdgeConfig) -> std::io::Result<Self> {
        let cache = if config.offline_mode {
            if let Some(cache_dir) = &config.cache_dir {
                Some(Arc::new(ModelCache::new(
                    cache_dir.clone(),
                    config.max_cache_size,
                )?))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            config,
            cache,
            current_memory: RwLock::new(0),
        })
    }

    /// Checks if a model can be loaded.
    pub fn can_load_model(&self, model_size: u64) -> bool {
        let current = *self.current_memory.read();
        current + model_size <= self.config.max_memory
    }

    /// Allocates memory for a model.
    pub fn allocate(&self, size: u64) -> bool {
        let mut current = self.current_memory.write();
        if *current + size <= self.config.max_memory {
            *current += size;
            true
        } else {
            false
        }
    }

    /// Deallocates memory.
    pub fn deallocate(&self, size: u64) {
        let mut current = self.current_memory.write();
        *current = current.saturating_sub(size);
    }

    /// Returns current memory usage.
    pub fn memory_usage(&self) -> MemoryUsage {
        let current = *self.current_memory.read();
        MemoryUsage {
            used: current,
            available: self.config.max_memory - current,
            total: self.config.max_memory,
            utilization: current as f64 / self.config.max_memory as f64,
        }
    }

    /// Gets model from cache if available.
    pub fn get_cached_model(&self, model_id: &str) -> Option<PathBuf> {
        self.cache.as_ref().and_then(|c| c.get_path(model_id))
    }

    /// Caches a model.
    pub fn cache_model(
        &self,
        model_id: &str,
        source_path: &std::path::Path,
    ) -> std::io::Result<Option<PathBuf>> {
        if let Some(cache) = &self.cache {
            Ok(Some(cache.add(model_id, source_path, self.config.quantization)?))
        } else {
            Ok(None)
        }
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Currently used memory in bytes.
    pub used: u64,
    /// Available memory in bytes.
    pub available: u64,
    /// Total memory in bytes.
    pub total: u64,
    /// Utilization (0.0-1.0).
    pub utilization: f64,
}

/// WASM-specific helpers.
#[cfg(target_arch = "wasm32")]
pub mod wasm {
    use super::*;

    /// Initializes the WASM environment.
    pub fn init() {
        // Set panic hook for better error messages in WASM
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();
    }

    /// Returns estimated available memory in WASM.
    pub fn available_memory() -> u64 {
        // WASM memory is limited, estimate based on typical browser limits
        2 * 1024 * 1024 * 1024 // 2GB typical limit
    }
}

/// Non-WASM stubs.
#[cfg(not(target_arch = "wasm32"))]
pub mod wasm {
    /// Initializes the WASM environment (no-op on non-WASM).
    pub fn init() {}

    /// Returns estimated available memory.
    pub fn available_memory() -> u64 {
        // On native, use system memory
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb * 1024;
                            }
                        }
                    }
                }
            }
        }

        // Default to 8GB
        8 * 1024 * 1024 * 1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_targets() {
        let wasm = EdgeTarget::Wasm;
        assert!(!wasm.has_gpu());
        assert_eq!(wasm.max_context_length(), 2048);

        let ios = EdgeTarget::Ios;
        assert!(ios.has_gpu());
        assert_eq!(ios.max_context_length(), 4096);
    }

    #[test]
    fn test_quantization_levels() {
        let q4 = QuantizationLevel::Q4_K_M;
        assert_eq!(q4.compression_ratio(), 4.0);
        assert!(q4.quality_factor() > 0.9);

        let q2 = QuantizationLevel::Q2_K;
        assert_eq!(q2.compression_ratio(), 8.0);
        assert!(q2.quality_factor() < 0.85);
    }

    #[test]
    fn test_edge_config() {
        let config = EdgeConfig::wasm();
        assert_eq!(config.target, EdgeTarget::Wasm);
        assert_eq!(config.num_threads, 1);
        assert!(!config.use_mmap);

        let config = EdgeConfig::mobile();
        assert!(config.use_mmap);
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_config_validation() {
        let mut config = EdgeConfig::wasm();
        config.max_model_size = config.max_memory + 1;

        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_lightweight_context() {
        let config = EdgeConfig::for_target(EdgeTarget::LightweightDesktop);
        let ctx = LightweightContext::new(config).unwrap();

        assert!(ctx.can_load_model(1024 * 1024 * 1024)); // 1GB
        assert!(ctx.allocate(500 * 1024 * 1024)); // 500MB

        let usage = ctx.memory_usage();
        assert_eq!(usage.used, 500 * 1024 * 1024);
    }
}
