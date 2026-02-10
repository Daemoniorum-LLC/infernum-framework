//! Lazy VarBuilder for on-demand tensor loading.
//!
//! This module provides a lazy alternative to Candle's `VarBuilder` that loads
//! tensors on-demand rather than requiring all tensors in memory upfront.
//!
//! ## Design
//!
//! Unlike `VarBuilder::from_tensors()` which requires all tensors to be loaded
//! into a HashMap before use, `LazyVarBuilder` loads tensors only when accessed
//! via `get()` or `pp()` methods.
//!
//! ## Features
//!
//! - **On-demand loading**: Tensors loaded only when accessed
//! - **LRU cache**: Recently used tensors cached to avoid reloading
//! - **Memory bounded**: Cache respects memory budget
//! - **Compatible API**: Works with existing model code expecting VarBuilder

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use candle_core::{DType, Device, Tensor};

use crate::hct::{HctError, HctLoader};

/// Trait for providing tensors on-demand.
///
/// This allows different backends (directory, network, etc.) to supply tensors
/// to the lazy VarBuilder.
pub trait TensorProvider: Send + Sync {
    /// Get a tensor by name, loading it if necessary.
    fn get(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor, HctError>;

    /// Check if a tensor exists without loading it.
    fn contains(&self, name: &str) -> bool;

    /// List all available tensor names.
    fn tensor_names(&self) -> Vec<String>;

    /// Clear cached tensors whose names start with the given prefix.
    ///
    /// This is used to free memory when layers are evicted.
    /// Returns (count, bytes) of evicted tensors.
    /// Default implementation does nothing (for providers without caching).
    fn clear_prefix(&self, _prefix: &str) -> (usize, u64) {
        (0, 0)
    }
}

/// Cache entry for loaded tensors.
#[derive(Debug)]
struct CacheEntry {
    tensor: Tensor,
    size_bytes: u64,
    last_access: std::time::Instant,
}

/// LRU cache configuration.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum memory for cached tensors.
    pub max_memory_bytes: u64,
    /// Maximum number of cached tensors.
    pub max_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            // 60GB cache - fits 405B model layers in 80GB RAM
            // Keeps reconstructed tensors cached between forward passes
            max_memory_bytes: 60 * 1024 * 1024 * 1024, // 60GB
            // 405B has 126 layers × ~100 tensors each = ~12,600 tensors
            max_entries: 20000,
        }
    }
}

/// Lazy VarBuilder that loads tensors on-demand.
///
/// Compatible with Candle model code that expects VarBuilder-like access patterns.
pub struct LazyVarBuilder {
    /// Tensor provider for loading.
    provider: Arc<dyn TensorProvider>,
    /// Current path prefix (for nested access).
    prefix: String,
    /// Target device.
    device: Device,
    /// Target dtype.
    dtype: DType,
    /// LRU cache for loaded tensors.
    cache: Arc<RwLock<LruCache>>,
    /// Cache configuration.
    cache_config: CacheConfig,
}

/// LRU cache for tensors.
struct LruCache {
    entries: HashMap<String, CacheEntry>,
    total_bytes: u64,
    max_bytes: u64,
    max_entries: usize,
}

impl LruCache {
    fn new(config: &CacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            total_bytes: 0,
            max_bytes: config.max_memory_bytes,
            max_entries: config.max_entries,
        }
    }

    fn get(&mut self, name: &str) -> Option<Tensor> {
        if let Some(entry) = self.entries.get_mut(name) {
            entry.last_access = std::time::Instant::now();
            Some(entry.tensor.clone())
        } else {
            None
        }
    }

    fn insert(&mut self, name: String, tensor: Tensor, size_bytes: u64) {
        // Skip insertion if cache is disabled (max_bytes=0 or max_entries=0)
        // This prevents double-caching when TieredHoloLoader already caches tensors
        if self.max_bytes == 0 || self.max_entries == 0 {
            return;
        }

        // Evict if necessary
        while self.total_bytes + size_bytes > self.max_bytes
            || self.entries.len() >= self.max_entries
        {
            if !self.evict_lru() {
                break; // No more entries to evict
            }
        }

        // Final check - don't insert if still over budget (shouldn't happen, but defensive)
        if self.total_bytes + size_bytes > self.max_bytes || self.entries.len() >= self.max_entries
        {
            return;
        }

        self.entries.insert(
            name,
            CacheEntry {
                tensor,
                size_bytes,
                last_access: std::time::Instant::now(),
            },
        );
        self.total_bytes += size_bytes;
    }

    fn evict_lru(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }

        // Find oldest entry
        let oldest = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(name, _)| name.clone());

        if let Some(name) = oldest {
            if let Some(entry) = self.entries.remove(&name) {
                self.total_bytes = self.total_bytes.saturating_sub(entry.size_bytes);
                tracing::debug!(
                    name = %name,
                    size_bytes = entry.size_bytes,
                    "Evicted tensor from cache"
                );
                return true;
            }
        }
        false
    }

    /// Clears all tensors whose names start with the given prefix.
    ///
    /// This is used to evict layer tensors when LazyLlama evicts a layer.
    fn clear_prefix(&mut self, prefix: &str) -> (usize, u64) {
        let keys_to_remove: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();

        let mut evicted_count = 0;
        let mut evicted_bytes = 0u64;

        for key in keys_to_remove {
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(entry.size_bytes);
                evicted_bytes += entry.size_bytes;
                evicted_count += 1;
            }
        }

        if evicted_count > 0 {
            tracing::debug!(
                prefix = %prefix,
                evicted_count = evicted_count,
                evicted_bytes = evicted_bytes,
                "Cleared tensors by prefix"
            );
        }

        (evicted_count, evicted_bytes)
    }

    fn memory_used(&self) -> u64 {
        self.total_bytes
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl LazyVarBuilder {
    /// Creates a new lazy VarBuilder from a tensor provider.
    pub fn new(provider: Arc<dyn TensorProvider>, device: Device, dtype: DType) -> Self {
        Self::with_cache_config(provider, device, dtype, CacheConfig::default())
    }

    /// Creates a new lazy VarBuilder with custom cache configuration.
    pub fn with_cache_config(
        provider: Arc<dyn TensorProvider>,
        device: Device,
        dtype: DType,
        cache_config: CacheConfig,
    ) -> Self {
        let cache = Arc::new(RwLock::new(LruCache::new(&cache_config)));
        Self {
            provider,
            prefix: String::new(),
            device,
            dtype,
            cache,
            cache_config,
        }
    }

    /// Creates a child VarBuilder with a path prefix.
    ///
    /// This is analogous to `VarBuilder::pp()` for accessing nested tensors.
    pub fn pp(&self, prefix: impl AsRef<str>) -> Self {
        let new_prefix = if self.prefix.is_empty() {
            prefix.as_ref().to_string()
        } else {
            format!("{}.{}", self.prefix, prefix.as_ref())
        };

        Self {
            provider: Arc::clone(&self.provider),
            prefix: new_prefix,
            device: self.device.clone(),
            dtype: self.dtype,
            cache: Arc::clone(&self.cache),
            cache_config: self.cache_config.clone(),
        }
    }

    /// Gets a tensor by name, loading it if not cached.
    ///
    /// The name is combined with the current prefix to form the full tensor path.
    pub fn get(&self, name: &str) -> Result<Tensor, HctError> {
        let full_name = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };

        // Check cache first
        {
            let mut cache = self.cache.write().map_err(|_| HctError::Tensor {
                message: "Cache lock poisoned".to_string(),
            })?;
            if let Some(tensor) = cache.get(&full_name) {
                return Ok(tensor);
            }
        }

        // Load from provider
        let tensor = self.provider.get(&full_name, &self.device, self.dtype)?;
        let size_bytes = tensor.elem_count() as u64 * dtype_size(tensor.dtype());

        // Cache the tensor
        {
            let mut cache = self.cache.write().map_err(|_| HctError::Tensor {
                message: "Cache lock poisoned".to_string(),
            })?;
            cache.insert(full_name, tensor.clone(), size_bytes);
        }

        Ok(tensor)
    }

    /// Gets a tensor with a specific shape, used for weight initialization.
    pub fn get_with_hints<S: Into<candle_core::Shape>>(
        &self,
        name: &str,
        _hint: S,
    ) -> Result<Tensor, HctError> {
        // For lazy loading, we ignore hints and just load the tensor
        self.get(name)
    }

    /// Checks if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        let full_name = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        self.provider.contains(&full_name)
    }

    /// Returns the current device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the current dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the current cache statistics.
    pub fn cache_stats(&self) -> (usize, u64) {
        if let Ok(cache) = self.cache.read() {
            (cache.len(), cache.memory_used())
        } else {
            (0, 0)
        }
    }

    /// Clears all cached tensors whose names start with the given prefix.
    ///
    /// This is used by LazyLlama to clear layer tensors when evicting a layer.
    /// Clears both the LazyVarBuilder cache AND the underlying provider's cache.
    /// Returns (evicted_count, evicted_bytes).
    pub fn clear_prefix(&self, prefix: &str) -> (usize, u64) {
        tracing::debug!(prefix = %prefix, "LazyVarBuilder: clear_prefix called");

        // Clear from our cache
        let (local_count, local_bytes) = if let Ok(mut cache) = self.cache.write() {
            cache.clear_prefix(prefix)
        } else {
            (0, 0)
        };
        tracing::debug!(
            local_count = local_count,
            "LazyVarBuilder: Cleared from local cache"
        );

        // Also clear from the provider's cache (e.g., TieredHoloLoader)
        let (provider_count, provider_bytes) = self.provider.clear_prefix(prefix);
        tracing::debug!(
            provider_count = provider_count,
            "LazyVarBuilder: Cleared from provider cache"
        );

        (local_count + provider_count, local_bytes + provider_bytes)
    }

    /// Clears all cached tensors.
    pub fn clear_all(&self) -> (usize, u64) {
        if let Ok(mut cache) = self.cache.write() {
            let count = cache.len();
            let bytes = cache.memory_used();
            cache.entries.clear();
            cache.total_bytes = 0;
            (count, bytes)
        } else {
            (0, 0)
        }
    }
}

/// Returns the size in bytes of a single element of the given dtype.
fn dtype_size(dtype: DType) -> u64 {
    match dtype {
        DType::F32 | DType::U32 => 4,
        DType::F64 | DType::I64 => 8,
        DType::F16 | DType::BF16 => 2,
        DType::U8 => 1,
        // Handle new candle_core DType variants (I16, I32, F8E4M3, etc.)
        _ => 4,
    }
}

/// Directory-based tensor provider using HCT files.
#[allow(dead_code)]
pub struct DirectoryTensorProvider {
    /// Directory containing HCT files.
    directory: PathBuf,
    /// Mapping from tensor names to file paths.
    tensor_files: HashMap<String, PathBuf>,
}

impl DirectoryTensorProvider {
    /// Creates a new directory provider.
    pub fn new(directory: impl AsRef<Path>) -> Result<Self, HctError> {
        let directory = directory.as_ref().to_path_buf();

        // Scan directory for HCT files
        let mut tensor_files = HashMap::new();

        for entry in std::fs::read_dir(&directory).map_err(|e| HctError::Io {
            path: directory.clone(),
            message: e.to_string(),
        })? {
            let entry = entry.map_err(|e| HctError::Io {
                path: directory.clone(),
                message: e.to_string(),
            })?;

            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "hct") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    let tensor_name = crate::hct::filename_to_tensor_name(name);
                    tensor_files.insert(tensor_name, path);
                }
            }
        }

        tracing::info!(
            count = tensor_files.len(),
            directory = %directory.display(),
            "Created directory tensor provider"
        );

        Ok(Self {
            directory,
            tensor_files,
        })
    }

    /// Returns the number of available tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensor_files.len()
    }
}

impl TensorProvider for DirectoryTensorProvider {
    fn get(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor, HctError> {
        let path = self
            .tensor_files
            .get(name)
            .ok_or_else(|| HctError::Tensor {
                message: format!("Tensor not found: {}", name),
            })?;

        let loader = HctLoader::from_file(path)?;
        loader.to_tensor(device, Some(dtype))
    }

    fn contains(&self, name: &str) -> bool {
        self.tensor_files.contains_key(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensor_files.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    /// Mock tensor provider for testing.
    struct MockTensorProvider {
        tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    }

    impl MockTensorProvider {
        fn new() -> Self {
            Self {
                tensors: HashMap::new(),
            }
        }

        fn add_tensor(&mut self, name: &str, shape: Vec<usize>, data: Vec<f32>) {
            self.tensors.insert(name.to_string(), (shape, data));
        }
    }

    impl TensorProvider for MockTensorProvider {
        fn get(&self, name: &str, device: &Device, _dtype: DType) -> Result<Tensor, HctError> {
            let (shape, data) = self.tensors.get(name).ok_or_else(|| HctError::Tensor {
                message: format!("Tensor not found: {}", name),
            })?;

            Tensor::from_vec(data.clone(), shape.as_slice(), device).map_err(|e| HctError::Tensor {
                message: format!("Failed to create tensor: {}", e),
            })
        }

        fn contains(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }

        fn tensor_names(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }
    }

    #[test]
    fn test_lazy_varbuilder_defers_loading() {
        let mut provider = MockTensorProvider::new();
        provider.add_tensor("weight", vec![4], vec![1.0, 2.0, 3.0, 4.0]);

        let vb = LazyVarBuilder::new(Arc::new(provider), Device::Cpu, DType::F32);

        // At this point, no tensor should be loaded
        let (cached_count, _) = vb.cache_stats();
        assert_eq!(cached_count, 0, "No tensors should be cached initially");

        // Now access the tensor
        let tensor = vb.get("weight").expect("load tensor");
        assert_eq!(tensor.dims(), &[4]);

        // Now it should be cached
        let (cached_count, _) = vb.cache_stats();
        assert_eq!(cached_count, 1, "One tensor should be cached after access");
    }

    #[test]
    fn test_lazy_varbuilder_caches_accessed_tensors() {
        let mut provider = MockTensorProvider::new();
        provider.add_tensor("tensor_a", vec![2], vec![1.0, 2.0]);
        provider.add_tensor("tensor_b", vec![2], vec![3.0, 4.0]);

        let vb = LazyVarBuilder::new(Arc::new(provider), Device::Cpu, DType::F32);

        // Access tensor_a
        let _a = vb.get("tensor_a").expect("load tensor_a");
        let (count, _) = vb.cache_stats();
        assert_eq!(count, 1);

        // Access tensor_a again (should hit cache)
        let a2 = vb.get("tensor_a").expect("load tensor_a again");
        let (count, _) = vb.cache_stats();
        assert_eq!(count, 1, "Should still be 1, tensor was cached");

        // Access tensor_b
        let _b = vb.get("tensor_b").expect("load tensor_b");
        let (count, _) = vb.cache_stats();
        assert_eq!(count, 2, "Should be 2 after loading tensor_b");
    }

    #[test]
    fn test_lazy_varbuilder_evicts_lru_when_full() {
        let mut provider = MockTensorProvider::new();
        // Each tensor is 4 floats = 16 bytes
        provider.add_tensor("tensor_1", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        provider.add_tensor("tensor_2", vec![4], vec![5.0, 6.0, 7.0, 8.0]);
        provider.add_tensor("tensor_3", vec![4], vec![9.0, 10.0, 11.0, 12.0]);

        // Set very small cache: max 32 bytes (2 tensors)
        let config = CacheConfig {
            max_memory_bytes: 32,
            max_entries: 2,
        };

        let vb =
            LazyVarBuilder::with_cache_config(Arc::new(provider), Device::Cpu, DType::F32, config);

        // Load tensors 1 and 2
        let _t1 = vb.get("tensor_1").expect("load tensor_1");
        let _t2 = vb.get("tensor_2").expect("load tensor_2");

        let (count, _) = vb.cache_stats();
        assert_eq!(count, 2);

        // Load tensor_3 - should evict tensor_1 (LRU)
        let _t3 = vb.get("tensor_3").expect("load tensor_3");

        let (count, _) = vb.cache_stats();
        assert_eq!(count, 2, "Should still be 2 after eviction");
    }

    #[test]
    fn test_lazy_varbuilder_pp_get_works_for_model_layers() {
        let mut provider = MockTensorProvider::new();
        provider.add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            vec![4],
            vec![1.0, 2.0, 3.0, 4.0],
        );
        provider.add_tensor(
            "model.layers.0.self_attn.k_proj.weight",
            vec![4],
            vec![5.0, 6.0, 7.0, 8.0],
        );
        provider.add_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            vec![4],
            vec![9.0, 10.0, 11.0, 12.0],
        );

        let vb = LazyVarBuilder::new(Arc::new(provider), Device::Cpu, DType::F32);

        // Navigate using pp()
        let layer_vb = vb.pp("model").pp("layers").pp("0");

        // Access self_attn tensors
        let attn_vb = layer_vb.pp("self_attn");
        let q = attn_vb.get("q_proj.weight").expect("load q_proj");
        let k = attn_vb.get("k_proj.weight").expect("load k_proj");

        assert_eq!(q.dims(), &[4]);
        assert_eq!(k.dims(), &[4]);

        // Access mlp tensors
        let mlp_vb = layer_vb.pp("mlp");
        let gate = mlp_vb.get("gate_proj.weight").expect("load gate_proj");

        assert_eq!(gate.dims(), &[4]);

        // Check all are cached
        let (count, _) = vb.cache_stats();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_lazy_varbuilder_contains() {
        let mut provider = MockTensorProvider::new();
        provider.add_tensor("existing", vec![2], vec![1.0, 2.0]);

        let vb = LazyVarBuilder::new(Arc::new(provider), Device::Cpu, DType::F32);

        assert!(vb.contains("existing"));
        assert!(!vb.contains("nonexistent"));
    }

    #[test]
    fn test_lru_cache_eviction_order() {
        let config = CacheConfig {
            max_memory_bytes: 100,
            max_entries: 3,
        };
        let mut cache = LruCache::new(&config);

        // Insert 3 tensors
        let t1 = Tensor::zeros(&[2], DType::F32, &Device::Cpu).expect("create t1");
        let t2 = Tensor::zeros(&[2], DType::F32, &Device::Cpu).expect("create t2");
        let t3 = Tensor::zeros(&[2], DType::F32, &Device::Cpu).expect("create t3");

        cache.insert("t1".to_string(), t1, 8);
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.insert("t2".to_string(), t2, 8);
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.insert("t3".to_string(), t3, 8);

        assert_eq!(cache.len(), 3);

        // Access t1 to make it most recently used
        let _ = cache.get("t1");

        // Insert t4 - should evict t2 (now LRU)
        let t4 = Tensor::zeros(&[2], DType::F32, &Device::Cpu).expect("create t4");
        cache.insert("t4".to_string(), t4, 8);

        assert_eq!(cache.len(), 3);
        assert!(cache.get("t1").is_some(), "t1 should still be in cache");
        assert!(cache.get("t2").is_none(), "t2 should have been evicted");
        assert!(cache.get("t3").is_some(), "t3 should still be in cache");
        assert!(cache.get("t4").is_some(), "t4 should be in cache");
    }
}
