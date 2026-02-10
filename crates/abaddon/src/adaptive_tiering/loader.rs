//! Adaptive tensor loader that uses allocation plans for intelligent caching.
//!
//! The AdaptiveLoader wraps an underlying `TensorProvider` and adds intelligent
//! caching based on an `AllocationPlan`. It manages VRAM and RAM caches according
//! to the planned allocation for each tensor.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};

use super::types::{AllocationPlan, MemoryTier, TensorAllocation, TensorPrecision};
use crate::hct::HctError;
use crate::lazy_varbuilder::TensorProvider;

/// Tensor provider backed by pre-loaded tensors in RAM.
///
/// This provider stores all tensors on CPU (RAM) and serves them without any
/// loading overhead. It's designed for the eager loading path where tensors
/// are decompressed upfront using `hct_sequential`.
///
/// The `AdaptiveLoader` can wrap this provider to add VRAM/RAM tiering on top
/// of the pre-loaded tensors.
pub struct EagerTensorProvider {
    /// Pre-loaded tensors stored on CPU.
    tensors: HashMap<String, Tensor>,
}

impl EagerTensorProvider {
    /// Creates a new eager tensor provider from pre-loaded tensors.
    ///
    /// # Arguments
    /// * `tensors` - HashMap of tensor name to tensor, loaded on CPU
    pub fn new(tensors: HashMap<String, Tensor>) -> Self {
        Self { tensors }
    }

    /// Returns the number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns true if there are no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns total size in bytes.
    pub fn total_bytes(&self) -> u64 {
        self.tensors
            .values()
            .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
            .sum()
    }
}

impl TensorProvider for EagerTensorProvider {
    fn get(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor, HctError> {
        let tensor = self.tensors.get(name).ok_or_else(|| HctError::Tensor {
            message: format!("tensor not found: {name}"),
        })?;

        // Convert dtype if needed
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype).map_err(|e| HctError::Tensor {
                message: format!("dtype conversion failed: {e}"),
            })?
        } else {
            tensor.clone()
        };

        // Move to device if needed (this is where VRAM transfer happens)
        if tensor.device().location() != device.location() {
            tensor.to_device(device).map_err(|e| HctError::Tensor {
                message: format!("device transfer failed: {e}"),
            })
        } else {
            Ok(tensor)
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
}

/// Statistics for the adaptive loader.
#[derive(Debug, Clone, Default)]
pub struct AdaptiveLoaderStats {
    /// Tensors served from VRAM cache.
    pub vram_hits: usize,
    /// Tensors served from RAM cache.
    pub ram_hits: usize,
    /// Tensors loaded from underlying provider.
    pub provider_loads: usize,
    /// Total bytes loaded.
    pub bytes_loaded: u64,
    /// Total load time in milliseconds.
    pub total_load_time_ms: u64,
}

/// Cached tensor entry.
struct CachedTensor {
    /// The tensor data.
    tensor: Tensor,
    /// Size in bytes.
    size_bytes: u64,
    /// Last access time for LRU eviction.
    last_access: Instant,
}

/// Adaptive tensor loader.
///
/// Wraps an underlying `TensorProvider` and adds intelligent caching based on
/// an allocation plan. Tensors are cached in VRAM or RAM according to their
/// planned tier.
pub struct AdaptiveLoader {
    /// The allocation plan determining placement.
    plan: AllocationPlan,
    /// Underlying tensor provider for loading.
    provider: Box<dyn TensorProvider>,
    /// VRAM tensor cache (resident tensors).
    vram_cache: RwLock<HashMap<String, CachedTensor>>,
    /// RAM tensor cache (warm tensors).
    ram_cache: RwLock<HashMap<String, CachedTensor>>,
    /// Target device for VRAM tensors.
    device: Device,
    /// Loading statistics.
    stats: RwLock<AdaptiveLoaderStats>,
}

/// Errors from the adaptive loader.
#[derive(Debug, thiserror::Error)]
pub enum AdaptiveLoaderError {
    /// Underlying provider error.
    #[error("provider error: {0}")]
    Provider(String),

    /// Tensor not found in plan.
    #[error("tensor not in allocation plan: {0}")]
    TensorNotInPlan(String),

    /// Cache lock error.
    #[error("cache lock error: {0}")]
    CacheLock(String),
}

impl AdaptiveLoader {
    /// Creates a new adaptive loader.
    ///
    /// # Arguments
    /// * `plan` - Allocation plan from the planner
    /// * `provider` - Underlying tensor provider (e.g., DirectoryTensorProvider)
    /// * `device` - Target device for VRAM tensors
    pub fn new(
        plan: AllocationPlan,
        provider: impl TensorProvider + 'static,
        device: Device,
    ) -> Self {
        Self {
            plan,
            provider: Box::new(provider),
            vram_cache: RwLock::new(HashMap::new()),
            ram_cache: RwLock::new(HashMap::new()),
            device,
            stats: RwLock::new(AdaptiveLoaderStats::default()),
        }
    }

    /// Returns current loading statistics.
    pub fn stats(&self) -> AdaptiveLoaderStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Returns the allocation plan.
    pub fn plan(&self) -> &AllocationPlan {
        &self.plan
    }

    /// Preloads tensors scheduled for VRAM.
    ///
    /// This should be called after construction to populate the VRAM cache
    /// before inference begins.
    ///
    /// # Errors
    /// Returns error if any tensor fails to load.
    pub fn preload_vram_tensors(&self) -> Result<(), AdaptiveLoaderError> {
        let vram_tensors: Vec<_> = self
            .plan
            .allocations
            .iter()
            .filter(|(_, alloc)| alloc.tier == MemoryTier::Vram)
            .map(|(name, _)| name.clone())
            .collect();

        tracing::info!(count = vram_tensors.len(), "Preloading VRAM tensors");

        for name in vram_tensors {
            let _ = self.load_tensor_internal(&name, &self.device, DType::BF16)?;
        }

        tracing::info!("VRAM preload complete");
        Ok(())
    }

    /// Gets current VRAM cache usage in bytes.
    pub fn vram_cache_usage(&self) -> u64 {
        self.vram_cache
            .read()
            .map(|cache| cache.values().map(|e| e.size_bytes).sum())
            .unwrap_or(0)
    }

    /// Gets current RAM cache usage in bytes.
    pub fn ram_cache_usage(&self) -> u64 {
        self.ram_cache
            .read()
            .map(|cache| cache.values().map(|e| e.size_bytes).sum())
            .unwrap_or(0)
    }

    /// Internal tensor loading with allocation-aware caching.
    fn load_tensor_internal(
        &self,
        name: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor, AdaptiveLoaderError> {
        let start = Instant::now();

        // Get allocation for this tensor (fallback to NVMe if not in plan)
        let alloc = self
            .plan
            .allocations
            .get(name)
            .cloned()
            .unwrap_or_else(|| TensorAllocation {
                tier: MemoryTier::Nvme,
                precision: TensorPrecision::BF16,
                priority: 0.0,
                prefetch: false,
                storage_size: 0,
            });

        // Check VRAM cache first
        if let Some(cached) = self.try_get_cached(name, MemoryTier::Vram)? {
            if let Ok(mut stats) = self.stats.write() {
                stats.vram_hits += 1;
            }
            return Ok(cached);
        }

        // Check RAM cache
        if let Some(cached) = self.try_get_cached(name, MemoryTier::Ram)? {
            if let Ok(mut stats) = self.stats.write() {
                stats.ram_hits += 1;
            }

            // If allocation says VRAM, promote from RAM
            if alloc.tier == MemoryTier::Vram {
                return self.promote_to_vram(name, cached, &alloc);
            }

            return Ok(cached);
        }

        // Load from underlying provider
        let tensor = self
            .provider
            .get(name, device, dtype)
            .map_err(|e| AdaptiveLoaderError::Provider(e.to_string()))?;

        let elapsed = start.elapsed();
        let size_bytes = tensor.elem_count() as u64 * dtype.size_in_bytes() as u64;

        if let Ok(mut stats) = self.stats.write() {
            stats.provider_loads += 1;
            stats.bytes_loaded += size_bytes;
            stats.total_load_time_ms += elapsed.as_millis() as u64;
        }

        // Cache according to allocation tier
        self.cache_tensor(name, &tensor, &alloc)?;

        Ok(tensor)
    }

    /// Tries to get a tensor from cache.
    fn try_get_cached(
        &self,
        name: &str,
        tier: MemoryTier,
    ) -> Result<Option<Tensor>, AdaptiveLoaderError> {
        let cache = match tier {
            MemoryTier::Vram => &self.vram_cache,
            MemoryTier::Ram => &self.ram_cache,
            MemoryTier::Nvme => return Ok(None),
        };

        let mut cache = cache
            .write()
            .map_err(|_| AdaptiveLoaderError::CacheLock("cache write lock".into()))?;

        if let Some(entry) = cache.get_mut(name) {
            entry.last_access = Instant::now();
            return Ok(Some(entry.tensor.clone()));
        }

        Ok(None)
    }

    /// Promotes a tensor from RAM to VRAM.
    fn promote_to_vram(
        &self,
        name: &str,
        tensor: Tensor,
        _alloc: &TensorAllocation,
    ) -> Result<Tensor, AdaptiveLoaderError> {
        // Move to VRAM device
        let vram_tensor = tensor
            .to_device(&self.device)
            .map_err(|e| AdaptiveLoaderError::Provider(format!("device transfer: {e}")))?;

        // Update caches
        if let Ok(mut ram_cache) = self.ram_cache.write() {
            ram_cache.remove(name);
        }

        let size_bytes =
            vram_tensor.elem_count() as u64 * vram_tensor.dtype().size_in_bytes() as u64;

        if let Ok(mut vram_cache) = self.vram_cache.write() {
            vram_cache.insert(
                name.to_string(),
                CachedTensor {
                    tensor: vram_tensor.clone(),
                    size_bytes,
                    last_access: Instant::now(),
                },
            );
        }

        Ok(vram_tensor)
    }

    /// Caches a tensor according to its allocation.
    fn cache_tensor(
        &self,
        name: &str,
        tensor: &Tensor,
        alloc: &TensorAllocation,
    ) -> Result<(), AdaptiveLoaderError> {
        let size_bytes = tensor.elem_count() as u64 * tensor.dtype().size_in_bytes() as u64;
        let entry = CachedTensor {
            tensor: tensor.clone(),
            size_bytes,
            last_access: Instant::now(),
        };

        match alloc.tier {
            MemoryTier::Vram => {
                if let Ok(mut cache) = self.vram_cache.write() {
                    cache.insert(name.to_string(), entry);
                }
            },
            MemoryTier::Ram => {
                if let Ok(mut cache) = self.ram_cache.write() {
                    cache.insert(name.to_string(), entry);
                }
            },
            MemoryTier::Nvme => {
                // Don't cache NVMe tensors
            },
        }

        Ok(())
    }

    /// Evicts tensors to free VRAM for KV cache growth.
    ///
    /// # Arguments
    /// * `bytes_needed` - Amount of VRAM to free
    ///
    /// # Returns
    /// Number of bytes actually freed.
    pub fn evict_for_kv_cache(&self, bytes_needed: u64) -> u64 {
        let mut freed = 0u64;

        if let Ok(mut vram_cache) = self.vram_cache.write() {
            // Sort by priority (lowest first) then by last access (oldest first)
            let mut candidates: Vec<_> = vram_cache
                .iter()
                .map(|(name, entry)| {
                    let priority = self
                        .plan
                        .allocations
                        .get(name)
                        .map(|a| a.priority)
                        .unwrap_or(0.0);
                    (name.clone(), priority, entry.last_access, entry.size_bytes)
                })
                .collect();

            candidates.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.2.cmp(&b.2))
            });

            for (name, _priority, _access, size) in candidates {
                if freed >= bytes_needed {
                    break;
                }

                if let Some(entry) = vram_cache.remove(&name) {
                    // Demote to RAM cache
                    if let Ok(mut ram_cache) = self.ram_cache.write() {
                        ram_cache.insert(
                            name,
                            CachedTensor {
                                tensor: entry.tensor,
                                size_bytes: entry.size_bytes,
                                last_access: entry.last_access,
                            },
                        );
                    }
                    freed += size;
                }
            }
        }

        freed
    }
}

impl TensorProvider for AdaptiveLoader {
    fn get(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor, HctError> {
        self.load_tensor_internal(name, device, dtype)
            .map_err(|e| HctError::Tensor {
                message: e.to_string(),
            })
    }

    fn contains(&self, name: &str) -> bool {
        self.provider.contains(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.provider.tensor_names()
    }

    fn clear_prefix(&self, prefix: &str) -> (usize, u64) {
        let mut count = 0usize;
        let mut bytes = 0u64;

        // Clear from VRAM cache
        if let Ok(mut cache) = self.vram_cache.write() {
            let to_remove: Vec<_> = cache
                .keys()
                .filter(|k| k.starts_with(prefix))
                .cloned()
                .collect();

            for key in to_remove {
                if let Some(entry) = cache.remove(&key) {
                    count += 1;
                    bytes += entry.size_bytes;
                }
            }
        }

        // Clear from RAM cache
        if let Ok(mut cache) = self.ram_cache.write() {
            let to_remove: Vec<_> = cache
                .keys()
                .filter(|k| k.starts_with(prefix))
                .cloned()
                .collect();

            for key in to_remove {
                if let Some(entry) = cache.remove(&key) {
                    count += 1;
                    bytes += entry.size_bytes;
                }
            }
        }

        // Also clear from underlying provider
        let (provider_count, provider_bytes) = self.provider.clear_prefix(prefix);
        count += provider_count;
        bytes += provider_bytes;

        (count, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_tiering::{
        AdaptiveTieringConfig, AllocationPlanner, ModelProfile, TensorInfo,
    };

    /// Mock tensor provider for testing.
    struct MockProvider {
        tensors: HashMap<String, (Vec<usize>, DType)>,
    }

    impl MockProvider {
        fn new() -> Self {
            Self {
                tensors: HashMap::new(),
            }
        }

        fn add_tensor(&mut self, name: &str, shape: Vec<usize>) {
            self.tensors.insert(name.to_string(), (shape, DType::BF16));
        }
    }

    impl TensorProvider for MockProvider {
        fn get(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor, HctError> {
            let (shape, _) = self.tensors.get(name).ok_or_else(|| HctError::Tensor {
                message: format!("tensor not found: {name}"),
            })?;
            Tensor::zeros(shape.as_slice(), dtype, device).map_err(|e| HctError::Tensor {
                message: e.to_string(),
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
    fn test_adaptive_loader_caching() {
        let mut provider = MockProvider::new();
        provider.add_tensor("model.embed_tokens.weight", vec![1000, 512]);
        provider.add_tensor("model.layers.0.self_attn.q_proj.weight", vec![512, 512]);

        let tensors = vec![
            TensorInfo::from_name("model.embed_tokens.weight", 1000 * 512 * 2),
            TensorInfo::from_name("model.layers.0.self_attn.q_proj.weight", 512 * 512 * 2),
        ];
        let profile = ModelProfile::new(tensors);

        let config = AdaptiveTieringConfig::with_budgets(22.0, 60.0);
        let planner = AllocationPlanner::new(config);
        let plan = planner.plan(&profile).expect("plan should succeed");

        let loader = AdaptiveLoader::new(plan, provider, Device::Cpu);

        // First load - should hit provider
        let _ = loader
            .get("model.embed_tokens.weight", &Device::Cpu, DType::BF16)
            .expect("load should succeed");

        let stats = loader.stats();
        assert_eq!(stats.provider_loads, 1);
        assert_eq!(stats.vram_hits, 0);

        // Second load - should hit cache
        let _ = loader
            .get("model.embed_tokens.weight", &Device::Cpu, DType::BF16)
            .expect("load should succeed");

        let stats = loader.stats();
        assert_eq!(stats.provider_loads, 1); // No new provider loads
        assert!(stats.vram_hits > 0 || stats.ram_hits > 0); // Hit cache
    }

    #[test]
    fn test_adaptive_loader_eviction() {
        let mut provider = MockProvider::new();
        provider.add_tensor("tensor1", vec![100, 100]);
        provider.add_tensor("tensor2", vec![100, 100]);

        // Create a plan where both tensors go to VRAM
        let mut plan = AllocationPlan::new();
        plan.allocations.insert(
            "tensor1".to_string(),
            TensorAllocation {
                tier: MemoryTier::Vram,
                precision: TensorPrecision::BF16,
                priority: 0.5,
                prefetch: false,
                storage_size: 100 * 100 * 2,
            },
        );
        plan.allocations.insert(
            "tensor2".to_string(),
            TensorAllocation {
                tier: MemoryTier::Vram,
                precision: TensorPrecision::BF16,
                priority: 0.9, // Higher priority
                prefetch: false,
                storage_size: 100 * 100 * 2,
            },
        );

        let loader = AdaptiveLoader::new(plan, provider, Device::Cpu);

        // Load both tensors
        loader
            .get("tensor1", &Device::Cpu, DType::BF16)
            .expect("load");
        loader
            .get("tensor2", &Device::Cpu, DType::BF16)
            .expect("load");

        // Evict - should evict tensor1 first (lower priority)
        let freed = loader.evict_for_kv_cache(100 * 100 * 2);
        assert!(freed >= 100 * 100 * 2);

        // tensor2 should still be in VRAM cache
        let _ = loader
            .get("tensor2", &Device::Cpu, DType::BF16)
            .expect("load");
        let stats = loader.stats();
        assert!(stats.vram_hits > 0, "tensor2 should be cache hit");
    }
}
