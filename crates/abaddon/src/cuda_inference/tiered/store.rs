//! TieredWeightStore - Central coordinator for 3-tier memory hierarchy.
//!
//! Manages weights across VRAM (hot), RAM (warm), and NVMe (cold) tiers with:
//! - Adaptive tier placement based on access patterns
//! - LRU eviction with priority-based ordering
//! - Prefetching for sequential layer access
//! - Support for both eager and progressive loading strategies

use std::collections::HashSet;
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use super::config::{LoadingStrategy, TieredConfig};
use super::error::TieredError;
use super::nvme_cache::NvmeCache;
use super::ram_cache::{CpuLayerWeights, RamCache};
use super::stats::TieredStats;
use super::vram_cache::{SharedWeights, VramCache};
use crate::adaptive_tiering::AllocationPlan;
use crate::cuda_inference::weight_store::LayerWeights;

/// State of a layer in the tiered system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerState {
    /// Layer is resident in VRAM (hot).
    Vram,
    /// Layer is in RAM, ready for fast upload (warm).
    Ram,
    /// Layer is on NVMe, requires disk read (cold).
    Nvme,
    /// Layer is not loaded anywhere.
    Unloaded,
}

/// TieredWeightStore manages weights across VRAM, RAM, and NVMe.
///
/// This is the main interface for the tiered memory system. It coordinates:
/// - Initial loading based on AllocationPlan
/// - Runtime tier transitions (promotion/demotion)
/// - Prefetching for upcoming layers
/// - Eviction when memory pressure increases
pub struct TieredWeightStore {
    /// CUDA device for GPU operations.
    device: Arc<CudaDevice>,

    /// VRAM cache for hot layers.
    vram: VramCache,

    /// RAM cache for warm layers.
    ram: RamCache,

    /// NVMe cache for cold layers (optional).
    nvme: Option<NvmeCache>,

    /// Allocation plan from adaptive tiering.
    plan: AllocationPlan,

    /// Configuration.
    config: TieredConfig,

    /// Statistics tracker.
    stats: Arc<TieredStats>,

    /// Total number of layers in the model.
    num_layers: usize,

    /// Size per layer in bytes (for uniform models).
    layer_size: Option<usize>,

    /// Loading strategy in use.
    strategy: LoadingStrategy,

    /// Layers currently being prefetched.
    prefetching: HashSet<usize>,
}

impl TieredWeightStore {
    /// Create a new TieredWeightStore.
    ///
    /// # Arguments
    /// * `device` - CUDA device for GPU operations
    /// * `plan` - Allocation plan from adaptive tiering analysis
    /// * `config` - Tiered storage configuration
    /// * `num_layers` - Total number of transformer layers
    pub fn new(
        device: Arc<CudaDevice>,
        plan: AllocationPlan,
        config: TieredConfig,
        num_layers: usize,
    ) -> Result<Self, TieredError> {
        let stats = Arc::new(TieredStats::new());

        // Determine loading strategy from plan
        let strategy = config.select_strategy(&plan);

        let vram_budget = config.hardware.vram_budget;
        let ram_budget = config.hardware.ram_budget;

        tracing::info!(
            ?strategy,
            num_layers,
            vram_budget_mb = vram_budget / (1024 * 1024),
            ram_budget_mb = ram_budget / (1024 * 1024),
            "Creating TieredWeightStore"
        );

        // Create VRAM cache
        let vram = VramCache::new(device.clone(), vram_budget, stats.clone());

        // Create RAM cache
        let ram = RamCache::new(ram_budget, config.hardware.use_pinned_memory, stats.clone());

        // Create NVMe cache if progressive loading is enabled
        let nvme = if config.progressive.enable_caching && strategy.needs_nvme() {
            let nvme_budget = if config.progressive.max_cache_size > 0 {
                config.progressive.max_cache_size
            } else {
                100 * 1024 * 1024 * 1024 // 100GB default
            };
            Some(NvmeCache::new(
                &config.progressive.cache_dir,
                nvme_budget,
                stats.clone(),
            )?)
        } else {
            None
        };

        Ok(Self {
            device,
            vram,
            ram,
            nvme,
            plan,
            config,
            stats,
            num_layers,
            layer_size: None,
            strategy,
            prefetching: HashSet::new(),
        })
    }

    /// Set shared weights (embeddings, final norm, lm_head).
    ///
    /// These are always VRAM-resident and never evicted.
    pub fn set_shared(&mut self, shared: SharedWeights) {
        self.vram.set_shared(shared);
    }

    /// Get shared weights reference.
    pub fn shared(&self) -> Option<&SharedWeights> {
        self.vram.shared()
    }

    /// Get the current state of a layer.
    pub fn layer_state(&self, layer_idx: usize) -> LayerState {
        if self.vram.contains(layer_idx) {
            LayerState::Vram
        } else if self.ram.contains(layer_idx) {
            LayerState::Ram
        } else if self
            .nvme
            .as_ref()
            .map(|n| n.contains(layer_idx))
            .unwrap_or(false)
        {
            LayerState::Nvme
        } else {
            LayerState::Unloaded
        }
    }

    /// Get a layer from VRAM, promoting from lower tiers if needed.
    ///
    /// This is the main access method during inference. It:
    /// 1. Returns immediately if layer is in VRAM
    /// 2. Promotes from RAM if available (fast path)
    /// 3. Loads from NVMe if needed (slow path)
    /// 4. Evicts lower-priority layers if VRAM is full
    pub fn get_layer(&mut self, layer_idx: usize) -> Result<&LayerWeights, TieredError> {
        // Fast path: already in VRAM
        if self.vram.contains(layer_idx) {
            return self
                .vram
                .get(layer_idx)
                .ok_or_else(|| TieredError::LayerNotFound(layer_idx, self.num_layers));
        }

        // Promote from RAM
        if self.ram.contains(layer_idx) {
            self.promote_from_ram(layer_idx)?;
            return self
                .vram
                .get(layer_idx)
                .ok_or_else(|| TieredError::LayerNotFound(layer_idx, self.num_layers));
        }

        // Load from NVMe
        if let Some(ref nvme) = self.nvme {
            if nvme.contains(layer_idx) {
                self.promote_from_nvme(layer_idx)?;
                return self
                    .vram
                    .get(layer_idx)
                    .ok_or_else(|| TieredError::LayerNotFound(layer_idx, self.num_layers));
            }
        }

        Err(TieredError::LayerNotFound(layer_idx, self.num_layers))
    }

    /// Check if a layer is available in VRAM (no promotion needed).
    pub fn is_vram_resident(&self, layer_idx: usize) -> bool {
        self.vram.contains(layer_idx)
    }

    /// Promote a layer from RAM to VRAM.
    fn promote_from_ram(&mut self, layer_idx: usize) -> Result<(), TieredError> {
        let cpu_weights = self
            .ram
            .remove(layer_idx)
            .ok_or_else(|| TieredError::LayerNotFound(layer_idx, self.num_layers))?;

        // Ensure VRAM has room
        let size = cpu_weights.size_bytes;
        if !self.vram.has_room_for(size) {
            self.evict_vram_for_space(size as u64)?;
        }

        // Upload to GPU
        let gpu_weights = self.upload_layer(&cpu_weights)?;
        let priority = self.plan.get_layer_priority(layer_idx).unwrap_or(0.5);
        self.vram.insert(layer_idx, gpu_weights, priority)?;

        self.stats.record_layer_loaded();

        tracing::debug!(
            layer_idx,
            size_mb = size / (1024 * 1024),
            "Promoted layer RAM→VRAM"
        );

        Ok(())
    }

    /// Promote a layer from NVMe to VRAM (via RAM).
    fn promote_from_nvme(&mut self, layer_idx: usize) -> Result<(), TieredError> {
        let nvme = self
            .nvme
            .as_ref()
            .ok_or_else(|| TieredError::nvme("NVMe cache not configured"))?;

        // Read from NVMe
        let data = nvme.read_layer(layer_idx)?;

        // Decompress if needed
        let entry = nvme.get_entry(layer_idx).unwrap();
        let cpu_weights = if entry.compressed {
            self.decompress_layer(layer_idx, &data)?
        } else {
            self.deserialize_layer(layer_idx, &data)?
        };

        // Upload to VRAM
        let size = cpu_weights.size_bytes;
        if !self.vram.has_room_for(size) {
            self.evict_vram_for_space(size as u64)?;
        }

        let gpu_weights = self.upload_layer(&cpu_weights)?;
        let priority = self.plan.get_layer_priority(layer_idx).unwrap_or(0.5);
        self.vram.insert(layer_idx, gpu_weights, priority)?;

        self.stats.record_layer_loaded();

        tracing::debug!(
            layer_idx,
            size_mb = size / (1024 * 1024),
            "Promoted layer NVMe→VRAM"
        );

        Ok(())
    }

    /// Evict layers from VRAM to make room.
    fn evict_vram_for_space(&mut self, bytes_needed: u64) -> Result<(), TieredError> {
        let evicted = self.vram.evict_for_space(bytes_needed, &self.plan);

        for (layer_idx, layer) in evicted {
            // Demote to RAM if there's room
            if self.ram.has_room_for(layer.size_bytes()) {
                let cpu_weights = self.download_layer(&layer)?;
                let priority = self.plan.get_layer_priority(layer_idx).unwrap_or(0.5);
                self.ram.insert(layer_idx, cpu_weights, priority)?;

                tracing::debug!(layer_idx, "Demoted layer VRAM→RAM");
            } else if self.nvme.is_some() {
                // Demote to NVMe - serialize first to avoid borrow conflict
                let data = self.serialize_layer(&layer)?;
                if let Some(ref mut nvme) = self.nvme {
                    nvme.write_layer(layer_idx, &data, false)?;
                }

                tracing::debug!(layer_idx, "Demoted layer VRAM→NVMe");
            }
            // Otherwise the layer is dropped
        }

        Ok(())
    }

    /// Upload CPU layer weights to GPU.
    fn upload_layer(&self, cpu_weights: &CpuLayerWeights) -> Result<LayerWeights, TieredError> {
        let start = std::time::Instant::now();

        // This is a placeholder - actual implementation will use cudarc
        // to upload each tensor from the CpuLayerWeights buffer
        let layer = self.create_gpu_layer_from_cpu(cpu_weights)?;

        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.stats
            .record_vram_upload(cpu_weights.size_bytes as u64, elapsed_ns);

        Ok(layer)
    }

    /// Download GPU layer weights to CPU.
    fn download_layer(&self, _layer: &LayerWeights) -> Result<CpuLayerWeights, TieredError> {
        // Placeholder - actual implementation will download tensors from GPU
        todo!("download_layer not yet implemented")
    }

    /// Create GPU layer from CPU weights.
    fn create_gpu_layer_from_cpu(
        &self,
        _cpu: &CpuLayerWeights,
    ) -> Result<LayerWeights, TieredError> {
        // Placeholder - actual implementation will create GPU tensors
        todo!("create_gpu_layer_from_cpu not yet implemented")
    }

    /// Serialize layer for NVMe storage.
    fn serialize_layer(&self, _layer: &LayerWeights) -> Result<Vec<u8>, TieredError> {
        // Placeholder - actual implementation will serialize to bytes
        todo!("serialize_layer not yet implemented")
    }

    /// Deserialize layer from NVMe storage.
    fn deserialize_layer(
        &self,
        _layer_idx: usize,
        _data: &[u8],
    ) -> Result<CpuLayerWeights, TieredError> {
        // Placeholder - actual implementation will deserialize bytes
        todo!("deserialize_layer not yet implemented")
    }

    /// Decompress HCT-compressed layer.
    fn decompress_layer(
        &self,
        layer_idx: usize,
        _data: &[u8],
    ) -> Result<CpuLayerWeights, TieredError> {
        // Placeholder - actual implementation will use HCT decompression
        Err(TieredError::decompress(
            format!("layer_{}", layer_idx),
            "HCT decompression not yet implemented",
        ))
    }

    /// Prefetch upcoming layers.
    ///
    /// Call this with the current layer index to prefetch the next N layers.
    /// Prefetching happens asynchronously and doesn't block the caller.
    pub fn prefetch(&mut self, current_layer: usize, lookahead: usize) {
        // Prefetching only makes sense for progressive loading
        if self.config.progressive.prefetch_depth == 0 {
            return;
        }

        for offset in 1..=lookahead {
            let target = current_layer + offset;
            if target >= self.num_layers {
                break;
            }

            // Skip if already in VRAM or being prefetched
            if self.vram.contains(target) || self.prefetching.contains(&target) {
                continue;
            }

            // Skip if not in a lower tier
            let state = self.layer_state(target);
            if state == LayerState::Unloaded {
                continue;
            }

            self.prefetching.insert(target);
            self.stats.record_prefetch_request();

            tracing::trace!(target, "Prefetching layer");

            // In a full implementation, this would spawn an async task
            // For now, we just mark it as prefetching
        }
    }

    /// Get statistics snapshot.
    pub fn stats(&self) -> &TieredStats {
        &self.stats
    }

    /// Get VRAM cache reference.
    pub fn vram_cache(&self) -> &VramCache {
        &self.vram
    }

    /// Get RAM cache reference.
    pub fn ram_cache(&self) -> &RamCache {
        &self.ram
    }

    /// Get NVMe cache reference.
    pub fn nvme_cache(&self) -> Option<&NvmeCache> {
        self.nvme.as_ref()
    }

    /// Get mutable VRAM cache access.
    pub fn vram_cache_mut(&mut self) -> &mut VramCache {
        &mut self.vram
    }

    /// Get mutable RAM cache access.
    pub fn ram_cache_mut(&mut self) -> &mut RamCache {
        &mut self.ram
    }

    /// Get mutable NVMe cache access.
    pub fn nvme_cache_mut(&mut self) -> Option<&mut NvmeCache> {
        self.nvme.as_mut()
    }

    /// Get the loading strategy.
    pub fn strategy(&self) -> &LoadingStrategy {
        &self.strategy
    }

    /// Get total number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get allocation plan.
    pub fn plan(&self) -> &AllocationPlan {
        &self.plan
    }

    /// Summary of current tier distribution.
    pub fn tier_summary(&self) -> TierSummary {
        let mut vram_count = 0;
        let mut ram_count = 0;
        let mut nvme_count = 0;
        let mut unloaded_count = 0;

        for i in 0..self.num_layers {
            match self.layer_state(i) {
                LayerState::Vram => vram_count += 1,
                LayerState::Ram => ram_count += 1,
                LayerState::Nvme => nvme_count += 1,
                LayerState::Unloaded => unloaded_count += 1,
            }
        }

        TierSummary {
            vram_layers: vram_count,
            ram_layers: ram_count,
            nvme_layers: nvme_count,
            unloaded_layers: unloaded_count,
            vram_usage: self.vram.total_usage(),
            ram_usage: self.ram.usage(),
            nvme_usage: self.nvme.as_ref().map(|n| n.usage()).unwrap_or(0),
        }
    }
}

/// Extension trait for AllocationPlan.
trait AllocationPlanExt {
    fn get_layer_priority(&self, layer_idx: usize) -> Option<f32>;
}

impl AllocationPlanExt for AllocationPlan {
    fn get_layer_priority(&self, layer_idx: usize) -> Option<f32> {
        let prefix = format!("model.layers.{layer_idx}.");
        self.allocations
            .iter()
            .find(|(name, _)| name.starts_with(&prefix))
            .map(|(_, alloc)| alloc.priority)
    }
}

/// Summary of layer distribution across tiers.
#[derive(Debug, Clone)]
pub struct TierSummary {
    /// Layers in VRAM.
    pub vram_layers: usize,
    /// Layers in RAM.
    pub ram_layers: usize,
    /// Layers in NVMe.
    pub nvme_layers: usize,
    /// Layers not loaded.
    pub unloaded_layers: usize,
    /// VRAM usage in bytes.
    pub vram_usage: usize,
    /// RAM usage in bytes.
    pub ram_usage: usize,
    /// NVMe usage in bytes.
    pub nvme_usage: u64,
}

impl std::fmt::Display for TierSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tiers: VRAM={} ({:.1}GB), RAM={} ({:.1}GB), NVMe={} ({:.1}GB), Unloaded={}",
            self.vram_layers,
            self.vram_usage as f64 / (1024.0 * 1024.0 * 1024.0),
            self.ram_layers,
            self.ram_usage as f64 / (1024.0 * 1024.0 * 1024.0),
            self.nvme_layers,
            self.nvme_usage as f64 / (1024.0 * 1024.0 * 1024.0),
            self.unloaded_layers,
        )
    }
}

impl std::fmt::Debug for TieredWeightStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TieredWeightStore")
            .field("num_layers", &self.num_layers)
            .field("strategy", &self.strategy)
            .field("vram", &self.vram)
            .field("ram", &self.ram)
            .field("nvme", &self.nvme)
            .finish()
    }
}
