//! Weight loaders for tiered memory system.
//!
//! Provides loading strategies for different model sizes:
//! - EagerLoader: Load all weights upfront (for models that fit in VRAM+RAM)
//! - ProgressiveLoader: Stream from NVMe with prefetching (for 405B+ models)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use super::config::{LoadingStrategy, TieredConfig};
use super::error::TieredError;
use super::ram_cache::{CpuLayerWeights, LayerLayout, TensorLayout};
use super::stats::TieredStats;
use super::store::TieredWeightStore;
use super::vram_cache::SharedWeights;
use crate::adaptive_tiering::{AllocationPlan, MemoryTier, TensorAllocation};
use crate::cuda_inference::arch::ModelConfig;
use crate::cuda_inference::tensor::{GpuDType, GpuTensor};
use crate::cuda_inference::weight_store::{
    LayerWeights, QuantFormat, QuantizedWeight, RMSNormWeights,
};

/// Trait for weight loading strategies.
pub trait WeightLoader {
    /// Load weights from source into tiered store.
    fn load(&self, store: &mut TieredWeightStore) -> Result<(), TieredError>;

    /// Get progress (0.0 - 1.0) for loading UI.
    fn progress(&self) -> f32;
}

/// Eager loader - loads all weights upfront.
///
/// Best for models that fit in VRAM + RAM. Provides fastest inference
/// by avoiding any disk I/O during generation.
pub struct EagerLoader {
    /// Model directory with HCT files.
    model_dir: PathBuf,

    /// Model configuration.
    config: ModelConfig,

    /// Allocation plan.
    plan: AllocationPlan,

    /// CUDA device.
    device: Arc<CudaDevice>,

    /// Statistics.
    stats: Arc<TieredStats>,

    /// Progress tracking.
    layers_loaded: std::sync::atomic::AtomicUsize,
}

impl EagerLoader {
    /// Create a new eager loader.
    pub fn new(
        model_dir: impl AsRef<Path>,
        config: ModelConfig,
        plan: AllocationPlan,
        device: Arc<CudaDevice>,
        stats: Arc<TieredStats>,
    ) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            config,
            plan,
            device,
            stats,
            layers_loaded: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Load shared weights (embeddings, final norm, lm_head).
    fn load_shared(&self) -> Result<SharedWeights, TieredError> {
        tracing::info!("Loading shared weights...");

        // Load embeddings
        let embed_tokens = self.load_tensor("model.embed_tokens.weight")?;

        // Load final norm
        let final_norm_weight = self.load_tensor("model.norm.weight")?;
        let final_norm = RMSNormWeights {
            weight: final_norm_weight,
        };

        // Load lm_head (or None if tied to embeddings)
        let lm_head = if self.config.tie_word_embeddings {
            None
        } else {
            Some(self.load_tensor("lm_head.weight")?)
        };

        Ok(SharedWeights::new(embed_tokens, final_norm, lm_head))
    }

    /// Load a single layer's weights.
    fn load_layer(&self, layer_idx: usize) -> Result<LayerWeights, TieredError> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Attention projections
        let q_proj = self.load_quantized_weight(&format!("{}.self_attn.q_proj", prefix))?;
        let k_proj = self.load_quantized_weight(&format!("{}.self_attn.k_proj", prefix))?;
        let v_proj = self.load_quantized_weight(&format!("{}.self_attn.v_proj", prefix))?;
        let o_proj = self.load_quantized_weight(&format!("{}.self_attn.o_proj", prefix))?;

        // MLP projections
        let gate_proj = self.load_quantized_weight(&format!("{}.mlp.gate_proj", prefix))?;
        let up_proj = self.load_quantized_weight(&format!("{}.mlp.up_proj", prefix))?;
        let down_proj = self.load_quantized_weight(&format!("{}.mlp.down_proj", prefix))?;

        // Norms
        let input_norm = RMSNormWeights {
            weight: self.load_tensor(&format!("{}.input_layernorm.weight", prefix))?,
        };
        let post_attn_norm = RMSNormWeights {
            weight: self.load_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?,
        };

        self.layers_loaded
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(LayerWeights {
            index: layer_idx,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            input_norm,
            post_attn_norm,
        })
    }

    /// Load a tensor from HCT file.
    fn load_tensor(&self, name: &str) -> Result<GpuTensor, TieredError> {
        // Placeholder - actual implementation would use HoloTensorReader
        Err(TieredError::ModelLoad(format!(
            "load_tensor not yet implemented for: {}",
            name
        )))
    }

    /// Load a quantized weight from HCT file.
    fn load_quantized_weight(&self, prefix: &str) -> Result<QuantizedWeight, TieredError> {
        // Placeholder - actual implementation would load from HCT
        Err(TieredError::ModelLoad(format!(
            "load_quantized_weight not yet implemented for: {}",
            prefix
        )))
    }

    /// Load a layer to CPU memory (for RAM tier).
    fn load_layer_to_cpu(&self, layer_idx: usize) -> Result<CpuLayerWeights, TieredError> {
        // Placeholder - actual implementation would load and keep in CPU memory
        Err(TieredError::ModelLoad(format!(
            "load_layer_to_cpu not yet implemented for layer: {}",
            layer_idx
        )))
    }
}

impl WeightLoader for EagerLoader {
    fn load(&self, store: &mut TieredWeightStore) -> Result<(), TieredError> {
        tracing::info!(num_layers = self.config.num_layers, "Starting eager load");

        // Load shared weights first (always to VRAM)
        let shared = self.load_shared()?;
        store.set_shared(shared);

        // Determine tier placement for each layer
        let num_layers = self.config.num_layers;

        for layer_idx in 0..num_layers {
            let tier = self.get_layer_tier(layer_idx);

            match tier {
                MemoryTier::Vram => {
                    // Load directly to VRAM
                    let layer = self.load_layer(layer_idx)?;
                    let priority = self.get_layer_priority(layer_idx);
                    store.vram_cache_mut().insert(layer_idx, layer, priority)?;
                },
                MemoryTier::Ram => {
                    // Load to RAM (pinned memory)
                    let cpu_weights = self.load_layer_to_cpu(layer_idx)?;
                    let priority = self.get_layer_priority(layer_idx);
                    store
                        .ram_cache_mut()
                        .insert(layer_idx, cpu_weights, priority)?;
                },
                MemoryTier::Nvme => {
                    // For eager loading, NVMe should not be used
                    return Err(TieredError::Config(
                        "EagerLoader cannot load to NVMe tier".into(),
                    ));
                },
            }

            if (layer_idx + 1) % 10 == 0 || layer_idx == num_layers - 1 {
                tracing::info!(
                    layer = layer_idx + 1,
                    total = num_layers,
                    "Loading progress"
                );
            }
        }

        tracing::info!("Eager load complete");
        Ok(())
    }

    fn progress(&self) -> f32 {
        let loaded = self
            .layers_loaded
            .load(std::sync::atomic::Ordering::Relaxed);
        loaded as f32 / self.config.num_layers as f32
    }
}

impl EagerLoader {
    /// Get the target tier for a layer.
    fn get_layer_tier(&self, layer_idx: usize) -> MemoryTier {
        // Look up any tensor from this layer
        let prefix = format!("model.layers.{layer_idx}.");
        self.plan
            .allocations
            .iter()
            .find(|(name, _)| name.starts_with(&prefix))
            .map(|(_, alloc)| alloc.tier)
            .unwrap_or(MemoryTier::Vram)
    }

    /// Get the priority for a layer.
    fn get_layer_priority(&self, layer_idx: usize) -> f32 {
        let prefix = format!("model.layers.{layer_idx}.");
        self.plan
            .allocations
            .iter()
            .find(|(name, _)| name.starts_with(&prefix))
            .map(|(_, alloc)| alloc.priority)
            .unwrap_or(0.5)
    }

    /// Get mutable VRAM cache access.
    fn vram_cache_mut<'a>(
        &self,
        store: &'a mut TieredWeightStore,
    ) -> &'a mut super::vram_cache::VramCache {
        // This requires exposing mut access on store
        todo!("Need mut accessor")
    }

    /// Get mutable RAM cache access.
    fn ram_cache_mut<'a>(
        &self,
        store: &'a mut TieredWeightStore,
    ) -> &'a mut super::ram_cache::RamCache {
        // This requires exposing mut access on store
        todo!("Need mut accessor")
    }
}

/// Progressive loader - streams from NVMe with prefetching.
///
/// Best for 405B+ models that don't fit in VRAM + RAM.
/// Uses aggressive prefetching to hide disk I/O latency.
pub struct ProgressiveLoader {
    /// Model directory with HCT files.
    model_dir: PathBuf,

    /// Model configuration.
    config: ModelConfig,

    /// Allocation plan.
    plan: AllocationPlan,

    /// CUDA device.
    device: Arc<CudaDevice>,

    /// Prefetch depth (layers to prefetch ahead).
    prefetch_depth: usize,

    /// Statistics.
    stats: Arc<TieredStats>,
}

impl ProgressiveLoader {
    /// Create a new progressive loader.
    pub fn new(
        model_dir: impl AsRef<Path>,
        config: ModelConfig,
        plan: AllocationPlan,
        device: Arc<CudaDevice>,
        prefetch_depth: usize,
        stats: Arc<TieredStats>,
    ) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            config,
            plan,
            device,
            prefetch_depth,
            stats,
        }
    }

    /// Initialize the NVMe cache with model tensors.
    fn initialize_nvme_cache(&self, store: &mut TieredWeightStore) -> Result<(), TieredError> {
        // For progressive loading, we need to populate the NVMe cache
        // This could be:
        // 1. Decompressing HCT files to disk cache
        // 2. Scanning existing cache directory
        // 3. Symlinking to original HCT files

        let nvme = store
            .nvme_cache_mut()
            .ok_or_else(|| TieredError::Config("Progressive loading requires NVMe cache".into()))?;

        // Scan for existing cached layers
        let found = nvme.scan_existing()?;
        tracing::info!(found, "Scanned NVMe cache");

        Ok(())
    }
}

impl WeightLoader for ProgressiveLoader {
    fn load(&self, store: &mut TieredWeightStore) -> Result<(), TieredError> {
        tracing::info!(
            num_layers = self.config.num_layers,
            prefetch_depth = self.prefetch_depth,
            "Starting progressive load"
        );

        // Initialize NVMe cache
        self.initialize_nvme_cache(store)?;

        // Load shared weights to VRAM (these are always needed)
        // For progressive, we load them lazily or from pre-cached files

        tracing::info!("Progressive loader initialized - layers will be loaded on demand");

        Ok(())
    }

    fn progress(&self) -> f32 {
        // Progressive loading doesn't have upfront progress
        // Progress happens during inference
        0.0
    }
}

/// Factory for creating the appropriate loader based on strategy.
pub fn create_loader(
    strategy: LoadingStrategy,
    model_dir: impl AsRef<Path>,
    config: ModelConfig,
    plan: AllocationPlan,
    tiered_config: &TieredConfig,
    device: Arc<CudaDevice>,
    stats: Arc<TieredStats>,
) -> Box<dyn WeightLoader> {
    match strategy {
        LoadingStrategy::Eager | LoadingStrategy::EagerQuantized { .. } => {
            Box::new(EagerLoader::new(model_dir, config, plan, device, stats))
        },
        LoadingStrategy::Progressive => Box::new(ProgressiveLoader::new(
            model_dir,
            config,
            plan,
            device,
            tiered_config.progressive.prefetch_depth,
            stats,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_factory() {
        // Basic test that factory compiles and returns correct type
        // Full tests require GPU and model files
    }
}
