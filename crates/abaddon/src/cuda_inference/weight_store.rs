//! GPU-resident weight storage for model inference.
//!
//! Loads quantized weights from HCT files directly to GPU memory,
//! organizing them by layer for efficient access during inference.
//!
//! Supports both standard HCT format (LZ4 compressed) and HoloTensor format
//! (holographic encoding with LRDF/SHE/RPH). Format is auto-detected by magic bytes.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use haagenti::holotensor::{HoloTensorDecoder, HoloTensorReader, HOLO_MAGIC};
use haagenti::tensor::HctReader;
use haagenti::Lz4Decompressor;

use crate::gpu_holo::cuda::GpuHoloContext;
use crate::hct::HctLoader;

use super::arch::{ModelArch, ModelConfig, WeightNameMap};
use super::tensor::{GpuDType, GpuTensor};
use super::InferenceError;

/// File format detected from magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileFormat {
    /// Standard HCT format (LZ4 compressed).
    StandardHct,
    /// HoloTensor format (holographic encoding).
    HoloTensor,
}

/// Quantization format for a weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// No quantization (F16).
    F16,
    /// No quantization (BF16).
    BF16,
    /// 8-bit integer quantization.
    Int8,
    /// 4-bit integer quantization (our HCT INT4 format).
    /// Symmetric quantization with per-group scales.
    Int4,
    /// GPTQ 4-bit quantization.
    /// Asymmetric quantization with per-group scales and zeros.
    /// Data packed into int32 (8 INT4 values per int32).
    Gptq,
    /// AWQ (Activation-aware Weight Quantization) 4-bit.
    /// Asymmetric quantization with activation-aware scales.
    /// Data packed into int32 (8 INT4 values per int32).
    Awq,
}

impl QuantFormat {
    /// Get bits per element for this format.
    pub fn bits(&self) -> usize {
        match self {
            QuantFormat::F16 | QuantFormat::BF16 => 16,
            QuantFormat::Int8 => 8,
            QuantFormat::Int4 | QuantFormat::Gptq | QuantFormat::Awq => 4,
        }
    }

    /// Whether this format uses zero points (asymmetric quantization).
    pub fn uses_zero_points(&self) -> bool {
        matches!(self, QuantFormat::Gptq | QuantFormat::Awq)
    }

    /// Whether this format uses g_idx for reordering.
    pub fn uses_g_idx(&self) -> bool {
        matches!(self, QuantFormat::Gptq)
    }
}

/// A quantized weight tensor with metadata.
#[derive(Debug)]
pub struct QuantizedWeight {
    /// Raw quantized data on GPU.
    pub data: GpuTensor,

    /// Quantization format.
    pub format: QuantFormat,

    /// Scale factors for dequantization (GPU-resident).
    pub scales: Option<GpuTensor>,

    /// Zero points for asymmetric quantization (GPU-resident).
    /// Used by GPTQ and AWQ.
    pub zero_points: Option<GpuTensor>,

    /// Group index for GPTQ reordering (GPU-resident).
    /// Maps each row to its quantization group.
    pub g_idx: Option<GpuTensor>,

    /// Original shape [out_features, in_features].
    pub shape: (usize, usize),

    /// Block/group size for per-group quantization.
    /// GPTQ typically uses 128, AWQ uses variable sizes.
    pub block_size: usize,

    /// Whether weight is stored transposed (PyTorch format [out, in] vs standard [in, out]).
    /// When true, GEMM uses A @ B^T instead of A @ B.
    pub transposed: bool,
}

impl QuantizedWeight {
    /// Number of output features.
    pub fn out_features(&self) -> usize {
        self.shape.0
    }

    /// Number of input features.
    pub fn in_features(&self) -> usize {
        self.shape.1
    }

    /// Whether this weight is quantized.
    pub fn is_quantized(&self) -> bool {
        !matches!(self.format, QuantFormat::F16 | QuantFormat::BF16)
    }
}

/// RMSNorm weights.
#[derive(Debug)]
pub struct RMSNormWeights {
    /// Gamma (scale) parameter.
    pub weight: GpuTensor,
}

/// Weights for a single transformer layer.
#[derive(Debug)]
pub struct LayerWeights {
    /// Layer index.
    pub index: usize,

    // === Attention ===
    /// Query projection [hidden_size, hidden_size] or [hidden_size, num_heads * head_dim].
    pub q_proj: QuantizedWeight,

    /// Key projection [hidden_size, num_kv_heads * head_dim].
    pub k_proj: QuantizedWeight,

    /// Value projection [hidden_size, num_kv_heads * head_dim].
    pub v_proj: QuantizedWeight,

    /// Output projection [num_heads * head_dim, hidden_size].
    pub o_proj: QuantizedWeight,

    // === MLP ===
    /// Gate projection (for SwiGLU: gate * up) [hidden_size, intermediate_size].
    pub gate_proj: QuantizedWeight,

    /// Up projection [hidden_size, intermediate_size].
    pub up_proj: QuantizedWeight,

    /// Down projection [intermediate_size, hidden_size].
    pub down_proj: QuantizedWeight,

    // === Norms ===
    /// Input layernorm (before attention).
    pub input_norm: RMSNormWeights,

    /// Post-attention layernorm (before MLP).
    pub post_attn_norm: RMSNormWeights,
}

impl LayerWeights {
    /// Get total size in bytes of all weights in this layer.
    pub fn size_bytes(&self) -> usize {
        self.q_proj.data.size_bytes()
            + self.k_proj.data.size_bytes()
            + self.v_proj.data.size_bytes()
            + self.o_proj.data.size_bytes()
            + self.gate_proj.data.size_bytes()
            + self.up_proj.data.size_bytes()
            + self.down_proj.data.size_bytes()
            + self.input_norm.weight.size_bytes()
            + self.post_attn_norm.weight.size_bytes()
            // Include scales/zero_points if present
            + self.q_proj.scales.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.q_proj.zero_points.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.k_proj.scales.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.k_proj.zero_points.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.v_proj.scales.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.v_proj.zero_points.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.o_proj.scales.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.o_proj.zero_points.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.gate_proj.scales.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.gate_proj.zero_points.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.up_proj.scales.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.up_proj.zero_points.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.down_proj.scales.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + self.down_proj.zero_points.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
    }
}

/// GPU-resident model weights.
#[derive(Debug)]
pub struct WeightStore {
    /// Model configuration.
    pub config: ModelConfig,

    /// CUDA device.
    device: Arc<CudaDevice>,

    /// Token embeddings [vocab_size, hidden_size].
    pub embed_tokens: GpuTensor,

    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,

    /// Final layer norm.
    pub final_norm: RMSNormWeights,

    /// LM head projection [hidden_size, vocab_size].
    /// None if tied to embed_tokens.
    pub lm_head: Option<GpuTensor>,

    /// Total GPU memory used (bytes).
    pub memory_used: usize,
}

impl WeightStore {
    /// Load model weights from HCT directory.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Directory containing .hct files and config.json
    /// * `arch` - Model architecture (or None to auto-detect)
    /// * `device_id` - CUDA device ID
    pub fn load_hct(
        model_dir: impl AsRef<Path>,
        arch: Option<ModelArch>,
        device_id: usize,
    ) -> Result<Self, InferenceError> {
        let model_dir = model_dir.as_ref();

        // Initialize CUDA
        let device =
            CudaDevice::new(device_id).map_err(|e| InferenceError::Device(e.to_string()))?;

        // Detect architecture
        let arch = match arch {
            Some(a) => a,
            None => ModelArch::detect(model_dir)?,
        };

        // Load config
        let config = Self::load_config(model_dir, arch)?;
        let weight_map = arch.weight_map();

        tracing::info!(
            arch = ?arch,
            hidden_size = config.hidden_size,
            num_layers = config.num_layers,
            "Loading model weights"
        );

        // Find all .hct files
        let hct_files = Self::find_hct_files(model_dir)?;

        tracing::info!(num_files = hct_files.len(), "Found HCT weight files");

        // Load all weights into a map
        let raw_weights = Self::load_raw_weights(&hct_files, &device)?;

        // Map HuggingFace names to internal names
        let weights = Self::map_weights(raw_weights, &weight_map);

        // Build weight store structure
        Self::build_weight_store(config, weights, device)
    }

    /// Load config.json if present, otherwise use defaults.
    fn load_config(model_dir: &Path, arch: ModelArch) -> Result<ModelConfig, InferenceError> {
        let config_path = model_dir.join("config.json");

        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| InferenceError::ModelLoad(e.to_string()))?;

            ModelConfig::from_json(&config_str, arch)
        } else {
            // Try to infer from model name
            let dir_name = model_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_lowercase();

            let (hidden_size, num_layers) = if dir_name.contains("7b") {
                (4096, 32)
            } else if dir_name.contains("13b") {
                (5120, 40)
            } else if dir_name.contains("70b") {
                (8192, 80)
            } else if dir_name.contains("1b") || dir_name.contains("1.5b") {
                (2048, 24)
            } else if dir_name.contains("3b") {
                (3200, 26)
            } else {
                (4096, 32) // Default to 7B-ish
            };

            Ok(ModelConfig::default_for_arch(arch, hidden_size, num_layers))
        }
    }

    /// Find all .hct files in directory.
    fn find_hct_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, InferenceError> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| InferenceError::ModelLoad(format!("Cannot read directory: {}", e)))?;

        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "hct"))
            .map(|e| e.path())
            .collect();

        files.sort();
        Ok(files)
    }

    /// Detect file format by reading magic bytes.
    fn detect_format(path: &Path) -> Result<FileFormat, InferenceError> {
        let mut file = File::open(path).map_err(|e| {
            InferenceError::ModelLoad(format!("Cannot open {}: {}", path.display(), e))
        })?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| InferenceError::ModelLoad(format!("Cannot read magic: {}", e)))?;

        if magic == HOLO_MAGIC {
            Ok(FileFormat::HoloTensor)
        } else {
            Ok(FileFormat::StandardHct)
        }
    }

    /// Load raw weights from HCT files (auto-detects format).
    fn load_raw_weights(
        hct_files: &[std::path::PathBuf],
        device: &Arc<CudaDevice>,
    ) -> Result<HashMap<String, LoadedWeight>, InferenceError> {
        let mut weights = HashMap::new();
        let mut total_bytes = 0usize;

        // Detect format from first file
        let format = if let Some(first) = hct_files.first() {
            Self::detect_format(first)?
        } else {
            return Err(InferenceError::ModelLoad("No HCT files found".to_string()));
        };

        match format {
            FileFormat::HoloTensor => {
                tracing::info!("Detected HoloTensor format, using GPU holographic reconstruction");
                Self::load_holotensor_weights(hct_files, device, &mut weights, &mut total_bytes)?;
            },
            FileFormat::StandardHct => {
                tracing::info!("Detected standard HCT format, using LZ4 decompression");
                Self::load_standard_hct_weights(hct_files, device, &mut weights, &mut total_bytes)?;
            },
        }

        tracing::info!(
            total_mb = total_bytes as f64 / 1024.0 / 1024.0,
            num_weights = weights.len(),
            "Loaded raw weights to GPU"
        );

        Ok(weights)
    }

    /// Load weights from standard HCT files (LZ4 compressed).
    fn load_standard_hct_weights(
        hct_files: &[std::path::PathBuf],
        device: &Arc<CudaDevice>,
        weights: &mut HashMap<String, LoadedWeight>,
        total_bytes: &mut usize,
    ) -> Result<(), InferenceError> {
        let decompressor = Lz4Decompressor::new();

        for path in hct_files {
            let loader = HctLoader::from_file(path).map_err(|e| {
                InferenceError::ModelLoad(format!("Failed to load {}: {}", path.display(), e))
            })?;

            // Convert underscore-separated filename to dot-separated HuggingFace name
            let name = filename_to_hf_name(&loader.metadata().name);
            let shape: Vec<usize> = loader
                .metadata()
                .shape
                .iter()
                .map(|&d| d as usize)
                .collect();
            let dtype = loader.metadata().dtype;

            // Open file and create reader
            let file = File::open(path).map_err(|e| InferenceError::ModelLoad(e.to_string()))?;
            let reader = BufReader::new(file);
            let mut hct_reader =
                HctReader::new(reader).map_err(|e| InferenceError::ModelLoad(e.to_string()))?;

            // Decompress
            let data = hct_reader
                .decompress_all(&decompressor)
                .map_err(|e| InferenceError::ModelLoad(format!("Decompress failed: {}", e)))?;

            // Determine format and upload to GPU
            let (gpu_data, format, scales, zero_points) =
                Self::process_weight_data(&data, &shape, dtype, device)?;

            *total_bytes += gpu_data.size_bytes();
            if let Some(ref s) = scales {
                *total_bytes += s.size_bytes();
            }
            if let Some(ref z) = zero_points {
                *total_bytes += z.size_bytes();
            }

            weights.insert(
                name,
                LoadedWeight {
                    data: gpu_data,
                    format,
                    scales,
                    zero_points,
                    shape,
                },
            );
        }

        Ok(())
    }

    /// Load weights from HoloTensor files using GPU holographic reconstruction.
    fn load_holotensor_weights(
        hct_files: &[std::path::PathBuf],
        device: &Arc<CudaDevice>,
        weights: &mut HashMap<String, LoadedWeight>,
        total_bytes: &mut usize,
    ) -> Result<(), InferenceError> {
        // Initialize GPU holographic context
        // TODO: Get device_id from CudaDevice - for now assume device 0
        let mut gpu_ctx = GpuHoloContext::with_device(Arc::clone(device), 0);

        // Load all reconstruction kernels (LRDF, RPH, Spectral, fused)
        gpu_ctx
            .load_all_kernels()
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to load GPU kernels: {}", e)))?;

        // Also load fused kernel for F32->F16 conversion
        gpu_ctx.load_fused_kernel().map_err(|e| {
            InferenceError::ModelLoad(format!("Failed to load fused kernel: {}", e))
        })?;

        tracing::info!("GPU holographic kernels loaded successfully");

        for path in hct_files {
            // Get weight name from filename
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(filename_to_hf_name)
                .ok_or_else(|| InferenceError::ModelLoad("Invalid filename".to_string()))?;

            // Open file and create HoloTensor reader
            let file = File::open(path).map_err(|e| {
                InferenceError::ModelLoad(format!("Cannot open {}: {}", path.display(), e))
            })?;
            let reader = BufReader::new(file);
            let mut holo_reader = HoloTensorReader::new(reader).map_err(|e| {
                InferenceError::ModelLoad(format!(
                    "Failed to parse HoloTensor {}: {}",
                    path.display(),
                    e
                ))
            })?;

            // Read header and all fragments
            let (header, fragments) = holo_reader.read_all().map_err(|e| {
                InferenceError::ModelLoad(format!(
                    "Failed to read fragments from {}: {}",
                    path.display(),
                    e
                ))
            })?;

            // Extract shape from header
            let shape: Vec<usize> = header.shape.iter().map(|&d| d as usize).collect();

            // For 1D tensors (biases, norms), use CPU reconstruction + GPU upload
            // GPU reconstruction requires 2D tensors
            let gpu_tensor = if shape.len() == 1 {
                // CPU reconstruction for 1D tensors
                let mut decoder = haagenti::holotensor::HoloTensorDecoder::new(header.clone());
                for fragment in &fragments {
                    decoder.add_fragment(fragment.clone()).map_err(|e| {
                        InferenceError::ModelLoad(format!("Failed to add fragment: {}", e))
                    })?;
                }
                let cpu_data = decoder.reconstruct().map_err(|e| {
                    InferenceError::ModelLoad(format!(
                        "CPU reconstruction failed for {}: {}",
                        path.display(),
                        e
                    ))
                })?;

                // Convert f32 to f16 and upload
                let f16_data: Vec<half::f16> =
                    cpu_data.iter().map(|&f| half::f16::from_f32(f)).collect();

                // Reinterpret as bytes and upload
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(f16_data.as_ptr() as *const u8, f16_data.len() * 2)
                };

                let gpu_data: CudaSlice<u8> = device.htod_sync_copy(bytes).map_err(|e| {
                    InferenceError::Memory(format!("Failed to upload 1D tensor: {}", e))
                })?;

                GpuTensor::from_cuda_slice(
                    gpu_data,
                    shape.clone(),
                    GpuDType::F16,
                    Arc::clone(device),
                )?
            } else {
                // GPU reconstruction for 2D+ tensors
                let gpu_data_f32: CudaSlice<f32> =
                    gpu_ctx.reconstruct(&header, &fragments).map_err(|e| {
                        InferenceError::ModelLoad(format!(
                            "GPU reconstruction failed for {}: {}",
                            path.display(),
                            e
                        ))
                    })?;

                // Convert f32 to f16 for inference (better memory efficiency)
                let gpu_data_f16 = gpu_ctx.convert_f32_to_f16(&gpu_data_f32).map_err(|e| {
                    InferenceError::ModelLoad(format!("F32->F16 conversion failed: {}", e))
                })?;

                GpuTensor::from_cuda_slice_f16(gpu_data_f16, shape.clone(), Arc::clone(device))?
            };

            *total_bytes += gpu_tensor.size_bytes();

            weights.insert(
                name,
                LoadedWeight {
                    data: gpu_tensor,
                    format: QuantFormat::F16,
                    scales: None,
                    zero_points: None,
                    shape,
                },
            );
        }

        Ok(())
    }

    /// Process weight data based on dtype and upload to GPU.
    fn process_weight_data(
        data: &[u8],
        shape: &[usize],
        dtype: haagenti::tensor::DType,
        device: &Arc<CudaDevice>,
    ) -> Result<(GpuTensor, QuantFormat, Option<GpuTensor>, Option<GpuTensor>), InferenceError>
    {
        use haagenti::tensor::DType;

        match dtype {
            DType::F16 => {
                // Direct F16, upload as-is
                let gpu_data = Self::upload_to_gpu(data, shape.to_vec(), GpuDType::F16, device)?;
                Ok((gpu_data, QuantFormat::F16, None, None))
            },
            DType::BF16 => {
                // Direct BF16
                let gpu_data = Self::upload_to_gpu(data, shape.to_vec(), GpuDType::BF16, device)?;
                Ok((gpu_data, QuantFormat::BF16, None, None))
            },
            DType::I4 => {
                // INT4 with scales and zero points
                // Format: [scales (f32)] [zero_points (i8)] [packed_int4_data]
                let num_elements: usize = shape.iter().product();
                let block_size = 128; // Our HCT INT4 block size
                let num_blocks = (num_elements + block_size - 1) / block_size;

                let scales_size = num_blocks * 4; // f32
                let zp_size = num_blocks; // i8
                let packed_size = (num_elements + 1) / 2; // 2 values per byte

                if data.len() < scales_size + zp_size + packed_size {
                    return Err(InferenceError::ModelLoad(format!(
                        "INT4 data too small: {} bytes for {} elements",
                        data.len(),
                        num_elements
                    )));
                }

                // Extract scales (convert f32 to f16 for GPU)
                let scales_f16: Vec<u8> = data[..scales_size]
                    .chunks_exact(4)
                    .flat_map(|c| {
                        let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                        half::f16::from_f32(f).to_le_bytes()
                    })
                    .collect();

                let scales_tensor =
                    Self::upload_to_gpu(&scales_f16, vec![num_blocks], GpuDType::F16, device)?;

                // Extract zero points
                let zp_data = &data[scales_size..scales_size + zp_size];
                let zp_tensor =
                    Self::upload_to_gpu(zp_data, vec![num_blocks], GpuDType::I8, device)?;

                // Extract packed INT4 data
                let packed_data = &data[scales_size + zp_size..];
                let packed_tensor =
                    Self::upload_to_gpu(packed_data, shape.to_vec(), GpuDType::I4, device)?;

                Ok((
                    packed_tensor,
                    QuantFormat::Int4,
                    Some(scales_tensor),
                    Some(zp_tensor),
                ))
            },
            DType::I8 => {
                // INT8 quantization
                let gpu_data = Self::upload_to_gpu(data, shape.to_vec(), GpuDType::I8, device)?;
                Ok((gpu_data, QuantFormat::Int8, None, None))
            },
            other => Err(InferenceError::ModelLoad(format!(
                "Unsupported dtype: {:?}",
                other
            ))),
        }
    }

    /// Upload data to GPU.
    fn upload_to_gpu(
        data: &[u8],
        shape: Vec<usize>,
        dtype: GpuDType,
        device: &Arc<CudaDevice>,
    ) -> Result<GpuTensor, InferenceError> {
        let gpu_data: CudaSlice<u8> = device
            .htod_sync_copy(data)
            .map_err(|e| InferenceError::Memory(e.to_string()))?;

        GpuTensor::from_cuda_slice(gpu_data, shape, dtype, Arc::clone(device))
    }

    /// Map HuggingFace weight names to internal names.
    fn map_weights(
        raw_weights: HashMap<String, LoadedWeight>,
        weight_map: &WeightNameMap,
    ) -> HashMap<String, LoadedWeight> {
        let mut mapped = HashMap::new();

        for (hf_name, weight) in raw_weights {
            if let Some(internal_name) = weight_map.map_name(&hf_name) {
                mapped.insert(internal_name, weight);
            } else {
                // Keep original name if no mapping
                tracing::debug!(name = %hf_name, "No mapping for weight, using original name");
                mapped.insert(hf_name, weight);
            }
        }

        mapped
    }

    /// Build the structured weight store from loaded weights.
    fn build_weight_store(
        config: ModelConfig,
        mut weights: HashMap<String, LoadedWeight>,
        device: Arc<CudaDevice>,
    ) -> Result<Self, InferenceError> {
        // Extract embeddings
        let embed_tokens = weights
            .remove("embed_tokens")
            .ok_or_else(|| InferenceError::ModelLoad("Missing embed_tokens".to_string()))?
            .data;

        // Extract final norm
        let final_norm = RMSNormWeights {
            weight: weights
                .remove("final_norm")
                .ok_or_else(|| InferenceError::ModelLoad("Missing final_norm".to_string()))?
                .data,
        };

        // Extract LM head (optional if tied)
        let lm_head = weights.remove("lm_head").map(|w| w.data);

        // Build layers
        let mut layers = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            let layer = Self::build_layer(&mut weights, i, &config)?;
            layers.push(layer);
        }

        // Calculate total memory
        let memory_used = embed_tokens.size_bytes()
            + final_norm.weight.size_bytes()
            + lm_head.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
            + layers.iter().map(|l| Self::layer_memory(l)).sum::<usize>();

        tracing::info!(
            memory_mb = memory_used as f64 / 1024.0 / 1024.0,
            num_layers = layers.len(),
            "Built weight store"
        );

        Ok(Self {
            config,
            device,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            memory_used,
        })
    }

    /// Build a single layer's weights.
    fn build_layer(
        weights: &mut HashMap<String, LoadedWeight>,
        index: usize,
        _config: &ModelConfig,
    ) -> Result<LayerWeights, InferenceError> {
        let mut get_weight = |name: &str| -> Result<LoadedWeight, InferenceError> {
            let key = format!("layers.{}.{}", index, name);
            weights
                .remove(&key)
                .ok_or_else(|| InferenceError::ModelLoad(format!("Missing weight: {}", key)))
        };

        let to_quantized = |w: LoadedWeight| -> QuantizedWeight {
            let shape = if w.shape.len() >= 2 {
                (w.shape[0], w.shape[1])
            } else {
                (w.shape[0], 1)
            };

            QuantizedWeight {
                data: w.data,
                format: w.format,
                scales: w.scales,
                zero_points: w.zero_points,
                g_idx: None, // GPTQ group index, not used for HCT
                shape,
                block_size: 128,   // Our HCT default
                transposed: false, // HCT loading transposes weights to [in, out] format
            }
        };

        let to_norm = |w: LoadedWeight| -> RMSNormWeights { RMSNormWeights { weight: w.data } };

        Ok(LayerWeights {
            index,
            q_proj: to_quantized(get_weight("q_proj")?),
            k_proj: to_quantized(get_weight("k_proj")?),
            v_proj: to_quantized(get_weight("v_proj")?),
            o_proj: to_quantized(get_weight("o_proj")?),
            gate_proj: to_quantized(get_weight("gate_proj")?),
            up_proj: to_quantized(get_weight("up_proj")?),
            down_proj: to_quantized(get_weight("down_proj")?),
            input_norm: to_norm(get_weight("input_norm")?),
            post_attn_norm: to_norm(get_weight("post_attn_norm")?),
        })
    }

    /// Calculate memory used by a layer.
    fn layer_memory(layer: &LayerWeights) -> usize {
        let qw_size = |qw: &QuantizedWeight| {
            qw.data.size_bytes()
                + qw.scales.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
                + qw.zero_points.as_ref().map(|t| t.size_bytes()).unwrap_or(0)
        };

        qw_size(&layer.q_proj)
            + qw_size(&layer.k_proj)
            + qw_size(&layer.v_proj)
            + qw_size(&layer.o_proj)
            + qw_size(&layer.gate_proj)
            + qw_size(&layer.up_proj)
            + qw_size(&layer.down_proj)
            + layer.input_norm.weight.size_bytes()
            + layer.post_attn_norm.weight.size_bytes()
    }

    /// Get CUDA device reference.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

/// Intermediate structure for loaded weights.
struct LoadedWeight {
    data: GpuTensor,
    format: QuantFormat,
    scales: Option<GpuTensor>,
    zero_points: Option<GpuTensor>,
    shape: Vec<usize>,
}

/// Convert underscore-separated HCT filename to dot-separated HuggingFace name.
///
/// Example: `model_layers_0_mlp_down_proj_weight` → `model.layers.0.mlp.down_proj.weight`
fn filename_to_hf_name(filename: &str) -> String {
    // Known component names that contain underscores (should NOT be split)
    const UNDERSCORE_COMPONENTS: &[&str] = &[
        "embed_tokens",
        "input_layernorm",
        "post_attention_layernorm",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "self_attn",
        "lm_head",
    ];

    let mut result = filename.to_string();

    // First, protect known underscore components by temporarily replacing them
    // Use markers that don't contain underscores so they survive the underscore->dot conversion
    let mut protected = Vec::new();
    for (i, comp) in UNDERSCORE_COMPONENTS.iter().enumerate() {
        let placeholder = format!("XPLACEHOLDER{}X", i);
        if result.contains(comp) {
            result = result.replace(comp, &placeholder);
            protected.push((placeholder, comp.to_string()));
        }
    }

    // Convert remaining underscores to dots
    result = result.replace('_', ".");

    // Restore protected components
    for (placeholder, original) in protected {
        result = result.replace(&placeholder, &original);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filename_to_hf_name() {
        assert_eq!(
            filename_to_hf_name("model_embed_tokens_weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            filename_to_hf_name("model_layers_0_self_attn_q_proj_weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            filename_to_hf_name("model_layers_29_mlp_down_proj_weight"),
            "model.layers.29.mlp.down_proj.weight"
        );
        assert_eq!(
            filename_to_hf_name("model_layers_0_input_layernorm_weight"),
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            filename_to_hf_name("model_norm_weight"),
            "model.norm.weight"
        );
    }
}
