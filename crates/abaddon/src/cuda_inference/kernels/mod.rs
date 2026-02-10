//! CUDA kernel implementations for inference operations.
//!
//! This module contains custom CUDA kernels for:
//! - Fused INT4 dequantization + GEMM
//! - RMSNorm
//! - RoPE (Rotary Position Embeddings)
//! - Flash Attention
//! - Activation functions (SiLU, GELU)
//! - Token sampling (softmax, top-k, top-p)
//! - Embedding lookup (GPU gather)

use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};

pub mod activations;
pub mod attention;
pub mod embedding;
pub mod fused_gemm;
pub mod fused_rmsnorm_proj;
pub mod rmsnorm;
pub mod rope;
pub mod sampling;

pub use activations::{ActivationKernel, ActivationType};
pub use attention::FlashAttentionKernel;
pub use embedding::EmbeddingKernel;
pub use fused_gemm::FusedGemmKernel;
pub use fused_rmsnorm_proj::FusedRMSNormProjKernel;
pub use rmsnorm::RMSNormKernel;
pub use rope::RoPEKernel;
pub use sampling::{RepetitionPenalty, SamplingKernel};

/// Compile CUDA C source to PTX with FP16 support.
///
/// This function sets up the include paths needed for cuda_fp16.h
/// and other standard CUDA headers.
pub(crate) fn compile_cuda_kernel(src: &str) -> Result<Ptx, cudarc::nvrtc::CompileError> {
    let opts = CompileOptions {
        include_paths: vec![
            "/usr/include".to_string(),
            "/usr/local/cuda/include".to_string(),
        ],
        ..Default::default()
    };
    compile_ptx_with_opts(src, opts)
}

/// Compile CUDA C source with tensor core (WMMA) support.
///
/// WMMA requires compute capability sm_70+ and proper CUDA include paths.
/// Returns None if compilation fails (e.g., older GPU or missing headers).
#[allow(dead_code)]
pub(crate) fn compile_cuda_kernel_wmma(src: &str) -> Option<Ptx> {
    // Try to find CUDA toolkit include path
    let cuda_paths = [
        "/usr/local/cuda/include",
        "/usr/local/cuda-12/include",
        "/usr/local/cuda-11/include",
        "/opt/cuda/include",
        "/usr/lib/wsl/lib/../include",
    ];

    let mut include_paths = vec!["/usr/include".to_string()];
    for path in cuda_paths {
        if std::path::Path::new(path).exists() {
            include_paths.push(path.to_string());
            break;
        }
    }

    // Try multiple compute capabilities for WMMA support
    for arch in ["sm_89", "sm_86", "sm_80", "sm_75", "sm_70"] {
        let opts = CompileOptions {
            include_paths: include_paths.clone(),
            arch: Some(arch),
            ..Default::default()
        };

        if let Ok(ptx) = compile_ptx_with_opts(src, opts) {
            tracing::info!("WMMA kernels compiled successfully for {}", arch);
            return Some(ptx);
        }
    }

    tracing::warn!("WMMA kernel compilation failed - tensor cores disabled");
    None
}
