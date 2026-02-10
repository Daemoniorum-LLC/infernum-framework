//! Quick diagnostic for HCT loading - compares direct haagenti vs HctLoader

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use abaddon::hct::HctLoader;
use candle_core::Device;
use haagenti::holotensor::{HoloTensorDecoder, HoloTensorReader, HolographicEncoding};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("Pass HCT file path")?;
    let path = Path::new(&path);

    // === Method 1: Direct haagenti HoloTensorDecoder ===
    println!("=== Method 1: Direct haagenti HoloTensorDecoder ===");
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut holo_reader = HoloTensorReader::new(reader)?;

    let header = holo_reader.header();
    println!("  encoding: {:?}", header.encoding);
    println!("  shape: {:?}", header.shape);
    println!("  total_fragments: {}", header.total_fragments);

    let (header, fragments) = holo_reader.read_all()?;
    println!("  loaded {} fragments", fragments.len());

    // Parse LRDF header from fragment 0
    if header.encoding == HolographicEncoding::LowRankDistributed && !fragments.is_empty() {
        let frag = &fragments[0];
        if frag.data.len() >= 12 {
            let rows = u32::from_le_bytes([frag.data[0], frag.data[1], frag.data[2], frag.data[3]]);
            let cols = u32::from_le_bytes([frag.data[4], frag.data[5], frag.data[6], frag.data[7]]);
            let num_comp =
                u32::from_le_bytes([frag.data[8], frag.data[9], frag.data[10], frag.data[11]]);
            let format = if num_comp == 0xFFFFFFFF { "RAW" } else { "SVD" };
            println!("  LRDF: rows={}, cols={}, format={}", rows, cols, format);
        }
    }

    let mut decoder = HoloTensorDecoder::new(header.clone());
    for frag in fragments {
        decoder.add_fragment(frag)?;
    }
    let haagenti_data = decoder.reconstruct()?;

    let h_min = haagenti_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let h_max = haagenti_data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let h_mean: f32 = haagenti_data.iter().sum::<f32>() / haagenti_data.len() as f32;
    println!(
        "  Result: {} values, min={:.6}, max={:.6}, mean={:.6}",
        haagenti_data.len(),
        h_min,
        h_max,
        h_mean
    );
    println!(
        "  First 5: {:?}",
        &haagenti_data[..5.min(haagenti_data.len())]
    );

    // === Method 2: HctLoader (what inference actually uses) ===
    println!("\n=== Method 2: HctLoader (inference path) ===");
    let loader = HctLoader::from_file(path)?;
    let meta = loader.metadata();
    println!("  is_holographic: {}", meta.is_holographic());
    println!("  dtype: {:?}", meta.dtype);
    println!("  shape: {:?}", meta.shape);

    let tensor = loader.to_tensor(&Device::Cpu, None)?;
    println!("  Tensor shape: {:?}", tensor.dims());
    println!("  Tensor dtype: {:?}", tensor.dtype());

    // Flatten to f32 for comparison
    let tensor_f32 = tensor.to_dtype(candle_core::DType::F32)?;
    let hct_data: Vec<f32> = tensor_f32.flatten_all()?.to_vec1()?;

    let l_min = hct_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let l_max = hct_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let l_mean: f32 = hct_data.iter().sum::<f32>() / hct_data.len() as f32;
    println!(
        "  Result: {} values, min={:.6}, max={:.6}, mean={:.6}",
        hct_data.len(),
        l_min,
        l_max,
        l_mean
    );
    println!("  First 5: {:?}", &hct_data[..5.min(hct_data.len())]);

    // === Comparison ===
    println!("\n=== Comparison ===");
    if haagenti_data.len() != hct_data.len() {
        println!(
            "  ERROR: Length mismatch! haagenti={} vs hctloader={}",
            haagenti_data.len(),
            hct_data.len()
        );
    } else {
        let mut max_diff = 0.0f32;
        let mut total_diff = 0.0f64;
        let mut mismatches = 0usize;

        for (i, (&a, &b)) in haagenti_data.iter().zip(hct_data.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            total_diff += diff as f64;
            if diff > 1e-5 {
                mismatches += 1;
                if mismatches <= 5 {
                    println!(
                        "  Mismatch at [{}]: haagenti={:.8} vs hctloader={:.8} (diff={:.8})",
                        i, a, b, diff
                    );
                }
            }
        }

        let mean_diff = total_diff / haagenti_data.len() as f64;
        println!("  Total elements: {}", haagenti_data.len());
        println!("  Mismatches (>1e-5): {}", mismatches);
        println!("  Max difference: {:.8}", max_diff);
        println!("  Mean difference: {:.8}", mean_diff);

        if mismatches == 0 && max_diff < 1e-5 {
            println!("  ✓ MATCH: Both methods produce identical results");
        } else {
            println!("  ✗ MISMATCH: Methods produce different results!");
        }
    }

    Ok(())
}
