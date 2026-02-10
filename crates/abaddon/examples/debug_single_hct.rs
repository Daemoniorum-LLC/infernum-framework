//! Debug single HCT file loading.
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use haagenti::holotensor::{HoloTensorDecoder, HoloTensorReader, HOLO_MAGIC};
use std::io::{Read, Seek, SeekFrom};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let default_path = "/tmp/llama1b-lossless/model_layers_0_self_attn_q_proj_weight.hct";
    let path_str = args.get(1).map(|s| s.as_str()).unwrap_or(default_path);
    let path = Path::new(path_str);

    eprintln!("Testing: {}", path.display());

    let mut file = File::open(path)?;

    // Check magic bytes
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    eprintln!("Magic bytes: {:?}", magic);
    eprintln!("Expected HTNS: {:?}", HOLO_MAGIC);

    if magic != HOLO_MAGIC {
        eprintln!("ERROR: Magic mismatch!");
        return Ok(());
    }

    file.seek(SeekFrom::Start(0))?;

    let reader = BufReader::new(file);
    eprintln!("Creating HoloTensorReader...");
    let mut holo_reader = HoloTensorReader::new(reader)?;

    eprintln!("Header created:");
    eprintln!("  encoding: {:?}", holo_reader.header().encoding);
    eprintln!("  compression: {:?}", holo_reader.header().compression);
    eprintln!("  shape: {:?}", holo_reader.header().shape);
    eprintln!("  dtype: {:?}", holo_reader.header().dtype);
    eprintln!(
        "  total_fragments: {}",
        holo_reader.header().total_fragments
    );

    eprintln!("\nCalling read_all()...");
    let (header, fragments) = holo_reader.read_all()?;
    eprintln!("read_all() returned {} fragments", fragments.len());

    if fragments.is_empty() {
        eprintln!("ERROR: No fragments returned!");
        return Ok(());
    }

    // Show first few fragments
    for (i, frag) in fragments.iter().take(3).enumerate() {
        eprintln!(
            "Fragment {}: index={}, flags=0x{:04x}, data_len={}",
            i,
            frag.index,
            frag.flags,
            frag.data.len()
        );
    }

    // Try decoding
    // Note: HoloTensorReader::read_all() now handles decompression automatically
    eprintln!("\nCreating decoder...");
    let mut decoder = HoloTensorDecoder::new(header.clone());

    for frag in fragments {
        decoder.add_fragment(frag)?;
    }

    eprintln!("\nReconstrucing...");
    let f32_data = decoder.reconstruct()?;
    eprintln!("Reconstructed {} f32 values", f32_data.len());

    // Check for zeros
    let non_zero = f32_data.iter().filter(|&&x| x.abs() > 1e-10).count();
    eprintln!("Non-zero values: {} / {}", non_zero, f32_data.len());

    if non_zero > 0 {
        let sum: f32 = f32_data.iter().sum();
        let min = f32_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = f32_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("Sum: {}, Range: [{}, {}]", sum, min, max);
    }

    Ok(())
}
