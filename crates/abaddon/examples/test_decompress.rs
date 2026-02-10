use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

fn main() {
    let mut f = File::open("/tmp/smollm2-hq/model_norm_weight.hct").unwrap();
    f.seek(SeekFrom::Start(120)).unwrap();

    let mut compressed = vec![0u8; 1208];
    f.read_exact(&mut compressed).unwrap();

    println!("Read {} compressed bytes", compressed.len());
    println!("First 32 bytes: {:02x?}", &compressed[..32]);

    // Try to decompress with the zstd crate
    match zstd::decode_all(&compressed[..]) {
        Ok(decompressed) => {
            println!("Decompressed to {} bytes", decompressed.len());
            println!(
                "First 16 bytes: {:02x?}",
                &decompressed[..16.min(decompressed.len())]
            );
        },
        Err(e) => {
            println!("Decompression error: {}", e);
        },
    }
}
