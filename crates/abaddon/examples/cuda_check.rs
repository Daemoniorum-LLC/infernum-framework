//! Quick CUDA availability check.

fn main() {
    println!("Checking CUDA availability...");

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaDevice;

        println!("cudarc feature enabled, attempting device creation...");

        match CudaDevice::new(0) {
            Ok(device) => {
                println!("SUCCESS! CUDA device created.");
                // Try to get device properties
                println!(
                    "Device pointer: {:?}",
                    std::sync::Arc::strong_count(&device)
                );
            },
            Err(e) => {
                println!("FAILED to create CUDA device:");
                println!("  Error: {:?}", e);
            },
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled!");
    }
}
