//! Build script for infernum-server.
//!
//! Currently empty - gRPC types are defined manually in src/grpc.rs
//! to avoid protoc dependency and provide better type integration
//! with existing REST API types.

fn main() {
    // No build-time proto compilation needed
    // gRPC service is defined manually using tonic's manual API
}
