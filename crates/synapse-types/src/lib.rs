pub mod ab_test;
pub mod backend;
pub mod dataset;

/// Generated protobuf types for the Synapse↔SY bridge protocol.
pub mod bridge {
    tonic::include_proto!("synapse.bridge");
}

/// Generated protobuf types for the core Synapse gRPC service.
pub mod synapse_proto {
    tonic::include_proto!("synapse");
}
pub mod distributed;
pub mod drift;
pub mod error;
pub mod eval;
pub mod experiment;
pub mod inference;
pub mod lineage;
pub mod marketplace;
pub mod model;
pub mod rag;
pub mod registry;
pub mod rlhf;
pub mod tenant;
pub mod training;
pub mod versioning;

// Re-export core types at crate root
pub use error::SynapseError;
pub use model::{ModelFormat, ModelId, ModelInfo, QuantLevel};
pub use tenant::TenantId;
