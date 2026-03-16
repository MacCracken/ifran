pub mod backend;

/// Generated protobuf types for the Synapse↔SY bridge protocol.
pub mod bridge {
    tonic::include_proto!("synapse.bridge");
}
pub mod distributed;
pub mod error;
pub mod eval;
pub mod experiment;
pub mod inference;
pub mod marketplace;
pub mod model;
pub mod rag;
pub mod registry;
pub mod rlhf;
pub mod tenant;
pub mod training;

// Re-export core types at crate root
pub use error::SynapseError;
pub use model::{ModelFormat, ModelId, ModelInfo, QuantLevel};
pub use tenant::TenantId;
