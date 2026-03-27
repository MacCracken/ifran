pub mod ab_test;
pub mod backend;
pub mod dataset;

/// Generated protobuf types for the Ifran↔SY bridge protocol.
pub mod bridge {
    tonic::include_proto!("ifran.bridge");
}

/// Generated protobuf types for the core Ifran gRPC service.
pub mod ifran_proto {
    tonic::include_proto!("ifran");
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
pub mod pagination;
pub mod rag;
pub mod registry;
pub mod rlhf;
pub mod tenant;
pub mod training;
pub mod versioning;

// Re-export core types at crate root
pub use error::IfranError;
pub use model::{ModelFormat, ModelId, ModelInfo, QuantLevel};
pub use pagination::PagedResult;
pub use tenant::TenantId;
