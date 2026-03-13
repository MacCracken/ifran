pub mod backend;
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
pub mod training;

// Re-export core types at crate root
pub use error::SynapseError;
pub use model::{ModelFormat, ModelId, ModelInfo, QuantLevel};
