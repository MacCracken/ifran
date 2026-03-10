pub mod backend;
pub mod error;
pub mod inference;
pub mod model;
pub mod registry;
pub mod training;

// Re-export core types at crate root
pub use error::SynapseError;
pub use model::{ModelFormat, ModelId, ModelInfo, QuantLevel};
