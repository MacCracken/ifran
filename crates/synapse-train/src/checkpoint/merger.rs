//! Checkpoint merger for combining LoRA adapter weights with base model weights.
//!
//! Merging is done by spawning a Python script that uses the PEFT library
//! to merge adapter weights into the base model and export the result.

use synapse_types::SynapseError;
use synapse_types::error::Result;
use tokio::process::Command;
use tracing::info;

/// Merge a LoRA adapter checkpoint into a base model.
///
/// - `base_model`: path or HuggingFace repo ID of the base model
/// - `adapter_path`: path to the LoRA adapter checkpoint directory
/// - `output_path`: where to write the merged model
pub async fn merge_lora(base_model: &str, adapter_path: &str, output_path: &str) -> Result<()> {
    info!(base = %base_model, adapter = %adapter_path, output = %output_path, "Merging LoRA adapter");

    let output = Command::new("python3")
        .args([
            "-c",
            &format!(
                r#"
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, "{adapter_path}")
merged = model.merge_and_unload()
merged.save_pretrained("{output_path}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
tokenizer.save_pretrained("{output_path}")
print("Merge complete")
"#
            ),
        ])
        .output()
        .await
        .map_err(|e| SynapseError::TrainingError(format!("Failed to run merge script: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SynapseError::TrainingError(format!(
            "Merge failed: {stderr}"
        )));
    }

    info!("LoRA merge complete: {output_path}");
    Ok(())
}
