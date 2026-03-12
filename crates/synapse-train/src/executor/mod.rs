pub mod docker;
pub mod native;
pub mod subprocess;

use async_trait::async_trait;
use synapse_types::error::Result;
use synapse_types::training::{TrainingJobConfig, TrainingJobId, TrainingMethod};

/// Which executor to use for training.
#[derive(Debug, Clone, Copy)]
pub enum ExecutorKind {
    Docker,
    Subprocess,
}

/// Map a training method to its Python training script.
pub fn script_for_method(method: TrainingMethod) -> &'static str {
    match method {
        TrainingMethod::Lora | TrainingMethod::Qlora => "scripts/train_sft.py",
        TrainingMethod::FullFineTune => "scripts/train_full.py",
        TrainingMethod::Dpo => "scripts/train_dpo.py",
        TrainingMethod::Rlhf => "scripts/train_rlhf.py",
        TrainingMethod::Distillation => "scripts/train_distill.py",
    }
}

/// Trait for training executors that launch and manage training workloads.
#[async_trait]
pub trait TrainingExecutor: Send + Sync {
    /// Run a training job to completion.
    async fn run(&self, config: &TrainingJobConfig, job_id: TrainingJobId) -> Result<()>;

    /// Cancel a running job.
    async fn cancel(&self, job_id: TrainingJobId) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_for_lora() {
        assert_eq!(script_for_method(TrainingMethod::Lora), "scripts/train_sft.py");
    }

    #[test]
    fn script_for_qlora() {
        assert_eq!(script_for_method(TrainingMethod::Qlora), "scripts/train_sft.py");
    }

    #[test]
    fn script_for_full() {
        assert_eq!(
            script_for_method(TrainingMethod::FullFineTune),
            "scripts/train_full.py"
        );
    }

    #[test]
    fn script_for_dpo() {
        assert_eq!(script_for_method(TrainingMethod::Dpo), "scripts/train_dpo.py");
    }

    #[test]
    fn script_for_rlhf() {
        assert_eq!(script_for_method(TrainingMethod::Rlhf), "scripts/train_rlhf.py");
    }

    #[test]
    fn script_for_distillation() {
        assert_eq!(
            script_for_method(TrainingMethod::Distillation),
            "scripts/train_distill.py"
        );
    }
}
