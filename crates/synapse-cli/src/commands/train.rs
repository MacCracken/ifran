/// Start or manage model training jobs.
use synapse_types::error::Result;

pub async fn execute(base_model: &str, dataset: &str, method: &str) -> Result<()> {
    use synapse_core::config::SynapseConfig;
    use synapse_train::executor::ExecutorKind;
    use synapse_train::job::manager::JobManager;
    use synapse_types::training::*;

    let config = SynapseConfig::discover();

    let executor_kind = match config.training.executor.as_str() {
        "docker" => ExecutorKind::Docker,
        _ => ExecutorKind::Subprocess,
    };

    let manager = JobManager::new(
        executor_kind,
        config.training.trainer_image.clone(),
        config.training.max_concurrent_jobs as usize,
    );

    let training_method = match method.to_lowercase().as_str() {
        "lora" => TrainingMethod::Lora,
        "qlora" => TrainingMethod::Qlora,
        "full" => TrainingMethod::FullFineTune,
        "dpo" => TrainingMethod::Dpo,
        "rlhf" => TrainingMethod::Rlhf,
        "distillation" => TrainingMethod::Distillation,
        _ => {
            eprintln!(
                "Unknown method '{method}'. Options: lora, qlora, full, dpo, rlhf, distillation"
            );
            return Ok(());
        }
    };

    let hyperparams = match training_method {
        TrainingMethod::Lora | TrainingMethod::Qlora => {
            synapse_train::methods::lora::default_hyperparams()
        }
        TrainingMethod::FullFineTune => synapse_train::methods::full::default_hyperparams(),
        TrainingMethod::Dpo => synapse_train::methods::dpo::default_hyperparams(),
        _ => synapse_train::methods::lora::default_hyperparams(),
    };

    let lora_config = match training_method {
        TrainingMethod::Lora | TrainingMethod::Qlora => {
            Some(synapse_train::methods::lora::default_lora_config())
        }
        _ => None,
    };

    let job_config = TrainingJobConfig {
        base_model: base_model.to_string(),
        dataset: DatasetConfig {
            path: dataset.to_string(),
            format: DatasetFormat::Jsonl,
            split: None,
            max_samples: None,
        },
        method: training_method,
        hyperparams,
        output_name: None,
        lora: lora_config,
    };

    let job_id = manager.create_job(job_config).await?;
    eprintln!("Created training job: {job_id}");

    manager.start_job(job_id).await?;
    eprintln!("Training job started. Use 'synapse status' to monitor progress.");

    Ok(())
}
