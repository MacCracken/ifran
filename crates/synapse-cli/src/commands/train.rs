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

/// Execute a distributed training job.
pub async fn execute_distributed(
    base_model: &str,
    dataset: &str,
    method: &str,
    world_size: u32,
    strategy: &str,
) -> Result<()> {
    use synapse_train::distributed::coordinator::DistributedCoordinator;
    use synapse_types::distributed::*;
    use synapse_types::training::*;

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

    let dist_strategy = match strategy {
        "model_parallel" => DistributedStrategy::ModelParallel,
        "pipeline_parallel" => DistributedStrategy::PipelineParallel,
        _ => DistributedStrategy::DataParallel,
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

    let config = DistributedTrainingConfig {
        base_config: TrainingJobConfig {
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
        },
        world_size,
        strategy: dist_strategy,
    };

    let instance_id =
        std::env::var("SYNAPSE_INSTANCE_ID").unwrap_or_else(|_| "local".to_string());

    let coordinator = DistributedCoordinator::new();
    let job_id = coordinator
        .create_job(config, &instance_id)
        .await?;

    eprintln!("Created distributed training job: {job_id}");
    eprintln!("Strategy: {strategy}");
    eprintln!("World size: {world_size}");

    // Auto-assign local worker as rank 0 (coordinator)
    coordinator
        .assign_worker(
            job_id,
            WorkerAssignment {
                rank: 0,
                instance_id: instance_id.clone(),
                endpoint: "local".to_string(),
                device_ids: vec![0],
            },
        )
        .await?;
    eprintln!("Assigned local worker as rank 0 (coordinator)");

    if world_size == 1 {
        coordinator.start_job(job_id).await?;
        eprintln!("Started distributed job with single worker.");
    } else {
        eprintln!(
            "Waiting for {} more workers to be assigned via API before starting.",
            world_size - 1
        );
        eprintln!(
            "Assign workers: POST /training/distributed/jobs/{job_id}/workers"
        );
    }

    Ok(())
}
