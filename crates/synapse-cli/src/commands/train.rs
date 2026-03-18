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

    let training_method = match parse_method(method) {
        Some(m) => m,
        None => {
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
        max_steps: None,
        time_budget_secs: None,
    };

    let job_id = manager
        .create_job(job_config, synapse_types::TenantId::default_tenant())
        .await?;
    eprintln!("Created training job: {job_id}");

    manager
        .start_job(job_id, &synapse_types::TenantId::default_tenant())
        .await?;
    eprintln!("Training job started. Use 'synapse status' to monitor progress.");

    Ok(())
}

/// Parse a training method string into a TrainingMethod enum.
fn parse_method(method: &str) -> Option<synapse_types::training::TrainingMethod> {
    use synapse_types::training::TrainingMethod;
    match method.to_lowercase().as_str() {
        "lora" => Some(TrainingMethod::Lora),
        "qlora" => Some(TrainingMethod::Qlora),
        "full" => Some(TrainingMethod::FullFineTune),
        "dpo" => Some(TrainingMethod::Dpo),
        "rlhf" => Some(TrainingMethod::Rlhf),
        "distillation" => Some(TrainingMethod::Distillation),
        _ => None,
    }
}

/// Parse a distributed strategy string into a DistributedStrategy enum.
fn parse_strategy(strategy: &str) -> synapse_types::distributed::DistributedStrategy {
    use synapse_types::distributed::DistributedStrategy;
    match strategy {
        "model_parallel" => DistributedStrategy::ModelParallel,
        "pipeline_parallel" => DistributedStrategy::PipelineParallel,
        _ => DistributedStrategy::DataParallel,
    }
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

    let training_method = match parse_method(method) {
        Some(m) => m,
        None => {
            eprintln!(
                "Unknown method '{method}'. Options: lora, qlora, full, dpo, rlhf, distillation"
            );
            return Ok(());
        }
    };

    let dist_strategy = parse_strategy(strategy);

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
            max_steps: None,
            time_budget_secs: None,
        },
        world_size,
        strategy: dist_strategy,
        placement_policy: None,
    };

    let instance_id = std::env::var("SYNAPSE_INSTANCE_ID").unwrap_or_else(|_| "local".to_string());

    let coordinator = DistributedCoordinator::new();
    let job_id = coordinator.create_job(config, &instance_id).await?;

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
        eprintln!("Assign workers: POST /training/distributed/jobs/{job_id}/workers");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_types::distributed::DistributedStrategy;
    use synapse_types::training::TrainingMethod;

    #[test]
    fn parse_method_all_variants() {
        assert!(matches!(parse_method("lora"), Some(TrainingMethod::Lora)));
        assert!(matches!(parse_method("qlora"), Some(TrainingMethod::Qlora)));
        assert!(matches!(
            parse_method("full"),
            Some(TrainingMethod::FullFineTune)
        ));
        assert!(matches!(parse_method("dpo"), Some(TrainingMethod::Dpo)));
        assert!(matches!(parse_method("rlhf"), Some(TrainingMethod::Rlhf)));
        assert!(matches!(
            parse_method("distillation"),
            Some(TrainingMethod::Distillation)
        ));
    }

    #[test]
    fn parse_method_case_insensitive() {
        assert!(matches!(parse_method("LORA"), Some(TrainingMethod::Lora)));
        assert!(matches!(parse_method("Qlora"), Some(TrainingMethod::Qlora)));
        assert!(matches!(
            parse_method("Full"),
            Some(TrainingMethod::FullFineTune)
        ));
        assert!(matches!(parse_method("DPO"), Some(TrainingMethod::Dpo)));
    }

    #[test]
    fn parse_method_invalid() {
        assert!(parse_method("unknown").is_none());
        assert!(parse_method("").is_none());
        assert!(parse_method("finetune").is_none());
    }

    #[test]
    fn parse_strategy_all_variants() {
        assert!(matches!(
            parse_strategy("model_parallel"),
            DistributedStrategy::ModelParallel
        ));
        assert!(matches!(
            parse_strategy("pipeline_parallel"),
            DistributedStrategy::PipelineParallel
        ));
        assert!(matches!(
            parse_strategy("data_parallel"),
            DistributedStrategy::DataParallel
        ));
    }

    #[test]
    fn parse_strategy_default_is_data_parallel() {
        assert!(matches!(
            parse_strategy("unknown"),
            DistributedStrategy::DataParallel
        ));
        assert!(matches!(
            parse_strategy(""),
            DistributedStrategy::DataParallel
        ));
    }
}
