use ifran_types::error::Result;
use ifran_types::experiment::ExperimentId;
/// Autonomous experiment commands — run hyperparameter sweeps, view leaderboards.
use std::sync::Arc;
use tokio::sync::Mutex;

/// Run an experiment from a TOML program file.
pub async fn run(program_path: &str) -> Result<()> {
    use ifran_core::config::IfranConfig;
    use ifran_core::experiment::store::ExperimentStore;
    use ifran_train::executor::ExecutorKind;
    use ifran_train::experiment::runner::ExperimentRunner;
    use ifran_train::job::manager::JobManager;
    use ifran_types::experiment::ExperimentProgram;

    let config = IfranConfig::discover();

    // Read and parse TOML program
    let toml_str = std::fs::read_to_string(program_path).map_err(|e| {
        ifran_types::IfranError::ConfigError(format!("Failed to read {program_path}: {e}"))
    })?;
    let program: ExperimentProgram = toml::from_str(&toml_str).map_err(|e| {
        ifran_types::IfranError::ConfigError(format!("Invalid experiment program: {e}"))
    })?;

    eprintln!("Experiment: {}", program.name);
    eprintln!("Model: {}", program.base_model);
    eprintln!("Method: {:?}", program.method);
    eprintln!("Time budget per trial: {}s", program.time_budget_secs);
    eprintln!(
        "Objective: {:?} ({:?})",
        program.objective.metric, program.objective.direction
    );
    eprintln!("Search space: {} parameters", program.search_space.len());

    let executor_kind = match config.training.executor.as_str() {
        "docker" => ExecutorKind::Docker,
        _ => ExecutorKind::Subprocess,
    };

    let job_manager = Arc::new(JobManager::new(
        executor_kind,
        config.training.trainer_image.clone(),
        config.training.max_concurrent_jobs as usize,
    ));

    let store_path = config.storage.database.with_file_name("experiments.db");
    let store = Arc::new(Mutex::new(ExperimentStore::open(&store_path)?));

    let handle = ExperimentRunner::start(job_manager, store.clone(), program).await?;
    eprintln!("Experiment started: {}", handle.experiment_id);
    eprintln!(
        "Use 'ifran experiment status {}' to monitor",
        handle.experiment_id
    );

    // Wait for completion by polling
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        let s = store.lock().await;
        let tenant = ifran_types::TenantId::default_tenant();
        match s.get_experiment(handle.experiment_id, &tenant) {
            Ok((_, _, _, status, _, best_score)) => {
                use ifran_types::experiment::ExperimentStatus;
                match status {
                    ExperimentStatus::Running => {
                        if let Some(score) = best_score {
                            eprint!("\rBest score so far: {score:.4}  ");
                        }
                    }
                    ExperimentStatus::Completed => {
                        eprintln!("\nExperiment completed!");
                        if let Some(score) = best_score {
                            eprintln!("Best score: {score:.4}");
                        }
                        return Ok(());
                    }
                    ExperimentStatus::Stopped => {
                        eprintln!("\nExperiment stopped.");
                        return Ok(());
                    }
                    ExperimentStatus::Failed => {
                        eprintln!("\nExperiment failed.");
                        return Ok(());
                    }
                    _ => {}
                }
            }
            Err(_) => break,
        }
    }

    Ok(())
}

/// List all experiments.
pub async fn list() -> Result<()> {
    use ifran_core::config::IfranConfig;
    use ifran_core::experiment::store::ExperimentStore;

    let config = IfranConfig::discover();
    let store_path = config.storage.database.with_file_name("experiments.db");

    if !store_path.exists() {
        eprintln!("No experiments found.");
        return Ok(());
    }

    let store = ExperimentStore::open(&store_path)?;
    let tenant = ifran_types::TenantId::default_tenant();
    let experiments = store.list_experiments(&tenant)?;

    if experiments.is_empty() {
        eprintln!("No experiments found.");
        return Ok(());
    }

    eprintln!("{:<36}  {:<20}  {:<10}  BEST SCORE", "ID", "NAME", "STATUS");
    eprintln!("{}", "-".repeat(80));
    for (id, name, status, best_score) in experiments {
        let score_str = best_score
            .map(|s| format!("{s:.4}"))
            .unwrap_or_else(|| "—".into());
        eprintln!("{:<36}  {:<20}  {:<10?}  {}", id, name, status, score_str);
    }

    Ok(())
}

/// Show experiment status and current trial info.
pub async fn status(id: Option<&str>) -> Result<()> {
    use ifran_core::config::IfranConfig;
    use ifran_core::experiment::store::ExperimentStore;

    let config = IfranConfig::discover();
    let store_path = config.storage.database.with_file_name("experiments.db");

    if !store_path.exists() {
        eprintln!("No experiments found.");
        return Ok(());
    }

    let store = ExperimentStore::open(&store_path)?;
    let tenant = ifran_types::TenantId::default_tenant();

    if let Some(id_str) = id {
        let experiment_id: ExperimentId = uuid::Uuid::parse_str(id_str)
            .map_err(|e| ifran_types::IfranError::ConfigError(format!("Invalid UUID: {e}")))?;

        let (_, name, program, exp_status, _best_trial, best_score) =
            store.get_experiment(experiment_id, &tenant)?;
        let trials = store.get_trials(experiment_id, &tenant)?;

        eprintln!("Experiment: {name}");
        eprintln!("ID: {experiment_id}");
        eprintln!("Status: {exp_status:?}");
        eprintln!("Model: {}", program.base_model);
        eprintln!(
            "Objective: {:?} ({:?})",
            program.objective.metric, program.objective.direction
        );
        if let Some(score) = best_score {
            eprintln!("Best score: {score:.4}");
        }
        eprintln!("\nTrials ({}):", trials.len());
        eprintln!(
            "  {:<4}  {:<10}  {:<12}  {:<12}  {:<8}  BEST",
            "#", "STATUS", "TRAIN LOSS", "EVAL SCORE", "SECS",
        );
        for t in &trials {
            let loss_str = t
                .train_loss
                .map(|l| format!("{l:.4}"))
                .unwrap_or_else(|| "—".into());
            let score_str = t
                .eval_score
                .map(|s| format!("{s:.4}"))
                .unwrap_or_else(|| "—".into());
            let dur_str = t
                .duration_secs
                .map(|d| format!("{d:.0}"))
                .unwrap_or_else(|| "—".into());
            let best_str = if t.is_best { "*" } else { "" };
            eprintln!(
                "  {:<4}  {:<10?}  {:<12}  {:<12}  {:<8}  {}",
                t.trial_number, t.status, loss_str, score_str, dur_str, best_str
            );
        }
    } else {
        // Show latest experiment
        let experiments = store.list_experiments(&tenant)?;
        if experiments.is_empty() {
            eprintln!("No experiments found.");
        } else {
            let (id, name, exp_status, best_score) = &experiments[0];
            eprintln!("Latest experiment: {name} ({id})");
            eprintln!("Status: {exp_status:?}");
            if let Some(score) = best_score {
                eprintln!("Best score: {score:.4}");
            }
            eprintln!("\nUse 'ifran experiment status {id}' for details.");
        }
    }

    Ok(())
}

/// Show the trial leaderboard for an experiment.
pub async fn leaderboard(id: &str, limit: usize) -> Result<()> {
    use ifran_core::config::IfranConfig;
    use ifran_core::experiment::store::ExperimentStore;

    let config = IfranConfig::discover();
    let store_path = config.storage.database.with_file_name("experiments.db");
    let store = ExperimentStore::open(&store_path)?;

    let experiment_id: ExperimentId = uuid::Uuid::parse_str(id)
        .map_err(|e| ifran_types::IfranError::ConfigError(format!("Invalid UUID: {e}")))?;

    let tenant = ifran_types::TenantId::default_tenant();
    let (_, name, program, _, _, _) = store.get_experiment(experiment_id, &tenant)?;
    let direction = program.objective.direction;
    let trials = store.get_leaderboard(experiment_id, direction, limit, &tenant)?;

    eprintln!(
        "Leaderboard: {name} ({:?} {:?})",
        program.objective.metric, direction
    );
    eprintln!(
        "{:<4}  {:<12}  {:<12}  {:<12}  {:<8}",
        "RANK", "EVAL SCORE", "TRAIN LOSS", "LR", "SECS"
    );
    eprintln!("{}", "-".repeat(56));
    for (rank, t) in trials.iter().enumerate() {
        let score_str = t
            .eval_score
            .map(|s| format!("{s:.4}"))
            .unwrap_or_else(|| "—".into());
        let loss_str = t
            .train_loss
            .map(|l| format!("{l:.4}"))
            .unwrap_or_else(|| "—".into());
        let dur_str = t
            .duration_secs
            .map(|d| format!("{d:.0}"))
            .unwrap_or_else(|| "—".into());
        eprintln!(
            "{:<4}  {:<12}  {:<12}  {:<12.2e}  {:<8}",
            rank + 1,
            score_str,
            loss_str,
            t.hyperparams.learning_rate,
            dur_str
        );
    }

    Ok(())
}

/// Stop a running experiment.
pub async fn stop(id: &str) -> Result<()> {
    use ifran_core::config::IfranConfig;
    use ifran_core::experiment::store::ExperimentStore;
    use ifran_types::experiment::ExperimentStatus;

    let config = IfranConfig::discover();
    let store_path = config.storage.database.with_file_name("experiments.db");
    let store = ExperimentStore::open(&store_path)?;

    let experiment_id: ExperimentId = uuid::Uuid::parse_str(id)
        .map_err(|e| ifran_types::IfranError::ConfigError(format!("Invalid UUID: {e}")))?;

    let tenant = ifran_types::TenantId::default_tenant();
    store.update_experiment_status(experiment_id, ExperimentStatus::Stopped, &tenant)?;
    eprintln!("Experiment {experiment_id} marked as stopped.");
    eprintln!("Note: Running trials will complete before the experiment fully stops.");

    Ok(())
}
