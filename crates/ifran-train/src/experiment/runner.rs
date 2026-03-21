//! Autonomous experiment runner — the core loop that ties training, eval, and
//! comparison together (inspired by Karpathy's autoresearch).

use std::sync::Arc;

use chrono::Utc;
use ifran_core::experiment::store::ExperimentStore;
use ifran_types::TenantId;
use ifran_types::error::Result;
use ifran_types::experiment::{
    ExperimentId, ExperimentProgram, ExperimentStatus, TrialResult, TrialStatus,
};
use ifran_types::training::{DatasetConfig, DatasetFormat, TrainingJobConfig, TrainingStatus};
use tokio::sync::{Mutex, watch};
use tracing::{info, warn};
use uuid::Uuid;

use crate::experiment::search::SearchSpace;
use crate::job::manager::JobManager;

/// Runs an experiment: generates trials, trains each one, evaluates, compares.
pub struct ExperimentRunner {
    job_manager: Arc<JobManager>,
    store: Arc<Mutex<ExperimentStore>>,
    stop_rx: watch::Receiver<bool>,
}

/// Handle returned to the caller for controlling a running experiment.
pub struct ExperimentHandle {
    pub experiment_id: ExperimentId,
    stop_tx: watch::Sender<bool>,
}

impl ExperimentHandle {
    /// Signal the experiment to stop after the current trial.
    pub fn stop(&self) {
        let _ = self.stop_tx.send(true);
    }
}

impl ExperimentRunner {
    pub fn new(
        job_manager: Arc<JobManager>,
        store: Arc<Mutex<ExperimentStore>>,
    ) -> (Self, watch::Sender<bool>) {
        let (stop_tx, stop_rx) = watch::channel(false);
        (
            Self {
                job_manager,
                store,
                stop_rx,
            },
            stop_tx,
        )
    }

    /// Start an experiment: register it in the store, then run the core loop.
    /// Returns the experiment ID and a handle for stopping.
    pub async fn start(
        job_manager: Arc<JobManager>,
        store: Arc<Mutex<ExperimentStore>>,
        program: ExperimentProgram,
    ) -> Result<ExperimentHandle> {
        let experiment_id = Uuid::new_v4();

        // Register experiment
        {
            let s = store.lock().await;
            let tenant = TenantId::default_tenant();
            s.insert_experiment(experiment_id, &program.name, &program, &tenant)?;
        }

        let (runner, stop_tx) = Self::new(job_manager, store.clone());
        let handle = ExperimentHandle {
            experiment_id,
            stop_tx,
        };

        // Spawn the loop in background
        let prog = program.clone();
        let store_for_error = store;
        tokio::spawn(async move {
            if let Err(e) = runner.run_loop(experiment_id, &prog).await {
                warn!(experiment_id = %experiment_id, error = %e, "Experiment failed");
                let s = store_for_error.lock().await;
                let tenant = TenantId::default_tenant();
                let _ =
                    s.update_experiment_status(experiment_id, ExperimentStatus::Failed, &tenant);
            }
        });

        Ok(handle)
    }

    async fn run_loop(
        mut self,
        experiment_id: ExperimentId,
        program: &ExperimentProgram,
    ) -> Result<()> {
        let tenant = TenantId::default_tenant();
        info!(experiment_id = %experiment_id, name = %program.name, "Starting experiment");

        // Generate trial hyperparams
        let search_space = SearchSpace::new(
            program.base_hyperparams.clone(),
            program.search_space.clone(),
            program.search.clone(),
        );
        let mut trial_hps = search_space.generate_trials();

        // Cap at max_trials
        if let Some(max) = program.max_trials {
            trial_hps.truncate(max as usize);
        }

        let total_trials = trial_hps.len();
        info!(experiment_id = %experiment_id, total_trials, "Generated trial configurations");

        let mut best_score: Option<f64> = None;
        let mut best_trial_id: Option<Uuid> = None;

        let dataset_format = match program.dataset_format.as_str() {
            "jsonl" => DatasetFormat::Jsonl,
            "csv" => DatasetFormat::Csv,
            "parquet" => DatasetFormat::Parquet,
            "huggingface" => DatasetFormat::HuggingFace,
            _ => DatasetFormat::Jsonl,
        };

        for (i, hp) in trial_hps.into_iter().enumerate() {
            // Check stop signal
            if *self.stop_rx.borrow() {
                info!(experiment_id = %experiment_id, "Stop signal received");
                let s = self.store.lock().await;
                s.update_experiment_status(experiment_id, ExperimentStatus::Stopped, &tenant)?;
                return Ok(());
            }

            let trial_number = (i + 1) as u32;
            let trial_id = Uuid::new_v4();
            let output_name = format!("{}-trial-{trial_number}", program.name);

            info!(
                experiment_id = %experiment_id,
                trial = trial_number,
                total = total_trials,
                lr = hp.learning_rate,
                "Starting trial"
            );

            // Record trial start
            let mut trial = TrialResult {
                trial_id,
                experiment_id,
                trial_number,
                hyperparams: hp.clone(),
                train_loss: None,
                eval_score: None,
                status: TrialStatus::Training,
                duration_secs: None,
                started_at: Some(Utc::now()),
                completed_at: None,
                checkpoint_path: None,
                is_best: false,
            };
            {
                let s = self.store.lock().await;
                s.insert_trial(&trial)?;
            }

            let start_time = std::time::Instant::now();

            // Build training config with time budget
            let job_config = TrainingJobConfig {
                base_model: program.base_model.clone(),
                dataset: DatasetConfig {
                    path: program.dataset_path.clone(),
                    format: dataset_format,
                    split: None,
                    max_samples: program.eval_sample_limit,
                },
                method: program.method,
                hyperparams: hp,
                output_name: Some(output_name.clone()),
                lora: None,
                max_steps: None,
                time_budget_secs: Some(program.time_budget_secs),
            };

            // Submit and run training
            let train_result = self.run_training_trial(&job_config).await;

            let duration = start_time.elapsed().as_secs_f64();
            trial.duration_secs = Some(duration);

            match train_result {
                Ok(final_loss) => {
                    trial.train_loss = final_loss;
                    trial.checkpoint_path = Some(format!("/workspace/checkpoints/{output_name}"));

                    // Use training loss as eval score for now
                    // (real eval would run perplexity/benchmark on the checkpoint)
                    let score = final_loss.unwrap_or(f64::MAX);
                    trial.eval_score = Some(score);
                    trial.status = TrialStatus::Completed;
                    trial.completed_at = Some(Utc::now());

                    // Compare with best
                    let is_new_best = match best_score {
                        Some(prev) => program.objective.direction.is_better(score, prev),
                        None => true,
                    };

                    if is_new_best {
                        best_score = Some(score);
                        best_trial_id = Some(trial_id);
                        trial.is_best = true;

                        info!(
                            experiment_id = %experiment_id,
                            trial = trial_number,
                            score,
                            "New best trial"
                        );

                        let s = self.store.lock().await;
                        s.update_best_trial(experiment_id, trial_id, score, &tenant)?;
                    }
                }
                Err(e) => {
                    warn!(
                        experiment_id = %experiment_id,
                        trial = trial_number,
                        error = %e,
                        "Trial failed"
                    );
                    trial.status = TrialStatus::Failed;
                    trial.completed_at = Some(Utc::now());
                }
            }

            // Update trial in store
            {
                let s = self.store.lock().await;
                s.update_trial(&trial)?;
            }
        }

        // Mark experiment completed
        {
            let s = self.store.lock().await;
            s.update_experiment_status(experiment_id, ExperimentStatus::Completed, &tenant)?;
        }

        if let (Some(id), Some(score)) = (best_trial_id, best_score) {
            info!(
                experiment_id = %experiment_id,
                best_trial = %id,
                best_score = score,
                "Experiment completed"
            );
        } else {
            info!(experiment_id = %experiment_id, "Experiment completed (no successful trials)");
        }

        Ok(())
    }

    /// Submit a training job, wait for it to finish, return final loss.
    async fn run_training_trial(&mut self, config: &TrainingJobConfig) -> Result<Option<f64>> {
        // Experiment runner operates at system level
        let tenant = ifran_types::TenantId::default_tenant();
        let job_id = self
            .job_manager
            .create_job(config.clone(), tenant.clone())
            .await?;
        self.job_manager.start_job(job_id, &tenant).await?;

        // Poll every 5 seconds until terminal, with wall-clock timeout
        let wall_timeout = config
            .time_budget_secs
            .map(|b| std::time::Duration::from_secs(b + 60))
            .unwrap_or(std::time::Duration::from_secs(3600));
        let deadline = std::time::Instant::now() + wall_timeout;

        loop {
            // Check stop signal
            if *self.stop_rx.borrow() {
                let _ = self.job_manager.cancel_job(job_id, &tenant).await;
                return Ok(None);
            }

            tokio::time::sleep(std::time::Duration::from_secs(5)).await;

            let job = self.job_manager.get_job(job_id, &tenant).await?;
            if job.is_terminal() {
                return match job.status {
                    TrainingStatus::Completed => Ok(job.current_loss),
                    TrainingStatus::Cancelled => Ok(job.current_loss),
                    _ => Err(ifran_types::IfranError::TrainingError(
                        job.error.unwrap_or_else(|| "Unknown training error".into()),
                    )),
                };
            }

            if std::time::Instant::now() > deadline {
                let _ = self.job_manager.cancel_job(job_id, &tenant).await;
                warn!(job_id = %job_id, "Training trial exceeded wall-clock timeout");
                return Ok(job.current_loss);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_handle_stop_sends_signal() {
        let (tx, rx) = watch::channel(false);
        let handle = ExperimentHandle {
            experiment_id: Uuid::new_v4(),
            stop_tx: tx,
        };
        assert!(!*rx.borrow());
        handle.stop();
        assert!(*rx.borrow());
    }
}
