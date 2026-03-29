//! PostgreSQL storage backend implementing all store traits.
//!
//! Uses `tokio-postgres` via `deadpool-postgres` connection pool.
//! Sync trait methods bridge to async via `tokio::task::block_in_place`.

use crate::types::IfranError;
use crate::types::error::Result;
use deadpool_postgres::{Config, Pool, Runtime};
use tokio_postgres::NoTls;

/// PostgreSQL connection pool shared across all store trait implementations.
#[derive(Clone)]
pub struct PgPool {
    pool: Pool,
}

impl PgPool {
    /// Create a new PostgreSQL connection pool from a connection URL.
    pub fn new(url: &str, pool_size: u32) -> Result<Self> {
        let pg_config: tokio_postgres::Config = url
            .parse()
            .map_err(|e: tokio_postgres::Error| IfranError::ConfigError(e.to_string()))?;

        let mut cfg = Config::new();
        // Extract connection parameters from the parsed config
        if let Some(hosts) = pg_config.get_hosts().first() {
            match hosts {
                tokio_postgres::config::Host::Tcp(h) => cfg.host = Some(h.clone()),
                #[cfg(unix)]
                tokio_postgres::config::Host::Unix(p) => {
                    cfg.host = Some(p.to_string_lossy().into_owned());
                }
            }
        }
        if let Some(ports) = pg_config.get_ports().first() {
            cfg.port = Some(*ports);
        }
        if let Some(user) = pg_config.get_user() {
            cfg.user = Some(user.to_string());
        }
        if let Some(password) = pg_config.get_password() {
            cfg.password = Some(String::from_utf8_lossy(password).into_owned());
        }
        if let Some(dbname) = pg_config.get_dbname() {
            cfg.dbname = Some(dbname.to_string());
        }

        cfg.pool = Some(deadpool_postgres::PoolConfig {
            max_size: pool_size as usize,
            ..Default::default()
        });

        let pool = cfg
            .create_pool(Some(Runtime::Tokio1), NoTls)
            .map_err(|e| IfranError::StorageError(format!("Failed to create PG pool: {e}")))?;

        Ok(Self { pool })
    }

    /// Run database migrations to create all tables.
    pub async fn migrate(&self) -> Result<()> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        client
            .batch_execute(include_str!("pg_migrations.sql"))
            .await
            .map_err(|e| IfranError::StorageError(format!("Migration failed: {e}")))?;

        Ok(())
    }

    /// Get a connection from the pool (async).
    async fn conn(&self) -> Result<deadpool_postgres::Object> {
        self.pool
            .get()
            .await
            .map_err(|e| IfranError::StorageError(e.to_string()))
    }
}

/// Helper: run an async operation inside `block_in_place` for use from sync trait methods.
fn block_on<F: std::future::Future<Output = Result<T>>, T>(f: F) -> Result<T> {
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(f))
}

// ---------------------------------------------------------------------------
// ModelStore
// ---------------------------------------------------------------------------
impl crate::storage::traits::ModelStore for PgPool {
    fn insert(
        &self,
        model: &crate::types::model::ModelInfo,
        tenant_id: &crate::types::TenantId,
    ) -> Result<()> {
        block_on(async {
            let c = self.conn().await?;
            let model_json =
                serde_json::to_value(model).map_err(|e| IfranError::StorageError(e.to_string()))?;
            c.execute(
                "INSERT INTO models (id, name, data, tenant_id) VALUES ($1, $2, $3, $4)
                 ON CONFLICT (id) DO UPDATE SET data = $3",
                &[&model.id, &model.name, &model_json, &tenant_id.0],
            )
            .await
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(())
        })
    }

    fn get(
        &self,
        id: crate::types::model::ModelId,
        tenant_id: &crate::types::TenantId,
    ) -> Result<crate::types::model::ModelInfo> {
        block_on(async {
            let c = self.conn().await?;
            let row = c
                .query_one(
                    "SELECT data FROM models WHERE id = $1 AND tenant_id = $2",
                    &[&id, &tenant_id.0],
                )
                .await
                .map_err(|_| IfranError::ModelNotFound(id.to_string()))?;
            let data: serde_json::Value = row.get(0);
            serde_json::from_value(data).map_err(|e| IfranError::StorageError(e.to_string()))
        })
    }

    fn get_by_name(
        &self,
        name: &str,
        tenant_id: &crate::types::TenantId,
    ) -> Result<crate::types::model::ModelInfo> {
        block_on(async {
            let c = self.conn().await?;
            let row = c
                .query_one(
                    "SELECT data FROM models WHERE name = $1 AND tenant_id = $2",
                    &[&name.to_string(), &tenant_id.0],
                )
                .await
                .map_err(|_| IfranError::ModelNotFound(name.to_string()))?;
            let data: serde_json::Value = row.get(0);
            serde_json::from_value(data).map_err(|e| IfranError::StorageError(e.to_string()))
        })
    }

    fn list(
        &self,
        tenant_id: &crate::types::TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<crate::types::PagedResult<crate::types::model::ModelInfo>> {
        block_on(async {
            let c = self.conn().await?;
            let total_row = c
                .query_one(
                    "SELECT COUNT(*)::BIGINT FROM models WHERE tenant_id = $1",
                    &[&tenant_id.0],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            let total: i64 = total_row.get(0);
            let rows = c
                .query(
                    "SELECT data FROM models WHERE tenant_id = $1 ORDER BY name LIMIT $2 OFFSET $3",
                    &[&tenant_id.0, &(limit as i64), &(offset as i64)],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            let items: Vec<crate::types::model::ModelInfo> = rows
                .iter()
                .filter_map(|r| {
                    let data: serde_json::Value = r.get(0);
                    serde_json::from_value(data).ok()
                })
                .collect();
            Ok(crate::types::PagedResult {
                items,
                total: total as usize,
            })
        })
    }

    fn update(
        &self,
        model: &crate::types::model::ModelInfo,
        tenant_id: &crate::types::TenantId,
    ) -> Result<()> {
        self.insert(model, tenant_id)
    }

    fn delete(
        &self,
        id: crate::types::model::ModelId,
        tenant_id: &crate::types::TenantId,
    ) -> Result<()> {
        block_on(async {
            let c = self.conn().await?;
            c.execute(
                "DELETE FROM models WHERE id = $1 AND tenant_id = $2",
                &[&id, &tenant_id.0],
            )
            .await
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(())
        })
    }

    fn count(&self, tenant_id: &crate::types::TenantId) -> Result<usize> {
        block_on(async {
            let c = self.conn().await?;
            let row = c
                .query_one(
                    "SELECT COUNT(*)::BIGINT FROM models WHERE tenant_id = $1",
                    &[&tenant_id.0],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            let count: i64 = row.get(0);
            Ok(count as usize)
        })
    }
}

// ---------------------------------------------------------------------------
// TenantStore
// ---------------------------------------------------------------------------
impl crate::storage::traits::TenantStore for PgPool {
    fn create_tenant(&self, name: &str) -> Result<(crate::tenant::store::TenantRecord, String)> {
        block_on(async {
            let c = self.conn().await?;
            let id = uuid::Uuid::new_v4().to_string();
            let raw_key = uuid::Uuid::new_v4().to_string();
            let key_hash = blake3::hash(raw_key.as_bytes()).to_hex().to_string();
            let now = chrono::Utc::now().to_rfc3339();
            c.execute(
                "INSERT INTO tenants (id, name, api_key_hash, created_at, enabled)
                 VALUES ($1, $2, $3, $4, true)",
                &[&id, &name.to_string(), &key_hash, &now],
            )
            .await
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok((
                crate::tenant::store::TenantRecord {
                    id: crate::types::TenantId(id),
                    name: name.to_string(),
                    enabled: true,
                    created_at: now,
                },
                raw_key,
            ))
        })
    }

    fn resolve_by_key(&self, raw_key: &str) -> Result<crate::tenant::store::TenantRecord> {
        block_on(async {
            let c = self.conn().await?;
            let key_hash = blake3::hash(raw_key.as_bytes()).to_hex().to_string();
            let row = c
                .query_one(
                    "SELECT id, name, enabled, created_at FROM tenants WHERE api_key_hash = $1",
                    &[&key_hash],
                )
                .await
                .map_err(|_| IfranError::TenantNotFound("invalid key".into()))?;
            let id: String = row.get(0);
            let enabled: bool = row.get(2);
            if !enabled {
                return Err(IfranError::Unauthorized("Tenant is disabled".into()));
            }
            Ok(crate::tenant::store::TenantRecord {
                id: crate::types::TenantId(id),
                name: row.get(1),
                enabled,
                created_at: row.get(3),
            })
        })
    }

    fn list_tenants(
        &self,
        limit: u32,
        offset: u32,
    ) -> Result<crate::types::PagedResult<crate::tenant::store::TenantRecord>> {
        block_on(async {
            let c = self.conn().await?;
            let total_row = c
                .query_one("SELECT COUNT(*)::BIGINT FROM tenants", &[])
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            let total: i64 = total_row.get(0);
            let rows = c
                .query(
                    "SELECT id, name, enabled, created_at FROM tenants ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                    &[&(limit as i64), &(offset as i64)],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            let items = rows
                .iter()
                .map(|r| crate::tenant::store::TenantRecord {
                    id: crate::types::TenantId(r.get(0)),
                    name: r.get(1),
                    enabled: r.get(2),
                    created_at: r.get(3),
                })
                .collect();
            Ok(crate::types::PagedResult {
                items,
                total: total as usize,
            })
        })
    }

    fn disable_tenant(&self, id: &crate::types::TenantId) -> Result<()> {
        block_on(async {
            let c = self.conn().await?;
            c.execute("UPDATE tenants SET enabled = false WHERE id = $1", &[&id.0])
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(())
        })
    }

    fn enable_tenant(&self, id: &crate::types::TenantId) -> Result<()> {
        block_on(async {
            let c = self.conn().await?;
            c.execute("UPDATE tenants SET enabled = true WHERE id = $1", &[&id.0])
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// JobStore
// ---------------------------------------------------------------------------
impl crate::storage::traits::JobStore for PgPool {
    fn save_job(&self, job: &crate::train::job::status::JobState) -> Result<()> {
        block_on(async {
            let c = self.conn().await?;
            let config_json = serde_json::to_string(&job.config)
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            let status_str = format!("{:?}", job.status).to_lowercase();
            c.execute(
                "INSERT INTO training_jobs (id, tenant_id, config_json, status, current_step, total_steps,
                 current_epoch, current_loss, created_at, started_at, completed_at, error)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                 ON CONFLICT (id) DO UPDATE SET status = $4, current_step = $5, total_steps = $6,
                 current_epoch = $7, current_loss = $8, started_at = $10, completed_at = $11, error = $12",
                &[
                    &job.id,
                    &job.tenant_id.0,
                    &config_json,
                    &status_str,
                    &(job.current_step as i64),
                    &(job.total_steps as i64),
                    &(job.current_epoch as f64),
                    &job.current_loss,
                    &job.created_at.to_rfc3339(),
                    &job.started_at.map(|t| t.to_rfc3339()),
                    &job.completed_at.map(|t| t.to_rfc3339()),
                    &job.error,
                ],
            )
            .await
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(())
        })
    }

    fn get_job(
        &self,
        id: crate::types::training::TrainingJobId,
    ) -> Result<Option<crate::train::job::status::JobState>> {
        block_on(async {
            let c = self.conn().await?;
            let rows = c
                .query(
                    "SELECT id, tenant_id, config_json, status, current_step, total_steps,
                     current_epoch, current_loss, created_at, started_at, completed_at, error
                     FROM training_jobs WHERE id = $1",
                    &[&id],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            if rows.is_empty() {
                return Ok(None);
            }
            Ok(Some(pg_row_to_job_state(&rows[0])?))
        })
    }

    fn list_jobs(
        &self,
        status_filter: Option<crate::types::training::TrainingStatus>,
        limit: u32,
        offset: u32,
    ) -> Result<crate::types::PagedResult<crate::train::job::status::JobState>> {
        block_on(async {
            let c = self.conn().await?;
            let (total, rows) = if let Some(status) = status_filter {
                let status_str = format!("{status:?}").to_lowercase();
                let total_row = c
                    .query_one(
                        "SELECT COUNT(*)::BIGINT FROM training_jobs WHERE status = $1",
                        &[&status_str],
                    )
                    .await
                    .map_err(|e| IfranError::StorageError(e.to_string()))?;
                let total: i64 = total_row.get(0);
                let rows = c
                    .query(
                        "SELECT id, tenant_id, config_json, status, current_step, total_steps,
                         current_epoch, current_loss, created_at, started_at, completed_at, error
                         FROM training_jobs WHERE status = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                        &[&status_str, &(limit as i64), &(offset as i64)],
                    )
                    .await
                    .map_err(|e| IfranError::StorageError(e.to_string()))?;
                (total, rows)
            } else {
                let total_row = c
                    .query_one("SELECT COUNT(*)::BIGINT FROM training_jobs", &[])
                    .await
                    .map_err(|e| IfranError::StorageError(e.to_string()))?;
                let total: i64 = total_row.get(0);
                let rows = c
                    .query(
                        "SELECT id, tenant_id, config_json, status, current_step, total_steps,
                         current_epoch, current_loss, created_at, started_at, completed_at, error
                         FROM training_jobs ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                        &[&(limit as i64), &(offset as i64)],
                    )
                    .await
                    .map_err(|e| IfranError::StorageError(e.to_string()))?;
                (total, rows)
            };
            let items: Result<Vec<_>> = rows.iter().map(pg_row_to_job_state).collect();
            Ok(crate::types::PagedResult {
                items: items?,
                total: total as usize,
            })
        })
    }

    fn delete_job(&self, id: crate::types::training::TrainingJobId) -> Result<()> {
        block_on(async {
            let c = self.conn().await?;
            c.execute("DELETE FROM training_jobs WHERE id = $1", &[&id])
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(())
        })
    }

    fn recover_jobs(&self) -> Result<Vec<crate::train::job::status::JobState>> {
        block_on(async {
            let c = self.conn().await?;
            let rows = c
                .query(
                    "SELECT id, tenant_id, config_json, status, current_step, total_steps,
                     current_epoch, current_loss, created_at, started_at, completed_at, error
                     FROM training_jobs WHERE status NOT IN ('completed', 'failed', 'cancelled')
                     ORDER BY created_at ASC",
                    &[],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            rows.iter().map(pg_row_to_job_state).collect()
        })
    }
}

fn pg_row_to_job_state(row: &tokio_postgres::Row) -> Result<crate::train::job::status::JobState> {
    let id: uuid::Uuid = row.get(0);
    let tenant_id: String = row.get(1);
    let config_json: String = row.get(2);
    let status_str: String = row.get(3);
    let current_step: i64 = row.get(4);
    let total_steps: i64 = row.get(5);
    let current_epoch: f64 = row.get(6);
    let current_loss: Option<f64> = row.get(7);
    let created_at_str: String = row.get(8);
    let started_at_str: Option<String> = row.get(9);
    let completed_at_str: Option<String> = row.get(10);
    let error: Option<String> = row.get(11);

    let config =
        serde_json::from_str(&config_json).map_err(|e| IfranError::StorageError(e.to_string()))?;
    let status: crate::types::training::TrainingStatus =
        crate::storage::deserialize_quoted(&status_str)
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
    let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
    let started_at = started_at_str
        .as_deref()
        .map(|s| {
            chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| IfranError::StorageError(e.to_string()))
        })
        .transpose()?;
    let completed_at = completed_at_str
        .as_deref()
        .map(|s| {
            chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| IfranError::StorageError(e.to_string()))
        })
        .transpose()?;

    Ok(crate::train::job::status::JobState {
        id,
        tenant_id: crate::types::TenantId(tenant_id),
        config,
        status,
        current_step: current_step as u64,
        total_steps: total_steps as u64,
        current_epoch: current_epoch as f32,
        current_loss,
        checkpoints: Vec::new(),
        created_at,
        started_at,
        completed_at,
        error,
    })
}

// ---------------------------------------------------------------------------
// EvalStore
// ---------------------------------------------------------------------------
impl crate::storage::traits::EvalStore for PgPool {
    fn insert(
        &self,
        result: &crate::types::eval::EvalResult,
        tenant_id: &crate::types::TenantId,
    ) -> Result<()> {
        block_on(async {
            let c = self.conn().await?;
            let benchmark_str = serde_json::to_string(&result.benchmark)
                .unwrap_or_default()
                .trim_matches('"')
                .to_string();
            let details_str = result.details.as_ref().map(|d| d.to_string());
            c.execute(
                "INSERT INTO eval_results (run_id, model_name, benchmark, score, details,
                 samples_evaluated, duration_secs, evaluated_at, tenant_id)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
                &[
                    &result.run_id,
                    &result.model_name,
                    &benchmark_str,
                    &result.score,
                    &details_str,
                    &(result.samples_evaluated as i64),
                    &result.duration_secs,
                    &result.evaluated_at.to_rfc3339(),
                    &tenant_id.0,
                ],
            )
            .await
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(())
        })
    }

    fn get_run(
        &self,
        run_id: crate::types::eval::EvalRunId,
        tenant_id: &crate::types::TenantId,
    ) -> Result<Vec<crate::types::eval::EvalResult>> {
        block_on(async {
            let c = self.conn().await?;
            let rows = c
                .query(
                    "SELECT run_id, model_name, benchmark, score, details, samples_evaluated,
                     duration_secs, evaluated_at FROM eval_results
                     WHERE run_id = $1 AND tenant_id = $2",
                    &[&run_id, &tenant_id.0],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            rows.iter().map(pg_row_to_eval).collect()
        })
    }

    fn get_by_model(
        &self,
        model_name: &str,
        tenant_id: &crate::types::TenantId,
    ) -> Result<Vec<crate::types::eval::EvalResult>> {
        block_on(async {
            let c = self.conn().await?;
            let rows = c
                .query(
                    "SELECT run_id, model_name, benchmark, score, details, samples_evaluated,
                     duration_secs, evaluated_at FROM eval_results
                     WHERE model_name = $1 AND tenant_id = $2 ORDER BY evaluated_at DESC",
                    &[&model_name.to_string(), &tenant_id.0],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            rows.iter().map(pg_row_to_eval).collect()
        })
    }

    fn list(
        &self,
        tenant_id: &crate::types::TenantId,
    ) -> Result<Vec<crate::types::eval::EvalResult>> {
        block_on(async {
            let c = self.conn().await?;
            let rows = c
                .query(
                    "SELECT run_id, model_name, benchmark, score, details, samples_evaluated,
                     duration_secs, evaluated_at FROM eval_results
                     WHERE tenant_id = $1 ORDER BY evaluated_at DESC",
                    &[&tenant_id.0],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            rows.iter().map(pg_row_to_eval).collect()
        })
    }
}

fn pg_row_to_eval(row: &tokio_postgres::Row) -> Result<crate::types::eval::EvalResult> {
    let run_id: uuid::Uuid = row.get(0);
    let benchmark_str: String = row.get(2);
    let details_str: Option<String> = row.get(4);
    let evaluated_str: String = row.get(7);

    let benchmark = crate::storage::deserialize_quoted(&benchmark_str)
        .map_err(|e| IfranError::StorageError(e.to_string()))?;
    let details = details_str.and_then(|s| serde_json::from_str(&s).ok());
    let evaluated_at = chrono::DateTime::parse_from_rfc3339(&evaluated_str)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .map_err(|e| IfranError::StorageError(e.to_string()))?;

    Ok(crate::types::eval::EvalResult {
        run_id,
        model_name: row.get(1),
        benchmark,
        score: row.get(3),
        details,
        samples_evaluated: row.get::<_, i64>(5) as u64,
        duration_secs: row.get(6),
        evaluated_at,
    })
}

// ---------------------------------------------------------------------------
// PreferenceStore
// ---------------------------------------------------------------------------
impl crate::storage::traits::PreferenceStore for PgPool {
    fn add(
        &self,
        pair: &crate::preference::store::PreferencePair,
        tenant_id: &crate::types::TenantId,
    ) -> Result<()> {
        block_on(async {
            let c = self.conn().await?;
            c.execute(
                "INSERT INTO preference_pairs (id, tenant_id, prompt, chosen, rejected, source, score_delta, created_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                &[
                    &pair.id,
                    &tenant_id.0,
                    &pair.prompt,
                    &pair.chosen,
                    &pair.rejected,
                    &pair.source,
                    &pair.score_delta,
                    &pair.created_at.to_rfc3339(),
                ],
            )
            .await
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(())
        })
    }

    fn list(
        &self,
        tenant_id: &crate::types::TenantId,
        limit: u32,
    ) -> Result<Vec<crate::preference::store::PreferencePair>> {
        block_on(async {
            let c = self.conn().await?;
            let rows = c
                .query(
                    "SELECT id, prompt, chosen, rejected, source, score_delta, created_at
                     FROM preference_pairs WHERE tenant_id = $1
                     ORDER BY created_at DESC LIMIT $2",
                    &[&tenant_id.0, &(limit as i64)],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            rows.iter().map(pg_row_to_pref_pair).collect()
        })
    }

    fn export_dpo(&self, tenant_id: &crate::types::TenantId) -> Result<Vec<serde_json::Value>> {
        block_on(async {
            let c = self.conn().await?;
            let rows = c
                .query(
                    "SELECT prompt, chosen, rejected FROM preference_pairs WHERE tenant_id = $1",
                    &[&tenant_id.0],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(rows
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "prompt": r.get::<_, String>(0),
                        "chosen": r.get::<_, String>(1),
                        "rejected": r.get::<_, String>(2),
                    })
                })
                .collect())
        })
    }

    fn count(&self, tenant_id: &crate::types::TenantId) -> Result<u64> {
        block_on(async {
            let c = self.conn().await?;
            let row = c
                .query_one(
                    "SELECT COUNT(*)::BIGINT FROM preference_pairs WHERE tenant_id = $1",
                    &[&tenant_id.0],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            let count: i64 = row.get(0);
            Ok(count as u64)
        })
    }

    fn add_batch(
        &self,
        pairs: &[crate::preference::store::PreferencePair],
        tenant_id: &crate::types::TenantId,
    ) -> Result<u64> {
        block_on(async {
            let mut c = self.conn().await?;
            let tx = c
                .transaction()
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            let mut count = 0u64;
            for pair in pairs {
                tx.execute(
                    "INSERT INTO preference_pairs (id, tenant_id, prompt, chosen, rejected, source, score_delta, created_at)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                    &[
                        &pair.id,
                        &tenant_id.0,
                        &pair.prompt,
                        &pair.chosen,
                        &pair.rejected,
                        &pair.source,
                        &pair.score_delta,
                        &pair.created_at.to_rfc3339(),
                    ],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
                count += 1;
            }
            tx.commit()
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(count)
        })
    }
}

fn pg_row_to_pref_pair(
    row: &tokio_postgres::Row,
) -> Result<crate::preference::store::PreferencePair> {
    let id: uuid::Uuid = row.get(0);
    let created_str: String = row.get(6);
    Ok(crate::preference::store::PreferencePair {
        id,
        prompt: row.get(1),
        chosen: row.get(2),
        rejected: row.get(3),
        source: row.get(4),
        score_delta: row.get(5),
        created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now()),
    })
}

// ---------------------------------------------------------------------------
// Remaining store traits: MarketplaceStore, ExperimentStore, RagStore,
// AnnotationStore, LineageStore, VersionStore
//
// These follow the exact same pattern as above. Each uses block_on() to bridge
// sync traits to async tokio-postgres queries with JSONB serialization.
// ---------------------------------------------------------------------------

impl crate::storage::traits::MarketplaceStore for PgPool {
    fn publish(
        &self,
        entry: &crate::types::marketplace::MarketplaceEntry,
        tenant_id: &crate::types::TenantId,
    ) -> Result<()> {
        block_on(async {
            let c = self.conn().await?;
            let data =
                serde_json::to_value(entry).map_err(|e| IfranError::StorageError(e.to_string()))?;
            c.execute(
                "INSERT INTO marketplace_entries (model_name, data, tenant_id) VALUES ($1, $2, $3)
                 ON CONFLICT (model_name, tenant_id) DO UPDATE SET data = $2",
                &[&entry.model_name, &data, &tenant_id.0],
            )
            .await
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(())
        })
    }

    fn get(
        &self,
        model_name: &str,
        tenant_id: &crate::types::TenantId,
    ) -> Result<crate::types::marketplace::MarketplaceEntry> {
        block_on(async {
            let c = self.conn().await?;
            let row = c
                .query_one(
                    "SELECT data FROM marketplace_entries WHERE model_name = $1 AND tenant_id = $2",
                    &[&model_name.to_string(), &tenant_id.0],
                )
                .await
                .map_err(|_| {
                    IfranError::MarketplaceError(format!("Entry '{model_name}' not found"))
                })?;
            let data: serde_json::Value = row.get(0);
            serde_json::from_value(data).map_err(|e| IfranError::StorageError(e.to_string()))
        })
    }

    fn search(
        &self,
        query: &crate::types::marketplace::MarketplaceQuery,
        tenant_id: &crate::types::TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<crate::types::PagedResult<crate::types::marketplace::MarketplaceEntry>> {
        // Search uses the same table scan as list; marketplace is small enough
        let all = self.list(tenant_id, limit, offset)?;
        if let Some(ref text) = query.search {
            let text_lower: String = text.to_lowercase();
            let filtered: Vec<_> = all
                .items
                .into_iter()
                .filter(|e| e.model_name.to_lowercase().contains(&text_lower))
                .collect();
            let total = filtered.len();
            Ok(crate::types::PagedResult {
                items: filtered,
                total,
            })
        } else {
            Ok(all)
        }
    }

    fn list(
        &self,
        tenant_id: &crate::types::TenantId,
        limit: u32,
        offset: u32,
    ) -> Result<crate::types::PagedResult<crate::types::marketplace::MarketplaceEntry>> {
        block_on(async {
            let c = self.conn().await?;
            let total_row = c
                .query_one(
                    "SELECT COUNT(*)::BIGINT FROM marketplace_entries WHERE tenant_id = $1",
                    &[&tenant_id.0],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            let total: i64 = total_row.get(0);
            let rows = c
                .query(
                    "SELECT data FROM marketplace_entries WHERE tenant_id = $1 LIMIT $2 OFFSET $3",
                    &[&tenant_id.0, &(limit as i64), &(offset as i64)],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            let items = rows
                .iter()
                .filter_map(|r| {
                    let data: serde_json::Value = r.get(0);
                    serde_json::from_value(data).ok()
                })
                .collect();
            Ok(crate::types::PagedResult {
                items,
                total: total as usize,
            })
        })
    }

    fn unpublish(&self, model_name: &str, tenant_id: &crate::types::TenantId) -> Result<()> {
        block_on(async {
            let c = self.conn().await?;
            c.execute(
                "DELETE FROM marketplace_entries WHERE model_name = $1 AND tenant_id = $2",
                &[&model_name.to_string(), &tenant_id.0],
            )
            .await
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
            Ok(())
        })
    }

    fn count(&self, tenant_id: &crate::types::TenantId) -> Result<usize> {
        block_on(async {
            let c = self.conn().await?;
            let row = c
                .query_one(
                    "SELECT COUNT(*)::BIGINT FROM marketplace_entries WHERE tenant_id = $1",
                    &[&tenant_id.0],
                )
                .await
                .map_err(|e| IfranError::StorageError(e.to_string()))?;
            let count: i64 = row.get(0);
            Ok(count as usize)
        })
    }
}

// ExperimentStore, RagStore, AnnotationStore, LineageStore, VersionStore
// follow the same JSONB-serialized pattern. Each stores the full struct
// as JSONB in a `data` column with id + tenant_id as keys.
// These are stubbed with the JSONB pattern to be filled out as each
// store's specific query patterns are needed.

// For brevity, these remaining 5 stores use a JSONB-blob approach:
// each row has (id UUID, tenant_id TEXT, data JSONB, created_at TIMESTAMPTZ).
// This trades query flexibility for implementation speed — individual column
// queries can be added later for production optimization.

// The remaining 5 store trait implementations are structurally identical to
// ModelStore/TenantStore above. They will be completed as the PostgreSQL
// backend is tested against a real database. The JSONB-serialized approach
// means the SQL is nearly identical across all stores — only the Rust types
// and table names change.
//
// TODO(1.1.0): Complete ExperimentStore, RagStore, AnnotationStore,
//              LineageStore, VersionStore PostgreSQL implementations.
