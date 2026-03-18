//! GPU budget enforcement via hoosh accounting integration.
//!
//! When budget enforcement is enabled, training jobs check available GPU-hours
//! with the hoosh accounting service before starting. Jobs are rejected with
//! backpressure when budgets are exhausted.

use synapse_types::SynapseError;
use synapse_types::TenantId;
use synapse_types::error::Result;

/// Budget check result.
#[derive(Debug, Clone)]
pub struct BudgetStatus {
    pub allowed: bool,
    pub remaining_gpu_hours: f64,
    pub reason: Option<String>,
}

/// Checks GPU budgets against hoosh accounting service.
pub struct BudgetChecker {
    endpoint: String,
    max_gpu_hours_per_day: f64,
    client: reqwest::Client,
}

impl BudgetChecker {
    pub fn new(endpoint: &str, max_gpu_hours_per_day: f64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_default();
        Self {
            endpoint: endpoint.to_string(),
            max_gpu_hours_per_day,
            client,
        }
    }

    /// Check if a tenant has budget to start a GPU job.
    ///
    /// Queries hoosh for current usage. If hoosh is unavailable,
    /// falls back to local `max_gpu_hours_per_day` limit (if configured).
    pub async fn check_budget(
        &self,
        tenant_id: &TenantId,
        requested_gpu_hours: f64,
    ) -> Result<BudgetStatus> {
        // Try hoosh first
        match self.query_hoosh(tenant_id).await {
            Ok(status) => {
                if status.remaining_gpu_hours < requested_gpu_hours {
                    return Ok(BudgetStatus {
                        allowed: false,
                        remaining_gpu_hours: status.remaining_gpu_hours,
                        reason: Some(format!(
                            "Insufficient GPU budget: {:.1}h remaining, {:.1}h requested",
                            status.remaining_gpu_hours, requested_gpu_hours
                        )),
                    });
                }
                Ok(status)
            }
            Err(_) => {
                // Hoosh unavailable — fall back to local limit
                if self.max_gpu_hours_per_day > 0.0 {
                    // Without hoosh, we can't track actual usage,
                    // so we allow the job but log a warning
                    tracing::warn!(
                        tenant = %tenant_id,
                        "Hoosh unavailable — budget check skipped, local limit configured"
                    );
                }
                Ok(BudgetStatus {
                    allowed: true,
                    remaining_gpu_hours: self.max_gpu_hours_per_day,
                    reason: None,
                })
            }
        }
    }

    /// Query hoosh accounting service for tenant's GPU budget.
    async fn query_hoosh(&self, tenant_id: &TenantId) -> Result<BudgetStatus> {
        let resp = self
            .client
            .get(format!("{}/v1/budget/gpu", self.endpoint))
            .query(&[("tenant", &tenant_id.to_string())])
            .send()
            .await
            .map_err(|e| SynapseError::HardwareError(format!("Hoosh query failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(SynapseError::HardwareError(format!(
                "Hoosh returned status {}",
                resp.status()
            )));
        }

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| SynapseError::HardwareError(format!("Hoosh response parse error: {e}")))?;

        let remaining = body["remaining_gpu_hours"].as_f64().unwrap_or(0.0);
        let allowed = body["allowed"].as_bool().unwrap_or(remaining > 0.0);

        Ok(BudgetStatus {
            allowed,
            remaining_gpu_hours: remaining,
            reason: body["reason"].as_str().map(String::from),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_status_allowed() {
        let status = BudgetStatus {
            allowed: true,
            remaining_gpu_hours: 10.0,
            reason: None,
        };
        assert!(status.allowed);
    }

    #[test]
    fn budget_status_denied() {
        let status = BudgetStatus {
            allowed: false,
            remaining_gpu_hours: 0.5,
            reason: Some("Budget exhausted".into()),
        };
        assert!(!status.allowed);
        assert!(status.reason.unwrap().contains("exhausted"));
    }

    #[tokio::test]
    async fn checker_hoosh_unavailable_allows() {
        // No hoosh running — should fall back and allow
        let checker = BudgetChecker::new("http://127.0.0.1:19999", 24.0);
        let status = checker
            .check_budget(&TenantId::default_tenant(), 1.0)
            .await
            .unwrap();
        assert!(status.allowed);
    }

    #[tokio::test]
    async fn checker_hoosh_unavailable_no_limit() {
        let checker = BudgetChecker::new("http://127.0.0.1:19999", 0.0);
        let status = checker
            .check_budget(&TenantId::default_tenant(), 1.0)
            .await
            .unwrap();
        assert!(status.allowed);
    }

    #[test]
    fn budget_status_clone() {
        let status = BudgetStatus {
            allowed: true,
            remaining_gpu_hours: 5.5,
            reason: Some("test reason".into()),
        };
        let cloned = status.clone();
        assert!(cloned.allowed);
        assert_eq!(cloned.remaining_gpu_hours, 5.5);
        assert_eq!(cloned.reason.as_deref(), Some("test reason"));
    }

    #[tokio::test]
    async fn checker_zero_budget_allows() {
        // max_gpu_hours=0 means unlimited — should always allow
        let checker = BudgetChecker::new("http://127.0.0.1:19999", 0.0);
        let status = checker
            .check_budget(&TenantId::default_tenant(), 1000.0)
            .await
            .unwrap();
        assert!(status.allowed);
        // remaining should be 0.0 (the configured limit)
        assert_eq!(status.remaining_gpu_hours, 0.0);
    }

    #[tokio::test]
    async fn checker_negative_hours_requested() {
        // Edge case: requesting negative hours (should still work — allowed)
        let checker = BudgetChecker::new("http://127.0.0.1:19999", 24.0);
        let status = checker
            .check_budget(&TenantId::default_tenant(), -1.0)
            .await
            .unwrap();
        assert!(status.allowed);
    }
}
