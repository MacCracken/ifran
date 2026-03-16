//! Approval gates for human-in-the-loop model deployment.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub type ApprovalId = Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: ApprovalId,
    pub pipeline_name: String,
    pub stage: String,
    pub artifact_ref: String,
    pub status: ApprovalStatus,
    pub reviewer: Option<String>,
    pub comment: Option<String>,
    pub created_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
}

/// Manages approval gates.
pub struct ApprovalGate {
    requests: HashMap<ApprovalId, ApprovalRequest>,
}

impl Default for ApprovalGate {
    fn default() -> Self {
        Self::new()
    }
}

impl ApprovalGate {
    pub fn new() -> Self {
        Self {
            requests: HashMap::new(),
        }
    }

    /// Create a new approval request (blocks pipeline until resolved).
    pub fn request_approval(
        &mut self,
        pipeline: &str,
        stage: &str,
        artifact: &str,
    ) -> ApprovalRequest {
        let req = ApprovalRequest {
            id: Uuid::new_v4(),
            pipeline_name: pipeline.into(),
            stage: stage.into(),
            artifact_ref: artifact.into(),
            status: ApprovalStatus::Pending,
            reviewer: None,
            comment: None,
            created_at: Utc::now(),
            resolved_at: None,
        };
        self.requests.insert(req.id, req.clone());
        req
    }

    /// Approve a pending request.
    pub fn approve(
        &mut self,
        id: ApprovalId,
        reviewer: &str,
        comment: Option<&str>,
    ) -> Option<&ApprovalRequest> {
        if let Some(req) = self.requests.get_mut(&id) {
            if req.status == ApprovalStatus::Pending {
                req.status = ApprovalStatus::Approved;
                req.reviewer = Some(reviewer.into());
                req.comment = comment.map(String::from);
                req.resolved_at = Some(Utc::now());
            }
        }
        self.requests.get(&id)
    }

    /// Reject a pending request.
    pub fn reject(
        &mut self,
        id: ApprovalId,
        reviewer: &str,
        comment: Option<&str>,
    ) -> Option<&ApprovalRequest> {
        if let Some(req) = self.requests.get_mut(&id) {
            if req.status == ApprovalStatus::Pending {
                req.status = ApprovalStatus::Rejected;
                req.reviewer = Some(reviewer.into());
                req.comment = comment.map(String::from);
                req.resolved_at = Some(Utc::now());
            }
        }
        self.requests.get(&id)
    }

    /// Get a request by ID.
    pub fn get(&self, id: ApprovalId) -> Option<&ApprovalRequest> {
        self.requests.get(&id)
    }

    /// List pending requests.
    pub fn pending(&self) -> Vec<&ApprovalRequest> {
        self.requests
            .values()
            .filter(|r| r.status == ApprovalStatus::Pending)
            .collect()
    }

    /// Check if a request is approved.
    pub fn is_approved(&self, id: ApprovalId) -> bool {
        self.requests
            .get(&id)
            .is_some_and(|r| r.status == ApprovalStatus::Approved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_approve() {
        let mut gate = ApprovalGate::new();
        let req = gate.request_approval("train-pipeline", "deploy", "model-v2");
        assert_eq!(req.status, ApprovalStatus::Pending);

        gate.approve(req.id, "admin", Some("Looks good"));
        assert!(gate.is_approved(req.id));
    }

    #[test]
    fn create_and_reject() {
        let mut gate = ApprovalGate::new();
        let req = gate.request_approval("pipe", "deploy", "model");
        gate.reject(req.id, "reviewer", Some("Needs more testing"));
        let fetched = gate.get(req.id).unwrap();
        assert_eq!(fetched.status, ApprovalStatus::Rejected);
        assert!(fetched.comment.as_ref().unwrap().contains("more testing"));
    }

    #[test]
    fn pending_list() {
        let mut gate = ApprovalGate::new();
        gate.request_approval("p1", "deploy", "m1");
        gate.request_approval("p2", "deploy", "m2");
        let req3 = gate.request_approval("p3", "deploy", "m3");
        gate.approve(req3.id, "admin", None);

        assert_eq!(gate.pending().len(), 2);
    }

    #[test]
    fn cannot_approve_twice() {
        let mut gate = ApprovalGate::new();
        let req = gate.request_approval("p", "s", "a");
        gate.approve(req.id, "user1", None);
        // Second approve is a no-op since already approved
        gate.reject(req.id, "user2", None);
        assert!(gate.is_approved(req.id));
    }

    #[test]
    fn status_serde() {
        for s in [
            ApprovalStatus::Pending,
            ApprovalStatus::Approved,
            ApprovalStatus::Rejected,
            ApprovalStatus::Expired,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: ApprovalStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }
}
