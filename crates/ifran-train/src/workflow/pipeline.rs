//! DAG-based ML workflow orchestration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub type StepId = Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    pub name: String,
    pub steps: Vec<PipelineStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStep {
    pub id: StepId,
    pub name: String,
    pub step_type: StepType,
    pub depends_on: Vec<StepId>,
    pub config: serde_json::Value,
    pub status: StepStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepType {
    Curate,
    Train,
    Evaluate,
    Approve,
    Deploy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
    WaitingApproval,
}

/// Find the next steps that are ready to run (all dependencies completed).
pub fn ready_steps(pipeline: &Pipeline) -> Vec<StepId> {
    let completed: std::collections::HashSet<StepId> = pipeline
        .steps
        .iter()
        .filter(|s| s.status == StepStatus::Completed)
        .map(|s| s.id)
        .collect();

    pipeline
        .steps
        .iter()
        .filter(|s| s.status == StepStatus::Pending)
        .filter(|s| s.depends_on.iter().all(|dep| completed.contains(dep)))
        .map(|s| s.id)
        .collect()
}

/// Validate a pipeline DAG (no cycles, all dependencies exist).
pub fn validate_dag(pipeline: &Pipeline) -> Result<(), String> {
    let ids: HashMap<StepId, usize> = pipeline
        .steps
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id, i))
        .collect();

    // Check all dependencies exist
    for step in &pipeline.steps {
        for dep in &step.depends_on {
            if !ids.contains_key(dep) {
                return Err(format!(
                    "Step '{}' depends on nonexistent step {dep}",
                    step.name
                ));
            }
        }
    }

    // Topological sort to detect cycles
    let mut visited = vec![0u8; pipeline.steps.len()]; // 0=unvisited, 1=visiting, 2=done
    for i in 0..pipeline.steps.len() {
        if visited[i] == 0 {
            if has_cycle(&pipeline.steps, &ids, i, &mut visited) {
                return Err("Pipeline contains a cycle".into());
            }
        }
    }
    Ok(())
}

fn has_cycle(
    steps: &[PipelineStep],
    ids: &HashMap<StepId, usize>,
    idx: usize,
    visited: &mut Vec<u8>,
) -> bool {
    visited[idx] = 1;
    for dep in &steps[idx].depends_on {
        if let Some(&dep_idx) = ids.get(dep) {
            if visited[dep_idx] == 1 {
                return true;
            }
            if visited[dep_idx] == 0 && has_cycle(steps, ids, dep_idx, visited) {
                return true;
            }
        }
    }
    visited[idx] = 2;
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(name: &str, step_type: StepType, deps: Vec<StepId>) -> PipelineStep {
        PipelineStep {
            id: Uuid::new_v4(),
            name: name.into(),
            step_type,
            depends_on: deps,
            config: serde_json::json!({}),
            status: StepStatus::Pending,
        }
    }

    #[test]
    fn valid_pipeline() {
        let s1 = step("curate", StepType::Curate, vec![]);
        let s2 = step("train", StepType::Train, vec![s1.id]);
        let s3 = step("eval", StepType::Evaluate, vec![s2.id]);
        let s4 = step("approve", StepType::Approve, vec![s3.id]);
        let s5 = step("deploy", StepType::Deploy, vec![s4.id]);

        let pipeline = Pipeline {
            name: "ml-pipeline".into(),
            steps: vec![s1, s2, s3, s4, s5],
        };
        assert!(validate_dag(&pipeline).is_ok());
    }

    #[test]
    fn ready_steps_finds_roots() {
        let s1 = step("curate", StepType::Curate, vec![]);
        let s2 = step("train", StepType::Train, vec![s1.id]);

        let pipeline = Pipeline {
            name: "p".into(),
            steps: vec![s1.clone(), s2],
        };
        let ready = ready_steps(&pipeline);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], s1.id);
    }

    #[test]
    fn ready_after_completion() {
        let mut s1 = step("curate", StepType::Curate, vec![]);
        s1.status = StepStatus::Completed;
        let s2 = step("train", StepType::Train, vec![s1.id]);

        let pipeline = Pipeline {
            name: "p".into(),
            steps: vec![s1, s2.clone()],
        };
        let ready = ready_steps(&pipeline);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], s2.id);
    }

    #[test]
    fn cycle_detection() {
        let mut s1 = step("a", StepType::Curate, vec![]);
        let s2 = step("b", StepType::Train, vec![s1.id]);
        s1.depends_on = vec![s2.id]; // cycle!
        let pipeline = Pipeline {
            name: "p".into(),
            steps: vec![s1, s2],
        };
        assert!(validate_dag(&pipeline).is_err());
    }

    #[test]
    fn missing_dependency() {
        let s1 = step("a", StepType::Curate, vec![Uuid::new_v4()]); // nonexistent dep
        let pipeline = Pipeline {
            name: "p".into(),
            steps: vec![s1],
        };
        assert!(validate_dag(&pipeline).is_err());
    }

    #[test]
    fn step_type_serde() {
        for t in [
            StepType::Curate,
            StepType::Train,
            StepType::Evaluate,
            StepType::Approve,
            StepType::Deploy,
        ] {
            let json = serde_json::to_string(&t).unwrap();
            let back: StepType = serde_json::from_str(&json).unwrap();
            assert_eq!(t, back);
        }
    }

    #[test]
    fn parallel_steps() {
        // Two steps with no dependencies — both should be ready.
        let s1 = step("curate-a", StepType::Curate, vec![]);
        let s2 = step("curate-b", StepType::Curate, vec![]);
        let pipeline = Pipeline {
            name: "p".into(),
            steps: vec![s1.clone(), s2.clone()],
        };
        let ready = ready_steps(&pipeline);
        assert_eq!(ready.len(), 2);
        assert!(ready.contains(&s1.id));
        assert!(ready.contains(&s2.id));
    }

    #[test]
    fn all_completed_nothing_ready() {
        let mut s1 = step("curate", StepType::Curate, vec![]);
        s1.status = StepStatus::Completed;
        let mut s2 = step("train", StepType::Train, vec![s1.id]);
        s2.status = StepStatus::Completed;
        let pipeline = Pipeline {
            name: "p".into(),
            steps: vec![s1, s2],
        };
        let ready = ready_steps(&pipeline);
        assert!(ready.is_empty());
    }

    #[test]
    fn empty_pipeline_valid() {
        let pipeline = Pipeline {
            name: "empty".into(),
            steps: vec![],
        };
        assert!(validate_dag(&pipeline).is_ok());
        assert!(ready_steps(&pipeline).is_empty());
    }

    #[test]
    fn diamond_dependency() {
        // A -> B, A -> C, B -> D, C -> D
        let a = step("A", StepType::Curate, vec![]);
        let b = step("B", StepType::Train, vec![a.id]);
        let c = step("C", StepType::Evaluate, vec![a.id]);
        let d = step("D", StepType::Deploy, vec![b.id, c.id]);

        let pipeline = Pipeline {
            name: "diamond".into(),
            steps: vec![a.clone(), b.clone(), c.clone(), d.clone()],
        };
        assert!(validate_dag(&pipeline).is_ok());

        // Initially only A is ready
        let ready = ready_steps(&pipeline);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], a.id);
    }

    #[test]
    fn single_step_pipeline() {
        let s = step("only", StepType::Deploy, vec![]);
        let pipeline = Pipeline {
            name: "single".into(),
            steps: vec![s.clone()],
        };
        assert!(validate_dag(&pipeline).is_ok());
        let ready = ready_steps(&pipeline);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], s.id);
    }

    #[test]
    fn step_status_serde() {
        for status in [
            StepStatus::Pending,
            StepStatus::Running,
            StepStatus::Completed,
            StepStatus::Failed,
            StepStatus::Skipped,
            StepStatus::WaitingApproval,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: StepStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, back);
        }
    }
}
