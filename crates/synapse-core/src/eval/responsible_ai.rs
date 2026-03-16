//! Responsible AI metrics — fairness analysis and cohort error rates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of a responsible AI audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsibleAiReport {
    pub model_name: String,
    pub total_samples: u64,
    pub overall_accuracy: f64,
    pub cohort_metrics: Vec<CohortMetric>,
    pub fairness: FairnessMetrics,
}

/// Per-cohort accuracy metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohortMetric {
    pub cohort_name: String,
    pub sample_count: u64,
    pub accuracy: f64,
    pub error_rate: f64,
}

/// Fairness metrics across cohorts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessMetrics {
    /// Max accuracy difference between any two cohorts.
    pub demographic_parity_gap: f64,
    /// Ratio of min to max accuracy across cohorts.
    pub disparate_impact_ratio: f64,
    /// Whether the model passes a fairness threshold (80% rule).
    pub passes_80_percent_rule: bool,
}

/// A labeled sample for cohort analysis.
pub struct LabeledSample {
    pub cohort: String,
    pub correct: bool,
}

/// Compute responsible AI metrics from labeled samples.
pub fn compute_report(model_name: &str, samples: &[LabeledSample]) -> ResponsibleAiReport {
    let total = samples.len() as u64;
    let total_correct = samples.iter().filter(|s| s.correct).count() as u64;
    let overall_accuracy = if total == 0 {
        0.0
    } else {
        total_correct as f64 / total as f64
    };

    // Group by cohort
    let mut cohorts: HashMap<&str, (u64, u64)> = HashMap::new();
    for sample in samples {
        let entry = cohorts.entry(&sample.cohort).or_insert((0, 0));
        entry.0 += 1; // total
        if sample.correct {
            entry.1 += 1;
        } // correct
    }

    let cohort_metrics: Vec<CohortMetric> = cohorts
        .iter()
        .map(|(name, (count, correct))| {
            let accuracy = if *count == 0 {
                0.0
            } else {
                *correct as f64 / *count as f64
            };
            CohortMetric {
                cohort_name: name.to_string(),
                sample_count: *count,
                accuracy,
                error_rate: 1.0 - accuracy,
            }
        })
        .collect();

    let fairness = compute_fairness(&cohort_metrics);

    ResponsibleAiReport {
        model_name: model_name.into(),
        total_samples: total,
        overall_accuracy,
        cohort_metrics,
        fairness,
    }
}

fn compute_fairness(cohorts: &[CohortMetric]) -> FairnessMetrics {
    if cohorts.is_empty() {
        return FairnessMetrics {
            demographic_parity_gap: 0.0,
            disparate_impact_ratio: 1.0,
            passes_80_percent_rule: true,
        };
    }

    let accuracies: Vec<f64> = cohorts.iter().map(|c| c.accuracy).collect();
    let max_acc = accuracies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_acc = accuracies.iter().cloned().fold(f64::INFINITY, f64::min);

    let gap = max_acc - min_acc;
    let ratio = if max_acc > 0.0 {
        min_acc / max_acc
    } else {
        1.0
    };

    FairnessMetrics {
        demographic_parity_gap: gap,
        disparate_impact_ratio: ratio,
        passes_80_percent_rule: ratio >= 0.8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(cohort: &str, correct: bool) -> LabeledSample {
        LabeledSample {
            cohort: cohort.into(),
            correct,
        }
    }

    #[test]
    fn fair_model() {
        let samples = vec![
            sample("A", true),
            sample("A", true),
            sample("A", false),
            sample("B", true),
            sample("B", true),
            sample("B", false),
        ];
        let report = compute_report("model", &samples);
        assert_eq!(report.total_samples, 6);
        assert!((report.overall_accuracy - 0.6667).abs() < 0.01);
        assert!(report.fairness.passes_80_percent_rule);
        assert!(report.fairness.demographic_parity_gap < 0.01);
    }

    #[test]
    fn unfair_model() {
        let samples = vec![
            sample("A", true),
            sample("A", true),
            sample("A", true), // 100%
            sample("B", false),
            sample("B", false),
            sample("B", true), // 33%
        ];
        let report = compute_report("model", &samples);
        assert!(!report.fairness.passes_80_percent_rule);
        assert!(report.fairness.demographic_parity_gap > 0.5);
    }

    #[test]
    fn empty_samples() {
        let report = compute_report("model", &[]);
        assert_eq!(report.total_samples, 0);
        assert_eq!(report.overall_accuracy, 0.0);
        assert!(report.fairness.passes_80_percent_rule);
    }

    #[test]
    fn single_cohort() {
        let samples = vec![sample("A", true), sample("A", false)];
        let report = compute_report("model", &samples);
        assert_eq!(report.cohort_metrics.len(), 1);
        assert_eq!(report.fairness.demographic_parity_gap, 0.0);
    }

    #[test]
    fn report_serde() {
        let samples = vec![sample("A", true), sample("B", false)];
        let report = compute_report("test", &samples);
        let json = serde_json::to_string(&report).unwrap();
        let back: ResponsibleAiReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_name, "test");
    }

    #[test]
    fn disparate_impact_ratio() {
        let samples = vec![
            sample("A", true),
            sample("A", true), // 100%
            sample("B", true),
            sample("B", false), // 50%
        ];
        let report = compute_report("model", &samples);
        assert!((report.fairness.disparate_impact_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn three_cohorts() {
        let samples = vec![
            // Cohort A: 2/2 = 100%
            sample("A", true),
            sample("A", true),
            // Cohort B: 1/2 = 50%
            sample("B", true),
            sample("B", false),
            // Cohort C: 0/2 = 0%
            sample("C", false),
            sample("C", false),
        ];
        let report = compute_report("model", &samples);
        assert_eq!(report.cohort_metrics.len(), 3);
        // Gap between best (100%) and worst (0%) = 1.0
        assert!((report.fairness.demographic_parity_gap - 1.0).abs() < 0.01);
        // 80% rule: min/max = 0/1 = 0 < 0.8
        assert!(!report.fairness.passes_80_percent_rule);
    }

    #[test]
    fn all_correct() {
        let samples = vec![
            sample("A", true),
            sample("A", true),
            sample("B", true),
            sample("B", true),
        ];
        let report = compute_report("model", &samples);
        assert_eq!(report.overall_accuracy, 1.0);
        assert!(report.fairness.passes_80_percent_rule);
        assert!(report.fairness.demographic_parity_gap < f64::EPSILON);
        assert!((report.fairness.disparate_impact_ratio - 1.0).abs() < f64::EPSILON);
    }
}
