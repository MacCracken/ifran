//! RAG retrieval optimizer using Thompson Sampling bandit.
//!
//! Selects the best ranking strategy for RAG queries by maintaining
//! Beta distribution parameters (alpha, beta) for each strategy and
//! sampling from them to decide which strategy to use.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A ranking strategy that can be selected by the bandit.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Strategy {
    pub name: String,
}

/// Beta distribution parameters for a strategy arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmState {
    pub alpha: f64, // successes + 1
    pub beta: f64,  // failures + 1
    pub pulls: u64,
}

impl Default for ArmState {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            pulls: 0,
        }
    }
}

impl ArmState {
    /// Expected value (mean of Beta distribution).
    pub fn expected_value(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Sample from the Beta distribution using a simple approximation.
    /// For a proper implementation, use a statistical library.
    pub fn sample(&self) -> f64 {
        // Use the mean +/- noise based on variance as a simple approximation
        let mean = self.expected_value();
        let variance = (self.alpha * self.beta)
            / ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.0));
        let noise = pseudo_normal() * variance.sqrt();
        (mean + noise).clamp(0.0, 1.0)
    }
}

/// Thompson Sampling bandit for RAG strategy selection.
pub struct RetrievalOptimizer {
    arms: HashMap<String, ArmState>,
}

impl Default for RetrievalOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl RetrievalOptimizer {
    pub fn new() -> Self {
        Self {
            arms: HashMap::new(),
        }
    }

    /// Register a new strategy arm.
    pub fn add_strategy(&mut self, name: &str) {
        self.arms.entry(name.into()).or_default();
    }

    /// Select the best strategy via Thompson Sampling.
    pub fn select(&self) -> Option<String> {
        // Sample each arm once, then pick the max
        let samples: Vec<(String, f64)> = self
            .arms
            .iter()
            .map(|(name, arm)| (name.clone(), arm.sample()))
            .collect();
        samples
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, _)| name)
    }

    /// Record a reward (1.0 = relevant result, 0.0 = irrelevant).
    pub fn record_reward(&mut self, strategy: &str, reward: f64) {
        if let Some(arm) = self.arms.get_mut(strategy) {
            arm.pulls += 1;
            arm.alpha += reward;
            arm.beta += 1.0 - reward;
        }
    }

    /// Get current state of all arms.
    pub fn arm_states(&self) -> &HashMap<String, ArmState> {
        &self.arms
    }

    /// Get the strategy with highest expected value.
    pub fn best_strategy(&self) -> Option<(String, f64)> {
        self.arms
            .iter()
            .max_by(|a, b| {
                a.1.expected_value()
                    .partial_cmp(&b.1.expected_value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, arm)| (name.clone(), arm.expected_value()))
    }
}

/// Simple pseudo-normal random value using Box-Muller-like approximation.
fn pseudo_normal() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let u1 = (hasher.finish() % 10000) as f64 / 10000.0;
    hasher.write_u64(u1.to_bits());
    let u2 = (hasher.finish() % 10000) as f64 / 10000.0;

    let u1 = u1.max(1e-10); // avoid log(0)
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_select() {
        let mut opt = RetrievalOptimizer::new();
        opt.add_strategy("cosine");
        opt.add_strategy("bm25");
        let selected = opt.select().unwrap();
        assert!(selected == "cosine" || selected == "bm25");
    }

    #[test]
    fn reward_shifts_preference() {
        let mut opt = RetrievalOptimizer::new();
        opt.add_strategy("good");
        opt.add_strategy("bad");

        // Give many rewards to "good"
        for _ in 0..50 {
            opt.record_reward("good", 1.0);
            opt.record_reward("bad", 0.0);
        }

        let (best, _) = opt.best_strategy().unwrap();
        assert_eq!(best, "good");
    }

    #[test]
    fn default_arm_state() {
        let arm = ArmState::default();
        assert_eq!(arm.alpha, 1.0);
        assert_eq!(arm.beta, 1.0);
        assert_eq!(arm.expected_value(), 0.5);
    }

    #[test]
    fn arm_sample_in_range() {
        let arm = ArmState {
            alpha: 10.0,
            beta: 5.0,
            pulls: 14,
        };
        for _ in 0..100 {
            let s = arm.sample();
            assert!((0.0..=1.0).contains(&s));
        }
    }

    #[test]
    fn empty_optimizer() {
        let opt = RetrievalOptimizer::new();
        assert!(opt.select().is_none());
        assert!(opt.best_strategy().is_none());
    }

    #[test]
    fn record_updates_pulls() {
        let mut opt = RetrievalOptimizer::new();
        opt.add_strategy("s1");
        opt.record_reward("s1", 1.0);
        assert_eq!(opt.arm_states()["s1"].pulls, 1);
        assert_eq!(opt.arm_states()["s1"].alpha, 2.0);
    }

    #[test]
    fn arm_state_serde() {
        let arm = ArmState {
            alpha: 5.0,
            beta: 3.0,
            pulls: 7,
        };
        let json = serde_json::to_string(&arm).unwrap();
        let back: ArmState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.alpha, 5.0);
        assert_eq!(back.pulls, 7);
    }

    #[test]
    fn three_strategies_reward_convergence() {
        let mut opt = RetrievalOptimizer::new();
        opt.add_strategy("cosine");
        opt.add_strategy("bm25");
        opt.add_strategy("hybrid");

        // Heavily reward "hybrid"
        for _ in 0..100 {
            opt.record_reward("hybrid", 1.0);
            opt.record_reward("cosine", 0.3);
            opt.record_reward("bm25", 0.1);
        }

        let (best, ev) = opt.best_strategy().unwrap();
        assert_eq!(best, "hybrid");
        assert!(ev > 0.8); // expected value should be high
    }

    #[test]
    fn arm_expected_value_after_rewards() {
        let mut opt = RetrievalOptimizer::new();
        opt.add_strategy("s1");

        // 10 rewards of 1.0 => alpha = 1 + 10 = 11, beta stays 1 + 0 = 1
        for _ in 0..10 {
            opt.record_reward("s1", 1.0);
        }
        let arm = &opt.arm_states()["s1"];
        assert_eq!(arm.alpha, 11.0);
        assert_eq!(arm.beta, 1.0);
        // expected_value = 11 / (11 + 1) = 11/12
        assert!((arm.expected_value() - 11.0 / 12.0).abs() < 0.001);
    }
}
