//! Cost-aware backend selection.

use std::collections::HashMap;

/// Cost configuration per backend.
#[derive(Debug, Clone)]
pub struct CostConfig {
    /// Cost per 1K tokens for each backend (by ID).
    costs: HashMap<String, f64>,
}

impl Default for CostConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl CostConfig {
    pub fn new() -> Self {
        Self {
            costs: HashMap::new(),
        }
    }

    pub fn set_cost(&mut self, backend_id: &str, cost_per_1k: f64) {
        self.costs.insert(backend_id.into(), cost_per_1k);
    }

    pub fn get_cost(&self, backend_id: &str) -> Option<f64> {
        self.costs.get(backend_id).copied()
    }

    /// Select the cheapest backend from a list of candidates.
    pub fn cheapest<'a>(&self, candidates: &[&'a str]) -> Option<&'a str> {
        candidates
            .iter()
            .min_by(|a, b| {
                let cost_a = self.costs.get(**a).copied().unwrap_or(f64::MAX);
                let cost_b = self.costs.get(**b).copied().unwrap_or(f64::MAX);
                cost_a
                    .partial_cmp(&cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    /// Select the best backend considering both cost and a preference.
    /// Returns the cheapest among those within `budget_per_1k`, or
    /// the cheapest overall if none are within budget.
    pub fn select_within_budget<'a>(
        &self,
        candidates: &[&'a str],
        budget_per_1k: f64,
    ) -> Option<&'a str> {
        let within_budget: Vec<&str> = candidates
            .iter()
            .filter(|id| self.costs.get(**id).copied().unwrap_or(0.0) <= budget_per_1k)
            .copied()
            .collect();

        if within_budget.is_empty() {
            self.cheapest(candidates)
        } else {
            self.cheapest(&within_budget)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cheapest_selection() {
        let mut config = CostConfig::new();
        config.set_cost("llamacpp", 0.0);
        config.set_cost("vllm", 0.002);
        config.set_cost("ollama", 0.0);

        let cheapest = config.cheapest(&["llamacpp", "vllm", "ollama"]);
        assert!(cheapest == Some("llamacpp") || cheapest == Some("ollama"));
    }

    #[test]
    fn within_budget() {
        let mut config = CostConfig::new();
        config.set_cost("cheap", 0.001);
        config.set_cost("expensive", 0.01);

        let selected = config.select_within_budget(&["cheap", "expensive"], 0.005);
        assert_eq!(selected, Some("cheap"));
    }

    #[test]
    fn over_budget_fallback() {
        let mut config = CostConfig::new();
        config.set_cost("a", 0.01);
        config.set_cost("b", 0.02);

        // Budget is 0.005 -- nothing fits, so return cheapest overall
        let selected = config.select_within_budget(&["a", "b"], 0.005);
        assert_eq!(selected, Some("a"));
    }

    #[test]
    fn empty_candidates() {
        let config = CostConfig::new();
        assert!(config.cheapest(&[]).is_none());
    }

    #[test]
    fn unknown_backend_defaults_to_max() {
        let mut config = CostConfig::new();
        config.set_cost("known", 0.001);
        let cheapest = config.cheapest(&["known", "unknown"]);
        assert_eq!(cheapest, Some("known"));
    }

    #[test]
    fn get_cost() {
        let mut config = CostConfig::new();
        config.set_cost("vllm", 0.005);
        assert_eq!(config.get_cost("vllm"), Some(0.005));
        assert_eq!(config.get_cost("nonexistent"), None);
    }

    #[test]
    fn nan_costs_do_not_panic() {
        let mut config = CostConfig::new();
        config.set_cost("a", f64::NAN);
        config.set_cost("b", 0.01);
        config.set_cost("c", f64::NAN);

        // Should not panic, and should return a valid result
        let result = config.cheapest(&["a", "b", "c"]);
        assert!(result.is_some());

        let result = config.select_within_budget(&["a", "b", "c"], 0.05);
        assert!(result.is_some());
    }

    #[test]
    fn set_cost_overwrite() {
        let mut config = CostConfig::new();
        config.set_cost("vllm", 0.01);
        assert_eq!(config.get_cost("vllm"), Some(0.01));
        config.set_cost("vllm", 0.05);
        assert_eq!(config.get_cost("vllm"), Some(0.05));
    }

    #[test]
    fn select_within_budget_all_free() {
        let mut config = CostConfig::new();
        config.set_cost("a", 0.0);
        config.set_cost("b", 0.0);
        config.set_cost("c", 0.0);

        // All costs are 0, any budget works.
        let selected = config.select_within_budget(&["a", "b", "c"], 0.001);
        assert!(selected.is_some());
        let s = selected.unwrap();
        assert!(s == "a" || s == "b" || s == "c");
    }
}
