//! A/B test traffic router.

use ifran_types::ab_test::AbTest;

/// Select which model variant to route a request to.
///
/// Uses a random number to decide based on the traffic split.
/// Returns the model name to use.
#[must_use]
pub fn select_variant(test: &AbTest) -> &str {
    let roll: f64 = rand_fraction();
    if roll < test.traffic_split {
        &test.model_b
    } else {
        &test.model_a
    }
}

/// Simple pseudo-random fraction [0, 1) using thread-local state.
fn rand_fraction() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let hash = hasher.finish();
    (hash % 10000) as f64 / 10000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ifran_types::ab_test::{AbTest, AbTestStatus};

    fn make_test(split: f64) -> AbTest {
        AbTest {
            id: uuid::Uuid::new_v4(),
            name: "test".into(),
            model_a: "model-a".into(),
            model_b: "model-b".into(),
            traffic_split: split,
            status: AbTestStatus::Active,
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn all_traffic_to_a() {
        let test = make_test(0.0);
        // With 0 split, always model_a
        for _ in 0..10 {
            assert_eq!(select_variant(&test), "model-a");
        }
    }

    #[test]
    fn all_traffic_to_b() {
        let test = make_test(1.0);
        // With 1.0 split, always model_b
        for _ in 0..10 {
            assert_eq!(select_variant(&test), "model-b");
        }
    }

    #[test]
    fn mixed_traffic() {
        let test = make_test(0.5);
        let variant = select_variant(&test);
        // Just verify it returns one of the two
        assert!(variant == "model-a" || variant == "model-b");
    }

    #[test]
    fn select_variant_returns_valid() {
        let test = make_test(0.5);
        // Run many iterations — every result must be one of the two models.
        for _ in 0..100 {
            let variant = select_variant(&test);
            assert!(
                variant == "model-a" || variant == "model-b",
                "unexpected variant: {variant}"
            );
        }
    }
}
