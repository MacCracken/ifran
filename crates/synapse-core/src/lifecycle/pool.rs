//! Hot model pool for managing loaded model instances.
//!
//! The pool keeps track of loaded models and supports hot-swapping:
//! replacing a running model with a new version without dropping requests.

use std::collections::HashMap;
use std::sync::Arc;
use synapse_types::model::ModelId;
use tokio::sync::RwLock;

/// A slot in the model pool.
#[derive(Debug, Clone)]
pub struct PoolSlot {
    /// The model ID loaded in this slot.
    pub model_id: ModelId,
    /// Friendly name / tag for routing (e.g. "llama-3.1-8b").
    pub name: String,
    /// Backend handle string used to send inference requests.
    pub handle: String,
    /// Whether this slot is actively serving requests.
    pub active: bool,
}

/// Pool of loaded models supporting hot-swap operations.
///
/// Models are addressed by a slot name (string key). Hot-swapping replaces
/// the model in a slot atomically so in-flight requests to the old model
/// can still complete while new requests go to the new model.
pub struct ModelPool {
    slots: Arc<RwLock<HashMap<String, PoolSlot>>>,
}

impl ModelPool {
    /// Create an empty model pool.
    pub fn new() -> Self {
        Self {
            slots: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a model to the pool under the given slot name.
    ///
    /// If a slot with this name already exists, it is replaced (hot-swap).
    /// Returns the previous slot contents, if any.
    pub async fn put(&self, slot_name: String, slot: PoolSlot) -> Option<PoolSlot> {
        let mut slots = self.slots.write().await;
        slots.insert(slot_name, slot)
    }

    /// Get a snapshot of a slot by name.
    pub async fn get(&self, slot_name: &str) -> Option<PoolSlot> {
        let slots = self.slots.read().await;
        slots.get(slot_name).cloned()
    }

    /// Remove a slot from the pool.
    pub async fn remove(&self, slot_name: &str) -> Option<PoolSlot> {
        let mut slots = self.slots.write().await;
        slots.remove(slot_name)
    }

    /// Set whether a slot is active (serving requests).
    pub async fn set_active(&self, slot_name: &str, active: bool) -> bool {
        let mut slots = self.slots.write().await;
        if let Some(slot) = slots.get_mut(slot_name) {
            slot.active = active;
            true
        } else {
            false
        }
    }

    /// List all slot names.
    pub async fn slot_names(&self) -> Vec<String> {
        let slots = self.slots.read().await;
        slots.keys().cloned().collect()
    }

    /// List all active slots.
    pub async fn active_slots(&self) -> Vec<PoolSlot> {
        let slots = self.slots.read().await;
        slots.values().filter(|s| s.active).cloned().collect()
    }

    /// Number of slots in the pool.
    pub async fn len(&self) -> usize {
        let slots = self.slots.read().await;
        slots.len()
    }

    /// Whether the pool is empty.
    pub async fn is_empty(&self) -> bool {
        let slots = self.slots.read().await;
        slots.is_empty()
    }

    /// Hot-swap a model in a slot: deactivate old, insert new, activate.
    ///
    /// Returns the old slot that was replaced, if any. The caller is
    /// responsible for unloading the old model from the backend after
    /// in-flight requests drain.
    pub async fn hot_swap(&self, slot_name: String, new_slot: PoolSlot) -> Option<PoolSlot> {
        let mut slots = self.slots.write().await;

        // Deactivate old slot if it exists
        let old = slots.remove(&slot_name);

        // Insert new slot as active
        let mut active_slot = new_slot;
        active_slot.active = true;
        slots.insert(slot_name, active_slot);

        old
    }
}

impl Default for ModelPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_slot(name: &str, model_id: ModelId) -> PoolSlot {
        PoolSlot {
            model_id,
            name: name.into(),
            handle: format!("handle-{name}"),
            active: true,
        }
    }

    #[tokio::test]
    async fn empty_pool() {
        let pool = ModelPool::new();
        assert!(pool.is_empty().await);
        assert_eq!(pool.len().await, 0);
        assert!(pool.get("any").await.is_none());
    }

    #[tokio::test]
    async fn put_and_get() {
        let pool = ModelPool::new();
        let id = uuid::Uuid::new_v4();
        let slot = make_slot("llama", id);

        let old = pool.put("primary".into(), slot).await;
        assert!(old.is_none());

        let got = pool.get("primary").await.unwrap();
        assert_eq!(got.model_id, id);
        assert_eq!(got.name, "llama");
        assert!(got.active);
    }

    #[tokio::test]
    async fn put_overwrites() {
        let pool = ModelPool::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();

        pool.put("slot".into(), make_slot("v1", id1)).await;
        let old = pool.put("slot".into(), make_slot("v2", id2)).await;

        assert!(old.is_some());
        assert_eq!(old.unwrap().model_id, id1);
        assert_eq!(pool.get("slot").await.unwrap().model_id, id2);
        assert_eq!(pool.len().await, 1);
    }

    #[tokio::test]
    async fn remove_slot() {
        let pool = ModelPool::new();
        let id = uuid::Uuid::new_v4();
        pool.put("slot".into(), make_slot("m", id)).await;

        let removed = pool.remove("slot").await;
        assert!(removed.is_some());
        assert!(pool.is_empty().await);
    }

    #[tokio::test]
    async fn remove_nonexistent() {
        let pool = ModelPool::new();
        assert!(pool.remove("nope").await.is_none());
    }

    #[tokio::test]
    async fn set_active() {
        let pool = ModelPool::new();
        let id = uuid::Uuid::new_v4();
        pool.put("slot".into(), make_slot("m", id)).await;

        assert!(pool.set_active("slot", false).await);
        assert!(!pool.get("slot").await.unwrap().active);

        assert!(pool.set_active("slot", true).await);
        assert!(pool.get("slot").await.unwrap().active);
    }

    #[tokio::test]
    async fn set_active_nonexistent() {
        let pool = ModelPool::new();
        assert!(!pool.set_active("nope", true).await);
    }

    #[tokio::test]
    async fn active_slots_filter() {
        let pool = ModelPool::new();
        pool.put("a".into(), make_slot("m1", uuid::Uuid::new_v4()))
            .await;
        pool.put("b".into(), make_slot("m2", uuid::Uuid::new_v4()))
            .await;
        pool.set_active("b", false).await;

        let active = pool.active_slots().await;
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].name, "m1");
    }

    #[tokio::test]
    async fn hot_swap() {
        let pool = ModelPool::new();
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();

        pool.put("primary".into(), make_slot("v1", id1)).await;
        let old = pool.hot_swap("primary".into(), make_slot("v2", id2)).await;

        assert_eq!(old.unwrap().model_id, id1);
        let current = pool.get("primary").await.unwrap();
        assert_eq!(current.model_id, id2);
        assert!(current.active);
    }

    #[tokio::test]
    async fn hot_swap_empty_slot() {
        let pool = ModelPool::new();
        let id = uuid::Uuid::new_v4();
        let old = pool.hot_swap("new-slot".into(), make_slot("m", id)).await;

        assert!(old.is_none());
        assert!(pool.get("new-slot").await.unwrap().active);
    }

    #[tokio::test]
    async fn slot_names() {
        let pool = ModelPool::new();
        pool.put("alpha".into(), make_slot("m1", uuid::Uuid::new_v4()))
            .await;
        pool.put("beta".into(), make_slot("m2", uuid::Uuid::new_v4()))
            .await;

        let mut names = pool.slot_names().await;
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[tokio::test]
    async fn concurrent_access() {
        let pool = Arc::new(ModelPool::new());
        let mut handles = vec![];

        for i in 0..20 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let id = uuid::Uuid::new_v4();
                pool.put(format!("slot-{i}"), make_slot(&format!("m-{i}"), id))
                    .await;
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(pool.len().await, 20);
    }
}
