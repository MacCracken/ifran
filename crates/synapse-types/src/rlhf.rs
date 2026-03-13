use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type AnnotationId = Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnnotationSessionStatus {
    Active,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationSession {
    pub id: Uuid,
    pub name: String,
    pub model_name: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub status: AnnotationSessionStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Preference {
    ResponseA,
    ResponseB,
    Tie,
    BothBad,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationPair {
    pub id: AnnotationId,
    pub session_id: Uuid,
    pub prompt: String,
    pub response_a: String,
    pub response_b: String,
    pub preference: Option<Preference>,
    pub annotated_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationStats {
    pub session_id: Uuid,
    pub total_pairs: usize,
    pub annotated_count: usize,
    pub remaining: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationExport {
    pub session_id: Uuid,
    pub pairs: Vec<AnnotationPair>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_status_serde_roundtrip() {
        let statuses = [
            AnnotationSessionStatus::Active,
            AnnotationSessionStatus::Completed,
        ];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let back: AnnotationSessionStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*s, back);
        }
    }

    #[test]
    fn session_status_json_values() {
        assert_eq!(
            serde_json::to_string(&AnnotationSessionStatus::Active).unwrap(),
            "\"active\""
        );
        assert_eq!(
            serde_json::to_string(&AnnotationSessionStatus::Completed).unwrap(),
            "\"completed\""
        );
    }

    #[test]
    fn preference_serde_roundtrip() {
        let prefs = [
            Preference::ResponseA,
            Preference::ResponseB,
            Preference::Tie,
            Preference::BothBad,
        ];
        for p in &prefs {
            let json = serde_json::to_string(p).unwrap();
            let back: Preference = serde_json::from_str(&json).unwrap();
            assert_eq!(*p, back);
        }
    }

    #[test]
    fn preference_json_values() {
        assert_eq!(
            serde_json::to_string(&Preference::ResponseA).unwrap(),
            "\"response_a\""
        );
        assert_eq!(
            serde_json::to_string(&Preference::ResponseB).unwrap(),
            "\"response_b\""
        );
        assert_eq!(serde_json::to_string(&Preference::Tie).unwrap(), "\"tie\"");
        assert_eq!(
            serde_json::to_string(&Preference::BothBad).unwrap(),
            "\"both_bad\""
        );
    }

    #[test]
    fn annotation_session_serde_roundtrip() {
        let session = AnnotationSession {
            id: Uuid::new_v4(),
            name: "test-session".into(),
            model_name: "llama-8b".into(),
            created_at: chrono::Utc::now(),
            status: AnnotationSessionStatus::Active,
        };
        let json = serde_json::to_string(&session).unwrap();
        let back: AnnotationSession = serde_json::from_str(&json).unwrap();
        assert_eq!(session.id, back.id);
        assert_eq!(session.name, back.name);
        assert_eq!(session.model_name, back.model_name);
    }

    #[test]
    fn annotation_pair_serde_roundtrip() {
        let pair = AnnotationPair {
            id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            prompt: "What is Rust?".into(),
            response_a: "A programming language".into(),
            response_b: "A systems language".into(),
            preference: Some(Preference::ResponseA),
            annotated_at: Some(chrono::Utc::now()),
        };
        let json = serde_json::to_string(&pair).unwrap();
        let back: AnnotationPair = serde_json::from_str(&json).unwrap();
        assert_eq!(pair.id, back.id);
        assert_eq!(pair.prompt, back.prompt);
        assert_eq!(pair.preference, back.preference);
    }

    #[test]
    fn annotation_pair_unannotated() {
        let pair = AnnotationPair {
            id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            prompt: "test".into(),
            response_a: "a".into(),
            response_b: "b".into(),
            preference: None,
            annotated_at: None,
        };
        let json = serde_json::to_string(&pair).unwrap();
        let back: AnnotationPair = serde_json::from_str(&json).unwrap();
        assert!(back.preference.is_none());
        assert!(back.annotated_at.is_none());
    }

    #[test]
    fn annotation_stats_serde_roundtrip() {
        let stats = AnnotationStats {
            session_id: Uuid::new_v4(),
            total_pairs: 100,
            annotated_count: 42,
            remaining: 58,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let back: AnnotationStats = serde_json::from_str(&json).unwrap();
        assert_eq!(back.total_pairs, 100);
        assert_eq!(back.annotated_count, 42);
        assert_eq!(back.remaining, 58);
    }

    #[test]
    fn annotation_export_serde_roundtrip() {
        let export = AnnotationExport {
            session_id: Uuid::new_v4(),
            pairs: vec![],
        };
        let json = serde_json::to_string(&export).unwrap();
        let back: AnnotationExport = serde_json::from_str(&json).unwrap();
        assert!(back.pairs.is_empty());
    }
}
