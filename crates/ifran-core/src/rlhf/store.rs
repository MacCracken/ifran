use rusqlite::{Connection, params};
use ifran_types::IfranError;
use ifran_types::TenantId;
use ifran_types::error::Result;
use ifran_types::rlhf::*;
use uuid::Uuid;

pub struct AnnotationStore {
    conn: Connection,
}

impl AnnotationStore {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(|e| IfranError::StorageError(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| IfranError::StorageError(e.to_string()))?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS annotation_sessions (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                model_name  TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active'
            );
            CREATE TABLE IF NOT EXISTS annotation_pairs (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                prompt      TEXT NOT NULL,
                response_a  TEXT NOT NULL,
                response_b  TEXT NOT NULL,
                preference  TEXT,
                annotated_at TEXT,
                FOREIGN KEY (session_id) REFERENCES annotation_sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_pairs_session ON annotation_pairs(session_id);",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        // Add tenant_id column to annotation_sessions (idempotent — ignore if already exists)
        let _ = self.conn.execute_batch(
            "ALTER TABLE annotation_sessions ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default';",
        );
        self.conn
            .execute_batch(
                "CREATE INDEX IF NOT EXISTS idx_annotation_sessions_tenant ON annotation_sessions(tenant_id);",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(())
    }

    pub fn create_session(
        &self,
        name: &str,
        model_name: &str,
        tenant_id: &TenantId,
    ) -> Result<AnnotationSession> {
        let id = Uuid::new_v4();
        let now = chrono::Utc::now();
        self.conn.execute(
            "INSERT INTO annotation_sessions (id, name, model_name, created_at, status, tenant_id) VALUES (?1, ?2, ?3, ?4, 'active', ?5)",
            params![id.to_string(), name, model_name, now.to_rfc3339(), tenant_id.0],
        ).map_err(|e| IfranError::StorageError(e.to_string()))?;
        Ok(AnnotationSession {
            id,
            name: name.to_string(),
            model_name: model_name.to_string(),
            created_at: now,
            status: AnnotationSessionStatus::Active,
        })
    }

    pub fn get_session(&self, id: Uuid, tenant_id: &TenantId) -> Result<AnnotationSession> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, model_name, created_at, status FROM annotation_sessions WHERE id = ?1 AND tenant_id = ?2"
        ).map_err(|e| IfranError::StorageError(e.to_string()))?;

        stmt.query_row(params![id.to_string(), tenant_id.0], |row| {
            let id_str: String = row.get(0)?;
            let name: String = row.get(1)?;
            let model_name: String = row.get(2)?;
            let created_at_str: String = row.get(3)?;
            let status_str: String = row.get(4)?;

            let id = Uuid::parse_str(&id_str).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
            let status: AnnotationSessionStatus =
                serde_json::from_str(&format!("\"{status_str}\"")).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        4,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;

            Ok(AnnotationSession {
                id,
                name,
                model_name,
                created_at,
                status,
            })
        })
        .map_err(|e| IfranError::StorageError(e.to_string()))
    }

    pub fn list_sessions(&self, tenant_id: &TenantId) -> Result<Vec<AnnotationSession>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, model_name, created_at, status FROM annotation_sessions WHERE tenant_id = ?1 ORDER BY created_at DESC"
        ).map_err(|e| IfranError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map(params![tenant_id.0], |row| {
                let id_str: String = row.get(0)?;
                let name: String = row.get(1)?;
                let model_name: String = row.get(2)?;
                let created_at_str: String = row.get(3)?;
                let status_str: String = row.get(4)?;

                let id = Uuid::parse_str(&id_str).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
                let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            3,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?;
                let status: AnnotationSessionStatus =
                    serde_json::from_str(&format!("\"{status_str}\"")).map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            4,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?;

                Ok(AnnotationSession {
                    id,
                    name,
                    model_name,
                    created_at,
                    status,
                })
            })
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(results)
    }

    pub fn add_pairs(&self, pairs: &[AnnotationPair]) -> Result<()> {
        for pair in pairs {
            self.conn.execute(
                "INSERT INTO annotation_pairs (id, session_id, prompt, response_a, response_b, preference, annotated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    pair.id.to_string(),
                    pair.session_id.to_string(),
                    pair.prompt,
                    pair.response_a,
                    pair.response_b,
                    pair.preference.map(|p| {
                        serde_json::to_string(&p).unwrap().trim_matches('"').to_string()
                    }),
                    pair.annotated_at.map(|t| t.to_rfc3339()),
                ],
            ).map_err(|e| IfranError::StorageError(e.to_string()))?;
        }
        Ok(())
    }

    pub fn get_pairs(&self, session_id: Uuid, tenant_id: &TenantId) -> Result<Vec<AnnotationPair>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT p.id, p.session_id, p.prompt, p.response_a, p.response_b, p.preference, p.annotated_at
             FROM annotation_pairs p
             JOIN annotation_sessions s ON p.session_id = s.id
             WHERE p.session_id = ?1 AND s.tenant_id = ?2",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map(params![session_id.to_string(), tenant_id.0], row_to_pair)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(results)
    }

    pub fn get_next_unannotated(
        &self,
        session_id: Uuid,
        tenant_id: &TenantId,
    ) -> Result<Option<AnnotationPair>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT p.id, p.session_id, p.prompt, p.response_a, p.response_b, p.preference, p.annotated_at
             FROM annotation_pairs p
             JOIN annotation_sessions s ON p.session_id = s.id
             WHERE p.session_id = ?1 AND s.tenant_id = ?2 AND p.preference IS NULL LIMIT 1",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![session_id.to_string(), tenant_id.0], row_to_pair)
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        match rows.next() {
            Some(Ok(pair)) => Ok(Some(pair)),
            Some(Err(e)) => Err(IfranError::StorageError(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn annotate_pair(&self, pair_id: Uuid, preference: Preference) -> Result<()> {
        let pref_str = serde_json::to_string(&preference)
            .unwrap()
            .trim_matches('"')
            .to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let affected = self
            .conn
            .execute(
                "UPDATE annotation_pairs SET preference = ?1, annotated_at = ?2 WHERE id = ?3",
                params![pref_str, now, pair_id.to_string()],
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;
        if affected == 0 {
            return Err(IfranError::StorageError(format!(
                "Pair {pair_id} not found"
            )));
        }
        Ok(())
    }

    pub fn get_stats(&self, session_id: Uuid, tenant_id: &TenantId) -> Result<AnnotationStats> {
        // Verify session belongs to tenant
        let _ = self.get_session(session_id, tenant_id)?;

        let total: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotation_pairs WHERE session_id = ?1",
                params![session_id.to_string()],
                |row| row.get(0),
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let annotated: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM annotation_pairs WHERE session_id = ?1 AND preference IS NOT NULL",
            params![session_id.to_string()],
            |row| row.get(0),
        ).map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(AnnotationStats {
            session_id,
            total_pairs: total as usize,
            annotated_count: annotated as usize,
            remaining: (total - annotated) as usize,
        })
    }

    pub fn export_session(
        &self,
        session_id: Uuid,
        tenant_id: &TenantId,
    ) -> Result<Vec<AnnotationPair>> {
        // Verify session belongs to tenant
        let _ = self.get_session(session_id, tenant_id)?;

        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, session_id, prompt, response_a, response_b, preference, annotated_at
             FROM annotation_pairs WHERE session_id = ?1 AND preference IS NOT NULL",
            )
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        let results = stmt
            .query_map(params![session_id.to_string()], row_to_pair)
            .map_err(|e| IfranError::StorageError(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| IfranError::StorageError(e.to_string()))?;

        Ok(results)
    }
}

fn row_to_pair(row: &rusqlite::Row) -> rusqlite::Result<AnnotationPair> {
    let id_str: String = row.get(0)?;
    let session_id_str: String = row.get(1)?;
    let prompt: String = row.get(2)?;
    let response_a: String = row.get(3)?;
    let response_b: String = row.get(4)?;
    let pref_str: Option<String> = row.get(5)?;
    let annotated_str: Option<String> = row.get(6)?;

    let id = Uuid::parse_str(&id_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let session_id = Uuid::parse_str(&session_id_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(1, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let preference = pref_str.map(|s| {
        serde_json::from_str::<Preference>(&format!("\"{s}\"")).unwrap_or(Preference::Tie)
    });
    let annotated_at = annotated_str.and_then(|s| {
        chrono::DateTime::parse_from_rfc3339(&s)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .ok()
    });

    Ok(AnnotationPair {
        id,
        session_id,
        prompt,
        response_a,
        response_b,
        preference,
        annotated_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_get_session() {
        let store = AnnotationStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let session = store.create_session("test", "llama-8b", &tenant).unwrap();
        assert_eq!(session.name, "test");
        assert_eq!(session.model_name, "llama-8b");
        assert_eq!(session.status, AnnotationSessionStatus::Active);

        let got = store.get_session(session.id, &tenant).unwrap();
        assert_eq!(got.name, "test");
    }

    #[test]
    fn list_sessions_empty() {
        let store = AnnotationStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let sessions = store.list_sessions(&tenant).unwrap();
        assert!(sessions.is_empty());
    }

    #[test]
    fn list_sessions_multiple() {
        let store = AnnotationStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        store.create_session("s1", "model-a", &tenant).unwrap();
        store.create_session("s2", "model-b", &tenant).unwrap();
        let sessions = store.list_sessions(&tenant).unwrap();
        assert_eq!(sessions.len(), 2);
    }

    #[test]
    fn add_and_get_pairs() {
        let store = AnnotationStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let session = store.create_session("test", "model", &tenant).unwrap();
        let pair = AnnotationPair {
            id: Uuid::new_v4(),
            session_id: session.id,
            prompt: "What is Rust?".into(),
            response_a: "A language".into(),
            response_b: "A systems lang".into(),
            preference: None,
            annotated_at: None,
        };
        store.add_pairs(std::slice::from_ref(&pair)).unwrap();
        let pairs = store.get_pairs(session.id, &tenant).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].prompt, "What is Rust?");
        assert!(pairs[0].preference.is_none());
    }

    #[test]
    fn annotate_pair_and_stats() {
        let store = AnnotationStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let session = store.create_session("test", "model", &tenant).unwrap();
        let pair1 = AnnotationPair {
            id: Uuid::new_v4(),
            session_id: session.id,
            prompt: "q1".into(),
            response_a: "a1".into(),
            response_b: "b1".into(),
            preference: None,
            annotated_at: None,
        };
        let pair2 = AnnotationPair {
            id: Uuid::new_v4(),
            session_id: session.id,
            prompt: "q2".into(),
            response_a: "a2".into(),
            response_b: "b2".into(),
            preference: None,
            annotated_at: None,
        };
        store.add_pairs(&[pair1.clone(), pair2]).unwrap();

        // Stats before annotation
        let stats = store.get_stats(session.id, &tenant).unwrap();
        assert_eq!(stats.total_pairs, 2);
        assert_eq!(stats.annotated_count, 0);
        assert_eq!(stats.remaining, 2);

        // Annotate one
        store
            .annotate_pair(pair1.id, Preference::ResponseA)
            .unwrap();

        let stats = store.get_stats(session.id, &tenant).unwrap();
        assert_eq!(stats.annotated_count, 1);
        assert_eq!(stats.remaining, 1);
    }

    #[test]
    fn get_next_unannotated() {
        let store = AnnotationStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let session = store.create_session("test", "model", &tenant).unwrap();

        // No pairs yet
        assert!(
            store
                .get_next_unannotated(session.id, &tenant)
                .unwrap()
                .is_none()
        );

        let pair = AnnotationPair {
            id: Uuid::new_v4(),
            session_id: session.id,
            prompt: "q".into(),
            response_a: "a".into(),
            response_b: "b".into(),
            preference: None,
            annotated_at: None,
        };
        store.add_pairs(std::slice::from_ref(&pair)).unwrap();

        let next = store.get_next_unannotated(session.id, &tenant).unwrap();
        assert!(next.is_some());

        store.annotate_pair(pair.id, Preference::Tie).unwrap();
        assert!(
            store
                .get_next_unannotated(session.id, &tenant)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn export_session_only_annotated() {
        let store = AnnotationStore::open_in_memory().unwrap();
        let tenant = TenantId::default_tenant();
        let session = store.create_session("test", "model", &tenant).unwrap();
        let pair1 = AnnotationPair {
            id: Uuid::new_v4(),
            session_id: session.id,
            prompt: "q1".into(),
            response_a: "a1".into(),
            response_b: "b1".into(),
            preference: None,
            annotated_at: None,
        };
        let pair2 = AnnotationPair {
            id: Uuid::new_v4(),
            session_id: session.id,
            prompt: "q2".into(),
            response_a: "a2".into(),
            response_b: "b2".into(),
            preference: None,
            annotated_at: None,
        };
        store.add_pairs(&[pair1.clone(), pair2]).unwrap();
        store
            .annotate_pair(pair1.id, Preference::ResponseB)
            .unwrap();

        let exported = store.export_session(session.id, &tenant).unwrap();
        assert_eq!(exported.len(), 1);
        assert_eq!(exported[0].preference, Some(Preference::ResponseB));
    }

    #[test]
    fn annotate_nonexistent_pair_fails() {
        let store = AnnotationStore::open_in_memory().unwrap();
        let result = store.annotate_pair(Uuid::new_v4(), Preference::Tie);
        assert!(result.is_err());
    }
}
