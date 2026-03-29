-- PostgreSQL schema for ifran store traits.
-- All tables use TEXT for tenant_id to match the SQLite schema.
-- Complex objects stored as JSONB for flexibility.

CREATE TABLE IF NOT EXISTS models (
    id          UUID PRIMARY KEY,
    name        TEXT NOT NULL,
    data        JSONB NOT NULL,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_models_tenant ON models(tenant_id);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name, tenant_id);

CREATE TABLE IF NOT EXISTS tenants (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    api_key_hash    TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL,
    enabled         BOOLEAN NOT NULL DEFAULT true
);
CREATE INDEX IF NOT EXISTS idx_tenants_key ON tenants(api_key_hash);

CREATE TABLE IF NOT EXISTS training_jobs (
    id              UUID PRIMARY KEY,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    config_json     TEXT NOT NULL,
    status          TEXT NOT NULL,
    current_step    BIGINT NOT NULL DEFAULT 0,
    total_steps     BIGINT NOT NULL DEFAULT 0,
    current_epoch   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    current_loss    DOUBLE PRECISION,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_tenant ON training_jobs(tenant_id);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id              UUID NOT NULL,
    model_name          TEXT NOT NULL,
    benchmark           TEXT NOT NULL,
    score               DOUBLE PRECISION NOT NULL,
    details             TEXT,
    samples_evaluated   BIGINT NOT NULL,
    duration_secs       DOUBLE PRECISION NOT NULL,
    evaluated_at        TEXT NOT NULL,
    tenant_id           TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (run_id, benchmark)
);
CREATE INDEX IF NOT EXISTS idx_eval_model ON eval_results(model_name);
CREATE INDEX IF NOT EXISTS idx_eval_tenant ON eval_results(tenant_id);

CREATE TABLE IF NOT EXISTS preference_pairs (
    id           UUID PRIMARY KEY,
    tenant_id    TEXT NOT NULL DEFAULT 'default',
    prompt       TEXT NOT NULL,
    chosen       TEXT NOT NULL,
    rejected     TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'manual',
    score_delta  DOUBLE PRECISION,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prefs_tenant ON preference_pairs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_prefs_source ON preference_pairs(source);

CREATE TABLE IF NOT EXISTS marketplace_entries (
    model_name  TEXT NOT NULL,
    data        JSONB NOT NULL,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (model_name, tenant_id)
);
CREATE INDEX IF NOT EXISTS idx_marketplace_tenant ON marketplace_entries(tenant_id);

CREATE TABLE IF NOT EXISTS experiments (
    id          UUID PRIMARY KEY,
    name        TEXT NOT NULL,
    program     JSONB NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    best_trial  UUID,
    best_score  DOUBLE PRECISION,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_experiments_tenant ON experiments(tenant_id);

CREATE TABLE IF NOT EXISTS experiment_trials (
    id              UUID PRIMARY KEY,
    experiment_id   UUID NOT NULL REFERENCES experiments(id),
    params          JSONB NOT NULL,
    score           DOUBLE PRECISION,
    status          TEXT NOT NULL DEFAULT 'pending',
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error           TEXT,
    tenant_id       TEXT NOT NULL DEFAULT 'default'
);
CREATE INDEX IF NOT EXISTS idx_trials_experiment ON experiment_trials(experiment_id);

CREATE TABLE IF NOT EXISTS rag_pipelines (
    id          UUID PRIMARY KEY,
    config      JSONB NOT NULL,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rag_tenant ON rag_pipelines(tenant_id);

CREATE TABLE IF NOT EXISTS rag_documents (
    id          UUID PRIMARY KEY,
    pipeline_id UUID NOT NULL REFERENCES rag_pipelines(id),
    data        JSONB NOT NULL,
    tenant_id   TEXT NOT NULL DEFAULT 'default'
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id          UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES rag_documents(id),
    data        JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS annotation_sessions (
    id          UUID PRIMARY KEY,
    name        TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_annotations_tenant ON annotation_sessions(tenant_id);

CREATE TABLE IF NOT EXISTS annotation_pairs (
    id          UUID PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES annotation_sessions(id),
    data        JSONB NOT NULL,
    preference  TEXT,
    tenant_id   TEXT NOT NULL DEFAULT 'default'
);
CREATE INDEX IF NOT EXISTS idx_pairs_session ON annotation_pairs(session_id);

CREATE TABLE IF NOT EXISTS lineage_nodes (
    id              UUID PRIMARY KEY,
    data            JSONB NOT NULL,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    artifact_ref    TEXT
);
CREATE INDEX IF NOT EXISTS idx_lineage_tenant ON lineage_nodes(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lineage_artifact ON lineage_nodes(artifact_ref);

CREATE TABLE IF NOT EXISTS model_versions (
    id          UUID PRIMARY KEY,
    data        JSONB NOT NULL,
    family      TEXT NOT NULL,
    tag         TEXT,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    parent_id   UUID,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (family, tag, tenant_id)
);
CREATE INDEX IF NOT EXISTS idx_versions_family ON model_versions(family, tenant_id);
CREATE INDEX IF NOT EXISTS idx_versions_tenant ON model_versions(tenant_id);
