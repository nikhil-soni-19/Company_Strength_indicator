-- Neon DB schema for Agent 7 (PESTEL edition)
-- Only risk_factors (10-K embeddings) and environment_runs are persisted here.
-- All market data (OHLCV, fundamentals, metadata) is fetched live from yfinance.

CREATE EXTENSION IF NOT EXISTS vector;

-- 10-K risk factor chunks with pgvector embeddings
CREATE TABLE IF NOT EXISTS risk_factors (
    ticker       TEXT NOT NULL,
    fiscal_year  INT  NOT NULL,
    filing_date  DATE,
    chunk_id     INT  NOT NULL,
    chunk_text   TEXT,
    is_material  BOOLEAN DEFAULT FALSE,
    embedding    VECTOR(1536),
    PRIMARY KEY (ticker, fiscal_year, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_rf_ticker ON risk_factors (ticker, fiscal_year);

-- Agent run results (PESTEL edition)
CREATE TABLE IF NOT EXISTS environment_runs (
    run_id            UUID PRIMARY KEY,
    ticker            TEXT NOT NULL,
    as_of_date        DATE NOT NULL,
    layer1_bundle     JSONB,
    quant_score       NUMERIC,
    qual_score        NUMERIC,
    environment_score NUMERIC,
    direction         TEXT,
    flags             TEXT[],
    narrative         TEXT,
    -- PESTEL dimension scores {P, E, S, T, En, L} each with {quant, qual, combined}
    pestel_scores     JSONB,
    created_at        TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_env_runs ON environment_runs (ticker, as_of_date DESC);

-- Migration for existing deployments (idempotent):
-- ALTER TABLE environment_runs ADD COLUMN IF NOT EXISTS pestel_scores JSONB;
