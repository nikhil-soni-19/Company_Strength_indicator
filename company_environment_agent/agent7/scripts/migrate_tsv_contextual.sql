-- ============================================================
-- Migration: make BM25 contextual (Section 5 of retrieval CONTRACT)
-- ============================================================
-- What:  The tsv generated column currently indexes only the chunk
--        body (text). This migration folds in the situating blurb
--        (context) so BM25 gets the same contextual signal that the
--        dense embeddings already have.
--
-- Why:   Ingestion stored embeddings as embed(context + "\n\n" + text).
--        Dense search is already contextual. BM25 is not — it misses
--        terms that appear only in the blurb. This fixes that gap.
--
-- Impact: No re-ingestion, no LLM calls, vectors untouched.
--         Postgres rewrites all 28,551 rows of the generated column.
--         On Neon this takes ~30–90 seconds.
--
-- Safety: COALESCE(context,'') degrades gracefully for summary rows
--         where context is NULL — they get the same tsv as before.
--
-- Run:   psql $DATABASE_URL_ONTOLOGY_LAB -f migrate_tsv_contextual.sql
-- ============================================================

BEGIN;

-- Step 1: Drop the view that depends on sec_filings.tsv.
--         narrative_summaries has no tsv column so it is unaffected.
--         Capture the current view definition first if you want to
--         inspect it:
--           SELECT pg_get_viewdef('ontology.narrative_chunks'::regclass, true);
DROP VIEW IF EXISTS ontology.narrative_chunks;

-- Step 2: Swap the generated column.
--         DROP also removes the idx_sec_filings_tsv GIN index automatically.
ALTER TABLE ontology.sec_filings DROP COLUMN IF EXISTS tsv;

ALTER TABLE ontology.sec_filings
    ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english',
            COALESCE(context, '') || ' ' || COALESCE(text, ''))
    ) STORED;

-- Step 3: Recreate the GIN index (dropped with the column in step 2).
CREATE INDEX idx_sec_filings_tsv
    ON ontology.sec_filings
    USING gin (tsv);

-- Step 4: Recreate the narrative_chunks view.
--         This is the view definition verified against the live DB
--         on 2026-06-18. It filters out parser-failure "apology" chunks
--         (those are rows where the body is an LLM apology string).
--         The view exposes tsv from the base table, so it now carries
--         the contextual blurb automatically.
CREATE VIEW ontology.narrative_chunks AS
SELECT
    sf.id,
    sf.chunk_id,
    sf.filing_id,
    sf.section,
    (sf.metadata->>'block_index')::int  AS block_index,
    sf.chunk_index,
    sf.text                             AS content,
    NULL::text                          AS contextualized,   -- vestigial alias, always NULL
    sf.embedding,
    sf.embedding_model,
    sf.metadata,
    sf.turn_id,
    sf.tsv,
    sf.parent_summary_id
FROM ontology.sec_filings sf
WHERE sf.level = 0
  AND sf.text IS NOT NULL
  AND sf.text NOT ILIKE '%I apologize%'
  AND sf.text NOT ILIKE '%unable to provide%'
  AND length(sf.text) > 50;

COMMIT;

-- Step 5 (run separately — not in transaction): update planner statistics.
ANALYZE ontology.sec_filings;

-- ============================================================
-- Verification queries (run after migration):
-- ============================================================

-- 1. Confirm tsv expression now includes context:
--    Should contain "context" in the expression string.
-- SELECT generation_expression
-- FROM information_schema.columns
-- WHERE table_schema = 'ontology'
--   AND table_name   = 'sec_filings'
--   AND column_name  = 'tsv';

-- 2. Confirm GIN index is back:
-- SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_sec_filings_tsv';

-- 3. Confirm view is recreated and row count matches (~25,973):
-- SELECT count(*) FROM ontology.narrative_chunks;

-- 4. Smoke test — a blurb-only phrase should now return rows:
--    Replace '<blurb phrase>' with a term you know is only in a context blurb.
-- SELECT count(*)
-- FROM ontology.narrative_chunks
-- WHERE tsv @@ websearch_to_tsquery('english', '<blurb phrase>');
