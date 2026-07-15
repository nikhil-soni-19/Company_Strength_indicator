"""Retrieve relevant 10-K risk factor excerpts.

Primary path: ontology-backed hybrid search (BGE 1024-dim + BM25, csi_ontology_lab).
Fallback: local agent7 risk_factors table (pgvector 1536-dim / tsvector FTS).

The primary path activates when DATABASE_URL_ONTOLOGY_LAB is set; if it is not
configured, or the filing is not yet ingested, the function falls back silently to
the local table.  Callers see no difference — always a list[str].
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.connection import get_conn, pgvector_available


def retrieve(
    ticker: str,
    fiscal_year: int,
    query: str,
    k: int = 5,
) -> list[str]:
    """
    Return up to k material risk-factor chunk texts for (ticker, fiscal_year).

    Tries the ontology-backed retrieval first (ten_k_retrieval.py).
    Falls back to the local risk_factors table if ontology is unavailable or empty.
    """
    # ── Primary: ontology-backed retrieval ───────────────────────────────────
    try:
        from layer2.ten_k_retrieval import retrieve_risk_factors
        result = retrieve_risk_factors(ticker, fiscal_year, query=query, k=k)
        if result:
            return result
    except Exception as e:
        print(f"  [RiskFactors] Ontology retrieval error, falling back: {e}")

    # ── Fallback: local risk_factors table ───────────────────────────────────
    return _retrieve_local(ticker, fiscal_year, query, k)


def _embed_query(query: str) -> list[float] | None:
    """Embed a query string using old MiniLM model (local table only)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = model.encode([query], show_progress_bar=False)[0].tolist()
        if len(vec) < 1536:
            vec = vec + [0.0] * (1536 - len(vec))
        return vec[:1536]
    except Exception:
        return None


def _retrieve_local(
    ticker: str,
    fiscal_year: int,
    query: str,
    k: int = 5,
) -> list[str]:
    """Query agent7's own risk_factors table — original fallback logic."""
    try:
        conn = get_conn()
    except Exception as e:
        print(f"  [RiskFactors] Local DB not available, skipping: {e}")
        return []
    try:
        if pgvector_available():
            vec = _embed_query(query)
            if vec is not None:
                vec_str = f"[{','.join(str(x) for x in vec)}]"
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT chunk_text FROM risk_factors
                        WHERE ticker = %s AND fiscal_year = %s AND is_material = TRUE
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (ticker, fiscal_year, vec_str, k),
                    )
                    rows = cur.fetchall()
                    if rows:
                        return [r["chunk_text"] for r in rows]

        # Fallback: full-text search
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_text FROM risk_factors
                WHERE ticker = %s AND fiscal_year = %s AND is_material = TRUE
                  AND to_tsvector('english', chunk_text) @@ plainto_tsquery('english', %s)
                LIMIT %s
                """,
                (ticker, fiscal_year, query, k),
            )
            rows = cur.fetchall()
            if rows:
                return [r["chunk_text"] for r in rows]

        # Last resort: return any material chunks
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_text FROM risk_factors
                WHERE ticker = %s AND fiscal_year = %s AND is_material = TRUE
                LIMIT %s
                """,
                (ticker, fiscal_year, k),
            )
            rows = cur.fetchall()
            return [r["chunk_text"] for r in rows]

    finally:
        conn.close()
