"""Hybrid BM25 + vector search over ontology.narrative_chunks.

Combines BM25 full-text ranking (tsv column) with BGE 1024-dim vector similarity,
fused via reciprocal rank fusion.

Key rules:
  - query_embedding must be 1024-dim BGE vectors, normalize_embeddings=True
  - BM25 uses websearch_to_tsquery('english', ...) — matches the GIN index config
  - Vector leg sets hnsw.ef_search=40
  - When filing_id is set it is the authoritative filter
  - Section filter drops and retries without if 0 results
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from retrieval.connection import get_ontology_conn
from retrieval.rrf import reciprocal_rank_fusion


def hybrid_search(
    query_text: str,
    query_embedding: List[float],
    *,
    filing_id: Optional[int] = None,
    ticker: Optional[str] = None,
    section: Optional[str] = None,
    doc_type: str = "10-K",
    top_k: int = 5,
    leg_k: int = 50,
) -> List[Dict[str, Any]]:
    """
    BM25 + vector RRF over ontology.narrative_chunks.

    Args:
        query_text:       Full-text search string (BM25 leg).
        query_embedding:  1024-dim BGE query vector (vector leg).
        filing_id:        Pin to one specific filing (authoritative).
        ticker:           Ticker filter (used when filing_id is None).
        section:          Section filter; drops and retries without if 0 rows.
        doc_type:         Filing type filter, default '10-K'.
        top_k:            Number of chunks to return.
        leg_k:            Candidate pool per retrieval leg.

    Returns:
        List of chunk dicts: id, chunk_text, chunk_index, filing_id,
        source_pdf, ticker, doc_type, period_end_date, section, relevance
    """
    try:
        conn = get_ontology_conn()
    except Exception as e:
        print(f"  [HybridSearch] Ontology DB not available: {e}")
        return []

    try:
        cur = conn.cursor()
        result = _search(
            cur, query_text, query_embedding,
            filing_id=filing_id, ticker=ticker,
            section=section, doc_type=doc_type,
            top_k=top_k, leg_k=leg_k,
        )
        if not result and section:
            print(f"  [HybridSearch] section='{section}' returned 0 — retrying without")
            result = _search(
                cur, query_text, query_embedding,
                filing_id=filing_id, ticker=ticker,
                section=None, doc_type=doc_type,
                top_k=top_k, leg_k=leg_k,
            )
        return result
    except Exception as e:
        print(f"  [HybridSearch] Search failed: {e}")
        return []
    finally:
        conn.close()


def _build_where(
    *,
    filing_id: Optional[int],
    ticker: Optional[str],
    section: Optional[str],
    doc_type: Optional[str],
    params: Dict[str, Any],
) -> str:
    clauses = ["1=1"]
    if filing_id is not None:
        clauses.append("nc.filing_id = %(filing_id)s")
        params["filing_id"] = filing_id
    else:
        if ticker:
            clauses.append(
                "EXISTS (SELECT 1 FROM ontology.filings f2 "
                "WHERE f2.filing_id = nc.filing_id "
                "AND UPPER(COALESCE(f2.canonical_ticker, f2.ticker)) = UPPER(%(ticker)s))"
            )
            params["ticker"] = ticker
        if doc_type:
            clauses.append(
                "EXISTS (SELECT 1 FROM ontology.filings f3 "
                "WHERE f3.filing_id = nc.filing_id "
                "AND LOWER(f3.filing_type) = LOWER(%(doc_type)s))"
            )
            params["doc_type"] = doc_type
    if section:
        clauses.append("LOWER(COALESCE(nc.section, '')) = LOWER(%(section)s)")
        params["section"] = section
    return " AND ".join(clauses)


def _fetch_hydrated(cur, doc_ids: List[int], qv_literal: str) -> List[Dict[str, Any]]:
    if not doc_ids:
        return []
    cur.execute(
        """
        SELECT
            nc.id,
            nc.content,
            nc.chunk_index,
            nc.section,
            nc.filing_id,
            f.source_pdf,
            COALESCE(f.canonical_ticker, f.ticker) AS canonical_ticker,
            f.filing_type                           AS doc_type,
            f.period_end_date,
            1 - (nc.embedding <=> %(qv)s::vector)   AS vec_similarity
        FROM ontology.narrative_chunks nc
        JOIN ontology.filings f ON f.filing_id = nc.filing_id
        WHERE nc.id = ANY(%(ids)s)
        """,
        {"ids": doc_ids, "qv": qv_literal},
    )
    return [dict(r) for r in cur.fetchall()]


def _search(
    cur,
    query_text: str,
    query_embedding: List[float],
    *,
    filing_id: Optional[int],
    ticker: Optional[str],
    section: Optional[str],
    doc_type: Optional[str],
    top_k: int,
    leg_k: int,
) -> List[Dict[str, Any]]:
    qv = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
    params: Dict[str, Any] = {"leg_k": leg_k, "qv": qv}
    where = _build_where(
        filing_id=filing_id, ticker=ticker,
        section=section, doc_type=doc_type,
        params=params,
    )
    q_clean = (query_text or "").strip()
    conn = cur.connection

    bm25_ranks: Dict[int, int] = {}
    if q_clean:
        params["q"] = q_clean
        try:
            cur.execute(
                f"""
                WITH scoped AS (
                    SELECT nc.id FROM ontology.narrative_chunks nc WHERE {where}
                ),
                query AS (
                    SELECT websearch_to_tsquery('english', %(q)s) AS q
                )
                SELECT nc.id AS doc_id,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(nc.tsv, query.q) DESC) AS rank
                FROM ontology.narrative_chunks nc
                INNER JOIN scoped s ON s.id = nc.id
                CROSS JOIN query
                WHERE nc.tsv @@ query.q
                LIMIT %(leg_k)s
                """,
                params,
            )
            bm25_ranks = {int(r["doc_id"]): int(r["rank"]) for r in cur.fetchall()}
        except Exception as e:
            print(f"  [HybridSearch] BM25 leg failed: {e}")
            try:
                conn.rollback()
            except Exception:
                pass

    vec_ranks: Dict[int, int] = {}
    try:
        cur.execute("SET LOCAL hnsw.ef_search = 40")
        cur.execute(
            f"""
            WITH scoped AS (
                SELECT nc.id FROM ontology.narrative_chunks nc WHERE {where}
            )
            SELECT nc.id AS doc_id,
                   ROW_NUMBER() OVER (ORDER BY nc.embedding <=> %(qv)s::vector) AS rank
            FROM ontology.narrative_chunks nc
            INNER JOIN scoped s ON s.id = nc.id
            LIMIT %(leg_k)s
            """,
            params,
        )
        vec_ranks = {int(r["doc_id"]): int(r["rank"]) for r in cur.fetchall()}
    except Exception as e:
        print(f"  [HybridSearch] Vector leg failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass

    if not bm25_ranks and not vec_ranks:
        return []

    rank_lists = [r for r in (bm25_ranks, vec_ranks) if r]
    fused = reciprocal_rank_fusion(rank_lists, k=60)
    top_ids = [doc_id for doc_id, _ in fused[: top_k * 2]]
    raw_rows = _fetch_hydrated(cur, top_ids, qv)
    rows_by_id = {r["id"]: r for r in raw_rows}

    result = []
    for doc_id, rrf_score in fused[:top_k]:
        row = rows_by_id.get(doc_id)
        if row:
            result.append({
                "id":           row.get("id"),
                "chunk_text":   row.get("content") or row.get("chunk_text") or "",
                "chunk_index":  row.get("chunk_index"),
                "filing_id":    row.get("filing_id"),
                "source_pdf":   row.get("source_pdf"),
                "ticker":       row.get("canonical_ticker") or row.get("company_symbol"),
                "doc_type":     row.get("doc_type") or row.get("document_type"),
                "period_end_date": row.get("period_end_date"),
                "section":      row.get("section"),
                "relevance":    rrf_score,
            })
    return result
