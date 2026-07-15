"""
retrieval.py — reusable spine-scoped hybrid retriever over ontology.sec_filings.

Replaces the per-agent ad-hoc ``ILIKE`` keyword search. Implements the
Retrieval & Routing design (§5), *adapted to the real Neon schema* audited
2026-05-19:

  - RAPTOR nodes live in ONE table ``ontology.sec_filings`` with a ``level``
    column (0 = leaf, 1-3 = summaries) and ``children text[]`` parent→child
    links — NOT the doc's separate narrative_chunks/narrative_summaries.
  - ``section`` exists and is clean (general/revenue/cash_and_capital/
    risk_factors/financials) → usable as a router filter.
  - No tsvector column / GIN index → BM25 leg uses on-the-fly
    ``to_tsvector`` + ``websearch_to_tsquery``. At ~2.2k leaf rows a seq
    scan is trivially fast, so no ingest migration is required for v1.
  - ``embedding`` is vector(1024). The vector leg embeds the query with
    BAAI/bge-large-en-v1.5 (settings.embedding_model_neon — the model that
    produced the stored vectors) and pgvector-searches it, fused with BM25
    via RRF. A runtime dim guard disables the leg (BM25-only) rather than
    querying with an incompatible vector.

Spine-first: everything is scoped to (ticker, period_end_date) via
``ontology.filings`` before any search.

Public API
----------
    rows = await retrieve_filing_evidence(
        "AAPL",
        query="services revenue growth drivers",
        hints=RetrievalHints(section=["revenue"]),
    )

Each returned row is a dict shaped to be a drop-in for the old chunk dicts
(keeps ``context`` / ``period_end_date`` / ``fiscal_year`` / ``fiscal_quarter``
keys) plus retrieval metadata (``chunk_id``, ``level``, ``section``,
``score``, ``source``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from sqlalchemy import text

import neon_reader
from neon_connection import get_neon_session, is_neon_available
from logger import get_logger

logger = get_logger(__name__)

# RRF constant (doc §5.3). k=60 is the standard default.
_RRF_K = 60

# Vector leg: the model that produced the stored 1024-dim Neon embeddings,
# documented in settings.py (`embedding_model_neon`, "1024-dim, matches Neon").
_NEON_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
_NEON_EMBED_DIM = 1024
# bge v1.x asymmetric retrieval: the query (not the passage) gets this prefix.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_bge_model = None  # lazy singleton (heavy ~1.3GB; loaded on first vector query)


def _get_bge():
    """Lazy-load the Neon-matching embedding model (cached). Returns None on
    failure so the vector leg degrades to BM25-only rather than crashing."""
    global _bge_model
    if _bge_model is not None:
        return _bge_model
    try:
        from settings import settings
        name = getattr(settings.llm, "embedding_model_neon", _NEON_EMBED_MODEL)
    except Exception:
        name = _NEON_EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading Neon-matching embedder: {name}")
        _bge_model = SentenceTransformer(name)
        dim = _bge_model.get_sentence_embedding_dimension()
        if dim != _NEON_EMBED_DIM:
            logger.warning(
                f"Embedder {name} is {dim}-dim, Neon expects "
                f"{_NEON_EMBED_DIM} — disabling vector leg (BM25 only)."
            )
            _bge_model = False  # sentinel: tried and unusable
        else:
            logger.info(f"Embedder loaded ({dim}-dim).")
    except Exception as e:
        logger.warning(f"Could not load embedder {name}: {e} — BM25 only.")
        _bge_model = False
    return _bge_model or None

# Valid section tags in this DB (audited). Used to validate router hints.
KNOWN_SECTIONS = {
    "general", "revenue", "cash_and_capital", "risk_factors", "financials",
}

# Query-breadth → RAPTOR start level (doc §5.2).
_BROAD_MARKERS = (
    "summarize", "summary", "overview", "overall", "across", "general",
    "annual report", "fiscal year", " fy", "10-k", "10k", "big picture",
    "risk factors", "everything", "all the", "in general",
)
_NEEDLE_MARKERS = (
    "how much", "exact", "exactly", "specific", "specifically", "quote",
    "what was the", "what is the", "figure", "number", "$", "percent",
    "how many", "value of",
)


@dataclass
class RetrievalHints:
    """Router-supplied retrieval hints (doc §3.1 retrieval_hints)."""
    section: Optional[List[str]] = None       # filter to these section tags
    raptor_start_level: Optional[int] = None  # override breadth heuristic
    top_k: int = 8                            # final fused results
    leg_limit: int = 50                       # candidates per leg before RRF
    enable_vector: bool = True                # bge-large-en-v1.5 pgvector leg; safe-degrades to BM25
    expand_parents: bool = True               # attach parent summary for leaf hits


def _classify_start_level(query: str) -> int:
    """
    Pick a RAPTOR start level from query breadth (doc §5.2):
      broad  → 2 (summary)   needle/specific → 0 (leaf)   thematic → 0 (default)
    """
    q = f" {query.lower()} "
    if any(m in q for m in _BROAD_MARKERS):
        return 2
    if any(m in q for m in _NEEDLE_MARKERS):
        return 0
    return 0  # thematic default: leaves (+ parent summaries via expand_parents)


def _rrf_fuse(*ranked_lists: List[str]) -> List[str]:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i(d)), best first."""
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


async def _scoped_filing_ids(
    session, ticker: str, period_end_date: Optional[date]
) -> List[int]:
    """Spine scope: filing_ids for (ticker[, period_end_date]) — spine-first."""
    if period_end_date is not None:
        sql = (
            "SELECT filing_id FROM ontology.filings "
            "WHERE ticker = :t AND period_end_date = :ped"
        )
        params = {"t": ticker, "ped": period_end_date}
    else:
        sql = "SELECT filing_id FROM ontology.filings WHERE ticker = :t"
        params = {"t": ticker}
    rows = await session.execute(text(sql), params)
    return [r.filing_id for r in rows.fetchall()]


async def _bm25_ranked(
    session,
    filing_ids: List[int],
    level: int,
    query: str,
    sections: Optional[List[str]],
    limit: int,
) -> List[str]:
    """
    BM25 leg via on-the-fly Postgres FTS (no tsvector column in this schema).
    Returns chunk_ids ordered best→worst. ``websearch_to_tsquery`` is used so
    arbitrary user text can never raise a tsquery syntax error.
    """
    sec_clause = ""
    params: Dict[str, Any] = {
        "fids": filing_ids,
        "lvl": level,
        "q": query,
        "lim": limit,
    }
    if sections:
        sec_clause = "AND sf.section = ANY(:sections)"
        params["sections"] = sections

    sql = f"""
        SELECT sf.chunk_id
        FROM ontology.sec_filings sf
        WHERE sf.filing_id = ANY(:fids)
          AND sf.level = :lvl
          {sec_clause}
          AND to_tsvector('english', coalesce(sf.text, sf.context, ''))
              @@ websearch_to_tsquery('english', :q)
        ORDER BY ts_rank_cd(
                   to_tsvector('english', coalesce(sf.text, sf.context, '')),
                   websearch_to_tsquery('english', :q)
                 ) DESC,
                 sf.id
        LIMIT :lim
    """
    rows = await session.execute(text(sql), params)
    return [r.chunk_id for r in rows.fetchall()]


async def _vector_ranked(
    session,
    filing_ids: List[int],
    level: int,
    query: str,
    sections: Optional[List[str]],
    limit: int,
) -> List[str]:
    """
    Vector leg — pgvector cosine search over the Neon-matching 1024-dim
    embeddings (BAAI/bge-large-en-v1.5, per settings.embedding_model_neon).

    Returns chunk_ids ranked most→least similar. NEVER raises and returns []
    on any failure (model load, dim mismatch, network) so RRF degrades to
    BM25-only rather than corrupting results with an incompatible vector.
    """
    model = _get_bge()
    if model is None:
        return []
    try:
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(
            None,
            lambda: model.encode(
                _BGE_QUERY_INSTRUCTION + query,
                normalize_embeddings=True,
            ).tolist(),
        )
        if len(vec) != _NEON_EMBED_DIM:  # guard: never query with wrong dim
            logger.warning(
                f"query embedding is {len(vec)}-dim, expected "
                f"{_NEON_EMBED_DIM} — skipping vector leg."
            )
            return []

        sec_clause = ""
        params: Dict[str, Any] = {
            "fids": filing_ids,
            "lvl": level,
            "vec": str(vec),
            "lim": limit,
        }
        if sections:
            sec_clause = "AND sf.section = ANY(:sections)"
            params["sections"] = sections

        # CAST(:vec AS vector) — NOT ":vec::vector": the `::` cast adjacent to
        # a bindparam breaks SQLAlchemy text() colon parsing.
        sql = f"""
            SELECT sf.chunk_id
            FROM ontology.sec_filings sf
            WHERE sf.filing_id = ANY(:fids)
              AND sf.level = :lvl
              AND sf.embedding IS NOT NULL
              {sec_clause}
            ORDER BY sf.embedding <=> CAST(:vec AS vector)
            LIMIT :lim
        """
        # SAVEPOINT: if anything here fails it must NOT poison the outer
        # retrieval transaction (BM25, parent-summary, hydrate run after).
        async with session.begin_nested():
            rows = await session.execute(text(sql), params)
            ranked = [r.chunk_id for r in rows.fetchall()]
        return ranked
    except Exception as e:
        logger.warning(f"vector leg failed ({e}) — BM25 only this query.")
        return []


async def _recent_fallback(
    session, filing_ids: List[int], level: int,
    sections: Optional[List[str]], limit: int,
) -> List[str]:
    """Empty query / no FTS hits → most-recent in-scope chunks (resilience
    parity with the old ILIKE path, which never returned []) ."""
    sec_clause = ""
    params: Dict[str, Any] = {"fids": filing_ids, "lvl": level, "lim": limit}
    if sections:
        sec_clause = "AND sf.section = ANY(:sections)"
        params["sections"] = sections
    sql = f"""
        SELECT sf.chunk_id
        FROM ontology.sec_filings sf
        JOIN ontology.filings f ON f.filing_id = sf.filing_id
        WHERE sf.filing_id = ANY(:fids)
          AND sf.level = :lvl
          {sec_clause}
        ORDER BY f.period_end_date DESC NULLS LAST, sf.id
        LIMIT :lim
    """
    rows = await session.execute(text(sql), params)
    return [r.chunk_id for r in rows.fetchall()]


async def _parent_summaries(
    session, filing_ids: List[int], leaf_chunk_ids: List[str], limit: int,
) -> List[str]:
    """RAPTOR expand: summary nodes whose children include any leaf hit."""
    if not leaf_chunk_ids:
        return []
    sql = """
        SELECT sf.chunk_id
        FROM ontology.sec_filings sf
        WHERE sf.filing_id = ANY(:fids)
          AND sf.level > 0
          AND sf.children && :leaves
        LIMIT :lim
    """
    rows = await session.execute(
        text(sql),
        {"fids": filing_ids, "leaves": leaf_chunk_ids, "lim": limit},
    )
    return [r.chunk_id for r in rows.fetchall()]


async def _hydrate(
    session, chunk_ids: List[str], source_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Fetch full rows + filing metadata for the fused chunk_ids, preserving order."""
    if not chunk_ids:
        return []
    sql = """
        SELECT sf.chunk_id, sf.level, sf.section, sf.speaker,
               coalesce(sf.text, sf.context, '') AS content,
               sf.period AS period_label, sf.filing_id,
               f.period_end_date, f.fiscal_year, f.fiscal_quarter,
               f.filing_type, f.filed_date
        FROM ontology.sec_filings sf
        JOIN ontology.filings f ON f.filing_id = sf.filing_id
        WHERE sf.chunk_id = ANY(:cids)
    """
    rows = await session.execute(text(sql), {"cids": chunk_ids})
    by_id = {r.chunk_id: r for r in rows.fetchall()}

    out: List[Dict[str, Any]] = []
    n = len(chunk_ids)
    for i, cid in enumerate(chunk_ids):
        r = by_id.get(cid)
        if r is None:
            continue
        content = r.content or ""
        out.append({
            "chunk_id": cid,
            "level": r.level,
            "section": r.section,
            "speaker": r.speaker,
            # drop-in keys for existing consumers:
            "context": content,
            "content": content,
            "period_end_date": r.period_end_date,
            "fiscal_year": r.fiscal_year,
            "fiscal_quarter": r.fiscal_quarter,
            "filing_type": r.filing_type,
            "period": r.period_label,
            "filing_id": r.filing_id,
            # retrieval metadata:
            "source": source_map.get(cid, "rrf"),
            "score": round((n - i) / n, 4),  # normalized fused rank
        })
    return out


async def retrieve_filing_evidence(
    ticker: str,
    *,
    query: str,
    period_end_date: Optional[date] = None,
    hints: Optional[RetrievalHints] = None,
) -> List[Dict[str, Any]]:
    """
    Spine-scoped hybrid retrieval over ontology.sec_filings.

    Steps: resolve canonical ticker → resolve period (latest if None) →
    scope filing_ids → choose RAPTOR level → BM25 (∪ vector stub) → RRF →
    optional parent-summary expansion → hydrate. NEVER raises; returns []
    on any failure so callers can fall back.
    """
    hints = hints or RetrievalHints()
    if not is_neon_available():
        return []

    query = (query or "").strip()

    sections = None
    if hints.section:
        sections = [s for s in hints.section if s in KNOWN_SECTIONS]
        if hints.section and not sections:
            logger.debug(f"retrieval: dropping unknown section hints {hints.section}")

    level = (
        hints.raptor_start_level
        if hints.raptor_start_level is not None
        else _classify_start_level(query)
    )

    try:
        canonical = await neon_reader.resolve_canonical_ticker(ticker)
        if period_end_date is None:
            period_end_date = await neon_reader.get_latest_period_end(canonical)

        async with get_neon_session() as session:
            filing_ids = await _scoped_filing_ids(session, canonical, period_end_date)
            if not filing_ids:
                logger.debug(f"retrieval: no filings in scope for {canonical} {period_end_date}")
                return []

            if query:
                bm25 = await _bm25_ranked(
                    session, filing_ids, level, query, sections, hints.leg_limit
                )
                vec = (
                    await _vector_ranked(
                        session, filing_ids, level, query,
                        sections, hints.leg_limit,
                    )
                    if hints.enable_vector else []
                )
                fused = _rrf_fuse(bm25, vec) if (bm25 or vec) else []
            else:
                fused = []

            if not fused:
                fused = await _recent_fallback(
                    session, filing_ids, level, sections, hints.top_k
                )

            top = fused[: hints.top_k]
            source_map = {c: ("rrf" if query else "recent") for c in top}

            # RAPTOR expand: pull parent summaries for leaf hits (doc §5.2).
            if hints.expand_parents and level == 0 and top:
                parents = await _parent_summaries(
                    session, filing_ids, top, limit=3
                )
                for p in parents:
                    if p not in source_map:
                        source_map[p] = "raptor_parent"
                        top.append(p)

            return await _hydrate(session, top, source_map)

    except Exception as e:
        logger.debug(f"retrieve_filing_evidence({ticker}): {e}")
        return []
