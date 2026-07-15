"""10-K risk factor retrieval for agent7 — ontology-backed.

This module is the primary interface for fetching 10-K risk factor excerpts
from the csi_ontology_lab database using the production retrieval primitives
(hybrid BM25 + BGE 1024-dim vector search via RRF).

Two entry points:

1. retrieve_risk_factors(ticker, fiscal_year, query, k)
   → List[str] of chunk texts — drop-in replacement for the old stub.

2. retrieve_pestel_risk_factors(ticker, fiscal_year)
   → Dict[str, List[str]] keyed by PESTEL dimension ("P","E","S","T","En","L")
   → Each value is a list of relevant chunk texts for that dimension.
   → Runs 6 targeted searches, one per PESTEL dimension.

Both functions:
  - Resolve the correct 10-K filing via resolve_10k_by_fiscal_year()
  - Use hybrid_search() (BM25 + vector RRF) within that filing
  - Fall back to [] / empty dict if DATABASE_URL_ONTOLOGY_LAB is not set
    or if the filing is not ingested (graceful degradation)

Quota note: retrieve_pestel_risk_factors() runs 6 vector+BM25 searches but
they all hit the same DB and are fast (<100ms each). No external API calls.
"""
from __future__ import annotations

from typing import Dict, List

from layer2.retrieval.connection import OntologyDBNotConfigured
from layer2.retrieval.embedder import embed_query
from layer2.retrieval.filing_resolver import (
    resolve_10k_by_fiscal_year,
    resolve_best_filing,
    FilingMatch,
)
from layer2.retrieval.hybrid_search import hybrid_search

# ── PESTEL dimension search queries ──────────────────────────────────────────
# Each query is designed to surface risk-factor language relevant to that
# PESTEL dimension. Broad enough to hit related language, specific enough to
# stay on-topic.
PESTEL_RF_QUERIES: Dict[str, str] = {
    "P": (
        "government regulation political risk trade tariffs sanctions "
        "geopolitical instability export controls policy uncertainty subsidies"
    ),
    "E": (
        "economic downturn recession interest rate risk inflation foreign exchange "
        "currency fluctuation credit risk macro economic conditions capital markets"
    ),
    "S": (
        "consumer preferences demographic trends workforce labor shortage "
        "social responsibility ESG reputation brand diversity inclusion talent"
    ),
    "T": (
        "technology disruption cybersecurity risk data breach artificial intelligence "
        "digital transformation innovation obsolescence intellectual property patents"
    ),
    "En": (
        "environmental regulation climate change carbon emissions greenhouse gas "
        "sustainability ESG net zero energy transition physical climate risk"
    ),
    "L": (
        "litigation regulatory enforcement antitrust compliance legal proceedings "
        "class action lawsuit government investigation data privacy GDPR"
    ),
}


def retrieve_risk_factors(
    ticker: str,
    fiscal_year: int,
    query: str,
    k: int = 5,
) -> List[str]:
    """
    Retrieve up to k risk-factor excerpts for (ticker, fiscal_year) using
    a single query string. Drop-in replacement for the old stub.

    Returns:
        List of chunk text strings (empty list if not available).
    """
    filing = _resolve(ticker, fiscal_year)
    if filing is None:
        return []

    try:
        qv = embed_query(query)
    except Exception as e:
        print(f"  [10K-Retrieval] Embedder failed: {e}")
        return []

    # Use the resolved doc_type (may be 10-Q or earnings_call if no 10-K exists)
    # Pass section=None for non-10-K types since they don't have a "risk_factors" section label
    section = "risk_factors" if filing.doc_type.upper().startswith("10-K") else None
    chunks = hybrid_search(
        query_text=query,
        query_embedding=qv,
        filing_id=filing.filing_id,
        section=section,
        doc_type=filing.doc_type,
        top_k=k,
    )
    return [c["chunk_text"] for c in chunks if c.get("chunk_text")]


def retrieve_pestel_risk_factors(
    ticker: str,
    fiscal_year: int,
    k_per_dim: int = 3,
) -> Dict[str, List[str]]:
    """
    Retrieve risk-factor excerpts structured by PESTEL dimension.

    Runs 6 targeted hybrid searches (one per dimension) pinned to the same
    10-K filing. Each search uses a dimension-specific query designed to
    surface the most relevant risk language for that PESTEL category.

    Args:
        ticker:      Company ticker (e.g. "AAPL").
        fiscal_year: Most recently completed fiscal year (e.g. 2025).
        k_per_dim:   Max chunks per PESTEL dimension (default 3).

    Returns:
        Dict keyed by dimension ("P","E","S","T","En","L"), each value a
        list of chunk text strings. Returns empty lists per dimension on
        any failure — never raises.
    """
    empty = {dim: [] for dim in PESTEL_RF_QUERIES}

    filing = _resolve(ticker, fiscal_year)
    if filing is None:
        return empty

    # Pre-embed all 6 queries in one batch (faster than 6 serial calls)
    dims = list(PESTEL_RF_QUERIES.keys())
    queries = [PESTEL_RF_QUERIES[d] for d in dims]
    try:
        from layer2.retrieval.embedder import embed_queries
        embeddings = embed_queries(queries)
    except Exception as e:
        print(f"  [10K-Retrieval] Batch embed failed: {e}")
        return empty

    # section="risk_factors" only applies to 10-K; use None for 10-Q/earnings_call
    section = "risk_factors" if filing.doc_type.upper().startswith("10-K") else None

    result: Dict[str, List[str]] = {}
    for dim, query, qv in zip(dims, queries, embeddings):
        try:
            chunks = hybrid_search(
                query_text=query,
                query_embedding=qv,
                filing_id=filing.filing_id,
                section=section,
                doc_type=filing.doc_type,
                top_k=k_per_dim,
            )
            result[dim] = [c["chunk_text"] for c in chunks if c.get("chunk_text")]
        except Exception as e:
            print(f"  [10K-Retrieval] Dim {dim} search failed: {e}")
            result[dim] = []

    return result


def _resolve(ticker: str, fiscal_year: int) -> FilingMatch | None:
    """Resolve the best available filing for (ticker, fiscal_year).

    Priority: 10-K for the given fiscal year → 10-K_A → most recent 10-Q
    → most recent earnings_call. Logs which doc type was actually used.
    """
    from datetime import date as _date
    try:
        # Primary: 10-K for the fiscal year
        filing = resolve_10k_by_fiscal_year(ticker, fiscal_year)
        if filing is not None:
            print(
                f"  [10K-Retrieval] Resolved {ticker} FY{fiscal_year} → "
                f"10-K filing_id={filing.filing_id} "
                f"period_end={filing.period_end_date}"
            )
            return filing

        # Fallback: best available filing before the fiscal year-end cutoff
        # Use Dec 31 of the fiscal year + 9 months to catch companies with
        # non-calendar fiscal years (e.g. Apple FY ends Sep, reports in Nov)
        fy_end = _date(fiscal_year + 1, 9, 30)
        filing = resolve_best_filing(ticker, as_of_date=fy_end)
        if filing is None:
            print(
                f"  [10K-Retrieval] No filings found for {ticker} in "
                f"csi_ontology_lab — excerpt retrieval skipped."
            )
        return filing

    except OntologyDBNotConfigured as e:
        print(f"  [10K-Retrieval] {e}")
        return None
    except Exception as e:
        print(f"  [10K-Retrieval] Filing resolution error: {e}")
        return None
