"""
10-K retrieval for Agent 6 — moat-focused.

Two targeted searches per filing:
  1. MOAT_CLAIMS   — Item 1 / Business section: competitive advantages,
                      pricing power, switching costs, network effects, brand, IP
  2. THREATS       — Item 1A / Risk factors: competitive risks, margin pressure,
                      substitutes, new entrants, disruption

Both use the same hybrid BM25 + BGE 1024-dim vector search as agent7.
"""
from __future__ import annotations

from typing import Dict, List

from layer2.retrieval.connection import OntologyDBNotConfigured
from layer2.retrieval.embedder import embed_queries
from layer2.retrieval.filing_resolver import (
    resolve_10k_by_fiscal_year,
    resolve_best_filing,
    FilingMatch,
)
from layer2.retrieval.hybrid_search import hybrid_search

MOAT_QUERIES: Dict[str, str] = {
    "moat_claims": (
        "competitive advantage moat pricing power switching costs "
        "network effects brand loyalty barriers to entry intellectual property "
        "unique capabilities customer retention market leadership"
    ),
    "threats": (
        "competitive risk market share loss pricing pressure substitute products "
        "new entrants disruption commoditization margin compression "
        "competitive intensity rivalry technological displacement"
    ),
    "transcript_moat": (
        "competitive advantage differentiation pricing power customer retention "
        "market position moat durable growth sustainable"
    ),
}


def retrieve_moat_context(
    ticker: str,
    fiscal_year: int,
    k_per_section: int = 4,
) -> Dict[str, List[str]]:
    """
    Retrieve moat claims + threats from the 10-K for (ticker, fiscal_year).

    Returns dict:
        {
            "moat_claims": [...],    # Item 1 — claimed competitive advantages
            "threats":     [...],    # Item 1A — risk factors
            "transcript_moat": [...] # earnings call moat language
        }
    All values are lists of chunk text strings (empty on failure).
    """
    empty: Dict[str, List[str]] = {k: [] for k in MOAT_QUERIES}

    filing = _resolve(ticker, fiscal_year)
    if filing is None:
        return empty

    dim_names = list(MOAT_QUERIES.keys())
    queries   = [MOAT_QUERIES[d] for d in dim_names]

    try:
        embeddings = embed_queries(queries)
    except Exception as e:
        print(f"  [10K-MoatRetrieval] Batch embed failed: {e}")
        return empty

    # Section routing:
    #   moat_claims  → try "business" first (Item 1), fallback to None
    #   threats      → try "risk_factors" (Item 1A), fallback to None
    #   transcript   → no section filter
    section_hints: Dict[str, str | None] = {
        "moat_claims":    "business"      if filing.doc_type.upper().startswith("10-K") else None,
        "threats":        "risk_factors"  if filing.doc_type.upper().startswith("10-K") else None,
        "transcript_moat": None,
    }

    result: Dict[str, List[str]] = {}
    for dim, query, qv in zip(dim_names, queries, embeddings):
        section = section_hints.get(dim)
        try:
            chunks = hybrid_search(
                query_text=query,
                query_embedding=qv,
                filing_id=filing.filing_id,
                section=section,
                doc_type=filing.doc_type,
                top_k=k_per_section,
            )
            result[dim] = [c["chunk_text"] for c in chunks if c.get("chunk_text")]
        except Exception as e:
            print(f"  [10K-MoatRetrieval] Dim '{dim}' failed: {e}")
            result[dim] = []

    return result


def _resolve(ticker: str, fiscal_year: int) -> FilingMatch | None:
    from datetime import date as _date
    try:
        filing = resolve_10k_by_fiscal_year(ticker, fiscal_year)
        if filing is not None:
            print(
                f"  [10K-MoatRetrieval] {ticker} FY{fiscal_year} → "
                f"{filing.doc_type} filing_id={filing.filing_id}"
            )
            return filing

        fy_end = _date(fiscal_year + 1, 9, 30)
        filing = resolve_best_filing(ticker, as_of_date=fy_end)
        if filing is None:
            print(f"  [10K-MoatRetrieval] No filings for {ticker} in ontology DB.")
        return filing

    except OntologyDBNotConfigured as e:
        print(f"  [10K-MoatRetrieval] {e}")
        return None
    except Exception as e:
        print(f"  [10K-MoatRetrieval] Filing resolution error: {e}")
        return None
