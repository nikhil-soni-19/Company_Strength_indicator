"""Resolve the most relevant 10-K filing for a given ticker + fiscal year.

Uses:
    ontology.filings        — one row per filing
    ontology.narrative_chunks — for chunk-count validation

Connection: DATABASE_URL_ONTOLOGY_LAB (via retrieval.connection)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from retrieval.connection import get_ontology_conn, OntologyDBNotConfigured


@dataclass
class FilingMatch:
    filing_id: int
    period_end_date: date
    doc_type: str
    canonical_ticker: str
    source_pdf: Optional[str] = None


_BASE_SELECT = """
SELECT
    f.filing_id,
    f.period_end_date,
    f.filing_type,
    COALESCE(f.canonical_ticker, f.ticker) AS canonical_ticker,
    f.source_pdf
FROM ontology.filings f
"""

_CHUNK_COUNT_SQL = """
SELECT COUNT(*) AS n
FROM ontology.narrative_chunks
WHERE filing_id = %(filing_id)s
"""

_PREFERRED_DOC_TYPES = ["10-K", "10-K_A", "10-Q", "earnings_call"]


def _has_chunks(cur, filing_id: int) -> bool:
    cur.execute(_CHUNK_COUNT_SQL, {"filing_id": filing_id})
    row = cur.fetchone()
    return bool(row and int(row["n"]) > 0)


def resolve_10k_by_fiscal_year(
    ticker: str,
    fiscal_year: int,
) -> Optional[FilingMatch]:
    """
    Find the most recent 10-K with chunks for (ticker, fiscal_year).
    Tries three strategies in order: fiscal_year column → period year → nearest before FY end.
    Returns None when no suitable filing exists; OntologyDBNotConfigured propagates upward.
    """
    conn = get_ontology_conn()
    try:
        with conn.cursor() as cur:
            for sql, params in [
                # Strategy 1: fiscal_year column match
                (
                    f"{_BASE_SELECT} WHERE UPPER(COALESCE(f.canonical_ticker, f.ticker)) = UPPER(%(t)s)"
                    " AND LOWER(f.filing_type) = '10-k' AND f.fiscal_year = %(fy)s"
                    " ORDER BY f.period_end_date DESC NULLS LAST, f.filing_id DESC LIMIT 5",
                    {"t": ticker, "fy": fiscal_year},
                ),
                # Strategy 2: period_end_date year
                (
                    f"{_BASE_SELECT} WHERE UPPER(COALESCE(f.canonical_ticker, f.ticker)) = UPPER(%(t)s)"
                    " AND LOWER(f.filing_type) = '10-k'"
                    " AND EXTRACT(YEAR FROM f.period_end_date) = %(yr)s"
                    " ORDER BY f.period_end_date DESC NULLS LAST, f.filing_id DESC LIMIT 5",
                    {"t": ticker, "yr": fiscal_year},
                ),
                # Strategy 3: nearest before FY end
                (
                    f"{_BASE_SELECT} WHERE UPPER(COALESCE(f.canonical_ticker, f.ticker)) = UPPER(%(t)s)"
                    " AND LOWER(f.filing_type) = '10-k' AND f.period_end_date <= %(cutoff)s"
                    " ORDER BY f.period_end_date DESC NULLS LAST, f.filing_id DESC LIMIT 5",
                    {"t": ticker, "cutoff": date(fiscal_year, 12, 31)},
                ),
            ]:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    match = _row_to_match(dict(row))
                    if _has_chunks(cur, match.filing_id):
                        return match
        return None
    except Exception as e:
        print(f"  [FilingResolver] resolve_10k_by_fiscal_year failed: {e}")
        return None
    finally:
        conn.close()


def resolve_best_filing(
    ticker: str,
    as_of_date: date,
    preferred_types: list[str] = _PREFERRED_DOC_TYPES,
) -> Optional[FilingMatch]:
    """
    Find the most recent filing with chunks for (ticker) before as_of_date.
    Tries doc types in priority order: 10-K → 10-K_A → 10-Q → earnings_call.
    """
    conn = get_ontology_conn()
    try:
        with conn.cursor() as cur:
            for doc_type in preferred_types:
                cur.execute(
                    f"{_BASE_SELECT}"
                    " WHERE UPPER(COALESCE(f.canonical_ticker, f.ticker)) = UPPER(%(t)s)"
                    " AND UPPER(f.filing_type) = UPPER(%(dt)s)"
                    " AND f.period_end_date <= %(cutoff)s"
                    " ORDER BY f.period_end_date DESC NULLS LAST, f.filing_id DESC LIMIT 5",
                    {"t": ticker, "dt": doc_type, "cutoff": as_of_date},
                )
                for row in cur.fetchall():
                    match = _row_to_match(dict(row))
                    if _has_chunks(cur, match.filing_id):
                        print(
                            f"  [FilingResolver] No 10-K for {ticker} — "
                            f"using {match.doc_type} filing_id={match.filing_id}"
                        )
                        return match
        return None
    except Exception as e:
        print(f"  [FilingResolver] resolve_best_filing failed: {e}")
        return None
    finally:
        conn.close()


def resolve_latest_earnings_call(
    ticker: str,
    as_of_date: date,
) -> Optional[FilingMatch]:
    """
    Find the most recent earnings_call filing with chunks for the given ticker.
    Returns None if none exists or the ontology DB is unavailable.
    """
    conn = get_ontology_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"{_BASE_SELECT}"
                " WHERE UPPER(COALESCE(f.canonical_ticker, f.ticker)) = UPPER(%(t)s)"
                " AND LOWER(f.filing_type) = 'earnings_call'"
                " AND f.period_end_date <= %(cutoff)s"
                " ORDER BY f.period_end_date DESC NULLS LAST, f.filing_id DESC LIMIT 5",
                {"t": ticker, "cutoff": as_of_date},
            )
            for row in cur.fetchall():
                match = _row_to_match(dict(row))
                if _has_chunks(cur, match.filing_id):
                    print(
                        f"  [FilingResolver] Earnings call: {ticker} "
                        f"filing_id={match.filing_id}  period={match.period_end_date}"
                    )
                    return match
        return None
    except Exception as e:
        print(f"  [FilingResolver] resolve_latest_earnings_call failed: {e}")
        return None
    finally:
        conn.close()


def _row_to_match(row: dict) -> FilingMatch:
    return FilingMatch(
        filing_id=int(row["filing_id"]),
        period_end_date=row["period_end_date"],
        doc_type=row["filing_type"],
        canonical_ticker=row["canonical_ticker"],
        source_pdf=row.get("source_pdf"),
    )
