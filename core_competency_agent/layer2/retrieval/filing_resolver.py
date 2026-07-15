"""Resolve a single 10-K filing for a given ticker + fiscal year.

Adapted from the retrieval schema's filing_resolver.py for agent7's use case:
we need to find the right 10-K filing by ticker + fiscal year so downstream
retrieval stays within the correct annual filing.

Uses:
    ontology.filings          — one row per filing
    ontology.v_filing_spine   — canonical_ticker join

Connection: DATABASE_URL_ONTOLOGY_LAB (via layer2.retrieval.connection)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from layer2.retrieval.connection import get_ontology_conn, OntologyDBNotConfigured


@dataclass
class FilingMatch:
    """One resolved filing — the fields agent7 needs for downstream filtering."""
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


def _has_chunks(cur, filing_id: int) -> bool:
    """Return True if narrative_chunks has at least one row for this filing_id.

    Guards against duplicate spine rows with 0 chunks (e.g. AAPL filing_id=167
    which has the same period as filing_id=285 but zero narrative_chunks rows).
    """
    cur.execute(_CHUNK_COUNT_SQL, {"filing_id": filing_id})
    row = cur.fetchone()
    return bool(row and int(row["n"]) > 0)


_PREFERRED_DOC_TYPES = ["10-K", "10-K_A", "10-Q", "earnings_call"]


def resolve_10k_by_fiscal_year(
    ticker: str,
    fiscal_year: int,
) -> Optional[FilingMatch]:
    """
    Find the most recent 10-K filing for (ticker, fiscal_year).

    Matches on:
      - fiscal_year column = fiscal_year   (preferred)
      - OR period_end_date year = fiscal_year
      - OR nearest 10-K before fiscal year-end
    Returns None if no 10-K exists — callers should then try resolve_best_filing().
    """
    # Let OntologyDBNotConfigured propagate — _resolve() in ten_k_retrieval.py
    # catches it and prints the right "DB not configured" message.
    conn = get_ontology_conn()

    try:
        with conn.cursor() as cur:
            # Try fiscal_year column first (most precise).
            # Use filing_id DESC as tiebreaker so duplicate spine rows with the
            # same period_end_date resolve to the higher (real) filing_id.
            cur.execute(
                f"""
                {_BASE_SELECT}
                WHERE UPPER(COALESCE(f.canonical_ticker, f.ticker)) = UPPER(%(t)s)
                  AND LOWER(f.filing_type) = '10-k'
                  AND f.fiscal_year = %(fy)s
                ORDER BY f.period_end_date DESC NULLS LAST, f.filing_id DESC
                LIMIT 5
                """,
                {"t": ticker, "fy": fiscal_year},
            )
            for row in cur.fetchall():
                match = _row_to_match(dict(row))
                if _has_chunks(cur, match.filing_id):
                    return match
            # (all candidates had 0 chunks — fall through to next strategy)

            # Fallback: match by period_end_date year
            cur.execute(
                f"""
                {_BASE_SELECT}
                WHERE UPPER(COALESCE(f.canonical_ticker, f.ticker)) = UPPER(%(t)s)
                  AND LOWER(f.filing_type) = '10-k'
                  AND EXTRACT(YEAR FROM f.period_end_date) = %(yr)s
                ORDER BY f.period_end_date DESC NULLS LAST, f.filing_id DESC
                LIMIT 5
                """,
                {"t": ticker, "yr": fiscal_year},
            )
            for row in cur.fetchall():
                match = _row_to_match(dict(row))
                if _has_chunks(cur, match.filing_id):
                    return match

            # Last resort: nearest 10-K before the fiscal year end
            cur.execute(
                f"""
                {_BASE_SELECT}
                WHERE UPPER(COALESCE(f.canonical_ticker, f.ticker)) = UPPER(%(t)s)
                  AND LOWER(f.filing_type) = '10-k'
                  AND f.period_end_date <= %(cutoff)s
                ORDER BY f.period_end_date DESC NULLS LAST, f.filing_id DESC
                LIMIT 5
                """,
                {"t": ticker, "cutoff": date(fiscal_year, 12, 31)},
            )
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
    Find the most recent relevant filing for (ticker) before as_of_date.

    Tries doc types in order: 10-K → 10-K_A → 10-Q → earnings_call.
    Returns the first match, or None.

    Used as a fallback when resolve_10k_by_fiscal_year() returns None
    (e.g. company has no 10-K ingested yet but has 10-Q or earnings calls).
    """
    conn = get_ontology_conn()
    try:
        with conn.cursor() as cur:
            for doc_type in preferred_types:
                cur.execute(
                    f"""
                    {_BASE_SELECT}
                    WHERE UPPER(COALESCE(f.canonical_ticker, f.ticker)) = UPPER(%(t)s)
                      AND UPPER(f.filing_type) = UPPER(%(dt)s)
                      AND f.period_end_date <= %(cutoff)s
                    ORDER BY f.period_end_date DESC NULLS LAST, f.filing_id DESC
                    LIMIT 5
                    """,
                    {"t": ticker, "dt": doc_type, "cutoff": as_of_date},
                )
                for row in cur.fetchall():
                    match = _row_to_match(dict(row))
                    if _has_chunks(cur, match.filing_id):
                        print(
                            f"  [FilingResolver] No 10-K for {ticker} — "
                            f"using {match.doc_type} "
                            f"(period_end={match.period_end_date}, "
                            f"filing_id={match.filing_id})"
                        )
                        return match
        return None
    except Exception as e:
        print(f"  [FilingResolver] resolve_best_filing failed: {e}")
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
