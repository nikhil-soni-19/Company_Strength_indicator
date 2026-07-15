"""
Neon DB reader — source of SEC-derived filing text and financial facts.

**Ground truth:** This module is coded against the tables that actually exist
in the lab Neon DB (see information_schema), not SCHEMA_DESIGN.md aspirational
tables. Current ontology objects:

  - ontology.companies
  - ontology.filings            (use source_pdf — there is no accession column)
  - ontology.sec_filings        (narrative: level 0 = leaf chunks; 1/2 = RAPTOR-style summaries)
  - ontology.financial_facts    (line_item, value_numeric, period_end_date; no XBRL concepts)

There is no ontology.narrative_chunks, narrative_summaries, xbrl_facts, or concepts
table in the migrated schema.

The agent's local DB (connection.py) still holds OHLCV, predictions, and EDGAR fallbacks.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from neon_connection import get_neon_session, is_neon_available
from logger import get_logger

logger = get_logger(__name__)


def _neon_ok(fn_name: str) -> bool:
    """Return False (and log once) if the circuit-breaker has tripped."""
    if not is_neon_available():
        logger.debug(f"neon_reader.{fn_name}: Neon unavailable — skipping, using local fallback.")
        return False
    return True


# ── Ticker helpers (ontology.companies.ticker) ───────────────────────────────


async def resolve_canonical_ticker(ticker: str) -> str:
    """
    Return the ticker as stored in ontology.companies (case-insensitive match).
    Falls back to the uppercased input if Neon is unavailable or ticker not found.
    NEVER raises — always returns a string.
    """
    t = ticker.upper()
    if not _neon_ok("resolve_canonical_ticker"):
        return t
    try:
        async with get_neon_session() as session:
            row = await session.execute(
                text("SELECT ticker FROM ontology.companies WHERE upper(ticker) = :t"),
                {"t": t},
            )
            result = row.scalar_one_or_none()
            if result:
                return result
    except Exception as e:
        logger.debug(f"resolve_canonical_ticker({ticker}): {e}")
    return t


async def get_available_periods(ticker: str) -> List[Dict]:
    """
    Distinct fiscal periods from ontology.filings for this ticker.
    Includes filed_date (earliest filing date for that period) so callers
    can use it as an announcement-date proxy without a second query.
    """
    if not _neon_ok("get_available_periods"):
        return []
    canonical = await resolve_canonical_ticker(ticker)
    try:
        async with get_neon_session() as session:
            rows = await session.execute(
                text("""
                    SELECT
                        period_end_date,
                        fiscal_year,
                        fiscal_quarter,
                        MIN(filed_date) AS filed_date
                    FROM ontology.filings
                    WHERE ticker = :t
                    GROUP BY period_end_date, fiscal_year, fiscal_quarter
                    ORDER BY period_end_date DESC
                """),
                {"t": canonical},
            )
            return [
                {
                    "period_end_date": r.period_end_date,
                    "fiscal_year": r.fiscal_year,
                    "fiscal_quarter": r.fiscal_quarter,
                    "filed_date": r.filed_date,
                }
                for r in rows.fetchall()
            ]
    except Exception as e:
        logger.debug(f"get_available_periods({ticker}): {e}")
        return []


async def get_latest_period_end(ticker: str) -> Optional[date]:
    """Most recent period_end_date from ontology.filings."""
    periods = await get_available_periods(ticker)
    return periods[0]["period_end_date"] if periods else None


async def _filing_id_for_period(
    ticker: str,
    period_end_date: date,
    filing_types: Tuple[str, ...] = ("10-Q", "10-K"),
) -> Optional[Any]:
    """Latest filing_id matching ticker, period, and form types."""
    if not _neon_ok("_filing_id_for_period"):
        return None
    canonical = await resolve_canonical_ticker(ticker)
    try:
        async with get_neon_session() as session:
            row = await session.execute(
                text("""
                    SELECT filing_id
                    FROM ontology.filings
                    WHERE ticker = :t
                      AND period_end_date = :ped
                      AND filing_type = ANY(:types)
                    ORDER BY filed_date DESC
                    LIMIT 1
                """),
                {
                    "t": canonical,
                    "ped": period_end_date,
                    "types": list(filing_types),
                },
            )
            return row.scalar_one_or_none()
    except Exception as e:
        logger.debug(f"_filing_id_for_period({ticker}): {e}")
        return None


# ── Filing metadata ───────────────────────────────────────────────────────────


async def get_filings(
    ticker: str,
    filing_types: Tuple[str, ...] = ("10-Q", "10-K"),
    limit: int = 10,
) -> List[Dict]:
    """
    Rows from ontology.filings. ``source_pdf`` substitutes for SEC accession in this DB.
    We alias ``source_pdf AS accession`` so callers expecting ``filing['accession']`` keep working.
    """
    if not _neon_ok("get_filings"):
        return []
    canonical = await resolve_canonical_ticker(ticker)
    try:
        async with get_neon_session() as session:
            rows = await session.execute(
                text("""
                    SELECT  filing_id,
                            ticker,
                            filing_type,
                            period_end_date,
                            fiscal_year,
                            fiscal_quarter,
                            filed_date,
                            source_pdf     AS accession,
                            source_pdf
                    FROM ontology.filings
                    WHERE ticker = :t
                      AND filing_type = ANY(:types)
                    ORDER BY filed_date DESC
                    LIMIT :lim
                """),
                {"t": canonical, "types": list(filing_types), "lim": limit},
            )
            return [dict(r._mapping) for r in rows.fetchall()]
    except Exception as e:
        logger.debug(f"get_filings({ticker}): {e}")
        return []


# ── Narrative: ontology.sec_filings (no narrative_chunks / no section column) ─


# Cache of actual ontology.sec_filings column names — populated on first call.
_SEC_FILINGS_COLS: Optional[set] = None


async def _get_sec_filings_cols(session) -> set:
    """Return the real column names of ontology.sec_filings (cached after first call)."""
    global _SEC_FILINGS_COLS
    if _SEC_FILINGS_COLS is None:
        rows = await session.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'ontology' AND table_name = 'sec_filings'
        """))
        _SEC_FILINGS_COLS = {r.column_name for r in rows.fetchall()}
        logger.info(f"ontology.sec_filings columns: {sorted(_SEC_FILINGS_COLS)}")
    return _SEC_FILINGS_COLS


def _pick(cols: set, *candidates: str) -> Optional[str]:
    """Return the first candidate that exists in cols, or None."""
    for c in candidates:
        if c in cols:
            return c
    return None


async def get_mda_chunks(
    ticker: str,
    period_end_date: Optional[date] = None,
    max_chunks: int = 500,
) -> List[Dict]:
    """
    Leaf narrative chunks (level = 0) from ontology.sec_filings.

    Column names are discovered at runtime from information_schema so the
    query never breaks when the actual DB schema differs from design-doc names.
    NEVER raises — returns [] on any error so callers fall back to local data.
    """
    if not _neon_ok("get_mda_chunks"):
        return []

    canonical = await resolve_canonical_ticker(ticker)

    if period_end_date is None:
        period_end_date = await get_latest_period_end(canonical)
        if period_end_date is None:
            logger.debug(f"No periods found in Neon for {ticker}")
            return []

    try:
      async with get_neon_session() as session:
        cols = await _get_sec_filings_cols(session)

        # ── Resolve each column to its real name ──────────────────────────
        # Text / content column
        content_col = _pick(cols, "context", "content", "text", "chunk_text", "body")
        # Ordering columns (may not exist — fall back to chunk_id order)
        block_col   = _pick(cols, "block_index", "block_idx", "block_num",
                                  "block", "page", "section_index")
        chunk_col   = _pick(cols, "chunk_index", "chunk_idx", "chunk_num",
                                  "chunk", "sequence", "seq", "position", "idx")
        # Optional columns — include only if present
        embed_col   = _pick(cols, "embedding")
        level_col   = _pick(cols, "level")
        role_col    = _pick(cols, "speaker_role", "role")

        if not content_col:
            logger.warning(
                "Cannot find a text column in ontology.sec_filings. "
                f"Available: {sorted(cols)}"
            )
            return []

        # ── Build SELECT list dynamically ─────────────────────────────────
        select_parts = [
            "sf.chunk_id",
            f"sf.{content_col} AS content",
            "sf.filing_id",
            "f.period_end_date",
            "f.filed_date",
            "f.filing_type",
        ]
        if block_col:
            select_parts.append(f"sf.{block_col} AS block_index")
        if chunk_col:
            select_parts.append(f"sf.{chunk_col} AS chunk_index")
        if embed_col:
            select_parts.append(f"sf.{embed_col}::text AS embedding_text")
        if level_col:
            select_parts.append(f"sf.{level_col}")
        if role_col:
            select_parts.append(f"sf.{role_col}")

        # ── Build WHERE / ORDER ───────────────────────────────────────────
        level_filter = f"AND sf.{level_col} = 0" if level_col else ""

        if block_col and chunk_col:
            order_clause = f"sf.{block_col}, sf.{chunk_col}"
        elif chunk_col:
            order_clause = f"sf.{chunk_col}"
        else:
            order_clause = "sf.chunk_id"

        query = f"""
            SELECT  {', '.join(select_parts)}
            FROM ontology.sec_filings sf
            JOIN ontology.filings f ON sf.filing_id = f.filing_id
            WHERE f.ticker = :t
              AND f.period_end_date = :ped
              {level_filter}
            ORDER BY {order_clause}
            LIMIT :lim
        """

        rows = await session.execute(
            text(query),
            {"t": canonical, "ped": period_end_date, "lim": max_chunks},
        )
        return [dict(r._mapping) for r in rows.fetchall()]
    except Exception as e:
        logger.debug(f"get_mda_chunks({ticker}): {e}")
        return []


async def get_filing_narrative_text(
    filing_id: int,
    max_chars: int = 12_000,
) -> Optional[str]:
    """
    Concatenate leaf-level sec_filings text for one filing_id (L2 milestone pass).
    Prefers ``text`` then ``context``. NEVER raises.
    """
    if not _neon_ok("get_filing_narrative_text"):
        return None
    try:
        async with get_neon_session() as session:
            cols = await _get_sec_filings_cols(session)
            content_col = _pick(cols, "text", "context", "content", "chunk_text", "body")
            if not content_col:
                return None
            level_col = _pick(cols, "level")
            level_filter = f"AND sf.{level_col} = 0" if level_col else ""
            rows = await session.execute(
                text(f"""
                    SELECT sf.{content_col} AS content
                    FROM ontology.sec_filings sf
                    WHERE sf.filing_id = :fid
                      {level_filter}
                    ORDER BY sf.chunk_id
                    LIMIT 500
                """),
                {"fid": filing_id},
            )
            parts: List[str] = []
            total = 0
            for r in rows.fetchall():
                t = (r.content or "").strip()
                if not t:
                    continue
                if total + len(t) > max_chars:
                    parts.append(t[: max_chars - total])
                    break
                parts.append(t)
                total += len(t)
            return "\n\n".join(parts) if parts else None
    except Exception as e:
        logger.debug(f"get_filing_narrative_text({filing_id}): {e}")
        return None


async def get_mda_text(
    ticker: str,
    period_end_date: Optional[date] = None,
    max_chars: int = 50_000,
) -> Optional[str]:
    """
    Join leaf sec_filings chunks into one string (MD&A substitute for the agent).
    Returns None if Neon is unavailable or has no data — caller should fall back
    to local filing.mda_text.  NEVER raises.
    """
    if not _neon_ok("get_mda_text"):
        return None
    try:
        chunks = await get_mda_chunks(ticker, period_end_date)
        if not chunks:
            return None

        parts: List[str] = []
        total = 0
        for chunk in chunks:
            text_to_use = chunk.get("content", "")
            if total + len(text_to_use) > max_chars:
                parts.append(text_to_use[: max_chars - total])
                break
            parts.append(text_to_use)
            total += len(text_to_use)

        return "\n\n".join(parts)
    except Exception as e:
        logger.debug(f"get_mda_text({ticker}): {e}")
        return None


async def get_raptor_summaries(
    ticker: str,
    period_end_date: Optional[date] = None,
    level: int = 2,
) -> List[Dict]:
    """
    Summary rows live in ontology.sec_filings with level 1 or 2 (not a separate narrative_summaries table).
    """
    canonical = await resolve_canonical_ticker(ticker)

    if period_end_date is None:
        period_end_date = await get_latest_period_end(canonical)
        if not period_end_date:
            return []

    async with get_neon_session() as session:
        rows = await session.execute(
            text("""
                SELECT  sf.chunk_id   AS summary_id,
                        sf.level,
                        sf.context    AS content,
                        sf.filing_id
                FROM ontology.sec_filings sf
                JOIN ontology.filings f ON sf.filing_id = f.filing_id
                WHERE f.ticker = :t
                  AND f.period_end_date = :ped
                  AND sf.level = :lvl
                ORDER BY sf.block, sf.chunk
            """),
            {"t": canonical, "ped": period_end_date, "lvl": level},
        )
        return [dict(r._mapping) for r in rows.fetchall()]


# ── Financial facts: ontology.financial_facts (no xbrl_facts / concepts) ─────


async def get_eps_facts(
    ticker: str,
    period_end_date: Optional[date] = None,
) -> Dict:
    """
    Pull numeric facts from ontology.financial_facts for the filing matching ticker + period.
    line_item is matched with ILIKE heuristics (no US-GAAP concept_id in this schema).
    NEVER raises — returns {} on any error.
    """
    if not _neon_ok("get_eps_facts"):
        return {}
    try:
        canonical = await resolve_canonical_ticker(ticker)

        if period_end_date is None:
            period_end_date = await get_latest_period_end(canonical)
            if not period_end_date:
                return {}

        fid = await _filing_id_for_period(canonical, period_end_date)
        if fid is None:
            return {"period_end_date": period_end_date}
    except Exception as e:
        logger.debug(f"get_eps_facts({ticker}) setup: {e}")
        return {}

    try:
      async with get_neon_session() as session:
        meta = await session.execute(
            text("""
                SELECT fiscal_year, fiscal_quarter
                FROM ontology.filings
                WHERE filing_id = :fid
            """),
            {"fid": fid},
        )
        filing_meta = meta.fetchone()

        eps_row = await session.execute(
            text("""
                SELECT value_numeric AS eps_value, period_end_date, period_type
                FROM ontology.financial_facts
                WHERE filing_id = :fid
                  AND (
                        line_item ILIKE '%earnings%per%share%diluted%'
                     OR line_item ILIKE '%eps%diluted%'
                     OR (line_item ILIKE '%earnings%per%share%' AND line_item ILIKE '%diluted%')
                  )
                ORDER BY period_end_date DESC
                LIMIT 1
            """),
            {"fid": fid},
        )
        eps_data = eps_row.fetchone()
        if eps_data is None:
            eps_row = await session.execute(
                text("""
                    SELECT value_numeric AS eps_value, period_end_date, period_type
                    FROM ontology.financial_facts
                    WHERE filing_id = :fid
                      AND line_item ILIKE '%earnings%per%share%'
                    ORDER BY period_end_date DESC
                    LIMIT 1
                """),
                {"fid": fid},
            )
            eps_data = eps_row.fetchone()

        rev_row = await session.execute(
            text("""
                SELECT value_numeric AS revenue, period_end_date
                FROM ontology.financial_facts
                WHERE filing_id = :fid
                  AND line_item ILIKE '%revenue%'
                  AND line_item NOT ILIKE '%cost%'
                  AND line_item NOT ILIKE '%deferred%'
                ORDER BY period_end_date DESC
                LIMIT 1
            """),
            {"fid": fid},
        )
        rev_data = rev_row.fetchone()

        result: Dict = {"period_end_date": period_end_date}
        if filing_meta:
            result["fiscal_year"] = filing_meta.fiscal_year
            result["fiscal_quarter"] = filing_meta.fiscal_quarter

        if eps_data and eps_data.eps_value is not None:
            result["reported_eps"] = float(eps_data.eps_value)
            if filing_meta is None:
                result["period_end_date"] = eps_data.period_end_date

        if rev_data and rev_data.revenue is not None:
            result["reported_revenue"] = float(rev_data.revenue)

        return result
    except Exception as e:
        logger.debug(f"get_eps_facts({ticker}) query: {e}")
        return {}


async def get_financial_history(
    ticker: str,
    concept_names: List[str],
    n_periods: int = 8,
) -> List[Dict]:
    """
    Time series from ontology.financial_facts joined to ontology.filings.

    ``concept_names`` are matched as case-insensitive substrings against ``line_item``
    (legacy API used US-GAAP concept names; adapt to your line_item strings as needed).

    Returns rows ordered by effective date descending (NULLS LAST).  Each row
    includes ``unit`` and ``currency`` so callers can scale value_numeric correctly
    (Apple reports in millions; ``unit`` will be e.g. "millions" or "USD Millions").
    """
    if not _neon_ok("get_financial_history"):
        return []
    canonical = await resolve_canonical_ticker(ticker)
    if not concept_names:
        return []

    patterns = [f"%{n}%" for n in concept_names]
    or_clauses = " OR ".join(
        [f"ff.line_item ILIKE :p{i}" for i in range(len(patterns))]
    )
    params: Dict[str, Any] = {"t": canonical, "n": n_periods}
    for i, p in enumerate(patterns):
        params[f"p{i}"] = p

    try:
        async with get_neon_session() as session:
            rows = await session.execute(
                text(f"""
                    SELECT  COALESCE(ff.period_end_date, f.period_end_date) AS period_end_date,
                            f.period_end_date   AS filing_period_end_date,
                            ff.line_item,
                            ff.value_numeric,
                            ff.period_type,
                            ff.unit,
                            ff.currency,
                            ff.statement_type
                    FROM ontology.financial_facts ff
                    JOIN ontology.filings f ON ff.filing_id = f.filing_id
                    WHERE f.ticker = :t
                      AND ({or_clauses})
                    ORDER BY COALESCE(ff.period_end_date, f.period_end_date) DESC NULLS LAST
                    LIMIT :n
                """),
                params,
            )
            return [dict(r._mapping) for r in rows.fetchall()]
    except Exception as e:
        logger.debug(f"get_financial_history({ticker}): {e}")
        return []


# ── Embedding / similarity (sec_filings.embedding, level 0) ───────────────────


async def search_similar_chunks_pgvector(
    query_embedding: List[float],
    ticker: Optional[str] = None,
    section: str = "mda",
    top_k: int = 10,
    score_threshold: float = 0.5,
) -> List[Dict]:
    """
    Pgvector search over ontology.sec_filings (level 0). No section column — ``section`` is ignored.
    Neon embeddings are 1024-dim; 768-dim agent vectors should skip (returns [] with warning).
    """
    del section  # schema has no MD&A section tag; full narrative is in sec_filings

    if len(query_embedding) != 1024:
        logger.warning(
            f"Embedding dimension mismatch: got {len(query_embedding)}, Neon expects 1024. "
            "Skipping pgvector search — using Qdrant fallback."
        )
        return []

    ticker_filter = ""
    params: Dict[str, Any] = {
        "vec": str(query_embedding),
        "k": top_k,
        "thresh": score_threshold,
    }
    if ticker:
        canonical = await resolve_canonical_ticker(ticker)
        ticker_filter = "AND f.ticker = :t"
        params["t"] = canonical

    async with get_neon_session() as session:
        rows = await session.execute(
            text(f"""
                SELECT  sf.chunk_id,
                        sf.context    AS content,
                        sf.filing_id,
                        f.ticker AS canonical_ticker,
                        f.period_end_date,
                        1 - (sf.embedding <=> :vec::vector) AS cosine_score
                FROM ontology.sec_filings sf
                JOIN ontology.filings f ON sf.filing_id = f.filing_id
                WHERE sf.level = 0
                  AND sf.embedding IS NOT NULL
                  {ticker_filter}
                  AND 1 - (sf.embedding <=> :vec::vector) >= :thresh
                ORDER BY sf.embedding <=> :vec::vector
                LIMIT :k
            """),
            params,
        )
        return [dict(r._mapping) for r in rows.fetchall()]


# ── Market tables: estimate_consensus / earnings_surprise / price_daily ───────


async def get_earnings_surprise(
    ticker: str,
    n: int = 8,
) -> list[dict]:
    """
    Return the last ``n`` reported earnings surprise rows for *ticker*
    from ``ontology.earnings_surprise``, newest first.

    Each row contains:
        fiscal_period, fiscal_year, fiscal_quarter, announcement_date,
        reported_eps, estimate_eps, surprise_pct, is_reported,
        price_change_pct, guidance_eps, guidance_surprise_pct
    """
    if not _neon_ok("get_earnings_surprise"):
        return []
    canonical = await resolve_canonical_ticker(ticker)
    async with get_neon_session() as session:
        rows = await session.execute(
            text("""
                SELECT fiscal_period, fiscal_year, fiscal_quarter,
                       announcement_date, reported_eps, estimate_eps,
                       surprise_pct, is_reported, price_change_pct,
                       guidance_eps, guidance_surprise_pct
                FROM   ontology.earnings_surprise
                WHERE  ticker = :t
                  AND  is_reported = true
                ORDER  BY announcement_date DESC
                LIMIT  :n
            """),
            {"t": canonical, "n": n},
        )
        return [dict(r._mapping) for r in rows.fetchall()]


async def get_eps_estimate_revision(
    ticker: str,
    target_period: str,
    lookback_weeks: int = 8,
) -> dict:
    """
    Compute EPS estimate drift for *target_period* over the last
    ``lookback_weeks`` weeks from ``ontology.estimate_consensus``.

    Returns a dict:
        {
          "revision_pct":   float | None,   # (latest - earliest) / |earliest|
          "latest_estimate": float | None,
          "earliest_estimate": float | None,
          "as_of_latest":   date  | None,
          "as_of_earliest": date  | None,
          "n_observations": int,
        }
    An upward revision → positive revision_pct → bullish signal.
    """
    if not _neon_ok("get_eps_estimate_revision"):
        return {"revision_pct": None, "n_observations": 0}
    canonical = await resolve_canonical_ticker(ticker)
    async with get_neon_session() as session:
        rows = await session.execute(
            text("""
                SELECT as_of_date, value_mean
                FROM   ontology.estimate_consensus
                WHERE  ticker        = :t
                  AND  target_period = :period
                  AND  as_of_date   >= current_date - (:weeks * 7)
                ORDER  BY as_of_date
            """),
            {"t": canonical, "period": target_period, "weeks": lookback_weeks},
        )
        data = [dict(r._mapping) for r in rows.fetchall()]

    if len(data) < 2:
        return {"revision_pct": None, "n_observations": len(data)}

    earliest_val = float(data[0]["value_mean"])
    latest_val   = float(data[-1]["value_mean"])
    revision_pct = (
        (latest_val - earliest_val) / abs(earliest_val)
        if earliest_val != 0 else None
    )
    return {
        "revision_pct":      revision_pct,
        "latest_estimate":   latest_val,
        "earliest_estimate": earliest_val,
        "as_of_latest":      data[-1]["as_of_date"],
        "as_of_earliest":    data[0]["as_of_date"],
        "n_observations":    len(data),
    }


async def get_price_daily(
    ticker: str,
    n_days: int = 30,
) -> list[dict]:
    """
    Return the last ``n_days`` daily close prices from
    ``ontology.price_daily``, newest first.

    Falls back gracefully to an empty list if not available.
    """
    if not _neon_ok("get_price_daily"):
        return []
    canonical = await resolve_canonical_ticker(ticker)
    async with get_neon_session() as session:
        rows = await session.execute(
            text("""
                SELECT price_date, close_px, currency
                FROM   ontology.price_daily
                WHERE  ticker = :t
                ORDER  BY price_date DESC
                LIMIT  :n
            """),
            {"t": canonical, "n": n_days},
        )
        return [dict(r._mapping) for r in rows.fetchall()]


# ── SECFiling-shaped dict for sec_pipeline / features ────────────────────────


async def build_filing_context(
    ticker: str,
    period_end_date: Optional[date] = None,
) -> Optional[Dict]:
    """
    Dict resembling the agent's SECFiling ORM. Accession substitute is filings.source_pdf.
    """
    canonical = await resolve_canonical_ticker(ticker)

    filings = await get_filings(canonical, filing_types=("10-Q", "10-K"), limit=1)
    if not filings:
        logger.warning(f"No filings in Neon for {ticker}")
        return None

    filing = filings[0]
    period = filing.get("period_end_date") or period_end_date

    mda_text = await get_mda_text(canonical, period)
    eps_facts = await get_eps_facts(canonical, period)

    accession = filing.get("source_pdf") or filing.get("accession")

    return {
        "ticker": canonical,
        "filing_id": filing.get("filing_id"),
        "filing_type": filing.get("filing_type"),
        "period_end_date": period,
        "filed_date": filing.get("filed_date"),
        "accession_number": accession,
        "mda_text": mda_text,
        "processed": mda_text is not None,
        "reported_eps": eps_facts.get("reported_eps"),
        "reported_revenue": eps_facts.get("reported_revenue"),
        "fiscal_year": eps_facts.get("fiscal_year"),
        "fiscal_quarter": eps_facts.get("fiscal_quarter"),
        "_source": "neon",
    }


# ── ERN: earnings history (ontology.earnings_surprise, BLOOMBERG_ERN) ─────────


async def get_earnings_history(ticker: str, n: int = 8) -> list[dict]:
    """
    Last ``n`` *reported* earnings rows (ERN), newest first — richer than
    get_earnings_surprise: includes pe_ratio, eps_ttm, period_end_date.
    """
    if not _neon_ok("get_earnings_history"):
        return []
    canonical = await resolve_canonical_ticker(ticker)
    try:
        async with get_neon_session() as session:
            rows = await session.execute(
                text("""
                    SELECT fiscal_period, fiscal_year, fiscal_quarter,
                           period_end_date, announcement_date,
                           reported_eps, estimate_eps, surprise_pct,
                           guidance_eps, guidance_surprise_pct,
                           price_change_pct, pe_ratio, eps_ttm
                    FROM   ontology.earnings_surprise
                    WHERE  ticker = :t AND is_reported = true
                    ORDER  BY announcement_date DESC
                    LIMIT  :n
                """),
                {"t": canonical, "n": n},
            )
            return [dict(r._mapping) for r in rows.fetchall()]
    except Exception as e:
        logger.debug(f"get_earnings_history({ticker}): {e}")
        return []


async def get_next_earnings_date(ticker: str) -> Optional[Dict]:
    """
    Earliest *unreported* future earnings row (ERN, is_reported=false) — the
    horizon anchor. Returns {announcement_date, fiscal_period, estimate_eps,
    pe_ratio} or None.
    """
    if not _neon_ok("get_next_earnings_date"):
        return None
    canonical = await resolve_canonical_ticker(ticker)
    try:
        async with get_neon_session() as session:
            row = await session.execute(
                text("""
                    SELECT announcement_date, fiscal_period, fiscal_year,
                           fiscal_quarter, estimate_eps, pe_ratio
                    FROM   ontology.earnings_surprise
                    WHERE  ticker = :t
                      AND  is_reported = false
                      AND  announcement_date >= current_date
                    ORDER  BY announcement_date ASC
                    LIMIT  1
                """),
                {"t": canonical},
            )
            r = row.fetchone()
            return dict(r._mapping) if r else None
    except Exception as e:
        logger.debug(f"get_next_earnings_date({ticker}): {e}")
        return None


# ── EEG: point-in-time consensus EPS trajectory (estimate_consensus) ──────────


async def get_estimate_trajectory(
    ticker: str,
    target_period: str,
    lookback_days: int = 730,
) -> Dict[str, Any]:
    """
    Point-in-time consensus EPS series for *target_period* (e.g. "FY-2026")
    over the last ``lookback_days`` (default 2 years), oldest→newest, plus a
    computed revision summary. NEVER raises.

    Returns:
      {
        "series": [{"as_of": date, "value": float}, ...],
        "earliest": float|None, "latest": float|None,
        "revision_pct": float|None,        # (latest-earliest)/|earliest|
        "recent_4w_delta_pct": float|None, # last vs ~4wk-ago
        "slope_per_quarter": float|None,   # ~linear drift / 90d
        "n_observations": int,
        "as_of_latest": date|None,
      }
    """
    empty = {
        "series": [], "earliest": None, "latest": None,
        "revision_pct": None, "recent_4w_delta_pct": None,
        "slope_per_quarter": None, "n_observations": 0,
        "as_of_latest": None,
    }
    if not _neon_ok("get_estimate_trajectory"):
        return empty
    canonical = await resolve_canonical_ticker(ticker)
    try:
        async with get_neon_session() as session:
            rows = await session.execute(
                text("""
                    SELECT as_of_date, value_mean
                    FROM   ontology.estimate_consensus
                    WHERE  ticker = :t
                      AND  upper(metric) = 'EPS'
                      AND  target_period = :tp
                      AND  as_of_date >= current_date - (:days * INTERVAL '1 day')
                      AND  value_mean IS NOT NULL
                    ORDER  BY as_of_date
                """),
                {"t": canonical, "tp": target_period, "days": lookback_days},
            )
            data = [(r.as_of_date, float(r.value_mean)) for r in rows.fetchall()]
    except Exception as e:
        logger.debug(f"get_estimate_trajectory({ticker},{target_period}): {e}")
        return empty

    if not data:
        return empty

    series = [{"as_of": d, "value": v} for d, v in data]
    earliest, latest = data[0][1], data[-1][1]
    revision_pct = (
        (latest - earliest) / abs(earliest) if earliest not in (0, None) else None
    )

    # recent ~4-week delta
    recent_4w = None
    cutoff = data[-1][0]
    prior = [v for (d, v) in data if (cutoff - d).days >= 28]
    if prior:
        base = prior[-1]
        recent_4w = (latest - base) / abs(base) if base else None

    # crude slope per ~quarter (90d) over the span
    slope_q = None
    span_days = (data[-1][0] - data[0][0]).days
    if span_days > 0:
        slope_q = (latest - earliest) / span_days * 90.0

    return {
        "series": series,
        "earliest": earliest,
        "latest": latest,
        "revision_pct": revision_pct,
        "recent_4w_delta_pct": recent_4w,
        "slope_per_quarter": slope_q,
        "n_observations": len(data),
        "as_of_latest": data[-1][0],
    }
