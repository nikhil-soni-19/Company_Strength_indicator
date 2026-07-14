"""
Phase 1 — Data contract: fetch and align all inputs for Agent 4.

Public API:
    fetch_inputs(ticker, n_quarters) -> InputBundle

All downstream code (Layer 1, Layer 2, fusion) touches only InputBundle —
never raw DB rows or yfinance DataFrames. This is the single data boundary.

Data sources and fallback hierarchy:
    Revenue + capex  : fundamentals.{ticker}_precomputed_metrics (DB primary)
                       → yfinance fallback if ticker not in DB
    R&D              : fundamentals.{ticker}_income_statement, field='researchAndDevelopment'
                       (wide format, columns named YYYY_MM_DD)
                       → yfinance gap-fill for any quarter where DB returns 0
    Holders          : yfinance always (not stored in DB)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from config import DEFAULT_QUARTERS
from retrieval.esg_fetcher import ESGData, fetch_esg_data

load_dotenv(Path(__file__).parent / ".env")

_DB_URL: str = os.environ.get("DATABASE_URL", "")


# ─── Output types ─────────────────────────────────────────────────────────────

@dataclass
class DataCoverage:
    """
    Records what data was actually returned for confidence guardrail use.
    All fields are set in fetch_inputs(); downstream code reads them but never writes.
    """
    quarters_returned: int
    rd_quarters_from_db: int
    rd_quarters_from_yf: int
    capex_found: bool     # True if at least one non-zero capex value was found
    holders_found: bool   # True if insider_pct was successfully retrieved
    source: str           # "db" | "yfinance" | "mixed"


@dataclass
class InputBundle:
    """
    Aligned financial inputs for one ticker. All numeric lists are oldest → newest.
    This is the only object Layer 1 and Layer 2 receive — never raw tables.
    """
    ticker: str
    periods: list[str]            # YYYY-MM-DD, quarterly
    revenue: list[float]          # total revenue, ≥ 0
    rd: list[float]               # R&D expense, ≥ 0
    capex: list[float]            # capital expenditure, ≥ 0 (sign-normalised)
    insider_pct: Optional[float]  # fraction in [0, 1]; None if unavailable
    institutional_top10: Optional[float]  # top-10 inst concentration; None if unavailable
    coverage: DataCoverage
    esg: Optional[ESGData] = None  # Bloomberg ESG annual data; None if not in DB


# ─── DB helpers ───────────────────────────────────────────────────────────────

def _get_conn():
    return psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _db_ticker_exists(ticker: str) -> bool:
    """Return True if fundamentals.{ticker}_precomputed_metrics exists in Neon."""
    if not _DB_URL:
        return False
    try:
        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_tables "
                "WHERE schemaname='fundamentals' AND tablename=%s LIMIT 1",
                (f"{ticker.lower()}_precomputed_metrics",),
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def _fetch_core_from_db(ticker: str, n_quarters: int) -> Optional[list[dict]]:
    """
    Pull (period, total_revenue, capex) from fundamentals.{ticker}_precomputed_metrics.
    Returns rows ordered oldest → newest, or None on failure.
    """
    table = f"fundamentals.{ticker.lower()}_precomputed_metrics"
    try:
        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT period, total_revenue, capex
                FROM (
                    SELECT period, total_revenue, capex
                    FROM {table}
                    ORDER BY period DESC
                    LIMIT %s
                ) sub
                ORDER BY period ASC
                """,
                (n_quarters,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows] if rows else None
    except Exception as e:
        print(f"  [DataContract] Core DB fetch failed for {ticker}: {e}")
        return None


def _fetch_rd_from_db(ticker: str, periods: list[str]) -> tuple[list[float], int]:
    """
    Pull R&D from fundamentals.{ticker}_income_statement (wide format).
    The table has one row per financial field; columns are date strings like '2025_09_30'.

    Returns:
        (rd_values aligned to periods, count of periods with a non-zero DB value)
    Any period not present in the DB gets a 0.0 placeholder for later gap-filling.
    """
    table = f"fundamentals.{ticker.lower()}_income_statement"
    try:
        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table} WHERE field='researchAndDevelopment'")
            row = cur.fetchone()
        if not row:
            return [0.0] * len(periods), 0

        result: list[float] = []
        count = 0
        for p in periods:
            col = p.replace("-", "_")   # '2025-09-30' → '2025_09_30'
            val = row.get(col)
            if val is not None and val != "":
                result.append(float(val))
                count += 1
            else:
                result.append(0.0)
        return result, count
    except Exception as e:
        print(f"  [DataContract] R&D DB fetch failed for {ticker}: {e}")
        return [0.0] * len(periods), 0


# ─── yfinance helpers ─────────────────────────────────────────────────────────

def _safe_get_yf(df: pd.DataFrame, *keys: str) -> list[float]:
    """Try multiple field names in a yfinance DataFrame; return zeros if none match."""
    n = len(df)
    for k in keys:
        if k in df.columns:
            return df[k].fillna(0).tolist()
    return [0.0] * n


def _fetch_from_yfinance(ticker: str, n_quarters: int) -> Optional[dict]:
    """
    Pull revenue, R&D, capex, and periods from yfinance quarterly statements.
    Returns None if no income statement data is available.
    """
    try:
        t = yf.Ticker(ticker)
        income = t.quarterly_income_stmt
        cashflow = t.quarterly_cashflow

        if income is None or income.empty:
            return None

        income = income.T.sort_index().tail(n_quarters)
        idx = income.index

        def _align(df: pd.DataFrame) -> pd.DataFrame:
            if df is not None and not df.empty:
                return df.T.sort_index().reindex(idx, fill_value=0)
            return pd.DataFrame(0.0, index=idx, columns=[])

        cashflow = _align(cashflow)

        revenue = _safe_get_yf(income, "Total Revenue", "Revenue")
        rd_raw  = _safe_get_yf(income, "Research And Development", "Research Development")
        rd      = [abs(v) for v in rd_raw]
        capex   = [abs(v) for v in _safe_get_yf(
            cashflow, "Capital Expenditure", "Purchases Of Property Plant And Equipment"
        )]

        return {
            "periods": [str(d)[:10] for d in idx.tolist()],
            "revenue": revenue,
            "rd":     rd,
            "capex":  capex,
        }
    except Exception as e:
        print(f"  [DataContract] yfinance fetch failed for {ticker}: {e}")
        return None


def _fill_rd_gaps(
    rd_db: list[float],
    periods: list[str],
    ticker: str,
) -> tuple[list[float], int]:
    """
    For quarters where DB returned 0 for R&D, try to fill from yfinance.
    Only fetches yfinance if there are actual gaps, to avoid unnecessary calls.

    Returns:
        (rd list with gaps filled where possible, count of periods filled from yfinance)
    """
    gap_indices = [i for i, v in enumerate(rd_db) if v == 0.0]
    if not gap_indices:
        return rd_db, 0

    try:
        t = yf.Ticker(ticker)
        income = t.quarterly_income_stmt
        if income is None or income.empty:
            return rd_db, 0

        income = income.T.sort_index()
        yf_rd: dict[str, float] = {}
        for col in ("Research And Development", "Research Development"):
            if col in income.columns:
                for date_idx, val in income[col].items():
                    k = str(date_idx)[:10]
                    if val is not None:
                        yf_rd[k] = abs(float(val))
                break

        filled = list(rd_db)
        count = 0
        for i in gap_indices:
            yf_val = yf_rd.get(periods[i], 0.0)
            if yf_val > 0:
                filled[i] = yf_val
                count += 1
        return filled, count
    except Exception:
        return rd_db, 0


def _fetch_holders(ticker: str) -> dict:
    """
    Pull insider ownership % and top-10 institutional concentration from yfinance.
    Returns floats in [0, 1] or None per field on failure.

    yfinance API note (≥ 0.2.x):
      major_holders  — DataFrame indexed by field name (e.g. 'insiderPercent'),
                       single column 'Value'. NOT the old 2-column layout.
      institutional_holders — pctHeld column (was '% Out' in older versions).
    """
    try:
        t = yf.Ticker(ticker)

        # ── Insider ownership ─────────────────────────────────────────────────
        insider_pct: Optional[float] = None
        try:
            mh = t.major_holders
            if mh is not None and not mh.empty:
                # New API: index contains field names like 'insiderPercent'
                # Old API: two columns [value, description] — no longer used.
                for key in ("insiderPercent", "insidersPercentHeld"):
                    if key in mh.index:
                        raw = mh.loc[key, "Value"]
                        insider_pct = float(raw)
                        # yfinance returns the value as a fraction (0.003) not %
                        if insider_pct > 1.0:          # guard: old % format
                            insider_pct /= 100.0
                        break
        except Exception as e:
            print(f"  [DataContract] major_holders parse failed for {ticker}: {e}")

        # ── Top-10 institutional concentration ────────────────────────────────
        institutional_top10: Optional[float] = None
        try:
            inst = t.institutional_holders
            if inst is not None and not inst.empty:
                # New API: column is 'pctHeld'; old API used '% Out'
                pct_col = None
                for col in ("pctHeld", "% Out", "Pct Held"):
                    if col in inst.columns:
                        pct_col = col
                        break
                if pct_col:
                    raw_sum = float(inst.head(10)[pct_col].sum())
                    # pctHeld is a fraction (0.07); '% Out' was also fraction
                    institutional_top10 = raw_sum
        except Exception as e:
            print(f"  [DataContract] institutional_holders parse failed for {ticker}: {e}")

        return {"insider_pct": insider_pct, "institutional_top10": institutional_top10}
    except Exception as e:
        print(f"  [DataContract] Holders fetch failed for {ticker}: {e}")
        return {"insider_pct": None, "institutional_top10": None}


# ─── Public API ───────────────────────────────────────────────────────────────

def fetch_inputs(ticker: str, n_quarters: int = DEFAULT_QUARTERS) -> InputBundle:
    """
    Fetch and align all inputs for Agent 4 for the given ticker.

    Tries the DB path first; falls back to yfinance if the ticker is absent.
    R&D gaps are always filled from yfinance regardless of primary source.
    Holder data is always from yfinance.

    Returns InputBundle with all lists oldest → newest and a DataCoverage summary
    that the confidence guardrail will inspect.

    Raises:
        ValueError: if no financial data can be obtained from any source.
    """
    t_upper = ticker.upper()

    # ── DB path ────────────────────────────────────────────────────────────────
    if _db_ticker_exists(t_upper):
        rows = _fetch_core_from_db(t_upper, n_quarters)
        if rows:
            periods = [str(r["period"])[:10] for r in rows]
            revenue = [float(r["total_revenue"] or 0) for r in rows]
            capex   = [abs(float(r["capex"] or 0)) for r in rows]

            rd_db, rd_db_count = _fetch_rd_from_db(t_upper, periods)
            rd, rd_yf_count    = _fill_rd_gaps(rd_db, periods, t_upper)

            holders      = _fetch_holders(t_upper)
            capex_found  = any(v > 0 for v in capex)
            holders_found = holders.get("insider_pct") is not None

            print(
                f"  [DataContract] {t_upper}: {len(periods)}Q from DB  |  "
                f"R&D: {rd_db_count}Q DB + {rd_yf_count}Q yfinance"
            )

            esg_data = fetch_esg_data(t_upper)

            return InputBundle(
                ticker=t_upper,
                periods=periods,
                revenue=revenue,
                rd=rd,
                capex=capex,
                insider_pct=holders.get("insider_pct"),
                institutional_top10=holders.get("institutional_top10"),
                coverage=DataCoverage(
                    quarters_returned=len(periods),
                    rd_quarters_from_db=rd_db_count,
                    rd_quarters_from_yf=rd_yf_count,
                    capex_found=capex_found,
                    holders_found=holders_found,
                    source="db",
                ),
                esg=esg_data,
            )

    # ── yfinance fallback ──────────────────────────────────────────────────────
    print(f"  [DataContract] {t_upper}: not in DB, falling back to yfinance")
    yf_data = _fetch_from_yfinance(t_upper, n_quarters)
    if not yf_data:
        raise ValueError(f"No financial data available for ticker: {t_upper}")

    periods = yf_data["periods"]
    revenue = yf_data["revenue"]
    rd      = yf_data["rd"]
    capex   = yf_data["capex"]
    holders = _fetch_holders(t_upper)

    rd_count     = sum(1 for v in rd if v > 0)
    capex_found  = any(v > 0 for v in capex)
    holders_found = holders.get("insider_pct") is not None

    print(f"  [DataContract] {t_upper}: {len(periods)}Q from yfinance")

    esg_data = fetch_esg_data(t_upper)

    return InputBundle(
        ticker=t_upper,
        periods=periods,
        revenue=revenue,
        rd=rd,
        capex=capex,
        insider_pct=holders.get("insider_pct"),
        institutional_top10=holders.get("institutional_top10"),
        coverage=DataCoverage(
            quarters_returned=len(periods),
            rd_quarters_from_db=0,
            rd_quarters_from_yf=rd_count,
            capex_found=capex_found,
            holders_found=holders_found,
            source="yfinance",
        ),
        esg=esg_data,
    )
