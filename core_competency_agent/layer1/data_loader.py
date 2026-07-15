"""
Fetch quarterly financial data for the company (8Q) and peers (latest Q).

Primary source: Neon DB — fundamentals.{ticker}_precomputed_metrics (up to 12Q).
Fallback:       yfinance (capped at 4Q by Yahoo's API).

Returns standardised dicts regardless of which fields Yahoo provides.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import yfinance as yf
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent))

_PEER_MAP_PATH = Path(__file__).parent.parent / "config" / "peer_map.yaml"
_PEER_MAP: dict | None = None

MAX_PEERS = 6  # cap to avoid rate-limiting

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_BgdTyxpXW3q4@ep-bitter-boat-aq1v8xns.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require",
)

# Cache of tickers confirmed to exist in Neon
_DB_TICKER_CACHE: set[str] = set()
_DB_TICKER_CACHE_LOADED = False


def _get_conn():
    return psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _db_ticker_exists(ticker: str) -> bool:
    """Check if fundamentals.{ticker}_precomputed_metrics exists in Neon."""
    global _DB_TICKER_CACHE, _DB_TICKER_CACHE_LOADED
    if not _DB_TICKER_CACHE_LOADED:
        try:
            with _get_conn() as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'fundamentals'
                      AND tablename LIKE '%_precomputed_metrics'
                """)
                _DB_TICKER_CACHE = {
                    r["tablename"].replace("_precomputed_metrics", "").upper()
                    for r in cur.fetchall()
                }
            _DB_TICKER_CACHE_LOADED = True
        except Exception as e:
            print(f"  [DataLoader] DB ticker cache failed: {e}")
            return False
    return ticker.upper() in _DB_TICKER_CACHE


def _fetch_tax_prov_from_db(ticker: str, periods: list[str]) -> list[float]:
    """
    Fetch incomeTaxExpense from fundamentals.{ticker}_income_statement
    and align to the given periods (format: 'YYYY-MM-DD').
    Falls back to 0.0 per quarter if missing — compute.py uses 21% statutory rate.
    """
    table = f"fundamentals.{ticker.lower()}_income_statement"
    try:
        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table} WHERE field = 'incomeTaxExpense'")
            row = cur.fetchone()
        if not row:
            return [0.0] * len(periods)
        # Income statement columns are named like '2026_03_31'; periods are 'YYYY-MM-DD'
        result = []
        for period in periods:
            col = period.replace("-", "_")
            val = row.get(col)
            result.append(float(val) if val is not None else 0.0)
        return result
    except Exception as e:
        print(f"  [DataLoader] Tax prov fetch failed for {ticker}: {e}")
        return [0.0] * len(periods)


def _fetch_from_db(ticker: str, n_quarters: int) -> Optional[list[dict]]:
    """
    Pull up to n_quarters rows from fundamentals.{ticker}_precomputed_metrics,
    ordered oldest → newest.
    """
    table = f"fundamentals.{ticker.lower()}_precomputed_metrics"
    try:
        with _get_conn() as conn, conn.cursor() as cur:
            cur.execute(f"""
                SELECT period, total_revenue, gross_profit, operating_income,
                       net_income, operating_cash_flow, capex, total_debt,
                       stockholders_equity
                FROM (
                    SELECT * FROM {table}
                    ORDER BY period DESC
                    LIMIT %s
                ) sub
                ORDER BY period ASC
            """, (n_quarters,))
            rows = cur.fetchall()
        return [dict(r) for r in rows] if rows else None
    except Exception as e:
        print(f"  [DataLoader] DB fetch failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Peer map
# ---------------------------------------------------------------------------

def _load_peer_map() -> dict:
    global _PEER_MAP
    if _PEER_MAP is None:
        raw = yaml.safe_load(_PEER_MAP_PATH.read_text())
        _PEER_MAP = {entry["ticker"]: entry for entry in raw.get("constituents", [])}
    return _PEER_MAP


def get_peers(ticker: str) -> list[str]:
    pm = _load_peer_map()
    entry = pm.get(ticker.upper())
    if not entry:
        return []
    peers = entry.get("peers", []) or []
    return [str(p) for p in peers[:MAX_PEERS]]


# ---------------------------------------------------------------------------
# yfinance helpers (fallback)
# ---------------------------------------------------------------------------

def _safe_get(df: pd.DataFrame, *keys) -> list[float]:
    n = len(df)
    for k in keys:
        if k in df.columns:
            return df[k].fillna(0).tolist()
    return [0.0] * n


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

def fetch_company_8q(ticker: str, n_quarters: int = 8) -> dict:
    """
    Fetch n_quarters of financials for the company. Oldest→newest.
    Uses Neon DB if available (up to 12Q), falls back to yfinance (4Q cap).
    tax_prov is not stored in DB — returns zeros, compute.py uses 21% fallback.
    """
    # --- DB path ---
    if _db_ticker_exists(ticker):
        rows = _fetch_from_db(ticker, n_quarters)
        if rows:
            periods      = [str(r["period"])[:10] for r in rows]
            revenue      = [float(r["total_revenue"]      or 0) for r in rows]
            gross_profit = [float(r["gross_profit"]       or 0) for r in rows]
            op_income    = [float(r["operating_income"]   or 0) for r in rows]
            net_income   = [float(r["net_income"]         or 0) for r in rows]
            ocf          = [float(r["operating_cash_flow"] or 0) for r in rows]
            capex        = [abs(float(r["capex"]          or 0)) for r in rows]
            total_debt   = [float(r["total_debt"]         or 0) for r in rows]
            total_equity = [float(r["stockholders_equity"] or 0) for r in rows]
            n = len(rows)
            tax_prov = _fetch_tax_prov_from_db(ticker, periods)
            print(f"  [DataLoader] {ticker}: loaded {n}Q from Neon DB")
            return {
                "ticker":       ticker,
                "periods":      periods,
                "n_quarters":   n,
                "revenue":      revenue,
                "gross_profit": gross_profit,
                "op_income":    op_income,
                "net_income":   net_income,
                "tax_prov":     tax_prov,
                "ocf":          ocf,
                "capex":        capex,
                "total_debt":   total_debt,
                "total_equity": total_equity,
            }

    # --- yfinance fallback ---
    print(f"  [DataLoader] {ticker}: not in Neon DB, falling back to yfinance (4Q cap)")
    t = yf.Ticker(ticker)
    income   = t.quarterly_income_stmt
    cashflow = t.quarterly_cashflow
    balance  = t.quarterly_balance_sheet

    if income is None or income.empty:
        raise ValueError(f"No income data for {ticker}")

    income = income.T.sort_index().tail(n_quarters)
    idx    = income.index

    def _align(df):
        if df is not None and not df.empty:
            return df.T.sort_index().reindex(idx, fill_value=0)
        return pd.DataFrame(0.0, index=idx, columns=[])

    cashflow = _align(cashflow)
    balance  = _align(balance)
    n        = len(idx)

    revenue      = _safe_get(income, "Total Revenue", "Revenue")
    cogs         = _safe_get(income, "Cost Of Revenue", "Cost of Revenue")
    gross_profit = _safe_get(income, "Gross Profit")
    op_income    = _safe_get(income, "Operating Income", "EBIT")
    net_income   = _safe_get(income, "Net Income", "Net Income Common Stockholders")
    tax_prov     = _safe_get(income, "Tax Provision", "Income Tax Expense")

    ocf   = _safe_get(cashflow, "Operating Cash Flow", "Cash From Operations")
    capex = [abs(v) for v in _safe_get(cashflow, "Capital Expenditure", "Purchases Of Property Plant And Equipment")]

    total_debt   = _safe_get(balance, "Total Debt", "Long Term Debt")
    total_equity = _safe_get(balance, "Total Stockholders Equity", "Stockholders Equity", "Common Stock Equity")

    if all(v == 0 for v in gross_profit):
        gross_profit = [r - c for r, c in zip(revenue, cogs)]

    return {
        "ticker":       ticker,
        "periods":      [str(d)[:10] for d in idx.tolist()],
        "n_quarters":   n,
        "revenue":      revenue,
        "gross_profit": gross_profit,
        "op_income":    op_income,
        "net_income":   net_income,
        "tax_prov":     tax_prov,
        "ocf":          ocf,
        "capex":        capex,
        "total_debt":   total_debt,
        "total_equity": total_equity,
    }


def fetch_peer_latest(ticker: str) -> Optional[dict]:
    """
    Fetch latest 4 quarters of financials for a peer.
    Uses Neon DB if available, falls back to yfinance.
    """
    # --- DB path ---
    if _db_ticker_exists(ticker):
        rows = _fetch_from_db(ticker, n_quarters=4)
        if rows:
            revenue      = [float(r["total_revenue"]      or 0) for r in rows]
            gross_profit = [float(r["gross_profit"]       or 0) for r in rows]
            op_income    = [float(r["operating_income"]   or 0) for r in rows]
            net_income   = [float(r["net_income"]         or 0) for r in rows]
            ocf          = [float(r["operating_cash_flow"] or 0) for r in rows]
            capex        = [abs(float(r["capex"]          or 0)) for r in rows]
            total_debt   = [float(r["total_debt"]         or 0) for r in rows]
            total_equity = [float(r["stockholders_equity"] or 0) for r in rows]
            return {
                "ticker":       ticker,
                "revenue":      revenue,
                "gross_profit": gross_profit,
                "op_income":    op_income,
                "net_income":   net_income,
                "tax_prov":     [0.0] * len(rows),
                "ocf":          ocf,
                "capex":        capex,
                "total_debt":   total_debt,
                "total_equity": total_equity,
            }

    # --- yfinance fallback ---
    try:
        t = yf.Ticker(ticker)
        income   = t.quarterly_income_stmt
        cashflow = t.quarterly_cashflow
        balance  = t.quarterly_balance_sheet

        if income is None or income.empty:
            return None

        income = income.T.sort_index().tail(4)
        idx    = income.index

        def _align(df):
            if df is not None and not df.empty:
                return df.T.sort_index().reindex(idx, fill_value=0)
            return pd.DataFrame(0.0, index=idx, columns=[])

        cashflow = _align(cashflow)
        balance  = _align(balance)

        revenue      = _safe_get(income, "Total Revenue", "Revenue")
        cogs         = _safe_get(income, "Cost Of Revenue", "Cost of Revenue")
        gross_profit = _safe_get(income, "Gross Profit")
        op_income    = _safe_get(income, "Operating Income", "EBIT")
        net_income   = _safe_get(income, "Net Income", "Net Income Common Stockholders")
        tax_prov     = _safe_get(income, "Tax Provision", "Income Tax Expense")
        ocf          = _safe_get(cashflow, "Operating Cash Flow", "Cash From Operations")
        capex        = [abs(v) for v in _safe_get(cashflow, "Capital Expenditure")]
        total_debt   = _safe_get(balance, "Total Debt", "Long Term Debt")
        total_equity = _safe_get(balance, "Total Stockholders Equity", "Stockholders Equity", "Common Stock Equity")

        if all(v == 0 for v in gross_profit):
            gross_profit = [r - c for r, c in zip(revenue, cogs)]

        return {
            "ticker":       ticker,
            "revenue":      revenue,
            "gross_profit": gross_profit,
            "op_income":    op_income,
            "net_income":   net_income,
            "tax_prov":     tax_prov,
            "ocf":          ocf,
            "capex":        capex,
            "total_debt":   total_debt,
            "total_equity": total_equity,
        }
    except Exception as e:
        print(f"  [DataLoader] Peer {ticker} failed: {e}")
        return None


def fetch_insider_ownership(ticker: str) -> dict:
    """
    Pull insider ownership % and top-10 institutional concentration from yfinance.
    Returns floats in range [0, 1] or None on failure.
    """
    try:
        t = yf.Ticker(ticker)
        holders = t.major_holders
        insider_pct = None
        if holders is not None and not holders.empty:
            for i in range(len(holders)):
                row = holders.iloc[i]
                label = str(row.iloc[1]).lower() if len(row) > 1 else ""
                if "insider" in label:
                    try:
                        insider_pct = float(str(row.iloc[0]).replace("%", "")) / 100
                    except (ValueError, TypeError):
                        pass
                    break

        institutional_top10 = None
        inst = t.institutional_holders
        if inst is not None and not inst.empty and "% Out" in inst.columns:
            top10 = inst.head(10)["% Out"].sum()
            institutional_top10 = float(top10)

        return {
            "insider_pct": insider_pct,
            "institutional_top10": institutional_top10,
        }
    except Exception as e:
        print(f"  [DataLoader] Insider fetch failed for {ticker}: {e}")
        return {"insider_pct": None, "institutional_top10": None}


def detect_leadership_change(ticker: str) -> dict:
    """
    Detect recent CEO/CFO changes via yfinance company officers.
    Returns {detected: bool, description: str|None}.
    This is a best-effort heuristic — yfinance doesn't provide change dates.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        officers = info.get("companyOfficers", [])
        roles = [o.get("title", "").lower() for o in officers]
        has_ceo = any("chief executive" in r or "ceo" in r for r in roles)
        has_cfo = any("chief financial" in r or "cfo" in r for r in roles)
        if not has_ceo:
            return {"detected": True, "description": "CEO role appears vacant or unlisted"}
        return {"detected": False, "description": None}
    except Exception:
        return {"detected": False, "description": None}
