"""Load price and fundamental data live from yfinance for Layer 1 computations.

Neon DB is used only for risk_factors (10-K embeddings) and environment_runs.
All market data is fetched live from yfinance; ticker metadata from peer_map.yaml.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

# ── Peer map cache ────────────────────────────────────────────────────────────
_PEER_MAP_PATH = Path(__file__).parent.parent / "config" / "peer_map.yaml"
_peer_map_cache: dict | None = None


def _get_peer_map() -> dict:
    """Load and cache the full GICS peer map keyed by ticker."""
    global _peer_map_cache
    if _peer_map_cache is None:
        with open(_PEER_MAP_PATH) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "constituents" in data:
            # Full GICS format: list of constituent dicts
            _peer_map_cache = {c["ticker"]: c for c in data["constituents"]}
        else:
            # Legacy simple format: {ticker: [peers]}
            _peer_map_cache = {
                t: {"ticker": t, "peers": p}
                for t, p in data.items()
                if not str(t).startswith("#")
            }
    return _peer_map_cache


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _extract_row(label: str, df: pd.DataFrame):
    for idx in df.index:
        if str(idx).lower().strip() == label.lower().strip():
            return df.loc[idx]
    return None


# ── yfinance MultiIndex normaliser ───────────────────────────────────────────
def _to_series(df: pd.DataFrame, col: str, ticker: str) -> pd.Series:
    """
    Extract a single price column from a yfinance DataFrame robustly.

    yfinance >=0.2.38 returns MultiIndex columns (col, ticker) even for a
    single ticker download, so df["Adj Close"] gives a one-column DataFrame
    instead of a Series.  This helper handles both layouts.
    """
    if df.empty:
        return pd.Series(dtype=float, name=ticker)

    # MultiIndex columns: ("Adj Close", "AAPL") style
    if isinstance(df.columns, pd.MultiIndex):
        if col in df.columns.get_level_values(0):
            sub = df[col]  # DataFrame with ticker columns
            # Pick the right ticker column, or first if not found
            if ticker in sub.columns:
                s = sub[ticker]
            else:
                s = sub.iloc[:, 0]
        else:
            return pd.Series(dtype=float, name=ticker)
    else:
        # Flat columns
        if col not in df.columns:
            col = "Close"
        raw = df[col]
        # Still could be a 1-col DataFrame from some yfinance builds
        s = raw.squeeze() if isinstance(raw, pd.DataFrame) else raw

    s = s.dropna()
    s.index = pd.DatetimeIndex(s.index)
    s.name = ticker
    return s


# ── Price data ────────────────────────────────────────────────────────────────
def load_prices(
    ticker: str,
    start: date,
    end: date,
) -> pd.Series:
    """Return adj_close series indexed by date (live from yfinance)."""
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    # Prefer Adj Close; fall back to Close
    flat_cols = (df.columns.get_level_values(0)
                 if isinstance(df.columns, pd.MultiIndex)
                 else df.columns)
    adj_col = "Adj Close" if "Adj Close" in flat_cols else "Close"
    return _to_series(df, adj_col, ticker)


def load_prices_multi(
    tickers: list[str],
    start: date,
    end: date,
) -> dict[str, pd.Series]:
    """Return dict of adj_close series for multiple tickers (live from yfinance)."""
    if not tickers:
        return {}
    if len(tickers) == 1:
        return {tickers[0]: load_prices(tickers[0], start, end)}

    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    result = {}
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.get_level_values(1):
                    result[t] = pd.Series(dtype=float, name=t)
                    continue
                sub = raw.xs(t, axis=1, level=1)  # flat DataFrame for this ticker
            else:
                sub = raw  # single-ticker flat layout

            flat = sub.columns.tolist()
            adj_col = "Adj Close" if "Adj Close" in flat else "Close"
            s = sub[adj_col].dropna()
            s.index = pd.DatetimeIndex(s.index)
            s.name = t
            result[t] = s
        except Exception:
            result[t] = pd.Series(dtype=float, name=t)
    return result


# ── Fundamental data ──────────────────────────────────────────────────────────
def _parse_quarter(ticker: str, fin: pd.DataFrame, col) -> dict:
    """Extract one quarter's fundamental fields from a yfinance financials DataFrame."""
    period_end = pd.Timestamp(col).date()

    def g(label):
        r = _extract_row(label, fin)
        return _safe_float(r[col]) if r is not None else None

    return {
        "ticker":              ticker,
        "period_end":          period_end,
        "fiscal_quarter":      str(col)[:7],
        "revenue":             g("Total Revenue"),
        "gross_profit":        g("Gross Profit"),
        "operating_income":    g("Operating Income") or g("Total Operating Income As Reported"),
        "ebitda":              g("EBITDA") or g("Normalized EBITDA"),
        "net_income":          g("Net Income") or g("Net Income Common Stockholders"),
        "net_interest_income": g("Net Interest Income"),
        "avg_earning_assets":  None,  # not in yfinance
        "ffo":                 None,  # not in yfinance
    }


def load_latest_fundamentals(ticker: str) -> dict | None:
    """Return the most recent quarterly fundamentals (live from yfinance)."""
    t = yf.Ticker(ticker)
    fin = t.quarterly_financials
    if fin is None or fin.empty:
        fin = t.quarterly_income_stmt
    if fin is None or fin.empty:
        return None
    col = fin.columns[0]
    return _parse_quarter(ticker, fin, col)


def load_ttm_quarters(ticker: str, n: int = 4) -> list[dict]:
    """Return last n quarterly fundamental rows newest-first (live from yfinance).
    yfinance reliably returns 4 quarters; used for margin calculation only."""
    t = yf.Ticker(ticker)
    fin = t.quarterly_financials
    if fin is None or fin.empty:
        fin = t.quarterly_income_stmt
    if fin is None or fin.empty:
        return []
    cols = fin.columns[:n]
    return [_parse_quarter(ticker, fin, col) for col in cols]


def load_annual_revenue(ticker: str) -> list[float | None]:
    """Return last 2 years of annual revenue from yfinance income_stmt.

    yfinance annual statements reliably return 4 fiscal years.
    Returns [revenue_year_0, revenue_year_1] newest-first (floats or None).
    Used for YoY revenue growth: (year_0 / year_1) - 1.
    """
    t = yf.Ticker(ticker)
    fin = t.income_stmt          # annual
    if fin is None or fin.empty:
        fin = t.financials       # fallback alias
    if fin is None or fin.empty:
        return []
    for idx in fin.index:
        label = str(idx).lower().strip()
        if label in ("total revenue", "revenue"):
            rows = [_safe_float(fin.loc[idx, c]) for c in fin.columns[:2]]
            return rows
    return []


# ── Ticker metadata ───────────────────────────────────────────────────────────
def load_ticker_metadata(ticker: str) -> dict | None:
    """Return GICS metadata for a ticker from peer_map.yaml."""
    entry = _get_peer_map().get(ticker)
    if entry is None:
        return None
    return {
        "ticker":      ticker,
        "name":        entry.get("name", ticker),
        "asset_class": "EQUITY",
        "gics_sector": entry.get("gics_sector"),
        "sector_etf":  entry.get("sector_etf"),
        "currency":    "USD",
    }


def load_peers(ticker: str) -> list[str]:
    """Return peer tickers from peer_map.yaml.

    Filters out any non-string values that PyYAML may have coerced from
    YAML 1.1 boolean keywords (e.g. unquoted 'ON' → True, 'NO' → False).
    """
    entry = _get_peer_map().get(ticker)
    if entry is None:
        return []
    peers = entry.get("peers", []) or []
    clean = []
    for p in peers:
        if isinstance(p, str):
            clean.append(p)
        else:
            print(f"  [WARN] peer_map.yaml: non-string peer {p!r} for {ticker} — "
                  f"quote it in the YAML (e.g. 'ON' → '\"ON\"')")
    return clean


# ── R&D and CapEx intensity ───────────────────────────────────────────────────
def load_rd_revenue_ratio(ticker: str) -> float | None:
    """Return R&D / Revenue from most recent annual income statement (live yfinance).
    Returns None if R&D line is not reported (common for non-R&D-intensive sectors)."""
    t = yf.Ticker(ticker)
    fin = t.income_stmt
    if fin is None or fin.empty:
        fin = t.financials
    if fin is None or fin.empty:
        return None
    col = fin.columns[0]  # most recent fiscal year

    rd = None
    rev = None
    for idx in fin.index:
        label = str(idx).lower().strip()
        if "research" in label and "development" in label:
            rd = _safe_float(fin.loc[idx, col])
        if label in ("total revenue", "revenue"):
            rev = _safe_float(fin.loc[idx, col])

    if rd is not None and rev and rev != 0:
        return abs(rd) / abs(rev)
    return None


def load_capex_revenue_ratio(ticker: str) -> float | None:
    """Return |CapEx| / Revenue from most recent annual cash flow + income statement (live yfinance)."""
    t = yf.Ticker(ticker)
    cf = t.cashflow
    fin = t.income_stmt
    if cf is None or cf.empty or fin is None or fin.empty:
        return None

    cf_col = cf.columns[0]
    capex = None
    for idx in cf.index:
        label = str(idx).lower().strip()
        if any(k in label for k in ("capital expenditure", "purchase of ppe",
                                     "purchases of property", "capital expenditures")):
            capex = _safe_float(cf.loc[idx, cf_col])

    fin_col = fin.columns[0]
    rev = None
    for idx in fin.index:
        label = str(idx).lower().strip()
        if label in ("total revenue", "revenue"):
            rev = _safe_float(fin.loc[idx, fin_col])

    if capex is not None and rev and rev != 0:
        return abs(capex) / abs(rev)
    return None


def load_macro_etf_prices(
    start: date,
    end: date,
) -> dict[str, pd.Series]:
    """
    Fetch macro ETF price series used by PESTEL dimensions.

    Tickers fetched:
      UUP   – Invesco DB US Dollar Index (USD strength proxy for Political)
      HYG   – iShares HY Bond ETF  ]
      IEF   – iShares 7-10yr Tsy  ] ratio = credit-spread proxy for Economic
      TIP   – iShares TIPS ETF    ] TIP/IEF ratio = breakeven inflation proxy
      XLY   – Consumer Discretionary ETF ]  ratio = consumer-sentiment
      XLP   – Consumer Staples ETF       ]  proxy for Social
    """
    tickers = ["UUP", "HYG", "IEF", "TIP", "XLY", "XLP"]
    return load_prices_multi(tickers, start, end)


# ── Macro rates ───────────────────────────────────────────────────────────────
def load_tnx_rate(end: date) -> float:
    """Return the latest ^TNX yield as decimal (e.g. 4.2 → 0.042) live from yfinance."""
    df = yf.download(
        "^TNX",
        start=(end - timedelta(days=10)).isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        return 0.04
    close = _to_series(df, "Close", "^TNX")
    if close.empty:
        return 0.04
    return float(close.iloc[-1]) / 100.0
