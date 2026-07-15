"""OHLCV ingestion from yfinance into ohlcv_daily."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.connection import get_conn

# Tickers whose close == adj_close (no dividend adjustment needed)
_RAW_CLOSE_PREFIXES = ("^",)
_RAW_CLOSE_SUFFIXES = ("=F",)

DEFAULT_MACRO_LONG = ["^VIX", "^TNX"]          # 5-year history
DEFAULT_UNIVERSE = [
    # SPDR ETFs
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
    # Indexes
    "^GSPC", "^IXIC", "^DJI",
    # Commodities
    "CL=F", "GC=F", "HG=F", "NG=F",
    # FX
    "DX-Y.NYB",
]

# Metadata for bootstrap tickers
_TICKER_META = {
    "XLK":     ("SPDR Technology ETF",               "ETF",       "Information Technology",  "XLK"),
    "XLF":     ("SPDR Financials ETF",                "ETF",       "Financials",              "XLF"),
    "XLE":     ("SPDR Energy ETF",                    "ETF",       "Energy",                  "XLE"),
    "XLV":     ("SPDR Health Care ETF",               "ETF",       "Health Care",             "XLV"),
    "XLY":     ("SPDR Consumer Discret ETF",          "ETF",       "Consumer Discretionary",  "XLY"),
    "XLP":     ("SPDR Consumer Staples ETF",          "ETF",       "Consumer Staples",        "XLP"),
    "XLI":     ("SPDR Industrials ETF",               "ETF",       "Industrials",             "XLI"),
    "XLB":     ("SPDR Materials ETF",                 "ETF",       "Materials",               "XLB"),
    "XLU":     ("SPDR Utilities ETF",                 "ETF",       "Utilities",               "XLU"),
    "XLRE":    ("SPDR Real Estate ETF",               "ETF",       "Real Estate",             "XLRE"),
    "XLC":     ("SPDR Comm Services ETF",             "ETF",       "Communication Services",  "XLC"),
    "^GSPC":   ("S&P 500 Index",                      "INDEX",     None,                      None),
    "^IXIC":   ("NASDAQ Composite",                   "INDEX",     None,                      None),
    "^DJI":    ("Dow Jones Industrial Average",       "INDEX",     None,                      None),
    "^VIX":    ("CBOE Volatility Index",              "VOL",       None,                      None),
    "^TNX":    ("10-Year Treasury Yield",             "RATE",      None,                      None),
    "DX-Y.NYB":("US Dollar Index",                    "FX",        None,                      None),
    "CL=F":    ("Crude Oil Futures",                  "COMMODITY", "Energy",                  None),
    "GC=F":    ("Gold Futures",                       "COMMODITY", "Materials",               None),
    "HG=F":    ("Copper Futures",                     "COMMODITY", "Materials",               None),
    "NG=F":    ("Natural Gas Futures",                "COMMODITY", "Energy",                  None),
}


def _is_raw_close(ticker: str) -> bool:
    return ticker.startswith(_RAW_CLOSE_PREFIXES) or ticker.endswith(_RAW_CLOSE_SUFFIXES)


def _normalize_ohlcv_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance multi-index columns to standard OHLCV names."""
    if df.empty:
        return df

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if ticker in out.columns.get_level_values(0):
            out = out[ticker].copy()
        else:
            out.columns = out.columns.get_level_values(-1)

    out.columns = [str(c) for c in out.columns]
    return out


def _upsert_metadata(conn, ticker: str) -> None:
    if ticker not in _TICKER_META:
        return
    name, asset_class, sector, etf = _TICKER_META[ticker]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ticker_metadata (ticker, name, asset_class, gics_sector, sector_etf)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
                name        = EXCLUDED.name,
                asset_class = EXCLUDED.asset_class,
                gics_sector = EXCLUDED.gics_sector,
                sector_etf  = EXCLUDED.sector_etf
            """,
            (ticker, name, asset_class, sector, etf),
        )


def ingest_ohlcv(
    tickers: list[str],
    start: date,
    end: date,
    conn=None,
) -> int:
    """
    Download OHLCV from yfinance and upsert into ohlcv_daily.
    Returns total rows upserted.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_conn()

    total = 0
    try:
        raw = yf.download(
            tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        if isinstance(tickers, str) or len(tickers) == 1:
            ticker_list = [tickers] if isinstance(tickers, str) else tickers
            frames = {ticker_list[0]: raw}
        else:
            frames = {t: raw[t] for t in tickers if t in raw.columns.get_level_values(0)}

        for ticker, df in frames.items():
            if df.empty:
                print(f"  [WARN] No data returned for {ticker}")
                continue

            df = _normalize_ohlcv_frame(df, ticker)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Forward-fill at most 2 consecutive NaN days; alert on longer gaps
            adj_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            gaps = df[adj_col].isna()
            consecutive = gaps.groupby((~gaps).cumsum()).cumsum()
            long_gaps = consecutive[consecutive > 2]
            if not long_gaps.empty:
                print(
                    f"  [WARN] {ticker} has {len(long_gaps)} trading day(s) with "
                    f">2 consecutive missing values — not forward-filled."
                )
            df = df.ffill(limit=2)

            # Use close as adj_close for index/commodity tickers
            use_raw = _is_raw_close(ticker)

            _upsert_metadata(conn, ticker)

            rows = []
            for dt, row in df.iterrows():
                close_val = float(row["Close"]) if not pd.isna(row.get("Close", float("nan"))) else None
                adj_close_val = (
                    close_val if use_raw
                    else (float(row[adj_col]) if not pd.isna(row.get(adj_col, float("nan"))) else close_val)
                )
                rows.append((
                    ticker,
                    dt.date(),
                    float(row["Open"])   if not pd.isna(row.get("Open",   float("nan"))) else None,
                    float(row["High"])   if not pd.isna(row.get("High",   float("nan"))) else None,
                    float(row["Low"])    if not pd.isna(row.get("Low",    float("nan"))) else None,
                    close_val,
                    adj_close_val,
                    int(row["Volume"])   if not pd.isna(row.get("Volume", float("nan"))) else None,
                ))

            with conn.cursor() as cur:
                from psycopg2.extras import execute_values
                execute_values(
                    cur,
                    """
                    INSERT INTO ohlcv_daily (ticker, date, open, high, low, close, adj_close, volume)
                    VALUES %s
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        open      = EXCLUDED.open,
                        high      = EXCLUDED.high,
                        low       = EXCLUDED.low,
                        close     = EXCLUDED.close,
                        adj_close = EXCLUDED.adj_close,
                        volume    = EXCLUDED.volume
                    """,
                    rows,
                )
            conn.commit()
            total += len(rows)
            print(f"  {ticker}: {len(rows)} rows upserted")

    finally:
        if own_conn:
            conn.close()

    return total


def bootstrap_universe(years_long: int = 5, months_short: int = 6) -> int:
    """Ingest the full default universe with correct lookback windows."""
    today = date.today()
    end = today

    start_long  = date(today.year - years_long, today.month, today.day)
    start_short = date(today.year, today.month, today.day) - timedelta(days=months_short * 30)

    print(f"Ingesting macro tickers (5y): {DEFAULT_MACRO_LONG}")
    total = ingest_ohlcv(DEFAULT_MACRO_LONG, start_long, end)

    print(f"Ingesting universe (6mo): {DEFAULT_UNIVERSE}")
    total += ingest_ohlcv(DEFAULT_UNIVERSE, start_short, end)

    return total
