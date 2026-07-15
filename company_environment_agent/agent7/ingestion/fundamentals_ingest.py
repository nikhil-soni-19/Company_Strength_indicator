"""Pull quarterly fundamentals from yfinance and upsert into fundamentals_quarterly."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.connection import get_conn


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _extract_row(label: str, df: pd.DataFrame) -> pd.Series | None:
    for idx in df.index:
        if str(idx).lower().strip() == label.lower().strip():
            return df.loc[idx]
    return None


def ingest_fundamentals(ticker: str, conn=None) -> int:
    """
    Pull quarterly financials and upsert into fundamentals_quarterly.
    Returns number of rows upserted.
    """
    own = conn is None
    if own:
        conn = get_conn()

    try:
        t = yf.Ticker(ticker)
        qf = t.quarterly_financials   # columns = period_end dates
        qi = t.quarterly_income_stmt  # alternate attribute

        # Prefer quarterly_financials; fall back to quarterly_income_stmt
        fin = qf if (qf is not None and not qf.empty) else qi

        if fin is None or fin.empty:
            print(f"  [WARN] No quarterly financials for {ticker}")
            return 0

        rows = []
        for col in fin.columns:
            period_end = pd.Timestamp(col).date()

            def g(label):
                r = _extract_row(label, fin)
                return _safe_float(r[col]) if r is not None else None

            revenue         = g("Total Revenue")
            gross_profit    = g("Gross Profit")
            operating_inc   = g("Operating Income") or g("Total Operating Income As Reported")
            ebitda          = g("EBITDA") or g("Normalized EBITDA")
            net_income      = g("Net Income") or g("Net Income Common Stockholders")

            # Bank: NII and earning assets
            nii = g("Net Interest Income")
            aea = None  # not reliably in yfinance; set NULL

            # REIT: FFO — not in yfinance, leave NULL
            ffo = None

            rows.append((
                ticker, period_end, str(col)[:7],
                revenue, gross_profit, operating_inc, ebitda, net_income,
                nii, aea, ffo,
            ))

        from psycopg2.extras import execute_values
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO fundamentals_quarterly
                  (ticker, period_end, fiscal_quarter,
                   revenue, gross_profit, operating_income, ebitda, net_income,
                   net_interest_income, avg_earning_assets, ffo)
                VALUES %s
                ON CONFLICT (ticker, period_end) DO UPDATE SET
                    fiscal_quarter      = EXCLUDED.fiscal_quarter,
                    revenue             = EXCLUDED.revenue,
                    gross_profit        = EXCLUDED.gross_profit,
                    operating_income    = EXCLUDED.operating_income,
                    ebitda              = EXCLUDED.ebitda,
                    net_income          = EXCLUDED.net_income,
                    net_interest_income = EXCLUDED.net_interest_income,
                    avg_earning_assets  = EXCLUDED.avg_earning_assets,
                    ffo                 = EXCLUDED.ffo
                """,
                rows,
            )
        conn.commit()
        print(f"  {ticker}: {len(rows)} fundamental rows upserted")
        return len(rows)

    finally:
        if own:
            conn.close()
