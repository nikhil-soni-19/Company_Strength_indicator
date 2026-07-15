"""Trading-day helpers.

The PDF explicitly warns: *"gaps in dates will break rolling-window logic
if you use strict calendar-day maths instead of trading-day index maths."*

We therefore drop weekends/NA bars and operate on a clean trading-day index.
"""

from __future__ import annotations

import pandas as pd


def ensure_trading_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a sorted, de-duplicated, NA-free
    trading-day index.

    Anything that is not an actual trading bar (rows where *all* OHLC columns
    are NA, or duplicated timestamps) is dropped, guaranteeing that rolling
    windows operate on adjacent trading days.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    ohlc_cols = [c for c in ("Open", "High", "Low", "Close") if c in out.columns]
    if ohlc_cols:
        out = out.dropna(subset=ohlc_cols, how="all")
    return out


def count_trading_days(df: pd.DataFrame) -> int:
    """Number of usable trading-day rows present in ``df``."""
    return len(ensure_trading_day_index(df))
