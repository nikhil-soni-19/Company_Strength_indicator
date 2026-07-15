"""Step 2.3 — Amihud Illiquidity Ratio.

For each day:

.. math::

    A_t = \\frac{|R_t|}{DV_t \\text{ (in millions)}}

where :math:`R_t = (C_t - C_{t-1}) / C_{t-1}`. We then take the mean over a
30-day rolling window.

Interpretation (from the design doc):
* ``A ≈ 0.005`` — highly liquid; takes ~$100M to move price 0.5%.
* ``A ≈ 2.5``   — extremely illiquid; just $1M moves price 2.5%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import AMIHUD_WINDOW


@dataclass
class AmihudResult:
    amihud_30d: Optional[float]
    daily_amihud: pd.Series


def compute_amihud(ohlcv: pd.DataFrame, window: int = AMIHUD_WINDOW) -> AmihudResult:
    if ohlcv.empty:
        return AmihudResult(None, pd.Series(dtype="float64"))

    close = ohlcv["Close"].astype("float64")
    volume = ohlcv["Volume"].astype("float64")

    daily_return = close.pct_change()
    daily_dv_millions = (close * volume) / 1_000_000.0

    safe_dv = daily_dv_millions.replace(0.0, np.nan)
    daily_amihud = (daily_return.abs() / safe_dv).rename("DailyAmihud")

    tail = daily_amihud.dropna().tail(window)
    if len(tail) < min(window, 5):
        amihud_30d: Optional[float] = None
    else:
        amihud_30d = float(tail.mean())

    return AmihudResult(amihud_30d=amihud_30d, daily_amihud=daily_amihud)
