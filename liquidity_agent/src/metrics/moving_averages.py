"""Step 1 — Simple and Exponential Moving Averages.

The PDF discusses both forms and notes that SMA "drops data off a cliff"
when a spike rolls out of the window, whereas EMA decays smoothly. We
expose both so downstream modules can pick the most appropriate one.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` trading days.

    .. math::

        SMA_t = \\frac{1}{n} \\sum_{i=0}^{n-1} P_{t-i}
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average with smoothing factor :math:`\\alpha = 2/(n+1)`.

    .. math::

        EMA_t = P_t \\cdot \\alpha + EMA_{t-1} \\cdot (1 - \\alpha)
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    return series.ewm(span=window, adjust=False, min_periods=window).mean()
