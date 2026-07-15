"""Outlier-handling utilities used across the metric pipeline."""

from __future__ import annotations

import pandas as pd


def clip_percentile(series: pd.Series, upper_q: float = 0.99) -> pd.Series:
    """Clip a numeric series at its ``upper_q`` quantile.

    Used in Step 2.1 of the design doc to neutralise a single anomalous
    dollar-volume day (e.g. earnings, meme-stock squeeze) before computing
    rolling averages.

    Parameters
    ----------
    series : pd.Series
        Numeric series to clip. NaNs are preserved.
    upper_q : float, default 0.99
        Upper quantile threshold in ``[0, 1]``.

    Returns
    -------
    pd.Series
        Series with values above the threshold replaced by the threshold.
    """
    if not 0.0 < upper_q <= 1.0:
        raise ValueError(f"upper_q must be in (0, 1], got {upper_q}")
    if series.dropna().empty:
        return series.copy()
    cap = series.quantile(upper_q)
    return series.clip(upper=cap)
