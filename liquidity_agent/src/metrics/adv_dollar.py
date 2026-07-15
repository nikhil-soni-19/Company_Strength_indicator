"""Step 2.1 — Average Daily Dollar Volume (ADV$).

Implements the calculation from the design document:

1. ``DV_t = C_t * V_t``
2. Clip ``DV_t`` at the 99th percentile to neutralise anomalies.
3. Compute 30-day and 90-day trailing means.

The downstream interpretation is also captured in the design doc:
*"If ADV$30 << ADV$90, it flags a structural draining of liquidity,
signaling your engine to expect higher slippage than historical baselines imply."*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import LONG_WINDOW, OUTLIER_CLIP_PERCENTILE, SHORT_WINDOW
from src.utils import clip_percentile


@dataclass
class ADVDollarResult:
    adv_dollar_30d: Optional[float]
    adv_dollar_90d: Optional[float]
    daily_dollar_volume: pd.Series
    daily_dollar_volume_clipped: pd.Series

    @property
    def liquidity_drain_ratio(self) -> Optional[float]:
        """``ADV$30 / ADV$90``. Values well below 1.0 indicate draining liquidity."""
        if not self.adv_dollar_30d or not self.adv_dollar_90d:
            return None
        if self.adv_dollar_90d == 0:
            return None
        return self.adv_dollar_30d / self.adv_dollar_90d


def compute_adv_dollar(
    ohlcv: pd.DataFrame,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
    clip_q: float = OUTLIER_CLIP_PERCENTILE,
) -> ADVDollarResult:
    """Compute clipped 30-day and 90-day Average Daily Dollar Volume."""
    if ohlcv.empty:
        empty = pd.Series(dtype="float64")
        return ADVDollarResult(None, None, empty, empty)

    close = ohlcv["Close"]
    volume = ohlcv["Volume"]
    daily_dv = (close * volume).rename("DailyDollarVolume")
    daily_dv_clipped = clip_percentile(daily_dv, upper_q=clip_q).rename("DailyDollarVolumeClipped")

    adv30 = _tail_mean(daily_dv_clipped, short_window)
    adv90 = _tail_mean(daily_dv_clipped, long_window)

    return ADVDollarResult(
        adv_dollar_30d=adv30,
        adv_dollar_90d=adv90,
        daily_dollar_volume=daily_dv,
        daily_dollar_volume_clipped=daily_dv_clipped,
    )


def _tail_mean(series: pd.Series, window: int) -> Optional[float]:
    tail = series.dropna().tail(window)
    if len(tail) < min(window, 5):
        return None
    return float(tail.mean())
