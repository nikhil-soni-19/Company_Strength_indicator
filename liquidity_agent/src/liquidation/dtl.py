"""Step 4 — Days To Liquidate (DTL) at a given hypothetical position size.

.. math::

    DTL = \\frac{N}{ADV \\times PR}

where:

* ``N`` is the *share* position size.
* ``ADV`` is the average daily *share* volume (not dollar volume — Step 2.1's
  ADV$ is a separate quantity).
* ``PR`` is the participation rate (default 10%).

The PDF asks for DTL at 1% and 5% of the *float*. We expose a general
helper plus a convenience wrapper that returns the standard pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    PARTICIPATION_RATE_DEFAULT,
    POSITION_PERCENTAGES,
    SHORT_WINDOW,
)


@dataclass
class DTLResult:
    dtl_by_position_pct: dict[float, Optional[float]]
    participation_rate: float
    adv_shares_30d: Optional[float]
    float_shares: Optional[float]

    @property
    def dtl_1pct(self) -> Optional[float]:
        return self.dtl_by_position_pct.get(0.01)

    @property
    def dtl_5pct(self) -> Optional[float]:
        return self.dtl_by_position_pct.get(0.05)


def _adv_shares(ohlcv: pd.DataFrame, window: int) -> Optional[float]:
    if ohlcv.empty or "Volume" not in ohlcv:
        return None
    tail = ohlcv["Volume"].dropna().tail(window)
    if len(tail) < min(window, 5):
        return None
    return float(tail.mean())


def compute_dtl(
    position_shares: float,
    adv_shares: Optional[float],
    participation_rate: float = PARTICIPATION_RATE_DEFAULT,
) -> Optional[float]:
    """Generic DTL: how many days to exit ``position_shares`` while staying
    under the participation-rate cap."""
    if adv_shares is None or adv_shares <= 0:
        return None
    if participation_rate <= 0:
        raise ValueError("participation_rate must be > 0")
    return float(position_shares) / (adv_shares * participation_rate)


def compute_dtl_for_positions(
    ohlcv: pd.DataFrame,
    float_shares: Optional[float],
    position_pcts: tuple[float, ...] = POSITION_PERCENTAGES,
    participation_rate: float = PARTICIPATION_RATE_DEFAULT,
    window: int = SHORT_WINDOW,
) -> DTLResult:
    """Compute DTL for each of ``position_pcts`` of the company's float."""
    adv = _adv_shares(ohlcv, window)

    results: dict[float, Optional[float]] = {}
    if float_shares is None or float_shares <= 0:
        for pct in position_pcts:
            results[pct] = None
    else:
        for pct in position_pcts:
            n_shares = float_shares * pct
            results[pct] = compute_dtl(n_shares, adv, participation_rate)

    return DTLResult(
        dtl_by_position_pct=results,
        participation_rate=participation_rate,
        adv_shares_30d=adv,
        float_shares=float_shares,
    )
