"""Step 2.2 — Volume Coefficient of Variation.

.. math::

    CV_k = \\frac{\\sigma(V)}{\\mu(V)}

The PDF interprets this as:

* ``CV < 0.4``  → steady, predictable trading traffic (safe).
* ``CV > 1.0``  → erratic; ADV is a "statistical illusion driven by a few
                  hyperactive days".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import LONG_WINDOW, SHORT_WINDOW


@dataclass
class VolumeCVResult:
    volume_cv_30d: Optional[float]
    volume_cv_90d: Optional[float]


def compute_volume_cv(
    ohlcv: pd.DataFrame,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
) -> VolumeCVResult:
    if ohlcv.empty:
        return VolumeCVResult(None, None)

    volume = ohlcv["Volume"].astype("float64")
    return VolumeCVResult(
        volume_cv_30d=_cv(volume, short_window),
        volume_cv_90d=_cv(volume, long_window),
    )


def _cv(series: pd.Series, window: int) -> Optional[float]:
    tail = series.dropna().tail(window)
    if len(tail) < min(window, 5):
        return None
    mean = float(tail.mean())
    if mean <= 0:
        return None
    std = float(tail.std(ddof=1))
    if np.isnan(std):
        return None
    return std / mean
