"""Shared pytest fixtures.

We synthesise OHLCV data with known statistical properties so every metric
in the pipeline has a deterministic, network-free target.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """100 trading days of stable, liquid OHLCV data."""
    rng = np.random.default_rng(seed=42)
    dates = pd.bdate_range(end="2026-05-20", periods=100)
    close = pd.Series(100.0 + rng.normal(0, 0.5, 100).cumsum(), index=dates)
    volume = pd.Series(rng.normal(1_000_000, 50_000, 100).clip(min=500_000), index=dates)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": volume,
        }
    )


@pytest.fixture
def thin_float_ohlcv() -> pd.DataFrame:
    """A low-volume, highly erratic stock — the kind that should land in Tier 4.

    Multiplicative random walk so prices never collapse to a floor, which would
    artificially zero out daily returns (and therefore the Amihud metric).
    """
    rng = np.random.default_rng(seed=7)
    dates = pd.bdate_range(end="2026-05-20", periods=100)
    log_returns = rng.normal(0.0, 0.06, 100)
    close = pd.Series(3.0 * np.exp(np.cumsum(log_returns)), index=dates)
    volume_base = rng.normal(30_000, 60_000, 100).clip(min=1_000)
    spike_idx = rng.choice(100, size=5, replace=False)
    volume_base[spike_idx] *= 50
    volume = pd.Series(volume_base, index=dates)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.05,
            "Low": close * 0.95,
            "Close": close,
            "Volume": volume,
        }
    )
