import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from layer1.regimes import vix_regime, rate_regime, market_trend


def _vix_series(spike_z: float = 3.0, n: int = 756):
    """Normal VIX around 18, last point z-scored at spike_z."""
    rng = np.random.default_rng(0)
    base = rng.normal(18, 4, n - 1)
    mu, sd = base.mean(), base.std()
    spike = mu + spike_z * sd
    vals = np.append(base, spike)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.Series(vals, index=idx)


def _tnx_series_rising(n: int = 756):
    """TNX series that ramps sharply over the final 63 days."""
    rng = np.random.default_rng(1)
    base = rng.normal(4.0, 0.1, n - 63)
    ramp = np.linspace(4.0, 6.0, 63)  # sharp rise
    vals = np.concatenate([base, ramp])
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.Series(vals, index=idx)


def test_vix_high_volatility():
    s = _vix_series(spike_z=3.0)
    result = vix_regime(s, thresh_high=1.5)
    assert result["vol_regime"] == "HIGH_VOLATILITY"
    assert abs(result["vix_zscore"] - 3.0) < 0.15


def test_vix_low_volatility():
    s = _vix_series(spike_z=-1.5)
    result = vix_regime(s, thresh_low=-1.0)
    assert result["vol_regime"] == "LOW_VOLATILITY"


def test_vix_normal():
    s = _vix_series(spike_z=0.0)
    result = vix_regime(s, thresh_high=1.5, thresh_low=-1.0)
    assert result["vol_regime"] == "NORMAL_VOLATILITY"


def test_rate_rising():
    s = _tnx_series_rising()
    result = rate_regime(s, sigma_mult=1.0)
    assert result["rate_regime"] == "RISING_RATE"


def test_market_trend_bull():
    # Price steadily rising: MA50 > MA200, price > MA200
    vals = np.linspace(100, 200, 250)
    idx = pd.date_range("2023-01-01", periods=250, freq="B")
    p = pd.Series(vals, index=idx)
    assert market_trend(p) == "BULL"


def test_market_trend_bear():
    vals = np.linspace(200, 100, 250)
    idx = pd.date_range("2023-01-01", periods=250, freq="B")
    p = pd.Series(vals, index=idx)
    assert market_trend(p) == "BEAR"


def test_market_trend_insufficient():
    p = pd.Series([100, 110], index=pd.date_range("2024-01-01", periods=2, freq="B"))
    assert market_trend(p) == "INSUFFICIENT_DATA"
