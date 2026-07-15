import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from layer1.alpha_beta import alpha_beta


def _make_returns(n=252, seed=42):
    rng = np.random.default_rng(seed)
    r_sector = rng.normal(0.0005, 0.01, n)
    daily_const = 0.0002  # ~5% annualised alpha
    r_company = 0.5 * r_sector + daily_const + rng.normal(0, 0.002, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(r_company, index=idx), pd.Series(r_sector, index=idx)


def test_beta_recovery():
    r_co, r_sec = _make_returns()
    result = alpha_beta(r_co, r_sec, rf_annual=0.0)
    assert result["beta"] is not None
    assert abs(result["beta"] - 0.5) < 0.05


def test_alpha_recovery():
    r_co, r_sec = _make_returns()
    result = alpha_beta(r_co, r_sec, rf_annual=0.0)
    assert result["alpha_annualised"] is not None
    expected_alpha = 0.0002 * 252
    assert abs(result["alpha_annualised"] - expected_alpha) < 0.015


def test_insufficient_data():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    r_co  = pd.Series(np.random.randn(10) * 0.01, index=idx)
    r_sec = pd.Series(np.random.randn(10) * 0.01, index=idx)
    result = alpha_beta(r_co, r_sec, rf_annual=0.04)
    assert result["beta"] is None
    assert result["alpha_annualised"] is None
