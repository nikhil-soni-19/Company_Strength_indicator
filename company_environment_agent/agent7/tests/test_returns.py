import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from layer1.returns import daily_returns, cumulative_return


def _make_series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_cumulative_return_flat():
    s = _make_series([100.0, 100.0, 100.0])
    assert abs(cumulative_return(s)) < 1e-9


def test_cumulative_return_up10():
    s = _make_series([100.0, 110.0])
    assert abs(cumulative_return(s) - 0.10) < 1e-9


def test_daily_returns_length():
    s = _make_series([100.0, 110.0, 121.0])
    r = daily_returns(s)
    assert len(r) == 2


def test_daily_returns_values():
    s = _make_series([100.0, 110.0, 121.0])
    r = daily_returns(s)
    assert abs(r.iloc[0] - 0.10) < 1e-9
    assert abs(r.iloc[1] - 0.10) < 1e-9
