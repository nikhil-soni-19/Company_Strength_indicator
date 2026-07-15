import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from layer1.relative_strength import relative_strength


def _series(start, end, n=10):
    vals = [start + (end - start) * i / (n - 1) for i in range(n)]
    return pd.Series(vals, index=pd.date_range("2024-01-01", periods=n, freq="B"))


def test_both_up_10pct():
    # sector +10%, benchmark +10% → RS == 1.0
    s = _series(100, 110)
    b = _series(100, 110)
    assert abs(relative_strength(s, b) - 1.0) < 1e-9


def test_sector_outperforms():
    # sector +20%, benchmark +10% → RS = 1.20 / 1.10 ≈ 1.0909
    s = _series(100, 120)
    b = _series(100, 110)
    expected = 1.20 / 1.10
    assert abs(relative_strength(s, b) - expected) < 1e-9


def test_sector_underperforms():
    # sector flat, benchmark +10% → RS ≈ 0.909
    s = _series(100, 100)
    b = _series(100, 110)
    expected = 1.0 / 1.10
    assert abs(relative_strength(s, b) - expected) < 1e-9
