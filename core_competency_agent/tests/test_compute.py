"""Unit tests for Layer 1 financial computations."""
import pytest
from layer1.compute import (
    margin_series,
    coefficient_of_variation,
    compute_roic,
    fcf_margin_series,
    peer_median_margin,
)


def test_margin_series_basic():
    gp  = [30.0, 35.0, 40.0]
    rev = [100.0, 100.0, 100.0]
    result = margin_series(gp, rev)
    assert result == pytest.approx([0.30, 0.35, 0.40])


def test_margin_series_zero_revenue():
    result = margin_series([10.0], [0.0])
    assert result == [0.0]


def test_cv_stable():
    series = [0.50, 0.51, 0.49, 0.50, 0.50, 0.51, 0.49, 0.50]
    cv = coefficient_of_variation(series)
    assert cv < 0.05  # very stable margins


def test_cv_volatile():
    series = [0.10, 0.40, 0.05, 0.45, 0.08, 0.42, 0.07, 0.43]
    cv = coefficient_of_variation(series)
    assert cv > 0.40  # highly volatile


def test_compute_roic_basic():
    # 4 quarters: op_income=10M, equity=80M, debt=20M (invested capital=100M)
    # NOPAT = 40M * (1-0.21) = 31.6M
    # ROIC = 31.6 / avg(100M) = 0.316 (TTM)
    op    = [10.0e6, 10.0e6, 10.0e6, 10.0e6]
    debt  = [20.0e6, 20.0e6, 20.0e6, 20.0e6]
    eq    = [80.0e6, 80.0e6, 80.0e6, 80.0e6]
    tax   = [0.0, 0.0, 0.0, 0.0]  # no tax provision → uses statutory rate
    rev   = [100.0e6, 100.0e6, 100.0e6, 100.0e6]
    roic  = compute_roic(op, debt, eq, tax, rev)
    assert roic is not None
    assert 0.30 < roic < 0.35


def test_compute_roic_insufficient_data():
    # Only 3 quarters
    assert compute_roic([10.0] * 3, [20.0] * 3, [80.0] * 3, [0.0] * 3, [100.0] * 3) is None


def test_fcf_margin_series():
    ocf   = [20.0, 25.0, 30.0]
    capex = [5.0, 5.0, 5.0]
    rev   = [100.0, 100.0, 100.0]
    result = fcf_margin_series(ocf, capex, rev)
    assert result == pytest.approx([0.15, 0.20, 0.25])


def test_peer_median_margin_basic():
    peers = [
        {"gross_profit": [30.0], "revenue": [100.0]},
        {"gross_profit": [50.0], "revenue": [100.0]},
        {"gross_profit": [40.0], "revenue": [100.0]},
    ]
    med = peer_median_margin(peers, "gross_profit")
    assert med == pytest.approx(0.40)


def test_peer_median_margin_empty():
    assert peer_median_margin([], "gross_profit") is None
