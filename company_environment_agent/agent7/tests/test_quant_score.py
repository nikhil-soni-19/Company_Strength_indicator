import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.quant_score import quant_score


def _neutral_bundle():
    return {
        "sector_rs_6m":             1.0,
        "company_alpha_annualised": 0.0,
        "vix_zscore":               0.0,
        "rate_slope_z":             0.0,
        "market_trend":             "TRANSITION",
        "commodity_impact_raw":     0.0,
        "peer_rev_growth_gap":      0.0,
        "peer_margin_gap":          0.0,
    }


def _supportive_bundle():
    return {
        "sector_rs_6m":             1.20,   # strong sector outperformance
        "company_alpha_annualised": 0.25,   # high alpha
        "vix_zscore":              -1.5,    # low vol (supportive)
        "rate_slope_z":            -1.5,    # falling rates (supportive)
        "market_trend":             "BULL",
        "commodity_impact_raw":     1.0,    # tailwind
        "peer_rev_growth_gap":      0.20,   # beating peers badly
        "peer_margin_gap":          0.10,   # margin leader
    }


def _hostile_bundle():
    return {
        "sector_rs_6m":             0.80,   # sector lagging badly
        "company_alpha_annualised":-0.25,   # negative alpha
        "vix_zscore":               2.5,    # high vol
        "rate_slope_z":             2.5,    # sharply rising rates
        "market_trend":             "BEAR",
        "commodity_impact_raw":    -1.0,    # headwind
        "peer_rev_growth_gap":     -0.20,   # losing to peers
        "peer_margin_gap":         -0.10,   # margin laggard
    }


def test_neutral_bundle_near_50():
    score = quant_score(_neutral_bundle())
    assert abs(score - 50.0) < 1.0


def test_supportive_bundle_above_80():
    score = quant_score(_supportive_bundle())
    assert score >= 80, f"Expected >= 80, got {score}"


def test_hostile_bundle_below_20():
    score = quant_score(_hostile_bundle())
    assert score <= 20, f"Expected <= 20, got {score}"


def test_score_in_range():
    for bundle in [_neutral_bundle(), _supportive_bundle(), _hostile_bundle()]:
        s = quant_score(bundle)
        assert 0 <= s <= 100


def test_none_values_handled():
    bundle = {k: None for k in _neutral_bundle()}
    score = quant_score(bundle)
    assert 0 <= score <= 100
