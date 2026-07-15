"""Unit tests for Layer 1 scorer."""
from layer1.scorer import compute_score


def test_score_high_moat():
    computed = {
        "avg_gross_margin_spread": 0.15,
        "avg_op_margin_spread": 0.10,
        "roic_spread": 0.12,
        "avg_fcf_margin_spread": 0.08,
        "gross_margin_cv": 0.03,
    }
    score = compute_score(computed, ["MARGIN_PREMIUM_SUSTAINED", "ROIC_ELITE", "INSIDER_CONVICTION_HIGH"])
    assert score >= 8.0


def test_score_no_moat():
    computed = {
        "avg_gross_margin_spread": -0.08,
        "avg_op_margin_spread": -0.05,
        "roic_spread": -0.08,
        "avg_fcf_margin_spread": -0.05,
        "gross_margin_cv": 0.20,
    }
    score = compute_score(computed, ["MARGIN_VOLATILE", "ROIC_BELOW_PEERS"])
    assert score <= 3.0


def test_score_clamped_0_10():
    computed = {
        "avg_gross_margin_spread": 0.50,  # absurdly high
        "avg_op_margin_spread": 0.50,
        "roic_spread": 0.50,
        "avg_fcf_margin_spread": 0.50,
        "gross_margin_cv": 0.0,
    }
    score = compute_score(computed, ["MARGIN_PREMIUM_SUSTAINED", "ROIC_ELITE", "INSIDER_CONVICTION_HIGH"])
    assert 0.0 <= score <= 10.0


def test_score_missing_optional_fields():
    computed = {
        "avg_gross_margin_spread": 0.05,
        "avg_op_margin_spread": 0.03,
        "gross_margin_cv": 0.06,
    }
    score = compute_score(computed, [])
    assert 0.0 <= score <= 10.0
