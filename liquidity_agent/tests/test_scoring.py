"""Smoke tests for Step-4 (DTL) and Step-5 (scoring + tier + mirage)."""

from __future__ import annotations

from src.liquidation import compute_dtl, compute_dtl_for_positions
from src.scoring import (
    apply_mirage_override,
    assign_tier,
    score_dimension,
)
from src.scoring.tiers import Tier


def test_dtl_formula():
    assert compute_dtl(position_shares=1_000_000, adv_shares=5_000_000, participation_rate=0.10) == 2.0


def test_dtl_returns_none_when_adv_missing():
    assert compute_dtl(position_shares=1_000_000, adv_shares=None) is None


def test_dtl_for_positions_uses_float(thin_float_ohlcv):
    result = compute_dtl_for_positions(thin_float_ohlcv, float_shares=10_000_000)
    assert result.dtl_5pct is not None
    assert result.dtl_5pct > result.dtl_1pct  # type: ignore[operator]


def test_assign_tier_boundaries():
    assert assign_tier(0).number == 1
    assert assign_tier(4).number == 1
    assert assign_tier(5).number == 2
    assert assign_tier(6).number == 2
    assert assign_tier(7).number == 3
    assert assign_tier(8).number == 3
    assert assign_tier(9).number == 4
    assert assign_tier(15).number == 4


def test_score_dimension_handles_missing_value():
    s = score_dimension("adv_dollar_30d", None)
    assert s.points == 3
    assert s.band == "Critical"


def test_score_dimension_buckets_correctly():
    assert score_dimension("adv_dollar_30d", 50_000_000).points == 0
    assert score_dimension("adv_dollar_30d", 5_000_000).points == 1
    assert score_dimension("adv_dollar_30d", 1_000_000).points == 2
    assert score_dimension("adv_dollar_30d", 100_000).points == 3

    assert score_dimension("amihud_30d", 0.001).points == 0
    assert score_dimension("amihud_30d", 0.30).points == 3

    assert score_dimension("volume_cv_30d", 0.1).points == 0
    assert score_dimension("volume_cv_30d", 1.5).points == 3


def test_mirage_override_downgrades_tier():
    base = assign_tier(0)  # Tier 1
    result = apply_mirage_override(
        base_tier=base,
        float_shares=5_000_000,
        short_percent_float=0.30,
    )
    assert result.triggered is True
    assert result.final_tier.number == 3  # +2 from Tier 1


def test_mirage_override_does_not_fire_on_healthy_stock():
    base = assign_tier(0)
    result = apply_mirage_override(base, float_shares=200_000_000, short_percent_float=0.05)
    assert result.triggered is False
    assert result.final_tier == base


def test_mirage_override_does_not_overshoot_tier_4():
    base = assign_tier(8)  # already Tier 4
    result = apply_mirage_override(base, float_shares=1_000_000, short_percent_float=0.40)
    assert isinstance(result.final_tier, Tier)
    assert result.final_tier.number == 4
