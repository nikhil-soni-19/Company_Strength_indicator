"""
Unit tests for layer1_deterministic.py.

All tests use synthetic InputBundles — no LLM, no DB, no network required.
Tests are organised into:
  1. Math helpers  (_safe_ratio, _ols_slope, _ratio_cagr)
  2. Flag logic    (_compute_flags) — each flag fires / does not fire
  3. Mutual exclusivity guarantee for capex flags
  4. Sector threshold override
  5. Integration: run_layer1 with known profiles + output shape / edge cases
"""
import sys
from pathlib import Path

# Allow running tests from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import config
from data_contract import DataCoverage, InputBundle
from layer1_deterministic import (
    Layer1Output,
    _compute_flags,
    _compute_l1_score,
    _ols_slope,
    _ratio_cagr,
    _safe_ratio,
    run_layer1,
)


# ─── Test helpers ─────────────────────────────────────────────────────────────

def _make_bundle(
    rd: list[float],
    revenue: list[float],
    capex: list[float],
    ticker: str = "TEST",
    insider_pct: float | None = None,
) -> InputBundle:
    """Build a minimal InputBundle from parallel lists (no DB or network needed)."""
    n = len(revenue)
    assert len(rd) == n and len(capex) == n, "All input lists must be the same length"

    # Generate placeholder period strings — exact dates don't matter for Layer 1.
    periods = [f"202{i // 4}-{(i % 4) * 3 + 3:02d}-30" for i in range(n)]

    coverage = DataCoverage(
        quarters_returned=n,
        rd_quarters_from_db=n,
        rd_quarters_from_yf=0,
        capex_found=any(v > 0 for v in capex),
        holders_found=insider_pct is not None,
        source="db",
    )
    return InputBundle(
        ticker=ticker,
        periods=periods,
        revenue=revenue,
        rd=rd,
        capex=capex,
        insider_pct=insider_pct,
        institutional_top10=None,
        coverage=coverage,
    )


# ─── 1. Math helpers ──────────────────────────────────────────────────────────

class TestSafeRatio:
    def test_normal(self):
        assert abs(_safe_ratio(2.0, 10.0) - 0.2) < 1e-9

    def test_zero_denominator(self):
        assert _safe_ratio(5.0, 0.0) == 0.0

    def test_negative_denominator(self):
        assert _safe_ratio(5.0, -1.0) == 0.0

    def test_zero_numerator(self):
        assert _safe_ratio(0.0, 10.0) == 0.0


class TestOlsSlope:
    def test_positive_slope(self):
        series = [0.00, 0.01, 0.02, 0.03, 0.04]
        assert _ols_slope(series) == pytest.approx(0.01, rel=1e-4)

    def test_negative_slope(self):
        series = [0.10, 0.08, 0.06, 0.04, 0.02]
        assert _ols_slope(series) < 0

    def test_flat_series(self):
        assert _ols_slope([0.05, 0.05, 0.05, 0.05]) == pytest.approx(0.0, abs=1e-9)

    def test_single_element(self):
        assert _ols_slope([0.1]) == 0.0

    def test_empty(self):
        assert _ols_slope([]) == 0.0

    def test_all_zeros(self):
        assert _ols_slope([0.0, 0.0, 0.0]) == 0.0


class TestRatioCagr:
    def test_doubling_over_two_years(self):
        # rd_rev grows from 0.05 to 0.10 over 9 quarters (2 years)
        series = [0.05, 0.056, 0.063, 0.070, 0.075, 0.080, 0.086, 0.093, 0.10]
        cagr = _ratio_cagr(series)
        # Expected: (0.10/0.05)^(1/2) - 1 ≈ 0.414
        assert 0.35 < cagr < 0.50

    def test_flat_series_returns_zero(self):
        assert _ratio_cagr([0.05, 0.05, 0.05, 0.05]) == pytest.approx(0.0, abs=1e-3)

    def test_zero_first_element(self):
        assert _ratio_cagr([0.0, 0.05, 0.10]) == 0.0

    def test_single_element(self):
        assert _ratio_cagr([0.05]) == 0.0

    def test_declining_ratio(self):
        series = [0.10, 0.09, 0.08, 0.07, 0.06]
        assert _ratio_cagr(series) < 0


# ─── 2. Flag logic ────────────────────────────────────────────────────────────

class TestRdIntensifyingFlag:
    def test_fires_just_above_threshold(self):
        slope = config.RD_INTENSIFYING_SLOPE_THRESHOLD + 0.0001
        flags = _compute_flags(0.05, slope, 0.05, 0.0)
        assert "R&D_INTENSIFYING" in flags

    def test_does_not_fire_just_below_threshold(self):
        slope = config.RD_INTENSIFYING_SLOPE_THRESHOLD - 0.0001
        flags = _compute_flags(0.05, slope, 0.05, 0.0)
        assert "R&D_INTENSIFYING" not in flags

    def test_does_not_fire_at_zero_slope(self):
        flags = _compute_flags(0.08, 0.0, 0.05, 0.0)
        assert "R&D_INTENSIFYING" not in flags

    def test_does_not_fire_on_negative_slope(self):
        flags = _compute_flags(0.08, -0.005, 0.05, 0.0)
        assert "R&D_INTENSIFYING" not in flags


class TestCapexReinvestmentStrongFlag:
    def test_fires_on_high_level(self):
        level = config.CAPEX_REINVESTMENT_STRONG_LEVEL + 0.01
        flags = _compute_flags(0.02, 0.0, level, 0.0)
        assert "CAPEX_REINVESTMENT_STRONG" in flags

    def test_fires_on_exactly_level_threshold(self):
        flags = _compute_flags(0.02, 0.0, config.CAPEX_REINVESTMENT_STRONG_LEVEL, 0.0)
        assert "CAPEX_REINVESTMENT_STRONG" in flags

    def test_fires_on_high_slope_even_if_level_is_low(self):
        level = config.CAPEX_REINVESTMENT_STRONG_LEVEL / 2
        slope = config.CAPEX_REINVESTMENT_STRONG_SLOPE + 0.001
        flags = _compute_flags(0.02, 0.0, level, slope)
        assert "CAPEX_REINVESTMENT_STRONG" in flags

    def test_does_not_fire_in_middle_range_with_flat_slope(self):
        mid = (config.CAPEX_LIGHT_FLOOR + config.CAPEX_REINVESTMENT_STRONG_LEVEL) / 2
        flags = _compute_flags(0.05, 0.0, mid, 0.0)
        assert "CAPEX_REINVESTMENT_STRONG" not in flags


class TestCapexLightBusinessFlag:
    def test_fires_clearly_below_floor(self):
        level = config.CAPEX_LIGHT_FLOOR / 2
        flags = _compute_flags(0.10, 0.0, level, 0.0)
        assert "CAPEX_LIGHT_BUSINESS" in flags

    def test_does_not_fire_at_floor(self):
        flags = _compute_flags(0.10, 0.0, config.CAPEX_LIGHT_FLOOR, 0.0)
        assert "CAPEX_LIGHT_BUSINESS" not in flags

    def test_does_not_fire_above_floor(self):
        level = config.CAPEX_LIGHT_FLOOR + 0.01
        flags = _compute_flags(0.10, 0.0, level, 0.0)
        assert "CAPEX_LIGHT_BUSINESS" not in flags


class TestCapexFlagsMutualExclusivity:
    """CAPEX_REINVESTMENT_STRONG and CAPEX_LIGHT_BUSINESS must never co-fire."""

    @pytest.mark.parametrize("level", [0.0, 0.005, 0.01, 0.015, 0.03, 0.05, 0.08, 0.15, 0.25])
    def test_mutually_exclusive_across_levels(self, level: float):
        flags = _compute_flags(0.05, 0.0, level, 0.0)
        both = "CAPEX_REINVESTMENT_STRONG" in flags and "CAPEX_LIGHT_BUSINESS" in flags
        assert not both, f"Both capex flags fired at capex_rev_level={level}"

    def test_mutually_exclusive_when_slope_triggers_strong(self):
        # Force STRONG via slope while level is well below floor
        level = config.CAPEX_LIGHT_FLOOR / 3
        slope = config.CAPEX_REINVESTMENT_STRONG_SLOPE + 0.01
        flags = _compute_flags(0.05, 0.0, level, slope)
        both = "CAPEX_REINVESTMENT_STRONG" in flags and "CAPEX_LIGHT_BUSINESS" in flags
        assert not both


# ─── 3. Sector threshold override ────────────────────────────────────────────

class TestSectorOverride:
    def test_raising_slope_threshold_suppresses_rd_flag(self):
        slope = config.RD_INTENSIFYING_SLOPE_THRESHOLD + 0.0001
        # At default this fires; with a higher override it should not.
        override = {"RD_INTENSIFYING_SLOPE_THRESHOLD": slope + 0.001}
        default_flags  = _compute_flags(0.05, slope, 0.05, 0.0)
        override_flags = _compute_flags(0.05, slope, 0.05, 0.0, override)
        assert "R&D_INTENSIFYING" in default_flags
        assert "R&D_INTENSIFYING" not in override_flags

    def test_lowering_capex_light_floor_suppresses_light_flag(self):
        level = config.CAPEX_LIGHT_FLOOR / 2  # fires at default
        override = {"CAPEX_LIGHT_FLOOR": level / 2}
        default_flags  = _compute_flags(0.05, 0.0, level, 0.0)
        override_flags = _compute_flags(0.05, 0.0, level, 0.0, override)
        assert "CAPEX_LIGHT_BUSINESS" in default_flags
        assert "CAPEX_LIGHT_BUSINESS" not in override_flags

    def test_unknown_override_key_is_ignored(self):
        # Should not crash on unexpected keys.
        slope = config.RD_INTENSIFYING_SLOPE_THRESHOLD + 0.0001
        flags = _compute_flags(0.05, slope, 0.05, 0.0, {"UNKNOWN_KEY": 999})
        assert "R&D_INTENSIFYING" in flags


# ─── 4. run_layer1 integration ────────────────────────────────────────────────

class TestRunLayer1:
    def test_heavy_industrial_profile(self):
        """
        Reinvestment-heavy industrial: capex/rev = 12% (above strong level),
        R&D rising → both CAPEX_REINVESTMENT_STRONG and R&D_INTENSIFYING fire.
        Score should be high (> 7).
        """
        n = 12
        rev   = [1_000_000_000.0] * n
        capex = [120_000_000.0] * n      # 12% — above 8% strong threshold
        # R&D rising: 2% → ~3.2% (slope ≈ +0.0011/quarter)
        rd = [20_000_000 + i * 1_100_000 for i in range(n)]

        out = run_layer1(_make_bundle(rd=rd, revenue=rev, capex=capex, ticker="HEAVY"))

        assert "CAPEX_REINVESTMENT_STRONG" in out.flags
        assert "R&D_INTENSIFYING" in out.flags
        assert "CAPEX_LIGHT_BUSINESS" not in out.flags
        assert out.l1_score > 7.0
        assert out.capex_rev_level == pytest.approx(0.12, rel=0.01)

    def test_asset_light_saas_profile(self):
        """
        Asset-light SaaS: capex/rev = 1% (below 2% floor), flat R&D
        → CAPEX_LIGHT_BUSINESS fires, R&D_INTENSIFYING does not.
        """
        n = 12
        rev   = [500_000_000.0] * n
        capex = [5_000_000.0] * n        # 1% — below light floor
        rd    = [40_000_000.0] * n       # flat at 8%

        out = run_layer1(_make_bundle(rd=rd, revenue=rev, capex=capex, ticker="SAAS"))

        assert "CAPEX_LIGHT_BUSINESS" in out.flags
        assert "CAPEX_REINVESTMENT_STRONG" not in out.flags
        assert abs(out.rd_rev_slope) < 0.001

    def test_semiconductor_profile(self):
        """
        Semiconductor: very high capex (20%) AND rising R&D → both flags, very high score.
        """
        n = 8
        rev   = [2_000_000_000.0] * n
        capex = [400_000_000.0] * n      # 20%
        rd    = [100_000_000 + i * 5_000_000 for i in range(n)]  # rising

        out = run_layer1(_make_bundle(rd=rd, revenue=rev, capex=capex, ticker="SEMI"))

        assert "CAPEX_REINVESTMENT_STRONG" in out.flags
        assert out.l1_score >= 8.0

    def test_output_list_lengths_consistent(self):
        n = 8
        rev   = [1e9] * n
        rd    = [5e7] * n
        capex = [8e7] * n
        out = run_layer1(_make_bundle(rd=rd, revenue=rev, capex=capex))

        assert len(out.rd_rev) == n
        assert len(out.capex_rev) == n
        assert len(out.periods) == n

    def test_score_clamped_to_zero_ten(self):
        # Extreme values should still produce a score in [0, 10].
        n = 4
        # Ridiculous R&D and capex levels
        rev   = [1e9] * n
        rd    = [2e9] * n    # 200% R&D ratio — impossible but should not crash
        capex = [3e9] * n
        out = run_layer1(_make_bundle(rd=rd, revenue=rev, capex=capex))
        assert 0.0 <= out.l1_score <= 10.0

    def test_zero_revenue_quarters_do_not_crash(self):
        """Zero revenue in a quarter should produce 0 ratio, not a crash or NaN."""
        n = 4
        rev   = [0.0, 1e9, 1e9, 1e9]
        rd    = [1e7] * n
        capex = [5e7] * n
        out = run_layer1(_make_bundle(rd=rd, revenue=rev, capex=capex))
        assert out.rd_rev[0] == 0.0
        assert out.capex_rev[0] == 0.0
        assert 0.0 <= out.l1_score <= 10.0

    def test_single_quarter_no_crash(self):
        """Single quarter: slope and CAGR must return 0.0, not crash."""
        out = run_layer1(_make_bundle(rd=[5e7], revenue=[1e9], capex=[8e7]))
        assert out.rd_rev_slope == 0.0
        assert out.rd_rev_cagr == 0.0
        assert out.capex_rev_slope == 0.0
        assert 0.0 <= out.l1_score <= 10.0

    def test_sector_override_propagates_through_run_layer1(self):
        """Sector threshold override must change flag output through the full pipeline."""
        n = 8
        rev   = [1e9] * n
        rd    = [1e7] * n   # 1% flat — no R&D_INTENSIFYING at default
        # Capex just below the light floor
        capex = [config.CAPEX_LIGHT_FLOOR * 0.9 * 1e9] * n

        out_default  = run_layer1(_make_bundle(rd=rd, revenue=rev, capex=capex))
        out_override = run_layer1(
            _make_bundle(rd=rd, revenue=rev, capex=capex),
            sector_thresholds={"CAPEX_LIGHT_FLOOR": config.CAPEX_LIGHT_FLOOR * 0.5},
        )

        assert "CAPEX_LIGHT_BUSINESS" in out_default.flags
        assert "CAPEX_LIGHT_BUSINESS" not in out_override.flags

    def test_coverage_passthrough(self):
        """DataCoverage from InputBundle must pass through to Layer1Output unchanged."""
        n = 6
        bundle = _make_bundle(rd=[5e7]*n, revenue=[1e9]*n, capex=[8e7]*n)
        out = run_layer1(bundle)
        assert out.data_coverage is bundle.coverage
