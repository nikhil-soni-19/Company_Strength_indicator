"""Unit tests for Layer 1 flag emission."""
import pytest
from layer1.flags import emit_flags


def _base(**overrides):
    defaults = dict(
        avg_gross_margin_spread=0.0,
        gross_margin_spread=[0.0] * 8,
        avg_op_margin_spread=0.0,
        roic_spread=None,
        roic_company=None,
        avg_fcf_margin_spread=None,
        insider_pct=None,
        gross_margin_cv=0.05,
    )
    defaults.update(overrides)
    return defaults


def test_margin_premium_sustained_fires():
    flags = emit_flags(**_base(
        avg_gross_margin_spread=0.08,
        gross_margin_spread=[0.05, 0.06, 0.07, 0.08, 0.07, 0.09, 0.10, 0.08],
    ))
    assert "MARGIN_PREMIUM_SUSTAINED" in flags


def test_margin_premium_sustained_fails_if_any_negative():
    flags = emit_flags(**_base(
        avg_gross_margin_spread=0.08,
        gross_margin_spread=[0.05, -0.01, 0.07, 0.08, 0.07, 0.09, 0.10, 0.08],
    ))
    assert "MARGIN_PREMIUM_SUSTAINED" not in flags


def test_roic_elite_fires():
    flags = emit_flags(**_base(
        roic_spread=0.07,
        roic_company=0.20,
    ))
    assert "ROIC_ELITE" in flags


def test_roic_elite_fails_below_absolute():
    flags = emit_flags(**_base(
        roic_spread=0.07,
        roic_company=0.12,  # below 15% absolute threshold
    ))
    assert "ROIC_ELITE" not in flags


def test_insider_conviction_high():
    flags = emit_flags(**_base(insider_pct=0.08))
    assert "INSIDER_CONVICTION_HIGH" in flags


def test_margin_volatile_fires():
    flags = emit_flags(**_base(gross_margin_cv=0.15))
    assert "MARGIN_VOLATILE" in flags


def test_roic_below_peers():
    flags = emit_flags(**_base(roic_spread=-0.05, roic_company=0.05))
    assert "ROIC_BELOW_PEERS" in flags


def test_no_flags_when_average():
    flags = emit_flags(**_base())
    assert flags == []
