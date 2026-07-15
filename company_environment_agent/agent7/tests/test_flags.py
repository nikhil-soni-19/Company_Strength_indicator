import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from layer1.flags import emit_flags


def test_sector_leading():
    flags = emit_flags(sector_rs_6m=1.05)
    assert "SECTOR_LEADING" in flags


def test_sector_lagging():
    flags = emit_flags(sector_rs_6m=0.95)
    assert "SECTOR_LAGGING" in flags


def test_rising_rate():
    flags = emit_flags(rate_regime="RISING_RATE")
    assert "RISING_RATE" in flags


def test_rate_sensitive():
    flags = emit_flags(beta_rate=-3.0)
    assert "RATE_SENSITIVE" in flags


def test_high_volatility():
    flags = emit_flags(vix_zscore=2.0)
    assert "HIGH_VOLATILITY" in flags


def test_low_volatility():
    flags = emit_flags(vix_zscore=-1.5)
    assert "LOW_VOLATILITY" in flags


def test_market_bullish():
    flags = emit_flags(market_trend="BULL")
    assert "MARKET_BULLISH" in flags


def test_commodity_headwind():
    flags = emit_flags(commodity_tag="COMMODITY_HEADWIND")
    assert "COMMODITY_HEADWIND" in flags


def test_peer_gaining_ground():
    flags = emit_flags(peer_rev_growth_gap=0.08)
    assert "PEER_GAINING_GROUND" in flags


def test_margin_laggard():
    flags = emit_flags(peer_margin_gap=-0.05)
    assert "MARGIN_LAGGARD" in flags


def test_neutral_no_flags():
    flags = emit_flags(sector_rs_6m=1.0, vix_zscore=0.0)
    assert len(flags) == 0
