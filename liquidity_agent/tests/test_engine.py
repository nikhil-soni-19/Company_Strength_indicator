"""End-to-end scoring engine test (no network)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.data_ingestion import MarketData
from src.output.confidence import evaluate_confidence
from src.output.narrative import build_narrative
from src.scoring import score_liquidity


def test_end_to_end_liquid_stock(synthetic_ohlcv):
    data = MarketData(
        ticker="SYNTH",
        ohlcv=synthetic_ohlcv,
        float_shares=200_000_000,
        shares_outstanding=200_000_000,
        short_percent_float=0.02,
        top10_institutional_pct=0.45,
        as_of=datetime(2026, 5, 21, tzinfo=timezone.utc),
    )
    result = score_liquidity(data)
    confidence = evaluate_confidence(data)

    assert result.final_tier.number in (1, 2)
    assert result.mirage.triggered is False
    assert confidence.score_pct >= 70

    narrative = build_narrative(result)
    assert "SYNTH" in narrative
    assert "ADV$" in narrative


def test_end_to_end_thin_float_triggers_mirage(thin_float_ohlcv):
    data = MarketData(
        ticker="THIN",
        ohlcv=thin_float_ohlcv,
        float_shares=4_000_000,
        shares_outstanding=4_000_000,
        short_percent_float=0.35,
        top10_institutional_pct=0.85,
        as_of=datetime(2026, 5, 21, tzinfo=timezone.utc),
    )
    result = score_liquidity(data)

    assert result.mirage.triggered is True
    assert result.final_tier.number >= result.base_tier.number
    assert result.final_tier.number >= 3
