"""Hermetic tests for the multi-ticker comparison path.

We monkeypatch ``OPENAI_API_KEY`` off so the deterministic fallback is
exercised. The synthetic fixtures from ``conftest.py`` provide both a
healthy and a thin-float stock.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.agent import LiquidityAgent
from src.agent.interpreter import interpret_comparison
from src.agent.ticker_resolver import TICKER_BLACKLIST
from src.data_ingestion import MarketData
from src.output.confidence import evaluate_confidence
from src.scoring import score_liquidity


def _build_market_data(ticker: str, ohlcv, **kwargs) -> MarketData:
    defaults = dict(
        float_shares=200_000_000,
        shares_outstanding=200_000_000,
        short_percent_float=0.02,
        top10_institutional_pct=0.4,
        as_of=datetime(2026, 5, 21, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return MarketData(ticker=ticker, ohlcv=ohlcv, **defaults)


def test_interpret_comparison_ranks_safe_first(synthetic_ohlcv, thin_float_ohlcv, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    safe = _build_market_data("SAFE", synthetic_ohlcv)
    thin = _build_market_data(
        "THIN",
        thin_float_ohlcv,
        float_shares=4_000_000,
        shares_outstanding=4_000_000,
        short_percent_float=0.35,
    )

    scored = [
        (score_liquidity(safe), evaluate_confidence(safe)),
        (score_liquidity(thin), evaluate_confidence(thin)),
    ]
    out = interpret_comparison(scored, question="Which is safer for a 5% stake?")

    assert out.used_llm is False
    assert "SAFE" in out.paragraph
    assert "THIN" in out.paragraph
    safe_idx = out.paragraph.index("SAFE")
    thin_idx = out.paragraph.index("THIN")
    assert safe_idx < thin_idx, "Safer ticker should appear first in the ranked paragraph"


def test_interpret_comparison_flags_mirage(thin_float_ohlcv, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    data = _build_market_data(
        "THIN",
        thin_float_ohlcv,
        float_shares=4_000_000,
        shares_outstanding=4_000_000,
        short_percent_float=0.40,
    )
    scored = [(score_liquidity(data), evaluate_confidence(data))]
    out = interpret_comparison(scored)
    assert "Mirage" in out.paragraph or "mirage" in out.paragraph


def test_extract_tickers_picks_up_multiple():
    tickers = LiquidityAgent._extract_tickers("Compare AAPL and GME, plus TSLA please")
    assert tickers == ["AAPL", "GME", "TSLA"]


def test_extract_tickers_dedupes_and_skips_words():
    tickers = LiquidityAgent._extract_tickers("Is AAPL more liquid than AAPL or GME?")
    assert tickers == ["AAPL", "GME"]


def test_blacklist_filters_common_words():
    tickers = LiquidityAgent._extract_tickers("Is GME safe for A short position?")
    assert "A" not in tickers
    assert "GME" in tickers
    assert "I" in TICKER_BLACKLIST and "THE" in TICKER_BLACKLIST
