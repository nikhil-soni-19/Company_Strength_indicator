"""Hermetic tests for the ticker resolver — alias matching + typo tolerance."""

from __future__ import annotations

import pytest

from src.agent import LiquidityAgent
from src.agent.ticker_resolver import resolve_ticker, resolve_tickers


@pytest.mark.parametrize(
    "text, expected",
    [
        ("AAPL", "AAPL"),
        ("apple", "AAPL"),
        ("Apple", "AAPL"),
        ("APPLE", "AAPL"),
        ("Apples", "AAPL"),
        ("Aple", "AAPL"),
        ("Aaple", "AAPL"),
        ("appl", "AAPL"),
        ("Microsoft", "MSFT"),
        ("microsft", "MSFT"),
        ("Goolge", "GOOGL"),
        ("amazn", "AMZN"),
        ("Tesla", "TSLA"),
        ("teslaa", "TSLA"),
        ("GameStop", "GME"),
        ("game stop", "GME"),
        ("nvidia", "NVDA"),
        ("nflx", "NFLX"),
        ("Netflix", "NFLX"),
    ],
)
def test_single_ticker_resolution(text: str, expected: str) -> None:
    assert resolve_ticker(text) == expected, f"{text!r} should resolve to {expected!r}"


def test_question_with_explicit_ticker() -> None:
    assert resolve_tickers("How liquid is AAPL right now?") == ["AAPL"]


def test_question_with_company_name() -> None:
    assert resolve_tickers("How liquid is Apple right now?") == ["AAPL"]


def test_question_with_typo() -> None:
    assert resolve_tickers("How liquid is Aple right now?") == ["AAPL"]


def test_multiple_tickers_natural_language() -> None:
    out = resolve_tickers("Compare apple, microsoft, and tesla for a 5% stake")
    assert out == ["AAPL", "MSFT", "TSLA"]


def test_mixed_explicit_and_named() -> None:
    out = resolve_tickers("Compare AAPL with microsoft and Goolge")
    assert out == ["AAPL", "MSFT", "GOOGL"]


def test_deduplicates_when_symbol_and_name_both_present() -> None:
    assert resolve_tickers("Is AAPL the same as Apple Inc?") == ["AAPL"]


def test_returns_empty_when_no_ticker_present() -> None:
    assert resolve_tickers("What is liquidity?") == []
    assert resolve_tickers("How are you today?") == []


def test_blacklist_filters_common_words() -> None:
    out = resolve_tickers("Is this a real position?")
    assert out == []


def test_short_words_do_not_false_match() -> None:
    out = resolve_tickers("an amp ape")
    assert "AAPL" not in out, "Three-letter fragment should not fuzzy-match Apple"


def test_extract_via_agent_uses_resolver() -> None:
    assert LiquidityAgent._extract_tickers("How liquid is Apple?") == ["AAPL"]
    assert LiquidityAgent._extract_ticker("How liquid is Aple?") == "AAPL"


def test_punctuation_tolerated() -> None:
    assert resolve_tickers("apple?") == ["AAPL"]
    assert resolve_tickers("apple, microsoft.") == ["AAPL", "MSFT"]
