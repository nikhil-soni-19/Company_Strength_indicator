"""
End-to-end test with mocked LLM, Tavily, and DB.
Verifies output contract shape and environment_runs persistence.
"""

from __future__ import annotations

import sys
import json
import uuid
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures / helpers ────────────────────────────────────────────────────────

MOCK_BUNDLE = {
    "ticker":                   "NVDA",
    "as_of_date":               "2026-06-05",
    "sector":                   "Information Technology",
    "sector_etf":               "XLK",
    "company_cum_return_6m":    0.25,
    "sector_cum_return_6m":     0.15,
    "sector_rs_6m":             1.08,
    "company_alpha_annualised": 0.12,
    "company_beta":             1.1,
    "ab_n_obs":                 120,
    "beta_rate":                -1.5,
    "vix_current":              14.0,
    "vix_zscore":              -0.5,
    "vol_regime":               "NORMAL_VOLATILITY",
    "rate_slope_3m":            0.001,
    "rate_slope_z":             0.3,
    "rate_regime":              "STABLE_RATE",
    "market_trend":             "BULL",
    "commodity_impact_raw":     0.0,
    "commodity_tag":            "NOT_APPLICABLE",
    "peer_rev_growth_gap":      0.07,
    "peer_margin_gap":          0.04,
    "margin_metric":            "gross_margin",
    "peer_margin_company":      0.65,
    "peer_margin_median":       0.61,
    "rf_annual":                0.043,
    "flags":                    ["SECTOR_LEADING", "MARKET_BULLISH", "PEER_GAINING_GROUND", "MARGIN_LEADER"],
}

MOCK_LLM_RESULT = {
    "qual_score": 68,
    "direction": "MIXED",
    "narrative": (
        "NVDA operates in a broadly supportive environment underpinned by AI demand tailwinds "
        "cited in [EXCERPT 1]. However, export control risks noted in [NEWS 2] introduce "
        "regulatory headwinds. Sector momentum is strong (RS 1.08 per Layer 1), but rising "
        "rate sensitivity (beta_rate = -1.5) warrants monitoring."
    ),
    "key_tailwinds": ["AI data-centre demand surge (EXCERPT 1)"],
    "key_risks":     ["US export controls on advanced chips (NEWS 2)"],
}

_PERSISTED: list[dict] = []


def _mock_persist(result: dict) -> None:
    _PERSISTED.append(result)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_output_contract_shape():
    """run_agent returns a dict matching the output contract."""
    _PERSISTED.clear()

    with (
        patch("agent.run.build_bundle",  return_value=MOCK_BUNDLE),
        patch("agent.run.top_news",      return_value=[{"title": "t", "url": "u", "snippet": "s", "published_at": "2026-06-01"}]),
        patch("agent.run.retrieve",      return_value=["Risk factor excerpt 1", "Risk factor excerpt 2"]),
        patch("agent.run.interpret",     return_value=MOCK_LLM_RESULT),
        patch("agent.run._persist",      side_effect=_mock_persist),
    ):
        from agent.run import run_agent
        result = run_agent("NVDA", as_of_date="2026-06-05")

    # Required top-level keys
    required_keys = {
        "ticker", "as_of_date", "environment_score", "direction",
        "quant_score", "qual_score", "flags", "layer1_bundle",
        "narrative", "key_tailwinds", "key_risks", "evidence",
    }
    assert required_keys.issubset(result.keys()), \
        f"Missing keys: {required_keys - result.keys()}"

    assert result["ticker"] == "NVDA"
    assert result["as_of_date"] == "2026-06-05"
    assert result["direction"] in {"SUPPORTIVE", "MIXED", "HOSTILE"}
    assert 0 <= result["environment_score"] <= 100
    assert 0 <= result["quant_score"] <= 100
    assert 0 <= result["qual_score"] <= 100
    assert isinstance(result["flags"], list)
    assert isinstance(result["key_tailwinds"], list)
    assert isinstance(result["key_risks"], list)
    assert "news_snippets" in result["evidence"]
    assert "risk_factor_excerpts" in result["evidence"]


def test_persistence_called():
    """run_agent persists a row to environment_runs."""
    _PERSISTED.clear()

    with (
        patch("agent.run.build_bundle",  return_value=MOCK_BUNDLE),
        patch("agent.run.top_news",      return_value=[]),
        patch("agent.run.retrieve",      return_value=[]),
        patch("agent.run.interpret",     return_value=MOCK_LLM_RESULT),
        patch("agent.run._persist",      side_effect=_mock_persist),
    ):
        from agent.run import run_agent
        run_agent("NVDA", as_of_date="2026-06-05")

    assert len(_PERSISTED) == 1
    row = _PERSISTED[0]
    assert row["ticker"] == "NVDA"
    assert row["direction"] in {"SUPPORTIVE", "MIXED", "HOSTILE"}
    assert "run_id" in row


def test_score_math():
    """quant + qual → combined score is correct."""
    from scoring.quant_score import quant_score
    from scoring.final_score import combine

    qs = quant_score(MOCK_BUNDLE)
    combined = combine(qs, 68.0)
    expected = round(0.5 * qs + 0.5 * 68.0, 2)
    assert abs(combined["environment_score"] - expected) < 0.01
    assert combined["direction"] in {"SUPPORTIVE", "MIXED", "HOSTILE"}


def test_direction_thresholds():
    from scoring.final_score import combine
    assert combine(100, 100)["direction"] == "SUPPORTIVE"
    assert combine(50, 50)["direction"] == "MIXED"
    assert combine(0, 0)["direction"] == "HOSTILE"
    assert combine(70, 70)["direction"] == "SUPPORTIVE"
    assert combine(69, 69)["direction"] == "MIXED"
    assert combine(30, 30)["direction"] == "MIXED"
    assert combine(29, 29)["direction"] == "HOSTILE"


def test_flags_in_output():
    """SECTOR_LEADING flag from bundle propagates to output."""
    _PERSISTED.clear()

    with (
        patch("agent.run.build_bundle",  return_value=MOCK_BUNDLE),
        patch("agent.run.top_news",      return_value=[]),
        patch("agent.run.retrieve",      return_value=[]),
        patch("agent.run.interpret",     return_value=MOCK_LLM_RESULT),
        patch("agent.run._persist",      side_effect=_mock_persist),
    ):
        from agent.run import run_agent
        result = run_agent("NVDA", as_of_date="2026-06-05")

    assert "SECTOR_LEADING" in result["flags"]
