"""Compute the *Data Confidence Score* shown on the dashboard.

Based on the PDF's spec:

* High (100%)        — data is fresh and complete.
* Moderate (~70%)    — some staleness or partial history.
* Poor (≤ 40%)       — stale feed or insufficient trading days.

This module isolates *all* data-quality logic so the scoring engine itself
doesn't need to reason about freshness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import CONFIDENCE_STALENESS_DAYS, MIN_TRADING_DAYS_FOR_VALID_SCORE
from src.data_ingestion import MarketData
from src.utils import count_trading_days


@dataclass
class ConfidenceReport:
    score_pct: int
    label: str
    warnings: list[str]

    @property
    def is_actionable(self) -> bool:
        return self.score_pct >= 70


def evaluate_confidence(data: MarketData) -> ConfidenceReport:
    warnings: list[str] = []
    score = 100

    trading_days = count_trading_days(data.ohlcv)
    if trading_days < MIN_TRADING_DAYS_FOR_VALID_SCORE:
        deficit = MIN_TRADING_DAYS_FOR_VALID_SCORE - trading_days
        penalty = min(60, deficit * 2)
        score -= penalty
        warnings.append(
            f"Only {trading_days} trading days of data available "
            f"(need {MIN_TRADING_DAYS_FOR_VALID_SCORE}). Scoring downgraded."
        )

    age = data.data_age_days
    if age is None:
        score -= 50
        warnings.append("Market data feed could not be timestamped.")
    elif age > CONFIDENCE_STALENESS_DAYS:
        score -= min(50, (age - CONFIDENCE_STALENESS_DAYS) * 10)
        warnings.append(
            f"Market data feed has not synced in {age} days. "
            "Do not deploy capital based on these metrics."
        )

    missing_structural: list[str] = []
    if data.float_shares is None:
        missing_structural.append("float")
    if data.short_percent_float is None:
        missing_structural.append("short%")
    if missing_structural:
        score -= 10 * len(missing_structural)
        warnings.append(
            "Missing structural data: " + ", ".join(missing_structural) + "."
        )

    if data.quarterly_cashflow is None or (
        hasattr(data.quarterly_cashflow, "empty") and data.quarterly_cashflow.empty
    ):
        score -= 5
        warnings.append(
            "Buyback cashflow data unavailable — BIR cannot be computed. "
            "Reported ADV$ may be inflated by an undetected repurchase programme."
        )

    score = max(0, min(100, score))
    label = _label_for_score(score)
    return ConfidenceReport(score_pct=score, label=label, warnings=warnings)


def _label_for_score(score: int) -> str:
    if score >= 90:
        return "High"
    if score >= 70:
        return "Moderate"
    if score >= 40:
        return "Low"
    return "Poor"
