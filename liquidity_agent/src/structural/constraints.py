"""Step 3 — Structural market constraints and buyback signal.

The PDF cites three structural risk factors that are *independent* of recent
trading activity but materially shape a stock's true liquidity profile:

* **Float** — shares actually available to the public.
* **Short Percentage** — a high value signals short-squeeze tail risk.
* **Top-10 Institutional Concentration** — heavily concentrated ownership
  shrinks the effective tradable float.

This module also computes the **Buyback Signal** — two metrics that measure
how much a company's own repurchase programme is inflating reported ADV:

* **Buyback Intensity Ratio (BIR)** — quarterly repurchase spend ÷ estimated
  quarterly dollar volume (ADV$ × 63 trading days).  BIR > 15 % means the
  company itself accounted for more than 15 % of market volume and the
  reported ADV$ overstates third-party liquidity.
* **Buyback Yield** — annualised repurchase spend ÷ market cap.  Captures the
  pace at which the float is being compressed.

Data source: ``yfinance`` ``quarterly_cashflow`` → row
``"Repurchase Of Capital Stock"`` (negative sign = cash out; we take the
absolute value).  If the row is absent or all-NaN no flag is raised and all
buyback fields are ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.data_ingestion import MarketData

# ── Buyback thresholds ────────────────────────────────────────────────────────

BIR_FLAG_THRESHOLD: float = 0.15       # raise flag above this
BIR_AGGRESSIVE_THRESHOLD: float = 0.30  # stronger language above this
_TRADING_DAYS_PER_QUARTER: int = 63


# ── BuybackResult ─────────────────────────────────────────────────────────────

@dataclass
class BuybackResult:
    """Output of :func:`compute_buyback`."""

    # Dollar amount repurchased in the most recent quarter (positive = cash spent)
    quarterly_spend: Optional[float]

    # Annualised repurchase spend (quarterly_spend × 4)
    annualized_spend: Optional[float]

    # BIR = quarterly_spend / (adv_dollar_30d × 63)
    bir: Optional[float]

    # Buyback yield = annualized_spend / market_cap
    buyback_yield: Optional[float]

    # True when BIR exceeds BIR_FLAG_THRESHOLD
    inflation_flag: bool

    # Human-readable reason when inflation_flag is True
    inflation_reason: Optional[str]


def compute_buyback(
    quarterly_cashflow: Optional[pd.DataFrame],
    adv_dollar_30d: Optional[float],
    market_cap: Optional[float],
) -> BuybackResult:
    """Compute buyback intensity ratio and buyback yield.

    Parameters
    ----------
    quarterly_cashflow:
        The ``yfinance`` ``quarterly_cashflow`` DataFrame for the ticker.
        May be ``None`` or empty if unavailable.
    adv_dollar_30d:
        30-day average daily dollar volume from :mod:`src.metrics`.
    market_cap:
        Current market capitalisation in dollars (from ``yfinance`` info).
    """
    quarterly_spend = _extract_quarterly_spend(quarterly_cashflow)

    if quarterly_spend is None:
        return BuybackResult(
            quarterly_spend=None,
            annualized_spend=None,
            bir=None,
            buyback_yield=None,
            inflation_flag=False,
            inflation_reason=None,
        )

    annualized_spend = quarterly_spend * 4

    bir: Optional[float] = None
    if adv_dollar_30d and adv_dollar_30d > 0:
        bir = quarterly_spend / (adv_dollar_30d * _TRADING_DAYS_PER_QUARTER)

    buyback_yield: Optional[float] = None
    if market_cap and market_cap > 0:
        buyback_yield = annualized_spend / market_cap

    inflation_flag = bir is not None and bir > BIR_FLAG_THRESHOLD
    inflation_reason: Optional[str] = None
    if inflation_flag and bir is not None:
        pct = bir * 100
        if bir > BIR_AGGRESSIVE_THRESHOLD:
            inflation_reason = (
                f"Aggressive buyback programme: the company accounted for "
                f"{pct:.0f}% of its own quarterly volume — reported ADV$ "
                f"materially overstates true third-party liquidity."
            )
        else:
            inflation_reason = (
                f"Buyback liquidity inflation: the company's repurchase "
                f"programme contributed {pct:.0f}% of quarterly volume — "
                f"reported ADV$ is modestly inflated."
            )

    return BuybackResult(
        quarterly_spend=quarterly_spend,
        annualized_spend=annualized_spend,
        bir=bir,
        buyback_yield=buyback_yield,
        inflation_flag=inflation_flag,
        inflation_reason=inflation_reason,
    )


def _extract_quarterly_spend(cf: Optional[pd.DataFrame]) -> Optional[float]:
    """Pull the most-recent-quarter buyback outflow from the cashflow DataFrame.

    yfinance uses the label "Repurchase Of Capital Stock" with a negative sign.
    We return the absolute value so callers always work with a positive spend
    figure.  Returns ``None`` if the row is absent or blank.
    """
    if cf is None or cf.empty:
        return None

    candidates = [
        "Repurchase Of Capital Stock",
        "RepurchaseOfCapitalStock",
        "repurchaseOfCapitalStock",
        "Common Stock Repurchased",
    ]
    row: Optional[pd.Series] = None
    for label in candidates:
        if label in cf.index:
            row = cf.loc[label]
            break

    if row is None:
        return None

    values = row.dropna()
    if values.empty:
        return None

    most_recent = values.sort_index(ascending=False).iloc[0]
    spend = abs(float(most_recent))
    return spend if spend > 0 else None


# ── StructuralConstraints ─────────────────────────────────────────────────────

@dataclass
class StructuralConstraints:
    float_shares: Optional[float]
    shares_outstanding: Optional[float]
    short_percent_float: Optional[float]
    top10_institutional_pct: Optional[float]

    @property
    def float_pct_of_outstanding(self) -> Optional[float]:
        """Float shares as a fraction of total shares outstanding."""
        if (
            self.float_shares is None
            or self.shares_outstanding is None
            or self.shares_outstanding == 0
        ):
            return None
        return self.float_shares / self.shares_outstanding

    @property
    def has_thin_float(self) -> bool:
        return self.float_shares is not None and self.float_shares < 10_000_000

    @property
    def has_high_short_interest(self) -> bool:
        return self.short_percent_float is not None and self.short_percent_float > 0.25


def compute_structural_constraints(data: MarketData) -> StructuralConstraints:
    """Project a :class:`MarketData` onto the structural-risk surface."""
    return StructuralConstraints(
        float_shares=data.float_shares,
        shares_outstanding=data.shares_outstanding,
        short_percent_float=data.short_percent_float,
        top10_institutional_pct=data.top10_institutional_pct,
    )
