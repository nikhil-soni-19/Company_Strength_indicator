"""End-to-end scoring orchestrator.

8 scoring dimensions (DTL removed; float%, short interest, top-10 institutional,
and buyback BIR added). Max raw score = 24.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.data_ingestion import MarketData
from src.metrics import (
    ADVDollarResult,
    AmihudResult,
    VolumeCVResult,
    compute_adv_dollar,
    compute_amihud,
    compute_volume_cv,
)
from src.scoring.overrides import MirageOverrideResult, apply_mirage_override
from src.scoring.rules import DimensionScore, score_all
from src.scoring.tiers import Tier, assign_tier
from src.structural import (
    BuybackResult,
    StructuralConstraints,
    compute_buyback,
    compute_structural_constraints,
)


@dataclass
class LiquidityScore:
    ticker: str
    raw_score: int
    base_tier: Tier
    final_tier: Tier
    dimension_scores: list[DimensionScore]
    adv: ADVDollarResult
    amihud: AmihudResult
    volume_cv: VolumeCVResult
    structural: StructuralConstraints
    mirage: MirageOverrideResult
    buyback: BuybackResult
    flags: list[str] = field(default_factory=list)

    @property
    def headline_metrics(self) -> dict[str, Optional[float]]:
        return {
            "adv_dollar_30d": self.adv.adv_dollar_30d,
            "adv_dollar_90d": self.adv.adv_dollar_90d,
            "amihud_30d": self.amihud.amihud_30d,
            "volume_cv_30d": self.volume_cv.volume_cv_30d,
            "float_shares": self.structural.float_shares,
            "float_pct_outstanding": self.structural.float_pct_of_outstanding,
            "short_percent_float": self.structural.short_percent_float,
            "top10_institutional_pct": self.structural.top10_institutional_pct,
            "buyback_bir": self.buyback.bir,
            "buyback_yield": self.buyback.buyback_yield,
        }


def score_liquidity(data: MarketData) -> LiquidityScore:
    adv = compute_adv_dollar(data.ohlcv)
    amihud = compute_amihud(data.ohlcv)
    cv = compute_volume_cv(data.ohlcv)
    structural = compute_structural_constraints(data)
    buyback = compute_buyback(
        quarterly_cashflow=data.quarterly_cashflow,
        adv_dollar_30d=adv.adv_dollar_30d,
        market_cap=data.market_cap,
    )

    dimension_values: dict[str, Optional[float]] = {
        "adv_dollar_30d": adv.adv_dollar_30d,
        "amihud_30d": amihud.amihud_30d,
        "volume_cv_30d": cv.volume_cv_30d,
        "free_float_shares": structural.float_shares,
        "float_pct_outstanding": structural.float_pct_of_outstanding,
        "short_percent_float": structural.short_percent_float,
        "top10_institutional_pct": structural.top10_institutional_pct,
        "buyback_bir": buyback.bir,
    }
    dim_scores = score_all(dimension_values)
    raw_score = sum(d.points for d in dim_scores)

    base_tier = assign_tier(raw_score)
    mirage = apply_mirage_override(
        base_tier=base_tier,
        float_shares=structural.float_shares,
        short_percent_float=structural.short_percent_float,
    )
    final_tier = mirage.final_tier

    _FLAGGABLE = frozenset({
        "volume_cv_30d", "free_float_shares",
        "short_percent_float", "top10_institutional_pct",
    })
    flags: list[str] = [
        f"{d.dimension}: {d.band}"
        for d in dim_scores
        if d.points >= 2 and d.dimension in _FLAGGABLE
    ]
    if mirage.triggered and mirage.reason:
        flags.insert(0, mirage.reason)
    if adv.liquidity_drain_ratio is not None and adv.liquidity_drain_ratio < 0.5:
        flags.append(
            f"Structural liquidity drain: ADV$30 is only "
            f"{adv.liquidity_drain_ratio:.0%} of ADV$90."
        )
    if buyback.inflation_flag and buyback.inflation_reason:
        flags.append(buyback.inflation_reason)

    return LiquidityScore(
        ticker=data.ticker,
        raw_score=raw_score,
        base_tier=base_tier,
        final_tier=final_tier,
        dimension_scores=dim_scores,
        adv=adv,
        amihud=amihud,
        volume_cv=cv,
        structural=structural,
        mirage=mirage,
        buyback=buyback,
        flags=flags,
    )
