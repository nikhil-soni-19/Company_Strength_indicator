"""Step 5 (edge case) — Short squeeze risk structural override.

Quoting the design doc:

    "If Float < 10M and Short% > 25%, automatically downgrade the stock by
    two risk tiers, no matter how high the current ADV$ is."

This is what catches a low-float penny stock being temporarily pumped on
social media: its ADV$ may *look* Tier-1, but the structural fragility
guarantees a liquidity vacuum the moment the short squeeze unwinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import MIRAGE_OVERRIDE
from src.scoring.tiers import Tier, assign_tier


@dataclass
class MirageOverrideResult:
    triggered: bool
    reason: Optional[str]
    original_tier: Tier
    final_tier: Tier
    downgrade_steps: int


def apply_mirage_override(
    base_tier: Tier,
    float_shares: Optional[float],
    short_percent_float: Optional[float],
) -> MirageOverrideResult:
    triggered = (
        float_shares is not None
        and short_percent_float is not None
        and float_shares < float(MIRAGE_OVERRIDE["max_float_shares"])
        and short_percent_float > float(MIRAGE_OVERRIDE["min_short_pct"])
    )

    if not triggered:
        return MirageOverrideResult(
            triggered=False,
            reason=None,
            original_tier=base_tier,
            final_tier=base_tier,
            downgrade_steps=0,
        )

    steps = int(MIRAGE_OVERRIDE["tier_downgrade"])
    new_number = min(4, base_tier.number + steps)

    bumped_score = max(base_tier.score, _min_score_for_tier(new_number))
    new_tier = assign_tier(bumped_score)

    reason = (
        f"Short squeeze risk override: float "
        f"{float_shares:,.0f} < 10M and short% "
        f"{short_percent_float:.1%} > 25%. "
        f"Downgraded by {steps} tier(s)."
    )
    return MirageOverrideResult(
        triggered=True,
        reason=reason,
        original_tier=base_tier,
        final_tier=new_tier,
        downgrade_steps=new_tier.number - base_tier.number,
    )


def _min_score_for_tier(tier_number: int) -> int:
    from config import TIER_BOUNDARIES
    for lo, _hi, number, _label in TIER_BOUNDARIES:
        if number == tier_number:
            return lo
    return 0
