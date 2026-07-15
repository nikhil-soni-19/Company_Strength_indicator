"""Step 5 (cont.) — Map raw score to a tier label and action policy."""

from __future__ import annotations

from dataclasses import dataclass

from config import TIER_BOUNDARIES


@dataclass(frozen=True)
class Tier:
    number: int
    label: str
    score: int
    action: str

    @property
    def badge(self) -> str:
        return f"Tier {self.number} — {self.label}"


_TIER_ACTIONS = {
    1: "Safe for automated market orders; up to 5% stake without pre-trade approval.",
    2: "Position-sizing caps; max 1% of float; Limit / TWAP only.",
    3: "Algorithmic execution only (VWAP, ≤5% participation); compliance sign-off required.",
    4: "Automated hard-block. Exit risk too high — do not enter.",
}


def assign_tier(score: int) -> Tier:
    """Map a raw score (sum of dimension points) onto a :class:`Tier`."""
    for lo, hi, number, label in TIER_BOUNDARIES:
        if lo <= score <= hi:
            return Tier(
                number=number,
                label=label,
                score=score,
                action=_TIER_ACTIONS.get(number, ""),
            )
    raise ValueError(f"Score {score} did not match any tier boundary")
