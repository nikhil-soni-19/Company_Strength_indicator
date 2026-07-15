"""Step 5 — Rule-based dimension scoring.

8 metrics (ADV$, Amihud, Volume CV, Free Float, Float%, Short Interest,
Top-10 Institutional, Buyback BIR) are each bucketed into one of four risk
bands worth 0, 1, 2 or 3 points. Max raw score = 24. The total is mapped
to a tier in :mod:`src.scoring.tiers`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import SCORING_THRESHOLDS

RuleSet = dict[str, list[tuple[Optional[float], Optional[float], int]]]


@dataclass
class DimensionScore:
    dimension: str
    value: Optional[float]
    points: int
    band: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        v = "n/a" if self.value is None else f"{self.value:.4g}"
        return f"{self.dimension}={v} -> {self.points}pt ({self.band})"


_BAND_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}


def score_dimension(
    dimension: str,
    value: Optional[float],
    rules: RuleSet = SCORING_THRESHOLDS,
) -> DimensionScore:
    """Score a single dimension against its rule table.

    A ``value`` of ``None`` is conservatively treated as the *highest* risk
    band for that dimension (mirroring the PDF's "Low Confidence" warning).
    """
    if dimension not in rules:
        raise KeyError(f"Unknown scoring dimension: {dimension!r}")

    if value is None:
        max_points = max(p for *_, p in rules[dimension])
        return DimensionScore(
            dimension=dimension,
            value=None,
            points=max_points,
            band=_BAND_LABELS.get(max_points, "Critical"),
        )

    for lo, hi, points in rules[dimension]:
        if _in_band(value, lo, hi):
            return DimensionScore(
                dimension=dimension,
                value=value,
                points=points,
                band=_BAND_LABELS.get(points, "Unknown"),
            )

    raise ValueError(
        f"Value {value} for dimension {dimension!r} did not match any band"
    )


def score_all(values: dict[str, Optional[float]], rules: RuleSet = SCORING_THRESHOLDS) -> list[DimensionScore]:
    return [score_dimension(dim, values.get(dim), rules) for dim in rules]


def _in_band(value: float, lo: Optional[float], hi: Optional[float]) -> bool:
    above = lo is None or value >= lo
    below = hi is None or value < hi
    return above and below
