"""
Layer 1 moat score: 0-10.

Weighted components:
  Gross margin premium   — 2.0 pts  (spread clamped [-10pp, +20pp] → [0, 1])
  Op margin premium      — 1.5 pts
  ROIC premium           — 2.5 pts  (spread clamped [-10pp, +20pp] → [0, 1])
  FCF margin premium     — 1.5 pts
  Margin stability       — 2.0 pts  (1 - clamp(CV/0.30, 0, 1))
  Insider bonus          — 0.5 pts  (all-or-nothing if INSIDER_CONVICTION_HIGH)

Max theoretical = 10.0
"""
from __future__ import annotations

from typing import Optional


def _spread_score(spread: Optional[float], lo: float = -0.10, hi: float = 0.20) -> float:
    if spread is None:
        return 0.5
    clamped = max(lo, min(hi, spread))
    return (clamped - lo) / (hi - lo)


def _stability_score(cv: float, max_cv: float = 0.30) -> float:
    return max(0.0, 1.0 - min(cv / max_cv, 1.0))


def compute_score(computed: dict, flags: list[str]) -> float:
    gm_score  = _spread_score(computed.get("avg_gross_margin_spread")) * 2.0
    opm_score = _spread_score(computed.get("avg_op_margin_spread"),    lo=-0.08, hi=0.15) * 1.5
    roic_score = _spread_score(computed.get("roic_spread"),            lo=-0.10, hi=0.20) * 2.5
    fcf_score  = _spread_score(computed.get("avg_fcf_margin_spread"),  lo=-0.08, hi=0.15) * 1.5
    stab_score = _stability_score(computed.get("gross_margin_cv", 0.0)) * 2.0
    insider_bonus = 0.5 if "INSIDER_CONVICTION_HIGH" in flags else 0.0

    # Penalty for negative flags
    volatile_penalty = -0.5 if "MARGIN_VOLATILE" in flags else 0.0
    roic_below_penalty = -0.5 if "ROIC_BELOW_PEERS" in flags else 0.0

    raw = (gm_score + opm_score + roic_score + fcf_score
           + stab_score + insider_bonus + volatile_penalty + roic_below_penalty)
    return round(max(0.0, min(10.0, raw)), 2)
