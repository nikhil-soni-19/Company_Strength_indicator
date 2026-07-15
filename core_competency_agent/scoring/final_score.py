"""
Fuse Layer 1 (55%) + Layer 2 (45%) into the final moat score.

Final moat_score is on a 0-100 scale.
When narrative contradicts numbers, the conflict is surfaced — not averaged away.
"""
from __future__ import annotations

from typing import Optional


def fuse(
    l1_score: float,
    l2_score: float,
    narrative_vs_numbers: str,
) -> dict:
    """
    Args:
        l1_score:              Layer 1 score, 0-10
        l2_score:              Layer 2 score, 0-10
        narrative_vs_numbers:  "consistent" | "conflict" | "insufficient_data"

    Returns:
        {moat_score, direction_override, conflict_penalty_applied}
    """
    blended = l1_score * 0.55 + l2_score * 0.45
    conflict_penalty_applied = False

    # When narrative contradicts numbers, reduce the score — don't silently average
    if narrative_vs_numbers == "conflict":
        blended = max(0.0, blended - 1.0)
        conflict_penalty_applied = True

    moat_score = round(blended * 10, 1)  # scale 0-10 → 0-100
    moat_score = max(0.0, min(100.0, moat_score))

    return {
        "moat_score": moat_score,
        "conflict_penalty_applied": conflict_penalty_applied,
    }


def score_to_label(moat_score: float) -> str:
    if moat_score >= 70:
        return "STRONG MOAT"
    elif moat_score >= 50:
        return "MODERATE MOAT"
    elif moat_score >= 30:
        return "NARROW MOAT"
    else:
        return "NO MOAT / CYCLICAL"
