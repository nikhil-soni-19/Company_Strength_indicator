"""Compute the overall quant_score (0-100) as a PESTEL-weighted blend.

PESTEL dimension weights:
  P  – Political       12 %   (quant signal limited to FX/trade)
  E  – Economic        30 %   (richest quant layer: momentum, regimes, spreads, peers)
  S  – Social          15 %   (consumer sentiment, R&D gap)
  T  – Technological   15 %   (R&D gap, CapEx gap, disruption exposure)
  En – Environmental   13 %   (commodity impact, carbon + env burden from config)
  L  – Legal           15 %   (regulatory complexity + litigation from config)
                      ─────
                      100 %

50 = neutral.  Each sub-score ∈ [0, 100].
"""

from __future__ import annotations

from scoring.pestel_score import all_pestel_quant_scores

# PESTEL weights (must sum to 1.0)
_WEIGHTS = {
    "P":  0.12,
    "E":  0.30,
    "S":  0.15,
    "T":  0.15,
    "En": 0.13,
    "L":  0.15,
}


def quant_score(bundle: dict) -> float:
    """
    Compute overall quant_score (0-100) as PESTEL-weighted average.

    Args:
        bundle: dict returned by layer1.bundle.build_bundle()

    Returns:
        float in [0, 100]; 50 = neutral environment.
    """
    scores = all_pestel_quant_scores(bundle)
    weighted = sum(_WEIGHTS[dim] * scores[dim] for dim in _WEIGHTS)
    return float(round(weighted, 2))
