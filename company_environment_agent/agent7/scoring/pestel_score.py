"""Compute 0-100 sub-scores for each PESTEL dimension from the Layer 1 bundle.

Each sub-score maps a raw ∈ [-1, +1] signal to [0, 100]:
    sub_score = 50 + 50 × raw

50 = neutral  |  >50 = tailwind  |  <50 = headwind

The six sub-scores feed into quant_score() in quant_score.py with PESTEL weights.
"""

from __future__ import annotations


def _to_score(raw: float | None, default: float = 0.0) -> float:
    """Map raw ∈ [-1, +1] → [0, 100].  None → neutral 50."""
    if raw is None:
        return 50.0
    clamped = max(-1.0, min(1.0, float(raw)))
    return round(50.0 + 50.0 * clamped, 2)


def political_sub_score(pestel: dict) -> float:
    """
    Political sub-score (0-100).

    Quant driver: political_fx_impact (FX headwind/tailwind from USD strength × trade sensitivity).
    Government dependency is contextual — captured by LLM; not directionally scored here.
    """
    return _to_score(pestel.get("political_score_raw"))


def economic_sub_score(pestel: dict) -> float:
    """
    Economic sub-score (0-100).

    Captures: sector momentum, alpha, VIX regime, rate regime, market trend,
              credit spread health, inflation pressure, peer revenue/margin gaps.
    This is the richest dimension quantitatively.
    """
    return _to_score(pestel.get("economic_score_raw"))


def social_sub_score(pestel: dict) -> float:
    """
    Social sub-score (0-100).

    Captures: XLY/XLP consumer sentiment z-score, R&D intensity gap vs peers.
    """
    return _to_score(pestel.get("social_score_raw"))


def technological_sub_score(pestel: dict) -> float:
    """
    Technological sub-score (0-100).

    Captures: R&D gap vs peers, CapEx gap vs peers, sector disruption exposure.
    """
    return _to_score(pestel.get("tech_score_raw"))


def environmental_sub_score(pestel: dict) -> float:
    """
    Environmental sub-score (0-100).

    Captures: sector carbon intensity, environmental regulatory burden,
              commodity energy-cost impact.
    """
    return _to_score(pestel.get("environmental_score_raw"))


def legal_sub_score(pestel: dict) -> float:
    """
    Legal sub-score (0-100).

    Captures: sector regulatory complexity, litigation exposure.
    LLM layer grounds these with 10-K risk-factor language and live regulatory news.
    """
    return _to_score(pestel.get("legal_score_raw"))


def all_pestel_quant_scores(bundle: dict) -> dict:
    """
    Compute all six PESTEL quant sub-scores from the bundle.

    Returns:
      {
        "P": <float>,   Political
        "E": <float>,   Economic
        "S": <float>,   Social
        "T": <float>,   Technological
        "En": <float>,  Environmental
        "L": <float>,   Legal
      }
    """
    pestel = bundle.get("pestel", {})
    return {
        "P":  political_sub_score(pestel),
        "E":  economic_sub_score(pestel),
        "S":  social_sub_score(pestel),
        "T":  technological_sub_score(pestel),
        "En": environmental_sub_score(pestel),
        "L":  legal_sub_score(pestel),
    }
