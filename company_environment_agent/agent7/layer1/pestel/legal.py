"""PESTEL – Legal dimension.

Quantitative signals:
  - Regulatory complexity (config-based, 0–1)
      Heavy compliance sectors (Financials, Health Care, Utilities) start with
      a structural Legal headwind even in benign news environments.
  - Litigation exposure (config-based, 0–1)
      High litigation sectors face contingent liability risk.

Both dimensions are headwinds — high scores lower the Legal sub-score.
The LLM layer grounds these config baselines with:
  • Live regulatory/antitrust/enforcement news (Tavily Legal search)
  • 10-K risk-factor language (legal proceedings, compliance cost disclosures)

legal_score_raw ∈ [-1, +1], where:
  +1 = very light regulatory touch, minimal litigation risk
  -1 = very heavily regulated with high active litigation exposure
"""
from __future__ import annotations


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def legal_signals(
    sector: str,
    cfg: dict,   # legal_regulatory_burden.yaml contents
) -> dict:
    """
    Returns legal quantitative signals.

    Keys:
      regulatory_complexity   float [0,1]  (config)
      litigation_exposure     float [0,1]  (config)
      legal_score_raw         float [-1,1]
    """
    sec_cfg    = cfg.get(sector, cfg.get("default", {}))
    reg_cmplx  = float(sec_cfg.get("regulatory_complexity", 0.48))
    lit_exp    = float(sec_cfg.get("litigation_exposure", 0.42))

    # Map [0,1] to [-1,0]: both are pure headwinds
    # Threshold of 0.30 = light-touch baseline (neutral); above it = headwind
    c_reg = _clip(-(reg_cmplx - 0.30) / 0.70)
    c_lit = _clip(-(lit_exp   - 0.30) / 0.70)

    # Equal weight: regulatory burden and litigation risk
    raw = 0.55 * c_reg + 0.45 * c_lit

    return {
        "regulatory_complexity": reg_cmplx,
        "litigation_exposure":   lit_exp,
        "legal_score_raw":       _clip(raw),
    }
