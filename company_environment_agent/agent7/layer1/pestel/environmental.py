"""PESTEL – Environmental dimension.

Quantitative signals:
  - Commodity impact (existing signal, repurposed here as energy/input cost signal)
  - Carbon intensity (config-based, 0–1)
      High emitters face structural regulatory headwinds and rising carbon cost risk
  - Environmental regulatory burden (config-based, 0–1)
      Sectors with heavy EPA/ESG oversight face compliance cost headwinds

environmental_score_raw ∈ [-1, +1], where:
  +1 = minimal environmental headwinds (clean sector, low burden)
  -1 = severe environmental headwinds (high carbon, heavy ESG regulation)

Note: LLM layer adjusts this with live climate/ESG regulatory news and
      company-specific 10-K environmental risk disclosures.
"""
from __future__ import annotations

import numpy as np


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def environmental_signals(
    sector: str,
    cfg: dict,                   # environmental_exposure.yaml contents
    commodity_impact_raw: float | None,  # existing commodity signal [-1,1]
) -> dict:
    """
    Returns environmental quantitative signals.

    Keys:
      carbon_intensity               float [0,1]  (config)
      environmental_regulatory_burden float [0,1] (config)
      commodity_env_impact           float [-1,1] (passed through)
      environmental_score_raw        float [-1,1]
    """
    def safe(x, default=0.0):
        return default if (x is None or (isinstance(x, float) and np.isnan(x))) else x

    sec_cfg   = cfg.get(sector, cfg.get("default", {}))
    carbon    = float(sec_cfg.get("carbon_intensity", 0.35))
    env_burden = float(sec_cfg.get("environmental_regulatory_burden", 0.38))

    # Config-based headwinds:
    # Carbon intensity: 0.0 → neutral, 1.0 → heavy headwind (−1 contribution)
    # Environmental burden: same logic
    # Both mapped from [0,1] to [-1,0] — they are purely headwinds in this model
    c_carbon = -(carbon - 0.20) / 0.80        # 0.20 threshold: below = neutral, above = headwind
    c_burden  = -(env_burden - 0.20) / 0.80

    c_carbon = _clip(c_carbon)
    c_burden  = _clip(c_burden)

    # Commodity impact: already computed in Layer 1 (energy/materials cost pressure)
    comm = safe(commodity_impact_raw)
    c_commodity = _clip(comm)

    # Weighted env score
    raw = 0.35 * c_carbon + 0.35 * c_burden + 0.30 * c_commodity

    return {
        "carbon_intensity":               carbon,
        "environmental_regulatory_burden": env_burden,
        "commodity_env_impact":           commodity_impact_raw,
        "environmental_score_raw":        _clip(raw),
    }
