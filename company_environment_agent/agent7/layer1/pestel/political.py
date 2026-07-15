"""PESTEL – Political dimension.

Quantitative signals:
  - Sector government dependency (config-based, 0–1)
  - Trade sensitivity (config-based, -1 to +1)
  - USD strength (UUP 6-month cumulative return)
  - FX impact = trade_sensitivity × (−uup_return / 0.08)
      normalised so that ±8% USD move ≈ ±1.0 raw impact unit

political_score_raw ∈ [-1, +1], where:
  +1 = strong political / FX tailwind
  -1 = strong political / FX headwind
"""
from __future__ import annotations

import pandas as pd


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def political_signals(
    sector: str,
    cfg: dict,          # political_exposure.yaml contents
    p_uup: pd.Series,   # UUP price series (~6 months)
) -> dict:
    """
    Returns a dict with political quantitative signals.

    Keys:
      political_govt_dependency   float [0,1]
      political_trade_sensitivity float [-1,1]
      uup_return_6m               float | None   (positive = USD appreciated)
      political_fx_impact         float [-1,1]   (positive = FX tailwind)
      political_score_raw         float [-1,1]
    """
    sec_cfg = cfg.get(sector, cfg.get("default", {}))
    govt_dep   = float(sec_cfg.get("govt_dependency", 0.40))
    trade_sens = float(sec_cfg.get("trade_sensitivity", -0.20))

    # USD strength: positive return = USD appreciated
    uup_ret: float | None = None
    if len(p_uup) > 1:
        uup_ret = float(p_uup.iloc[-1] / p_uup.iloc[0] - 1)

    # FX impact: negative trade_sensitivity (exporter) × strong USD → headwind (negative score)
    # Normalised: 8% USD move → full ±|trade_sens| contribution
    fx_impact = 0.0
    if uup_ret is not None:
        raw = trade_sens * uup_ret / 0.08
        fx_impact = _clip(raw)

    # political_score_raw: FX is the main quant driver; govt dependency is context for LLM
    political_score_raw = fx_impact

    return {
        "political_govt_dependency":   govt_dep,
        "political_trade_sensitivity": trade_sens,
        "uup_return_6m":               uup_ret,
        "political_fx_impact":         fx_impact,
        "political_score_raw":         political_score_raw,
    }
