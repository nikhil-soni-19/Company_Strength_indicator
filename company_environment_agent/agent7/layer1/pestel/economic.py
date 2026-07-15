"""PESTEL – Economic dimension.

Consolidates all macro / market-structure signals (existing + new):

Existing (passed through from bundle):
  sector_rs_6m, company_alpha_annualised, vix_zscore, rate_slope_z,
  market_trend, peer_rev_growth_gap, peer_margin_gap

New (derived from macro ETF prices):
  credit_spread_z  – z-score of HYG/IEF price ratio vs trailing window
                     Negative z → widening spreads → economic stress
  inflation_z      – z-score of TIP/IEF price ratio vs trailing window
                     Positive z → rising inflation expectations
                     Impact is sector-dependent; here we treat it as mild headwind
                     (cost pressure) unless caller overrides

economic_score_raw ∈ [-1, +1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _ratio_zscore(p_num: pd.Series, p_den: pd.Series) -> float | None:
    """Z-score of the current price ratio vs its own trailing history."""
    common = p_num.index.intersection(p_den.index)
    if len(common) < 30:
        return None
    ratio = p_num.loc[common] / p_den.loc[common]
    mu = float(ratio.mean())
    sd = float(ratio.std())
    if sd == 0:
        return None
    return float((ratio.iloc[-1] - mu) / sd)


def economic_signals(
    sector_rs_6m: float | None,
    alpha_annualised: float | None,
    vix_zscore: float | None,
    rate_slope_z: float | None,
    market_trend: str | None,
    peer_rev_growth_gap: float | None,
    peer_margin_gap: float | None,
    p_hyg: pd.Series,   # HYG prices
    p_ief: pd.Series,   # IEF prices
    p_tip: pd.Series,   # TIP prices
) -> dict:
    """
    Returns economic quantitative signals.

    Keys:
      credit_spread_z    float | None   (negative = spread widening = stress)
      inflation_z        float | None   (positive = rising breakeven inflation)
      economic_score_raw float [-1,1]
    """
    def safe(x, default=0.0):
        return default if (x is None or (isinstance(x, float) and np.isnan(x))) else x

    # ── New macro signals ──────────────────────────────────────────────────────
    credit_spread_z = _ratio_zscore(p_hyg, p_ief)   # positive = HY outperforming = tight spreads
    inflation_z     = _ratio_zscore(p_tip, p_ief)    # positive = inflation expectations rising

    # ── Component contributions (each ∈ [-1, 1]) ─────────────────────────────
    # Relative strength vs sector
    c_rs = _clip((safe(sector_rs_6m, 1.0) - 1.0) / 0.08)

    # Alpha (annualised; 10% alpha ≈ ±1 unit)
    c_alpha = _clip(safe(alpha_annualised) / 0.10)

    # Macro regime: low VIX + flat-to-falling rates + bull market = positive
    trend_pt = {"BULL": 1.0, "TRANSITION": 0.0, "BEAR": -1.0}.get(market_trend or "", 0.0)
    c_macro = _clip((-safe(vix_zscore) - safe(rate_slope_z) + trend_pt) / 3.0)

    # Credit spreads: tight spreads (positive z) = positive economic signal
    c_credit = _clip(safe(credit_spread_z) / 1.5)

    # Inflation: mild headwind regardless of direction (cost uncertainty)
    c_inflation = _clip(-abs(safe(inflation_z)) / 2.0)

    # Peer fundamentals
    c_growth = _clip(safe(peer_rev_growth_gap) / 0.10)
    c_margin = _clip(safe(peer_margin_gap) / 0.05)

    # Weighted economic raw score
    # Weights: momentum/alpha 25%, macro regime 25%, credit 15%, inflation 5%, peers 30%
    raw = (
        0.15 * c_rs     +
        0.10 * c_alpha  +
        0.25 * c_macro  +
        0.15 * c_credit +
        0.05 * c_inflation +
        0.15 * c_growth +
        0.15 * c_margin
    )

    return {
        "credit_spread_z":    credit_spread_z,
        "inflation_z":        inflation_z,
        "economic_score_raw": _clip(raw),
    }
