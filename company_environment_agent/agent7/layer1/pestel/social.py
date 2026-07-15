"""PESTEL – Social dimension.

Quantitative signals:
  - Consumer sentiment proxy: XLY/XLP price ratio z-score
      XLY outperforming XLP → consumer spending strength → positive
  - R&D intensity gap vs peers: company R&D/revenue − peer median R&D/revenue
      Positive gap → company investing more in human capital / innovation talent
  - Labor cost intensity: SG&A/revenue (proxy for workforce investment)
      Sector-relative gap signals competitive social positioning

social_score_raw ∈ [-1, +1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _ratio_zscore(p_num: pd.Series, p_den: pd.Series) -> float | None:
    common = p_num.index.intersection(p_den.index)
    if len(common) < 30:
        return None
    ratio = p_num.loc[common] / p_den.loc[common]
    mu = float(ratio.mean())
    sd = float(ratio.std())
    if sd == 0:
        return None
    return float((ratio.iloc[-1] - mu) / sd)


def social_signals(
    p_xly: pd.Series,             # XLY price series
    p_xlp: pd.Series,             # XLP price series
    rd_ratio_company: float | None,   # company R&D / revenue
    rd_ratio_peers: list[float | None],  # peer R&D / revenue ratios
) -> dict:
    """
    Returns social quantitative signals.

    Keys:
      consumer_sentiment_z   float | None  (positive = consumer spending strong)
      rd_intensity_company   float | None
      rd_intensity_peer_med  float | None
      rd_intensity_gap       float | None  (positive = company outspends peers on R&D)
      social_score_raw       float [-1,1]
    """
    def safe(x, default=0.0):
        return default if (x is None or (isinstance(x, float) and np.isnan(x))) else x

    # ── Consumer sentiment ────────────────────────────────────────────────────
    consumer_z = _ratio_zscore(p_xly, p_xlp)   # positive = XLY outperforms = optimism
    c_consumer = _clip(safe(consumer_z) / 1.5)

    # ── R&D intensity gap ─────────────────────────────────────────────────────
    valid_peers = [r for r in rd_ratio_peers if r is not None]
    import statistics
    rd_peer_med: float | None = statistics.median(valid_peers) if valid_peers else None
    rd_gap: float | None = None
    if rd_ratio_company is not None and rd_peer_med is not None:
        rd_gap = rd_ratio_company - rd_peer_med

    # Normalise: 5pp gap ≈ ±1 unit (R&D/rev gaps rarely exceed 10pp)
    c_rd = _clip(safe(rd_gap) / 0.05)

    # ── Social raw score ──────────────────────────────────────────────────────
    # Consumer sentiment weighted higher (observable macro signal)
    raw = 0.60 * c_consumer + 0.40 * c_rd

    return {
        "consumer_sentiment_z":  consumer_z,
        "rd_intensity_company":  rd_ratio_company,
        "rd_intensity_peer_med": rd_peer_med,
        "rd_intensity_gap":      rd_gap,
        "social_score_raw":      _clip(raw),
    }
