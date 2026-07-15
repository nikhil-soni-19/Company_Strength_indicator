"""PESTEL – Technological dimension.

Quantitative signals:
  - R&D intensity gap vs peers (company R&D/rev − peer median)
      Positive = company outspending peers on R&D → tech advantage
  - CapEx intensity gap vs peers (company CapEx/rev − peer median)
      Positive = heavier capital investment → future capacity / innovation
  - Sector tech disruption exposure (config-based, 0–1)
      Low disruption exposure + positive R&D/CapEx gap → strong tech score
  - rd_capex_importance (config-based, 0–1)
      Scales how much R&D/CapEx gaps matter in this sector

tech_score_raw ∈ [-1, +1]
"""
from __future__ import annotations

import statistics

import numpy as np


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def technological_signals(
    sector: str,
    cfg: dict,                               # tech_disruption.yaml contents
    rd_ratio_company: float | None,          # company R&D / revenue
    rd_ratio_peers: list[float | None],
    capex_ratio_company: float | None,       # company CapEx / revenue
    capex_ratio_peers: list[float | None],
) -> dict:
    """
    Returns technological quantitative signals.

    Keys:
      tech_disruption_exposure   float [0,1]   (config)
      rd_capex_importance        float [0,1]   (config)
      rd_intensity_gap           float | None
      capex_intensity_company    float | None
      capex_intensity_peer_med   float | None
      capex_intensity_gap        float | None
      tech_score_raw             float [-1,1]
    """
    def safe(x, default=0.0):
        return default if (x is None or (isinstance(x, float) and np.isnan(x))) else x

    sec_cfg = cfg.get(sector, cfg.get("default", {}))
    disruption  = float(sec_cfg.get("tech_disruption_exposure", 0.40))
    rd_cap_imp  = float(sec_cfg.get("rd_capex_importance", 0.50))

    # ── R&D gap ───────────────────────────────────────────────────────────────
    valid_rd = [r for r in rd_ratio_peers if r is not None]
    rd_peer_med: float | None = statistics.median(valid_rd) if valid_rd else None
    rd_gap: float | None = None
    if rd_ratio_company is not None and rd_peer_med is not None:
        rd_gap = rd_ratio_company - rd_peer_med

    # Normalise: 5pp R&D/rev gap ≈ ±1 unit
    c_rd = _clip(safe(rd_gap) / 0.05) * rd_cap_imp

    # ── CapEx gap ─────────────────────────────────────────────────────────────
    valid_cx = [r for r in capex_ratio_peers if r is not None]
    capex_peer_med: float | None = statistics.median(valid_cx) if valid_cx else None
    capex_gap: float | None = None
    if capex_ratio_company is not None and capex_peer_med is not None:
        capex_gap = capex_ratio_company - capex_peer_med

    # Normalise: 3pp CapEx/rev gap ≈ ±1 unit
    c_capex = _clip(safe(capex_gap) / 0.03) * rd_cap_imp

    # ── Disruption penalty ────────────────────────────────────────────────────
    # High disruption sector + low R&D investment = structural headwind
    # Maps disruption [0,1] to [-1,0] range (always non-positive)
    c_disruption = -(disruption - 0.40)   # neutral at 0.40; ranges from +0.40 to -0.60

    # ── Tech raw score ────────────────────────────────────────────────────────
    raw = 0.40 * c_rd + 0.30 * c_capex + 0.30 * c_disruption

    return {
        "tech_disruption_exposure":  disruption,
        "rd_capex_importance":       rd_cap_imp,
        "rd_intensity_gap":          rd_gap,
        "capex_intensity_company":   capex_ratio_company,
        "capex_intensity_peer_med":  capex_peer_med,
        "capex_intensity_gap":       capex_gap,
        "tech_score_raw":            _clip(raw),
    }
