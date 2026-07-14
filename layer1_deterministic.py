"""
Phase 2 — Layer 1: deterministic capability-stack engine (60% spine).

This module is intentionally isolated: no LLM, no DB, no network.
It runs purely from an InputBundle in memory and produces a Layer1Output struct.

Design principle: compute everything deterministically here; let Layer 2 interpret
only what genuinely requires judgment (narrative, ESG, governance signals).

Thresholds are defined in config.py and overridable per-sector via sector_thresholds
dict passed to run_layer1() — no need to modify this file for sector calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    L1_R2_LOW_THRESHOLD,
    CAPEX_LIGHT_FLOOR,
    CAPEX_REINVESTMENT_STRONG_LEVEL,
    CAPEX_REINVESTMENT_STRONG_SLOPE,
    INSIDER_CONVICTION_THRESHOLD,
    INST_CONCENTRATION_THRESHOLD,
    L1_BASE_SCORE,
    L1_BONUS_PER_PCT_CAPEX,
    L1_BONUS_PER_PCT_RD,
    L1_DELTA_CAPEX_LIGHT,
    L1_DELTA_CAPEX_STRONG,
    L1_DELTA_INSIDER_CONVICTION,
    L1_DELTA_INST_CONCENTRATION,
    L1_DELTA_RD_INTENSIFYING,
    RD_INTENSIFYING_SLOPE_THRESHOLD,
)
from data_contract import DataCoverage, InputBundle


# ─── Output type ──────────────────────────────────────────────────────────────

@dataclass
class Layer1Output:
    ticker: str
    periods: list[str]       # YYYY-MM-DD, oldest → newest

    # Per-quarter ratios
    rd_rev: list[float]      # R&D / revenue
    capex_rev: list[float]   # capex / revenue

    # Summary statistics — most recent value + trend
    rd_rev_level: float      # current (most recent quarter)
    rd_rev_slope: float      # OLS slope per quarter-index
    rd_rev_cagr: float       # annualised CAGR of the ratio

    capex_rev_level: float
    capex_rev_slope: float
    capex_rev_cagr: float

    # Governance signals (from yfinance; may be None)
    insider_pct: Optional[float]
    institutional_top10: Optional[float]

    # ── Time-series analytics (all derived from the per-quarter series above) ──
    # TTM: sum of last 4Q numerator / sum of last 4Q denominator — removes seasonal distortion
    rd_rev_ttm: float
    capex_rev_ttm: float
    # YoY: current Q minus same Q last year — isolates real trend from adjacent-quarter noise
    rd_rev_yoy: Optional[float]    # None if fewer than 5 quarters available
    capex_rev_yoy: Optional[float]
    # CV: coefficient of variation (std / mean) — rewards steady execution over jumpy series
    rd_rev_cv: float
    capex_rev_cv: float
    # R²: how well the OLS line fits — low R² means the reported slope is noise
    rd_rev_r2: float
    capex_rev_r2: float
    # Self-relative percentile: where current value sits in its own 3-year range
    # 1.0 = at historic high, 0.0 = at historic low
    rd_rev_pct: float
    capex_rev_pct: float

    # Boolean flags
    flags: list[str]
    # Possible flags:
    #   R&D_INTENSIFYING          — R&D/rev slope is accelerating
    #   CAPEX_REINVESTMENT_STRONG — heavy or rapidly rising capex intensity
    #   CAPEX_LIGHT_BUSINESS      — asset-light model (mutually exclusive with above)
    #   INSIDER_CONVICTION_HIGH   — insider ownership > threshold (skin-in-the-game)
    #   INST_CONCENTRATION_HIGH   — top-10 institutions own > threshold of shares

    # Layer 1 capability score [0, 10]
    l1_score: float

    # Pass-through for fusion + guardrail
    data_coverage: DataCoverage


# ─── Math helpers ─────────────────────────────────────────────────────────────

def _safe_ratio(num: float, den: float) -> float:
    """Return num/den, or 0.0 if denominator is zero or negative."""
    return num / den if den > 1e-9 else 0.0


def _ols_slope(series: list[float]) -> float:
    """
    OLS slope of series vs quarter-index [0, 1, …, n-1].
    Returns 0.0 for series shorter than 2 elements or all-zero series.
    """
    n = len(series)
    if n < 2:
        return 0.0
    y = np.array(series, dtype=float)
    if np.allclose(y, 0):
        return 0.0
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _ratio_cagr(series: list[float]) -> float:
    """
    Annualised CAGR of a quarterly ratio series.
        n_years = (n_quarters - 1) / 4
    Returns 0.0 if the first value is zero, the last is non-positive, or the
    series is too short to compute a meaningful rate.
    """
    n = len(series)
    if n < 2:
        return 0.0
    first, last = series[0], series[-1]
    if first <= 1e-9 or last <= 0:
        return 0.0
    n_years = (n - 1) / 4.0
    if n_years < 0.01:
        return 0.0
    return float((last / first) ** (1.0 / n_years) - 1.0)


def _ttm_ratio(num_series: list[float], den_series: list[float]) -> float:
    """
    Trailing 4-quarter (TTM) ratio: sum(last 4Q num) / sum(last 4Q den).
    Eliminates single-quarter seasonal distortion (e.g. December revenue spike
    artificially compressing the R&D/Rev ratio for tech companies).
    Falls back to all available quarters if fewer than 4.
    """
    n = min(4, len(num_series))
    return _safe_ratio(sum(num_series[-n:]), sum(den_series[-n:]))


def _yoy_change(series: list[float]) -> Optional[float]:
    """
    Year-over-year change: current quarter minus same quarter last year.
    Isolates real trend from adjacent-quarter seasonal noise.
    Returns None when fewer than 5 quarters are available.
    """
    if len(series) < 5:
        return None
    return series[-1] - series[-5]


def _coeff_variation(series: list[float]) -> float:
    """
    Coefficient of variation (std / mean) of the ratio series.
    Measures execution consistency — a rising-but-jumpy series scores lower
    than a rising-and-smooth one. Returns 0.0 if the mean is near zero.
    """
    if len(series) < 2:
        return 0.0
    arr = np.array(series, dtype=float)
    m = np.mean(arr)
    return float(np.std(arr) / m) if m > 1e-9 else 0.0


def _ols_r2(series: list[float]) -> float:
    """
    R² of the OLS linear fit (value vs quarter-index).
    Measures how well the slope actually describes the data.
    High R² (> 0.5) → slope is a reliable trend.
    Low R² (< 0.3)  → slope is noise; down-weight slope-based flags.
    Returns 0.0 for series shorter than 3 elements or with zero variance.
    """
    n = len(series)
    if n < 3:
        return 0.0
    y = np.array(series, dtype=float)
    if np.allclose(y, 0):
        return 0.0
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def _self_percentile(series: list[float]) -> float:
    """
    Where the current (last) value sits within the series' own historical range.
    1.0 = at or above historic high (3-year high).
    0.0 = at or below historic low.
    0.5 returned for flat or too-short series.
    Needs no peer data — the 12-quarter series is its own baseline.
    """
    if len(series) < 2:
        return 0.5
    arr = np.array(series, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return 0.5
    return float((arr[-1] - mn) / (mx - mn))


# ─── Flag logic ───────────────────────────────────────────────────────────────

def _compute_flags(
    rd_rev_level: float,
    rd_rev_slope: float,
    capex_rev_level: float,
    capex_rev_slope: float,
    insider_pct: Optional[float],
    institutional_top10: Optional[float],
    thresholds: dict | None = None,
) -> list[str]:
    """
    Evaluate all capability flags. All thresholds are pulled from config.py by
    default; pass a dict with any subset of threshold keys to override for
    sector-specific calibration.

    R&D_INTENSIFYING
        R&D/revenue OLS slope exceeds the threshold — active intensification.

    CAPEX_REINVESTMENT_STRONG
        Capex/rev level ≥ strong level OR slope ≥ strong slope.
        Signals heavy, sustained physical reinvestment.

    CAPEX_LIGHT_BUSINESS
        Capex/rev below the asset-light floor. Mutually exclusive with
        CAPEX_REINVESTMENT_STRONG by construction.

    INSIDER_CONVICTION_HIGH
        Insider ownership fraction exceeds the threshold. Signals management
        has meaningful skin in the game — a positive governance signal.

    INST_CONCENTRATION_HIGH
        Top-10 institutional holders own more than the threshold fraction of
        shares outstanding. Signals strong smart-money conviction.
    """
    t = thresholds or {}

    rd_slope_thresh    = t.get("RD_INTENSIFYING_SLOPE_THRESHOLD",  RD_INTENSIFYING_SLOPE_THRESHOLD)
    capex_strong_level = t.get("CAPEX_REINVESTMENT_STRONG_LEVEL",  CAPEX_REINVESTMENT_STRONG_LEVEL)
    capex_strong_slope = t.get("CAPEX_REINVESTMENT_STRONG_SLOPE",  CAPEX_REINVESTMENT_STRONG_SLOPE)
    capex_light_floor  = t.get("CAPEX_LIGHT_FLOOR",                CAPEX_LIGHT_FLOOR)
    insider_thresh     = t.get("INSIDER_CONVICTION_THRESHOLD",     INSIDER_CONVICTION_THRESHOLD)
    inst_thresh        = t.get("INST_CONCENTRATION_THRESHOLD",     INST_CONCENTRATION_THRESHOLD)

    flags: list[str] = []

    if rd_rev_slope > rd_slope_thresh:
        flags.append("R&D_INTENSIFYING")

    if capex_rev_level >= capex_strong_level or capex_rev_slope > capex_strong_slope:
        flags.append("CAPEX_REINVESTMENT_STRONG")
    elif capex_rev_level < capex_light_floor:
        # Only evaluated in the else-branch — guarantees mutual exclusivity.
        flags.append("CAPEX_LIGHT_BUSINESS")

    if insider_pct is not None and insider_pct > insider_thresh:
        flags.append("INSIDER_CONVICTION_HIGH")

    if institutional_top10 is not None and institutional_top10 > inst_thresh:
        flags.append("INST_CONCENTRATION_HIGH")

    return flags


# ─── Layer 1 scoring ──────────────────────────────────────────────────────────

def _compute_l1_score(
    rd_rev_level: float,
    capex_rev_level: float,
    flags: list[str],
) -> float:
    """
    Map ratios and flags to a [0, 10] capability score.

    Structure: base score + flag deltas + continuous bonuses.
    Flag deltas encode the direction and magnitude of each signal.
    Continuous bonuses ensure a high-intensity firm still scores well
    even if it doesn't clear the slope threshold for the binary flag.
    Score is clamped to [0, 10].
    """
    score = L1_BASE_SCORE

    if "R&D_INTENSIFYING" in flags:
        score += L1_DELTA_RD_INTENSIFYING
    if "CAPEX_REINVESTMENT_STRONG" in flags:
        score += L1_DELTA_CAPEX_STRONG
    if "CAPEX_LIGHT_BUSINESS" in flags:
        score += L1_DELTA_CAPEX_LIGHT
    if "INSIDER_CONVICTION_HIGH" in flags:
        score += L1_DELTA_INSIDER_CONVICTION
    if "INST_CONCENTRATION_HIGH" in flags:
        score += L1_DELTA_INST_CONCENTRATION

    # Continuous bonus: each percentage point of ratio intensity adds a small amount.
    score += rd_rev_level * 100.0 * L1_BONUS_PER_PCT_RD
    score += capex_rev_level * 100.0 * L1_BONUS_PER_PCT_CAPEX

    return float(max(0.0, min(10.0, score)))


# ─── Public entry point ───────────────────────────────────────────────────────

def run_layer1(
    bundle: InputBundle,
    sector_thresholds: dict | None = None,
) -> Layer1Output:
    """
    Compute all Layer 1 metrics from an InputBundle.

    Pure Python + NumPy: no LLM, no DB, no network calls.
    Safe to import and run in offline tests or CI without any credentials.

    Args:
        bundle:            InputBundle from data_contract.fetch_inputs().
        sector_thresholds: Optional dict overriding specific threshold keys.
                           Example: {"CAPEX_LIGHT_FLOOR": 0.03} for a sector
                           where 3% capex/rev is the asset-light boundary.

    Returns:
        Layer1Output with per-quarter ratios, summary statistics, flags, and
        a [0, 10] l1_score.
    """
    n = len(bundle.periods)

    # ── Ratios ────────────────────────────────────────────────────────────────
    rd_rev    = [_safe_ratio(bundle.rd[i],    bundle.revenue[i]) for i in range(n)]
    capex_rev = [_safe_ratio(bundle.capex[i], bundle.revenue[i]) for i in range(n)]

    # ── Summary statistics ────────────────────────────────────────────────────
    rd_rev_level    = rd_rev[-1]    if rd_rev    else 0.0
    capex_rev_level = capex_rev[-1] if capex_rev else 0.0

    rd_rev_slope    = _ols_slope(rd_rev)
    capex_rev_slope = _ols_slope(capex_rev)

    rd_rev_cagr    = _ratio_cagr(rd_rev)
    capex_rev_cagr = _ratio_cagr(capex_rev)

    # ── Time-series analytics ─────────────────────────────────────────────────
    rd_rev_ttm  = _ttm_ratio(bundle.rd, bundle.revenue)
    rd_rev_yoy  = _yoy_change(rd_rev)
    rd_rev_cv   = _coeff_variation(rd_rev)
    rd_rev_r2   = _ols_r2(rd_rev)
    rd_rev_pct  = _self_percentile(rd_rev)

    capex_rev_ttm = _ttm_ratio(bundle.capex, bundle.revenue)
    capex_rev_yoy = _yoy_change(capex_rev)
    capex_rev_cv  = _coeff_variation(capex_rev)
    capex_rev_r2  = _ols_r2(capex_rev)
    capex_rev_pct = _self_percentile(capex_rev)

    # ── Flags ─────────────────────────────────────────────────────────────────
    flags = _compute_flags(
        rd_rev_level, rd_rev_slope,
        capex_rev_level, capex_rev_slope,
        bundle.insider_pct,
        bundle.institutional_top10,
        sector_thresholds,
    )

    # ── Score ─────────────────────────────────────────────────────────────────
    l1_score = _compute_l1_score(rd_rev_level, capex_rev_level, flags)

    return Layer1Output(
        ticker=bundle.ticker,
        periods=bundle.periods,
        rd_rev=rd_rev,
        capex_rev=capex_rev,
        rd_rev_level=rd_rev_level,
        rd_rev_slope=rd_rev_slope,
        rd_rev_cagr=rd_rev_cagr,
        capex_rev_level=capex_rev_level,
        capex_rev_slope=capex_rev_slope,
        capex_rev_cagr=capex_rev_cagr,
        insider_pct=bundle.insider_pct,
        institutional_top10=bundle.institutional_top10,
        rd_rev_ttm=rd_rev_ttm,
        capex_rev_ttm=capex_rev_ttm,
        rd_rev_yoy=rd_rev_yoy,
        capex_rev_yoy=capex_rev_yoy,
        rd_rev_cv=rd_rev_cv,
        capex_rev_cv=capex_rev_cv,
        rd_rev_r2=rd_rev_r2,
        capex_rev_r2=capex_rev_r2,
        rd_rev_pct=rd_rev_pct,
        capex_rev_pct=capex_rev_pct,
        flags=flags,
        l1_score=l1_score,
        data_coverage=bundle.coverage,
    )
