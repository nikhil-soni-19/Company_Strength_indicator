"""Emit boolean flag labels from Layer 1 computed values.

Flags are grouped by PESTEL dimension so the LLM can anchor
its per-dimension narrative to specific signal conditions.
"""

from __future__ import annotations


def emit_flags(
    # ── Existing signals ──────────────────────────────────────────────────────
    sector_rs_6m: float | None = None,
    rate_regime: str | None = None,
    beta_rate: float | None = None,
    vix_zscore: float | None = None,
    market_trend: str | None = None,
    commodity_tag: str | None = None,
    peer_rev_growth_gap: float | None = None,
    peer_margin_gap: float | None = None,
    # Existing thresholds
    rs_lead_thresh: float = 1.0,
    rs_lag_thresh: float = 1.0,
    rate_beta_thresh: float = -2.0,
    vix_high: float = 1.0,
    vix_low: float = -1.0,
    rev_growth_gap_pos: float = 0.05,
    rev_growth_gap_neg: float = -0.05,
    margin_gap_pos: float = 0.03,
    margin_gap_neg: float = -0.03,
    # ── PESTEL signals ────────────────────────────────────────────────────────
    # Political
    political_fx_impact: float | None = None,
    political_govt_dependency: float | None = None,
    # Economic
    credit_spread_z: float | None = None,
    inflation_z: float | None = None,
    # Social
    consumer_sentiment_z: float | None = None,
    rd_intensity_gap: float | None = None,
    # Technological
    capex_intensity_gap: float | None = None,
    tech_disruption_exposure: float | None = None,
    # Environmental
    environmental_score_raw: float | None = None,
    # Legal
    legal_score_raw: float | None = None,
    # PESTEL thresholds
    fx_impact_thresh: float = 0.25,         # |fx_impact| > threshold → flag
    credit_spread_z_thresh: float = -1.0,   # z < threshold → CREDIT_STRESS
    inflation_z_thresh: float = 1.5,        # z > threshold → INFLATION_PRESSURE
    consumer_z_thresh: float = 1.0,         # |z| > threshold → consumer flag
    rd_gap_pos_thresh: float = 0.02,        # rd_gap > thresh → TECH_LEADER
    rd_gap_neg_thresh: float = -0.02,       # rd_gap < thresh → TECH_LAGGARD
    capex_gap_pos_thresh: float = 0.02,
    capex_gap_neg_thresh: float = -0.02,
    env_headwind_thresh: float = -0.30,     # env_score_raw < thresh → ENV_BURDEN_HIGH
    legal_headwind_thresh: float = -0.25,   # legal_score_raw < thresh → LEGAL_RISK_HIGH
    govt_dep_thresh: float = 0.60,          # govt_dependency > thresh → HIGH_GOVT_DEPENDENCY
) -> list[str]:
    flags: list[str] = []

    # ── Economic (existing) ───────────────────────────────────────────────────
    if sector_rs_6m is not None:
        if sector_rs_6m > rs_lead_thresh:
            flags.append("SECTOR_LEADING")
        elif sector_rs_6m < rs_lag_thresh:
            flags.append("SECTOR_LAGGING")

    if rate_regime == "RISING_RATE":
        flags.append("RISING_RATE")
    elif rate_regime == "FALLING_RATE":
        flags.append("FALLING_RATE")

    if beta_rate is not None and beta_rate < rate_beta_thresh:
        flags.append("RATE_SENSITIVE")

    if vix_zscore is not None:
        if vix_zscore > vix_high:
            flags.append("HIGH_VOLATILITY")
        elif vix_zscore < vix_low:
            flags.append("LOW_VOLATILITY")

    if market_trend == "BULL":
        flags.append("MARKET_BULLISH")
    elif market_trend == "BEAR":
        flags.append("MARKET_BEARISH")

    if commodity_tag == "COMMODITY_TAILWIND":
        flags.append("COMMODITY_TAILWIND")
    elif commodity_tag == "COMMODITY_HEADWIND":
        flags.append("COMMODITY_HEADWIND")

    if peer_rev_growth_gap is not None:
        if peer_rev_growth_gap > rev_growth_gap_pos:
            flags.append("PEER_GAINING_GROUND")
        elif peer_rev_growth_gap < rev_growth_gap_neg:
            flags.append("PEER_LOSING_GROUND")

    if peer_margin_gap is not None:
        if peer_margin_gap > margin_gap_pos:
            flags.append("MARGIN_LEADER")
        elif peer_margin_gap < margin_gap_neg:
            flags.append("MARGIN_LAGGARD")

    # ── Political flags ───────────────────────────────────────────────────────
    if political_fx_impact is not None:
        if political_fx_impact < -fx_impact_thresh:
            flags.append("STRONG_USD_HEADWIND")     # strong USD hurting exporter
        elif political_fx_impact > fx_impact_thresh:
            flags.append("WEAK_USD_TAILWIND")       # weak USD helping exporter

    if political_govt_dependency is not None and political_govt_dependency > govt_dep_thresh:
        flags.append("HIGH_GOVT_DEPENDENCY")        # revenue materially tied to government

    # ── Economic flags (new) ──────────────────────────────────────────────────
    if credit_spread_z is not None and credit_spread_z < credit_spread_z_thresh:
        flags.append("CREDIT_STRESS")               # HY spreads widening → economic stress

    if inflation_z is not None and inflation_z > inflation_z_thresh:
        flags.append("INFLATION_PRESSURE")          # breakeven inflation elevated

    # ── Social flags ──────────────────────────────────────────────────────────
    if consumer_sentiment_z is not None:
        if consumer_sentiment_z > consumer_z_thresh:
            flags.append("CONSUMER_STRONG")         # XLY significantly outperforming XLP
        elif consumer_sentiment_z < -consumer_z_thresh:
            flags.append("CONSUMER_WEAK")           # XLY lagging XLP → defensive rotation

    if rd_intensity_gap is not None:
        if rd_intensity_gap > rd_gap_pos_thresh:
            flags.append("RD_LEADER")               # company outspending peers on R&D
        elif rd_intensity_gap < rd_gap_neg_thresh:
            flags.append("RD_LAGGARD")              # underinvesting in R&D vs peers

    # ── Technological flags ───────────────────────────────────────────────────
    if capex_intensity_gap is not None:
        if capex_intensity_gap > capex_gap_pos_thresh:
            flags.append("CAPEX_LEADER")            # above-peer capital investment
        elif capex_intensity_gap < capex_gap_neg_thresh:
            flags.append("CAPEX_LAGGARD")           # below-peer capital investment

    if tech_disruption_exposure is not None and tech_disruption_exposure > 0.55:
        flags.append("HIGH_DISRUPTION_RISK")        # sector structurally at risk of tech disruption

    # ── Environmental flags ───────────────────────────────────────────────────
    if environmental_score_raw is not None and environmental_score_raw < env_headwind_thresh:
        flags.append("ENV_BURDEN_HIGH")             # carbon intensity + regulatory burden elevated

    # ── Legal flags ───────────────────────────────────────────────────────────
    if legal_score_raw is not None and legal_score_raw < legal_headwind_thresh:
        flags.append("LEGAL_RISK_HIGH")             # heavy regulatory + litigation exposure

    return flags
