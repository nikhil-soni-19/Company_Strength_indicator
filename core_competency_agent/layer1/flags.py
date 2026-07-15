"""
Emit boolean moat flags from computed Layer 1 signals.

Thresholds (all explicit — no magic numbers buried in logic):
  MARGIN_PREMIUM_SUSTAINED  — avg gross margin spread >= 5pp, no quarter < 0
  OP_MARGIN_PREMIUM         — avg op margin spread >= 3pp
  ROIC_ELITE                — ROIC > peer median by 5pp AND ROIC > 15%
  FCF_YIELD_STRONG          — avg FCF margin spread >= 3pp
  INSIDER_CONVICTION_HIGH   — insider ownership >= 5%
  MARGIN_VOLATILE           — gross margin CV > 0.10  (inverse durability)
  ROIC_BELOW_PEERS          — ROIC < peer median by 3pp  (negative)
"""
from __future__ import annotations

from typing import Optional

MARGIN_PREMIUM_THRESHOLD    = 0.05   # 5 pp gross margin lead, sustained
OP_MARGIN_PREMIUM_THRESHOLD = 0.03   # 3 pp op margin lead
ROIC_SPREAD_THRESHOLD       = 0.05   # 5 pp ROIC lead over peers
ROIC_ABSOLUTE_THRESHOLD     = 0.15   # 15% absolute ROIC (elite threshold)
FCF_SPREAD_THRESHOLD        = 0.03   # 3 pp FCF margin lead
INSIDER_PCT_THRESHOLD       = 0.05   # 5% insider ownership
MARGIN_CV_VOLATILE          = 0.10   # gross margin CV above 10% = volatile
ROIC_BELOW_THRESHOLD        = -0.03  # 3 pp ROIC below peers


def emit_flags(
    avg_gross_margin_spread: float,
    gross_margin_spread: list[float],
    avg_op_margin_spread: float,
    roic_spread: Optional[float],
    roic_company: Optional[float],
    avg_fcf_margin_spread: Optional[float],
    insider_pct: Optional[float],
    gross_margin_cv: float,
) -> list[str]:
    flags: list[str] = []

    # MARGIN_PREMIUM_SUSTAINED: above-peer gross margins every quarter
    if (avg_gross_margin_spread >= MARGIN_PREMIUM_THRESHOLD
            and all(s >= 0 for s in gross_margin_spread)):
        flags.append("MARGIN_PREMIUM_SUSTAINED")

    # OP_MARGIN_PREMIUM
    if avg_op_margin_spread >= OP_MARGIN_PREMIUM_THRESHOLD:
        flags.append("OP_MARGIN_PREMIUM")

    # ROIC_ELITE
    if (roic_spread is not None and roic_spread >= ROIC_SPREAD_THRESHOLD
            and roic_company is not None and roic_company >= ROIC_ABSOLUTE_THRESHOLD):
        flags.append("ROIC_ELITE")

    # FCF_YIELD_STRONG
    if avg_fcf_margin_spread is not None and avg_fcf_margin_spread >= FCF_SPREAD_THRESHOLD:
        flags.append("FCF_YIELD_STRONG")

    # INSIDER_CONVICTION_HIGH
    if insider_pct is not None and insider_pct >= INSIDER_PCT_THRESHOLD:
        flags.append("INSIDER_CONVICTION_HIGH")

    # MARGIN_VOLATILE (negative signal)
    if gross_margin_cv > MARGIN_CV_VOLATILE:
        flags.append("MARGIN_VOLATILE")

    # ROIC_BELOW_PEERS (negative signal)
    if roic_spread is not None and roic_spread <= ROIC_BELOW_THRESHOLD:
        flags.append("ROIC_BELOW_PEERS")

    return flags
