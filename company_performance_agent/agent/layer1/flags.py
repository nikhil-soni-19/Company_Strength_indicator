"""
Emit boolean flags from computed metrics.
All thresholds are calibrated for US public equities.
Adjust thresholds per your universe if needed.
"""


def emit_flags(computed: dict) -> list[str]:
    """Return list of flag strings based on computed metrics."""
    flags = []
    ol = computed["ol_delta"]
    ol_slope = computed["ol_slope"]
    op_slope = computed["op_margin_slope"]
    gross_slope = computed["gross_margin_slope"]
    fcf_ni = computed["fcf_ni_ratio"]
    rev_yoy = computed["rev_yoy_pct"]
    rev_accel = computed["rev_accel"]
    ol_consistency = computed["ol_consistency"]
    ccc_delta = computed["ccc_delta"]

    # ── Operating Leverage ─────────────────────────────────────────────
    if ol_consistency >= 0.75 and ol[-1] > 0:
        flags.append("OP_LEVERAGE_POSITIVE")

    if ol_slope < -0.005 and (ol[-1] < ol[-2] if len(ol) >= 2 else False):
        flags.append("OP_LEVERAGE_DETERIORATING")

    # ── Margins ────────────────────────────────────────────────────────
    if op_slope < -0.003:
        flags.append("MARGIN_COMPRESSING")

    if op_slope > 0.003 and gross_slope >= 0:
        flags.append("MARGIN_EXPANDING")

    if gross_slope < -0.004:
        flags.append("GROSS_MARGIN_PRESSURE")

    if gross_slope >= 0 and op_slope < -0.003:
        flags.append("OPEX_OVERHEAD_BLOAT")

    # ── Revenue Growth ─────────────────────────────────────────────────
    decel_count = sum(1 for a in rev_accel if a < -2.0)
    if decel_count >= 3:
        flags.append("REV_DECELERATING")

    accel_count = sum(1 for a in rev_accel if a > 1.5)
    if accel_count >= 3:
        flags.append("REV_ACCELERATING")

    # ── Cash Quality ───────────────────────────────────────────────────
    recent_fcf_ni = fcf_ni[-4:] if len(fcf_ni) >= 4 else fcf_ni
    weak_count = sum(1 for r in recent_fcf_ni if r < 0.8)
    if weak_count >= 3:
        flags.append("FCF_WEAK")

    fcf_values = computed.get("fcf", [])
    if len(fcf_values) >= 3 and all(v < 0 for v in fcf_values[-3:]):
        flags.append("FCF_NEGATIVE")

    # ── Working Capital ────────────────────────────────────────────────
    if ccc_delta is not None and ccc_delta > 5:
        flags.append("DSO_RISING")

    return flags
