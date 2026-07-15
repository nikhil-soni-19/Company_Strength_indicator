"""
Compute a 0-10 Layer 1 score from normalized signals.
Weights reflect the importance of each signal cluster.
"""
import numpy as np


WEIGHTS = {
    "operating_leverage": 0.25,
    "margin_trajectory":  0.25,
    "revenue_growth":     0.20,
    "cash_conversion":    0.20,
    "working_capital":    0.10,
}

# Min/max bounds for normalization (calibrated for US large-cap equities)
BOUNDS = {
    "ol_consistency":      (0.0, 1.0),
    "ol_slope":            (-3.0, 3.0),
    "op_margin_slope":     (-0.01, 0.01),
    "gross_margin_slope":  (-0.01, 0.01),
    "rev_yoy_latest":      (-20.0, 40.0),
    "rev_accel_avg":       (-5.0, 5.0),
    "fcf_ni_ratio_latest": (-0.5, 1.5),
    "ccc_improvement":     (-20.0, 20.0),
}


def compute_score(computed: dict) -> float:
    """Return a 0-10 composite Layer 1 score."""

    def norm(value: float, key: str) -> float:
        lo, hi = BOUNDS[key]
        return float(np.clip((value - lo) / (hi - lo + 1e-9), 0.0, 1.0))

    rev_yoy = computed["rev_yoy_pct"]
    rev_accel = computed["rev_accel"]
    ccc_delta = computed.get("ccc_delta") or 0.0

    s_ol = (
        norm(computed["ol_consistency"], "ol_consistency") * 0.6 +
        norm(computed["ol_slope"], "ol_slope") * 0.4
    )
    s_margin = (
        norm(computed["op_margin_slope"], "op_margin_slope") * 0.6 +
        norm(computed["gross_margin_slope"], "gross_margin_slope") * 0.4
    )
    s_rev = (
        norm(rev_yoy[-1] if rev_yoy else 0, "rev_yoy_latest") * 0.5 +
        norm(np.mean(rev_accel) if rev_accel else 0, "rev_accel_avg") * 0.5
    )
    s_cash = norm(computed["fcf_ni_ratio_latest"], "fcf_ni_ratio_latest")
    s_wc   = norm(-ccc_delta, "ccc_improvement")  # negative delta is good

    raw = (
        WEIGHTS["operating_leverage"] * s_ol +
        WEIGHTS["margin_trajectory"]  * s_margin +
        WEIGHTS["revenue_growth"]     * s_rev +
        WEIGHTS["cash_conversion"]    * s_cash +
        WEIGHTS["working_capital"]    * s_wc
    )

    return round(raw * 10, 2)
