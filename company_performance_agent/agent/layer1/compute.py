"""
Layer 1 computations. All math, no LLM.
Input: raw financials dict from fetcher.py
Output: computed metrics dict
"""
import numpy as np
from scipy.stats import linregress


def compute_all(data: dict) -> dict:
    """Run all Layer 1 calculations on raw financial data."""
    n = data["n_quarters"]
    rev = np.array(data["revenue"])
    opex = np.array(data["opex"])
    gross = np.array(data["gross_profit"])
    op_inc = np.array(data["op_income"])
    net_inc = np.array(data["net_income"])
    ocf = np.array(data["ocf"])
    capex = np.array(data["capex"])
    ar = np.array(data["ar"])
    inventory = np.array(data["inventory"])
    ap = np.array(data["ap"])

    # ── FCF ──────────────────────────────────────────────────────────────
    fcf = ocf - capex

    # ── YoY growth (need at least 5Q: 4 for YoY pairs) ───────────────────
    rev_yoy = _yoy_pct(rev, n)
    opex_yoy = _yoy_pct(opex, n)

    # ── Operating leverage delta ──────────────────────────────────────────
    ol_delta = [r - o for r, o in zip(rev_yoy, opex_yoy)]

    # ── Revenue acceleration (delta of delta) ────────────────────────────
    rev_accel = [rev_yoy[i] - rev_yoy[i-1] for i in range(1, len(rev_yoy))]

    # ── Margins ───────────────────────────────────────────────────────────
    gross_margin = _safe_divide(gross, rev)
    op_margin = _safe_divide(op_inc, rev)
    net_margin = _safe_divide(net_inc, rev)

    # ── Slopes via linear regression over all n quarters ─────────────────
    x = np.arange(n)
    rev_slope = _slope(x, rev / 1e6)          # normalized to millions
    op_margin_slope = _slope(x, op_margin)
    gross_margin_slope = _slope(x, gross_margin)
    ol_slope = _slope(np.arange(len(ol_delta)), ol_delta) if len(ol_delta) > 1 else 0.0

    # ── Cash conversion quality ───────────────────────────────────────────
    fcf_ni_ratio = _safe_divide(fcf, net_inc)

    # ── Working capital (Days) — only if data available ───────────────────
    days = 91  # approximate days per quarter
    dso = (ar / np.maximum(rev, 1)) * days
    dio = (inventory / np.maximum(np.array(data["cogs"]), 1)) * days
    dpo = (ap / np.maximum(np.array(data["cogs"]), 1)) * days
    ccc = dso + dio - dpo
    ccc_delta = float(ccc[-1] - ccc[-5]) if n >= 5 else None

    # ── OL consistency (fraction of quarters with positive OL) ───────────
    ol_all = _yoy_pct(rev, n, full=True)
    opex_yoy_all = _yoy_pct(opex, n, full=True)
    ol_all_delta = [r - o for r, o in zip(ol_all, opex_yoy_all)]
    ol_consistency = sum(1 for x in ol_all_delta if x > 0) / max(len(ol_all_delta), 1)

    return {
        # Series
        "rev_yoy_pct": rev_yoy,
        "opex_yoy_pct": opex_yoy,
        "ol_delta": ol_delta,
        "rev_accel": rev_accel,
        "gross_margin_series": gross_margin.tolist(),
        "op_margin_series": op_margin.tolist(),
        "net_margin_series": net_margin.tolist(),
        "fcf": fcf.tolist(),
        "fcf_ni_ratio": fcf_ni_ratio.tolist(),
        "dso": dso.tolist(),
        "ccc": ccc.tolist(),

        # Latest values
        "gross_margin_latest": float(gross_margin[-1]),
        "op_margin_latest": float(op_margin[-1]),
        "net_margin_latest": float(net_margin[-1]),
        "fcf_ni_ratio_latest": float(fcf_ni_ratio[-1]),

        # Slopes
        "rev_slope": float(rev_slope),
        "op_margin_slope": float(op_margin_slope),
        "gross_margin_slope": float(gross_margin_slope),
        "ol_slope": float(ol_slope),

        # Summary
        "ol_consistency": float(ol_consistency),
        "ccc_delta": ccc_delta,
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _yoy_pct(arr: np.ndarray, n: int, full: bool = False) -> list[float]:
    """Compute YoY % change. Returns last 4 quarters unless full=True."""
    result = []
    for i in range(4, n):
        prev = arr[i - 4]
        curr = arr[i]
        if abs(prev) > 0:
            result.append(((curr - prev) / abs(prev)) * 100)
        else:
            result.append(0.0)
    return result if full else result[-4:]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return slope of linear regression."""
    if len(x) < 2 or np.all(y == 0):
        return 0.0
    slope, _, _, _, _ = linregress(x, y)
    return float(slope)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element-wise division, returns 0 for zero denominators."""
    return np.where(np.abs(den) > 0, num / den, 0.0)
