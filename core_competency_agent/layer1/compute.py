"""
Compute all moat-relevant financial signals from raw 8Q data.
All ratios are expressed as decimals (0.30 = 30%).
"""
from __future__ import annotations

import statistics
from typing import Optional


_TAX_RATE = 0.21  # US statutory rate — used when company rate unavailable


def _safe_div(num: float, den: float) -> Optional[float]:
    if den and abs(den) > 1e-9:
        return num / den
    return None


def _series_ratio(num: list[float], den: list[float]) -> list[float]:
    return [_safe_div(n, d) or 0.0 for n, d in zip(num, den)]


def margin_series(numerator: list[float], revenue: list[float]) -> list[float]:
    return _series_ratio(numerator, revenue)


def coefficient_of_variation(series: list[float]) -> float:
    """std/|mean|. Low CV = stable margins = durability signal."""
    clean = [v for v in series if v is not None]
    if len(clean) < 2:
        return 0.0
    mu = statistics.mean(clean)
    if abs(mu) < 1e-9:
        return 0.0
    return statistics.stdev(clean) / abs(mu)


def compute_roic(
    op_income: list[float],
    total_debt: list[float],
    total_equity: list[float],
    tax_prov: list[float],
    revenue: list[float],
) -> Optional[float]:
    """TTM ROIC = NOPAT_TTM / Invested_Capital_avg."""
    n = len(op_income)
    if n < 4:
        return None

    ttm_op   = sum(op_income[-4:])
    ttm_rev  = sum(revenue[-4:])
    ttm_tax  = sum(tax_prov[-4:])

    if ttm_tax > 0 and ttm_op > 0:
        eff_rate = max(0.0, min(0.45, ttm_tax / ttm_op))
    else:
        eff_rate = _TAX_RATE

    nopat = ttm_op * (1 - eff_rate)

    # Average invested capital over last 2 periods
    ic_vals = [d + e for d, e in zip(total_debt[-4:], total_equity[-4:])]
    ic_vals = [v for v in ic_vals if v > 1e-6]
    if not ic_vals:
        return None
    avg_ic = statistics.mean(ic_vals)

    return _safe_div(nopat, avg_ic)


def compute_roe(
    net_income: list[float],
    total_equity: list[float],
) -> Optional[float]:
    """TTM ROE = NI_TTM / avg_equity."""
    if len(net_income) < 4:
        return None
    ttm_ni = sum(net_income[-4:])
    eq_vals = [v for v in total_equity[-4:] if v > 1e-6]
    if not eq_vals:
        return None
    return _safe_div(ttm_ni, statistics.mean(eq_vals))


def fcf_margin_series(
    ocf: list[float],
    capex: list[float],
    revenue: list[float],
) -> list[float]:
    fcf = [o - c for o, c in zip(ocf, capex)]
    return _series_ratio(fcf, revenue)


def peer_median_margin(peer_data: list[dict], margin_key: str) -> Optional[float]:
    """Compute median of a margin across all peers (latest quarter)."""
    vals = []
    for p in peer_data:
        rev = p.get("revenue", [])
        num = p.get(margin_key, [])
        if rev and num and rev[-1] and abs(rev[-1]) > 1e-6:
            vals.append(num[-1] / rev[-1])
    if not vals:
        return None
    return statistics.median(vals)


def peer_median_roic(peer_data: list[dict]) -> Optional[float]:
    vals = []
    for p in peer_data:
        roic = compute_roic(
            p.get("op_income", []),
            p.get("total_debt", []),
            p.get("total_equity", []),
            p.get("tax_prov", []),
            p.get("revenue", []),
        )
        if roic is not None:
            vals.append(roic)
    return statistics.median(vals) if vals else None


def peer_median_roe(peer_data: list[dict]) -> Optional[float]:
    vals = []
    for p in peer_data:
        roe = compute_roe(p.get("net_income", []), p.get("total_equity", []))
        if roe is not None:
            vals.append(roe)
    return statistics.median(vals) if vals else None


def peer_median_fcf_margin(peer_data: list[dict]) -> Optional[float]:
    """FCF margin of each peer (TTM if 4Q available, else latest Q)."""
    vals = []
    for p in peer_data:
        rev  = p.get("revenue", [])
        ocf  = p.get("ocf", [])
        capex = p.get("capex", [])
        if not (rev and ocf and capex):
            continue
        n = min(len(rev), len(ocf), len(capex), 4)
        ttm_fcf = sum(o - c for o, c in zip(ocf[-n:], capex[-n:]))
        ttm_rev = sum(rev[-n:])
        m = _safe_div(ttm_fcf, ttm_rev)
        if m is not None:
            vals.append(m)
    return statistics.median(vals) if vals else None


def compute_all(company: dict, peer_data: list[dict]) -> dict:
    """
    Run all Layer 1 computations.
    Returns a dict with every computed signal.
    """
    rev  = company["revenue"]
    gp   = company["gross_profit"]
    op   = company["op_income"]
    ni   = company["net_income"]
    tax  = company["tax_prov"]
    ocf  = company["ocf"]
    cap  = company["capex"]
    debt = company["total_debt"]
    eq   = company["total_equity"]

    gm_series  = margin_series(gp, rev)
    opm_series = margin_series(op, rev)
    fcf_series = fcf_margin_series(ocf, cap, rev)

    gm_cv  = coefficient_of_variation(gm_series)
    opm_cv = coefficient_of_variation(opm_series)

    roic = compute_roic(op, debt, eq, tax, rev)
    roe  = compute_roe(ni, eq)

    peer_gm_median  = peer_median_margin(peer_data, "gross_profit") or 0.0
    peer_opm_median = peer_median_margin(peer_data, "op_income")    or 0.0
    peer_roic       = peer_median_roic(peer_data)
    peer_roe        = peer_median_roe(peer_data)
    peer_fcf_margin = peer_median_fcf_margin(peer_data)

    gm_spread  = [g - peer_gm_median  for g in gm_series]
    opm_spread = [o - peer_opm_median for o in opm_series]

    avg_gm_spread  = statistics.mean(gm_spread)  if gm_spread  else 0.0
    avg_opm_spread = statistics.mean(opm_spread) if opm_spread else 0.0

    roic_spread = None
    if roic is not None and peer_roic is not None:
        roic_spread = roic - peer_roic

    avg_fcf_spread = None
    if fcf_series and peer_fcf_margin is not None:
        avg_fcf = statistics.mean(fcf_series)
        avg_fcf_spread = avg_fcf - peer_fcf_margin

    return {
        "gross_margin_series":     gm_series,
        "gross_margin_peer_median": peer_gm_median,
        "gross_margin_spread":     gm_spread,
        "avg_gross_margin_spread": avg_gm_spread,

        "op_margin_series":        opm_series,
        "op_margin_peer_median":   peer_opm_median,
        "op_margin_spread":        opm_spread,
        "avg_op_margin_spread":    avg_opm_spread,

        "roic_company":     roic,
        "roic_peer_median": peer_roic,
        "roic_spread":      roic_spread,

        "roe_company":     roe,
        "roe_peer_median": peer_roe,

        "fcf_margin_series":    fcf_series,
        "fcf_margin_peer_median": peer_fcf_margin,
        "avg_fcf_margin_spread":  avg_fcf_spread,

        "gross_margin_cv":    gm_cv,
        "op_margin_cv":       opm_cv,
    }
