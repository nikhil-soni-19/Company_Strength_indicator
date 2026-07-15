from __future__ import annotations
import statistics


def compute_margin(row: dict, metric: str) -> float | None:
    if metric == "nim":
        nii, aea = row.get("net_interest_income"), row.get("avg_earning_assets")
        return (nii / aea) if (nii and aea and aea != 0) else None
    if metric == "gross_margin":
        gp, rev = row.get("gross_profit"), row.get("revenue")
        return (gp / rev) if (gp and rev) else None
    if metric == "operating_margin":
        oi, rev = row.get("operating_income"), row.get("revenue")
        return (oi / rev) if (oi and rev) else None
    if metric == "ebitda_margin":
        e, rev = row.get("ebitda"), row.get("revenue")
        return (e / rev) if (e and rev) else None
    if metric == "ffo_margin":
        ffo, rev = row.get("ffo"), row.get("revenue")
        return (ffo / rev) if (ffo and rev) else None
    return None


def ttm_revenue_growth(
    quarters_now: list[float],
    quarters_prior: list[float],
) -> float | None:
    """quarters_now = trailing 4 quarters revenue; quarters_prior = the 4 before that."""
    if len(quarters_now) < 4 or len(quarters_prior) < 4:
        return None
    return sum(quarters_now) / sum(quarters_prior) - 1


def peer_gaps(company_fund: dict, peer_funds: list[dict], margin_metric: str) -> dict:
    """
    company_fund / peer_funds: dicts with TTM-built fields.
    Excludes peers whose latest filing is >6 months stale (caller filters).
    """
    g_co = ttm_revenue_growth(company_fund["ttm_now"], company_fund["ttm_prior"])
    g_pe = [ttm_revenue_growth(p["ttm_now"], p["ttm_prior"]) for p in peer_funds]
    g_pe = [x for x in g_pe if x is not None]

    m_co = compute_margin(company_fund["latest"], margin_metric)
    m_pe = [compute_margin(p["latest"], margin_metric) for p in peer_funds]
    m_pe = [x for x in m_pe if x is not None]

    return {
        "rev_growth_company":     g_co,
        "rev_growth_peer_median": statistics.median(g_pe) if g_pe else None,
        "rev_growth_gap":         (g_co - statistics.median(g_pe)) if (g_co is not None and g_pe) else None,
        "margin_metric":          margin_metric,
        "margin_company":         m_co,
        "margin_peer_median":     statistics.median(m_pe) if m_pe else None,
        "margin_gap":             (m_co - statistics.median(m_pe)) if (m_co is not None and m_pe) else None,
    }
