import pandas as pd


def commodity_impact(
    sector: str,
    sensitivity_map: dict,
    commodity_prices: dict[str, pd.Series],
) -> dict:
    """
    sensitivity_map[sector] = {commodity_ticker: +1 / -1}
    commodity_prices[ticker] = price series over window.
    Impact_c = sign(s,c) * sign(return_c).
    Aggregate = mean of Impact_c.
    """
    linked = sensitivity_map.get(sector, {})
    if not linked:
        return {"commodity_impact_raw": 0.0, "commodity_tag": "NOT_APPLICABLE"}
    impacts = []
    detail = {}
    for c, sign in linked.items():
        if c not in commodity_prices:
            continue
        s = commodity_prices[c]
        ret = s.iloc[-1] / s.iloc[0] - 1
        imp = sign * (1 if ret > 0 else -1 if ret < 0 else 0)
        impacts.append(imp)
        detail[c] = {"return": float(ret), "sensitivity": sign, "impact": imp}
    if not impacts:
        return {"commodity_impact_raw": 0.0, "commodity_tag": "NO_DATA"}
    agg = sum(impacts) / len(impacts)
    tag = (
        "COMMODITY_TAILWIND" if agg > 0
        else "COMMODITY_HEADWIND" if agg < 0
        else "COMMODITY_NEUTRAL"
    )
    return {"commodity_impact_raw": agg, "commodity_tag": tag, "detail": detail}
