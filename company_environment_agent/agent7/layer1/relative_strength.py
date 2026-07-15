import pandas as pd


def relative_strength(p_company: pd.Series, p_sector: pd.Series) -> float:
    """
    RS = (P_company_t / P_company_0) / (P_sector_t / P_sector_0)

    RS > 1.0 → company outperforming its sector ETF  → LEADING
    RS < 1.0 → company underperforming its sector ETF → LAGGING
    """
    company_ratio = p_company.iloc[-1] / p_company.iloc[0]
    sector_ratio  = p_sector.iloc[-1]  / p_sector.iloc[0]
    return float(company_ratio / sector_ratio)
