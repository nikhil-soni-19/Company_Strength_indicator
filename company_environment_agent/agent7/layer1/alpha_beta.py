import pandas as pd
import statsmodels.api as sm


def alpha_beta(
    r_company: pd.Series,
    r_sector: pd.Series,
    rf_annual: float,
) -> dict:
    """
    OLS on daily EXCESS returns:
        (R_company - Rf) = α + β · (R_sector - Rf) + ε

    β  = Cov(excess_company, excess_sector) / Var(excess_sector)
    α_daily = mean(excess_company) - β · mean(excess_sector)
    α_annualised = (1 + α_daily)^252 - 1   [compound, not linear ×252]

    rf_annual: ^TNX yield as decimal (e.g. 0.042 for 4.2%)
    """
    rf_daily = rf_annual / 252.0

    df = pd.concat([r_company, r_sector], axis=1).dropna()
    if len(df) < 40:
        return {"alpha_annualised": None, "beta": None, "n_obs": len(df)}

    exc_company = df.iloc[:, 0] - rf_daily
    exc_sector  = df.iloc[:, 1] - rf_daily

    X = sm.add_constant(exc_sector.values)
    y = exc_company.values
    model = sm.OLS(y, X).fit()

    beta        = float(model.params[1])
    alpha_daily = float(exc_company.mean()) - beta * float(exc_sector.mean())

    # Compound annualisation — (1 + α_daily)^252 - 1
    alpha_annual = float((1 + alpha_daily) ** 252 - 1)

    return {
        "alpha_annualised": alpha_annual,
        "beta":             beta,
        "n_obs":            len(df),
    }
