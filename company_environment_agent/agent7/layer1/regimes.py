import numpy as np
import pandas as pd


def vix_regime(
    vix_series_3y: pd.Series,
    thresh_high: float = 1.0,
    thresh_low: float = -1.0,
) -> dict:
    """
    Z-score of current VIX against 3-year μ/σ baseline.
    Thresholds: ±1.0 (symmetric).
    """
    mu = vix_series_3y.mean()
    sd = vix_series_3y.std()
    z  = (vix_series_3y.iloc[-1] - mu) / sd
    if   z >  thresh_high: tag = "HIGH_VOLATILITY"
    elif z <  thresh_low:  tag = "LOW_VOLATILITY"
    else:                  tag = "NORMAL_VOLATILITY"
    return {
        "vix_current": float(vix_series_3y.iloc[-1]),
        "vix_zscore":  float(z),
        "vol_regime":  tag,
    }


def _rolling_ols_slope(series: pd.Series, window: int = 21) -> pd.Series:
    """
    For each day t, fit OLS(y=series[t-window:t], x=range(window))
    and return the slope coefficient.
    This is the 21-day local linear trend in yield level.
    """
    slopes = np.full(len(series), np.nan)
    x = np.arange(window, dtype=float)
    X = np.column_stack([np.ones(window), x])
    XtX_inv = np.linalg.inv(X.T @ X)

    vals = series.values
    for i in range(window - 1, len(vals)):
        y = vals[i - window + 1: i + 1]
        if np.any(np.isnan(y)):
            continue
        b = XtX_inv @ X.T @ y
        slopes[i] = b[1]   # slope coefficient

    return pd.Series(slopes, index=series.index)


def rate_regime(
    tnx_series_3y: pd.Series,
    sigma_mult: float = 1.0,
) -> dict:
    """
    Rate regime via 21-day rolling OLS slope on TNX yield levels.

    Slope_t = OLS slope of TNX over the trailing 21 trading days.
    Build a 3-year distribution of this slope series.
    Z_slope = (slope_current - μ_slope) / σ_slope

    RISING_RATE  if Z_slope > +sigma_mult
    FALLING_RATE if Z_slope < -sigma_mult
    STABLE_RATE  otherwise
    """
    slope_series = _rolling_ols_slope(tnx_series_3y, window=21).dropna()
    if len(slope_series) < 30:
        return {
            "rate_slope_3m":    None,
            "rate_slope_mu":    None,
            "rate_slope_sigma": None,
            "rate_regime":      "INSUFFICIENT_DATA",
        }

    mu      = float(slope_series.mean())
    sd      = float(slope_series.std())
    current = float(slope_series.iloc[-1])
    z       = (current - mu) / sd if sd > 0 else 0.0

    if   z >  sigma_mult: tag = "RISING_RATE"
    elif z < -sigma_mult: tag = "FALLING_RATE"
    else:                 tag = "STABLE_RATE"

    return {
        "rate_slope_3m":    current,
        "rate_slope_mu":    mu,
        "rate_slope_sigma": sd,
        "rate_regime":      tag,
    }


def market_trend(p_gspc: pd.Series) -> str:
    if len(p_gspc) < 200:
        return "INSUFFICIENT_DATA"
    ma50  = p_gspc.rolling(50).mean().iloc[-1]
    ma200 = p_gspc.rolling(200).mean().iloc[-1]
    last  = p_gspc.iloc[-1]
    if last > ma200 and ma50 > ma200: return "BULL"
    if last < ma200 and ma50 < ma200: return "BEAR"
    return "TRANSITION"
