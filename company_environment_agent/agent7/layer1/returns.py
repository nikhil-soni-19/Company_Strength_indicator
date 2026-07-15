import pandas as pd


def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def cumulative_return(prices: pd.Series) -> float:
    return prices.iloc[-1] / prices.iloc[0] - 1
