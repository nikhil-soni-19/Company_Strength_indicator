"""
Fetch 8 quarters of financial data from yfinance.
Returns a standardized dict regardless of which fields Yahoo provides.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional


def fetch_financials(ticker: str, n_quarters: int = 8) -> dict:
    """
    Fetch quarterly financials from yfinance.
    Returns a dict with lists of n_quarters values (oldest to newest).
    Raises ValueError if insufficient data is available.
    """
    t = yf.Ticker(ticker)

    income = t.quarterly_income_stmt
    cashflow = t.quarterly_cashflow
    balance = t.quarterly_balance_sheet

    if income is None or income.empty:
        raise ValueError(f"No income statement data found for ticker: {ticker}")

    # yfinance returns columns newest-first — reverse to oldest-first
    income = income.T.sort_index().tail(n_quarters)

    # Align cashflow and balance sheet to exactly the same periods as income.
    # yfinance can return different numbers of rows for each statement;
    # mismatched lengths cause numpy broadcast errors in Layer 1.
    income_index = income.index

    if cashflow is not None and not cashflow.empty:
        cashflow = cashflow.T.sort_index()
        # Keep only rows whose dates appear in the income index; fill missing with 0
        cashflow = cashflow.reindex(income_index, fill_value=0)
    else:
        cashflow = pd.DataFrame(0.0, index=income_index, columns=[])

    if balance is not None and not balance.empty:
        balance = balance.T.sort_index()
        balance = balance.reindex(income_index, fill_value=0)
    else:
        balance = pd.DataFrame(0.0, index=income_index, columns=[])

    n = len(income_index)

    def safe_get(df: pd.DataFrame, *possible_keys) -> list[float]:
        """Try multiple possible field names, return zeros if not found."""
        for key in possible_keys:
            if key in df.columns:
                return df[key].fillna(0).tolist()
        return [0.0] * n

    periods = [str(d)[:10] for d in income_index.tolist()]

    revenue      = safe_get(income, "Total Revenue", "Revenue")
    cogs         = safe_get(income, "Cost Of Revenue", "Cost of Revenue", "Cost Of Goods Sold")
    gross_profit = safe_get(income, "Gross Profit")
    opex         = safe_get(income, "Total Expenses", "Operating Expense", "Total Operating Expenses")
    op_income    = safe_get(income, "Operating Income", "EBIT")
    net_income   = safe_get(income, "Net Income", "Net Income Common Stockholders")
    sga          = safe_get(income, "Selling General And Administrative", "SGA Expense", "Selling General Administrative")
    rd           = safe_get(income, "Research And Development", "Research Development")

    ocf   = safe_get(cashflow, "Operating Cash Flow", "Cash From Operations")
    capex = safe_get(cashflow, "Capital Expenditure", "Purchases Of Property Plant And Equipment")
    capex = [abs(v) for v in capex]  # capex is negative in yfinance

    ar         = safe_get(balance, "Accounts Receivable", "Net Receivables")
    inventory  = safe_get(balance, "Inventory")
    ap         = safe_get(balance, "Accounts Payable", "Payables")
    total_debt = safe_get(balance, "Total Debt", "Long Term Debt")

    # Compute gross profit if not directly available
    if all(v == 0 for v in gross_profit):
        gross_profit = [r - c for r, c in zip(revenue, cogs)]

    # Compute opex if not directly available (COGS + SGA + R&D)
    if all(v == 0 for v in opex):
        opex = [c + s + r for c, s, r in zip(cogs, sga, rd)]

    return {
        "ticker": ticker,
        "periods": periods,
        "revenue": revenue,
        "cogs": cogs,
        "gross_profit": gross_profit,
        "opex": opex,
        "op_income": op_income,
        "net_income": net_income,
        "sga": sga,
        "rd": rd,
        "ocf": ocf,
        "capex": capex,
        "ar": ar,
        "inventory": inventory,
        "ap": ap,
        "total_debt": total_debt,
        "n_quarters": len(periods),
    }
