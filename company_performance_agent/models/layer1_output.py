from pydantic import BaseModel
from typing import Optional

class Layer1Output(BaseModel):
    ticker: str
    period_latest: str                  # e.g. "Q4-2024"
    periods: list[str]                  # all 8 quarter labels

    # Raw series (8 values each, oldest to newest)
    revenue: list[float]
    opex: list[float]
    gross_profit: list[float]
    operating_income: list[float]
    net_income: list[float]
    ocf: list[float]
    capex: list[float]

    # Computed — YoY for last 4 quarters
    rev_yoy_pct: list[float]            # 4 values
    opex_yoy_pct: list[float]
    ol_delta: list[float]               # revenue YoY - opex YoY

    # Slopes (linear regression over 8Q)
    rev_slope: float
    op_margin_slope: float
    gross_margin_slope: float
    ol_slope: float

    # Margins — latest quarter
    gross_margin: float
    op_margin: float
    net_margin: float

    # Cash quality
    fcf: list[float]                    # 8 values
    fcf_ni_ratio: list[float]           # 8 values
    ccc_delta: Optional[float] = None   # cash conversion cycle change

    # Acceleration
    rev_accel: list[float]              # 3 values (Q-on-Q change in YoY)
    ol_consistency: float               # 0-1: fraction of 8Q with positive OL

    # Score and flags
    score: float                        # 0-10
    flags: list[str]                    # e.g. ["OP_LEVERAGE_DETERIORATING", ...]
