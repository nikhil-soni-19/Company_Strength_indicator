from pydantic import BaseModel
from typing import Optional


class Layer1MoatOutput(BaseModel):
    ticker: str
    peers: list[str]
    periods: list[str]                      # 8Q oldest→newest

    # Gross margin
    gross_margin_series: list[float]        # company, per quarter
    gross_margin_peer_median: float         # peer median (latest quarter)
    gross_margin_spread: list[float]        # company - peer_median, per quarter
    avg_gross_margin_spread: float

    # Operating margin
    op_margin_series: list[float]
    op_margin_peer_median: float
    op_margin_spread: list[float]
    avg_op_margin_spread: float

    # ROIC (TTM, annualised)
    roic_company: Optional[float]
    roic_peer_median: Optional[float]
    roic_spread: Optional[float]

    # ROE (TTM)
    roe_company: Optional[float]
    roe_peer_median: Optional[float]

    # FCF margin
    fcf_margin_series: list[float]          # per quarter
    fcf_margin_peer_median: Optional[float]
    avg_fcf_margin_spread: Optional[float]

    # Margin stability
    gross_margin_cv: float                  # std/mean — lower = more durable
    op_margin_cv: float

    # Insider / institutional ownership
    insider_ownership_pct: Optional[float]
    institutional_concentration_top10: Optional[float]

    # Leadership change detection
    leadership_change_detected: bool = False
    leadership_change_description: Optional[str] = None

    # Layer 1 score and flags
    score: float                            # 0-10
    flags: list[str]
