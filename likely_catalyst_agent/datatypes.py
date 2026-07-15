"""
Plain Python dataclasses replacing the SQLAlchemy ORM models.
Used as data containers throughout the pipeline — no DB dependency.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from enums import DriftLabel, FilingType


@dataclass
class EarningsEvent:
    """Earnings announcement data, populated from yfinance or Neon."""
    ticker: str
    announcement_date: datetime
    reported_eps: Optional[float] = None
    estimated_eps: Optional[float] = None
    eps_surprise: Optional[float] = None
    eps_surprise_pct: Optional[float] = None
    revenue_surprise_pct: Optional[float] = None
    earnings_beat: Optional[bool] = None
    return_3day: Optional[float] = None
    return_20day: Optional[float] = None
    return_60day: Optional[float] = None
    abnormal_return_60day: Optional[float] = None
    drift_label: Optional[DriftLabel] = None
    filing_lag_days: Optional[int] = None


@dataclass
class SECFiling:
    """SEC filing metadata and extracted MD&A text."""
    ticker: str
    mda_text: Optional[str] = None
    accession_number: Optional[str] = None
    filing_date: Optional[datetime] = None
    filing_type: FilingType = FilingType.FORM_10Q
    cik: str = ""
    processed: bool = False


@dataclass
class UpcomingEvent:
    """An upcoming earnings announcement from the calendar."""
    ticker: str
    expected_date: datetime
    cik: Optional[str] = None
    fiscal_quarter: Optional[str] = None
    analyst_eps_estimate: Optional[float] = None
    analyst_revenue_estimate: Optional[float] = None
    source: str = "yfinance"
