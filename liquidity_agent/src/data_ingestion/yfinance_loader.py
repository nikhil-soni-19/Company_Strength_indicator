"""Step 1 — Pull the last 90 trading days of OHLCV + structural data.

This is the only entry-point for live market data in the pipeline. Every
downstream module accepts a :class:`MarketData` and never re-fetches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from config import LOOKBACK_DAYS
from src.utils import ensure_trading_day_index


@dataclass
class MarketData:
    """Bundle of everything the scoring engine needs about a single ticker."""

    ticker: str
    ohlcv: pd.DataFrame
    float_shares: Optional[float]
    shares_outstanding: Optional[float]
    short_percent_float: Optional[float]
    top10_institutional_pct: Optional[float]
    # Buyback signal inputs
    quarterly_cashflow: Optional[pd.DataFrame] = field(default=None)
    market_cap: Optional[float] = field(default=None)
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def last_bar_date(self) -> Optional[pd.Timestamp]:
        if self.ohlcv.empty:
            return None
        return pd.Timestamp(self.ohlcv.index.max())

    @property
    def data_age_days(self) -> Optional[int]:
        last = self.last_bar_date
        if last is None:
            return None
        last_naive = last.tz_localize(None) if last.tzinfo else last
        return (pd.Timestamp(self.as_of).tz_localize(None) - last_naive).days


class LiveDataLoader:
    """Thin wrapper around ``yfinance`` exposing the exact data the agent needs.

    The PDF specifies a 90-day lookback. We request a slightly larger window
    (default ~130 calendar days) so that, after weekends/holidays are dropped,
    we still end up with at least ``LOOKBACK_DAYS`` trading rows.
    """

    def __init__(self, lookback_days: int = LOOKBACK_DAYS, calendar_pad: float = 1.45):
        self.lookback_days = lookback_days
        self._calendar_pad = calendar_pad

    def fetch(self, ticker: str) -> MarketData:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=int(self.lookback_days * self._calendar_pad))

        tk = yf.Ticker(ticker)
        ohlcv = tk.history(start=start, end=end, auto_adjust=False)
        ohlcv = ensure_trading_day_index(ohlcv)

        ohlcv = ohlcv.tail(self.lookback_days)

        info = self._safe_get_info(tk)
        quarterly_cashflow = self._safe_get_quarterly_cashflow(tk)

        return MarketData(
            ticker=ticker.upper(),
            ohlcv=ohlcv,
            float_shares=_coerce_float(info.get("floatShares")),
            shares_outstanding=_coerce_float(info.get("sharesOutstanding")),
            short_percent_float=_coerce_float(info.get("shortPercentOfFloat")),
            top10_institutional_pct=self._calc_top10_institutional_pct(tk, info),
            quarterly_cashflow=quarterly_cashflow,
            market_cap=_coerce_float(info.get("marketCap")),
        )

    @staticmethod
    def _safe_get_info(tk: "yf.Ticker") -> dict:
        try:
            info = tk.get_info() if hasattr(tk, "get_info") else tk.info  # type: ignore[attr-defined]
        except Exception:
            info = {}
        return info or {}

    @staticmethod
    def _calc_top10_institutional_pct(tk: "yf.Ticker", info: dict) -> Optional[float]:
        """Sum the top-10 institutional holders' pct_held to get a true top-10 figure.

        Falls back to ``heldPercentInstitutions`` (total institutional %) only if
        the holders table cannot be fetched.
        """
        try:
            holders = tk.institutional_holders
            if holders is not None and not holders.empty:
                # yfinance returns a '% Out' column (values 0-1 or 0-100; normalise both)
                pct_col = None
                for col in holders.columns:
                    if "%" in str(col) or "pct" in str(col).lower() or "out" in str(col).lower():
                        pct_col = col
                        break
                if pct_col is not None:
                    top10 = holders.head(10)[pct_col].dropna()
                    total = float(top10.sum())
                    # normalise: yfinance sometimes returns 0-100, sometimes 0-1
                    if total > 1.5:
                        total = total / 100.0
                    return total if 0.0 < total <= 1.0 else None
        except Exception:
            pass
        # fallback: total institutional %
        return _coerce_float(info.get("heldPercentInstitutions"))

    @staticmethod
    def _safe_get_quarterly_cashflow(tk: "yf.Ticker") -> Optional[pd.DataFrame]:
        """Fetch the quarterly cashflow statement, returning None on any failure."""
        try:
            cf = tk.quarterly_cashflow
            if cf is None or (hasattr(cf, "empty") and cf.empty):
                return None
            return cf
        except Exception:
            return None


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f
