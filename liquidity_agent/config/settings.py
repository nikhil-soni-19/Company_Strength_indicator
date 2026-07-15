"""
Central configuration for the Liquidity Agent.

Thresholds are calibrated for institutional-grade stocks (Russell 1000 universe).
DTL has been removed; 4 new structural dimensions have been added.

8 scoring dimensions, max raw score = 24.
Editing thresholds here changes engine behaviour without touching application code.
"""

from __future__ import annotations

LOOKBACK_DAYS: int = 90
SHORT_WINDOW: int = 30
LONG_WINDOW: int = 90
AMIHUD_WINDOW: int = 30

OUTLIER_CLIP_PERCENTILE: float = 0.99

PARTICIPATION_RATE_DEFAULT: float = 0.10
POSITION_PERCENTAGES: tuple[float, ...] = (0.01, 0.05)


SCORING_THRESHOLDS: dict[str, list[tuple[float | None, float | None, int]]] = {
    # (lower_inclusive, upper_exclusive, points)
    # A bound of None means open-ended on that side.
    # Higher score = higher liquidity risk.

    # ── Market activity ───────────────────────────────────────────────────────
    # Calibrated for institutional universe: $500M+ ADV = truly unrestricted.
    "adv_dollar_30d": [
        (500_000_000, None, 0),          # mega/large-cap: AAPL ~$13.87B
        (50_000_000, 500_000_000, 1),    # large/mid-cap
        (5_000_000, 50_000_000, 2),      # small/mid-cap
        (None, 5_000_000, 3),            # micro/nano-cap
    ],
    # Amihud scaled per $M traded. Large-caps typically <0.0001.
    "amihud_30d": [
        (None, 0.0001, 0),
        (0.0001, 0.005, 1),
        (0.005, 0.05, 2),
        (0.05, None, 3),
    ],
    # CV of daily volume over 30d. Bands widened vs. prior calibration.
    "volume_cv_30d": [
        (None, 0.40, 0),
        (0.40, 0.70, 1),
        (0.70, 1.20, 2),
        (1.20, None, 3),
    ],

    # ── Float & structural ────────────────────────────────────────────────────
    # Thresholds raised 10× vs. prior; 500M+ shares covers large-caps properly.
    "free_float_shares": [
        (500_000_000, None, 0),          # AAPL ~15.11B
        (100_000_000, 500_000_000, 1),
        (20_000_000, 100_000_000, 2),
        (None, 20_000_000, 3),
    ],
    # Float as fraction of shares outstanding (higher = better = fewer points).
    "float_pct_outstanding": [
        (0.90, None, 0),                 # AAPL 99.83%
        (0.75, 0.90, 1),
        (0.50, 0.75, 2),
        (None, 0.50, 3),
    ],
    # Short interest as fraction of float (e.g. 0.05 = 5%).
    "short_percent_float": [
        (None, 0.05, 0),                 # AAPL ~0.95%
        (0.05, 0.15, 1),
        (0.15, 0.25, 2),
        (0.25, None, 3),
    ],
    # Top-10 institutional concentration as fraction (e.g. 0.314 = 31.4%).
    "top10_institutional_pct": [
        (None, 0.20, 0),
        (0.20, 0.35, 1),                 # AAPL ~31.4% → 1 pt
        (0.35, 0.50, 2),
        (0.50, None, 3),
    ],

    # ── Buyback ───────────────────────────────────────────────────────────────
    # BIR = quarterly buyback spend / (ADV$30 × 63 trading days).
    "buyback_bir": [
        (None, 0.05, 0),                 # AAPL ~1%
        (0.05, 0.15, 1),
        (0.15, 0.30, 2),
        (0.30, None, 3),
    ],
}

TIER_BOUNDARIES: list[tuple[int, int, int, str]] = [
    # 8 dimensions × 3 pts max = 24 total. Boundaries scaled proportionally.
    # (min_score_inclusive, max_score_inclusive, tier_number, label)
    (0, 6,  1, "Unrestricted"),
    (7, 10, 2, "Position Sizing Caps"),
    (11, 18, 3, "Algorithmic Execution Only"),
    (19, 10_000, 4, "Blacklist"),
]


MIRAGE_OVERRIDE: dict[str, float | int] = {
    "max_float_shares": 10_000_000,
    "min_short_pct": 0.25,
    "tier_downgrade": 2,
}


CONFIDENCE_STALENESS_DAYS: int = 5
MIN_TRADING_DAYS_FOR_VALID_SCORE: int = 60
