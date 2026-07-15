"""
Neon DB writer — persists the agent's output signals to live.catalyst_snapshot.

This is the WRITE side of the Neon connection.  The agent runs its full
prediction pipeline locally, then calls write_catalyst_snapshot() to push
the result into the shared Neon DB so other agents (sentiment, fundamentals,
orchestrator) can read the catalyst signal via the cross-agent spine.

Schema used (from Schema Design.md / Spine.md):
    live.catalyst_snapshot  keyed by (canonical_ticker, as_of_date)

The table is created here if it doesn't exist (safe to run repeatedly — uses
CREATE TABLE IF NOT EXISTS).

Usage:
    from decision_engine import CatalystSignal
    from neon_writer import write_catalyst_snapshot

    signal: CatalystSignal = engine.build_signal(...)
    await write_catalyst_snapshot(signal)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Dict, Any
import json

from sqlalchemy import text

from neon_connection import get_neon_session
from logger import get_logger

logger = get_logger(__name__)


# ── DDL — run once ───────────────────────────────────────────────────────────

_LIVE_SCHEMA_DDL = """
CREATE SCHEMA IF NOT EXISTS live;

CREATE TABLE IF NOT EXISTS live.catalyst_snapshot (
    -- Spine keys (required by Spine.md for cross-agent joins)
    canonical_ticker    TEXT    NOT NULL,
    as_of_date          DATE    NOT NULL,

    -- Signal probabilities
    bullish_probability NUMERIC(6,4),
    bearish_probability NUMERIC(6,4),
    no_drift_probability NUMERIC(6,4),
    confidence          NUMERIC(6,4),

    -- Decision
    decision            TEXT,               -- BUY / HOLD / SELL
    expected_drift_window TEXT,             -- "20-40 days"

    -- Catalyst analysis
    catalyst_type       TEXT,
    bullish_catalysts   JSONB,              -- list of strings
    bearish_catalysts   JSONB,
    risk_signals        JSONB,
    sentiment_score     NUMERIC(6,4),
    narrative_summary   TEXT,

    -- Key feature snapshot (for audit / debugging)
    eps_surprise_pct    NUMERIC(8,4),
    return_3day         NUMERIC(8,4),
    pre_volatility_20d  NUMERIC(8,4),

    -- Metadata
    model_version       TEXT,
    generated_at        TIMESTAMPTZ DEFAULT now(),

    PRIMARY KEY (canonical_ticker, as_of_date)
);

CREATE INDEX IF NOT EXISTS ix_catalyst_ticker
    ON live.catalyst_snapshot (canonical_ticker);
CREATE INDEX IF NOT EXISTS ix_catalyst_date
    ON live.catalyst_snapshot (as_of_date);
"""

_UPCOMING_DDL = """
CREATE TABLE IF NOT EXISTS live.upcoming_prediction (
    canonical_ticker    TEXT    NOT NULL,
    event_date          DATE    NOT NULL,       -- expected earnings announcement date
    as_of_date          DATE    NOT NULL,       -- date prediction was made

    -- Inputs
    analyst_eps_estimate    NUMERIC(10,4),
    hist_avg_eps_surprise   NUMERIC(8,4),       -- trailing 4-quarter avg surprise %
    days_until_event        INT,

    -- Model output
    predicted_return        NUMERIC(8,4),       -- point estimate (e.g. 0.058 = +5.8%)
    return_low              NUMERIC(8,4),       -- lower bound of range
    return_high             NUMERIC(8,4),       -- upper bound of range
    bullish_probability     NUMERIC(6,4),
    bearish_probability     NUMERIC(6,4),
    confidence              NUMERIC(6,4),
    decision                TEXT,               -- BUY / HOLD / SELL
    drift_window            TEXT,
    catalyst_type           TEXT,

    -- Outcome (filled in retrospectively by outcome_tracker.py)
    actual_return_20d       NUMERIC(8,4),
    actual_return_60d       NUMERIC(8,4),
    prediction_error        NUMERIC(8,4),
    outcome_recorded_at     TIMESTAMPTZ,

    model_version           TEXT,
    created_at              TIMESTAMPTZ DEFAULT now(),

    PRIMARY KEY (canonical_ticker, event_date, as_of_date)
);
"""


async def ensure_live_tables() -> None:
    """Create live.catalyst_snapshot and live.upcoming_prediction if they don't exist."""
    async with get_neon_session() as session:
        await session.execute(text(_LIVE_SCHEMA_DDL))
        await session.execute(text(_UPCOMING_DDL))
    logger.info("live.* tables verified / created in Neon.")


# ── Writers ──────────────────────────────────────────────────────────────────

async def write_catalyst_snapshot(
    signal,                         # CatalystSignal dataclass from decision_engine.py
    as_of_date: Optional[date] = None,
    canonical_ticker: Optional[str] = None,
) -> None:
    """
    Upsert a CatalystSignal into live.catalyst_snapshot.

    The ticker is resolved to canonical_ticker via the spine if possible.
    If live tables don't exist yet, they are created automatically.

    Args:
        signal:           CatalystSignal object from decision_engine.build_signal()
        as_of_date:       Override for the as_of_date (defaults to today UTC)
        canonical_ticker: Override ticker (defaults to signal.ticker)
    """
    await ensure_live_tables()

    from neon_reader import resolve_canonical_ticker

    ticker = canonical_ticker or signal.ticker
    canonical = await resolve_canonical_ticker(ticker)
    snapshot_date = as_of_date or date.today()

    features: Dict[str, Any] = signal.features_used or {}

    async with get_neon_session() as session:
        await session.execute(
            text("""
                INSERT INTO live.catalyst_snapshot (
                    canonical_ticker, as_of_date,
                    bullish_probability, bearish_probability, no_drift_probability,
                    confidence, decision, expected_drift_window,
                    catalyst_type, bullish_catalysts, bearish_catalysts,
                    risk_signals, sentiment_score, narrative_summary,
                    eps_surprise_pct, return_3day, pre_volatility_20d,
                    model_version, generated_at
                ) VALUES (
                    :ticker, :aod,
                    :bull, :bear, :neutral,
                    :conf, :decision, :window,
                    :ctype, :bcats::jsonb, :bcats_neg::jsonb,
                    :risks::jsonb, :sent, :summary,
                    :eps_pct, :r3d, :vol20,
                    :ver, now()
                )
                ON CONFLICT (canonical_ticker, as_of_date)
                DO UPDATE SET
                    bullish_probability  = EXCLUDED.bullish_probability,
                    bearish_probability  = EXCLUDED.bearish_probability,
                    no_drift_probability = EXCLUDED.no_drift_probability,
                    confidence           = EXCLUDED.confidence,
                    decision             = EXCLUDED.decision,
                    expected_drift_window = EXCLUDED.expected_drift_window,
                    catalyst_type        = EXCLUDED.catalyst_type,
                    bullish_catalysts    = EXCLUDED.bullish_catalysts,
                    bearish_catalysts    = EXCLUDED.bearish_catalysts,
                    risk_signals         = EXCLUDED.risk_signals,
                    sentiment_score      = EXCLUDED.sentiment_score,
                    narrative_summary    = EXCLUDED.narrative_summary,
                    eps_surprise_pct     = EXCLUDED.eps_surprise_pct,
                    return_3day          = EXCLUDED.return_3day,
                    pre_volatility_20d   = EXCLUDED.pre_volatility_20d,
                    model_version        = EXCLUDED.model_version,
                    generated_at         = now()
            """),
            {
                "ticker": canonical,
                "aod": snapshot_date,
                "bull": signal.bullish_probability,
                "bear": signal.bearish_probability,
                "neutral": signal.no_drift_probability,
                "conf": signal.confidence,
                "decision": signal.decision,
                "window": signal.expected_drift_window,
                "ctype": signal.catalyst_type,
                "bcats": json.dumps(signal.bullish_catalysts or []),
                "bcats_neg": json.dumps(signal.bearish_catalysts or []),
                "risks": json.dumps(signal.risk_signals or []),
                "sent": signal.sentiment_score,
                "summary": signal.narrative_summary,
                "eps_pct": features.get("eps_surprise_pct"),
                "r3d": features.get("return_3day"),
                "vol20": features.get("pre_volatility_20d"),
                "ver": signal.model_version,
            },
        )

    logger.info(
        f"✅ Wrote catalyst snapshot to Neon: {canonical} / {snapshot_date} / {signal.decision}"
    )


async def write_upcoming_prediction(prediction: Dict) -> None:
    """
    Upsert an upcoming event prediction into live.upcoming_prediction.
    prediction should be the dict returned by upcoming_predictor.predict_upcoming().
    """
    await ensure_live_tables()

    from neon_reader import resolve_canonical_ticker

    canonical = await resolve_canonical_ticker(prediction["ticker"])

    async with get_neon_session() as session:
        await session.execute(
            text("""
                INSERT INTO live.upcoming_prediction (
                    canonical_ticker, event_date, as_of_date,
                    analyst_eps_estimate, hist_avg_eps_surprise, days_until_event,
                    predicted_return, return_low, return_high,
                    bullish_probability, bearish_probability, confidence,
                    decision, drift_window, catalyst_type,
                    model_version
                ) VALUES (
                    :ticker, :edate, :aod,
                    :eps_est, :hist_surp, :days,
                    :pred_ret, :ret_low, :ret_high,
                    :bull, :bear, :conf,
                    :decision, :window, :ctype,
                    :ver
                )
                ON CONFLICT (canonical_ticker, event_date, as_of_date)
                DO UPDATE SET
                    predicted_return     = EXCLUDED.predicted_return,
                    return_low           = EXCLUDED.return_low,
                    return_high          = EXCLUDED.return_high,
                    bullish_probability  = EXCLUDED.bullish_probability,
                    bearish_probability  = EXCLUDED.bearish_probability,
                    confidence           = EXCLUDED.confidence,
                    decision             = EXCLUDED.decision,
                    drift_window         = EXCLUDED.drift_window,
                    catalyst_type        = EXCLUDED.catalyst_type,
                    model_version        = EXCLUDED.model_version
            """),
            {
                "ticker": canonical,
                "edate": prediction.get("event_date"),
                "aod": date.today(),
                "eps_est": prediction.get("analyst_eps_estimate"),
                "hist_surp": prediction.get("hist_avg_eps_surprise_pct"),
                "days": prediction.get("days_until_event"),
                "pred_ret": prediction.get("predicted_return"),
                "ret_low": prediction.get("return_low"),
                "ret_high": prediction.get("return_high"),
                "bull": prediction.get("bullish_probability"),
                "bear": prediction.get("bearish_probability"),
                "conf": prediction.get("confidence"),
                "decision": prediction.get("decision"),
                "window": prediction.get("drift_window"),
                "ctype": prediction.get("catalyst_type"),
                "ver": prediction.get("model_version", "1.0.0"),
            },
        )

    logger.info(
        f"✅ Upcoming prediction written to Neon: "
        f"{canonical} event on {prediction.get('event_date')} → {prediction.get('decision')}"
    )


async def write_upcoming_predictions_bulk(predictions: list) -> int:
    """Write a list of upcoming predictions, returns count of rows written."""
    import asyncio
    tasks = [write_upcoming_prediction(p) for p in predictions]
    await asyncio.gather(*tasks, return_exceptions=True)
    return len(predictions)
