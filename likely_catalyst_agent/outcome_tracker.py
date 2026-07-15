"""
Phase 5: Retrospective accuracy tracker for upcoming predictions.
Runs after event dates pass: attaches actual returns, computes error,
and feeds labeled outcomes back into Qdrant for similarity search.
"""

from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

from settings import settings
from enums import DriftLabel
from datatypes import SECFiling
from market_pipeline import MarketDataPipeline
from embeddings import get_embedding_generator
from qdrant_manager import get_qdrant_manager
from logger import get_logger

logger = get_logger(__name__)


class OutcomeTracker:
    """
    Closes the feedback loop on UpcomingPrediction records.

    1. Find predictions where event_date has passed and outcomes are missing
    2. Compute forward returns via MarketDataPipeline
    3. Update prediction_error = predicted - actual
    4. Upsert labeled filing embeddings into Qdrant
    """

    def __init__(self):
        self.market = MarketDataPipeline()
        self.embedding_gen = get_embedding_generator()
        self.qdrant = get_qdrant_manager()

    async def track_completed_events(
        self,
        grace_days: int = 25,
        min_forward_days: int = 20,
    ) -> Dict:
        """
        Process all untracked predictions whose event_date is in the past.

        ``grace_days``: minimum calendar days after event before tracking
        ``min_forward_days``: require at least this many trading days of data
        """
        cutoff = datetime.utcnow() - timedelta(days=grace_days)
        pending = await self._load_pending_predictions(cutoff)

        if not pending:
            logger.info("No pending upcoming predictions to track")
            return {"tracked": 0, "errors": 0}

        tracked = 0
        errors = 0
        for pred in pending:
            try:
                updated = await self._track_single(pred, min_forward_days)
                if updated:
                    tracked += 1
            except Exception as e:
                errors += 1
                logger.error(f"[{pred.ticker}] Outcome tracking failed: {e}")

        logger.info(f"Outcome tracker: {tracked} updated, {errors} errors")
        return {"tracked": tracked, "errors": errors, "pending": len(pending)}

    async def _load_pending_predictions(
        self, event_before: datetime
    ) -> List[Dict]:
        """Load untracked upcoming predictions from Neon live.upcoming_prediction."""
        from neon_connection import get_neon_session
        from sqlalchemy import text
        try:
            async with get_neon_session() as session:
                rows = await session.execute(text("""
                    SELECT canonical_ticker AS ticker, event_date, as_of_date,
                           predicted_return, actual_return_20d, actual_return_60d,
                           catalyst_type
                    FROM live.upcoming_prediction
                    WHERE event_date <= :cutoff
                      AND outcome_recorded_at IS NULL
                """), {"cutoff": event_before.date()})
                return [dict(r._mapping) for r in rows.fetchall()]
        except Exception as e:
            logger.warning(f"Could not load pending predictions from Neon: {e}")
            return []

    async def _track_single(
        self, pred: Dict, min_forward_days: int
    ) -> bool:
        ticker = pred["ticker"]
        event_date = pred["event_date"]
        if isinstance(event_date, date) and not isinstance(event_date, datetime):
            event_date = datetime(event_date.year, event_date.month, event_date.day)

        returns = await self.market.compute_forward_returns(ticker, event_date)
        actual_20 = returns.get("return_20day")
        actual_60 = returns.get("return_60day")

        if actual_20 is None and actual_60 is None:
            return False

        primary_actual = actual_20 if actual_20 is not None else actual_60
        prediction_error = None
        predicted = pred.get("predicted_return")
        if predicted is not None and primary_actual is not None:
            prediction_error = float(predicted) - primary_actual

        # Write outcome back to Neon
        from neon_connection import get_neon_session
        from sqlalchemy import text
        async with get_neon_session() as session:
            await session.execute(text("""
                UPDATE live.upcoming_prediction
                SET actual_return_20d   = :r20,
                    actual_return_60d   = :r60,
                    prediction_error    = :err,
                    outcome_recorded_at = now()
                WHERE canonical_ticker = :ticker
                  AND event_date       = :edate
            """), {
                "r20": actual_20, "r60": actual_60,
                "err": prediction_error,
                "ticker": ticker,
                "edate": pred["event_date"],
            })

        await self._enrich_qdrant(
            ticker, event_date, primary_actual, pred.get("catalyst_type")
        )
        return True

    async def _enrich_qdrant(
        self,
        ticker: str,
        event_date: datetime,
        actual_return: Optional[float],
        catalyst_type: Optional[str],
    ) -> None:
        """Add labeled outcome to Qdrant for future similarity queries."""
        filing = await self._get_filing_near_event(ticker, event_date)
        if not filing or not filing.mda_text:
            return

        drift_label = self._return_to_label(actual_return)
        try:
            embedding = await self.embedding_gen.encode_async(filing.mda_text[:4000])
            point_id = f"{ticker}_{event_date.strftime('%Y%m%d')}_outcome"
            await self.qdrant.upsert_embedding(
                collection="filings",
                point_id=point_id,
                vector=embedding,
                payload={
                    "ticker": ticker,
                    "filing_date": filing.filing_date.isoformat() if filing.filing_date else None,
                    "event_date": event_date.isoformat(),
                    "catalyst_type": catalyst_type or "Unknown",
                    "drift_label": drift_label,
                    "actual_return_20d": actual_return,
                    "source": "upcoming_outcome_tracker",
                },
            )
        except Exception as e:
            logger.warning(f"Qdrant enrichment failed for {ticker}: {e}")

    async def _get_filing_near_event(
        self, ticker: str, event_date: datetime
    ) -> Optional[SECFiling]:
        """Fetch filing MD&A from Neon."""
        import neon_reader
        filing_context = await neon_reader.build_filing_context(ticker)
        if not filing_context:
            return None
        return SECFiling(
            ticker=ticker,
            mda_text=filing_context.get("mda_text"),
            accession_number=filing_context.get("accession_number"),
        )

    @staticmethod
    def _return_to_label(actual_return: Optional[float]) -> str:
        if actual_return is None:
            return DriftLabel.NEUTRAL.value
        threshold = settings.market.drift_threshold
        if actual_return > threshold:
            return DriftLabel.BULLISH.value
        if actual_return < -threshold:
            return DriftLabel.BEARISH.value
        return DriftLabel.NEUTRAL.value


_tracker: Optional[OutcomeTracker] = None


def get_outcome_tracker() -> OutcomeTracker:
    global _tracker
    if _tracker is None:
        _tracker = OutcomeTracker()
    return _tracker
