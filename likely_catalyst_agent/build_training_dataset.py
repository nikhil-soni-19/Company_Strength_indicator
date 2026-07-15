"""
Training dataset builder — converts all Neon filings into labeled examples
and trains the XGBoost model.

What this script does:
  1. Reads every available filing period for each ticker from Neon
  2. For each period, fetches MD&A text + financial facts from Neon
  3. Fetches post-event price returns from yfinance (price data only — no EPS from yfinance)
  4. Builds a feature vector using FeatureEngineer
  5. Labels each example (BULLISH / BEARISH / NEUTRAL) from 60-day abnormal return
  6. Splits into train/val sets (temporal split — older = train, recent = val)
  7. Trains the XGBoost model and saves it to ./models/saved/xgb_catalyst.pkl

Run:
    cd torch_intern
    python build_training_dataset.py

Expected output with 2 tickers × ~10-12 quarters each:
    ~20-24 training examples (small but enough to verify the pipeline works)
    Model saved → ./models/saved/xgb_catalyst.pkl
"""

import asyncio
import os
import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from neon_connection import verify_neon_connection
from neon_reader import get_available_periods, get_mda_text, get_eps_facts
from market_pipeline import MarketDataPipeline
from feature import FeatureEngineer, FEATURE_SCHEMA
from enums import DriftLabel, FilingType
from datatypes import EarningsEvent, SECFiling
from xgb_models import DriftPredictionModel, get_prediction_model
from settings import settings
from logger import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT"]   # extend when more tickers are added to Neon

# Temporal split: periods before this date → train, on/after → validation
VAL_CUTOFF = datetime(2024, 1, 1).date()

MODEL_DIR = Path("./models/saved")
MODEL_PATH = MODEL_DIR / "xgb_catalyst.pkl"
DATASET_CSV = MODEL_DIR / "training_dataset.csv"   # saved for inspection


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _build_example(
    ticker: str,
    period: Dict,
    market_pipeline: MarketDataPipeline,
    feature_engineer: FeatureEngineer,
) -> Optional[Dict]:
    """
    Build one training example for a (ticker, period_end_date) pair.
    All EPS data comes from Neon (ontology.financial_facts).
    Price return data comes from yfinance (OHLCV only — no EPS).
    Returns a dict with features + label, or None if data is insufficient.
    """
    period_end_date = period["period_end_date"]
    logger.info(f"  Building example: {ticker} / {period_end_date}")

    # ── 1. Get MD&A text from Neon ────────────────────────────────────────
    mda_text = await get_mda_text(ticker, period_end_date)

    # ── 2. Get financial facts from Neon ──────────────────────────────────
    neon_facts = await get_eps_facts(ticker, period_end_date)
    reported_eps = float(neon_facts.get("reported_eps") or 0.0)

    # ── 3. Announcement date: filed_date is now embedded in period dict ───
    raw_date = period.get("filed_date")
    if raw_date is None:
        logger.warning(f"    No filed_date in Neon for {ticker}/{period_end_date} — skipping")
        return None
    announcement_date = (
        datetime(raw_date.year, raw_date.month, raw_date.day)
        if not isinstance(raw_date, datetime)
        else raw_date
    )

    # Analyst estimates not yet in Neon — default to zeros until added.
    estimated_eps    = 0.0
    eps_surprise     = 0.0
    eps_surprise_pct = 0.0
    earnings_beat    = False

    # ── 4. Compute forward returns (60-day PEAD window) from yfinance ─────
    forward_returns = await market_pipeline.compute_forward_returns(
        ticker, announcement_date
    )
    abnormal_return = forward_returns.get("abnormal_return_60day")
    if abnormal_return is None or math.isnan(abnormal_return):
        logger.warning(f"    No 60-day return for {ticker}/{announcement_date} — skipping")
        return None

    # ── 5. Assign drift label ─────────────────────────────────────────────
    drift_label = market_pipeline.assign_drift_label(abnormal_return, earnings_beat)

    # ── 6. Build synthetic ORM-like objects for FeatureEngineer ──────────
    synthetic_event = EarningsEvent(
        ticker=ticker,
        announcement_date=announcement_date,
        reported_eps=reported_eps,
        estimated_eps=estimated_eps,
        eps_surprise=eps_surprise,
        eps_surprise_pct=eps_surprise_pct,
        earnings_beat=earnings_beat,
        return_3day=forward_returns.get("return_3day") or 0.0,
        return_20day=forward_returns.get("return_20day"),
        return_60day=forward_returns.get("return_60day"),
        abnormal_return_60day=abnormal_return,
        drift_label=drift_label,
    )

    synthetic_filing = SECFiling(
        ticker=ticker,
        cik="",
        filing_type=FilingType.FORM_10Q,
        accession_number=f"neon-{ticker}-{period_end_date}",
        filing_date=announcement_date,
        mda_text=mda_text or "",
    )

    # ── 7. Extract full feature vector ────────────────────────────────────
    try:
        features = await feature_engineer.extract_all_features(
            ticker=ticker,
            filing=synthetic_filing,
            earnings_event=synthetic_event,
        )
    except Exception as e:
        logger.warning(f"    Feature extraction failed for {ticker}/{period_end_date}: {e}")
        return None

    return {
        "ticker": ticker,
        "period_end_date": str(period_end_date),
        "announcement_date": str(announcement_date.date()),
        "drift_label": drift_label.value,
        "abnormal_return_60day": abnormal_return,
        "eps_surprise_pct": eps_surprise_pct,
        **features,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("\n=== Training Dataset Builder ===\n")

    neon_ok = await verify_neon_connection()
    if not neon_ok:
        print("❌ Neon connection failed — cannot build dataset")
        return

    market_pipeline   = MarketDataPipeline()
    feature_engineer  = FeatureEngineer()

    all_examples: List[Dict] = []

    for ticker in TICKERS:
        print(f"\n── {ticker} ──────────────────────────────────")

        # Available periods in Neon
        periods = await get_available_periods(ticker)
        print(f"  Periods in Neon: {len(periods)}")
        if not periods:
            print(f"  ⚠️  No periods found for {ticker} — skipping")
            continue

        for period in periods:
            try:
                example = await _build_example(
                    ticker, period, market_pipeline, feature_engineer
                )
                if example:
                    all_examples.append(example)
                    print(f"    ✅ {period['period_end_date']} → {example['drift_label']} "
                          f"(abnormal_ret={example['abnormal_return_60day']:.3f})")
            except Exception as e:
                logger.warning(f"  Failed {ticker}/{period['period_end_date']}: {e}")

    print(f"\n\nTotal training examples built: {len(all_examples)}")

    if len(all_examples) < 4:
        print("⚠️  Too few examples to train meaningfully (need at least 4).")
        print("   Check that yfinance earnings dates overlap with your Neon periods.")
        return

    # ── Save dataset CSV for inspection ──────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_examples)
    df.to_csv(DATASET_CSV, index=False)
    print(f"Dataset saved → {DATASET_CSV}")
    print(f"\nLabel distribution:\n{df['drift_label'].value_counts().to_string()}")

    # ── Temporal train / val split ────────────────────────────────────────
    df["period_end_date"] = pd.to_datetime(df["period_end_date"]).dt.date
    train_df = df[df["period_end_date"] < VAL_CUTOFF]
    val_df   = df[df["period_end_date"] >= VAL_CUTOFF]

    print(f"\nTrain size : {len(train_df)}  (before {VAL_CUTOFF})")
    print(f"Val size   : {len(val_df)}  (from {VAL_CUTOFF} onwards)")

    if len(train_df) < 3:
        print("⚠️  Not enough training rows before val cutoff — using all data as train.")
        train_df = df
        val_df   = pd.DataFrame()

    # ── Extract features + labels ─────────────────────────────────────────
    def to_feature_list(frame: pd.DataFrame) -> Tuple[List[Dict], List[str]]:
        feat_cols = [c for c in FEATURE_SCHEMA if c in frame.columns]
        feats  = frame[feat_cols].fillna(0).to_dict(orient="records")
        labels = frame["drift_label"].tolist()
        return feats, labels

    train_feats, train_labels = to_feature_list(train_df)
    val_feats,   val_labels   = to_feature_list(val_df) if not val_df.empty else (None, None)

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n── Training XGBoost model ──────────────────────────────────")
    model = DriftPredictionModel()
    metrics = model.train(
        train_features=train_feats,
        train_labels=train_labels,
        val_features=val_feats if val_feats else None,
        val_labels=val_labels if val_labels else None,
    )
    print(f"Training metrics: {metrics}")

    # ── Save model ────────────────────────────────────────────────────────
    model.save(str(MODEL_PATH))
    print(f"\n✅ Model saved → {MODEL_PATH}")
    print("\nNow run:  python run_smoke.py")
    print("The agent will load the trained model and produce real predictions.\n")


if __name__ == "__main__":
    asyncio.run(main())
