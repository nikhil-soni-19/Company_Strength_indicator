"""CLI: load Bloomberg EEG/ERN workbooks into ontology market tables.

Examples:
    python -m src.ingest_estimates --dry-run
    python -m src.ingest_estimates --tickers AAPL
    python -m src.ingest_estimates --eeg path/to/EEG.xlsx --ern path/to/ERN.xlsx
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from config import COMPANIES
from src.db import get_engine
from src.market_estimates import parse_eeg, parse_eeg_price, parse_ern
from src.ontology import (
    init_ontology_schema,
    upsert_company,
    upsert_earnings_surprise,
    upsert_estimate_consensus,
    upsert_price_daily,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv(_REPO_ROOT / ".env.ingestion", override=True)
    cloud = _REPO_ROOT / "database_cloud.env"
    if cloud.exists():
        load_dotenv(cloud, override=True)


def _company_cfg(ticker: str) -> dict:
    for co in COMPANIES:
        if co["ticker"] == ticker:
            return co
    raise SystemExit(f"Unknown ticker {ticker!r}; not in config.COMPANIES")


def _default_paths(ticker: str) -> tuple[Path, Path]:
    base = _REPO_ROOT / "data" / "source" / ticker / f"{ticker}-EEG-ERN"
    return (
        base / f"{ticker} - EEG.xlsx",
        base / f"{ticker} - ERN.xlsx",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m src.ingest_estimates")
    p.add_argument(
        "--tickers",
        type=str,
        default="AAPL",
        help="Comma-separated tickers (default: AAPL)",
    )
    p.add_argument("--eeg", type=Path, default=None, help="Override EEG workbook path")
    p.add_argument("--ern", type=Path, default=None, help="Override ERN workbook path")
    p.add_argument(
        "--skip-price",
        action="store_true",
        help="Skip price_daily load from EEG cols G/H",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print counts only; no DB writes",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def _print_sample(label: str, rows: list[dict], *, n: int = 1) -> None:
    if not rows:
        print(f"  {label}: (empty)")
        return
    print(f"  {label} first: {rows[0]}")
    if len(rows) > 1:
        print(f"  {label} last:  {rows[-1]}")


def _ingest_ticker(
    ticker: str,
    *,
    eeg_path: Path,
    ern_path: Path,
    skip_price: bool,
    dry_run: bool,
) -> dict[str, int]:
    if not eeg_path.is_file():
        raise FileNotFoundError(f"EEG workbook not found: {eeg_path}")
    if not ern_path.is_file():
        raise FileNotFoundError(f"ERN workbook not found: {ern_path}")

    estimates = parse_eeg(eeg_path, ticker=ticker)
    prices = [] if skip_price else parse_eeg_price(eeg_path, ticker=ticker)
    surprises = parse_ern(ern_path, ticker=ticker)

    by_period = Counter(r["target_period"] for r in estimates)
    reported_n = sum(1 for r in surprises if r["is_reported"])
    forward_n = len(surprises) - reported_n

    print(f"\n[{ticker}] EEG: {eeg_path.name}")
    print(f"  estimate_consensus: {len(estimates)} total")
    for period, count in sorted(by_period.items()):
        print(f"    {period}: {count}")
    if not skip_price:
        print(f"  price_daily: {len(prices)}")
    print(f"[{ticker}] ERN: {ern_path.name}")
    print(f"  earnings_surprise: {len(surprises)} ({reported_n} reported + {forward_n} forward)")

    if dry_run:
        _print_sample("estimate", estimates)
        if prices:
            _print_sample("price", prices)
        _print_sample("surprise", surprises)
        return {
            "estimate_consensus": len(estimates),
            "price_daily": len(prices),
            "earnings_surprise": len(surprises),
        }

    engine = get_engine()
    init_ontology_schema(engine)
    co = _company_cfg(ticker)
    upsert_company(
        engine,
        ticker,
        legal_name=co.get("legal_name"),
        cik=co.get("cik"),
        fiscal_year_end=co.get("fiscal_year_end"),
        hq_country=co.get("hq_country"),
    )

    n_est = upsert_estimate_consensus(engine, estimates)
    n_px = upsert_price_daily(engine, prices) if prices else 0
    n_sur = upsert_earnings_surprise(engine, surprises)
    print(f"  inserted estimate_consensus: {n_est} new rows")
    print(f"  inserted price_daily: {n_px} new rows")
    print(f"  upserted earnings_surprise: {n_sur} rows touched")
    return {
        "estimate_consensus": len(estimates),
        "price_daily": len(prices),
        "earnings_surprise": len(surprises),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _load_env()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    totals: dict[str, int] = Counter()

    for ticker in tickers:
        default_eeg, default_ern = _default_paths(ticker)
        eeg_path = args.eeg or default_eeg
        ern_path = args.ern or default_ern
        counts = _ingest_ticker(
            ticker,
            eeg_path=eeg_path,
            ern_path=ern_path,
            skip_price=args.skip_price,
            dry_run=args.dry_run,
        )
        totals.update(counts)

    if args.dry_run:
        print("\n[dry-run] no DB writes")
        print(f"  totals: {dict(totals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
