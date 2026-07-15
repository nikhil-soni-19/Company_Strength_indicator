#!/usr/bin/env python3
"""Agent 7 CLI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")


def cmd_init_db(args):
    from db.connection import init_schema
    init_schema()


def cmd_ingest_10k(args):
    """Chunk, embed, and upsert a 10-K text file into risk_factors."""
    from ingestion.risk_factors_ingest import load_from_filing
    filing_date = date.fromisoformat(args.filing_date) if args.filing_date else None
    n = load_from_filing(
        ticker=args.ticker,
        fiscal_year=int(args.year),
        text_path=args.file,
        filing_date=filing_date,
    )
    print(f"Done. {n} chunks upserted for {args.ticker} FY{args.year}.")


def cmd_chat(args):
    from agent.chat import run_chat
    run_chat(lookback_days=int(args.lookback) if args.lookback else 126)


def cmd_run(args):
    from agent.run import run_agent
    print(f"Running Agent 7 for {args.ticker}...")
    result = run_agent(
        ticker=args.ticker,
        as_of_date=args.as_of or None,
        lookback_days=int(args.lookback) if args.lookback else 126,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_backtest(args):
    from agent.run import run_agent

    ticker = args.ticker
    start  = date.fromisoformat(args.start)
    end    = date.fromisoformat(args.end)
    freq   = args.freq or "monthly"

    as_of_dates = []
    current = start
    while current <= end:
        as_of_dates.append(current)
        if freq == "monthly":
            m = current.month % 12 + 1
            y = current.year + (1 if current.month == 12 else 0)
            try:
                current = current.replace(year=y, month=m)
            except ValueError:
                import calendar
                last_day = calendar.monthrange(y, m)[1]
                current = current.replace(year=y, month=m, day=last_day)
        elif freq == "weekly":
            current = current + timedelta(weeks=1)
        elif freq == "daily":
            current = current + timedelta(days=1)
        else:
            current = current + timedelta(days=30)

    results = []
    for d in as_of_dates:
        print(f"  Backtesting {ticker} as of {d}...")
        try:
            r = run_agent(ticker=ticker, as_of_date=d)
            results.append({
                "as_of_date":        r["as_of_date"],
                "environment_score": r["environment_score"],
                "direction":         r["direction"],
                "quant_score":       r["quant_score"],
                "qual_score":        r["qual_score"],
                "flags":             r["flags"],
            })
        except Exception as e:
            print(f"  [ERROR] {d}: {e}")
            results.append({"as_of_date": d.isoformat(), "error": str(e)})

    out_path = Path(f"backtest_{ticker}_{start}_{end}.json")
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Backtest results written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Agent 7 — Company Environment")
    sub = parser.add_subparsers(dest="command", required=True)

    # init-db
    p_init = sub.add_parser("init-db", help="Initialise Neon DB schema (risk_factors + environment_runs)")
    p_init.set_defaults(func=cmd_init_db)

    # ingest-10k
    p_10k = sub.add_parser("ingest-10k", help="Ingest a 10-K filing text file into risk_factors")
    p_10k.add_argument("--ticker",       required=True, help="Company ticker (e.g. AAPL)")
    p_10k.add_argument("--year",         required=True, help="Fiscal year (e.g. 2024)")
    p_10k.add_argument("--file",         required=True, help="Path to plain-text 10-K file")
    p_10k.add_argument("--filing-date",  dest="filing_date", default=None,
                       help="Filing date YYYY-MM-DD (optional)")
    p_10k.set_defaults(func=cmd_ingest_10k)

    # chat  ← main interactive entry point
    p_chat = sub.add_parser("chat", help="Interactive query loop — ask anything, get environment analysis")
    p_chat.add_argument("--lookback", default="126",
                        help="Lookback trading days (default 126 = ~6 months)")
    p_chat.set_defaults(func=cmd_chat)

    # run
    p_run = sub.add_parser("run", help="Run the environment agent for a ticker")
    p_run.add_argument("--ticker",   required=True, help="Company ticker symbol")
    p_run.add_argument("--as-of",    dest="as_of", default=None,
                       help="As-of date (YYYY-MM-DD, default today)")
    p_run.add_argument("--lookback", default="126",
                       help="Lookback trading days (default 126 = ~6 months)")
    p_run.set_defaults(func=cmd_run)

    # backtest
    p_bt = sub.add_parser("backtest", help="Run agent across a date range")
    p_bt.add_argument("--ticker", required=True)
    p_bt.add_argument("--start",  required=True, help="Start date YYYY-MM-DD")
    p_bt.add_argument("--end",    required=True, help="End date YYYY-MM-DD")
    p_bt.add_argument("--freq",   default="monthly",
                      choices=["monthly", "weekly", "daily"])
    p_bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
