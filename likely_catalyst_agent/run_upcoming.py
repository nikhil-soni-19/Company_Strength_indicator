"""
Run the pre-event upcoming earnings predictor and print JSON results.

Usage:
  python run_upcoming.py
  python run_upcoming.py AAPL NVDA MSFT
  python run_upcoming.py AAPL --days 45
"""

import argparse
import asyncio
import json
import sys

from catalyst_agent import LikelyCatalystAgent
from neon_connection import verify_neon_connection
from qdrant_manager import get_qdrant_manager


async def main() -> int:
    parser = argparse.ArgumentParser(description="Upcoming earnings price predictor")
    parser.add_argument(
        "tickers",
        nargs="*",
        default=["AAPL", "NVDA", "MSFT"],
        help="Ticker symbols (default: AAPL NVDA MSFT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Look ahead window in days (default: 30)",
    )
    args = parser.parse_args()

    if not await verify_neon_connection():
        print("ERROR: Cannot connect to Neon. Check NEON_* env vars.", file=sys.stderr)
        return 1

    print("Connecting to Qdrant...")
    qdrant = get_qdrant_manager()
    if not await qdrant.health_check():
        print("ERROR: Cannot connect to Qdrant. Start Qdrant or set QDRANT_HOST/QDRANT_PORT.", file=sys.stderr)
        return 1
    await qdrant.initialize_collections()

    tickers = [t.upper() for t in args.tickers]
    print(f"Running upcoming predictor for {tickers} (next {args.days} days)...\n")

    agent = LikelyCatalystAgent()
    results = await agent.upcoming(tickers, days_ahead=args.days)

    if not results:
        print(
            "No upcoming earnings found in that window for these tickers.\n"
            "Try: more tickers, --days 60, or check yfinance earnings dates.",
            file=sys.stderr,
        )
        return 0

    print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
    print(f"\n{len(results)} prediction(s) saved to upcoming_predictions table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
