"""CLI entry point for the Liquidity Agent.

Examples
--------

::

    python main.py score AAPL
    python main.py score GME --json
    python main.py ask "How liquid is TSLA for a 5% position?"
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from src.agent import LiquidityAgent
from src.data_ingestion import LiveDataLoader
from src.output import evaluate_confidence, render_dashboard
from src.scoring import score_liquidity

load_dotenv()

app = typer.Typer(help="Institutional liquidity & exit-risk scoring engine.")
console = Console()


@app.command()
def score(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON instead of a rendered dashboard."),
) -> None:
    """Score a single ticker end-to-end."""
    loader = LiveDataLoader()
    data = loader.fetch(ticker)
    result = score_liquidity(data)
    confidence = evaluate_confidence(data)

    if as_json:
        payload = {
            "ticker": result.ticker,
            "tier": result.final_tier.number,
            "tier_label": result.final_tier.label,
            "raw_score": result.raw_score,
            "confidence_pct": confidence.score_pct,
            "metrics": result.headline_metrics,
            "flags": result.flags,
            "warnings": confidence.warnings,
            "mirage_triggered": result.mirage.triggered,
        }
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    render_dashboard(result, confidence, console=console)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Free-form question, e.g. 'How liquid is GME?'"),
    ticker: Optional[str] = typer.Option(None, "--ticker", help="Override ticker extraction."),
) -> None:
    """Ask the LLM-backed agent a natural-language question."""
    agent = LiquidityAgent()
    response = agent.ask(question, ticker=ticker)

    if response.ticker is None:
        console.print(f"[red]{response.answer}[/red]")
        sys.exit(2)

    console.rule(f"[bold]{response.ticker}[/bold]  " + ("(LLM)" if response.used_llm else "(deterministic)"))
    console.print(response.answer)

    if response.score and response.confidence:
        console.rule("Backing metrics")
        render_dashboard(response.score, response.confidence, console=console)


if __name__ == "__main__":
    app()
