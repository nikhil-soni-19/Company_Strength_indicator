"""CLI entry point for the Liquidity Agent.

Examples
--------

::

    python main.py score AAPL
    python main.py score GME --json
    python main.py ask "How liquid is TSLA for a 5% position?"
    python main.py compare AAPL GME TSLA
    python main.py compare AAPL GME TSLA -q "Which is safest for a 5% stake?"
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass

from src.agent import (
    AgentResponse,
    ComparisonResponse,
    LiquidityAgent,
    interpret,
    interpret_comparison,
)
from src.data_ingestion import LiveDataLoader
from src.output import evaluate_confidence, render_comparison, render_dashboard
from src.scoring import score_liquidity

load_dotenv()

app = typer.Typer(help="Institutional liquidity & exit-risk scoring engine.")
console = Console()


@app.command()
def score(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON instead of a rendered dashboard."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the LLM interpretation paragraph."),
) -> None:
    """Score a single ticker end-to-end."""
    loader = LiveDataLoader()
    data = loader.fetch(ticker)
    result = score_liquidity(data)
    confidence = evaluate_confidence(data)
    interpretation = None if no_llm else interpret(result, confidence)

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
            "interpretation": None if interpretation is None else {
                "paragraph": interpretation.paragraph,
                "used_llm": interpretation.used_llm,
                "model": interpretation.model,
            },
        }
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    render_dashboard(result, confidence, console=console, interpretation=interpretation)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Free-form question, e.g. 'How liquid is GME?'"),
    ticker: Optional[str] = typer.Option(None, "--ticker", help="Override ticker extraction (single-ticker)."),
    no_details: bool = typer.Option(False, "--no-details", help="Omit the backing-metrics dashboard."),
) -> None:
    """Ask the agent a natural-language liquidity question.

    The LLM Interpretation panel in the output answers your question
    directly in plain English. If your question mentions two or more
    tickers, the agent automatically routes through the comparison flow.
    """
    agent = LiquidityAgent()
    response = agent.ask(question, ticker=ticker)

    if isinstance(response, ComparisonResponse):
        if response.error and not response.scored:
            console.print(f"[red]{response.error}[/red]")
            sys.exit(2)
        console.rule(
            f"[bold]{' vs '.join(response.tickers)}[/bold]  "
            + ("(LLM)" if response.used_llm else "(deterministic)")
        )
        render_comparison(response.scored, interpretation=response.interpretation, console=console)
        if not no_details:
            console.rule("Per-ticker detail")
            for score_, confidence_ in response.scored:
                render_dashboard(score_, confidence_, console=console)
        if response.error:
            console.print(f"[yellow]{response.error}[/yellow]")
        return

    if response.ticker is None:
        console.print(f"[red]{response.answer}[/red]")
        sys.exit(2)

    console.rule(
        f"[bold]{response.ticker}[/bold]  "
        + ("(LLM)" if response.used_llm else "(deterministic)")
    )
    if response.score and response.confidence:
        render_dashboard(
            response.score,
            response.confidence,
            console=console,
            interpretation=response.interpretation,
        )
    else:
        console.print(response.answer)


@app.command()
def compare(
    tickers: list[str] = typer.Argument(..., help="Two or more tickers, e.g. AAPL GME TSLA"),
    question: Optional[str] = typer.Option(
        None, "--question", "-q",
        help="Optional natural-language question to focus the comparison.",
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the LLM comparison paragraph."),
    details: bool = typer.Option(False, "--details", help="Also print the full single-ticker dashboard for each ticker."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON instead of a rendered dashboard."),
) -> None:
    """Compare the liquidity of two or more tickers side-by-side."""
    if len(tickers) < 2:
        console.print("[red]compare requires at least two tickers, e.g. `compare AAPL GME`.[/red]")
        sys.exit(2)

    loader = LiveDataLoader()
    scored: list[tuple] = []
    errors: list[str] = []
    for tkr in tickers:
        try:
            data = loader.fetch(tkr)
            scored.append((score_liquidity(data), evaluate_confidence(data)))
        except Exception as exc:  # pragma: no cover - depends on yfinance
            errors.append(f"{tkr}: {exc!s}")

    if not scored:
        console.print(f"[red]No tickers could be fetched.[/red]\n" + "\n".join(errors))
        sys.exit(2)

    interpretation = (
        None if no_llm else interpret_comparison(scored, question=question)
    )

    if as_json:
        payload = {
            "question": question,
            "tickers": [s.ticker for s, _ in scored],
            "results": [
                {
                    "ticker": s.ticker,
                    "tier": s.final_tier.number,
                    "tier_label": s.final_tier.label,
                    "raw_score": s.raw_score,
                    "confidence_pct": c.score_pct,
                    "metrics": s.headline_metrics,
                    "flags": s.flags,
                    "mirage_triggered": s.mirage.triggered,
                }
                for s, c in scored
            ],
            "interpretation": None if interpretation is None else {
                "paragraph": interpretation.paragraph,
                "used_llm": interpretation.used_llm,
                "model": interpretation.model,
            },
            "errors": errors,
        }
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    render_comparison(scored, interpretation=interpretation, console=console)
    if details:
        console.rule("Per-ticker detail")
        for s, c in scored:
            render_dashboard(s, c, console=console)
    if errors:
        console.print(f"[yellow]Partial failures: {' | '.join(errors)}[/yellow]")


@app.command()
def chat(
    no_details: bool = typer.Option(False, "--no-details", help="Omit per-ticker dashboards in comparison answers and the startup tier-reference card."),
) -> None:
    """Start an interactive chat session in the terminal.

    Type free-form liquidity questions and the agent will answer each one.
    Mention one ticker (or company name) for a focused answer, or two-plus
    tickers to get a side-by-side comparison automatically.

    In-session commands:
      :help          Show this help.
      :tiers         Re-display the tier reference card.
      :clear         Clear the screen.
      :model <name>  Switch the LLM model for subsequent answers.
      :exit / :quit  Leave the chat (Ctrl-C also works).
    """
    agent = LiquidityAgent()
    _print_chat_banner(agent)
    if not no_details:
        _print_tier_reference()

    while True:
        try:
            raw = console.input("[bold cyan]liquidity>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not raw:
            continue

        lowered = raw.lower()
        if lowered in (":exit", ":quit", "exit", "quit", ":q"):
            console.print("[dim]Session ended.[/dim]")
            break
        if lowered in (":help", "?"):
            _print_chat_help()
            continue
        if lowered in (":tiers", ":tier"):
            _print_tier_reference()
            continue
        if lowered == ":clear":
            console.clear()
            _print_chat_banner(agent)
            if not no_details:
                _print_tier_reference()
            continue
        if lowered.startswith(":model"):
            parts = raw.split(maxsplit=1)
            if len(parts) == 2:
                agent.llm_model = parts[1].strip()
                console.print(f"[green]Model set to[/green] {agent.llm_model}")
            else:
                console.print(f"Current model: [bold]{agent.llm_model}[/bold]")
            continue

        try:
            response = agent.ask(raw)
        except Exception as exc:  # pragma: no cover - depends on yfinance / network
            console.print(f"[red]Error: {exc!s}[/red]")
            continue

        if isinstance(response, ComparisonResponse):
            if response.error and not response.scored:
                console.print(f"[red]{response.error}[/red]")
                continue
            console.rule(
                f"[bold]{' vs '.join(response.tickers)}[/bold]  "
                + ("(LLM)" if response.used_llm else "(deterministic)")
            )
            render_comparison(response.scored, interpretation=response.interpretation, console=console)
            if not no_details:
                console.rule("Per-ticker detail")
                for s, c in response.scored:
                    render_dashboard(s, c, console=console)
            if response.error:
                console.print(f"[yellow]{response.error}[/yellow]")
            continue

        if response.ticker is None:
            console.print(f"[red]{response.answer}[/red]")
            continue

        console.rule(
            f"[bold]{response.ticker}[/bold]  "
            + ("(LLM)" if response.used_llm else "(deterministic)")
        )
        if response.score and response.confidence:
            render_dashboard(
                response.score,
                response.confidence,
                console=console,
                interpretation=response.interpretation,
            )
        else:
            console.print(response.answer)


def _print_chat_banner(agent: LiquidityAgent) -> None:
    has_key = bool(__import__("os").getenv("OPENAI_API_KEY"))
    mode = f"LLM: {agent.llm_model}" if has_key else "LLM: [yellow]offline (deterministic fallback)[/yellow]"
    console.rule("[bold]Liquidity Agent — interactive chat[/bold]")
    console.print(
        "Ask about one ticker for a focused answer, or two+ tickers for a comparison.\n"
        f"{mode}   |   Type [bold]:help[/bold] for in-session commands, [bold]:exit[/bold] to quit."
    )
    console.rule()


def _print_chat_help() -> None:
    console.print(
        "[bold]In-session commands[/bold]\n"
        "  [bold]:help[/bold]          Show this help.\n"
        "  [bold]:tiers[/bold]         Re-display the tier reference card.\n"
        "  [bold]:clear[/bold]         Clear the screen.\n"
        "  [bold]:model <name>[/bold]  Switch the LLM model for subsequent answers.\n"
        "  [bold]:exit[/bold] / [bold]:quit[/bold]  Leave the chat (Ctrl-C also works).\n\n"
        "[bold]Example questions[/bold]\n"
        "  How liquid is AAPL for a 5% position?\n"
        "  Is apple safe to accumulate right now?\n"
        "  Compare apple, tesla and microsft.\n"
        "  Which is the safest among AAPL, GME and TSLA?"
    )


def _print_tier_reference() -> None:
    """Brief explainer card: tiers, scoring direction, short squeeze override."""
    body = (
        "[bold]How to read the score[/bold]\n"
        "Each of 5 dimensions (ADV$, Amihud, Volume CV, DTL@5%, Free Float)\n"
        "contributes 0-3 points. [bold]Higher total = higher liquidity / slippage risk[/bold].\n\n"
        "[bold green]Tier 1 — Unrestricted[/bold green]            "
        "[dim]0-4 pts[/dim]   Safe for market orders; up to 5% stake.\n"
        "[bold yellow]Tier 2 — Position-sizing caps[/bold yellow]    "
        "[dim]5-6 pts[/dim]   Max 1% of float; Limit / TWAP only.\n"
        "[bold dark_orange]Tier 3 — Algorithmic execution[/bold dark_orange]   "
        "[dim]7-8 pts[/dim]   VWAP, <=5% participation; compliance sign-off.\n"
        "[bold red]Tier 4 — Blacklist[/bold red]                "
        "[dim]9+ pts[/dim]    Hard-block. Do not enter.\n\n"
        "[bold]Short Squeeze Risk Override[/bold] — if Float < 10M shares and Short% > 25%,\n"
        "the engine downgrades the tier by 2 (short squeeze liquidity guard)."
    )
    console.print(Panel(body, title="Tier Reference", border_style="blue", expand=False))


if __name__ == "__main__":
    app()
