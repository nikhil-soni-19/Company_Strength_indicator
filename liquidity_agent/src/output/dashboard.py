"""Terminal-side rendering of the liquidity dashboard.

Display policy
--------------
Default Liquidity Metrics table — exactly 4 rows:
  ADV$ (30d) | Float % of Outstanding | Short Interest (% Float) | Top-10 Institutional Holdings

Scored dimensions NOT in the default table:
  * Volume CV (30d)  — shown only when band = Critical or user explicitly asks
  * Free Float (shares) — shown only when band = Critical or user explicitly asks
  * amihud_30d — shown only when user explicitly asks (handled by interpreter)

Score column removed. Scoring still runs in full; the numeric score and tier
appear in the top header panel on every query.

Short Squeeze Risk is always shown in the header panel and in the comparison
summary table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.output.confidence import ConfidenceReport
from src.output.narrative import build_narrative
from src.scoring.engine import LiquidityScore

if TYPE_CHECKING:
    from src.agent.interpreter import Interpretation


_TIER_COLORS = {1: "bold green", 2: "bold yellow", 3: "bold dark_orange", 4: "bold red"}


# ── Dimension helpers ─────────────────────────────────────────────────────────

def _dim_band(score: LiquidityScore, dimension: str) -> str:
    for d in score.dimension_scores:
        if d.dimension == dimension:
            return d.band
    return "Low"


def _dim_is_critical(score: LiquidityScore, dimension: str) -> bool:
    return _dim_band(score, dimension) == "Critical"


# ── Single-ticker dashboard ───────────────────────────────────────────────────

def render_dashboard(
    score: LiquidityScore,
    confidence: ConfidenceReport,
    console: Console | None = None,
    interpretation: Optional[Interpretation] = None,
) -> None:
    console = console or Console()
    color = _TIER_COLORS.get(score.final_tier.number, "white")

    # ── Header panel: Ticker · Tier · Score · Confidence · Short Squeeze Risk ──
    squeeze_status = (
        Text("⚠ Short Squeeze Risk: TRIGGERED", style="bold red")
        if score.mirage.triggered
        else Text("Short Squeeze Risk: ok", style="green")
    )
    conf_style = (
        "green" if confidence.score_pct >= 90
        else "yellow" if confidence.score_pct >= 70
        else "red"
    )
    header = Text.assemble(
        Text(f"{score.ticker}  ", style="bold white"),
        Text(score.final_tier.badge, style=color),
        Text(f"  ·  Score: {score.raw_score}", style="dim white"),
        Text("  ·  "),
        Text(f"Confidence: {confidence.label} ({confidence.score_pct}%)", style=conf_style),
        Text("  ·  "),
        squeeze_status,
    )
    console.print(Panel(header, expand=False, border_style=color))

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(Panel(build_narrative(score), title="Summary", expand=False, border_style="cyan"))

    # ── LLM / Analyst Interpretation ─────────────────────────────────────────
    if interpretation is not None:
        title = "LLM Interpretation" if interpretation.used_llm else "Analyst Interpretation (offline)"
        subtitle = interpretation.model if interpretation.used_llm and interpretation.model else None
        console.print(
            Panel(
                interpretation.paragraph,
                title=title,
                subtitle=subtitle,
                border_style="magenta",
                expand=False,
            )
        )

    # ── Liquidity Metrics table ───────────────────────────────────────────────
    metrics_table = Table(title="Liquidity Metrics", show_lines=False, expand=False)
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")

    # ADV$ (30d) — always shown
    metrics_table.add_row("ADV$ (30d)", _fmt_adv(score.adv.adv_dollar_30d))

    # Volume CV — only if its scoring band is Critical
    if _dim_is_critical(score, "volume_cv_30d"):
        cv = score.volume_cv.volume_cv_30d
        metrics_table.add_row(
            "[red]Volume CV (30d) ⚠ Critical[/red]",
            f"[red]{cv:.4g}[/red]" if cv is not None else "n/a",
        )

    # Free Float share count — only if Critical band
    if _dim_is_critical(score, "free_float_shares"):
        ff = score.structural.float_shares
        metrics_table.add_row(
            "[red]Free Float (shares) ⚠ Critical[/red]",
            f"[red]{ff:,.0f}[/red]" if ff is not None else "n/a",
        )

    # Structural metrics — always shown (no scoring band, just context)
    metrics_table.add_row(
        "Float % of Outstanding",
        _fmt_pct(score.structural.float_pct_of_outstanding),
    )
    metrics_table.add_row(
        "Short Interest (% Float)",
        _fmt_pct(score.structural.short_percent_float),
    )
    metrics_table.add_row(
        "Top-10 Institutional Holdings",
        _fmt_pct(score.structural.top10_institutional_pct),
    )

    # Buyback metrics — quarterly spend + yield always shown; BIR always shown
    # with colour coding: green (<10%), yellow (10-15%), red (>15% = flagged).
    spend = score.buyback.quarterly_spend
    if spend is not None:
        metrics_table.add_row(
            "Buyback Spend (latest quarter)",
            _fmt_adv(spend),
        )
    byield = score.buyback.buyback_yield
    metrics_table.add_row(
        "Buyback Yield (annualised)",
        _fmt_pct(byield) if byield is not None else "n/a",
    )
    bir = score.buyback.bir
    if bir is not None:
        if score.buyback.inflation_flag:
            bir_label = "[red]Buyback Intensity (BIR) ⚠[/red]"
            bir_value = f"[red]{bir:.0%}[/red]"
        elif bir >= 0.10:
            bir_label = "[yellow]Buyback Intensity (BIR)[/yellow]"
            bir_value = f"[yellow]{bir:.0%}[/yellow]"
        else:
            bir_label = "Buyback Intensity (BIR)"
            bir_value = f"{bir:.0%}"
        metrics_table.add_row(bir_label, bir_value)
    else:
        metrics_table.add_row(
            "Buyback Intensity (BIR)",
            "n/a — cashflow unavailable",
        )

    console.print(metrics_table)

    # ── Short Squeeze detail panel (only when triggered) ──────────────────────
    if score.mirage.triggered and score.mirage.reason:
        console.print(
            Panel(
                f"[bold red]⚠ SHORT SQUEEZE RISK — OVERRIDE ACTIVE[/bold red]\n"
                f"{score.mirage.reason}\n"
                f"[dim]Tier downgraded by {score.mirage.downgrade_steps} step(s) "
                f"from {score.mirage.original_tier.badge} → "
                f"{score.mirage.final_tier.badge}.[/dim]",
                border_style="red",
                expand=False,
            )
        )

    # ── Other flags ───────────────────────────────────────────────────────────
    if score.flags:
        other_flags = [
            f for f in score.flags
            if "short squeeze" not in f.lower() and "mirage" not in f.lower()
        ]
        if other_flags:
            flag_body = "\n".join(f"• {f}" for f in other_flags)
            console.print(Panel(flag_body, title="Flags Raised", border_style="red", expand=False))

    # ── Data quality warnings ─────────────────────────────────────────────────
    if confidence.warnings:
        warn_body = "\n".join(f"• {w}" for w in confidence.warnings)
        console.print(Panel(warn_body, title="Data Quality Warnings", border_style="yellow", expand=False))

    console.print(Text(f"Action: {score.final_tier.action}", style="italic"))


# ── Multi-ticker comparison ───────────────────────────────────────────────────

def render_comparison(
    scored: Sequence[tuple[LiquidityScore, ConfidenceReport]],
    interpretation: Optional[Interpretation] = None,
    console: Console | None = None,
) -> None:
    """Render a side-by-side comparison of multiple tickers."""
    console = console or Console()
    if not scored:
        console.print("[red]No tickers to compare.[/red]")
        return

    # Header table: Ticker · Tier · Score · Confidence · Short Squeeze Risk
    header_table = Table(title="Liquidity Comparison", expand=False, show_lines=False)
    header_table.add_column("Ticker", style="bold white")
    header_table.add_column("Tier")
    header_table.add_column("Score", justify="right")
    header_table.add_column("Confidence")
    header_table.add_column("Short Squeeze Risk")

    for s, c in scored:
        color = _TIER_COLORS.get(s.final_tier.number, "white")
        header_table.add_row(
            s.ticker,
            f"[{color}]{s.final_tier.badge}[/{color}]",
            str(s.raw_score),
            f"{c.label} ({c.score_pct}%)",
            "[red]⚠ TRIGGERED[/red]" if s.mirage.triggered else "[green]ok[/green]",
        )
    console.print(header_table)

    # LLM / Analyst comparison paragraph
    if interpretation is not None:
        title = "LLM Comparison" if interpretation.used_llm else "Analyst Comparison (offline)"
        subtitle = interpretation.model if interpretation.used_llm and interpretation.model else None
        console.print(
            Panel(
                interpretation.paragraph,
                title=title,
                subtitle=subtitle,
                border_style="magenta",
                expand=False,
            )
        )

    # Metric grid — same 4-row policy as single-ticker view
    metrics_table = Table(title="Metric Grid", expand=False, show_lines=False)
    metrics_table.add_column("Metric", style="bold")
    for s, _ in scored:
        metrics_table.add_column(s.ticker, justify="right")

    # ADV$ — always
    adv_row = ["ADV$ (30d)"]
    for s, _ in scored:
        adv_row.append(_fmt_adv(s.adv.adv_dollar_30d))
    metrics_table.add_row(*adv_row)

    # Volume CV — only if Critical for any ticker
    if any(_dim_is_critical(s, "volume_cv_30d") for s, _ in scored):
        cv_row = ["Volume CV (30d)"]
        for s, _ in scored:
            cv = s.volume_cv.volume_cv_30d
            is_crit = _dim_is_critical(s, "volume_cv_30d")
            val = (f"[red]{cv:.4g} ⚠[/red]" if is_crit else f"{cv:.4g}") if cv is not None else "n/a"
            cv_row.append(val)
        metrics_table.add_row(*cv_row)

    # Free Float shares — only if Critical for any ticker
    if any(_dim_is_critical(s, "free_float_shares") for s, _ in scored):
        ff_row = ["Free Float (shares)"]
        for s, _ in scored:
            ff = s.structural.float_shares
            is_crit = _dim_is_critical(s, "free_float_shares")
            val = (f"[red]{ff:,.0f} ⚠[/red]" if is_crit else f"{ff:,.0f}") if ff is not None else "n/a"
            ff_row.append(val)
        metrics_table.add_row(*ff_row)

    # Structural rows — always shown
    float_pct_row = ["Float % of Outstanding"]
    for s, _ in scored:
        float_pct_row.append(_fmt_pct(s.structural.float_pct_of_outstanding))
    metrics_table.add_row(*float_pct_row)

    short_pct_row = ["Short Interest (% Float)"]
    for s, _ in scored:
        short_pct_row.append(_fmt_pct(s.structural.short_percent_float))
    metrics_table.add_row(*short_pct_row)

    inst_pct_row = ["Top-10 Institutional Holdings"]
    for s, _ in scored:
        inst_pct_row.append(_fmt_pct(s.structural.top10_institutional_pct))
    metrics_table.add_row(*inst_pct_row)

    # Buyback Spend — shown when at least one ticker has data
    if any(s.buyback.quarterly_spend is not None for s, _ in scored):
        spend_row = ["Buyback Spend (latest quarter)"]
        for s, _ in scored:
            spend = s.buyback.quarterly_spend
            spend_row.append(_fmt_adv(spend) if spend is not None else "n/a")
        metrics_table.add_row(*spend_row)

    # Buyback Yield — always shown in comparison
    byield_row = ["Buyback Yield (annualised)"]
    for s, _ in scored:
        byield = s.buyback.buyback_yield
        byield_row.append(_fmt_pct(byield) if byield is not None else "n/a")
    metrics_table.add_row(*byield_row)

    # BIR — always shown with colour coding: green (<10%), yellow (10-15%), red (>15%)
    bir_row = ["Buyback Intensity (BIR)"]
    for s, _ in scored:
        bir = s.buyback.bir
        if bir is None:
            bir_row.append("n/a")
        elif s.buyback.inflation_flag:
            bir_row.append(f"[red]{bir:.0%} ⚠[/red]")
        elif bir >= 0.10:
            bir_row.append(f"[yellow]{bir:.0%}[/yellow]")
        else:
            bir_row.append(f"{bir:.0%}")
    metrics_table.add_row(*bir_row)

    console.print(metrics_table)

    # Other flags (non-short-squeeze)
    per_ticker_flag_lines: list[str] = []
    for s, _ in scored:
        other_flags = [
            f for f in s.flags
            if "short squeeze" not in f.lower() and "mirage" not in f.lower()
        ]
        if other_flags:
            per_ticker_flag_lines.append(f"[bold]{s.ticker}[/bold]")
            per_ticker_flag_lines.extend(f"  • {f}" for f in other_flags)
    if per_ticker_flag_lines:
        console.print(
            Panel("\n".join(per_ticker_flag_lines), title="Flags Raised", border_style="red", expand=False)
        )

    # Short squeeze callout for triggered tickers
    squeeze_hits = [(s, c) for s, c in scored if s.mirage.triggered]
    if squeeze_hits:
        lines = [
            f"[bold red]{s.ticker}[/bold red] — {s.mirage.reason}  "
            f"[dim]({s.mirage.original_tier.badge} → {s.mirage.final_tier.badge})[/dim]"
            for s, _ in squeeze_hits
        ]
        console.print(
            Panel(
                "\n".join(lines),
                title="[bold red]⚠ Short Squeeze Risk — Override Triggered[/bold red]",
                border_style="red",
                expand=False,
            )
        )

    # Data quality warnings
    warning_lines: list[str] = []
    for s, c in scored:
        if c.warnings:
            warning_lines.append(f"[bold]{s.ticker}[/bold]")
            warning_lines.extend(f"  • {w}" for w in c.warnings)
    if warning_lines:
        console.print(
            Panel("\n".join(warning_lines), title="Data Quality Warnings", border_style="yellow", expand=False)
        )


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_adv(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.2f}M"
    if v >= 1e3:
        return f"${v / 1e3:.1f}K"
    return f"${v:.0f}"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2%}"


# kept for any callers that still reference it
def _fmt_dollars(v: float) -> str:
    return _fmt_adv(v)


def _value_for_dimension(score: LiquidityScore, dimension: str) -> Optional[float]:
    if dimension == "adv_dollar_30d":
        return score.adv.adv_dollar_30d
    if dimension == "volume_cv_30d":
        return score.volume_cv.volume_cv_30d
    if dimension == "free_float_shares":
        return score.structural.float_shares
    if dimension == "amihud_30d":
        return score.amihud.amihud_30d
    if dimension == "float_pct_outstanding":
        return score.structural.float_pct_of_outstanding
    if dimension == "short_percent_float":
        return score.structural.short_percent_float
    if dimension == "top10_institutional_pct":
        return score.structural.top10_institutional_pct
    if dimension == "buyback_bir":
        return score.buyback.bir
    return None
