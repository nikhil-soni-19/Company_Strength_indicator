"""
Agent 4 — Capability Stack: orchestrator and CLI entry point.

Usage:
    python agent.py AAPL
    python agent.py MSFT --quarters 12

Programmatic:
    from agent import run
    result = run("AAPL")        # returns validated dict
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from config import DEFAULT_QUARTERS
from data_contract import fetch_inputs
from fusion import fuse
from layer1_deterministic import Layer1Output, run_layer1
from layer2_llm import run_layer2
from schema import FinalOutputSchema, validate_output


# ─── Display helpers ──────────────────────────────────────────────────────────

W = 72
_line  = "═" * (W - 2)
_dline = "─" * (W - 2)
_blank = " " * (W - 2)


def _section(title: str) -> None:
    print(f"\n╔{_line}╗")
    print(f"║  {title:<{W-4}}║")
    print(f"╠{_line}╣")


def _section_end() -> None:
    print(f"╚{_line}╝")


def _divider() -> None:
    print(f"╠{_dline}╣")


def _blank_row() -> None:
    print(f"║{_blank}║")


def _row(label: str, value: str, lw: int = 26) -> None:
    print(f"║  {label:<{lw}}{value:<{W-lw-4}}║")


def _wrap_row(label: str, text: str, lw: int = 26) -> None:
    content_w = W - lw - 4
    words = (text or "").split()
    lines, cur = [], []
    for w in words:
        if sum(len(x) + 1 for x in cur) + len(w) > content_w:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    for i, line in enumerate(lines or [""]):
        lbl = label if i == 0 else ""
        print(f"║  {lbl:<{lw}}{line:<{content_w}}║")


def _bullet_rows(items: list[str], indent: int = 4) -> None:
    for item in items:
        content_w = W - indent - 2
        words, wlines, cur = item.split(), [], []
        for w in words:
            if sum(len(x) + 1 for x in cur) + len(w) > content_w - 3:
                wlines.append(" ".join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            wlines.append(" ".join(cur))
        for i, line in enumerate(wlines or [""]):
            prefix = "•  " if i == 0 else "   "
            print(f"║{' '*indent}{prefix}{line:<{content_w-3}}║")


def _score_bar(score: float, max_score: float = 10.0, width: int = 20) -> str:
    filled = int(round((score / max_score) * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.1f}/{max_score:.0f}"


def _conf_bar(conf: float, width: int = 20) -> str:
    filled = int(round(conf * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {conf:.0%}"


def _flag_icon(flag: str) -> str:
    return {
        "R&D_INTENSIFYING":          "📈",
        "CAPEX_REINVESTMENT_STRONG": "🏗 ",
        "CAPEX_LIGHT_BUSINESS":      "☁ ",
        "INSIDER_CONVICTION_HIGH":   "👤",
        "INST_CONCENTRATION_HIGH":   "🏦",
    }.get(flag, "⚑ ")


def _pct(v, decimals: int = 1) -> str:
    return f"{v * 100:.{decimals}f}%" if v is not None else "N/A"


def _slope(v) -> str:
    return f"{v:+.5f}/Q" if v is not None else "N/A"


def _cagr(v) -> str:
    return f"{v:+.1%}/yr" if v is not None and v != 0.0 else "N/A"


# ─── Layer 1 display ──────────────────────────────────────────────────────────

def print_layer1(l1: Layer1Output) -> None:
    period_range = (
        f"{l1.periods[0]} → {l1.periods[-1]}" if l1.periods else "N/A"
    )
    _section(f"LAYER 1 — {l1.ticker}  ({period_range})")

    # Per-quarter series
    _row("R&D / Rev (quarterly)",   "  ".join(_pct(v) for v in l1.rd_rev))
    _row("Capex / Rev (quarterly)", "  ".join(_pct(v) for v in l1.capex_rev))
    _divider()

    # Summary statistics
    _row("R&D/Rev — current",    _pct(l1.rd_rev_level))
    _row("R&D/Rev — slope",      _slope(l1.rd_rev_slope))
    _row("R&D/Rev — CAGR",       _cagr(l1.rd_rev_cagr))
    _blank_row()
    _row("Capex/Rev — current",  _pct(l1.capex_rev_level))
    _row("Capex/Rev — slope",    _slope(l1.capex_rev_slope))
    _row("Capex/Rev — CAGR",     _cagr(l1.capex_rev_cagr))
    _divider()

    # Time-series analytics
    def _r2(v) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    def _cv(v) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    def _pct_rank(v) -> str:
        if v is None:
            return "N/A"
        pct = int(round(v * 100))
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        return f"[{bar}] {pct}th pct"

    _row("R&D/Rev — TTM",        _pct(l1.rd_rev_ttm))
    yoy_rd = f"{l1.rd_rev_yoy*100:+.2f}pp YoY" if l1.rd_rev_yoy is not None else "N/A (<5Q)"
    _row("R&D/Rev — YoY Δ",      yoy_rd)
    _row("R&D/Rev — OLS R²",     _r2(l1.rd_rev_r2) + "  (slope reliability)")
    _row("R&D/Rev — CV",         _cv(l1.rd_rev_cv) + "  (consistency; lower=steadier)")
    _row("R&D/Rev — Percentile", _pct_rank(l1.rd_rev_pct))
    _blank_row()
    _row("Capex/Rev — TTM",        _pct(l1.capex_rev_ttm))
    yoy_cx = f"{l1.capex_rev_yoy*100:+.2f}pp YoY" if l1.capex_rev_yoy is not None else "N/A (<5Q)"
    _row("Capex/Rev — YoY Δ",      yoy_cx)
    _row("Capex/Rev — OLS R²",     _r2(l1.capex_rev_r2) + "  (slope reliability)")
    _row("Capex/Rev — CV",         _cv(l1.capex_rev_cv) + "  (consistency; lower=steadier)")
    _row("Capex/Rev — Percentile", _pct_rank(l1.capex_rev_pct))
    _divider()

    # Governance signals + data quality
    _row("Insider Ownership",    _pct(l1.insider_pct))
    inst = (
        _pct(l1.institutional_top10)
        if l1.institutional_top10 is not None else "N/A"
    )
    _row("Inst. Top-10 Conc.",   inst)
    _row("Data Source",          l1.data_coverage.source)
    _row("Quarters Returned",    str(l1.data_coverage.quarters_returned))
    _divider()

    # Score + flags
    _row("Layer 1 Score", _score_bar(l1.l1_score))
    _blank_row()
    if l1.flags:
        for flag in l1.flags:
            print(f"║    {_flag_icon(flag)}  {flag:<{W-12}}║")
    else:
        _row("Flags", "(none)")

    _section_end()


# ─── Verdict display (Layer 2 + fusion) ───────────────────────────────────────

_THEME_LABELS = {
    "tech":       "Tech Adoption",
    "capacity":   "Capacity / Ops",
    "esg":        "ESG",
    "governance": "Governance",
}


def print_verdict(v: FinalOutputSchema) -> None:
    _section(f"CAPABILITY VERDICT — {v.ticker}")

    # Fused score breakdown
    _row("Overall Score",  _score_bar(v.overall.score) + f"  ← {v.overall.verdict}")
    _row("  L1 (60%)",     _score_bar(v.overall.l1_score))
    _row("  L2 (40%)",     _score_bar(v.overall.l2_score))
    _row("Confidence",     _conf_bar(v.overall.confidence))
    _divider()

    # Per-theme Layer 2 judgements
    for theme_key, label in _THEME_LABELS.items():
        t = v.themes.get(theme_key)
        if t is None:
            continue
        conf_str = f"conf={t.confidence:.2f}"
        _row(label, _score_bar(t.score) + f"  {conf_str}")
        _wrap_row("", t.rationale, lw=4)
        # Show up to 2 evidence quotes per theme
        for ev in (t.evidence_used or [])[:2]:
            _wrap_row("", f'"{ev}"', lw=6)
        _blank_row()

    _divider()

    # Low-evidence themes
    if v.low_evidence_themes:
        low_str = ", ".join(v.low_evidence_themes)
        print(f"║  {'⚠  Low-Evidence Themes (excluded from L2 avg):':<{W-4}}║")
        _row("", low_str)
        _blank_row()

    # Guardrail notes
    if v.guardrail_notes:
        print(f"║  {'⚠  Guardrail Notes':<{W-4}}║")
        _bullet_rows(v.guardrail_notes)
        _blank_row()

    _section_end()


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def _run_pipeline(
    ticker: str,
    n_quarters: int = DEFAULT_QUARTERS,
) -> tuple[FinalOutputSchema, Layer1Output]:
    """
    Run the full pipeline and return (validated_schema, l1_output).
    Internal — both the CLI and run() call this.
    """
    print(f"\n{'='*60}")
    print(f"  Agent 4 — Capability Stack  |  {ticker.upper()}")
    print(f"{'='*60}")

    print("\n[1/4] Fetching inputs...")
    bundle = fetch_inputs(ticker.upper(), n_quarters)

    print("\n[2/4] Running Layer 1 (deterministic)...")
    l1 = run_layer1(bundle)
    print(f"  L1 score: {l1.l1_score:.1f}/10  |  flags: {l1.flags or '(none)'}")

    print("\n[3/4] Running Layer 2 (LLM interpretation)...")
    l2 = run_layer2(ticker.upper(), l1, esg=bundle.esg)

    print("\n[4/4] Fusing + applying guardrail...")
    fusion_result = fuse(l1, l2)
    validated = validate_output(fusion_result)

    return validated, l1


def run(ticker: str, n_quarters: int = DEFAULT_QUARTERS) -> dict:
    """
    Full Agent 4 pipeline for the given ticker.

    Returns:
        Validated dict matching FinalOutputSchema. Suitable for JSON serialisation.

    Raises:
        ValueError:  if no financial data is available for the ticker.
        pydantic.ValidationError: if fused output violates schema constraints.
    """
    validated, _ = _run_pipeline(ticker, n_quarters)
    return validated.model_dump()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Agent 4 — Capability Stack: assess whether a company "
                    "has the internal engine to keep executing."
    )
    parser.add_argument("ticker", help="Company ticker (e.g. AAPL)")
    parser.add_argument(
        "--quarters", type=int, default=DEFAULT_QUARTERS,
        help=f"Quarters of history to fetch (default: {DEFAULT_QUARTERS})"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Write JSON output to this file path"
    )
    args = parser.parse_args()

    validated, l1 = _run_pipeline(args.ticker, n_quarters=args.quarters)

    print()
    print_layer1(l1)
    print_verdict(validated)

    if args.out:
        output_json = json.dumps(validated.model_dump(), indent=2, default=str)
        Path(args.out).write_text(output_json)
        print(f"\nOutput written to {args.out}")


if __name__ == "__main__":
    _cli()
