"""
Chat loop and output renderer for Agent 7.

Entry point: run_chat()
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.query_parser import parse_ticker
from agent.run import run_agent

# ── ANSI colours (gracefully disabled if terminal doesn't support them) ───────
_USE_COLOUR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")
def green(t):  return _c(t, "32")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def cyan(t):   return _c(t, "36")
def blue(t):   return _c(t, "34")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _direction_colour(d: str) -> str:
    return {"SUPPORTIVE": green, "MIXED": yellow, "HOSTILE": red}.get(d, lambda x: x)(d)


def _bar(val: float | None, lo: float = 0, hi: float = 100, width: int = 20) -> str:
    """Simple ASCII progress bar."""
    if val is None:
        return dim("─" * width + "  n/a")
    pct = max(0, min(1, (val - lo) / (hi - lo)))
    filled = round(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {val:.1f}"


def _fmt_pct(val: float | None, suffix: str = "%") -> str:
    if val is None:
        return dim("n/a")
    sign = "+" if val > 0 else ""
    s = f"{sign}{val*100:.1f}{suffix}" if suffix == "%" else f"{sign}{val:.2f}"
    return green(s) if val > 0 else red(s) if val < 0 else s


def _fmt_score(val: float | None) -> str:
    if val is None:
        return dim("n/a")
    if val >= 70:   return green(f"{val:.0f}")
    elif val >= 30: return yellow(f"{val:.0f}")
    else:           return red(f"{val:.0f}")


def _divider(char: str = "─", width: int = 60) -> str:
    return dim(char * width)


# ── Main renderer ─────────────────────────────────────────────────────────────
def render_result(result: dict) -> None:
    ticker   = result.get("ticker", "?")
    score    = result.get("environment_score")
    dirn     = result.get("direction", "MIXED")
    qs       = result.get("quant_score")
    qual     = result.get("qual_score")
    flags    = result.get("flags", [])
    narrative = result.get("narrative", "")
    bundle   = result.get("layer1_bundle", {})
    evidence = result.get("evidence", {})

    print()
    print(_divider("═", 60))
    dir_str = _direction_colour(dirn)
    score_str = _fmt_score(score)
    print(bold(f"  {ticker}  ·  {dir_str}  ·  {score_str} / 100"))
    print(_divider("═", 60))

    # ── LAYER 1 — QUANTITATIVE ────────────────────────────────────────────────
    print()
    print(bold(cyan("  LAYER 1 — QUANTITATIVE")))
    print(_divider())

    def row(label: str, val: str, pad: int = 20):
        print(f"  {label:<{pad}}  {val}")

    # Returns
    co_ret  = bundle.get("company_cum_return_6m")
    sec_ret = bundle.get("sector_cum_return_6m")
    row("Returns (6m)",
        f"company {_fmt_pct(co_ret)}  ·  sector {_fmt_pct(sec_ret)}")

    # Relative strength
    rs = bundle.get("sector_rs_6m")
    rs_label = "LEADING" if rs and rs >= 1.03 else "LAGGING" if rs and rs <= 0.97 else "IN-LINE"
    rs_col = (green if rs_label == "LEADING" else red if rs_label == "LAGGING" else dim)(rs_label)
    row("Relative Strength", f"{rs_col}  (rs={rs:.3f})" if rs else dim("n/a"))

    # Alpha / Beta
    alpha = bundle.get("company_alpha_annualised")
    beta  = bundle.get("company_beta")
    alpha_str = _fmt_pct(alpha) if alpha is not None else dim("n/a")
    beta_str  = f"{beta:.2f}" if beta is not None else dim("n/a")
    row("Alpha / Beta", f"α={alpha_str}  β={beta_str}")

    # Rate beta
    beta_rate = bundle.get("beta_rate")
    row("Rate Beta", f"{beta_rate:.2f}" if beta_rate is not None else dim("n/a"))

    # Volatility regime
    vol_regime = bundle.get("vol_regime", dim("n/a"))
    vix_z      = bundle.get("vix_zscore")
    vix_str    = f"  (z={vix_z:.2f})" if vix_z is not None else ""
    row("Vol Regime", f"{vol_regime}{vix_str}")

    # Rate regime
    rate_regime = bundle.get("rate_regime", dim("n/a"))
    slope_z     = bundle.get("rate_slope_z")
    rate_str    = f"  (slope z={slope_z:.2f})" if slope_z is not None else ""
    row("Rate Regime", f"{rate_regime}{rate_str}")

    # Market trend
    trend = bundle.get("market_trend", dim("n/a"))
    row("Market Trend", str(trend))

    # Commodity
    comm_tag = bundle.get("commodity_tag", dim("n/a"))
    row("Commodity", str(comm_tag))

    # Peer gaps
    rev_gap = bundle.get("peer_rev_growth_gap")
    mar_gap = bundle.get("peer_margin_gap")
    row("Peer Rev Growth Gap", _fmt_pct(rev_gap, "pp") if rev_gap is not None else dim("n/a"))
    row("Peer Margin Gap",     _fmt_pct(mar_gap, "pp") if mar_gap is not None else dim("n/a"))

    print()
    print(f"  {'Quant Score':<20}  {_bar(qs)}  /100")

    if flags:
        print()
        flag_str = "  ".join(yellow(f) for f in flags)
        print(f"  {'Flags':<20}  {flag_str}")

    # ── LAYER 2 — QUALITATIVE ─────────────────────────────────────────────────
    print()
    print(bold(cyan("  LAYER 2 — QUALITATIVE")))
    print(_divider())

    # news_by_pestel_dim: {P: [...], E: [...], ...}  — flatten for display
    news_by_dim = evidence.get("news_by_pestel_dim", {})
    all_news = [art for articles in news_by_dim.values() for art in articles]
    excerpts = evidence.get("risk_factor_excerpts", [])

    if all_news:
        print(f"  {'News':<20}  {len(all_news)} article(s) retrieved ({len(news_by_dim)} PESTEL dims)")
        for i, n in enumerate(all_news[:3], 1):
            title = n.get("title", "")[:70]
            published = n.get("published_at", "")[:10]
            print(f"  {'':<20}  [{i}] {dim(title)}  {dim(published)}")
    else:
        print(f"  {'News':<20}  {dim('none retrieved')}")

    print()
    if excerpts:
        print(f"  {'10-K Excerpts':<20}  {len(excerpts)} chunk(s) retrieved from Neon")
        for i, ex in enumerate(excerpts[:2], 1):
            snippet = ex[:120].replace("\n", " ")
            print(f"  {'':<20}  [{i}] {dim(snippet)}...")
    else:
        print(f"  {'10-K Excerpts':<20}  {dim('none — no filing in DB for this ticker/year')}")

    print()
    print(f"  {'Qual Score':<20}  {_bar(qual)}  /100")

    # ── NARRATIVE ─────────────────────────────────────────────────────────────
    print()
    print(bold(cyan("  NARRATIVE")))
    print(_divider())

    def _wrap(text: str, width: int = 70) -> list[str]:
        """Word-wrap a string to lines of at most `width` chars."""
        words = text.split()
        line, lines = [], []
        for w in words:
            line.append(w)
            if sum(len(x) + 1 for x in line) > width:
                lines.append(" ".join(line[:-1]))
                line = [w]
        if line:
            lines.append(" ".join(line))
        return lines

    # Per-dimension paragraphs (preferred)
    _DIM_LABELS = [
        ("Political",      "P"),
        ("Economic",       "E"),
        ("Social",         "S"),
        ("Technological",  "T"),
        ("Environmental",  "En"),
        ("Legal",          "L"),
    ]
    narrative_by_dim = result.get("narrative_by_dim", {})
    pestel_scores    = result.get("pestel_scores", {})

    if narrative_by_dim and any(narrative_by_dim.values()):
        for label, key in _DIM_LABELS:
            text = (narrative_by_dim.get(label) or "").strip()
            if not text:
                continue
            # Score badge next to label
            dim_score = pestel_scores.get(key, {}).get("combined")
            badge = f"  {dim_score:.0f}/100" if dim_score is not None else ""
            print(f"  {bold(label)}{dim(badge)}")
            for l in _wrap(text):
                print(f"    {l}")
            print()
    elif narrative:
        # Fallback: single overall paragraph
        for l in _wrap(narrative):
            print(f"  {l}")
    else:
        print(f"  {dim('No narrative generated.')}")

    # Tailwinds / risks
    tailwinds = result.get("key_tailwinds", [])
    risks     = result.get("key_risks", [])
    if tailwinds:
        print()
        print(f"  {bold('Tailwinds')}")
        for tw in tailwinds:
            print(f"  {green('▲')}  {tw}")
    if risks:
        print()
        print(f"  {bold('Risks')}")
        for rk in risks:
            print(f"  {red('▼')}  {rk}")

    print()
    print(_divider("─", 60))
    print(dim(f"  Run ID: {result.get('run_id', '?')}  ·  As-of: {result.get('as_of_date', '?')}"))
    print()


# ── Progress printer ──────────────────────────────────────────────────────────
def _step(n: int, total: int, msg: str) -> None:
    print(f"  {dim(f'[{n}/{total}]')} {msg}")


# ── Chat loop ─────────────────────────────────────────────────────────────────
def run_chat(lookback_days: int = 126) -> None:
    print()
    print(bold("  Agent 7 — Company Environment  ") + dim("(type 'exit' to quit)"))
    print(_divider())
    print(dim("  Ask anything: 'analyse AAPL', 'what's the environment for Microsoft?'"))
    print(dim("  'compare NVDA and AMD' (runs first ticker), 'run Tesla today'"))
    print()

    while True:
        try:
            query = input(bold("  > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(dim("  Goodbye."))
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q", "bye"}:
            print(dim("  Goodbye."))
            break

        # ── Step 1: Parse ticker
        _step(1, 4, "Parsing query...")
        ticker = parse_ticker(query)
        if not ticker:
            print(red("  Could not identify a ticker. Try: 'analyse AAPL' or 'run Apple'."))
            print()
            continue
        print(f"       → {bold(ticker)}")

        # ── Step 2: Fetch data (yfinance is live, no pre-ingestion needed)
        _step(2, 4, f"Fetching prices & fundamentals for {ticker}...")

        # ── Step 3: Layer 1
        _step(3, 4, "Running Layer 1 computations (returns, regimes, peer gaps)...")

        # ── Step 4: Layer 2
        _step(4, 4, "Running Layer 2 (news + 10-K retrieval + LLM)...")

        try:
            result = run_agent(
                ticker=ticker,
                as_of_date=date.today(),
                lookback_days=lookback_days,
            )
            render_result(result)
        except Exception as e:
            print()
            print(red(f"  Error running agent for {ticker}: {e}"))
            import traceback
            print(dim(traceback.format_exc()))
            print()
