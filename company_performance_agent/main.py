"""
Agent 10 — Company Performance Agent
Interactive mode: python main.py
One-shot mode:    python main.py "Analyse MSFT"
"""
import sys
from agent.orchestrator import run

W = 72   # display width

_line = '═' * (W-2)
_blank = ' ' * (W-2)
BANNER = f"""
╔{_line}╗
║{'Agent 10 — Company Performance Analyser'.center(W-2)}║
║{_blank}║
║{'Examples:'.center(W-2)}║
║{'Analyse MSFT business execution over the last year'.center(W-2)}║
║{'Is AAPL showing positive operating leverage?'.center(W-2)}║
║{'Did NVDA management narrative match their results?'.center(W-2)}║
║{_blank}║
║{'Type  exit  or  quit  to stop.'.center(W-2)}║
╚{_line}╝"""


# ── Formatting helpers ────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n╔{'═' * (W-2)}╗")
    print(f"║  {title:<{W-4}}║")
    print(f"╠{'═' * (W-2)}╣")

def section_end():
    print(f"╚{'═' * (W-2)}╝")

def row(label: str, value: str, width: int = 22):
    print(f"║  {label:<{width}}{value:<{W-width-4}}║")

def divider():
    print(f"╠{'─' * (W-2)}╣")

def blank():
    print(f"║{' ' * (W-2)}║")

def wrap_row(label: str, text: str, label_width: int = 22):
    """Print a label + wrapped text inside the box."""
    words = text.split()
    content_w = W - label_width - 4
    lines, cur = [], []
    for w in words:
        if sum(len(x)+1 for x in cur) + len(w) > content_w:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    for i, line in enumerate(lines):
        lbl = label if i == 0 else ""
        print(f"║  {lbl:<{label_width}}{line:<{content_w}}║")

def fmt_pct(v, decimals=1):
    if v is None: return "  N/A  "
    return f"{v:+.{decimals}f}%"

def fmt_val(v, decimals=1):
    if v is None: return "  N/A  "
    return f"{v:.{decimals}f}"

def fmt_money(v):
    """Format large numbers as $B or $M."""
    if v is None: return "N/A"
    b = abs(v) / 1e9
    if b >= 1:
        return f"${v/1e9:+.1f}B"
    return f"${v/1e6:+.1f}M"

def flag_emoji(flag: str) -> str:
    icons = {
        "OP_LEVERAGE_POSITIVE":    "✅",
        "OP_LEVERAGE_DETERIORATING": "⚠️",
        "MARGIN_EXPANDING":        "📈",
        "MARGIN_COMPRESSING":      "📉",
        "GROSS_MARGIN_PRESSURE":   "⚠️",
        "OPEX_OVERHEAD_BLOAT":     "🔴",
        "REV_ACCELERATING":        "🚀",
        "REV_DECELERATING":        "📉",
        "FCF_WEAK":                "⚠️",
        "FCF_NEGATIVE":            "🔴",
        "DSO_RISING":              "⚠️",
    }
    return icons.get(flag, "⚑ ")

def direction_icon(d: str) -> str:
    return {"improving": "📈 IMPROVING", "stable": "➡️  STABLE", "deteriorating": "📉 DETERIORATING"}.get(d, d.upper())

def verdict_icon(v: str) -> str:
    return {
        "narrative_credible":     "✅  NARRATIVE CREDIBLE",
        "narrative_not_credible": "❌  NARRATIVE NOT CREDIBLE",
        "insufficient_data":      "⚠️   INSUFFICIENT DATA",
    }.get(v, v.upper())

def score_bar(score: float, max_score: float = 10, width: int = 20) -> str:
    filled = int(round((score / max_score) * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.2f}/10"


# ── Layer 1 table ─────────────────────────────────────────────────────────────

def print_layer1(l1, computed):
    periods = l1.periods
    # Show last 4 periods for the YoY columns (they have fewer entries)
    yoy_periods = periods[-4:] if len(periods) >= 4 else periods

    section(f"LAYER 1 METRICS — {l1.ticker}  ({periods[0]} → {periods[-1]})")

    # ── Revenue ──────────────────────────────────────────────────────────────
    COL = 10
    header = f"  {'Metric':<26}" + "".join(f"{p[-7:]:<{COL}}" for p in yoy_periods)
    print(f"║{header:<{W-2}}║")
    print(f"╠{'─' * (W-2)}╣")

    def metric_row(label, values, fmt_fn):
        vals = "".join(f"{fmt_fn(v):<{COL}}" for v in values)
        print(f"║  {label:<26}{vals:<{(W-2)-28}}║")

    metric_row("Revenue YoY",    l1.rev_yoy_pct,  fmt_pct)
    metric_row("OpEx YoY",       l1.opex_yoy_pct, fmt_pct)
    metric_row("OL Delta (pp)",  l1.ol_delta,     lambda v: f"{v:+.1f}pp")
    divider()

    # Margins — use last 4 quarters of series
    gm  = computed["gross_margin_series"][-4:]
    opm = computed["op_margin_series"][-4:]
    nm  = computed["net_margin_series"][-4:]
    metric_row("Gross Margin",   [v*100 for v in gm],  lambda v: f"{v:.1f}%")
    metric_row("Op Margin",      [v*100 for v in opm], lambda v: f"{v:.1f}%")
    metric_row("Net Margin",     [v*100 for v in nm],  lambda v: f"{v:.1f}%")
    divider()

    # FCF
    fcf_ni = computed["fcf_ni_ratio"][-4:]
    fcf    = computed["fcf"][-4:]
    metric_row("FCF/NI Ratio",   fcf_ni, lambda v: f"{v:.2f}x")
    metric_row("FCF",            fcf,    lambda v: fmt_money(v))

    divider()

    # Slopes + summary
    def slope_dir(s): return "↑" if s > 0 else ("↓" if s < 0 else "→")

    row("Rev Slope (M/Q)",   f"{slope_dir(l1.rev_slope)} {l1.rev_slope:+.1f}")
    row("Op Margin Slope",   f"{slope_dir(l1.op_margin_slope)} {l1.op_margin_slope:+.4f}/Q")
    row("Gross Margin Slope",f"{slope_dir(l1.gross_margin_slope)} {l1.gross_margin_slope:+.4f}/Q")
    row("OL Slope",          f"{slope_dir(l1.ol_slope)} {l1.ol_slope:+.4f}/Q")
    row("OL Consistency",    f"{l1.ol_consistency:.0%} of quarters positive")
    if l1.ccc_delta is not None:
        row("CCC Delta",     f"{l1.ccc_delta:+.1f} days")
    divider()

    row("Layer 1 Score",     score_bar(l1.score))
    blank()

    if l1.flags:
        print(f"║  {'Flags':<{W-4}}║")
        for flag in l1.flags:
            icon = flag_emoji(flag)
            print(f"║    {icon}  {flag:<{W-10}}║")
    else:
        row("Flags", "None")

    section_end()


# ── RAG sources table ─────────────────────────────────────────────────────────

def print_rag_sources(rag):
    section(f"RAG SOURCES  ({'enabled' if rag.rag_enabled else 'disabled'} | {len(rag.passages)} passages | credibility: {rag.credibility_track_record:.0%})")

    if not rag.rag_enabled:
        row("", "RAG not enabled — set RAG_ENABLED=true in .env")
        section_end()
        return

    if not rag.passages:
        row("", "No passages retrieved from Neon")
        section_end()
        return

    # Header
    print(f"║  {'#':<3}{'Source':<18}{'Period':<12}{'Section':<16}{'Score':<8}Preview{'':>{W-63}}║")
    print(f"╠{'─' * (W-2)}╣")

    for i, p in enumerate(rag.passages, 1):
        source  = (p.source_type or "")[:16]
        period  = (p.period or "")[:10]
        section_= (p.section or "")[:14]
        score   = f"{p.similarity_score:.2f}"
        preview = (p.text or "")[:28].replace("\n", " ").strip() + "…"
        line    = f"  {i:<3}{source:<18}{period:<12}{section_:<16}{score:<8}{preview}"
        print(f"║{line:<{W-2}}║")

        # Show speaker if available
        if p.speaker:
            print(f"║{'':4}{'Speaker: ' + p.speaker:<{W-6}}║")

    if rag.guidance_matches:
        divider()
        print(f"║  {'Guidance Track Record':<{W-4}}║")
        for g in rag.guidance_matches:
            gline = f"  {g.made_in_period}: \"{g.guided_value}\" → {g.outcome.upper()}"
            print(f"║{gline:<{W-2}}║")

    section_end()


# ── Final verdict ─────────────────────────────────────────────────────────────

def print_verdict(verdict):
    section(f"FINAL VERDICT — {verdict.ticker}  ({verdict.period})")

    row("Direction",        direction_icon(verdict.direction))
    row("Verdict",          verdict_icon(verdict.verdict))
    divider()
    row("Execution Score",  score_bar(verdict.execution_score))
    row("  └ Layer1 base",  score_bar(verdict.layer1_score))
    row("Credibility Score",score_bar(verdict.credibility_score))
    divider()

    blank()
    wrap_row("LLM Interpretation", verdict.reasoning)
    blank()

    if verdict.key_contradiction:
        divider()
        wrap_row("⚠  Key Contradiction", verdict.key_contradiction)

    if verdict.counterargument:
        divider()
        wrap_row("🔄 Counterargument", verdict.counterargument)

    if verdict.sources_cited:
        divider()
        row("Sources cited", ", ".join(verdict.sources_cited))

    section_end()


# ── Interactive loop ──────────────────────────────────────────────────────────

def interactive_loop():
    print(BANNER)
    while True:
        try:
            query = input(f"\n{'─'*W}\nQuery > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        try:
            verdict, l1, rag, computed = run(query)
            print()
            print_layer1(l1, computed)
            print_rag_sources(rag)
            print_verdict(verdict)
        except Exception as e:
            print(f"\n  ✗ Error: {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        verdict, l1, rag, computed = run(query)
        print()
        print_layer1(l1, computed)
        print_rag_sources(rag)
        print_verdict(verdict)
    else:
        interactive_loop()
