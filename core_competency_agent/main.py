"""
Agent 6 — Core Competency (Moat Analysis)
Interactive mode: python main.py
One-shot mode:    python main.py "Does AAPL have a durable moat?"
"""
import sys
from agent.run import run
from scoring.final_score import score_to_label

W = 72
_line  = "═" * (W - 2)
_dline = "─" * (W - 2)
_blank = " " * (W - 2)

BANNER = f"""
╔{_line}╗
║{"Agent 6 — Core Competency Analyser".center(W-2)}║
║{_blank}║
║{"Examples:".center(W-2)}║
║{"Does AAPL have a durable moat?".center(W-2)}║
║{"What is MSFT's competitive advantage?".center(W-2)}║
║{"Is NVDA's moat sustainable or just a good cycle?".center(W-2)}║
║{_blank}║
║{"Type  exit  or  quit  to stop.".center(W-2)}║
╚{_line}╝"""


# ── Formatting helpers ────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n╔{_line}╗")
    print(f"║  {title:<{W-4}}║")
    print(f"╠{_line}╣")

def section_end():
    print(f"╚{_line}╝")

def divider():
    print(f"╠{_dline}╣")

def row(label: str, value: str, lw: int = 26):
    print(f"║  {label:<{lw}}{value:<{W-lw-4}}║")

def blank():
    print(f"║{_blank}║")

def wrap_row(label: str, text: str, lw: int = 26):
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
    for i, line in enumerate(lines):
        lbl = label if i == 0 else ""
        print(f"║  {lbl:<{lw}}{line:<{content_w}}║")

def bullet_rows(items: list[str], indent: int = 4):
    for item in items:
        content_w = W - indent - 2
        wrapped = _wrap(item, content_w)
        for i, line in enumerate(wrapped):
            prefix = "•  " if i == 0 else "   "
            print(f"║{' '*indent}{prefix}{line:<{content_w-3}}║")

def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], []
    for w in words:
        if sum(len(x) + 1 for x in cur) + len(w) > width:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]

def score_bar(score: float, max_score: float = 10.0, width: int = 20) -> str:
    filled = int(round((score / max_score) * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.1f}/{max_score:.0f}"

def moat_bar(score: float) -> str:
    return score_bar(score, max_score=100, width=20)

def flag_icon(flag: str) -> str:
    icons = {
        "MARGIN_PREMIUM_SUSTAINED": "✅",
        "OP_MARGIN_PREMIUM":        "✅",
        "ROIC_ELITE":               "✅",
        "FCF_YIELD_STRONG":         "✅",
        "INSIDER_CONVICTION_HIGH":  "✅",
        "MARGIN_VOLATILE":          "⚠️ ",
        "ROIC_BELOW_PEERS":         "🔴",
    }
    return icons.get(flag, "⚑ ")

def direction_icon(d: str) -> str:
    return {
        "strengthening": "📈 STRENGTHENING",
        "stable":        "➡️  STABLE",
        "eroding":       "📉 ERODING",
    }.get(d, d.upper())

def nvn_icon(nvn: str) -> str:
    return {
        "consistent":        "✅  CONSISTENT",
        "conflict":          "⚠️   CONFLICT",
        "insufficient_data": "⚠️   INSUFFICIENT DATA",
    }.get(nvn, nvn.upper())


# ── Layer 1 display ───────────────────────────────────────────────────────────

def print_layer1(l1, computed):
    section(f"LAYER 1 — {l1.ticker}  ({l1.periods[0]} → {l1.periods[-1]})")

    gm  = [f"{v*100:.1f}%" for v in l1.gross_margin_series]
    opm = [f"{v*100:.1f}%" for v in l1.op_margin_series]
    spr = [f"{v*100:+.1f}pp" for v in l1.gross_margin_spread]

    row("Gross Margin Series",  "  ".join(gm))
    row("Op Margin Series",     "  ".join(opm))
    row("GM Spread vs Peers",   "  ".join(spr))
    divider()

    row("Avg GM Spread",    f"{l1.avg_gross_margin_spread*100:+.1f}pp  (vs peer median {l1.gross_margin_peer_median*100:.1f}%)")
    row("Avg OpM Spread",   f"{l1.avg_op_margin_spread*100:+.1f}pp  (vs peer median {l1.op_margin_peer_median*100:.1f}%)")
    divider()

    roic_str = f"{l1.roic_company*100:.1f}%" if l1.roic_company is not None else "N/A"
    roic_pm  = f"{l1.roic_peer_median*100:.1f}%" if l1.roic_peer_median is not None else "N/A"
    spr_str  = f"{l1.roic_spread*100:+.1f}pp" if l1.roic_spread is not None else "N/A"
    row("ROIC (TTM)",         f"{roic_str}  (peer median {roic_pm}, spread {spr_str})")

    fcf_spr  = f"{l1.avg_fcf_margin_spread*100:+.1f}pp" if l1.avg_fcf_margin_spread is not None else "N/A"
    row("FCF Margin Spread",  fcf_spr)
    divider()

    row("Gross Margin CV",   f"{l1.gross_margin_cv:.3f}  (lower = more stable)")
    row("Op Margin CV",      f"{l1.op_margin_cv:.3f}")
    divider()

    ins_str  = f"{l1.insider_ownership_pct*100:.1f}%" if l1.insider_ownership_pct is not None else "N/A"
    row("Insider Ownership", ins_str)
    if l1.leadership_change_detected:
        row("Leadership Change", f"⚠ {l1.leadership_change_description or 'Detected'}")
    divider()

    row("Layer 1 Score",   score_bar(l1.score))
    blank()

    if l1.flags:
        for flag in l1.flags:
            print(f"║    {flag_icon(flag)}  {flag:<{W-12}}║")
    else:
        row("Flags", "None")

    section_end()


# ── Verdict display ───────────────────────────────────────────────────────────

def print_verdict(v):
    label = score_to_label(v.moat_score)
    section(f"MOAT VERDICT — {v.ticker}  ({v.period})")

    row("Moat Score",    moat_bar(v.moat_score) + f"  ← {label}")
    row("  L1 (55%)",   score_bar(v.layer1_score))
    row("  L2 (45%)",   score_bar(v.layer2_score))
    row("Direction",     direction_icon(v.direction))
    row("Narrative",     nvn_icon(v.narrative_vs_numbers))
    divider()

    if v.key_sources:
        print(f"║  {'Key Moat Sources':<{W-4}}║")
        bullet_rows(v.key_sources)
        blank()

    if v.key_threats:
        print(f"║  {'Key Threats':<{W-4}}║")
        bullet_rows(v.key_threats)
        blank()

    divider()

    if v.bull_case:
        wrap_row("Bull Case", v.bull_case)
        blank()
    if v.bear_case:
        wrap_row("Bear Case", v.bear_case)
        blank()

    divider()
    wrap_row("Reasoning", v.reasoning)

    if v.conflict_description:
        blank()
        divider()
        wrap_row("⚠ Conflict", v.conflict_description)

    if v.sources_cited:
        divider()
        row("Sources cited", ", ".join(v.sources_cited))

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
            verdict, l1, computed = run(query)
            print()
            print_layer1(l1, computed)
            print_verdict(verdict)
        except Exception as e:
            print(f"\n  ✗ Error: {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        verdict, l1, computed = run(query)
        print()
        print_layer1(l1, computed)
        print_verdict(verdict)
    else:
        interactive_loop()
