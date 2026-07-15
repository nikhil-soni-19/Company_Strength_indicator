"""LLM-backed natural-language interpretation of liquidity scores.

Three public entry points:

* :func:`interpret` — single-ticker. If a ``question`` is provided, the LLM
  answers it plainly in natural language using only the computed metrics.
  If no question is provided, the LLM writes a general qualitative analyst
  paragraph.
* :func:`interpret_comparison` — multi-ticker. Ranks and compares the
  liquidity profile of several stocks in a single paragraph. Also accepts
  an optional ``question`` so the comparison can directly answer the
  trader.

In all cases, if ``OPENAI_API_KEY`` is missing or the LLM call fails, we
fall back to a deterministic paragraph that still references the actual
numbers — so the agent is fully runnable offline.

Display policy
--------------
Amihud price-impact ratio and DTL (days-to-liquidate) / exit-risk figures
are **not volunteered** in any output panel — Summary, LLM Interpretation,
or Analyst Interpretation — unless the user's question explicitly asks about
price impact, exit speed, or liquidation time.  The LLM system prompts
enforce this; the deterministic fallback uses keyword detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

from src.output.confidence import ConfidenceReport
from src.scoring.engine import LiquidityScore


@dataclass
class Interpretation:
    paragraph: str
    used_llm: bool
    model: Optional[str] = None
    question: Optional[str] = None


# ── LLM system prompts ────────────────────────────────────────────────────────

_ANALYST_PROMPT = (
    "You are a senior buy-side liquidity analyst writing a short qualitative "
    "commentary paragraph for an execution trader.\n\n"
    "The trader already has a table of the raw numbers in front of them. "
    "Your job is NOT to repeat those numbers — it is to explain what they "
    "mean in plain English: the character of the stock's liquidity, what "
    "structural forces are at play, and what a trader should actually do.\n\n"
    "Hard rules:\n"
    "- Ground your commentary in the data provided but translate it into "
    "qualitative insight, not a recitation of figures.\n"
    "- Mention the tier by name so the trader knows the risk classification. "
    "Do not quote the raw score or repeat metric values.\n"
    "- Write ONE paragraph, 4 to 6 sentences. No bullet points, no headings.\n"
    "- Cover: (1) what kind of stock this is from a liquidity standpoint — "
    "deep and institutional, workable but demanding, or thin and fragile; "
    "(2) what the ownership and short-interest structure signals about "
    "potential volatility or crowding risk; (3) if the short squeeze risk "
    "override is triggered, explain the danger in plain terms — give it "
    "prominent weight; (4) what execution approach the tier calls for.\n"
    "- Mention Volume CV only if it is in the Critical band — frame it as "
    "'erratic trading patterns' not as a number.\n"
    "- If the buyback inflation flag is triggered, mention it plainly: the "
    "reported ADV$ is artificially supported and may not reflect true "
    "third-party liquidity.\n"
    "- Do NOT mention Amihud unless the trader asks.\n"
    "- Tone: conversational but professional — like a trusted colleague "
    "giving a straight read, not a compliance report. No hype, no disclaimers."
)

_QUESTION_PROMPT = (
    "You are a senior buy-side liquidity analyst answering an execution "
    "trader's question in plain, conversational English.\n\n"
    "The trader already sees the raw numbers in a table. Your job is to give "
    "them the qualitative answer — what the data means, not what it says.\n\n"
    "Hard rules:\n"
    "- Ground your answer in the provided data, but express it as insight "
    "rather than a data readout. Avoid repeating metric values.\n"
    "- Answer the question directly and concisely in 3 to 5 sentences.\n"
    "- Name the tier so the trader knows the risk classification.\n"
    "- If the question cannot be answered from the available data, say so "
    "briefly and redirect to what the data does show.\n"
    "- Mention Volume CV only if it is Critical — frame it qualitatively.\n"
    "- Do NOT discuss Amihud unless the question explicitly asks about price impact.\n"
    "- If the short squeeze risk override is triggered, flag the danger "
    "clearly in plain language.\n"
    "- If the buyback inflation flag is triggered, note that the ADV$ is "
    "artificially supported by the company's own repurchase programme.\n"
    "- No bullet points, no headings, no markdown, no disclaimers.\n"
    "- Tone: direct, warm, like a knowledgeable colleague — not a report."
)

_COMPARISON_PROMPT_BASE = (
    "You are a senior buy-side liquidity analyst comparing multiple stocks "
    "for an execution trader in plain, conversational English.\n\n"
    "The trader already sees all the raw numbers. Give them the qualitative "
    "read — which stock is more tradeable, why, and what to watch out for.\n\n"
    "Hard rules:\n"
    "- Express your ranking in terms of tradability and risk character, "
    "not as a list of metric values. Mention tier names, not raw scores.\n"
    "- If any ticker triggered the short squeeze risk override, explain "
    "the structural danger in plain terms — give it prominent weight.\n"
    "- Write ONE paragraph, 5 to 8 sentences. No bullet points, no headings.\n"
    "- Mention Volume CV only if it is Critical for a ticker — frame it "
    "qualitatively as erratic or unpredictable trading patterns.\n"
    "- If any ticker has the buyback inflation flag triggered, note the "
    "ADV$ inflation risk for that name.\n"
    "- Do NOT discuss Amihud unless the question asks about price impact.\n"
    "- Tone: direct, collegial, conversational — not a compliance memo."
)


# ── Keyword sets that gate conditional metric disclosure ─────────────────────

_AMIHUD_KEYWORDS = frozenset({"amihud", "price impact", "impact", "slippage"})
_CV_KEYWORDS     = frozenset({"volume cv", "volume coefficient", "coefficient of variation",
                               "cv ", " cv", "erratic", "volume variability", "volume stability"})
_FLOAT_SH_KEYWORDS = frozenset({"float shares", "free float shares", "float size",
                                 "number of shares", "shares outstanding"})
_BUYBACK_KEYWORDS  = frozenset({"buyback", "buy back", "buy-back", "repurchase",
                                 "share repurchase", "buyback yield", "bir",
                                 "buyback intensity", "repurchase programme"})


def _asks_amihud(question: Optional[str]) -> bool:
    if not question:
        return False
    q = question.lower()
    return any(kw in q for kw in _AMIHUD_KEYWORDS)


def _asks_cv(question: Optional[str]) -> bool:
    if not question:
        return False
    q = question.lower()
    return any(kw in q for kw in _CV_KEYWORDS)


def _asks_float_shares(question: Optional[str]) -> bool:
    if not question:
        return False
    q = question.lower()
    return any(kw in q for kw in _FLOAT_SH_KEYWORDS)


def _asks_buyback(question: Optional[str]) -> bool:
    if not question:
        return False
    q = question.lower()
    return any(kw in q for kw in _BUYBACK_KEYWORDS)


def _dim_is_critical(score: LiquidityScore, dimension: str) -> bool:
    """Return True if the named scoring dimension reached the Critical band."""
    for d in score.dimension_scores:
        if d.dimension == dimension:
            return d.band == "Critical"
    return False


# ── Public entry points ───────────────────────────────────────────────────────

def interpret(
    score: LiquidityScore,
    confidence: ConfidenceReport,
    question: Optional[str] = None,
    model: Optional[str] = None,
) -> Interpretation:
    """Single-ticker interpretation. If ``question`` is given, answers it."""
    api_key = os.getenv("OPENAI_API_KEY")
    resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            paragraph = _llm_single(score, confidence, question, api_key, resolved_model)
            return Interpretation(
                paragraph=paragraph,
                used_llm=True,
                model=resolved_model,
                question=question,
            )
        except Exception as exc:  # pragma: no cover - depends on external API
            fallback = _deterministic_single(score, confidence, question)
            return Interpretation(
                paragraph=f"{fallback}\n\n[LLM unavailable: {exc!s}]",
                used_llm=False,
                question=question,
            )

    return Interpretation(
        paragraph=_deterministic_single(score, confidence, question),
        used_llm=False,
        question=question,
    )


def interpret_comparison(
    scored: Sequence[tuple[LiquidityScore, ConfidenceReport]],
    question: Optional[str] = None,
    model: Optional[str] = None,
) -> Interpretation:
    """Multi-ticker comparison interpretation."""
    if not scored:
        raise ValueError("interpret_comparison requires at least one (score, confidence) pair")

    api_key = os.getenv("OPENAI_API_KEY")
    resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            paragraph = _llm_comparison(scored, question, api_key, resolved_model)
            return Interpretation(
                paragraph=paragraph,
                used_llm=True,
                model=resolved_model,
                question=question,
            )
        except Exception as exc:  # pragma: no cover - depends on external API
            fallback = _deterministic_comparison(scored, question)
            return Interpretation(
                paragraph=f"{fallback}\n\n[LLM unavailable: {exc!s}]",
                used_llm=False,
                question=question,
            )

    return Interpretation(
        paragraph=_deterministic_comparison(scored, question),
        used_llm=False,
        question=question,
    )


# ── LLM call helpers ──────────────────────────────────────────────────────────

def _llm_single(
    score: LiquidityScore,
    confidence: ConfidenceReport,
    question: Optional[str],
    api_key: str,
    model: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    system_prompt = _QUESTION_PROMPT if question else _ANALYST_PROMPT
    context = _format_single_context(score, confidence, question)
    user_content = (
        f"Question: {question}\n\n{context}" if question else context
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.5,
    )
    return (completion.choices[0].message.content or "").strip()


def _llm_comparison(
    scored: Sequence[tuple[LiquidityScore, ConfidenceReport]],
    question: Optional[str],
    api_key: str,
    model: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    system_prompt = _COMPARISON_PROMPT_BASE
    if question:
        system_prompt += (
            "\n- A user question is provided. Answer it directly while "
            "still delivering the ranked comparison."
        )

    context = _format_comparison_context(scored, question)
    user_content = (
        f"Question: {question}\n\n{context}" if question else context
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.5,
    )
    return (completion.choices[0].message.content or "").strip()


# ── Qualitative vocabulary helpers ───────────────────────────────────────────

def _adv_character(adv: Optional[float]) -> str:
    if adv is None:
        return "liquidity depth is unknown"
    if adv >= 1e9:
        return "deep, institutional-grade liquidity with ample capacity to absorb large orders"
    if adv >= 100e6:
        return "solid liquidity with enough daily turnover to work meaningful block orders"
    if adv >= 10e6:
        return "moderate liquidity — workable for smaller institutional orders but demanding for anything larger"
    if adv >= 2e6:
        return "thin liquidity where even modest position changes can move the market"
    return "very limited liquidity — this name trades in a near-vacuum and execution risk is high"


def _short_character(pct: Optional[float]) -> str:
    if pct is None:
        return None
    if pct > 0.30:
        return (
            "short interest is heavily elevated, signalling crowded bearish positioning "
            "and meaningful short-squeeze vulnerability"
        )
    if pct > 0.15:
        return (
            "short interest is notably high, which creates the risk of sharp covering "
            "rallies that can distort the apparent liquidity"
        )
    if pct > 0.05:
        return "short interest is modest and unlikely to materially affect execution"
    return "short interest is negligible"


def _float_character(pct: Optional[float]) -> str:
    if pct is None:
        return None
    if pct > 0.85:
        return "the share structure is freely floating with broad public availability"
    if pct > 0.60:
        return "the float is reasonably accessible, though some shares are locked with insiders or strategic holders"
    if pct > 0.40:
        return "a moderately constrained float limits the effective supply available to traders"
    return "the float is tightly controlled, which amplifies price impact and squeezes tradable supply"


def _inst_character(pct: Optional[float]) -> str:
    if pct is None:
        return None
    if pct > 0.70:
        return (
            "heavy institutional concentration means order flow tends to arrive in large "
            "blocks around rebalancing events, creating episodic liquidity gaps"
        )
    if pct > 0.40:
        return (
            "significant institutional ownership can produce lumpy, unevenly distributed "
            "order flow — particularly around earnings and index events"
        )
    return "institutional ownership is moderate and unlikely to create structural flow distortions"


def _tier_narrative(tier_number: int) -> str:
    return {
        1: (
            "This is a clean, unrestricted name — standard market orders are appropriate "
            "and position sizes up to a meaningful percentage of float are manageable."
        ),
        2: (
            "This name warrants measured position sizing — limit orders or TWAP spread "
            "across multiple sessions will reduce market impact meaningfully."
        ),
        3: (
            "Full algorithmic execution is required here — VWAP with strict participation "
            "limits and compliance sign-off before any order is placed."
        ),
        4: (
            "This name is effectively off-limits. Any entry carries unacceptable execution "
            "risk — the position should be hard-blocked until conditions improve."
        ),
    }.get(tier_number, "")


# ── Deterministic fallbacks ───────────────────────────────────────────────────

def _deterministic_single(
    score: LiquidityScore,
    confidence: ConfidenceReport,
    question: Optional[str],
) -> str:
    want_amihud   = _asks_amihud(question)
    want_cv       = _asks_cv(question) or _dim_is_critical(score, "volume_cv_30d")
    want_float_sh = _asks_float_shares(question) or _dim_is_critical(score, "free_float_shares")

    parts: list[str] = []
    if question:
        parts.append(f"On your question — \"{question}\" — here is the read.")

    # Opening: tier classification + liquidity character
    adv_desc = _adv_character(score.adv.adv_dollar_30d)
    parts.append(
        f"{score.ticker} sits in {score.final_tier.badge} — "
        f"it has {adv_desc}."
    )

    # Structural narrative: float, short interest, institutional ownership
    float_desc = _float_character(score.structural.float_pct_of_outstanding)
    short_desc  = _short_character(score.structural.short_percent_float)
    inst_desc   = _inst_character(score.structural.top10_institutional_pct)

    structural_lines: list[str] = []
    if float_desc:
        structural_lines.append(float_desc)
    if short_desc:
        structural_lines.append(short_desc)
    if inst_desc:
        structural_lines.append(inst_desc)
    if structural_lines:
        parts.append(". ".join(s.capitalize() for s in structural_lines) + ".")

    # Volume behaviour — only mention CV if Critical or asked
    if want_cv:
        cv = score.volume_cv.volume_cv_30d
        if cv is not None and cv > 1.0:
            parts.append(
                "Trading patterns are erratic — volume swings wildly day to day, "
                "which makes arrival-price estimates unreliable and increases the "
                "chance of realised slippage exceeding the model."
            )

    # Free float share count context — only if Critical or explicitly asked
    if want_float_sh and _dim_is_critical(score, "free_float_shares"):
        parts.append(
            "The absolute float is critically small, meaning even a modestly sized "
            "institutional order represents a disproportionate share of the available "
            "supply — structural fragility risk is high."
        )

    # Amihud — only if user explicitly asks
    if want_amihud:
        amihud = score.amihud.amihud_30d
        if amihud is not None:
            if amihud < 0.01:
                am_desc = "price impact per dollar traded is negligible — large orders move the market very little"
            elif amihud < 0.05:
                am_desc = "moderate price sensitivity — orders of meaningful size will leave a visible footprint"
            elif amihud < 0.20:
                am_desc = "elevated price impact — even mid-sized orders can cause noticeable price dislocation"
            else:
                am_desc = "extreme price sensitivity — the stock moves sharply in response to relatively small order flow"
            parts.append(f"On price impact: {am_desc}.")

    # Short squeeze warning — most important, given heavy emphasis
    if score.mirage.triggered:
        parts.append(
            "⚠ The short squeeze risk override is active — the combination of a thin float "
            "and heavy short positioning is a classic setup for an artificial liquidity "
            "illusion. Volume may look healthy, but it can evaporate instantly when the "
            "squeeze unwinds. The tier has been downgraded to reflect this structural danger."
        )

    # Closing: what to actually do
    parts.append(_tier_narrative(score.final_tier.number))
    return " ".join(parts)


def _deterministic_comparison(
    scored: Sequence[tuple[LiquidityScore, ConfidenceReport]],
    question: Optional[str],
) -> str:
    ranked = sorted(
        scored,
        key=lambda pair: (pair[0].final_tier.number, -(pair[0].adv.adv_dollar_30d or 0)),
    )

    parts: list[str] = []
    if question:
        parts.append(f"On your question — \"{question}\" — here is the comparative read.")

    # Open with the qualitative ranking
    ranking_desc = ", ".join(
        f"{s.ticker} ({s.final_tier.badge})" for s, _ in ranked
    )
    parts.append(
        f"From most to least tradeable: {ranking_desc}."
    )

    # Characterise the top and bottom picks
    best  = ranked[0][0]
    worst = ranked[-1][0]
    if best.ticker != worst.ticker:
        best_adv  = _adv_character(best.adv.adv_dollar_30d)
        worst_adv = _adv_character(worst.adv.adv_dollar_30d)
        parts.append(
            f"{best.ticker} is the clear preference — it offers {best_adv}, "
            f"making it the natural home for larger or time-sensitive orders. "
            f"{worst.ticker}, by contrast, has {worst_adv}, so execution there "
            f"demands proportionally more patience and care."
        )

    # Short squeeze danger — prominent if any ticker is affected
    squeeze_hits = [(s, c) for s, c in scored if s.mirage.triggered]
    if squeeze_hits:
        hit_names = ", ".join(s.ticker for s, _ in squeeze_hits)
        parts.append(
            f"⚠ {hit_names} {'have' if len(squeeze_hits) > 1 else 'has'} triggered the short squeeze "
            f"risk override — a thin float combined with heavy short positioning creates an "
            f"artificial liquidity illusion that can collapse without warning. "
            f"Any volume strength in {'these names' if len(squeeze_hits) > 1 else 'this name'} "
            f"should be treated with deep scepticism."
        )

    # Closing execution guidance
    parts.append(
        "When working across these names in a basket, the most restrictive tier "
        "sets the pace — do not let the liquid leg distract from the constraints "
        "the illiquid leg imposes on overall position sizing."
    )
    return " ".join(parts)


# ── Context formatters (passed as user message to the LLM) ───────────────────

def _format_single_context(
    score: LiquidityScore,
    confidence: ConfidenceReport,
    question: Optional[str] = None,
) -> str:
    """Build the structured metrics block sent to the LLM as user content.

    All computed numbers are included so the LLM can answer any question
    accurately.  The system prompt controls which ones to volunteer
    proactively vs. only on explicit request.
    """
    lines: list[str] = [
        f"Ticker: {score.ticker}",
        f"Assigned tier: {score.final_tier.badge}",
        f"Raw risk score: {score.raw_score}",
        f"Base tier (pre-override): {score.base_tier.badge}",
        f"Short squeeze risk override triggered: {score.mirage.triggered}",
        f"Data confidence: {confidence.label} ({confidence.score_pct}%)",
        "",
        "Computed metrics:",
        f"  ADV$ (30d): {_fmt_dollar_m(score.adv.adv_dollar_30d)}",
        f"  ADV$ (90d): {_fmt_dollar_m(score.adv.adv_dollar_90d)}",
        f"  Amihud (30d): {_fmt_or_na(score.amihud.amihud_30d, '.5f')}",
        f"  Volume CV (30d): {_fmt_or_na(score.volume_cv.volume_cv_30d, '.3f')}",
        f"  Float shares: {_fmt_or_na(score.structural.float_shares, ',.0f')}",
        f"  Float % of outstanding: {_fmt_pct(score.structural.float_pct_of_outstanding)}",
        f"  Short interest (% of float): {_fmt_pct(score.structural.short_percent_float)}",
        f"  Top-10 institutional holdings: {_fmt_pct(score.structural.top10_institutional_pct)}",
        f"  Buyback quarterly spend: {_fmt_dollar_m(score.buyback.quarterly_spend)}",
        f"  Buyback yield (annualised): {_fmt_pct(score.buyback.buyback_yield)}",
        f"  Buyback intensity ratio (BIR): {_fmt_pct(score.buyback.bir)}",
        f"  Buyback inflation flag: {score.buyback.inflation_flag}",
        "",
        "Per-dimension scores:",
    ]
    for d in score.dimension_scores:
        lines.append(f"  - {d}")

    if score.flags:
        lines.append("")
        lines.append("Flags raised:")
        lines.extend(f"  - {f}" for f in score.flags)

    if confidence.warnings:
        lines.append("")
        lines.append("Data-quality warnings:")
        lines.extend(f"  - {w}" for w in confidence.warnings)

    lines.append("")
    lines.append(f"Tier action policy: {score.final_tier.action}")

    if score.mirage.triggered and score.mirage.reason:
        lines.append("")
        lines.append(f"Short squeeze risk detail: {score.mirage.reason}")

    return "\n".join(lines)


def _format_comparison_context(
    scored: Sequence[tuple[LiquidityScore, ConfidenceReport]],
    question: Optional[str] = None,
) -> str:
    blocks: list[str] = []
    for score, confidence in scored:
        blocks.append(
            f"=== {score.ticker} ===\n"
            + _format_single_context(score, confidence, question)
        )
    return "\n\n".join(blocks)


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_dollar_m(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def _fmt_or_na(v: Optional[float], spec: str) -> str:
    if v is None:
        return "n/a"
    return format(v, spec)


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2%}"
