"""
Layer 2: LLM reasoning over Layer 1 numbers + RAG passages.
Reads LLM provider and API key from .env.
"""
import os
import json
from dotenv import load_dotenv
from models.layer1_output import Layer1Output
from models.rag_output import RAGOutput
from models.intent import QueryIntent
from models.verdict import FinalVerdict

load_dotenv()

REASONING_PROMPT = """
You are a financial execution analyst. Your job is to assess whether a company's
management narrative is consistent with its financial trajectory.

You do NOT forecast future performance. You do NOT give buy/sell recommendations.
You assess execution quality and narrative credibility based only on the data provided.

## Original Query
{raw_query}

## Hypothesis to Test
{layer2_question}

## Layer 1 Computed Signals (last 4 quarters, oldest to newest)
- Revenue YoY growth:        {rev_yoy}
- OpEx YoY growth:           {opex_yoy}
- Operating Leverage Delta:  {ol_delta}
- OL Slope (8Q trend):       {ol_slope:.4f} (negative = worsening)
- Operating Margin Slope:    {op_margin_slope:.4f} (negative = compressing)
- Gross Margin (latest):     {gross_margin:.1%}
- Operating Margin (latest): {op_margin:.1%}
- FCF/NI Ratio (latest):     {fcf_ni:.2f} (<0.8 = low cash quality)
- OL Consistency (8Q):       {ol_consistency:.0%} of quarters positive

## Flags Emitted
{flags}

## Evidence from Neon Database
The following passages come from multiple sources in our database.
Each is labelled by source type so you can weight them appropriately:
  - "10-K/Q" / "sec_filing"   → SEC filing narrative text (high authority)
  - "earnings_call"            → Earnings call transcript quotes (management voice)
  - "employee_esg"             → Employee headcount + ESG metrics (workforce trends)
  - "geographic_segment"       → Revenue by geography
  - "product_segment"          → Revenue by product line
  - "financial_facts"          → Structured financial line items
  - "analyst_consensus"        → Street estimates vs actuals (external validation)

Use these to cross-check whether management's stated narrative matches the numbers.
Pay particular attention to:
  - Employee headcount trends vs revenue growth (productivity signal)
  - Earnings call language vs actual financial trajectory
  - Analyst consensus beat/miss pattern (credibility signal)

{rag_context}

## Your Task
Reason step by step:
1. What does the quantitative data (Layer 1) say about execution quality?
2. Does the management narrative from earnings calls / SEC filings match the numbers?
3. What do employee trends or geographic/product breakdowns reveal that the headline numbers hide?
4. What is the single most important contradiction or confirmation across all sources?
5. What is one counterargument you cannot rule out from this data alone?

Then return ONLY valid JSON:
{{
  "execution_score": <float 0-10: overall execution quality>,
  "credibility_score": <float 0-10: management narrative vs reality>,
  "direction": <"improving" | "stable" | "deteriorating">,
  "verdict": <"narrative_credible" | "narrative_not_credible" | "insufficient_data">,
  "key_contradiction": <string or null>,
  "counterargument": <string or null>,
  "reasoning": <2-3 sentence plain English summary>,
  "sources_cited": [<list of "Q3-2024 10-Q" style citations>]
}}
"""


def reason(intent: QueryIntent, l1: Layer1Output, rag: RAGOutput) -> FinalVerdict:
    """Run Layer 2 LLM reasoning and return a FinalVerdict."""
    rag_context = _format_rag_context(rag)

    prompt = REASONING_PROMPT.format(
        raw_query=intent.raw_query,
        layer2_question=intent.layer2_question or "Does execution quality match management claims?",
        rev_yoy=_fmt_list(l1.rev_yoy_pct, "%", 1),
        opex_yoy=_fmt_list(l1.opex_yoy_pct, "%", 1),
        ol_delta=_fmt_list(l1.ol_delta, "pp", 1),
        ol_slope=l1.ol_slope,
        op_margin_slope=l1.op_margin_slope,
        gross_margin=l1.gross_margin,
        op_margin=l1.op_margin,
        fcf_ni=l1.fcf_ni_ratio[-1] if l1.fcf_ni_ratio else 0,
        ol_consistency=l1.ol_consistency,
        flags=", ".join(l1.flags) if l1.flags else "None",
        rag_context=rag_context,
    )

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    response_text = _call_llm(prompt, provider)

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        result = {
            "execution_score": l1.score,
            "credibility_score": 5.0,
            "direction": "stable",
            "verdict": "insufficient_data",
            "key_contradiction": None,
            "counterargument": "LLM response parsing failed.",
            "reasoning": "Layer 2 reasoning failed. Layer 1 score only.",
            "sources_cited": [],
        }

    # Blend Layer 1 score (65%) with Layer 2 execution score (35%)
    blended_score = round(l1.score * 0.65 + result["execution_score"] * 0.35, 2)

    return FinalVerdict(
        ticker=intent.ticker,
        period=l1.period_latest,
        execution_score=blended_score,
        layer1_score=l1.score,
        credibility_score=result["credibility_score"],
        direction=result["direction"],
        verdict=result["verdict"],
        flags=l1.flags,
        key_contradiction=result.get("key_contradiction"),
        counterargument=result.get("counterargument"),
        reasoning=result["reasoning"],
        rag_passages_used=len(rag.passages),
        sources_cited=result.get("sources_cited", []),
    )


def _format_rag_context(rag: RAGOutput) -> str:
    if not rag.rag_enabled or not rag.passages:
        return "No database passages available. Reason purely from Layer 1 numbers."

    # Group passages by source type so the LLM sees them in logical order
    order = ["10-K/Q", "sec_filing", "earnings_call", "employee_esg",
             "analyst_consensus", "geographic_segment", "product_segment",
             "financial_facts", "event"]
    grouped: dict[str, list] = {}
    for p in rag.passages:
        grouped.setdefault(p.source_type or "other", []).append(p)

    lines = []
    for src_type in order + [k for k in grouped if k not in order]:
        if src_type not in grouped:
            continue
        lines.append(f"\n### {src_type.replace('_', ' ').upper()}")
        for p in grouped[src_type]:
            header = f"[{p.period}"
            if p.speaker:
                header += f" | {p.speaker}"
            if p.section:
                header += f" | {p.section}"
            header += "]"
            lines.append(f"{header}\n{p.text}\n")

    if rag.guidance_matches:
        lines.append("\n### GUIDANCE VS ACTUALS")
        for g in rag.guidance_matches:
            lines.append(
                f"- {g.made_in_period}: \"{g.guided_value}\" → "
                f"Actual {g.actual_period}: {g.actual_value} — {g.outcome.upper()}"
            )

    lines.append(f"\n[Credibility track record: {rag.credibility_track_record:.0%} based on earnings beat rate]")
    return "\n".join(lines)


def _fmt_list(values: list, suffix: str = "", decimals: int = 1) -> str:
    return "[" + ", ".join(f"{v:+.{decimals}f}{suffix}" for v in values) + "]"


def _call_llm(prompt: str, provider: str) -> str:
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return resp.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6"),
            max_tokens=1024,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
