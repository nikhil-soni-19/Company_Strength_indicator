"""
Layer 2 LLM interpreter — adversarial moat analysis.
Uses Anthropic claude-sonnet-4-6.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from layer2.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
_VALID_DIRECTIONS = {"strengthening", "stable", "eroding"}
_VALID_NVN = {"consistent", "conflict", "insufficient_data"}


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v*100:+.1f}%"


def _fmt_series(series: List[float]) -> str:
    return "[" + ", ".join(f"{v*100:.1f}%" for v in series) + "]"


def _fmt_news(articles: List[dict]) -> str:
    if not articles:
        return "(no articles retrieved)"
    return "\n".join(
        f"[{i+1}] {a['title']} ({a.get('published_at', '')}) — {a.get('snippet', '')}"
        for i, a in enumerate(articles)
    )


def _fmt_excerpts(chunks: List[str]) -> str:
    if not chunks:
        return "(no excerpts retrieved)"
    return "\n".join(f"[EXCERPT {i+1}] {c[:600]}" for i, c in enumerate(chunks))


def _parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _validate(parsed: dict) -> dict:
    score = float(parsed.get("moat_score_l2", 5.0))
    parsed["moat_score_l2"] = max(0.0, min(10.0, score))

    direction = str(parsed.get("direction", "stable")).lower()
    parsed["direction"] = direction if direction in _VALID_DIRECTIONS else "stable"

    nvn = str(parsed.get("narrative_vs_numbers", "insufficient_data")).lower()
    parsed["narrative_vs_numbers"] = nvn if nvn in _VALID_NVN else "insufficient_data"

    for list_field in ("key_sources", "key_threats", "claimed_moat_sources", "sources_cited"):
        val = parsed.get(list_field, [])
        parsed[list_field] = val if isinstance(val, list) else []

    return parsed


def interpret(
    ticker: str,
    peers: List[str],
    l1_computed: dict,
    flags: List[str],
    moat_context: Dict[str, List[str]],
    competitive_news: List[dict],
) -> dict:
    """
    Call Anthropic LLM with adversarial moat prompt.
    Returns validated dict matching the prompt's JSON schema.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    user_msg = USER_PROMPT_TEMPLATE.format(
        ticker=ticker,
        as_of_date=date.today().isoformat(),
        peers=", ".join(peers) if peers else "none found",
        gross_margin_series=_fmt_series(l1_computed.get("gross_margin_series", [])),
        gross_margin_peer_median=_fmt_pct(l1_computed.get("gross_margin_peer_median")),
        avg_gross_margin_spread=_fmt_pct(l1_computed.get("avg_gross_margin_spread")),
        gross_margin_cv=f"{l1_computed.get('gross_margin_cv', 0.0):.3f}",
        op_margin_series=_fmt_series(l1_computed.get("op_margin_series", [])),
        avg_op_margin_spread=_fmt_pct(l1_computed.get("avg_op_margin_spread")),
        roic_company=_fmt_pct(l1_computed.get("roic_company")),
        roic_peer_median=_fmt_pct(l1_computed.get("roic_peer_median")),
        roic_spread=_fmt_pct(l1_computed.get("roic_spread")),
        avg_fcf_margin_spread=_fmt_pct(l1_computed.get("avg_fcf_margin_spread")),
        insider_pct=_fmt_pct(l1_computed.get("insider_pct")),
        flags=", ".join(flags) if flags else "(none)",
        moat_claims=_fmt_excerpts(moat_context.get("moat_claims", [])),
        risk_factors=_fmt_excerpts(moat_context.get("threats", [])),
        transcript_moat=_fmt_excerpts(moat_context.get("transcript_moat", [])),
        competitive_context=_fmt_news(competitive_news),
    )

    _fallback = {
        "moat_score_l2": 5.0,
        "direction": "stable",
        "key_sources": [],
        "key_threats": [],
        "claimed_moat_sources": [],
        "narrative_vs_numbers": "insufficient_data",
        "conflict_description": None,
        "bull_case": None,
        "bear_case": None,
        "reasoning": "LLM response could not be parsed.",
        "sources_cited": [],
    }

    for attempt in range(2):
        suffix = "" if attempt == 0 else (
            "\n\nIMPORTANT: Return ONLY valid JSON. No text before or after the JSON object."
        )
        response = client.messages.create(
            model=_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg + suffix}],
        )
        raw = response.content[0].text
        try:
            return _validate(_parse_response(raw))
        except Exception as e:
            if attempt == 1:
                print(f"  [LLM] Parse failed after retry: {e}\nRaw: {raw[:300]}")
                return _fallback

    return _fallback
