"""Call the Anthropic LLM to generate a PESTEL-structured qualitative environment score."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from layer2.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")
_VALID_DIRECTIONS = {"SUPPORTIVE", "MIXED", "HOSTILE"}
_PESTEL_DIMS = ("P", "E", "S", "T", "En", "L")


def _format_news_dim(articles: list[dict]) -> str:
    """Format a list of news dicts for one PESTEL dimension."""
    if not articles:
        return "(no articles retrieved)"
    return "\n".join(
        f"[{i+1}] {a.get('title', '')} ({a.get('published_at', '')}) — {a.get('snippet', '')}"
        for i, a in enumerate(articles)
    )


_DIM_MAP = {
    "P": "rf_political",
    "E": "rf_economic",
    "S": "rf_social",
    "T": "rf_technological",
    "En": "rf_environmental",
    "L": "rf_legal",
}


def _format_rf_dim(chunks: list[str]) -> str:
    """Format a list of 10-K chunk texts for one PESTEL dimension."""
    if not chunks:
        return "(no excerpts retrieved)"
    return "\n".join(
        f"[EXCERPT {i+1}] {c[:600]}" for i, c in enumerate(chunks)
    )


def _build_user_prompt(
    bundle: dict,
    flags: list[str],
    pestel_news: dict[str, list[dict]],
    excerpts: list[str],
    pestel_excerpts: dict[str, list[str]] | None = None,
) -> str:
    """Build the LLM user prompt.

    If pestel_excerpts (per-dimension dict from ten_k_retrieval) is provided,
    it populates 6 aligned excerpt blocks. Otherwise falls back to distributing
    the flat excerpts list across all dimensions.
    """
    # Slim the bundle: omit raw price-series-derived noise, keep PESTEL sub-bundle
    slim_bundle = {
        k: v for k, v in bundle.items()
        if k not in ("flags",)
    }

    # Resolve per-dimension excerpt blocks
    if pestel_excerpts:
        rf_blocks = {
            template_key: _format_rf_dim(pestel_excerpts.get(dim, []))
            for dim, template_key in _DIM_MAP.items()
        }
    else:
        # Fallback: show flat excerpts in every dimension block
        flat = "\n".join(
            f"[EXCERPT {i+1}] {e[:600]}" for i, e in enumerate(excerpts)
        ) or "(no 10-K excerpts available)"
        rf_blocks = {v: flat for v in _DIM_MAP.values()}

    return USER_PROMPT_TEMPLATE.format(
        layer1_bundle_json=json.dumps(slim_bundle, indent=2, default=str),
        flags=", ".join(flags) if flags else "(none)",
        news_political=_format_news_dim(pestel_news.get("P", [])),
        news_economic=_format_news_dim(pestel_news.get("E", [])),
        news_social=_format_news_dim(pestel_news.get("S", [])),
        news_technological=_format_news_dim(pestel_news.get("T", [])),
        news_environmental=_format_news_dim(pestel_news.get("En", [])),
        news_legal=_format_news_dim(pestel_news.get("L", [])),
        **rf_blocks,
    )


def _parse_response(text: str) -> dict:
    """Strip markdown fences, isolate the JSON object, and parse."""
    text = text.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()
    # Isolate the outermost {...} — guards against trailing prose
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


_NARRATIVE_DIMS = ("Political", "Economic", "Social", "Technological", "Environmental", "Legal")


def _validate(parsed: dict) -> dict:
    """Validate and normalise the LLM response dict.  Raises ValueError on bad data."""
    # qual_score
    qs = int(parsed.get("qual_score", -1))
    if not (0 <= qs <= 100):
        raise ValueError(f"qual_score out of range: {qs}")
    parsed["qual_score"] = qs

    # direction
    direction = str(parsed.get("direction", "")).strip().upper()
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"Invalid direction: {direction}")
    parsed["direction"] = direction

    # pestel_scores (optional — tolerate missing dims; default to 50)
    ps = parsed.get("pestel_scores", {})
    if not isinstance(ps, dict):
        ps = {}
    for dim in _PESTEL_DIMS:
        val = ps.get(dim, 50)
        try:
            val = int(val)
        except (TypeError, ValueError):
            val = 50
        val = max(0, min(100, val))
        ps[dim] = val
    parsed["pestel_scores"] = ps

    # narrative_by_dim — tolerate missing keys, default to empty string
    nbd = parsed.get("narrative_by_dim", {})
    if not isinstance(nbd, dict):
        nbd = {}
    for dim in _NARRATIVE_DIMS:
        nbd.setdefault(dim, "")
    parsed["narrative_by_dim"] = nbd

    return parsed


def interpret(
    bundle: dict,
    flags: list[str],
    pestel_news: dict[str, list[dict]],
    excerpts: list[str],
    pestel_excerpts: dict[str, list[str]] | None = None,
) -> dict:
    """
    Call Anthropic API with PESTEL-structured context.

    Args:
        bundle:           Full Layer 1 bundle (including pestel sub-dict).
        flags:            List of flag strings from emit_flags().
        pestel_news:      Dict keyed by PESTEL dim — news dicts from tavily_pestel.py.
        excerpts:         Flat list of 10-K risk factor chunk texts (legacy / fallback).
        pestel_excerpts:  Per-dimension 10-K excerpts from ten_k_retrieval.py.
                          When provided, supersedes the flat excerpts list.

    Returns dict with:
        qual_score, direction, pestel_scores, narrative_by_dim, narrative,
        key_tailwinds, key_risks
    """
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    user_msg = _build_user_prompt(bundle, flags, pestel_news, excerpts, pestel_excerpts)

    _fallback = {
        "qual_score": 50,
        "direction": "MIXED",
        "pestel_scores": {dim: 50 for dim in _PESTEL_DIMS},
        "narrative_by_dim": {dim: "" for dim in _NARRATIVE_DIMS},
        "narrative": "LLM response could not be parsed.",
        "key_tailwinds": [],
        "key_risks": [],
    }

    for attempt in range(2):
        extra = "" if attempt == 0 else (
            "\n\nIMPORTANT: Return ONLY valid JSON. No text before or after the JSON object."
        )
        response = client.messages.create(
            model=_MODEL,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg + extra}],
        )
        raw = response.content[0].text
        try:
            parsed = _validate(_parse_response(raw))
            return parsed
        except Exception as e:
            if attempt == 1:
                print(f"  [ERROR] LLM parse failed after retry: {e}\nRaw: {raw[:300]}")
                return _fallback

    return _fallback
