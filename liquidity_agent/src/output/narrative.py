"""Builds the 2-3 sentence plain-English summary shown beneath the tier badge.

This is the deterministic, template-based summary. The LLM agent in
:mod:`src.agent.llm_agent` consumes both the metrics *and* this narrative
when answering free-form natural-language questions.
"""

from __future__ import annotations

from typing import Optional

from src.scoring.engine import LiquidityScore


def build_narrative(score: LiquidityScore) -> str:
    """Return a 2-3 sentence English description of the score.

    Amihud price-impact and DTL exit-risk are intentionally omitted here;
    they are surfaced only when a user query explicitly requests them
    (handled by the interpreter layer).
    """
    parts: list[str] = []
    parts.append(_liquidity_sentence(score))

    structural_sentence = _structural_sentence(score)
    if structural_sentence:
        parts.append(structural_sentence)

    buyback_sentence = _buyback_sentence(score)
    if buyback_sentence:
        parts.append(buyback_sentence)

    return " ".join(parts)


def _liquidity_sentence(score: LiquidityScore) -> str:
    adv_str = _fmt_dollar_m(score.adv.adv_dollar_30d)

    if score.final_tier.number == 1:
        descriptor = "highly liquid"
    elif score.final_tier.number == 2:
        descriptor = "moderately liquid"
    elif score.final_tier.number == 3:
        descriptor = "thinly traded"
    else:
        descriptor = "structurally illiquid"

    return f"{score.ticker} is {descriptor} with a 30-day ADV$ of {adv_str}."


def _structural_sentence(score: LiquidityScore) -> Optional[str]:
    if score.mirage.triggered:
        return (
            "Short squeeze risk override active: the thin float and elevated "
            "short interest signal a short squeeze condition, so the tier has "
            "been downgraded regardless of recent volume."
        )
    # Volume CV is only surfaced in the Summary when it reaches the Critical band
    # (> 1.0), signalling genuinely erratic trading that warrants trader attention.
    cv = score.volume_cv.volume_cv_30d
    if cv is not None and cv > 1.0:
        return (
            f"⚠ Volume CV of {cv:.2f} is in the Critical band — "
            f"erratic trading traffic may materially inflate realised slippage."
        )
    return None


def _buyback_sentence(score: LiquidityScore) -> Optional[str]:
    """Surface buyback inflation only when BIR exceeds the flag threshold."""
    if not score.buyback.inflation_flag:
        return None
    bir = score.buyback.bir
    if bir is None:
        return None
    if bir > 0.30:
        return (
            f"⚠ The company's buyback programme is aggressive — it accounted "
            f"for {bir:.0%} of its own quarterly volume, meaning the reported "
            f"ADV$ materially overstates third-party tradeable liquidity."
        )
    return (
        f"⚠ Active buyback programme ({bir:.0%} of quarterly volume) is "
        f"modestly inflating the reported ADV$ figure."
    )


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


def _fmt_amihud(v: Optional[float]) -> str:
    """Format an Amihud ratio value — used by the interpreter on explicit request."""
    if v is None:
        return "n/a"
    return f"{v:.4f}"
