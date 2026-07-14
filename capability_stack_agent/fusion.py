"""
Phase 4 — Fusion: combine Layer 1 + Layer 2 and apply the confidence guardrail.

Fusion formula (60/40):
    combined_score = LAYER1_WEIGHT * l1_score + LAYER2_WEIGHT * mean(l2_theme_scores)

Confidence guardrail:
    Base = mean(l2_theme_confidences)
    Discounts applied (subtractive, never additive) for:
      1. Thin data coverage   — fewer quarters than minimum
      2. Missing capex data   — capex_found=False
      3. Sparse RAG retrieval — any theme below RAG_MIN_CHUNKS_PER_THEME
      4. L1/L2 flag conflicts — deterministic flags contradict narrative scores
      5. Suspect high scores  — theme score > 7 with LLM confidence < 0.4
    Floor: 0.1

The guardrail logs every discount it applies via guardrail_notes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import (
    CAPEX_LIGHT_FLOOR,
    CAPEX_REINVESTMENT_STRONG_LEVEL,
    GUARDRAIL_COVERAGE_DISCOUNT,
    GUARDRAIL_FLAG_CONFLICT_DISCOUNT,
    GUARDRAIL_THIN_RAG_DISCOUNT,
    L1_R2_LOW_THRESHOLD,
    L2_CONFIDENCE_GATE,
    LAYER1_WEIGHT,
    LAYER2_WEIGHT,
    MIN_QUARTERS_FULL_CONFIDENCE,
    RAG_MIN_CHUNKS_PER_THEME,
)
from layer1_deterministic import Layer1Output
from layer2_llm import Layer2Output, ThemeAssessment

_THEMES = ["tech", "capacity", "esg", "governance"]

# Conflict detection: flag → (theme, direction, score_threshold, description)
#   direction "low"  = flag fired but narrative score is too low  (false negative in narrative)
#   direction "high" = flag fired but narrative score is too high (false positive in narrative)
_FLAG_CONFLICT_RULES: list[tuple[str, str, str, float, str]] = [
    # R&D_INTENSIFYING says R&D is rising fast — tech score should not be very low
    ("R&D_INTENSIFYING",           "tech",     "low",  4.0,
     "R&D_INTENSIFYING flag fired but LLM tech score is {actual:.0f}/10 (< 4) — "
     "narrative may be underweighting the quantitative R&D trend"),
    # CAPEX_REINVESTMENT_STRONG says capex is heavy — capacity score should not be very low
    ("CAPEX_REINVESTMENT_STRONG",  "capacity", "low",  4.0,
     "CAPEX_REINVESTMENT_STRONG flag fired but LLM capacity score is {actual:.0f}/10 (< 4) — "
     "narrative may not reflect heavy reinvestment"),
    # CAPEX_LIGHT_BUSINESS says capex is minimal — capacity score should not be very high
    ("CAPEX_LIGHT_BUSINESS",       "capacity", "high", 8.0,
     "CAPEX_LIGHT_BUSINESS flag fired but LLM capacity score is {actual:.0f}/10 (> 8) — "
     "narrative may be overstating physical capacity"),
]


# ─── Output type ──────────────────────────────────────────────────────────────

@dataclass
class FusionOutput:
    ticker: str
    # ── Layer 1 pass-through ──────────────────────────────────────────────────
    metrics: dict           # rd_rev_level, rd_rev_slope, capex_rev_level, capex_rev_slope
    flags: list[str]
    # ── Layer 2 pass-through ──────────────────────────────────────────────────
    themes: dict[str, ThemeAssessment]
    low_evidence_themes: list[str]  # themes excluded from L2 score (confidence < gate)
    # ── Fusion ────────────────────────────────────────────────────────────────
    l1_score: float         # [0, 10]
    l2_score: float         # mean of theme scores, [0, 10]
    combined_score: float   # 60/40 weighted, [0, 10]
    verdict: str
    # ── Confidence guardrail ──────────────────────────────────────────────────
    base_confidence: float  # mean of theme confidences before guardrail
    final_confidence: float # after all discounts, floor 0.1
    guardrail_notes: list[str]
    # ── Coverage ──────────────────────────────────────────────────────────────
    data_coverage: dict     # quarters_returned, capex_found, holders_found


# ─── Scoring helpers ──────────────────────────────────────────────────────────

def _compute_l2_score(
    themes: dict[str, ThemeAssessment],
) -> tuple[float, list[str]]:
    """
    Confidence-weighted mean of theme scores, with low-evidence gating.

    Themes with LLM confidence < L2_CONFIDENCE_GATE are excluded from the
    denominator entirely — they are marked 'low evidence' rather than scored
    as a real 3.0 or 4.0 that would drag the average down.

    Returns:
        (weighted_score, list_of_excluded_theme_names)
    """
    included = {k: v for k, v in themes.items() if v.confidence >= L2_CONFIDENCE_GATE}
    excluded = [k for k, v in themes.items() if v.confidence < L2_CONFIDENCE_GATE]

    if not included:
        return 5.0, list(themes.keys())

    total_weight = sum(t.confidence for t in included.values())
    weighted_sum = sum(t.score * t.confidence for t in included.values())
    return round(weighted_sum / total_weight, 4), excluded


def _verdict(score: float) -> str:
    if score >= 7.5:
        return "Strong execution engine"
    if score >= 6.0:
        return "Solid operating capacity"
    if score >= 4.5:
        return "Adequate"
    if score >= 3.0:
        return "Weak investment signals"
    return "Capability concerns"


# ─── L1 confidence ────────────────────────────────────────────────────────────

def _compute_l1_confidence(l1_output: Layer1Output) -> tuple[float, list[str]]:
    """
    Compute a Layer 1 signal quality confidence from data properties.

    Discounts applied:
      - yfinance-only source (no DB): shallower history, less reliable
      - Slope-based flag fired but OLS R² is low: the trend may be noise
    """
    confidence = 1.0
    notes: list[str] = []

    if l1_output.data_coverage.source == "yfinance":
        confidence -= 0.10
        notes.append(
            "L1 data from yfinance only — DB unavailable; "
            "history limited to ~4Q and may be less accurate"
        )

    if (
        "R&D_INTENSIFYING" in l1_output.flags
        and l1_output.rd_rev_r2 < L1_R2_LOW_THRESHOLD
    ):
        confidence -= 0.10
        notes.append(
            f"R&D_INTENSIFYING fired but OLS R²={l1_output.rd_rev_r2:.2f} "
            f"(< {L1_R2_LOW_THRESHOLD}) — slope may not reflect a real trend"
        )

    if (
        "CAPEX_REINVESTMENT_STRONG" in l1_output.flags
        and l1_output.capex_rev_r2 < L1_R2_LOW_THRESHOLD
    ):
        confidence -= 0.10
        notes.append(
            f"CAPEX_REINVESTMENT_STRONG fired but OLS R²={l1_output.capex_rev_r2:.2f} "
            f"(< {L1_R2_LOW_THRESHOLD}) — slope may not reflect a real trend"
        )

    return max(0.10, round(confidence, 4)), notes


# ─── Self-consistency audit ────────────────────────────────────────────────────

_POSITIVE_WORDS = {
    "exceptional", "outstanding", "significant", "strong", "robust",
    "excellent", "leading", "world-class", "best-in-class", "impressive",
    "substantial", "notably", "remarkable", "disciplined", "rigorous",
}
_NEGATIVE_WORDS = {
    "lacking", "weak", "poor", "limited", "absent", "minimal",
    "inadequate", "no evidence", "insufficient", "sparse", "vague",
    "boilerplate", "generic", "no specific", "not disclosed",
}


def _self_consistency_check(themes: dict[str, ThemeAssessment]) -> list[str]:
    """
    Flag themes where the LLM's reasoning/rationale text sentiment diverges
    from the numeric score it assigned — catches calibration errors using
    only the text the LLM already generated.

    Example: reasoning says "exceptional technology deployment" but score = 3.
    """
    flags: list[str] = []
    for theme_name, t in themes.items():
        text = (t.reasoning + " " + t.rationale).lower()
        pos_hits = sum(1 for w in _POSITIVE_WORDS if w in text)
        neg_hits = sum(1 for w in _NEGATIVE_WORDS if w in text)

        if pos_hits >= 2 and t.score < 5.0:
            flags.append(
                f"Self-consistency: '{theme_name}' reasoning is positive "
                f"({pos_hits} positive signals) but score={t.score:.0f}/10 — "
                "possible under-scoring"
            )
        if neg_hits >= 2 and t.score > 7.0:
            flags.append(
                f"Self-consistency: '{theme_name}' reasoning is negative "
                f"({neg_hits} negative signals) but score={t.score:.0f}/10 — "
                "possible over-scoring"
            )
    return flags


# ─── Guardrail ────────────────────────────────────────────────────────────────

def _apply_guardrail(
    l1_output: Layer1Output,
    l2_output: Layer2Output,
    base_confidence: float,
) -> tuple[float, list[str]]:
    """
    Apply subtractive discounts to base_confidence and return
    (clamped_confidence, list_of_notes).

    Guardrail is the single most important reliability feature:
    it must make confidence go DOWN on thin/conflicting evidence — never up.
    Every triggered discount is recorded in notes for transparency.
    """
    confidence = base_confidence
    notes: list[str] = []

    # ── 1. Data coverage ──────────────────────────────────────────────────────
    qs = l1_output.data_coverage.quarters_returned
    if qs < MIN_QUARTERS_FULL_CONFIDENCE:
        confidence -= GUARDRAIL_COVERAGE_DISCOUNT
        notes.append(
            f"Thin data coverage: only {qs} quarters returned "
            f"(minimum for full confidence: {MIN_QUARTERS_FULL_CONFIDENCE}) "
            f"— confidence discounted by {GUARDRAIL_COVERAGE_DISCOUNT:.0%}"
        )

    if not l1_output.data_coverage.capex_found:
        confidence -= 0.05
        notes.append(
            "Capex data absent — capex/revenue ratios defaulted to zero; "
            "CAPEX flags and scores are unreliable"
        )

    # ── 2. Sparse RAG retrieval ────────────────────────────────────────────────
    for theme in _THEMES:
        chunk_count = l2_output.rag_chunks_per_theme.get(theme, 0)
        if chunk_count < RAG_MIN_CHUNKS_PER_THEME:
            confidence -= GUARDRAIL_THIN_RAG_DISCOUNT
            notes.append(
                f"Thin RAG for theme '{theme}': only {chunk_count} chunks retrieved "
                f"(minimum: {RAG_MIN_CHUNKS_PER_THEME}) — LLM evidence is sparse"
            )

    # ── 3. L1 flag / L2 narrative conflicts ───────────────────────────────────
    for flag, theme, direction, threshold, msg_template in _FLAG_CONFLICT_RULES:
        if flag not in l1_output.flags:
            continue
        theme_obj = l2_output.themes.get(theme)
        if theme_obj is None:
            continue
        conflict = (
            (direction == "low"  and theme_obj.score < threshold) or
            (direction == "high" and theme_obj.score > threshold)
        )
        if conflict:
            confidence -= GUARDRAIL_FLAG_CONFLICT_DISCOUNT
            notes.append(msg_template.format(actual=theme_obj.score))

    # ── 4. Suspect high scores (high score, very low LLM confidence) ──────────
    for theme_name in _THEMES:
        theme_obj = l2_output.themes.get(theme_name)
        if theme_obj and theme_obj.score > 7.0 and theme_obj.confidence < 0.4:
            confidence -= 0.10
            notes.append(
                f"Theme '{theme_name}' has a high score ({theme_obj.score:.0f}/10) "
                f"but very low LLM confidence ({theme_obj.confidence:.2f}) — "
                f"score may not be reliable"
            )

    # ── 5. Industry context absent ────────────────────────────────────────────
    if l2_output.industry_context_count == 0:
        confidence -= 0.05
        notes.append("No Tavily industry context retrieved — tech theme evidence is limited to 10-K text")

    final = max(0.10, round(confidence, 4))
    return final, notes


# ─── Public entry point ───────────────────────────────────────────────────────

def fuse(l1_output: Layer1Output, l2_output: Layer2Output) -> FusionOutput:
    """
    Combine Layer 1 and Layer 2 into the final FusionOutput.

    Fusion:
        combined_score = 0.60 * l1_score + 0.40 * l2_score

    Confidence pipeline:
        base        = mean(l2_theme_confidences, gated themes only)
        guardrail   = base − coverage/RAG/conflict discounts
        l1_discount = additional discount if L1 data quality is poor
        dispersion  = additional discount if theme scores are inconsistent
        final       = max(0.10, guardrail − l1_discount − dispersion_penalty)

    All discounts are logged in guardrail_notes for transparency.
    """
    # ── L2 score (confidence-weighted, low-evidence themes excluded) ──────────
    l2_score, low_evidence_themes = _compute_l2_score(l2_output.themes)

    # ── Fusion ────────────────────────────────────────────────────────────────
    l1_score = l1_output.l1_score
    combined_score = round(LAYER1_WEIGHT * l1_score + LAYER2_WEIGHT * l2_score, 4)
    verdict = _verdict(combined_score)

    # ── Base confidence = mean of *all* theme confidences (pre-gate) ─────────
    # We use all themes here (not just gated ones) so the base reflects how
    # much evidence the LLM actually found across the full assessment.
    theme_confidences = [t.confidence for t in l2_output.themes.values()]
    base_confidence = (
        sum(theme_confidences) / len(theme_confidences) if theme_confidences else 0.5
    )

    # ── Primary guardrail (coverage, RAG quality, flag conflicts) ────────────
    final_confidence, guardrail_notes = _apply_guardrail(
        l1_output, l2_output, base_confidence
    )

    # ── L1 signal quality discount ────────────────────────────────────────────
    l1_conf, l1_notes = _compute_l1_confidence(l1_output)
    if l1_conf < 1.0:
        l1_discount = round(1.0 - l1_conf, 4)
        final_confidence = max(0.10, round(final_confidence - l1_discount, 4))
        guardrail_notes.extend(l1_notes)

    # ── Theme score dispersion penalty ────────────────────────────────────────
    # High variance across theme scores (std > 2.5) indicates the LLM's
    # judgments are inconsistent — lower confidence in the aggregate L2 score.
    included_scores = [
        t.score for k, t in l2_output.themes.items()
        if k not in low_evidence_themes
    ]
    if len(included_scores) >= 2:
        score_std = float(np.std(included_scores))
        if score_std > 2.5:
            dispersion_discount = 0.10
            final_confidence = max(0.10, round(final_confidence - dispersion_discount, 4))
            guardrail_notes.append(
                f"High theme score dispersion (std={score_std:.2f}) — "
                f"LLM judgments inconsistent across themes; "
                f"confidence discounted by {dispersion_discount:.0%}"
            )

    # ── Self-consistency audit ────────────────────────────────────────────────
    # Does not affect the score — purely diagnostic; flags anomalies for review.
    consistency_flags = _self_consistency_check(l2_output.themes)
    if consistency_flags:
        guardrail_notes.extend(consistency_flags)

    # ── Build output ──────────────────────────────────────────────────────────
    metrics = {
        "rd_rev_level":    round(l1_output.rd_rev_level, 6),
        "rd_rev_slope":    round(l1_output.rd_rev_slope, 6),
        "capex_rev_level": round(l1_output.capex_rev_level, 6),
        "capex_rev_slope": round(l1_output.capex_rev_slope, 6),
    }

    coverage = {
        "quarters_returned": l1_output.data_coverage.quarters_returned,
        "capex_found":       l1_output.data_coverage.capex_found,
        "holders_found":     l1_output.data_coverage.holders_found,
    }

    return FusionOutput(
        ticker=l1_output.ticker,
        metrics=metrics,
        flags=list(l1_output.flags),
        themes=dict(l2_output.themes),
        low_evidence_themes=low_evidence_themes,
        l1_score=round(l1_score, 4),
        l2_score=round(l2_score, 4),
        combined_score=combined_score,
        verdict=verdict,
        base_confidence=round(base_confidence, 4),
        final_confidence=final_confidence,
        guardrail_notes=guardrail_notes,
        data_coverage=coverage,
    )
