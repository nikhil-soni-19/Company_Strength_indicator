"""
Phase 5 — Output contract: Pydantic schema for the final agent output.

The schema is the stable interface the rest of the system consumes.
validate_output() converts a FusionOutput to this schema and raises
ValidationError if any field violates its constraint.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from fusion import FusionOutput
from layer2_llm import ThemeAssessment


# ─── Schema ───────────────────────────────────────────────────────────────────

class MetricsSchema(BaseModel):
    rd_rev_level:    float
    rd_rev_slope:    float
    capex_rev_level: float
    capex_rev_slope: float


class ThemeSchema(BaseModel):
    score:          float = Field(ge=0.0, le=10.0)
    rationale:      str
    evidence_used:  list[str]
    confidence:     float = Field(ge=0.0, le=1.0)


class OverallSchema(BaseModel):
    verdict:    str
    score:      float = Field(ge=0.0, le=10.0)   # combined 60/40 score
    l1_score:   float = Field(ge=0.0, le=10.0)
    l2_score:   float = Field(ge=0.0, le=10.0)
    confidence: float = Field(ge=0.0, le=1.0)    # guardrail-adjusted


class DataCoverageSchema(BaseModel):
    quarters_returned: int  = Field(ge=0)
    capex_found:       bool
    holders_found:     bool


class FinalOutputSchema(BaseModel):
    ticker:               str
    metrics:              MetricsSchema
    flags:                list[str]
    themes:               dict[str, ThemeSchema]
    low_evidence_themes:  list[str]   # excluded from L2 avg (confidence < gate)
    overall:              OverallSchema
    data_coverage:        DataCoverageSchema
    guardrail_notes:      list[str]

    @model_validator(mode="after")
    def _check_themes(self) -> "FinalOutputSchema":
        required = {"tech", "capacity", "esg", "governance"}
        missing = required - self.themes.keys()
        if missing:
            raise ValueError(f"themes dict is missing required keys: {missing}")
        return self


# ─── Conversion ───────────────────────────────────────────────────────────────

def _theme_to_schema(t: ThemeAssessment) -> ThemeSchema:
    return ThemeSchema(
        score=t.score,
        rationale=t.rationale,
        evidence_used=t.evidence_used,
        confidence=t.confidence,
    )


def validate_output(fusion: FusionOutput) -> FinalOutputSchema:
    """
    Convert a FusionOutput to FinalOutputSchema and validate all constraints.
    Raises pydantic.ValidationError on invalid data — caller should handle.
    """
    return FinalOutputSchema(
        ticker=fusion.ticker,
        metrics=MetricsSchema(**fusion.metrics),
        flags=fusion.flags,
        themes={k: _theme_to_schema(v) for k, v in fusion.themes.items()},
        low_evidence_themes=fusion.low_evidence_themes,
        overall=OverallSchema(
            verdict=fusion.verdict,
            score=fusion.combined_score,
            l1_score=fusion.l1_score,
            l2_score=fusion.l2_score,
            confidence=fusion.final_confidence,
        ),
        data_coverage=DataCoverageSchema(**fusion.data_coverage),
        guardrail_notes=fusion.guardrail_notes,
    )
