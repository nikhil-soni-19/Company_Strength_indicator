"""
Unit tests for fusion.py — guardrail logic and score computation.
No LLM, no DB, no network. All inputs are synthetic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import config
from data_contract import DataCoverage, InputBundle
from fusion import FusionOutput, _apply_guardrail, _compute_l2_score, _verdict, fuse
from layer1_deterministic import Layer1Output, run_layer1
from layer2_llm import Layer2Output, ThemeAssessment


# ─── Builders ─────────────────────────────────────────────────────────────────

def _make_coverage(
    quarters: int = 12,
    capex_found: bool = True,
    holders_found: bool = True,
) -> DataCoverage:
    return DataCoverage(
        quarters_returned=quarters,
        rd_quarters_from_db=quarters,
        rd_quarters_from_yf=0,
        capex_found=capex_found,
        holders_found=holders_found,
        source="db",
    )


def _make_l1(
    flags: list[str] | None = None,
    quarters: int = 12,
    capex_found: bool = True,
    rd_rev_level: float = 0.08,
    capex_rev_level: float = 0.10,
    l1_score: float = 7.0,
) -> Layer1Output:
    coverage = _make_coverage(quarters=quarters, capex_found=capex_found)
    return Layer1Output(
        ticker="TEST",
        periods=["2024-01-01"] * quarters,
        rd_rev=[rd_rev_level] * quarters,
        capex_rev=[capex_rev_level] * quarters,
        rd_rev_level=rd_rev_level,
        rd_rev_slope=0.0,
        rd_rev_cagr=0.0,
        capex_rev_level=capex_rev_level,
        capex_rev_slope=0.0,
        capex_rev_cagr=0.0,
        insider_pct=None,
        institutional_top10=None,
        flags=flags or [],
        l1_score=l1_score,
        data_coverage=coverage,
    )


def _make_theme(score: float, confidence: float) -> ThemeAssessment:
    return ThemeAssessment(
        score=score,
        rationale="test rationale",
        evidence_used=["evidence"],
        confidence=confidence,
    )


def _make_l2(
    themes: dict[str, tuple[float, float]] | None = None,
    rag_counts: dict[str, int] | None = None,
    industry_count: int = 5,
) -> Layer2Output:
    default_themes = {
        "tech":       (7.0, 0.8),
        "capacity":   (6.0, 0.7),
        "esg":        (5.0, 0.6),
        "governance": (6.0, 0.7),
    }
    t = themes or default_themes
    return Layer2Output(
        ticker="TEST",
        themes={k: _make_theme(s, c) for k, (s, c) in t.items()},
        rag_chunks_per_theme=rag_counts or {k: 5 for k in t},
        industry_context_count=industry_count,
        raw_llm_response="{}",
    )


# ─── Score computation ────────────────────────────────────────────────────────

class TestL2Score:
    def test_average_of_four_themes(self):
        l2 = _make_l2(themes={
            "tech": (8.0, 0.9), "capacity": (6.0, 0.8),
            "esg": (4.0, 0.5), "governance": (6.0, 0.7),
        })
        assert _compute_l2_score(l2.themes) == pytest.approx(6.0)

    def test_all_same_score(self):
        l2 = _make_l2(themes={k: (7.0, 0.8) for k in ["tech", "capacity", "esg", "governance"]})
        assert _compute_l2_score(l2.themes) == pytest.approx(7.0)

    def test_empty_themes_returns_5(self):
        assert _compute_l2_score({}) == pytest.approx(5.0)


class TestVerdictThresholds:
    @pytest.mark.parametrize("score,expected", [
        (8.0, "Strong execution engine"),
        (7.5, "Strong execution engine"),
        (7.4, "Solid operating capacity"),
        (6.0, "Solid operating capacity"),
        (5.9, "Adequate"),
        (4.5, "Adequate"),
        (4.4, "Weak investment signals"),
        (3.0, "Weak investment signals"),
        (2.9, "Capability concerns"),
        (0.0, "Capability concerns"),
    ])
    def test_verdict(self, score: float, expected: str):
        assert _verdict(score) == expected


class TestFusionWeights:
    def test_60_40_formula(self):
        l1 = _make_l1(l1_score=8.0)
        l2 = _make_l2(themes={k: (6.0, 0.8) for k in ["tech", "capacity", "esg", "governance"]})
        result = fuse(l1, l2)
        # 0.6 * 8.0 + 0.4 * 6.0 = 4.8 + 2.4 = 7.2
        assert result.combined_score == pytest.approx(7.2, rel=1e-4)

    def test_l1_weight_dominates_when_l2_is_neutral(self):
        l1_high = _make_l1(l1_score=9.0)
        l1_low  = _make_l1(l1_score=3.0)
        l2 = _make_l2(themes={k: (5.0, 0.8) for k in ["tech", "capacity", "esg", "governance"]})
        high = fuse(l1_high, l2).combined_score
        low  = fuse(l1_low,  l2).combined_score
        # high should be higher
        assert high > low
        # difference should be 60% of the L1 difference: 0.6 * 6 = 3.6
        assert (high - low) == pytest.approx(3.6, rel=1e-3)


# ─── Confidence guardrail ─────────────────────────────────────────────────────

class TestGuardrail:
    def _run_guardrail(self, l1: Layer1Output, l2: Layer2Output, base: float = 0.75):
        return _apply_guardrail(l1, l2, base)

    # ── 1. Data coverage ─────────────────────────────────────────────────────

    def test_thin_coverage_discounts_confidence(self):
        l1 = _make_l1(quarters=config.MIN_QUARTERS_FULL_CONFIDENCE - 1)
        l2 = _make_l2()
        conf, notes = self._run_guardrail(l1, l2, 0.8)
        assert conf < 0.8
        assert any("Thin data coverage" in n for n in notes)

    def test_full_coverage_no_discount(self):
        l1 = _make_l1(quarters=config.MIN_QUARTERS_FULL_CONFIDENCE)
        l2 = _make_l2()
        conf, notes = self._run_guardrail(l1, l2, 0.8)
        assert not any("Thin data coverage" in n for n in notes)

    def test_missing_capex_discounts_confidence(self):
        l1 = _make_l1(capex_found=False)
        l2 = _make_l2()
        conf, notes = self._run_guardrail(l1, l2, 0.8)
        assert conf < 0.8
        assert any("Capex data absent" in n for n in notes)

    # ── 2. Sparse RAG ─────────────────────────────────────────────────────────

    def test_sparse_rag_discounts_confidence(self):
        l1 = _make_l1()
        l2 = _make_l2(rag_counts={"tech": 1, "capacity": 5, "esg": 5, "governance": 5})
        conf, notes = self._run_guardrail(l1, l2, 0.8)
        assert conf < 0.8
        assert any("Thin RAG for theme 'tech'" in n for n in notes)

    def test_all_themes_thin_rag_multiple_discounts(self):
        l1 = _make_l1()
        l2 = _make_l2(rag_counts={k: 0 for k in ["tech", "capacity", "esg", "governance"]})
        conf, notes = self._run_guardrail(l1, l2, 0.8)
        thin_notes = [n for n in notes if "Thin RAG" in n]
        assert len(thin_notes) == 4   # one per theme
        assert conf < 0.8 - 3 * config.GUARDRAIL_THIN_RAG_DISCOUNT  # at least 3 applied

    # ── 3. Flag / narrative conflicts ─────────────────────────────────────────

    def test_rd_intensifying_conflicts_with_low_tech_score(self):
        l1 = _make_l1(flags=["R&D_INTENSIFYING"])
        l2 = _make_l2(themes={
            "tech": (3.0, 0.8),   # < 4 → conflict
            "capacity": (6.0, 0.8), "esg": (5.0, 0.7), "governance": (6.0, 0.7),
        })
        conf, notes = self._run_guardrail(l1, l2, 0.8)
        assert any("R&D_INTENSIFYING" in n for n in notes)
        assert conf < 0.8

    def test_rd_intensifying_no_conflict_when_tech_score_is_adequate(self):
        l1 = _make_l1(flags=["R&D_INTENSIFYING"])
        l2 = _make_l2(themes={
            "tech": (5.0, 0.8),   # >= 4 → no conflict
            "capacity": (6.0, 0.8), "esg": (5.0, 0.7), "governance": (6.0, 0.7),
        })
        _, notes = self._run_guardrail(l1, l2, 0.8)
        assert not any("R&D_INTENSIFYING" in n for n in notes)

    def test_capex_strong_conflicts_with_low_capacity_score(self):
        l1 = _make_l1(flags=["CAPEX_REINVESTMENT_STRONG"])
        l2 = _make_l2(themes={
            "capacity": (3.0, 0.8),  # < 4 → conflict
            "tech": (7.0, 0.8), "esg": (5.0, 0.7), "governance": (6.0, 0.7),
        })
        conf, notes = self._run_guardrail(l1, l2, 0.8)
        assert any("CAPEX_REINVESTMENT_STRONG" in n for n in notes)
        assert conf < 0.8

    def test_capex_light_conflicts_with_high_capacity_score(self):
        l1 = _make_l1(flags=["CAPEX_LIGHT_BUSINESS"])
        l2 = _make_l2(themes={
            "capacity": (9.0, 0.8),  # > 8 → conflict
            "tech": (7.0, 0.8), "esg": (5.0, 0.7), "governance": (6.0, 0.7),
        })
        conf, notes = self._run_guardrail(l1, l2, 0.8)
        assert any("CAPEX_LIGHT_BUSINESS" in n for n in notes)
        assert conf < 0.8

    def test_no_flag_no_conflict_notes(self):
        l1 = _make_l1(flags=[])
        l2 = _make_l2()
        _, notes = self._run_guardrail(l1, l2, 0.8)
        conflict_notes = [n for n in notes if "flag fired" in n.lower()]
        assert len(conflict_notes) == 0

    # ── 4. Suspect high scores ────────────────────────────────────────────────

    def test_high_score_low_confidence_triggers_discount(self):
        l1 = _make_l1()
        l2 = _make_l2(themes={
            "tech": (8.5, 0.2),   # score > 7 AND confidence < 0.4 → suspect
            "capacity": (6.0, 0.7), "esg": (5.0, 0.6), "governance": (6.0, 0.7),
        })
        conf, notes = self._run_guardrail(l1, l2, 0.8)
        assert any("high score" in n and "very low LLM confidence" in n for n in notes)
        assert conf < 0.8

    def test_high_score_adequate_confidence_no_discount(self):
        l1 = _make_l1()
        l2 = _make_l2(themes={
            "tech": (8.5, 0.7),   # score > 7 but confidence OK → no suspect note
            "capacity": (6.0, 0.7), "esg": (5.0, 0.6), "governance": (6.0, 0.7),
        })
        _, notes = self._run_guardrail(l1, l2, 0.8)
        assert not any("very low LLM confidence" in n for n in notes)

    # ── 5. Guardrail never raises confidence ─────────────────────────────────

    def test_guardrail_never_raises_confidence(self):
        """No combination of inputs should result in confidence > base."""
        l1 = _make_l1(flags=[], quarters=20, capex_found=True)
        l2 = _make_l2(rag_counts={k: 10 for k in ["tech", "capacity", "esg", "governance"]})
        base = 0.6
        conf, _ = self._run_guardrail(l1, l2, base)
        assert conf <= base

    def test_guardrail_floored_at_0_1(self):
        """Extreme discounts must not push confidence below 0.1."""
        # Force every possible discount
        l1 = _make_l1(
            flags=["R&D_INTENSIFYING", "CAPEX_REINVESTMENT_STRONG", "CAPEX_LIGHT_BUSINESS"],
            quarters=1,
            capex_found=False,
        )
        # Thin RAG everywhere, high scores with low confidence
        l2 = _make_l2(
            themes={k: (9.0, 0.1) for k in ["tech", "capacity", "esg", "governance"]},
            rag_counts={k: 0 for k in ["tech", "capacity", "esg", "governance"]},
            industry_count=0,
        )
        conf, _ = self._run_guardrail(l1, l2, 0.5)
        assert conf >= 0.10

    # ── 6. Full fuse() integration ────────────────────────────────────────────

    def test_fuse_produces_valid_output(self):
        l1 = _make_l1(flags=["CAPEX_LIGHT_BUSINESS"], l1_score=5.0)
        l2 = _make_l2()
        result = fuse(l1, l2)

        assert isinstance(result, FusionOutput)
        assert 0.0 <= result.combined_score <= 10.0
        assert 0.1 <= result.final_confidence <= 1.0
        assert result.final_confidence <= result.base_confidence
        assert result.verdict in {
            "Strong execution engine", "Solid operating capacity",
            "Adequate", "Weak investment signals", "Capability concerns",
        }

    def test_fuse_heavy_investor_scores_high(self):
        """A company with R&D_INTENSIFYING + CAPEX_REINVESTMENT_STRONG should score > 7."""
        l1 = _make_l1(flags=["R&D_INTENSIFYING", "CAPEX_REINVESTMENT_STRONG"], l1_score=9.0)
        l2 = _make_l2(themes={k: (8.0, 0.8) for k in ["tech", "capacity", "esg", "governance"]})
        result = fuse(l1, l2)
        assert result.combined_score > 7.0

    def test_fuse_asset_light_scores_lower(self):
        """Asset-light, flat R&D, neutral L2 → combined score in the middle range."""
        l1 = _make_l1(flags=["CAPEX_LIGHT_BUSINESS"], l1_score=4.5)
        l2 = _make_l2(themes={k: (5.0, 0.7) for k in ["tech", "capacity", "esg", "governance"]})
        result = fuse(l1, l2)
        # 0.6 * 4.5 + 0.4 * 5.0 = 2.7 + 2.0 = 4.7
        assert 4.0 <= result.combined_score <= 6.0
