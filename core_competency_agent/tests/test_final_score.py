"""Unit tests for final score fusion."""
from scoring.final_score import fuse, score_to_label


def test_fuse_consistent():
    result = fuse(8.0, 7.0, "consistent")
    expected = round((8.0 * 0.55 + 7.0 * 0.45) * 10, 1)
    assert result["moat_score"] == expected
    assert not result["conflict_penalty_applied"]


def test_fuse_conflict_penalty():
    r_no_conflict  = fuse(7.0, 7.0, "consistent")
    r_with_conflict = fuse(7.0, 7.0, "conflict")
    assert r_with_conflict["moat_score"] < r_no_conflict["moat_score"]
    assert r_with_conflict["conflict_penalty_applied"]


def test_fuse_clamped_0_100():
    # Very high scores should stay at 100
    result = fuse(10.0, 10.0, "consistent")
    assert result["moat_score"] <= 100.0
    # Very low scores should stay at 0
    result2 = fuse(0.0, 0.0, "conflict")
    assert result2["moat_score"] >= 0.0


def test_score_labels():
    assert score_to_label(80) == "STRONG MOAT"
    assert score_to_label(55) == "MODERATE MOAT"
    assert score_to_label(35) == "NARROW MOAT"
    assert score_to_label(15) == "NO MOAT / CYCLICAL"
