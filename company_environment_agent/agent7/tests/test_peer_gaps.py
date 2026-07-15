import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from layer1.peer_gaps import compute_margin, peer_gaps, ttm_revenue_growth


def _fund(ttm_now=None, ttm_prior=None, latest=None):
    return {
        "ttm_now":  ttm_now or [100, 100, 100, 100],
        "ttm_prior": ttm_prior or [90, 90, 90, 90],
        "latest": latest or {},
    }


# ── compute_margin ─────────────────────────────────────────────────────────────

def test_nim_formula():
    row = {"net_interest_income": 10.0, "avg_earning_assets": 200.0}
    assert abs(compute_margin(row, "nim") - 0.05) < 1e-9


def test_gross_margin():
    row = {"gross_profit": 30.0, "revenue": 100.0}
    assert abs(compute_margin(row, "gross_margin") - 0.30) < 1e-9


def test_ffo_margin():
    row = {"ffo": 20.0, "revenue": 100.0}
    assert abs(compute_margin(row, "ffo_margin") - 0.20) < 1e-9


def test_operating_margin():
    row = {"operating_income": 15.0, "revenue": 100.0}
    assert abs(compute_margin(row, "operating_margin") - 0.15) < 1e-9


def test_ebitda_margin():
    row = {"ebitda": 25.0, "revenue": 100.0}
    assert abs(compute_margin(row, "ebitda_margin") - 0.25) < 1e-9


def test_nim_missing_returns_none():
    assert compute_margin({}, "nim") is None


# ── ttm_revenue_growth ─────────────────────────────────────────────────────────

def test_ttm_growth():
    now   = [110, 110, 110, 110]
    prior = [100, 100, 100, 100]
    assert abs(ttm_revenue_growth(now, prior) - 0.10) < 1e-9


def test_ttm_growth_insufficient():
    assert ttm_revenue_growth([100, 100], [100, 100]) is None


# ── peer_gaps ──────────────────────────────────────────────────────────────────

def test_peer_gaps_bank_nim():
    company = _fund(
        ttm_now=[110]*4, ttm_prior=[100]*4,
        latest={"net_interest_income": 10, "avg_earning_assets": 200},
    )
    peer = _fund(
        ttm_now=[105]*4, ttm_prior=[100]*4,
        latest={"net_interest_income": 8, "avg_earning_assets": 200},
    )
    result = peer_gaps(company, [peer], "nim")
    assert result["margin_metric"] == "nim"
    assert result["margin_company"] is not None
    # company NIM = 0.05, peer NIM = 0.04 → gap > 0
    assert result["margin_gap"] > 0


def test_peer_gaps_tech_gross_margin():
    company = _fund(
        ttm_now=[120]*4, ttm_prior=[100]*4,
        latest={"gross_profit": 50, "revenue": 100},
    )
    peer = _fund(
        ttm_now=[110]*4, ttm_prior=[100]*4,
        latest={"gross_profit": 40, "revenue": 100},
    )
    result = peer_gaps(company, [peer], "gross_margin")
    assert abs(result["margin_company"] - 0.50) < 1e-9


def test_peer_gaps_reit_ffo():
    company = _fund(
        ttm_now=[100]*4, ttm_prior=[90]*4,
        latest={"ffo": 20, "revenue": 100},
    )
    result = peer_gaps(company, [], "ffo_margin")
    assert result["margin_company"] is not None
    assert result["margin_peer_median"] is None
