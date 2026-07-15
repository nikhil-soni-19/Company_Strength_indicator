"""Orchestrate all Layer 1 computations into a single bundle dict."""
from __future__ import annotations

from layer1.data_loader import (
    fetch_company_8q,
    fetch_peer_latest,
    fetch_insider_ownership,
    detect_leadership_change,
    get_peers,
)
from layer1.compute import compute_all
from layer1.flags import emit_flags
from layer1.scorer import compute_score
from models.layer1_output import Layer1MoatOutput


def build_layer1(ticker: str, n_quarters: int = 8) -> tuple[Layer1MoatOutput, dict]:
    """
    Full Layer 1 pipeline for a ticker.
    Returns (Layer1MoatOutput, raw_computed_dict).
    """
    print(f"\n[Layer 1] Fetching company data for {ticker}...")
    company = fetch_company_8q(ticker, n_quarters)
    print(f"  Quarters: {company['periods'][0]} → {company['periods'][-1]}")

    peers = get_peers(ticker)
    print(f"  Peers ({len(peers)}): {peers}")

    peer_data = []
    for p in peers:
        print(f"  Fetching peer {p}...")
        d = fetch_peer_latest(p)
        if d:
            peer_data.append(d)

    print(f"  Peer data fetched: {len(peer_data)}/{len(peers)}")

    print("[Layer 1] Computing signals...")
    computed = compute_all(company, peer_data)

    print("[Layer 1] Emitting flags...")
    flags = emit_flags(
        avg_gross_margin_spread=computed["avg_gross_margin_spread"],
        gross_margin_spread=computed["gross_margin_spread"],
        avg_op_margin_spread=computed["avg_op_margin_spread"],
        roic_spread=computed.get("roic_spread"),
        roic_company=computed.get("roic_company"),
        avg_fcf_margin_spread=computed.get("avg_fcf_margin_spread"),
        insider_pct=None,  # filled in below
        gross_margin_cv=computed["gross_margin_cv"],
    )

    print("[Layer 1] Fetching insider ownership...")
    ownership = fetch_insider_ownership(ticker)
    insider_pct = ownership.get("insider_pct")
    institutional_top10 = ownership.get("institutional_top10")

    # Re-emit flags with insider data
    flags = emit_flags(
        avg_gross_margin_spread=computed["avg_gross_margin_spread"],
        gross_margin_spread=computed["gross_margin_spread"],
        avg_op_margin_spread=computed["avg_op_margin_spread"],
        roic_spread=computed.get("roic_spread"),
        roic_company=computed.get("roic_company"),
        avg_fcf_margin_spread=computed.get("avg_fcf_margin_spread"),
        insider_pct=insider_pct,
        gross_margin_cv=computed["gross_margin_cv"],
    )
    print(f"  Flags: {flags}")

    print("[Layer 1] Detecting leadership changes...")
    leadership = detect_leadership_change(ticker)

    score = compute_score(computed, flags)
    print(f"  Layer 1 Score: {score}/10")

    l1 = Layer1MoatOutput(
        ticker=ticker,
        peers=peers,
        periods=company["periods"],

        gross_margin_series=computed["gross_margin_series"],
        gross_margin_peer_median=computed["gross_margin_peer_median"],
        gross_margin_spread=computed["gross_margin_spread"],
        avg_gross_margin_spread=computed["avg_gross_margin_spread"],

        op_margin_series=computed["op_margin_series"],
        op_margin_peer_median=computed["op_margin_peer_median"],
        op_margin_spread=computed["op_margin_spread"],
        avg_op_margin_spread=computed["avg_op_margin_spread"],

        roic_company=computed.get("roic_company"),
        roic_peer_median=computed.get("roic_peer_median"),
        roic_spread=computed.get("roic_spread"),

        roe_company=computed.get("roe_company"),
        roe_peer_median=computed.get("roe_peer_median"),

        fcf_margin_series=computed["fcf_margin_series"],
        fcf_margin_peer_median=computed.get("fcf_margin_peer_median"),
        avg_fcf_margin_spread=computed.get("avg_fcf_margin_spread"),

        gross_margin_cv=computed["gross_margin_cv"],
        op_margin_cv=computed["op_margin_cv"],

        insider_ownership_pct=insider_pct,
        institutional_concentration_top10=institutional_top10,

        leadership_change_detected=leadership["detected"],
        leadership_change_description=leadership.get("description"),

        score=score,
        flags=flags,
    )

    return l1, computed
