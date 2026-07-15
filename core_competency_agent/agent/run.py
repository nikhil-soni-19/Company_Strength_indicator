"""
Agent 6 — Core Competency runner.
Orchestrates: Layer 1 → RAG retrieval → Layer 2 → final fusion → MoatVerdict.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.query_parser import parse_ticker
from layer1.bundle import build_layer1
from layer2.ten_k_moat_retrieval import retrieve_moat_context
from layer2.tavily_competitive import fetch_competitive_context
from layer2.llm_interpreter import interpret
from scoring.final_score import fuse
from models.intent import MoatIntent
from models.moat_verdict import MoatVerdict

RAG_ENABLED = True  # set False to skip ontology DB


def run(query: str) -> tuple[MoatVerdict, object, dict]:
    """
    Full Agent 6 pipeline.
    Returns (MoatVerdict, Layer1MoatOutput, raw_computed_dict).
    """
    print(f"\n{'='*60}")
    print(f"Agent 6 — Core Competency: {query}")
    print("=" * 60)

    # Step 0: Parse query
    print("\n[Step 0] Parsing query...")
    ticker = parse_ticker(query)
    if not ticker:
        raise ValueError(f"Could not extract a ticker from: '{query}'")
    print(f"  Ticker: {ticker}")

    intent = MoatIntent(ticker=ticker, raw_query=query)

    # Step 1: Layer 1 — financial signals
    l1, computed = build_layer1(ticker)
    # Inject insider_pct into computed so LLM sees it
    computed["insider_pct"] = l1.insider_ownership_pct

    # Step 2: RAG retrieval — moat claims + threats from 10-K
    fiscal_year = date.today().year - 1
    moat_context: dict = {}
    if RAG_ENABLED:
        print(f"\n[Step 2] Retrieving 10-K moat context for FY{fiscal_year}...")
        moat_context = retrieve_moat_context(ticker, fiscal_year, k_per_section=4)
        for k, v in moat_context.items():
            print(f"  {k}: {len(v)} chunks")
    else:
        moat_context = {"moat_claims": [], "threats": [], "transcript_moat": []}
        print("\n[Step 2] RAG disabled — skipping 10-K retrieval")

    # Step 3: Tavily competitive context
    print("\n[Step 3] Fetching competitive context (Tavily)...")
    company_name = _get_company_name(ticker)
    competitive_news = fetch_competitive_context(
        ticker, company_name, peers=l1.peers, n=9
    )
    print(f"  Competitive articles: {len(competitive_news)}")

    # Step 4: Layer 2 — LLM adversarial interpretation
    print("\n[Step 4] Running Layer 2 LLM (adversarial moat analysis)...")
    l2_result = interpret(
        ticker=ticker,
        peers=l1.peers,
        l1_computed=computed,
        flags=l1.flags,
        moat_context=moat_context,
        competitive_news=competitive_news,
    )
    l2_score = l2_result["moat_score_l2"]
    print(f"  L2 Moat Score: {l2_score}/10")
    print(f"  Direction:     {l2_result['direction']}")
    print(f"  Narrative vs Numbers: {l2_result['narrative_vs_numbers']}")

    # Step 5: Fuse Layer 1 + Layer 2
    print("\n[Step 5] Fusing Layer 1 + Layer 2 (55/45)...")
    fusion = fuse(
        l1_score=l1.score,
        l2_score=l2_score,
        narrative_vs_numbers=l2_result["narrative_vs_numbers"],
    )
    moat_score = fusion["moat_score"]
    print(f"  Final Moat Score: {moat_score}/100")
    if fusion["conflict_penalty_applied"]:
        print("  ⚠ Conflict penalty applied — narrative contradicts numbers")

    # Assemble verdict
    verdict = MoatVerdict(
        ticker=ticker,
        period=l1.periods[-1] if l1.periods else "unknown",
        moat_score=moat_score,
        direction=l2_result["direction"],
        layer1_score=l1.score,
        layer2_score=l2_score,
        key_sources=l2_result.get("key_sources", []),
        key_threats=l2_result.get("key_threats", []),
        flags=l1.flags,
        margin_premium_sustained="MARGIN_PREMIUM_SUSTAINED" in l1.flags,
        roic_elite="ROIC_ELITE" in l1.flags,
        insider_conviction_high="INSIDER_CONVICTION_HIGH" in l1.flags,
        claimed_moat_sources=l2_result.get("claimed_moat_sources", []),
        narrative_vs_numbers=l2_result["narrative_vs_numbers"],
        conflict_description=l2_result.get("conflict_description"),
        bull_case=l2_result.get("bull_case"),
        bear_case=l2_result.get("bear_case"),
        reasoning=l2_result.get("reasoning", ""),
        sources_cited=l2_result.get("sources_cited", []),
        passages_used=sum(len(v) for v in moat_context.values()),
    )

    return verdict, l1, computed


def _get_company_name(ticker: str) -> str:
    """Resolve full company name — DB first, yfinance fallback."""
    try:
        import psycopg2
        import os
        conn = psycopg2.connect(os.environ.get(
            "DATABASE_URL",
            "postgresql://neondb_owner:npg_BgdTyxpXW3q4@ep-bitter-boat-aq1v8xns.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require"
        ))
        cur = conn.cursor()
        cur.execute("SELECT name FROM ticker_metadata WHERE ticker = %s LIMIT 1", (ticker.upper(),))
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker
