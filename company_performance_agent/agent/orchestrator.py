"""
Orchestrator: runs the full pipeline for a query.
Step 0 → Layer 1 → RAG → Layer 2 → FinalVerdict
"""
from agent.router import parse_query
from agent.layer1.fetcher import fetch_financials
from agent.layer1.compute import compute_all
from agent.layer1.flags import emit_flags
from agent.layer1.scorer import compute_score
from agent.rag.retriever import retrieve
from agent.layer2.reasoner import reason
from models.layer1_output import Layer1Output
from models.verdict import FinalVerdict


def run(query: str) -> tuple:
    """Run the full Agent 10 pipeline for a natural language query."""

    print(f"\n{'='*60}")
    print(f"Agent 10 — Processing: {query}")
    print('='*60)

    # Step 0: Parse query
    print("\n[Step 0] Parsing query...")
    intent = parse_query(query)
    print(f"  Ticker: {intent.ticker}")
    print(f"  Hypothesis: {intent.hypothesis}")
    print(f"  RAG keywords: {intent.rag_keywords}")

    # Step 1: Fetch data
    print("\n[Step 1] Fetching data from yfinance...")
    raw = fetch_financials(intent.ticker, n_quarters=8)
    print(f"  Fetched {raw['n_quarters']} quarters: {raw['periods'][0]} → {raw['periods'][-1]}")

    # Step 2: Layer 1 computations
    print("\n[Step 2] Running Layer 1 computations...")
    computed = compute_all(raw)
    flags = emit_flags(computed)
    score = compute_score(computed)
    print(f"  Layer 1 Score: {score}/10")
    print(f"  Flags: {flags}")

    # Assemble Layer1Output
    l1 = Layer1Output(
        ticker=intent.ticker,
        period_latest=raw["periods"][-1],
        periods=raw["periods"],
        revenue=raw["revenue"],
        opex=raw["opex"],
        gross_profit=raw["gross_profit"],
        operating_income=raw["op_income"],
        net_income=raw["net_income"],
        ocf=raw["ocf"],
        capex=raw["capex"],
        rev_yoy_pct=computed["rev_yoy_pct"],
        opex_yoy_pct=computed["opex_yoy_pct"],
        ol_delta=computed["ol_delta"],
        rev_slope=computed["rev_slope"],
        op_margin_slope=computed["op_margin_slope"],
        gross_margin_slope=computed["gross_margin_slope"],
        ol_slope=computed["ol_slope"],
        gross_margin=computed["gross_margin_latest"],
        op_margin=computed["op_margin_latest"],
        net_margin=computed["net_margin_latest"],
        fcf=computed["fcf"],
        fcf_ni_ratio=computed["fcf_ni_ratio"],
        ccc_delta=computed.get("ccc_delta"),
        rev_accel=computed["rev_accel"],
        ol_consistency=computed["ol_consistency"],
        score=score,
        flags=flags,
    )

    # Step 3: RAG retrieval
    print("\n[Step 3] RAG retrieval...")
    rag = retrieve(intent, flags)
    print(f"  RAG enabled: {rag.rag_enabled} | Passages: {len(rag.passages)}")

    # Step 4: Layer 2 reasoning
    print("\n[Step 4] Layer 2 LLM reasoning...")
    verdict = reason(intent, l1, rag)
    print(f"  Execution Score: {verdict.execution_score}/10")
    print(f"  Credibility Score: {verdict.credibility_score}/10")
    print(f"  Direction: {verdict.direction}")
    print(f"  Verdict: {verdict.verdict}")

    return verdict, l1, rag, computed
