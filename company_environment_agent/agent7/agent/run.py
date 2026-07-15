"""Agent 7 — Company Environment runner."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from layer1.bundle import build_bundle
from layer2.tavily_pestel import pestel_news
from layer2.risk_factor_retrieval import retrieve
from layer2.ten_k_retrieval import retrieve_pestel_risk_factors
from layer2.llm_interpreter import interpret
from scoring.quant_score import quant_score
from scoring.pestel_score import all_pestel_quant_scores
from scoring.final_score import combine
from db.connection import get_conn


def _ensure_pestel_column(conn) -> bool:
    """Add pestel_scores column if it doesn't exist. Returns True if column is present."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                ALTER TABLE environment_runs
                ADD COLUMN IF NOT EXISTS pestel_scores JSONB
                """
            )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"  [WARN] Could not add pestel_scores column: {e}")
        return False


def _persist(result: dict) -> None:
    conn = get_conn()
    try:
        has_pestel_col = _ensure_pestel_column(conn)

        if has_pestel_col:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO environment_runs
                      (run_id, ticker, as_of_date, layer1_bundle, quant_score,
                       qual_score, environment_score, direction, flags, narrative,
                       pestel_scores)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    (
                        result["run_id"],
                        result["ticker"],
                        result["as_of_date"],
                        json.dumps(result.get("layer1_bundle"), default=str),
                        result.get("quant_score"),
                        result.get("qual_score"),
                        result.get("environment_score"),
                        result.get("direction"),
                        result.get("flags", []),
                        result.get("narrative"),
                        json.dumps(result.get("pestel_scores"), default=str),
                    ),
                )
        else:
            # Fallback: insert without pestel_scores
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO environment_runs
                      (run_id, ticker, as_of_date, layer1_bundle, quant_score,
                       qual_score, environment_score, direction, flags, narrative)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    (
                        result["run_id"],
                        result["ticker"],
                        result["as_of_date"],
                        json.dumps(result.get("layer1_bundle"), default=str),
                        result.get("quant_score"),
                        result.get("qual_score"),
                        result.get("environment_score"),
                        result.get("direction"),
                        result.get("flags", []),
                        result.get("narrative"),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def run_agent(
    ticker: str,
    as_of_date: str | date | None = None,
    lookback_days: int = 126,
) -> dict:
    """
    Full PESTEL pipeline:
    1) Build Layer 1 bundle (including PESTEL sub-bundle)
    2) Compute PESTEL quant sub-scores + overall quant_score
    3) Fetch PESTEL-structured news (6 Tavily searches, cached 24h)
    4) Retrieve 10-K risk factor excerpts
    5) Call LLM (returns qual scores per PESTEL dimension)
    6) Combine quant + qual into environment_score
    7) Persist to environment_runs
    8) Return output contract dict
    """
    if isinstance(as_of_date, str):
        as_of_date = date.fromisoformat(as_of_date)
    if as_of_date is None:
        as_of_date = date.today()

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    bundle = build_bundle(ticker, as_of_date, lookback_days)
    flags  = bundle.get("flags", [])
    qs     = quant_score(bundle)
    pestel_quant = all_pestel_quant_scores(bundle)

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    sector = bundle.get("sector") or ""
    news   = pestel_news(sector, ticker, n_per_dim=3)

    fiscal_year = as_of_date.year - 1

    # Per-dimension 10-K excerpts from ontology DB (primary path)
    pestel_excerpts = retrieve_pestel_risk_factors(ticker, fiscal_year, k_per_dim=3)

    # Flat excerpt list: flatten pestel_excerpts or fall back to local table
    flat_excerpts = [chunk for chunks in pestel_excerpts.values() for chunk in chunks]
    if not flat_excerpts:
        flat_excerpts = retrieve(
            ticker,
            fiscal_year,
            query=f"{sector} regulatory risk environment political legal climate technology",
            k=5,
        )

    llm_result = interpret(bundle, flags, news, flat_excerpts, pestel_excerpts=pestel_excerpts)

    qual_score_val  = llm_result.get("qual_score", 50)
    pestel_qual     = llm_result.get("pestel_scores", {})

    # ── Combine ──────────────────────────────────────────────────────────────
    combined = combine(qs, float(qual_score_val))

    # Merge quant + qual PESTEL scores into a unified view per dimension
    pestel_scores_combined = {
        dim: {
            "quant": pestel_quant.get(dim, 50.0),
            "qual":  pestel_qual.get(dim, 50),
            # Simple average — LLM weights by materiality in its overall qual_score
            "combined": round(0.5 * pestel_quant.get(dim, 50.0) + 0.5 * pestel_qual.get(dim, 50), 2),
        }
        for dim in ("P", "E", "S", "T", "En", "L")
    }

    result = {
        "run_id":            str(uuid.uuid4()),
        "ticker":            ticker,
        "as_of_date":        as_of_date.isoformat(),
        "environment_score": combined["environment_score"],
        "direction":         combined["direction"],
        "quant_score":       qs,
        "qual_score":        qual_score_val,
        # PESTEL dimension breakdown
        "pestel_scores":     pestel_scores_combined,
        "flags":             flags,
        "layer1_bundle":     bundle,
        "narrative_by_dim":  llm_result.get("narrative_by_dim", {}),
        "narrative":         llm_result.get("narrative", ""),
        "key_tailwinds":     llm_result.get("key_tailwinds", []),
        "key_risks":         llm_result.get("key_risks", []),
        "evidence": {
            "news_by_pestel_dim": news,
            "risk_factor_excerpts_by_dim": pestel_excerpts,
            "risk_factor_excerpts": flat_excerpts,
        },
    }

    _persist(result)
    return result
