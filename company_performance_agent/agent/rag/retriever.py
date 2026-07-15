"""
RAG Retriever — Neon PostgreSQL

Pulls from all available data sources for MSFT and AAPL:
  1. ontology.sec_filings          — SEC filing text (10-K, 10-Q)
  2. company_performance.<t>_evt   — Earnings call transcripts / events
  3. company_performance.<t>_fa_esg — Employee headcount + ESG metrics
  4. company_performance.<t>_fa_geo — Geographic revenue breakdown
  5. company_performance.<t>_fa_prod — Product segment breakdown
  6. ontology.financial_facts       — Structured financial line items
  7. ontology.estimate_consensus    — Analyst estimates vs actuals
  8. ontology.earnings_surprise     — Beat/miss history (credibility scoring)
"""

import os
from typing import Optional
from dotenv import load_dotenv

from models.intent import QueryIntent
from models.rag_output import RAGOutput, RAGPassage, GuidanceMatch

load_dotenv()

RAG_ENABLED = os.getenv("RAG_ENABLED", "false").lower() == "true"
NEON_DSN    = os.getenv("DATABASE_URL_ONTOLOGY_LAB") or os.getenv("NEON_DATABASE_URL")
TOP_K       = int(os.getenv("NEON_TOP_K", "12"))

_col_cache: dict[str, set] = {}


# ── Public entry point ────────────────────────────────────────────────────────

def retrieve(intent: QueryIntent, layer1_flags: list[str]) -> RAGOutput:
    if RAG_ENABLED:
        if not NEON_DSN:
            raise EnvironmentError("RAG_ENABLED=true but DATABASE_URL_ONTOLOGY_LAB is not set.")
        return _retrieve_from_neon(intent, layer1_flags)
    return _retrieve_stub(intent, layer1_flags)


def _retrieve_stub(intent: QueryIntent, layer1_flags: list[str]) -> RAGOutput:
    return RAGOutput(ticker=intent.ticker, passages=[], guidance_matches=[],
                     credibility_track_record=0.5, rag_enabled=False)


# ── Connection helpers ────────────────────────────────────────────────────────

def _connect():
    import psycopg2
    return psycopg2.connect(NEON_DSN)

def _cols(conn, schema: str, table: str) -> set:
    key = f"{schema}.{table}"
    if key in _col_cache:
        return _col_cache[key]
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s;
        """, [schema, table])
        result = {r[0] for r in cur.fetchall()}
    _col_cache[key] = result
    return result

def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s LIMIT 1;
        """, [schema, table])
        return cur.fetchone() is not None

def _pick(cols: set, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None

def _keyword_score(text: str, keywords: list) -> float:
    if not keywords or not text:
        return 0.65
    hits = sum(1 for kw in keywords if kw.lower() in text.lower())
    return round(0.60 + (hits / len(keywords)) * 0.35, 3)

def _safe_run(fn, label):
    """Run a retrieval function, return [] on any error."""
    try:
        return fn()
    except Exception as e:
        print(f"  [RAG] {label} error: {e}")
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _retrieve_from_neon(intent: QueryIntent, layer1_flags: list[str]) -> RAGOutput:
    ticker   = intent.ticker.upper()
    keywords = intent.rag_keywords
    conn     = _connect()
    passages: list[RAGPassage] = []

    try:
        # ── Source 1: SEC filings ─────────────────────────────────────────────
        sec = _safe_run(lambda: _query_sec_filings(conn, ticker, keywords), "sec_filings")
        passages += sec
        print(f"  [RAG] SEC filings     : {len(sec)} passages")

        # ── Source 2: Earnings call transcripts ───────────────────────────────
        ec = _safe_run(lambda: _query_earnings_calls(conn, ticker, keywords), "earnings_calls")
        passages += ec
        print(f"  [RAG] Earnings calls  : {len(ec)} passages")

        # ── Source 3: Employee / ESG data ─────────────────────────────────────
        emp = _safe_run(lambda: _query_employee_data(conn, ticker), "employee_data")
        passages += emp
        print(f"  [RAG] Employee/ESG    : {len(emp)} passages")

        # ── Source 4: Geographic breakdown ────────────────────────────────────
        geo = _safe_run(lambda: _query_segment(conn, ticker, "fa_geo", "geographic"), "fa_geo")
        passages += geo
        print(f"  [RAG] Geographic      : {len(geo)} passages")

        # ── Source 5: Product segment breakdown ───────────────────────────────
        prod = _safe_run(lambda: _query_segment(conn, ticker, "fa_prod", "product"), "fa_prod")
        passages += prod
        print(f"  [RAG] Product segs    : {len(prod)} passages")

        # ── Source 6: Financial facts ─────────────────────────────────────────
        ff = _safe_run(lambda: _query_financial_facts(conn, ticker), "financial_facts")
        passages += ff
        print(f"  [RAG] Financial facts : {len(ff)} passages")

        # ── Source 7: Analyst consensus ───────────────────────────────────────
        cons = _safe_run(lambda: _query_estimate_consensus(conn, ticker), "estimate_consensus")
        passages += cons
        print(f"  [RAG] Analyst consenus: {len(cons)} passages")

        # Deduplicate, rank, keep top K
        seen: dict[str, RAGPassage] = {}
        for p in passages:
            if p.chunk_id not in seen or p.similarity_score > seen[p.chunk_id].similarity_score:
                seen[p.chunk_id] = p
        top = sorted(seen.values(), key=lambda x: -x.similarity_score)[:TOP_K]
        print(f"  [RAG] Total passages  : {len(top)} (after dedup)")

        # ── Source 8: Credibility from earnings surprise ──────────────────────
        guidance_matches, credibility = _earnings_credibility(conn, ticker)

        return RAGOutput(ticker=ticker, passages=top, guidance_matches=guidance_matches,
                         credibility_track_record=credibility, rag_enabled=True)

    finally:
        conn.close()


# ── Source 1: SEC filings ─────────────────────────────────────────────────────

def _query_sec_filings(conn, ticker: str, keywords: list) -> list[RAGPassage]:
    """10-K / 10-Q narrative text filtered by ticker and keywords."""
    if not _table_exists(conn, "ontology", "sec_filings"):
        return []

    cols       = _cols(conn, "ontology", "sec_filings")
    col_id     = _pick(cols, ["id", "filing_id", "chunk_id"])
    col_ticker = _pick(cols, ["ticker", "canonical_ticker", "symbol", "company_ticker"])
    col_text   = _pick(cols, ["text", "content", "body", "filing_text", "chunk_text", "excerpt"])
    col_period = _pick(cols, ["period", "period_end_date", "fiscal_period", "filed_date", "date"])
    col_type   = _pick(cols, ["filing_type", "source_type", "form_type", "type", "doc_type"])
    col_section= _pick(cols, ["section", "section_type", "part"])

    if not col_text:
        return []

    kw_clauses = [f"{col_text} ILIKE %s" for kw in keywords[:4]]
    kw_params  = [f"%{kw}%" for kw in keywords[:4]]
    kw_filter  = f"AND ({' OR '.join(kw_clauses)})" if kw_clauses else ""
    t_filter   = f"AND {col_ticker} ILIKE %s" if col_ticker else ""
    t_param    = [f"%{ticker}%"] if col_ticker else []

    id_expr    = f"{col_id}::TEXT" if col_id else "ROW_NUMBER() OVER ()::TEXT"
    type_expr  = f"{col_type}::TEXT" if col_type else "'sec_filing'"
    period_expr= f"{col_period}::TEXT" if col_period else "'unknown'"
    sec_expr   = f"{col_section}::TEXT" if col_section else "NULL::TEXT"
    order_expr = f"{col_period} DESC," if col_period else ""
    id_order   = f"{col_id} DESC" if col_id else "1"

    sql = f"""
        SELECT {id_expr}, {type_expr}, {period_expr}, {sec_expr},
               LEFT({col_text}, 900) AS text
        FROM ontology.sec_filings
        WHERE TRUE {t_filter} {kw_filter}
        ORDER BY {order_expr} {id_order}
        LIMIT {TOP_K};
    """

    with conn.cursor() as cur:
        cur.execute(sql, t_param + kw_params)
        rows = cur.fetchall()

    return [RAGPassage(
        chunk_id    = f"sec_{ticker}_{i}",
        ticker      = ticker,
        source_type = str(row[1]) if row[1] else "10-K/Q",
        period      = str(row[2]) if row[2] else "unknown",
        speaker     = None,
        section     = str(row[3]) if row[3] else "filing",
        text        = str(row[4]),
        similarity_score = _keyword_score(str(row[4]), keywords),
    ) for i, row in enumerate(rows)]


# ── Source 2: Earnings call transcripts ──────────────────────────────────────

def _query_earnings_calls(conn, ticker: str, keywords: list) -> list[RAGPassage]:
    """
    Pull from company_performance.<ticker>_evt.
    Prioritises rows that look like transcript/earnings call entries.
    Auto-detects available columns and builds the richest query possible.
    """
    table = f"{ticker.lower()}_evt"
    if not _table_exists(conn, "company_performance", table):
        return []

    cols       = _cols(conn, "company_performance", table)
    col_id     = _pick(cols, ["id", "event_id", "row_id"])
    col_period = _pick(cols, ["period", "date", "event_date", "quarter", "fiscal_period", "period_end"])
    col_type   = _pick(cols, ["event_type", "type", "category", "source_type", "doc_type"])
    col_speaker= _pick(cols, ["speaker", "speaker_name", "speaker_role", "executive", "analyst"])
    col_text   = _pick(cols, ["text", "content", "body", "transcript", "quote", "statement",
                               "description", "summary", "comment", "note", "excerpt"])
    col_section= _pick(cols, ["section", "segment", "part", "topic"])

    # If there's a dedicated text column use it, otherwise concat all non-system columns
    if col_text:
        text_expr = f"LEFT({col_text}, 900)"
    else:
        non_sys = sorted(c for c in cols if c not in ("id", "embedding", "created_at", "updated_at"))
        text_expr = "CONCAT_WS(' | ', " + ", ".join(
            f"'{c}: ' || COALESCE({c}::TEXT, '')" for c in non_sys[:10]
        ) + ")"

    id_expr     = f"{col_id}::TEXT" if col_id else "ROW_NUMBER() OVER ()::TEXT"
    type_expr   = f"{col_type}::TEXT" if col_type else "'earnings_call'"
    period_expr = f"{col_period}::TEXT" if col_period else "'unknown'"
    speaker_expr= f"{col_speaker}::TEXT" if col_speaker else "NULL::TEXT"
    section_expr= f"{col_section}::TEXT" if col_section else "NULL::TEXT"
    order_expr  = f"{col_period} DESC," if col_period else ""
    id_order    = f"{col_id} DESC" if col_id else "1"

    # Keyword filter if text col exists
    kw_filter  = ""
    kw_params  = []
    if col_text and keywords:
        clauses   = [f"{col_text} ILIKE %s" for kw in keywords[:3]]
        kw_params = [f"%{kw}%" for kw in keywords[:3]]
        kw_filter = f"AND ({' OR '.join(clauses)})"

    sql = f"""
        SELECT {id_expr}, {type_expr}, {period_expr}, {speaker_expr},
               {section_expr}, {text_expr}
        FROM company_performance.{table}
        WHERE TRUE {kw_filter}
        ORDER BY {order_expr} {id_order}
        LIMIT {TOP_K};
    """

    with conn.cursor() as cur:
        cur.execute(sql, kw_params)
        rows = cur.fetchall()

    passages = []
    for i, row in enumerate(rows):
        text = str(row[5]) if row[5] else ""
        if not text.strip():
            continue
        passages.append(RAGPassage(
            chunk_id     = f"ec_{ticker}_{i}",
            ticker       = ticker,
            source_type  = str(row[1]) if row[1] else "earnings_call",
            period       = str(row[2]) if row[2] else "unknown",
            speaker      = str(row[3]) if row[3] else None,
            section      = str(row[4]) if row[4] else "transcript",
            text         = text,
            similarity_score = _keyword_score(text, keywords),
        ))
    return passages


# ── Source 3: Employee / ESG data ─────────────────────────────────────────────

def _query_employee_data(conn, ticker: str) -> list[RAGPassage]:
    """
    Pull from company_performance.<ticker>_fa_esg.
    Explicitly surfaces employee headcount and workforce-related fields
    as labelled passages so the LLM can reason about workforce trends.
    """
    table = f"{ticker.lower()}_fa_esg"
    if not _table_exists(conn, "company_performance", table):
        return []

    cols       = _cols(conn, "company_performance", table)
    col_period = _pick(cols, ["period", "date", "fiscal_period", "period_end", "quarter", "year"])

    # Identify employee-related columns
    emp_cols = [c for c in cols if any(k in c.lower() for k in
                ["employee", "headcount", "workforce", "staff", "fte", "personnel", "hire"])]

    # All other ESG columns
    other_cols = sorted(c for c in cols if c not in emp_cols
                        and c not in ("id", "embedding", "created_at", "updated_at")
                        and c != col_period)

    all_data_cols = emp_cols + other_cols[:8]  # employees first, then rest
    if not all_data_cols:
        return []

    concat_expr = "CONCAT_WS(chr(10), " + ", ".join(
        f"'{c}: ' || COALESCE({c}::TEXT, 'N/A')" for c in all_data_cols
    ) + ")"

    period_select = f"{col_period}::TEXT" if col_period else "'unknown'"
    order_expr    = f"ORDER BY {col_period} DESC" if col_period else ""

    sql = f"""
        SELECT {period_select}, {concat_expr}
        FROM company_performance.{table}
        {order_expr}
        LIMIT 6;
    """

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    passages = []
    for i, row in enumerate(rows):
        period = str(row[0]) if row[0] else "unknown"
        body   = str(row[1]) if row[1] else ""
        if not body.strip():
            continue

        # Label it clearly so the LLM knows what this is
        label = "Employee & ESG data" if emp_cols else "ESG data"
        text  = f"{label} for {ticker} ({period}):\n{body}"

        passages.append(RAGPassage(
            chunk_id     = f"esg_{ticker}_{i}",
            ticker       = ticker,
            source_type  = "employee_esg",
            period       = period,
            speaker      = None,
            section      = "employee_headcount" if emp_cols else "esg",
            text         = text,
            similarity_score = 0.75,
        ))
    return passages


# ── Source 4 & 5: Segment breakdowns (geo / prod) ─────────────────────────────

def _query_segment(conn, ticker: str, suffix: str, label: str) -> list[RAGPassage]:
    """Generic segment table query for fa_geo and fa_prod."""
    table = f"{ticker.lower()}_{suffix}"
    if not _table_exists(conn, "company_performance", table):
        return []

    cols       = _cols(conn, "company_performance", table)
    col_period = _pick(cols, ["period", "date", "fiscal_period", "period_end", "quarter"])
    data_cols  = sorted(c for c in cols
                        if c not in ("id", "embedding", "created_at", "updated_at")
                        and c != col_period)[:12]

    if not data_cols:
        return []

    concat_expr = "CONCAT_WS(chr(10), " + ", ".join(
        f"'{c}: ' || COALESCE({c}::TEXT, 'N/A')" for c in data_cols
    ) + ")"
    period_sel  = f"{col_period}::TEXT" if col_period else "'unknown'"
    order_expr  = f"ORDER BY {col_period} DESC" if col_period else ""

    sql = f"""
        SELECT {period_sel}, {concat_expr}
        FROM company_performance.{table}
        {order_expr}
        LIMIT 4;
    """

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    passages = []
    for i, row in enumerate(rows):
        period = str(row[0]) if row[0] else "unknown"
        body   = str(row[1]) if row[1] else ""
        if not body.strip():
            continue
        passages.append(RAGPassage(
            chunk_id     = f"{suffix}_{ticker}_{i}",
            ticker       = ticker,
            source_type  = f"{label}_segment",
            period       = period,
            speaker      = None,
            section      = suffix,
            text         = f"{label.capitalize()} breakdown for {ticker} ({period}):\n{body}",
            similarity_score = 0.70,
        ))
    return passages


# ── Source 6: Financial facts ─────────────────────────────────────────────────

def _query_financial_facts(conn, ticker: str) -> list[RAGPassage]:
    """Structured financial line items grouped by period."""
    if not _table_exists(conn, "ontology", "financial_facts"):
        return []

    cols       = _cols(conn, "ontology", "financial_facts")
    col_ticker = _pick(cols, ["ticker", "canonical_ticker", "symbol"])
    col_period = _pick(cols, ["period", "period_end_date", "fiscal_period", "date"])
    col_metric = _pick(cols, ["metric", "concept", "line_item", "name", "label", "fact_name"])
    col_value  = _pick(cols, ["value", "value_numeric", "amount", "fact_value"])
    col_unit   = _pick(cols, ["unit", "currency"])

    if not col_metric or not col_value:
        return []

    t_filter  = f"WHERE {col_ticker} ILIKE %s" if col_ticker else "WHERE TRUE"
    t_param   = [f"%{ticker}%"] if col_ticker else []
    unit_sel  = f", {col_unit}::TEXT" if col_unit else ""
    order_exp = f"ORDER BY {col_period} DESC," if col_period else "ORDER BY"
    period_sel= f"COALESCE({col_period}::TEXT, 'N/A')" if col_period else "'N/A'"

    sql = f"""
        SELECT {period_sel}, {col_metric}::TEXT, {col_value}::TEXT {unit_sel}
        FROM ontology.financial_facts
        {t_filter}
        {order_exp} {col_metric}
        LIMIT 24;
    """

    with conn.cursor() as cur:
        cur.execute(sql, t_param)
        rows = cur.fetchall()

    if not rows:
        return []

    by_period: dict[str, list] = {}
    for row in rows:
        period = str(row[0])
        unit   = str(row[3]) if len(row) > 3 else ""
        by_period.setdefault(period, []).append(
            f"{row[1]}: {row[2]} {unit}".strip()
        )

    return [
        RAGPassage(
            chunk_id     = f"ff_{ticker}_{i}",
            ticker       = ticker,
            source_type  = "financial_facts",
            period       = period,
            speaker      = None,
            section      = "financials",
            text         = f"Financial facts for {ticker} ({period}):\n" + "\n".join(items),
            similarity_score = 0.72,
        )
        for i, (period, items) in enumerate(sorted(by_period.items(), reverse=True)[:4])
    ]


# ── Source 7: Analyst consensus ───────────────────────────────────────────────

def _query_estimate_consensus(conn, ticker: str) -> list[RAGPassage]:
    """
    Pull from ontology.estimate_consensus.
    Formats analyst estimates vs actuals as a passage so the LLM
    can assess whether management delivered on street expectations.
    """
    if not _table_exists(conn, "ontology", "estimate_consensus"):
        return []

    cols       = _cols(conn, "ontology", "estimate_consensus")
    col_ticker = _pick(cols, ["ticker", "canonical_ticker", "symbol"])
    col_period = _pick(cols, ["period", "period_end_date", "date", "quarter", "fiscal_period"])
    col_metric = _pick(cols, ["metric", "concept", "estimate_type", "name"])
    col_est    = _pick(cols, ["estimate", "consensus_estimate", "mean_estimate", "expected"])
    col_actual = _pick(cols, ["actual", "actual_value", "reported"])
    col_diff   = _pick(cols, ["surprise", "surprise_pct", "beat", "difference", "delta"])

    if not col_est:
        return []

    t_filter  = f"WHERE {col_ticker} ILIKE %s" if col_ticker else "WHERE TRUE"
    t_param   = [f"%{ticker}%"] if col_ticker else []
    order_exp = f"ORDER BY {col_period} DESC" if col_period else ""

    # Build a readable per-row summary
    parts = []
    if col_metric:  parts.append(f"{col_metric}::TEXT")
    if col_est:     parts.append(f"'Estimate: ' || {col_est}::TEXT")
    if col_actual:  parts.append(f"'Actual: '   || {col_actual}::TEXT")
    if col_diff:    parts.append(f"'Surprise: ' || {col_diff}::TEXT")
    row_expr = "CONCAT_WS('  |  ', " + ", ".join(parts) + ")" if parts else "'N/A'"
    period_sel = f"{col_period}::TEXT" if col_period else "'unknown'"

    sql = f"""
        SELECT {period_sel}, {row_expr}
        FROM ontology.estimate_consensus
        {t_filter}
        {order_exp}
        LIMIT 12;
    """

    with conn.cursor() as cur:
        cur.execute(sql, t_param)
        rows = cur.fetchall()

    if not rows:
        return []

    by_period: dict[str, list] = {}
    for row in rows:
        by_period.setdefault(str(row[0]), []).append(str(row[1]))

    return [
        RAGPassage(
            chunk_id     = f"cons_{ticker}_{i}",
            ticker       = ticker,
            source_type  = "analyst_consensus",
            period       = period,
            speaker      = "analyst_consensus",
            section      = "estimates",
            text         = f"Analyst consensus for {ticker} ({period}):\n" + "\n".join(items),
            similarity_score = 0.73,
        )
        for i, (period, items) in enumerate(sorted(by_period.items(), reverse=True)[:4])
    ]


# ── Source 8: Earnings credibility ────────────────────────────────────────────

def _earnings_credibility(conn, ticker: str) -> tuple[list[GuidanceMatch], float]:
    """Beat rate from ontology.earnings_surprise → credibility score 0.4–0.9."""
    if not _table_exists(conn, "ontology", "earnings_surprise"):
        return [], 0.5

    cols         = _cols(conn, "ontology", "earnings_surprise")
    col_ticker   = _pick(cols, ["ticker", "canonical_ticker", "symbol"])
    col_surprise = _pick(cols, ["surprise", "surprise_pct", "eps_surprise", "beat"])
    col_period   = _pick(cols, ["period", "period_end_date", "date", "quarter"])

    if not col_surprise:
        return [], 0.5

    t_filter = f"WHERE {col_ticker} ILIKE %s" if col_ticker else "WHERE TRUE"
    t_param  = [f"%{ticker}%"] if col_ticker else []
    order    = f"ORDER BY {col_period} DESC" if col_period else ""

    sql = f"""
        SELECT {col_surprise} FROM ontology.earnings_surprise
        {t_filter} {order} LIMIT 8;
    """

    with conn.cursor() as cur:
        cur.execute(sql, t_param)
        rows = cur.fetchall()

    values = []
    for row in rows:
        try:
            values.append(float(row[0]))
        except (TypeError, ValueError):
            pass

    if not values:
        return [], 0.5

    beat_rate   = sum(1 for v in values if v >= 0) / len(values)
    credibility = round(0.4 + beat_rate * 0.5, 2)
    print(f"  [RAG] Beat rate ({ticker}): {beat_rate:.0%} → credibility {credibility}")
    return [], credibility
