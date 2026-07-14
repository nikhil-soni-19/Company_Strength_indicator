"""
Phase 3 — Layer 2: LLM interpretation (40% of final score).

Two steps:
  1. Retrieval  — pull 10-K chunks for four capability themes (BM25 + vector)
                  + Tavily web-search for industry tech-adoption context
  2. Reasoning  — LLM receives Layer 1 numbers + flags FIRST, then retrieved
                  evidence, reasons theme-by-theme, returns strict JSON

The LLM must ground every claim in provided excerpts — no outside knowledge.
Thin evidence for a theme → the LLM must say so and set confidence ≤ 0.4.
Malformed JSON → one automatic retry with an explicit JSON-only instruction.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    RAG_TOP_K,
    RAG_TOP_K_FINAL,
    RAG_TOP_K_SUBQUERY,
    TAVILY_MAX_RESULTS,
)
from layer1_deterministic import Layer1Output
from retrieval.connection import OntologyDBNotConfigured
from retrieval.esg_fetcher import ESGData
from retrieval.embedder import embed_queries
from retrieval.filing_resolver import (
    resolve_10k_by_fiscal_year,
    resolve_best_filing,
    resolve_latest_earnings_call,
)
from retrieval.hybrid_search import hybrid_search

load_dotenv(Path(__file__).parent / ".env")

# ── Four capability themes and their targeted retrieval queries ───────────────
# Queries are designed to surface capability-stack evidence (operating engine),
# not competitive moat evidence. Kept long and descriptive to maximise recall.

_THEME_QUERIES: Dict[str, list[str]] = {
    "tech": [
        # What are they building / deploying?
        "AI machine learning automation software deployment product innovation "
        "technology stack production system digital platform",
        # How much are they spending on it and on what?
        "R&D investment research spending digital transformation platform "
        "modernization technology budget roadmap initiative",
    ],
    "capacity": [
        # Physical assets and operational scale
        "manufacturing production facilities expansion throughput utilization "
        "workforce capacity output supply chain operations",
        # Capital spending and infrastructure investment
        "capital expenditure capex infrastructure investment scalability "
        "operational efficiency plant equipment logistics",
    ],
    "esg": [
        # Environmental commitments
        "environmental sustainability carbon emissions climate targets net-zero "
        "renewable energy waste reduction energy efficiency",
        # Social and governance responsibility
        "social responsibility diversity inclusion community workforce safety "
        "ESG commitments human rights supplier standards",
    ],
    "governance": [
        # Board and leadership structure
        "board composition independent directors executive leadership succession "
        "planning oversight accountability audit committee",
        # Compensation and performance alignment
        "executive compensation pay performance KPIs incentives alignment "
        "risk management shareholder interests capital allocation",
    ],
}

# Section routing: where each theme is most likely to appear in a 10-K.
_THEME_SECTIONS: Dict[str, Optional[str]] = {
    "tech":       "business",       # Item 1 — Business description
    "capacity":   "business",
    "esg":        None,             # may appear in multiple sections
    "governance": None,             # Item 10/11/13 (proxy sections not always tagged)
}

_SYSTEM_PROMPT = """\
You are a capability-stack analyst assessing whether a company has the internal \
engine to keep executing. You are NOT assessing competitive moat or market position. \
You are judging operating capacity: R&D investment intensity, physical reinvestment \
patterns, technology adoption, governance quality, and ESG commitments.

You will receive:
1. Layer 1 deterministic metrics (computed from financial data — treat as fact)
2. Excerpts from the company's 10-K filing (retrieved evidence)
3. Earnings call transcript excerpts (management's forward-looking commentary)
4. Industry context from web search

Rules you must follow:
- For EACH theme, write a "reasoning" block BEFORE assigning a score. The reasoning
  must cover three things: (a) what the Layer 1 numbers imply about this theme,
  (b) what the filing and transcript evidence confirms or contradicts, and
  (c) any gaps or conflicts between data and narrative. Your score MUST follow from
  this reasoning — never precede it.
- If a Layer 1 flag contradicts the filing evidence (e.g. CAPEX_LIGHT_BUSINESS flag
  but the 10-K describes heavy infrastructure spend), call this out explicitly in
  your reasoning and explain which source you trust more and why.
- Ground every claim in the provided evidence. Do not use outside knowledge.
- If evidence for a theme is thin or absent, say so in reasoning and set confidence ≤ 0.4.
- Return ONLY valid JSON — no preamble, no explanation outside the JSON object.\
"""

_USER_TEMPLATE = """\
Ticker: {ticker}
As of: {as_of_date}

=== LAYER 1 — DETERMINISTIC METRICS ===
R&D / Revenue — TTM (trailing 4Q):  {rd_rev_ttm}   ← use this as the primary level; removes seasonal distortion
R&D / Revenue — current quarter:    {rd_rev_level}
R&D / Revenue — YoY change:         {rd_rev_yoy}   (same quarter last year comparison)
R&D / Revenue — OLS slope:          {rd_rev_slope} per quarter  (R²={rd_rev_r2})
R&D / Revenue — CAGR (annualised):  {rd_rev_cagr}
R&D / Revenue — consistency (CV):   {rd_rev_cv}    (lower = steadier execution)
R&D / Revenue — 3yr percentile:     {rd_rev_pct}   (1.0 = at 3-year high)

Capex / Revenue — TTM (trailing 4Q): {capex_rev_ttm}
Capex / Revenue — current quarter:   {capex_rev_level}
Capex / Revenue — YoY change:        {capex_rev_yoy}
Capex / Revenue — OLS slope:         {capex_rev_slope} per quarter  (R²={capex_rev_r2})
Capex / Revenue — CAGR:              {capex_rev_cagr}
Capex / Revenue — consistency (CV):  {capex_rev_cv}
Capex / Revenue — 3yr percentile:    {capex_rev_pct}

NOTE: Prefer TTM ratios over single-quarter values. A slope with R² < 0.30 is likely noise.
A high 3yr percentile (≥ 0.80) means the company is investing at its highest rate in 3 years.

Flags fired:                      {flags}
Quarters of data:                 {quarters}
Insider ownership:                {insider_pct}

Flag definitions — each flag maps to the theme shown. Use these in your reasoning:
  R&D_INTENSIFYING          [Tech]       R&D/rev OLS slope is rising — company is actively intensifying research spend
  CAPEX_REINVESTMENT_STRONG [Capacity]   capex/rev ≥8% OR slope rising rapidly — heavy physical reinvestment
  CAPEX_LIGHT_BUSINESS      [Capacity]   capex/rev <2% — asset-light model; mutually exclusive with above
  INSIDER_CONVICTION_HIGH   [Governance] insiders own >5% of shares — management has meaningful skin in the game
  INST_CONCENTRATION_HIGH   [Governance] top-10 institutions own >50% of float — strong smart-money conviction

Conflict rule — for every flag that fired:
  Your reasoning for the mapped theme MUST state whether the filing evidence
  confirms or contradicts the flag, and which source you trust more and why.
  Example: "R&D_INTENSIFYING is set but the 10-K mentions no specific R&D programs —
  this suggests the spend increase may be in headcount rather than disclosed initiatives."
For flags that did NOT fire: do not penalise automatically — check the filing evidence
independently before scoring.

=== STRUCTURED ESG DATA (Bloomberg, annual time series) ===
NOTE: This Bloomberg data is more reliable than RAG excerpts for ESG and Governance themes.
Use these numbers directly in your reasoning for both the "esg" and "governance" themes.
{structured_esg}

=== 10-K EVIDENCE ===

[TECH ADOPTION]
{tech_chunks}

[CAPACITY / OPERATIONS]
{capacity_chunks}

[ESG]
{esg_chunks}

[GOVERNANCE]
{governance_chunks}

=== EARNINGS CALL TRANSCRIPT ===
(Management's own words on R&D plans, capex guidance, and technology priorities)
{transcript_chunks}

=== INDUSTRY CONTEXT (web search) ===
{industry_context}

=== TASK ===
Using ONLY the evidence above, assess each theme. For each theme write "reasoning"
FIRST — then derive "score", "rationale", "evidence_used", and "confidence" from it.

Return JSON with this exact structure:

{{
  "tech": {{
    "reasoning": "<3–5 sentences: what do L1 metrics + filing evidence together say? are R&D slope/flags consistent with what the 10-K describes? any conflicts?>",
    "score": <integer 0–10>,
    "rationale": "<2–3 sentences summarising your reasoning for display>",
    "evidence_used": ["<exact quote or close paraphrase from above, with source e.g. [10-K 2] or [transcript 1]>", ...],
    "confidence": <float 0.0–1.0>
  }},
  "capacity": {{
    "reasoning": "<3–5 sentences: what does capex/rev level + slope say? does the filing describe specific facilities, expansions, or supply-chain investments?>",
    "score": <integer 0–10>,
    "rationale": "<2–3 sentences>",
    "evidence_used": [...],
    "confidence": <float 0.0–1.0>
  }},
  "esg": {{
    "reasoning": "<3–5 sentences: are ESG commitments specific with timelines, or boilerplate? any third-party verification mentioned?>",
    "score": <integer 0–10>,
    "rationale": "<2–3 sentences>",
    "evidence_used": [...],
    "confidence": <float 0.0–1.0>
  }},
  "governance": {{
    "reasoning": "<3–5 sentences: what does insider ownership % signal? is board composition described? is pay linked to named KPIs?>",
    "score": <integer 0–10>,
    "rationale": "<2–3 sentences>",
    "evidence_used": [...],
    "confidence": <float 0.0–1.0>
  }}
}}

Scoring anchors — your score MUST map to one of these levels (interpolate between):

  Tech (R&D execution + technology adoption):
     1 = no R&D; pure distribution or services with zero technology investment mentioned
     4 = R&D exists but no specific programs or initiatives named in filings
     7 = named technology initiatives with stated goals and deployment evidence; R&D/rev rising per L1
    10 = quantified revenue or efficiency gains directly attributed to specific technology investments

  Capacity (capital reinvestment + operational scalability):
     1 = fully outsourced; capex near zero; no operational scale investment discussed
     4 = capex mentioned generically; no specific projects, timelines, or capacity targets
     7 = specific facility, line, or supply-chain investment named with timeline or budget
    10 = capacity expansion completed with measurable output or efficiency improvement cited

  ESG (environmental + social commitments):
     1 = no ESG discussion or pure legal-compliance boilerplate only
     4 = high-level aspirations only; no baselines, targets, or timelines stated
     7 = specific targets with baselines and timelines (e.g. net-zero by 2040, -30% water by 2030)
    10 = third-party verified progress reported; ESG performance tied to executive compensation

  Governance (board quality + management accountability):
     1 = concentrated control; no independent oversight; no accountability mechanisms disclosed
     4 = standard SEC disclosures only; basic audit and compensation committees stated
     7 = independent board majority; compensation explicitly linked to named performance KPIs
    10 = robust succession planning disclosed; multiple named KPIs with track record of capital discipline

{capacity_rubric_override}
Confidence: 0=no evidence, 0.4=1 weak excerpt, 0.7=2–3 direct excerpts, 1.0=multiple direct excerpts confirm claim.
Set confidence ≤ 0.4 when fewer than 2 excerpts support a theme.\
"""


# ─── Dynamic rubric ───────────────────────────────────────────────────────────

def _build_capacity_rubric_override(flags: list[str]) -> str:
    """
    Return a capacity scoring rubric override based on the company's detected archetype.

    CAPEX_LIGHT_BUSINESS → asset-light company: score on operational efficiency,
        not capital deployment. Prevents penalising Apple/SaaS for low capex.
    CAPEX_REINVESTMENT_STRONG → capital-intensive: score on quality and returns
        of capital deployed, not just presence of capex.
    No flag → return empty string (default rubric applies).
    """
    if "CAPEX_LIGHT_BUSINESS" in flags:
        return """
CAPACITY SCORING OVERRIDE — CAPEX_LIGHT_BUSINESS is set (asset-light model detected):
This company operates with minimal physical capital by design. Do NOT penalise low capex/rev.
Instead score Capacity on operational efficiency and execution quality:
   1 = costs growing faster than revenue; no operational discipline
   4 = adequate execution; no notable efficiency improvement cited
   7 = documented gains: margin improvement, automation, throughput per employee, or unit cost reduction
  10 = quantified productivity gains with named initiatives and measurable before/after output
"""
    if "CAPEX_REINVESTMENT_STRONG" in flags:
        return """
CAPACITY SCORING OVERRIDE — CAPEX_REINVESTMENT_STRONG is set (capital-intensive model):
This company makes heavy physical investment. Score Capacity on quality and returns of capital:
   1 = capex with no stated purpose or return expectation
   4 = capex described but no specific project, timeline, or capacity target named
   7 = specific capital projects with stated capacity or output targets
  10 = completed projects with quantified capacity expansion or return on investment cited
"""
    return ""


# ─── Output types ─────────────────────────────────────────────────────────────

@dataclass
class ThemeAssessment:
    score: float         # [0, 10]
    rationale: str
    evidence_used: list[str]
    confidence: float    # [0, 1]
    reasoning: str = "" # CoT scratchpad written before the score


@dataclass
class Layer2Output:
    ticker: str
    themes: Dict[str, ThemeAssessment]   # keys: tech, capacity, esg, governance
    rag_chunks_per_theme: Dict[str, int]  # chunk count retrieved per theme (for guardrail)
    transcript_chunks_count: int          # earnings call chunks retrieved
    industry_context_count: int           # Tavily articles found (for guardrail)
    raw_llm_response: str


_FALLBACK_THEME = ThemeAssessment(
    score=5.0,
    rationale="LLM response could not be parsed.",
    evidence_used=[],
    confidence=0.1,
)

_THEMES = list(_THEME_QUERIES.keys())


# ─── Retrieval ────────────────────────────────────────────────────────────────

def _resolve_filing(ticker: str) -> object:
    """Resolve the most recent available 10-K for ticker."""
    fy = date.today().year - 1
    try:
        filing = resolve_10k_by_fiscal_year(ticker, fy)
        if filing:
            print(f"  [Layer2] {ticker}: resolved 10-K FY{fy} → filing_id={filing.filing_id}")
            return filing
        # Try current year in case fiscal year already ended
        filing = resolve_10k_by_fiscal_year(ticker, fy + 1)
        if filing:
            print(f"  [Layer2] {ticker}: resolved 10-K FY{fy+1} → filing_id={filing.filing_id}")
            return filing
        # Fall back to any recent filing
        filing = resolve_best_filing(ticker, as_of_date=date.today())
        if filing:
            return filing
        print(f"  [Layer2] {ticker}: no filing found in ontology DB")
        return None
    except OntologyDBNotConfigured as e:
        print(f"  [Layer2] {e}")
        return None
    except Exception as e:
        print(f"  [Layer2] Filing resolution error: {e}")
        return None


def retrieve_capability_context(
    ticker: str,
    k_per_theme: int = RAG_TOP_K,
) -> tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Retrieve 10-K chunks for all four capability themes via hybrid BM25+vector search.

    Each theme uses two focused sub-queries (one for 'what they build', one for
    'how much they spend'). All sub-queries are embedded in a single batch call.
    Results are merged per theme and deduplicated by chunk text before being capped
    at RAG_TOP_K_FINAL chunks for the LLM.

    Returns:
        chunks_by_theme: {theme_name: [chunk_text, ...]}
        counts_by_theme: {theme_name: int}  — for confidence guardrail
    """
    empty_chunks: Dict[str, List[str]] = {t: [] for t in _THEMES}
    empty_counts: Dict[str, int] = {t: 0 for t in _THEMES}

    filing = _resolve_filing(ticker)
    if filing is None:
        return empty_chunks, empty_counts

    # Flatten all sub-queries into one list for a single batch embed call
    theme_subqueries: list[tuple[str, str]] = [
        (theme, q)
        for theme in _THEMES
        for q in _THEME_QUERIES[theme]
    ]
    all_queries = [q for _, q in theme_subqueries]

    try:
        embeddings = embed_queries(all_queries)
    except Exception as e:
        print(f"  [Layer2] Batch embed failed: {e}")
        return empty_chunks, empty_counts

    # Collect raw results per theme across all sub-queries
    raw_by_theme: Dict[str, list[dict]] = {t: [] for t in _THEMES}

    for (theme, query), qvec in zip(theme_subqueries, embeddings):
        section = _THEME_SECTIONS.get(theme)
        try:
            results = hybrid_search(
                query_text=query,
                query_embedding=qvec,
                filing_id=filing.filing_id,
                section=section,
                doc_type=filing.doc_type,
                top_k=RAG_TOP_K_SUBQUERY,
            )
            raw_by_theme[theme].extend(results)
        except Exception as e:
            print(f"  [Layer2] Theme '{theme}' sub-query failed: {e}")

    # Deduplicate by chunk_text, then cap at RAG_TOP_K_FINAL per theme
    chunks_by_theme: Dict[str, List[str]] = {}
    counts_by_theme: Dict[str, int] = {}

    for theme in _THEMES:
        seen: set[str] = set()
        deduped: List[str] = []
        for chunk in raw_by_theme[theme]:
            text = chunk.get("chunk_text", "")
            if text and text not in seen:
                seen.add(text)
                deduped.append(text)
            if len(deduped) >= RAG_TOP_K_FINAL:
                break
        chunks_by_theme[theme] = deduped
        counts_by_theme[theme] = len(deduped)
        n_candidates = len(raw_by_theme[theme])
        print(f"  [Layer2] {ticker} theme='{theme}': {len(deduped)} chunks "
              f"(from {n_candidates} candidates across {len(_THEME_QUERIES[theme])} sub-queries)")

    return chunks_by_theme, counts_by_theme


# Transcript query: focused on forward-looking management commentary
_TRANSCRIPT_QUERY = (
    "R&D investment research development technology roadmap AI automation "
    "capital expenditure capex infrastructure investment plans guidance "
    "capacity expansion operational efficiency innovation strategy"
)


def retrieve_transcript_context(
    ticker: str,
    k: int = RAG_TOP_K,
) -> tuple[List[str], int]:
    """
    Retrieve chunks from the most recent earnings call transcript for the ticker.
    Uses the same hybrid BM25+vector search as 10-K retrieval.

    Returns:
        (list of chunk texts, count retrieved)
    """
    try:
        filing = resolve_latest_earnings_call(ticker, date.today())
    except OntologyDBNotConfigured as e:
        print(f"  [Layer2] {e}")
        return [], 0
    except Exception as e:
        print(f"  [Layer2] Earnings call resolution error: {e}")
        return [], 0

    if filing is None:
        print(f"  [Layer2] {ticker}: no earnings call transcript found in ontology DB")
        return [], 0

    try:
        embeddings = embed_queries([_TRANSCRIPT_QUERY])
        qvec = embeddings[0]
    except Exception as e:
        print(f"  [Layer2] Transcript embed failed: {e}")
        return [], 0

    try:
        results = hybrid_search(
            query_text=_TRANSCRIPT_QUERY,
            query_embedding=qvec,
            filing_id=filing.filing_id,
            doc_type=filing.doc_type,
            top_k=k,
        )
        texts = [c["chunk_text"] for c in results if c.get("chunk_text")]
        print(f"  [Layer2] {ticker} earnings call: {len(texts)} chunks "
              f"(filing_id={filing.filing_id}, period={filing.period_end_date})")
        return texts, len(texts)
    except Exception as e:
        print(f"  [Layer2] Transcript retrieval failed: {e}")
        return [], 0


def fetch_industry_context(ticker: str) -> list[dict]:
    """
    Tavily web search for industry tech-adoption context for the company's sector.
    24-hour in-process cache to avoid duplicate calls within a session.
    Returns list of article dicts; empty list on failure or missing API key.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []

    year = date.today().year
    queries = [
        f"{ticker} technology adoption AI innovation R&D investment {year}",
        f"{ticker} capital investment infrastructure capacity expansion {year}",
    ]

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        results: list[dict] = []
        per_q = max(3, TAVILY_MAX_RESULTS // len(queries))

        for i, q in enumerate(queries):
            try:
                resp = client.search(
                    query=q,
                    max_results=per_q,
                    search_depth="advanced",
                    include_answer=False,
                )
                for r in resp.get("results", []):
                    results.append({
                        "title":    r.get("title", ""),
                        "url":      r.get("url", ""),
                        "snippet":  r.get("content", "")[:600],
                        "pub_date": r.get("published_date", ""),
                    })
                print(f"  [Layer2] Tavily query {i+1}/{len(queries)}: {len(resp.get('results', []))} results")
            except Exception as e:
                print(f"  [Layer2] Tavily query {i+1} failed: {e}")

        # Deduplicate by URL
        seen: set[str] = set()
        deduped = []
        for a in results:
            if a["url"] not in seen:
                seen.add(a["url"])
                deduped.append(a)

        return deduped[:TAVILY_MAX_RESULTS]
    except Exception as e:
        print(f"  [Layer2] Tavily failed: {e}")
        return []


# ─── LLM call ─────────────────────────────────────────────────────────────────

def _fmt_chunks(chunks: list[str]) -> str:
    if not chunks:
        return "(no excerpts retrieved)"
    return "\n".join(f"[{i+1}] {c[:700]}" for i, c in enumerate(chunks))


def _fmt_industry(articles: list[dict]) -> str:
    if not articles:
        return "(no industry context retrieved)"
    return "\n".join(
        f"[{i+1}] {a['title']} ({a.get('pub_date', '')})\n    {a['snippet']}"
        for i, a in enumerate(articles)
    )


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.1%}" if v is not None else "N/A"


def _fmt_esg_structured(esg: Optional[ESGData], n_years: int = 5) -> str:
    """
    Format Bloomberg ESG annual data as a compact table for the LLM prompt.

    Shows the most recent n_years of data. Returns a placeholder string if
    no ESG data is available so the template always renders cleanly.
    """
    if esg is None:
        return "(Bloomberg ESG data not available for this ticker)"

    # Take the last n_years
    fys = esg.fiscal_years[-n_years:]
    n = len(fys)
    if n == 0:
        return "(Bloomberg ESG data found but contains no fiscal years)"

    def _row(label: str, series: list, fmt: str = "num", suffix: str = "") -> str:
        """Format one metric row aligned to fys."""
        raw = (series or [])[-n_years:]
        raw = [None] * n + raw          # left-pad with None if series is short
        raw = raw[-n:]                  # take last n
        parts = []
        for v in raw:
            if v is None:
                parts.append("   N/A  ")
            elif fmt == "pct":
                parts.append(f"{v:6.1f}%")
            elif fmt == "ratio":
                parts.append(f"{v:6.0f}x ")
            elif fmt == "score":
                parts.append(f"{v:6.2f}  ")
            else:
                parts.append(f"{v:8.2f}")
        cols = "  ".join(parts)
        return f"  {label:<42}{cols}{suffix}"

    header = "  " + " " * 42 + "  ".join(f"{y:>8}" for y in fys)

    lines = [
        header,
        "",
        "  Pillar Scores (Bloomberg BESG, 0–10):",
        _row("Environmental Score:", esg.environmental_score, "score"),
        _row("Social Score:", esg.social_score, "score"),
        "",
        "  Governance Metrics:",
        _row("% Independent Directors:", esg.pct_independent_directors, "pct"),
        _row("Say-on-Pay Support:", esg.say_on_pay_support, "pct"),
        _row("% Women on Board:", esg.pct_women_on_board, "pct"),
        _row("CEO Pay Ratio (vs median employee):", esg.ceo_pay_ratio, "ratio"),
        _row("Board Average Age:", esg.board_average_age, "num"),
        "",
        "  Social Metrics:",
        _row("Employee Turnover %:", esg.employee_turnover_pct, "pct"),
        _row("Safety Incident Rate (TRIR):", esg.safety_incident_rate, "score"),
        _row("% Women in Workforce:", esg.pct_women_employees, "pct"),
        _row("% Women in Management:", esg.pct_women_mgmt, "pct"),
        "",
        "  Environmental Metrics:",
        _row("CO2 Scope 1+2 (metric tons, millions):", esg.co2_total),
        _row("Energy Consumed (GWh):", esg.energy_consumed),
        "",
        f"  Source: Bloomberg ({', '.join(esg.tables_used)})",
    ]
    return "\n".join(lines)


def _parse_llm_response(text: str) -> dict:
    """Extract and parse JSON from raw LLM output. Strips markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start: end + 1]
    return json.loads(text)


def _validate_theme(raw: dict, theme: str) -> ThemeAssessment:
    """Validate and clamp one theme object from the LLM response."""
    score = float(raw.get("score", 5.0))
    score = max(0.0, min(10.0, score))

    confidence = float(raw.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    reasoning = str(raw.get("reasoning", "")).strip()
    rationale = str(raw.get("rationale", "")).strip() or f"No rationale for {theme}."

    ev = raw.get("evidence_used", [])
    evidence_used = ev if isinstance(ev, list) else []

    return ThemeAssessment(
        score=score,
        rationale=rationale,
        evidence_used=evidence_used,
        confidence=confidence,
        reasoning=reasoning,
    )


def _call_llm(user_msg: str) -> str:
    """Call Anthropic API and return raw text. Raises on total failure."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    response = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", LLM_MODEL),
        max_tokens=LLM_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text


# ─── Public entry point ───────────────────────────────────────────────────────

def run_layer2(
    ticker: str,
    l1_output: Layer1Output,
    esg: Optional[ESGData] = None,
    k_per_theme: int = RAG_TOP_K,
) -> Layer2Output:
    """
    Run Layer 2 for the given ticker.

    Step 1 — Retrieval: pull 10-K chunks for four capability themes + Tavily context.
    Step 2 — LLM reasoning: hand L1 numbers + flags first, then evidence;
              parse strict JSON; retry once on malformed output.

    Args:
        ticker:      Company ticker (e.g. 'AAPL').
        l1_output:   Layer1Output from run_layer1() — provides numbers + flags.
        k_per_theme: Chunks to retrieve per theme (default from config.RAG_TOP_K).

    Returns:
        Layer2Output with per-theme ThemeAssessments and metadata for the guardrail.
    """
    # ── Step 1: retrieval ─────────────────────────────────────────────────────
    chunks_by_theme, counts_by_theme = retrieve_capability_context(ticker, k_per_theme)
    transcript_chunks, transcript_count = retrieve_transcript_context(ticker, k_per_theme)
    industry_articles = fetch_industry_context(ticker)

    # ── Build user message ────────────────────────────────────────────────────
    def _opt_pct(v) -> str:
        return f"{v:+.2%}" if v is not None else "N/A (< 5Q data)"

    user_msg = _USER_TEMPLATE.format(
        ticker=ticker,
        as_of_date=date.today().isoformat(),
        # TTM and current quarter
        rd_rev_ttm=f"{l1_output.rd_rev_ttm:.2%}",
        rd_rev_level=f"{l1_output.rd_rev_level:.2%}",
        rd_rev_yoy=_opt_pct(l1_output.rd_rev_yoy),
        rd_rev_slope=f"{l1_output.rd_rev_slope:+.5f}",
        rd_rev_r2=f"{l1_output.rd_rev_r2:.2f}",
        rd_rev_cagr=f"{l1_output.rd_rev_cagr:+.1%}/yr" if l1_output.rd_rev_cagr else "N/A",
        rd_rev_cv=f"{l1_output.rd_rev_cv:.2f}",
        rd_rev_pct=f"{l1_output.rd_rev_pct:.0%}",
        capex_rev_ttm=f"{l1_output.capex_rev_ttm:.2%}",
        capex_rev_level=f"{l1_output.capex_rev_level:.2%}",
        capex_rev_yoy=_opt_pct(l1_output.capex_rev_yoy),
        capex_rev_slope=f"{l1_output.capex_rev_slope:+.5f}",
        capex_rev_r2=f"{l1_output.capex_rev_r2:.2f}",
        capex_rev_cagr=f"{l1_output.capex_rev_cagr:+.1%}/yr" if l1_output.capex_rev_cagr else "N/A",
        capex_rev_cv=f"{l1_output.capex_rev_cv:.2f}",
        capex_rev_pct=f"{l1_output.capex_rev_pct:.0%}",
        flags=", ".join(l1_output.flags) if l1_output.flags else "(none)",
        quarters=l1_output.data_coverage.quarters_returned,
        insider_pct=_fmt_pct(l1_output.insider_pct),
        capacity_rubric_override=_build_capacity_rubric_override(l1_output.flags),
        structured_esg=_fmt_esg_structured(esg),
        tech_chunks=_fmt_chunks(chunks_by_theme.get("tech", [])),
        capacity_chunks=_fmt_chunks(chunks_by_theme.get("capacity", [])),
        esg_chunks=_fmt_chunks(chunks_by_theme.get("esg", [])),
        governance_chunks=_fmt_chunks(chunks_by_theme.get("governance", [])),
        transcript_chunks=_fmt_chunks(transcript_chunks),
        industry_context=_fmt_industry(industry_articles),
    )

    # ── Step 2: LLM call with one retry ───────────────────────────────────────
    fallback_themes = {t: _FALLBACK_THEME for t in _THEMES}
    raw_text = ""

    for attempt in range(2):
        retry_suffix = (
            "" if attempt == 0
            else "\n\nIMPORTANT: Return ONLY the JSON object. "
                 "No preamble, no markdown, no explanation outside the JSON."
        )
        try:
            raw_text = _call_llm(user_msg + retry_suffix)
            parsed = _parse_llm_response(raw_text)
            themes = {
                t: _validate_theme(parsed.get(t, {}), t)
                for t in _THEMES
            }
            print(
                f"  [Layer2] {ticker}: LLM parse succeeded on attempt {attempt + 1}  |  "
                + "  ".join(f"{t}={themes[t].score:.0f}({themes[t].confidence:.2f})" for t in _THEMES)
            )
            return Layer2Output(
                ticker=ticker,
                themes=themes,
                rag_chunks_per_theme=counts_by_theme,
                transcript_chunks_count=transcript_count,
                industry_context_count=len(industry_articles),
                raw_llm_response=raw_text,
            )
        except json.JSONDecodeError as e:
            print(f"  [Layer2] {ticker}: JSON parse failed (attempt {attempt + 1}): {e}")
            if attempt == 1:
                print(f"  [Layer2] {ticker}: falling back to default scores")
        except Exception as e:
            print(f"  [Layer2] {ticker}: LLM call failed (attempt {attempt + 1}): {e}")
            if attempt == 1:
                break

    return Layer2Output(
        ticker=ticker,
        themes=fallback_themes,
        rag_chunks_per_theme=counts_by_theme,
        transcript_chunks_count=transcript_count,
        industry_context_count=len(industry_articles),
        raw_llm_response=raw_text,
    )
