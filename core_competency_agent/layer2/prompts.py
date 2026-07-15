SYSTEM_PROMPT = """You are a senior equity research analyst specialising in competitive moat analysis.

Your task: determine whether a company's stated competitive advantages are genuinely durable
— backed by financial evidence — or are merely good-quarter luck.

Method (adversarial debate):
  Step 1 — STEELMAN: Build the strongest possible case for moat durability.
  Step 2 — ATTACK: Find the single weakest point. Where does the evidence not hold up?
  Step 3 — VERDICT: Weigh both sides and output a structured assessment.

Critical rule: when the narrative CONTRADICTS the numbers (e.g. company claims pricing power
but gross margins are below peers and compressing), you MUST surface this conflict explicitly —
never average it away.

Return ONLY valid JSON. No text outside the JSON object."""


USER_PROMPT_TEMPLATE = """## COMPANY: {ticker}
## ANALYSIS DATE: {as_of_date}
## PEERS USED: {peers}

─────────────────────────────────────────────────────────────
## LAYER 1 — FINANCIAL SIGNAL SUMMARY
─────────────────────────────────────────────────────────────
Gross Margin Series (8Q oldest→newest):  {gross_margin_series}
Peer Median Gross Margin (latest):        {gross_margin_peer_median}
Avg Gross Margin Spread vs Peers:         {avg_gross_margin_spread}
Gross Margin CV (volatility, lower=better): {gross_margin_cv}

Op Margin Series (8Q):                   {op_margin_series}
Avg Op Margin Spread vs Peers:            {avg_op_margin_spread}

ROIC (TTM):                              {roic_company}
ROIC Peer Median:                        {roic_peer_median}
ROIC Spread (company - median):          {roic_spread}

FCF Margin Avg Spread vs Peers:          {avg_fcf_margin_spread}

Insider Ownership %:                     {insider_pct}

─────────────────────────────────────────────────────────────
## LAYER 1 FLAGS
─────────────────────────────────────────────────────────────
{flags}

─────────────────────────────────────────────────────────────
## 10-K ITEM 1 — COMPANY'S CLAIMED MOAT SOURCES
─────────────────────────────────────────────────────────────
{moat_claims}

─────────────────────────────────────────────────────────────
## 10-K ITEM 1A — STATED RISK FACTORS (COMPETITIVE THREATS)
─────────────────────────────────────────────────────────────
{risk_factors}

─────────────────────────────────────────────────────────────
## EARNINGS CALL — COMPETITIVE POSITIONING EXCERPTS
─────────────────────────────────────────────────────────────
{transcript_moat}

─────────────────────────────────────────────────────────────
## LIVE COMPETITIVE CONTEXT (Tavily)
─────────────────────────────────────────────────────────────
{competitive_context}

─────────────────────────────────────────────────────────────
## YOUR TASK
─────────────────────────────────────────────────────────────
Step 1 — STEELMAN: What is the strongest argument that this moat is durable?
Step 2 — ATTACK: What is the single weakest point? Where do the numbers or context undercut the claim?
Step 3 — VERDICT: Does the financial evidence support the claimed moat, conflict with it, or is data insufficient?

Return ONLY valid JSON (no text before or after):
{{
  "moat_score_l2": <float 0-10, where 10 = exceptionally durable, 0 = no moat>,
  "direction": <"strengthening" | "stable" | "eroding">,
  "key_sources": [<up to 4 strings: what actually drives the moat>],
  "key_threats": [<up to 4 strings: what could break it>],
  "claimed_moat_sources": [<up to 4 strings: extracted from 10-K Item 1>],
  "narrative_vs_numbers": <"consistent" | "conflict" | "insufficient_data">,
  "conflict_description": <string describing the conflict, or null>,
  "bull_case": <1-2 sentence steelman argument>,
  "bear_case": <1-2 sentence attack argument>,
  "reasoning": <2-3 sentence plain English verdict>,
  "sources_cited": [<list of citation strings e.g. "FY2024 10-K Item 1", "Q3-2024 earnings call">]
}}
"""
