"""
Centralized configuration for Agent 4 — Capability Stack.

All thresholds, scoring weights, model names, and tunable constants live here.
Downstream code imports from this module; nothing is hardcoded inline.
To calibrate for a specific sector, pass a dict of overrides to run_layer1().
"""

# ── Fusion weights ────────────────────────────────────────────────────────────
LAYER1_WEIGHT: float = 0.60  # deterministic R&D/capex spine
LAYER2_WEIGHT: float = 0.40  # LLM narrative interpretation

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL: str = "claude-sonnet-4-6"
LLM_MAX_TOKENS: int = 3500

# ── Data ─────────────────────────────────────────────────────────────────────
# Target window is 5Y / 20Q; DB typically has ~12Q — coverage tracks actual count.
DEFAULT_QUARTERS: int = 20

# ── Layer 1: R&D thresholds ───────────────────────────────────────────────────
# R&D_INTENSIFYING fires when the OLS slope of (R&D / revenue) exceeds this value.
# Unit: change in ratio per quarter-index.
# 0.001 = +0.1pp per quarter ≈ +0.4pp per year — a meaningful intensification signal.
RD_INTENSIFYING_SLOPE_THRESHOLD: float = 0.001

# ── Layer 1: capex thresholds ─────────────────────────────────────────────────
# CAPEX_REINVESTMENT_STRONG fires when capex/rev level OR slope exceeds these values.
# 0.08 = 8% of revenue — characteristic of asset-intensive businesses (semis, industrials).
CAPEX_REINVESTMENT_STRONG_LEVEL: float = 0.08
# 0.002 = +0.2pp per quarter — rapidly rising capex intensity also triggers the flag.
CAPEX_REINVESTMENT_STRONG_SLOPE: float = 0.002

# CAPEX_LIGHT_BUSINESS fires when capex/rev is below this floor.
# 0.02 = 2% of revenue — characteristic of asset-light businesses (SaaS, marketplaces).
CAPEX_LIGHT_FLOOR: float = 0.02

# ── Layer 1: governance thresholds (from yfinance) ────────────────────────────
# INSIDER_CONVICTION_HIGH fires when insider ownership exceeds this fraction.
# 0.05 = 5% — meaningful skin-in-the-game at large-cap scale.
INSIDER_CONVICTION_THRESHOLD: float = 0.05

# INST_CONCENTRATION_HIGH fires when the top-10 institutional holders own
# more than this fraction of shares outstanding.
# 0.50 = 50% — signals strong smart-money conviction in the company's execution.
INST_CONCENTRATION_THRESHOLD: float = 0.50

# ── Layer 1: scoring ─────────────────────────────────────────────────────────
L1_BASE_SCORE: float = 5.0          # neutral starting point before adjustments

# Flag-based score deltas (additive).
L1_DELTA_RD_INTENSIFYING: float = +1.5
L1_DELTA_CAPEX_STRONG: float = +1.5
L1_DELTA_CAPEX_LIGHT: float = -0.5  # asset-light is not strongly negative — neutral-ish
L1_DELTA_INSIDER_CONVICTION: float = +0.5   # management has skin in the game
L1_DELTA_INST_CONCENTRATION: float = +0.3   # smart-money concentration signal

# Continuous bonuses: each percentage point of intensity above zero adds this much.
# Ensures a high-R&D firm still scores well even without the slope flag.
L1_BONUS_PER_PCT_RD: float = 0.05    # 10% R&D/rev → +0.5 pts
L1_BONUS_PER_PCT_CAPEX: float = 0.03 # 10% capex/rev → +0.3 pts

# ── L2 theme gating ──────────────────────────────────────────────────────────
# Themes with LLM confidence below this are excluded from the L2 average
# and marked as "low evidence" rather than being scored as a real 3.0 / 4.0.
L2_CONFIDENCE_GATE: float = 0.30

# ── L1 signal reliability ─────────────────────────────────────────────────────
# When a slope-based flag fires but the OLS R² is below this threshold,
# the flag may be noise — apply a confidence discount.
L1_R2_LOW_THRESHOLD: float = 0.30

# ── Confidence guardrail ──────────────────────────────────────────────────────
# Coverage: fewer than this many quarters → apply data-thinness discount.
MIN_QUARTERS_FULL_CONFIDENCE: int = 12
# RAG: fewer than this many chunks returned per theme → thin evidence discount.
RAG_MIN_CHUNKS_PER_THEME: int = 3

# Per-trigger discount amounts (subtracted from confidence, floored at 0.1).
GUARDRAIL_COVERAGE_DISCOUNT: float = 0.15    # weak data coverage
GUARDRAIL_THIN_RAG_DISCOUNT: float = 0.10    # sparse 10-K retrieval
GUARDRAIL_FLAG_CONFLICT_DISCOUNT: float = 0.15  # L1 flags contradict L2 narrative

# ── RAG ──────────────────────────────────────────────────────────────────────
RAG_TOP_K: int = 5           # kept for transcript retrieval (single query)
RAG_TOP_K_SUBQUERY: int = 6  # chunks fetched per sub-query (2 sub-queries per theme)
RAG_TOP_K_FINAL: int = 8     # chunks passed to LLM per theme after merge + dedup

# ── Tavily ────────────────────────────────────────────────────────────────────
TAVILY_MAX_RESULTS: int = 8
