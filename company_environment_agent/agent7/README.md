# Agent 7 — Company Environment Agent

Agent 7 answers one question for any covered equity: **"Is this company operating in a supportive, mixed, or hostile external environment?"**

It produces an `environment_score` (0–100) and a categorical direction (`SUPPORTIVE` / `MIXED` / `HOSTILE`) by combining a quantitative **Layer 1** signal bundle with a qualitative **Layer 2** LLM interpretation grounded in live news and 10-K risk-factor excerpts. Every claim in the narrative cites a specific source — news snippet or 10-K passage — so the score is fully auditable.

---

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Configure
cp .env.example .env        # fill in DATABASE_URL, ANTHROPIC_API_KEY, TAVILY_API_KEY,
                             # DATABASE_URL_ONTOLOGY_LAB (see Environment Variables below)

# 3. Initialise the database schema
python cli.py init-db

# 4. Run the interactive chat interface
python cli.py chat
```

Then at the prompt:

```
> analyse AMD
> what's the environment for Microsoft?
> run Tesla today
```

---

## Environment variables

Add these to `agent7/.env`:

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | ✅ | Main Neon PostgreSQL DB — stores `environment_runs`. Format: `postgresql://user:pass@host/db?sslmode=require` |
| `DATABASE_URL_ONTOLOGY_LAB` | ✅ | Neon DB holding ingested 10-K filings (`csi_ontology_lab`). Can be the same Neon instance as `DATABASE_URL`. |
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API key for the Layer 2 LLM call |
| `TAVILY_API_KEY` | ✅ | Tavily API key for live PESTEL news searches |
| `ANTHROPIC_MODEL` | optional | Model ID, defaults to `claude-sonnet-4-6` |

> **Note:** `DATABASE_URL` and `DATABASE_URL_ONTOLOGY_LAB` can point to the same Neon connection string if both the main schema and the ontology schema live in the same database.

---

## Architecture

```
INPUT: ticker + as_of_date
         │
         ▼
┌────────────────────────────────────────┐
│  LAYER 1 — Quantitative bundle         │
│                                        │
│  • Returns & momentum (6m)             │
│  • Alpha / Beta vs sector ETF          │
│  • Vol regime  (VIX z-score)           │
│  • Rate regime (^TNX slope z-score)    │
│  • Market trend (SPY momentum)         │
│  • Commodity sensitivity               │
│  • Peer revenue & margin gaps          │
│  • R&D / CapEx intensity vs peers      │
│  • PESTEL quant sub-scores (6 dims)    │
└────────────────────────────────────────┘
         │ quant_score  (0-100)
         ▼
┌────────────────────────────────────────┐
│  LAYER 2 — Qualitative                 │
│                                        │
│  News: Tavily (6 searches × dim)       │
│  10-K:  hybrid BM25 + vector search    │
│         (BGE 1024-dim, csi_ontology_lab│
│          per-dimension PESTEL queries) │
│                                        │
│  LLM (Claude): scores each PESTEL dim  │
│  + writes per-dimension narratives     │
│  + identifies key tailwinds & risks    │
└────────────────────────────────────────┘
         │ qual_score  (0-100)
         ▼
┌────────────────────────────────────────┐
│  FINAL SCORE                           │
│  environment_score = 0.5×Q1 + 0.5×Q2  │
│  ≥70 → SUPPORTIVE                      │
│  30–69 → MIXED                         │
│  <30 → HOSTILE                         │
└────────────────────────────────────────┘
```

### PESTEL dimension weights (quant scoring)

| Dimension | Weight | Key quant signals |
|---|---|---|
| Political (P) | 12% | USD strength × trade sensitivity |
| Economic (E) | 30% | Sector alpha, credit spread, rate regime, peer gaps |
| Social (S) | 15% | XLY/XLP consumer sentiment, R&D intensity |
| Technological (T) | 15% | R&D gap, CapEx gap, sector disruption exposure |
| Environmental (En) | 13% | Sector carbon intensity, energy-cost sensitivity |
| Legal (L) | 15% | Regulatory complexity, litigation exposure |

---

## Key files

```
agent7/
├── cli.py                        # Entry point: chat / run / init-db / ingest
├── agent/
│   ├── run.py                    # Full pipeline: Layer1 → Layer2 → score → persist
│   └── chat.py                   # Interactive REPL and formatted output
├── layer1/
│   ├── bundle.py                 # Assembles all Layer 1 signals into one dict
│   ├── data_loader.py            # yfinance price/fundamental fetches, peer map
│   ├── returns.py                # 6-month return, relative strength, alpha/beta
│   ├── regimes.py                # VIX vol regime, rate regime, market trend
│   ├── peer_gaps.py              # Revenue growth gap, margin gap vs peers
│   ├── flags.py                  # PESTEL-aware qualitative flags
│   └── pestel/                   # Per-dimension quant signal modules
│       ├── political.py          # FX/trade impact
│       ├── economic.py           # Credit spread, macro momentum
│       ├── social.py             # Consumer sentiment ratio
│       ├── technological.py      # R&D/CapEx intensity
│       ├── environmental.py      # Carbon/energy exposure
│       └── legal.py              # Regulatory burden
├── layer2/
│   ├── tavily_pestel.py          # 6 targeted Tavily searches (1 per PESTEL dim)
│   ├── ten_k_retrieval.py        # Hybrid search interface → csi_ontology_lab
│   ├── risk_factor_retrieval.py  # Orchestrates 10-K retrieval with fallback
│   ├── llm_interpreter.py        # Builds prompt, calls Claude, parses response
│   ├── prompts.py                # System + user prompt templates
│   └── retrieval/
│       ├── connection.py         # DATABASE_URL_ONTOLOGY_LAB connector
│       ├── filing_resolver.py    # 10-K → 10-K_A → 10-Q → earnings_call fallback
│       ├── embedder.py           # BGE bge-large-en-v1.5 (1024-dim)
│       └── hybrid_search.py      # BM25 + vector RRF(k=60) over narrative_chunks
├── scoring/
│   ├── quant_score.py            # Weighted PESTEL quant → quant_score (0-100)
│   ├── pestel_score.py           # Per-dimension 0-100 sub-scores
│   └── final_score.py            # 50/50 blend → environment_score + direction
├── config/
│   ├── peer_map.yaml             # Ticker → GICS sector + peers (⚠ see note below)
│   ├── political_exposure.yaml   # Per-sector trade/FX sensitivity weights
│   ├── tech_disruption.yaml      # Per-sector technology disruption scores
│   ├── environmental_exposure.yaml
│   └── legal_regulatory_burden.yaml
├── db/
│   ├── connection.py             # Main DB connector (DATABASE_URL)
│   └── schema.sql                # environment_runs table definition
└── scripts/
    └── debug_ontology.py         # Diagnostic: inspect csi_ontology_lab for a ticker
```

---

## 10-K retrieval

The 10-K retrieval system connects to a separate Neon database (`csi_ontology_lab`) that holds pre-ingested SEC filings chunked and embedded with **BGE bge-large-en-v1.5** (1024-dim vectors). For each PESTEL dimension, it runs a **hybrid BM25 + vector search** fused via Reciprocal Rank Fusion, pinned to the correct filing.

Filing resolution order: `10-K → 10-K_A → 10-Q → earnings_call`. Companies with non-calendar fiscal years (e.g. AAPL ends in September) fall through to the most recent quarterly filing automatically.

If `DATABASE_URL_ONTOLOGY_LAB` is not set, or a ticker has no filing in the DB, retrieval falls back silently to the local `risk_factors` table (if populated).

To diagnose retrieval issues for a specific ticker:

```bash
python scripts/debug_ontology.py AMD
```

---

## Output structure

```json
{
  "ticker": "AMD",
  "as_of_date": "2026-06-16",
  "environment_score": 58,
  "direction": "MIXED",
  "quant_score": 59.9,
  "qual_score": 57.0,
  "pestel_scores": {
    "P": {"quant": 45.0, "qual": 48, "combined": 46.5},
    "E": {"quant": 68.0, "qual": 65, "combined": 66.5},
    "S": {"quant": 52.0, "qual": 55, "combined": 53.5},
    "T": {"quant": 72.0, "qual": 70, "combined": 71.0},
    "En": {"quant": 55.0, "qual": 60, "combined": 57.5},
    "L":  {"quant": 40.0, "qual": 42, "combined": 41.0}
  },
  "flags": ["SECTOR_LEADING", "RD_LEADER", "MARGIN_LAGGARD", "CAPEX_LAGGARD"],
  "narrative": "AMD's external environment is MIXED...",
  "narrative_by_dim": {
    "Political": "Export control tightening...",
    "Economic": "Fixed business investment in tech...",
    "Social": "AI adoption accelerating...",
    "Technological": "AMD's R&D intensity 7pp above peers...",
    "Environmental": "AMD has a proactive Climate Transition Plan...",
    "Legal": "Intensifying antitrust enforcement..."
  },
  "key_tailwinds": ["..."],
  "key_risks": ["..."],
  "evidence": {
    "news_by_pestel_dim": {
      "P": [{"title": "...", "url": "...", "snippet": "...", "published_at": "..."}],
      "E": [...], "S": [...], "T": [...], "En": [...], "L": [...]
    },
    "risk_factor_excerpts_by_dim": {
      "P": ["...", "..."], "E": [...], "S": [...], "T": [...], "En": [...], "L": [...]
    },
    "risk_factor_excerpts": ["..."]
  }
}
```

---

## CLI reference

```bash
python cli.py chat                          # Interactive mode (recommended)
python cli.py run --ticker AMD              # Single run, prints JSON
python cli.py run --ticker AAPL --as-of 2026-01-15
python cli.py init-db                       # Create/migrate schema
python cli.py ingest --tickers AAPL,MSFT --years 1 --fundamentals
python cli.py backtest --ticker NVDA --start 2024-01-01 --end 2026-01-01 --freq monthly
```

---

## Configuration notes

### `config/peer_map.yaml`

> ⚠ **Requires stakeholder review before production use.** Peer groupings were seeded from GICS sector classifications and are a starting point only. Incorrect peer groups distort revenue growth gap and margin gap signals.

YAML 1.1 parses certain unquoted strings as booleans (`ON` → `true`, `NO` → `false`, `YES` → `true`). Tickers that are also YAML 1.1 keywords **must be quoted**:

```yaml
# Wrong
peers: [NVDA, AMD, ON, INTC]

# Correct
peers: [NVDA, AMD, "ON", INTC]
```

The loader filters non-string values and warns when it encounters them, but quoting the YAML is the proper fix.

---

## Limitations

- **Tavily quota**: The free tier allows ~1,000 searches/month. Each agent run makes 6 Tavily searches (one per PESTEL dimension). Results are cached for 24 hours per (sector, ticker, date) so repeated same-day runs are free.
- **10-K coverage**: Retrieval only works for tickers ingested into `csi_ontology_lab`. Tickers not in the DB fall back to the local `risk_factors` table or produce narrative from news only.
- **LLM cost**: Each run makes one Anthropic API call (~2–3k tokens). At current pricing this is ~$0.01–0.05 per run.
- **yfinance lag**: Fundamental data from yfinance may lag SEC filings by days to weeks. Backtest scores near the ingestion date may use slightly stale fundamentals.
- **pgvector optional**: If pgvector is not installed on the main DB, the local risk-factor fallback uses Postgres full-text search instead of vector similarity.
