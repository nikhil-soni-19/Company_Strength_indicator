# Torch Intern — Likely Catalyst Agent

Production-oriented research stack for **earnings-catalyst analysis** on US large-cap tech. The repo combines:

- **Neon PostgreSQL** (ontology DB: SEC filings, Bloomberg ERN/EEG, financial facts)
- **Vector + BM25 retrieval** over filing narratives
- **FinBERT / XGBoost** models for post-earnings drift (legacy pipeline)
- **Two-tier LLMs** (OpenAI light + Anthropic Sonnet thinking)
- **Optional Tavily** live news

Supported tickers in the query agent: **AAPL**, **MSFT** (more can be added via config and data ingest).

---

## Two agents (pick the right one)

| Agent | Entry script | Purpose |
|-------|--------------|---------|
| **Likely Catalyst (v2)** | `upcoming_catalyst_agent.py` | Natural-language Q&A: *“What could drive Apple’s growth?”* — filings + Bloomberg + news, **no BUY/SELL probability** |
| **Post-earnings / PEAD** | `catalyst_agent.py` | Full ML pipeline: SEC → features → XGBoost → `DecisionEngine` → BUY/HOLD/SELL-style signals |
| **Upcoming predictor** | `run_upcoming.py` | Pre-event price forecasts for upcoming earnings (uses `catalyst_agent` + Qdrant) |
| **Unified report** | `run_full_report.py` / `unified_report.py` | Merges pre-event + post-event views per ticker |

Deep dive for the query agent: **[system.md](system.md)**.

---

## Quick start

### 1. Python environment

```powershell
cd c:\Palash\task_manager_agent\torch_intern
python -m venv .venv-win
.\.venv-win\Scripts\pip install -r requirements.txt
pip install python-dotenv openai anthropic   # LLM + .env loading (if not already installed)
```

### 2. Secrets (never commit)

Create or edit **`.env`** at the project root (gitignored):

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
TAVILY_API_KEY=your-tavily-key

LLM_LIGHT_PROVIDER=openai
LLM_LIGHT_MODEL=gpt-4o-mini

LLM_THINKING_PROVIDER=anthropic
LLM_THINKING_MODEL=claude-sonnet-4-6
```

Create **`database_cloud.env`** (also gitignored) with Neon:

```env
DATABASE_URL_ONTOLOGY_LAB=postgresql://USER:PASSWORD@HOST/neondb?sslmode=require
```

### 3. Verify Neon

```powershell
.\.venv-win\Scripts\python.exe verify_neon.py
.\.venv-win\Scripts\python.exe quick_test.py
```

### 4. Run the Likely Catalyst agent

```powershell
# One-shot
.\.venv-win\Scripts\python.exe upcoming_catalyst_agent.py "What could be the catalyst for Apple's growth?"

# Interactive REPL
.\.venv-win\Scripts\python.exe upcoming_catalyst_agent.py
```

---

## Architecture (high level)

```mermaid
flowchart LR
    subgraph inputs [Inputs]
        Q[User query]
        Neon[(Neon Postgres)]
        Tavily[Tavily API]
    end

    subgraph v2 [upcoming_catalyst_agent]
        Parse[Query parse - GPT-4o mini]
        S1[Stage 1: RRF filings + ERN + EEG]
        S2[Stage 2: Tavily news]
        S3[Stage 3: Sonnet synthesis]
        Report[CatalystReport]
    end

    Q --> Parse --> S1
    Neon --> S1
    S1 --> S2
    Tavily --> S2
    S1 --> S3
    S2 --> S3
    S3 --> Report
```

```mermaid
flowchart LR
    subgraph legacy [catalyst_agent pipeline]
        SEC[sec_pipeline]
        MKT[market_pipeline]
        FEAT[feature + FinBERT + narrative]
        XGB[xgb_models]
        DEC[decision_engine]
    end

    SEC --> FEAT --> XGB --> DEC
    MKT --> FEAT
    Neon2[(Neon)] --> FEAT
```

---

## Project layout

```
torch_intern/
├── upcoming_catalyst_agent.py   # Main NL catalyst agent (v2) — START HERE for Q&A
├── catalyst_agent.py              # Post-earnings ML orchestrator
├── system.md                      # Detailed design doc for upcoming_catalyst_agent
│
├── llm.py                         # OpenAI (light) + Anthropic (thinking) router
├── retrieval.py                   # BM25 + pgvector RRF over sec_filings
├── neon_connection.py             # Async Neon engine (database_cloud.env)
├── neon_reader.py                 # Read ERN, EEG, MD&A, financial_facts
├── neon_writer.py                 # Write catalyst snapshots to Neon
├── tavily_news.py                 # Stage-2 news (1 credit/query, cached)
├── provenance.py                  # Evidence ledger (trust / source tags)
├── run_context.py                 # Degradation / limitations tracking
│
├── sec_pipeline.py                # SEC EDGAR fetch + parse
├── market_pipeline.py             # Prices, earnings surprise (yfinance fallback)
├── feature.py                     # Feature engineering for XGBoost
├── finbert_analyzer.py            # FinBERT sentiment on filing text
├── narrative.py                   # Catalyst taxonomy (rule-based + optional LLM)
├── embeddings.py                  # Sentence-transformer embeddings
├── rule_based_scorer.py           # Transparent bull/bear probabilities (no XGB)
├── decision_engine.py             # BUY / HOLD / SELL from signal scores
├── xgb_models.py                  # Trained classifier (PEAD)
├── upcoming_predictor.py          # Pre-event earnings forecasts
├── unified_report.py              # Merge pre + post event reports
├── pre_event_features.py          # Features before announcement
├── outcome_tracker.py             # Track prediction outcomes
├── engine.py                      # Backtest metrics (BHAR, Sharpe, etc.)
│
├── run_upcoming.py                # CLI: upcoming predictor JSON
├── run_full_report.py             # CLI: unified upcoming / retrospective reports
├── run_smoke.py                   # Smoke: Neon + Qdrant + numeric signal
├── quick_test.py                  # Stack sanity checks
├── verify_neon.py                 # Neon connection + table counts
├── db_audit.py                    # Raw AAPL audit (financial_facts, ERN, prices)
├── build_training_dataset.py      # Training data for XGBoost
├── build_catalyst_pptx.py           # Generate presentation assets
│
├── database/                      # Sync ingest utilities + sample Bloomberg xlsx
│   ├── ontology.py                # Schema DDL + upserts
│   ├── ingest_estimates.py        # Load EEG/ERN Excel → Neon
│   ├── market_estimates.py        # Parse Bloomberg workbooks
│   ├── pipeline.py                # Filing ingestion pipeline
│   ├── db.py                      # Sync SQLAlchemy engine
│   ├── companies.py               # Company config
│   ├── AAPL - EEG.xlsx            # Sample Bloomberg EEG export
│   └── AAPL - ERN.xlsx            # Sample Bloomberg ERN export
│
├── datatypes.py                   # SECFiling, EarningsEvent dataclasses
├── enums.py                       # DriftLabel, FilingType
├── settings.py                    # Central env-based config
├── logger.py                      # Logging helper
├── data_quality.py                # Financial row validation
├── retry.py                       # Retry utilities
├── upcoming_calendar.py           # Earnings calendar helpers
├── eps_models.py                  # EPS-related model helpers
├── init.py                        # Package version marker
│
├── requirements.txt               # Core Python dependencies
├── database_cloud.env             # Neon DSN (gitignored — create locally)
├── .env                           # API keys (gitignored)
├── .gitignore
│
├── catalyst_agent_presentation.*    # LaTeX / PPTX deck assets
├── pead_agent_presentation.tex
└── .tavily_cache/                 # Same-day Tavily response cache
```

Virtual environments (`.venv/`, `.venv-win/`) are local — not part of source control.

---

## Data in Neon (`ontology` schema)

Data lives in **PostgreSQL on Neon**, not Neo4j.

| Logical name | Bloomberg / source | Table | Used by |
|--------------|----------------------|-------|---------|
| **ERN** | ERN workbook | `ontology.earnings_surprise` | Beat history, surprise %, next earnings date |
| **EEG** | EEG workbook | `ontology.estimate_consensus` | Point-in-time consensus EPS revisions |
| **Filings** | SEC + earnings calls | `ontology.filings`, `ontology.sec_filings` | RRF retrieval, MD&A text |
| **Facts** | XBRL / extracted | `ontology.financial_facts` | Revenue, margins (joined via `filing_id`) |
| **Prices** | Bloomberg BDH | `ontology.price_daily` | Market pipeline, charts |

Example SQL (AAPL ERN):

```sql
SELECT fiscal_period, announcement_date, surprise_pct, price_change_pct, pe_ratio
FROM ontology.earnings_surprise
WHERE ticker = 'AAPL' AND is_reported = true
ORDER BY announcement_date DESC
LIMIT 8;
```

Example SQL (AAPL EEG FY-2026):

```sql
SELECT as_of_date, value_mean
FROM ontology.estimate_consensus
WHERE ticker = 'AAPL' AND target_period = 'FY-2026' AND upper(metric) = 'EPS'
ORDER BY as_of_date;
```

---

## Ingesting Bloomberg ERN / EEG

Sample workbooks are under `database/`. To load into Neon (sync path):

```powershell
# From repo root — adjust module path if your layout uses src.*
python database/ingest_estimates.py --tickers AAPL --dry-run
python database/ingest_estimates.py --tickers AAPL
```

Requires `DATABASE_URL_ONTOLOGY_LAB` in `.env` or `.env.ingestion` / `database_cloud.env`.

---

## Scripts reference

| Script | What it does |
|--------|----------------|
| `upcoming_catalyst_agent.py` | **Likely Catalyst** interactive / one-shot Q&A |
| `catalyst_agent.py` | Import as library; `LikelyCatalystAgent.predict()` |
| `run_upcoming.py` | Upcoming earnings JSON predictions (needs Qdrant) |
| `run_full_report.py` | Full unified report (`--retro` for past events) |
| `run_smoke.py` | Quick Neon + Qdrant + `predict_numeric` smoke test |
| `quick_test.py` | Multi-check health of ontology tables |
| `verify_neon.py` | Connection test + row counts |
| `db_audit.py` | Deep AAPL financial_facts / ERN / price audit |
| `build_training_dataset.py` | Build XGB training set from Neon |
| `build_catalyst_pptx.py` | Build PowerPoint from stack stats |

---

## Module responsibilities (core library)

| Module | Role |
|--------|------|
| `upcoming_catalyst_agent.py` | 3-stage catalyst workflow + `CatalystReport` |
| `llm.py` | `complete_light()` / `complete_thinking()` with `.env` routing |
| `retrieval.py` | `retrieve_filing_evidence()` — BM25 + 1024-dim vector RRF |
| `neon_reader.py` | Async readers: ERN, EEG, MD&A, financial history |
| `neon_connection.py` | Async SQLAlchemy + circuit breaker |
| `tavily_news.py` | Post–last-report news; 1 Tavily call per query |
| `provenance.py` | `Ledger` — DB vs LLM vs fallback trust table |
| `run_context.py` | Records parser/news/Neon degradations for the report |
| `catalyst_agent.py` | SEC + market + FinBERT + XGB + decision engine |
| `decision_engine.py` | Maps probabilities → BUY / HOLD / SELL |
| `rule_based_scorer.py` | Explainable bull/bear weights (no ML file) |
| `finbert_analyzer.py` | `ProsusAI/finbert` document sentiment |
| `narrative.py` | Catalyst keyword taxonomy + optional BART |
| `feature.py` | Feature vector builder for training/inference |
| `xgb_models.py` | Load/predict saved XGBoost catalyst model |
| `qdrant_manager.py` | Vector DB for filing embeddings (legacy pipeline) |
| `settings.py` | Env defaults: DB, Qdrant, models, thresholds |

---

## External services

| Service | Required for | Config |
|---------|----------------|--------|
| **Neon Postgres** | All DB-backed features | `database_cloud.env` |
| **OpenAI** | Query parse, themes (light tier) | `OPENAI_API_KEY` |
| **Anthropic** | Catalyst synthesis (thinking tier) | `ANTHROPIC_API_KEY` |
| **Tavily** | Live news (optional) | `TAVILY_API_KEY` |
| **Qdrant** | `catalyst_agent` / `run_upcoming` embeddings | `QDRANT_HOST`, `QDRANT_PORT` |
| **SEC EDGAR** | Filing fetch | `SEC_USER_AGENT` in `settings.py` |

---

## Dependencies (`requirements.txt`)

Core stack: `sqlalchemy[asyncio]`, `asyncpg`, `qdrant-client`, `sentence-transformers`, `transformers`, `torch`, `xgboost`, `scikit-learn`, `pandas`, `yfinance`, `httpx`, `beautifulsoup4`, `scipy`.

Install LLM extras separately if needed: `openai`, `anthropic`, `python-dotenv`.

First run downloads **FinBERT** and **BGE-large-en-v1.5** (~GB) for sentiment and retrieval.

---

## Output: Likely Catalyst report

The v2 agent prints a structured report including:

- **Likely catalyst** (one sentence) + directional lean + horizon (next earnings from ERN)
- **Evidence** tagged `[DB·ERN]`, `[DB·EEG]`, `[DB·filing]`, `[DB·risk]`, `[LLM]`, `[news]`
- **Downside catalysts to watch** (always populated)
- **Trust summary** — HIGH for Bloomberg/DB rows, LOW for Tavily
- **Limitations** — keyword fallback, missing news, etc.

Evidence trust and directional lean strength are **separate** concepts (see `system.md`).

---

## Security

- **Do not commit** `.env`, `database_cloud.env`, or API keys.
- `.gitignore` excludes `*.env`, models (`*.pkl`), and virtualenvs.
- Rotate keys if they were ever pasted into chat or committed by mistake.

---

## Documentation

| Doc | Contents |
|-----|----------|
| [system.md](system.md) | End-to-end flow of `upcoming_catalyst_agent.py` |
| This README | Repo-wide map, setup, scripts, data model |

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `Could not resolve authentication` (Anthropic/OpenAI) | Fill `.env` keys; restart shell so dotenv loads |
| `PARSE=KEYWORD` in report | Light LLM failed — check `OPENAI_API_KEY` |
| No Tavily news | Missing `TAVILY_API_KEY` or no items after last report date |
| Neon connection errors | Check `database_cloud.env` host/SSL; run `verify_neon.py` |
| Vector search slow first time | BGE model loading — normal on first query |
| `db_audit.py` Unicode error on Windows | Set `$env:PYTHONIOENCODING='utf-8'` |

---

## License / status

Internal research / intern project. Not a production trading system. Model outputs are interpretive — verify Bloomberg and filing numbers in Neon before acting on them.
