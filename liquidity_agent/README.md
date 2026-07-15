# Liquidity Agent

An institutional-grade equity **liquidity & exit-risk scoring engine** with a natural-language
agent interface, built from the design described in `Documentation/Torch___Plan_for_Liquidity_Agent.pdf`.

The agent ingests live OHLCV data from `yfinance`, computes a battery of liquidity, volatility
and market-structure metrics, runs them through a rule-based scoring matrix, and emits a
tiered go / no-go signal (`Tier 1` Unrestricted → `Tier 4` Blacklist) accompanied by a
plain-English narrative and a confidence score.

---

## Pipeline (mirrors the PDF)

| Stage | Module | What it does |
|------:|--------|--------------|
| 1 | `src/data_ingestion/yfinance_loader.py` | Pull last 90 trading days of OHLCV + float / short% / institutional data |
| 2.1 | `src/metrics/adv_dollar.py` | Average Daily Dollar Volume (30d / 90d), 99th-percentile clipped |
| 2.2 | `src/metrics/volume_cv.py` | Volume Coefficient of Variation (30d / 90d) |
| 2.3 | `src/metrics/amihud.py` | Amihud illiquidity ratio (30d rolling) |
| 3 | `src/structural/constraints.py` | Float, short interest, top-10 institutional concentration |
| 4 | `src/liquidation/dtl.py` | Days-To-Liquidate at 1% and 5% positions |
| 5 | `src/scoring/rules.py`, `tiers.py`, `overrides.py` | Rule-based scoring + Tier assignment + false-liquidity-mirage override |
| Out | `src/output/dashboard.py`, `narrative.py`, `confidence.py` | Tier badge, narrative summary, data-confidence flagging |
| Agent | `src/agent/llm_agent.py` | Natural-language query layer (LLM-as-a-Judge ready) |

---

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt

# Optional: copy and fill in API keys for the LLM agent layer
copy .env.example .env

# Score a single ticker
python main.py score AAPL

# Ask a natural-language question (the LLM Interpretation panel
# directly answers your question in plain English)
python main.py ask "Is GME safe for a 5% position right now?"

# Compare multiple tickers side-by-side
python main.py compare AAPL GME TSLA

# Ask a comparison question (auto-routes through compare)
python main.py ask "Compare liquidity of AAPL and GME for a 5% stake"
python main.py compare AAPL GME TSLA -q "Which is safest for a 5% stake?"

# Start an interactive chat session (REPL)
python main.py chat
liquidity> How liquid is Aple right now?
liquidity> Compare apple, tesla and microsft.
liquidity> :exit
```

The agent tolerates typos and accepts company names interchangeably with
tickers — `apple`, `Apples`, `Aple`, `Aaple`, `microsft`, `Goolge` all
resolve to their correct symbols.

---

## Scoring matrix (from the PDF)

| Dimension          | Metric        | +0 (Low)   | +1 (Medium) | +2 (High)  | +3 (Critical) |
|--------------------|---------------|-----------:|------------:|-----------:|--------------:|
| Absolute Liquidity | ADV$ (30d)    | > $10M     | $2-10M      | $500K-2M   | < $500K       |
| Price Impact       | Amihud Ratio  | ≤ 0.01     | 0.01-0.05   | 0.05-0.20  | > 0.20        |
| Volume Stability   | Volume CV     | < 0.3      | 0.3-0.6     | 0.6-1.0    | > 1.0         |
| Exit Capacity      | DTL₅%         | < 1 day    | 1-3 days    | 3-5 days   | > 5 days      |
| Market Structure   | Free Float    | > 50M      | 20-50M      | 5-20M      | < 5M          |

| Score | Tier | Action |
|------:|------|--------|
| 0-4   | **Tier 1 — Unrestricted** | Safe for automated market orders. |
| 5-6   | **Tier 2 — Position-sizing caps** | Max 1% of float; TWAP / Limit only. |
| 7-8   | **Tier 3 — Algorithmic execution only** | VWAP, ≤5% participation, compliance sign-off. |
| 9+    | **Tier 4 — Blacklist** | Hard-block. |

**Structural override:** if `Float < 10M shares` and `Short% > 25%`, downgrade by **2 tiers**
(false-liquidity-mirage guard).

---

## Project layout

```
Liquidity Agent/
├── Documentation/                      Original plan PDF
├── config/
│   └── settings.py                     Thresholds, lookback windows, tier rules
├── src/
│   ├── data_ingestion/yfinance_loader.py
│   ├── metrics/{moving_averages,adv_dollar,volume_cv,amihud}.py
│   ├── structural/constraints.py
│   ├── liquidation/dtl.py
│   ├── scoring/{rules,tiers,overrides}.py
│   ├── output/{dashboard,narrative,confidence}.py
│   ├── agent/llm_agent.py
│   └── utils/{trading_days,outliers}.py
├── tests/                              pytest smoke tests
├── data/{raw,processed}/               Local caches (gitignored)
├── notebooks/                          Exploration & validation
├── main.py                             CLI entry point
└── requirements.txt
```

---

## Status

| Stage                                  | Progress |
|----------------------------------------|----------|
| Data ingestion (live yfinance)         | ✓        |
| Core metrics (ADV$, CV, Amihud, DTL)   | ✓        |
| Rule-based scoring + tier override     | ✓        |
| CLI + narrative dashboard              | ✓        |
| LLM agent (natural-language queries)   | scaffold |
| LLM-as-a-Judge evaluation harness      | todo     |
