"""
upcoming_catalyst_agent.py — Query-driven Likely Catalyst Agent (v2).

3-stage workflow:
  Stage 1  Company's own voice  — spine-scoped BM25+vector RRF over the last
           reported 10-Q/8-K MD&A + earnings-call transcript (what the company
           itself said), plus ERN (Bloomberg earnings history) and EEG
           (point-in-time consensus EPS trajectory, 2-yr).
  Stage 2  Grounded live news   — themes extracted from Stage 1 steer a single
           budget-frugal Tavily search, time-boxed to AFTER the last report
           date (point-in-time discipline).
  Stage 3  Synthesis            — Claude Sonnet fuses Stage 1 + Stage 2 + ERN +
           EEG into a structured CatalystReport: catalyst, direction, horizon
           (anchored to the next earnings date), provenance-tagged evidence,
           qualitative confidence, limitations. NO probability / BUY-HOLD.

Usage:
    python upcoming_catalyst_agent.py "What could be the catalyst for Apple's growth?"
    python upcoming_catalyst_agent.py            # interactive REPL
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
    else:
        print(
            f"[llm-config] WARNING: {_ENV_PATH} not found — LLM/Tavily keys "
            "not loaded; agent will use free fallbacks.",
            file=sys.stderr,
        )
except ImportError:
    print(
        "[llm-config] WARNING: python-dotenv not installed — .env NOT loaded. "
        "Run: pip install python-dotenv",
        file=sys.stderr,
    )

import horizons
import llm
import neon_reader
import tavily_news
from logger import get_logger
from neon_connection import is_neon_available
from provenance import Ledger, Source, Trust
from run_context import RunContext, Severity

logger = get_logger(__name__)

# ── Taxonomy ──────────────────────────────────────────────────────────────────

_TICKER_KEYWORDS: Dict[str, str] = {
    "apple": "AAPL", "aapl": "AAPL", "iphone": "AAPL", "ipad": "AAPL",
    "ios": "AAPL", "macos": "AAPL", "mac": "AAPL", "airpods": "AAPL",
    "app store": "AAPL", "tim cook": "AAPL", "vision pro": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT", "azure": "MSFT", "windows": "MSFT",
    "copilot": "MSFT", "teams": "MSFT", "office": "MSFT", "xbox": "MSFT",
    "satya": "MSFT", "nadella": "MSFT", "365": "MSFT",
}

_FY_END_MONTH: Dict[str, int] = {"AAPL": 9, "MSFT": 6}

EVENT_TYPES = [
    "product_launch", "earnings", "guidance", "macro", "regulatory", "partnership",
]

INTENTS = [
    "growth_catalyst", "risk", "earnings",
    "product_launch", "guidance", "valuation", "general",
]

_INTENT_KEYWORDS: Dict[str, tuple] = {
    "growth_catalyst": ("catalyst", "grow", "growth", "upside", "tailwind",
                        "drive", "bull case", "accelerat", "expand",
                        "opportunit", "what could", "boost", "momentum"),
    "risk":            ("risk", "downside", "headwind", "threat", "bear case",
                        "concern", "slowdown", "decline", "weaken", "pressure",
                        "litigation", "antitrust", "regulat"),
    "earnings":        ("earnings", "eps", "beat", "miss", "quarter result",
                        "report card", "results", "surprise"),
    "product_launch":  ("launch", "release", "unveil", "introduce", "ship "),
    "guidance":        ("guidance", "outlook", "forecast", "raise", "cut"),
    "valuation":       ("valuation", "multiple", "p/e", "pe ratio", "expensive",
                        "cheap", "overvalued", "undervalued", "fair value"),
}

_INTENT_SECTIONS: Dict[str, List[str]] = {
    "growth_catalyst": ["revenue"],
    "risk":            ["risk_factors"],
    "earnings":        ["financials", "revenue"],
    "guidance":        ["revenue"],
    "valuation":       ["financials"],
    "product_launch":  ["revenue"],
    "general":         [],
}

_INTENT_DEFAULT_SUBJECT: Dict[str, str] = {
    "growth_catalyst": "growth drivers",
    "risk":            "risk factors",
    "earnings":        "quarterly earnings",
    "product_launch":  "product launch",
    "guidance":        "forward guidance",
    "valuation":       "valuation",
    "general":         "overall performance",
}


def _classify_intent(query: str) -> str:
    q = f" {query.lower()} "
    scores = {
        intent: sum(1 for kw in kws if kw in q)
        for intent, kws in _INTENT_KEYWORDS.items()
    }
    best = max(scores, key=lambda i: scores[i])
    return best if scores[best] > 0 else "general"


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class CatalystReport:
    """Structured catalyst output — a statement + evidence, NOT a probability."""
    query: str
    ticker: str
    intent: str
    parse_mode: str
    catalyst: str
    direction: str           # bullish | bearish | mixed
    horizon: str
    confidence: str          # directional-lean strength: high|medium|low
    rationale: List[str]
    evidence: List[str]
    overall_trust: str
    downside: List[str] = field(default_factory=list)   # always populated
    caveat: str = ""         # flagged when headline rests on unverified news
    trust_summary: str = ""
    limitations_text: str = ""
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[Dict[str, Any]] = field(default_factory=list)
    data_sources: Dict[str, int] = field(default_factory=dict)
    horizons_profile: Optional[Dict[str, Any]] = None
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "ticker": self.ticker,
            "intent": self.intent,
            "parse_mode": self.parse_mode,
            "likely_catalyst": {
                "catalyst": self.catalyst,
                "direction": self.direction,
                "horizon": self.horizon,
                "directional_lean_strength": self.confidence,
                "rationale": self.rationale,
                "evidence": self.evidence,
                "downside_to_watch": self.downside,
                "caveat": self.caveat,
            },
            "evidence_trust": self.overall_trust,
            "provenance": self.provenance,
            "limitations": self.limitations,
            "data_sources": self.data_sources,
            "horizons": self.horizons_profile,
            "generated_at": self.generated_at,
        }

    def to_text(self) -> str:
        W = 66
        pad = lambda n: "─" * max(0, n)
        parse_tag = "" if self.parse_mode == "llm" else "  ·  PARSE=KEYWORD"
        lines = [
            "═" * W,
            f"  LIKELY CATALYST  ·  {self.ticker}  ·  "
            f"{self.intent.upper().replace('_', ' ')}{parse_tag}",
            f"  {self.generated_at[:19]} UTC  ·  lean={self.direction.upper()}"
            f"/{self.confidence.upper()}  ·  evidence-trust="
            f"{self.overall_trust.upper()}",
            "═" * W,
            "",
            f"  QUERY   {self.query}",
            "",
            f"  ── LIKELY CATALYST {pad(W - 20)}",
            f"  {self.catalyst}",
            "",
            f"  Directional lean : {self.direction.upper()} "
            f"(strength {self.confidence.upper()}, qualitative — not a probability)",
            f"  Horizon          : {self.horizon}",
            f"  Evidence trust   : {self.overall_trust.upper()}  "
            "(how verifiable the evidence is — independent of the lean)",
        ]
        if self.horizons_profile and self.horizons_profile.get("buckets"):
            lines += ["", f"  ── HORIZON BUCKETS {pad(W - 19)}"]
            for bkt in ("0-6m", "6-24m", "structural", "unscheduled"):
                items = self.horizons_profile["buckets"].get(bkt) or []
                if not items:
                    continue
                lines.append(f"  [{bkt}]")
                for m in items[:4]:
                    desc = str(m.get("description", ""))[:70]
                    ev = str(m.get("evidence_span", ""))[:50]
                    src = m.get("source", "filing")
                    lines.append(f"    • {desc}  ({src})")
                    if ev and src != "ern":
                        lines.append(f"      \"{ev}…\"")
        if self.caveat:
            lines += ["", f"  ⚠ KEY-CLAIM CAVEAT: {self.caveat}"]
        if self.limitations_text:
            lines += ["", self.limitations_text]
        lines += ["", f"  ── EVIDENCE {pad(W - 13)}"]
        for b in self.evidence:
            lines.append(f"  • {b}")
        if not self.evidence:
            lines.append("  • Insufficient data for this query")
        lines += ["", f"  ── DOWNSIDE CATALYSTS TO WATCH {pad(W - 31)}"]
        for d in self.downside:
            lines.append(f"  • {d}")
        if not self.downside:
            lines.append("  • None judged material in the available data "
                          "(actively checked, not skipped)")
        lines += ["", f"  ── RATIONALE {pad(W - 14)}"]
        rat = self.rationale if isinstance(self.rationale, list) else [self.rationale]
        for bullet in rat:
            # Wrap long bullets at ~62 chars so they fit the report width
            words = str(bullet).split()
            line = "  • "
            for w in words:
                if len(line) + len(w) + 1 > W - 2 and line.strip() != "•":
                    lines.append(line.rstrip())
                    line = "    " + w + " "
                else:
                    line += w + " "
            if line.strip():
                lines.append(line.rstrip())
        lines.append("")
        if self.trust_summary:
            lines += [self.trust_summary, ""]
        lines.append(f"  ── DATA SOURCES {pad(W - 16)}")
        for src, count in self.data_sources.items():
            lines.append(f"  {src:<44} {count:>4}")
        lines.append("═" * W)
        return "\n".join(lines)


# ── Agent ─────────────────────────────────────────────────────────────────────

class UpcomingCatalystAgent:
    """3-stage query-driven likely-catalyst agent."""

    # subject keyword → audited sec_filings.section tag
    _SECTION_HINTS = {
        "risk_factors": ("risk", "lawsuit", "litigation", "regulat",
                         "antitrust", "headwind", "investigation"),
        "revenue": ("revenue", "sales", "iphone", "ipad", "mac", "services",
                    "azure", "office", "windows", "xbox", "segment",
                    "product", "growth", "demand"),
        "cash_and_capital": ("cash", "capital", "buyback", "repurchase",
                             "dividend", "liquidity", "debt", "free cash flow"),
        "financials": ("margin", "eps", "earnings", "net income",
                       "operating income", "gross margin", "profit"),
    }

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze(self, query: str) -> CatalystReport:
        logger.info(f"Analyzing: {query!r}")
        ctx = RunContext()
        ledger = Ledger()

        parsed = await self._parse_query(query, ctx)
        ticker = parsed["ticker"]
        intent = parsed.get("intent", "general")
        key_subject = parsed["key_subject"]
        parse_mode = parsed.get("parse_mode", "llm")
        logger.info(
            f"  ticker={ticker} intent={intent} subject={key_subject!r} "
            f"parse={parse_mode}"
        )
        if parse_mode == "llm":
            ledger.add("Query understanding", f"intent={intent}",
                       source=Source.LLM, trust=Trust.MEDIUM)
        else:
            ledger.add("Query understanding", f"intent={intent} (keyword)",
                       source=Source.FALLBACK, trust=Trust.LOW,
                       note="LLM parse unavailable; framing approximate")

        # ── STAGE 1: company's own voice + ERN + EEG ─────────────────────────
        chunks = await self._search_relevant_chunks(ticker, key_subject, intent, 6)
        mda_text = await neon_reader.get_mda_text(ticker)
        if not is_neon_available():
            ctx.degrade("neon", "Neon unreachable",
                        "Built from cached/partial data; figures may be "
                        "incomplete.", Severity.CRITICAL)

        s1_text = self._pick_relevant_text(mda_text, chunks, key_subject)
        quote = ""
        if chunks:
            c0 = chunks[0]
            fy, fq = c0.get("fiscal_year", ""), c0.get("fiscal_quarter", "")
            plabel = (f"FY{fy}{'Q'+str(fq) if fq else ''}" if fy
                      else str(c0.get("period_end_date", "prior period")))
            snip = (c0.get("context") or "").replace("\n", " ").strip()[:160]
            if snip:
                quote = f'{ticker} {plabel} filing: "{snip}…"'
                ledger.add("Filing quote (company's own words)",
                           f"section={c0.get('section')}, level={c0.get('level')}",
                           source=Source.DB_VERIFIED, trust=Trust.HIGH,
                           as_of=c0.get("period_end_date"))

        # Dedicated BEAR pass — runs regardless of query intent so the
        # downside is never excluded by phrasing (Job-A two-sided requirement).
        bear_chunks = await self._search_bear_chunks(ticker, key_subject)
        bear_quote = ""
        if bear_chunks:
            b0 = bear_chunks[0]
            bsnip = (b0.get("context") or "").replace("\n", " ").strip()[:160]
            if bsnip:
                bear_quote = f'{ticker} risk_factors: "{bsnip}…"'
                ledger.add("Risk-factor quote (bear pass)",
                           f"section={b0.get('section')}, level={b0.get('level')}",
                           source=Source.DB_VERIFIED, trust=Trust.HIGH,
                           as_of=b0.get("period_end_date"))

        ern = await neon_reader.get_earnings_history(ticker, 8)
        nxt = await neon_reader.get_next_earnings_date(ticker)
        tps = self._fy_targets(ticker)
        eeg = {tp: await neon_reader.get_estimate_trajectory(ticker, tp, 730)
               for tp in tps}

        ern_summary = self._summarise_ern(ern)
        last_report = ern[0]["announcement_date"] if ern else None
        if ern_summary:
            ledger.add("Earnings history (ERN, Bloomberg)",
                       ern_summary["headline"],
                       source=Source.DB_VERIFIED, trust=Trust.HIGH,
                       as_of=last_report)
        for tp, tr in eeg.items():
            if tr["n_observations"]:
                ledger.add(f"Consensus EPS trajectory {tp} (EEG, 2-yr)",
                           f"revision {self._pct(tr['revision_pct'])}, "
                           f"{tr['n_observations']} obs",
                           source=Source.DB_VERIFIED, trust=Trust.HIGH,
                           as_of=tr["as_of_latest"])

        themes = await self._extract_themes(s1_text, ticker, key_subject)

        # ── Horizons: L2 milestone extract (cached per accession) + ERN anchor ─
        horizons_prof = await horizons.build_horizons_profile(
            ticker, date.today(), nxt, max_filings=2,
        )
        horizon = horizons.format_horizon_summary(horizons_prof)
        n_milestones = len(horizons_prof.all_milestones)
        if horizons_prof.anchor:
            ledger.add(
                "Horizon anchor (ERN next earnings)",
                str(horizons_prof.anchor.get("event_date", "")),
                source=Source.DB_VERIFIED, trust=Trust.HIGH,
                as_of=horizons_prof.anchor.get("event_date"),
            )
        if n_milestones:
            ledger.add(
                "Future milestones (L2 filing extract)",
                f"{n_milestones} items; {horizons_prof.llm_extractions} new LLM, "
                f"{horizons_prof.cache_hits} cached",
                source=Source.LLM if horizons_prof.llm_extractions else Source.COMPUTED,
                trust=Trust.MEDIUM if horizons_prof.llm_extractions else Trust.HIGH,
                note="bucketed 0-6m / 6-24m / structural / unscheduled",
            )
        elif not llm.thinking_available():
            ctx.degrade(
                "horizons",
                "Milestone L2 pass skipped (thinking LLM unavailable)",
                "Horizon uses ERN earnings anchor only.",
                Severity.INFO,
            )

        # ── STAGE 2: grounded live news (budget-frugal, point-in-time) ───────
        # Two parallel Tavily calls:
        #   A) theme-grounded — anchored to what the filing discussed
        #   B) event-grounded — forward-looking: CEO changes, conferences,
        #      product launches, partnerships — NOT derivable from the filing
        news_themes = (themes.get("themes", [])[:3]
                       + themes.get("risk_themes", [])[:3])
        news: List[Dict[str, Any]] = []
        event_news: List[Dict[str, Any]] = []
        if tavily_news.available():
            loop = asyncio.get_event_loop()
            # Run both fetches concurrently (each is synchronous/blocking,
            # so we use run_in_executor for each and gather the futures).
            news_fut = loop.run_in_executor(
                None,
                lambda: tavily_news.fetch_news(ticker, news_themes, last_report),
            )
            event_fut = loop.run_in_executor(
                None,
                lambda: tavily_news.fetch_upcoming_events(ticker),
            )
            news, event_news = await asyncio.gather(news_fut, event_fut)
            news = news or []
            event_news = event_news or []

            if news:
                ledger.add("Live news (Tavily, theme-grounded)",
                           f"{len(news)} items since {last_report}",
                           source=Source.LLM, trust=Trust.LOW,
                           as_of=date.today(),
                           note="web sources, unverified; post last-report only")
            else:
                ctx.degrade("news", "No relevant news returned",
                            "Catalyst rests on filings + ERN/EEG only.",
                            Severity.INFO)

            if event_news:
                ledger.add(
                    "Upcoming events (Tavily, forward-looking)",
                    f"{len(event_news)} items — CEO changes, launches, "
                    f"conferences, partnerships (90-day window)",
                    source=Source.LLM, trust=Trust.LOW,
                    as_of=date.today(),
                    note="web sources, unverified; event-focused, not filing-anchored",
                )
        else:
            ctx.degrade("news", "Tavily key absent",
                        "Live-news leg skipped; catalyst built from the "
                        "company's filings + ERN/EEG only.", Severity.WARN)

        # ── STAGE 3: synthesis → structured catalyst ─────────────────────────
        synth = None
        if llm.thinking_available():
            synth = await self._synthesize_catalyst(
                query, ticker, intent, key_subject, quote, bear_quote,
                themes, ern_summary, eeg, nxt, news, event_news,
                horizons_prof, ctx,
            )
        if synth:
            ledger.add("Catalyst synthesis", "thinking tier",
                       source=Source.LLM, trust=Trust.MEDIUM,
                       note="grounded in the evidence above; interpretation")
        else:
            synth = self._fallback_catalyst(themes, ern_summary, eeg)
            ledger.add("Catalyst synthesis", "deterministic fallback",
                       source=Source.FALLBACK, trust=Trust.LOW,
                       note="thinking tier unavailable; numbers-only heuristic")

        evidence = self._compose_evidence(
            quote, bear_quote, ern_summary, eeg, news, event_news,
            synth, horizons_prof,
        )

        return CatalystReport(
            query=query,
            ticker=ticker,
            intent=intent,
            parse_mode=parse_mode,
            catalyst=synth.get("catalyst", "No clear catalyst in available data"),
            direction=synth.get("direction", "mixed"),
            horizon=horizon,
            confidence=synth.get("confidence", "low"),
            rationale=synth.get("rationale", []),
            evidence=evidence,
            downside=synth.get("downside", []),
            caveat=synth.get("caveat", ""),
            overall_trust=ledger.overall_trust().value,
            trust_summary=ledger.trust_table(),
            limitations_text=ctx.limitations_block(),
            provenance=ledger.to_dict(),
            limitations=ctx.to_dict(),
            horizons_profile=horizons_prof.to_dict(),
            data_sources={
                "sec_filings chunks (RRF)":    len(chunks),
                "ERN earnings rows":           len(ern),
                "EEG series points":           sum(t["n_observations"] for t in eeg.values()),
                "Tavily news items":           len(news),
                "Tavily upcoming event items": len(event_news),
                "mda_text chars":              len(mda_text or ""),
                "Horizon milestones":          n_milestones,
            },
        )

    # ── Stage helpers ─────────────────────────────────────────────────────────

    def _fy_targets(self, ticker: str) -> List[str]:
        today = date.today()
        fy_end = _FY_END_MONTH.get(ticker, 12)
        fy = today.year + 1 if today.month > fy_end else today.year
        return [f"FY-{fy}", f"FY-{fy + 1}"]

    @staticmethod
    def _pct(v: Optional[float]) -> str:
        return f"{v*100:+.1f}%" if v is not None else "N/A"

    def _summarise_ern(self, ern: List[Dict]) -> Dict[str, Any]:
        if not ern:
            return {}
        n = len(ern)
        beats = sum(1 for r in ern
                    if r.get("surprise_pct") is not None and float(r["surprise_pct"]) > 0)
        latest = ern[0]
        sp = latest.get("surprise_pct")
        sp = (float(sp) / 100.0 if sp is not None and abs(float(sp)) > 1.5
              else (float(sp) if sp is not None else None))
        gap = None
        if latest.get("guidance_eps") is not None and latest.get("estimate_eps") is not None:
            gap = float(latest["guidance_eps"]) - float(latest["estimate_eps"])
        return {
            "headline": f"{beats}/{n} beats; latest surprise {self._pct(sp)}",
            "beats": beats, "n": n, "latest_surprise_pct": sp,
            "latest_guidance_eps": _f(latest.get("guidance_eps")),
            "latest_consensus_eps": _f(latest.get("estimate_eps")),
            "guidance_vs_consensus": _f(gap),
            "latest_pe_ratio": _f(latest.get("pe_ratio")),
            "latest_reaction_pct": _f(latest.get("price_change_pct")),
        }

    def _horizon_str(self, nxt: Optional[Dict]) -> str:
        if nxt and nxt.get("announcement_date"):
            d = nxt["announcement_date"]
            try:
                days = (d - date.today()).days
                return (f"next earnings ≈ {d} (~{days}d) — "
                        f"{nxt.get('fiscal_period', '')}".strip(" —"))
            except Exception:
                return f"next earnings ≈ {d}"
        return "next 1–2 quarters (no scheduled date in data)"

    # Canonical bear themes — used so the bear pass never depends solely on a
    # bull-skewed filing. Specifics come from the risk_factors retrieval.
    _RISK_THEMES = [
        "regulation antitrust App Store DMA", "China demand weakness",
        "tariffs", "gross margin pressure", "competition", "supply chain risk",
    ]

    async def _extract_themes(
        self, s1_text: Optional[str], ticker: str, key_subject: str
    ) -> Dict[str, List[str]]:
        """Light-tier extraction of BOTH growth themes and risk themes the
        filing raises. Always returns risk themes (canonical fallback) so the
        bear side is never silently dropped."""
        bull_fb = [t for t in (
            [key_subject] + (self._merged_section_hints(key_subject, "general") or [])
        ) if t][:4] or [key_subject]
        out = {"themes": bull_fb, "risk_themes": list(self._RISK_THEMES)}
        if not s1_text or not llm.light_available():
            return out
        system = (
            "From the company's own filing text, extract two lists. Return ONLY "
            'JSON: {"themes": [3-5 growth/product/segment drivers], '
            '"risk_themes": [3-5 risks/headwinds the filing itself raises — '
            "tariffs, regulation, China, margins, competition, etc.]}"
        )
        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: llm.complete_light(
                    system, f"{ticker} filing text:\n{s1_text[:3000]}",
                    json_mode=True,
                ),
            )
            data = llm.parse_json(raw)
            th = [str(t).strip() for t in (data.get("themes") or []) if str(t).strip()]
            rk = [str(t).strip() for t in (data.get("risk_themes") or []) if str(t).strip()]
            if th:
                out["themes"] = th[:5]
            # union filing-specific risks with the canonical floor
            out["risk_themes"] = (rk[:5] or []) + [
                r for r in self._RISK_THEMES if r not in rk
            ]
            out["risk_themes"] = out["risk_themes"][:6]
        except Exception as e:
            logger.warning(f"theme extraction failed ({e}); heuristic fallback")
        return out

    async def _search_bear_chunks(
        self, ticker: str, key_subject: str
    ) -> List[Dict]:
        """Dedicated bear-side retrieval over risk_factors — runs regardless of
        the query's intent so downside is never excluded by phrasing."""
        from retrieval import retrieve_filing_evidence, RetrievalHints
        bear_q = f"risks headwinds {key_subject} regulation competition decline pressure"
        return await retrieve_filing_evidence(
            ticker, query=bear_q,
            hints=RetrievalHints(section=["risk_factors"], top_k=4),
        )

    _SYNTH_SYSTEM = (
        "You are an equity catalyst analyst. Inputs:\n"
        "  (a) what the company said last quarter [filing]\n"
        "  (b) its OWN risk-factor language [filing/risk]\n"
        "  (c) Bloomberg earnings history [ERN]\n"
        "  (d) point-in-time consensus EPS revision trajectory [EEG, 2-yr]\n"
        "  (e) theme-grounded news published AFTER the last report [news]\n"
        "  (f) upcoming_events_news — forward-looking events NOT in the filing:\n"
        "      CEO/leadership changes, developer conferences (WWDC etc.),\n"
        "      major product launches, AI/chip partnerships, acquisitions,\n"
        "      regulatory decisions [upcoming_events]\n"
        "  (g) next earnings date and bucketed future milestones [horizons]\n"
        "  (h) known limitations\n\n"

        "STEP 1 — SCAN upcoming_events_news FIRST (before reading anything else).\n"
        "For every item in upcoming_events_news ask: does this describe a\n"
        "scheduled event within the next 6 months? If YES, it MUST appear as\n"
        "the primary or co-primary catalyst — it is more actionable than any\n"
        "filing theme because it is imminent and post-filing. Examples of\n"
        "mandatory primary catalysts: CEO/executive change, annual developer\n"
        "conference (WWDC, Build, Google I/O), new product launch cycle,\n"
        "major AI/chip partnership announcement, regulatory ruling.\n"
        "NEVER produce a report that omits a scheduled event present in\n"
        "upcoming_events_news — if it is there, it must appear in 'catalyst'\n"
        "and in 'rationale'.\n\n"

        "STEP 2 — Cross-check with ERN/EEG for confirmation or conflict.\n"
        "Interpret news AGAINST the EEG revision trend (agreement =\n"
        "confirmation; against a flat trend = unconfirmed).\n\n"

        "STEP 3 — Produce the output. Use ONLY the provided evidence — never\n"
        "invent numbers/dates/facts, never output probabilities, respect\n"
        "the limitations.\n\n"

        "TWO-SIDED REQUIREMENT: always populate 'downside' with concrete bear\n"
        "catalysts/risks, drawn from risk-factor evidence and risk_themes,\n"
        "EVEN IF the question is bullishly phrased. Never return empty downside.\n\n"

        "CAVEAT REQUIREMENT: if the headline catalyst depends on a figure from\n"
        "news NOT corroborated by ERN/EEG, state that in 'caveat'. Otherwise\n"
        "set 'caveat' to \"\".\n\n"

        "Return ONLY JSON:\n"
        "{\n"
        '  "catalyst": "one concise sentence naming the primary catalyst",\n'
        '  "direction": "bullish|bearish|mixed",\n'
        '  "confidence": "high|medium|low",\n'
        '  "rationale": [\n'
        '    "• [source] Specific data point or event + exactly why it answers '
        'the user query. Source tags: [upcoming_events] [news] [ERN] [EEG] '
        '[filing] [filing/risk]",\n'
        '    "• [source] Second distinct point — different source or angle",\n'
        '    "• [source] Third point — supporting, qualifying, or conflicting",\n'
        '    "• [source] Fourth point (optional) — only if it adds new info"\n'
        "  ],\n"
        '  "evidence": ["short attributed bullet per piece of evidence used"],\n'
        '  "downside": ["concrete bear catalyst or risk to watch"],\n'
        '  "caveat": "unverified-headline note, or empty string"\n'
        "}"
    )

    async def _synthesize_catalyst(
        self, query, ticker, intent, key_subject, quote, bear_quote, themes,
        ern_summary, eeg, nxt, news, event_news, horizons_prof, ctx,
    ) -> Optional[Dict[str, Any]]:
        bundle = {
            "user_query": query, "ticker": ticker, "intent": intent,
            "key_subject": key_subject,

            # ── SCAN THIS FIRST (system prompt Step 1) ──────────────────────
            # Forward-looking events NOT derivable from the filing:
            # CEO transitions, WWDC/conferences, product launches, partnerships.
            "upcoming_events_news": [
                {
                    "title":        n["title"],
                    "published":    n.get("published_date"),
                    "snippet":      n.get("content", "")[:350],
                    "url":          n.get("url"),
                    "source_tier":  n.get("source_tier", "Upcoming Events"),
                }
                for n in (event_news or [])
            ],

            # ── Filing-grounded evidence ─────────────────────────────────────
            "company_said": {
                "quote": quote,
                "growth_themes": themes.get("themes", []),
            },
            "company_risk_factors": {
                "quote": bear_quote,
                "risk_themes": themes.get("risk_themes", []),
            },
            "ern_earnings_history": ern_summary,
            "eeg_consensus_trajectory_2yr": {
                tp: {k: tr[k] for k in (
                    "revision_pct", "recent_4w_delta_pct", "slope_per_quarter",
                    "latest", "n_observations")}
                for tp, tr in eeg.items()
            },
            "next_earnings": (
                {"date": str(nxt.get("announcement_date")),
                 "fiscal_period": nxt.get("fiscal_period")}
                if nxt else None
            ),
            "horizon_milestones": horizons_prof.to_dict(),

            # ── Theme-grounded news (post-last-report) ───────────────────────
            "news_since_last_report": [
                {"title": n["title"], "published": n.get("published_date"),
                 "snippet": n.get("content", "")[:300], "url": n.get("url")}
                for n in (news or [])
            ],
            "known_limitations": ctx.to_dict(),
        }
        prompt = (
            "Evidence (JSON):\n"
            + json.dumps(bundle, indent=2, default=str, ensure_ascii=False)
            + "\n\nReturn the JSON described in the system instructions."
        )
        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: llm.complete_thinking(
                    self._SYNTH_SYSTEM, prompt, json_mode=True
                ),
            )
            d = llm.parse_json(raw)
            ev = d.get("evidence") or []
            dn = d.get("downside") or []
            # rationale: accept list (new format) or fall back to splitting a string
            raw_rat = d.get("rationale") or []
            if isinstance(raw_rat, list):
                rationale_bullets = [str(x).strip() for x in raw_rat if str(x).strip()][:4]
            else:
                # Legacy string: split on ". " to make bullets
                sentences = [s.strip() for s in str(raw_rat).split(". ") if s.strip()]
                rationale_bullets = [s if s.endswith(".") else s + "." for s in sentences][:4]
            return {
                "catalyst": str(d.get("catalyst", "")).strip(),
                "direction": str(d.get("direction", "mixed")).strip().lower(),
                "confidence": str(d.get("confidence", "low")).strip().lower(),
                "rationale": rationale_bullets,
                "evidence": [str(x).strip() for x in ev if str(x).strip()][:6],
                "downside": [str(x).strip() for x in dn if str(x).strip()][:5],
                "caveat": str(d.get("caveat", "")).strip(),
            }
        except Exception as e:
            reason = str(e).split("\n")[0][:120]
            logger.warning(f"Catalyst synthesis failed: {e}")
            ctx.degrade("synthesis",
                        f"Thinking-tier synthesis unavailable ({reason})",
                        "Fell back to a numbers-only heuristic catalyst.",
                        Severity.WARN)
            return None

    def _fallback_catalyst(self, themes, ern_summary, eeg) -> Dict[str, Any]:
        """No thinking tier: derive a coarse direction from EEG revision sign +
        guidance gap. Honest and conservative."""
        near = next(iter(eeg.values()), {})
        rev = near.get("revision_pct")
        gap = (ern_summary or {}).get("guidance_vs_consensus")
        score = 0.0
        if rev is not None:
            score += 1 if rev > 0.01 else (-1 if rev < -0.01 else 0)
        if gap is not None:
            score += 1 if gap > 0 else (-1 if gap < 0 else 0)
        direction = "bullish" if score > 0 else "bearish" if score < 0 else "mixed"
        gthemes = (themes or {}).get("themes") or []
        rthemes = (themes or {}).get("risk_themes") or []
        cat = (gthemes[0] if gthemes else "upcoming earnings")
        return {
            "catalyst": f"{cat} — direction inferred from consensus revision "
                        "and guidance gap (no LLM synthesis available).",
            "direction": direction,
            "confidence": "low",
            "rationale": [
                "[EEG] Thinking-tier LLM unavailable — rationale is a "
                "deterministic read of the consensus EPS revision sign and "
                "the guidance-vs-consensus gap only; no narrative synthesis.",
            ],
            "evidence": [],
            "downside": rthemes[:5] or ["risk themes unavailable"],
            "caveat": "",
        }

    def _compose_evidence(
        self, quote, bear_quote, ern_summary, eeg, news, event_news,
        synth, horizons_prof=None,
    ) -> List[str]:
        ev: List[str] = []

        # ── Upcoming events go FIRST — most actionable, most likely to be
        #    missed by filing-grounded analysis ──────────────────────────────
        for n in (event_news or [])[:4]:
            pub = n.get("published_date") or "n/a"
            title = (n.get("title") or "").strip()
            snippet = (n.get("content") or "").strip()
            # Extract the first informative sentence from the snippet
            sentences = [s.strip() for s in snippet.split(".") if len(s.strip()) > 25]
            detail = sentences[0][:120] + "." if sentences else ""
            ev.append(
                f"[upcoming_events] {title} ({pub})"
                + (f" — {detail}" if detail else "")
            )

        # ── LLM synthesis evidence bullets ───────────────────────────────────
        if synth.get("evidence"):
            ev.extend(f"[LLM] {e}" for e in synth["evidence"])

        # ── Verified DB evidence ──────────────────────────────────────────────
        if ern_summary:
            ev.append(
                f"[DB·ERN] {ern_summary['headline']}; guidance vs consensus "
                f"{ern_summary.get('guidance_vs_consensus')}, "
                f"P/E {ern_summary.get('latest_pe_ratio')}, "
                f"reaction {ern_summary.get('latest_reaction_pct')}"
            )
        for tp, tr in eeg.items():
            if tr["n_observations"]:
                ev.append(
                    f"[DB·EEG] {tp} consensus EPS revised "
                    f"{self._pct(tr['revision_pct'])} over 2y "
                    f"(recent 4w {self._pct(tr['recent_4w_delta_pct'])})"
                )
        if horizons_prof and horizons_prof.anchor:
            ev.append(
                f"[DB·ERN·horizon] Next earnings "
                f"{horizons_prof.anchor.get('event_date')} "
                f"({horizons_prof.anchor.get('description', '')})"
            )
        if horizons_prof:
            for bkt in ("0-6m", "6-24m", "structural"):
                for m in (horizons_prof.buckets.get(bkt) or [])[:2]:
                    if m.get("source") == "ern":
                        continue
                    ev.append(
                        f"[horizon·{bkt}] {m.get('description', '')[:80]}"
                    )
        if quote:
            ev.append(f"[DB·filing] {quote}")
        if bear_quote:
            ev.append(f"[DB·risk] {bear_quote}")

        # ── Theme-grounded news ───────────────────────────────────────────────
        for n in (news or [])[:3]:
            ev.append(f"[news] {n['title']} ({n.get('published_date') or 'n/a'})")

        return ev[:16]

    # ── Query parsing ─────────────────────────────────────────────────────────

    async def _parse_query(self, query: str, ctx: RunContext) -> Dict[str, str]:
        try:
            return await self._parse_with_llm(query)
        except Exception as e:
            reason = str(e).split("\n")[0][:120]
            logger.warning(f"LLM parse unavailable ({e}), keyword fallback")
            ctx.degrade(
                "query_parser",
                f"LLM query parse unavailable ({reason})",
                "Intent, subject and ticker inferred by keyword heuristics; "
                "framing approximate.", Severity.WARN,
            )
            return self._parse_with_keywords(query)

    async def _parse_with_llm(self, query: str) -> Dict[str, str]:
        system = (
            "You are a financial query parser. Return ONLY valid JSON, no "
            "markdown."
        )
        user = (
            "Extract from the query:\n"
            f'  "ticker": one of ["AAPL","MSFT"]\n'
            f'  "intent": one of {INTENTS}\n'
            '  "key_subject": the specific thing asked about\n'
            '  "sentiment_hint": one of ["positive","negative","neutral"]\n\n'
            f"Query: {query}"
        )
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, lambda: llm.complete_light(system, user, json_mode=True)
        )
        parsed = llm.parse_json(raw)
        ticker = str(parsed.get("ticker", "AAPL")).upper()
        if ticker not in _FY_END_MONTH:
            ticker = self._infer_ticker(query)
        intent = str(parsed.get("intent", "")).strip()
        if intent not in INTENTS:
            intent = _classify_intent(query)
        key_subject = str(parsed.get("key_subject", "")).strip() \
            or _INTENT_DEFAULT_SUBJECT.get(intent, "overall performance")
        return {
            "ticker": ticker, "intent": intent, "key_subject": key_subject,
            "sentiment_hint": str(parsed.get("sentiment_hint", "neutral")),
            "parse_mode": "llm",
        }

    def _parse_with_keywords(self, query: str) -> Dict[str, str]:
        q = query.lower()
        ticker = self._infer_ticker(query)
        intent = _classify_intent(query)
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
        if quoted:
            key_subject = (quoted[0][0] or quoted[0][1]).strip()
        else:
            kws = ["iphone 18", "iphone", "ipad", "mac", "vision pro", "airpods",
                   "services", "azure", "copilot", "windows", "office", "xbox",
                   "revenue", "earnings", "eps", "margin", "guidance", "cash flow"]
            key_subject = _INTENT_DEFAULT_SUBJECT.get(intent, "overall performance")
            for kw in kws:
                if kw in q:
                    key_subject = kw
                    break
        return {
            "ticker": ticker, "intent": intent, "key_subject": key_subject,
            "sentiment_hint": "neutral", "parse_mode": "keyword",
        }

    def _infer_ticker(self, query: str) -> str:
        q = query.lower()
        for kw in sorted(_TICKER_KEYWORDS, key=len, reverse=True):
            if kw in q:
                return _TICKER_KEYWORDS[kw]
        return "AAPL"

    # ── Retrieval (Stage 1) ───────────────────────────────────────────────────

    def _section_hints(self, key_subject: str) -> Optional[List[str]]:
        q = key_subject.lower()
        hits = [sec for sec, kws in self._SECTION_HINTS.items()
                if any(k in q for k in kws)]
        return hits or None

    def _merged_section_hints(
        self, key_subject: str, intent: str
    ) -> Optional[List[str]]:
        subj = set(self._section_hints(key_subject) or [])
        subj.update(_INTENT_SECTIONS.get(intent, []))
        return sorted(subj) or None

    async def _search_relevant_chunks(
        self, ticker: str, key_subject: str, intent: str = "general",
        limit: int = 6,
    ) -> List[Dict]:
        from retrieval import retrieve_filing_evidence, RetrievalHints
        return await retrieve_filing_evidence(
            ticker, query=key_subject,
            hints=RetrievalHints(
                section=self._merged_section_hints(key_subject, intent),
                top_k=limit,
            ),
        )

    def _pick_relevant_text(
        self, mda_text: Optional[str], chunks: List[Dict], key_subject: str
    ) -> Optional[str]:
        parts: List[str] = []
        for c in chunks[:3]:
            t = (c.get("context") or "").strip()
            if t:
                parts.append(t[:800])
        if mda_text and len(" ".join(parts)) < 1000:
            kl = key_subject.lower()
            for para in mda_text.split("\n\n"):
                if kl in para.lower() and len(para) > 50:
                    parts.append(para[:800])
                    if len(" ".join(parts)) > 3000:
                        break
            if len(" ".join(parts)) < 500:
                parts.append(mda_text[:2000])
        return "\n\n".join(parts) if parts else mda_text


def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None else round(float(v), 4)
    except (TypeError, ValueError):
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────

_EXIT_WORDS = {"exit", "quit", "q", ":q"}
_EXAMPLES = (
    "Examples:\n"
    "  What could be the catalyst for Apple's growth?\n"
    "  What are the risks to MSFT next quarter?\n"
    "  How is Azure trending into the next earnings?"
)


async def _interactive(agent: UpcomingCatalystAgent) -> None:
    loop = asyncio.get_event_loop()
    print("=" * 66)
    print("  Likely Catalyst Agent — interactive mode")
    print("  Ask about an AAPL or MSFT catalyst. Type 'exit' or Ctrl+C to quit.")
    print("=" * 66)
    print(_EXAMPLES)
    while True:
        try:
            query = await loop.run_in_executor(None, input, "\nQuery> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        query = query.strip()
        if not query:
            continue
        if query.lower() in _EXIT_WORDS:
            print("Bye.")
            return
        try:
            print((await agent.analyze(query)).to_text())
        except Exception as e:
            logger.exception("Query failed")
            print(f"\n[error] Could not analyze that query: {e}")


async def _main() -> None:
    agent = UpcomingCatalystAgent()
    if len(sys.argv) >= 2:
        print((await agent.analyze(" ".join(sys.argv[1:]))).to_text())
        return
    await _interactive(agent)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nBye.")
