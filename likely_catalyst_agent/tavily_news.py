"""
tavily_news.py — grounded, multi-source news fetch for the catalyst agent.

Stage-2 of the v2 workflow: given themes the company itself emphasised
(Stage-1), pull recent news *about those themes*, time-boxed to after the
last report date so it is genuinely forward/live (point-in-time discipline).

Source strategy (10 items total):
  Tier 1 — IR / company website (3 items)
    Query targets investor-relations language: press releases, earnings
    calls, product launches, conferences.  Tavily naturally surfaces the
    company's own IR pages and newswire filings for these queries.

  Tier 2 — Trusted financial/market outlets (4 items)
    include_domains restricted to Bloomberg, Yahoo Finance, Reuters, WSJ,
    CNBC, FT, MarketWatch, Barron's, Seeking Alpha.

  Tier 3 — YouTube + open-web fallback (remaining slots up to 10)
    YouTube-domain search first; then any remaining slots filled from
    unrestricted open-web to hit the 10-item cap.

Recency bias:
  Every item's raw Tavily relevance score is multiplied by an exponential
  decay weight with a 21-day half-life:
      recency_weight = exp(-ln2 × days_old / 21)
  Result: a 1-day-old article scores ≈ 97% of face value; a 3-week-old
  article scores 50%; a 3-month-old article scores ≈ 5%.

Budget discipline (1000-credit free trial):
  - ``search_depth="basic"`` → 1 credit per call
  - up to 3 Tavily calls per ticker per run (3 credits)
  - same-day per-segment on-disk cache → repeat runs cost 0 credits
  - never raises: any error / missing key → [] so the agent degrades
    gracefully to filings + ERN + EEG only.

Called via raw httpx (httpx is already a dependency; the Tavily SDK is not)
for precise control over depth/credits.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from logger import get_logger

logger = get_logger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"
_CACHE_DIR = Path(__file__).resolve().parent / ".tavily_cache"
_MAX_DAYS = 365  # Tavily news recency cap

# ── Recency-decay constants ────────────────────────────────────────────────────
_RECENCY_HALF_LIFE_DAYS: int = 21   # 21 days → 50% weight; 3 months → ~5%
_UNKNOWN_DATE_WEIGHT: float = 0.30  # penalty for items with no published_date

# ── Trusted financial outlet domains ──────────────────────────────────────────
_FINANCIAL_DOMAINS: List[str] = [
    "bloomberg.com",
    "finance.yahoo.com",
    "reuters.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "marketwatch.com",
    "barrons.com",
    "seekingalpha.com",
    "thestreet.com",
]

# ── YouTube domains ────────────────────────────────────────────────────────────
_YOUTUBE_DOMAINS: List[str] = ["youtube.com", "youtu.be"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _api_key() -> Optional[str]:
    return os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY") or None


def available() -> bool:
    return bool(_api_key())


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def _cache_key(segment: str, query: str, days: int) -> str:
    """
    Date-scoped cache key per segment so each is cached independently and
    invalidated daily.
    """
    raw = f"{date.today().isoformat()}|{days}|{segment}|{query}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _recency_weight(published_date_str: Optional[str]) -> float:
    """
    Exponential-decay recency weight.

    Examples (half-life = 21 days):
        0 days old → 1.000
        7 days old → 0.794
       21 days old → 0.500
       60 days old → 0.117
      180 days old → 0.008
    """
    if not published_date_str:
        return _UNKNOWN_DATE_WEIGHT
    try:
        # Tavily returns "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS[Z]"
        dt_date = datetime.fromisoformat(published_date_str[:10]).date()
        days_old = max(0, (date.today() - dt_date).days)
        return math.exp(-math.log(2) * days_old / _RECENCY_HALF_LIFE_DAYS)
    except Exception:
        return _UNKNOWN_DATE_WEIGHT


def _compute_weighted_score(score: Optional[float], published_date: Optional[str]) -> float:
    """Combine Tavily relevance score with recency decay."""
    relevance = score if (score is not None) else 0.5
    return round(relevance * _recency_weight(published_date), 4)


def _top_n_by_score(items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Return the top-n items ranked by weighted_score descending."""
    return sorted(items, key=lambda x: x.get("weighted_score", 0.0), reverse=True)[:n]


def _source_tier_from_url(url: Optional[str]) -> str:
    """Infer a human-readable source tier from a result URL."""
    if not url:
        return "Web"
    url_lower = url.lower()
    if any(d in url_lower for d in _YOUTUBE_DOMAINS):
        return "YouTube"
    if any(d in url_lower for d in _FINANCIAL_DOMAINS):
        return "Financial News"
    if any(kw in url_lower for kw in ("investor", "ir.", "/ir/", "investor-relations",
                                       "investor_relations", "press-release", "newsroom")):
        return "IR/Company"
    return "Web"


# ── Per-segment fetch ──────────────────────────────────────────────────────────

def _fetch_segment(
    api_key: str,
    query: str,
    days: int,
    segment: str,
    *,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    max_results: int = 6,
    topic: str = "news",
) -> List[Dict[str, Any]]:
    """
    Single Tavily call for one source tier.  Returns normalised item dicts with
    ``weighted_score`` pre-computed.  Returns [] on any failure (never raises).

    Cache is per-segment so each tier can be reused independently.
    """
    ck = _cache_key(segment, query, days)
    cpath = _cache_path(ck)
    if cpath.exists():
        try:
            cached = json.loads(cpath.read_text(encoding="utf-8"))
            logger.info(f"Tavily cache hit [{segment}] ({len(cached)} items, 0 credits).")
            return cached
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "query": query[:400],
        "search_depth": "basic",   # 1 credit
        "topic": topic,
        "days": days,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    try:
        resp = httpx.post(
            _TAVILY_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Tavily [{segment}] fetch failed ({e}) — segment skipped.")
        return []

    items: List[Dict[str, Any]] = []
    for r in (data.get("results") or [])[:max_results]:
        pub_date = r.get("published_date")
        score = r.get("score")
        items.append({
            "title":          (r.get("title") or "").strip(),
            "url":            r.get("url"),
            "published_date": pub_date,
            "content":        (r.get("content") or "").strip()[:600],
            "score":          score,
            "weighted_score": _compute_weighted_score(score, pub_date),
        })

    # Persist per-segment cache
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        cpath.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.debug(f"Tavily cache write [{segment}] failed: {e}")

    logger.info(f"Tavily [{segment}]: {len(items)} items (1 credit, query={query[:60]!r}).")
    return items


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_news(
    ticker: str,
    themes: List[str],
    since_date: Optional[date] = None,
    *,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Return up to ``max_results`` (default 10) recent news items about ``themes``
    for ``ticker``, published after ``since_date``.

    Items are drawn from three source tiers in priority order:

      • Tier 1 — IR / company website (up to 3 items)
        Searches for investor-relations content: press releases, earnings
        conferences, product launches announced by the company.

      • Tier 2 — Trusted financial outlets (up to 4 items)
        Restricted to Bloomberg, Yahoo Finance, Reuters, WSJ, CNBC, FT,
        MarketWatch, Barron's, Seeking Alpha.

      • Tier 3 — YouTube + open-web fallback (remaining slots)
        YouTube channel videos first; then any unrestricted open-web
        results to fill up to ``max_results``.

    Within each tier, items are ranked by:
        weighted_score = tavily_relevance × exp(-ln2 × days_old / 21)
    so that more recent articles rank higher than older ones with the
    same relevance score.

    Each returned dict contains:
        title, url, published_date, content (≤600 chars), score,
        weighted_score, source_tier

    Returns [] on any API failure or missing key.
    """
    key = _api_key()
    if not key:
        logger.info("Tavily key absent — skipping news leg (degraded).")
        return []

    theme_str = " ".join(t.strip() for t in themes if t and t.strip())[:280]
    base_query = f"{ticker} {theme_str}".strip()
    if not base_query:
        return []

    today = date.today()
    days = _MAX_DAYS
    if since_date:
        days = max(1, min(_MAX_DAYS, (today - since_date).days))

    # ── Tier 1: IR / company website ──────────────────────────────────────────
    # Include leadership and event terms so CEO changes, WWDC announcements,
    # and partnerships in the company's own newsroom get surfaced here.
    ir_query = (
        f"{ticker} investor relations press release earnings conference "
        f"CEO leadership executive product launch partnership announcement "
        f"{theme_str}"
    )
    ir_raw = _fetch_segment(key, ir_query, days, segment="ir", max_results=6)
    tier1 = _top_n_by_score(ir_raw, 3)

    # ── Tier 2: Trusted financial outlets ─────────────────────────────────────
    fin_raw = _fetch_segment(
        key,
        base_query,
        days,
        segment="financial",
        include_domains=_FINANCIAL_DOMAINS,
        max_results=6,
    )
    tier2 = _top_n_by_score(fin_raw, 4)

    # ── Tier 3: YouTube + open-web fallback ───────────────────────────────────
    seen_urls = {item["url"] for item in tier1 + tier2}
    remaining = max_results - len(tier1) - len(tier2)
    tier3: List[Dict[str, Any]] = []

    if remaining > 0:
        # YouTube first
        yt_raw = _fetch_segment(
            key,
            base_query,
            days,
            segment="youtube",
            include_domains=_YOUTUBE_DOMAINS,
            max_results=4,
            topic="general",
        )
        for item in _top_n_by_score(yt_raw, min(remaining, 3)):
            if item["url"] not in seen_urls:
                tier3.append(item)
                seen_urls.add(item["url"])

        # Open-web fallback to reach the cap
        still_left = remaining - len(tier3)
        if still_left > 0:
            fallback_raw = _fetch_segment(
                key, base_query, days, segment="fallback", max_results=still_left + 3
            )
            for item in _top_n_by_score(fallback_raw, still_left + 3):
                if item["url"] not in seen_urls and len(tier3) < remaining:
                    tier3.append(item)
                    seen_urls.add(item["url"])

    # ── Merge, tag with source_tier, and return ────────────────────────────────
    results: List[Dict[str, Any]] = []

    for item in tier1:
        tier_label = _source_tier_from_url(item.get("url"))
        # If Tavily returned a financial outlet in an IR query, label it accurately
        if tier_label == "Web":
            tier_label = "IR/Company"
        results.append({**item, "source_tier": tier_label})

    for item in tier2:
        results.append({**item, "source_tier": "Financial News"})

    for item in tier3:
        tier_label = _source_tier_from_url(item.get("url"))
        results.append({**item, "source_tier": tier_label if tier_label != "Web" else "Open Web"})

    total_credits = sum(
        1 for seg in ("ir", "financial", "youtube", "fallback")
        if any(True for _ in [None])   # count non-cached calls — approximate
    )
    logger.info(
        f"Tavily: {len(results)} news items total for {ticker} "
        f"(≤4 credits, {len(tier1)} IR, {len(tier2)} financial, {len(tier3)} fallback/YT)."
    )
    return results


def fetch_upcoming_events(
    ticker: str,
    *,
    year: Optional[int] = None,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch forward-looking catalyst events for ``ticker`` — completely
    independent of the last 10-Q filing themes.

    This is a dedicated search tier for events the filing can never know
    about: CEO/leadership changes, developer conferences (e.g. WWDC),
    major product launches, AI/chip partnerships, regulatory decisions.
    All of these are announced AFTER the filing and are missed by the
    standard theme-grounded fetch_news() call.

    Two targeted Tavily calls are made (2 credits/day, cached):

      Call A — Corporate events: CEO transitions, conferences, partnerships,
        acquisitions, regulatory rulings.  Query avoids stock-price language
        that attracts evergreen price-prediction articles.

      Call B — Product & technology events: new product launches, developer
        conferences (WWDC, Build, Google I/O), AI feature announcements,
        chip/foundry deals, hardware cycle news.

    Price-prediction and stock-forecast sites are explicitly excluded via
    exclude_domains so results stay event-focused, not price-target focused.

    Results are merged, deduplicated, and ranked by recency-weighted score.
    All items tagged source_tier="Upcoming Events".

    Returns [] on any failure — never raises.
    """
    key = _api_key()
    if not key:
        return []

    yr = year or date.today().year
    next_yr = yr + 1

    # Sites that publish evergreen stock price predictions — these dominate
    # generic "catalyst" queries and crowd out actual event news.
    _PRICE_PREDICTION_SITES: List[str] = [
        "marketbeat.com", "stockanalysis.com", "wisesheets.io",
        "coincodex.com", "cryptopolitan.com", "fxopen.com",
        "walletinvestor.com", "tradingeconomics.com", "macrotrends.net",
        "predictionsforecast.com", "longforecast.com", "stockforecasttoday.com",
        "moneycontrol.com", "investing.com",
    ]

    # ── Call A: Corporate events ───────────────────────────────────────────
    # Targets: CEO/leadership transitions, developer conferences, major
    # partnerships/deals, acquisitions, regulatory decisions.
    corp_query = (
        f"{ticker} {yr} {next_yr} "
        f"CEO leadership executive transition conference "
        f"partnership deal acquisition regulatory ruling announcement"
    )
    corp_raw = _fetch_segment(
        key,
        corp_query,
        days=90,
        segment=f"upcoming_corp_{ticker.lower()}",
        exclude_domains=_PRICE_PREDICTION_SITES,
        max_results=max_results + 2,
        topic="news",
    )

    # ── Call B: Product & technology launches ─────────────────────────────
    # Targets: WWDC, foldable iPhone, AI feature roadmap, new hardware
    # cycle, chip/foundry deals, developer tools announcements.
    product_query = (
        f"{ticker} {yr} {next_yr} "
        f"new product launch foldable developer conference WWDC "
        f"AI feature Siri chip foundry hardware roadmap announcement"
    )
    product_raw = _fetch_segment(
        key,
        product_query,
        days=90,
        segment=f"upcoming_product_{ticker.lower()}",
        exclude_domains=_PRICE_PREDICTION_SITES,
        max_results=max_results + 2,
        topic="news",
    )

    # ── Merge, deduplicate, rank by recency-weighted score ────────────────
    seen_urls: set = set()
    combined: List[Dict[str, Any]] = []
    for item in corp_raw + product_raw:
        url = item.get("url") or ""
        if url not in seen_urls:
            seen_urls.add(url)
            combined.append(item)

    top = _top_n_by_score(combined, max_results)
    result = [{**item, "source_tier": "Upcoming Events"} for item in top]

    logger.info(
        f"Tavily [upcoming_events]: {len(result)} forward-looking items "
        f"for {ticker} (2 credits, {len(corp_raw)} corp + {len(product_raw)} product)."
    )
    return result
