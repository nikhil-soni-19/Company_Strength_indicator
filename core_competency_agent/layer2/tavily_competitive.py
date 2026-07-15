"""Tavily search for competitive context — 24-hour in-process cache.

Runs 3 targeted queries per company instead of one generic search:
  1. Moat / pricing power angle   — company-specific strengths
  2. Competitor comparison angle  — company vs named peers
  3. Threat / risk angle          — recent headwinds and market share pressure
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_CACHE: dict[tuple, tuple[datetime, list[dict]]] = {}
_TTL = timedelta(hours=24)


def _search(client, query: str, n: int) -> list[dict]:
    """Run one Tavily search and return normalised article dicts."""
    resp = client.search(
        query=query,
        max_results=n,
        search_depth="advanced",
        include_answer=False,
    )
    return [
        {
            "title":        r.get("title", ""),
            "url":          r.get("url", ""),
            "snippet":      r.get("content", "")[:600],
            "published_at": r.get("published_date", ""),
        }
        for r in resp.get("results", [])
    ]


def _dedupe(articles: list[dict]) -> list[dict]:
    """Remove duplicate URLs, preserving order."""
    seen, out = set(), []
    for a in articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            out.append(a)
    return out


def fetch_competitive_context(
    ticker: str,
    company_name: str,
    peers: list[str] | None = None,
    n: int = 9,
) -> list[dict]:
    """
    Fetch competitive context via 3 targeted Tavily searches.
    Returns up to n deduplicated articles, ordered: moat → competitor → threat.

    Args:
        ticker:       e.g. "AAPL"
        company_name: e.g. "Apple Inc."
        peers:        e.g. ["DELL", "HPQ", "MSFT"] — used in competitor query
        n:            max articles to return (default 9, ~3 per angle)
    """
    cache_key = (ticker, date.today().isoformat())
    now = datetime.utcnow()

    if cache_key in _CACHE:
        ts, results = _CACHE[cache_key]
        if now - ts < _TTL:
            return results[:n]

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []

    year = date.today().year
    peer_str = " ".join(peers[:3]) if peers else "competitors"

    queries = [
        # 1. Moat / pricing power — what sustains this company's advantage
        (
            f"{company_name} pricing power competitive moat switching costs "
            f"market share {year}"
        ),
        # 2. Competitor comparison — direct head-to-head dynamics
        (
            f"{company_name} vs {peer_str} competition market share "
            f"revenue growth {year}"
        ),
        # 3. Threat / risk — what could erode the moat
        (
            f"{company_name} {ticker} competitive threat disruption risk "
            f"margin pressure headwinds {year}"
        ),
    ]

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)

        results: list[dict] = []
        per_query = max(3, n // len(queries))

        for i, query in enumerate(queries):
            try:
                hits = _search(client, query, per_query)
                results.extend(hits)
                print(f"  [Tavily] Query {i+1}/3: {len(hits)} results")
            except Exception as e:
                print(f"  [Tavily] Query {i+1} failed: {e}")

        results = _dedupe(results)[:n]
        _CACHE[cache_key] = (now, results)
        return results

    except Exception as e:
        print(f"  [Tavily] Competitive context failed: {e}")
        return []
