"""Structured PESTEL news retrieval via Tavily.

Fetches 6 targeted searches — one per PESTEL dimension — and returns a dict
keyed by dimension letter ("P", "E", "S", "T", "En", "L").

⚠ Quota note: Each run() call makes up to 6 Tavily searches (vs 1 in the original
  top_news() function).  Results are cached per (sector, company, today) for 24 hours
  so repeated same-day runs are free.  On Tavily's free tier (~1,000/month), run
  PESTEL news only when the full Layer 2 pipeline is needed (not in backtests unless
  you have a paid Tavily plan).
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_CACHE: dict[tuple, tuple[datetime, dict[str, list[dict]]]] = {}
_CACHE_TTL = timedelta(hours=24)

# Per-dimension search templates.
# {company}, {sector}, {year} are filled at call time.
_QUERIES: dict[str, str] = {
    "P": (
        "{company} {sector} government policy trade tariffs geopolitical political risk "
        "regulatory reform subsidies sanctions {year}"
    ),
    "E": (
        "{company} {sector} economic outlook interest rates inflation GDP consumer spending "
        "credit conditions macro environment {year}"
    ),
    "S": (
        "{company} {sector} consumer trends demographics workforce labor ESG social "
        "brand reputation diversity sustainability {year}"
    ),
    "T": (
        "{company} {sector} technology disruption innovation AI digital transformation "
        "R&D patents automation competitive technology {year}"
    ),
    "En": (
        "{company} {sector} climate change ESG environmental regulation carbon emissions "
        "net-zero sustainability energy transition {year}"
    ),
    "L": (
        "{company} {sector} regulation compliance antitrust litigation legal enforcement "
        "lawsuit regulatory scrutiny {year}"
    ),
}


def _search_one(client, query: str, n: int) -> list[dict]:
    """Execute a single Tavily search and normalise results."""
    try:
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
                "snippet":      r.get("content", "")[:500],
                "published_at": r.get("published_date", ""),
            }
            for r in resp.get("results", [])
        ]
    except Exception as e:
        print(f"  [WARN] Tavily PESTEL search failed: {e}")
        return []


def pestel_news(
    sector: str,
    company: str,
    n_per_dim: int = 3,
) -> dict[str, list[dict]]:
    """
    Fetch PESTEL-structured news for (sector, company).

    Returns:
        {
          "P":  [list of news dicts],
          "E":  [...],
          "S":  [...],
          "T":  [...],
          "En": [...],
          "L":  [...],
        }

    Each news dict: {title, url, snippet, published_at}
    """
    cache_key = (sector, company, date.today().isoformat())
    now = datetime.utcnow()

    if cache_key in _CACHE:
        ts, cached = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            return {dim: results[:n_per_dim] for dim, results in cached.items()}

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return {dim: [] for dim in _QUERIES}

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
    except Exception as e:
        print(f"  [WARN] Tavily client init failed: {e}")
        return {dim: [] for dim in _QUERIES}

    year = date.today().year
    all_results: dict[str, list[dict]] = {}

    for dim, template in _QUERIES.items():
        query = template.format(company=company, sector=sector, year=year)
        all_results[dim] = _search_one(client, query, n_per_dim)

    _CACHE[cache_key] = (now, all_results)
    return all_results
