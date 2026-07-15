"""Tavily search wrapper with 24-hour in-process cache."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_CACHE: dict[tuple, tuple[datetime, list[dict]]] = {}
_CACHE_TTL = timedelta(hours=24)


def top_news(sector: str, company: str, n: int = 5) -> list[dict]:
    """
    Fetch top regulatory / sector news from Tavily.
    Results are cached by (sector, today_date) for 24 hours.
    Each result: {title, url, snippet, published_at}
    """
    cache_key = (sector, company, date.today().isoformat())
    now = datetime.utcnow()

    if cache_key in _CACHE:
        ts, results = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            return results[:n]

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        query = (
            f"{company} {sector} sector regulatory environment operating conditions "
            f"headwinds tailwinds {date.today().year}"
        )
        resp = client.search(
            query=query,
            max_results=n,
            search_depth="advanced",
            include_answer=False,
        )
        results = []
        for r in resp.get("results", []):
            results.append({
                "title":        r.get("title", ""),
                "url":          r.get("url", ""),
                "snippet":      r.get("content", "")[:500],
                "published_at": r.get("published_date", ""),
            })
        _CACHE[cache_key] = (now, results)
        return results[:n]
    except Exception as e:
        print(f"  [WARN] Tavily search failed: {e}")
        return []
