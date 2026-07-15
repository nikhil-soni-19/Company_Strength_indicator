"""
horizons.py — L2 milestone extractor for future-dated company events.

One Sonnet (thinking-tier) call per filing accession, cached on disk.
Buckets milestones relative to as_of_date:
  0-6m | 6-24m | structural | unscheduled

Deterministic anchor: ERN forward row (is_reported=false) → next earnings date.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import llm
import neon_reader
from logger import get_logger

logger = get_logger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent / ".milestone_cache"
_BUCKETS = ("0-6m", "6-24m", "structural", "unscheduled")

_EXTRACT_SYSTEM = (
    "You extract future-dated events the company mentions in a filing or "
    "earnings-call transcript.\n"
    "Examples: product launch H2 2026, fab online 2027, Phase 3 readout, "
    "debt repayment due 2028, regulatory approval expected next year.\n"
    "Return ONLY valid JSON (no markdown):\n"
    '{"milestones": [{"description": "short label", "year": 2027 or null, '
    '"quarter_if_known": "Q3"|"H1"|"H2"|null, '
    '"evidence_span": "verbatim phrase from the text"}]}\n'
    "Rules:\n"
    "- Include only events the company itself flags as upcoming/future.\n"
    "- year/quarter_if_known when stated or clearly implied; else null.\n"
    "- evidence_span must be copied from the provided text.\n"
    "- If none found, return {\"milestones\": []}."
)


@dataclass
class HorizonsProfile:
    """Bucketed future milestones + ERN earnings anchor."""
    as_of_date: date
    anchor: Optional[Dict[str, Any]] = None
    buckets: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    all_milestones: List[Dict[str, Any]] = field(default_factory=list)
    filings_scanned: int = 0
    cache_hits: int = 0
    llm_extractions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_date": str(self.as_of_date),
            "anchor": self.anchor,
            "buckets": self.buckets,
            "all_milestones": self.all_milestones,
            "filings_scanned": self.filings_scanned,
            "cache_hits": self.cache_hits,
            "llm_extractions": self.llm_extractions,
        }


def _cache_path(accession: str) -> Path:
    key = hashlib.sha256(accession.encode()).hexdigest()[:32]
    return _CACHE_DIR / f"{key}.json"


def _load_cache(accession: str) -> Optional[List[Dict[str, Any]]]:
    path = _cache_path(accession)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("accession") == accession:
            return data.get("milestones") or []
    except Exception as e:
        logger.debug(f"milestone cache read failed ({accession}): {e}")
    return None


def _save_cache(accession: str, milestones: List[Dict[str, Any]]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "accession": accession,
        "extracted_at": date.today().isoformat(),
        "milestones": milestones,
    }
    _cache_path(accession).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _extract_milestones_llm(
    accession: str,
    filing_meta: Dict[str, Any],
    text: str,
) -> List[Dict[str, Any]]:
    """Sync Sonnet call — run inside executor from async code."""
    if not llm.thinking_available():
        return []
    filed = filing_meta.get("filed_date") or filing_meta.get("period_end_date")
    user = (
        f"Accession/source: {accession}\n"
        f"Filing type: {filing_meta.get('filing_type')}\n"
        f"Filed/period: {filed}\n\n"
        f"Text:\n{text[:12_000]}"
    )
    raw = llm.complete_thinking(_EXTRACT_SYSTEM, user, json_mode=True)
    data = llm.parse_json(raw)
    out: List[Dict[str, Any]] = []
    for m in data.get("milestones") or []:
        if not isinstance(m, dict):
            continue
        desc = str(m.get("description") or "").strip()
        if not desc:
            continue
        year_raw = m.get("year")
        year = int(year_raw) if year_raw is not None else None
        out.append({
            "description": desc,
            "year": year,
            "quarter_if_known": m.get("quarter_if_known"),
            "evidence_span": str(m.get("evidence_span") or "")[:300],
            "accession": accession,
            "source": "filing_llm",
        })
    return out


def _parse_quarter_month(quarter_if_known: Optional[str]) -> int:
    """Map Q1/H1/etc. to a representative month (1-indexed)."""
    if not quarter_if_known:
        return 7
    q = str(quarter_if_known).upper()
    if re.search(r"\bQ1\b|1Q", q):
        return 2
    if re.search(r"\bQ2\b|2Q", q):
        return 5
    if re.search(r"\bQ3\b|3Q", q):
        return 8
    if re.search(r"\bQ4\b|4Q", q):
        return 11
    if "H1" in q or "1H" in q:
        return 3
    if "H2" in q or "2H" in q:
        return 9
    return 7


def _months_until(as_of: date, event: date) -> int:
    return (event.year - as_of.year) * 12 + (event.month - as_of.month)


def _infer_event_date(m: Dict[str, Any], as_of: date) -> Optional[date]:
    explicit = m.get("event_date")
    if explicit:
        try:
            if isinstance(explicit, date):
                return explicit
            return date.fromisoformat(str(explicit)[:10])
        except (TypeError, ValueError):
            pass
    year = m.get("year")
    if year is None:
        return None
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    month = _parse_quarter_month(m.get("quarter_if_known"))
    try:
        return date(y, month, 15)
    except ValueError:
        return date(y, 6, 15)


def bucket_milestone(m: Dict[str, Any], as_of: date) -> str:
    """Assign 0-6m / 6-24m / structural / unscheduled."""
    ed = _infer_event_date(m, as_of)
    if ed is None:
        return "unscheduled"
    months = _months_until(as_of, ed)
    if months < 0:
        return "unscheduled"
    if months <= 6:
        return "0-6m"
    if months <= 24:
        return "6-24m"
    return "structural"


def _dedupe_milestones(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for m in items:
        key = (
            str(m.get("description", "")).lower()[:80],
            m.get("year"),
            m.get("quarter_if_known"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _bucket_all(milestones: List[Dict[str, Any]], as_of: date) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {b: [] for b in _BUCKETS}
    for m in milestones:
        b = bucket_milestone(m, as_of)
        enriched = {**m, "bucket": b}
        if _infer_event_date(m, as_of):
            enriched["inferred_date"] = str(_infer_event_date(m, as_of))
        buckets[b].append(enriched)
    return buckets


def _ern_anchor(next_ern: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not next_ern or not next_ern.get("announcement_date"):
        return None
    ann = next_ern["announcement_date"]
    if hasattr(ann, "date") and callable(getattr(ann, "date", None)):
        try:
            ann = ann.date()  # type: ignore[union-attr]
        except Exception:
            pass
    if not isinstance(ann, date):
        try:
            ann = date.fromisoformat(str(ann)[:10])
        except (TypeError, ValueError):
            return None
    fp = next_ern.get("fiscal_period") or "next earnings"
    return {
        "description": f"Next earnings ({fp})",
        "year": ann.year,
        "quarter_if_known": None,
        "event_date": str(ann),
        "evidence_span": "Bloomberg ERN forward row (is_reported=false)",
        "source": "ern",
        "accession": None,
    }


def format_horizon_summary(profile: HorizonsProfile) -> str:
    """Human-readable horizon line for the report header."""
    parts: List[str] = []
    if profile.anchor:
        ann = profile.anchor.get("event_date", "?")
        fp = profile.anchor.get("description", "next earnings")
        try:
            d = date.fromisoformat(str(ann)[:10])
            days = (d - profile.as_of_date).days
            parts.append(f"next earnings ≈ {ann} (~{days}d) — {fp}")
        except (TypeError, ValueError):
            parts.append(f"next earnings ≈ {ann}")

    for label, title in (
        ("0-6m", "near-term"),
        ("6-24m", "medium-term"),
        ("structural", "structural"),
        ("unscheduled", "unscheduled"),
    ):
        items = profile.buckets.get(label) or []
        if not items:
            continue
        names = "; ".join(
            str(m.get("description", ""))[:60] for m in items[:3]
        )
        if len(items) > 3:
            names += f" (+{len(items) - 3} more)"
        parts.append(f"{title}: {names}")

    return " | ".join(parts) if parts else "next 1–2 quarters (no scheduled dates in data)"


async def build_horizons_profile(
    ticker: str,
    as_of_date: Optional[date] = None,
    next_ern: Optional[Dict[str, Any]] = None,
    *,
    max_filings: int = 2,
) -> HorizonsProfile:
    """
    L2 pass: extract milestones from the newest filings (cached per accession),
    bucket vs as_of_date, prepend ERN earnings anchor.
    """
    as_of = as_of_date or date.today()
    profile = HorizonsProfile(as_of_date=as_of)
    profile.anchor = _ern_anchor(next_ern)

    filings = await neon_reader.get_filings(
        ticker,
        filing_types=("10-Q", "10-K", "8-K"),
        limit=max_filings,
    )
    profile.filings_scanned = len(filings)

    extracted: List[Dict[str, Any]] = []
    loop = asyncio.get_event_loop()

    for fil in filings:
        accession = fil.get("accession") or fil.get("source_pdf")
        if not accession:
            continue
        cached = _load_cache(str(accession))
        if cached is not None:
            profile.cache_hits += 1
            extracted.extend(cached)
            continue

        fid = fil.get("filing_id")
        if fid is None:
            continue
        text = await neon_reader.get_filing_narrative_text(int(fid))
        if not text or len(text.strip()) < 100:
            continue

        try:
            milestones = await loop.run_in_executor(
                None,
                _extract_milestones_llm,
                str(accession),
                fil,
                text,
            )
            profile.llm_extractions += 1
            _save_cache(str(accession), milestones)
            extracted.extend(milestones)
        except Exception as e:
            logger.warning(f"milestone extract failed ({accession}): {e}")

    if profile.anchor:
        extracted.insert(0, profile.anchor)

    profile.all_milestones = _dedupe_milestones(extracted)
    profile.buckets = _bucket_all(profile.all_milestones, as_of)
    return profile
