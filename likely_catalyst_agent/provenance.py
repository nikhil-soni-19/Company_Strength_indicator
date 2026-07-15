"""
provenance.py — shared evidence / trust layer for all catalyst agents.

Problem this solves: agents mix raw DB rows, keyword-rule tags, model outputs
and degraded fallbacks into one report with uniform authority, so a reader
cannot tell a verified fact from a heuristic guess. This makes provenance a
first-class value: every datum an agent surfaces is wrapped in an ``Evidence``
record, and a ``Ledger`` auto-renders the trust-summary table + JSON (the kind
of table a human otherwise has to reconstruct by hand).

Dependency-free on purpose so any agent can import it.

    ledger = Ledger()
    ledger.add("EPS beat history", "8/8 quarters beat",
               source=Source.DB_VERIFIED, trust=Trust.HIGH, as_of=date(2026,4,30))
    print(ledger.trust_table())
    payload["provenance"] = ledger.to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Source(Enum):
    """Where a value came from — loosely most→least authoritative."""
    DB_VERIFIED = "db_verified"      # raw DB row that passed data_quality checks
    DB_UNVERIFIED = "db_unverified"  # raw DB row that failed/skipped validation
    MODEL = "model"                  # XGBoost / FinBERT inference
    COMPUTED = "computed"            # arithmetic / softmax over other evidence
    NLP_RULE = "nlp_rule"            # keyword / regex extraction (narrative.py)
    LLM = "llm"                      # LLM-parsed / produced
    FALLBACK = "fallback"            # produced by a degraded code path

    @property
    def label(self) -> str:
        return {
            "db_verified": "DB (verified)",
            "db_unverified": "DB (unverified)",
            "model": "Model output",
            "computed": "Computed",
            "nlp_rule": "Keyword rule",
            "llm": "LLM",
            "fallback": "Fallback",
        }[self.value]


class Trust(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNTRUSTED = "untrusted"

    @property
    def rank(self) -> int:
        return {"high": 3, "medium": 2, "low": 1, "untrusted": 0}[self.value]

    @property
    def marker(self) -> str:
        """Compact inline tag for report bullets."""
        return {
            "high": "[DB✓]",
            "medium": "[~]",
            "low": "[?]",
            "untrusted": "[⚠]",
        }[self.value]

    @classmethod
    def worst(cls, trusts: "List[Trust]") -> "Trust":
        """A chain is only as trustworthy as its weakest link."""
        if not trusts:
            return cls.MEDIUM
        return min(trusts, key=lambda t: t.rank)


def _as_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return datetime.fromisoformat(str(v)[:19]).date()
    except (ValueError, TypeError):
        return None


@dataclass
class Evidence:
    """One claim surfaced by an agent, with its provenance.

    ``excluded=True`` means this evidence did NOT feed the final answer
    (e.g. a quarantined bad row, or a keyword seed superseded by an LLM
    synthesis). Excluded items are still shown for transparency but do not
    drag ``Ledger.overall_trust()`` — confidence reflects only used evidence.
    """
    label: str
    detail: str = ""
    source: Source = Source.COMPUTED
    trust: Trust = Trust.MEDIUM
    as_of: Optional[date] = None
    note: str = ""
    excluded: bool = False

    def tag(self) -> str:
        return self.trust.marker

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "detail": self.detail,
            "source": self.source.value,
            "trust": self.trust.value,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "note": self.note,
            "excluded": self.excluded,
        }


class Ledger:
    """Collects Evidence as an agent runs; renders the trust summary."""

    def __init__(self) -> None:
        self._items: List[Evidence] = []

    def add(
        self,
        label: str,
        detail: str = "",
        *,
        source: Source,
        trust: Trust,
        as_of: Any = None,
        note: str = "",
        excluded: bool = False,
    ) -> Evidence:
        ev = Evidence(
            label, detail, source, trust, _as_date(as_of), note, excluded
        )
        self._items.append(ev)
        return ev

    @property
    def items(self) -> List[Evidence]:
        return list(self._items)

    def overall_trust(self) -> Trust:
        """Trust the consumer should place in the headline decision: the
        weakest link among the evidence that actually fed it. Excluded
        evidence (quarantined / superseded) is not counted — catching and
        discarding a bad row should not penalise a conclusion that never
        used it."""
        used = [e.trust for e in self._items if not e.excluded]
        return Trust.worst(used) if used else Trust.MEDIUM

    def trust_table(self) -> str:
        """Aligned text table — the report's auto-generated trust summary."""
        if not self._items:
            return ""
        rows = []
        for e in self._items:
            asof = e.as_of.isoformat() if e.as_of else "—"
            trust_col = (
                e.trust.value.upper() + " (excluded)"
                if e.excluded else e.trust.value.upper()
            )
            rows.append(
                (e.label, e.source.label, trust_col, asof,
                 e.note or e.detail)
            )
        headers = ("Section", "Source", "Trust", "As of")
        widths = [
            max(len(headers[i]), max(len(r[i]) for r in rows))
            for i in range(4)
        ]
        fmt = "  {:<{w0}}  {:<{w1}}  {:<{w2}}  {:<{w3}}"
        lines = [
            "  TRUST SUMMARY",
            "  " + "─" * 62,
            fmt.format(*headers, w0=widths[0], w1=widths[1],
                       w2=widths[2], w3=widths[3]),
        ]
        for r in rows:
            lines.append(fmt.format(r[0], r[1], r[2], r[3], w0=widths[0],
                                    w1=widths[1], w2=widths[2], w3=widths[3]))
            if r[4]:
                lines.append(f"      ↳ {r[4][:90]}")
        lines.append(
            f"  Evidence trust (weakest used signal): "
            f"{self.overall_trust().value.upper()}"
        )
        return "\n".join(lines)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._items]
