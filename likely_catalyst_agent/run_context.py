"""
run_context.py — degradation tracking for catalyst agents.

Why: when Claude parsing fails the agent silently falls back to keyword
parsing (and a growth question becomes an earnings report). The only trace is
a log line the consumer never sees. ``RunContext`` records every fallback as
it happens and renders a "LIMITATIONS & CONFIDENCE" block so degradation is
visible in the output itself, not just the logs.

    ctx = RunContext()
    ctx.degrade("query_parser", "Claude API auth unavailable",
                "intent/subject inferred by keyword heuristics — framing "
                "may be less precise", Severity.WARN)
    if ctx.degraded:
        print(ctx.limitations_block())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class Severity(Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"

    @property
    def icon(self) -> str:
        return {"info": "·", "warn": "⚠", "critical": "✗"}[self.value]


@dataclass
class Degradation:
    component: str   # "query_parser", "neon", "finbert", "financial_facts"...
    reason: str      # what failed
    impact: str      # what it means for the answer
    severity: Severity = Severity.WARN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "reason": self.reason,
            "impact": self.impact,
            "severity": self.severity.value,
        }


class RunContext:
    """Accumulates degradations encountered during one agent run."""

    def __init__(self) -> None:
        self._events: List[Degradation] = []

    def degrade(
        self,
        component: str,
        reason: str,
        impact: str,
        severity: Severity = Severity.WARN,
    ) -> None:
        self._events.append(Degradation(component, reason, impact, severity))

    @property
    def degraded(self) -> bool:
        return bool(self._events)

    @property
    def worst_severity(self) -> Severity:
        if not self._events:
            return Severity.INFO
        order = {Severity.INFO: 0, Severity.WARN: 1, Severity.CRITICAL: 2}
        return max((e.severity for e in self._events), key=lambda s: order[s])

    @property
    def events(self) -> List[Degradation]:
        return list(self._events)

    def limitations_block(self) -> str:
        """Rendered section for the report; '' when nothing degraded."""
        if not self._events:
            return ""
        lines = [
            "  LIMITATIONS & CONFIDENCE",
            "  " + "─" * 62,
        ]
        for e in self._events:
            lines.append(f"  {e.severity.icon} {e.component}: {e.reason}")
            lines.append(f"      → {e.impact}")
        return "\n".join(lines)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._events]
