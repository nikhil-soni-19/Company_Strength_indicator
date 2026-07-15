"""Top-level natural-language agent.

Two public methods:

* :meth:`LiquidityAgent.ask` — single-ticker question. Extracts the ticker
  from the user's text (or accepts an override), scores it, and delegates
  to :func:`src.agent.interpreter.interpret` so the answer is a clean
  natural-language paragraph addressing the question.
* :meth:`LiquidityAgent.compare` — multi-ticker comparison. Scores every
  ticker and delegates to :func:`src.agent.interpreter.interpret_comparison`
  to produce a single ranked paragraph.

If the user's question mentions two or more tickers, ``ask`` automatically
routes through ``compare`` so the UX matches the trader's intent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from src.agent.interpreter import Interpretation, interpret, interpret_comparison
from src.agent.ticker_resolver import (
    TICKER_BLACKLIST as _TICKER_BLACKLIST,
    resolve_tickers,
    resolve_tickers_smart,
)
from src.data_ingestion import LiveDataLoader
from src.output.confidence import ConfidenceReport, evaluate_confidence
from src.scoring.engine import LiquidityScore, score_liquidity


@dataclass
class AgentResponse:
    question: str
    ticker: Optional[str]
    score: Optional[LiquidityScore]
    confidence: Optional[ConfidenceReport]
    interpretation: Optional[Interpretation]
    error: Optional[str] = None

    @property
    def answer(self) -> str:
        if self.interpretation is not None:
            return self.interpretation.paragraph
        return self.error or ""

    @property
    def used_llm(self) -> bool:
        return self.interpretation is not None and self.interpretation.used_llm


@dataclass
class ComparisonResponse:
    question: Optional[str]
    tickers: list[str]
    scored: list[tuple[LiquidityScore, ConfidenceReport]] = field(default_factory=list)
    interpretation: Optional[Interpretation] = None
    error: Optional[str] = None

    @property
    def answer(self) -> str:
        if self.interpretation is not None:
            return self.interpretation.paragraph
        return self.error or ""

    @property
    def used_llm(self) -> bool:
        return self.interpretation is not None and self.interpretation.used_llm


class LiquidityAgent:
    """Composes data ingestion, scoring, and LLM interpretation."""

    def __init__(
        self,
        loader: Optional[LiveDataLoader] = None,
        llm_model: Optional[str] = None,
    ):
        self.loader = loader or LiveDataLoader()
        self.llm_model = llm_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def ask(
        self,
        question: str,
        ticker: Optional[str] = None,
    ) -> AgentResponse | ComparisonResponse:
        """Answer a free-form question.

        If the question (or explicit override) contains two or more tickers,
        the call is automatically routed through :meth:`compare`.
        """
        if ticker:
            tickers = [ticker.upper()]
        else:
            tickers = self._extract_tickers(question)

        if not tickers:
            return AgentResponse(
                question=question,
                ticker=None,
                score=None,
                confidence=None,
                interpretation=None,
                error=(
                    "I could not identify a company or ticker in your question. "
                    "Try naming a company (e.g. 'How liquid is ZScaler?') or "
                    "using its ticker symbol (e.g. 'How liquid is ZS?')."
                ),
            )

        if len(tickers) >= 2:
            return self.compare(tickers, question=question)

        only_ticker = tickers[0]
        data = self.loader.fetch(only_ticker)
        score = score_liquidity(data)
        confidence = evaluate_confidence(data)
        interpretation = interpret(
            score=score,
            confidence=confidence,
            question=question,
            model=self.llm_model,
        )
        return AgentResponse(
            question=question,
            ticker=only_ticker,
            score=score,
            confidence=confidence,
            interpretation=interpretation,
        )

    def compare(
        self,
        tickers: list[str],
        question: Optional[str] = None,
    ) -> ComparisonResponse:
        """Score multiple tickers and produce a comparative interpretation."""
        cleaned = [t.upper() for t in tickers if t]
        if len(cleaned) < 1:
            return ComparisonResponse(
                question=question,
                tickers=[],
                error="compare() requires at least one ticker.",
            )

        scored: list[tuple[LiquidityScore, ConfidenceReport]] = []
        errors: list[str] = []
        for tkr in cleaned:
            try:
                data = self.loader.fetch(tkr)
                score = score_liquidity(data)
                confidence = evaluate_confidence(data)
                scored.append((score, confidence))
            except Exception as exc:  # pragma: no cover - depends on yfinance
                errors.append(f"{tkr}: {exc!s}")

        if not scored:
            return ComparisonResponse(
                question=question,
                tickers=cleaned,
                error="No tickers could be fetched. " + " | ".join(errors),
            )

        interpretation = interpret_comparison(
            scored=scored,
            question=question,
            model=self.llm_model,
        )
        return ComparisonResponse(
            question=question,
            tickers=[s.ticker for s, _ in scored],
            scored=scored,
            interpretation=interpretation,
            error=("Partial failures: " + " | ".join(errors)) if errors else None,
        )

    def _extract_tickers(self, question: str) -> list[str]:
        """Return the unique tickers referenced in ``question``, in order.

        Uses the four-layer smart resolver:
          1. Direct uppercase symbol regex (e.g. ``ZS``, ``AAPL``)
          2. Curated alias dictionary (e.g. "apple" → ``AAPL``)
          3. Fuzzy difflib matching (e.g. "Apples" → ``AAPL``)
          4. LLM fallback — only if layers 1-3 all return nothing (e.g.
             "ZScaler" → ``ZS``).  Each LLM result is confirmed against
             yfinance before being accepted to prevent hallucination.
        """
        return resolve_tickers_smart(
            question,
            api_key=os.getenv("OPENAI_API_KEY"),
            llm_model=self.llm_model,
        )

    def _extract_ticker(self, question: str) -> Optional[str]:
        """Backwards-compatible single-ticker extractor."""
        tickers = self._extract_tickers(question)
        return tickers[0] if tickers else None
