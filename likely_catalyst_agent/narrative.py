"""
Narrative understanding layer using BART/Llama.
Extracts structured catalyst information from MD&A text:
- Bullish catalysts (strong guidance, demand growth, margin improvement)
- Bearish catalysts (declining sales, cost pressures, litigation risks)
- Forward guidance signals
- Uncertainty language
- Catalyst type classification
"""

import re
import json
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch

from settings import settings
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class CatalystExtractionResult:
    """Structured output from catalyst extraction."""
    catalyst_type: str
    bullish_catalysts: List[str]
    bearish_catalysts: List[str]
    risk_signals: List[str]
    forward_guidance: List[str]
    uncertainty_signals: List[str]
    narrative_summary: str
    bullish_strength: float   # 0-1
    bearish_strength: float   # 0-1


# Catalyst taxonomy based on the research paper
CATALYST_TAXONOMY = {
    "bullish": [
        "AI Demand Expansion",
        "Revenue Growth Beat",
        "Margin Improvement",
        "New Product Launch",
        "Market Share Expansion",
        "Cost Reduction Achievement",
        "Strong Forward Guidance",
        "International Expansion",
        "M&A Synergies",
        "Operating Leverage",
    ],
    "bearish": [
        "Revenue Miss",
        "Margin Compression",
        "Demand Slowdown",
        "Competitive Pressure",
        "Cost Escalation",
        "Regulatory Risk",
        "Litigation Exposure",
        "Guidance Cut",
        "Liquidity Concern",
        "Supply Chain Disruption",
    ],
    "neutral": [
        "Mixed Signals",
        "Transition Period",
        "Stable Operations",
        "Restructuring",
    ],
}


class NarrativeAnalyzer:
    """
    Uses BART for summarization and narrative understanding.
    Falls back to rule-based extraction when LLM is unavailable.
    """

    def __init__(self):
        self._summarizer = None
        self._classifier = None
        self.device = self._resolve_device()

    def _resolve_device(self) -> str:
        if settings.llm.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return settings.llm.device

    def load_summarizer(self):
        """Lazy-load BART summarization pipeline."""
        if self._summarizer is None:
            logger.info(f"Loading BART: {settings.llm.bart_model}")
            self._summarizer = pipeline(
                "summarization",
                model=settings.llm.bart_model,
                device=0 if self.device == "cuda" else -1,
                truncation=True,
            )
            logger.info("BART loaded.")

    def summarize(
        self, text: str, max_length: int = 300, min_length: int = 80
    ) -> str:
        """Generate BART summary of MD&A text."""
        try:
            self.load_summarizer()
            # BART has 1024 token limit
            text_truncated = text[:3000]
            result = self._summarizer(
                text_truncated,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
            )
            return result[0]["summary_text"] if result else text[:500]
        except Exception as e:
            logger.warning(f"BART summarization failed: {e}, using truncated text")
            return text[:500]

    def extract_catalysts_rule_based(self, text: str) -> CatalystExtractionResult:
        """
        Rule-based catalyst extraction as reliable fallback.
        Uses keyword matching and pattern recognition.
        """
        text_lower = text.lower()

        # Bullish pattern matching
        bullish_patterns = {
            "AI Demand Expansion": [
                "artificial intelligence", "ai demand", "machine learning",
                "data center", "gpu demand", "ai infrastructure",
            ],
            "Revenue Growth Beat": [
                "revenue exceeded", "beat expectations", "record revenue",
                "above guidance", "revenue growth", "top line growth",
            ],
            "Margin Improvement": [
                "margin expansion", "gross margin increased", "operating leverage",
                "improved profitability", "cost efficiencies",
            ],
            "Strong Forward Guidance": [
                "raised guidance", "increased outlook", "strong pipeline",
                "expect growth", "confident in", "positive momentum",
            ],
            "Market Share Expansion": [
                "gained market share", "new customers", "customer wins",
                "expanded presence", "competitive win",
            ],
        }

        bearish_patterns = {
            "Revenue Miss": [
                "missed expectations", "below consensus", "revenue decline",
                "shortfall", "below guidance", "revenue decreased",
            ],
            "Margin Compression": [
                "margin pressure", "gross margin declined", "pricing pressure",
                "cost inflation", "increased costs", "margin headwinds",
            ],
            "Guidance Cut": [
                "lowered guidance", "reduced outlook", "below prior guidance",
                "revised downward", "cut guidance",
            ],
            "Regulatory Risk": [
                "regulatory", "investigation", "compliance", "sec inquiry",
                "antitrust", "litigation",
            ],
            "Demand Slowdown": [
                "demand weakness", "slower demand", "reduced orders",
                "inventory buildup", "macro headwinds",
            ],
        }

        risk_keywords = [
            "risk", "uncertainty", "challenge", "adverse", "volatility",
            "exposure", "litigation", "regulatory", "competition",
        ]

        guidance_keywords = [
            "expect", "anticipate", "guidance", "outlook", "forecast",
            "target", "project", "plan", "intend",
        ]

        # Extract matched catalysts
        found_bullish = []
        bullish_strength = 0.0
        for catalyst, keywords in bullish_patterns.items():
            if any(kw in text_lower for kw in keywords):
                found_bullish.append(catalyst)
                bullish_strength += 1.0 / len(bullish_patterns)

        found_bearish = []
        bearish_strength = 0.0
        for catalyst, keywords in bearish_patterns.items():
            if any(kw in text_lower for kw in keywords):
                found_bearish.append(catalyst)
                bearish_strength += 1.0 / len(bearish_patterns)

        # Extract risk signals (sentences containing risk language)
        sentences = re.split(r"[.!?]\s+", text)
        risk_signals = [
            s.strip()[:150]
            for s in sentences
            if any(k in s.lower() for k in risk_keywords) and len(s) > 30
        ][:5]

        forward_guidance = [
            s.strip()[:150]
            for s in sentences
            if any(k in s.lower() for k in guidance_keywords) and len(s) > 30
        ][:5]

        uncertainty_signals = [
            s.strip()[:150]
            for s in sentences
            if any(k in s.lower() for k in ["uncertain", "may", "could", "no assurance"])
            and len(s) > 30
        ][:3]

        # Determine primary catalyst type
        if found_bullish and bullish_strength > bearish_strength:
            catalyst_type = found_bullish[0]
        elif found_bearish and bearish_strength > bullish_strength:
            catalyst_type = found_bearish[0]
        else:
            catalyst_type = "Mixed Signals"

        # Generate simple narrative summary
        summary_parts = []
        if found_bullish:
            summary_parts.append(f"Bullish drivers: {', '.join(found_bullish[:3])}.")
        if found_bearish:
            summary_parts.append(f"Risk factors: {', '.join(found_bearish[:3])}.")
        narrative_summary = " ".join(summary_parts) or "No clear directional catalysts identified."

        return CatalystExtractionResult(
            catalyst_type=catalyst_type,
            bullish_catalysts=found_bullish,
            bearish_catalysts=found_bearish,
            risk_signals=risk_signals,
            forward_guidance=forward_guidance,
            uncertainty_signals=uncertainty_signals,
            narrative_summary=narrative_summary,
            bullish_strength=min(bullish_strength, 1.0),
            bearish_strength=min(bearish_strength, 1.0),
        )

    async def analyze_filing(
        self, mda_text: str, use_llm: bool = True
    ) -> CatalystExtractionResult:
        """
        Main entry point for filing analysis.
        Combines BART summarization with rule-based catalyst extraction.
        """
        if not mda_text:
            return CatalystExtractionResult(
                catalyst_type="No Data",
                bullish_catalysts=[],
                bearish_catalysts=[],
                risk_signals=[],
                forward_guidance=[],
                uncertainty_signals=[],
                narrative_summary="",
                bullish_strength=0.0,
                bearish_strength=0.0,
            )

        loop = asyncio.get_event_loop()

        # Run rule-based extraction (fast and reliable)
        catalyst_result = await loop.run_in_executor(
            None, self.extract_catalysts_rule_based, mda_text
        )

        # Add BART summary if available
        if use_llm:
            try:
                summary = await loop.run_in_executor(None, self.summarize, mda_text)
                catalyst_result.narrative_summary = summary
            except Exception as e:
                logger.warning(f"BART summarization unavailable: {e}")

        return catalyst_result

    def build_feature_dict(self, result: CatalystExtractionResult) -> Dict[str, float]:
        """Convert catalyst extraction result to numerical features."""
        return {
            "bullish_catalyst_count": float(len(result.bullish_catalysts)),
            "bearish_catalyst_count": float(len(result.bearish_catalysts)),
            "risk_signal_count": float(len(result.risk_signals)),
            "forward_guidance_count": float(len(result.forward_guidance)),
            "uncertainty_count": float(len(result.uncertainty_signals)),
            "bullish_strength": result.bullish_strength,
            "bearish_strength": result.bearish_strength,
            "net_catalyst_score": result.bullish_strength - result.bearish_strength,
        }


# Singleton
_analyzer: Optional[NarrativeAnalyzer] = None


def get_narrative_analyzer() -> NarrativeAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = NarrativeAnalyzer()
    return _analyzer