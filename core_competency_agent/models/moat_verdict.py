from pydantic import BaseModel
from typing import Optional


class MoatVerdict(BaseModel):
    ticker: str
    period: str                             # latest quarter

    # Final fused score (0-100)
    moat_score: float
    direction: str                          # "strengthening" | "stable" | "eroding"

    # Layer breakdown
    layer1_score: float                     # 0-10
    layer2_score: float                     # 0-10

    # Moat drivers and threats
    key_sources: list[str]
    key_threats: list[str]

    # Boolean flags from Layer 1
    flags: list[str]
    margin_premium_sustained: bool
    roic_elite: bool
    insider_conviction_high: bool

    # Layer 2 narrative assessment
    claimed_moat_sources: list[str]        # from 10-K Item 1
    narrative_vs_numbers: str              # "consistent" | "conflict" | "insufficient_data"
    conflict_description: Optional[str] = None

    # Adversarial analysis
    bull_case: Optional[str] = None
    bear_case: Optional[str] = None

    reasoning: str
    sources_cited: list[str]
    passages_used: int
