from pydantic import BaseModel
from typing import Optional

class FinalVerdict(BaseModel):
    ticker: str
    period: str
    execution_score: float             # 0-10 (weighted L1 + L2)
    layer1_score: float
    credibility_score: float           # 0-10
    direction: str                     # "improving", "stable", "deteriorating"
    verdict: str                       # "narrative_credible", "narrative_not_credible", "insufficient_data"
    flags: list[str]
    key_contradiction: Optional[str] = None
    counterargument: Optional[str] = None
    reasoning: str                     # LLM's full reasoning paragraph
    rag_passages_used: int             # 0 when stub
    sources_cited: list[str]           # ["Q3-2024 10-Q", "Q2-2024 transcript"]
