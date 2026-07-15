from pydantic import BaseModel
from typing import Optional

class QueryIntent(BaseModel):
    ticker: str
    raw_query: str
    time_scope_quarters: int = 4       # "last year" = 4, "last 2 years" = 8
    slope_window: int = 8              # always 8 for regression
    primary_signals: list[str] = []    # ["operating_leverage", "margins", ...]
    hypothesis: Optional[str] = None   # e.g. "management_credibility"
    rag_keywords: list[str] = []       # seeded from hypothesis in query
    layer2_question: str = ""          # specific question for the LLM
