from pydantic import BaseModel
from typing import Optional


class MoatIntent(BaseModel):
    ticker: str
    raw_query: str
    fiscal_year: Optional[int] = None
    rag_keywords: list[str] = [
        "competitive advantage", "moat", "pricing power",
        "switching costs", "network effects", "market share",
        "brand", "barriers to entry",
    ]
