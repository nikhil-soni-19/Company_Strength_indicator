from pydantic import BaseModel
from typing import Optional

class RAGPassage(BaseModel):
    chunk_id: str
    ticker: str
    source_type: str                   # "10-Q", "transcript", "10-K"
    period: str
    speaker: Optional[str] = None      # "CEO", "CFO", "management"
    section: Optional[str] = None      # "mda", "qa_transcript"
    text: str
    similarity_score: float

class GuidanceMatch(BaseModel):
    made_in_period: str
    guided_metric: str
    guided_value: str
    actual_period: str
    actual_value: Optional[float] = None
    outcome: str                       # "beat", "met", "miss", "pending", "withdrawn"
    source_text: str

class RAGOutput(BaseModel):
    ticker: str
    passages: list[RAGPassage]
    guidance_matches: list[GuidanceMatch]
    credibility_track_record: float    # 0-1 score based on guidance delivery
    rag_enabled: bool                  # False when stub, True when Neon connected
