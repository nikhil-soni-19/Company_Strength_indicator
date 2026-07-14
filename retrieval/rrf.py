"""Reciprocal rank fusion — pure function, no DB calls."""
from __future__ import annotations

from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    rank_lists: List[Dict[int, int]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Fuse multiple ranked lists via RRF.

    Args:
        rank_lists: list of {doc_id: rank} dicts — one per ranker (BM25, vector, …)
        k:          smoothing constant (default 60, standard in literature)
    """
    scores: Dict[int, float] = {}
    for ranks in rank_lists:
        for doc_id, rank in ranks.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])
