"""BGE-large-en-v1.5 embedder — produces 1024-dim vectors.

The ontology DB stores embeddings produced by BAAI/bge-large-en-v1.5.
Passing any other dimension will crash with a vector dimension mismatch error.
Model is loaded once and cached for the process lifetime.
"""
from __future__ import annotations

from typing import List

_MODEL = None
_MODEL_NAME = "BAAI/bge-large-en-v1.5"


def _get_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer(_MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Could not load BGE embedder '{_MODEL_NAME}'. "
                f"Ensure sentence-transformers>=3.0 is installed. Error: {e}"
            ) from e
    return _MODEL


def embed_query(text: str) -> List[float]:
    """Embed a single query string → 1024-dim BGE vector (normalised L2)."""
    model = _get_model()
    vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return vec.tolist()


def embed_queries(texts: List[str]) -> List[List[float]]:
    """Batch embed multiple query strings → list of 1024-dim vectors."""
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]
