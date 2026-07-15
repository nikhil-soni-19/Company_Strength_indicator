"""
Stub ingestion for 10-K risk factor text.

Actual SEC filing text is assumed already in DB (per spec §5.3).
Provides:
  - load_from_filing(ticker, fiscal_year, text_path) — chunk, embed, insert
  - mark_material(ticker, fiscal_year, chunk_ids) — flag material chunks
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.connection import get_conn, pgvector_available

_CHUNK_TOKENS = 500
_CHARS_PER_TOKEN = 4  # rough approximation


def _chunk_text(text: str, chunk_size: int = _CHUNK_TOKENS * _CHARS_PER_TOKEN) -> list[str]:
    words = text.split()
    chunks, current = [], []
    char_count = 0
    for word in words:
        current.append(word)
        char_count += len(word) + 1
        if char_count >= chunk_size:
            chunks.append(" ".join(current))
            current, char_count = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def _embed(texts: list[str]) -> list[list[float] | None]:
    """Embed texts. Tries Anthropic voyage API first, falls back to sentence-transformers."""
    try:
        import anthropic
        import os
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        # Anthropic does not expose a direct embedding API in the standard SDK;
        # use sentence-transformers as primary fallback.
        raise NotImplementedError("Use sentence-transformers")
    except Exception:
        pass

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = model.encode(texts, show_progress_bar=False)
        # Pad/truncate to 1536 dims to match schema
        import numpy as np
        result = []
        for v in vecs:
            arr = v.tolist()
            if len(arr) < 1536:
                arr = arr + [0.0] * (1536 - len(arr))
            else:
                arr = arr[:1536]
            result.append(arr)
        return result
    except Exception as e:
        print(f"  [WARN] Embedding failed: {e}. Storing NULL embeddings.")
        return [None] * len(texts)


def load_from_filing(
    ticker: str,
    fiscal_year: int,
    text_path: str | Path,
    filing_date: date | None = None,
) -> int:
    """
    Chunk text from a 10-K filing, embed, and insert into risk_factors.
    Returns number of chunks inserted.
    """
    text = Path(text_path).read_text(encoding="utf-8")
    chunks = _chunk_text(text)
    embeddings = _embed(chunks)

    conn = get_conn()
    try:
        from psycopg2.extras import execute_values
        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            emb_val = f"[{','.join(str(x) for x in emb)}]" if emb else None
            rows.append((ticker, fiscal_year, filing_date, i, chunk, False, emb_val))

        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO risk_factors
                      (ticker, fiscal_year, filing_date, chunk_id, chunk_text, is_material, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (ticker, fiscal_year, chunk_id) DO UPDATE SET
                        filing_date = EXCLUDED.filing_date,
                        chunk_text  = EXCLUDED.chunk_text,
                        is_material = EXCLUDED.is_material,
                        embedding   = EXCLUDED.embedding
                    """,
                    row,
                )
        conn.commit()
        print(f"  {ticker} FY{fiscal_year}: {len(rows)} chunks upserted")
        return len(rows)
    finally:
        conn.close()


def mark_material(ticker: str, fiscal_year: int, chunk_ids: list[int]) -> None:
    """Flag specific chunks as material (litigation / material risk factor)."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE risk_factors SET is_material = TRUE
                WHERE ticker = %s AND fiscal_year = %s AND chunk_id = ANY(%s)
                """,
                (ticker, fiscal_year, chunk_ids),
            )
        conn.commit()
        print(f"  Marked {len(chunk_ids)} chunks as material for {ticker} FY{fiscal_year}")
    finally:
        conn.close()
