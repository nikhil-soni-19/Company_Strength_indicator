"""Debug script — inspect csi_ontology_lab to diagnose 10-K retrieval failures.

Run from the agent7/ directory:
    python scripts/debug_ontology.py [TICKER]

Defaults to AAPL if no ticker given.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import psycopg2
from psycopg2.extras import RealDictCursor

ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
url = os.environ.get("DATABASE_URL_ONTOLOGY_LAB", "").strip()
if not url:
    print("ERROR: DATABASE_URL_ONTOLOGY_LAB not set in .env")
    sys.exit(1)

conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
cur = conn.cursor()

SEP = "─" * 64

# ── 1. Schema: what columns exist on ontology.filings? ───────────────────────
print(f"\n{SEP}")
print("1. COLUMNS in ontology.filings")
print(SEP)
cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'ontology' AND table_name = 'filings'
    ORDER BY ordinal_position
""")
for row in cur.fetchall():
    print(f"  {row['column_name']:<35} {row['data_type']}")

# ── 2. Distinct filing_type values ───────────────────────────────────────────
print(f"\n{SEP}")
print("2. DISTINCT filing_type values (up to 20)")
print(SEP)
cur.execute("""
    SELECT DISTINCT filing_type, COUNT(*) AS n
    FROM ontology.filings
    GROUP BY filing_type
    ORDER BY n DESC
    LIMIT 20
""")
for row in cur.fetchall():
    print(f"  {str(row['filing_type']):<20}  count={row['n']}")

# ── 3. Search for the ticker in all ticker-like columns ──────────────────────
print(f"\n{SEP}")
print(f"3. ROWS matching '{ticker}' (any ticker column, case-insensitive)")
print(SEP)
cur.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'ontology'
      AND table_name = 'filings'
      AND (column_name ILIKE '%ticker%'
           OR column_name ILIKE '%symbol%'
           OR column_name ILIKE '%company%')
    ORDER BY ordinal_position
""")
ticker_cols = [r["column_name"] for r in cur.fetchall()]
print(f"  Ticker-like columns found: {ticker_cols}")

for col in ticker_cols:
    cur.execute(f"""
        SELECT filing_id,
               {col},
               filing_type,
               period_end_date,
               fiscal_year
        FROM ontology.filings
        WHERE UPPER({col}::text) = %s
        ORDER BY period_end_date DESC NULLS LAST
        LIMIT 10
    """, (ticker,))
    rows = cur.fetchall()
    print(f"\n  Column '{col}' = '{ticker}'  →  {len(rows)} rows")
    for r in rows:
        print(f"    filing_id={r['filing_id']}  "
              f"filing_type={r.get('filing_type')!r}  "
              f"period_end_date={r.get('period_end_date')}  "
              f"fiscal_year={r.get('fiscal_year')}")

# ── 4. Fuzzy: rows containing ticker as substring ────────────────────────────
print(f"\n{SEP}")
print(f"4. FUZZY search — '{ticker}' as substring across ticker-like columns")
print(SEP)
for col in ticker_cols:
    cur.execute(f"""
        SELECT DISTINCT {col}::text AS val, COUNT(*) AS n
        FROM ontology.filings
        WHERE {col}::text ILIKE %s
        GROUP BY {col}::text
        ORDER BY n DESC
        LIMIT 10
    """, (f"%{ticker}%",))
    rows = cur.fetchall()
    if rows:
        for r in rows:
            print(f"  col={col}  value={r['val']!r}  count={r['n']}")

# ── 5. Sample 10 rows to see raw data ────────────────────────────────────────
print(f"\n{SEP}")
print("5. SAMPLE 10 rows from ontology.filings (any ticker)")
print(SEP)
cur.execute("""
    SELECT filing_id, filing_type, period_end_date, fiscal_year
    FROM ontology.filings
    ORDER BY filing_id DESC
    LIMIT 10
""")
# Also grab one ticker column if any exist
for r in cur.fetchall():
    print(f"  {dict(r)}")

# ── 6. Check narrative_chunks exist for any 10-K ─────────────────────────────
print(f"\n{SEP}")
print("6. narrative_chunks count by filing_type (top 5)")
print(SEP)
cur.execute("""
    SELECT f.filing_type, COUNT(nc.id) AS chunk_count
    FROM ontology.narrative_chunks nc
    JOIN ontology.filings f ON f.filing_id = nc.filing_id
    GROUP BY f.filing_type
    ORDER BY chunk_count DESC
    LIMIT 5
""")
for r in cur.fetchall():
    print(f"  {r['filing_type']:<20}  chunks={r['chunk_count']}")

# ── 7a. Verify vector operator works with a dummy query ──────────────────────
print(f"\n{SEP}")
print("7a. VECTOR operator smoke test (dummy 1024-dim zero vector against AMD 10-K)")
print(SEP)
dummy_vec = "[" + ",".join(["0.0"] * 1024) + "]"
try:
    cur.execute("""
        SELECT nc.id,
               1 - (nc.embedding <=> %(qv)s::vector) AS similarity
        FROM ontology.narrative_chunks nc
        JOIN ontology.filings f ON f.filing_id = nc.filing_id
        WHERE f.ticker = %(ticker)s
          AND LOWER(f.filing_type) = '10-k'
        ORDER BY nc.embedding <=> %(qv)s::vector
        LIMIT 3
    """, {"qv": dummy_vec, "ticker": ticker})
    rows = cur.fetchall()
    if rows:
        for r in rows:
            print(f"  id={r['id']}  similarity={r['similarity']:.4f}")
        print("  OK  vector <=> operator works")
    else:
        print("  (0 rows — filing has no chunks? check filing_id)")
except Exception as e:
    print(f"  ERR ({type(e).__name__}): {e}")
    conn.rollback()

# ── 7b. BM25 smoke test ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("7b. BM25 smoke test (tsv search for 'risk' against AMD 10-K)")
print(SEP)
try:
    cur.execute("""
        SELECT nc.id,
               ts_rank_cd(nc.tsv, plainto_tsquery('english', 'risk')) AS rank
        FROM ontology.narrative_chunks nc
        JOIN ontology.filings f ON f.filing_id = nc.filing_id
        WHERE f.ticker = %(ticker)s
          AND LOWER(f.filing_type) = '10-k'
          AND nc.tsv @@ plainto_tsquery('english', 'risk')
        ORDER BY rank DESC
        LIMIT 3
    """, {"ticker": ticker})
    rows = cur.fetchall()
    if rows:
        for r in rows:
            print(f"  id={r['id']}  rank={r['rank']:.4f}")
        print("  OK  tsv BM25 works")
    else:
        print("  (0 rows — chunks exist but no 'risk' hits?)")
except Exception as e:
    print(f"  ERR ({type(e).__name__}): {e}")
    conn.rollback()

# ── 7. Check narrative_chunks schema (embedding dim + tsv column) ─────────────
print(f"\n{SEP}")
print("7. COLUMNS in ontology.narrative_chunks (first 15)")
print(SEP)
cur.execute("""
    SELECT column_name, data_type, udt_name
    FROM information_schema.columns
    WHERE table_schema = 'ontology' AND table_name = 'narrative_chunks'
    ORDER BY ordinal_position
    LIMIT 15
""")
for row in cur.fetchall():
    print(f"  {row['column_name']:<35} {row['data_type']} ({row['udt_name']})")

# Check embedding dimension
print(f"\n{SEP}")
print("8. EMBEDDING dimension sample (first row)")
print(SEP)
cur.execute("""
    SELECT vector_dims(embedding) AS dims
    FROM ontology.narrative_chunks
    WHERE embedding IS NOT NULL
    LIMIT 1
""")
row = cur.fetchone()
if row:
    print(f"  Embedding dimensions: {row['dims']}")
else:
    print("  (no rows with embedding found)")

conn.close()
print(f"\n{SEP}")
print("Done.")
