"""
Quick connectivity test for the Neon RAG layer.
Run from the agent10/ directory:
    python test_neon_connection.py

What it checks:
  1. Can we connect to Neon at all?
  2. What schemas and tables exist?
  3. Which schema mode will the retriever use? (acsi or simple)
  4. How many rows are in the relevant chunk table?
  5. Does a sample vector similarity query return results for a test ticker?
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("DATABASE_URL_ONTOLOGY_LAB") or os.getenv("NEON_DATABASE_URL")

if not DSN:
    print("ERROR: DATABASE_URL_ONTOLOGY_LAB not set in .env")
    sys.exit(1)

print("=" * 60)
print("Neon Connection Test")
print("=" * 60)

# ── Step 1: Connect ───────────────────────────────────────────────────────────
print("\n[1] Connecting to Neon...")
try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    conn = psycopg2.connect(DSN)
    register_vector(conn)
    print("    ✓ Connected")
except Exception as e:
    print(f"    ✗ Connection failed: {e}")
    sys.exit(1)

cur = conn.cursor()

# ── Step 2: List schemas ──────────────────────────────────────────────────────
print("\n[2] Schemas in database:")
cur.execute("""
    SELECT schema_name FROM information_schema.schemata
    WHERE schema_name NOT IN ('pg_catalog','information_schema','pg_toast')
    ORDER BY schema_name;
""")
for row in cur.fetchall():
    print(f"    • {row[0]}")

# ── Step 3: List all tables with row counts ───────────────────────────────────
print("\n[3] Tables and row counts:")
cur.execute("""
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema NOT IN ('pg_catalog','information_schema','pg_toast')
    AND table_type = 'BASE TABLE'
    ORDER BY table_schema, table_name;
""")
tables = cur.fetchall()
for schema, table in tables:
    try:
        cur.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
        count = cur.fetchone()[0]
        print(f"    • {schema}.{table}  ({count:,} rows)")
    except Exception as e:
        print(f"    • {schema}.{table}  (error: {e})")
        conn.rollback()

# ── Step 4: Detect schema mode ────────────────────────────────────────────────
print("\n[4] Schema mode detection:")
table_set = {f"{s}.{t}" for s, t in tables}
if any("narrative_chunks" in t for t in table_set):
    mode = "acsi"
    chunk_table = next(t for t in table_set if "narrative_chunks" in t)
elif any("filing_chunks" in t for t in table_set):
    mode = "simple"
    chunk_table = "filing_chunks"
else:
    mode = "unknown"
    chunk_table = None

print(f"    Mode: {mode}")
print(f"    Chunk table: {chunk_table}")

# ── Step 5: Inspect chunk table columns ──────────────────────────────────────
if chunk_table:
    schema_name, table_name = chunk_table.split(".", 1) if "." in chunk_table else ("public", chunk_table)
    print(f"\n[5] Columns in {chunk_table}:")
    cur.execute("""
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position;
    """, [schema_name, table_name])
    for col, dtype, maxlen in cur.fetchall():
        print(f"    • {col}  ({dtype}{'(' + str(maxlen) + ')' if maxlen else ''})")

# ── Step 6: Sample distinct tickers ──────────────────────────────────────────
print("\n[6] Sample tickers in the database:")
try:
    if mode == "acsi":
        cur.execute("""
            SELECT DISTINCT f.canonical_ticker, COUNT(nc.chunk_id) as chunks
            FROM ontology.narrative_chunks nc
            JOIN ontology.filings f ON f.filing_id = nc.filing_id
            GROUP BY f.canonical_ticker
            ORDER BY chunks DESC
            LIMIT 10;
        """)
    else:
        cur.execute("""
            SELECT DISTINCT ticker, COUNT(*) as chunks
            FROM filing_chunks
            GROUP BY ticker
            ORDER BY chunks DESC
            LIMIT 10;
        """)
    rows = cur.fetchall()
    for ticker, count in rows:
        print(f"    • {ticker}  ({count:,} chunks)")
except Exception as e:
    print(f"    Could not list tickers: {e}")
    conn.rollback()

# ── Step 7: Sample similarity query ──────────────────────────────────────────
print("\n[7] Sample vector similarity query (requires OPENAI_API_KEY):")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("sk-..."):
    print("    Skipped — set OPENAI_API_KEY in .env to test this")
else:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        dim = int(os.getenv("NEON_EMBEDDING_DIM", "1024"))
        model = os.getenv("NEON_EMBEDDING_MODEL", "text-embedding-3-small")

        kwargs = {"input": "operating leverage margin expansion", "model": model}
        if dim != 1536:
            kwargs["dimensions"] = dim
        vec = client.embeddings.create(**kwargs).data[0].embedding
        print(f"    Embedded test query ({len(vec)}-dim)")

        # Pick first available ticker
        test_ticker = rows[0][0] if rows else "MSFT"

        if mode == "acsi":
            cur.execute("""
                SELECT nc.chunk_id, f.filing_type, nc.section,
                       SUBSTRING(COALESCE(nc.contextualized, nc.content), 1, 120) AS preview,
                       1 - (nc.embedding <=> %s::vector) AS sim
                FROM ontology.narrative_chunks nc
                JOIN ontology.filings f ON f.filing_id = nc.filing_id
                WHERE f.canonical_ticker = %s
                ORDER BY nc.embedding <=> %s::vector
                LIMIT 3;
            """, [vec, test_ticker, vec])
        else:
            cur.execute("""
                SELECT chunk_id, source_type, section,
                       SUBSTRING(text, 1, 120) AS preview,
                       1 - (embedding <=> %s::vector) AS sim
                FROM filing_chunks
                WHERE ticker = %s
                ORDER BY embedding <=> %s::vector
                LIMIT 3;
            """, [vec, test_ticker, vec])

        results = cur.fetchall()
        if results:
            print(f"    Top 3 results for ticker={test_ticker}:")
            for chunk_id, source, section, preview, sim in results:
                print(f"      sim={sim:.3f}  [{source}/{section}]")
                print(f"      \"{preview.strip()}...\"")
        else:
            print(f"    No results for ticker={test_ticker} — check if data is loaded")
    except Exception as e:
        print(f"    Query failed: {e}")
        conn.rollback()

conn.close()
print("\n" + "=" * 60)
print("Test complete.")
print("=" * 60)
