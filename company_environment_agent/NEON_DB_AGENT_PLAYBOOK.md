# Neon DB × Agent Playbook
### Exact patterns for connecting agents to Neon PostgreSQL — distilled from agent7

---

## 1. Install dependencies

```bash
pip install psycopg2-binary python-dotenv sqlalchemy
# If using pgvector:
pip install pgvector
```

---

## 2. .env file

Always keep two separate connection strings if your agent uses more than one database (e.g. an app DB + an ingested-data DB):

```env
# Main app database — stores runs, results, state
DATABASE_URL=postgresql://user:pass@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require

# Separate read DB (e.g. ingested documents, embeddings)
DATABASE_URL_SECONDARY=postgresql://user:pass@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require
```

> Both can point to the same Neon instance — Neon separates data via PostgreSQL schemas (`public`, `ontology`, `app`, etc.), not separate connection strings.

---

## 3. Connection module — the correct pattern

Create `db/connection.py`. The **critical rule**: read the URL at **call time**, not at module import time. If you cache `_URL = os.environ.get(...)` at the top of the module, you get an empty string whenever the module is imported before `load_dotenv` runs — causing a cryptic local-socket connection error.

```python
# db/connection.py
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def _get_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env file."
        )
    return url


def get_conn():
    """Raw psycopg2 connection with RealDictCursor (rows as dicts)."""
    return psycopg2.connect(_get_url(), cursor_factory=RealDictCursor)
```

For a secondary database (e.g. an ontology/embedding store):

```python
# db/ontology_connection.py
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class SecondaryDBNotConfigured(RuntimeError):
    pass


def get_secondary_conn():
    import psycopg2
    from psycopg2.extras import RealDictCursor

    url = os.environ.get("DATABASE_URL_SECONDARY", "").strip()
    if not url:
        raise SecondaryDBNotConfigured(
            "DATABASE_URL_SECONDARY is not set."
        )
    conn = psycopg2.connect(url, cursor_factory=RealDictCursor)

    # Set search_path if you have a custom schema (e.g. 'ontology')
    # Always include 'public' — that's where pgvector's <=> lives
    with conn.cursor() as cur:
        cur.execute("SET search_path TO public, ontology")
    conn.commit()
    return conn
```

---

## 4. RealDictCursor — always use named keys

`RealDictCursor` makes every row a dict. **Never use positional indexing.**

```python
# ❌ WRONG — KeyError(0), str(KeyError(0)) == "0", silent failure
doc_id = row[0]
rank   = row[1]

# ✅ CORRECT
doc_id = row["doc_id"]
rank   = row["rank"]

# ✅ CORRECT — converting to plain dict
result = dict(row)          # RealDictRow is already a dict subclass
```

This is the most common silent bug when porting code that used a plain cursor. The error manifests as `KeyError: 0` but `str(KeyError(0))` prints as `"0"`, making it look like a numeric value error.

---

## 5. pgvector setup

### In your schema

```sql
-- Run once on the database
CREATE EXTENSION IF NOT EXISTS vector;

-- Column in your table
embedding vector(1024)   -- match your model's output dimension
```

### search_path is mandatory

pgvector's `<=>` operator lives in the `public` schema. If your tables are in a custom schema (`ontology`, `app`, etc.), you must include both in `search_path` or the operator won't resolve:

```python
cur.execute("SET search_path TO public, ontology")
conn.commit()
```

### Embedding models → dimensions

| Model | Dimensions |
|---|---|
| `BAAI/bge-large-en-v1.5` | **1024** |
| `text-embedding-ada-002` (OpenAI) | 1536 |
| `all-MiniLM-L6-v2` | 384 |
| `text-embedding-3-small` (OpenAI) | 1536 |

Always match the vector column dimension exactly. Inserting a 1024-dim vector into a `vector(1536)` column raises a dimension mismatch error.

---

## 6. Hybrid search pattern (BM25 + vector)

For retrieval-heavy agents, BM25 full-text + vector similarity fused via Reciprocal Rank Fusion outperforms either alone.

### Schema additions

```sql
-- Generated tsvector column for BM25 (auto-updates)
ALTER TABLE chunks ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX ON chunks USING GIN(tsv);
CREATE INDEX ON chunks USING ivfflat(embedding vector_cosine_ops);
```

### RRF fusion

```python
def reciprocal_rank_fusion(rank_lists: list[dict[int, int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for doc_id, rank in ranks.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Transaction safety

When running BM25 + vector as separate queries in the same connection, a failure in one leg leaves the transaction in an error state. The second leg will then also fail with `InFailedSqlTransaction`. Always rollback after a leg failure:

```python
try:
    cur.execute(bm25_query, params)
    bm25_ranks = {r["doc_id"]: r["rank"] for r in cur.fetchall()}
except Exception as e:
    print(f"BM25 leg failed: {e}")
    conn.rollback()   # ← mandatory — clears the failed transaction state

try:
    cur.execute(vector_query, params)
    vec_ranks = {r["doc_id"]: r["rank"] for r in cur.fetchall()}
except Exception as e:
    print(f"Vector leg failed: {e}")
    conn.rollback()
```

---

## 7. Schema migrations — safe ALTER TABLE

Never drop and recreate tables in production. Use `IF NOT EXISTS` / `IF EXISTS` guards:

```python
def _ensure_column(conn, table: str, column: str, col_type: str) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[WARN] Could not add column {column}: {e}")
```

Call this in your persist function before every insert so new columns are added automatically without a separate migration step.

---

## 8. Filing / document resolver pattern

When your agent retrieves documents and needs a fallback chain (e.g. prefer a 10-K but accept a 10-Q):

```python
_PREFERRED_TYPES = ["10-K", "10-K_A", "10-Q", "earnings_call"]

def resolve_best_filing(ticker: str, as_of_date: date, preferred_types=_PREFERRED_TYPES):
    conn = get_secondary_conn()
    with conn.cursor() as cur:
        for doc_type in preferred_types:
            cur.execute(
                """
                SELECT filing_id, filing_type, period_end_date
                FROM ontology.filings
                WHERE UPPER(ticker) = UPPER(%(t)s)
                  AND UPPER(filing_type) = UPPER(%(dt)s)
                  AND period_end_date <= %(cutoff)s
                ORDER BY period_end_date DESC
                LIMIT 1
                """,
                {"t": ticker, "dt": doc_type, "cutoff": as_of_date},
            )
            row = cur.fetchone()
            if row:
                return dict(row)
    return None
```

For companies with non-calendar fiscal years (e.g. Apple ends in September), use a generous cutoff — `date(fiscal_year + 1, 9, 30)` covers any FY end within the fiscal year.

---

## 9. YAML configuration — watch out for boolean keywords

PyYAML (YAML 1.1) silently converts certain unquoted strings to booleans:

| Unquoted | Parsed as |
|---|---|
| `ON`, `YES`, `TRUE` | `True` |
| `OFF`, `NO`, `FALSE` | `False` |

This bites you when ticker symbols like `ON` (ON Semiconductor) appear in config files. Always quote them:

```yaml
# ❌ breaks
peers: [NVDA, AMD, ON, INTC]

# ✅ correct
peers: [NVDA, AMD, "ON", INTC]
```

Add a defensive filter wherever you read YAML-sourced lists of strings:

```python
clean = [p for p in raw_peers if isinstance(p, str)]
```

---

## 10. Debugging checklist

When a Neon connection fails with `connection to server on socket "/tmp/.s.PGSQL.5432" failed`:

1. **Check that the env var is set**: `print(os.environ.get("DATABASE_URL"))` — if `None` or `""`, the URL is missing.
2. **Check that `load_dotenv` ran before the connection call** — or better, read the URL at call time (see §3).
3. **Check `.env` file for accidental overwrites** — when adding a new var, you may have replaced an existing one.

When `pgvector <=>` raises `operator does not exist`:
- Run `SET search_path TO public, <your_schema>` after connecting (see §5).

When you get `KeyError: 0` or silent empty results from queries:
- You're indexing a `RealDictRow` positionally — use named keys (see §4).

When the second query in a transaction fails with `InFailedSqlTransaction`:
- The first query failed but you didn't `conn.rollback()` — add rollback after each failing leg (see §6).

---

## 11. Quick-reference template

Minimal `db/connection.py` to copy into any new agent:

```python
import os
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def get_conn(env_var: str = "DATABASE_URL"):
    url = os.environ.get(env_var, "").strip()
    if not url:
        raise RuntimeError(f"{env_var} is not set in .env")
    conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
    # Uncomment if using pgvector with a custom schema:
    # with conn.cursor() as cur:
    #     cur.execute("SET search_path TO public, your_schema")
    # conn.commit()
    return conn
```

Usage:

```python
conn = get_conn()                           # uses DATABASE_URL
conn = get_conn("DATABASE_URL_SECONDARY")  # uses a secondary DB
```
