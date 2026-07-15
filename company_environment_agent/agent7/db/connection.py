import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text

load_dotenv(Path(__file__).parent.parent / ".env")

def _get_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your agent7/.env file."
        )
    return url


def get_conn():
    """Return a raw psycopg2 connection."""
    return psycopg2.connect(_get_url(), cursor_factory=RealDictCursor)


def get_engine():
    """Return a SQLAlchemy engine (for pandas read_sql / to_sql)."""
    return create_engine(_get_url())


def init_schema():
    """Run db/schema.sql against the connected database."""
    schema_path = Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("Schema initialised.")


def pgvector_available() -> bool:
    """Check whether pgvector extension is present."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                )
                return cur.fetchone() is not None
    except Exception:
        return False
