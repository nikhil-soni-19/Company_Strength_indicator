"""Connection to the csi_ontology_lab database (10-K filings)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


class OntologyDBNotConfigured(RuntimeError):
    pass


def get_ontology_conn():
    url = os.environ.get("DATABASE_URL_ONTOLOGY_LAB", "").strip()
    if not url:
        raise OntologyDBNotConfigured(
            "DATABASE_URL_ONTOLOGY_LAB is not set. "
            "Add it to agent6/.env to enable 10-K retrieval."
        )
    import psycopg2
    from psycopg2.extras import RealDictCursor
    conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO public, ontology")
    conn.commit()
    return conn


def ontology_available() -> bool:
    try:
        conn = get_ontology_conn()
        conn.close()
        return True
    except Exception:
        return False
