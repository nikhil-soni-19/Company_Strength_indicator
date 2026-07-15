"""Connection to the csi_ontology_lab database (10-K filings / RAPTOR tree).

Uses DATABASE_URL_ONTOLOGY_LAB — a separate Neon database from agent7's
primary DATABASE_URL (which stores environment_runs and risk_factors).

If DATABASE_URL_ONTOLOGY_LAB is not set, get_ontology_conn() raises
OntologyDBNotConfigured so callers can fall back gracefully.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


class OntologyDBNotConfigured(RuntimeError):
    """Raised when DATABASE_URL_ONTOLOGY_LAB is not set in the environment."""


def get_ontology_conn():
    """Return a psycopg2 connection to csi_ontology_lab.

    Raises OntologyDBNotConfigured if the env var is missing so callers can
    detect the condition and fall back to the legacy local risk_factors table.
    """
    # Check env var FIRST — before importing psycopg2 — so OntologyDBNotConfigured
    # is raised cleanly even if psycopg2 isn't installed in the environment.
    url = os.environ.get("DATABASE_URL_ONTOLOGY_LAB", "").strip()
    if not url:
        raise OntologyDBNotConfigured(
            "DATABASE_URL_ONTOLOGY_LAB is not set. "
            "Add it to your .env to enable ontology-backed 10-K retrieval."
        )
    import psycopg2
    from psycopg2.extras import RealDictCursor
    conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
    # Ensure pgvector's <=> operator (installed in public schema) and
    # the ontology schema are both in the search_path.
    with conn.cursor() as cur:
        cur.execute("SET search_path TO public, ontology")
    conn.commit()
    return conn


def ontology_available() -> bool:
    """Return True if DATABASE_URL_ONTOLOGY_LAB is set and reachable."""
    try:
        conn = get_ontology_conn()
        conn.close()
        return True
    except Exception:
        return False
