"""Lazy, memoized SQLAlchemy engine for the ontology lab DB.

Loads `.env` (API keys, harmless here) and `.env.ingestion` (the lab DSN).
Only `DATABASE_URL_ONTOLOGY_LAB` is consumed — never `DATABASE_URL`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / ".env.ingestion")

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = os.getenv("DATABASE_URL_ONTOLOGY_LAB")
        if not url:
            raise RuntimeError(
                "DATABASE_URL_ONTOLOGY_LAB is not set; check .env.ingestion in the project root"
            )
        _engine = create_engine(url, pool_pre_ping=True, future=True)
    return _engine
