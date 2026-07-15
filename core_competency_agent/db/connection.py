import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine

load_dotenv(Path(__file__).parent.parent / ".env")


def _get_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Add it to agent6/.env")
    return url


def get_conn():
    return psycopg2.connect(_get_url(), cursor_factory=RealDictCursor)


def get_engine():
    return create_engine(_get_url())
