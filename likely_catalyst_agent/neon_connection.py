"""
Neon (cloud Postgres) connection layer.

Loads credentials from database_cloud.env and creates a separate async engine
for the ontology/spine database.  The local agent DB (connection.py) and the
Neon DB are intentionally kept as TWO separate engines:

  - Local DB  → agent writes predictions, backtest results, price cache
  - Neon DB   → agent READS sec filings, financial facts, embeddings
              → agent WRITES live.catalyst_snapshot outputs

Usage:
    from neon_connection import get_neon_session

    async with get_neon_session() as session:
        result = await session.execute(text("SELECT current_database()"))
        print(result.scalar())   # → "neondb"
"""

import os
import re
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy import text

from logger import get_logger

logger = get_logger(__name__)

# ── Credential loading ──────────────────────────────────────────────────────

def _load_neon_env() -> str:
    """
    Load database_cloud.env, searching both cwd and parent dir so the
    code works whether run from torch_intern/ or a notebooks/ sub-folder.
    Returns the raw DATABASE_URL_ONTOLOGY_LAB value.
    """
    here = Path(__file__).parent
    candidates = [
        here / "database_cloud.env",
        here.parent / "database_cloud.env",
    ]

    env_path = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        raise FileNotFoundError(
            "database_cloud.env not found. Drop it in the project root "
            f"(looked in: {[str(p) for p in candidates]})"
        )

    # Manual parse — avoids a dotenv import requirement at module level.
    # os.environ is only mutated with override=False semantics (won't stomp
    # on values already set by the shell, e.g. in CI).
    raw_url: Optional[str] = None
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "DATABASE_URL_ONTOLOGY_LAB":
                raw_url = value
            # Only set in os.environ if not already present
            if key not in os.environ:
                os.environ[key] = value

    if not raw_url:
        raise ValueError(
            "DATABASE_URL_ONTOLOGY_LAB not found in database_cloud.env"
        )
    return raw_url


def _build_asyncpg_url(raw_url: str) -> str:
    """
    Convert the psycopg2-style URL from the env file to asyncpg format.

    postgresql://user:pass@host/db?sslmode=require
      →
    postgresql+asyncpg://user:pass@host/db
    (SSL is passed as a connect_arg, not a query param)
    """
    url = raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    url = re.sub(r"\?.*$", "", url)          # strip ?sslmode=require etc.
    return url


# ── Engine singleton + circuit-breaker ──────────────────────────────────────

_neon_engine: Optional[AsyncEngine] = None
_neon_session_factory: Optional[async_sessionmaker] = None

# Circuit-breaker: set to False after the first unrecoverable connection error.
# All neon_reader / neon_writer calls check this before attempting a session,
# so a DNS failure on first use doesn't spam the logs on every subsequent call.
_neon_available: bool = True


def is_neon_available() -> bool:
    """Return False if Neon has already failed — callers should skip Neon reads."""
    return _neon_available


def _mark_neon_unavailable(reason: str) -> None:
    global _neon_available
    if _neon_available:
        logger.warning(
            f"Neon marked UNAVAILABLE — agent will fall back to local data. "
            f"Reason: {reason}"
        )
        _neon_available = False


def get_neon_engine() -> AsyncEngine:
    """Return (or lazily create) the Neon async engine."""
    global _neon_engine
    if _neon_engine is None:
        raw_url = _load_neon_env()
        async_url = _build_asyncpg_url(raw_url)

        # Log hostname (no password) so connection errors are diagnosable
        try:
            from urllib.parse import urlparse
            host = urlparse(raw_url).hostname
            logger.info(f"Neon engine target: {host}")
        except Exception:
            pass

        _neon_engine = create_async_engine(
            async_url,
            connect_args={"ssl": True},   # Neon requires TLS
            pool_size=3,
            max_overflow=3,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False,
        )
        logger.info("Neon engine created (ontology DB).")
    return _neon_engine


def get_neon_session_factory() -> async_sessionmaker:
    global _neon_session_factory
    if _neon_session_factory is None:
        _neon_session_factory = async_sessionmaker(
            bind=get_neon_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _neon_session_factory


# Error types that indicate the endpoint is unreachable (DNS / network / SSL).
# On these we trip the circuit-breaker so the agent stops retrying.
_UNRECOVERABLE_ERRORS = (
    "nodename nor servname provided",   # macOS errno 8  — hostname DNS fail
    "Name or service not known",        # Linux DNS fail
    "Connection refused",
    "Network is unreachable",
    "getaddrinfo failed",
    "SSL",
)


@asynccontextmanager
async def get_neon_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager — yields a session to the Neon DB.

    If the connection is unrecoverable (DNS / SSL failure), trips the
    circuit-breaker so the agent falls back to local data on all
    subsequent calls without spamming retry attempts.
    """
    factory = get_neon_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            err_str = str(e)
            if any(tag in err_str for tag in _UNRECOVERABLE_ERRORS):
                _mark_neon_unavailable(err_str[:120])
            else:
                logger.error(f"Neon session rollback: {e}")
            raise
        finally:
            await session.close()


async def verify_neon_connection() -> bool:
    """
    Ground-truth check from the setup instructions:
    Expected: (None or '169.254.254.254', 'neondb')
    """
    try:
        async with get_neon_session() as session:
            row = await session.execute(
                text("SELECT inet_server_addr()::text, current_database()")
            )
            addr, db = row.fetchone()
            if db == "neondb":
                logger.info(f"✅ Connected to Neon  addr={addr}  db={db}")
                return True
            else:
                logger.warning(f"❌ Unexpected DB: addr={addr}  db={db}")
                return False
    except Exception as e:
        logger.error(f"Neon connection check failed: {e}")
        return False


async def close_neon():
    """Dispose the Neon engine on shutdown."""
    global _neon_engine
    if _neon_engine:
        await _neon_engine.dispose()
        _neon_engine = None
        logger.info("Neon connections closed.")
