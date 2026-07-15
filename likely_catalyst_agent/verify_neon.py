"""
Quick connection + data check for the Neon DB.
Run from torch_intern/:   python verify_neon.py

Expected output (if everything is wired correctly):
    ✅ Connected to Neon  addr=...  db=neondb
    Companies : [('AAPL',), ('MSFT',)]
    Filings   : 78
    Facts     : ~5900
    sec_filings rows : ~2544
    AAPL latest period : 2024-09-28
    AAPL MD&A preview  : 'The following discussion ...'
"""

import asyncio
import socket
from urllib.parse import urlparse

from sqlalchemy import text
from neon_connection import (
    get_neon_session,
    verify_neon_connection,
    _load_neon_env,
)
from neon_reader import get_available_periods, get_mda_text, get_eps_facts

# Endpoint hostnames vary by Neon project; we only flag obvious paste corruption.
def diagnose_url() -> bool:
    """
    Pre-flight: print repr(hostname) (no password) and catch mangled env files.
    Returns True if the hostname looks structurally valid.
    """
    raw = _load_neon_env()
    host = urlparse(raw).hostname

    print(f"hostname: {repr(host)}  len={len(host or '')}")

    if not host:
        print("❌ Could not parse hostname from DATABASE_URL_ONTOLOGY_LAB")
        return False

    if "\n" in raw or "\r" in raw:
        print("❌ URL contains a line break — re-copy database_cloud.env from a vault/file attachment")
        return False

    if " " in raw:
        print("❌ URL contains spaces — likely pasted incorrectly")
        return False

    if len(host) < 12 or "." not in (host or ""):
        print(f"⚠️  Hostname looks unusual: {repr(host)} (check DATABASE_URL_ONTOLOGY_LAB)")

    bad = [c for c in host if ord(c) < 33 or ord(c) > 126]
    if bad:
        print(f"❌ Non-printable characters in hostname: {[hex(ord(c)) for c in bad]}")
        return False

    return True


def diagnose_dns(hostname: str) -> bool:
    """Returns True if getaddrinfo resolves the Neon host."""
    try:
        socket.getaddrinfo(hostname, 5432, type=socket.SOCK_STREAM)
        print(f"✅ DNS resolves {hostname}")
        return True
    except socket.gaierror as e:
        print(f"❌ DNS failed for {hostname}: {e}")
        print(
            "   If hostname repr() above looks correct, this is network/DNS — "
            "try disabling VPN, switching network (hotspot), or DNS 1.1.1.1 / 8.8.8.8."
        )
        return False


async def main():
    print("── URL pre-flight (password-free) ──")
    if not diagnose_url():
        return

    raw = _load_neon_env()
    host = urlparse(raw).hostname
    print("── DNS pre-flight ──")
    if host and not diagnose_dns(host):
        return

    # Ground-truth connection check
    ok = await verify_neon_connection()
    if not ok:
        print("❌ Connection failed — hostname/DNS OK; check password or SSL")
        return

    async with get_neon_session() as session:

        rows = await session.execute(
            text("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema IN ('ontology', 'live')
                ORDER BY 1, 2
            """)
        )
        tables = rows.fetchall()
        print(f"\nTables (ontology + live): {tables}")

        # 2. Companies
        rows = await session.execute(
            text("SELECT ticker FROM ontology.companies ORDER BY 1")
        )
        companies = rows.fetchall()
        print(f"\nCompanies (ontology.companies) : {companies}")

        rows = await session.execute(text("SELECT COUNT(*) FROM ontology.filings"))
        print(f"Filings   : {rows.scalar()}")

        rows = await session.execute(text("SELECT COUNT(*) FROM ontology.financial_facts"))
        print(f"Facts     : {rows.scalar()}")

        rows = await session.execute(text("SELECT COUNT(*) FROM ontology.sec_filings"))
        print(f"sec_filings (narrative chunks): {rows.scalar()}")

    # 6. AAPL period spine
    periods = await get_available_periods("AAPL")
    if periods:
        print(f"\nAAPL latest period : {periods[0]['period_end_date']}")
    else:
        print("\nAAPL: no periods found")

    # 7. MD&A text preview
    mda = await get_mda_text("AAPL")
    if mda:
        print(f"AAPL MD&A preview  : '{mda[:120].strip()}...'")
    else:
        print("AAPL MD&A          : not found")

    # 8. EPS facts
    eps = await get_eps_facts("AAPL")
    print(f"AAPL EPS facts     : {eps}")

    print("\n✅ Neon verification complete.")


if __name__ == "__main__":
    asyncio.run(main())
