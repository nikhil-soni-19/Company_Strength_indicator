"""
Inspect ESG table contents fully.
Usage: python check_esg_schema.py
"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import os, psycopg2
from psycopg2.extras import RealDictCursor

conn = psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=RealDictCursor)
conn.autocommit = True
cur = conn.cursor()

# 1. Show ALL rows of aapl_fa_esg (it's the combined table for AAPL)
print("=== company_performance.aapl_fa_esg — ALL ROWS ===")
cur.execute('SELECT * FROM company_performance.aapl_fa_esg')
for r in cur.fetchall():
    print(dict(r))

# 2. Show ALL rows of amd_fa_esge/esgg/esgs to understand the split-table structure
for suffix in ['esge', 'esgg', 'esgs']:
    table = f'amd_fa_{suffix}'
    print(f"\n=== company_performance.{table} — ALL ROWS ===")
    cur.execute(f'SELECT * FROM company_performance."{table}"')
    for r in cur.fetchall():
        print(dict(r))

conn.close()
