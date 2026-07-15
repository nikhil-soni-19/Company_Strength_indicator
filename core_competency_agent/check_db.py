import psycopg2

conn = psycopg2.connect('postgresql://neondb_owner:npg_BgdTyxpXW3q4@ep-bitter-boat-aq1v8xns.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require')
cur = conn.cursor()

# cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
# print('TABLES:', [r[0] for r in cur.fetchall()])

# cur.execute("""
#     SELECT table_name, column_name, data_type 
#     FROM information_schema.columns 
#     WHERE table_schema='public' 
#     ORDER BY table_name, ordinal_position
# """)
# for row in cur.fetchall():
#     print(row)

# cur.execute("""
#     SELECT relname, n_live_tup 
#     FROM pg_stat_user_tables 
#     ORDER BY relname
# """)
# print('ROW COUNTS:', cur.fetchall())

# conn.close()
# cur.execute("SELECT * FROM aapl_precomputed_metrics")
# cols = [d[0] for d in cur.description]
# rows = cur.fetchall()
# print('COLUMNS:', cols)
# for r in rows:
#     print(r)

# conn.close()

# cur.execute("""
#     SELECT schemaname, tablename 
#     FROM pg_tables 
#     WHERE tablename LIKE '%precomputed%' OR tablename LIKE '%income_statement%'
#     ORDER BY schemaname, tablename
# """)
# print('FOUND IN SCHEMAS:')
# for r in cur.fetchall():
#     print(r)

# conn.close()

for t in ['aapl_precomputed_metrics', 'aapl_income_statement', 'aapl_cash_flow', 'aapl_balance_sheet']:
    cur.execute(f"SELECT * FROM fundamentals.{t} LIMIT 3")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f'\n--- {t} ---')
    print('COLUMNS:', cols)
    for r in rows:
        print(r)

conn.close()