import psycopg2

conn = psycopg2.connect('postgresql://neondb_owner:npg_BgdTyxpXW3q4@ep-bitter-boat-aq1v8xns.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require')
cur = conn.cursor()

# All fields in income statement
cur.execute("SELECT field FROM fundamentals.aapl_income_statement")
fields = [r[0] for r in cur.fetchall()]
print("ALL FIELDS:")
for f in fields:
    print(" ", f)

# Check if any tax-related fields exist
print("\nTAX-RELATED:")
tax_fields = [f for f in fields if 'tax' in f.lower() or 'provision' in f.lower()]
for f in tax_fields:
    print(" ", f)

conn.close()
