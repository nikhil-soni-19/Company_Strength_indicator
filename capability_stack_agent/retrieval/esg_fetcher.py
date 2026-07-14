"""
Bloomberg ESG data fetcher for Agent 4 — Capability Stack.

Fetches annual ESG data from the company_performance schema in Neon.
Handles two table layouts:
  - Split: {ticker}_fa_esge + {ticker}_fa_esgg + {ticker}_fa_esgs
  - Combined: {ticker}_fa_esg  (used for AAPL, MSFT)

The Bloomberg tables are wide-format:
  unnamed_0 = metric label
  unnamed_1 = Bloomberg field code
  unnamed_2..unnamed_N = FY values (left = oldest, right = newest)

All numeric series are returned oldest → newest, aligned to `fiscal_years`.
None represents a missing or '—' value for that year.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_DB_URL: str = os.environ.get("DATABASE_URL", "")

# ─── Output type ──────────────────────────────────────────────────────────────

@dataclass
class ESGData:
    ticker: str
    fiscal_years: list[str]        # e.g. ['FY 2021', 'FY 2022', 'FY 2023', 'FY 2024']
    period_ends: list[str]         # actual fiscal year end dates matching fiscal_years
    tables_used: list[str]         # which Bloomberg table(s) were read

    # ── Bloomberg pillar scores (0–10 scale, from BESG model) ────────────────
    environmental_score: list[Optional[float]]  # ENVIRONMENTAL_SCORE
    social_score: list[Optional[float]]         # SOCIAL_SCORE
    disclosure_score: list[Optional[float]]     # ENV_DISCLOSURE_SCORE or SOCIAL_DISCLOSURE_SCORE

    # ── Governance (quantitative, multi-year) ─────────────────────────────────
    pct_independent_directors: list[Optional[float]]  # PCT_INDEPENDENT_DIRECTORS
    say_on_pay_support: list[Optional[float]]          # SAY_PAY_SUPPORT_LEVEL (%)
    pct_women_on_board: list[Optional[float]]          # PCT_WOMEN_ON_BOARD
    ceo_pay_ratio: list[Optional[float]]               # CEO_PAY_RATIO_MEDIAN
    board_average_age: list[Optional[float]]           # BOARD_AVERAGE_AGE

    # ── Social (quantitative, multi-year) ─────────────────────────────────────
    employee_turnover_pct: list[Optional[float]]       # EMPLOYEE_TURNOVER_PCT
    safety_incident_rate: list[Optional[float]]        # TOTAL_RECORDABLE_INCIDENT_RATE
    pct_women_employees: list[Optional[float]]         # PCT_WOMEN_EMPLOYEES
    pct_women_mgmt: list[Optional[float]]              # PCT_WOMEN_SENIOR_MGT

    # ── Environmental (quantitative, multi-year) ──────────────────────────────
    co2_total: list[Optional[float]]                   # CO2_SCOPE1_AND_2 or similar
    energy_consumed: list[Optional[float]]             # ENERGY_CONSUMED


# ─── Parsing helpers ──────────────────────────────────────────────────────────

def _to_float(val) -> Optional[float]:
    """Convert a Bloomberg cell value to float. Returns None for missing/dash."""
    if val is None:
        return None
    s = str(val).strip()
    if s in ("—", "", "None", "N/A", "n/a"):
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def _parse_wide_table(rows: list[dict]) -> dict:
    """
    Parse a Bloomberg ESG wide-format table into a structured dict.

    Returns:
        {
          'fy_labels':   ['FY 2021', 'FY 2022', ...],
          'period_ends': ['09/25/2021', ...],          # may be shorter if row absent
          'metrics':     {bloomberg_code: [val, ...]}  # aligned to fy_labels
        }
    """
    # ── Step 1: find which columns map to fiscal years ─────────────────────────
    # The header row has 'FY XXXX' in unnamed_2..unnamed_N.
    fy_col_indices: list[int] = []
    fy_labels: list[str] = []

    for row in rows:
        val2 = row.get("unnamed_2", "")
        if val2 and str(val2).startswith("FY "):
            for i in range(2, 20):
                col = f"unnamed_{i}"
                if col not in row:
                    break
                v = row.get(col)
                if v and str(v).startswith("FY "):
                    fy_col_indices.append(i)
                    fy_labels.append(str(v))
            break

    if not fy_col_indices:
        return {"fy_labels": [], "period_ends": [], "metrics": {}}

    # ── Step 2: find period-end dates ─────────────────────────────────────────
    period_ends: list[str] = []
    for row in rows:
        v0 = row.get("unnamed_0", "")
        if v0 and "12 Months Ending" in str(v0):
            for i in fy_col_indices:
                v = row.get(f"unnamed_{i}")
                period_ends.append(str(v) if v else "")
            break

    # ── Step 3: extract metrics keyed by Bloomberg code ───────────────────────
    metrics: dict[str, list] = {}
    for row in rows:
        code = row.get("unnamed_1")
        if not code:
            continue
        code = str(code).strip()
        if not code or code == "None":
            continue
        values = [row.get(f"unnamed_{i}") for i in fy_col_indices]
        # If the same code appears twice (e.g. HUMAN_RIGHTS_POLICY duplicated at
        # bottom of AMD esgs), keep whichever has more non-None values.
        if code in metrics:
            existing_count = sum(1 for v in metrics[code] if v is not None)
            new_count = sum(1 for v in values if v is not None)
            if new_count <= existing_count:
                continue
        metrics[code] = values

    return {"fy_labels": fy_labels, "period_ends": period_ends, "metrics": metrics}


# ─── DB fetch ─────────────────────────────────────────────────────────────────

def _read_table(conn, schema: str, table: str) -> Optional[list[dict]]:
    """
    Read all rows from schema.table. Returns None if table does not exist.
    Uses autocommit=True connection — caller must provide such a connection.
    """
    cur = conn.cursor()
    try:
        cur.execute(f'SELECT * FROM {schema}."{table}"')
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception:
        # Table not found or other error — rollback is not needed (autocommit).
        return None


def fetch_esg_data(ticker: str) -> Optional[ESGData]:
    """
    Fetch Bloomberg ESG time-series data for the given ticker.

    Tries split tables ({ticker}_fa_esge/esgg/esgs) first, then falls back
    to the combined table ({ticker}_fa_esg).

    Returns None if no ESG data is found or the DB is not configured.
    Silently swallows connection errors so the rest of the pipeline keeps running.
    """
    if not _DB_URL:
        return None

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(_DB_URL, cursor_factory=RealDictCursor)
        conn.autocommit = True          # prevent transaction abort cascade
    except Exception as e:
        print(f"  [ESGFetcher] DB connection failed: {e}")
        return None

    ticker_l = ticker.lower()
    schema = "company_performance"
    all_metrics: dict[str, list] = {}
    fy_labels: list[str] = []
    period_ends: list[str] = []
    tables_used: list[str] = []

    try:
        # ── Attempt 1: split tables ────────────────────────────────────────────
        for suffix in ("esge", "esgg", "esgs"):
            table_name = f"{ticker_l}_fa_{suffix}"
            rows = _read_table(conn, schema, table_name)
            if rows:
                parsed = _parse_wide_table(rows)
                if parsed["fy_labels"]:
                    if not fy_labels:
                        fy_labels = parsed["fy_labels"]
                        period_ends = parsed["period_ends"]
                    # Merge metrics; split tables share the same year columns
                    all_metrics.update(parsed["metrics"])
                    tables_used.append(table_name)

        # ── Attempt 2: combined table (AAPL, MSFT style) ──────────────────────
        if not tables_used:
            table_name = f"{ticker_l}_fa_esg"
            rows = _read_table(conn, schema, table_name)
            if rows:
                parsed = _parse_wide_table(rows)
                if parsed["fy_labels"]:
                    fy_labels = parsed["fy_labels"]
                    period_ends = parsed["period_ends"]
                    all_metrics = parsed["metrics"]
                    tables_used.append(table_name)

        if not fy_labels:
            print(f"  [ESGFetcher] No ESG tables found for {ticker.upper()}")
            return None

        # ── Build typed time series ────────────────────────────────────────────
        def series(code: str) -> list[Optional[float]]:
            raw = all_metrics.get(code, [None] * len(fy_labels))
            # Pad / trim to match fy_labels length
            raw = list(raw) + [None] * len(fy_labels)
            return [_to_float(v) for v in raw[: len(fy_labels)]]

        esg = ESGData(
            ticker=ticker.upper(),
            fiscal_years=fy_labels,
            period_ends=period_ends,
            tables_used=tables_used,
            # Pillar scores
            environmental_score=series("ENVIRONMENTAL_SCORE"),
            social_score=series("SOCIAL_SCORE"),
            disclosure_score=series("ENV_DISCLOSURE_SCORE") or series("SOCIAL_DISCLOSURE_SCORE"),
            # Governance
            pct_independent_directors=series("PCT_INDEPENDENT_DIRECTORS"),
            say_on_pay_support=series("SAY_PAY_SUPPORT_LEVEL"),
            pct_women_on_board=series("PCT_WOMEN_ON_BOARD"),
            ceo_pay_ratio=series("CEO_PAY_RATIO_MEDIAN"),
            board_average_age=series("BOARD_AVERAGE_AGE"),
            # Social
            employee_turnover_pct=series("EMPLOYEE_TURNOVER_PCT"),
            safety_incident_rate=series("TOTAL_RECORDABLE_INCIDENT_RATE"),
            pct_women_employees=series("PCT_WOMEN_EMPLOYEES"),
            pct_women_mgmt=series("PCT_WOMEN_SENIOR_MGT"),
            # Environmental
            co2_total=series("CO2_SCOPE1_AND_2"),
            energy_consumed=series("ENERGY_CONSUMED"),
        )

        print(
            f"  [ESGFetcher] {ticker.upper()}: {len(fy_labels)} years loaded "
            f"from {tables_used}  |  "
            f"{sum(1 for c in ['ENVIRONMENTAL_SCORE','SOCIAL_SCORE','PCT_INDEPENDENT_DIRECTORS'] if all_metrics.get(c))} "
            f"key metrics populated"
        )
        return esg

    except Exception as e:
        print(f"  [ESGFetcher] Unexpected error for {ticker.upper()}: {e}")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
