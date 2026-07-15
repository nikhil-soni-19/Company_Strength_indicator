"""Ontology entities: companies + filings.

P0 schema layer for multi-company ingestion.

Tables:
    ontology.companies   ticker (PK), legal_name, cik, sector, industry,
                         fiscal_year_end, hq_country, created_at
    ontology.filings     filing_id (BIGSERIAL PK), ticker (FK->companies),
                         filing_type, period_end_date, fiscal_year,
                         fiscal_quarter, filed_date, source_pdf, parsed_at
                         UNIQUE (ticker, filing_type, period_end_date)

Public API:
    init_ontology_schema(engine)        — idempotent DDL, safe to call repeatedly
    upsert_company(engine, ticker, **)  — UPSERT with COALESCE (only non-null
                                          arguments overwrite existing values)
    upsert_filing(engine, ticker, type, period_end_date, **) -> filing_id
    delete_filing_data(engine, filing_id) — removes child rows in every
                                            ontology.* table that has a
                                            filing_id column (auto-discovered
                                            via information_schema). Does NOT
                                            delete the filings row itself.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine


SCHEMA: str = "ontology"

# Column lists used by upsert_company. Keep in sync with the DDL below.
_COMPANY_OVERWRITE_COLS: tuple[str, ...] = (
    "legal_name",
    "cik",
    "sector",
    "industry",
    "fiscal_year_end",
    "hq_country",
)

_FILING_OVERWRITE_COLS: tuple[str, ...] = (
    "fiscal_year",
    "fiscal_quarter",
    "filed_date",
    "source_pdf",
)


def init_ontology_schema(engine: Engine) -> None:
    """Create companies + filings tables. Idempotent (IF NOT EXISTS everywhere)."""
    with engine.begin() as conn:
        conn.execute(sa_text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        conn.execute(
            sa_text(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.companies (
                    ticker            TEXT PRIMARY KEY,
                    legal_name        TEXT,
                    cik               TEXT,
                    sector            TEXT,
                    industry          TEXT,
                    fiscal_year_end   TEXT,
                    hq_country        TEXT,
                    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            sa_text(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.filings (
                    filing_id         BIGSERIAL PRIMARY KEY,
                    ticker            TEXT NOT NULL REFERENCES {SCHEMA}.companies(ticker),
                    filing_type       TEXT NOT NULL,
                    period_end_date   DATE,
                    fiscal_year       INTEGER,
                    fiscal_quarter    INTEGER,
                    filed_date        DATE,
                    source_pdf        TEXT,
                    parsed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (ticker, filing_type, period_end_date)
                )
                """
            )
        )
        conn.execute(
            sa_text(
                f'CREATE INDEX IF NOT EXISTS filings_ticker_idx '
                f'ON {SCHEMA}.filings (ticker)'
            )
        )
        conn.execute(
            sa_text(
                f'CREATE INDEX IF NOT EXISTS filings_type_idx '
                f'ON {SCHEMA}.filings (filing_type)'
            )
        )

        # --- financial_facts: long/tidy fact table replacing the five wide tables ---
        conn.execute(
            sa_text(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.financial_facts (
                    fact_id           BIGSERIAL PRIMARY KEY,
                    filing_id         BIGINT NOT NULL
                                      REFERENCES {SCHEMA}.filings(filing_id) ON DELETE CASCADE,
                    statement_type    TEXT NOT NULL,
                    line_item         TEXT NOT NULL,
                    concept_id        BIGINT,
                    period_end_date   DATE,
                    period_type       TEXT,
                    dim_segment       TEXT,
                    dim_geography     TEXT,
                    dim_product       TEXT,
                    value_numeric     NUMERIC,
                    value_text        TEXT,
                    unit              TEXT,
                    currency          TEXT DEFAULT 'USD',
                    source            TEXT NOT NULL DEFAULT 'pdf_table',
                    raw_period_header TEXT
                )
                """
            )
        )
        # Unique index with COALESCE to handle NULLs in dimension columns.
        conn.execute(
            sa_text(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS financial_facts_uq_idx
                ON {SCHEMA}.financial_facts (
                    filing_id, statement_type, line_item,
                    COALESCE(period_end_date, '1900-01-01'),
                    COALESCE(dim_segment,''), COALESCE(dim_geography,''),
                    COALESCE(dim_product,'')
                )
                """
            )
        )
        conn.execute(
            sa_text(
                f'CREATE INDEX IF NOT EXISTS financial_facts_filing_idx '
                f'ON {SCHEMA}.financial_facts (filing_id)'
            )
        )
        conn.execute(
            sa_text(
                f'CREATE INDEX IF NOT EXISTS financial_facts_stmt_period_idx '
                f'ON {SCHEMA}.financial_facts (statement_type, period_end_date)'
            )
        )
        has_trgm = conn.execute(
            sa_text("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
        ).scalar()
        if has_trgm:
            conn.execute(
                sa_text(
                    f'CREATE INDEX IF NOT EXISTS financial_facts_line_item_trgm_idx '
                    f'ON {SCHEMA}.financial_facts USING gin (line_item gin_trgm_ops)'
                )
            )
        else:
            conn.execute(
                sa_text(
                    f'CREATE INDEX IF NOT EXISTS financial_facts_line_item_idx '
                    f'ON {SCHEMA}.financial_facts (lower(line_item))'
                )
            )

        # --- market data (Bloomberg EEG/ERN; no filing_id) ---
        conn.execute(
            sa_text(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.estimate_consensus (
                    estimate_id       BIGSERIAL PRIMARY KEY,
                    ticker            TEXT NOT NULL
                                      REFERENCES {SCHEMA}.companies(ticker),
                    metric            TEXT NOT NULL DEFAULT 'EPS',
                    target_period     TEXT NOT NULL,
                    target_period_type TEXT NOT NULL,
                    as_of_date        DATE NOT NULL,
                    value_mean        NUMERIC,
                    value_high        NUMERIC,
                    value_low         NUMERIC,
                    value_median      NUMERIC,
                    n_estimates       INTEGER,
                    value_stdev       NUMERIC,
                    currency          TEXT DEFAULT 'USD',
                    source            TEXT NOT NULL DEFAULT 'BLOOMBERG_BQL',
                    source_field      TEXT,
                    pull_params       JSONB,
                    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (ticker, metric, target_period, as_of_date, source)
                )
                """
            )
        )
        conn.execute(
            sa_text(
                f'CREATE INDEX IF NOT EXISTS estimate_consensus_lookup_idx '
                f'ON {SCHEMA}.estimate_consensus '
                f'(ticker, metric, target_period, as_of_date)'
            )
        )
        conn.execute(
            sa_text(
                f'CREATE INDEX IF NOT EXISTS estimate_consensus_target_idx '
                f'ON {SCHEMA}.estimate_consensus (target_period)'
            )
        )
        conn.execute(
            sa_text(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.earnings_surprise (
                    surprise_id           BIGSERIAL PRIMARY KEY,
                    ticker                TEXT NOT NULL
                                          REFERENCES {SCHEMA}.companies(ticker),
                    fiscal_period         TEXT NOT NULL,
                    fiscal_year           INTEGER,
                    fiscal_quarter        INTEGER,
                    period_end_date       DATE,
                    announcement_date     DATE NOT NULL,
                    reported_eps          NUMERIC,
                    comparable_eps        NUMERIC,
                    estimate_eps          NUMERIC,
                    surprise_pct          NUMERIC,
                    guidance_eps          NUMERIC,
                    guidance_surprise_pct NUMERIC,
                    price_change_pct      NUMERIC,
                    eps_ttm               NUMERIC,
                    pe_ratio              NUMERIC,
                    is_reported           BOOLEAN NOT NULL,
                    source                TEXT DEFAULT 'BLOOMBERG_ERN',
                    pull_params           JSONB,
                    ingested_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (ticker, fiscal_period, announcement_date)
                )
                """
            )
        )
        conn.execute(
            sa_text(
                f'CREATE INDEX IF NOT EXISTS earnings_surprise_ticker_ann_idx '
                f'ON {SCHEMA}.earnings_surprise (ticker, announcement_date)'
            )
        )
        conn.execute(
            sa_text(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.price_daily (
                    ticker            TEXT NOT NULL
                                      REFERENCES {SCHEMA}.companies(ticker),
                    price_date        DATE NOT NULL,
                    close_px          NUMERIC NOT NULL,
                    currency          TEXT DEFAULT 'USD',
                    source            TEXT DEFAULT 'BLOOMBERG_BDH',
                    source_field      TEXT DEFAULT 'PR005',
                    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (ticker, price_date, source)
                )
                """
            )
        )

        # Drop legacy wide tables only if they are base tables (not views).
        _LEGACY = ("income_statement", "balance_sheet", "cash_flow",
                    "segment_revenue", "product_revenue")
        for t in _LEGACY:
            conn.execute(sa_text(f"""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = '{SCHEMA}'
                          AND c.relname = '{t}'
                          AND c.relkind = 'r'
                    ) THEN
                        EXECUTE 'DROP TABLE {SCHEMA}."{t}" CASCADE';
                    END IF;
                END $$;
            """))
        for t in _LEGACY:
            conn.execute(
                sa_text(
                    f'CREATE OR REPLACE VIEW {SCHEMA}."{t}" AS '
                    f"SELECT * FROM {SCHEMA}.financial_facts "
                    f"WHERE statement_type = '{t}'"
                )
            )


def upsert_company(
    engine: Engine,
    ticker: str,
    *,
    legal_name: str | None = None,
    cik: str | None = None,
    sector: str | None = None,
    industry: str | None = None,
    fiscal_year_end: str | None = None,
    hq_country: str | None = None,
) -> str:
    """INSERT or UPDATE a company row.

    Non-null arguments overwrite existing values; NULL arguments preserve
    whatever is currently in the row (COALESCE semantics). Returns the ticker.
    """
    set_clause = ", ".join(
        f"{col} = COALESCE(EXCLUDED.{col}, {SCHEMA}.companies.{col})"
        for col in _COMPANY_OVERWRITE_COLS
    )
    sql = sa_text(
        f"""
        INSERT INTO {SCHEMA}.companies
            (ticker, legal_name, cik, sector, industry, fiscal_year_end, hq_country)
        VALUES
            (:ticker, :legal_name, :cik, :sector, :industry, :fiscal_year_end, :hq_country)
        ON CONFLICT (ticker) DO UPDATE SET
            {set_clause}
        """
    )
    params: dict[str, Any] = {
        "ticker": ticker,
        "legal_name": legal_name,
        "cik": cik,
        "sector": sector,
        "industry": industry,
        "fiscal_year_end": fiscal_year_end,
        "hq_country": hq_country,
    }
    with engine.begin() as conn:
        conn.execute(sql, params)
    return ticker


def upsert_filing(
    engine: Engine,
    ticker: str,
    filing_type: str,
    period_end_date: date | None,
    *,
    fiscal_year: int | None = None,
    fiscal_quarter: int | None = None,
    filed_date: date | None = None,
    source_pdf: str | None = None,
) -> int:
    """INSERT or UPDATE a filings row, return its filing_id.

    Idempotent: same (ticker, filing_type, period_end_date) always resolves to
    the same filing_id. Non-null arguments overwrite; NULL preserves.
    ``parsed_at`` is always refreshed on update.
    """
    set_clause = ", ".join(
        f"{col} = COALESCE(EXCLUDED.{col}, {SCHEMA}.filings.{col})"
        for col in _FILING_OVERWRITE_COLS
    )
    sql = sa_text(
        f"""
        INSERT INTO {SCHEMA}.filings
            (ticker, filing_type, period_end_date,
             fiscal_year, fiscal_quarter, filed_date, source_pdf)
        VALUES
            (:ticker, :filing_type, :period_end_date,
             :fiscal_year, :fiscal_quarter, :filed_date, :source_pdf)
        ON CONFLICT (ticker, filing_type, period_end_date) DO UPDATE SET
            {set_clause},
            parsed_at = now()
        RETURNING filing_id
        """
    )
    params: dict[str, Any] = {
        "ticker": ticker,
        "filing_type": filing_type,
        "period_end_date": period_end_date,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "filed_date": filed_date,
        "source_pdf": source_pdf,
    }
    with engine.begin() as conn:
        row = conn.execute(sql, params).fetchone()
    if row is None:
        raise RuntimeError(
            "upsert_filing returned no filing_id; check unique key collision logic"
        )
    return int(row[0])


_SURPRISE_UPSERT_COLS: tuple[str, ...] = (
    "fiscal_year",
    "fiscal_quarter",
    "period_end_date",
    "reported_eps",
    "comparable_eps",
    "estimate_eps",
    "surprise_pct",
    "guidance_eps",
    "guidance_surprise_pct",
    "price_change_pct",
    "eps_ttm",
    "pe_ratio",
    "pull_params",
)


def upsert_estimate_consensus(engine: Engine, rows: list[dict[str, Any]]) -> int:
    """Bulk INSERT point-in-time consensus rows; duplicates are skipped."""
    if not rows:
        return 0
    sql = sa_text(
        f"""
        INSERT INTO {SCHEMA}.estimate_consensus (
            ticker, metric, target_period, target_period_type, as_of_date,
            value_mean, value_high, value_low, value_median,
            n_estimates, value_stdev, currency, source, source_field, pull_params
        ) VALUES (
            :ticker, :metric, :target_period, :target_period_type, :as_of_date,
            :value_mean, :value_high, :value_low, :value_median,
            :n_estimates, :value_stdev, :currency, :source, :source_field,
            CAST(:pull_params AS jsonb)
        )
        ON CONFLICT (ticker, metric, target_period, as_of_date, source) DO NOTHING
        """
    )
    params = [_json_row(r) for r in rows]
    with engine.begin() as conn:
        result = conn.execute(sql, params)
    return result.rowcount or 0


def upsert_price_daily(engine: Engine, rows: list[dict[str, Any]]) -> int:
    """Bulk INSERT daily prices; duplicates are skipped."""
    if not rows:
        return 0
    sql = sa_text(
        f"""
        INSERT INTO {SCHEMA}.price_daily (
            ticker, price_date, close_px, currency, source, source_field
        ) VALUES (
            :ticker, :price_date, :close_px, :currency, :source, :source_field
        )
        ON CONFLICT (ticker, price_date, source) DO NOTHING
        """
    )
    with engine.begin() as conn:
        result = conn.execute(sql, [_json_row(r) for r in rows])
    return result.rowcount or 0


def upsert_earnings_surprise(engine: Engine, rows: list[dict[str, Any]]) -> int:
    """INSERT or UPDATE earnings surprise rows (forward quarters may later report)."""
    if not rows:
        return 0
    coalesce_set = ", ".join(
        f"{col} = COALESCE(EXCLUDED.{col}, {SCHEMA}.earnings_surprise.{col})"
        for col in _SURPRISE_UPSERT_COLS
    )
    sql = sa_text(
        f"""
        INSERT INTO {SCHEMA}.earnings_surprise (
            ticker, fiscal_period, fiscal_year, fiscal_quarter, period_end_date,
            announcement_date, reported_eps, comparable_eps, estimate_eps,
            surprise_pct, guidance_eps, guidance_surprise_pct, price_change_pct,
            eps_ttm, pe_ratio, is_reported, source, pull_params
        ) VALUES (
            :ticker, :fiscal_period, :fiscal_year, :fiscal_quarter, :period_end_date,
            :announcement_date, :reported_eps, :comparable_eps, :estimate_eps,
            :surprise_pct, :guidance_eps, :guidance_surprise_pct, :price_change_pct,
            :eps_ttm, :pe_ratio, :is_reported, :source,
            CAST(:pull_params AS jsonb)
        )
        ON CONFLICT (ticker, fiscal_period, announcement_date) DO UPDATE SET
            {coalesce_set},
            is_reported = EXCLUDED.is_reported
        """
    )
    with engine.begin() as conn:
        result = conn.execute(sql, [_json_row(r) for r in rows])
    return result.rowcount or 0


def _json_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    pp = out.get("pull_params")
    if pp is not None and not isinstance(pp, str):
        out["pull_params"] = json.dumps(pp)
    return out


def _tables_with_filing_id(engine: Engine) -> list[str]:
    """Auto-discover every ontology.* BASE TABLE that has a filing_id column.

    Excludes ``filings`` itself (whose filing_id is the PK, not an FK) and
    views (DELETE FROM a view would fail).
    """
    with engine.connect() as conn:
        rows = conn.execute(
            sa_text(
                """
                SELECT c.table_name
                FROM information_schema.columns c
                JOIN information_schema.tables t
                  ON t.table_schema = c.table_schema
                 AND t.table_name  = c.table_name
                WHERE c.table_schema = :s
                  AND c.column_name  = 'filing_id'
                  AND c.table_name NOT IN ('filings', 'financial_facts')
                  AND t.table_type   = 'BASE TABLE'
                ORDER BY c.table_name
                """
            ),
            {"s": SCHEMA},
        ).fetchall()
    return [r[0] for r in rows]


def delete_filing_data(engine: Engine, filing_id: int) -> dict[str, int]:
    """Remove all rows in every ontology.* table where filing_id = X.

    Auto-discovers child tables via information_schema (so new financial
    tables added later are picked up automatically). Does NOT delete the
    parent ``filings`` row.

    Returns a {table_name: rows_deleted} report.
    """
    deleted: dict[str, int] = {}
    with engine.begin() as conn:
        for name in _tables_with_filing_id(engine):
            result = conn.execute(
                sa_text(f'DELETE FROM {SCHEMA}."{name}" WHERE filing_id = :fid'),
                {"fid": filing_id},
            )
            deleted[name] = result.rowcount or 0
    return deleted
