"""End-to-end ingestion driver. One filing → one unit of work.

Shared singletons (parser, embedder, tree_builder, store, table_proc) are
constructed once by ``ingest_all`` and threaded through so in-memory caches
survive across filings.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from config.filing_types import FILING_TYPES

from src.classify import classify_table
from src.db import get_engine
from src.embeddings import ContextualEmbedder, RAPTORTree, VectorStore
from src.manifest import RunManifest
from src.ontology import (
    delete_filing_data,
    init_ontology_schema,
    upsert_company,
    upsert_filing,
)
from src.parser import DocumentParser
from src.processor import Chunk, TableProcessor, TextChunker

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_DIR = _REPO_ROOT / "data" / "source"
_PROC_DIR = _REPO_ROOT / "data" / "processed"


def _build_speaker_rules(company_cfg: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Map ``speaker_keywords`` from config to ``TextChunker`` rule tuples."""
    sk = company_cfg.get("speaker_keywords") or {}
    return [(role, kws) for role, kws in sk.items() if kws]


def _derive_period(fiscal_year: int, fiscal_quarter: int | None) -> str:
    """Human-readable period label. ``fiscal_quarter`` None → annual."""
    return f"FY-{fiscal_year}" if fiscal_quarter is None else f"Q{fiscal_quarter}-{fiscal_year}"


def ingest_filing(
    *,
    engine: Engine,
    parser: DocumentParser,
    table_proc: TableProcessor,
    chunker: TextChunker,
    embedder: ContextualEmbedder | None,
    tree_builder: RAPTORTree | None,
    store: VectorStore | None,
    company_cfg: dict[str, Any],
    filing: dict[str, Any],
    skip_embed: bool = False,
    manifest: RunManifest | None = None,
) -> dict[str, Any]:
    """Run the pipeline for ONE filing. Returns a report dict."""
    ticker = company_cfg["ticker"]
    ftype = filing["filing_type"]
    period = _derive_period(filing["fiscal_year"], filing.get("fiscal_quarter"))
    pdf_path = _SOURCE_DIR / filing["pdf_relative"]
    spec = FILING_TYPES[ftype]
    t0 = time.monotonic()
    log.info("[%s %s %s] start  pdf=%s", ticker, ftype, period, filing["pdf_relative"])

    filing_id = upsert_filing(
        engine,
        ticker=ticker,
        filing_type=ftype,
        period_end_date=filing["period_end_date"],
        fiscal_year=filing["fiscal_year"],
        fiscal_quarter=filing.get("fiscal_quarter"),
        filed_date=filing["filed_date"],
        source_pdf=filing["pdf_relative"],
    )

    parsed = parser.parse(str(pdf_path))

    writes: Counter[str] = Counter()
    unmatched: list[tuple[int, str]] = []
    meta_base = {"ticker": ticker, "period": period, "filing_type": ftype}
    if spec["has_tables"]:
        for i, md in enumerate(parsed["tables"]):
            stmt = classify_table(md, company_cfg)
            if stmt is None:
                snippet = md[:200].replace("\n", " ")
                unmatched.append((i, snippet))
                if manifest is not None:
                    manifest.record_unmatched(
                        ticker=ticker,
                        filing_type=ftype,
                        pdf_relative=filing["pdf_relative"],
                        table_index=i,
                        snippet=snippet,
                    )
                continue
            try:
                df = table_proc.markdown_to_df(md)
                if df.empty:
                    continue
                table_proc.store(
                    df, statement_type=stmt, filing_id=filing_id, meta=meta_base
                )
                writes[stmt] += 1
            except Exception as e:
                log.warning(
                    "[%s %s] table#%d → %s failed: %s: %s",
                    ticker,
                    ftype,
                    i,
                    stmt,
                    type(e).__name__,
                    e,
                )
    else:
        if parsed["tables"]:
            log.debug(
                "[%s %s] has_tables=False but parser produced %d tables; ignoring",
                ticker,
                ftype,
                len(parsed["tables"]),
            )

    base_meta = {**meta_base, "filing_id": filing_id}
    speaker_rules = _build_speaker_rules(company_cfg)
    leaves: list[Chunk] = chunker.chunk(
        parsed["text"],
        base_meta=base_meta,
        speaker_rules=speaker_rules or None,
    )

    nodes: list[Chunk] = []
    if skip_embed:
        log.info(
            "[%s %s] --skip-embed: %d leaves, no embed/RAPTOR/upsert",
            ticker,
            ftype,
            len(leaves),
        )
    else:
        assert embedder is not None and tree_builder is not None and store is not None
        full_doc = (_PROC_DIR / f"{pdf_path.stem}.md").read_text(encoding="utf-8")
        embedder.embed(leaves, full_doc=full_doc, contextualize=True)
        nodes = tree_builder.build(leaves, max_levels=3)
        delete_filing_data(engine, filing_id)
        store.upsert(nodes)

    postprocess_reports: dict[str, Any] = {}
    if ftype in {"10-K", "10-Q"}:
        if company_cfg.get("needs_bare_year_fix"):
            from src.postprocess import fix_bare_year_periods

            postprocess_reports["bare_year"] = fix_bare_year_periods(
                engine, filing_id, company_cfg, filing
            )
        if company_cfg.get("needs_segment_reclassify"):
            from src.postprocess import reclassify_segment_headers

            postprocess_reports["segment"] = reclassify_segment_headers(
                engine, filing_id, company_cfg, filing
            )

    elapsed = time.monotonic() - t0
    log.info(
        "[%s %s %s] done  filing_id=%d  tables: %s (unmatched=%d)  "
        "leaves=%d  nodes=%d  %.1fs",
        ticker,
        ftype,
        period,
        filing_id,
        dict(writes),
        len(unmatched),
        len(leaves),
        len(nodes),
        elapsed,
    )
    report: dict[str, Any] = {
        "ticker": ticker,
        "filing_type": ftype,
        "period": period,
        "filing_id": filing_id,
        "writes": dict(writes),
        "unmatched": unmatched,
        "n_leaves": len(leaves),
        "n_nodes": len(nodes),
        "elapsed_sec": elapsed,
    }
    if postprocess_reports:
        report["postprocess"] = postprocess_reports
    return report


def ingest_company(
    *,
    engine: Engine,
    parser: DocumentParser,
    table_proc: TableProcessor,
    chunker: TextChunker,
    embedder: ContextualEmbedder | None,
    tree_builder: RAPTORTree | None,
    store: VectorStore | None,
    company_cfg: dict[str, Any],
    filing_type_filter: set[str] | None,
    skip_embed: bool,
    manifest: RunManifest | None = None,
) -> list[dict[str, Any]]:
    upsert_company(
        engine,
        ticker=company_cfg["ticker"],
        legal_name=company_cfg["legal_name"],
        cik=company_cfg.get("cik"),
        fiscal_year_end=company_cfg.get("fiscal_year_end"),
        hq_country=company_cfg.get("hq_country"),
    )
    reports: list[dict[str, Any]] = []
    for filing in company_cfg["filings"]:
        if filing_type_filter and filing["filing_type"] not in filing_type_filter:
            continue
        try:
            r = ingest_filing(
                engine=engine,
                parser=parser,
                table_proc=table_proc,
                chunker=chunker,
                embedder=embedder,
                tree_builder=tree_builder,
                store=store,
                company_cfg=company_cfg,
                filing=filing,
                skip_embed=skip_embed,
                manifest=manifest,
            )
            if manifest is not None:
                manifest.record_filing(r)
            reports.append(r)
        except Exception as e:
            log.exception(
                "[%s %s] FAILED: %s",
                company_cfg["ticker"],
                filing["filing_type"],
                e,
            )
            if manifest is not None:
                manifest.record_failure(
                    ticker=company_cfg["ticker"],
                    filing_type=filing["filing_type"],
                    pdf_relative=filing["pdf_relative"],
                    exc=e,
                    filing_meta=filing,
                )
            reports.append(
                {
                    "ticker": company_cfg["ticker"],
                    "filing_type": filing["filing_type"],
                    "pdf_relative": filing["pdf_relative"],
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    return reports


def ingest_all(
    companies: list[dict[str, Any]],
    *,
    ticker_filter: set[str] | None = None,
    filing_type_filter: set[str] | None = None,
    skip_embed: bool = False,
    manifest: RunManifest | None = None,
) -> list[dict[str, Any]]:
    """Build shared singletons once, then loop companies."""
    engine = get_engine()
    init_ontology_schema(engine)
    parser = DocumentParser()
    table_proc = TableProcessor(engine)
    chunker = TextChunker(size=512, overlap=50)
    embedder = ContextualEmbedder() if not skip_embed else None
    tree_builder = RAPTORTree(embedder) if not skip_embed else None
    store = VectorStore(engine) if not skip_embed else None

    all_reports: list[dict[str, Any]] = []
    for co in companies:
        if ticker_filter and co["ticker"] not in ticker_filter:
            continue
        all_reports.extend(
            ingest_company(
                engine=engine,
                parser=parser,
                table_proc=table_proc,
                chunker=chunker,
                embedder=embedder,
                tree_builder=tree_builder,
                store=store,
                company_cfg=co,
                filing_type_filter=filing_type_filter,
                skip_embed=skip_embed,
                manifest=manifest,
            )
        )
    return all_reports
