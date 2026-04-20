#!/usr/bin/env python3
"""
Stage 2 — Parse + Chunk + LLM Tag

Reads Stage 1 raw data, produces:
  processed/case_summaries.jsonl   (one row per case)
  processed/case_law.jsonl         (one row per chunk)
  processed/statutes.jsonl         (one row per statute section)

Safe to kill and resume — all processed opinion_ids tracked in state.db.

Usage:
    python ingestion/stage2_process.py
    python ingestion/stage2_process.py --cl-only
    python ingestion/stage2_process.py --pol-only
    python ingestion/stage2_process.py --statutes-only
    python ingestion/stage2_process.py --skip-llm     # deterministic summaries only
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import os
import re
import time
from typing import IO

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from ingestion.parsers.opinion_parser import chunk_opinion, load_splitter, token_count
from ingestion.parsers.statute_parser import parse_all_statutes
from ingestion.taggers.groq_tagger import make_deterministic_summary, tag_opinion
from ingestion.utils.rate_limiter import TokenBucketRateLimiter
from ingestion.utils.state import (
    init_db,
    is_opinion_processed,
    mark_opinion_processed,
)
from ingestion.utils.practice_area import classify_practice_area as _classify_practice_area

log = logging.getLogger(__name__)

RAW_CL = Path("raw/cl")
RAW_POL = Path("raw/pol")
PROCESSED = Path("processed")

# ---------------------------------------------------------------------------
# Sampling targets for CL raw opinions
# ---------------------------------------------------------------------------
CL_TARGETS: dict[str, int] = {
    "scotus": 50,
    "ca9": 450,
    "cal": 100,
    "cacd": 200,
    "cand": 200,
    "caed": 150,
    "casd": 100,
    "calctapp": 50,
}
CL_TOTAL_TARGET = 1300

# ---------------------------------------------------------------------------
# Court metadata lookup tables
# ---------------------------------------------------------------------------
COURT_TIER: dict[str, int] = {
    "scotus": 3,
    "ca9": 2, "cal": 2,
    "cacd": 1, "cand": 1, "caed": 1, "casd": 1, "calctapp": 1,
}
COURT_JURISDICTION: dict[str, str] = {
    "scotus": "federal", "ca9": "federal",
    "cacd": "federal", "cand": "federal", "caed": "federal", "casd": "federal",
    "cal": "CA", "calctapp": "CA",
}
COURT_ABBR: dict[str, str] = {
    "scotus": "U.S.", "ca9": "9th Cir.",
    "cacd": "C.D. Cal.", "cand": "N.D. Cal.", "caed": "E.D. Cal.", "casd": "S.D. Cal.",
    "cal": "Cal.", "calctapp": "Cal. Ct. App.",
}


# ---------------------------------------------------------------------------
# CL cluster metadata fetching
# ---------------------------------------------------------------------------

def _make_cl_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Token {token}",
        "User-Agent": "LegalMind-Ingestion/2.0",
    })
    return s


def _court_id_from_url(court_url: str) -> str:
    """Extract court slug from CL court URL like .../courts/ca9/."""
    return court_url.rstrip("/").split("/")[-1]


def _build_cite_str(case_name: str, citations: list, court_abbr: str, year: int) -> str:
    if citations:
        c = citations[0]
        vol = c.get("volume", "")
        rep = c.get("reporter", "")
        page = c.get("page", "")
        if vol and rep and page:
            return f"{case_name}, {vol} {rep} {page} ({court_abbr} {year})"
    return f"{case_name} ({court_abbr} {year})"


def fetch_cluster(cluster_id: int, session: requests.Session,
                  limiter: TokenBucketRateLimiter) -> dict | None:
    """Fetch CL v4 cluster object. Returns None on failure."""
    limiter.acquire()
    url = f"https://www.courtlistener.com/api/rest/v4/clusters/{cluster_id}/"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 429:
            log.warning("CL 429 on cluster %d — sleeping 60s", cluster_id)
            time.sleep(60)
            return fetch_cluster(cluster_id, session, limiter)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("Failed to fetch cluster %d: %s", cluster_id, exc)
        return None


def fetch_docket(docket_id: int, session: requests.Session,
                 limiter: TokenBucketRateLimiter) -> dict | None:
    """Fetch CL v4 docket to get court_id (v4 clusters no longer carry court directly)."""
    limiter.acquire()
    url = f"https://www.courtlistener.com/api/rest/v4/dockets/{docket_id}/"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 429:
            log.warning("CL 429 on docket %d — sleeping 60s", docket_id)
            time.sleep(60)
            return fetch_docket(docket_id, session, limiter)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("Failed to fetch docket %d: %s", docket_id, exc)
        return None


def fetch_opinion_v4(opinion_id: str, session: requests.Session,
                     limiter: TokenBucketRateLimiter) -> dict | None:
    """Fetch CL v4 opinion to get cluster_id (needed for pol CL opinions)."""
    limiter.acquire()
    url = f"https://www.courtlistener.com/api/rest/v4/opinions/{opinion_id}/"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 429:
            log.warning("CL 429 on opinion %s — sleeping 60s", opinion_id)
            time.sleep(60)
            return fetch_opinion_v4(opinion_id, session, limiter)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("Failed to fetch opinion %s: %s", opinion_id, exc)
        return None


def extract_metadata_from_cluster(cluster: dict, docket: dict | None = None) -> dict:
    """Parse a CL cluster response into the fields we need."""
    case_name = (cluster.get("case_name") or "Unknown v. Unknown")[:512]
    date_filed = cluster.get("date_filed") or ""
    year_match = re.search(r"(\d{4})", date_filed)
    year = int(year_match.group(1)) if year_match else 2000

    # v4 clusters no longer carry court directly — it lives on the docket
    court_raw = cluster.get("court") or ""
    if court_raw and "/" in court_raw:
        court_id = _court_id_from_url(court_raw)
    elif court_raw:
        court_id = court_raw
    elif docket:
        court_id = docket.get("court_id") or ""
        if not court_id:
            court_url = docket.get("court") or ""
            court_id = _court_id_from_url(court_url) if court_url else ""
    else:
        court_id = ""

    citations = cluster.get("citations") or []
    if isinstance(citations, list) and citations and isinstance(citations[0], str):
        # Some versions return formatted strings; parse them
        citations = []

    negative = cluster.get("negative_treatment") or ""
    is_good_law = not bool(negative.strip())

    court_abbr = COURT_ABBR.get(court_id, court_id)
    return {
        "court_id": court_id,
        "case_name": case_name,
        "year": year,
        "citations": citations,
        "is_good_law": is_good_law,
        "court_abbr": court_abbr,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_jsonl(fh: IO, obj: dict) -> None:
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()


def _build_case_summary_row(case_id: str, case_name: str, citation: str,
                             court_id: str, jurisdiction: str, practice_area: str,
                             year: int, is_good_law: bool, full_cite_str: str,
                             outcome: str, ai_summary: str,
                             key_concepts: list[str]) -> dict:
    return {
        "case_id": case_id,
        "case_name": case_name,
        "citation": citation,
        "court": court_id,
        "court_tier": COURT_TIER.get(court_id, 1),
        "jurisdiction": jurisdiction,
        "practice_area": practice_area,
        "year": year,
        "is_good_law": is_good_law,
        "outcome": outcome,
        "ai_summary": ai_summary,
        "key_concepts": key_concepts,
        "full_cite_str": full_cite_str,
    }


def _build_chunk_rows(case_id: str, chunks: list[str], primary_idx: int,
                      case_name: str, citation: str, court_id: str,
                      jurisdiction: str, practice_area: str, year: int,
                      is_good_law: bool, full_cite_str: str) -> list[dict]:
    rows = []
    for i, chunk in enumerate(chunks):
        rows.append({
            "id": f"{case_id}_{i}",
            "case_id": case_id,
            "case_name": case_name,
            "citation": citation,
            "court": court_id,
            "court_tier": COURT_TIER.get(court_id, 1),
            "jurisdiction": jurisdiction,
            "practice_area": practice_area,
            "year": year,
            "is_good_law": is_good_law,
            "is_primary_holding": (i == primary_idx),
            "holding_text": chunk,
            "full_cite_str": full_cite_str,
        })
    return rows


# ---------------------------------------------------------------------------
# Opinion processing core
# ---------------------------------------------------------------------------

def process_opinion(
    opinion_id: str,
    plain_text: str,
    case_id: str,
    case_name: str,
    citation: str,
    court_id: str,
    jurisdiction: str,
    practice_area: str,
    year: int,
    is_good_law: bool,
    full_cite_str: str,
    splitter,
    skip_llm: bool,
    summaries_fh: IO,
    chunks_fh: IO,
    conn,
    source: str,
) -> bool:
    """Chunk + tag one opinion and write to output files.

    Returns True on success, False if skipped (already processed or empty text).
    """
    if is_opinion_processed(conn, opinion_id):
        return False

    if not plain_text or len(plain_text.strip()) < 100:
        log.debug("Skipping %s — empty/short text", opinion_id)
        return False

    chunks = chunk_opinion(splitter, plain_text)
    if not chunks:
        log.debug("Skipping %s — no chunks after cleaning", opinion_id)
        return False

    short_opinion = token_count(plain_text) < 300

    if skip_llm or short_opinion:
        tags = make_deterministic_summary(case_name, court_id, year, plain_text)
    else:
        try:
            tags = tag_opinion(chunks)
        except Exception as exc:
            log.warning("Groq failed for %s: %s — using deterministic fallback", opinion_id, exc)
            tags = make_deterministic_summary(case_name, court_id, year, plain_text)

    primary_idx = tags["primary_holding_chunk_index"]

    summary_row = _build_case_summary_row(
        case_id=case_id,
        case_name=case_name,
        citation=citation,
        court_id=court_id,
        jurisdiction=jurisdiction,
        practice_area=practice_area,
        year=year,
        is_good_law=is_good_law,
        full_cite_str=full_cite_str,
        outcome=tags["outcome"],
        ai_summary=tags["ai_summary"],
        key_concepts=tags["key_concepts"],
    )
    _write_jsonl(summaries_fh, summary_row)

    chunk_rows = _build_chunk_rows(
        case_id=case_id,
        chunks=chunks,
        primary_idx=primary_idx,
        case_name=case_name,
        citation=citation,
        court_id=court_id,
        jurisdiction=jurisdiction,
        practice_area=practice_area,
        year=year,
        is_good_law=is_good_law,
        full_cite_str=full_cite_str,
    )
    for row in chunk_rows:
        _write_jsonl(chunks_fh, row)

    mark_opinion_processed(conn, opinion_id, source)
    return True


# ---------------------------------------------------------------------------
# Phase 1 — CL raw opinions (sample 1,300)
# ---------------------------------------------------------------------------

def run_cl_phase(splitter, skip_llm: bool, cl_session: requests.Session,
                 cl_limiter: TokenBucketRateLimiter, summaries_fh: IO,
                 chunks_fh: IO, conn) -> None:
    log.info("=== Phase 1: CL raw opinions (target: %d) ===", CL_TOTAL_TARGET)

    page_files = sorted(RAW_CL.glob("page_*.json"))
    if not page_files:
        log.warning("No CL page files found in %s", RAW_CL)
        return

    court_counts: dict[str, int] = {k: 0 for k in CL_TARGETS}
    total_processed = 0

    # Count already-processed CL opinions to resume correctly
    already_done = conn.execute(
        "SELECT COUNT(*) FROM processed_opinions WHERE source = 'cl'"
    ).fetchone()[0]
    total_processed = already_done
    log.info("Already processed %d CL opinions — resuming", already_done)

    pbar = tqdm(total=CL_TOTAL_TARGET, initial=already_done, desc="CL opinions", unit="op")

    for page_path in page_files:
        if total_processed >= CL_TOTAL_TARGET:
            break

        try:
            page_data = json.loads(page_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping page %s: %s", page_path, exc)
            continue

        for opinion in page_data.get("results", []):
            if total_processed >= CL_TOTAL_TARGET:
                break

            opinion_id = str(opinion.get("id", ""))
            cluster_id = opinion.get("cluster_id")
            plain_text = (opinion.get("plain_text") or "").strip()

            if not opinion_id or not cluster_id or not plain_text:
                continue

            if is_opinion_processed(conn, f"cl_{opinion_id}"):
                continue

            # Practice area filter (fast path before API call)
            areas = _classify_practice_area(plain_text.lower())
            if not areas:
                continue
            practice_area = areas[0]

            # Fetch cluster metadata (court_id lives on docket in v4)
            cluster = fetch_cluster(cluster_id, cl_session, cl_limiter)
            if not cluster:
                continue

            docket = None
            if not cluster.get("court") and cluster.get("docket_id"):
                docket = fetch_docket(cluster["docket_id"], cl_session, cl_limiter)
            meta = extract_metadata_from_cluster(cluster, docket)
            court_id = meta["court_id"]

            # Court-level sampling cap
            if court_id not in court_counts:
                court_counts[court_id] = 0
            target = CL_TARGETS.get(court_id, 0)
            if target == 0 or court_counts[court_id] >= target:
                continue

            jurisdiction = COURT_JURISDICTION.get(court_id, "federal")
            court_abbr = meta["court_abbr"]
            case_name = meta["case_name"]
            year = meta["year"]
            is_good_law = meta["is_good_law"]
            citation_raw = meta["citations"][0] if meta["citations"] else {}
            citation = (f"{citation_raw.get('volume','')} {citation_raw.get('reporter','')} "
                        f"{citation_raw.get('page','')}").strip() if citation_raw else ""
            full_cite_str = _build_cite_str(case_name, meta["citations"], court_abbr, year)
            case_id = f"cl_{opinion_id}"

            ok = process_opinion(
                opinion_id=f"cl_{opinion_id}",
                plain_text=plain_text,
                case_id=case_id,
                case_name=case_name,
                citation=citation,
                court_id=court_id,
                jurisdiction=jurisdiction,
                practice_area=practice_area,
                year=year,
                is_good_law=is_good_law,
                full_cite_str=full_cite_str,
                splitter=splitter,
                skip_llm=skip_llm,
                summaries_fh=summaries_fh,
                chunks_fh=chunks_fh,
                conn=conn,
                source="cl",
            )
            if ok:
                court_counts[court_id] += 1
                total_processed += 1
                pbar.update(1)

    pbar.close()
    log.info("Phase 1 done: %d CL opinions processed", total_processed)


# ---------------------------------------------------------------------------
# Phase 2 — pol courtlisteneropinions
# ---------------------------------------------------------------------------

def run_pol_cl_phase(splitter, skip_llm: bool, cl_session: requests.Session,
                     cl_limiter: TokenBucketRateLimiter, summaries_fh: IO,
                     chunks_fh: IO, conn) -> None:
    jsonl_path = RAW_POL / "courtlisteneropinions.jsonl"
    if not jsonl_path.exists():
        log.warning("Missing %s — skipping pol CL phase", jsonl_path)
        return

    log.info("=== Phase 2: pol courtlisteneropinions ===")
    lines = jsonl_path.read_text().splitlines()
    pbar = tqdm(lines, desc="pol CL opinions", unit="op")

    for line in pbar:
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue

        url = doc.get("url", "")
        plain_text = (doc.get("text") or "").strip()

        m = re.search(r"/opinions/(\d+)/?", url)
        if not m or not plain_text:
            continue
        raw_opinion_id = m.group(1)
        opinion_id = f"pol_cl_{raw_opinion_id}"

        if is_opinion_processed(conn, opinion_id):
            continue

        practice_area_matches: list[str] = doc.get("practice_area_matches", [])
        if not practice_area_matches:
            practice_area_matches = _classify_practice_area(plain_text.lower())
        if not practice_area_matches:
            continue
        practice_area = practice_area_matches[0]

        # Fetch v4 opinion to get cluster_id
        opinion_v4 = fetch_opinion_v4(raw_opinion_id, cl_session, cl_limiter)
        if not opinion_v4:
            continue
        cluster_id = opinion_v4.get("cluster_id")
        if not cluster_id:
            continue

        cluster = fetch_cluster(cluster_id, cl_session, cl_limiter)
        if not cluster:
            continue

        docket = None
        if not cluster.get("court") and cluster.get("docket_id"):
            docket = fetch_docket(cluster["docket_id"], cl_session, cl_limiter)
        meta = extract_metadata_from_cluster(cluster, docket)
        court_id = meta["court_id"]
        jurisdiction = COURT_JURISDICTION.get(court_id, "federal")
        court_abbr = meta["court_abbr"]
        case_name = meta["case_name"]
        year = meta["year"]
        is_good_law = meta["is_good_law"]
        citation_raw = meta["citations"][0] if meta["citations"] else {}
        citation = (f"{citation_raw.get('volume','')} {citation_raw.get('reporter','')} "
                    f"{citation_raw.get('page','')}").strip() if citation_raw else ""
        full_cite_str = _build_cite_str(case_name, meta["citations"], court_abbr, year)
        case_id = f"cl_{raw_opinion_id}"

        process_opinion(
            opinion_id=opinion_id,
            plain_text=plain_text,
            case_id=case_id,
            case_name=case_name,
            citation=citation,
            court_id=court_id,
            jurisdiction=jurisdiction,
            practice_area=practice_area,
            year=year,
            is_good_law=is_good_law,
            full_cite_str=full_cite_str,
            splitter=splitter,
            skip_llm=skip_llm,
            summaries_fh=summaries_fh,
            chunks_fh=chunks_fh,
            conn=conn,
            source="pol_cl",
        )

    pbar.close()
    log.info("Phase 2 done")


# ---------------------------------------------------------------------------
# Phase 3 — pol nlrb_decisions
# ---------------------------------------------------------------------------

def _parse_nlrb_metadata(text: str, doc_index: int) -> dict:
    name_match = re.search(
        r"(?:NATIONAL\s+LABOR\s+RELATIONS\s+BOARD\s*)([\s\S]{5,200}?)\.?\s*Case\s+[\d\w\-]+",
        text, re.IGNORECASE,
    )
    case_name = (
        name_match.group(1).replace("\n", " ").strip()
        if name_match else f"NLRB Decision {doc_index}"
    )

    num_match = re.search(r"\bCase\s+([\d\w][\d\w\-]*)", text, re.IGNORECASE)
    case_number = num_match.group(1) if num_match else str(doc_index)

    year_match = re.search(
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+(\d{4})",
        text, re.IGNORECASE,
    )
    year = int(year_match.group(1)) if year_match else 1980

    return {
        "case_id": f"nlrb_{case_number}",
        "case_name": case_name[:512],
        "citation": f"NLRB Case {case_number}",
        "court": "nlrb",
        "court_tier": 1,
        "jurisdiction": "federal",
        "practice_area": "employment",
        "year": year,
        "is_good_law": True,
        "full_cite_str": f"{case_name[:80]}, NLRB Case {case_number} ({year})",
    }


def run_pol_nlrb_phase(splitter, skip_llm: bool, summaries_fh: IO,
                       chunks_fh: IO, conn) -> None:
    jsonl_path = RAW_POL / "nlrb_decisions.jsonl"
    if not jsonl_path.exists():
        log.warning("Missing %s — skipping NLRB phase", jsonl_path)
        return

    log.info("=== Phase 3: pol nlrb_decisions ===")
    lines = jsonl_path.read_text().splitlines()
    pbar = tqdm(lines, desc="NLRB decisions", unit="op")

    for idx, line in enumerate(pbar):
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue

        plain_text = (doc.get("text") or "").strip()
        if not plain_text:
            continue

        meta = _parse_nlrb_metadata(plain_text, idx)
        opinion_id = f"nlrb_{meta['case_id']}"

        if is_opinion_processed(conn, opinion_id):
            continue

        process_opinion(
            opinion_id=opinion_id,
            plain_text=plain_text,
            case_id=meta["case_id"],
            case_name=meta["case_name"],
            citation=meta["citation"],
            court_id=meta["court"],
            jurisdiction=meta["jurisdiction"],
            practice_area=meta["practice_area"],
            year=meta["year"],
            is_good_law=meta["is_good_law"],
            full_cite_str=meta["full_cite_str"],
            splitter=splitter,
            skip_llm=skip_llm,
            summaries_fh=summaries_fh,
            chunks_fh=chunks_fh,
            conn=conn,
            source="pol_nlrb",
        )

    pbar.close()
    log.info("Phase 3 done")


# ---------------------------------------------------------------------------
# Phase 4 — Statutes
# ---------------------------------------------------------------------------

def run_statutes_phase(statutes_fh: IO) -> None:
    log.info("=== Phase 4: Statutes ===")
    rows = parse_all_statutes()
    log.info("Parsed %d statute sections total", len(rows))

    # Deduplicate against already-written rows (simple: check if file has content)
    written = 0
    for row in tqdm(rows, desc="Statutes", unit="sec"):
        _write_jsonl(statutes_fh, row)
        written += 1

    log.info("Phase 4 done: %d statute rows written", written)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    args = _parse_args()
    _setup_logging()

    run_cl = args.cl_only or not any([args.pol_only, args.statutes_only])
    run_pol = args.pol_only or not any([args.cl_only, args.statutes_only])
    run_statutes = args.statutes_only or not any([args.cl_only, args.pol_only])

    cl_token = os.environ.get("CL_API_TOKEN")
    if not cl_token and (run_cl or run_pol):
        log.error("CL_API_TOKEN not set in .env — required for cluster metadata fetches")
        sys.exit(1)

    if (run_cl or run_pol) and not args.skip_llm:
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            log.error("GROQ_API_KEY not set in .env")
            sys.exit(1)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    conn = init_db()

    cl_session = _make_cl_session(cl_token) if cl_token else None
    cl_limiter = TokenBucketRateLimiter(rate=45.0, burst=10)

    splitter = None
    if run_cl or run_pol:
        if args.skip_llm:
            log.info("Loading fast sentence splitter (no embeddings)...")
            splitter = load_splitter(semantic=False)
        else:
            log.info("Loading BGE-M3 embed model for semantic chunking...")
            splitter = load_splitter(semantic=True)
        log.info("Splitter ready.")

    summaries_path = PROCESSED / "case_summaries.jsonl"
    chunks_path = PROCESSED / "case_law.jsonl"
    statutes_path = PROCESSED / "statutes.jsonl"

    # Open in append mode so resume doesn't overwrite existing rows
    with (
        open(summaries_path, "a") as summaries_fh,
        open(chunks_path, "a") as chunks_fh,
        open(statutes_path, "a") as statutes_fh,
    ):
        if run_cl:
            run_cl_phase(splitter, args.skip_llm, cl_session, cl_limiter,
                         summaries_fh, chunks_fh, conn)

        if run_pol:
            run_pol_cl_phase(splitter, args.skip_llm, cl_session, cl_limiter,
                             summaries_fh, chunks_fh, conn)
            run_pol_nlrb_phase(splitter, args.skip_llm,
                               summaries_fh, chunks_fh, conn)

        if run_statutes:
            run_statutes_phase(statutes_fh)

    conn.close()
    log.info("=== Stage 2 complete ===")
    _print_summary()


def _print_summary() -> None:
    for label, path in [
        ("case_summaries", PROCESSED / "case_summaries.jsonl"),
        ("case_law chunks", PROCESSED / "case_law.jsonl"),
        ("statutes", PROCESSED / "statutes.jsonl"),
    ]:
        if path.exists():
            count = sum(1 for _ in open(path))
            log.info("  %-20s %d rows", label, count)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: Parse, chunk, and LLM-tag legal data.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cl-only", action="store_true")
    group.add_argument("--pol-only", action="store_true")
    group.add_argument("--statutes-only", action="store_true")
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Use deterministic summaries instead of calling Groq (fast, for testing)",
    )
    return parser.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("stage2.log"),
        ],
    )


if __name__ == "__main__":
    main()
