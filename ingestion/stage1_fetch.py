#!/usr/bin/env python3
# Ensure project root is on sys.path when run as `python ingestion/stage1_fetch.py`
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Stage 1 — Fetch Raw

Downloads all raw data to disk without processing, embedding, or inserting.
Safe to kill and resume at any point.

Usage:
    python ingestion/stage1_fetch.py
    python ingestion/stage1_fetch.py --cl-only
    python ingestion/stage1_fetch.py --pol-only
    python ingestion/stage1_fetch.py --statutes-only
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from ingestion.fetchers.courtlistener import fetch_courtlistener
from ingestion.fetchers.hf_streaming import PRACTICE_AREA_KEYWORDS, stream_pile_of_law
from ingestion.fetchers.statutes import fetch_ca_statutes, fetch_federal_statutes
from ingestion.utils.rate_limiter import TokenBucketRateLimiter
from ingestion.utils.state import init_db, load_state


def main() -> None:
    load_dotenv()

    args = _parse_args()
    run_cl = args.cl_only or not any([args.pol_only, args.statutes_only])
    run_pol = args.pol_only or not any([args.cl_only, args.statutes_only])
    run_statutes = args.statutes_only or not any([args.cl_only, args.pol_only])

    _setup_logging()

    log = logging.getLogger(__name__)
    log.info("=== Stage 1: Fetch Raw ===")
    log.info("Fetchers: cl=%s  pol=%s  statutes=%s", run_cl, run_pol, run_statutes)

    # Create output directories upfront
    for d in ["raw/cl", "raw/pol", "raw/statutes/federal", "raw/statutes/ca"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    state = load_state()
    conn = init_db()

    # Rate limiters
    cl_limiter = TokenBucketRateLimiter(rate=45.0, burst=10)
    fed_limiter = TokenBucketRateLimiter(rate=5.0, burst=1)
    ca_limiter = TokenBucketRateLimiter(rate=1.0, burst=1)

    # Shared HTTP session (no auth by default; CL fetcher sets its own header)
    session = requests.Session()
    session.headers.update({"User-Agent": "LegalMind-Ingestion/1.0"})

    cl_token = os.environ.get("CL_API_TOKEN")
    if not cl_token and run_cl:
        log.error("CL_API_TOKEN not set. Add it to .env: CL_API_TOKEN=Token <your_token>")
        sys.exit(1)

    # Execution order: federal statutes first (quick network check), then CL, CA, HF
    if run_statutes:
        log.info("--- Federal Statutes ---")
        fetch_federal_statutes(conn, session, fed_limiter)

    # if run_cl:
    #     log.info("--- CourtListener ---")
    #     cl_session = requests.Session()
    #     cl_session.headers.update({
    #         "Authorization": f"Token {cl_token}",
    #         "User-Agent": "LegalMind-Ingestion/1.0",
    #     })
    #     fetch_courtlistener(state, conn, cl_limiter, cl_session)

    if run_statutes:
        log.info("--- California Statutes ---")
        fetch_ca_statutes(conn, session, ca_limiter)

    if run_pol:
        log.info("--- HuggingFace pile-of-law ---")
        stream_pile_of_law(conn, PRACTICE_AREA_KEYWORDS)

    conn.close()
    log.info("=== Stage 1 complete ===")
    log.info("Verify with:")
    log.info("  ls raw/cl/ | wc -l          # expect ~180")
    log.info("  wc -l raw/pol/*.jsonl       # expect 1500 / 500 / 200")
    log.info("  ls -lh raw/statutes/federal/ # expect 3 ZIP files")
    log.info("  find raw/statutes/ca/ -name '*.html' | wc -l  # expect ~1100+")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Fetch raw legal data to disk.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cl-only", action="store_true", help="Only fetch CourtListener opinions")
    group.add_argument("--pol-only", action="store_true", help="Only stream pile-of-law")
    group.add_argument("--statutes-only", action="store_true", help="Only download statutes")
    return parser.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("stage1.log"),
        ],
    )


if __name__ == "__main__":
    main()
