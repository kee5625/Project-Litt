import json
import logging
import sqlite3
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ingestion.utils.rate_limiter import TokenBucketRateLimiter
from ingestion.utils.state import (
    is_cl_page_written,
    mark_cl_page_written,
    save_state,
)

log = logging.getLogger(__name__)

CL_BASE_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"
COURT_IDS = ["scotus", "ca9", "cacd", "cand", "caed", "casd", "cal", "calctapp"]
# Correct filter field: opinions relate to courts via cluster → docket → court
COURT_FILTER_FIELD = "cluster__docket__court"
OUTPUT_DIR = Path("raw/cl")
PAGE_SIZE = 20


def fetch_courtlistener(
    state: dict,
    conn: sqlite3.Connection,
    limiter: TokenBucketRateLimiter,
    session: requests.Session,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cursor = state.get("cl_cursor")
    page_num = state.get("cl_page", 0)

    if state.get("cl_done"):
        log.info("CourtListener fetch already complete — skipping.")
        return

    log.info("Starting CourtListener fetch from page %d (cursor=%s)", page_num, cursor)
    total_docs = 0

    while True:
        if is_cl_page_written(conn, page_num):
            log.debug("Page %d already written — skipping.", page_num)
            # Still need to advance cursor: read it from state
            cursor = state.get("cl_cursor")
            if not cursor and page_num > 0:
                log.warning("Page %d skipped but no cursor saved — restarting from page 0.", page_num)
                page_num = 0
                cursor = None
            page_num += 1
            continue

        limiter.acquire()
        params = _build_params(cursor)

        try:
            data = _get_page(session, params)
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                log.warning("HTTP 429 from CourtListener — sleeping 60s.")
                time.sleep(60)
                continue
            raise

        results = data.get("results", [])
        next_url = data.get("next")

        out_path = OUTPUT_DIR / f"page_{page_num:04d}.json"
        with open(out_path, "w") as f:
            json.dump(data, f)
            f.flush()

        mark_cl_page_written(conn, page_num, cursor, len(results))
        total_docs += len(results)

        next_cursor = _extract_cursor(next_url)
        state["cl_cursor"] = next_cursor
        state["cl_page"] = page_num + 1
        if not next_url:
            state["cl_done"] = True
        save_state(state)

        log.info(
            "Page %04d written — %d docs (total so far: %d)", page_num, len(results), total_docs
        )

        if not next_url:
            log.info("CourtListener fetch complete. Total pages: %d, docs: %d", page_num + 1, total_docs)
            break

        page_num += 1
        cursor = next_cursor


def _build_params(cursor: str | None) -> list[tuple]:
    params: list[tuple] = [(COURT_FILTER_FIELD, c) for c in COURT_IDS]
    params += [("format", "json"), ("page_size", str(PAGE_SIZE))]
    if cursor:
        params.append(("cursor", cursor))
    return params


def _extract_cursor(next_url: str | None) -> str | None:
    if not next_url:
        return None
    qs = parse_qs(urlparse(next_url).query)
    cursors = qs.get("cursor", [])
    return cursors[0] if cursors else None


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        return exc.response is None or exc.response.status_code != 429
    return True


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _get_page(session: requests.Session, params: dict) -> dict:
    resp = session.get(CL_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()
