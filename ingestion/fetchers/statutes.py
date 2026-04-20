import logging
import re
import sqlite3
import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.utils.rate_limiter import TokenBucketRateLimiter
from ingestion.utils.state import is_statute_written, mark_statute_written

log = logging.getLogger(__name__)

FED_OUTPUT_DIR = Path("raw/statutes/federal")
CA_OUTPUT_DIR = Path("raw/statutes/ca")

FEDERAL_STATUTE_URLS: dict[str, str] = {
    "usc28": "https://uscode.house.gov/download/releasepoints/us/pl/119/83/xml_usc28@119-83.zip",
    "usc29": "https://uscode.house.gov/download/releasepoints/us/pl/119/83/xml_usc29@119-83.zip",
    "usc42": "https://uscode.house.gov/download/releasepoints/us/pl/119/83/xml_usc42@119-83.zip",
}

# (law_code, section_range, group_name)
CA_STATUTE_TARGETS: list[tuple[str, range, str]] = [
    ("LAB", range(98, 133), "LAB_98_132"),
    ("LAB", range(1100, 1200), "LAB_1100_1199"),
    ("FAM", range(3000, 4001), "FAM_3000_4000"),
    ("PEN", range(832, 848), "PEN_832_847"),
    ("CIV", range(1714, 1726), "CIV_1714_1725"),
]

CA_BASE_URL = "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml"
USCODE_CONTENTS_URL = "https://uscode.house.gov/download/downloadcontents.htm"

CHUNK_SIZE = 8 * 1024  # 8 KB


def fetch_federal_statutes(
    conn: sqlite3.Connection,
    session: requests.Session,
    limiter: TokenBucketRateLimiter,
) -> None:
    FED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    release = _resolve_latest_uscode_release(session)
    urls = _build_uscode_urls(release)

    for name, url in urls.items():
        out_path = FED_OUTPUT_DIR / f"{name}.zip"
        if is_statute_written(conn, str(out_path)):
            log.info("Federal statute %s already downloaded — skipping.", name)
            continue

        log.info("Downloading %s from %s", name, url)
        limiter.acquire()
        try:
            _stream_download(session, url, out_path)
            mark_statute_written(conn, str(out_path))
            size_mb = out_path.stat().st_size / (1024 * 1024)
            log.info("  %s written (%.1f MB)", out_path, size_mb)
        except Exception:
            log.exception("Failed to download %s", url)


def fetch_ca_statutes(
    conn: sqlite3.Connection,
    session: requests.Session,
    ca_limiter: TokenBucketRateLimiter,
) -> None:
    CA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for law_code, sections, group_name in CA_STATUTE_TARGETS:
        group_dir = CA_OUTPUT_DIR / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        total = len(sections)
        written = 0
        skipped_gaps = 0

        log.info("Fetching CA %s (%d sections)...", group_name, total)

        for section_num in sections:
            out_path = group_dir / f"section_{section_num:05d}.html"
            if is_statute_written(conn, str(out_path)):
                written += 1
                continue

            ca_limiter.acquire()
            try:
                html = _fetch_ca_section(session, law_code, section_num)
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    log.debug("CA %s §%d: 404 — section gap, skipping.", law_code, section_num)
                    skipped_gaps += 1
                    continue
                log.warning("CA %s §%d: HTTP error %s", law_code, section_num, exc)
                continue
            except Exception:
                log.exception("CA %s §%d: unexpected error", law_code, section_num)
                continue

            if len(html) < 500:
                log.debug("CA %s §%d: body too short (%d bytes) — section gap.", law_code, section_num, len(html))
                skipped_gaps += 1
                continue

            out_path.write_bytes(html)
            mark_statute_written(conn, str(out_path))
            written += 1

            if written % 50 == 0:
                log.info("  %s: %d/%d sections written", group_name, written, total)

        log.info(
            "CA %s complete: %d written, %d gaps skipped", group_name, written, skipped_gaps
        )


def _resolve_latest_uscode_release(session: requests.Session) -> str:
    fallback = "us/pl/119/83"
    try:
        resp = session.get(USCODE_CONTENTS_URL, timeout=15)
        resp.raise_for_status()
        match = re.search(r"xml_usc28@(\d+-\d+)\.zip", resp.text)
        if match:
            # Build path segment from "119-83" → "us/pl/119/83"
            parts = match.group(1).split("-")
            if len(parts) == 2:
                release = f"us/pl/{parts[0]}/{parts[1]}"
                log.info("Resolved US Code release point: %s", release)
                return release
    except Exception:
        log.warning("Could not resolve US Code release point — using fallback %s", fallback)
    return fallback


def _build_uscode_urls(release: str) -> dict[str, str]:
    base = f"https://uscode.house.gov/download/releasepoints/{release}"
    rp = release.replace("us/pl/", "").replace("/", "-")
    return {
        "usc28": f"{base}/xml_usc28@{rp}.zip",
        "usc29": f"{base}/xml_usc29@{rp}.zip",
        "usc42": f"{base}/xml_usc42@{rp}.zip",
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _stream_download(session: requests.Session, url: str, out_path: Path) -> None:
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
            f.flush()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_ca_section(session: requests.Session, law_code: str, section_num: int) -> bytes:
    params = {"sectionNum": f"{section_num}.", "lawCode": law_code}
    resp = session.get(CA_BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.content
