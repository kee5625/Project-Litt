from __future__ import annotations

import concurrent.futures
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests

from .config import CourtListenerConfig
from .models import NormalizedRecord
from .utils import (
    classify_practice_area,
    extract_citation,
    extract_first_non_empty,
    normalize_whitespace,
    safe_int,
    stable_json_dump,
    timestamp_slug,
)


def _headers(cfg: CourtListenerConfig) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if cfg.api_token:
        headers["Authorization"] = f"Token {cfg.api_token}"
    return headers


def _request_json(
    url: str, params: dict[str, Any], cfg: CourtListenerConfig
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(cfg.retry_count + 1):
        try:
            response = requests.get(
                url, params=params, headers=_headers(cfg), timeout=cfg.timeout_seconds
            )
            if response.status_code == 429:
                sleep_seconds = min(2**attempt, 20)
                time.sleep(sleep_seconds)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # requests exposes many exception subclasses
            last_error = exc
            time.sleep(min(2**attempt, 10))
    if last_error:
        raise last_error
    raise RuntimeError("CourtListener request failed with unknown error")


# Rate limiter for concurrent page fetches (8 concurrent requests max)
_rate_limiter = threading.Semaphore(8)


def _fetch_page(
    base_url: str,
    page: int,
    page_size: int,
    cfg: CourtListenerConfig,
    run_slug: str,
    raw_dir: Path,
) -> tuple[int, list[dict[str, Any]]]:
    """Fetch a single page with rate limiting. Returns (page_num, results)."""
    with _rate_limiter:
        try:
            params = {"page": page, "page_size": page_size}
            payload = _request_json(base_url, params=params, cfg=cfg)
            results = payload.get("results", [])
            if not isinstance(results, list):
                return page, []

            raw_out = raw_dir / f"courtlistener_rest_{run_slug}_p{page:03d}.json"
            stable_json_dump(raw_out, payload)
            return page, results
        except Exception:
            return page, []


def fetch_courtlistener_rest(
    cfg: CourtListenerConfig,
    raw_dir: Path,
    max_records: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    if not cfg.rest_enabled:
        return []

    opinions: list[dict[str, Any]] = []
    base_url = cfg.base_url.rstrip("/") + "/opinions/"
    run_slug = timestamp_slug()

    if dry_run:
        # Single request for dry run
        try:
            params = {"page": 1, "page_size": cfg.page_size}
            payload = _request_json(base_url, params=params, cfg=cfg)
            results = payload.get("results", [])
            if isinstance(results, list):
                raw_out = raw_dir / f"courtlistener_rest_{run_slug}_p001.json"
                stable_json_dump(raw_out, payload)
                return results[:max_records]
        except Exception:
            pass
        return []

    # Parallel page fetching with rate limiter
    opinions_dict: dict[int, list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                _fetch_page, base_url, page, cfg.page_size, cfg, run_slug, raw_dir
            ): page
            for page in range(1, cfg.max_pages + 1)
        }

        for future in concurrent.futures.as_completed(futures):
            if len(opinions) >= max_records:
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

            page, results = future.result()
            opinions_dict[page] = results
            opinions.extend(results)

    return opinions[:max_records]


def _load_bulk_files(bulk_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not bulk_path.exists():
        return records

    for file_path in sorted(bulk_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                records.extend([p for p in payload if isinstance(p, dict)])
            elif isinstance(payload, dict):
                if isinstance(payload.get("results"), list):
                    records.extend(
                        [p for p in payload["results"] if isinstance(p, dict)]
                    )
                else:
                    records.append(payload)
        elif file_path.suffix.lower() in {".jsonl", ".ndjson"}:
            for line in file_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        records.append(payload)
                except json.JSONDecodeError:
                    continue
    return records


def fetch_courtlistener_bulk(
    cfg: CourtListenerConfig, max_records: int
) -> list[dict[str, Any]]:
    if not cfg.bulk_enabled:
        return []
    return _load_bulk_files(cfg.bulk_path)[:max_records]


def _extract_citation_cl(opinion: dict[str, Any]) -> str:
    """Extract citation using CourtListener-specific fields first, then unified patterns."""
    citations = opinion.get("citations")
    if isinstance(citations, list) and citations:
        first = citations[0]
        if isinstance(first, dict):
            cite = first.get("cite")
            if cite:
                return cite
        if first:
            return str(first)

    # Use unified extraction from utils (tries HTML then reporter patterns)
    plain_text = extract_first_non_empty([opinion.get("plain_text")])
    html = extract_first_non_empty([opinion.get("html_with_citations")])
    return extract_citation(plain_text, html)


def _extract_year(opinion: dict[str, Any]) -> int:
    date_value = extract_first_non_empty(
        [
            opinion.get("date_filed"),
            opinion.get("date_created"),
            opinion.get("date_modified"),
        ]
    )
    if len(date_value) >= 4 and date_value[:4].isdigit():
        return safe_int(date_value[:4], 0)
    return _extract_year_from_text(_extract_text(opinion))


def _extract_text(opinion: dict[str, Any]) -> str:
    return normalize_whitespace(
        extract_first_non_empty(
            [
                opinion.get("plain_text"),
                opinion.get("html_with_citations"),
                opinion.get("html"),
                opinion.get("text"),
            ]
        )
    )


def _extract_case_name(opinion: dict[str, Any], text: str) -> str:
    explicit = extract_first_non_empty(
        [
            opinion.get("case_name"),
            opinion.get("case_name_full"),
            opinion.get("caseName"),
            opinion.get("name"),
        ]
    )
    if explicit:
        return explicit

    raw_text = opinion.get("html_with_citations") or text
    cite_match = re.search(r"\[Cite as\s+([^,\]]+)", raw_text)
    if cite_match:
        return normalize_whitespace(cite_match.group(1))

    lines = [normalize_whitespace(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    for index, line in enumerate(lines[:40]):
        if line.lower() in {"v.", "v", "vs.", "vs"}:
            plaintiff = ""
            defendant = ""
            for left in reversed(lines[:index]):
                if left and not re.fullmatch(
                    r"(?i)(plaintiff|appellant|petitioner|debtor|respondent|defendant|appellee|claimant|trustee|creditor|petitioner-appellant|plaintiff-appellant|defendant-appellee)",
                    left,
                ):
                    plaintiff = left.strip(" ,;:()")
                    break
            for right in lines[index + 1 : index + 8]:
                if right and not re.fullmatch(
                    r"(?i)(plaintiff|appellant|petitioner|debtor|respondent|defendant|appellee|claimant|trustee|creditor|petitioner-appellant|plaintiff-appellant|defendant-appellee)",
                    right,
                ):
                    defendant = right.strip(" ,;:()")
                    break
            if plaintiff and defendant:
                return f"{plaintiff} v. {defendant}"

    in_re_match = re.search(r"\bIn re\s+([^\n\r]+)", text, flags=re.IGNORECASE)
    if in_re_match:
        return normalize_whitespace(f"In re {in_re_match.group(1)}")

    return ""


def _extract_court_from_text(text: str) -> str:
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    for line in lines[:40]:
        match = re.match(
            r"(?i)^(?:in the\s+)?(.+?(?:court|tribunal|panel).*?)(?:\s{2,}|$)", line
        )
        if match:
            court = normalize_whitespace(match.group(1))
            if court:
                return court
    header_match = re.search(r"(?i)\b(?:in the|court of)\b[^\n\r]{0,120}", text)
    if header_match:
        return normalize_whitespace(header_match.group(0))
    return ""


def _extract_year_from_text(text: str) -> int:
    header_text = text[:2000]
    for pattern in (
        r"(?i)(?:filed|dated|date filed|date created|date modified)[:\s]+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
        r"(?i)\b(?:filed|dated)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    ):
        match = re.search(pattern, header_text)
        if match:
            year_match = re.search(r"(19|20)\d{2}", match.group(1))
            if year_match:
                return safe_int(year_match.group(0), 0)

    year_match = re.search(r"\b(19|20)\d{2}\b", header_text)
    if year_match:
        return safe_int(year_match.group(0), 0)
    return 0


def _infer_jurisdiction(court_value: str) -> str:
    c = court_value.lower()
    ca_patterns = ["calif", "ca app", "cal.", "los angeles", "san francisco", "sacramento"]
    federal_patterns = ["circuit", "district", "supreme court", "scotus", "u.s. court", "federal"]
    if any(p in c for p in ca_patterns):
        return "CA"
    if any(p in c for p in federal_patterns):
        return "federal"
    return "state"


def normalize_courtlistener_record(opinion: dict[str, Any]) -> NormalizedRecord:
    text = _extract_text(opinion)
    court_value = extract_first_non_empty(
        [
            opinion.get("court"),
            opinion.get("court_id"),
            _extract_court_from_text(text),
        ]
    )
    docket = extract_first_non_empty([opinion.get("docket")])

    return NormalizedRecord(
        source="courtlistener",
        source_id=extract_first_non_empty(
            [opinion.get("id"), opinion.get("cluster_id")]
        ),
        case_name=_extract_case_name(opinion, text),
        citation=_extract_citation_cl(opinion),
        court=court_value,
        jurisdiction=_infer_jurisdiction(court_value),
        practice_area=classify_practice_area(text, docket),
        year=_extract_year(opinion),
        published_status=extract_first_non_empty(
            [
                opinion.get("status"),
                opinion.get("publication_status"),
                opinion.get("status_text"),
            ]
        ),
        precedential_status=extract_first_non_empty(
            [
                opinion.get("precedential_status"),
                opinion.get("precedential"),
            ]
        ),
        plain_text=text,
        source_url=extract_first_non_empty(
            [
                opinion.get("absolute_url"),
                opinion.get("download_url"),
                opinion.get("resource_uri"),
            ]
        ),
        metadata={
            "docket": docket,
            "judges": opinion.get("judges"),
            "panel": opinion.get("panel"),
            "raw": opinion,
        },
    )
