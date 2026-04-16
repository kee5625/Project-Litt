from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def timestamp_slug() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def stable_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def total_file_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for file_path in root.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def extract_first_non_empty(values: list[Any], default: str = "") -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value:
            return str(value)
    return default


def classify_practice_area(text: str, docket: str = "") -> str:
    """Unified weighted keyword classifier. Returns one of 5 areas or 'general'."""
    WEIGHTED_KEYWORDS = {
        "employment": [
            ("wrongful termination", 4), ("title vii", 4), ("ada", 3),
            ("discrimination", 3), ("harassment", 3), ("retaliation", 3),
            ("wage theft", 3), ("overtime", 2), ("flsa", 2), ("fmla", 2),
            ("workers compensation", 2), ("wage", 1), ("hour", 1),
        ],
        "family": [
            ("child custody", 4), ("divorce", 3), ("domestic violence", 3),
            ("child support", 3), ("alimony", 2), ("custody", 2), ("adoption", 2),
        ],
        "ip": [
            ("patent infringement", 4), ("trademark infringement", 4),
            ("copyright infringement", 4), ("trade secret", 3),
            ("patent", 2), ("trademark", 2), ("copyright", 2), ("infringement", 1),
        ],
        "criminal": [
            ("fourth amendment", 3), ("criminal conviction", 3),
            ("sentencing guidelines", 3), ("indictment", 2), ("guilty plea", 2),
            ("conviction", 2), ("sentencing", 2), ("plea", 1), ("dui", 1),
        ],
        "contracts": [
            ("breach of contract", 4), ("promissory estoppel", 4),
            ("indemnification clause", 3), ("arbitration clause", 3),
            ("contractual", 2), ("indemnity", 2), ("arbitration", 2),
            ("breach", 1), ("consideration", 1),
        ],
    }
    corpus = f"{text} {docket}".lower()
    scores = {
        area: sum(corpus.count(kw) * w for kw, w in kws)
        for area, kws in WEIGHTED_KEYWORDS.items()
    }
    best_area, best_score = max(scores.items(), key=lambda x: x[1])
    return best_area if best_score > 0 else "general"


def extract_citation(text: str, html: str = "") -> str:
    """Unified citation extraction. Tries HTML spans then reporter patterns."""
    # HTML span extraction
    if html:
        m = re.search(
            r'<span class="citation[^"]*"[^>]*>\s*<a [^>]*>([^<]+)</a>', html
        )
        if m:
            return m.group(1).strip()

    # Reporter patterns (ordered most specific to least)
    REPORTER_PATTERNS = [
        r"\b\d+\s+U\.S\.\s+\d+\b",  # 477 U.S. 242
        r"\b\d+\s+S\.\s*Ct\.\s+\d+\b",  # 127 S. Ct. 1955
        r"\b\d+\s+F\.[234]d\s+\d+\b",  # 123 F.3d 456
        r"\b\d+\s+F\.\s*(?:Supp|App)\.\s*(?:\d+d\s+)?\d+\b",  # 123 F. Supp. 3d 789
        r"\b\d+\s+Cal(?:\.\s*App)?\.(?:\s*[234]d)?\s+\d+\b",  # Cal.App.3d
        r"\b\d+\s+P\.[234]d\s+\d+\b",  # Pacific Reporter
        r"\b\d+\s+S\.W\.[234]d\s+\d+\b",  # South Western Reporter
        r"\b\d+\s+N\.E\.[234]d\s+\d+\b",  # North Eastern Reporter
        r"\b\d+\s+[A-Z][A-Za-z\.]+\s+\d+\b",  # fallback generic reporter
    ]

    for pattern in REPORTER_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(0).strip()

    return ""
