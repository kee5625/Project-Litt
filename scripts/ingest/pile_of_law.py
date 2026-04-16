from __future__ import annotations

import json
import lzma
import random
import re
from pathlib import Path
from typing import Any

from .config import PileOfLawConfig
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


def _iter_dataset(cfg: PileOfLawConfig):
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies: install 'pandas' and 'huggingface-hub' for pile-of-law ingestion"
        ) from exc

    repo_id = cfg.dataset_name
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = [
        file_name for file_name in files if file_name.lower().endswith(".parquet")
    ]
    split_prefix = f"data/{cfg.dataset_split}."
    jsonl_xz_files = [
        file_name
        for file_name in files
        if file_name.startswith(split_prefix)
        and file_name.lower().endswith(".jsonl.xz")
    ]
    if not parquet_files and not jsonl_xz_files:
        raise RuntimeError(
            f"No parquet or {cfg.dataset_split}.jsonl.xz files found for dataset {repo_id}"
        )

    # Prefer deterministic order but sample file list for diversity when many shards exist.
    random.seed(42)
    candidate_files = parquet_files if parquet_files else jsonl_xz_files
    priority_tokens = [
        "courtlisteneropinions",
        "state_code",
        "uscode",
        "federal_register",
        "scotus",
    ]
    prioritized = [
        name
        for name in candidate_files
        if any(token in name for token in priority_tokens)
    ]
    remaining = [name for name in candidate_files if name not in prioritized]
    ordered_candidates = sorted(prioritized) + sorted(remaining)
    if len(ordered_candidates) > 40:
        head = ordered_candidates[:20]
        tail_pool = ordered_candidates[20:]
        selected_files = head + sorted(random.sample(tail_pool, 20))
    else:
        selected_files = ordered_candidates

    rows: list[dict[str, Any]] = []
    max_rows = max((cfg.curated_target + cfg.diverse_target) * 3, 1500)

    for file_name in selected_files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=file_name,
        )
        if file_name.lower().endswith(".parquet"):
            frame = pd.read_parquet(local_path)
            frame_rows = frame.to_dict(orient="records")
            rows.extend([{str(k): v for k, v in row.items()} for row in frame_rows])
        else:
            with lzma.open(local_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        rows.append({str(k): v for k, v in payload.items()})
                    if len(rows) >= max_rows:
                        break
        if len(rows) >= max_rows:
            break

    return rows


def _extract_court_from_text(text: str) -> str:
    """Extract court name from text (shared with courtlistener.py)."""
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


def _extract_case_name_from_text(text: str) -> str:
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    lines = [line for line in lines if line]

    for line in lines[:40]:
        in_re = re.search(r"\bIn re\s+(.+)", line, flags=re.IGNORECASE)
        if in_re:
            return normalize_whitespace(f"In re {in_re.group(1)}")

    for index, line in enumerate(lines[:50]):
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

    for line in lines[:40]:
        if re.search(r"(?i)\bv\.\b", line):
            cleaned = normalize_whitespace(re.sub(r"\s+", " ", line))
            cleaned = re.sub(
                r"(?i)\b(plaintiff|appellant|petitioner|debtor|respondent|defendant|appellee|claimant|trustee|creditor)\b.*$",
                "",
                cleaned,
            ).strip()
            if " v. " in cleaned and len(cleaned) > 6:
                return cleaned

    return ""


def _extract_year_from_text(text: str) -> int:
    header_text = text[:2000]
    for pattern in (
        r"(?i)(?:filed|dated|downloaded)[:\s]+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
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


def _normalize(record: dict[str, Any]) -> NormalizedRecord:
    text = normalize_whitespace(
        extract_first_non_empty(
            [
                record.get("text"),
                record.get("opinion_text"),
                record.get("plain_text"),
                record.get("body"),
            ]
        )
    )
    case_name = extract_first_non_empty(
        [
            record.get("case_name"),
            record.get("name"),
            record.get("title"),
            _extract_case_name_from_text(text),
        ]
    )
    citation = extract_first_non_empty(
        [
            record.get("citation"),
            record.get("cite"),
        ]
    )
    if not citation:
        # Use unified citation extraction
        citation = extract_citation(text)
    court = extract_first_non_empty(
        [
            record.get("court"),
            record.get("court_id"),
            _extract_court_from_text(text),
        ]
    )
    year = safe_int(
        extract_first_non_empty(
            [
                record.get("year"),
                record.get("decision_year"),
            ]
        ),
        0,
    ) or _extract_year_from_text(text)

    # Improved jurisdiction inference
    c = court.lower()
    ca_patterns = ["calif", "ca app", "cal.", "los angeles", "san francisco", "sacramento"]
    federal_patterns = ["circuit", "district", "supreme court", "scotus", "u.s. court", "federal"]
    if any(p in c for p in ca_patterns):
        jurisdiction = "CA"
    elif any(p in c for p in federal_patterns):
        jurisdiction = "federal"
    else:
        jurisdiction = "state"

    return NormalizedRecord(
        source="pile_of_law",
        source_id=extract_first_non_empty(
            [
                record.get("id"),
                record.get("opinion_id"),
                record.get("uuid"),
            ]
        ),
        case_name=case_name,
        citation=citation,
        court=court,
        jurisdiction=jurisdiction,
        practice_area=classify_practice_area(f"{case_name} {text}"),
        year=year,
        published_status=extract_first_non_empty(
            [record.get("status"), record.get("publication_status")]
        ),
        precedential_status=extract_first_non_empty(
            [record.get("precedential_status")]
        ),
        plain_text=text,
        source_url=extract_first_non_empty(
            [record.get("url"), record.get("source_url")]
        ),
        metadata={"raw": record},
    )


def fetch_pile_of_law(
    cfg: PileOfLawConfig,
    raw_dir: Path,
    dry_run: bool,
) -> tuple[list[NormalizedRecord], list[NormalizedRecord]]:
    dataset_rows = _iter_dataset(cfg)
    run_slug = timestamp_slug()

    curated: list[NormalizedRecord] = []
    diverse: list[NormalizedRecord] = []

    # Keep only enough records to hit targets plus a buffer for quality filtering later.
    target_total = cfg.curated_target + cfg.diverse_target
    buffer_limit = max(target_total * 3, 1000)

    sampled_raw: list[dict[str, Any]] = []
    CURATED_AREAS = {"employment", "family", "ip", "criminal", "contracts"}

    for idx, item in enumerate(dataset_rows):
        if idx >= buffer_limit:
            break
        if not isinstance(item, dict):
            continue

        sampled_raw.append(item)
        norm = _normalize(item)

        # Curated: matches one of the 5 practice areas; Diverse: everything else (or overflow)
        if norm.practice_area in CURATED_AREAS and len(curated) < cfg.curated_target:
            curated.append(norm)
        elif len(diverse) < cfg.diverse_target:
            diverse.append(norm)

        if dry_run and (curated or diverse):
            break

        if len(curated) >= cfg.curated_target and len(diverse) >= cfg.diverse_target:
            break

    stable_json_dump(raw_dir / f"pile_of_law_sample_{run_slug}.json", sampled_raw)
    return curated, diverse
