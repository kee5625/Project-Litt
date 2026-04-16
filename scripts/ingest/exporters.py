from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import IngestionOutputs, NormalizedRecord
from .utils import total_file_size_bytes, utc_now_iso, write_csv


CSV_FIELDS = [
    "source",
    "source_id",
    "case_name",
    "citation",
    "court",
    "jurisdiction",
    "practice_area",
    "year",
    "published_status",
    "precedential_status",
    "source_url",
    "plain_text",
    "quality_score",
    "quality_flags",
    "dedupe_hash",
]


def record_to_row(record: NormalizedRecord) -> dict[str, Any]:
    return {
        "source": record.source,
        "source_id": record.source_id,
        "case_name": record.case_name,
        "citation": record.citation,
        "court": record.court,
        "jurisdiction": record.jurisdiction,
        "practice_area": record.practice_area,
        "year": record.year,
        "published_status": record.published_status,
        "precedential_status": record.precedential_status,
        "source_url": record.source_url,
        "plain_text": record.plain_text,
        "quality_score": f"{record.quality_score:.4f}",
        "quality_flags": ";".join(record.quality_flags),
        "dedupe_hash": record.dedupe_hash,
    }


def write_outputs(
    output_dir: Path,
    prefix: str,
    outputs: IngestionOutputs,
) -> tuple[Path, Path]:
    accepted_path = output_dir / f"{prefix}_accepted.csv"
    quarantine_path = output_dir / f"{prefix}_quarantine.csv"

    write_csv(accepted_path, [record_to_row(r) for r in outputs.accepted], CSV_FIELDS)
    write_csv(
        quarantine_path, [record_to_row(r) for r in outputs.quarantined], CSV_FIELDS
    )
    return accepted_path, quarantine_path


def write_manifest(
    manifest_path: Path,
    payload: dict[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


def build_feasibility_manifest(
    runtime_seconds: float,
    output_root: Path,
    counters: dict[str, Any],
) -> dict[str, Any]:
    output_bytes = total_file_size_bytes(output_root)
    return {
        "generated_at": utc_now_iso(),
        "runtime_seconds": round(runtime_seconds, 3),
        "runtime_minutes": round(runtime_seconds / 60.0, 3),
        "output_bytes": output_bytes,
        "output_mb": round(output_bytes / (1024 * 1024), 3),
        "within_runtime_budget_60_min": runtime_seconds <= 3600,
        "within_storage_budget_15_gb": output_bytes <= 15 * 1024 * 1024 * 1024,
        "counters": counters,
    }
