from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedRecord:
    source: str
    source_id: str
    case_name: str
    citation: str
    court: str
    jurisdiction: str
    practice_area: str
    year: int
    published_status: str
    precedential_status: str
    plain_text: str
    source_url: str
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    quality_flags: list[str] = field(default_factory=list)
    dedupe_hash: str = ""


@dataclass
class IngestionCounters:
    fetched: int = 0
    normalized: int = 0
    accepted: int = 0
    quarantined: int = 0
    rejected: int = 0


@dataclass
class IngestionOutputs:
    accepted: list[NormalizedRecord] = field(default_factory=list)
    quarantined: list[NormalizedRecord] = field(default_factory=list)
    rejected_reasons: dict[str, int] = field(default_factory=dict)
