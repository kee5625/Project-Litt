from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceConfig:
    enabled: bool
    target_count: int


@dataclass
class CourtListenerConfig:
    base_url: str
    api_token: str
    page_size: int
    max_pages: int
    timeout_seconds: int
    retry_count: int
    rest_enabled: bool
    bulk_enabled: bool
    bulk_path: Path


@dataclass
class PileOfLawConfig:
    dataset_name: str
    dataset_split: str
    curated_target: int
    diverse_target: int
    streaming: bool


@dataclass
class QualityConfig:
    require_plain_text: bool
    strict_citation: bool
    quarantine_missing_citation: bool
    dedupe_enabled: bool
    min_text_chars: int
    adaptive_recency: bool
    max_domain_share: float


@dataclass
class OutputConfig:
    root: Path
    raw_courtlistener_dir: Path
    raw_pile_of_law_dir: Path
    processed_dir: Path
    manifests_dir: Path


@dataclass
class AppConfig:
    courtlistener: CourtListenerConfig
    pile_of_law: PileOfLawConfig
    quality: QualityConfig
    output: OutputConfig
    target_courtlistener_high_value: int


def _to_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_config(project_root: Path) -> AppConfig:
    load_env_file(project_root / "scripts" / ".env")
    load_env_file(project_root / "scripts" / ".env.example")

    output_root = project_root / "scripts" / "output"
    return AppConfig(
        courtlistener=CourtListenerConfig(
            base_url=os.getenv(
                "COURTLISTENER_BASE_URL", "https://www.courtlistener.com/api/rest/v4"
            ),
            api_token=os.getenv("COURTLISTENER_API_TOKEN", ""),
            page_size=int(os.getenv("COURTLISTENER_PAGE_SIZE", "100")),
            max_pages=int(os.getenv("COURTLISTENER_MAX_PAGES", "120")),
            timeout_seconds=int(os.getenv("COURTLISTENER_TIMEOUT_SECONDS", "45")),
            retry_count=int(os.getenv("COURTLISTENER_RETRY_COUNT", "4")),
            rest_enabled=_to_bool(os.getenv("COURTLISTENER_REST_ENABLED"), True),
            bulk_enabled=_to_bool(os.getenv("COURTLISTENER_BULK_ENABLED"), True),
            bulk_path=Path(
                os.getenv(
                    "COURTLISTENER_BULK_PATH",
                    str(output_root / "raw" / "courtlistener" / "bulk"),
                )
            ),
        ),
        pile_of_law=PileOfLawConfig(
            dataset_name=os.getenv(
                "PILE_OF_LAW_DATASET_NAME", "pile-of-law/pile-of-law"
            ),
            dataset_split=os.getenv("PILE_OF_LAW_DATASET_SPLIT", "train"),
            curated_target=int(os.getenv("PILE_OF_LAW_CURATED_TARGET", "300")),
            diverse_target=int(os.getenv("PILE_OF_LAW_DIVERSE_TARGET", "250")),
            streaming=_to_bool(os.getenv("PILE_OF_LAW_STREAMING"), True),
        ),
        quality=QualityConfig(
            require_plain_text=_to_bool(os.getenv("QUALITY_REQUIRE_PLAIN_TEXT"), True),
            strict_citation=_to_bool(os.getenv("QUALITY_STRICT_CITATION"), True),
            quarantine_missing_citation=_to_bool(
                os.getenv("QUALITY_QUARANTINE_MISSING_CITATION"), True
            ),
            dedupe_enabled=_to_bool(os.getenv("QUALITY_DEDUPE_ENABLED"), True),
            min_text_chars=int(os.getenv("QUALITY_MIN_TEXT_CHARS", "2000")),
            adaptive_recency=_to_bool(os.getenv("QUALITY_ADAPTIVE_RECENCY"), True),
            max_domain_share=float(os.getenv("QUALITY_MAX_DOMAIN_SHARE", "0.40")),
        ),
        output=OutputConfig(
            root=output_root,
            raw_courtlistener_dir=output_root / "raw" / "courtlistener",
            raw_pile_of_law_dir=output_root / "raw" / "pile_of_law",
            processed_dir=output_root / "processed",
            manifests_dir=output_root / "manifests",
        ),
        target_courtlistener_high_value=int(
            os.getenv("TARGET_COURTLISTENER_HIGH_VALUE", "500")
        ),
    )
