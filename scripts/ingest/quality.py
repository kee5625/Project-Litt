from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Iterable

from .config import QualityConfig
from .models import IngestionOutputs, NormalizedRecord
from .utils import normalize_whitespace, sha1_text


def _is_precedential(value: str) -> bool:
    lc = (value or "").lower()
    return (
        any(token in lc for token in ["precedential", "published", "for publication"])
        and "non" not in lc
    )


def _base_quality_score(record: NormalizedRecord) -> float:
    score = 0.0
    if record.citation:
        score += 3.0
    if _is_precedential(record.precedential_status) or _is_precedential(
        record.published_status
    ):
        score += 2.0
    if len(record.plain_text) >= 5000:
        score += 2.0
    elif len(record.plain_text) >= 2500:
        score += 1.0
    if record.year >= 2020:
        score += 2.0
    elif record.year >= 2015:
        score += 1.5
    elif record.year >= 2010:
        score += 1.0
    elif record.year >= 2000:
        score += 0.5
    if record.case_name:
        score += 0.5
    if record.court:
        score += 0.5
    return score


def _adaptive_recency_boost(records: list[NormalizedRecord]) -> dict[str, float]:
    per_domain_years: dict[str, list[int]] = defaultdict(list)
    for record in records:
        if record.year > 0:
            per_domain_years[record.practice_area].append(record.year)

    boosts: dict[str, float] = {}
    for domain, years in per_domain_years.items():
        if not years:
            boosts[domain] = 0.0
            continue
        median_year = sorted(years)[len(years) // 2]
        if median_year >= 2018:
            boosts[domain] = 0.0
        elif median_year >= 2010:
            boosts[domain] = 0.5
        else:
            boosts[domain] = 1.0
    return boosts


def _dedupe_hash(record: NormalizedRecord) -> str:
    fingerprint = "|".join(
        [
            normalize_whitespace(record.case_name.lower()),
            normalize_whitespace(record.citation.lower()),
            str(record.year),
            normalize_whitespace(record.plain_text[:2500].lower()),
        ]
    )
    return sha1_text(fingerprint)


def _near_duplicate(a: NormalizedRecord, b: NormalizedRecord) -> bool:
    if a.citation and b.citation and a.citation.lower() == b.citation.lower():
        return True
    if a.case_name and b.case_name and a.year == b.year:
        ratio = SequenceMatcher(None, a.case_name.lower(), b.case_name.lower()).ratio()
        if ratio >= 0.95:
            return True
    return False


def _apply_soft_balancing(
    records: list[NormalizedRecord], max_domain_share: float
) -> list[NormalizedRecord]:
    """Enforce domain balance: hard cap per domain, then fill remaining slots from overflow."""
    if not records:
        return []

    # Total target is the actual input size (enforce cap)
    total_target = len(records)
    max_per_domain = max(1, int(total_target * max_domain_share))

    selected: list[NormalizedRecord] = []
    overflow: list[NormalizedRecord] = []
    domain_counts: Counter[str] = Counter()

    for record in records:
        if domain_counts[record.practice_area] < max_per_domain:
            selected.append(record)
            domain_counts[record.practice_area] += 1
        else:
            overflow.append(record)

    # Fill remaining slots from overflow (best scores first)
    remaining_slots = total_target - len(selected)
    if remaining_slots > 0:
        overflow.sort(key=lambda r: r.quality_score, reverse=True)
        selected.extend(overflow[:remaining_slots])

    selected.sort(key=lambda r: r.quality_score, reverse=True)
    return selected


def score_and_filter(
    records: Iterable[NormalizedRecord], cfg: QualityConfig
) -> IngestionOutputs:
    outputs = IngestionOutputs()
    accepted_candidates: list[NormalizedRecord] = []
    seen_hashes: set[str] = set()
    seen_citations: set[str] = set()
    seen_case_names: set[str] = set()

    records_list = list(records)
    domain_boosts = (
        _adaptive_recency_boost(records_list) if cfg.adaptive_recency else {}
    )

    for record in records_list:
        record.quality_flags = []
        text_len = len(record.plain_text)

        if cfg.require_plain_text and not record.plain_text:
            outputs.rejected_reasons["missing_text"] = (
                outputs.rejected_reasons.get("missing_text", 0) + 1
            )
            continue
        if text_len < cfg.min_text_chars:
            outputs.rejected_reasons["short_text"] = (
                outputs.rejected_reasons.get("short_text", 0) + 1
            )
            continue

        if cfg.strict_citation and not record.citation:
            if cfg.quarantine_missing_citation:
                record.quality_flags.append("missing_citation")
                outputs.quarantined.append(record)
                continue
            outputs.rejected_reasons["missing_citation"] = (
                outputs.rejected_reasons.get("missing_citation", 0) + 1
            )
            continue

        record.quality_score = _base_quality_score(record) + domain_boosts.get(
            record.practice_area, 0.0
        )

        record.dedupe_hash = _dedupe_hash(record)
        if cfg.dedupe_enabled and record.dedupe_hash in seen_hashes:
            outputs.rejected_reasons["exact_duplicate"] = (
                outputs.rejected_reasons.get("exact_duplicate", 0) + 1
            )
            continue

        # Check near-duplicates against full set (not just 150-record window)
        duplicate_found = False
        if cfg.dedupe_enabled:
            if record.citation and record.citation.lower() in seen_citations:
                duplicate_found = True
            elif record.case_name:
                norm_name = record.case_name.lower().strip()
                if norm_name in seen_case_names and record.year > 0:
                    # Case name match + same year = near duplicate
                    for existing in accepted_candidates:
                        if (existing.case_name.lower().strip() == norm_name
                            and existing.year == record.year):
                            duplicate_found = True
                            break

            if duplicate_found:
                outputs.rejected_reasons["near_duplicate"] = (
                    outputs.rejected_reasons.get("near_duplicate", 0) + 1
                )
                continue

        # Record accepted
        seen_hashes.add(record.dedupe_hash)
        if record.citation:
            seen_citations.add(record.citation.lower())
        if record.case_name:
            seen_case_names.add(record.case_name.lower().strip())
        accepted_candidates.append(record)

    accepted_candidates.sort(key=lambda r: r.quality_score, reverse=True)
    outputs.accepted = _apply_soft_balancing(accepted_candidates, cfg.max_domain_share)
    return outputs
