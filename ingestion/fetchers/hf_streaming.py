import json
import logging
import lzma
import sqlite3
from pathlib import Path

from huggingface_hub import hf_hub_url
from ingestion.utils.state import get_pol_doc_count, mark_pol_doc_written
from ingestion.utils.practice_area import PRACTICE_AREA_KEYWORDS, classify_practice_area

log = logging.getLogger(__name__)

POL_OUTPUT_DIR = Path("raw/pol")

# Keys are the file-name stem used in pile-of-law (train.{key}.*.jsonl.xz).
# Values are target doc counts after keyword filtering.
HF_SUBSETS: dict[str, int] = {
    "courtlisteneropinions": 2000,
    "us_bills": 200,
    "nlrb_decisions": 300,
}


def stream_pile_of_law(conn: sqlite3.Connection, practice_keywords: dict) -> None:
    POL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subset, target in HF_SUBSETS.items():
        already_written = get_pol_doc_count(conn, subset)
        if already_written >= target:
            log.info("Subset %s already complete (%d docs) — skipping.", subset, already_written)
            continue

        log.info("Streaming subset: %s (target: %d, already have: %d)", subset, target, already_written)

        try:
            _stream_subset(conn, subset, target, practice_keywords)
        except Exception:
            log.exception("Failed streaming subset %s — continuing to next.", subset)


def _shard_urls(subset: str) -> list[str]:
    from huggingface_hub import list_repo_files

    prefix = f"data/train.{subset}."
    files = [
        f for f in list_repo_files("pile-of-law/pile-of-law", repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".jsonl.xz")
    ]
    return [
        hf_hub_url("pile-of-law/pile-of-law", filename=f, repo_type="dataset")
        for f in sorted(files)
    ]


def _iter_shards(urls: list[str]):
    import requests

    for url in urls:
        log.info("    downloading shard: %s", url.split("/")[-1])
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        with lzma.open(__import__("io").BytesIO(resp.content)) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _stream_subset(
    conn: sqlite3.Connection,
    subset: str,
    target: int,
    practice_keywords: dict,
) -> None:
    out_path = POL_OUTPUT_DIR / f"{subset}.jsonl"

    written = 0
    docs_seen = 0

    urls = _shard_urls(subset)
    if not urls:
        log.warning("  %s: no shard files found in repo — skipping.", subset)
        return
    log.info("  %s: found %d shard(s)", subset, len(urls))

    with open(out_path, "w") as fh:
        for doc in _iter_shards(urls):
            docs_seen += 1
            search_text = (
                (doc.get("text") or "") + " " + str(doc.get("meta") or "")
            ).lower()

            matched = classify_practice_area(search_text)
            if not matched:
                continue

            doc["practice_area_matches"] = matched
            fh.write(json.dumps(doc) + "\n")
            fh.flush()
            mark_pol_doc_written(conn, subset, written)
            written += 1

            if written % 100 == 0:
                log.info("  %s: %d/%d docs written (seen %d)", subset, written, target, docs_seen)

            if written >= target:
                break

    log.info(
        "Subset %s done: %d docs written from %d seen (%.1f%% match rate)",
        subset, written, docs_seen, 100 * written / max(docs_seen, 1),
    )


def _classify_practice_area(text: str, keywords: dict) -> list[str]:
    matched = []
    for area, kws in keywords.items():
        if any(kw in text for kw in kws):
            matched.append(area)
    return matched
