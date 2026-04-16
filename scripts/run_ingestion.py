from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import time
from pathlib import Path

from ingest.config import load_config
from ingest.courtlistener import (
    fetch_courtlistener_bulk,
    fetch_courtlistener_rest,
    normalize_courtlistener_record,
)
from ingest.exporters import build_feasibility_manifest, write_manifest, write_outputs
from ingest.pile_of_law import fetch_pile_of_law
from ingest.quality import score_and_filter
from ingest.utils import ensure_dirs, timestamp_slug


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch legal datasets and export quality-filtered CSV outputs."
    )
    parser.add_argument("--project-root", default=".", help="Project root path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate sources and run minimal sample"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["courtlistener", "pile-of-law", "all"],
        default=["all"],
        help="Sources to ingest",
    )
    return parser


def _run_courtlistener(config, run_slug: str, dry_run: bool) -> tuple[list, dict]:
    """Fetch, normalize, and score CourtListener data."""
    try:
        raw_rest = fetch_courtlistener_rest(
            cfg=config.courtlistener,
            raw_dir=config.output.raw_courtlistener_dir,
            max_records=config.target_courtlistener_high_value * 3,
            dry_run=dry_run,
        )
        if (
            len(raw_rest) < config.target_courtlistener_high_value
            and config.courtlistener.bulk_enabled
        ):
            needed = config.target_courtlistener_high_value * 3 - len(raw_rest)
            raw_bulk = fetch_courtlistener_bulk(
                config.courtlistener, max_records=max(needed, 0)
            )
            raw_cl = raw_rest + raw_bulk
        else:
            raw_cl = raw_rest

        norm_cl = [normalize_courtlistener_record(item) for item in raw_cl]
        scored_cl = score_and_filter(norm_cl, config.quality)
        scored_cl.accepted = scored_cl.accepted[: config.target_courtlistener_high_value]

        write_outputs(config.output.processed_dir, f"courtlistener_{run_slug}", scored_cl)

        return scored_cl.accepted, {
            "fetched": len(raw_cl),
            "accepted": len(scored_cl.accepted),
            "quarantined": len(scored_cl.quarantined),
            "rejected_reasons": scored_cl.rejected_reasons,
        }
    except Exception as exc:
        return [], {"error": str(exc)}


def _run_pile_of_law(config, run_slug: str, dry_run: bool) -> tuple[list, dict]:
    """Fetch, normalize, and score pile-of-law data."""
    try:
        curated, diverse = fetch_pile_of_law(
            cfg=config.pile_of_law,
            raw_dir=config.output.raw_pile_of_law_dir,
            dry_run=dry_run,
        )
        norm_pol = curated + diverse
        scored_pol = score_and_filter(norm_pol, config.quality)
        write_outputs(config.output.processed_dir, f"pile_of_law_{run_slug}", scored_pol)

        return scored_pol.accepted, {
            "fetched": len(norm_pol),
            "accepted": len(scored_pol.accepted),
            "quarantined": len(scored_pol.quarantined),
            "rejected_reasons": scored_pol.rejected_reasons,
        }
    except Exception as exc:
        return [], {"error": str(exc)}


def run() -> int:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    config = load_config(project_root)
    run_slug = timestamp_slug()

    ensure_dirs(
        [
            config.output.root,
            config.output.raw_courtlistener_dir,
            config.output.raw_pile_of_law_dir,
            config.output.processed_dir,
            config.output.manifests_dir,
            config.courtlistener.bulk_path,
        ]
    )

    selected_sources = set(args.sources)
    if "all" in selected_sources:
        selected_sources = {"courtlistener", "pile-of-law"}

    start = time.perf_counter()
    counters = {
        "courtlistener_fetched": 0,
        "courtlistener_accepted": 0,
        "courtlistener_quarantined": 0,
        "pile_of_law_fetched": 0,
        "pile_of_law_accepted": 0,
        "pile_of_law_quarantined": 0,
    }
    warnings: list[str] = []
    merged_records = []

    # Parallel execution of CourtListener and pile-of-law sources
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}

        if "courtlistener" in selected_sources:
            futures["cl"] = executor.submit(_run_courtlistener, config, run_slug, args.dry_run)

        if "pile-of-law" in selected_sources:
            futures["pol"] = executor.submit(_run_pile_of_law, config, run_slug, args.dry_run)

        # Collect results as they complete
        for source_key, future in futures.items():
            try:
                accepted_records, source_counters = future.result()
                if "error" in source_counters:
                    warning = f"{source_key} ingestion failed: {source_counters['error']}"
                    warnings.append(warning)
                    print(warning)
                    continue

                merged_records.extend(accepted_records)

                if source_key == "cl":
                    counters["courtlistener_fetched"] = source_counters.get("fetched", 0)
                    counters["courtlistener_accepted"] = source_counters.get("accepted", 0)
                    counters["courtlistener_quarantined"] = source_counters.get("quarantined", 0)
                    counters["courtlistener_rejected_reasons"] = source_counters.get(
                        "rejected_reasons", {}
                    )
                else:
                    counters["pile_of_law_fetched"] = source_counters.get("fetched", 0)
                    counters["pile_of_law_accepted"] = source_counters.get("accepted", 0)
                    counters["pile_of_law_quarantined"] = source_counters.get("quarantined", 0)
                    counters["pile_of_law_rejected_reasons"] = source_counters.get(
                        "rejected_reasons", {}
                    )
            except Exception as exc:
                warning = f"{source_key} ingestion exception: {exc}"
                warnings.append(warning)
                print(warning)

    # Merge without redundant dedup (records already scored and deduped per-source)
    merge_cfg = dataclasses.replace(config.quality, dedupe_enabled=False)
    merged_outputs = score_and_filter(merged_records, merge_cfg)
    write_outputs(config.output.processed_dir, f"merged_{run_slug}", merged_outputs)

    runtime = time.perf_counter() - start
    manifest = build_feasibility_manifest(runtime, config.output.root, counters)
    manifest["run_id"] = run_slug
    manifest["dry_run"] = bool(args.dry_run)
    manifest["source_selection"] = sorted(selected_sources)
    manifest["merged_accepted"] = len(merged_outputs.accepted)
    manifest["merged_quarantined"] = len(merged_outputs.quarantined)
    manifest["warnings"] = warnings

    write_manifest(
        config.output.manifests_dir / f"ingestion_manifest_{run_slug}.json", manifest
    )

    print("Ingestion run complete")
    print(f"Run ID: {run_slug}")
    print(f"Dry run: {args.dry_run}")
    print(f"Merged accepted records: {len(merged_outputs.accepted)}")
    print(f"Merged quarantined records: {len(merged_outputs.quarantined)}")
    print(f"Runtime seconds: {manifest['runtime_seconds']}")
    print(f"Output MB: {manifest['output_mb']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
