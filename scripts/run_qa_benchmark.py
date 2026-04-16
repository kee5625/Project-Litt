#!/usr/bin/env python3
"""Run a retrieval benchmark over the accepted legal corpus."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CSV = (
    Path(__file__).parent
    / "output"
    / "processed"
    / "merged_20260416T030110Z_accepted.csv"
)
DEFAULT_BENCHMARK = Path(__file__).parent / "qa_benchmark.jsonl"
SEMANTIC_WEIGHT = 0.75
LEXICAL_WEIGHT = 0.25

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass
class BenchmarkItem:
    id: str
    question: str
    answer: str
    acceptable_indices: list[int]
    tags: list[str]


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_benchmark(benchmark_path: Path) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with benchmark_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            items.append(
                BenchmarkItem(
                    id=payload["id"],
                    question=payload["question"],
                    answer=payload["answer"],
                    acceptable_indices=list(payload["acceptable_indices"]),
                    tags=list(payload.get("tags", [])),
                )
            )
    return items


def build_record_text(row: dict[str, str]) -> str:
    parts = [
        row.get("citation", ""),
        row.get("court", ""),
        row.get("jurisdiction", ""),
        row.get("practice_area", ""),
        row.get("source_url", ""),
        row.get("plain_text", "")[:6000],
    ]
    return "\n".join(part for part in parts if part)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def lexical_overlap_score(question_tokens: set[str], record_tokens: set[str]) -> float:
    if not question_tokens:
        return 0.0
    return len(question_tokens & record_tokens) / len(question_tokens)


def run_benchmark(
    csv_path: Path, benchmark_path: Path, model_name: str, top_k: int
) -> int:
    rows = load_rows(csv_path)
    benchmark_items = load_benchmark(benchmark_path)

    model = SentenceTransformer(model_name)

    record_texts = [build_record_text(row) for row in rows]
    question_texts = [item.question for item in benchmark_items]
    record_tokens = [tokenize(text) for text in record_texts]
    question_tokens = [tokenize(text) for text in question_texts]

    record_vectors = normalize_rows(
        model.encode(record_texts, convert_to_numpy=True, show_progress_bar=True)
    )
    question_vectors = normalize_rows(
        model.encode(question_texts, convert_to_numpy=True, show_progress_bar=True)
    )

    top1_hits = 0
    topk_hits = 0

    for item, question_vector, q_tokens in zip(
        benchmark_items, question_vectors, question_tokens
    ):
        semantic_scores = record_vectors @ question_vector
        lexical_scores = np.array(
            [lexical_overlap_score(q_tokens, tokens) for tokens in record_tokens],
            dtype=float,
        )
        scores = (SEMANTIC_WEIGHT * semantic_scores) + (LEXICAL_WEIGHT * lexical_scores)
        ranked_indices = np.argsort(-scores)
        top_indices = ranked_indices[:top_k].tolist()
        top_index = int(ranked_indices[0])

        hit_top1 = top_index in item.acceptable_indices
        hit_topk = any(index in item.acceptable_indices for index in top_indices)

        top1_hits += int(hit_top1)
        topk_hits += int(hit_topk)

        print(f"[{item.id}] {item.question}")
        print(f"  expected: {item.answer}")
        print(
            f"  top1: row {top_index} score={scores[top_index]:.4f} "
            f"semantic={semantic_scores[top_index]:.4f} lexical={lexical_scores[top_index]:.4f} hit={hit_top1}"
        )
        print("  top candidates:")
        for rank, row_index in enumerate(top_indices, 1):
            row = rows[row_index]
            label = f"{row.get('source', '')} | {row.get('citation', '')} | {row.get('practice_area', '')}"
            print(
                f"    {rank}. row {row_index} score={scores[row_index]:.4f} "
                f"semantic={semantic_scores[row_index]:.4f} lexical={lexical_scores[row_index]:.4f} | {label[:140]}"
            )
        print()

    total = len(benchmark_items)
    top1_rate = top1_hits / total if total else 0.0
    topk_rate = topk_hits / total if total else 0.0

    print(
        f"Summary: top1={top1_hits}/{total} ({top1_rate:.1%}), top{top_k}={topk_hits}/{total} ({topk_rate:.1%})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Q&A retrieval benchmark against accepted corpus"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Accepted CSV to benchmark against",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Q&A benchmark JSONL file",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="How many top matches to report"
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1
    if not args.benchmark.exists():
        print(f"Benchmark not found: {args.benchmark}", file=sys.stderr)
        return 1

    return run_benchmark(args.csv, args.benchmark, args.model, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
