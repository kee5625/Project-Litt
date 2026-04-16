from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any

from actian_vectorai import (
    Field,
    FilterBuilder,
    ScoredPoint,
    VectorAIClient,
    reciprocal_rank_fusion,
)

SERVER = os.getenv("ACTIAN_SERVER", "localhost:50051")
COLLECTION = os.getenv("ACTIAN_COLLECTION", "case_law")
DEFAULT_LIMIT = 10
MAX_LIMIT = 50
KEYWORD_CANDIDATE_LIMIT = 200
KEYWORD_TERMS_LIMIT = 8


def _clamp_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    return max(1, min(limit, MAX_LIMIT))


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def _tokenize_query(text: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9\.\-]+", text.lower())
    deduped: list[str] = []
    for term in terms:
        if len(term) < 3:
            continue
        if term in deduped:
            continue
        deduped.append(term)
        if len(deduped) >= KEYWORD_TERMS_LIMIT:
            break
    return deduped


def _build_filter(filters: dict[str, Any] | None):
    if not filters:
        return None

    builder = FilterBuilder()

    jurisdiction = filters.get("jurisdiction")
    if jurisdiction:
        builder.must(Field("jurisdiction").eq(jurisdiction))

    court_level = filters.get("court_level")
    if court_level:
        builder.must(Field("court_level").eq(court_level))

    year_from = filters.get("year_from")
    if year_from is not None:
        builder.must(Field("year").gte(int(year_from)))

    year_to = filters.get("year_to")
    if year_to is not None:
        builder.must(Field("year").lte(int(year_to)))

    outcome = filters.get("outcome")
    if outcome:
        builder.must(Field("outcome").eq(outcome))

    is_good_law = filters.get("is_good_law")
    if is_good_law is not None:
        builder.must(Field("is_good_law").eq(bool(is_good_law)))

    if builder.is_empty():
        return None
    return builder.build()


def _build_keyword_filter(
    terms: list[str],
    base_filters: dict[str, Any] | None,
):
    builder = FilterBuilder()

    if base_filters:
        jurisdiction = base_filters.get("jurisdiction") if base_filters else None
        if jurisdiction:
            builder.must(Field("jurisdiction").eq(jurisdiction))
        court_level = base_filters.get("court_level") if base_filters else None
        if court_level:
            builder.must(Field("court_level").eq(court_level))
        year_from = base_filters.get("year_from") if base_filters else None
        if year_from is not None:
            builder.must(Field("year").gte(int(year_from)))
        year_to = base_filters.get("year_to") if base_filters else None
        if year_to is not None:
            builder.must(Field("year").lte(int(year_to)))
        outcome = base_filters.get("outcome") if base_filters else None
        if outcome:
            builder.must(Field("outcome").eq(outcome))
        is_good_law = base_filters.get("is_good_law") if base_filters else None
        if is_good_law is not None:
            builder.must(Field("is_good_law").eq(bool(is_good_law)))

    for term in terms:
        builder.should(Field("holding_text").text(term))
        builder.should(Field("case_name").text(term))
        builder.should(Field("citation").text(term))

    if terms:
        builder.min_should(
            [Field("holding_text").text(t) for t in terms],
            min_count=1,
        )

    if builder.is_empty():
        return None
    return builder.build()


def _build_query_vector(search_query: str, dim: int) -> list[float]:
    # Deterministic hashed pseudo-embedding for query-side retrieval only.
    # This allows searching an already indexed collection without local model deps.
    seed = abs(hash(search_query)) % (2**32)
    values: list[float] = []
    state = seed
    for _ in range(dim):
        state = (1664525 * state + 1013904223) % (2**32)
        values.append((state / (2**32)) * 2.0 - 1.0)
    return values


def _extract_dim(client: VectorAIClient) -> int:
    info = client.collections.get_info(COLLECTION)
    config = getattr(getattr(info, "config", None), "params", None)
    vectors_cfg = getattr(config, "vectors", None) if config is not None else None
    if hasattr(vectors_cfg, "size"):
        return int(vectors_cfg.size)
    if isinstance(vectors_cfg, dict) and vectors_cfg:
        first = next(iter(vectors_cfg.values()))
        return int(first.size)

    env_dim = os.getenv("ACTIAN_VECTOR_DIM")
    if env_dim:
        try:
            return int(env_dim)
        except ValueError as exc:
            raise ValueError("ACTIAN_VECTOR_DIM must be an integer") from exc

    # Fallback for servers that do not return collection config in get_info.
    # Matches scripts/actian_loader.py EMBED_DIM default.
    return 384


def _point_to_result(point: ScoredPoint, source: str) -> dict[str, Any]:
    payload = point.payload or {}
    return {
        "id": str(point.id),
        "score": float(point.score),
        "source": source,
        "case_name": payload.get("case_name"),
        "citation": payload.get("citation"),
        "court": payload.get("court"),
        "jurisdiction": payload.get("jurisdiction"),
        "court_level": payload.get("court_level"),
        "year": payload.get("year"),
        "outcome": payload.get("outcome"),
        "is_good_law": payload.get("is_good_law"),
        "source_url": payload.get("source_url"),
        "holding_text": payload.get("holding_text"),
        "full_cite_str": payload.get("full_cite_str"),
    }


def _keyword_hit_count(payload: dict[str, Any], terms: list[str]) -> int:
    haystack = " ".join(
        str(payload.get(field, "") or "")
        for field in ("holding_text", "case_name", "citation", "court")
    ).lower()
    return sum(1 for t in terms if t in haystack)


async def vector_search(
    search_query: str,
    filters: dict[str, Any] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    query = _normalize_query(search_query)
    if not query:
        return {"results": [], "count": 0, "latency_ms": 0}

    bounded_limit = _clamp_limit(limit)
    filter_obj = _build_filter(filters)
    started = time.perf_counter()

    with VectorAIClient(SERVER) as client:
        dim = _extract_dim(client)
        query_vector = _build_query_vector(query, dim)
        scored = client.points.search(
            COLLECTION,
            vector=query_vector,
            limit=bounded_limit,
            filter=filter_obj,
            with_payload=True,
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    results = [_point_to_result(point, "vector") for point in scored]
    return {"results": results, "count": len(results), "latency_ms": latency_ms}


async def keyword_search(
    search_query: str,
    filters: dict[str, Any] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    query = _normalize_query(search_query)
    if not query:
        return {"results": [], "count": 0, "latency_ms": 0, "terms": []}

    bounded_limit = _clamp_limit(limit)
    terms = _tokenize_query(query)
    keyword_filter = _build_keyword_filter(terms, filters)
    started = time.perf_counter()

    with VectorAIClient(SERVER) as client:
        dim = _extract_dim(client)
        broad_vector = _build_query_vector("keyword " + query, dim)
        candidates = client.points.search(
            COLLECTION,
            vector=broad_vector,
            limit=max(bounded_limit * 4, KEYWORD_CANDIDATE_LIMIT),
            filter=keyword_filter,
            with_payload=True,
        )

    rescored = []
    for point in candidates:
        payload = point.payload or {}
        hits = _keyword_hit_count(payload, terms)
        if terms and hits == 0:
            continue
        combined_score = float(hits) * 10.0 + float(point.score)
        rescored.append((combined_score, point))

    rescored.sort(key=lambda item: item[0], reverse=True)
    top_points = [point for _, point in rescored[:bounded_limit]]
    latency_ms = int((time.perf_counter() - started) * 1000)

    results = [_point_to_result(point, "keyword") for point in top_points]
    return {
        "results": results,
        "count": len(results),
        "latency_ms": latency_ms,
        "terms": terms,
    }


async def search_all(
    search_query: str,
    filters: dict[str, Any] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    bounded_limit = _clamp_limit(limit)
    started = time.perf_counter()

    vector_task = vector_search(search_query, filters=filters, limit=bounded_limit * 3)
    keyword_task = keyword_search(
        search_query, filters=filters, limit=bounded_limit * 3
    )
    vector_data, keyword_data = await asyncio.gather(vector_task, keyword_task)

    vector_points = [
        ScoredPoint(
            id=item["id"],
            score=item["score"],
            payload={
                k: v for k, v in item.items() if k not in {"id", "score", "source"}
            },
        )
        for item in vector_data["results"]
    ]
    keyword_points = [
        ScoredPoint(
            id=item["id"],
            score=item["score"],
            payload={
                k: v for k, v in item.items() if k not in {"id", "score", "source"}
            },
        )
        for item in keyword_data["results"]
    ]

    fused = reciprocal_rank_fusion(
        [vector_points, keyword_points],
        limit=bounded_limit,
        weights=[0.65, 0.35],
    )

    vector_ids = {str(item["id"]) for item in vector_data["results"]}
    keyword_ids = {str(item["id"]) for item in keyword_data["results"]}

    final_results: list[dict[str, Any]] = []
    for point in fused:
        source_tags: list[str] = []
        pid = str(point.id)
        if pid in vector_ids:
            source_tags.append("semantic")
        if pid in keyword_ids:
            source_tags.append("keyword")
        item = _point_to_result(point, "hybrid")
        item["match_reasons"] = source_tags
        final_results.append(item)

    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "results": final_results,
        "count": len(final_results),
        "latency_ms": latency_ms,
        "components": {
            "vector_count": vector_data["count"],
            "keyword_count": keyword_data["count"],
            "keyword_terms": keyword_data.get("terms", []),
        },
    }
