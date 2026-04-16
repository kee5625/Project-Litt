import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import app as app_module
import search


class _FakeCollections:
    def get_info(self, _collection_name):
        vectors = SimpleNamespace(size=64)
        params = SimpleNamespace(vectors=vectors)
        config = SimpleNamespace(params=params)
        return SimpleNamespace(config=config)


class _FakePoints:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def search(self, collection_name, **kwargs):
        self.calls.append({"collection": collection_name, **kwargs})
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.collections = _FakeCollections()
        self.points = _FakePoints(responses)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class SearchPipelineTests(IsolatedAsyncioTestCase):
    async def test_vector_search_returns_normalized_results(self):
        points = [
            SimpleNamespace(
                id="a1",
                score=0.91,
                payload={
                    "case_name": "Anderson v. Liberty Lobby",
                    "citation": "477 U.S. 242",
                    "court": "SCOTUS",
                    "jurisdiction": "federal",
                    "court_level": "supreme",
                    "year": 1986,
                    "outcome": "affirmed",
                    "is_good_law": True,
                    "source_url": "https://example.com/a1",
                    "holding_text": "Summary judgment standard...",
                },
            )
        ]
        fake_client = _FakeClient([points])

        with patch("search.VectorAIClient", return_value=fake_client):
            data = await search.vector_search(
                "summary judgment standard",
                filters={"jurisdiction": "federal", "year_from": 1980},
                limit=5,
            )

        self.assertEqual(data["count"], 1)
        self.assertEqual(data["results"][0]["case_name"], "Anderson v. Liberty Lobby")
        self.assertEqual(data["results"][0]["source"], "vector")
        self.assertEqual(fake_client.points.calls[0]["collection"], search.COLLECTION)
        self.assertEqual(fake_client.points.calls[0]["limit"], 5)
        self.assertIsNotNone(fake_client.points.calls[0]["filter"])

    async def test_keyword_search_reranks_by_term_hits(self):
        candidates = [
            SimpleNamespace(
                id="doc-low",
                score=0.92,
                payload={
                    "holding_text": "summary only",
                    "case_name": "Low Hit Case",
                    "citation": "1 F.3d 1",
                    "court": "9th Cir.",
                },
            ),
            SimpleNamespace(
                id="doc-high",
                score=0.10,
                payload={
                    "holding_text": "summary judgment standard for summary proceedings",
                    "case_name": "High Hit Case",
                    "citation": "2 F.3d 2",
                    "court": "2nd Cir.",
                },
            ),
        ]
        fake_client = _FakeClient([candidates])

        with patch("search.VectorAIClient", return_value=fake_client):
            data = await search.keyword_search("summary judgment", limit=2)

        self.assertEqual(data["count"], 2)
        self.assertEqual(data["results"][0]["id"], "doc-high")
        self.assertEqual(data["results"][1]["id"], "doc-low")
        self.assertIn("summary", data["terms"])
        self.assertIn("judgment", data["terms"])

    async def test_search_all_combines_sources_and_sets_match_reasons(self):
        vector_data = {
            "results": [
                {
                    "id": "same",
                    "score": 0.8,
                    "source": "vector",
                    "case_name": "Case A",
                    "citation": "1 F.3d 1",
                    "court": "9th Cir.",
                    "jurisdiction": "federal",
                    "court_level": "circuit",
                    "year": 2020,
                    "outcome": "affirmed",
                    "is_good_law": True,
                    "source_url": "https://example.com/a",
                    "holding_text": "text a",
                    "full_cite_str": "Case A, 1 F.3d 1",
                }
            ],
            "count": 1,
            "latency_ms": 5,
        }
        keyword_data = {
            "results": [
                {
                    "id": "same",
                    "score": 0.7,
                    "source": "keyword",
                    "case_name": "Case A",
                    "citation": "1 F.3d 1",
                    "court": "9th Cir.",
                    "jurisdiction": "federal",
                    "court_level": "circuit",
                    "year": 2020,
                    "outcome": "affirmed",
                    "is_good_law": True,
                    "source_url": "https://example.com/a",
                    "holding_text": "text a",
                    "full_cite_str": "Case A, 1 F.3d 1",
                },
                {
                    "id": "kw-only",
                    "score": 0.6,
                    "source": "keyword",
                    "case_name": "Case B",
                    "citation": "2 F.3d 2",
                    "court": "2nd Cir.",
                    "jurisdiction": "federal",
                    "court_level": "circuit",
                    "year": 2021,
                    "outcome": "reversed",
                    "is_good_law": True,
                    "source_url": "https://example.com/b",
                    "holding_text": "text b",
                    "full_cite_str": "Case B, 2 F.3d 2",
                },
            ],
            "count": 2,
            "latency_ms": 4,
            "terms": ["summary", "judgment"],
        }

        fused_points = [
            SimpleNamespace(id="same", score=1.0, payload=vector_data["results"][0]),
            SimpleNamespace(
                id="kw-only", score=0.8, payload=keyword_data["results"][1]
            ),
        ]

        with (
            patch("search.vector_search", new=AsyncMock(return_value=vector_data)),
            patch("search.keyword_search", new=AsyncMock(return_value=keyword_data)),
            patch("search.reciprocal_rank_fusion", return_value=fused_points),
        ):
            data = await search.search_all("summary judgment", limit=5)

        self.assertEqual(data["count"], 2)
        first = data["results"][0]
        second = data["results"][1]
        self.assertEqual(first["id"], "same")
        self.assertEqual(first["source"], "hybrid")
        self.assertEqual(set(first["match_reasons"]), {"semantic", "keyword"})
        self.assertEqual(second["id"], "kw-only")
        self.assertEqual(second["match_reasons"], ["keyword"])


class ApiSearchRouteTests(TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)

    def test_api_search_maps_query_params_to_filters(self):
        mocked = {
            "count": 1,
            "latency_ms": 12,
            "components": {"vector_count": 1, "keyword_count": 1, "keyword_terms": []},
            "results": [{"id": "1", "case_name": "Case A", "source": "hybrid"}],
        }
        with patch("app.search_all", new=AsyncMock(return_value=mocked)) as mock_search:
            response = self.client.get(
                "/api/search",
                params={
                    "query": "summary judgment",
                    "jurisdiction": "federal",
                    "court_level": "district",
                    "year_from": 2015,
                    "year_to": 2025,
                    "outcome": "affirmed",
                    "good_law_only": "true",
                    "limit": 7,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["query"], "summary judgment")
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["applied_filters"]["is_good_law"], True)
        self.assertEqual(body["applied_filters"]["court_level"], "district")

        _, kwargs = mock_search.call_args
        self.assertEqual(kwargs["limit"], 7)
        self.assertEqual(kwargs["filters"]["jurisdiction"], "federal")

    def test_api_search_returns_500_on_search_failure(self):
        with patch("app.search_all", new=AsyncMock(side_effect=RuntimeError("boom"))):
            response = self.client.get(
                "/api/search", params={"query": "ada accommodations"}
            )

        self.assertEqual(response.status_code, 500)
        self.assertIn("Search failed", response.json()["detail"])
