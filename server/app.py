import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from search import search_all

FRONTEND_ORIGIN = "http://localhost:3000"
PORT = 8000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Project Litt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/search")
async def api_search(
    query: str = Query(..., min_length=1),
    jurisdiction: str | None = None,
    court_level: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    outcome: str | None = None,
    good_law_only: bool | None = None,
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    filters: dict[str, Any] = {
        "jurisdiction": jurisdiction,
        "court_level": court_level,
        "year_from": year_from,
        "year_to": year_to,
        "outcome": outcome,
        "is_good_law": good_law_only,
    }
    filters = {k: v for k, v in filters.items() if v is not None}

    try:
        data = await search_all(query, filters=filters, limit=limit)
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return {
        "query": query,
        "applied_filters": filters,
        "count": data["count"],
        "latency_ms": data["latency_ms"],
        "components": data.get("components", {}),
        "results": data["results"],
    }


if __name__ == "__main__":
    import uvicorn

    try:
        port_value = int(PORT)
    except ValueError:
        port_value = 8000

    uvicorn.run(app, host="0.0.0.0", port=port_value)
