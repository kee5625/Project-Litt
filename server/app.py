import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gremlin_python.driver.client import Client
from integrations.azure.openai_client import close_openai_client, get_openai_client

from api import router as api_router

FRONTEND_ORIGIN = "http://localhost:3000"
PORT = 8000

# ── Logging setup ─────────────────────────────────────────────────────────────
# Root logger at INFO with a compact format — no headers, no request bodies.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Keep uvicorn's lifespan messages (startup/shutdown) visible
logging.getLogger("uvicorn").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI(title="Project Litt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/ingest")
async def index(files: List[File]):
    return
    
@app.get("/api/search")
async def api_search(query: str):
    return 

if __name__ == "__main__":
    import uvicorn

    try:
        port_value = int(PORT)
    except ValueError:
        port_value = 8000

    uvicorn.run(app, host="0.0.0.0", port=port_value)
