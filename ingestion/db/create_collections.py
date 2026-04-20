#!/usr/bin/env python3
"""
Create Actian VectorAI collections for LegalMind.

Run once before Stage 4. Docker must be running:
    docker compose -f actian-db/docker-compose.yml up -d
    python ingestion/db/create_collections.py
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from actian_vectorai import AsyncVectorAIClient, Distance, VectorParams

SERVER = "localhost:50051"
VECTOR_DIM = 1024
COLLECTIONS = ["case_summaries", "case_law", "statutes", "clause_templates"]


def str_to_int_id(s: str) -> int:
    """Convert a string ID to a stable positive integer for Actian PointStruct.id."""
    return abs(hash(s)) % (2**53)


async def main() -> None:
    async with AsyncVectorAIClient(SERVER) as client:
        for name in COLLECTIONS:
            if await client.collections.exists(name):
                print(f"  already exists: {name}")
            else:
                await client.collections.create(
                    name,
                    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.Cosine),
                )
                print(f"  created: {name}")

        names = await client.collections.list()
        print(f"\nCollections on server: {names}")


if __name__ == "__main__":
    asyncio.run(main())
