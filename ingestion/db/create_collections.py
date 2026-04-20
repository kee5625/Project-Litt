#!/usr/bin/env python3
"""
Create Actian VectorAI collections for LegalMind.

Run once before Stage 4. Docker must be running:
    docker compose -f actian-db/docker-compose.yml up -d
    python ingestion/db/create_collections.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from actian_vectorai import Distance, VectorAIClient, VectorParams

SERVER = "localhost:50051"
VECTOR_DIM = 768
COLLECTIONS = ["case_summaries", "case_law", "statutes", "clause_templates"]


def str_to_int_id(s: str) -> int:
    """Convert a string ID to a stable positive integer for Actian PointStruct.id."""
    return abs(hash(s)) % (2**53)


def main() -> None:
    with VectorAIClient(SERVER) as client:
        for name in COLLECTIONS:
            if client.collections.exists(name):
                print(f"  already exists: {name}")
            else:
                client.collections.create(
                    name,
                    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.Cosine),
                )
                print(f"  created: {name}")

        existing = client.collections.get_all()
        print(f"\nCollections on server: {[c.name for c in existing]}")


if __name__ == "__main__":
    main()
