import asyncio 

from dotenv import load_dotenv
from __future__ import annotations

import random

from actian_vectorai import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    SearchParams,
    VectorAIClient,
    VectorParams,
)   
from actian_vectorai.exceptions import VectorAIError

load_dotenv()

SERVER = "localhost:50051"
COLLECTION = "search_demo"
DIM = 64
VOCAB_SIZE = 1000

fmt = "\n=== {:50} ==="

async def vector_search(search_query: str) -> dict:
    random.seed(42)
    
    with VectorAIClient(SERVER) as client:
        # Setup
        if client.collections.exists(COLLECTION):
            client.collections.delete(COLLECTION)
        client.collections.create(
            COLLECTION,
            vectors_config=VectorParams(size=DIM, distance=Distance.Cosine),
        )
        
        # Insert 100 points with category metadata
        points = [
            PointStruct(
                id=i,
                vector=[random.gauss(0, 1) for _ in range(DIM)],
                payload={
                    "category":["electronics", "clothing", "books"][i % 3],
                    "price": round(random.uniform(5,200), 2),
                },
            )
            for i in range(1, 101)
        ]
        client.points.upsert(COLLECTION, points)
        print("Inserted 100 points")
        
        query = [random.gauss(0, 1) for _ in range(DIM)]
        
        # Basic Search
        print(fmt.format("Basic search (top 5)"))
        results = client.points.search(COLLECTION, vector=query, limit=5)
        for r in results:
            print(f" id={r.id:3d} score={r.score:.4f} cat={r.payload.get('category')}")
            
        # Search with score threshold
        print(fmt.format("Paginated search (page 1 & 2)"))
        page1 = client.points.search(COLLECTION, vector=query, limit=3, offset=0)
        page2 = client.points.search(COLLECTION, vector=query, limit=3, offset=3)
        print(f" Page 1: {[r.id for r in page1]}")
        print(f" Page 2: {[r.id for r in page2]}")
        
        # Search with custom HNSW params 
        print(fmt.format("Search with custom HNSW ef"))
        results = client.points.search(
            COLLECTION,
            vector=query,
            limit=5,
            params=SearchParams(hnsw_ef=256, exact=False),
        )
        print(f" Found {len(results)} results with ef=256")
        
        # Exact Search
        print(fmt.format("Exact brute-force search"))
        results = client.points.search(
            COLLECTION,
            vector=query,
            limit=5,
            params=SearchParams(exact=True),
        )
        for r in results:
            print(f" id={r.id:3d} score={r.score:.4f}")
        
        # Batch Search
        print(fmt.format("Batch search (3 queries at once)"))
        queries = [
            {"vector": [random.gauss(0, 1) for _ in range(DIM)], "limit": 3} for _ in range(3)
        ]
        batch_results = client.points.search_batch(COLLECTION, queries)
        for i, results in enumerate(batch_results):
            ids = [r.id for r in results]
            print(f" Query {i + 1}: top IDs = {ids}")
        
        # Cleanup
        client.collections.delete(COLLECTION)
        print("\n Cleaned up")
        
# HELPERS
def sparse_from_text(text: str, vocab_size: int = VOCAB_SIZE) -> SparseVector:
    

async def keyword_search(search_query: str) -> dict:
    

async def search_all(search_query: str) -> dict:
    return

if __name__ == "__main__":
    search_query = sys.argv[1] if len(sys.argv) > 1 else "case law 1"

    print("=== Testing search (RRF) ===")
    result = asyncio.run(search(search_query))
    print(f"Results count: {len(result.get('results', []))}")
    for item in result.get("results", [])[:10]:
        print(item)
