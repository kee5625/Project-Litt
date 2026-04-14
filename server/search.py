if __name__ == "__main__":
    search_query = sys.argv[1] if len(sys.argv) > 1 else "case law 1"

    print("=== Testing search (RRF) ===")
    result = asyncio.run(search(search_query))
    print(f"Results count: {len(result.get('results', []))}")
    for item in result.get("results", [])[:10]:
        print(item)
