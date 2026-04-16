#!/usr/bin/env python3
"""
Load ingestion pipeline output into Actian VectorAI DB.

Reads cleaned CSVs from run_ingestion.py, chunks plain_text using semantic boundaries,
embeds with sentence-transformers, and inserts into Actian with all metadata.
"""

import argparse
import asyncio
import csv
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# Import Actian client
try:
    from actian_vectorai import (
        AsyncVectorAIClient,
        Distance,
        Field,
        FilterBuilder,
        PointStruct,
        VectorAIClient,
        VectorParams,
    )
except ImportError:
    print("ERROR: actian_vectorai not installed. Run:")
    print("  pip install ../actian-db/actian_vectorai-0.1.0b2-py3-none-any.whl")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
DEFAULT_COLLECTION = "case_law"
DEFAULT_SERVER = "localhost:50051"
DEFAULT_EMBED_BATCH = 64
DEFAULT_UPLOAD_BATCH = 256
DEFAULT_MIN_CHUNK_CHARS = 600
CHUNK_TARGET_SIZE = 2000
MAX_METADATA_FIELD_CHARS = 4096

# Semantic chunking thresholds
SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # Split when similarity drops below this
MIN_SENTENCES_PER_CHUNK = 2

# Test queries to validate insertion
TEST_QUERIES = [
    {
        "text": "employment wrongful termination california",
        "filters": {"practice_area": "employment", "jurisdiction": "CA"},
        "limit": 3,
    },
    {
        "text": "ADA reasonable accommodation remote work",
        "filters": {"practice_area": "employment"},
        "limit": 3,
    },
    {"text": "summary judgment standard 9th circuit", "filters": {}, "limit": 3},
]

# ============================================================================
# Sentence Splitting
# ============================================================================


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences. Handles common abbreviations.
    Returns list of non-empty sentences.
    """
    import re

    # Replace common abbreviations to preserve them
    text = text.replace("U.S.", "U_S_")
    text = text.replace("U.S.C.", "U_S_C_")
    text = text.replace("etc.", "etc_")
    text = text.replace("e.g.", "e_g_")
    text = text.replace("i.e.", "i_e_")
    text = text.replace("vs.", "vs_")
    text = text.replace("v.", "v_")

    # Split on period, question mark, exclamation
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Restore abbreviations
    sentences = [
        s.replace("U_S_", "U.S.")
        .replace("U_S_C_", "U.S.C.")
        .replace("etc_", "etc.")
        .replace("e_g_", "e.g.")
        .replace("i_e_", "i.e.")
        .replace("vs_", "vs.")
        .replace("v_", "v.")
        for s in sentences
    ]

    return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# Semantic Chunking
# ============================================================================


class SemanticChunker:
    """Chunk text based on semantic similarity between sentences."""

    def __init__(
        self,
        embed_model: SentenceTransformer,
        threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    ):
        self.embed_model = embed_model
        self.threshold = threshold

    def chunk_text(
        self, text: str, min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS
    ) -> list[str]:
        """
        Chunk text using semantic similarity.
        Splits when similarity between consecutive sentences drops below threshold.
        Respects paragraph boundaries and minimum size constraints.
        """
        if not text or len(text.strip()) < min_chunk_chars:
            return []

        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # For each paragraph, split into sentences and apply semantic chunking
        all_chunks = []
        for para in paragraphs:
            para_chunks = self._chunk_paragraph(para, min_chunk_chars)
            all_chunks.extend(para_chunks)

        return all_chunks

    def _chunk_paragraph(self, para_text: str, min_chunk_chars: int) -> list[str]:
        """Chunk a single paragraph using semantic similarity."""
        sentences = split_into_sentences(para_text)
        if not sentences:
            return []

        if len(sentences) <= MIN_SENTENCES_PER_CHUNK:
            # Too short to split further
            chunk_text = " ".join(sentences)
            if len(chunk_text) >= min_chunk_chars:
                return [chunk_text]
            return []

        # Encode all sentences for similarity comparison
        sentence_embeddings = self.embed_model.encode(sentences, convert_to_numpy=True)

        # Find semantic break points
        chunks = []
        current_chunk_idx = 0

        for i in range(1, len(sentences)):
            # Cosine similarity between consecutive sentences
            sim = np.dot(sentence_embeddings[i - 1], sentence_embeddings[i]) / (
                np.linalg.norm(sentence_embeddings[i - 1])
                * np.linalg.norm(sentence_embeddings[i])
            )

            # Check if we should split here
            should_split = False

            # Split if similarity drops below threshold
            if sim < self.threshold:
                should_split = True

            # Also split if chunk is getting too large
            current_chunk_text = " ".join(sentences[current_chunk_idx:i])
            if (
                len(current_chunk_text) >= CHUNK_TARGET_SIZE
                and i > current_chunk_idx + MIN_SENTENCES_PER_CHUNK
            ):
                should_split = True

            if should_split:
                chunk_text = " ".join(sentences[current_chunk_idx:i])
                if len(chunk_text) >= min_chunk_chars:
                    chunks.append(chunk_text)
                current_chunk_idx = i

        # Don't forget the final chunk
        final_chunk_text = " ".join(sentences[current_chunk_idx:])
        if len(final_chunk_text) >= min_chunk_chars:
            chunks.append(final_chunk_text)

        return chunks


# ============================================================================
# CSV I/O
# ============================================================================


def find_latest_csv(csv_dir: Path) -> Optional[Path]:
    """Find the latest merged_*_accepted.csv in the output/processed directory."""
    merged_files = list(csv_dir.glob("merged_*_accepted.csv"))
    if not merged_files:
        return None
    return sorted(merged_files)[-1]  # Most recent by name (run ID is timestamp-based)


def load_csv(csv_path: Path) -> list[dict]:
    """Load CSV file, return list of dicts."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


# ============================================================================
# Embedding
# ============================================================================


class EmbeddingModel:
    """Wrapper around sentence-transformers for batch encoding."""

    def __init__(self, model_name: str):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode_batch(
        self, texts: list[str], batch_size: int = DEFAULT_EMBED_BATCH
    ) -> np.ndarray:
        """Encode list of texts, return (N, EMBED_DIM) numpy array."""
        logger.info(f"Encoding {len(texts)} texts in batches of {batch_size}")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings

    def encode_single(self, text: str) -> list[float]:
        """Encode single text, return list[float]."""
        emb = self.model.encode([text], convert_to_numpy=False)[0]
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def get_model(self):
        """Return the underlying model for semantic chunking."""
        return self.model


# ============================================================================
# Actian Loader
# ============================================================================


class ActianLoader:
    """Load chunks + embeddings into Actian VectorAI DB."""

    def __init__(
        self,
        server: str,
        collection: str,
        recreate: bool = False,
    ):
        self.server = server
        self.collection = collection
        self.recreate = recreate
        self.client = None

    def connect(self):
        """Connect to Actian server."""
        logger.info(f"Connecting to {self.server}")
        try:
            self.client = VectorAIClient(self.server)
            # Newer client versions require an explicit connect() call.
            self.client.connect()
            _ = self.client.collections.list()
            logger.info("Connected successfully")
        except Exception as sync_error:
            logger.warning(f"Sync client connection failed: {sync_error}")
            logger.info(
                "Trying AsyncVectorAIClient context-manager connectivity test..."
            )
            try:
                asyncio.run(self._probe_async_connection())
                logger.error(
                    "Async connection test succeeded, but sync client could not connect. "
                    "Please keep using async context-manager flow."
                )
            except Exception as async_error:
                logger.error(
                    "Failed to connect with both sync and async clients. "
                    f"sync_error={sync_error} async_error={async_error}"
                )
            raise

    async def _probe_async_connection(self):
        """Probe connectivity using async context manager."""
        async with AsyncVectorAIClient(self.server) as client:
            await client.collections.list()

    def close(self):
        """Close connection."""
        if self.client:
            self.client.close()

    def setup_collection(self):
        """Create or recreate collection."""
        if self.recreate:
            logger.info(f"Dropping collection '{self.collection}'")
            try:
                self.client.collections.delete(self.collection, strict=False)
            except Exception as e:
                logger.debug(f"Delete raised (might not exist): {e}")

        logger.info(
            f"Creating/verifying collection '{self.collection}' with dim={EMBED_DIM}"
        )
        try:
            self.client.collections.create(
                self.collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.Cosine),
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(
                    f"Collection '{self.collection}' already exists; reusing it"
                )
            else:
                raise

    def load_chunks(
        self,
        chunks_with_metadata: list[dict],
        embeddings: np.ndarray,
        upload_batch: int = DEFAULT_UPLOAD_BATCH,
    ) -> int:
        """Bulk insert chunks + embeddings."""
        logger.info(f"Building {len(chunks_with_metadata)} PointStructs")
        points = []
        for chunk_data, emb in zip(chunks_with_metadata, embeddings):
            payload = self._sanitize_payload(chunk_data)
            pt = PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist() if isinstance(emb, np.ndarray) else list(emb),
                payload=payload,
            )
            points.append(pt)

        logger.info(f"Uploading {len(points)} points in batches of {upload_batch}")
        total = self.client.upload_points(
            self.collection, points, batch_size=upload_batch
        )
        logger.info(f"Uploaded {total} points")

        return total

    def _sanitize_payload(self, payload: dict) -> dict:
        """Drop malformed oversized metadata fields before upload.

        Some source rows contain a field that accidentally captures the full
        opinion text instead of a short metadata value. The payload store in
        Actian rejects those oversized records, so we keep only concise fields.
        """
        sanitized = {}
        for key, value in payload.items():
            if key == "__record_idx":
                continue
            if isinstance(value, str) and len(value) > MAX_METADATA_FIELD_CHARS:
                logger.warning(
                    "Dropping oversized payload field %s (len=%s)",
                    key,
                    len(value),
                )
                continue
            sanitized[key] = value
        return sanitized

    def test_queries(self, embed_fn) -> int:
        """Run test queries to validate insertion. Returns count of successful queries."""
        logger.info("=" * 70)
        logger.info("RUNNING TEST QUERIES")
        logger.info("=" * 70)

        success_count = 0
        for i, query_spec in enumerate(TEST_QUERIES, 1):
            query_text = query_spec["text"]
            query_filters = query_spec["filters"]
            limit = query_spec["limit"]

            logger.info(f"\nTest {i}: '{query_text}'")
            if query_filters:
                logger.info(f"  Filters: {query_filters}")

            # Encode query
            query_vec = embed_fn(query_text)

            # Build filter if needed
            query_filter = None
            if query_filters:
                fb = FilterBuilder()
                for key, value in query_filters.items():
                    fb = fb.must(Field(key).eq(value))
                query_filter = fb.build()

            # Search
            try:
                results = self.client.points.search(
                    self.collection,
                    vector=query_vec,
                    filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    score_threshold=0.0,  # Accept all results
                )

                if results:
                    logger.info(f"  Found {len(results)} result(s):")
                    for j, r in enumerate(results, 1):
                        p = r.payload or {}
                        logger.info(
                            f"    {j}. score={r.score:.4f} | "
                            f"{p.get('case_name', 'N/A')} "
                            f"({p.get('citation', 'N/A')}) | "
                            f"{p.get('court', 'N/A')} {p.get('year', '?')} | "
                            f"chunk {p.get('chunk_index', '?')}/{p.get('chunk_total', '?')}"
                        )
                    success_count += 1
                else:
                    logger.warning(f"  No results found")
            except Exception as e:
                logger.error(f"  Query failed: {e}")

        logger.info("\n" + "=" * 70)
        logger.info(f"Test queries: {success_count}/{len(TEST_QUERIES)} successful")
        logger.info("=" * 70)

        return success_count


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Load ingestion pipeline output into Actian VectorAI DB"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to CSV file (auto-detects latest if not provided)",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Actian collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"Actian server address (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--embed-batch",
        type=int,
        default=DEFAULT_EMBED_BATCH,
        help=f"Embedding batch size (default: {DEFAULT_EMBED_BATCH})",
    )
    parser.add_argument(
        "--upload-batch",
        type=int,
        default=DEFAULT_UPLOAD_BATCH,
        help=f"Upload batch size (default: {DEFAULT_UPLOAD_BATCH})",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=DEFAULT_MIN_CHUNK_CHARS,
        help=f"Minimum chunk size in chars (default: {DEFAULT_MIN_CHUNK_CHARS})",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SEMANTIC_SIMILARITY_THRESHOLD,
        help=f"Semantic similarity threshold for splitting (default: {SEMANTIC_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Drop and recreate the collection"
    )

    args = parser.parse_args()

    # Find CSV
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            logger.error(f"CSV not found: {csv_path}")
            sys.exit(1)
    else:
        csv_dir = Path(__file__).parent / "output" / "processed"
        csv_path = find_latest_csv(csv_dir)
        if not csv_path:
            logger.error(f"No merged_*_accepted.csv found in {csv_dir}")
            logger.error("Run: python run_ingestion.py --sources all --project-root ..")
            sys.exit(1)

    logger.info(f"Using CSV: {csv_path}")
    logger.info(f"Target collection: {args.collection}")
    logger.info(f"Server: {args.server}")

    # Load CSV
    logger.info(f"Loading CSV...")
    records = load_csv(csv_path)
    logger.info(f"Loaded {len(records)} records")

    # Initialize embedding model (shared for semantic chunking and final embedding)
    logger.info("Initializing embedding model...")
    embed_model = EmbeddingModel(EMBED_MODEL_NAME)

    # Initialize semantic chunker
    chunker = SemanticChunker(
        embed_model.get_model(), threshold=args.similarity_threshold
    )

    # Chunk + prepare metadata
    logger.info(f"Chunking {len(records)} documents using semantic boundaries...")
    chunks_with_metadata = []
    all_chunk_texts = []

    for doc_idx, record in enumerate(records):
        if (doc_idx + 1) % max(1, len(records) // 10) == 0:
            logger.info(f"  Processing document {doc_idx + 1}/{len(records)}")

        plain_text = record.get("plain_text", "")
        chunks = chunker.chunk_text(plain_text, min_chunk_chars=args.min_chunk_chars)

        if not chunks:
            logger.debug(f"  Doc {doc_idx}: no valid chunks")
            continue

        for chunk_idx, chunk_text_content in enumerate(chunks):
            metadata = {
                "chunk_text": chunk_text_content,
                "chunk_index": chunk_idx,
                "chunk_total": len(chunks),
            }

            # Copy all metadata fields from record
            for key in record.keys():
                if key != "plain_text":  # Don't duplicate the full text
                    metadata[key] = record[key]

            # Try to convert year and quality_score to numeric
            if "year" in metadata and metadata["year"]:
                try:
                    metadata["year"] = int(metadata["year"])
                except (ValueError, TypeError):
                    pass

            if "quality_score" in metadata and metadata["quality_score"]:
                try:
                    metadata["quality_score"] = float(metadata["quality_score"])
                except (ValueError, TypeError):
                    pass

            chunks_with_metadata.append(metadata)
            all_chunk_texts.append(chunk_text_content)

    logger.info(
        f"Created {len(chunks_with_metadata)} chunks from {len(records)} documents"
    )

    # Embed chunks
    embeddings = embed_model.encode_batch(all_chunk_texts, batch_size=args.embed_batch)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Connect to Actian
    loader = ActianLoader(args.server, args.collection, recreate=args.recreate)
    try:
        loader.connect()
        loader.setup_collection()

        # Upload
        uploaded = loader.load_chunks(
            chunks_with_metadata, embeddings, upload_batch=args.upload_batch
        )
        logger.info(f"Successfully uploaded {uploaded} chunks")

        # Test
        test_success = loader.test_queries(embed_model.encode_single)

        if test_success == len(TEST_QUERIES):
            logger.info("\n✓ All test queries passed!")
            sys.exit(0)
        else:
            logger.warning(
                f"\n⚠ {len(TEST_QUERIES) - test_success} test queries did not return results"
            )
            sys.exit(0)  # Still exit 0, data was loaded

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        loader.close()


if __name__ == "__main__":
    main()
