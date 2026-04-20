"""
Opinion text cleaning, semantic chunking, and size guard.

The BGE-M3 embed model (via Ollama) is loaded ONCE via load_splitter() and reused for
all opinions. SemanticSplitterNodeParser uses it to detect sentence-boundary breakpoints —
this is separate from the Stage 3 embedding step.
"""
from __future__ import annotations

import re

import tiktoken

STRIP_PATTERNS = [
    re.compile(r"^\d+\s*$", re.MULTILINE),
    re.compile(r"^\*\d+\s*$", re.MULTILINE),
    re.compile(r"\(c\)\s*\d{4}\s+Thomson Reuters.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"WESTLAW.*?END OF DOCUMENT", re.IGNORECASE | re.DOTALL),
    re.compile(r"^FOR PUBLICATION[^\n]*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^NOT FOR PUBLICATION[^\n]*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*[-_=]{5,}\s*$", re.MULTILINE),
]

_enc = tiktoken.get_encoding("cl100k_base")


def clean_opinion_text(text: str) -> str:
    for pat in STRIP_PATTERNS:
        text = pat.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def token_count(text: str) -> int:
    return len(_enc.encode(text))


def load_splitter(semantic: bool = True):
    """Return a text splitter for chunking opinions.

    semantic=True  — SemanticSplitterNodeParser (BGE-M3 via Ollama, higher quality, slow)
    semantic=False — SentenceSplitter (pure Python, ~300 tok chunks, fast)
    """
    if not semantic:
        from llama_index.core.node_parser import SentenceSplitter
        return SentenceSplitter(chunk_size=300, chunk_overlap=40)

    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.ollama import OllamaEmbedding

    embed_model = OllamaEmbedding(model_name="bge-m3")
    return SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=90,
        embed_model=embed_model,
    )


def chunk_text(splitter, text: str) -> list[str]:
    from llama_index.core import Document

    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])
    return [n.get_content() for n in nodes]


def apply_size_guard(chunks: list[str]) -> list[str]:
    """Merge chunks < 80 tokens with next; halve chunks > 600 tokens."""
    result: list[str] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        tc = token_count(chunk)

        if tc < 80 and i + 1 < len(chunks):
            chunks[i + 1] = chunk + "\n" + chunks[i + 1]
            i += 1
            continue

        if tc > 600:
            mid = len(chunk) // 2
            split_at = chunk.rfind(". ", 0, mid)
            split_at = split_at + 2 if split_at > 10 else mid
            first, second = chunk[:split_at].strip(), chunk[split_at:].strip()
            if first:
                result.append(first)
            if second:
                result.append(second)
        else:
            result.append(chunk)

        i += 1

    return [c for c in result if c.strip()]


def chunk_opinion(splitter, text: str) -> list[str]:
    """Clean, chunk, and apply size guard. Returns final chunk list."""
    cleaned = clean_opinion_text(text)
    if not cleaned:
        return []
    raw_chunks = chunk_text(splitter, cleaned)
    return apply_size_guard(raw_chunks)
