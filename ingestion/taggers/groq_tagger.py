"""
Groq LLM tagging for court opinions.

Model: llama-3.1-8b-instant (free tier)
Rate limit: 20 RPM enforced via sleep; 429 triggers 65s backoff via tenacity.
"""
from __future__ import annotations

import json
import logging
import os
import time

from groq import Groq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

log = logging.getLogger(__name__)

_CALL_INTERVAL = 3.0   # 20 RPM → 3s between calls
_last_call_time = 0.0
_client: Groq | None = None

VALID_OUTCOMES = {"plaintiff_won", "defendant_won", "mixed", "unknown"}

SYSTEM_PROMPT = """\
You are a legal analyst. Given one or more excerpts from a court opinion, return ONLY valid JSON with exactly these keys:
{
  "ai_summary": "<4 sentences: (1) key facts, (2) legal question presented, (3) holding, (4) significance or rule established>",
  "key_concepts": ["<legal concept or statute citation>"],
  "outcome": "plaintiff_won" | "defendant_won" | "mixed" | "unknown",
  "primary_holding_chunk_index": <integer, 0-based index of the chunk most containing the ratio decidendi>
}
Return nothing else. No markdown, no explanation."""


def make_deterministic_summary(case_name: str, court: str, year: int,
                                plain_text: str) -> dict:
    snippet = plain_text[:800].replace("\n", " ").strip()
    return {
        "ai_summary": f"{case_name} ({court}, {year}). {snippet}",
        "key_concepts": [],
        "outcome": "unknown",
        "primary_holding_chunk_index": 0,
    }


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(65),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def tag_opinion(chunks: list[str]) -> dict:
    """Send up to 3 largest chunks to Groq and return structured tags.

    Enforces 20 RPM via blocking sleep before each call. On HTTP 429 or
    any error, tenacity retries up to 3 times with 65s waits.
    """
    global _last_call_time

    elapsed = time.monotonic() - _last_call_time
    if elapsed < _CALL_INTERVAL:
        time.sleep(_CALL_INTERVAL - elapsed)

    selected = sorted(chunks, key=len, reverse=True)[:3]
    user_content = "\n\n---\n\n".join(
        f"[Chunk {i}]:\n{c[:2000]}" for i, c in enumerate(selected)
    )

    _last_call_time = time.monotonic()
    resp = _get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
        max_tokens=512,
    )

    raw = resp.choices[0].message.content
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("Groq returned non-JSON: %s — exc: %s", raw[:200], exc)
        raise

    outcome = result.get("outcome", "unknown")
    if outcome not in VALID_OUTCOMES:
        outcome = "unknown"

    try:
        primary_idx = int(result.get("primary_holding_chunk_index", 0))
    except (TypeError, ValueError):
        primary_idx = 0
    primary_idx = max(0, min(primary_idx, len(chunks) - 1))

    concepts = result.get("key_concepts", [])
    if not isinstance(concepts, list):
        concepts = []

    return {
        "ai_summary": str(result.get("ai_summary", ""))[:1500],
        "key_concepts": [str(c) for c in concepts],
        "outcome": outcome,
        "primary_holding_chunk_index": primary_idx,
    }
