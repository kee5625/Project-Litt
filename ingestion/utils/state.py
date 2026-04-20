import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

STATE_JSON_PATH = Path("state.json")
STATE_DB_PATH = Path("state.db")


def load_state() -> dict:
    if not STATE_JSON_PATH.exists():
        return {}
    with open(STATE_JSON_PATH) as f:
        return json.load(f)


def save_state(state: dict, path: Path = STATE_JSON_PATH) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def init_db(path: Path = STATE_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cl_pages (
            page_num   INTEGER PRIMARY KEY,
            cursor     TEXT,
            n_docs     INTEGER,
            written_at TEXT
        );
        CREATE TABLE IF NOT EXISTS pol_docs (
            subset    TEXT,
            doc_index INTEGER,
            PRIMARY KEY (subset, doc_index)
        );
        CREATE TABLE IF NOT EXISTS statute_files (
            path       TEXT PRIMARY KEY,
            written_at TEXT
        );
        CREATE TABLE IF NOT EXISTS processed_opinions (
            opinion_id   TEXT PRIMARY KEY,
            source       TEXT,
            processed_at TEXT
        );
    """)
    conn.commit()
    return conn


def is_cl_page_written(conn: sqlite3.Connection, page_num: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM cl_pages WHERE page_num = ?", (page_num,)
    ).fetchone()
    return row is not None


def mark_cl_page_written(
    conn: sqlite3.Connection, page_num: int, cursor: str | None, n_docs: int
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO cl_pages (page_num, cursor, n_docs, written_at) VALUES (?, ?, ?, ?)",
        (page_num, cursor, n_docs, _now()),
    )
    conn.commit()


def get_pol_doc_count(conn: sqlite3.Connection, subset: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM pol_docs WHERE subset = ?", (subset,)
    ).fetchone()
    return row[0] if row else 0


def mark_pol_doc_written(conn: sqlite3.Connection, subset: str, doc_index: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO pol_docs (subset, doc_index) VALUES (?, ?)",
        (subset, doc_index),
    )
    conn.commit()


def is_statute_written(conn: sqlite3.Connection, path: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM statute_files WHERE path = ?", (path,)
    ).fetchone()
    return row is not None


def mark_statute_written(conn: sqlite3.Connection, path: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO statute_files (path, written_at) VALUES (?, ?)",
        (path, _now()),
    )
    conn.commit()


def is_opinion_processed(conn: sqlite3.Connection, opinion_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM processed_opinions WHERE opinion_id = ?", (opinion_id,)
    ).fetchone() is not None


def mark_opinion_processed(conn: sqlite3.Connection, opinion_id: str, source: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO processed_opinions (opinion_id, source, processed_at) VALUES (?, ?, ?)",
        (opinion_id, source, _now()),
    )
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
