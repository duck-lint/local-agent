from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional


SCHEMA_VERSION = 1
ALLOWED_MEMORY_TYPES = {
    "preference",
    "project_state",
    "constraint",
    "workflow_rule",
    "user_fact",
}
ALLOWED_MEMORY_SOURCES = {"manual", "derived_from_evidence"}


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), factory=_ClosingConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    with connect_db(db_path) as conn:
        row = conn.execute("PRAGMA user_version").fetchone()
        version = int(row[0]) if row is not None else 0
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Memory DB schema version {version} is newer than supported {SCHEMA_VERSION}"
            )
        if version < 1:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory (
                    memory_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_evidence (
                    memory_id TEXT NOT NULL REFERENCES memory(memory_id) ON DELETE CASCADE,
                    chunk_key TEXT NOT NULL,
                    PRIMARY KEY(memory_id, chunk_key)
                );

                CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(type);
                CREATE INDEX IF NOT EXISTS idx_memory_source ON memory(source);
                CREATE INDEX IF NOT EXISTS idx_memory_evidence_chunk_key ON memory_evidence(chunk_key);
                """
            )
            conn.execute("PRAGMA user_version = 1")
            set_meta(conn, "schema_version", str(SCHEMA_VERSION))
        conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    value = row["value"]
    return str(value) if value is not None else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def add_memory(
    conn: sqlite3.Connection,
    *,
    memory_type: str,
    content: str,
    source: str,
    chunk_keys: Iterable[str],
    memory_id: Optional[str] = None,
) -> str:
    memory_type = memory_type.strip()
    source = source.strip()
    content = content.strip()
    if memory_type not in ALLOWED_MEMORY_TYPES:
        raise ValueError(f"Unsupported memory type: {memory_type}")
    if source not in ALLOWED_MEMORY_SOURCES:
        raise ValueError(f"Unsupported memory source: {source}")
    if not content:
        raise ValueError("memory content must be non-empty")

    keys = sorted({str(k).strip() for k in chunk_keys if str(k).strip()})
    if source == "derived_from_evidence" and not keys:
        raise ValueError("derived_from_evidence memory requires at least one chunk_key")

    now = time.time()
    record_id = memory_id or str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO memory(memory_id, type, content, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (record_id, memory_type, content, source, now, now),
    )
    for key in keys:
        conn.execute(
            """
            INSERT INTO memory_evidence(memory_id, chunk_key)
            VALUES (?, ?)
            ON CONFLICT(memory_id, chunk_key) DO NOTHING
            """,
            (record_id, key),
        )
    return record_id


def delete_memory(conn: sqlite3.Connection, memory_id: str) -> bool:
    cur = conn.execute("DELETE FROM memory WHERE memory_id = ?", (memory_id,))
    return int(cur.rowcount) > 0


def list_memory(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT memory_id, type, content, source, created_at, updated_at
        FROM memory
        ORDER BY updated_at DESC, memory_id
        """
    ).fetchall()

    out: list[dict[str, object]] = []
    for row in rows:
        memory_id = str(row["memory_id"])
        evidence_rows = conn.execute(
            "SELECT chunk_key FROM memory_evidence WHERE memory_id = ? ORDER BY chunk_key",
            (memory_id,),
        ).fetchall()
        out.append(
            {
                "memory_id": memory_id,
                "type": str(row["type"]),
                "content": str(row["content"]),
                "source": str(row["source"]),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
                "chunk_keys": [str(e["chunk_key"]) for e in evidence_rows],
            }
        )
    return out


def export_memory(conn: sqlite3.Connection, target_path: Path) -> dict[str, object]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": time.time(),
        "items": list_memory(conn),
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def iter_evidence_chunk_keys(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT chunk_key FROM memory_evidence ORDER BY chunk_key").fetchall()
    return [str(row["chunk_key"]) for row in rows]
