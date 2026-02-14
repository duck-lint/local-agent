from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional


SCHEMA_VERSION = 1


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
                f"Embeddings DB schema version {version} is newer than supported {SCHEMA_VERSION}"
            )
        if version < 1:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_key TEXT PRIMARY KEY,
                    embed_sig TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    preprocess_sig TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    embedded_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_id);
                CREATE INDEX IF NOT EXISTS idx_embeddings_preprocess ON embeddings(preprocess_sig);
                CREATE INDEX IF NOT EXISTS idx_embeddings_dim ON embeddings(dim);
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


def upsert_embedding(
    conn: sqlite3.Connection,
    *,
    chunk_key: str,
    embed_sig: str,
    model_id: str,
    dim: int,
    preprocess_sig: str,
    vector_blob: bytes,
    embedded_at: Optional[float] = None,
) -> None:
    ts = time.time() if embedded_at is None else float(embedded_at)
    conn.execute(
        """
        INSERT INTO embeddings(
            chunk_key, embed_sig, model_id, dim, preprocess_sig, vector, embedded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_key) DO UPDATE SET
            embed_sig = excluded.embed_sig,
            model_id = excluded.model_id,
            dim = excluded.dim,
            preprocess_sig = excluded.preprocess_sig,
            vector = excluded.vector,
            embedded_at = excluded.embedded_at
        """,
        (
            chunk_key,
            embed_sig,
            model_id,
            int(dim),
            preprocess_sig,
            sqlite3.Binary(vector_blob),
            ts,
        ),
    )


def count_embeddings(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM embeddings").fetchone()
    return int(row["c"]) if row is not None else 0


def fetch_embeddings_map(conn: sqlite3.Connection, chunk_keys: Iterable[str]) -> dict[str, sqlite3.Row]:
    keys = sorted(set(str(k) for k in chunk_keys if str(k)))
    if not keys:
        return {}
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"SELECT * FROM embeddings WHERE chunk_key IN ({placeholders})",
        keys,
    ).fetchall()
    return {str(r["chunk_key"]): r for r in rows}


def iter_embeddings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM embeddings ORDER BY chunk_key"))
