from __future__ import annotations

import sqlite3
import time
import hashlib
from pathlib import Path
from typing import Iterable, Optional

from agent.chunking import Chunk


SCHEMA_VERSION = 3


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


def _apply_migration_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            root TEXT NOT NULL,
            kind TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY,
            source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            rel_path TEXT NOT NULL,
            abs_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            is_markdown INTEGER NOT NULL,
            yaml_present INTEGER,
            yaml_parse_ok INTEGER,
            yaml_error TEXT,
            required_keys_present INTEGER,
            frontmatter_json TEXT NOT NULL,
            discovered_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(source_id, rel_path)
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            text TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(doc_id, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_docs_sha256 ON docs(sha256);
        CREATE INDEX IF NOT EXISTS idx_docs_source_id ON docs(source_id);
        CREATE INDEX IF NOT EXISTS idx_docs_rel_path ON docs(rel_path);
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
        """
    )
    conn.execute("PRAGMA user_version = 1")


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    out: set[str] = set()
    for row in rows:
        # row[1] is column name for PRAGMA table_info result.
        out.add(str(row[1]))
    return out


def _apply_migration_v2(conn: sqlite3.Connection) -> None:
    chunk_cols = _column_names(conn, "chunks")
    if "scheme" not in chunk_cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN scheme TEXT NOT NULL DEFAULT 'fixed_window_v1'")
    if "heading_path" not in chunk_cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN heading_path TEXT")
    if "chunk_key" not in chunk_cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN chunk_key TEXT")
    conn.execute("PRAGMA user_version = 2")


def _apply_migration_v3(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    chunk_cols = _column_names(conn, "chunks")
    required = {"chunk_key", "sha256", "scheme", "heading_path", "chunk_index"}
    if required.issubset(chunk_cols):
        rows = conn.execute(
            """
            SELECT
                chunks.id AS chunk_id,
                chunks.sha256 AS digest,
                chunks.scheme AS scheme,
                chunks.heading_path AS heading_path,
                chunks.chunk_index AS chunk_index,
                docs.rel_path AS rel_path
            FROM chunks
            INNER JOIN docs ON docs.id = chunks.doc_id
            WHERE chunks.chunk_key IS NULL OR trim(chunks.chunk_key) = ''
            """
        ).fetchall()
        updates: list[tuple[str, int]] = []
        for row in rows:
            rel_path = str(row["rel_path"])
            scheme = str(row["scheme"] or "fixed_window_v1")
            heading_path = str(row["heading_path"] or "")
            digest = str(row["digest"] or "")
            chunk_index = int(row["chunk_index"])
            chunk_key_src = f"{rel_path}|{scheme}|{heading_path}|0|{chunk_index}|{digest}"
            chunk_key = hashlib.sha256(chunk_key_src.encode("utf-8", errors="replace")).hexdigest()[:32]
            updates.append((chunk_key, int(row["chunk_id"])))
        if updates:
            conn.executemany("UPDATE chunks SET chunk_key = ? WHERE id = ?", updates)

    conn.execute("PRAGMA user_version = 3")


def init_db(db_path: Path) -> None:
    with connect_db(db_path) as conn:
        row = conn.execute("PRAGMA user_version").fetchone()
        version = int(row[0]) if row is not None else 0
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Database schema version {version} is newer than supported {SCHEMA_VERSION}"
            )
        if version < 1:
            _apply_migration_v1(conn)
            version = 1
        if version < 2:
            _apply_migration_v2(conn)
            version = 2
        if version < 3:
            _apply_migration_v3(conn)
        conn.commit()


def upsert_source(
    conn: sqlite3.Connection,
    *,
    name: str,
    root: str,
    kind: str,
    now_ts: Optional[float] = None,
) -> int:
    ts = time.time() if now_ts is None else now_ts
    conn.execute(
        """
        INSERT INTO sources(name, root, kind, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            root=excluded.root,
            kind=excluded.kind
        """,
        (name, root, kind, ts),
    )
    row = conn.execute("SELECT id FROM sources WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to upsert source: {name}")
    return int(row["id"])


def get_doc_by_source_rel_path(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    rel_path: str,
) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, sha256
        FROM docs
        WHERE source_id = ? AND rel_path = ?
        """,
        (source_id, rel_path),
    ).fetchone()


def upsert_doc(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    rel_path: str,
    abs_path: str,
    sha256: str,
    mtime: float,
    size: int,
    is_markdown: int,
    yaml_present: Optional[int],
    yaml_parse_ok: Optional[int],
    yaml_error: Optional[str],
    required_keys_present: Optional[int],
    frontmatter_json: str,
    now_ts: Optional[float] = None,
) -> tuple[int, bool]:
    ts = time.time() if now_ts is None else now_ts
    existing = get_doc_by_source_rel_path(conn, source_id=source_id, rel_path=rel_path)
    if existing is None:
        conn.execute(
            """
            INSERT INTO docs(
                source_id, rel_path, abs_path, sha256, mtime, size,
                is_markdown, yaml_present, yaml_parse_ok, yaml_error,
                required_keys_present, frontmatter_json, discovered_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                rel_path,
                abs_path,
                sha256,
                float(mtime),
                int(size),
                int(is_markdown),
                yaml_present,
                yaml_parse_ok,
                yaml_error,
                required_keys_present,
                frontmatter_json,
                ts,
                ts,
            ),
        )
        row = get_doc_by_source_rel_path(conn, source_id=source_id, rel_path=rel_path)
        if row is None:
            raise RuntimeError(f"Failed to insert doc: {rel_path}")
        return int(row["id"]), True

    doc_id = int(existing["id"])
    previous_sha = str(existing["sha256"])
    changed = previous_sha != sha256
    conn.execute(
        """
        UPDATE docs
        SET
            abs_path = ?,
            sha256 = ?,
            mtime = ?,
            size = ?,
            is_markdown = ?,
            yaml_present = ?,
            yaml_parse_ok = ?,
            yaml_error = ?,
            required_keys_present = ?,
            frontmatter_json = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            abs_path,
            sha256,
            float(mtime),
            int(size),
            int(is_markdown),
            yaml_present,
            yaml_parse_ok,
            yaml_error,
            required_keys_present,
            frontmatter_json,
            ts,
            doc_id,
        ),
    )
    return doc_id, changed


def replace_doc_chunks(
    conn: sqlite3.Connection,
    *,
    doc_id: int,
    doc_relpath: str,
    scheme: str,
    chunks: Iterable[Chunk],
    now_ts: Optional[float] = None,
) -> int:
    ts = time.time() if now_ts is None else now_ts
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    count = 0
    for chunk in chunks:
        digest = hashlib.sha256(chunk.text.encode("utf-8", errors="replace")).hexdigest()
        heading_path = chunk.heading_path or ""
        chunk_key_src = (
            f"{doc_relpath}|{scheme}|{heading_path}|"
            f"{chunk.section_ordinal}|{chunk.chunk_ordinal}|{digest}"
        )
        chunk_key = hashlib.sha256(chunk_key_src.encode("utf-8", errors="replace")).hexdigest()[:32]
        conn.execute(
            """
            INSERT INTO chunks(
                doc_id, chunk_index, start_char, end_char, text, sha256,
                scheme, heading_path, chunk_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                int(chunk.chunk_index),
                int(chunk.start_char),
                int(chunk.end_char),
                chunk.text,
                digest,
                scheme,
                chunk.heading_path,
                chunk_key,
                ts,
            ),
        )
        count += 1
    return count


def prune_docs_not_in_source(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    keep_rel_paths: set[str],
) -> int:
    if not keep_rel_paths:
        row = conn.execute("SELECT COUNT(*) AS c FROM docs WHERE source_id = ?", (source_id,)).fetchone()
        count = int(row["c"]) if row is not None else 0
        conn.execute("DELETE FROM docs WHERE source_id = ?", (source_id,))
        return count

    placeholders = ",".join("?" for _ in keep_rel_paths)
    params = [source_id, *sorted(keep_rel_paths)]
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS c
        FROM docs
        WHERE source_id = ?
          AND rel_path NOT IN ({placeholders})
        """,
        params,
    ).fetchone()
    count = int(row["c"]) if row is not None else 0
    conn.execute(
        f"""
        DELETE FROM docs
        WHERE source_id = ?
          AND rel_path NOT IN ({placeholders})
        """,
        params,
    )
    return count


def query_chunks_lexical(
    conn: sqlite3.Connection,
    *,
    query_text: str,
    limit: int = 5,
) -> list[sqlite3.Row]:
    safe_limit = max(1, int(limit))
    return list(
        conn.execute(
            """
            SELECT
                chunks.id AS chunk_id,
                chunks.chunk_index AS chunk_index,
                chunks.start_char AS start_char,
                chunks.end_char AS end_char,
                chunks.text AS chunk_text,
                chunks.scheme AS scheme,
                chunks.heading_path AS heading_path,
                chunks.chunk_key AS chunk_key,
                docs.id AS doc_id,
                docs.rel_path AS rel_path,
                docs.abs_path AS abs_path,
                docs.yaml_present AS yaml_present,
                docs.yaml_parse_ok AS yaml_parse_ok,
                docs.required_keys_present AS required_keys_present,
                docs.yaml_error AS yaml_error,
                docs.frontmatter_json AS frontmatter_json,
                sources.name AS source_name,
                sources.kind AS source_kind
            FROM chunks
            INNER JOIN docs ON docs.id = chunks.doc_id
            INNER JOIN sources ON sources.id = docs.source_id
            WHERE instr(lower(chunks.text), lower(?)) > 0
            ORDER BY
                instr(lower(chunks.text), lower(?)) ASC,
                length(chunks.text) ASC,
                docs.id ASC,
                chunks.chunk_index ASC
            LIMIT ?
            """,
            (query_text, query_text, safe_limit),
        )
    )


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


def count_docs(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM docs").fetchone()
    return int(row["c"]) if row is not None else 0


def count_chunks(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()
    return int(row["c"]) if row is not None else 0
