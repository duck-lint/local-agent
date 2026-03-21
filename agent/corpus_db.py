from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

from agent.app_types import ChunkRecord, DocumentRecord


SCHEMA_VERSION = 4


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


def _drop_existing_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS documents;
        DROP TABLE IF EXISTS docs;
        DROP TABLE IF EXISTS sources;
        DROP TABLE IF EXISTS meta;
        """
    )


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE sources (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            root TEXT NOT NULL,
            kind TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            doc_key TEXT NOT NULL UNIQUE,
            rel_path TEXT NOT NULL,
            source_uri TEXT NOT NULL,
            abs_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            title TEXT NOT NULL,
            folder TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            sensitivity TEXT NOT NULL,
            entry_date TEXT,
            source_date TEXT,
            yaml_present INTEGER NOT NULL,
            yaml_parse_ok INTEGER,
            yaml_error TEXT,
            frontmatter_json TEXT NOT NULL,
            discovered_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(source_id, rel_path)
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_key TEXT NOT NULL UNIQUE,
            doc_key TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            section_index INTEGER NOT NULL,
            heading_path TEXT NOT NULL,
            chunk_anchor TEXT NOT NULL,
            chunk_title TEXT NOT NULL,
            text TEXT NOT NULL,
            chunk_hash TEXT NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            out_links_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(doc_id, chunk_index)
        );

        CREATE INDEX idx_documents_source_id ON documents(source_id);
        CREATE INDEX idx_documents_rel_path ON documents(rel_path);
        CREATE INDEX idx_documents_doc_key ON documents(doc_key);
        CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
        CREATE INDEX idx_chunks_doc_key ON chunks(doc_key);
        CREATE INDEX idx_chunks_heading_path ON chunks(heading_path);
        """
    )
    conn.execute("PRAGMA user_version = 4")
    set_meta(conn, "schema_version", str(SCHEMA_VERSION))


def init_db(db_path: Path) -> None:
    with connect_db(db_path) as conn:
        row = conn.execute("PRAGMA user_version").fetchone()
        version = int(row[0]) if row is not None else 0
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Corpus DB schema version {version} is newer than supported {SCHEMA_VERSION}"
            )
        if version != SCHEMA_VERSION:
            _drop_existing_schema(conn)
            _create_schema(conn)
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
            root = excluded.root,
            kind = excluded.kind
        """,
        (name, root, kind, ts),
    )
    row = conn.execute("SELECT id FROM sources WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise RuntimeError(f"Failed to upsert source: {name}")
    return int(row["id"])


def get_document_by_source_rel_path(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    rel_path: str,
) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, source_hash, doc_key
        FROM documents
        WHERE source_id = ? AND rel_path = ?
        """,
        (source_id, rel_path),
    ).fetchone()


def upsert_document(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    record: DocumentRecord,
    now_ts: Optional[float] = None,
) -> tuple[int, bool]:
    ts = time.time() if now_ts is None else now_ts
    existing = get_document_by_source_rel_path(conn, source_id=source_id, rel_path=record.rel_path)
    frontmatter_json = json.dumps(record.frontmatter, ensure_ascii=False, sort_keys=True, default=str)
    if existing is None:
        conn.execute(
            """
            INSERT INTO documents(
                source_id, doc_key, rel_path, source_uri, abs_path, source_hash, mtime, size,
                title, folder, doc_type, sensitivity, entry_date, source_date,
                yaml_present, yaml_parse_ok, yaml_error, frontmatter_json,
                discovered_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                record.doc_key,
                record.rel_path,
                record.source_uri,
                record.abs_path,
                record.source_hash,
                float(record.mtime),
                int(record.size),
                record.title,
                record.folder,
                record.doc_type,
                record.sensitivity,
                record.entry_date,
                record.source_date,
                int(record.yaml_present),
                record.yaml_parse_ok,
                record.yaml_error,
                frontmatter_json,
                ts,
                ts,
            ),
        )
        row = get_document_by_source_rel_path(conn, source_id=source_id, rel_path=record.rel_path)
        if row is None:
            raise RuntimeError(f"Failed to insert document: {record.rel_path}")
        return int(row["id"]), True

    doc_id = int(existing["id"])
    changed = str(existing["source_hash"]) != record.source_hash or str(existing["doc_key"]) != record.doc_key
    conn.execute(
        """
        UPDATE documents
        SET
            doc_key = ?,
            source_uri = ?,
            abs_path = ?,
            source_hash = ?,
            mtime = ?,
            size = ?,
            title = ?,
            folder = ?,
            doc_type = ?,
            sensitivity = ?,
            entry_date = ?,
            source_date = ?,
            yaml_present = ?,
            yaml_parse_ok = ?,
            yaml_error = ?,
            frontmatter_json = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            record.doc_key,
            record.source_uri,
            record.abs_path,
            record.source_hash,
            float(record.mtime),
            int(record.size),
            record.title,
            record.folder,
            record.doc_type,
            record.sensitivity,
            record.entry_date,
            record.source_date,
            int(record.yaml_present),
            record.yaml_parse_ok,
            record.yaml_error,
            frontmatter_json,
            ts,
            doc_id,
        ),
    )
    return doc_id, changed


def replace_document_chunks(
    conn: sqlite3.Connection,
    *,
    doc_id: int,
    chunks: Iterable[ChunkRecord],
    now_ts: Optional[float] = None,
) -> int:
    ts = time.time() if now_ts is None else now_ts
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    count = 0
    for chunk in chunks:
        conn.execute(
            """
            INSERT INTO chunks(
                doc_id, chunk_key, doc_key, chunk_index, section_index, heading_path, chunk_anchor,
                chunk_title, text, chunk_hash, start_char, end_char, out_links_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                chunk.chunk_key,
                chunk.doc_key,
                int(chunk.chunk_index),
                int(chunk.section_index),
                chunk.heading_path,
                chunk.chunk_anchor,
                chunk.chunk_title,
                chunk.text,
                chunk.chunk_hash,
                int(chunk.start_char),
                int(chunk.end_char),
                json.dumps(chunk.out_links, ensure_ascii=False, sort_keys=True),
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
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM documents WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        count = int(row["c"]) if row is not None else 0
        conn.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))
        return count

    conn.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS tmp_keep_rel_paths(
            rel_path TEXT PRIMARY KEY
        )
        """
    )
    conn.execute("DELETE FROM tmp_keep_rel_paths")
    keep_paths = sorted(str(path) for path in keep_rel_paths)
    batch_size = 900
    for start in range(0, len(keep_paths), batch_size):
        batch = keep_paths[start : start + batch_size]
        conn.executemany(
            "INSERT OR IGNORE INTO tmp_keep_rel_paths(rel_path) VALUES (?)",
            [(path,) for path in batch],
        )

    row = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM documents d
        LEFT JOIN tmp_keep_rel_paths k ON d.rel_path = k.rel_path
        WHERE d.source_id = ?
          AND k.rel_path IS NULL
        """,
        (source_id,),
    ).fetchone()
    count = int(row["c"]) if row is not None else 0
    conn.execute(
        """
        DELETE FROM documents
        WHERE source_id = ?
          AND rel_path IN (
              SELECT d.rel_path
              FROM documents d
              LEFT JOIN tmp_keep_rel_paths k ON d.rel_path = k.rel_path
              WHERE d.source_id = ?
                AND k.rel_path IS NULL
          )
        """,
        (source_id, source_id),
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
                chunks.chunk_key AS chunk_key,
                chunks.chunk_index AS chunk_index,
                chunks.section_index AS section_index,
                chunks.heading_path AS heading_path,
                chunks.chunk_anchor AS chunk_anchor,
                chunks.chunk_title AS chunk_title,
                chunks.text AS chunk_text,
                chunks.chunk_hash AS chunk_hash,
                documents.doc_key AS doc_key,
                documents.rel_path AS rel_path,
                documents.source_uri AS source_uri,
                documents.title AS document_title,
                documents.folder AS folder,
                documents.doc_type AS doc_type,
                documents.sensitivity AS sensitivity,
                documents.entry_date AS entry_date,
                documents.source_date AS source_date,
                documents.frontmatter_json AS frontmatter_json,
                sources.name AS source_name,
                sources.kind AS source_kind
            FROM chunks
            INNER JOIN documents ON documents.id = chunks.doc_id
            INNER JOIN sources ON sources.id = documents.source_id
            WHERE instr(lower(chunks.text), lower(?)) > 0
            ORDER BY
                instr(lower(chunks.text), lower(?)) ASC,
                length(chunks.text) ASC,
                documents.id ASC,
                chunks.chunk_index ASC
            LIMIT ?
            """,
            (query_text, query_text, safe_limit),
        )
    )


def count_docs(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
    return int(row["c"]) if row is not None else 0


def count_chunks(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()
    return int(row["c"]) if row is not None else 0
