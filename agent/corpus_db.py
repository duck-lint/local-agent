from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

from agent.app_types import ChunkRecord, DocumentRecord
from agent.chunking import (
    CHUNK_KIND_METADATA,
    LEXICAL_PROJECTION_VERSION,
    parse_string_field,
    parse_string_list_field,
)

SCHEMA_VERSION = 7


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
        DROP TABLE IF EXISTS chunk_search_fts;
        DROP TABLE IF EXISTS chunk_search;
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
            chunk_kind TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            section_index INTEGER NOT NULL,
            section_ordinal INTEGER,
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

        CREATE TABLE chunk_search (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_key TEXT NOT NULL UNIQUE,
            chunk_kind TEXT NOT NULL,
            rel_path TEXT NOT NULL,
            body_text TEXT NOT NULL,
            chunk_title TEXT NOT NULL,
            heading_path TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            aliases_text TEXT NOT NULL,
            tags_text TEXT NOT NULL,
            note_type TEXT NOT NULL,
            journal_entry_date TEXT,
            layer TEXT NOT NULL,
            register TEXT NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE INDEX idx_documents_source_id ON documents(source_id);
        CREATE INDEX idx_documents_rel_path ON documents(rel_path);
        CREATE INDEX idx_documents_doc_key ON documents(doc_key);
        CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
        CREATE INDEX idx_chunks_doc_key ON chunks(doc_key);
        CREATE INDEX idx_chunks_kind ON chunks(chunk_kind);
        CREATE INDEX idx_chunks_heading_path ON chunks(heading_path);
        CREATE INDEX idx_chunk_search_doc_id ON chunk_search(doc_id);
        CREATE INDEX idx_chunk_search_kind ON chunk_search(chunk_kind);
        """
    )
    conn.execute("PRAGMA user_version = 7")
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


def get_document_by_doc_key(
    conn: sqlite3.Connection,
    *,
    doc_key: str,
) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT documents.id, documents.doc_key, documents.rel_path, sources.name AS source_name
        FROM documents
        INNER JOIN sources ON sources.id = documents.source_id
        WHERE documents.doc_key = ?
        """,
        (doc_key,),
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
                doc_id, chunk_key, doc_key, chunk_kind, chunk_index, section_index, section_ordinal,
                heading_path, chunk_anchor, chunk_title, text, chunk_hash, start_char, end_char,
                out_links_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                chunk.chunk_key,
                chunk.doc_key,
                chunk.chunk_kind,
                int(chunk.chunk_index),
                int(chunk.section_index),
                None if chunk.section_ordinal is None else int(chunk.section_ordinal),
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
) -> list[dict[str, object]]:
    safe_limit = max(1, int(limit))
    raw_query = str(query_text or "").strip()
    if not raw_query:
        return []

    fetch_limit = max(50, safe_limit * 10)
    backend_mode = get_meta(conn, "lexical_backend_mode") or (
        "fts5" if _table_exists(conn, "chunk_search_fts") else "projection_substring"
    )
    backend_warning = get_meta(conn, "lexical_backend_warning") or ""
    actual_backend = backend_mode

    candidates: list[dict[str, object]] = []
    if backend_mode == "fts5":
        fts_query = _build_fts_query(raw_query)
        if fts_query:
            try:
                candidates = _query_chunk_search_fts(conn, fts_query=fts_query, limit=fetch_limit)
            except sqlite3.DatabaseError:
                candidates = []
        if not candidates:
            actual_backend = "projection_substring"
            if not backend_warning:
                backend_warning = "FTS unavailable or yielded no tokenized query; using projection substring fallback."
            candidates = _query_chunk_search_fallback(conn, query_text=raw_query, limit=fetch_limit)
    else:
        actual_backend = "projection_substring"
        candidates = _query_chunk_search_fallback(conn, query_text=raw_query, limit=fetch_limit)

    if not candidates:
        return []

    ranked: list[tuple[tuple[int, int, float, str], dict[str, object]]] = []
    for row in candidates:
        exact_rank, exact_field = _exact_match_info(raw_query, row)
        kind_priority = 0 if exact_field and str(row.get("chunk_kind") or "") == CHUNK_KIND_METADATA else 1
        backend_score = float(row.get("backend_score") or 0.0)
        chunk_key = str(row.get("chunk_key") or "")
        row["lexical_exact_match_field"] = exact_field
        row["lexical_backend_mode"] = actual_backend
        row["lexical_backend_warning"] = backend_warning
        ranked.append(((exact_rank, kind_priority, -backend_score, chunk_key), row))

    ranked.sort(key=lambda item: item[0])
    selected = [row for _, row in ranked[:safe_limit]]
    chunk_rows = _fetch_chunk_rows_for_keys(conn, [str(row.get("chunk_key") or "") for row in selected])
    out: list[dict[str, object]] = []
    for row in selected:
        chunk_key = str(row.get("chunk_key") or "")
        base = chunk_rows.get(chunk_key)
        if base is None:
            continue
        merged = dict(base)
        merged["canonical_name"] = row.get("canonical_name")
        merged["aliases_text"] = row.get("aliases_text")
        merged["tags_text"] = row.get("tags_text")
        merged["note_type"] = row.get("note_type")
        merged["journal_entry_date"] = row.get("journal_entry_date")
        merged["layer"] = row.get("layer")
        merged["register"] = row.get("register")
        merged["lexical_backend_mode"] = row.get("lexical_backend_mode")
        merged["lexical_backend_warning"] = row.get("lexical_backend_warning")
        merged["lexical_backend_score"] = row.get("backend_score")
        merged["lexical_exact_match_field"] = row.get("lexical_exact_match_field")
        out.append(merged)
    return out


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_chunk_search_fts(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search_fts USING fts5(
                chunk_key UNINDEXED,
                body_text,
                chunk_title,
                heading_path,
                rel_path,
                canonical_name,
                aliases_text,
                tags_text,
                note_type,
                journal_entry_date,
                layer,
                register
            )
            """
        )
        return True
    except sqlite3.OperationalError:
        conn.execute("DROP TABLE IF EXISTS chunk_search_fts")
        return False


def rebuild_chunk_search(conn: sqlite3.Connection) -> str:
    conn.execute("DELETE FROM chunk_search")
    use_fts = _ensure_chunk_search_fts(conn)
    if use_fts:
        conn.execute("DELETE FROM chunk_search_fts")

    rows = conn.execute(
        """
        SELECT
            chunks.doc_id AS doc_id,
            chunks.chunk_key AS chunk_key,
            chunks.chunk_kind AS chunk_kind,
            chunks.chunk_title AS chunk_title,
            chunks.heading_path AS heading_path,
            chunks.text AS chunk_text,
            documents.rel_path AS rel_path,
            documents.title AS document_title,
            documents.entry_date AS entry_date,
            documents.frontmatter_json AS frontmatter_json
        FROM chunks
        INNER JOIN documents ON documents.id = chunks.doc_id
        ORDER BY chunks.doc_id ASC, chunks.chunk_index ASC
        """
    ).fetchall()

    ts = time.time()
    projection_rows: list[tuple[object, ...]] = []
    fts_rows: list[tuple[str, str, str, str, str, str, str, str, str, str, str, str]] = []
    for row in rows:
        frontmatter = _load_frontmatter_json(str(row["frontmatter_json"] or ""))
        aliases = "\n".join(parse_string_list_field(frontmatter, "aliases"))
        tags = "\n".join(parse_string_list_field(frontmatter, "tags"))
        chunk_kind = str(row["chunk_kind"] or "")
        if chunk_kind == CHUNK_KIND_METADATA:
            body_text = ""
            chunk_title = ""
            heading_path = ""
            canonical_name = parse_string_field(frontmatter, "canonical_name") or str(row["document_title"] or "")
            note_type = parse_string_field(frontmatter, "note_type")
            journal_entry_date = str(row["entry_date"] or "") or None
            layer = parse_string_field(frontmatter, "layer")
            register = parse_string_field(frontmatter, "register")
        else:
            body_text = str(row["chunk_text"] or "")
            chunk_title = str(row["chunk_title"] or "")
            heading_path = str(row["heading_path"] or "")
            canonical_name = ""
            aliases = ""
            tags = ""
            note_type = ""
            journal_entry_date = None
            layer = ""
            register = ""

        rel_path = str(row["rel_path"] or "")
        projection_rows.append(
            (
                int(row["doc_id"]),
                str(row["chunk_key"]),
                chunk_kind,
                rel_path,
                body_text,
                chunk_title,
                heading_path,
                canonical_name,
                aliases,
                tags,
                note_type,
                journal_entry_date,
                layer,
                register,
                ts,
            )
        )
        if use_fts:
            fts_rows.append(
                (
                    str(row["chunk_key"]),
                    body_text,
                    chunk_title,
                    heading_path,
                    rel_path,
                    canonical_name,
                    aliases,
                    tags,
                    note_type,
                    str(journal_entry_date or ""),
                    layer,
                    register,
                )
            )

    if projection_rows:
        conn.executemany(
            """
            INSERT INTO chunk_search(
                doc_id, chunk_key, chunk_kind, rel_path, body_text, chunk_title, heading_path,
                canonical_name, aliases_text, tags_text, note_type, journal_entry_date, layer, register, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            projection_rows,
        )
    if use_fts and fts_rows:
        conn.executemany(
            """
            INSERT INTO chunk_search_fts(
                chunk_key, body_text, chunk_title, heading_path, rel_path,
                canonical_name, aliases_text, tags_text, note_type, journal_entry_date, layer, register
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fts_rows,
        )

    backend_mode = "fts5" if use_fts else "projection_substring"
    set_meta(conn, "lexical_backend_mode", backend_mode)
    set_meta(
        conn,
        "lexical_backend_warning",
        "" if use_fts else "FTS5 unavailable; using declared projection substring fallback.",
    )
    set_meta(conn, "lexical_projection_version", LEXICAL_PROJECTION_VERSION)
    return backend_mode


def _load_frontmatter_json(raw_value: str) -> dict[str, object]:
    if not raw_value.strip():
        return {}
    try:
        loaded = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _build_fts_query(query_text: str) -> str:
    tokens = [token for token in re.findall(r"\w+", str(query_text or "").lower()) if token]
    if not tokens:
        return ""
    return " AND ".join(f'"{token}"' for token in tokens)


def _query_chunk_search_fts(
    conn: sqlite3.Connection,
    *,
    fts_query: str,
    limit: int,
) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT
            chunk_search.chunk_key AS chunk_key,
            chunk_search.chunk_kind AS chunk_kind,
            chunk_search.rel_path AS rel_path,
            chunk_search.body_text AS body_text,
            chunk_search.chunk_title AS chunk_title,
            chunk_search.heading_path AS heading_path,
            chunk_search.canonical_name AS canonical_name,
            chunk_search.aliases_text AS aliases_text,
            chunk_search.tags_text AS tags_text,
            chunk_search.note_type AS note_type,
            chunk_search.journal_entry_date AS journal_entry_date,
            chunk_search.layer AS layer,
            chunk_search.register AS register,
            (0.0 - bm25(chunk_search_fts, 1.0, 2.5, 2.0, 1.5, 5.0, 4.0, 3.0, 2.0, 1.5, 1.5, 1.5)) AS backend_score
        FROM chunk_search_fts
        INNER JOIN chunk_search ON chunk_search.chunk_key = chunk_search_fts.chunk_key
        WHERE chunk_search_fts MATCH ?
        LIMIT ?
        """,
        (fts_query, max(1, int(limit))),
    ).fetchall()
    return [dict(row) for row in rows]


def _query_chunk_search_fallback(
    conn: sqlite3.Connection,
    *,
    query_text: str,
    limit: int,
) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT
            chunk_key,
            chunk_kind,
            rel_path,
            body_text,
            chunk_title,
            heading_path,
            canonical_name,
            aliases_text,
            tags_text,
            note_type,
            journal_entry_date,
            layer,
            register
        FROM chunk_search
        WHERE
            instr(lower(body_text), lower(?)) > 0
            OR instr(lower(chunk_title), lower(?)) > 0
            OR instr(lower(heading_path), lower(?)) > 0
            OR instr(lower(rel_path), lower(?)) > 0
            OR instr(lower(canonical_name), lower(?)) > 0
            OR instr(lower(aliases_text), lower(?)) > 0
            OR instr(lower(tags_text), lower(?)) > 0
            OR instr(lower(note_type), lower(?)) > 0
            OR instr(lower(journal_entry_date), lower(?)) > 0
            OR instr(lower(layer), lower(?)) > 0
            OR instr(lower(register), lower(?)) > 0
        ORDER BY chunk_key
        LIMIT ?
        """,
        (
            query_text,
            query_text,
            query_text,
            query_text,
            query_text,
            query_text,
            query_text,
            query_text,
            query_text,
            query_text,
            query_text,
            max(1, int(limit)),
        ),
    ).fetchall()
    out: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item["backend_score"] = _fallback_backend_score(query_text, item)
        out.append(item)
    return out


def _fallback_backend_score(query_text: str, row: dict[str, object]) -> float:
    lowered_query = str(query_text or "").strip().lower()
    if not lowered_query:
        return 0.0
    weights = (
        ("canonical_name", 6000.0),
        ("aliases_text", 5500.0),
        ("tags_text", 5250.0),
        ("chunk_title", 5000.0),
        ("heading_path", 4500.0),
        ("rel_path", 4000.0),
        ("note_type", 3500.0),
        ("journal_entry_date", 3250.0),
        ("layer", 3000.0),
        ("register", 2750.0),
        ("body_text", 2500.0),
    )
    best = 0.0
    for field_name, weight in weights:
        haystack = str(row.get(field_name) or "").lower()
        pos = haystack.find(lowered_query)
        if pos < 0:
            continue
        score = weight - min(pos, int(weight) - 1)
        if score > best:
            best = float(score)
    return best


def _normalize_match_text(value: object, *, path: bool = False) -> str:
    text = str(value or "").strip().lower()
    if path:
        text = text.replace("\\", "/")
    return " ".join(text.split())


def _exact_match_info(query_text: str, row: dict[str, object]) -> tuple[int, str]:
    query_norm = _normalize_match_text(query_text)
    if not query_norm:
        return 99, ""
    if _normalize_match_text(row.get("canonical_name")) == query_norm:
        return 0, "canonical_name"
    aliases = [line for line in str(row.get("aliases_text") or "").splitlines() if line.strip()]
    if any(_normalize_match_text(alias) == query_norm for alias in aliases):
        return 1, "aliases"
    tags = [line for line in str(row.get("tags_text") or "").splitlines() if line.strip()]
    if any(_normalize_match_text(tag) == query_norm for tag in tags):
        return 2, "tags"
    if _normalize_match_text(row.get("heading_path")) == query_norm:
        return 3, "heading_path"
    if _normalize_match_text(row.get("rel_path"), path=True) == _normalize_match_text(query_text, path=True):
        return 4, "rel_path"
    if _normalize_match_text(row.get("note_type")) == query_norm:
        return 5, "note_type"
    if _normalize_match_text(row.get("journal_entry_date")) == query_norm:
        return 6, "journal_entry_date"
    if _normalize_match_text(row.get("layer")) == query_norm:
        return 7, "layer"
    if _normalize_match_text(row.get("register")) == query_norm:
        return 8, "register"
    return 99, ""


def _fetch_chunk_rows_for_keys(
    conn: sqlite3.Connection,
    chunk_keys: list[str],
) -> dict[str, dict[str, object]]:
    unique_keys = [key for key in sorted({str(key).strip() for key in chunk_keys if str(key).strip()})]
    if not unique_keys:
        return {}
    placeholders = ",".join("?" for _ in unique_keys)
    rows = conn.execute(
        f"""
        SELECT
            chunks.id AS chunk_id,
            chunks.chunk_key AS chunk_key,
            chunks.doc_key AS doc_key,
            chunks.chunk_kind AS chunk_kind,
            chunks.chunk_index AS chunk_index,
            chunks.section_index AS section_index,
            chunks.heading_path AS heading_path,
            chunks.chunk_anchor AS chunk_anchor,
            chunks.chunk_title AS chunk_title,
            chunks.text AS chunk_text,
            chunks.chunk_hash AS chunk_hash,
            documents.rel_path AS rel_path,
            documents.source_uri AS source_uri,
            documents.title AS document_title,
            documents.folder AS folder,
            documents.doc_type AS doc_type,
            documents.sensitivity AS sensitivity,
            documents.entry_date AS entry_date,
            documents.source_date AS source_date,
            documents.mtime AS mtime,
            documents.frontmatter_json AS frontmatter_json,
            sources.name AS source_name,
            sources.kind AS source_kind
        FROM chunks
        INNER JOIN documents ON documents.id = chunks.doc_id
        INNER JOIN sources ON sources.id = documents.source_id
        WHERE chunks.chunk_key IN ({placeholders})
        """,
        unique_keys,
    ).fetchall()
    return {str(row["chunk_key"]): dict(row) for row in rows}


def count_docs(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
    return int(row["c"]) if row is not None else 0


def count_chunks(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()
    return int(row["c"]) if row is not None else 0


def list_chunk_keys(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT chunk_key FROM chunks ORDER BY chunk_key").fetchall()
    return [str(row["chunk_key"]) for row in rows]


def fetch_existing_chunk_keys(
    conn: sqlite3.Connection,
    chunk_keys: Iterable[str],
) -> set[str]:
    keys = sorted({str(key).strip() for key in chunk_keys if str(key).strip()})
    if not keys:
        return set()
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"SELECT chunk_key FROM chunks WHERE chunk_key IN ({placeholders})",
        keys,
    ).fetchall()
    return {str(row["chunk_key"]) for row in rows}
