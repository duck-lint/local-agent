from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional


SCHEMA_VERSION = 2
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
        if version != SCHEMA_VERSION:
            conn.executescript(
                """
                DROP TABLE IF EXISTS memory_evidence;
                DROP TABLE IF EXISTS memory;
                DROP TABLE IF EXISTS meta;

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
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            set_meta(conn, "schema_version", str(SCHEMA_VERSION))
        # Phase 4 sidecar: created opportunistically; does not bump SCHEMA_VERSION
        # so that test_memory_contract.py's assertion of payload.schema_version == 2
        # continues to hold for the export payload.
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_promotion (
                memory_id TEXT PRIMARY KEY REFERENCES memory(memory_id) ON DELETE CASCADE,
                session_id TEXT NOT NULL,
                triggering_query_ids TEXT NOT NULL,
                evidence_bundle_ids TEXT NOT NULL,
                promoted_by TEXT NOT NULL,
                promoted_at REAL NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_promotion_session ON memory_promotion(session_id);
            """
        )
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
    allowed_chunk_keys: Optional[Iterable[str]] = None,
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
    if allowed_chunk_keys is not None:
        allowed = {str(k).strip() for k in allowed_chunk_keys if str(k).strip()}
        missing = [key for key in keys if key not in allowed]
        if missing:
            rendered = ", ".join(missing)
            raise ValueError(f"memory evidence chunk_keys are not present in the current corpus: {rendered}")

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


def export_memory(
    conn: sqlite3.Connection,
    target_path: Path,
    *,
    corpus_contract_sig: Optional[str] = None,
    valid_chunk_keys: Optional[Iterable[str]] = None,
) -> dict[str, object]:
    checked_against_current_corpus = valid_chunk_keys is not None
    dangling_evidence_chunk_keys: list[str] = []
    if valid_chunk_keys is not None:
        valid = {str(key).strip() for key in valid_chunk_keys if str(key).strip()}
        dangling_evidence_chunk_keys = [key for key in iter_evidence_chunk_keys(conn) if key not in valid]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": time.time(),
        "provenance": {
            "corpus_contract_sig": corpus_contract_sig,
        },
        "validation": {
            "checked_against_current_corpus": checked_against_current_corpus,
            "dangling_evidence_chunk_keys": dangling_evidence_chunk_keys,
        },
        "items": list_memory(conn),
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def iter_evidence_chunk_keys(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT chunk_key FROM memory_evidence ORDER BY chunk_key").fetchall()
    return [str(row["chunk_key"]) for row in rows]


# --- Phase 4: promotion provenance (sidecar table) ---------------------------

ALLOWED_PROMOTED_BY = {"user", "llm_suggested_user_confirmed"}


def add_promoted_memory(
    conn: sqlite3.Connection,
    *,
    memory_type: str,
    content: str,
    chunk_keys: Iterable[str],
    session_id: str,
    triggering_query_ids: Iterable[str],
    evidence_bundle_ids: Iterable[str],
    promoted_by: str,
    payload: Optional[dict] = None,
    allowed_chunk_keys: Optional[Iterable[str]] = None,
    memory_id: Optional[str] = None,
) -> str:
    """Insert a promoted memory record + sidecar provenance row.

    Validates promoted_by; reuses add_memory(source='derived_from_evidence') so
    chunk_key validation, ALLOWED_MEMORY_TYPES checks, and timestamps stay
    consistent with manual entries.
    """
    promoted_by = (promoted_by or "").strip()
    if promoted_by not in ALLOWED_PROMOTED_BY:
        raise ValueError(f"Unsupported promoted_by: {promoted_by!r}")
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id must be non-empty")
    record_id = add_memory(
        conn,
        memory_type=memory_type,
        content=content,
        source="derived_from_evidence",
        chunk_keys=chunk_keys,
        allowed_chunk_keys=allowed_chunk_keys,
        memory_id=memory_id,
    )
    triggering_ids = sorted({str(x).strip() for x in triggering_query_ids if str(x).strip()})
    bundle_ids = sorted({str(x).strip() for x in evidence_bundle_ids if str(x).strip()})
    conn.execute(
        """
        INSERT INTO memory_promotion(
            memory_id, session_id, triggering_query_ids,
            evidence_bundle_ids, promoted_by, promoted_at, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_id,
            sid,
            json.dumps(triggering_ids, ensure_ascii=False),
            json.dumps(bundle_ids, ensure_ascii=False),
            promoted_by,
            time.time(),
            json.dumps(payload or {}, ensure_ascii=False),
        ),
    )
    return record_id


def get_promotion_provenance(
    conn: sqlite3.Connection, memory_id: str
) -> Optional[dict[str, object]]:
    row = conn.execute(
        """
        SELECT memory_id, session_id, triggering_query_ids,
               evidence_bundle_ids, promoted_by, promoted_at, payload_json
        FROM memory_promotion WHERE memory_id = ?
        """,
        (memory_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        triggering_ids = json.loads(row["triggering_query_ids"])
    except (TypeError, ValueError):
        triggering_ids = []
    try:
        bundle_ids = json.loads(row["evidence_bundle_ids"])
    except (TypeError, ValueError):
        bundle_ids = []
    try:
        payload = json.loads(row["payload_json"])
    except (TypeError, ValueError):
        payload = {}
    return {
        "memory_id": str(row["memory_id"]),
        "session_id": str(row["session_id"]),
        "triggering_query_ids": list(triggering_ids),
        "evidence_bundle_ids": list(bundle_ids),
        "promoted_by": str(row["promoted_by"]),
        "promoted_at": float(row["promoted_at"]),
        "payload": payload if isinstance(payload, dict) else {},
    }


def list_promotions_for_session(
    conn: sqlite3.Connection, session_id: str
) -> list[dict[str, object]]:
    rows = conn.execute(
        "SELECT memory_id FROM memory_promotion WHERE session_id = ? ORDER BY promoted_at, memory_id",
        ((session_id or "").strip(),),
    ).fetchall()
    out: list[dict[str, object]] = []
    for row in rows:
        prov = get_promotion_provenance(conn, str(row["memory_id"]))
        if prov is not None:
            out.append(prov)
    return out
