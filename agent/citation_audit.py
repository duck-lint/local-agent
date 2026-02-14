from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.index_db import connect_db as connect_index_db
from agent.retrieval import RetrievedChunk


@dataclass(frozen=True)
class ChunkAuditRow:
    chunk_key: str
    rel_path: str
    heading_path: str
    chunk_sha: str
    text: str


def fetch_chunk_rows_for_keys(*, index_db_path: Path, chunk_keys: list[str]) -> dict[str, ChunkAuditRow]:
    unique_keys = sorted({key for key in chunk_keys if key})
    if not unique_keys:
        return {}
    placeholders = ",".join("?" for _ in unique_keys)
    with connect_index_db(index_db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                chunks.chunk_key AS chunk_key,
                docs.rel_path AS rel_path,
                COALESCE(chunks.heading_path, '') AS heading_path,
                chunks.sha256 AS chunk_sha,
                chunks.text AS chunk_text
            FROM chunks
            INNER JOIN docs ON docs.id = chunks.doc_id
            WHERE chunks.chunk_key IN ({placeholders})
            """,
            unique_keys,
        ).fetchall()
    out: dict[str, ChunkAuditRow] = {}
    for row in rows:
        chunk_key = str(row["chunk_key"] or "")
        if not chunk_key:
            continue
        out[chunk_key] = ChunkAuditRow(
            chunk_key=chunk_key,
            rel_path=str(row["rel_path"] or ""),
            heading_path=str(row["heading_path"] or ""),
            chunk_sha=str(row["chunk_sha"] or ""),
            text=str(row["chunk_text"] or ""),
        )
    return out


def build_evidence_log_entries(
    *,
    candidates: Sequence[RetrievedChunk],
    chunk_rows: Mapping[str, ChunkAuditRow],
    max_total_chars: int,
    max_excerpt_chars: int,
) -> tuple[list[dict[str, Any]], bool, int]:
    total_cap = max(0, int(max_total_chars))
    excerpt_cap = max(0, int(max_excerpt_chars))

    entries: list[dict[str, Any]] = []
    total_chars = 0
    logging_truncated_total = False
    omitted_count = 0

    for idx, item in enumerate(candidates):
        remaining = total_cap - total_chars
        if remaining <= 0:
            logging_truncated_total = True
            omitted_count = len(candidates) - idx
            break

        row = chunk_rows.get(item.chunk_key)
        rel_path = row.rel_path if row is not None else item.rel_path
        heading_path = row.heading_path if row is not None else item.heading_path
        chunk_sha = row.chunk_sha if row is not None else ""
        source_text = row.text if row is not None and row.text else item.text

        per_item_cap = min(excerpt_cap, remaining)
        excerpt = source_text[:per_item_cap]
        intended_without_total_cap = min(excerpt_cap, len(source_text))
        if len(excerpt) < intended_without_total_cap:
            logging_truncated_total = True

        excerpt_truncated = len(source_text) > len(excerpt)
        total_chars += len(excerpt)
        entries.append(
            {
                "chunk_key": item.chunk_key,
                "rel_path": rel_path,
                "heading_path": heading_path,
                "method": item.method,
                "scores": {
                    "merged": float(item.score),
                    "lexical": float(item.lexical_score),
                    "vector": float(item.vector_score),
                },
                "chunk_sha": chunk_sha,
                "excerpt": excerpt,
                "excerpt_truncated": bool(excerpt_truncated),
                "excerpt_sha": hashlib.sha256(excerpt.encode("utf-8", errors="replace")).hexdigest(),
            }
        )

    return entries, logging_truncated_total, omitted_count
