from __future__ import annotations

import hashlib
import re
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


@dataclass(frozen=True)
class ParsedCitation:
    chunk_key: str
    rel_path: str
    heading_path: str


_CITATION_PATTERN = re.compile(
    r"\[source:\s*(?P<path>[^|\]]+?)\s*\|\s*(?P<chunk_key>[0-9a-f]{32})\s*\]"
)


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


def parse_citations(answer: str) -> list[ParsedCitation]:
    out: list[ParsedCitation] = []
    for match in _CITATION_PATTERN.finditer(answer or ""):
        raw_path = str(match.group("path") or "").strip()
        chunk_key = str(match.group("chunk_key") or "").strip()
        if not raw_path or not chunk_key:
            continue
        rel_path, heading_path = _split_rel_and_heading(raw_path)
        out.append(
            ParsedCitation(
                chunk_key=chunk_key,
                rel_path=rel_path,
                heading_path=_normalize_heading(heading_path),
            )
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


def validate_citations(
    *,
    parsed_citations: Sequence[ParsedCitation],
    index_db_path: Path,
    retrieval_snapshot_sha_by_key: Mapping[str, str],
    enabled: bool,
    strict: bool,
) -> dict[str, Any]:
    parsed_payload = [
        {
            "chunk_key": item.chunk_key,
            "rel_path": item.rel_path,
            "heading_path": item.heading_path,
        }
        for item in parsed_citations
    ]
    if not enabled:
        return {
            "enabled": False,
            "strict": bool(strict),
            "parsed_citations": parsed_payload,
            "missing_chunk_keys": [],
            "mismatched_sha": [],
            "path_mismatches": [],
            "sha_unchecked_chunk_keys": [],
            "valid": True,
        }

    cited_keys = sorted({item.chunk_key for item in parsed_citations if item.chunk_key})
    db_rows = fetch_chunk_rows_for_keys(index_db_path=index_db_path, chunk_keys=cited_keys)

    missing_keys: set[str] = set()
    mismatched_sha: list[dict[str, str]] = []
    path_mismatches: list[dict[str, str]] = []
    sha_unchecked: set[str] = set()
    seen_sha_mismatch: set[str] = set()

    for citation in parsed_citations:
        row = db_rows.get(citation.chunk_key)
        if row is None:
            missing_keys.add(citation.chunk_key)
            continue

        expected_rel = row.rel_path
        expected_heading = _normalize_heading(row.heading_path)
        if citation.rel_path != expected_rel or _normalize_heading(citation.heading_path) != expected_heading:
            path_mismatches.append(
                {
                    "chunk_key": citation.chunk_key,
                    "expected_rel_path": expected_rel,
                    "got_rel_path": citation.rel_path,
                    "expected_heading_path": expected_heading,
                    "got_heading_path": _normalize_heading(citation.heading_path),
                }
            )

        snapshot_sha = str(retrieval_snapshot_sha_by_key.get(citation.chunk_key) or "")
        if not snapshot_sha:
            sha_unchecked.add(citation.chunk_key)
            continue
        if snapshot_sha != row.chunk_sha and citation.chunk_key not in seen_sha_mismatch:
            mismatched_sha.append(
                {
                    "chunk_key": citation.chunk_key,
                    "expected": row.chunk_sha,
                    "got": snapshot_sha,
                }
            )
            seen_sha_mismatch.add(citation.chunk_key)

    return {
        "enabled": True,
        "strict": bool(strict),
        "parsed_citations": parsed_payload,
        "missing_chunk_keys": sorted(missing_keys),
        "mismatched_sha": mismatched_sha,
        "path_mismatches": path_mismatches,
        "sha_unchecked_chunk_keys": sorted(sha_unchecked),
        "valid": not missing_keys and not mismatched_sha and not path_mismatches,
    }


def _split_rel_and_heading(raw_path: str) -> tuple[str, str]:
    rel, sep, heading = raw_path.partition("#")
    if not sep:
        return rel.strip(), ""
    return rel.strip(), heading.strip()


def _normalize_heading(heading_path: str) -> str:
    heading = (heading_path or "").strip()
    if heading.lower() == "root":
        return ""
    return heading
