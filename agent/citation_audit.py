from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.corpus_db import connect_db as connect_corpus_db
from agent.retrieval import RetrievedChunk


@dataclass(frozen=True)
class ChunkAuditRow:
    chunk_key: str
    rel_path: str
    heading_path: str
    chunk_hash: str
    text: str


@dataclass(frozen=True)
class ParsedCitation:
    chunk_key: str
    rel_path: str
    heading_path: str


_CITATION_PATTERN = re.compile(
    r"\[source:\s*(?P<path>[^|\]]+?)\s*\|\s*(?P<chunk_key>[0-9a-f]{32})\s*\]"
)


def fetch_chunk_rows_for_keys(*, corpus_db_path: Path, chunk_keys: list[str]) -> dict[str, ChunkAuditRow]:
    unique_keys = sorted({key for key in chunk_keys if key})
    if not unique_keys:
        return {}
    placeholders = ",".join("?" for _ in unique_keys)
    with connect_corpus_db(corpus_db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                chunks.chunk_key AS chunk_key,
                documents.rel_path AS rel_path,
                chunks.heading_path AS heading_path,
                chunks.chunk_hash AS chunk_hash,
                chunks.text AS chunk_text
            FROM chunks
            INNER JOIN documents ON documents.id = chunks.doc_id
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
            chunk_hash=str(row["chunk_hash"] or ""),
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
                heading_path=_normalize_citation_heading_token(heading_path),
            )
        )
    return out


def count_citation_markers(answer: str) -> int:
    return (answer or "").count("[source:")


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

    for index, item in enumerate(candidates):
        remaining = total_cap - total_chars
        if remaining <= 0:
            logging_truncated_total = True
            omitted_count = len(candidates) - index
            break
        row = chunk_rows.get(item.chunk_key)
        rel_path = row.rel_path if row is not None else item.rel_path
        heading_path = row.heading_path if row is not None else item.heading_path
        chunk_hash = row.chunk_hash if row is not None else ""
        source_text = row.text if row is not None and row.text else item.text
        per_item_cap = min(excerpt_cap, remaining)
        excerpt = source_text[:per_item_cap]
        intended_without_total_cap = min(excerpt_cap, len(source_text))
        if len(excerpt) < intended_without_total_cap:
            logging_truncated_total = True
        total_chars += len(excerpt)
        entries.append(
            {
                "chunk_key": item.chunk_key,
                "rel_path": rel_path,
                "heading_path": heading_path,
                "chunk_anchor": item.chunk_anchor,
                "chunk_title": item.chunk_title,
                "method": item.method,
                "scores": {
                    "merged": float(item.score),
                    "lexical": float(item.lexical_score),
                    "vector": float(item.vector_score),
                },
                "chunk_hash": chunk_hash,
                "excerpt": excerpt,
                "excerpt_truncated": len(source_text) > len(excerpt),
                "excerpt_hash": hashlib.sha256(excerpt.encode("utf-8", errors="replace")).hexdigest(),
            }
        )
    return entries, logging_truncated_total, omitted_count


def validate_citations(
    *,
    parsed_citations: Sequence[ParsedCitation],
    corpus_db_path: Path,
    retrieval_snapshot_hash_by_key: Mapping[str, str],
    enabled: bool,
    strict: bool,
    require_in_snapshot: bool,
    heading_match: str = "prefix",
    normalize_heading: bool = True,
    citation_markers_found: int | None = None,
) -> dict[str, Any]:
    heading_strategy = str(heading_match or "prefix").strip().lower() or "prefix"
    if heading_strategy not in {"exact", "prefix", "ignore"}:
        heading_strategy = "prefix"
    normalize_heading_flag = bool(normalize_heading)
    parsed_payload = [
        {
            "chunk_key": item.chunk_key,
            "rel_path": item.rel_path,
            "heading_path": item.heading_path,
        }
        for item in parsed_citations
    ]
    parsed_citations_count = len(parsed_payload)
    marker_count = (
        max(0, int(citation_markers_found))
        if citation_markers_found is not None
        else parsed_citations_count
    )
    unparseable_citations_count = max(0, marker_count - parsed_citations_count)
    all_citations_unparseable = marker_count > 0 and parsed_citations_count == 0
    if not enabled:
        report = {
            "enabled": False,
            "strict": bool(strict),
            "require_in_snapshot": bool(require_in_snapshot),
            "heading_match_strategy": heading_strategy,
            "normalize_heading": normalize_heading_flag,
            "parsed_citations": parsed_payload,
            "parsed_citations_count": parsed_citations_count,
            "citation_markers_found": marker_count,
            "unparseable_citations_count": unparseable_citations_count,
            "all_citations_unparseable": all_citations_unparseable,
            "not_in_snapshot_chunk_keys": [],
            "missing_chunk_keys": [],
            "mismatched_hash": [],
            "path_mismatches": [],
            "hash_unchecked_chunk_keys": [],
            "valid": True,
        }
        counts = citation_validation_counts(report)
        report["counts"] = counts
        return report

    cited_keys = sorted({item.chunk_key for item in parsed_citations if item.chunk_key})
    db_rows = fetch_chunk_rows_for_keys(corpus_db_path=corpus_db_path, chunk_keys=cited_keys)
    missing_keys: set[str] = set()
    mismatched_hash: list[dict[str, str]] = []
    path_mismatches: list[dict[str, str]] = []
    hash_unchecked: set[str] = set()
    not_in_snapshot: set[str] = set()
    invalid_path_mismatch = False
    seen_hash_mismatch: set[str] = set()
    seen_path_mismatch: set[tuple[str, str, str, str, str, str]] = set()
    snapshot_keys = set(retrieval_snapshot_hash_by_key.keys())

    for citation in parsed_citations:
        if citation.chunk_key not in snapshot_keys:
            not_in_snapshot.add(citation.chunk_key)

        row = db_rows.get(citation.chunk_key)
        if row is None:
            missing_keys.add(citation.chunk_key)
            continue

        expected_rel = row.rel_path
        rel_mismatch = citation.rel_path != expected_rel
        expected_heading_raw = _normalize_citation_heading_token(row.heading_path)
        got_heading_raw = _normalize_citation_heading_token(citation.heading_path)
        heading_invalid = False
        heading_mismatch_reason = ""
        heading_mismatch_for_report = False
        expected_heading_cmp = _canonicalize_heading_path(expected_heading_raw, normalize_heading=normalize_heading_flag)
        got_heading_cmp = _canonicalize_heading_path(got_heading_raw, normalize_heading=normalize_heading_flag)

        if heading_strategy == "ignore":
            heading_mismatch_for_report = expected_heading_raw != got_heading_raw
            if heading_mismatch_for_report:
                heading_mismatch_reason = "ignored"
        else:
            expected_segs = _split_heading_path(expected_heading_raw, normalize_heading=normalize_heading_flag)
            got_segs = _split_heading_path(got_heading_raw, normalize_heading=normalize_heading_flag)
            if heading_strategy == "exact":
                heading_invalid = expected_segs != got_segs
                heading_mismatch_for_report = heading_invalid
                if heading_invalid:
                    heading_mismatch_reason = _heading_mismatch_reason(
                        expected_heading_raw=expected_heading_raw,
                        got_heading_raw=got_heading_raw,
                        expected_segs=expected_segs,
                        got_segs=got_segs,
                        strategy=heading_strategy,
                    )
            else:
                if not got_segs:
                    heading_invalid = bool(expected_segs)
                elif len(got_segs) > len(expected_segs):
                    heading_invalid = True
                else:
                    heading_invalid = expected_segs[: len(got_segs)] != got_segs
                heading_mismatch_for_report = heading_invalid
                if heading_invalid:
                    heading_mismatch_reason = "prefix"

        mismatch_kind = "none"
        if rel_mismatch and heading_mismatch_for_report:
            mismatch_kind = "both"
        elif rel_mismatch:
            mismatch_kind = "rel_path"
        elif heading_mismatch_for_report:
            mismatch_kind = "heading"

        if mismatch_kind != "none":
            mismatch_key = (
                citation.chunk_key,
                expected_rel,
                citation.rel_path,
                expected_heading_cmp,
                got_heading_cmp,
                mismatch_kind,
            )
            if mismatch_key not in seen_path_mismatch:
                path_mismatches.append(
                    {
                        "chunk_key": citation.chunk_key,
                        "expected_rel_path": expected_rel,
                        "got_rel_path": citation.rel_path,
                        "expected_heading_path": expected_heading_raw,
                        "got_heading_path": got_heading_raw,
                        "normalized_expected_heading_path": expected_heading_cmp,
                        "normalized_got_heading_path": got_heading_cmp,
                        "mismatch_kind": mismatch_kind,
                        "heading_invalid": bool(heading_invalid),
                        "heading_match_strategy": heading_strategy,
                        "normalize_heading": normalize_heading_flag,
                        "heading_mismatch_reason": heading_mismatch_reason,
                    }
                )
                seen_path_mismatch.add(mismatch_key)

        if rel_mismatch or heading_invalid:
            invalid_path_mismatch = True

        snapshot_hash = str(retrieval_snapshot_hash_by_key.get(citation.chunk_key) or "")
        if not snapshot_hash:
            hash_unchecked.add(citation.chunk_key)
            continue
        if snapshot_hash != row.chunk_hash and citation.chunk_key not in seen_hash_mismatch:
            mismatched_hash.append(
                {
                    "chunk_key": citation.chunk_key,
                    "expected": row.chunk_hash,
                    "got": snapshot_hash,
                }
            )
            seen_hash_mismatch.add(citation.chunk_key)

    report = {
        "enabled": True,
        "strict": bool(strict),
        "require_in_snapshot": bool(require_in_snapshot),
        "heading_match_strategy": heading_strategy,
        "normalize_heading": normalize_heading_flag,
        "parsed_citations": parsed_payload,
        "parsed_citations_count": parsed_citations_count,
        "citation_markers_found": marker_count,
        "unparseable_citations_count": unparseable_citations_count,
        "all_citations_unparseable": all_citations_unparseable,
        "not_in_snapshot_chunk_keys": sorted(not_in_snapshot),
        "missing_chunk_keys": sorted(missing_keys),
        "mismatched_hash": mismatched_hash,
        "path_mismatches": path_mismatches,
        "hash_unchecked_chunk_keys": sorted(hash_unchecked),
        "valid": (
            not missing_keys
            and not mismatched_hash
            and not invalid_path_mismatch
            and unparseable_citations_count == 0
            and (not require_in_snapshot or not not_in_snapshot)
        ),
    }
    report["counts"] = citation_validation_counts(report)
    return report


def citation_validation_counts(report: Mapping[str, Any]) -> dict[str, int]:
    return {
        "missing": len(list(report.get("missing_chunk_keys") or [])) + int(report.get("unparseable_citations_count") or 0),
        "path_mismatches": len(list(report.get("path_mismatches") or [])),
        "hash_mismatches": len(list(report.get("mismatched_hash") or [])),
        "not_in_snapshot": len(list(report.get("not_in_snapshot_chunk_keys") or [])),
    }


def format_citation_validation_footer(report: Mapping[str, Any]) -> str:
    counts = citation_validation_counts(report)
    return (
        f"(missing={counts['missing']}, path_mismatches={counts['path_mismatches']}, "
        f"hash_mismatches={counts['hash_mismatches']}, not_in_snapshot={counts['not_in_snapshot']})"
    )


def _split_rel_and_heading(raw_path: str) -> tuple[str, str]:
    rel, sep, heading = raw_path.partition("#")
    if not sep:
        return rel.strip(), ""
    return rel.strip(), heading.strip()


def _normalize_citation_heading_token(heading_path: str) -> str:
    heading = (heading_path or "").strip()
    if heading.lower() == "root":
        return ""
    return heading


def _normalize_heading_segment(segment: str, *, normalize_heading: bool) -> str:
    text = str(segment or "").strip()
    if not normalize_heading:
        return text
    text = " ".join(text.split())
    if ":" in text:
        prefix, title = text.split(":", 1)
        prefix = prefix.strip()
        title = " ".join(title.split()).rstrip(":;,.!?").strip()
        if title:
            return f"{prefix}: {title}"
        return prefix.rstrip(":;,.!?")
    return text.rstrip(":;,.!?").strip()


def _split_heading_path(path: str, *, normalize_heading: bool) -> list[str]:
    raw = _normalize_citation_heading_token(path)
    if not raw:
        return []
    parts = [_normalize_heading_segment(segment, normalize_heading=normalize_heading) for segment in raw.split(">")]
    return [part for part in parts if part]


def _canonicalize_heading_path(path: str, *, normalize_heading: bool) -> str:
    return " > ".join(_split_heading_path(path, normalize_heading=normalize_heading))


def _heading_mismatch_reason(
    *,
    expected_heading_raw: str,
    got_heading_raw: str,
    expected_segs: list[str],
    got_segs: list[str],
    strategy: str,
) -> str:
    if strategy == "prefix":
        return "prefix"
    if got_segs and len(got_segs) < len(expected_segs) and expected_segs[: len(got_segs)] == got_segs:
        return "prefix"
    expected_norm = _canonicalize_heading_path(expected_heading_raw, normalize_heading=True)
    got_norm = _canonicalize_heading_path(got_heading_raw, normalize_heading=True)
    expected_raw = _canonicalize_heading_path(expected_heading_raw, normalize_heading=False)
    got_raw = _canonicalize_heading_path(got_heading_raw, normalize_heading=False)
    if expected_norm == got_norm and expected_raw != got_raw:
        return "punctuation"
    return "other"
