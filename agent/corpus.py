from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from agent.app_types import ChunkRecord, CorpusConfig, CorpusSyncResult, DocumentRecord, SourceConfig
from agent.chunking import (
    CHUNK_KIND_CONTENT,
    CHUNK_KIND_METADATA,
    LEXICAL_PROJECTION_VERSION,
    METADATA_CHUNK_ANCHOR,
    METADATA_CHUNK_INDEX,
    METADATA_CHUNK_TITLE,
    METADATA_HEADING_PATH,
    METADATA_PROJECTION_VERSION,
    METADATA_SECTION_INDEX,
    build_markdown_chunks,
    build_metadata_projection,
    canonicalize_source_uri,
    infer_document_title,
    normalize_doc_type,
    parse_date_field,
    parse_source_date,
    parse_yaml_frontmatter,
    sha256_text,
    split_frontmatter,
    stable_chunk_key,
    stable_doc_key_from_rel_path,
    split_into_sections,
)
from agent.corpus_db import (
    connect_db,
    count_chunks,
    count_docs,
    get_meta,
    init_db,
    prune_docs_not_in_source,
    query_chunks_lexical,
    rebuild_chunk_search,
    replace_document_chunks,
    set_meta,
    upsert_document,
    upsert_source,
)
from agent.tools import (
    ToolError,
    get_read_text_file_policy,
    get_workspace_root,
    resolve_and_validate_path,
)


def compute_corpus_contract_sig(*, max_chars: int, overlap: int) -> str:
    payload = {
        "chunk_profile": "obsidian_v1",
        "chunk_impl": "vault_corpus_v2_metadata",
        "max_chars": int(max_chars),
        "overlap": int(overlap),
        "metadata_chunk_policy": {
            "enabled": True,
            "kind": CHUNK_KIND_METADATA,
            "heading_path": METADATA_HEADING_PATH,
            "anchor": METADATA_CHUNK_ANCHOR,
            "title": METADATA_CHUNK_TITLE,
            "chunk_index": METADATA_CHUNK_INDEX,
            "section_index": METADATA_SECTION_INDEX,
            "projection_version": METADATA_PROJECTION_VERSION,
        },
        "lexical_projection_version": LEXICAL_PROJECTION_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()[:32]


def _is_within(candidate: Path, base: Path) -> bool:
    try:
        candidate.relative_to(base)
        return True
    except ValueError:
        return False


def _resolve_source_root(source_root: str, security_root: Path) -> Path:
    candidate = Path(source_root).expanduser()
    if not candidate.is_absolute():
        candidate = security_root / candidate
    return candidate.resolve()


def _validate_source_root_allowlisted(source_root: Path) -> None:
    policy = get_read_text_file_policy()
    allowed_roots = [root.resolve() for root in policy.allowed_roots]
    if not allowed_roots:
        raise ValueError("Security policy has no allowlisted roots configured")
    if not any(_is_within(source_root, base) for base in allowed_roots):
        rendered = ", ".join(str(path) for path in allowed_roots)
        raise ValueError(
            f"Source root escapes allowlisted roots: {source_root} (allowed: {rendered})"
        )


def _iter_markdown_files(source_root: Path) -> list[Path]:
    paths = [
        path.resolve()
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".md"
    ]
    return sorted(paths, key=lambda path: str(path).lower())


def _validated_readable_path(path: Path) -> Path:
    workspace_root = get_workspace_root().resolve()
    policy = get_read_text_file_policy()
    try:
        rel = os.path.relpath(str(path), str(workspace_root))
    except ValueError as exc:
        raise ToolError(
            "PATH_DENIED",
            f"Cannot ingest path across drives: {path} (workspace_root={workspace_root})",
        ) from exc
    return resolve_and_validate_path(Path(rel).as_posix(), policy)


def _frontmatter_status(yaml_block: str) -> tuple[int, Optional[int], Optional[str], dict[str, object]]:
    if not yaml_block.strip():
        return 0, None, None, {}
    frontmatter, error = parse_yaml_frontmatter(yaml_block)
    if error is not None:
        return 1, 0, error, frontmatter
    return 1, 1, None, frontmatter


def _document_record_from_file(
    *,
    source_name: str,
    rel_path: str,
    safe_path: Path,
    text: str,
    max_chars: int,
    overlap: int,
) -> tuple[DocumentRecord, list[ChunkRecord]]:
    yaml_block, body = split_frontmatter(text)
    yaml_present, yaml_parse_ok, yaml_error, frontmatter = _frontmatter_status(yaml_block)
    raw_doc_key = str(frontmatter.get("uuid") or "").strip()
    doc_key = raw_doc_key or stable_doc_key_from_rel_path(rel_path, source_name=source_name)
    source_uri = canonicalize_source_uri(rel_path)
    chunk_source_uri = canonicalize_source_uri(f"{source_name}/{rel_path}") if source_name else source_uri
    source_hash = sha256_text(text)
    sections = split_into_sections(body)
    title = infer_document_title(frontmatter, rel_path, sections)
    folder = Path(rel_path).parts[0] if Path(rel_path).parts else ""
    entry_date = parse_date_field(frontmatter, "journal_entry_date")
    source_date = parse_source_date(frontmatter, Path(rel_path).name)
    doc_type = normalize_doc_type(frontmatter, folder=folder, entry_date=entry_date)
    sensitivity = str(frontmatter.get("sensitivity") or "").strip() or "private"
    stat = safe_path.stat()

    metadata_projection = build_metadata_projection(
        meta=frontmatter,
        document_title=title,
        doc_type=doc_type,
        entry_date=entry_date,
        source_date=source_date,
    )
    body_chunks = build_markdown_chunks(body_text=body, max_chars=max_chars, overlap=overlap)
    chunk_records: list[ChunkRecord] = []
    chunk_records.append(
        ChunkRecord(
            chunk_key=stable_chunk_key(
                source_uri=chunk_source_uri,
                chunk_kind=CHUNK_KIND_METADATA,
                heading_path=[METADATA_HEADING_PATH],
                section_index=METADATA_SECTION_INDEX,
                chunk_index=METADATA_CHUNK_INDEX,
            ),
            doc_key=doc_key,
            chunk_kind=CHUNK_KIND_METADATA,
            chunk_index=METADATA_CHUNK_INDEX,
            section_index=METADATA_SECTION_INDEX,
            heading_path=METADATA_HEADING_PATH,
            chunk_anchor=METADATA_CHUNK_ANCHOR,
            chunk_title=METADATA_CHUNK_TITLE,
            text=metadata_projection.text,
            chunk_hash=sha256_text(metadata_projection.text),
            start_char=0,
            end_char=0,
            out_links=[],
        )
    )
    for draft in body_chunks:
        chunk_records.append(
            ChunkRecord(
                chunk_key=stable_chunk_key(
                    source_uri=chunk_source_uri,
                    chunk_kind=CHUNK_KIND_CONTENT,
                    heading_path=draft.heading_path.split(" > ") if draft.heading_path else [],
                    section_index=draft.section_index,
                    chunk_index=draft.chunk_index,
                ),
                doc_key=doc_key,
                chunk_kind=CHUNK_KIND_CONTENT,
                chunk_index=draft.chunk_index,
                section_index=draft.section_index,
                heading_path=draft.heading_path,
                chunk_anchor=draft.chunk_anchor,
                chunk_title=draft.chunk_title,
                text=draft.text,
                chunk_hash=sha256_text(draft.text),
                start_char=draft.start_char,
                end_char=draft.end_char,
                out_links=draft.out_links,
            )
        )

    document = DocumentRecord(
        doc_key=doc_key,
        source_name=source_name,
        rel_path=rel_path,
        source_uri=source_uri,
        source_hash=source_hash,
        abs_path=str(safe_path),
        title=title,
        folder=folder,
        doc_type=doc_type,
        sensitivity=sensitivity,
        entry_date=entry_date,
        source_date=source_date,
        frontmatter={str(key): value for key, value in frontmatter.items()},
        yaml_present=yaml_present,
        yaml_parse_ok=yaml_parse_ok,
        yaml_error=yaml_error,
        mtime=float(stat.st_mtime),
        size=int(stat.st_size),
    )
    return document, chunk_records


def _document_requires_reingest(conn, *, doc_id: int) -> bool:
    row = conn.execute("SELECT COUNT(*) AS c FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()
    return row is None or int(row["c"]) == 0


def sync_corpus(
    *,
    db_path: Path,
    source_specs: list[SourceConfig],
    security_root: Path,
    corpus_config: CorpusConfig,
    force_rebuild: bool = False,
) -> CorpusSyncResult:
    init_db(db_path)
    errors: list[str] = []
    docs_scanned = 0
    docs_changed = 0
    docs_unchanged = 0
    docs_pruned = 0
    chunks_written = 0
    contract_sig = compute_corpus_contract_sig(
        max_chars=corpus_config.max_chars,
        overlap=corpus_config.overlap,
    )

    with connect_db(db_path) as conn:
        stored_sig = get_meta(conn, "corpus_contract_sig")
        force_refresh_all = bool(force_rebuild or stored_sig != contract_sig)
        set_meta(conn, "corpus_contract_sig", contract_sig)
        set_meta(conn, "chunk_profile", "obsidian_v1")
        set_meta(conn, "metadata_projection_version", METADATA_PROJECTION_VERSION)
        set_meta(conn, "lexical_projection_version", LEXICAL_PROJECTION_VERSION)

        for source in source_specs:
            source_root = _resolve_source_root(source.root, security_root.resolve())
            if not source_root.exists() or not source_root.is_dir():
                errors.append(f"source '{source.name}' root does not exist or is not a directory: {source_root}")
                continue
            try:
                _validate_source_root_allowlisted(source_root)
            except Exception as exc:
                errors.append(f"source '{source.name}' denied: {exc}")
                continue

            source_id = upsert_source(conn, name=source.name, root=str(source_root), kind=source.kind)
            seen_rel_paths: set[str] = set()

            for file_path in _iter_markdown_files(source_root):
                rel_path = file_path.relative_to(source_root).as_posix()
                seen_rel_paths.add(rel_path)
                docs_scanned += 1

                try:
                    safe_path = _validated_readable_path(file_path)
                    text = safe_path.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    errors.append(f"{source.name}:{rel_path}: {exc}")
                    continue

                try:
                    document, chunks = _document_record_from_file(
                        source_name=source.name,
                        rel_path=rel_path,
                        safe_path=safe_path,
                        text=text,
                        max_chars=corpus_config.max_chars,
                        overlap=corpus_config.overlap,
                    )
                except Exception as exc:
                    errors.append(f"{source.name}:{rel_path}: failed to prepare document: {exc}")
                    continue

                try:
                    doc_id, changed = upsert_document(conn, source_id=source_id, record=document)
                except Exception as exc:
                    errors.append(f"{source.name}:{rel_path}: failed to upsert document: {exc}")
                    continue

                needs_reingest = changed or force_refresh_all
                if not needs_reingest:
                    needs_reingest = _document_requires_reingest(conn, doc_id=doc_id)

                if needs_reingest:
                    try:
                        written = replace_document_chunks(conn, doc_id=doc_id, chunks=chunks)
                        chunks_written += written
                        docs_changed += 1
                    except Exception as exc:
                        errors.append(f"{source.name}:{rel_path}: failed to write chunks: {exc}")
                else:
                    docs_unchanged += 1

            try:
                docs_pruned += prune_docs_not_in_source(
                    conn,
                    source_id=source_id,
                    keep_rel_paths=seen_rel_paths,
                )
            except Exception as exc:
                errors.append(f"source '{source.name}': failed to prune removed documents: {exc}")

        try:
            rebuild_chunk_search(conn)
        except Exception as exc:
            errors.append(f"lexical projection rebuild failed: {exc}")
        conn.commit()
        total_docs = count_docs(conn)
        total_chunks = count_chunks(conn)

    return CorpusSyncResult(
        sources_total=len(source_specs),
        docs_scanned=docs_scanned,
        docs_changed=docs_changed,
        docs_unchanged=docs_unchanged,
        docs_pruned=docs_pruned,
        chunks_written=chunks_written,
        total_docs=total_docs,
        total_chunks=total_chunks,
        errors=errors,
        corpus_db_path=str(db_path),
        corpus_contract_sig=contract_sig,
    )


def lexical_query(
    *,
    db_path: Path,
    query_text: str,
    limit: int = 5,
) -> list[dict[str, object]]:
    init_db(db_path)
    with connect_db(db_path) as conn:
        rows = query_chunks_lexical(conn, query_text=query_text, limit=limit)
    return list(rows)
