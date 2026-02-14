from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from agent.chunking import (
    chunk_markdown_fixed_window_v1,
    chunk_markdown_obsidian_v1,
    split_frontmatter,
)
from agent.index_db import (
    connect_db,
    count_chunks,
    count_docs,
    get_meta,
    init_db,
    prune_docs_not_in_source,
    replace_doc_chunks,
    set_meta,
    upsert_doc,
    upsert_source,
)
from agent.tools import (
    ToolError,
    get_read_text_file_policy,
    get_workspace_root,
    resolve_and_validate_path,
)

try:
    import yaml as _yaml
except Exception:  # pragma: no cover - fallback path is tested by behavior, not import failure.
    _yaml = None


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: str
    kind: str


@dataclass(frozen=True)
class FrontmatterParseResult:
    yaml_present: int
    yaml_parse_ok: Optional[int]
    yaml_error: Optional[str]
    required_keys_present: Optional[int]
    frontmatter: Dict[str, Any]


@dataclass(frozen=True)
class IndexSummary:
    sources_total: int
    docs_scanned: int
    docs_changed: int
    docs_unchanged: int
    docs_pruned: int
    chunks_written: int
    total_docs: int
    total_chunks: int
    errors: list[str]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _is_within(candidate: Path, base: Path) -> bool:
    try:
        candidate.relative_to(base)
        return True
    except ValueError:
        return False


def _sanitize_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_sanitize_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_jsonable(v) for k, v in value.items()}
    return str(value)


def _parse_scalar_token(token: str) -> Any:
    raw = token.strip()
    if not raw:
        return None
    lower = raw.lower()
    if lower in {"null", "~"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if raw.startswith("'") and raw.endswith("'") and len(raw) >= 2:
        return raw[1:-1]
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return raw[1:-1]
    if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
        try:
            return int(raw)
        except ValueError:
            return raw
    try:
        return float(raw)
    except ValueError:
        return raw


def _fallback_parse_key_values(frontmatter_text: str) -> tuple[Dict[str, Any], int]:
    out: Dict[str, Any] = {}
    unsupported = 0
    for line in frontmatter_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith(" ") or line.startswith("\t"):
            unsupported += 1
            continue
        if ":" not in line:
            unsupported += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            unsupported += 1
            continue
        out[key] = _parse_scalar_token(value)
    return out, unsupported


def _extract_frontmatter_block(text: str) -> tuple[int, str, Optional[str]]:
    if not text:
        return 0, "", None

    raw = text[1:] if text.startswith("\ufeff") else text
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return 0, "", None

    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return 1, "".join(lines[1:i]), None

    return 1, "".join(lines[1:]), "frontmatter block is not closed"


def parse_frontmatter(text: str) -> FrontmatterParseResult:
    yaml_present, block, structural_error = _extract_frontmatter_block(text)
    if yaml_present == 0:
        return FrontmatterParseResult(
            yaml_present=0,
            yaml_parse_ok=None,
            yaml_error=None,
            required_keys_present=None,
            frontmatter={},
        )

    if structural_error is not None:
        partial, _ = _fallback_parse_key_values(block)
        return FrontmatterParseResult(
            yaml_present=1,
            yaml_parse_ok=0,
            yaml_error=structural_error,
            required_keys_present=None,
            frontmatter=partial,
        )

    if _yaml is not None:
        try:
            loaded = _yaml.safe_load(block)
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                return FrontmatterParseResult(
                    yaml_present=1,
                    yaml_parse_ok=0,
                    yaml_error="frontmatter must be a mapping",
                    required_keys_present=None,
                    frontmatter={},
                )
            frontmatter = _sanitize_jsonable(loaded)
            uuid_value = frontmatter.get("uuid")
            if uuid_value is None:
                required = 0
            elif isinstance(uuid_value, str):
                required = 1 if uuid_value.strip() else 0
            else:
                required = 1 if str(uuid_value).strip() else 0
            return FrontmatterParseResult(
                yaml_present=1,
                yaml_parse_ok=1,
                yaml_error=None,
                required_keys_present=required,
                frontmatter=frontmatter,
            )
        except Exception as exc:
            partial, _ = _fallback_parse_key_values(block)
            return FrontmatterParseResult(
                yaml_present=1,
                yaml_parse_ok=0,
                yaml_error=str(exc),
                required_keys_present=None,
                frontmatter=partial,
            )

    partial, unsupported = _fallback_parse_key_values(block)
    parse_ok = 1 if unsupported == 0 else 0
    err = None if parse_ok == 1 else "fallback frontmatter parser could not parse all lines"
    required_value = partial.get("uuid")
    if parse_ok == 1:
        if required_value is None:
            required = 0
        elif isinstance(required_value, str):
            required = 1 if required_value.strip() else 0
        else:
            required = 1 if str(required_value).strip() else 0
    else:
        required = None
    return FrontmatterParseResult(
        yaml_present=1,
        yaml_parse_ok=parse_ok,
        yaml_error=err,
        required_keys_present=required,
        frontmatter=partial,
    )


def parse_sources_from_config(phase2_cfg: Dict[str, Any]) -> list[SourceSpec]:
    raw_sources = phase2_cfg.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("phase2.sources must be a non-empty list")

    specs: list[SourceSpec] = []
    seen_names: set[str] = set()
    for item in raw_sources:
        if not isinstance(item, dict):
            raise ValueError("phase2.sources entries must be objects")
        name = str(item.get("name", "")).strip()
        root = str(item.get("root", "")).strip()
        kind = str(item.get("kind", "")).strip()
        if not name or not root or not kind:
            raise ValueError("phase2.sources entries require name, root, and kind")
        if name in seen_names:
            raise ValueError(f"Duplicate phase2 source name: {name}")
        seen_names.add(name)
        specs.append(SourceSpec(name=name, root=root, kind=kind))
    return specs


def _resolve_source_root(source_root: str, security_root: Path) -> Path:
    p = Path(source_root).expanduser()
    if not p.is_absolute():
        p = security_root / p
    return p.resolve()


def _validate_source_root_allowlisted(source_root: Path) -> None:
    policy = get_read_text_file_policy()
    allowed_roots = [root.resolve() for root in policy.allowed_roots]
    if not allowed_roots:
        raise ValueError("Security policy has no allowlisted roots configured")
    if not any(_is_within(source_root, base) for base in allowed_roots):
        roots_rendered = ", ".join(str(p) for p in allowed_roots)
        raise ValueError(
            f"Source root escapes allowlisted roots: {source_root} (allowed: {roots_rendered})"
        )


def _iter_markdown_files(source_root: Path) -> list[Path]:
    paths = [
        p.resolve()
        for p in source_root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".md"
    ]
    return sorted(paths, key=lambda p: str(p).lower())


def _validated_readable_path(path: Path) -> Path:
    workspace_root = get_workspace_root().resolve()
    policy = get_read_text_file_policy()
    try:
        rel = os.path.relpath(str(path), str(workspace_root))
    except ValueError as exc:
        raise ToolError(
            "PATH_DENIED",
            f"Cannot index path across drives: {path} (workspace_root={workspace_root})",
        ) from exc
    return resolve_and_validate_path(Path(rel).as_posix(), policy)


def _compute_chunker_sig(*, scheme: str, max_chars: int, overlap: int) -> str:
    payload = {
        "scheme": scheme,
        "max_chars": int(max_chars),
        "overlap": int(overlap),
        "chunker_impl": "phase2_chunking_v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()[:32]


def doc_needs_rechunk(
    conn,
    *,
    doc_id: int,
    expected_scheme: str,
) -> bool:
    row = conn.execute("SELECT COUNT(*) AS c FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()
    if row is None or int(row["c"]) == 0:
        return True

    row = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM chunks
        WHERE doc_id = ?
          AND (chunk_key IS NULL OR trim(chunk_key) = '')
        """,
        (doc_id,),
    ).fetchone()
    if row is not None and int(row["c"]) > 0:
        return True

    scheme_rows = conn.execute(
        "SELECT DISTINCT scheme FROM chunks WHERE doc_id = ?",
        (doc_id,),
    ).fetchall()
    schemes = {str(r["scheme"]) for r in scheme_rows if r["scheme"] is not None}
    if not schemes:
        return True
    return schemes != {expected_scheme}


def index_sources(
    *,
    db_path: Path,
    source_specs: Sequence[SourceSpec],
    security_root: Path,
    scheme: str,
    max_chars: int,
    overlap: int,
    force_rebuild: bool = False,
) -> IndexSummary:
    if scheme not in {"fixed_window_v1", "obsidian_v1"}:
        raise ValueError(f"Unsupported chunking scheme: {scheme}")

    init_db(db_path)
    errors: list[str] = []
    docs_scanned = 0
    docs_changed = 0
    docs_unchanged = 0
    docs_pruned = 0
    chunks_written = 0

    with connect_db(db_path) as conn:
        chunker_sig = _compute_chunker_sig(scheme=scheme, max_chars=max_chars, overlap=overlap)
        stored_sig = get_meta(conn, "chunker_sig")
        force_rechunk_all = bool(force_rebuild)
        if stored_sig != chunker_sig:
            force_rechunk_all = True
        set_meta(conn, "chunker_sig", chunker_sig)

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

            source_id = upsert_source(
                conn,
                name=source.name,
                root=str(source_root),
                kind=source.kind,
            )
            seen_rel_paths: set[str] = set()

            for file_path in _iter_markdown_files(source_root):
                rel_path = file_path.relative_to(source_root).as_posix()
                seen_rel_paths.add(rel_path)
                docs_scanned += 1

                try:
                    safe_path = _validated_readable_path(file_path)
                    text = safe_path.read_text(encoding="utf-8", errors="replace")
                    stat = safe_path.stat()
                except Exception as exc:
                    errors.append(f"{source.name}:{rel_path}: {exc}")
                    continue

                parsed = parse_frontmatter(text)
                frontmatter_json = json.dumps(parsed.frontmatter, ensure_ascii=False, sort_keys=True)
                doc_sha = _sha256_text(text)

                try:
                    doc_id, changed = upsert_doc(
                        conn,
                        source_id=source_id,
                        rel_path=rel_path,
                        abs_path=str(safe_path),
                        sha256=doc_sha,
                        mtime=float(stat.st_mtime),
                        size=int(stat.st_size),
                        is_markdown=1,
                        yaml_present=parsed.yaml_present,
                        yaml_parse_ok=parsed.yaml_parse_ok,
                        yaml_error=parsed.yaml_error,
                        required_keys_present=parsed.required_keys_present,
                        frontmatter_json=frontmatter_json,
                    )
                except Exception as exc:
                    errors.append(f"{source.name}:{rel_path}: failed to upsert doc: {exc}")
                    continue

                needs_rechunk = changed or force_rechunk_all
                if (not needs_rechunk) and (not changed):
                    try:
                        needs_rechunk = doc_needs_rechunk(
                            conn,
                            doc_id=doc_id,
                            expected_scheme=scheme,
                        )
                    except Exception as exc:
                        errors.append(f"{source.name}:{rel_path}: failed rechunk sanity check: {exc}")
                        continue

                if needs_rechunk:
                    _, body_text = split_frontmatter(text)
                    if scheme == "fixed_window_v1":
                        chunks = chunk_markdown_fixed_window_v1(
                            body_text=body_text,
                            max_chars=max_chars,
                            overlap=overlap,
                        )
                    else:
                        chunks = chunk_markdown_obsidian_v1(
                            body_text=body_text,
                            max_chars=max_chars,
                            overlap=overlap,
                        )
                    try:
                        written = replace_doc_chunks(
                            conn,
                            doc_id=doc_id,
                            doc_relpath=rel_path,
                            scheme=scheme,
                            chunks=chunks,
                        )
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
                errors.append(f"source '{source.name}': failed to prune removed docs: {exc}")

        conn.commit()
        total_docs = count_docs(conn)
        total_chunks = count_chunks(conn)

    return IndexSummary(
        sources_total=len(source_specs),
        docs_scanned=docs_scanned,
        docs_changed=docs_changed,
        docs_unchanged=docs_unchanged,
        docs_pruned=docs_pruned,
        chunks_written=chunks_written,
        total_docs=total_docs,
        total_chunks=total_chunks,
        errors=errors,
    )
