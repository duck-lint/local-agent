from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.app_types import AppConfig, DoctorCheck, DoctorReport
from agent.corpus import compute_corpus_contract_sig
from agent.corpus_db import connect_db as connect_corpus_db
from agent.corpus_db import get_meta as get_corpus_meta
from agent.embeddings import (
    create_embedder,
    load_corpus_chunks,
    parse_embed_runtime,
    resolve_embeddings_db_path,
    resolve_memory_db_path,
)
from agent.embedding_fingerprint import compute_embed_sig
from agent.embeddings_db import (
    connect_db as connect_embeddings_db,
    count_embeddings,
    count_orphan_embeddings,
    fetch_embeddings_map,
    get_meta as get_embeddings_meta,
)
from agent.memory_db import connect_db as connect_memory_db
from agent.memory_db import iter_evidence_chunk_keys, init_db as init_memory_db
from agent.ollama_client import ensure_ollama_up
from agent.retrieval import retrieve
from agent.tools import get_read_text_file_policy


def _ok(code: str, message: str) -> DoctorCheck:
    return DoctorCheck(ok=True, code=code, message=message)


def _fail(code: str, message: str, suggested_fix: str | None = None) -> DoctorCheck:
    return DoctorCheck(ok=False, code=code, message=message, suggested_fix=suggested_fix)


def _append_memory_checks(
    *,
    checks: list[DoctorCheck],
    summary: dict[str, object],
    app_config: AppConfig,
    security_root: Path,
    chunk_keys: list[str],
) -> None:
    memory_db_path = resolve_memory_db_path(app_config.memory.db_path, security_root)
    summary["memory_db_path"] = str(memory_db_path)
    if not app_config.memory.enabled:
        checks.append(_ok("DOCTOR_MEMORY_DISABLED", "Memory is disabled; memory checks skipped."))
        return
    init_memory_db(memory_db_path)
    with connect_memory_db(memory_db_path) as mem_conn:
        dangling = [key for key in iter_evidence_chunk_keys(mem_conn) if key not in set(chunk_keys)]
    summary["dangling_memory_evidence"] = len(dangling)
    if dangling:
        checks.append(
            _fail(
                "DOCTOR_MEMORY_DANGLING_EVIDENCE",
                f"Memory references {len(dangling)} chunk keys that are not present in the current corpus.",
                suggested_fix="Reset memory state or reattach evidence under the rebuilt corpus.",
            )
        )
    else:
        checks.append(_ok("DOCTOR_MEMORY_EVIDENCE_OK", "Memory evidence links point at current corpus chunks."))


def _report_ok(checks: list[DoctorCheck], *, require_grounding: bool) -> bool:
    ok = all(check.ok or check.code.endswith("_WARN") for check in checks)
    if require_grounding:
        ok = all(check.ok for check in checks)
    return ok


def run_doctor(
    *,
    app_config: AppConfig,
    security_root: Path,
    corpus_db_path: Path,
    check_ollama: bool = True,
    require_grounding: bool = False,
) -> DoctorReport:
    checks: list[DoctorCheck] = []
    summary: dict[str, object] = {
        "require_grounding": bool(require_grounding),
        "corpus_db_path": str(corpus_db_path),
    }

    policy = get_read_text_file_policy()
    if not policy.allowed_roots:
        checks.append(_fail("DOCTOR_SECURITY_ROOTS_MISSING", "No valid allowlisted roots are configured."))
    else:
        checks.append(_ok("DOCTOR_SECURITY_ROOTS_OK", "Security allowlisted roots are configured."))

    if not corpus_db_path.exists():
        checks.append(
            _fail(
                "DOCTOR_CORPUS_DB_MISSING",
                f"Corpus DB does not exist at {corpus_db_path}.",
                suggested_fix="Run: local-agent index --rebuild --json",
            )
        )
        return DoctorReport(ok=_report_ok(checks, require_grounding=require_grounding), checks=checks, summary=summary)

    try:
        with connect_corpus_db(corpus_db_path) as corpus_conn:
            docs_total = int(corpus_conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"])
            chunks_total = int(corpus_conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"])
            stored_contract_sig = get_corpus_meta(corpus_conn, "corpus_contract_sig") or ""
            chunk_kind_rows = corpus_conn.execute(
                "SELECT chunk_kind, COUNT(*) AS c FROM chunks GROUP BY chunk_kind ORDER BY chunk_kind"
            ).fetchall()
            chunk_search_rows = int(corpus_conn.execute("SELECT COUNT(*) AS c FROM chunk_search").fetchone()["c"])
            stored_metadata_projection_version = get_corpus_meta(corpus_conn, "metadata_projection_version") or ""
            stored_lexical_projection_version = get_corpus_meta(corpus_conn, "lexical_projection_version") or ""
            lexical_backend_mode = get_corpus_meta(corpus_conn, "lexical_backend_mode") or "projection_substring"
            lexical_backend_warning = get_corpus_meta(corpus_conn, "lexical_backend_warning") or ""
    except sqlite3.DatabaseError as exc:
        summary["corpus_db_error"] = str(exc)
        checks.append(
            _fail(
                "DOCTOR_CORPUS_DB_INVALID",
                "Corpus DB exists but does not match the current runtime schema.",
                suggested_fix="Run: local-agent index --rebuild --json",
            )
        )
        return DoctorReport(ok=_report_ok(checks, require_grounding=require_grounding), checks=checks, summary=summary)
    summary["documents_total"] = docs_total
    summary["chunks_total"] = chunks_total
    summary["corpus_contract_sig"] = stored_contract_sig
    summary["chunk_kind_counts"] = {str(row["chunk_kind"]): int(row["c"]) for row in chunk_kind_rows}
    summary["metadata_projection_version"] = stored_metadata_projection_version
    summary["lexical_projection_version"] = stored_lexical_projection_version
    summary["chunk_search_rows"] = chunk_search_rows
    summary["lexical_backend_mode"] = lexical_backend_mode
    summary["lexical_backend_warning"] = lexical_backend_warning
    expected_contract_sig = compute_corpus_contract_sig(
        max_chars=app_config.corpus.max_chars,
        overlap=app_config.corpus.overlap,
    )
    if chunks_total <= 0:
        checks.append(
            _fail(
                "DOCTOR_CORPUS_EMPTY",
                "No indexed chunks were found.",
                suggested_fix="Run: local-agent index --rebuild --json",
            )
        )
    else:
        checks.append(_ok("DOCTOR_CORPUS_READY", f"Corpus index has {chunks_total} chunks across {docs_total} documents."))
    if stored_contract_sig and stored_contract_sig != expected_contract_sig:
        checks.append(
            _fail(
                "DOCTOR_CORPUS_CONTRACT_MISMATCH",
                "Corpus contract signature does not match current configuration.",
                suggested_fix="Run: local-agent index --rebuild --json",
            )
        )
    else:
        checks.append(_ok("DOCTOR_CORPUS_CONTRACT_OK", "Corpus contract matches current configuration."))
    if chunk_search_rows != chunks_total:
        checks.append(
            _fail(
                "DOCTOR_LEXICAL_PROJECTION_MISMATCH",
                f"Lexical projection has {chunk_search_rows} rows for {chunks_total} corpus chunks.",
                suggested_fix="Run: local-agent index --rebuild --json",
            )
        )
    else:
        checks.append(_ok("DOCTOR_LEXICAL_PROJECTION_READY", "Lexical projection rows match current corpus chunks."))
    if lexical_backend_mode == "fts5":
        checks.append(_ok("DOCTOR_LEXICAL_BACKEND_FTS5", "Lexical backend is using FTS5."))
    else:
        message = lexical_backend_warning or "Lexical backend is using projection substring fallback."
        checks.append(_ok("DOCTOR_LEXICAL_BACKEND_FALLBACK", message))

    chunks = load_corpus_chunks(corpus_db_path)
    chunk_keys = [chunk["chunk_key"] for chunk in chunks]
    embeddings_db_path = resolve_embeddings_db_path(app_config.embeddings, security_root)
    summary["embeddings_db_path"] = str(embeddings_db_path)
    if not embeddings_db_path.exists():
        code = "DOCTOR_EMBEDDINGS_MISSING" if require_grounding else "DOCTOR_EMBEDDINGS_MISSING_WARN"
        checks.append(
            _fail(
                code,
                f"Embeddings DB does not exist at {embeddings_db_path}.",
                suggested_fix="Run: local-agent embed --json",
            )
        )
        _append_memory_checks(
            checks=checks,
            summary=summary,
            app_config=app_config,
            security_root=security_root,
            chunk_keys=chunk_keys,
        )
        return DoctorReport(ok=_report_ok(checks, require_grounding=require_grounding), checks=checks, summary=summary)

    provider, model_id, preprocess_name, chunk_preprocess_sig, query_preprocess_sig, _ = parse_embed_runtime(
        app_config.embeddings
    )
    try:
        with connect_embeddings_db(embeddings_db_path) as embed_conn:
            embeddings_total = count_embeddings(embed_conn)
            existing_rows = fetch_embeddings_map(embed_conn, chunk_keys)
            orphan_embeddings = count_orphan_embeddings(embed_conn, chunk_keys)
            stored_provider = get_embeddings_meta(embed_conn, "embed_provider") or provider
            stored_model_id = get_embeddings_meta(embed_conn, "embed_model_id") or model_id
            stored_chunk_preprocess_sig = get_embeddings_meta(embed_conn, "chunk_preprocess_sig") or chunk_preprocess_sig
            stored_query_preprocess_sig = get_embeddings_meta(embed_conn, "query_preprocess_sig") or query_preprocess_sig
    except sqlite3.DatabaseError as exc:
        summary["embeddings_db_error"] = str(exc)
        checks.append(
            _fail(
                "DOCTOR_EMBEDDINGS_DB_INVALID",
                "Embeddings DB exists but does not match the current runtime schema.",
                suggested_fix="Run: local-agent embed --rebuild --json",
            )
        )
        _append_memory_checks(
            checks=checks,
            summary=summary,
            app_config=app_config,
            security_root=security_root,
            chunk_keys=chunk_keys,
        )
        return DoctorReport(ok=_report_ok(checks, require_grounding=require_grounding), checks=checks, summary=summary)
    summary["embeddings_total"] = embeddings_total
    summary["orphan_embeddings"] = orphan_embeddings
    summary["embed_provider"] = stored_provider
    summary["embed_model_id"] = stored_model_id

    missing_embeddings = 0
    outdated_embeddings = 0
    for chunk in chunks:
        row = existing_rows.get(chunk["chunk_key"])
        if row is None:
            missing_embeddings += 1
            continue
        expected_sig = compute_embed_sig(
            chunk_key=chunk["chunk_key"],
            chunk_sha=chunk["chunk_hash"],
            model_id=model_id,
            dim=int(row["dim"]),
            chunk_preprocess_sig=chunk_preprocess_sig,
        )
        if (
            str(row["embed_sig"]) != expected_sig
            or str(row["model_id"]) != model_id
            or str(row["preprocess_sig"]) != chunk_preprocess_sig
        ):
            outdated_embeddings += 1

    summary["missing_embeddings"] = missing_embeddings
    summary["outdated_embeddings"] = outdated_embeddings
    if stored_provider != provider or stored_model_id != model_id:
        checks.append(
            _fail(
                "DOCTOR_EMBED_CONFIG_MISMATCH",
                "Stored embedding provider/model does not match current configuration.",
                suggested_fix="Run: local-agent embed --rebuild --json",
            )
        )
    else:
        checks.append(_ok("DOCTOR_EMBED_CONFIG_OK", "Embedding provider and model match current configuration."))

    if stored_chunk_preprocess_sig != chunk_preprocess_sig or stored_query_preprocess_sig != query_preprocess_sig:
        checks.append(
            _fail(
                "DOCTOR_EMBED_PREPROCESS_MISMATCH",
                "Stored embedding preprocess signatures do not match current configuration.",
                suggested_fix="Run: local-agent embed --rebuild --json",
            )
        )
    else:
        checks.append(_ok("DOCTOR_EMBED_PREPROCESS_OK", "Embedding preprocess signatures match current configuration."))

    if missing_embeddings > 0:
        code = "DOCTOR_EMBEDDINGS_MISSING_CHUNKS" if require_grounding else "DOCTOR_EMBEDDINGS_MISSING_CHUNKS_WARN"
        checks.append(
            _fail(
                code,
                f"Missing embeddings for {missing_embeddings} corpus chunks.",
                suggested_fix="Run: local-agent embed --json",
            )
        )
    else:
        checks.append(_ok("DOCTOR_EMBEDDINGS_COMPLETE", "Embeddings exist for all indexed chunks."))

    if outdated_embeddings > 0:
        code = "DOCTOR_EMBEDDINGS_OUTDATED" if require_grounding else "DOCTOR_EMBEDDINGS_OUTDATED_WARN"
        checks.append(
            _fail(
                code,
                f"{outdated_embeddings} embeddings do not match current chunk hashes or preprocess settings.",
                suggested_fix="Run: local-agent embed --rebuild --json",
            )
        )
    else:
        checks.append(_ok("DOCTOR_EMBEDDINGS_FRESH", "Embeddings match current chunk hashes and preprocess settings."))

    if orphan_embeddings > 0:
        code = "DOCTOR_EMBEDDINGS_ORPHANED" if require_grounding else "DOCTOR_EMBEDDINGS_ORPHANED_WARN"
        checks.append(
            _fail(
                code,
                f"{orphan_embeddings} embedding rows are no longer referenced by the corpus.",
                suggested_fix="Run: local-agent embed --rebuild --json",
            )
        )
    else:
        checks.append(_ok("DOCTOR_EMBEDDINGS_ORPHANED_OK", "No orphaned embedding rows were found."))

    if check_ollama:
        try:
            ensure_ollama_up(app_config.ollama_base_url, timeout_s=app_config.timeout_s)
            checks.append(_ok("DOCTOR_OLLAMA_OK", f"Ollama reachable at {app_config.ollama_base_url}."))
            retrieval_ready = missing_embeddings == 0 and outdated_embeddings == 0 and orphan_embeddings == 0 and chunks_total > 0
            if retrieval_ready:
                embedder = create_embedder(
                    embeddings_cfg=app_config.embeddings,
                    base_url=app_config.ollama_base_url,
                    timeout_s=app_config.timeout_s,
                )
                result = retrieve(
                    "diagnostic retrieval smoke",
                    corpus_db_path=corpus_db_path,
                    embeddings_db_path=embeddings_db_path,
                    embedder=embedder,
                    embed_model_id=model_id,
                    preprocess_name=preprocess_name,
                    chunk_preprocess_sig=chunk_preprocess_sig,
                    query_preprocess_sig=query_preprocess_sig,
                    lexical_k=min(5, app_config.retrieval.lexical_k),
                    vector_k=min(5, app_config.retrieval.vector_k),
                    vector_fetch_k=app_config.retrieval.vector_fetch_k,
                    rel_path_prefix=app_config.retrieval.rel_path_prefix,
                    fusion=app_config.retrieval.fusion,
                )
                if result.candidates:
                    checks.append(_ok("DOCTOR_RETRIEVAL_READY", "Retrieval smoke returned candidates."))
                else:
                    checks.append(
                        _fail(
                            "DOCTOR_RETRIEVAL_NOT_READY",
                            "Retrieval smoke returned no candidates.",
                            suggested_fix="Rebuild corpus and embeddings, then rerun doctor.",
                        )
                    )
        except Exception as exc:
            checks.append(_fail("DOCTOR_OLLAMA_UNREACHABLE", str(exc)))
    else:
        checks.append(_ok("DOCTOR_OLLAMA_SKIPPED", "Ollama network checks were skipped."))

    _append_memory_checks(
        checks=checks,
        summary=summary,
        app_config=app_config,
        security_root=security_root,
        chunk_keys=chunk_keys,
    )

    ok = _report_ok(checks, require_grounding=require_grounding)
    return DoctorReport(ok=ok, checks=checks, summary=summary)
