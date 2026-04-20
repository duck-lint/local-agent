from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

from agent.app_types import AppConfig, EmbeddingSyncResult, EmbeddingsConfig
from agent.corpus_db import connect_db as connect_corpus_db
from agent.corpus_db import init_db as init_corpus_db
from agent.embedder import Embedder
from agent.embedding_fingerprint import (
    compute_chunk_preprocess_sig,
    compute_embed_sig,
    compute_query_preprocess_sig,
    normalize_vector,
    pack_vector_f32_le,
    preprocess_chunk_text,
)
from agent.embedders.ollama import OllamaEmbedder
from agent.embedders.torch_embedder import TorchEmbedder
from agent.embeddings_db import (
    count_embeddings,
    count_orphan_embeddings,
    connect_db as connect_embeddings_db,
    delete_orphan_embeddings,
    fetch_embeddings_map,
    get_meta as get_embeddings_meta,
    init_db as init_embeddings_db,
    set_meta as set_embeddings_meta,
    upsert_embedding,
)
from agent.manifests import (
    append_manifest_index,
    git_info,
    stable_settings_hash,
    system_info,
    utc_iso,
    utc_now,
    write_run_manifest,
)
from datetime import datetime


def ensure_runtime_dirs(security_root: Path) -> None:
    (security_root / "embeddings" / "db").mkdir(parents=True, exist_ok=True)
    (security_root / "embeddings" / "manifests").mkdir(parents=True, exist_ok=True)
    (security_root / "memory").mkdir(parents=True, exist_ok=True)


def resolve_embeddings_db_path(embeddings_cfg: EmbeddingsConfig, security_root: Path) -> Path:
    candidate = Path(embeddings_cfg.db_path).expanduser()
    if not candidate.is_absolute():
        candidate = security_root / candidate
    return candidate.resolve()


def resolve_memory_db_path(memory_db_path: str, security_root: Path) -> Path:
    candidate = Path(memory_db_path).expanduser()
    if not candidate.is_absolute():
        candidate = security_root / candidate
    return candidate.resolve()


def parse_embed_runtime(embeddings_cfg: EmbeddingsConfig) -> tuple[str, str, str, str, str, int]:
    provider = embeddings_cfg.provider.strip().lower()
    model_id = embeddings_cfg.model_id.strip()
    preprocess_name = embeddings_cfg.preprocess.strip()
    batch_size = int(embeddings_cfg.batch_size)
    if batch_size <= 0:
        raise ValueError("embeddings.batch_size must be > 0")
    if provider not in {"ollama", "torch"}:
        raise ValueError("embeddings.provider must be one of ollama|torch")
    if preprocess_name != "obsidian_v1":
        raise ValueError("embeddings.preprocess must be obsidian_v1")
    computed_chunk_sig = compute_chunk_preprocess_sig(preprocess_name)
    computed_query_sig = compute_query_preprocess_sig(preprocess_name)
    if embeddings_cfg.chunk_preprocess_sig and embeddings_cfg.chunk_preprocess_sig != computed_chunk_sig:
        raise ValueError(
            "embeddings.chunk_preprocess_sig does not match computed signature "
            f"(configured={embeddings_cfg.chunk_preprocess_sig}, computed={computed_chunk_sig})"
        )
    if embeddings_cfg.query_preprocess_sig and embeddings_cfg.query_preprocess_sig != computed_query_sig:
        raise ValueError(
            "embeddings.query_preprocess_sig does not match computed signature "
            f"(configured={embeddings_cfg.query_preprocess_sig}, computed={computed_query_sig})"
        )
    return provider, model_id, preprocess_name, computed_chunk_sig, computed_query_sig, batch_size


def create_embedder(
    *,
    embeddings_cfg: EmbeddingsConfig,
    base_url: str,
    timeout_s: int,
) -> Embedder:
    if embeddings_cfg.provider == "ollama":
        return OllamaEmbedder(base_url=base_url, model_id=embeddings_cfg.model_id, timeout_s=timeout_s)
    torch_cfg = embeddings_cfg.torch
    return TorchEmbedder(
        model_id=embeddings_cfg.model_id,
        local_model_path=torch_cfg.local_model_path,
        cache_dir=torch_cfg.cache_dir,
        device=torch_cfg.device,
        dtype=torch_cfg.dtype,
        batch_size=torch_cfg.batch_size,
        max_length=torch_cfg.max_length,
        pooling=torch_cfg.pooling,
        normalize=torch_cfg.normalize,
        trust_remote_code=torch_cfg.trust_remote_code,
        offline_only=torch_cfg.offline_only,
    )


def load_corpus_chunks(corpus_db_path: Path) -> list[dict[str, str]]:
    init_corpus_db(corpus_db_path)
    with connect_corpus_db(corpus_db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                chunks.chunk_key AS chunk_key,
                chunks.chunk_hash AS chunk_hash,
                chunks.text AS chunk_text,
                chunks.heading_path AS heading_path,
                documents.rel_path AS rel_path
            FROM chunks
            INNER JOIN documents ON documents.id = chunks.doc_id
            ORDER BY chunks.chunk_key
            """
        ).fetchall()
    return [
        {
            "chunk_key": str(row["chunk_key"]),
            "chunk_hash": str(row["chunk_hash"]),
            "chunk_text": str(row["chunk_text"]),
            "heading_path": str(row["heading_path"]),
            "rel_path": str(row["rel_path"]),
        }
        for row in rows
    ]


def _summarize_embedding_drift(
    *,
    chunks: list[dict[str, str]],
    existing_rows: dict[str, Any],
    model_id: str,
    chunk_preprocess_sig: str,
    dim: int,
    rebuild: bool,
) -> tuple[list[dict[str, str]], int, int, int]:
    to_process: list[dict[str, str]] = []
    missing = 0
    outdated = 0
    skipped_ok = 0
    for chunk in chunks:
        key = chunk["chunk_key"]
        row = existing_rows.get(key)
        expected_sig = compute_embed_sig(
            chunk_key=key,
            chunk_sha=chunk["chunk_hash"],
            model_id=model_id,
            dim=dim,
            chunk_preprocess_sig=chunk_preprocess_sig,
        )
        row_missing = row is None
        row_outdated = False
        if not row_missing:
            row_outdated = (
                str(row["embed_sig"]) != expected_sig
                or str(row["model_id"]) != model_id
                or str(row["preprocess_sig"]) != chunk_preprocess_sig
                or int(row["dim"]) != int(dim)
            )
        if rebuild or row_missing or row_outdated:
            to_process.append(chunk)
        if row_missing:
            missing += 1
        elif row_outdated:
            outdated += 1
        else:
            skipped_ok += 1
    if rebuild:
        skipped_ok = 0
    return to_process, missing, outdated, skipped_ok


def _finalize_embedding_run(
    *,
    security_root: Path,
    started_at: datetime,
    start_perf: float,
    app_config: AppConfig,
    rebuild: bool,
    dry_run: bool,
    result: EmbeddingSyncResult,
) -> EmbeddingSyncResult:
    finished_at = utc_now()
    duration_s = time.perf_counter() - start_perf
    settings_payload = {
        "provider": result.provider,
        "model_id": result.model_id,
        "dim": result.dim,
        "chunk_preprocess_sig": result.chunk_preprocess_sig,
        "query_preprocess_sig": result.query_preprocess_sig,
        "embed_runtime_fingerprint": result.embed_runtime_fingerprint,
        "rebuild": bool(rebuild),
        "dry_run": bool(dry_run),
    }
    settings_hash_short, settings_hash_full = stable_settings_hash(settings_payload)
    git_commit, git_dirty = git_info(security_root)
    run_id = f"{started_at.strftime('%Y%m%d_%H%M%S')}_{settings_hash_short}"
    payload = {
        "run_id": run_id,
        "kind": "embed",
        "started_at_utc": utc_iso(started_at),
        "finished_at_utc": utc_iso(finished_at),
        "duration_s": duration_s,
        "settings_for_hash": settings_payload,
        "settings_hash_short": settings_hash_short,
        "settings_hash_full": settings_hash_full,
        "system": system_info(),
        "repo": {"git_commit": git_commit, "git_dirty": git_dirty},
        "input_provenance": {
            "embeddings_db_path_abs": result.embeddings_db_path,
        },
        "outcomes": {
            "total_chunks": result.total_chunks,
            "existing_embeddings": result.existing_embeddings,
            "embeddings_total_before": result.embeddings_total_before,
            "embeddings_total_after": result.embeddings_total_after,
            "orphan_embeddings_before": result.orphan_embeddings_before,
            "orphan_embeddings_pruned": result.orphan_embeddings_pruned,
            "missing": result.missing,
            "outdated": result.outdated,
            "embedded_written": result.embedded_written,
            "skipped_ok": result.skipped_ok,
            "errors_count": len(result.errors),
        },
    }
    manifest_dir = security_root / "embeddings" / "manifests"
    try:
        manifest_path = write_run_manifest(
            manifest_dir=manifest_dir,
            kind="embed",
            settings_hash_short=settings_hash_short,
            finished_at=finished_at,
            payload=payload,
        )
        append_manifest_index(
            manifest_dir,
            {
                "manifest_filename": manifest_path.name,
                "run_id": run_id,
                "kind": "embed",
                "started_at_utc": payload["started_at_utc"],
                "finished_at_utc": payload["finished_at_utc"],
                "duration_s": duration_s,
                "settings_hash_short": settings_hash_short,
                "provider": result.provider,
                "model_id": result.model_id,
                "dim": result.dim,
                "embedded_written": result.embedded_written,
                "missing": result.missing,
                "outdated": result.outdated,
                "orphan_embeddings_pruned": result.orphan_embeddings_pruned,
                "errors_count": len(result.errors),
                "git_commit": git_commit,
                "git_dirty": git_dirty,
            },
        )
    except Exception:
        pass
    return result


def sync_embeddings(
    *,
    app_config: AppConfig,
    security_root: Path,
    corpus_db_path: Path,
    rebuild: bool = False,
    limit: Optional[int] = None,
    dry_run: bool = False,
    prune_orphans: bool = True,
    embedder_factory: Optional[Callable[..., Any]] = None,
) -> EmbeddingSyncResult:
    ensure_runtime_dirs(security_root)
    started_at = utc_now()
    start_perf = time.perf_counter()
    chunks = load_corpus_chunks(corpus_db_path)
    if limit is not None:
        chunks = chunks[: max(0, int(limit))]
    total_chunks = len(chunks)

    provider, model_id, preprocess_name, chunk_preprocess_sig, query_preprocess_sig, batch_size = parse_embed_runtime(
        app_config.embeddings
    )
    embeddings_db_path = resolve_embeddings_db_path(app_config.embeddings, security_root)
    init_embeddings_db(embeddings_db_path)

    chunk_keys = [chunk["chunk_key"] for chunk in chunks]
    with connect_embeddings_db(embeddings_db_path) as embed_conn:
        embeddings_total_before = count_embeddings(embed_conn)
        orphan_embeddings_before = count_orphan_embeddings(embed_conn, chunk_keys)
        stored_provider = get_embeddings_meta(embed_conn, "embed_provider")
        stored_runtime = get_embeddings_meta(embed_conn, "embed_runtime_fingerprint")

    if total_chunks == 0:
        orphan_embeddings_pruned = 0
        embeddings_total_after = embeddings_total_before
        if prune_orphans and not dry_run:
            with connect_embeddings_db(embeddings_db_path) as embed_conn:
                orphan_embeddings_pruned = delete_orphan_embeddings(embed_conn, chunk_keys)
                embeddings_total_after = count_embeddings(embed_conn)
                embed_conn.commit()
        return _finalize_embedding_run(
            security_root=security_root,
            started_at=started_at,
            start_perf=start_perf,
            app_config=app_config,
            rebuild=rebuild,
            dry_run=dry_run,
            result=EmbeddingSyncResult(
                total_chunks=0,
                existing_embeddings=0,
                embeddings_total_before=embeddings_total_before,
                embeddings_total_after=embeddings_total_after,
                orphan_embeddings_before=orphan_embeddings_before,
                orphan_embeddings_pruned=orphan_embeddings_pruned,
                missing=0,
                outdated=0,
                embedded_written=0,
                skipped_ok=0,
                errors=[],
                dim=None,
                provider=provider,
                model_id=model_id,
                embed_runtime_fingerprint="",
                chunk_preprocess_sig=chunk_preprocess_sig,
                query_preprocess_sig=query_preprocess_sig,
                vectors_normalized=True,
                embeddings_db_path=str(embeddings_db_path),
            ),
        )

    factory = embedder_factory
    if factory is None:
        factory = create_embedder
    embedder = factory(
        embeddings_cfg=app_config.embeddings,
        base_url=app_config.ollama_base_url,
        timeout_s=app_config.timeout_s,
    )
    runtime_fingerprint = str(getattr(embedder, "runtime_fingerprint", lambda: "")() or "")

    first = chunks[0]
    probe_text = preprocess_chunk_text(
        rel_path=first["rel_path"],
        heading_path=first["heading_path"],
        chunk_text=first["chunk_text"],
        preprocess_name=preprocess_name,
    )
    probe_vectors = embedder.embed_texts([probe_text])
    if len(probe_vectors) != 1:
        raise ValueError(f"Embedding probe returned unexpected count: {len(probe_vectors)}")
    probe_vector = normalize_vector(probe_vectors[0])
    dim = int(getattr(embedder, "embed_dim", 0)) or len(probe_vector)
    if dim <= 0:
        raise ValueError("Embedding dimension must be > 0")

    if embeddings_total_before > 0 and not rebuild:
        if stored_provider and stored_provider != provider:
            raise RuntimeError(
                "Embedding provider changed; rebuild embeddings to refresh derived state "
                f"(stored={stored_provider}, current={provider})."
            )
        if stored_runtime and runtime_fingerprint and stored_runtime != runtime_fingerprint:
            raise RuntimeError(
                "Embedding runtime changed; rebuild embeddings to refresh derived state "
                "(embed_runtime_fingerprint mismatch)."
            )

    with connect_embeddings_db(embeddings_db_path) as embed_conn:
        orphan_embeddings_pruned = 0
        if prune_orphans and not dry_run:
            orphan_embeddings_pruned = delete_orphan_embeddings(embed_conn, chunk_keys)
        embeddings_total_after = count_embeddings(embed_conn)
        existing_rows = fetch_embeddings_map(embed_conn, chunk_keys)
        existing_embeddings = len(existing_rows)
        embed_conn.commit()

    to_process, missing, outdated, skipped_ok = _summarize_embedding_drift(
        chunks=chunks,
        existing_rows=existing_rows,
        model_id=model_id,
        chunk_preprocess_sig=chunk_preprocess_sig,
        dim=dim,
        rebuild=rebuild,
    )

    if dry_run:
        return _finalize_embedding_run(
            security_root=security_root,
            started_at=started_at,
            start_perf=start_perf,
            app_config=app_config,
            rebuild=rebuild,
            dry_run=dry_run,
            result=EmbeddingSyncResult(
                total_chunks=total_chunks,
                existing_embeddings=existing_embeddings,
                embeddings_total_before=embeddings_total_before,
                embeddings_total_after=embeddings_total_after,
                orphan_embeddings_before=orphan_embeddings_before,
                orphan_embeddings_pruned=orphan_embeddings_pruned,
                missing=missing,
                outdated=outdated,
                embedded_written=0,
                skipped_ok=skipped_ok,
                errors=[],
                dim=dim,
                provider=provider,
                model_id=model_id,
                embed_runtime_fingerprint=runtime_fingerprint,
                chunk_preprocess_sig=chunk_preprocess_sig,
                query_preprocess_sig=query_preprocess_sig,
                vectors_normalized=True,
                embeddings_db_path=str(embeddings_db_path),
            ),
        )

    written = 0
    errors: list[str] = []
    prefetched_by_key = {first["chunk_key"]: probe_vector}
    with connect_embeddings_db(embeddings_db_path) as embed_conn:
        for start in range(0, len(to_process), batch_size):
            batch = to_process[start : start + batch_size]
            texts: list[str] = []
            uncached: list[dict[str, str]] = []
            vectors_by_key: dict[str, list[float]] = {}
            for chunk in batch:
                cached = prefetched_by_key.get(chunk["chunk_key"])
                if cached is not None:
                    vectors_by_key[chunk["chunk_key"]] = cached
                    continue
                texts.append(
                    preprocess_chunk_text(
                        rel_path=chunk["rel_path"],
                        heading_path=chunk["heading_path"],
                        chunk_text=chunk["chunk_text"],
                        preprocess_name=preprocess_name,
                    )
                )
                uncached.append(chunk)
            if texts:
                vectors = embedder.embed_texts(texts)
                if len(vectors) != len(uncached):
                    raise ValueError(
                        f"Embedding batch size mismatch: requested={len(uncached)} got={len(vectors)}"
                    )
                for chunk, vector in zip(uncached, vectors):
                    normalized = normalize_vector(vector)
                    if len(normalized) != dim:
                        raise ValueError(
                            f"Embedding dimension mismatch for chunk {chunk['chunk_key']}: "
                            f"expected={dim} got={len(normalized)}"
                        )
                    vectors_by_key[chunk["chunk_key"]] = normalized

            for chunk in batch:
                key = chunk["chunk_key"]
                vector = vectors_by_key.get(key)
                if vector is None:
                    errors.append(f"Missing vector for chunk {key}")
                    continue
                embed_sig = compute_embed_sig(
                    chunk_key=key,
                    chunk_sha=chunk["chunk_hash"],
                    model_id=model_id,
                    dim=dim,
                    chunk_preprocess_sig=chunk_preprocess_sig,
                )
                upsert_embedding(
                    embed_conn,
                    chunk_key=key,
                    embed_sig=embed_sig,
                    model_id=model_id,
                    dim=dim,
                    preprocess_sig=chunk_preprocess_sig,
                    vector_blob=pack_vector_f32_le(vector),
                )
                written += 1

        set_embeddings_meta(embed_conn, "schema_version", "2")
        set_embeddings_meta(embed_conn, "embed_provider", provider)
        set_embeddings_meta(embed_conn, "embed_model_id", model_id)
        set_embeddings_meta(embed_conn, "embed_dim", str(dim))
        set_embeddings_meta(embed_conn, "embed_runtime_fingerprint", runtime_fingerprint)
        set_embeddings_meta(embed_conn, "chunk_preprocess_sig", chunk_preprocess_sig)
        set_embeddings_meta(embed_conn, "query_preprocess_sig", query_preprocess_sig)
        set_embeddings_meta(embed_conn, "vectors_normalized", "1")
        set_embeddings_meta(embed_conn, "updated_at", str(time.time()))
        embeddings_total_after = count_embeddings(embed_conn)
        embed_conn.commit()

    return _finalize_embedding_run(
        security_root=security_root,
        started_at=started_at,
        start_perf=start_perf,
        app_config=app_config,
        rebuild=rebuild,
        dry_run=dry_run,
        result=EmbeddingSyncResult(
            total_chunks=total_chunks,
            existing_embeddings=existing_embeddings,
            embeddings_total_before=embeddings_total_before,
            embeddings_total_after=embeddings_total_after,
            orphan_embeddings_before=orphan_embeddings_before,
            orphan_embeddings_pruned=orphan_embeddings_pruned,
            missing=missing,
            outdated=outdated,
            embedded_written=written,
            skipped_ok=skipped_ok,
            errors=errors,
            dim=dim,
            provider=provider,
            model_id=model_id,
            embed_runtime_fingerprint=runtime_fingerprint,
            chunk_preprocess_sig=chunk_preprocess_sig,
            query_preprocess_sig=query_preprocess_sig,
            vectors_normalized=True,
            embeddings_db_path=str(embeddings_db_path),
        ),
    )
