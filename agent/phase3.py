from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from agent.embedding_fingerprint import (
    compute_chunk_preprocess_sig,
    compute_embed_sig,
    compute_query_preprocess_sig,
    normalize_vector,
    pack_vector_f32_le,
    preprocess_chunk_text,
)
from agent.embedders.ollama import OllamaEmbedder
from agent.embeddings_db import (
    connect_db as connect_embeddings_db,
    count_embeddings,
    fetch_embeddings_map,
    get_meta as get_embeddings_meta,
    init_db as init_embeddings_db,
    set_meta as set_embeddings_meta,
    upsert_embedding,
)
from agent.index_db import connect_db as connect_index_db
from agent.index_db import init_db as init_index_db


DEFAULT_PHASE3: dict[str, Any] = {
    "embeddings_db_path": "embeddings/db/embeddings.sqlite",
    "embed": {
        "provider": "ollama",
        "model_id": "nomic-embed-text-v1.5",
        "preprocess": "obsidian_v1",
        "chunk_preprocess_sig": "",
        "query_preprocess_sig": "",
        "batch_size": 32,
    },
    "retrieve": {
        "lexical_k": 20,
        "vector_k": 20,
        "vector_fetch_k": 0,
        "rel_path_prefix": "",
        "fusion": "simple_union",
    },
    "memory": {
        "durable_db_path": "memory/durable.sqlite",
        "enabled": True,
    },
}


@dataclass(frozen=True)
class EmbedSummary:
    total_chunks: int
    existing_embeddings: int
    missing: int
    outdated: int
    embedded_written: int
    skipped_ok: int
    errors: list[str]
    dim: Optional[int]
    model_id: str
    chunk_preprocess_sig: str
    query_preprocess_sig: str
    vectors_normalized: bool
    embeddings_db_path: str


def _string(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def build_phase3_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    base = {
        "embeddings_db_path": DEFAULT_PHASE3["embeddings_db_path"],
        "embed": dict(DEFAULT_PHASE3["embed"]),
        "retrieve": dict(DEFAULT_PHASE3["retrieve"]),
        "memory": dict(DEFAULT_PHASE3["memory"]),
    }
    raw = cfg.get("phase3")
    if not isinstance(raw, dict):
        return base

    if "embeddings_db_path" in raw:
        base["embeddings_db_path"] = raw.get("embeddings_db_path")

    raw_embed = raw.get("embed")
    if isinstance(raw_embed, dict):
        embed = dict(base["embed"])
        embed.update(raw_embed)
        base["embed"] = embed

    raw_retrieve = raw.get("retrieve")
    if isinstance(raw_retrieve, dict):
        retrieve = dict(base["retrieve"])
        retrieve.update(raw_retrieve)
        base["retrieve"] = retrieve

    raw_memory = raw.get("memory")
    if isinstance(raw_memory, dict):
        memory = dict(base["memory"])
        memory.update(raw_memory)
        base["memory"] = memory

    return base


def resolve_embeddings_db_path(phase3_cfg: dict[str, Any], security_root: Path) -> Path:
    raw = _string(phase3_cfg.get("embeddings_db_path"), DEFAULT_PHASE3["embeddings_db_path"])
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = security_root / p
    return p.resolve()


def resolve_memory_db_path(phase3_cfg: dict[str, Any], security_root: Path) -> Path:
    memory_cfg = phase3_cfg.get("memory") if isinstance(phase3_cfg.get("memory"), dict) else {}
    raw = _string(memory_cfg.get("durable_db_path"), DEFAULT_PHASE3["memory"]["durable_db_path"])
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = security_root / p
    return p.resolve()


def phase3_memory_enabled(phase3_cfg: dict[str, Any]) -> bool:
    memory_cfg = phase3_cfg.get("memory") if isinstance(phase3_cfg.get("memory"), dict) else {}
    return _as_bool(memory_cfg.get("enabled"), True)


def ensure_phase3_dirs(security_root: Path) -> None:
    (security_root / "embeddings" / "db").mkdir(parents=True, exist_ok=True)
    (security_root / "embeddings" / "manifests").mkdir(parents=True, exist_ok=True)
    (security_root / "memory").mkdir(parents=True, exist_ok=True)
    (security_root / "memory" / "session").mkdir(parents=True, exist_ok=True)


def parse_embed_runtime(
    phase3_cfg: dict[str, Any],
    *,
    model_override: Optional[str],
    batch_size_override: Optional[int],
) -> tuple[str, str, str, str, str, int]:
    embed_cfg = phase3_cfg.get("embed") if isinstance(phase3_cfg.get("embed"), dict) else {}
    provider = _string(embed_cfg.get("provider"), "ollama").lower()
    model_id = _string(model_override or embed_cfg.get("model_id"), "nomic-embed-text-v1.5")
    preprocess_name = _string(embed_cfg.get("preprocess"), "obsidian_v1")
    configured_chunk_sig = _string(embed_cfg.get("chunk_preprocess_sig"), "")
    configured_query_sig = _string(embed_cfg.get("query_preprocess_sig"), "")
    batch_size = _as_int(batch_size_override if batch_size_override is not None else embed_cfg.get("batch_size"), 32)
    if batch_size <= 0:
        raise ValueError("phase3.embed.batch_size must be > 0")

    computed_chunk_sig = compute_chunk_preprocess_sig(preprocess_name)
    computed_query_sig = compute_query_preprocess_sig(preprocess_name)
    if configured_chunk_sig and configured_chunk_sig != computed_chunk_sig:
        raise ValueError(
            "phase3.embed.chunk_preprocess_sig does not match computed signature "
            f"(configured={configured_chunk_sig}, computed={computed_chunk_sig})"
        )
    if configured_query_sig and configured_query_sig != computed_query_sig:
        raise ValueError(
            "phase3.embed.query_preprocess_sig does not match computed signature "
            f"(configured={configured_query_sig}, computed={computed_query_sig})"
        )

    return provider, model_id, preprocess_name, computed_chunk_sig, computed_query_sig, batch_size


def create_embedder(
    *,
    provider: str,
    model_id: str,
    base_url: str,
    timeout_s: int,
):
    if provider != "ollama":
        raise ValueError(f"Unsupported embedding provider: {provider}")
    return OllamaEmbedder(base_url=base_url, model_id=model_id, timeout_s=timeout_s)


def load_phase2_chunks(phase2_db_path: Path) -> list[dict[str, str]]:
    init_index_db(phase2_db_path)
    with connect_index_db(phase2_db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                chunks.chunk_key AS chunk_key,
                chunks.sha256 AS chunk_sha,
                chunks.text AS chunk_text,
                COALESCE(chunks.heading_path, '') AS heading_path,
                docs.rel_path AS rel_path
            FROM chunks
            INNER JOIN docs ON docs.id = chunks.doc_id
            WHERE chunks.chunk_key IS NOT NULL
              AND trim(chunks.chunk_key) != ''
            ORDER BY chunks.chunk_key
            """
        ).fetchall()
    return [
        {
            "chunk_key": str(row["chunk_key"]),
            "chunk_sha": str(row["chunk_sha"]),
            "chunk_text": str(row["chunk_text"]),
            "heading_path": str(row["heading_path"]),
            "rel_path": str(row["rel_path"]),
        }
        for row in rows
    ]


def summarize_embedding_drift(
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
            chunk_sha=chunk["chunk_sha"],
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


def run_embed_phase(
    *,
    cfg: dict[str, Any],
    security_root: Path,
    phase2_db_path: Path,
    phase3_cfg: dict[str, Any],
    model_override: Optional[str] = None,
    rebuild: bool = False,
    batch_size_override: Optional[int] = None,
    json_limit: Optional[int] = None,
    dry_run: bool = False,
    embedder_factory: Optional[Callable[[str, str, str, int], Any]] = None,
) -> EmbedSummary:
    ensure_phase3_dirs(security_root)
    chunks = load_phase2_chunks(phase2_db_path)
    if json_limit is not None:
        chunks = chunks[: max(0, int(json_limit))]
    total_chunks = len(chunks)

    provider, model_id, preprocess_name, chunk_preprocess_sig, query_preprocess_sig, batch_size = parse_embed_runtime(
        phase3_cfg,
        model_override=model_override,
        batch_size_override=batch_size_override,
    )
    embeddings_db_path = resolve_embeddings_db_path(phase3_cfg, security_root)
    init_embeddings_db(embeddings_db_path)

    if total_chunks == 0:
        return EmbedSummary(
            total_chunks=0,
            existing_embeddings=0,
            missing=0,
            outdated=0,
            embedded_written=0,
            skipped_ok=0,
            errors=[],
            dim=None,
            model_id=model_id,
            chunk_preprocess_sig=chunk_preprocess_sig,
            query_preprocess_sig=query_preprocess_sig,
            vectors_normalized=True,
            embeddings_db_path=str(embeddings_db_path),
        )

    timeout_s = _as_int(cfg.get("timeout_s"), 300)
    base_url = _string(cfg.get("ollama_base_url"), "http://127.0.0.1:11434")
    factory = embedder_factory
    if factory is None:
        def _default_factory(p: str, m: str, b: str, t: int):
            return create_embedder(provider=p, model_id=m, base_url=b, timeout_s=t)
        factory = _default_factory
    embedder = factory(provider, model_id, base_url, timeout_s)

    first = chunks[0]
    first_text = preprocess_chunk_text(
        preprocess_name=preprocess_name,
        rel_path=first["rel_path"],
        heading_path=first["heading_path"],
        chunk_text=first["chunk_text"],
    )
    probe_vectors = embedder.embed_texts([first_text])
    if len(probe_vectors) != 1:
        raise ValueError(f"Embedding probe returned unexpected count: {len(probe_vectors)}")
    probe_vector = normalize_vector(probe_vectors[0])
    dim = len(probe_vector)
    if dim <= 0:
        raise ValueError("Embedding dimension must be > 0")

    with connect_embeddings_db(embeddings_db_path) as embed_conn:
        existing_rows = fetch_embeddings_map(embed_conn, [chunk["chunk_key"] for chunk in chunks])
        existing_embeddings = len(existing_rows)

    to_process, missing, outdated, skipped_ok = summarize_embedding_drift(
        chunks=chunks,
        existing_rows=existing_rows,
        model_id=model_id,
        chunk_preprocess_sig=chunk_preprocess_sig,
        dim=dim,
        rebuild=rebuild,
    )

    if dry_run:
        return EmbedSummary(
            total_chunks=total_chunks,
            existing_embeddings=existing_embeddings,
            missing=missing,
            outdated=outdated,
            embedded_written=0,
            skipped_ok=skipped_ok,
            errors=[],
            dim=dim,
            model_id=model_id,
            chunk_preprocess_sig=chunk_preprocess_sig,
            query_preprocess_sig=query_preprocess_sig,
            vectors_normalized=True,
            embeddings_db_path=str(embeddings_db_path),
        )

    prefetched_by_key = {first["chunk_key"]: probe_vector}
    written = 0
    errors: list[str] = []
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
                        preprocess_name=preprocess_name,
                        rel_path=chunk["rel_path"],
                        heading_path=chunk["heading_path"],
                        chunk_text=chunk["chunk_text"],
                    )
                )
                uncached.append(chunk)
            if texts:
                vectors = embedder.embed_texts(texts)
                if len(vectors) != len(uncached):
                    raise ValueError(f"Embedding batch size mismatch: requested={len(uncached)} got={len(vectors)}")
                for chunk, vec in zip(uncached, vectors):
                    normalized_vec = normalize_vector(vec)
                    if len(normalized_vec) != dim:
                        raise ValueError(
                            f"Embedding dimension mismatch for chunk {chunk['chunk_key']}: expected={dim} got={len(normalized_vec)}"
                        )
                    vectors_by_key[chunk["chunk_key"]] = normalized_vec

            for chunk in batch:
                key = chunk["chunk_key"]
                vec = vectors_by_key.get(key)
                if vec is None:
                    errors.append(f"Missing vector for chunk {key}")
                    continue
                embed_sig = compute_embed_sig(
                    chunk_key=key,
                    chunk_sha=chunk["chunk_sha"],
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
                    vector_blob=pack_vector_f32_le(vec),
                )
                written += 1

        set_embeddings_meta(embed_conn, "schema_version", "1")
        set_embeddings_meta(embed_conn, "embed_model_id", model_id)
        set_embeddings_meta(embed_conn, "embed_dim", str(dim))
        set_embeddings_meta(embed_conn, "chunk_preprocess_sig", chunk_preprocess_sig)
        set_embeddings_meta(embed_conn, "query_preprocess_sig", query_preprocess_sig)
        set_embeddings_meta(embed_conn, "vectors_normalized", "1")
        set_embeddings_meta(embed_conn, "updated_at", str(time.time()))

        verify_rows = fetch_embeddings_map(embed_conn, [chunk["chunk_key"] for chunk in to_process])
        verified = 0
        for chunk in to_process:
            row = verify_rows.get(chunk["chunk_key"])
            if row is None:
                continue
            expected_sig = compute_embed_sig(
                chunk_key=chunk["chunk_key"],
                chunk_sha=chunk["chunk_sha"],
                model_id=model_id,
                dim=dim,
                chunk_preprocess_sig=chunk_preprocess_sig,
            )
            if (
                str(row["embed_sig"]) == expected_sig
                and str(row["model_id"]) == model_id
                and str(row["preprocess_sig"]) == chunk_preprocess_sig
                and int(row["dim"]) == dim
            ):
                verified += 1
        if verified != len(to_process):
            raise RuntimeError(f"Embedding write verification failed (verified={verified}, expected={len(to_process)})")
        embed_conn.commit()

    return EmbedSummary(
        total_chunks=total_chunks,
        existing_embeddings=existing_embeddings,
        missing=missing,
        outdated=outdated,
        embedded_written=written,
        skipped_ok=skipped_ok,
        errors=errors,
        dim=dim,
        model_id=model_id,
        chunk_preprocess_sig=chunk_preprocess_sig,
        query_preprocess_sig=query_preprocess_sig,
        vectors_normalized=True,
        embeddings_db_path=str(embeddings_db_path),
    )


def read_embeddings_meta_summary(embeddings_db_path: Path) -> dict[str, Optional[str]]:
    if not embeddings_db_path.exists():
        return {
            "embed_model_id": None,
            "embed_dim": None,
            "chunk_preprocess_sig": None,
            "query_preprocess_sig": None,
            "vectors_normalized": None,
            "schema_version": None,
        }
    with connect_embeddings_db(embeddings_db_path) as conn:
        return {
            "embed_model_id": get_embeddings_meta(conn, "embed_model_id"),
            "embed_dim": get_embeddings_meta(conn, "embed_dim"),
            "chunk_preprocess_sig": get_embeddings_meta(conn, "chunk_preprocess_sig"),
            "query_preprocess_sig": get_embeddings_meta(conn, "query_preprocess_sig"),
            "vectors_normalized": get_embeddings_meta(conn, "vectors_normalized"),
            "schema_version": get_embeddings_meta(conn, "schema_version"),
        }


def embeddings_total_count(embeddings_db_path: Path) -> int:
    if not embeddings_db_path.exists():
        return 0
    with connect_embeddings_db(embeddings_db_path) as conn:
        return count_embeddings(conn)
