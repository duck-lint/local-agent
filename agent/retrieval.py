from __future__ import annotations

from array import array
import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agent.embedder import Embedder
from agent.embedding_fingerprint import (
    normalize_vector,
    preprocess_query_text,
    unpack_vector_f32_le,
)
from agent.embeddings_db import connect_db as connect_embeddings_db
from agent.index_db import connect_db as connect_index_db
from agent.index_db import get_meta as get_index_meta
from agent.index_db import query_chunks_lexical

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _np = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_key: str
    rel_path: str
    heading_path: str
    text: str
    score: float
    method: str
    lexical_score: float
    vector_score: float


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunker_sig: str
    embed_model_id: str
    chunk_preprocess_sig: str
    query_preprocess_sig: str
    embed_db_schema_version: int
    vector_fetch_k_used: int
    vector_candidates_scored: int
    vector_candidates_prefilter: int
    vector_candidates_postfilter: int
    rel_path_prefix_applied: bool
    vector_filter_warning: str
    candidates: list[RetrievedChunk]


def retrieve(
    query: str,
    *,
    index_db_path: Path,
    embeddings_db_path: Path,
    embedder: Embedder,
    embed_model_id: str,
    preprocess_name: str,
    chunk_preprocess_sig: str,
    query_preprocess_sig: str,
    lexical_k: int,
    vector_k: int,
    vector_fetch_k: int = 0,
    rel_path_prefix: str = "",
    fusion: str,
) -> RetrievalResult:
    if fusion != "simple_union":
        raise ValueError(f"Unsupported fusion strategy: {fusion}")

    lexical_limit = max(1, int(lexical_k))
    vector_limit = max(1, int(vector_k))
    prefix = rel_path_prefix.replace("\\", "/").strip()
    prefix_applied = bool(prefix)
    configured_fetch_k = int(vector_fetch_k)
    if prefix_applied:
        fetch_k_used = max(1, configured_fetch_k) if configured_fetch_k > 0 else max(50, vector_limit * 5)
    else:
        fetch_k_used = vector_limit

    with connect_index_db(index_db_path) as index_conn:
        lexical_rows = [dict(row) for row in query_chunks_lexical(index_conn, query_text=query, limit=lexical_limit)]
        chunker_sig = get_index_meta(index_conn, "chunker_sig") or ""

    lexical_ranked: dict[str, float] = {}
    lexical_meta: dict[str, dict[str, str]] = {}
    lexical_count = max(1, len(lexical_rows))
    for rank, row in enumerate(lexical_rows, start=1):
        chunk_key = str(row.get("chunk_key") or "")
        if not chunk_key:
            continue
        score = 1.0 - ((rank - 1) / lexical_count)
        lexical_ranked[chunk_key] = max(score, lexical_ranked.get(chunk_key, 0.0))
        lexical_meta[chunk_key] = {
            "rel_path": str(row.get("rel_path") or ""),
            "heading_path": str(row.get("heading_path") or ""),
            "text": str(row.get("chunk_text") or ""),
        }

    query_input = preprocess_query_text(query=query, preprocess_name=preprocess_name)
    query_vectors = embedder.embed_texts([query_input])
    if not query_vectors:
        raise ValueError("Embedder returned no query vector")
    query_vector = normalize_vector(query_vectors[0])
    query_dim = len(query_vector)
    if query_dim <= 0:
        raise ValueError("Query vector dimension must be > 0")

    scored, scored_count, vector_normalized = _compute_vector_candidates(
        embeddings_db_path=embeddings_db_path,
        query_vector=query_vector,
        query_dim=query_dim,
        model_id=embed_model_id,
        chunk_preprocess_sig=chunk_preprocess_sig,
        fetch_k=fetch_k_used,
    )
    prefilter_count = len(scored)

    metadata_rows = (
        _fetch_chunk_metadata(index_db_path=index_db_path, chunk_keys=[key for _, key in scored])
        if scored
        else {}
    )
    filtered_scored = [(score, key) for score, key in scored if key in metadata_rows]
    orphan_dropped = prefilter_count - len(filtered_scored)

    warning_parts: list[str] = []
    if prefix_applied and filtered_scored:
        allowed = {key for key, row in metadata_rows.items() if row["rel_path"].replace("\\", "/").startswith(prefix)}
        filtered_scored = [(score, key) for score, key in filtered_scored if key in allowed]
        if len(filtered_scored) < vector_limit:
            warning_parts.append(
                f"rel_path_prefix reduced vector results: {len(filtered_scored)}/{vector_limit} "
                f"(fetched {fetch_k_used})"
            )
    if orphan_dropped > 0:
        warning_parts.append(f"dropped orphan vector candidates: {orphan_dropped}")
    filter_warning = "; ".join(warning_parts)
    vector_top = filtered_scored[:vector_limit]
    vector_ranked = {chunk_key: (score + 1.0) / 2.0 for score, chunk_key in vector_top}

    with connect_embeddings_db(embeddings_db_path) as embed_conn:
        row = embed_conn.execute("PRAGMA user_version").fetchone()
        embed_schema_version = int(row[0]) if row is not None else 0

    merged = _fuse_candidates(
        index_db_path=index_db_path,
        lexical_ranked=lexical_ranked,
        lexical_meta=lexical_meta,
        vector_ranked=vector_ranked,
    )
    return RetrievalResult(
        query=query,
        chunker_sig=chunker_sig,
        embed_model_id=embed_model_id,
        chunk_preprocess_sig=chunk_preprocess_sig,
        query_preprocess_sig=query_preprocess_sig,
        embed_db_schema_version=embed_schema_version,
        vector_fetch_k_used=fetch_k_used,
        vector_candidates_scored=scored_count,
        vector_candidates_prefilter=prefilter_count,
        vector_candidates_postfilter=len(filtered_scored),
        rel_path_prefix_applied=prefix_applied,
        vector_filter_warning=filter_warning,
        candidates=merged,
    )


def _compute_vector_candidates(
    *,
    embeddings_db_path: Path,
    query_vector: list[float],
    query_dim: int,
    model_id: str,
    chunk_preprocess_sig: str,
    fetch_k: int,
) -> tuple[list[tuple[float, str]], int, bool]:
    fetch_limit = max(1, int(fetch_k))
    scored_heap: list[tuple[float, str, str]] = []
    scored_count = 0
    vector_normalized = True

    q_np = None
    q_arr: Optional[array] = None
    q_norm = _l2_norm(query_vector)
    if _np is not None:
        q_np = _np.asarray(query_vector, dtype=_np.float32)
    else:
        q_arr = array("f", query_vector)

    with connect_embeddings_db(embeddings_db_path) as embed_conn:
        vectors_row = embed_conn.execute("SELECT value FROM meta WHERE key = 'vectors_normalized'").fetchone()
        vectors_normalized_meta = str(vectors_row["value"]) if vectors_row is not None else "0"
        vector_normalized = vectors_normalized_meta == "1"
        rows = embed_conn.execute(
            """
            SELECT chunk_key, vector
            FROM embeddings
            WHERE model_id = ? AND preprocess_sig = ? AND dim = ?
            """,
            (model_id, chunk_preprocess_sig, int(query_dim)),
        )
        for row in rows:
            blob = bytes(row["vector"])
            if len(blob) != query_dim * 4:
                continue
            if q_np is not None and _np is not None:
                vec_np = _np.frombuffer(blob, dtype=_np.float32)
                if vector_normalized:
                    score = float(_np.dot(q_np, vec_np))
                else:
                    denom = float(_np.linalg.norm(vec_np)) * max(q_norm, 1e-12)
                    score = float(_np.dot(q_np, vec_np) / denom) if denom > 0.0 else 0.0
            else:
                vec_arr = unpack_vector_f32_le(blob)
                if vector_normalized:
                    score = _dot_array(q_arr, vec_arr)
                else:
                    denom = _l2_norm_arr(vec_arr) * max(q_norm, 1e-12)
                    score = _dot_array(q_arr, vec_arr) / denom if denom > 0.0 else 0.0
            if not math.isfinite(score):
                continue
            scored_count += 1
            chunk_key = str(row["chunk_key"])
            pair = (score, _reverse_chunk_key(chunk_key), chunk_key)
            if len(scored_heap) < fetch_limit:
                heapq.heappush(scored_heap, pair)
            else:
                if pair > scored_heap[0]:
                    heapq.heapreplace(scored_heap, pair)
    ranked = sorted([(score, chunk_key) for score, _, chunk_key in scored_heap], key=lambda item: (-item[0], item[1]))
    return ranked, scored_count, vector_normalized


def _fetch_chunk_metadata(*, index_db_path: Path, chunk_keys: list[str]) -> dict[str, dict[str, str]]:
    unique_keys = sorted({key for key in chunk_keys if key})
    if not unique_keys:
        return {}
    placeholders = ",".join("?" for _ in unique_keys)
    with connect_index_db(index_db_path) as index_conn:
        rows = index_conn.execute(
            f"""
            SELECT
                chunks.chunk_key AS chunk_key,
                docs.rel_path AS rel_path,
                COALESCE(chunks.heading_path, '') AS heading_path,
                chunks.text AS chunk_text
            FROM chunks
            INNER JOIN docs ON docs.id = chunks.doc_id
            WHERE chunks.chunk_key IN ({placeholders})
            """,
            unique_keys,
        ).fetchall()
    return {
        str(row["chunk_key"]): {
            "rel_path": str(row["rel_path"]),
            "heading_path": str(row["heading_path"]),
            "text": str(row["chunk_text"]),
        }
        for row in rows
    }


def _fuse_candidates(
    *,
    index_db_path: Path,
    lexical_ranked: dict[str, float],
    lexical_meta: dict[str, dict[str, str]],
    vector_ranked: dict[str, float],
) -> list[RetrievedChunk]:
    all_keys = sorted(set(lexical_ranked) | set(vector_ranked))
    fetched = _fetch_chunk_metadata(index_db_path=index_db_path, chunk_keys=all_keys)
    out: list[RetrievedChunk] = []
    for chunk_key in all_keys:
        lex = lexical_ranked.get(chunk_key, 0.0)
        vec = vector_ranked.get(chunk_key, 0.0)
        if lex > 0 and vec > 0:
            method = "both"
            merged_score = (lex + vec) / 2.0
        elif lex > 0:
            method = "lexical"
            merged_score = lex
        else:
            method = "vector"
            merged_score = vec

        meta = fetched.get(chunk_key) or lexical_meta.get(chunk_key) or {}
        if not meta and lex <= 0.0:
            continue
        out.append(
            RetrievedChunk(
                chunk_key=chunk_key,
                rel_path=str(meta.get("rel_path") or ""),
                heading_path=str(meta.get("heading_path") or ""),
                text=str(meta.get("text") or ""),
                score=merged_score,
                method=method,
                lexical_score=lex,
                vector_score=vec,
            )
        )
    out.sort(
        key=lambda item: (
            0 if item.method == "both" else 1,
            -item.score,
            item.chunk_key,
        )
    )
    return out


def _dot_array(a: Optional[array], b: array) -> float:
    if a is None:
        return 0.0
    total = 0.0
    for left, right in zip(a, b):
        total += left * right
    return total


def _l2_norm(values: list[float]) -> float:
    total = 0.0
    for value in values:
        total += float(value) * float(value)
    return math.sqrt(total)


def _l2_norm_arr(values: array) -> float:
    total = 0.0
    for value in values:
        total += value * value
    return math.sqrt(total)


def _reverse_chunk_key(chunk_key: str) -> str:
    if not chunk_key:
        return ""
    lowered = chunk_key.lower()
    if all(ch in "0123456789abcdef" for ch in lowered):
        complement = {
            "0": "f",
            "1": "e",
            "2": "d",
            "3": "c",
            "4": "b",
            "5": "a",
            "6": "9",
            "7": "8",
            "8": "7",
            "9": "6",
            "a": "5",
            "b": "4",
            "c": "3",
            "d": "2",
            "e": "1",
            "f": "0",
        }
        return "".join(complement[ch] for ch in lowered)
    return bytes(255 - b for b in chunk_key.encode("utf-8", errors="replace")).decode("latin1")
