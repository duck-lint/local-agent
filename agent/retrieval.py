from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agent.embedder import Embedder
from agent.embedding_fingerprint import (
    build_query_embedding_input,
    compute_embed_preprocess_sig,
    unpack_vector_f32_le,
)
from agent.embeddings_db import connect_db as connect_embeddings_db
from agent.embeddings_db import get_meta as get_embed_meta
from agent.index_db import connect_db as connect_index_db
from agent.index_db import get_meta as get_index_meta
from agent.index_db import query_chunks_lexical


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
    preprocess_sig: str
    embed_db_schema_version: int
    candidates: list[RetrievedChunk]


def retrieve(
    query: str,
    *,
    index_db_path: Path,
    embeddings_db_path: Path,
    embedder: Embedder,
    embed_model_id: str,
    preprocess_name: str,
    lexical_k: int,
    vector_k: int,
    fusion: str,
) -> RetrievalResult:
    if fusion != "simple_union":
        raise ValueError(f"Unsupported fusion strategy: {fusion}")

    preprocess_sig = compute_embed_preprocess_sig(preprocess_name)
    lexical_limit = max(1, int(lexical_k))
    vector_limit = max(1, int(vector_k))

    with connect_index_db(index_db_path) as index_conn:
        lexical_rows = [dict(row) for row in query_chunks_lexical(index_conn, query_text=query, limit=lexical_limit)]
        chunker_sig = get_index_meta(index_conn, "chunker_sig") or ""
        chunk_rows = index_conn.execute(
            """
            SELECT chunks.chunk_key AS chunk_key,
                   chunks.text AS chunk_text,
                   COALESCE(chunks.heading_path, '') AS heading_path,
                   docs.rel_path AS rel_path
            FROM chunks
            INNER JOIN docs ON docs.id = chunks.doc_id
            WHERE chunks.chunk_key IS NOT NULL AND trim(chunks.chunk_key) != ''
            """
        ).fetchall()
    chunk_map = {
        str(row["chunk_key"]): {
            "text": str(row["chunk_text"]),
            "heading_path": str(row["heading_path"]),
            "rel_path": str(row["rel_path"]),
        }
        for row in chunk_rows
    }

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

    query_input = build_query_embedding_input(preprocess_name=preprocess_name, query=query)
    query_vectors = embedder.embed_texts([query_input])
    if not query_vectors:
        raise ValueError("Embedder returned no query vector")
    query_vector = query_vectors[0]
    query_dim = len(query_vector)
    if query_dim <= 0:
        raise ValueError("Query vector dimension must be > 0")

    vector_ranked = _compute_vector_candidates(
        embeddings_db_path=embeddings_db_path,
        chunk_map=chunk_map,
        query_vector=query_vector,
        query_dim=query_dim,
        model_id=embed_model_id,
        preprocess_sig=preprocess_sig,
        limit=vector_limit,
    )

    with connect_embeddings_db(embeddings_db_path) as embed_conn:
        row = embed_conn.execute("PRAGMA user_version").fetchone()
        embed_schema_version = int(row[0]) if row is not None else 0

    merged = _fuse_candidates(lexical_ranked, vector_ranked, lexical_meta, chunk_map)
    return RetrievalResult(
        query=query,
        chunker_sig=chunker_sig,
        embed_model_id=embed_model_id,
        preprocess_sig=preprocess_sig,
        embed_db_schema_version=embed_schema_version,
        candidates=merged,
    )


def _compute_vector_candidates(
    *,
    embeddings_db_path: Path,
    chunk_map: dict[str, dict[str, str]],
    query_vector: list[float],
    query_dim: int,
    model_id: str,
    preprocess_sig: str,
    limit: int,
) -> dict[str, float]:
    scored_heap: list[tuple[float, str]] = []

    query_norm = _l2_norm(query_vector)
    if query_norm == 0.0:
        return {}

    with connect_embeddings_db(embeddings_db_path) as embed_conn:
        for row in embed_conn.execute("SELECT chunk_key, model_id, dim, preprocess_sig, vector FROM embeddings"):
            chunk_key = str(row["chunk_key"])
            if chunk_key not in chunk_map:
                continue
            if str(row["model_id"]) != model_id:
                continue
            if str(row["preprocess_sig"]) != preprocess_sig:
                continue

            dim = int(row["dim"])
            if dim != query_dim:
                continue
            blob = bytes(row["vector"])
            vec = unpack_vector_f32_le(blob)
            if len(vec) != dim:
                continue
            sim = _cosine_similarity(query_vector, vec, query_norm=query_norm)
            if len(scored_heap) < limit:
                heapq.heappush(scored_heap, (sim, chunk_key))
            else:
                heapq.heappushpop(scored_heap, (sim, chunk_key))

    ranked = sorted(scored_heap, key=lambda item: item[0], reverse=True)
    return {chunk_key: (score + 1.0) / 2.0 for score, chunk_key in ranked}


def _fuse_candidates(
    lexical_ranked: dict[str, float],
    vector_ranked: dict[str, float],
    lexical_meta: dict[str, dict[str, str]],
    chunk_map: dict[str, dict[str, str]],
) -> list[RetrievedChunk]:
    all_keys = sorted(set(lexical_ranked) | set(vector_ranked))
    out: list[RetrievedChunk] = []
    for chunk_key in all_keys:
        lex = lexical_ranked.get(chunk_key, 0.0)
        vec = vector_ranked.get(chunk_key, 0.0)
        if lex > 0 and vec > 0:
            method = "both"
        elif lex > 0:
            method = "lexical"
        else:
            method = "vector"

        merged_score = (lex + vec) / (2.0 if (lex > 0 and vec > 0) else 1.0)

        meta = lexical_meta.get(chunk_key) or chunk_map.get(chunk_key) or {}
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
            1 if item.method == "both" else 0,
            item.score,
        ),
        reverse=True,
    )
    return out


def _l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def _cosine_similarity(a: list[float], b: list[float], *, query_norm: Optional[float] = None) -> float:
    b_list = [float(x) for x in b]
    a_norm = query_norm if query_norm is not None else _l2_norm(a)
    b_norm = _l2_norm(b_list)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b_list))
    return dot / (a_norm * b_norm)
