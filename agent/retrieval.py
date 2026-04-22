from __future__ import annotations

from array import array
from datetime import date, datetime, timezone
import dataclasses
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agent.embedder import Embedder
from agent.embedding_fingerprint import normalize_vector, preprocess_query_text, unpack_vector_f32_le
from agent.embeddings_db import connect_db as connect_embeddings_db
from agent.corpus_db import connect_db as connect_corpus_db
from agent.corpus_db import get_meta as get_corpus_meta
from agent.corpus_db import query_chunks_lexical

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_key: str
    doc_key: str
    chunk_kind: str
    rel_path: str
    heading_path: str
    chunk_anchor: str
    chunk_title: str
    text: str
    score: float
    method: str
    lexical_score: float
    vector_score: float
    expansion_source: Optional[str] = None


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    corpus_contract_sig: str
    embed_model_id: str
    chunk_preprocess_sig: str
    query_preprocess_sig: str
    embed_db_schema_version: int
    lexical_backend_mode: str
    lexical_backend_warning: str
    vector_fetch_k_used: int
    vector_candidates_scored: int
    vector_candidates_prefilter: int
    vector_candidates_postfilter: int
    rel_path_prefix_applied: bool
    vector_filter_warning: str
    rerank_applied: bool
    rerank_intent: str
    rerank_signals_available: bool
    neighbor_expansion_applied: bool = False
    neighbor_scope: str = ""
    neighbor_chunks_added: int = 0
    neighbor_warnings: list[str] = field(default_factory=list)
    candidates: list[RetrievedChunk] = field(default_factory=list)


def retrieve(
    query: str,
    *,
    corpus_db_path: Path,
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
    rrf_k: int = 60,
) -> RetrievalResult:
    if fusion not in ("simple_union", "rrf"):
        raise ValueError(f"Unsupported fusion strategy: {fusion}")
    rrf_k_value = max(1, int(rrf_k))

    lexical_limit = max(1, int(lexical_k))
    vector_limit = max(1, int(vector_k))
    prefix = rel_path_prefix.replace("\\", "/").strip()
    prefix_applied = bool(prefix)
    configured_fetch_k = int(vector_fetch_k)
    if prefix_applied:
        fetch_k_used = max(1, configured_fetch_k) if configured_fetch_k > 0 else max(50, vector_limit * 5)
    else:
        fetch_k_used = vector_limit

    with connect_corpus_db(corpus_db_path) as corpus_conn:
        lexical_rows = query_chunks_lexical(corpus_conn, query_text=query, limit=lexical_limit)
        corpus_contract_sig = get_corpus_meta(corpus_conn, "corpus_contract_sig") or ""
        lexical_backend_mode = get_corpus_meta(corpus_conn, "lexical_backend_mode") or "projection_substring"
        lexical_backend_warning = get_corpus_meta(corpus_conn, "lexical_backend_warning") or ""
    if lexical_rows:
        lexical_backend_mode = str(lexical_rows[0].get("lexical_backend_mode") or lexical_backend_mode)
        lexical_backend_warning = str(lexical_rows[0].get("lexical_backend_warning") or lexical_backend_warning)

    lexical_ranked: dict[str, float] = {}
    lexical_ranks: dict[str, int] = {}
    lexical_meta: dict[str, dict[str, object]] = {}
    lexical_count = max(1, len(lexical_rows))
    for rank, row in enumerate(lexical_rows, start=1):
        chunk_key = str(row.get("chunk_key") or "")
        if not chunk_key:
            continue
        score = 1.0 - ((rank - 1) / lexical_count)
        if score > lexical_ranked.get(chunk_key, -1.0):
            lexical_ranked[chunk_key] = score
            lexical_ranks[chunk_key] = rank
        lexical_meta[chunk_key] = {
            "doc_key": str(row.get("doc_key") or ""),
            "chunk_kind": str(row.get("chunk_kind") or ""),
            "rel_path": str(row.get("rel_path") or ""),
            "heading_path": str(row.get("heading_path") or ""),
            "chunk_anchor": str(row.get("chunk_anchor") or ""),
            "chunk_title": str(row.get("chunk_title") or ""),
            "text": str(row.get("chunk_text") or ""),
            "note_type": str(row.get("note_type") or ""),
            "journal_entry_date": str(row.get("journal_entry_date") or ""),
            "mtime": row.get("mtime") or 0.0,
        }

    query_input = preprocess_query_text(query=query, preprocess_name=preprocess_name)
    query_vectors = embedder.embed_texts([query_input])
    if not query_vectors:
        raise ValueError("Embedder returned no query vector")
    query_vector = normalize_vector(query_vectors[0])
    query_dim = len(query_vector)
    if query_dim <= 0:
        raise ValueError("Query vector dimension must be > 0")

    scored, scored_count, _ = _compute_vector_candidates(
        embeddings_db_path=embeddings_db_path,
        query_vector=query_vector,
        query_dim=query_dim,
        model_id=embed_model_id,
        chunk_preprocess_sig=chunk_preprocess_sig,
        fetch_k=fetch_k_used,
    )
    prefilter_count = len(scored)

    metadata_rows = (
        _fetch_chunk_metadata(corpus_db_path=corpus_db_path, chunk_keys=[key for _, key in scored]) if scored else {}
    )
    filtered_scored = [(score, key) for score, key in scored if key in metadata_rows]
    orphan_dropped = prefilter_count - len(filtered_scored)

    warning_parts: list[str] = []
    if prefix_applied and filtered_scored:
        allowed = {
            key
            for key, row in metadata_rows.items()
            if row["rel_path"].replace("\\", "/").startswith(prefix)
        }
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
    vector_ranks = {chunk_key: idx for idx, (_, chunk_key) in enumerate(vector_top, start=1)}

    with connect_embeddings_db(embeddings_db_path) as embed_conn:
        row = embed_conn.execute("PRAGMA user_version").fetchone()
        embed_schema_version = int(row[0]) if row is not None else 0

    merged = _fuse_candidates(
        corpus_db_path=corpus_db_path,
        lexical_ranked=lexical_ranked,
        lexical_meta=lexical_meta,
        vector_ranked=vector_ranked,
        lexical_ranks=lexical_ranks,
        vector_ranks=vector_ranks,
        strategy=fusion,
        rrf_k=rrf_k_value,
    )
    reranked, rerank_applied, rerank_intent, rerank_signals_available = _apply_bounded_rerank(
        query=query,
        candidates=merged,
        chunk_meta={**metadata_rows, **lexical_meta},
    )
    return RetrievalResult(
        query=query,
        corpus_contract_sig=corpus_contract_sig,
        embed_model_id=embed_model_id,
        chunk_preprocess_sig=chunk_preprocess_sig,
        query_preprocess_sig=query_preprocess_sig,
        embed_db_schema_version=embed_schema_version,
        lexical_backend_mode=lexical_backend_mode,
        lexical_backend_warning=lexical_backend_warning,
        vector_fetch_k_used=fetch_k_used,
        vector_candidates_scored=scored_count,
        vector_candidates_prefilter=prefilter_count,
        vector_candidates_postfilter=len(filtered_scored),
        rel_path_prefix_applied=prefix_applied,
        vector_filter_warning=filter_warning,
        rerank_applied=rerank_applied,
        rerank_intent=rerank_intent,
        rerank_signals_available=rerank_signals_available,
        neighbor_expansion_applied=False,
        neighbor_scope="",
        neighbor_chunks_added=0,
        neighbor_warnings=[],
        candidates=reranked,
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
    vectors_normalized = True

    q_np = None
    q_arr: array | None = None
    q_norm = _l2_norm(query_vector)
    if _np is not None:
        q_np = _np.asarray(query_vector, dtype=_np.float32)
    else:
        q_arr = array("f", query_vector)

    with connect_embeddings_db(embeddings_db_path) as embed_conn:
        row = embed_conn.execute("SELECT value FROM meta WHERE key = 'vectors_normalized'").fetchone()
        vectors_normalized = str(row["value"]) == "1" if row is not None else True
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
                if vectors_normalized:
                    score = float(_np.dot(q_np, vec_np))
                else:
                    denom = float(_np.linalg.norm(vec_np)) * max(q_norm, 1e-12)
                    score = float(_np.dot(q_np, vec_np) / denom) if denom > 0.0 else 0.0
            else:
                vec_arr = unpack_vector_f32_le(blob)
                if vectors_normalized:
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
            elif pair > scored_heap[0]:
                heapq.heapreplace(scored_heap, pair)

    ranked = sorted(
        [(score, chunk_key) for score, _, chunk_key in scored_heap],
        key=lambda item: (-item[0], item[1]),
    )
    return ranked, scored_count, vectors_normalized


def _fetch_chunk_metadata(*, corpus_db_path: Path, chunk_keys: list[str]) -> dict[str, dict[str, object]]:
    unique_keys = sorted({key for key in chunk_keys if key})
    if not unique_keys:
        return {}
    placeholders = ",".join("?" for _ in unique_keys)
    with connect_corpus_db(corpus_db_path) as corpus_conn:
        rows = corpus_conn.execute(
            f"""
            SELECT
                chunks.chunk_key AS chunk_key,
                chunks.doc_key AS doc_key,
                chunks.chunk_kind AS chunk_kind,
                documents.rel_path AS rel_path,
                chunks.heading_path AS heading_path,
                chunks.chunk_anchor AS chunk_anchor,
                chunks.chunk_title AS chunk_title,
                chunks.text AS chunk_text,
                documents.entry_date AS entry_date,
                documents.mtime AS mtime,
                documents.frontmatter_json AS frontmatter_json
            FROM chunks
            INNER JOIN documents ON documents.id = chunks.doc_id
            WHERE chunks.chunk_key IN ({placeholders})
            """,
            unique_keys,
        ).fetchall()
    return {
        str(row["chunk_key"]): {
            "doc_key": str(row["doc_key"]),
            "chunk_kind": str(row["chunk_kind"]),
            "rel_path": str(row["rel_path"]),
            "heading_path": str(row["heading_path"]),
            "chunk_anchor": str(row["chunk_anchor"]),
            "chunk_title": str(row["chunk_title"]),
            "text": str(row["chunk_text"]),
            "note_type": _note_type_from_frontmatter(str(row["frontmatter_json"] or "")),
            "journal_entry_date": str(row["entry_date"] or ""),
            "mtime": row["mtime"] or 0.0,
        }
        for row in rows
    }


def _fuse_candidates(
    *,
    corpus_db_path: Path,
    lexical_ranked: dict[str, float],
    lexical_meta: dict[str, dict[str, object]],
    vector_ranked: dict[str, float],
    lexical_ranks: dict[str, int] | None = None,
    vector_ranks: dict[str, int] | None = None,
    strategy: str = "simple_union",
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    if strategy not in ("simple_union", "rrf"):
        raise ValueError(f"Unsupported fusion strategy: {strategy}")
    lex_ranks = lexical_ranks or {}
    vec_ranks = vector_ranks or {}
    rrf_k_value = max(1, int(rrf_k))
    all_keys = sorted(set(lexical_ranked) | set(vector_ranked))
    fetched = _fetch_chunk_metadata(corpus_db_path=corpus_db_path, chunk_keys=all_keys)
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

        if strategy == "rrf":
            merged_score = 0.0
            lex_rank = lex_ranks.get(chunk_key)
            vec_rank = vec_ranks.get(chunk_key)
            if lex_rank is not None:
                merged_score += 1.0 / (rrf_k_value + lex_rank)
            if vec_rank is not None:
                merged_score += 1.0 / (rrf_k_value + vec_rank)
        else:
            if method == "both":
                merged_score = (lex + vec) / 2.0
            elif method == "lexical":
                merged_score = lex
            else:
                merged_score = vec

        meta = fetched.get(chunk_key) or lexical_meta.get(chunk_key) or {}
        if not meta and lex <= 0.0:
            continue
        out.append(
            RetrievedChunk(
                chunk_key=chunk_key,
                doc_key=str(meta.get("doc_key") or ""),
                chunk_kind=str(meta.get("chunk_kind") or ""),
                rel_path=str(meta.get("rel_path") or ""),
                heading_path=str(meta.get("heading_path") or ""),
                chunk_anchor=str(meta.get("chunk_anchor") or ""),
                chunk_title=str(meta.get("chunk_title") or ""),
                text=str(meta.get("text") or ""),
                score=merged_score,
                method=method,
                lexical_score=lex,
                vector_score=vec,
            )
        )
    out.sort(key=lambda item: (0 if item.method == "both" else 1, -item.score, item.chunk_key))
    return out


def _apply_bounded_rerank(
    *,
    query: str,
    candidates: list[RetrievedChunk],
    chunk_meta: dict[str, dict[str, object]],
) -> tuple[list[RetrievedChunk], bool, str, bool]:
    intent = _detect_rerank_intent(query)
    if not intent or not candidates:
        return candidates, False, "", False

    original_order = {item.chunk_key: index for index, item in enumerate(candidates)}
    signals_available = False

    def sort_key(item: RetrievedChunk) -> tuple[int, int, int, int]:
        nonlocal signals_available
        meta = chunk_meta.get(item.chunk_key, {})
        note_type = str(meta.get("note_type") or "").strip().lower()
        journal_entry_date = _date_ordinal(str(meta.get("journal_entry_date") or ""))
        mtime_date = _mtime_ordinal(meta.get("mtime"))
        best_date = journal_entry_date or mtime_date
        class_match = note_type in {"journal", "journal_entry", "journal-entry"}

        if intent in {"journal", "journal_recent"} and class_match:
            signals_available = True
        if intent in {"recent", "journal_recent"} and best_date > 0:
            signals_available = True

        class_bucket = 0 if intent not in {"journal", "journal_recent"} or class_match else 1
        if intent in {"recent", "journal_recent"}:
            date_bucket = 0 if best_date > 0 else 1
            recency_key = -best_date if best_date > 0 else 0
        else:
            date_bucket = 0
            recency_key = 0
        return (
            class_bucket,
            date_bucket,
            recency_key,
            original_order.get(item.chunk_key, 0),
        )

    reranked = sorted(candidates, key=sort_key)
    applied = signals_available and [item.chunk_key for item in reranked] != [item.chunk_key for item in candidates]
    return reranked, applied, intent, signals_available


def _detect_rerank_intent(query: str) -> str:
    lowered = str(query or "").strip().lower()
    if not lowered:
        return ""
    wants_recent = any(token in lowered for token in ("most recent", "latest", "newest", "recent"))
    wants_journal = any(token in lowered for token in ("journal entries", "journal entry", "journal"))
    if wants_recent and wants_journal:
        return "journal_recent"
    if wants_journal:
        return "journal"
    if wants_recent:
        return "recent"
    return ""


def _date_ordinal(raw_value: str) -> int:
    text = str(raw_value or "").strip()
    if not text:
        return 0
    try:
        return date.fromisoformat(text[:10]).toordinal()
    except ValueError:
        return 0


def _mtime_ordinal(raw_value: object) -> int:
    try:
        timestamp = float(raw_value or 0.0)
    except (TypeError, ValueError):
        return 0
    if timestamp <= 0.0:
        return 0
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().toordinal()
    except (OverflowError, OSError, ValueError):
        return 0


def _note_type_from_frontmatter(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(loaded, dict):
        return ""
    return str(loaded.get("note_type") or "").strip()


def _dot_array(a: array | None, b: array) -> float:
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
    return bytes(255 - b for b in chunk_key.encode("utf-8", errors="replace")).decode("latin1")


NEIGHBOR_SCOPES = ("adjacent_only", "same_section", "same_heading_path")


def expand_neighbors(
    result: RetrievalResult,
    *,
    corpus_db_path: Path,
    scope: str,
) -> RetrievalResult:
    """Append same-document neighbor chunks to result.candidates based on scope.

    Pure: only reads the corpus DB; does not mutate the input.
    Dedup: a neighbor whose chunk_key already appears in result.candidates is skipped.
    Original candidates retain their order and scores; neighbors are appended after them,
    sorted deterministically by (rel_path, chunk_index). Each neighbor is constructed
    with score=0.0, method="neighbor", lexical_score=0.0, vector_score=0.0,
    expansion_source=<scope>.
    """
    if scope not in NEIGHBOR_SCOPES:
        raise ValueError(
            f"Unknown neighbor expansion scope: {scope!r} (expected one of {NEIGHBOR_SCOPES})"
        )

    from agent.corpus_db import fetch_neighbor_chunks

    candidate_keys = {chunk.chunk_key for chunk in result.candidates}
    warnings: list[str] = []

    try:
        with connect_corpus_db(corpus_db_path) as conn:
            neighbor_dicts = fetch_neighbor_chunks(
                conn,
                chunk_keys=[chunk.chunk_key for chunk in result.candidates],
                scope=scope,
            )
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - degrade per plan, surface in warnings
        warnings.append(f"neighbor_expansion_db_error: {exc}")
        return dataclasses.replace(
            result,
            neighbor_expansion_applied=False,
            neighbor_scope=scope,
            neighbor_chunks_added=0,
            neighbor_warnings=warnings,
        )

    seen: set[str] = set(candidate_keys)
    new_neighbors: list[RetrievedChunk] = []
    for nd in neighbor_dicts:
        ckey = str(nd.get("chunk_key", ""))
        if not ckey or ckey in seen:
            continue
        seen.add(ckey)
        new_neighbors.append(
            RetrievedChunk(
                chunk_key=ckey,
                doc_key=str(nd.get("doc_key", "")),
                chunk_kind=str(nd.get("chunk_kind", "")),
                rel_path=str(nd.get("rel_path", "")),
                heading_path=str(nd.get("heading_path", "")),
                chunk_anchor=str(nd.get("chunk_anchor", "")),
                chunk_title=str(nd.get("chunk_title", "")),
                text=str(nd.get("text", "")),
                score=0.0,
                method="neighbor",
                lexical_score=0.0,
                vector_score=0.0,
                expansion_source=scope,
            )
        )

    chunk_index_map = {str(d.get("chunk_key", "")): int(d.get("chunk_index", 0)) for d in neighbor_dicts}
    new_neighbors.sort(key=lambda c: (c.rel_path, chunk_index_map.get(c.chunk_key, 0)))

    return dataclasses.replace(
        result,
        neighbor_expansion_applied=True,
        neighbor_scope=scope,
        neighbor_chunks_added=len(new_neighbors),
        neighbor_warnings=warnings,
        candidates=list(result.candidates) + new_neighbors,
    )


@dataclass(frozen=True)
class RefinementRound:
    round_index: int
    query_used: str
    rewrite: Optional[dict]
    result: RetrievalResult


@dataclass(frozen=True)
class RefinementOutcome:
    final_result: RetrievalResult
    rounds: list[RefinementRound]
    coverage: Optional[dict]
    rewritten_query: str
    refinement_applied: bool


def retrieve_with_refinement(
    query: str,
    *,
    app_config,
    corpus_db_path: Path,
    embeddings_db_path: Path,
    embedder: Embedder,
    embed_model_id: str,
    preprocess_name: str,
    chunk_preprocess_sig: str,
    query_preprocess_sig: str,
    memory: object | None = None,
) -> RefinementOutcome:
    """Phase 2 retrieval orchestrator.

    1. Run round 1 with the original query.
    2. If refinement_round_enabled: compute coverage on round-1 top_n candidates.
    3. If coverage.should_refine AND rewrite.rule_based_enabled AND rewrite is non-identity:
       run round 2 with the rewritten query, then merge candidates (round 1 first,
       round 2 only contributes new chunk_keys). If round 2 adds zero new candidates,
       degrade to round-1 result (refinement_applied stays False).
    4. If neighbor_expansion_enabled: expand neighbors on the FINAL merged result
       (so neighbors are sourced from both rounds' chunks combined).
    """
    retrieval_cfg = app_config.retrieval
    common_kwargs = dict(
        corpus_db_path=corpus_db_path,
        embeddings_db_path=embeddings_db_path,
        embedder=embedder,
        embed_model_id=embed_model_id,
        preprocess_name=preprocess_name,
        chunk_preprocess_sig=chunk_preprocess_sig,
        query_preprocess_sig=query_preprocess_sig,
        lexical_k=retrieval_cfg.lexical_k,
        vector_k=retrieval_cfg.vector_k,
        vector_fetch_k=retrieval_cfg.vector_fetch_k,
        rel_path_prefix=retrieval_cfg.rel_path_prefix,
        fusion=retrieval_cfg.fusion,
        rrf_k=retrieval_cfg.rrf_k,
    )

    r1 = retrieve(query, **common_kwargs)
    rounds: list[RefinementRound] = [
        RefinementRound(round_index=1, query_used=query, rewrite=None, result=r1)
    ]
    coverage_dict: Optional[dict] = None
    rewritten_query = ""
    refinement_applied = False
    final_result = r1

    if retrieval_cfg.refinement_round_enabled:
        from agent.coverage import compute_coverage

        cov_cfg = retrieval_cfg.coverage_predicate
        top_n = max(1, int(cov_cfg.top_n))
        # Effective memory_weight: session-level override takes precedence when
        # memory is actually provided AND non-zero; falls back to the static
        # coverage_predicate.memory_weight otherwise.
        session_weight = float(getattr(getattr(app_config, "session", None), "coverage_memory_weight", 0.0) or 0.0)
        effective_memory_weight = session_weight if (memory is not None and session_weight > 0.0) else float(cov_cfg.memory_weight)
        cov = compute_coverage(
            query,
            r1.candidates[:top_n],
            lexical_threshold=float(cov_cfg.lexical_threshold),
            vector_threshold=float(cov_cfg.vector_threshold),
            memory=memory,
            memory_weight=effective_memory_weight,
        )
        coverage_dict = cov.to_dict()

        if cov.should_refine and retrieval_cfg.rewrite.rule_based_enabled:
            from agent.rewrite import rule_based_rewrite

            # Memory contributes seeds to rewrite only when session.memory_rewrite_enabled.
            memory_for_rewrite = memory if bool(getattr(getattr(app_config, "session", None), "memory_rewrite_enabled", False)) else None
            rw = rule_based_rewrite(
                query,
                acronyms=retrieval_cfg.rewrite.acronyms,
                synonyms=retrieval_cfg.rewrite.synonyms,
                memory=memory_for_rewrite,
            )
            if not rw.is_identity() and rw.rewritten and rw.rewritten != query:
                rewritten_query = rw.rewritten
                r2 = retrieve(rw.rewritten, **common_kwargs)
                rounds.append(
                    RefinementRound(
                        round_index=2,
                        query_used=rw.rewritten,
                        rewrite=rw.to_dict(),
                        result=r2,
                    )
                )
                seen = {c.chunk_key for c in r1.candidates}
                added = [c for c in r2.candidates if c.chunk_key not in seen]
                if added:
                    refinement_applied = True
                    final_result = dataclasses.replace(
                        r1, candidates=list(r1.candidates) + added
                    )
                # else: degrade to r1, refinement_applied stays False

    if retrieval_cfg.neighbor_expansion_enabled:
        final_result = expand_neighbors(
            final_result,
            corpus_db_path=corpus_db_path,
            scope=retrieval_cfg.neighbor_scope,
        )

    return RefinementOutcome(
        final_result=final_result,
        rounds=rounds,
        coverage=coverage_dict,
        rewritten_query=rewritten_query,
        refinement_applied=refinement_applied,
    )

