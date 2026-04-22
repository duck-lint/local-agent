from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from agent.retrieval import RetrievedChunk


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


@dataclass(frozen=True)
class CoverageScore:
    lexical_token_coverage: float
    vector_top_score: float
    memory_overlap: float
    lexical_threshold: float
    vector_threshold: float
    memory_weight: float
    should_refine: bool
    query_token_count: int
    matched_token_count: int
    chunks_considered: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "lexical_token_coverage": self.lexical_token_coverage,
            "vector_top_score": self.vector_top_score,
            "memory_overlap": self.memory_overlap,
            "lexical_threshold": self.lexical_threshold,
            "vector_threshold": self.vector_threshold,
            "memory_weight": self.memory_weight,
            "should_refine": self.should_refine,
            "query_token_count": self.query_token_count,
            "matched_token_count": self.matched_token_count,
            "chunks_considered": self.chunks_considered,
            "notes": list(self.notes),
        }


def _memory_signals(memory: object | None) -> tuple[list[str], set[str]]:
    """Extract (topic_summary_terms, active_ref_chunk_keys) from a memory object.

    Accepts SessionState (attributes) or a plain mapping. Missing → empty.
    """
    if memory is None:
        return [], set()
    if isinstance(memory, dict):
        ts_raw = memory.get("topic_summary") or []
        refs_raw = memory.get("active_refs") or []
    else:
        ts_raw = getattr(memory, "topic_summary", None) or []
        refs_raw = getattr(memory, "active_refs", None) or []
    topic_terms = [str(t) for t in ts_raw if isinstance(t, str) and t.strip()]
    keys: set[str] = set()
    for ref in refs_raw:
        if isinstance(ref, dict):
            k = str(ref.get("chunk_key") or "").strip()
        else:
            k = str(getattr(ref, "chunk_key", "") or "").strip()
        if k:
            keys.add(k)
    return topic_terms, keys


def compute_coverage(
    query: str,
    top_chunks: Sequence[RetrievedChunk],
    *,
    lexical_threshold: float,
    vector_threshold: float,
    memory: object | None = None,
    memory_weight: float = 0.0,
) -> CoverageScore:
    """Deterministic coverage predicate.

    lexical_token_coverage = |query_tokens ∩ chunks_tokens| / max(1, |query_tokens|).
    vector_top_score = max chunk.vector_score across top_chunks (0.0 if empty).
    memory_overlap (Phase 4): when `memory` is provided, computed as the max of
      - topic_overlap_lexical = |topic_summary_tokens ∩ chunks_tokens|
                                  / max(1, |topic_summary_tokens|)
      - active_ref_hit_rate   = |active_ref_chunk_keys ∩ round_1_chunk_keys|
                                  / max(1, |active_ref_chunk_keys|)
      Returns 0.0 if memory is None or contains no signal.

    should_refine is True (i.e., round 2 should run) when EITHER:
      - lexical_token_coverage < effective_lexical_threshold, OR
      - vector_top_score < vector_threshold.

    Memory raises the bar: when memory_weight > 0 AND memory_overlap > 0,
    effective lexical_threshold is bumped to lexical_threshold + memory_weight*memory_overlap
    (capped at 1.0). This makes follow-ups more likely to refine.
    """
    notes: list[str] = []
    query_tokens = _tokens(query)

    chunks_tokens: set[str] = set()
    chunk_keys_round1: set[str] = set()
    vector_top_score = 0.0
    for chunk in top_chunks:
        chunks_tokens |= _tokens(chunk.text)
        if chunk.chunk_title:
            chunks_tokens |= _tokens(chunk.chunk_title)
        if chunk.heading_path:
            chunks_tokens |= _tokens(chunk.heading_path)
        ck = getattr(chunk, "chunk_key", "")
        if ck:
            chunk_keys_round1.add(str(ck))
        try:
            vs = float(chunk.vector_score)
        except (TypeError, ValueError):
            vs = 0.0
        if vs > vector_top_score:
            vector_top_score = vs

    matched = query_tokens & chunks_tokens
    qcount = len(query_tokens)
    if qcount == 0:
        lexical_coverage = 0.0
        notes.append("empty_query_tokens")
    else:
        lexical_coverage = len(matched) / qcount

    memory_overlap = 0.0
    if memory is not None:
        topic_terms, ref_keys = _memory_signals(memory)
        topic_tokens: set[str] = set()
        for term in topic_terms:
            topic_tokens |= _tokens(term)
        topic_overlap = (
            len(topic_tokens & chunks_tokens) / len(topic_tokens) if topic_tokens else 0.0
        )
        ref_hit_rate = (
            len(ref_keys & chunk_keys_round1) / len(ref_keys) if ref_keys else 0.0
        )
        memory_overlap = max(topic_overlap, ref_hit_rate)
        if not topic_terms and not ref_keys:
            notes.append("memory_provided_but_empty")

    effective_lex_threshold = lexical_threshold
    if memory_weight > 0.0 and memory_overlap > 0.0:
        effective_lex_threshold = min(1.0, lexical_threshold + memory_weight * memory_overlap)

    lex_pass = lexical_coverage >= effective_lex_threshold
    vec_pass = vector_top_score >= vector_threshold
    should_refine = not (lex_pass and vec_pass)

    return CoverageScore(
        lexical_token_coverage=lexical_coverage,
        vector_top_score=vector_top_score,
        memory_overlap=memory_overlap,
        lexical_threshold=lexical_threshold,
        vector_threshold=vector_threshold,
        memory_weight=memory_weight,
        should_refine=should_refine,
        query_token_count=qcount,
        matched_token_count=len(matched),
        chunks_considered=len(top_chunks),
        notes=notes,
    )
