"""Phase 4: memory-aware coverage predicate."""

from __future__ import annotations

import unittest

from agent.coverage import compute_coverage
from agent.retrieval import RetrievedChunk
from agent.session_memory import ChunkRef, SessionState


def _chunk(chunk_key: str, text: str, *, vector_score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_key=chunk_key,
        doc_key="d1",
        chunk_kind="section",
        rel_path="alpha.md",
        heading_path="Alpha",
        chunk_anchor="a",
        chunk_title="Alpha",
        text=text,
        score=1.0,
        method="fused",
        lexical_score=1.0,
        vector_score=vector_score,
    )


def _state(*, topic_summary=(), active_refs=()) -> SessionState:
    return SessionState(
        session_id="t",
        topic_summary=list(topic_summary),
        active_refs=list(active_refs),
    )


class CoverageMemoryTests(unittest.TestCase):
    def test_memory_none_matches_phase2_behavior(self) -> None:
        chunks = [_chunk("c1", "alpha evidence text")]
        cov_off = compute_coverage(
            "alpha evidence",
            chunks,
            lexical_threshold=0.5,
            vector_threshold=0.5,
            memory=None,
            memory_weight=0.0,
        )
        self.assertEqual(cov_off.memory_overlap, 0.0)
        self.assertFalse(cov_off.should_refine)

    def test_memory_weight_zero_leaves_threshold_unchanged(self) -> None:
        """Parity: memory provided but weight=0 → zero behavioral effect."""
        chunks = [_chunk("c1", "alpha evidence")]
        state = _state(topic_summary=["alpha", "evidence"])
        cov_off = compute_coverage(
            "alpha evidence",
            chunks,
            lexical_threshold=0.5,
            vector_threshold=0.5,
            memory=None,
            memory_weight=0.0,
        )
        cov_memory = compute_coverage(
            "alpha evidence",
            chunks,
            lexical_threshold=0.5,
            vector_threshold=0.5,
            memory=state,
            memory_weight=0.0,
        )
        self.assertEqual(cov_off.should_refine, cov_memory.should_refine)
        self.assertEqual(cov_off.lexical_token_coverage, cov_memory.lexical_token_coverage)

    def test_topic_overlap_signal(self) -> None:
        chunks = [_chunk("c1", "alpha evidence embeddings retrieval")]
        state = _state(topic_summary=["retrieval", "embeddings"])
        cov = compute_coverage(
            "alpha",
            chunks,
            lexical_threshold=0.1,
            vector_threshold=0.5,
            memory=state,
            memory_weight=0.5,
        )
        # Both topic tokens appear in the chunk text → overlap == 1.0
        self.assertEqual(cov.memory_overlap, 1.0)

    def test_active_ref_hit_rate_signal(self) -> None:
        chunks = [_chunk("c1", "text")]
        state = _state(active_refs=[ChunkRef(chunk_key="c1", doc_key="d1")])
        cov = compute_coverage(
            "whatever",
            chunks,
            lexical_threshold=0.1,
            vector_threshold=0.5,
            memory=state,
            memory_weight=0.5,
        )
        self.assertEqual(cov.memory_overlap, 1.0)

    def test_memory_raises_round2_threshold_for_followups(self) -> None:
        """Follow-up with memory signal should be more likely to refine than the
        same query with memory off (same chunks)."""
        chunks = [_chunk("c1", "alpha beta gamma delta")]
        # Query has two tokens but only one is in the chunk → lexical = 0.5.
        state = _state(topic_summary=["alpha"])
        cov_off = compute_coverage(
            "alpha zulu",
            chunks,
            lexical_threshold=0.4,
            vector_threshold=0.5,
            memory=None,
            memory_weight=0.0,
        )
        cov_on = compute_coverage(
            "alpha zulu",
            chunks,
            lexical_threshold=0.4,
            vector_threshold=0.5,
            memory=state,
            memory_weight=1.0,
        )
        # lexical=0.5 ≥ 0.4 → cov_off does not refine.
        self.assertFalse(cov_off.should_refine)
        # With memory, effective threshold = min(1.0, 0.4+1.0*1.0) = 1.0 → lex fails.
        self.assertTrue(cov_on.should_refine)
        self.assertGreater(cov_on.memory_overlap, 0.0)

    def test_memory_overlap_capped_at_one(self) -> None:
        chunks = [_chunk("c1", "alpha")]
        state = _state(
            topic_summary=["alpha", "beta"],
            active_refs=[ChunkRef(chunk_key="c1", doc_key="d")],
        )
        cov = compute_coverage(
            "alpha",
            chunks,
            lexical_threshold=0.5,
            vector_threshold=0.5,
            memory=state,
            memory_weight=1.0,
        )
        self.assertLessEqual(cov.memory_overlap, 1.0)


if __name__ == "__main__":
    unittest.main()
